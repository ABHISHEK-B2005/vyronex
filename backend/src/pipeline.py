"""Top-level per-frame analysis pipeline.

CAMERA -> POSE -> LANDMARKS -> BIOMECHANICAL FEATURES -> SQUAT DETECTION
-> FORM ANALYSIS -> RISK ENGINE -> EXPLAINABLE FEEDBACK

This module glues together the smaller, independently-testable modules
(geometry, landmarks, squat_detector, feature_engineering, form_analysis,
risk_engine, feedback_engine) into the single call the Streamlit app
needs per frame: ``AnalysisPipeline.process_frame(landmarks, timestamp)``.
"""

from __future__ import annotations

import time
from collections import deque
from math import hypot, sqrt
from dataclasses import dataclass
from typing import Optional

from src.feature_engineering import RepFeatures, extract_rep_features
from src.feedback_engine import Feedback, generate_feedback
from src.form_analysis import FormScores, analyze_alignment, analyze_form
from src.geometry import calculate_angle, calculate_angle_3d, compute_alignment, detect_view
from src.landmarks import BodyLandmarks
from src.risk_engine import RiskResult, RuleBasedRiskEngine, get_risk_engine
from src.squat_detector import CompletedRep, RepFrameSample, SquatDetector, SquatPhase
from src.config import (
    MAX_KNEE_DRIFT_LEG_RATIO,
    MAX_FRAME_JUMP_TORSO_RATIO,
    POSITION_MEDIAN_WINDOW,
    POSITION_SMOOTHING_ALPHA,
)


REQUIRED_FOR_SQUAT = [
    "left_shoulder", "right_shoulder",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]


@dataclass
class RepResult:
    """Bundle returned whenever a repetition completes."""
    rep: CompletedRep
    features: RepFeatures
    form_scores: FormScores
    risk_result: RiskResult
    feedback: Feedback


@dataclass
class FrameStatus:
    """Bundle returned every frame (whether or not a rep just completed)."""
    landmarks_ok: bool
    phase: SquatPhase
    rep_count: int
    avg_knee_angle: Optional[float] = None
    completed_rep: Optional[RepResult] = None
    message: Optional[str] = None
    tracking_error: bool = False
    view: str = "unknown"
    alignment: Optional[dict] = None


def _compute_torso_angle(landmarks: BodyLandmarks) -> Optional[float]:
    left_shoulder = landmarks.point3d("left_shoulder")
    right_shoulder = landmarks.point3d("right_shoulder")
    left_hip = landmarks.point3d("left_hip")
    right_hip = landmarks.point3d("right_hip")
    if any(point is None for point in (left_shoulder, right_shoulder, left_hip, right_hip)):
        return None
    shoulder_mid = tuple((a + b) / 2 for a, b in zip(left_shoulder, right_shoulder))
    hip_mid_point = tuple((a + b) / 2 for a, b in zip(left_hip, right_hip))
    vertical_reference = (hip_mid_point[0], hip_mid_point[1] - 100, hip_mid_point[2])
    angle_from_vertical = calculate_angle_3d(shoulder_mid, hip_mid_point, vertical_reference)
    torso_angle = 90.0 - min(angle_from_vertical, 90.0)
    return round(max(0.0, torso_angle), 2)


def _body_lateral_axis(landmarks: BodyLandmarks) -> Optional[tuple[float, float, float]]:
    left = landmarks.point3d("left_hip")
    right = landmarks.point3d("right_hip")
    if left is None or right is None:
        return None
    lateral = (right[0] - left[0], 0.0, right[2] - left[2])
    length = sqrt(sum(value * value for value in lateral))
    if length <= 1e-6:
        return None
    return tuple(value / length for value in lateral)


def _sagittal_joint_angle(
    point_a: tuple[float, float, float],
    point_b: tuple[float, float, float],
    point_c: tuple[float, float, float],
    lateral: tuple[float, float, float],
) -> float:
    vectors = []
    for point in (point_a, point_c):
        vector = tuple(value - centre for value, centre in zip(point, point_b))
        lateral_component = sum(value * axis for value, axis in zip(vector, lateral))
        vectors.append(tuple(value - lateral_component * axis for value, axis in zip(vector, lateral)))
    origin = (0.0, 0.0, 0.0)
    return calculate_angle_3d(vectors[0], origin, vectors[1])


def _compute_knee_alignment_shift(landmarks: BodyLandmarks) -> float:
    """Measure knee movement toward the body's midline relative to each leg."""
    lateral = _body_lateral_axis(landmarks)
    if lateral is None:
        return 0.0

    shifts = []
    for side in ("left", "right"):
        hip = landmarks.point3d(f"{side}_hip")
        knee = landmarks.point3d(f"{side}_knee")
        ankle = landmarks.point3d(f"{side}_ankle")
        if not hip or not knee or not ankle:
            continue
        leg_vector = tuple(a - h for a, h in zip(ankle, hip))
        leg_length = sqrt(sum(value * value for value in leg_vector))
        if leg_length <= 1e-6:
            continue
        t = sum((k - h) * leg for k, h, leg in zip(knee, hip, leg_vector)) / (leg_length * leg_length)
        t = max(0.0, min(1.0, t))
        expected = tuple(h + t * leg for h, leg in zip(hip, leg_vector))
        deviation = tuple(k - e for k, e in zip(knee, expected))
        toward_midline = sum(a * b for a, b in zip(deviation, lateral))
        shifts.append((toward_midline if side == "left" else -toward_midline) / leg_length)
    if not shifts:
        return 0.0
    return max(0.0, max(shifts))


def build_frame_sample(
    landmarks: BodyLandmarks,
    timestamp: Optional[float] = None,
    camera_view: Optional[str] = None,
) -> Optional[RepFrameSample]:
    """Convert a single frame's BodyLandmarks into a RepFrameSample, or
    None if required landmarks are missing/low-confidence.
    """
    if not landmarks.has_all(REQUIRED_FOR_SQUAT):
        return None

    left_hip = landmarks.point3d("left_hip")
    right_hip = landmarks.point3d("right_hip")
    left_knee = landmarks.point3d("left_knee")
    right_knee = landmarks.point3d("right_knee")
    left_ankle = landmarks.point3d("left_ankle")
    right_ankle = landmarks.point3d("right_ankle")
    left_shoulder = landmarks.point3d("left_shoulder")
    right_shoulder = landmarks.point3d("right_shoulder")

    lateral = _body_lateral_axis(landmarks) or (1.0, 0.0, 0.0)
    angle = (
        calculate_angle
        if landmarks.low_confidence
        else lambda a, b, c: _sagittal_joint_angle(a, b, c, lateral)
    )
    if landmarks.low_confidence:
        left_knee_angle = angle(left_hip[:2], left_knee[:2], left_ankle[:2])
        right_knee_angle = angle(right_hip[:2], right_knee[:2], right_ankle[:2])
        left_hip_angle = angle(left_shoulder[:2], left_hip[:2], left_knee[:2])
        right_hip_angle = angle(right_shoulder[:2], right_hip[:2], right_knee[:2])
    else:
        left_knee_angle = angle(left_hip, left_knee, left_ankle)
        right_knee_angle = angle(right_hip, right_knee, right_ankle)
        left_hip_angle = angle(left_shoulder, left_hip, left_knee)
        right_hip_angle = angle(right_shoulder, right_hip, right_knee)

    torso_angle = _compute_torso_angle(landmarks)
    if torso_angle is None:
        torso_angle = 90.0

    knee_alignment_shift = _compute_knee_alignment_shift(landmarks)
    hip_mid = tuple((a + b) / 2 for a, b in zip(left_hip, right_hip))

    avg_knee_angle = (left_knee_angle + right_knee_angle) / 2.0
    angles = (left_knee_angle, right_knee_angle, left_hip_angle, right_hip_angle)
    if any(angle < 20.0 or angle > 180.0 for angle in angles):
        return None
    if camera_view == "front" and knee_alignment_shift > MAX_KNEE_DRIFT_LEG_RATIO:
        return None

    resolved_view = camera_view or detect_view(landmarks)
    return RepFrameSample(
        timestamp=timestamp if timestamp is not None else time.time(),
        avg_knee_angle=avg_knee_angle,
        left_knee_angle=left_knee_angle,
        right_knee_angle=right_knee_angle,
        left_hip_angle=left_hip_angle,
        right_hip_angle=right_hip_angle,
        torso_angle=torso_angle,
        knee_alignment_shift=knee_alignment_shift,
        hip_midpoint=hip_mid,
        camera_view=resolved_view,
        low_confidence=landmarks.low_confidence,
        alignment=compute_alignment(landmarks, resolved_view),
    )


class AnalysisPipeline:
    """Stateful per-session pipeline: owns the SquatDetector and the
    currently-selected risk engine, and produces a FrameStatus each time
    a frame is processed.
    """

    def __init__(self, risk_mode: str = "rule_based", smooth_signals: bool = False):
        self.squat_detector = SquatDetector()
        self.risk_mode = risk_mode
        self.smooth_signals = smooth_signals
        self._risk_engine = get_risk_engine(risk_mode)
        self._angle_history = {
            name: deque(maxlen=POSITION_MEDIAN_WINDOW)
            for name in ("avg_knee_angle", "left_knee_angle", "right_knee_angle",
                         "left_hip_angle", "right_hip_angle", "torso_angle",
                         "knee_alignment_shift")
        }
        self._angle_lowpass: dict[str, float] = {}
        self._previous_landmarks: Optional[BodyLandmarks] = None

    def _smooth_sample(self, sample: RepFrameSample) -> RepFrameSample:
        from statistics import median

        values = {
            name: getattr(sample, name)
            for name in self._angle_history
        }
        smoothed = {}
        for name, value in values.items():
            history = self._angle_history[name]
            history.append(value)
            filtered = float(median(history))
            previous = self._angle_lowpass.get(name, filtered)
            current = previous + POSITION_SMOOTHING_ALPHA * (filtered - previous)
            self._angle_lowpass[name] = current
            smoothed[name] = current
        return RepFrameSample(
            **{
                **sample.__dict__,
                **{name: round(value, 3) for name, value in smoothed.items()},
            }
        )

    def set_risk_mode(self, mode: str) -> None:
        if mode != self.risk_mode:
            self.risk_mode = mode
            self._risk_engine = get_risk_engine(mode)

    def reset(self) -> None:
        self.squat_detector.reset()
        for history in self._angle_history.values():
            history.clear()
        self._angle_lowpass.clear()
        self._previous_landmarks = None

    def _has_implausible_position_jump(self, landmarks: BodyLandmarks) -> bool:
        previous = self._previous_landmarks
        self._previous_landmarks = landmarks
        if previous is None:
            return False
        torso_points = (
            landmarks.point3d("left_shoulder"),
            landmarks.point3d("right_shoulder"),
            landmarks.point3d("left_hip"),
            landmarks.point3d("right_hip"),
        )
        if any(point is None for point in torso_points):
            return False
        shoulder_mid = tuple((a + b) / 2 for a, b in zip(torso_points[0], torso_points[1]))
        hip_mid = tuple((a + b) / 2 for a, b in zip(torso_points[2], torso_points[3]))
        torso_length = sqrt(sum((a - b) ** 2 for a, b in zip(shoulder_mid, hip_mid)))
        if torso_length <= 1e-6:
            return False
        for name in ("left_shoulder", "right_shoulder", "left_hip", "right_hip",
                     "left_knee", "right_knee", "left_ankle", "right_ankle"):
            current_point = landmarks.point3d(name)
            previous_point = previous.point3d(name)
            if current_point is None or previous_point is None:
                continue
            distance = sqrt(sum((a - b) ** 2 for a, b in zip(current_point, previous_point)))
            if distance / torso_length > MAX_FRAME_JUMP_TORSO_RATIO:
                return True
        return False

    def process_frame(
        self,
        landmarks: Optional[BodyLandmarks],
        timestamp: Optional[float] = None,
        camera_view: Optional[str] = None,
    ) -> FrameStatus:
        if landmarks is None:
            return FrameStatus(
                landmarks_ok=False,
                phase=self.squat_detector.phase,
                rep_count=self.squat_detector.rep_count,
                message="No person detected in frame.",
            )

        if self._has_implausible_position_jump(landmarks):
            return FrameStatus(
                landmarks_ok=False,
                phase=self.squat_detector.phase,
                rep_count=self.squat_detector.rep_count,
                message="Tracking error: joint position jumped between frames.",
                tracking_error=True,
            )

        sample = build_frame_sample(landmarks, timestamp, camera_view)
        if sample is None:
            return FrameStatus(
                landmarks_ok=False,
                phase=self.squat_detector.phase,
                rep_count=self.squat_detector.rep_count,
                message="Tracking error: implausible pose geometry or low landmark confidence.",
                tracking_error=True,
            )

        if self.smooth_signals:
            sample = self._smooth_sample(sample)
        completed_rep = self.squat_detector.update(sample)
        rep_result: Optional[RepResult] = None

        if completed_rep is not None:
            features = extract_rep_features(completed_rep)
            features.alignment = analyze_alignment(features)
            form_scores = analyze_form(features)
            risk_result = self._risk_engine.assess(form_scores, features)
            feedback = generate_feedback(risk_result, form_scores, features)
            rep_result = RepResult(
                rep=completed_rep,
                features=features,
                form_scores=form_scores,
                risk_result=risk_result,
                feedback=feedback,
            )

        return FrameStatus(
            landmarks_ok=True,
            phase=self.squat_detector.phase,
            rep_count=self.squat_detector.rep_count,
            avg_knee_angle=round(sample.avg_knee_angle, 1),
            completed_rep=rep_result,
            view=sample.camera_view,
            alignment=sample.alignment,
        )
