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
from dataclasses import dataclass
from typing import Optional

from src.feature_engineering import RepFeatures, extract_rep_features
from src.feedback_engine import Feedback, generate_feedback
from src.form_analysis import FormScores, analyze_form
from src.geometry import calculate_angle, midpoint
from src.landmarks import BodyLandmarks
from src.risk_engine import RiskResult, RuleBasedRiskEngine, get_risk_engine
from src.squat_detector import CompletedRep, RepFrameSample, SquatDetector, SquatPhase


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


def _compute_torso_angle(landmarks: BodyLandmarks) -> Optional[float]:
    shoulder_mid = midpoint(landmarks.point("left_shoulder"), landmarks.point("right_shoulder"))
    hip_mid = landmarks.point("left_hip"), landmarks.point("right_hip")
    if None in hip_mid:
        return None
    hip_mid_point = midpoint(*hip_mid)
    vertical_reference = (hip_mid_point[0], hip_mid_point[1] - 100)
    angle_from_vertical = calculate_angle(shoulder_mid, hip_mid_point, vertical_reference)
    torso_angle = 90.0 - min(angle_from_vertical, 90.0)
    return round(max(0.0, torso_angle), 2)


def _compute_knee_alignment_shift(landmarks: BodyLandmarks) -> float:
    """Rough knee-valgus proxy: horizontal deviation of each knee from the
    straight line between hip and ankle, normalized by hip width.
    Returns the larger of the two sides (worst case).
    """
    shifts = []
    for side in ("left", "right"):
        hip = landmarks.point(f"{side}_hip")
        knee = landmarks.point(f"{side}_knee")
        ankle = landmarks.point(f"{side}_ankle")
        if not hip or not knee or not ankle:
            continue
        expected_x = (hip[0] + ankle[0]) / 2.0
        left_hip = landmarks.point("left_hip")
        right_hip = landmarks.point("right_hip")
        hip_width = abs(left_hip[0] - right_hip[0]) if left_hip and right_hip else 1.0
        hip_width = max(hip_width, 1e-3)
        shift = (knee[0] - expected_x) / hip_width
        shifts.append(shift)
    if not shifts:
        return 0.0
    return max(shifts, key=abs)


def build_frame_sample(landmarks: BodyLandmarks, timestamp: Optional[float] = None) -> Optional[RepFrameSample]:
    """Convert a single frame's BodyLandmarks into a RepFrameSample, or
    None if required landmarks are missing/low-confidence.
    """
    if not landmarks.has_all(REQUIRED_FOR_SQUAT):
        return None

    left_hip = landmarks.point("left_hip")
    right_hip = landmarks.point("right_hip")
    left_knee = landmarks.point("left_knee")
    right_knee = landmarks.point("right_knee")
    left_ankle = landmarks.point("left_ankle")
    right_ankle = landmarks.point("right_ankle")
    left_shoulder = landmarks.point("left_shoulder")
    right_shoulder = landmarks.point("right_shoulder")

    left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
    right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
    left_hip_angle = calculate_angle(left_shoulder, left_hip, left_knee)
    right_hip_angle = calculate_angle(right_shoulder, right_hip, right_knee)

    torso_angle = _compute_torso_angle(landmarks)
    if torso_angle is None:
        torso_angle = 90.0

    knee_alignment_shift = _compute_knee_alignment_shift(landmarks)
    hip_mid = midpoint(left_hip, right_hip)

    avg_knee_angle = (left_knee_angle + right_knee_angle) / 2.0

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
    )


class AnalysisPipeline:
    """Stateful per-session pipeline: owns the SquatDetector and the
    currently-selected risk engine, and produces a FrameStatus each time
    a frame is processed.
    """

    def __init__(self, risk_mode: str = "rule_based"):
        self.squat_detector = SquatDetector()
        self.risk_mode = risk_mode
        self._risk_engine = get_risk_engine(risk_mode)

    def set_risk_mode(self, mode: str) -> None:
        if mode != self.risk_mode:
            self.risk_mode = mode
            self._risk_engine = get_risk_engine(mode)

    def reset(self) -> None:
        self.squat_detector.reset()

    def process_frame(
        self,
        landmarks: Optional[BodyLandmarks],
        timestamp: Optional[float] = None,
    ) -> FrameStatus:
        if landmarks is None:
            return FrameStatus(
                landmarks_ok=False,
                phase=self.squat_detector.phase,
                rep_count=self.squat_detector.rep_count,
                message="No person detected in frame.",
            )

        sample = build_frame_sample(landmarks, timestamp)
        if sample is None:
            return FrameStatus(
                landmarks_ok=False,
                phase=self.squat_detector.phase,
                rep_count=self.squat_detector.rep_count,
                message="Landmarks visible but confidence too low for analysis.",
            )

        completed_rep = self.squat_detector.update(sample)
        rep_result: Optional[RepResult] = None

        if completed_rep is not None:
            features = extract_rep_features(completed_rep)
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
        )
