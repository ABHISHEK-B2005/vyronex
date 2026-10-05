"""Pose-estimation wrapper around MediaPipe Pose.

Designed so that:
  * The (expensive) MediaPipe model is initialized ONCE and reused
    (never re-created per frame).
  * If MediaPipe is not installed / not available on the machine, the
    application degrades gracefully instead of crashing, and callers get
    a clear ``PoseEstimatorUnavailable`` signal instead of an exception
    mid-pipeline.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from src.config import (
    POSE_MIN_DETECTION_CONFIDENCE,
    POSE_MIN_TRACKING_CONFIDENCE,
)
from src.landmarks import BodyLandmarks

try:
    import mediapipe as mp

    MEDIAPIPE_AVAILABLE = True
except Exception:  # pragma: no cover - environment dependent
    MEDIAPIPE_AVAILABLE = False
    mp = None  # type: ignore


class PoseEstimatorUnavailable(RuntimeError):
    """Raised when no pose-estimation backend could be initialized."""


class PoseEstimator:
    """Thin, cached wrapper around MediaPipe's Pose solution.

    Usage:
        estimator = PoseEstimator()
        landmarks, annotated_frame, person_count = estimator.process(frame_bgr)
    """

    def __init__(
        self,
        min_detection_confidence: float = POSE_MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence: float = POSE_MIN_TRACKING_CONFIDENCE,
        model_complexity: int = 1,
    ) -> None:
        if not MEDIAPIPE_AVAILABLE:
            raise PoseEstimatorUnavailable(
                "MediaPipe is not installed or failed to import. "
                "Install it with `pip install mediapipe` to enable live "
                "pose detection, or use Demo Mode."
            )

        self._mp_pose = mp.solutions.pose
        self._mp_drawing = mp.solutions.drawing_utils
        self._mp_drawing_styles = mp.solutions.drawing_styles

        # Created ONCE and reused across frames (important for performance).
        self._pose = self._mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            smooth_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process(self, frame_bgr: np.ndarray) -> Tuple[Optional[BodyLandmarks], object, bool]:
        """Run pose estimation on a single BGR frame.

        Returns:
            (BodyLandmarks or None, raw_mediapipe_results, person_detected)
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return None, None, False

        frame_rgb = frame_bgr[:, :, ::-1]  # BGR -> RGB without extra copy cost
        frame_rgb.flags.writeable = False
        results = self._pose.process(frame_rgb)

        if not results.pose_landmarks:
            return None, results, False

        height, width = frame_bgr.shape[:2]
        landmarks = BodyLandmarks.from_mediapipe(results.pose_landmarks, width, height)
        return landmarks, results, True

    def draw_skeleton(self, frame_bgr: np.ndarray, results, color_bgr: Tuple[int, int, int]) -> np.ndarray:
        """Draw the pose skeleton onto a copy of the frame using a
        risk-dependent color (green/orange/red).
        """
        if results is None or not getattr(results, "pose_landmarks", None):
            return frame_bgr

        annotated = frame_bgr.copy()
        landmark_spec = self._mp_drawing.DrawingSpec(color=color_bgr, thickness=3, circle_radius=4)
        connection_spec = self._mp_drawing.DrawingSpec(color=color_bgr, thickness=2)

        self._mp_drawing.draw_landmarks(
            annotated,
            results.pose_landmarks,
            self._mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=landmark_spec,
            connection_drawing_spec=connection_spec,
        )
        return annotated

    def close(self) -> None:
        try:
            self._pose.close()
        except Exception:
            pass
