"""Body-landmark abstraction layer.

Wraps MediaPipe Pose landmarks (or any 33-point pose output) into a safe,
typed structure so the rest of the pipeline never has to deal with raw
MediaPipe enums or worry about missing/low-confidence points directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from src.config import MIN_LANDMARK_VISIBILITY

Point = Tuple[float, float]

# Names we care about for squat analysis (subset of MediaPipe's 33 points)
REQUIRED_LANDMARK_NAMES = [
    "nose",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

# MediaPipe Pose landmark indices for the names above
MEDIAPIPE_INDEX_MAP = {
    "nose": 0,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
    "left_foot_index": 31,
    "right_foot_index": 32,
}


@dataclass
class LandmarkPoint:
    x: float
    y: float
    visibility: float
    z: float = 0.0
    image_x: float | None = None
    image_y: float | None = None

    @property
    def as_tuple(self) -> Point:
        return (self.x, self.y)

    @property
    def is_visible(self) -> bool:
        return self.visibility >= MIN_LANDMARK_VISIBILITY


class BodyLandmarks:
    """Safe container for a single frame's detected body landmarks."""

    def __init__(self, points: Dict[str, LandmarkPoint], low_confidence: bool = False):
        self._points = points
        self.low_confidence = low_confidence

    @classmethod
    def from_mediapipe(
        cls,
        pose_landmarks,
        frame_width: int,
        frame_height: int,
        world_landmarks=None,
    ) -> "BodyLandmarks":
        """Prefer hip-centered world coordinates; use image coordinates as fallback."""
        points: Dict[str, LandmarkPoint] = {}
        image_points = getattr(pose_landmarks, "landmark", pose_landmarks)
        world_points = getattr(world_landmarks, "landmark", world_landmarks)
        low_confidence = world_points is None
        for name, idx in MEDIAPIPE_INDEX_MAP.items():
            try:
                image_lm = image_points[idx]
                lm = world_points[idx] if world_points is not None else image_lm
                points[name] = LandmarkPoint(
                    x=float(lm.x) if world_points is not None else float(lm.x) * frame_width,
                    y=float(lm.y) if world_points is not None else float(lm.y) * frame_height,
                    visibility=float(getattr(image_lm, "visibility", getattr(lm, "visibility", 1.0))),
                    z=float(getattr(lm, "z", 0.0)) if world_points is not None else 0.0,
                    image_x=float(image_lm.x) * frame_width,
                    image_y=float(image_lm.y) * frame_height,
                )
            except (IndexError, AttributeError):
                # Missing landmark -> mark as invisible instead of crashing
                points[name] = LandmarkPoint(x=0.0, y=0.0, visibility=0.0)
        return cls(points, low_confidence=low_confidence)

    def get(self, name: str) -> Optional[LandmarkPoint]:
        return self._points.get(name)

    def point(self, name: str) -> Optional[Point]:
        """Return (x, y) for a landmark, or None if missing/low-confidence."""
        lm = self._points.get(name)
        if lm is None or not lm.is_visible:
            return None
        return lm.as_tuple

    def point3d(self, name: str) -> Optional[Tuple[float, float, float]]:
        lm = self._points.get(name)
        if lm is None or not lm.is_visible:
            return None
        return (lm.x, lm.y, lm.z)

    def numeric_dict(self) -> Dict[str, Dict[str, float]]:
        return {
            name: {
                "x": float(point.x),
                "y": float(point.y),
                "z": float(point.z),
                "visibility": float(point.visibility),
            }
            for name, point in self._points.items()
        }

    def has_all(self, names) -> bool:
        """True only if every requested landmark is present & confident."""
        return all(self.point(n) is not None for n in names)

    def visibility_ratio(self) -> float:
        """Fraction of required landmarks that are currently visible.
        Used to decide whether a frame is usable at all.
        """
        if not self._points:
            return 0.0
        visible = sum(1 for lm in self._points.values() if lm.is_visible)
        return visible / len(self._points)

    def as_dict(self) -> Dict[str, LandmarkPoint]:
        return dict(self._points)
