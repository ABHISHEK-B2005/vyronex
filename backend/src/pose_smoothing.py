from __future__ import annotations

from collections import deque
from statistics import median

from src.config import POSITION_MEDIAN_WINDOW, POSITION_SMOOTHING_ALPHA
from src.geometry import distance_3d
from src.landmarks import BodyLandmarks, LandmarkPoint


class PoseSmoother:
    """Median-filter landmarks, then apply a light exponential low-pass."""

    def __init__(self) -> None:
        self._history: dict[str, deque[tuple[float, float, float]]] = {}
        self._lowpass: dict[str, tuple[float, float, float]] = {}

    def reset(self) -> None:
        self._history.clear()
        self._lowpass.clear()

    def smooth(self, landmarks: BodyLandmarks | None) -> BodyLandmarks | None:
        if landmarks is None:
            self.reset()
            return None
        points = {}
        for name, point in landmarks.as_dict().items():
            history = self._history.setdefault(name, deque(maxlen=POSITION_MEDIAN_WINDOW))
            value = (point.x, point.y, point.z)
            history.append(value)
            filtered = tuple(median(axis) for axis in zip(*history))
            previous = self._lowpass.get(name, filtered)
            current = tuple(
                old + POSITION_SMOOTHING_ALPHA * (new - old)
                for old, new in zip(previous, filtered)
            )
            self._lowpass[name] = current
            points[name] = LandmarkPoint(
                x=current[0],
                y=current[1],
                visibility=point.visibility,
                z=current[2],
                image_x=point.image_x,
                image_y=point.image_y,
            )
        return BodyLandmarks(points, low_confidence=landmarks.low_confidence)


def landmark_visibility(landmarks: BodyLandmarks | None) -> float:
    if landmarks is None:
        return 0.0
    values = landmarks.as_dict().values()
    return sum(point.is_visible for point in values) / max(1, len(landmarks.as_dict()))


def pose_jump_ratio(previous: BodyLandmarks | None, current: BodyLandmarks | None) -> float:
    """Largest frame-to-frame joint jump after removing hip-centre motion."""
    if previous is None or current is None:
        return 0.0
    prev_hips = [previous.point3d(name) for name in ("left_hip", "right_hip")]
    curr_hips = [current.point3d(name) for name in ("left_hip", "right_hip")]
    if any(point is None for point in prev_hips + curr_hips):
        return 0.0
    prev_centre = tuple(sum(p[i] for p in prev_hips) / 2 for i in range(3))
    curr_centre = tuple(sum(p[i] for p in curr_hips) / 2 for i in range(3))
    shoulders = [current.point3d(name) for name in ("left_shoulder", "right_shoulder")]
    if any(point is None for point in shoulders):
        return 0.0
    torso_length = distance_3d(curr_centre, tuple(sum(p[i] for p in shoulders) / 2 for i in range(3)))
    if torso_length <= 1e-6:
        return 0.0
    largest = 0.0
    for name in ("left_shoulder", "right_shoulder", "left_hip", "right_hip",
                 "left_knee", "right_knee", "left_ankle", "right_ankle"):
        old, new = previous.point3d(name), current.point3d(name)
        if old is None or new is None:
            continue
        old_relative = tuple(a - b for a, b in zip(old, prev_centre))
        new_relative = tuple(a - b for a, b in zip(new, curr_centre))
        largest = max(largest, distance_3d(old_relative, new_relative) / torso_length)
    return largest
