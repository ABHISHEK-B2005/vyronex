"""Reusable geometry / biomechanics math utilities.

All functions operate on simple (x, y) tuples (normalized or pixel
coordinates — callers should be consistent) so they are easy to unit test
without any dependency on MediaPipe or OpenCV.
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple

Point = Tuple[float, float]


def calculate_angle(point_a: Point, point_b: Point, point_c: Point) -> float:
    """Return the angle (in degrees) at vertex ``point_b`` formed by the
    rays ``point_b -> point_a`` and ``point_b -> point_c``.

    This is the standard joint-angle calculation used throughout the
    biomechanics pipeline (e.g. knee angle = angle at knee formed by
    hip-knee-ankle).
    """
    ax, ay = point_a
    bx, by = point_b
    cx, cy = point_c

    ba = (ax - bx, ay - by)
    bc = (cx - bx, cy - by)

    ba_norm = math.hypot(*ba)
    bc_norm = math.hypot(*bc)

    if ba_norm == 0 or bc_norm == 0:
        return 0.0

    cos_angle = (ba[0] * bc[0] + ba[1] * bc[1]) / (ba_norm * bc_norm)
    cos_angle = max(-1.0, min(1.0, cos_angle))  # clamp for float safety
    angle_rad = math.acos(cos_angle)
    return math.degrees(angle_rad)


def distance(point_a: Point, point_b: Point) -> float:
    """Euclidean distance between two points."""
    return math.hypot(point_a[0] - point_b[0], point_a[1] - point_b[1])


def midpoint(point_a: Point, point_b: Point) -> Point:
    """Midpoint between two points."""
    return ((point_a[0] + point_b[0]) / 2.0, (point_a[1] + point_b[1]) / 2.0)


def normalize_point(point: Point, frame_width: int, frame_height: int) -> Point:
    """Convert pixel coordinates to normalized (0-1) coordinates."""
    if frame_width == 0 or frame_height == 0:
        return (0.0, 0.0)
    return (point[0] / frame_width, point[1] / frame_height)


def denormalize_point(point: Point, frame_width: int, frame_height: int) -> Point:
    """Convert normalized (0-1) coordinates to pixel coordinates."""
    return (point[0] * frame_width, point[1] * frame_height)


def symmetry_difference(value_left: float, value_right: float) -> float:
    """Absolute difference between a left-side and right-side measurement
    (e.g. left knee angle vs right knee angle). Used as a simple symmetry
    indicator — smaller is more symmetric.
    """
    return abs(value_left - value_right)


def movement_velocity(
    point_prev: Point, point_curr: Point, dt_seconds: float
) -> float:
    """Frame-to-frame velocity of a landmark, in distance-units / second.

    ``dt_seconds`` must be > 0; if it is not, returns 0.0 to avoid
    division-by-zero blowing up the pipeline.
    """
    if dt_seconds <= 0:
        return 0.0
    return distance(point_prev, point_curr) / dt_seconds


def frame_displacement(point_prev: Point, point_curr: Point) -> float:
    """Raw frame-to-frame displacement (no time normalization)."""
    return distance(point_prev, point_curr)


def average_point(points: Sequence[Point]) -> Point:
    """Average (centroid) of a sequence of points. Useful for smoothing
    or estimating a body-center reference point.
    """
    if not points:
        return (0.0, 0.0)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def clamp(value: float, low: float, high: float) -> float:
    """Clamp ``value`` into the inclusive range [low, high]."""
    return max(low, min(high, value))
