"""Reusable geometry / biomechanics math utilities.

All functions operate on simple (x, y) tuples (normalized or pixel
coordinates — callers should be consistent) so they are easy to unit test
without any dependency on MediaPipe or OpenCV.
"""

from __future__ import annotations

import math

from typing import Any, Sequence, Tuple

from src.config import (
    ALIGNMENT_GOOD_SCORE,
    ALIGNMENT_IDEAL_ANKLE_HIP_WIDTH_RATIO,
    ALIGNMENT_MAX_KNEE_DRIFT_RATIO,
    ALIGNMENT_MAX_KNEE_OVER_TOE_RATIO,
    ALIGNMENT_MAX_LEVEL_TILT_DEGREES,
    ALIGNMENT_MAX_NECK_TORSO_DEGREES,
    ALIGNMENT_IDEAL_BOTTOM_KNEE_ANGLE,
    ALIGNMENT_MAX_BOTTOM_KNEE_ANGLE_DEVIATION,
    ALIGNMENT_MAX_TORSO_LEAN_DEGREES,
    ALIGNMENT_MAX_WEIGHT_SHIFT_DEGREES,
    ALIGNMENT_MAX_ANKLE_HIP_WIDTH_DEVIATION,
    ALIGNMENT_WATCH_SCORE,
    SQUAT_DOWN_ANGLE,
    VIEW_FRONT_MIN_RATIO,
    VIEW_SIDE_MAX_RATIO,
)

Point = Tuple[float, float]
Point3D = Tuple[float, float, float]

_ALIGNMENT_METRICS = {
    "knee_alignment": ("Knee alignment", "% leg length", "knee"),
    "shoulder_level": ("Shoulder level", "degrees", "shoulder"),
    "hip_level": ("Hip level", "degrees", "hip"),
    "weight_shift": ("Left/right knee difference", "degrees", "hip"),
    "foot_ankle_alignment": ("Foot / ankle alignment", "ratio", "ankle"),
    "torso_lean": ("Torso lean", "degrees", "lower_back"),
    "neck_torso_alignment": ("Neck-torso alignment", "degrees", "lower_back"),
    "knee_over_toe": ("Knee over toe", "% leg length", "knee"),
    "hip_knee_ankle_stacking": ("Hip-knee-ankle stacking", "degrees", "hip"),
}


def alignment_status(score: float | None) -> str:
    if score is None:
        return "Not assessed"
    if score >= ALIGNMENT_GOOD_SCORE:
        return "Good"
    if score >= ALIGNMENT_WATCH_SCORE:
        return "Watch"
    return "Fix"


def _score_with_limit(value: float, limit: float) -> float:
    return round(max(0.0, min(100.0, 100.0 * (1.0 - abs(value) / limit))), 1)


def _level_tilt(left: Point3D, right: Point3D, lateral_width: float) -> float:
    vertical_delta = right[1] - left[1]
    return math.degrees(math.atan2(abs(vertical_delta), max(lateral_width, 1e-6)))


def _horizontal_distance(point_a: Point3D, point_b: Point3D) -> float:
    return math.hypot(point_a[0] - point_b[0], point_a[2] - point_b[2])


def _body_midpoint(left: Point3D, right: Point3D) -> Point3D:
    return tuple((a + b) / 2.0 for a, b in zip(left, right))


def _metric(key: str, value: float | None, score: float | None) -> dict[str, Any]:
    label, unit, body_area = _ALIGNMENT_METRICS[key]
    assessed = value is not None and score is not None and math.isfinite(value) and math.isfinite(score)
    return {
        "key": key,
        "label": label,
        "value": round(value, 2) if assessed else None,
        "unit": unit,
        "score": round(score, 1) if assessed else None,
        "status": alignment_status(score) if assessed else "Not assessed",
        "assessed": assessed,
        "body_area": body_area,
    }


def alignment_summary(metrics: list[dict[str, Any]], view: str) -> dict[str, Any]:
    assessed = [metric for metric in metrics if metric.get("assessed") and metric.get("score") is not None]
    overall = round(sum(metric["score"] for metric in assessed) / len(assessed), 1) if assessed else None
    cues = {
        "knee_alignment": "Push your knees out and track them over your feet.",
        "shoulder_level": "Keep your shoulders level.",
        "hip_level": "Keep your hips level.",
        "weight_shift": "Share your weight evenly between both legs.",
        "foot_ankle_alignment": "Keep your feet evenly spaced.",
        "torso_lean": "Keep your back tall and brace your core.",
        "neck_torso_alignment": "Keep your head in line with your torso.",
        "knee_over_toe": "Keep your knees moving in line with your toes.",
        "hip_knee_ankle_stacking": "Keep your hips, knees and ankles aligned.",
    }
    issues = [
        cues.get(metric["key"], f"Review {metric['label'].lower()}.")
        for metric in assessed
        if metric["status"] in {"Watch", "Fix"}
    ]
    return {
        "overall": overall,
        "status": alignment_status(overall),
        "metrics": metrics,
        "issues": issues,
        "view": view,
    }


def compute_alignment(landmarks, view: str | None = None) -> dict[str, Any]:
    """Compute only camera-view-supported alignment metrics from 3D points."""
    view = view or detect_view(landmarks)
    values: dict[str, tuple[float | None, float | None]] = {}
    side_names = (
        ("left_shoulder", "right_shoulder", "left_hip", "right_hip",
         "left_knee", "right_knee", "left_ankle", "right_ankle")
    )
    if view == "front" and landmarks.has_all(side_names):
        left_shoulder, right_shoulder = (landmarks.point3d(name) for name in side_names[:2])
        left_hip, right_hip = (landmarks.point3d(name) for name in side_names[2:4])
        shoulder_width = _horizontal_distance(left_shoulder, right_shoulder)
        hip_width = _horizontal_distance(left_hip, right_hip)
        knee_angles = []
        knee_shifts = []
        for side, inward_sign in (("left", 1.0), ("right", -1.0)):
            hip = landmarks.point3d(f"{side}_hip")
            knee = landmarks.point3d(f"{side}_knee")
            ankle = landmarks.point3d(f"{side}_ankle")
            leg_length = distance_3d(hip, ankle)
            if leg_length <= 1e-6:
                continue
            # Body lateral axis is the hip-to-hip vector in the horizontal plane.
            lateral = (right_hip[0] - left_hip[0], 0.0, right_hip[2] - left_hip[2])
            lateral_length = math.sqrt(lateral[0] ** 2 + lateral[2] ** 2)
            if lateral_length <= 1e-6:
                continue
            lateral = (lateral[0] / lateral_length, 0.0, lateral[2] / lateral_length)
            leg = tuple(a - h for a, h in zip(ankle, hip))
            t = max(0.0, min(1.0, sum((k - h) * d for k, h, d in zip(knee, hip, leg)) / (leg_length ** 2)))
            expected = tuple(h + t * d for h, d in zip(hip, leg))
            deviation = tuple(k - e for k, e in zip(knee, expected))
            inward = sum(a * b for a, b in zip(deviation, lateral)) * inward_sign / leg_length
            knee_shifts.append(max(0.0, inward))
            knee_angles.append(calculate_angle_3d(hip, knee, ankle))
        knee_drift = max(knee_shifts) if knee_shifts else None
        shoulder_tilt = _level_tilt(left_shoulder, right_shoulder, shoulder_width)
        hip_tilt = _level_tilt(left_hip, right_hip, hip_width)
        angle_diff = abs(knee_angles[0] - knee_angles[1]) if len(knee_angles) == 2 else None
        ankle_width = _horizontal_distance(
            landmarks.point3d("left_ankle"), landmarks.point3d("right_ankle")
        )
        ankle_ratio = ankle_width / hip_width if hip_width > 1e-6 else None
        values.update({
            "knee_alignment": (knee_drift * 100.0, _score_with_limit(knee_drift, ALIGNMENT_MAX_KNEE_DRIFT_RATIO)) if knee_drift is not None and knee_drift <= 0.25 else (None, None),
            "shoulder_level": (shoulder_tilt, _score_with_limit(shoulder_tilt, ALIGNMENT_MAX_LEVEL_TILT_DEGREES)),
            "hip_level": (hip_tilt, _score_with_limit(hip_tilt, ALIGNMENT_MAX_LEVEL_TILT_DEGREES)),
            "weight_shift": (angle_diff, _score_with_limit(angle_diff, ALIGNMENT_MAX_WEIGHT_SHIFT_DEGREES)) if angle_diff is not None else (None, None),
            "foot_ankle_alignment": (
                (ankle_ratio, _score_with_limit(ankle_ratio - ALIGNMENT_IDEAL_ANKLE_HIP_WIDTH_RATIO, ALIGNMENT_MAX_ANKLE_HIP_WIDTH_DEVIATION))
                if ankle_ratio is not None else (None, None)
            ),
        })
        selected_keys = ("knee_alignment", "shoulder_level", "hip_level", "weight_shift", "foot_ankle_alignment")
    elif view == "side":
        shoulder_mid = _body_midpoint(
            landmarks.point3d("left_shoulder"), landmarks.point3d("right_shoulder")
        ) if landmarks.has_all(("left_shoulder", "right_shoulder")) else None
        hip_mid = _body_midpoint(
            landmarks.point3d("left_hip"), landmarks.point3d("right_hip")
        ) if landmarks.has_all(("left_hip", "right_hip")) else None
        lean = None
        neck_angle = None
        if shoulder_mid and hip_mid:
            torso = tuple(s - h for s, h in zip(shoulder_mid, hip_mid))
            lean = math.degrees(math.atan2(math.hypot(torso[0], torso[2]), max(abs(torso[1]), 1e-6)))
            if landmarks.has_all(("nose",)):
                nose = landmarks.point3d("nose")
                neck = tuple(n - s for n, s in zip(nose, shoulder_mid))
                neck_angle = math.degrees(math.atan2(math.hypot(neck[0], neck[2]), max(abs(neck[1]), 1e-6)))
        values["torso_lean"] = (lean, _score_with_limit(lean, ALIGNMENT_MAX_TORSO_LEAN_DEGREES)) if lean is not None else (None, None)
        values["neck_torso_alignment"] = (neck_angle, _score_with_limit(neck_angle, ALIGNMENT_MAX_NECK_TORSO_DEGREES)) if neck_angle is not None else (None, None)
        toe_offsets = []
        knee_angles = []
        for side in ("left", "right"):
            base_names = (f"{side}_knee", f"{side}_ankle", f"{side}_hip")
            if landmarks.has_all(base_names):
                knee = landmarks.point3d(f"{side}_knee")
                ankle = landmarks.point3d(f"{side}_ankle")
                hip = landmarks.point3d(f"{side}_hip")
                angle = calculate_angle_3d(hip, knee, ankle)
                knee_angles.append(angle)
                toe_name = f"{side}_foot_index"
                if landmarks.has_all((*base_names[:2], toe_name)):
                    toe = landmarks.point3d(toe_name)
                    foot_axis = tuple(t - a for t, a in zip(toe, ankle))
                    foot_length = distance_3d(toe, ankle)
                    leg_length = distance_3d(hip, ankle)
                    if foot_length > 1e-6 and leg_length > 1e-6:
                        projection = sum((k - a) * d for k, a, d in zip(knee, ankle, foot_axis)) / foot_length
                        toe_offsets.append(max(0.0, (projection - foot_length) / leg_length))
        knee_to_toe = max(toe_offsets) if toe_offsets else None
        bottom_knee_angle = sum(knee_angles) / len(knee_angles) if knee_angles else None
        values["knee_over_toe"] = (
            (knee_to_toe * 100.0, _score_with_limit(knee_to_toe, ALIGNMENT_MAX_KNEE_OVER_TOE_RATIO))
            if knee_to_toe is not None and knee_to_toe <= 0.25
            else (None, None)
        )
        values["hip_knee_ankle_stacking"] = (
            (bottom_knee_angle, _score_with_limit(
                bottom_knee_angle - ALIGNMENT_IDEAL_BOTTOM_KNEE_ANGLE,
                ALIGNMENT_MAX_BOTTOM_KNEE_ANGLE_DEVIATION,
            ))
            if bottom_knee_angle is not None
            and bottom_knee_angle <= SQUAT_DOWN_ANGLE
            and 20 <= bottom_knee_angle <= 180
            else (None, None)
        )
        selected_keys = ("torso_lean", "neck_torso_alignment", "knee_over_toe", "hip_knee_ankle_stacking")
    else:
        selected_keys = ()
    metrics = [
        _metric(key, *(values.get(key, (None, None)) if key in selected_keys else (None, None)))
        for key in _ALIGNMENT_METRICS
    ]
    return alignment_summary(metrics, view)


def aggregate_alignment(samples: list[dict[str, Any]], view: str) -> dict[str, Any]:
    """Keep each rep's worst assessed per-metric score without scoring missing data."""
    metrics = []
    for key, (label, unit, body_area) in _ALIGNMENT_METRICS.items():
        candidates = [
            metric for sample in samples for metric in sample.get("metrics", [])
            if metric.get("key") == key and metric.get("assessed")
        ]
        if key == "hip_knee_ankle_stacking" and candidates:
            chosen = min(candidates, key=lambda metric: metric["value"])
        else:
            chosen = min(candidates, key=lambda metric: metric["score"]) if candidates else None
        metrics.append({
            "key": key,
            "label": label,
            "value": chosen.get("value") if chosen else None,
            "unit": unit,
            "score": chosen.get("score") if chosen else None,
            "status": chosen.get("status", "Not assessed") if chosen else "Not assessed",
            "assessed": chosen is not None,
            "body_area": body_area,
        })
    return alignment_summary(metrics, view)


def distance_3d(point_a: Point3D, point_b: Point3D) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(point_a, point_b)))


def detect_view(landmarks) -> str:
    """Estimate camera view from shoulder/hip width relative to torso length."""
    ratio = view_width_ratio(landmarks)
    if ratio is None:
        return "angled"
    if ratio > VIEW_FRONT_MIN_RATIO:
        return "front"
    if ratio < VIEW_SIDE_MAX_RATIO:
        return "side"
    return "angled"


def view_width_ratio(landmarks) -> float | None:
    """Return mean shoulder/hip width divided by torso length."""
    points = {
        name: landmarks.point3d(name)
        for name in ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
    }
    if any(point is None for point in points.values()):
        return None
    shoulder_mid = tuple((a + b) / 2 for a, b in zip(points["left_shoulder"], points["right_shoulder"]))
    hip_mid = tuple((a + b) / 2 for a, b in zip(points["left_hip"], points["right_hip"]))
    torso_length = distance_3d(shoulder_mid, hip_mid)
    if torso_length <= 1e-6:
        return None
    shoulder_width = distance_3d(points["left_shoulder"], points["right_shoulder"])
    hip_width = distance_3d(points["left_hip"], points["right_hip"])
    return ((shoulder_width + hip_width) / 2) / torso_length


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


def calculate_angle_3d(
    point_a: tuple[float, float, float],
    point_b: tuple[float, float, float],
    point_c: tuple[float, float, float],
) -> float:
    """Return the joint angle between three points in 3D space."""
    ba = tuple(a - b for a, b in zip(point_a, point_b))
    bc = tuple(c - b for c, b in zip(point_c, point_b))
    ba_norm = math.sqrt(sum(value * value for value in ba))
    bc_norm = math.sqrt(sum(value * value for value in bc))
    if ba_norm == 0 or bc_norm == 0:
        return 0.0
    cosine = sum(a * b for a, b in zip(ba, bc)) / (ba_norm * bc_norm)
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


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
