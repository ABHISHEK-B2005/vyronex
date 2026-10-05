"""Demo Mode: replay a prepared movement sequence without needing a live
webcam. Used for pitch demonstrations when the camera/environment may be
imperfect (see project spec section 26/27).

Generates a synthetic sequence of BodyLandmarks representing:
  GOOD MOVEMENT -> RISK DETECTION -> WARNING -> CORRECTION

All data here is SYNTHETIC / SAMPLE DATA for demonstration and must be
labeled as such in the UI.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

from src.landmarks import BodyLandmarks, LandmarkPoint


@dataclass
class DemoStep:
    label: str
    landmarks: BodyLandmarks


def _knee_and_hip_for_angle(knee_xy, angle_target_deg: float, thigh_len: float = 100.0):
    theta = math.radians(180.0 - angle_target_deg)
    dx = thigh_len * math.sin(theta)
    dy = -thigh_len * math.cos(theta)
    return (knee_xy[0] + dx, knee_xy[1] + dy)


def _build_pose(
    knee_angle_target: float,
    torso_lean_deg: float = 5.0,
    knee_inward_shift: float = 0.0,
) -> BodyLandmarks:
    """Build one synthetic pose frame.

    knee_angle_target: knee bend in degrees (170 = standing, ~85 = deep squat)
    torso_lean_deg: forward lean of the torso in degrees (0 = perfectly upright)
    knee_inward_shift: fraction of hip-width the knees shift inward (valgus proxy)
    """
    left_knee = (290.0, 400.0)
    right_knee = (350.0, 400.0)
    left_ankle = (290.0, 480.0)
    right_ankle = (350.0, 480.0)

    hip_width = abs(right_knee[0] - left_knee[0])
    shift_px = knee_inward_shift * hip_width
    left_knee = (left_knee[0] + shift_px, left_knee[1])
    right_knee = (right_knee[0] - shift_px, right_knee[1])

    left_hip = _knee_and_hip_for_angle(left_knee, knee_angle_target)
    right_hip = _knee_and_hip_for_angle(right_knee, knee_angle_target)

    # Apply forward lean by shifting shoulders forward (x) relative to hips
    lean_shift = math.tan(math.radians(torso_lean_deg)) * 150.0
    shoulder_y = min(left_hip[1], right_hip[1]) - 150.0

    points = {
        "nose": LandmarkPoint(320 + lean_shift, shoulder_y - 60, 0.98),
        "left_shoulder": LandmarkPoint(280 + lean_shift, shoulder_y, 0.98),
        "right_shoulder": LandmarkPoint(360 + lean_shift, shoulder_y, 0.98),
        "left_elbow": LandmarkPoint(270 + lean_shift, shoulder_y + 60, 0.9),
        "right_elbow": LandmarkPoint(370 + lean_shift, shoulder_y + 60, 0.9),
        "left_wrist": LandmarkPoint(265 + lean_shift, shoulder_y + 120, 0.9),
        "right_wrist": LandmarkPoint(375 + lean_shift, shoulder_y + 120, 0.9),
        "left_hip": LandmarkPoint(left_hip[0], left_hip[1], 0.98),
        "right_hip": LandmarkPoint(right_hip[0], right_hip[1], 0.98),
        "left_knee": LandmarkPoint(left_knee[0], left_knee[1], 0.98),
        "right_knee": LandmarkPoint(right_knee[0], right_knee[1], 0.98),
        "left_ankle": LandmarkPoint(left_ankle[0], left_ankle[1], 0.98),
        "right_ankle": LandmarkPoint(right_ankle[0], right_ankle[1], 0.98),
    }
    return BodyLandmarks(points)


def _squat_cycle(
    torso_lean_deg: float, knee_inward_shift: float, frames_per_phase: int = 5
) -> List[BodyLandmarks]:
    """One full standing->bottom->standing cycle at the given form quality."""
    angles = (
        [170] * 2
        + list(range(170, 85, -8))
        + [85] * 3
        + list(range(85, 170, 8))
        + [170] * 2
    )
    frames = []
    for angle in angles:
        for _ in range(max(1, frames_per_phase // 3)):
            frames.append(_build_pose(angle, torso_lean_deg, knee_inward_shift))
    return frames


def build_demo_sequence() -> List[DemoStep]:
    """Build the full pitch-demo sequence:
        GOOD MOVEMENT -> RISK DETECTION -> WARNING -> CORRECTION
    as described in the project spec (Pitch Demonstration Flow).
    """
    steps: List[DemoStep] = []

    # STEP 3: correct squat (low risk) x2 reps
    for frame in _squat_cycle(torso_lean_deg=3.0, knee_inward_shift=0.0) * 2:
        steps.append(DemoStep(label="GOOD_MOVEMENT", landmarks=frame))

    # STEP 4: intentionally poor movement (high risk) x2 reps
    for frame in _squat_cycle(torso_lean_deg=35.0, knee_inward_shift=0.12) * 2:
        steps.append(DemoStep(label="POOR_MOVEMENT", landmarks=frame))

    # STEP 7: corrected movement (back to low/moderate risk) x2 reps
    for frame in _squat_cycle(torso_lean_deg=8.0, knee_inward_shift=0.02) * 2:
        steps.append(DemoStep(label="CORRECTED_MOVEMENT", landmarks=frame))

    return steps
