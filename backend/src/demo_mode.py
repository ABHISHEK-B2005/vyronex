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
    hip_width = 0.34
    shift = knee_inward_shift * hip_width
    knee_bend = 0.45 * math.tan(math.radians((180.0 - knee_angle_target) / 2.0))
    left_knee = (-hip_width / 2 + shift, 0.45, knee_bend)
    right_knee = (hip_width / 2 - shift, 0.45, knee_bend)
    left_ankle = (-hip_width / 2, 0.90, 0.0)
    right_ankle = (hip_width / 2, 0.90, 0.0)
    left_hip = (-hip_width / 2, 0.0, 0.0)
    right_hip = (hip_width / 2, 0.0, 0.0)

    shoulder_width = 0.42
    lean_shift = math.tan(math.radians(torso_lean_deg)) * 0.5
    shoulder_y = -0.50

    points = {
        "nose": LandmarkPoint(0.0, -0.78, 0.98, lean_shift),
        "left_shoulder": LandmarkPoint(-shoulder_width / 2, shoulder_y, 0.98, lean_shift),
        "right_shoulder": LandmarkPoint(shoulder_width / 2, shoulder_y, 0.98, lean_shift),
        "left_elbow": LandmarkPoint(-shoulder_width / 2, -0.30, 0.9, lean_shift),
        "right_elbow": LandmarkPoint(shoulder_width / 2, -0.30, 0.9, lean_shift),
        "left_wrist": LandmarkPoint(-shoulder_width / 2, -0.10, 0.9, lean_shift),
        "right_wrist": LandmarkPoint(shoulder_width / 2, -0.10, 0.9, lean_shift),
        "left_hip": LandmarkPoint(*left_hip[:2], 0.98, left_hip[2]),
        "right_hip": LandmarkPoint(*right_hip[:2], 0.98, right_hip[2]),
        "left_knee": LandmarkPoint(*left_knee[:2], 0.98, left_knee[2]),
        "right_knee": LandmarkPoint(*right_knee[:2], 0.98, right_knee[2]),
        "left_ankle": LandmarkPoint(*left_ankle[:2], 0.98, left_ankle[2]),
        "right_ankle": LandmarkPoint(*right_ankle[:2], 0.98, right_ankle[2]),
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
