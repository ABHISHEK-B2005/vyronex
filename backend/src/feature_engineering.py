"""Biomechanical feature extraction.

Converts a ``CompletedRep`` (a sequence of raw per-frame measurements)
into a single feature dictionary summarizing that repetition, which is
then consumed by the form-analysis and risk engines (and can also be
logged to CSV for ML training).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List

from src.geometry import symmetry_difference
from src.squat_detector import CompletedRep


@dataclass
class RepFeatures:
    rep_number: int
    min_knee_angle: float
    max_knee_angle: float
    avg_knee_angle: float
    knee_symmetry: float          # abs(left-right) knee angle diff, avg over rep
    hip_angle: float              # min hip angle reached (depth indicator)
    hip_symmetry: float
    torso_angle: float            # average torso angle during descent/bottom
    forward_lean: float           # how far torso deviates from upright (90deg)
    knee_alignment: float         # max inward knee shift observed (0=perfect,1=bad)
    rep_duration: float
    movement_speed: float         # degrees/second of knee angle change (avg)
    movement_smoothness: float    # lower variance in velocity = smoother

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def extract_rep_features(rep: CompletedRep) -> RepFeatures:
    """Compute a RepFeatures summary from a CompletedRep's raw samples.

    Designed to be robust to short reps (few samples) by falling back to
    sane defaults rather than raising.
    """
    samples = rep.samples
    if not samples:
        return RepFeatures(
            rep_number=rep.rep_number,
            min_knee_angle=180.0,
            max_knee_angle=180.0,
            avg_knee_angle=180.0,
            knee_symmetry=0.0,
            hip_angle=180.0,
            hip_symmetry=0.0,
            torso_angle=90.0,
            forward_lean=0.0,
            knee_alignment=0.0,
            rep_duration=0.0,
            movement_speed=0.0,
            movement_smoothness=100.0,
        )

    knee_angles = [s.avg_knee_angle for s in samples]
    left_knee = [s.left_knee_angle for s in samples]
    right_knee = [s.right_knee_angle for s in samples]
    left_hip = [s.left_hip_angle for s in samples]
    right_hip = [s.right_hip_angle for s in samples]
    torso_angles = [s.torso_angle for s in samples]
    alignment_shifts = [abs(s.knee_alignment_shift) for s in samples]

    knee_sym_values = [
        symmetry_difference(l, r) for l, r in zip(left_knee, right_knee)
    ]
    hip_sym_values = [
        symmetry_difference(l, r) for l, r in zip(left_hip, right_hip)
    ]

    min_knee = min(knee_angles)
    max_knee = max(knee_angles)
    avg_knee = sum(knee_angles) / len(knee_angles)
    hip_angle_min = min((l + r) / 2.0 for l, r in zip(left_hip, right_hip))
    avg_torso = sum(torso_angles) / len(torso_angles)
    forward_lean = max(0.0, 90.0 - avg_torso)
    max_alignment_shift = max(alignment_shifts) if alignment_shifts else 0.0

    duration = rep.duration_seconds
    knee_rom = max_knee - min_knee  # range of motion covered
    movement_speed = (knee_rom / duration) if duration > 0 else 0.0

    # Smoothness: variance of frame-to-frame angle deltas (lower = smoother)
    deltas = [
        abs(knee_angles[i] - knee_angles[i - 1]) for i in range(1, len(knee_angles))
    ]
    if deltas:
        mean_delta = sum(deltas) / len(deltas)
        variance = sum((d - mean_delta) ** 2 for d in deltas) / len(deltas)
    else:
        variance = 0.0
    smoothness_score = max(0.0, 100.0 - variance * 10.0)  # heuristic scale

    return RepFeatures(
        rep_number=rep.rep_number,
        min_knee_angle=round(min_knee, 2),
        max_knee_angle=round(max_knee, 2),
        avg_knee_angle=round(avg_knee, 2),
        knee_symmetry=round(sum(knee_sym_values) / len(knee_sym_values), 2) if knee_sym_values else 0.0,
        hip_angle=round(hip_angle_min, 2),
        hip_symmetry=round(sum(hip_sym_values) / len(hip_sym_values), 2) if hip_sym_values else 0.0,
        torso_angle=round(avg_torso, 2),
        forward_lean=round(forward_lean, 2),
        knee_alignment=round(max_alignment_shift, 4),
        rep_duration=round(duration, 2),
        movement_speed=round(movement_speed, 2),
        movement_smoothness=round(smoothness_score, 2),
    )
