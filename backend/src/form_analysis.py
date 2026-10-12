"""Form-analysis engine.

Converts a RepFeatures object into 0-100 category scores:
  - Knee Alignment
  - Knee Depth
  - Back Posture
  - Movement Control
  - Symmetry

...and an Overall Form Score (weighted average, see src.config.FORM_WEIGHTS).

These are PROTOTYPE movement-quality scores based on engineering
assumptions about "ideal" squat mechanics. They are NOT medically
validated metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from src.config import (
    FORM_WEIGHTS,
    IDEAL_MIN_KNEE_ANGLE,
    IDEAL_REP_DURATION,
    IDEAL_TORSO_ANGLE,
    MAX_HIP_SYMMETRY_DIFF,
    MAX_KNEE_SYMMETRY_DIFF,
    MAX_KNEE_VALGUS_SHIFT_LEG_RATIO,
)
from src.feature_engineering import RepFeatures
from src.geometry import alignment_status, alignment_summary, clamp


@dataclass
class FormScores:
    knee_alignment: float | None
    knee_depth: float | None
    back_posture: float | None
    movement_control: float | None
    symmetry: float | None
    overall: float | None

    def to_dict(self) -> Dict[str, float | None]:
        return {
            "knee_alignment": self.knee_alignment,
            "knee_depth": self.knee_depth,
            "back_posture": self.back_posture,
            "movement_control": self.movement_control,
            "symmetry": self.symmetry,
            "overall": self.overall,
        }


def analyze_alignment(features: RepFeatures) -> dict:
    """Recalculate a rep's alignment average from assessed metrics only."""
    alignment = features.alignment if isinstance(features.alignment, dict) else {}
    metrics = alignment.get("metrics", [])
    assessed_scores = [
        metric["score"] for metric in metrics
        if metric.get("assessed") and isinstance(metric.get("score"), (int, float))
    ]
    overall = round(sum(assessed_scores) / len(assessed_scores), 1) if assessed_scores else None
    return {
        **alignment_summary(metrics, features.camera_view),
        "overall": overall,
        "status": alignment_status(overall),
    }


def _score_knee_alignment(features: RepFeatures) -> float | None:
    """Penalize inward/outward knee shift (knee valgus proxy)."""
    if features.camera_view != "front":
        return None
    ratio = features.knee_alignment / MAX_KNEE_VALGUS_SHIFT_LEG_RATIO
    penalty = clamp(ratio, 0.0, 1.0) * 100.0
    return round(clamp(100.0 - penalty, 0.0, 100.0), 1)


def _score_knee_depth(features: RepFeatures) -> float | None:
    """Score how close the minimum knee angle is to the ideal squat-depth
    range. Too shallow (angle too high) or too deep is both penalized
    less harshly than being wildly off.
    """
    if features.camera_view not in {"side", "angled"}:
        return None
    low, high = IDEAL_MIN_KNEE_ANGLE
    angle = features.min_knee_angle
    if low <= angle <= high:
        return 100.0
    if angle > high:
        # too shallow (didn't squat deep enough)
        overshoot = angle - high
        penalty = clamp(overshoot / 40.0, 0.0, 1.0) * 100.0
    else:
        # deeper than the "ideal" range - mild penalty only
        overshoot = low - angle
        penalty = clamp(overshoot / 60.0, 0.0, 1.0) * 40.0
    return round(clamp(100.0 - penalty, 0.0, 100.0), 1)


def _score_back_posture(features: RepFeatures) -> float | None:
    """Score torso angle against the ideal upright range."""
    if features.camera_view != "side":
        return None
    low, high = IDEAL_TORSO_ANGLE
    angle = features.torso_angle
    if low <= angle <= high:
        return 100.0
    deviation = low - angle if angle < low else angle - high
    penalty = clamp(deviation / 45.0, 0.0, 1.0) * 100.0
    return round(clamp(100.0 - penalty, 0.0, 100.0), 1)


def _score_movement_control(features: RepFeatures) -> float:
    """Combine rep-duration sanity + smoothness into a control score."""
    low, high = IDEAL_REP_DURATION
    duration = features.rep_duration
    if low <= duration <= high:
        duration_score = 100.0
    else:
        deviation = low - duration if duration < low else duration - high
        duration_penalty = clamp(deviation / 3.0, 0.0, 1.0) * 100.0
        duration_score = clamp(100.0 - duration_penalty, 0.0, 100.0)

    smoothness_score = clamp(features.movement_smoothness, 0.0, 100.0)
    return round((duration_score * 0.5) + (smoothness_score * 0.5), 1)


def _score_symmetry(features: RepFeatures) -> float | None:
    if features.camera_view != "front":
        return None
    knee_ratio = features.knee_symmetry / MAX_KNEE_SYMMETRY_DIFF
    hip_ratio = features.hip_symmetry / MAX_HIP_SYMMETRY_DIFF
    penalty = clamp(max(knee_ratio, hip_ratio), 0.0, 1.0) * 100.0
    return round(clamp(100.0 - penalty, 0.0, 100.0), 1)


def analyze_form(features: RepFeatures) -> FormScores:
    """Compute all category scores + weighted overall score for a rep."""
    knee_alignment = _score_knee_alignment(features)
    knee_depth = _score_knee_depth(features)
    back_posture = _score_back_posture(features)
    movement_control = (
        _score_movement_control(features)
        if features.camera_view in {"front", "side"}
        else None
    )
    symmetry = _score_symmetry(features)

    assessed = [
        (score, FORM_WEIGHTS[name])
        for name, score in (
            ("knee_alignment", knee_alignment),
            ("knee_depth", knee_depth),
            ("back_posture", back_posture),
            ("movement_control", movement_control),
            ("symmetry", symmetry),
        )
        if score is not None
    ]
    overall = (
        round(sum(score * weight for score, weight in assessed) / sum(weight for _, weight in assessed), 1)
        if assessed
        else None
    )

    return FormScores(
        knee_alignment=knee_alignment,
        knee_depth=knee_depth,
        back_posture=back_posture,
        movement_control=movement_control,
        symmetry=symmetry,
        overall=overall,
    )
