"""Explainable feedback engine.

Turns a RiskResult + FormScores + RepFeatures into a human-readable
"why" explanation and a concrete correction — always derived from the
actual detected feature that caused the warning, never a random pick.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from src.feature_engineering import RepFeatures
from src.form_analysis import FormScores
from src.risk_engine import RiskResult

# Category -> (reason template, recommendation) — filled in with real
# measured values at generation time so feedback is always grounded in
# the actual detected feature.
_CATEGORY_EXPLANATIONS = {
    "knee_alignment": {
        "reason": "Knee alignment changed significantly during the lowering phase (inward/outward shift of {value:.1%}).",
        "recommendation": "Keep the knee tracking in line with the foot and avoid letting it collapse inward.",
        "warning_tag": "KNEE ALIGNMENT WARNING",
    },
    "knee_depth": {
        "reason": "Squat depth (minimum knee angle {value:.0f}°) was outside the comfortable target range.",
        "recommendation": "Use a controlled, comfortable range of motion — avoid squatting too shallow or excessively deep.",
        "warning_tag": "SQUAT DEPTH WARNING",
    },
    "back_posture": {
        "reason": "Torso angle averaged {value:.0f}°, indicating excessive forward lean.",
        "recommendation": "Maintain a more neutral, upright torso position throughout the movement.",
        "warning_tag": "BACK POSTURE WARNING",
    },
    "movement_control": {
        "reason": "Movement speed/smoothness during the repetition ({value:.0f}°/s) suggests reduced control.",
        "recommendation": "Reduce movement speed and control the descent phase.",
        "warning_tag": "MOVEMENT CONTROL WARNING",
    },
    "symmetry": {
        "reason": "Left/right knee-angle difference of {value:.1f}° indicates asymmetric movement.",
        "recommendation": "Focus on even weight distribution and symmetry between both legs.",
        "warning_tag": "SYMMETRY WARNING",
    },
}

_FEATURE_LOOKUP = {
    "knee_alignment": lambda f: f.knee_alignment,
    "knee_depth": lambda f: f.min_knee_angle,
    "back_posture": lambda f: f.torso_angle,
    "movement_control": lambda f: f.movement_speed,
    "symmetry": lambda f: f.knee_symmetry,
}


@dataclass
class Feedback:
    risk_level: str
    risk_label: str
    warning_tag: Optional[str]
    reason: str
    recommendation: str
    all_issues: List[str]


def generate_feedback(
    risk_result: RiskResult, form_scores: FormScores, features: RepFeatures
) -> Feedback:
    """Generate an explanation grounded in the actual primary_issue /
    contributing factors computed by the risk engine.
    """
    if risk_result.risk_level == "LOW" or not risk_result.primary_issue:
        return Feedback(
            risk_level=risk_result.risk_level,
            risk_label=risk_result.risk_label,
            warning_tag=None,
            reason="No significant movement-risk indicators detected in this repetition.",
            recommendation="Good form — maintain this technique.",
            all_issues=[],
        )

    primary = risk_result.primary_issue
    template = _CATEGORY_EXPLANATIONS.get(primary)
    if template is None:
        return Feedback(
            risk_level=risk_result.risk_level,
            risk_label=risk_result.risk_label,
            warning_tag=None,
            reason="Elevated movement-risk indicators detected.",
            recommendation="Focus on controlled, symmetric movement with a neutral spine.",
            all_issues=[],
        )

    value = _FEATURE_LOOKUP[primary](features)
    reason = template["reason"].format(value=value)
    recommendation = template["recommendation"]
    warning_tag = template["warning_tag"]

    # Collect all meaningfully-contributing issues (not just the primary one)
    sorted_factors = sorted(
        risk_result.contributing_factors.items(),
        key=lambda kv: kv[1] if isinstance(kv[1], (int, float)) else 0,
        reverse=True,
    )
    all_issues = [
        _CATEGORY_EXPLANATIONS[cat]["warning_tag"]
        for cat, contribution in sorted_factors
        if isinstance(contribution, (int, float))
        and contribution >= 2.0
        and cat in _CATEGORY_EXPLANATIONS
    ]

    return Feedback(
        risk_level=risk_result.risk_level,
        risk_label=risk_result.risk_label,
        warning_tag=warning_tag,
        reason=reason,
        recommendation=recommendation,
        all_issues=all_issues,
    )
