"""Base classes and utility functions for athletics event biomechanical analysis."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.geometry import calculate_angle, midpoint
from src.landmarks import BodyLandmarks


@dataclass
class MetricEvaluation:
    name: str
    phase: str
    measured_value: float
    reference_mean: float
    reference_std: float
    z_score: float
    unit: str
    explanation: str
    is_injury_risk: bool = False


@dataclass
class AttemptResult:
    event: str
    attempt_number: int
    overall_technique_score: float  # 0 to 100 (100 = perfect alignment with reference)
    risk_level: str  # LOW, MODERATE, HIGH
    metric_evaluations: List[MetricEvaluation] = field(default_factory=list)
    injury_warnings: List[str] = field(default_factory=list)
    explanations: List[str] = field(default_factory=list)
    phases_detected: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event": self.event,
            "attempt_number": self.attempt_number,
            "overall_technique_score": round(self.overall_technique_score, 1),
            "risk_level": self.risk_level,
            "injury_warnings": self.injury_warnings,
            "explanations": self.explanations,
            "phases_detected": self.phases_detected,
            "metrics": [
                {
                    "name": m.name,
                    "phase": m.phase,
                    "measured": round(m.measured_value, 2),
                    "reference_mean": m.reference_mean,
                    "reference_std": m.reference_std,
                    "z_score": round(m.z_score, 2),
                    "unit": m.unit,
                    "explanation": m.explanation,
                    "is_injury_risk": m.is_injury_risk,
                }
                for m in self.metric_evaluations
            ],
        }


def evaluate_z_score(
    name: str,
    phase: str,
    measured: float,
    mean: float,
    std: float,
    unit: str = "deg",
    injury_z_thresh: float = 2.5,
    is_critical_joint: bool = False,
) -> MetricEvaluation:
    """Compute z-score and generate plain-language explanation."""
    std_safe = max(std, 1e-4)
    z = (measured - mean) / std_safe
    abs_z = abs(z)

    # Explanation text
    direction = "higher" if z > 0 else "lower"
    readable_name = name.replace("_", " ")

    if abs_z < 1.0:
        exp = f"{phase.capitalize()} {readable_name} ({measured:.1f}{unit}) is within optimal reference range ({mean:.1f} ± {std:.1f}{unit})."
    elif abs_z < 2.0:
        exp = f"{phase.capitalize()} {readable_name} ({measured:.1f}{unit}) is {abs_z:.1f} standard deviations {direction} than optimal ({mean:.1f}{unit})."
    else:
        exp = f"{phase.capitalize()} {readable_name} ({measured:.1f}{unit}) is {abs_z:.1f} standard deviations from the reference ({mean:.1f}{unit})."

    is_injury = is_critical_joint and (abs_z >= injury_z_thresh)

    return MetricEvaluation(
        name=name,
        phase=phase,
        measured_value=measured,
        reference_mean=mean,
        reference_std=std,
        z_score=z,
        unit=unit,
        explanation=exp,
        is_injury_risk=is_injury,
    )
