"""Injury-RISK engine.

Two interchangeable "risk providers" are supported, selectable at runtime:
  1. RuleBasedRiskEngine  - transparent, weighted-scoring engine (default)
  2. MLRiskEngine         - loads a trained scikit-learn model (ml/predict.py)

Both implement the same interface: ``assess(form_scores, features) -> RiskResult``
so the Streamlit app / feedback engine can treat them interchangeably.

DISCLAIMER: All thresholds are prototype engineering assumptions, not
medically validated cut-offs. See src/config.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from src.config import (
    RISK_LABELS,
    RISK_LOW_MAX,
    RISK_MODERATE_MAX,
    RISK_WEIGHTS,
)
from src.feature_engineering import RepFeatures
from src.form_analysis import FormScores


@dataclass
class RiskResult:
    risk_score: float          # 0-100, higher = riskier
    risk_level: str            # "LOW" | "MODERATE" | "HIGH"
    risk_label: str            # human readable, e.g. "HIGH MOVEMENT RISK"
    contributing_factors: Dict[str, float]  # category -> risk contribution (0-100)
    primary_issue: Optional[str]  # category name most responsible for risk
    engine: str                # "rule_based" or "ml"

    def to_dict(self) -> dict:
        return {
            "risk_score": self.risk_score,
            "risk_level": self.risk_level,
            "risk_label": self.risk_label,
            "contributing_factors": self.contributing_factors,
            "primary_issue": self.primary_issue,
            "engine": self.engine,
        }


def classify_risk_score(score: float) -> str:
    """Map a 0-100 risk score to LOW / MODERATE / HIGH using the
    configurable thresholds in src.config.
    """
    if score <= RISK_LOW_MAX:
        return "LOW"
    if score <= RISK_MODERATE_MAX:
        return "MODERATE"
    return "HIGH"


class RuleBasedRiskEngine:
    """Transparent weighted risk engine (Priority Mode 1).

    risk_contribution(category) = (100 - form_score(category)) * weight(category)
    risk_score = sum(risk_contribution) over all categories
    """

    name = "rule_based"

    def assess(self, form_scores: FormScores, features: RepFeatures) -> RiskResult:
        contributions: Dict[str, float] = {}
        for category, weight in RISK_WEIGHTS.items():
            form_value = getattr(form_scores, category)
            deficiency = 100.0 - form_value  # how "bad" this category is
            contributions[category] = round(deficiency * weight, 2)

        risk_score = round(sum(contributions.values()), 1)
        risk_score = max(0.0, min(100.0, risk_score))
        risk_level = classify_risk_score(risk_score)
        risk_label = RISK_LABELS[risk_level]

        primary_issue = None
        if contributions:
            primary_issue = max(contributions, key=contributions.get)
            # Only report a primary issue if it's a meaningful contributor
            if contributions[primary_issue] < 2.0:
                primary_issue = None

        return RiskResult(
            risk_score=risk_score,
            risk_level=risk_level,
            risk_label=risk_label,
            contributing_factors=contributions,
            primary_issue=primary_issue,
            engine=self.name,
        )


class MLRiskEngine:
    """Wraps a trained ML model (see ml/train_model.py, ml/predict.py).

    Falls back to raising a clear error if no trained model is found, so
    the UI can show a friendly message ("Train a model first") instead of
    crashing.
    """

    name = "ml"

    def __init__(self) -> None:
        from ml.predict import RiskModelPredictor  # local import to avoid hard dep

        self._predictor = RiskModelPredictor()

    def assess(self, form_scores: FormScores, features: RepFeatures) -> RiskResult:
        feature_dict = {
            "knee_angle": features.min_knee_angle,
            "hip_angle": features.hip_angle,
            "torso_angle": features.torso_angle,
            "knee_alignment": features.knee_alignment,
            "symmetry": max(features.knee_symmetry, features.hip_symmetry),
            "movement_speed": features.movement_speed,
            "rep_duration": features.rep_duration,
        }
        pred_res = self._predictor.predict(feature_dict)
        if len(pred_res) == 3:
            label, probabilities, explanations = pred_res
        else:
            label, probabilities = pred_res
            explanations = []

        # Map classifier probability of risk to a 0-100 continuous score
        risk_prob = probabilities.get("HIGH", probabilities.get("HIGH_RISK", 0.0)) * 1.0 + probabilities.get(
            "MODERATE", probabilities.get("MODERATE_RISK", 0.0)
        ) * 0.5
        risk_score = round(risk_prob * 100.0, 1)
        risk_level = "LOW" if label in ("LOW", "SAFE") else ("HIGH" if label in ("HIGH", "HIGH_RISK") else "MODERATE")
        risk_label = RISK_LABELS[risk_level]

        contributing = {"ml_predicted_class": label}
        if explanations:
            contributing["shap_top_factors"] = explanations

        return RiskResult(
            risk_score=risk_score,
            risk_level=risk_level,
            risk_label=risk_label,
            contributing_factors=contributing,
            primary_issue=explanations[0]["feature"] if explanations else None,
            engine=self.name,
        )



def get_risk_engine(mode: str = "rule_based"):
    """Factory returning the requested risk engine instance."""
    if mode == "ml":
        return MLRiskEngine()
    return RuleBasedRiskEngine()
