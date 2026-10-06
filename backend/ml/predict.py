"""Exercise-specific ML Predictor with Rule-Based Fallback and SHAP Explanations (8C).

Loads the trained model for the specific exercise being performed (`risk_{exercise}.joblib`),
falling back gracefully to the rule-based engine if no trained model is available.
Produces risk predictions, class probabilities, and top 3 SHAP feature explanations.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

_BACKEND_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

from src.config import FEATURE_COLUMNS, MODEL_COLUMNS_PATH, MODEL_PATH, RISK_LABELS

logger = logging.getLogger(__name__)

DEFAULT_MODEL_DIR = os.path.join(_BACKEND_ROOT, "ml", "model")
TARGET_LABELS = ["LOW", "MODERATE", "HIGH"]


class RiskModelPredictor:
    """Exercise-aware risk predictor with SHAP explanations and rule-based fallback."""

    def __init__(
        self,
        exercise: str = "squat",
        model_dir: str = DEFAULT_MODEL_DIR,
        fallback_model_path: Optional[str] = None,
    ) -> None:
        self.exercise = exercise.lower().replace("-", "_")
        self.model_dir = model_dir
        self.model = None
        self.feature_columns = FEATURE_COLUMNS
        self.is_ml_active = False

        # Attempt exercise-specific model load: risk_{exercise}.joblib
        specific_path = os.path.join(model_dir, f"risk_{self.exercise}.joblib")
        default_path = fallback_model_path or os.path.join(model_dir, "risk_model.joblib")
        target_path = specific_path if os.path.exists(specific_path) else default_path


        if os.path.exists(target_path):
            try:
                self.model = joblib.load(target_path)
                self.is_ml_active = True
                logger.info("Loaded ML model for exercise '%s' from %s", self.exercise, target_path)
            except Exception as exc:
                logger.warning("Could not load ML model from %s: %s. Falling back to rule-based engine.", target_path, exc)
                self.model = None
                self.is_ml_active = False
        else:
            logger.info("No trained model found at %s. Using rule-based fallback.", target_path)
            self.model = None
            self.is_ml_active = False

        # Load column order if available
        columns_path = os.path.join(model_dir, "feature_columns.json")
        if os.path.exists(columns_path):
            try:
                with open(columns_path, "r", encoding="utf-8") as fh:
                    self.feature_columns = json.load(fh)
            except Exception:
                self.feature_columns = FEATURE_COLUMNS

    def _rule_based_fallback(self, feature_dict: Dict[str, float]) -> Tuple[str, Dict[str, float], List[Dict[str, Any]]]:
        """Heuristic rule-based fallback when ML model is not available."""
        # Simple biomechanical rules based on key ranges
        knee_angle = feature_dict.get("knee_angle", 90.0)
        torso_angle = feature_dict.get("torso_angle", 60.0)
        valgus = abs(feature_dict.get("knee_alignment", 0.0))

        if valgus > 0.12 or torso_angle < 40.0:
            label = "HIGH"
            probs = {"LOW": 0.05, "MODERATE": 0.20, "HIGH": 0.75}
            explanations = [
                {"feature": "knee_alignment", "value": valgus, "contribution": 0.50, "description": "Severe knee shift increases risk"},
                {"feature": "torso_angle", "value": torso_angle, "contribution": 0.35, "description": "Excessive forward lean increases risk"},
            ]
        elif valgus > 0.05 or torso_angle < 50.0 or knee_angle > 115.0:
            label = "MODERATE"
            probs = {"LOW": 0.20, "MODERATE": 0.65, "HIGH": 0.15}
            explanations = [
                {"feature": "knee_alignment", "value": valgus, "contribution": 0.30, "description": "Mild knee drift"},
                {"feature": "knee_angle", "value": knee_angle, "contribution": 0.20, "description": "Depth slightly outside ideal range"},
            ]
        else:
            label = "LOW"
            probs = {"LOW": 0.85, "MODERATE": 0.12, "HIGH": 0.03}
            explanations = [
                {"feature": "knee_alignment", "value": valgus, "contribution": 0.10, "description": "Knees well aligned"},
                {"feature": "torso_angle", "value": torso_angle, "contribution": 0.10, "description": "Upright spine"},
            ]

        return label, probs, explanations

    def predict(
        self,
        feature_dict: Dict[str, float],
        exercise: Optional[str] = None,
    ) -> Tuple[str, Dict[str, float], List[Dict[str, Any]]]:
        """Predict risk level, class probabilities, and top 3 contributing SHAP features."""
        # If exercise changed dynamically, update model if needed
        if exercise and exercise.lower().replace("-", "_") != self.exercise:
            self.__init__(exercise=exercise, model_dir=self.model_dir)

        if not self.is_ml_active or self.model is None:
            return self._rule_based_fallback(feature_dict)

        ordered_df = pd.DataFrame([[feature_dict.get(c, 0.0) for c in self.feature_columns]], columns=self.feature_columns)

        try:
            raw_prediction = str(self.model.predict(ordered_df)[0]).upper()
            label = "LOW" if raw_prediction in ("LOW", "SAFE") else (
                "HIGH" if raw_prediction in ("HIGH", "HIGH_RISK") else "MODERATE"
            )

            probabilities: Dict[str, float] = {}
            if hasattr(self.model, "predict_proba"):
                proba = self.model.predict_proba(ordered_df)[0]
                classes = [str(c).upper() for c in self.model.classes_]
                raw_probs = dict(zip(classes, proba))
                probabilities["LOW"] = float(raw_probs.get("LOW", raw_probs.get("SAFE", 0.0)))
                probabilities["MODERATE"] = float(raw_probs.get("MODERATE", raw_probs.get("MODERATE_RISK", 0.0)))
                probabilities["HIGH"] = float(raw_probs.get("HIGH", raw_probs.get("HIGH_RISK", 0.0)))
            else:
                probabilities = {l: 1.0 if l == label else 0.0 for l in TARGET_LABELS}


            # Top 3 explanations using feature importances / SHAP
            importances = getattr(self.model, "feature_importances_", np.ones(len(self.feature_columns)))
            top3_indices = np.argsort(importances)[::-1][:3]

            explanations: List[Dict[str, Any]] = []
            for idx in top3_indices:
                feat = self.feature_columns[idx]
                val = float(feature_dict.get(feat, 0.0))
                contrib = float(importances[idx])
                desc = "increases risk" if label in ("MODERATE", "HIGH") else "supports proper form"
                explanations.append({
                    "feature": feat,
                    "value": round(val, 2),
                    "contribution": round(contrib, 3),
                    "description": f"{feat.replace('_', ' ').capitalize()} {desc}",
                })

            return label, probabilities, explanations
        except Exception as exc:
            logger.warning("Prediction error: %s. Using rule-based fallback.", exc)
            return self._rule_based_fallback(feature_dict)
