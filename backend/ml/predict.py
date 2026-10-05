"""Loads the trained RandomForest model and makes predictions.

Used by src.risk_engine.MLRiskEngine (MODE 2 in the app). Raises a clear,
catchable error if no trained model exists yet, so the UI can tell the
user to run ``python ml/train_model.py`` first instead of crashing.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Dict, Tuple

import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import ML_LABELS, MODEL_COLUMNS_PATH, MODEL_PATH  # noqa: E402


class ModelNotTrainedError(RuntimeError):
    """Raised when the ML risk model file is missing."""


class RiskModelPredictor:
    """Thin wrapper that loads the model ONCE and reuses it for
    predictions (never re-loads from disk per-frame).
    """

    def __init__(self, model_path: str = MODEL_PATH, columns_path: str = MODEL_COLUMNS_PATH) -> None:
        if not os.path.exists(model_path):
            raise ModelNotTrainedError(
                f"No trained model found at '{model_path}'. "
                f"Run `python ml/train_model.py` first, or use the "
                f"Prototype Rule-Based Risk Engine mode instead."
            )
        if not os.path.exists(columns_path):
            raise ModelNotTrainedError(
                f"Feature-column metadata not found at '{columns_path}'. "
                f"Re-run `python ml/train_model.py` to regenerate it."
            )

        self.model = joblib.load(model_path)
        with open(columns_path, "r", encoding="utf-8") as fh:
            self.feature_columns = json.load(fh)

    def predict(self, feature_dict: Dict[str, float]) -> Tuple[str, Dict[str, float]]:
        """Predict a risk label + class probabilities for one rep's
        features. ``feature_dict`` must contain (at least) all keys in
        ``self.feature_columns``.
        """
        missing = [c for c in self.feature_columns if c not in feature_dict]
        if missing:
            raise ValueError(f"Missing required features for prediction: {missing}")

        ordered_values = [[feature_dict[c] for c in self.feature_columns]]

        prediction = self.model.predict(ordered_values)[0]
        proba = self.model.predict_proba(ordered_values)[0]
        class_labels = list(self.model.classes_)
        probabilities = {label: float(p) for label, p in zip(class_labels, proba)}

        # Ensure all expected labels are present in output (0.0 if unseen)
        for label in ML_LABELS:
            probabilities.setdefault(label, 0.0)

        return str(prediction), probabilities
