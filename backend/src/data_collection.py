"""Optional data-collection mode.

Lets the user label completed reps as SAFE / MODERATE_RISK / HIGH_RISK
and appends the corresponding feature row to a CSV, building a real
dataset over time that can later be used with ml/train_model.py.
"""

from __future__ import annotations

import csv
import os
from typing import Optional

from src.config import COLLECTED_DATA_PATH, FEATURE_COLUMNS, ML_LABELS
from src.feature_engineering import RepFeatures


def features_to_ml_row(features: RepFeatures) -> dict:
    """Map the rich RepFeatures dataclass down to the compact ML feature
    schema used by the training pipeline (src.config.FEATURE_COLUMNS).
    """
    return {
        "knee_angle": features.min_knee_angle,
        "hip_angle": features.hip_angle,
        "torso_angle": features.torso_angle,
        "knee_alignment": features.knee_alignment,
        "symmetry": max(features.knee_symmetry, features.hip_symmetry),
        "movement_speed": features.movement_speed,
        "rep_duration": features.rep_duration,
    }


def save_labeled_sample(features: RepFeatures, label: str, path: str = COLLECTED_DATA_PATH) -> str:
    """Append one labeled rep to the collected-data CSV. Creates the file
    (with header) if it doesn't exist yet.
    """
    if label not in ML_LABELS:
        raise ValueError(f"Invalid label '{label}'. Must be one of {ML_LABELS}.")

    row = features_to_ml_row(features)
    row["label"] = label

    file_exists = os.path.exists(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FEATURE_COLUMNS + ["label"])
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    return path


def count_collected_samples(path: str = COLLECTED_DATA_PATH) -> Optional[int]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return max(0, sum(1 for _ in fh) - 1)  # minus header
