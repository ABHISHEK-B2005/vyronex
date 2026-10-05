"""Train a RandomForestClassifier on biomechanical feature CSV data.

This is the training pipeline referenced in the project spec (section 16).
It is intentionally kept simple and transparent for a hackathon prototype.

Usage:
    python ml/train_model.py
    python ml/train_model.py --data data/collected_features.csv

IMPORTANT: The default sample CSV (data/sample_features.csv) is SYNTHETIC
demonstration data only — a model trained on it is for pipeline-testing
purposes, not a validated injury-risk predictor.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import FEATURE_COLUMNS, MODEL_COLUMNS_PATH, MODEL_PATH, SAMPLE_DATA_PATH  # noqa: E402


def load_and_validate(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Training CSV not found at '{csv_path}'. "
            f"Run `python data/generate_sample_data.py` to create sample data, "
            f"or collect real data via Data Collection Mode in the app."
        )

    df = pd.read_csv(csv_path)

    required_columns = set(FEATURE_COLUMNS + ["label"])
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Training CSV is missing required columns: {sorted(missing)}")

    df = df.dropna(subset=FEATURE_COLUMNS + ["label"])
    if df.empty:
        raise ValueError("Training CSV has no valid rows after dropping missing values.")

    return df


def train(csv_path: str, model_path: str, columns_path: str, test_size: float = 0.25, random_state: int = 42) -> None:
    print(f"Loading data from: {csv_path}")
    df = load_and_validate(csv_path)
    print(f"Loaded {len(df)} rows. Label distribution:\n{df['label'].value_counts()}\n")

    X = df[FEATURE_COLUMNS]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=random_state,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(y_test, y_pred, zero_division=0))

    print("=" * 60)
    print("CONFUSION MATRIX")
    print("=" * 60)
    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"true_{l}" for l in labels], columns=[f"pred_{l}" for l in labels])
    print(cm_df)
    print()

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    with open(columns_path, "w", encoding="utf-8") as fh:
        json.dump(FEATURE_COLUMNS, fh, indent=2)

    print(f"Model saved to: {model_path}")
    print(f"Feature column order saved to: {columns_path}")
    print("\nNOTE: This model was trained on the data above. If that data is")
    print("the synthetic sample dataset, this model is for PIPELINE")
    print("DEMONSTRATION ONLY and is not a validated injury-risk predictor.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the injury-risk RandomForest model.")
    parser.add_argument("--data", default=SAMPLE_DATA_PATH, help="Path to training CSV.")
    parser.add_argument("--model-out", default=MODEL_PATH, help="Path to save trained model.")
    parser.add_argument("--columns-out", default=MODEL_COLUMNS_PATH, help="Path to save feature column order.")
    args = parser.parse_args()

    train(args.data, args.model_out, args.columns_out)


if __name__ == "__main__":
    main()
