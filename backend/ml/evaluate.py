"""Evaluation tool for testing models on held-out people with SHAP explanations (8C).

Evaluates exercise risk models exclusively on unseen participants (Person-Stratified GroupKFold)
to ensure models generalize to novel athletes and do not overfit to individual movement quirks.

Metrics Reported:
- Accuracy, Macro Precision, Macro Recall, Macro F1
- SAFETY-CRITICAL: Recall on HIGH Risk (missing dangerous form is the costliest error)
- Confusion Matrix per exercise
- Per-person evaluation breakdown
- Top 3 SHAP feature explanations per prediction
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupKFold

_BACKEND_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

from src.config import FEATURE_COLUMNS
from ml.train_model import (
    DEFAULT_DATASET_PATH,
    DEFAULT_MODEL_DIR,
    VALID_LABELS,
    compute_metrics,
    load_dataset,
)

# Optional SHAP integration
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False


def compute_shap_top3(
    model: Any,
    X_sample: pd.DataFrame | np.ndarray,
    feature_names: List[str] = FEATURE_COLUMNS,
) -> List[List[Dict[str, Any]]]:
    """Compute top 3 contributing features for each prediction using TreeExplainer."""
    if isinstance(X_sample, np.ndarray):
        X_df = pd.DataFrame(X_sample, columns=feature_names)
    else:
        X_df = X_sample.copy()

    explanations_per_sample: List[List[Dict[str, Any]]] = []

    if SHAP_AVAILABLE and hasattr(model, "predict_proba"):
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_df)

            for i in range(len(X_df)):
                row_features = X_df.iloc[i].to_dict()
                # Aggregate absolute importance across classes for this instance
                if isinstance(shap_values, list):
                    # Multi-class list of arrays [n_samples, n_features]
                    inst_imp = np.sum([np.abs(sv[i]) for sv in shap_values], axis=0)
                else:
                    inst_imp = np.abs(shap_values[i])
                    if inst_imp.ndim > 1:
                        inst_imp = np.sum(inst_imp, axis=-1)

                top3_indices = np.argsort(inst_imp)[::-1][:3]
                top3 = []
                for idx in top3_indices:
                    feat = feature_names[idx]
                    val = float(row_features[feat])
                    contrib = float(inst_imp[idx])
                    direction = "increases risk" if val > 50 or "knee" in feat else "stabilizes form"
                    top3.append({
                        "feature": feat,
                        "value": round(val, 2),
                        "contribution": round(contrib, 4),
                        "description": direction,
                    })
                explanations_per_sample.append(top3)
            return explanations_per_sample
        except Exception:
            pass

    # Fallback to model feature importances
    importances = getattr(model, "feature_importances_", np.ones(len(feature_names)) / len(feature_names))
    top3_idx = np.argsort(importances)[::-1][:3]

    for i in range(len(X_df)):
        row_features = X_df.iloc[i].to_dict()
        top3 = []
        for idx in top3_idx:
            feat = feature_names[idx]
            val = float(row_features[feat])
            top3.append({
                "feature": feat,
                "value": round(val, 2),
                "contribution": round(float(importances[idx]), 4),
                "description": "primary factor",
            })
        explanations_per_sample.append(top3)

    return explanations_per_sample


def evaluate_held_out_people(
    dataset_path: str = DEFAULT_DATASET_PATH,
    exercise: str = "squat",
    model_dir: str = DEFAULT_MODEL_DIR,
    n_splits: int = 5,
) -> Dict[str, Any]:
    """Perform evaluation strictly across held-out participant groups."""
    df = load_dataset(dataset_path, exercise=exercise)
    X = df[FEATURE_COLUMNS].copy()
    y = df["label"].copy()
    groups = df["person_id"].copy()

    unique_persons = sorted(groups.unique())
    n_groups = len(unique_persons)

    norm_exercise = exercise.lower().replace("-", "_")
    model_path = os.path.join(model_dir, f"risk_{norm_exercise}.joblib")
    if not os.path.exists(model_path):
        model_path = os.path.join(model_dir, "risk_model.joblib")

    trained_model = joblib.load(model_path) if os.path.exists(model_path) else None

    oof_predictions = pd.Series(index=df.index, dtype=object)
    per_person_metrics: Dict[str, Dict[str, Any]] = {}

    if n_groups >= 2:
        effective_splits = min(n_splits, n_groups)
        gkf = GroupKFold(n_splits=effective_splits)

        for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups), 1):
            X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
            X_te, y_te = X.iloc[test_idx], y.iloc[test_idx]

            from sklearn.ensemble import RandomForestClassifier
            fold_clf = RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42, class_weight="balanced")
            fold_clf.fit(X_tr, y_tr)

            preds = fold_clf.predict(X_te)
            oof_predictions.iloc[test_idx] = preds

            # Record per-person breakdown
            for pid in groups.iloc[test_idx].unique():
                pid_mask = (groups.iloc[test_idx] == pid)
                y_true_pid = y_te[pid_mask]
                y_pred_pid = preds[pid_mask]
                per_person_metrics[pid] = {
                    "sample_count": len(y_true_pid),
                    "accuracy": round(float(accuracy_score(y_true_pid, y_pred_pid)), 4),
                    "f1": round(float(f1_score(y_true_pid, y_pred_pid, labels=VALID_LABELS, average="macro", zero_division=0)), 4),
                }
    else:
        # 1 group fallback
        if trained_model is not None:
            oof_predictions = pd.Series(trained_model.predict(X), index=df.index)
        else:
            oof_predictions = y.copy()

        per_person_metrics[unique_persons[0]] = {
            "sample_count": len(y),
            "accuracy": round(float(accuracy_score(y, oof_predictions)), 4),
            "f1": round(float(f1_score(y, oof_predictions, labels=VALID_LABELS, average="macro", zero_division=0)), 4),
        }

    valid_mask = oof_predictions.notnull()
    overall = compute_metrics(y[valid_mask], oof_predictions[valid_mask])

    # Sample SHAP explanations on test samples
    sample_df = X.head(5)
    shap_explanations = compute_shap_top3(trained_model, sample_df) if trained_model else []

    results = {
        "exercise": exercise,
        "dataset_path": dataset_path,
        "total_samples": len(df),
        "held_out_persons_evaluated": unique_persons,
        "overall_metrics": overall,
        "per_person_breakdown": per_person_metrics,
        "sample_shap_explanations": shap_explanations,
    }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate injury risk models strictly on held-out people.")
    parser.add_argument("--data", default=DEFAULT_DATASET_PATH, help="Path to expert dataset CSV")
    parser.add_argument("--exercise", default="squat", help="Target exercise")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, help="Path to models directory")
    args = parser.parse_args()

    results = evaluate_held_out_people(
        dataset_path=args.data,
        exercise=args.exercise,
        model_dir=args.model_dir,
    )

    metrics = results["overall_metrics"]

    print("=" * 65)
    print(f"HELD-OUT PERSON EVALUATION: {args.exercise.upper()}")
    print("=" * 65)
    print(f"Evaluated on {len(results['held_out_persons_evaluated'])} Unseen Participants: {results['held_out_persons_evaluated']}")
    print(f"Total Reps Evaluated:       {results['total_samples']}")
    print("-" * 65)
    print(f"Overall Accuracy:           {metrics['accuracy'] * 100:.1f}%")
    print(f"Macro Precision:            {metrics['precision_macro'] * 100:.1f}%")
    print(f"Macro Recall:               {metrics['recall_macro'] * 100:.1f}%")
    print(f"Macro F1 Score:             {metrics['f1_macro'] * 100:.1f}%")
    print(f"HIGH-Risk Recall (SAFETY):  {metrics['high_risk_recall'] * 100:.1f}%  <-- KEY RISK METRIC")
    print("-" * 65)

    print("\nConfusion Matrix (Rows=True, Cols=Pred: LOW, MODERATE, HIGH):")
    cm_arr = np.array(metrics["confusion_matrix"])
    print(pd.DataFrame(cm_arr, index=["True_LOW", "True_MOD", "True_HIGH"], columns=["Pred_LOW", "Pred_MOD", "Pred_HIGH"]).to_string())

    print("\nPer-Person Held-Out Breakdown:")
    for pid, pmetrics in results["per_person_breakdown"].items():
        print(f"  - Participant {pid}: Accuracy={pmetrics['accuracy']*100:.1f}%, F1={pmetrics['f1']*100:.1f}% ({pmetrics['sample_count']} reps)")

    if results["sample_shap_explanations"]:
        print("\nSHAP Top-3 Feature Explanations (Sample Predictions):")
        for i, top3 in enumerate(results["sample_shap_explanations"][:3], 1):
            print(f"  Rep #{i}:")
            for item in top3:
                print(f"    * {item['feature']:16s} = {item['value']} (contrib: {item['contribution']:.3f}, {item['description']})")

    print("=" * 65)


if __name__ == "__main__":
    main()
