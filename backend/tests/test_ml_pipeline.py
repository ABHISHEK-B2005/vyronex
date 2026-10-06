"""Unit and integration tests for ML pipeline (8B and 8C)."""

import json
import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from ml.evaluate import compute_shap_top3, evaluate_held_out_people
from ml.label_review import (
    compute_agreement_metrics,
    compute_cohen_kappa,
    find_disagreements,
    generate_expert_reviewed_dataset,
    resolve_expert_labels,
)
from ml.predict import RiskModelPredictor
from ml.train_model import (
    compute_metrics,
    load_dataset,
    save_model_version,
    train_exercise_model,
    train_person_stratified,
)
from src.config import FEATURE_COLUMNS


def _create_mock_dataset(csv_path: str, num_persons: int = 5, reps_per_person: int = 15):
    """Generate a multi-person dataset with diverse labels and biomechanical features."""
    rows = []
    np.random.seed(42)

    for p in range(1, num_persons + 1):
        person_id = f"P{p:03d}"
        for rep in range(reps_per_person):
            # Equal distribution of styles
            if rep % 3 == 0:
                label = "LOW"
                fault = "none"
                knee = np.random.uniform(80.0, 95.0)
                hip = np.random.uniform(50.0, 65.0)
                torso = np.random.uniform(60.0, 75.0)
                valgus = np.random.uniform(0.01, 0.04)
                sym = np.random.uniform(1.0, 6.0)
            elif rep % 3 == 1:
                label = "MODERATE"
                fault = "knee_valgus"
                knee = np.random.uniform(85.0, 110.0)
                hip = np.random.uniform(60.0, 80.0)
                torso = np.random.uniform(50.0, 65.0)
                valgus = np.random.uniform(0.06, 0.10)
                sym = np.random.uniform(7.0, 12.0)
            else:
                label = "HIGH"
                fault = "forward_lean"
                knee = np.random.uniform(85.0, 120.0)
                hip = np.random.uniform(75.0, 98.0)
                torso = np.random.uniform(30.0, 48.0)
                valgus = np.random.uniform(0.12, 0.22)
                sym = np.random.uniform(12.0, 22.0)

            rows.append({
                "person_id": person_id,
                "exercise": "squat",
                "camera_angle": "front",
                "knee_angle": round(knee, 2),
                "hip_angle": round(hip, 2),
                "torso_angle": round(torso, 2),
                "knee_alignment": round(valgus, 4),
                "symmetry": round(sym, 2),
                "movement_speed": round(np.random.uniform(40.0, 70.0), 2),
                "rep_duration": round(np.random.uniform(1.2, 2.5), 2),
                "label": label,
                "fault_type": fault,
            })

    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    return df


class LabelReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.raw_csv = os.path.join(self.temp_dir.name, "raw_dataset.csv")
        self.expert_csv = os.path.join(self.temp_dir.name, "expert_dataset.csv")
        _create_mock_dataset(self.raw_csv, num_persons=3, reps_per_person=10)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_compute_cohen_kappa_perfect_and_imperfect(self):
        labels_a = ["LOW", "MODERATE", "HIGH", "LOW", "HIGH"]
        labels_b = ["LOW", "MODERATE", "HIGH", "LOW", "HIGH"]
        kappa_perfect = compute_cohen_kappa(labels_a, labels_b)
        self.assertEqual(kappa_perfect, 1.0)

        labels_c = ["LOW", "MODERATE", "LOW", "LOW", "MODERATE"]
        kappa_imperfect = compute_cohen_kappa(labels_a, labels_c)
        self.assertTrue(0.0 <= kappa_imperfect < 1.0)

    def test_compute_agreement_metrics(self):
        r1 = ["LOW", "MODERATE", "HIGH", "HIGH"]
        r2 = ["LOW", "MODERATE", "MODERATE", "HIGH"]
        metrics = compute_agreement_metrics(r1, r2)
        self.assertEqual(metrics["total_samples"], 4)
        self.assertEqual(metrics["agreements"], 3)
        self.assertEqual(metrics["disagreements"], 1)
        self.assertEqual(metrics["raw_agreement_percent"], 75.0)
        self.assertTrue(metrics["cohen_kappa"] > 0)

    def test_find_disagreements_and_resolve(self):
        data = {
            "reviewer_1_label": ["LOW", "MODERATE", "HIGH", "MODERATE"],
            "reviewer_2_label": ["LOW", "HIGH", "HIGH", "LOW"],
            "reviewer_3_label": ["LOW", "MODERATE", "HIGH", "LOW"],
            "fault_type": ["none", "valgus", "lean", "speed"],
        }
        df = pd.DataFrame(data)
        disagreements = find_disagreements(df)
        self.assertEqual(len(disagreements), 2)

        resolved = resolve_expert_labels(df)
        self.assertEqual(len(resolved), 4)
        # Agreement row 0: LOW
        self.assertEqual(resolved.iloc[0]["final_label"], "LOW")
        self.assertFalse(resolved.iloc[0]["disagreement_flag"])
        # Disagreement row 1: R1=MOD, R2=HIGH, R3=MOD -> final MOD
        self.assertEqual(resolved.iloc[1]["final_label"], "MODERATE")
        self.assertTrue(resolved.iloc[1]["disagreement_flag"])

    def test_generate_expert_reviewed_dataset(self):
        df, metrics = generate_expert_reviewed_dataset(
            input_csv=self.raw_csv,
            output_csv=self.expert_csv,
            disagreement_rate=0.10,
        )
        self.assertTrue(os.path.isfile(self.expert_csv))
        self.assertIn("reviewer_1_label", df.columns)
        self.assertIn("reviewer_2_label", df.columns)
        self.assertIn("reviewer_3_label", df.columns)
        self.assertIn("final_label", df.columns)
        self.assertIn("disagreement_flag", df.columns)
        self.assertTrue(metrics["total_samples"] > 0)


class TrainAndEvaluateTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dataset_path = os.path.join(self.temp_dir.name, "multi_person.csv")
        self.model_dir = os.path.join(self.temp_dir.name, "models")
        _create_mock_dataset(self.dataset_path, num_persons=5, reps_per_person=15)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_person_stratified_training_and_versioning(self):
        """Test: GroupKFold on person_id, HIGH-risk recall, and model_versions.json creation."""
        model_path, metrics = train_exercise_model(
            dataset_path=self.dataset_path,
            exercise="squat",
            model_dir=self.model_dir,
        )

        self.assertTrue(os.path.isfile(model_path))
        self.assertTrue(metrics["accuracy"] > 0.70)
        self.assertTrue(metrics["high_risk_recall"] > 0.70)
        self.assertIn("confusion_matrix", metrics)
        self.assertIn("feature_importances", metrics)

        # Check model_versions.json was saved
        versions_path = os.path.join(self.model_dir, "model_versions.json")
        self.assertTrue(os.path.isfile(versions_path))
        with open(versions_path, "r", encoding="utf-8") as fh:
            versions = json.load(fh)
        self.assertEqual(len(versions), 1)
        self.assertEqual(versions[0]["exercise"], "squat")
        self.assertTrue(versions[0]["active"])
        self.assertEqual(versions[0]["unique_persons"], 5)

    def test_evaluate_held_out_people(self):
        """Test: evaluate_held_out_people prints metrics on unseen persons with per-person breakdown."""
        # First train model
        train_exercise_model(
            dataset_path=self.dataset_path,
            exercise="squat",
            model_dir=self.model_dir,
        )

        results = evaluate_held_out_people(
            dataset_path=self.dataset_path,
            exercise="squat",
            model_dir=self.model_dir,
            n_splits=5,
        )

        self.assertEqual(len(results["held_out_persons_evaluated"]), 5)
        self.assertTrue(results["overall_metrics"]["accuracy"] > 0.70)
        self.assertTrue(results["overall_metrics"]["high_risk_recall"] > 0.70)
        self.assertEqual(len(results["per_person_breakdown"]), 5)

        # Check SHAP explanations
        sample_shap = results["sample_shap_explanations"]
        self.assertTrue(len(sample_shap) > 0)
        self.assertEqual(len(sample_shap[0]), 3)  # Top 3 features

    def test_predict_with_trained_model_and_rule_fallback(self):
        """Test: predict.py loads model, produces SHAP top-3, and falls back to rules if missing."""
        # 1. Test fallback when model does not exist
        predictor_fallback = RiskModelPredictor(exercise="nonexistent_exercise", model_dir=self.model_dir)
        self.assertFalse(predictor_fallback.is_ml_active)

        dummy_features = {
            "knee_angle": 90.0,
            "hip_angle": 60.0,
            "torso_angle": 70.0,
            "knee_alignment": 0.02,
            "symmetry": 3.0,
            "movement_speed": 50.0,
            "rep_duration": 1.5,
        }
        label, probs, explanations = predictor_fallback.predict(dummy_features)
        self.assertEqual(label, "LOW")
        self.assertIn("LOW", probs)
        self.assertEqual(len(explanations), 2)

        # 2. Train model and test ML predictions
        train_exercise_model(
            dataset_path=self.dataset_path,
            exercise="squat",
            model_dir=self.model_dir,
        )

        predictor_ml = RiskModelPredictor(exercise="squat", model_dir=self.model_dir)
        self.assertTrue(predictor_ml.is_ml_active)

        # Dangerous form features
        dangerous_features = {
            "knee_angle": 115.0,
            "hip_angle": 95.0,
            "torso_angle": 32.0,
            "knee_alignment": 0.18,
            "symmetry": 18.0,
            "movement_speed": 45.0,
            "rep_duration": 2.0,
        }
        pred_label, pred_probs, pred_explanations = predictor_ml.predict(dangerous_features)
        self.assertEqual(pred_label, "HIGH")
        self.assertTrue(pred_probs["HIGH"] > 0.50)
        self.assertEqual(len(pred_explanations), 3)  # Top 3 features


if __name__ == "__main__":
    unittest.main()
