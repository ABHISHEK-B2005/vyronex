"""Unit and integration tests for ml/collect.py data collection tool (8A)."""

import csv
import os
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from ml.collect import (
    DATASET_COLUMNS,
    append_dataset_row,
    collect_from_sequence,
    features_to_dataset_row,
    generate_style_squat_frames,
    normalize_label,
    record_person_protocol,
    save_landmarks_archive,
)
from src.config import FEATURE_COLUMNS
from src.demo_mode import build_demo_sequence


class CollectToolTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.csv_path = os.path.join(self.temp_dir.name, "test_collected_features.csv")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_normalize_label(self):
        self.assertEqual(normalize_label("correct"), "LOW")
        self.assertEqual(normalize_label("proper"), "LOW")
        self.assertEqual(normalize_label("minor_error"), "MODERATE")
        self.assertEqual(normalize_label("minor error"), "MODERATE")
        self.assertEqual(normalize_label("dangerous_form"), "HIGH")
        self.assertEqual(normalize_label("dangerous"), "HIGH")
        self.assertEqual(normalize_label("LOW"), "LOW")
        self.assertEqual(normalize_label("MODERATE"), "MODERATE")
        self.assertEqual(normalize_label("HIGH"), "HIGH")

        with self.assertRaises(ValueError):
            normalize_label("invalid_style_name")

    def test_record_one_person_10_reps_per_style_produces_30_clean_rows(self):
        """Test: record one person doing 10 reps in each style and check the CSV has 30 clean rows."""
        person_id = "P001"
        exercise = "squat"
        camera_angle = "front"

        # Record full protocol: 10 Correct (LOW), 10 Minor (MODERATE), 10 Dangerous (HIGH)
        results = record_person_protocol(
            person_id=person_id,
            exercise=exercise,
            camera_angle=camera_angle,
            reps_per_style=10,
            output_csv=self.csv_path,
        )

        self.assertIn("correct", results)
        self.assertIn("minor_error", results)
        self.assertIn("dangerous_form", results)

        self.assertEqual(len(results["correct"]), 10)
        self.assertEqual(len(results["minor_error"]), 10)
        self.assertEqual(len(results["dangerous_form"]), 10)

        # Verify CSV exists and read using pandas & csv
        self.assertTrue(os.path.isfile(self.csv_path))

        df = pd.read_csv(self.csv_path)

        # Exactly 30 rows
        self.assertEqual(len(df), 30)

        # Columns match the 12-column schema exactly
        self.assertEqual(list(df.columns), DATASET_COLUMNS)

        # Label counts: exactly 10 LOW, 10 MODERATE, 10 HIGH
        counts = df["label"].value_counts().to_dict()
        self.assertEqual(counts.get("LOW"), 10)
        self.assertEqual(counts.get("MODERATE"), 10)
        self.assertEqual(counts.get("HIGH"), 10)

        # Check person_id, exercise, camera_angle
        self.assertTrue((df["person_id"] == "P001").all())
        self.assertTrue((df["exercise"] == "squat").all())
        self.assertTrue((df["camera_angle"] == "front").all())

        # Check biomechanical feature values are clean, valid, and positive/finite
        for col in FEATURE_COLUMNS:
            self.assertIn(col, df.columns)
            self.assertFalse(df[col].isnull().any(), f"Column {col} contains null values")
            self.assertTrue((df[col] >= 0).all() or col == "knee_alignment", f"Column {col} contains invalid values")

        # Rep durations are all positive
        self.assertTrue((df["rep_duration"] > 0).all())

        # Check fault types
        self.assertEqual(len(df[df["label"] == "LOW"]["fault_type"].unique()), 1)
        self.assertEqual(df[df["label"] == "LOW"]["fault_type"].iloc[0], "none")

    def test_save_landmarks_archive(self):
        landmarks_dir = os.path.join(self.temp_dir.name, "landmarks")
        sample_landmarks = [
            {"frame": 0, "timestamp": 0.0, "points": {"nose": {"x": 100, "y": 100, "visibility": 1.0}}}
        ]
        target_file = save_landmarks_archive(
            landmarks_list=sample_landmarks,
            person_id="P002",
            exercise="squat",
            style="correct",
            rep_number=1,
            output_dir=landmarks_dir,
        )
        self.assertTrue(os.path.isfile(target_file))
        self.assertTrue(target_file.endswith(".json"))


if __name__ == "__main__":
    unittest.main()
