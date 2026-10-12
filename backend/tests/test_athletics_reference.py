"""Unit and integration tests for Athletics Reference Profiles & Analyzers (Chunk 8D)."""

import os
import tempfile
import unittest

from ml.build_reference import build_default_athletics_reference_profiles, save_reference_profiles
from src.athletics.analyzer import AthleticsAnalyzer
from src.athletics.javelin import JavelinAnalyzer
from src.athletics.long_jump import LongJumpAnalyzer
from src.athletics.shot_put import ShotPutAnalyzer
from src.landmarks import BodyLandmarks, LandmarkPoint


def _make_landmarks(points_dict: dict) -> BodyLandmarks:
    lms = {}
    for name, (x, y) in points_dict.items():
        lms[name] = LandmarkPoint(x=float(x), y=float(y), visibility=0.99)
    return BodyLandmarks(lms)


class AthleticsReferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.profiles_path = os.path.join(self.temp_dir.name, "reference_profiles.json")
        profiles = build_default_athletics_reference_profiles()
        save_reference_profiles(profiles, self.profiles_path)
        self.analyzer = AthleticsAnalyzer(self.profiles_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_long_jump_good_attempt_scores_close_to_reference(self):
        """Test: a coach-approved attempt scores close to the reference."""
        # Good takeoff: knee ~165 deg, lean ~8 deg
        # Shoulder center ~ (345, 200), Hip center ~ (325, 350) -> 8 deg torso lean
        # Knee angle: hip (300, 350), knee (300, 450), ankle (326, 546) -> 165 deg knee
        takeoff_points = {
            "left_shoulder": (320, 200),
            "right_shoulder": (370, 200),
            "left_hip": (300, 350),
            "right_hip": (350, 350),
            "left_knee": (300, 450),
            "right_knee": (350, 450),
            "left_ankle": (326, 546),
            "right_ankle": (376, 546),
        }
        # Good landing: symmetric knees, small valgus
        landing_points = {
            "left_hip": (280, 360),
            "right_hip": (340, 360),
            "left_knee": (280, 440),
            "right_knee": (340, 440),
            "left_ankle": (280, 520),
            "right_ankle": (340, 520),
        }

        takeoff_lm = _make_landmarks(takeoff_points)
        landing_lm = _make_landmarks(landing_points)

        result = self.analyzer.analyze("long_jump", takeoff_lm, landing_lm, attempt_number=1)

        self.assertEqual(result.event, "long_jump")
        self.assertEqual(result.risk_level, "LOW")
        self.assertTrue(result.overall_technique_score >= 80.0)
        self.assertEqual(len(result.injury_warnings), 0)
        self.assertTrue(len(result.explanations) >= 3)

    def test_long_jump_poor_attempt_flags_landing_asymmetry_and_knee_collapse(self):
        """Test: a deliberately poor attempt scores far from reference and flags injury risk."""
        # Poor takeoff: knee bent ~120 deg
        takeoff_points = {
            "left_shoulder": (300, 200),
            "right_shoulder": (350, 200),
            "left_hip": (300, 350),
            "right_hip": (350, 350),
            "left_knee": (220, 430),  # heavily bent
            "right_knee": (260, 430),
            "left_ankle": (300, 550),
            "right_ankle": (350, 550),
        }
        # Dangerous landing: huge knee asymmetry + severe inward valgus collapse
        landing_points = {
            "left_hip": (280, 360),
            "right_hip": (340, 360),
            "left_knee": (320, 440),  # severe inward shift (valgus)
            "right_knee": (300, 440),
            "left_ankle": (260, 520),
            "right_ankle": (360, 520),
        }

        takeoff_lm = _make_landmarks(takeoff_points)
        landing_lm = _make_landmarks(landing_points)

        result = self.analyzer.analyze("long_jump", takeoff_lm, landing_lm, attempt_number=2)

        self.assertEqual(result.risk_level, "HIGH")
        self.assertTrue(result.overall_technique_score < 70.0)
        self.assertTrue(len(result.injury_warnings) > 0)
        # Verify explanation contains z-score deviation
        has_z_explanation = any("standard deviations" in exp for exp in result.explanations)
        self.assertTrue(has_z_explanation)

    def test_javelin_brace_knee_collapse_flags_high_risk(self):
        """Test: javelin brace knee collapse (< 140 deg) flags critical knee injury risk."""
        # Collapsed plant leg
        plant_points = {
            "left_shoulder": (300, 200),
            "right_shoulder": (350, 200),
            "left_hip": (300, 350),
            "right_hip": (350, 350),
            "left_knee": (230, 420),  # severely bent brace knee (~115 deg)
            "left_ankle": (300, 550),
            "right_knee": (350, 450),
            "right_ankle": (350, 550),
        }
        delivery_points = {
            "right_shoulder": (350, 200),
            "right_elbow": (380, 240),
            "right_wrist": (420, 230),
        }

        plant_lm = _make_landmarks(plant_points)
        deliv_lm = _make_landmarks(delivery_points)

        jav_analyzer = JavelinAnalyzer()
        result = jav_analyzer.analyze_attempt(plant_lm, deliv_lm)

        self.assertEqual(result.risk_level, "HIGH")
        self.assertTrue(any("Brace knee collapsed" in w for w in result.injury_warnings))


if __name__ == "__main__":
    unittest.main()
