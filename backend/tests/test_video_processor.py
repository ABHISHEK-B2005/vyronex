"""Unit tests for services/video_processor.py."""

import asyncio
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from services.video_processor import (
    InvalidVideoFileError,
    NoPersonDetectedError,
    VideoProcessingError,
    VideoProcessor,
    analyze_pose_sequence,
    build_checklist,
    serialize_rep,
)
from services.report_recompute import recompute_report
from src.demo_mode import build_demo_sequence
from src.feature_engineering import extract_rep_features
from src.form_analysis import FormScores, analyze_alignment
from src.geometry import alignment_summary, compute_alignment
from src.landmarks import BodyLandmarks, LandmarkPoint
from src.squat_detector import RepFrameSample, SquatDetector
from src.pipeline import _compute_knee_alignment_shift
from src.pipeline import build_frame_sample
from src.pipeline import RepResult
from src.risk_engine import RiskResult
from src.squat_detector import CompletedRep


class FakeCollection:
    def __init__(self, documents=None):
        self.documents = list(documents or [])
        self.updates = []

    async def insert_one(self, document):
        self.documents.append(dict(document))

    async def update_one(self, query, update, upsert=False):
        self.updates.append((query, update))
        matched = False
        for doc in self.documents:
            match = all(doc.get(k) == v for k, v in query.items())
            if match:
                matched = True
                if "$set" in update:
                    doc.update(update["$set"])
        if upsert and not matched:
            self.documents.append(dict(update.get("$set", {})))


    async def find_one(self, query):
        for doc in self.documents:
            match = True
            for k, v in query.items():
                if doc.get(k) != v:
                    match = False
                    break
            if match:
                return doc
        return None

    async def delete_many(self, query):
        self.documents = [
            document for document in self.documents
            if not all(document.get(key) == value for key, value in query.items())
        ]

    def find(self, query, projection=None):
        matched = [doc for doc in self.documents if all(doc.get(k) == v for k, v in query.items())]

        class FakeCursor:
            def __init__(self, items):
                self.items = items

            async def to_list(self, length=1000):
                return list(self.items[:length])

            def sort(self, key, direction=1):
                self.items.sort(key=lambda item: item.get(key, 0), reverse=direction < 0)
                return self

            def limit(self, count):
                self.items = self.items[:count]
                return self

        return FakeCursor(matched)


def _create_synthetic_video(file_path: str, num_frames: int = 60, fps: float = 30.0, width: int = 640, height: int = 480):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(file_path, fourcc, fps, (width, height))
    for i in range(num_frames):
        # Create a plain frame
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()


def _synthetic_squat_sequence(
    view: str,
    duration: float = 16.5,
    fps: int = 30,
    inward_knee_shift: float = 0.0,
    torso_lean_degrees: float = 8.0,
):
    widths = {
        "front": (0.42, 0.34),
        "side": (0.12, 0.10),
        "angled": (0.25, 0.22),
    }
    shoulder_width, hip_width = widths[view]
    lean = 0.5 * np.tan(np.radians(torso_lean_degrees))
    frames = []
    for index in range(int(duration * fps)):
        phase = (index / fps % 3.3) / 3.3
        if phase < 0.30:
            amount = phase / 0.30
            angle = 170 - 80 * amount
        elif phase < 0.45:
            angle = 90
        elif phase < 0.75:
            amount = (phase - 0.45) / 0.30
            angle = 90 + 80 * amount
        else:
            angle = 170
        knee_bend = 0.45 * np.tan(np.radians((180 - angle) / 2))
        left_inward = inward_knee_shift if view == "front" else 0.0
        right_inward = -inward_knee_shift if view == "front" else 0.0
        coords = {
            "nose": (0.0, -0.78, lean),
            "left_shoulder": (-shoulder_width / 2, -0.50, lean),
            "right_shoulder": (shoulder_width / 2, -0.50, lean),
            "left_elbow": (-shoulder_width / 2, -0.30, lean),
            "right_elbow": (shoulder_width / 2, -0.30, lean),
            "left_wrist": (-shoulder_width / 2, -0.10, lean),
            "right_wrist": (shoulder_width / 2, -0.10, lean),
            "left_hip": (-hip_width / 2, 0.0, 0.0),
            "right_hip": (hip_width / 2, 0.0, 0.0),
            "left_knee": (-hip_width / 2 + left_inward, 0.45, knee_bend),
            "right_knee": (hip_width / 2 + right_inward, 0.45, knee_bend),
            "left_ankle": (-hip_width / 2, 0.90, 0.0),
            "right_ankle": (hip_width / 2, 0.90, 0.0),
        }
        points = {
            name: LandmarkPoint(x, y, 0.99, z)
            for name, (x, y, z) in coords.items()
        }
        frames.append({
            "timestamp": index / fps,
            "landmarks": BodyLandmarks(points),
        })
    return frames


class VideoProcessorTests(unittest.TestCase):
    def setUp(self):
        self.database = SimpleNamespace(
            users=FakeCollection(),
            sessions=FakeCollection(),
            reps=FakeCollection(),
            reports=FakeCollection(),
            pose_frames=FakeCollection(),
        )
        self.processor = VideoProcessor()

    def test_nonexistent_video_file_raises_error(self):
        with self.assertRaises(InvalidVideoFileError):
            asyncio.run(self.processor.process_video("non_existent_file_path_123.mp4"))

    def test_world_landmarks_are_used_and_image_fallback_is_marked_low_confidence(self):
        image_points = [
            SimpleNamespace(x=0.25, y=0.75, visibility=0.9, z=0.0)
            for _ in range(33)
        ]
        world_points = [
            SimpleNamespace(x=0.12, y=-0.34, z=0.56, visibility=0.99)
            for _ in range(33)
        ]
        image = SimpleNamespace(landmark=image_points)
        world = SimpleNamespace(landmark=world_points)

        world_body = BodyLandmarks.from_mediapipe(image, 640, 480, world)
        image_fallback = BodyLandmarks.from_mediapipe(image, 640, 480)

        self.assertEqual(world_body.point3d("nose"), (0.12, -0.34, 0.56))
        self.assertEqual(world_body.get("nose").image_x, 160.0)
        self.assertFalse(world_body.low_confidence)
        self.assertEqual(image_fallback.point3d("nose"), (160.0, 360.0, 0.0))
        self.assertTrue(image_fallback.low_confidence)

    def test_knee_alignment_uses_position_along_slanted_leg(self):
        landmarks = BodyLandmarks({
            "left_hip": LandmarkPoint(100.0, 100.0, 1.0),
            "left_knee": LandmarkPoint(133.3, 200.0, 1.0),
            "left_ankle": LandmarkPoint(200.0, 400.0, 1.0),
            "right_hip": LandmarkPoint(300.0, 100.0, 1.0),
            "right_knee": LandmarkPoint(333.3, 200.0, 1.0),
            "right_ankle": LandmarkPoint(400.0, 400.0, 1.0),
        })

        self.assertAlmostEqual(_compute_knee_alignment_shift(landmarks), 0.0, places=3)

    def test_knee_alignment_ignores_forward_motion_but_detects_inward_motion(self):
        def pose(left_knee_z, right_knee_z):
            return BodyLandmarks({
                "left_shoulder": LandmarkPoint(100.0, 20.0, 1.0, 0.0),
                "right_shoulder": LandmarkPoint(100.0, 20.0, 1.0, 60.0),
                "left_hip": LandmarkPoint(100.0, 100.0, 1.0, 0.0),
                "right_hip": LandmarkPoint(100.0, 100.0, 1.0, 60.0),
                "left_knee": LandmarkPoint(180.0, 250.0, 1.0, left_knee_z),
                "right_knee": LandmarkPoint(180.0, 250.0, 1.0, right_knee_z),
                "left_ankle": LandmarkPoint(100.0, 400.0, 1.0, 0.0),
                "right_ankle": LandmarkPoint(100.0, 400.0, 1.0, 60.0),
            })

        neutral_pose = pose(0.0, 60.0)
        self.assertAlmostEqual(_compute_knee_alignment_shift(neutral_pose), 0.0, places=3)

        inward_pose = pose(20.0, 40.0)
        self.assertGreater(_compute_knee_alignment_shift(inward_pose), 0.05)

    def test_empty_rep_features_are_not_zeroed(self):
        features = extract_rep_features(CompletedRep(rep_number=1, samples=[]))
        self.assertIsNone(features.knee_symmetry)
        self.assertIsNone(features.hip_symmetry)
        self.assertIsNone(features.knee_alignment)
        self.assertIsNone(features.torso_angle)

    def test_incomplete_reps_are_finalized_when_video_ends_mid_cycle(self):
        detector = SquatDetector()
        samples = [
            RepFrameSample(0.00, 170.0, 170.0, 170.0, 170.0, 170.0, 84.0, 0.0, (0.0, 0.0, 0.0)),
            RepFrameSample(0.15, 145.0, 145.0, 145.0, 145.0, 145.0, 82.0, 0.02, (0.0, 0.0, 0.0)),
            RepFrameSample(0.30, 145.0, 145.0, 145.0, 145.0, 145.0, 82.5, 0.02, (0.0, 0.0, 0.0)),
            RepFrameSample(0.45, 145.0, 145.0, 145.0, 145.0, 145.0, 83.0, 0.02, (0.0, 0.0, 0.0)),
            RepFrameSample(0.60, 100.0, 100.0, 100.0, 100.0, 100.0, 85.0, 0.04, (0.0, 0.0, 0.0)),
            RepFrameSample(0.75, 100.0, 100.0, 100.0, 100.0, 100.0, 85.1, 0.04, (0.0, 0.0, 0.0)),
            RepFrameSample(0.90, 100.0, 100.0, 100.0, 100.0, 100.0, 85.2, 0.04, (0.0, 0.0, 0.0)),
            RepFrameSample(1.05, 125.0, 125.0, 125.0, 125.0, 125.0, 86.0, 0.03, (0.0, 0.0, 0.0)),
            RepFrameSample(1.20, 125.0, 125.0, 125.0, 125.0, 125.0, 86.1, 0.03, (0.0, 0.0, 0.0)),
            RepFrameSample(1.35, 125.0, 125.0, 125.0, 125.0, 125.0, 86.2, 0.03, (0.0, 0.0, 0.0)),
            RepFrameSample(1.50, 138.0, 138.0, 138.0, 138.0, 138.0, 87.0, 0.02, (0.0, 0.0, 0.0)),
            RepFrameSample(1.65, 138.0, 138.0, 138.0, 138.0, 138.0, 87.1, 0.02, (0.0, 0.0, 0.0)),
            RepFrameSample(1.80, 138.0, 138.0, 138.0, 138.0, 138.0, 87.2, 0.02, (0.0, 0.0, 0.0)),
        ]
        for sample in samples:
            detector.update(sample)
        final_rep = detector.finalize_current_rep(samples[-1].timestamp)
        self.assertIsNotNone(final_rep)
        self.assertEqual(detector.rep_count, 1)
        self.assertGreaterEqual(final_rep.duration_seconds, 0.8)

    def test_front_view_counts_reps_and_assesses_alignment_and_symmetry(self):
        reps, _debug, summary = analyze_pose_sequence(_synthetic_squat_sequence("front"))
        self.assertEqual(summary["camera_view"], "front")
        self.assertGreaterEqual(len(reps), 4)
        self.assertGreater(reps[0]["form_scores"]["knee_alignment"], 85)
        self.assertGreater(reps[0]["form_scores"]["symmetry"], 90)
        self.assertIsNone(reps[0]["form_scores"]["knee_depth"])
        self.assertIsNone(reps[0]["form_scores"]["back_posture"])

    def test_front_view_flags_inward_knee_collapse(self):
        reps, _debug, _summary = analyze_pose_sequence(
            _synthetic_squat_sequence("front", inward_knee_shift=0.10)
        )
        self.assertTrue(reps)
        self.assertLess(reps[0]["form_scores"]["knee_alignment"], 20)

    def test_front_alignment_is_high_for_balanced_pose_and_flags_knee_collapse(self):
        good = compute_alignment(_synthetic_squat_sequence("front", duration=1)[0]["landmarks"], "front")
        self.assertGreater(good["overall"], 90)
        self.assertTrue(all(metric["assessed"] for metric in good["metrics"] if metric["key"] in {
            "knee_alignment", "shoulder_level", "hip_level", "weight_shift", "foot_ankle_alignment"
        }))

        caving_pose = _synthetic_squat_sequence("front", duration=1, inward_knee_shift=0.10)[0]["landmarks"]
        caving = compute_alignment(caving_pose, "front")
        knee_metric = next(metric for metric in caving["metrics"] if metric["key"] == "knee_alignment")
        self.assertEqual(knee_metric["status"], "Fix")
        self.assertLess(knee_metric["score"], 60)

    def test_front_alignment_flags_uneven_hips(self):
        pose = _synthetic_squat_sequence("front", duration=1)[0]["landmarks"]
        pose.as_dict()["right_hip"].y += 0.12
        alignment = compute_alignment(pose, "front")
        hip_level = next(metric for metric in alignment["metrics"] if metric["key"] == "hip_level")
        self.assertIn(hip_level["status"], {"Watch", "Fix"})

    def test_side_alignment_assesses_spine_and_not_front_metrics(self):
        good_pose = _synthetic_squat_sequence("side", duration=1)[0]["landmarks"]
        good = compute_alignment(good_pose, "side")
        self.assertEqual(good["view"], "side")
        self.assertIsNone(next(m for m in good["metrics"] if m["key"] == "knee_alignment")["score"])

        leaning_pose = _synthetic_squat_sequence(
            "side", duration=1, torso_lean_degrees=38
        )[0]["landmarks"]
        leaning = compute_alignment(leaning_pose, "side")
        torso = next(metric for metric in leaning["metrics"] if metric["key"] == "torso_lean")
        self.assertEqual(torso["status"], "Fix")

    def test_alignment_excludes_impossible_knee_drift_and_ignores_unassessed_metrics(self):
        pose = _synthetic_squat_sequence("front", duration=1, inward_knee_shift=0.30)[0]["landmarks"]
        alignment = compute_alignment(pose, "front")
        knee = next(metric for metric in alignment["metrics"] if metric["key"] == "knee_alignment")
        self.assertFalse(knee["assessed"])
        summary = alignment_summary([
            {"key": "assessed", "score": 90.0, "assessed": True, "status": "Good", "label": "Metric"},
            {"key": "missing", "score": None, "assessed": False, "status": "Not assessed", "label": "Missing"},
        ], "front")
        self.assertEqual(summary["overall"], 90.0)
        self.assertEqual(summary["status"], "Good")

    def test_analyze_alignment_averages_only_assessed_rep_metrics(self):
        features = extract_rep_features(CompletedRep(1))
        features.camera_view = "front"
        features.alignment = {
            "metrics": [
                {"key": "knee", "score": 84.0, "assessed": True, "label": "Knee", "unit": "", "body_area": "knee", "value": 1.0, "status": "Good"},
                {"key": "side", "score": None, "assessed": False, "label": "Torso", "unit": "", "body_area": "lower_back", "value": None, "status": "Not assessed"},
            ]
        }
        result = analyze_alignment(features)
        self.assertEqual(result["overall"], 84.0)

    def test_side_view_assesses_depth_and_lean_not_front_metrics(self):
        reps, _debug, summary = analyze_pose_sequence(_synthetic_squat_sequence("side"))
        scores = reps[0]["form_scores"]
        self.assertEqual(summary["camera_view"], "side")
        self.assertIsNone(scores["knee_alignment"])
        self.assertIsNone(scores["symmetry"])
        self.assertGreater(scores["knee_depth"], 90)
        self.assertGreater(scores["back_posture"], 80)

    def test_side_view_flags_forward_lean(self):
        reps, _debug, _summary = analyze_pose_sequence(
            _synthetic_squat_sequence("side", torso_lean_degrees=38)
        )
        self.assertTrue(reps)
        self.assertLess(reps[0]["form_scores"]["back_posture"], 80)

    def test_angled_noisy_sequence_does_not_score_knee_alignment(self):
        frames = _synthetic_squat_sequence("angled")
        for index, frame in enumerate(frames):
            point = frame["landmarks"].as_dict()["left_knee"]
            point.z += (0.03 if index % 2 else -0.03)
        reps, _debug, summary = analyze_pose_sequence(frames)
        self.assertEqual(summary["camera_view"], "angled")
        self.assertTrue(reps)
        self.assertIsNone(reps[0]["form_scores"]["knee_alignment"])
        self.assertIsNone(reps[0]["form_scores"]["symmetry"])
        self.assertIsNone(reps[0]["form_scores"]["movement_control"])

    def test_impossible_geometry_is_tracking_error_not_a_form_fault(self):
        frames = _synthetic_squat_sequence("front", duration=2.0)
        frames[20]["landmarks"].as_dict()["left_knee"].x = 100.0
        self.assertIsNone(build_frame_sample(frames[20]["landmarks"], 20 / 30, "front"))
        _reps, debug, summary = analyze_pose_sequence(frames)
        self.assertTrue(debug[20]["tracking_error"])
        self.assertFalse(debug[20]["usable"])
        self.assertGreater(summary["tracking_quality"], 0.9)

    def test_excessive_unusable_frames_returns_friendly_tracking_error(self):
        frames = _synthetic_squat_sequence("front", duration=2.0)
        for frame in frames[:25]:
            frame["landmarks"] = None
        with self.assertRaisesRegex(
            VideoProcessingError,
            "We could not track you reliably",
        ):
            analyze_pose_sequence(frames)

    def test_video_with_no_person_raises_no_person_detected_error(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            temp_path = f.name

        try:
            _create_synthetic_video(temp_path, num_frames=30, fps=30.0)
            with self.assertRaises(NoPersonDetectedError):
                asyncio.run(self.processor.process_video(temp_path, activity="squat"))
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_video_with_synthetic_squats_produces_reps_and_persists_session(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            temp_path = f.name

        try:
            demo_steps = build_demo_sequence()
            # 600 frames at 30 fps = 20 seconds
            total_frames = 600
            fps = 30.0
            _create_synthetic_video(temp_path, num_frames=total_frames, fps=fps)

            # Mock PoseEstimator to feed demo steps sequentially
            step_idx = 0

            def fake_process(frame):
                nonlocal step_idx
                step = demo_steps[step_idx % len(demo_steps)]
                step_idx += 1
                return step.landmarks, None, True

            with patch("services.video_processor.PoseEstimator") as MockPoseEstimator:
                mock_inst = MagicMock()
                mock_inst.process.side_effect = fake_process
                MockPoseEstimator.return_value = mock_inst

                progress_calls = []

                def on_progress(p):
                    progress_calls.append(p)

                result = asyncio.run(
                    self.processor.process_video(
                        video_path=temp_path,
                        activity="squat",
                        user_id="gym-user-42",
                        database=self.database,
                        progress_callback=on_progress,
                    )
                )

                self.assertIn("session_id", result)
                self.assertIn("summary", result)
                self.assertIn("reps", result)
                self.assertTrue(len(result["reps"]) > 0)
                self.assertEqual(result["total_frames"], total_frames)
                self.assertEqual(result["fps"], fps)
                self.assertTrue(result["duration_seconds"] > 0)
                self.assertTrue(len(progress_calls) > 0)
                self.assertEqual(progress_calls[-1], 100.0)

                # Check rep details and timestamps
                for rep in result["reps"]:
                    self.assertIn("rep_number", rep)
                    self.assertIn("duration_seconds", rep)
                    self.assertIn("timestamp_sec", rep)
                    self.assertIn("form_scores", rep)
                    self.assertIn("risk", rep)
                    self.assertIn("feedback", rep)
                    self.assertIn("features", rep)
                    self.assertIn("checklist", rep)
                    self.assertTrue(rep["duration_seconds"] >= 0)
                    self.assertTrue(rep["timestamp_sec"] >= 0)

                # Check database records
                session_doc = self.database.sessions.documents[0]
                self.assertEqual(session_doc["_id"], result["session_id"])
                self.assertEqual(session_doc["user_id"], "gym-user-42")
                self.assertEqual(session_doc["activity"], "SQUAT")
                self.assertEqual(session_doc["source"], "upload")
                self.assertIsNotNone(session_doc["ended_at"])

                # Check reps in database
                self.assertEqual(len(self.database.reps.documents), len(result["reps"]))
                self.assertEqual(len(self.database.pose_frames.documents), total_frames)
                for rep_doc in self.database.reps.documents:
                    self.assertEqual(rep_doc["session_id"], result["session_id"])
                    self.assertEqual(rep_doc["user_id"], "gym-user-42")
                    self.assertTrue(rep_doc["timestamp_sec"] >= 0)
                    self.assertIn("status", rep_doc)
                    self.assertIn("form_scores", rep_doc)

                # Check report in database
                report_doc = self.database.reports.documents[0]
                self.assertEqual(report_doc["session_id"], result["session_id"])
                self.assertEqual(report_doc["user_id"], "gym-user-42")
                self.assertEqual(report_doc["totals"]["total_reps"], len(result["reps"]))

                recomputed = asyncio.run(
                    recompute_report(result["session_id"], self.database, "gym-user-42")
                )
                self.assertEqual(recomputed["totals"]["reps"], len(result["reps"]))
                self.assertEqual(recomputed["camera_view"], result["camera_view"])
                self.assertEqual(len(self.database.pose_frames.documents), total_frames)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


if __name__ == "__main__":
    unittest.main()
