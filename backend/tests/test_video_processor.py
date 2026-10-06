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
    VideoProcessor,
    build_checklist,
    serialize_rep,
)
from src.demo_mode import build_demo_sequence
from src.form_analysis import FormScores
from src.pipeline import RepResult
from src.risk_engine import RiskResult
from src.squat_detector import CompletedRep


class FakeCollection:
    def __init__(self, documents=None):
        self.documents = list(documents or [])
        self.updates = []

    async def insert_one(self, document):
        self.documents.append(dict(document))

    async def update_one(self, query, update):
        self.updates.append((query, update))
        for doc in self.documents:
            match = all(doc.get(k) == v for k, v in query.items())
            if match:
                if "$set" in update:
                    doc.update(update["$set"])


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

    def find(self, query):
        matched = [doc for doc in self.documents if all(doc.get(k) == v for k, v in query.items())]

        class FakeCursor:
            def __init__(self, items):
                self.items = items

            async def to_list(self, length=1000):
                return list(self.items[:length])

        return FakeCursor(matched)


def _create_synthetic_video(file_path: str, num_frames: int = 60, fps: float = 30.0, width: int = 640, height: int = 480):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(file_path, fourcc, fps, (width, height))
    for i in range(num_frames):
        # Create a plain frame
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()


class VideoProcessorTests(unittest.TestCase):
    def setUp(self):
        self.database = SimpleNamespace(
            users=FakeCollection(),
            sessions=FakeCollection(),
            reps=FakeCollection(),
            reports=FakeCollection(),
        )
        self.processor = VideoProcessor(frame_step=2)

    def test_nonexistent_video_file_raises_error(self):
        with self.assertRaises(InvalidVideoFileError):
            asyncio.run(self.processor.process_video("non_existent_file_path_123.mp4"))

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
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


if __name__ == "__main__":
    unittest.main()
