"""Integration and route tests for upload and video analysis endpoints."""

import io
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient

from core.deps import get_current_user, get_current_user_optional
from routes.upload import _jobs, router as upload_router
from src.demo_mode import build_demo_sequence


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

    def find(self, query, projection=None):
        matched = [doc for doc in self.documents if all(doc.get(k) == v for k, v in query.items())]
        if projection:
            included = {key for key, enabled in projection.items() if enabled}
            matched = [
                {key: value for key, value in doc.items() if key in included}
                for doc in matched
            ]

        class FakeCursor:
            def __init__(self, items):
                self.items = items

            async def to_list(self, length=1000):
                return list(self.items[:length])

            def sort(self, key, direction=1):
                self.items.sort(key=lambda item: item.get(key, 0), reverse=direction < 0)
                return self

        return FakeCursor(matched)


def _generate_mp4_bytes(num_frames: int = 60, fps: float = 30.0, width: int = 320, height: int = 240) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        temp_path = f.name
    try:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))
        for _ in range(num_frames):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            writer.write(frame)
        writer.release()
        with open(temp_path, "rb") as rf:
            return rf.read()
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


class UploadRouteTests(unittest.TestCase):
    def setUp(self):
        _jobs.clear()
        self.database = SimpleNamespace(
            users=FakeCollection([{"_id": "test-user-1", "role": "gym"}]),
            sessions=FakeCollection(),
            reps=FakeCollection(),
            reports=FakeCollection(),
            pose_frames=FakeCollection(),
        )
        self.app = FastAPI()
        self.app.state.database = self.database
        self.app.include_router(upload_router)
        self.app.dependency_overrides[get_current_user_optional] = lambda: {"_id": "test-user-1", "role": "gym"}
        self.app.dependency_overrides[get_current_user] = lambda: {"_id": "test-user-1", "role": "gym"}
        self.client = TestClient(self.app)

    def test_upload_pdf_rejected_with_415(self):
        pdf_bytes = b"%PDF-1.4 sample pdf content"
        response = self.client.post(
            "/api/upload/analyze",
            files={"file": ("document.pdf", pdf_bytes, "application/pdf")},
            data={"activity": "squat"},
        )
        self.assertEqual(response.status_code, 415)
        self.assertIn("Unsupported media type", response.json()["detail"])

    def test_upload_unsupported_extension_rejected_with_415(self):
        text_bytes = b"Hello world"
        response = self.client.post(
            "/api/upload/analyze",
            files={"file": ("workout.txt", text_bytes, "text/plain")},
            data={"activity": "squat"},
        )
        self.assertEqual(response.status_code, 415)

    def test_non_squat_upload_is_rejected_instead_of_scored_as_a_squat(self):
        response = self.client.post(
            "/api/upload/analyze",
            files={"file": ("jump.mp4", b"not processed", "video/mp4")},
            data={"activity": "long_jump"},
        )

        self.assertEqual(response.status_code, 422)
        self.assertIn("supports squats only", response.json()["detail"])

    def test_upload_status_unknown_job_returns_404(self):
        response = self.client.get("/api/upload/status/unknown-job-id-999")
        self.assertEqual(response.status_code, 404)

    def test_debug_endpoint_returns_json_and_csv_rows(self):
        session_id = "debug-session"
        self.database.sessions.documents.append({
            "_id": session_id,
            "user_id": "test-user-1",
        })
        self.database.pose_frames.documents.append({
            "session_id": session_id,
            "timestamp": 0.1,
            "knee_angle": 120.0,
            "torso_angle": 80.0,
            "view": "front",
            "visibility": 1.0,
            "usable": True,
            "tracking_error": False,
            "rep_phase": "Descending",
            "landmarks": {"nose": {"x": 0.1}},
        })

        json_response = self.client.get(f"/api/upload/debug/{session_id}")
        csv_response = self.client.get(f"/api/upload/debug/{session_id}?format=csv")

        self.assertEqual(json_response.status_code, 200)
        self.assertEqual(len(json_response.json()["frames"]), 1)
        self.assertNotIn("landmarks", json_response.json()["frames"][0])
        self.assertEqual(csv_response.status_code, 200)
        self.assertIn("timestamp,knee_angle,torso_angle", csv_response.text)

    def test_upload_clip_with_no_person_reports_clean_error(self):
        video_bytes = _generate_mp4_bytes(num_frames=30, fps=30.0)

        response = self.client.post(
            "/api/upload/analyze",
            files={"file": ("empty_room.mp4", video_bytes, "video/mp4")},
            data={"activity": "squat"},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("job_id", data)
        job_id = data["job_id"]

        # Poll status
        status_res = self.client.get(f"/api/upload/status/{job_id}")
        self.assertEqual(status_res.status_code, 200)
        status_data = status_res.json()
        self.assertEqual(status_data["status"], "failed")
        self.assertIn(
            "We could not track you reliably",
            status_data["error"],
        )

    def test_upload_20_second_squat_clip_completes_and_returns_session(self):
        # 600 frames at 30 fps = 20 seconds
        total_frames = 600
        fps = 30.0
        video_bytes = _generate_mp4_bytes(num_frames=total_frames, fps=fps)

        demo_steps = build_demo_sequence()
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

            response = self.client.post(
                "/api/upload/analyze",
                files={"file": ("20s_squat.mp4", video_bytes, "video/mp4")},
                data={"activity": "squat"},
            )
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertIn("job_id", data)
            job_id = data["job_id"]

            # Poll status
            status_res = self.client.get(f"/api/upload/status/{job_id}")
            self.assertEqual(status_res.status_code, 200)
            status_data = status_res.json()

            self.assertEqual(status_data["status"], "completed")
            self.assertEqual(status_data["progress"], 100.0)
            self.assertIsNotNone(status_data["session_id"])
            self.assertIsNotNone(status_data["report_id"])
            self.assertIsNotNone(status_data["summary"])
            self.assertIsNotNone(status_data["reps"])
            self.assertTrue(len(status_data["reps"]) > 0)

            # Validate reps contain real timestamps and matching live mode format
            for rep in status_data["reps"]:
                self.assertIn("rep_number", rep)
                self.assertIn("duration_seconds", rep)
                self.assertIn("timestamp_sec", rep)
                self.assertIn("form_scores", rep)
                self.assertIn("risk", rep)
                self.assertIn("feedback", rep)
                self.assertIn("features", rep)
                self.assertTrue(rep["duration_seconds"] > 0)
                self.assertTrue(rep["timestamp_sec"] > 0)

            # Validate database records were created (Chunk 7)
            session_doc = self.database.sessions.documents[0]
            self.assertEqual(session_doc["_id"], status_data["session_id"])
            self.assertEqual(session_doc["source"], "upload")
            self.assertEqual(session_doc["user_id"], "test-user-1")

            self.assertEqual(len(self.database.reps.documents), len(status_data["reps"]))
            self.assertEqual(len(self.database.reports.documents), 1)


if __name__ == "__main__":
    unittest.main()
