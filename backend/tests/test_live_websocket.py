import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from core.security import create_access_token
from routes.ws import router as live_router


JWT_SECRET = "test-secret-that-is-long-enough-for-hmac"


class FakeCollection:
    def __init__(self, documents=None):
        self.documents = list(documents or [])
        self.updates = []

    async def find_one(self, query):
        return next(
            (doc for doc in self.documents if doc.get("_id") == query.get("_id")),
            None,
        )

    async def insert_one(self, document):
        self.documents.append(dict(document))

    async def update_one(self, query, update):
        self.updates.append((query, update))

    def find(self, query):
        matches = [
            document
            for document in self.documents
            if all(document.get(key) == value for key, value in query.items())
        ]

        class FakeCursor:
            async def to_list(self, length=None):
                return list(matches if length is None else matches[:length])

        return FakeCursor()


class LiveWebSocketTests(unittest.TestCase):
    def setUp(self):
        self.user = {"_id": "gym-user", "role": "gym"}
        self.database = SimpleNamespace(
            users=FakeCollection([self.user]),
            sessions=FakeCollection(),
            reps=FakeCollection(),
            reports=FakeCollection(),
        )
        self.app = FastAPI()
        self.app.state.database = self.database
        self.app.include_router(live_router)
        self.client = TestClient(self.app)
        self.secret_patch = patch.dict(os.environ, {"JWT_SECRET": JWT_SECRET})
        self.secret_patch.start()
        self.addCleanup(self.secret_patch.stop)
        self.token = create_access_token("gym-user", "gym", JWT_SECRET)

    def test_invalid_token_is_closed_with_4401(self):
        with self.client.websocket_connect(
            "/ws/live?token=not-a-valid-token&activity=squat",
        ) as websocket:
            with self.assertRaises(WebSocketDisconnect) as disconnected:
                websocket.receive_json()

        self.assertEqual(disconnected.exception.code, 4401)
        self.assertEqual(self.database.sessions.documents, [])

    def test_frames_return_guidance_and_end_persists_session_and_report(self):
        with self.client.websocket_connect(
            f"/ws/live?token={self.token}&activity=squat",
        ) as websocket:
            ready = websocket.receive_json()
            self.assertEqual(ready["type"], "ready")

            websocket.send_json({"landmarks": None, "t": 1.0})
            frame = websocket.receive_json()
            self.assertEqual(frame["type"], "frame")
            self.assertEqual(frame["phase"], "Standing")
            self.assertEqual(frame["rep_count"], 0)
            self.assertFalse(frame["landmarks_ok"])
            self.assertIn("Step back", frame["cue"])

            websocket.send_json({"type": "end"})
            ended = websocket.receive_json()

        self.assertEqual(ended["type"], "ended")
        self.assertEqual(ended["summary"]["total_reps"], 0)
        self.assertEqual(len(self.database.sessions.documents), 1)
        self.assertIsNotNone(self.database.sessions.updates[0][1]["$set"]["ended_at"])
        self.assertEqual(len(self.database.reports.documents), 1)
        self.assertEqual(
            self.database.reports.documents[0]["session_id"],
            ready["session_id"],
        )

    def test_five_squats_count_and_knee_cave_returns_warning_cue(self):
        with self.client.websocket_connect(
            f"/ws/live?token={self.token}&activity=squat",
        ) as websocket:
            websocket.receive_json()
            timestamp = 0.0
            latest = None
            completed_reps = []

            def send_frames(knee_y, knee_shift=0, shoulder_shift=0, repeats=5):
                nonlocal timestamp, latest
                for _ in range(repeats):
                    landmarks = {
                        "nose": {"x": 400, "y": 40},
                        "left_shoulder": {"x": 300 + shoulder_shift, "y": 100},
                        "right_shoulder": {"x": 500 + shoulder_shift, "y": 100},
                        "left_elbow": {"x": 270, "y": 150},
                        "right_elbow": {"x": 530, "y": 150},
                        "left_wrist": {"x": 260, "y": 200},
                        "right_wrist": {"x": 540, "y": 200},
                        "left_hip": {"x": 300, "y": 200},
                        "right_hip": {"x": 500, "y": 200},
                        "left_knee": {
                            "x": 350 + knee_shift,
                            "y": knee_y,
                        },
                        "right_knee": {
                            "x": 450 - knee_shift,
                            "y": knee_y,
                        },
                        "left_ankle": {"x": 400, "y": 400},
                        "right_ankle": {"x": 400, "y": 400},
                    }
                    websocket.send_json({
                        "landmarks": landmarks,
                        "t": timestamp,
                    })
                    timestamp += 1 / 15
                    latest = websocket.receive_json()
                    if latest["completed_rep"] is not None:
                        completed_reps.append(latest["completed_rep"])

            send_frames(300, repeats=4)
            for rep_index in range(5):
                send_frames(210)
                is_knee_cave_rep = rep_index == 2
                send_frames(
                    190,
                    knee_shift=40 if is_knee_cave_rep else 0,
                    shoulder_shift=150 if is_knee_cave_rep else 0,
                )
                send_frames(210)
                send_frames(300)

            self.assertEqual(latest["rep_count"], 5)
            self.assertEqual(len(completed_reps), 5)
            cave_rep = completed_reps[2]
            self.assertIn(cave_rep["risk"]["risk_level"], {"MODERATE", "HIGH"})
            self.assertEqual(
                cave_rep["feedback"]["warning_tag"],
                "KNEE ALIGNMENT WARNING",
            )
            self.assertIn("knee", cave_rep["feedback"]["recommendation"].lower())

            websocket.send_json({"type": "end"})
            websocket.receive_json()

    def test_non_gym_user_is_rejected(self):
        self.user["role"] = "athlete"
        athlete_token = create_access_token("gym-user", "athlete", JWT_SECRET)
        with self.client.websocket_connect(
            f"/ws/live?token={athlete_token}&activity=squat",
        ) as websocket:
            with self.assertRaises(WebSocketDisconnect) as disconnected:
                websocket.receive_json()

        self.assertEqual(disconnected.exception.code, 4403)


if __name__ == "__main__":
    unittest.main()
