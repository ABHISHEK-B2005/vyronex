"""Unit and integration tests for Rep Feedback, Expert Review, and Retraining Loop (Chunk 8E)."""

import os
import unittest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from app import app
from core.security import create_token
from ml.retrain import execute_retraining_pipeline, load_model_registry


class TestDatabase:
    """Mock MongoDB database for testing."""
    def __init__(self):
        self.users_data = {}
        self.reps_data = {}
        self.sessions_data = {}
        self.reports_data = {}

        self.users = MagicMock()
        self.users.find_one = AsyncMock(side_effect=self._user_find_one)
        self.users.update_one = AsyncMock(side_effect=self._user_update_one)
        self.users.create_index = AsyncMock()

        self.reps = MagicMock()
        self.reps.find_one = AsyncMock(side_effect=self._rep_find_one)
        self.reps.update_one = AsyncMock(side_effect=self._rep_update_one)
        self.reps.find = MagicMock(side_effect=self._rep_find)

        self.sessions = MagicMock()
        self.sessions.find_one = AsyncMock(side_effect=self._session_find_one)

        self.reports = MagicMock()
        self.reports.find_one = AsyncMock(return_value=None)
        self.reports.update_one = AsyncMock()

    async def _user_find_one(self, query):
        if "_id" in query:
            return self.users_data.get(query["_id"])
        if "email" in query:
            for u in self.users_data.values():
                if u.get("email") == query["email"]:
                    return u
        return None

    async def _user_update_one(self, query, update):
        uid = query.get("_id")
        if uid in self.users_data:
            self.users_data[uid].update(update.get("$set", {}))

    async def _rep_find_one(self, query):
        return self.reps_data.get(query.get("_id"))

    async def _rep_update_one(self, query, update):
        rid = query.get("_id")
        if rid in self.reps_data:
            self.reps_data[rid].update(update.get("$set", {}))

    def _rep_find(self, query=None):
        cursor = MagicMock()
        res = list(self.reps_data.values())
        cursor.sort = MagicMock(return_value=cursor)
        cursor.limit = MagicMock(return_value=cursor)
        cursor.to_list = AsyncMock(return_value=res)
        return cursor

    async def _session_find_one(self, query):
        return self.sessions_data.get(query.get("_id"))


class FeedbackAndExpertTests(unittest.TestCase):
    def setUp(self):
        self.db = TestDatabase()
        app.state.database = self.db
        self.client = TestClient(app)

        # Seed athlete user
        self.athlete_id = "athlete_001"
        self.athlete_token = create_token(self.athlete_id, "athlete")
        self.db.users_data[self.athlete_id] = {
            "_id": self.athlete_id,
            "name": "Alex Runner",
            "email": "alex@example.com",
            "role": "athlete",
            "data_consent": True,
        }

        # Seed expert user
        self.expert_id = "expert_001"
        self.expert_token = create_token(self.expert_id, "expert")
        self.db.users_data[self.expert_id] = {
            "_id": self.expert_id,
            "name": "Dr. Smith Coach",
            "email": "smith@coach.com",
            "role": "expert",
        }

        # Seed a session and reps
        self.session_id = "sess_001"
        self.db.sessions_data[self.session_id] = {
            "_id": self.session_id,
            "user_id": self.athlete_id,
            "activity": "squat",
            "started_at": datetime.now(timezone.utc),
        }

        self.rep_id = "rep_001"
        self.db.reps_data[self.rep_id] = {
            "_id": self.rep_id,
            "user_id": self.athlete_id,
            "session_id": self.session_id,
            "rep_number": 1,
            "timestamp_sec": 5.0,
            "status": "defective",
            "defect_type": "knee_valgus",
            "body_area": "knee",
            "risk_level": "MODERATE",
            "form_scores": {"overall": 70.0},
            "features": {"knee_angle": 92.0, "torso_angle": 60.0, "knee_alignment": 0.08, "symmetry": 0.04, "movement_speed": 1.2, "rep_duration": 2.2},
            "model_version": "v1.0.0",
            "model_confidence": 0.55,  # low confidence
            "disputed": False,
            "expert_reviewed": False,
        }

    def test_submit_rep_feedback_dispute(self):
        """Athletes can vote No on rep rating, flagging it for expert review."""
        res = self.client.post(
            f"/api/feedback/rep/{self.rep_id}",
            headers={"Authorization": f"Bearer {self.athlete_token}"},
            json={"is_correct": False, "comment": "I kept my knees straight!"},
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["disputed"], True)
        self.assertEqual(data["enqueued_for_expert_review"], True)

        # Check DB updated
        saved_rep = self.db.reps_data[self.rep_id]
        self.assertEqual(saved_rep["disputed"], True)
        self.assertEqual(saved_rep["user_feedback"], "incorrect")

    def test_expert_queue_and_labeling(self):
        """Expert views queue and sets ground-truth label."""
        # 1. Fetch queue
        queue_res = self.client.get(
            "/api/expert/queue",
            headers={"Authorization": f"Bearer {self.expert_token}"},
        )
        self.assertEqual(queue_res.status_code, 200)
        queue_data = queue_res.json()
        self.assertTrue(queue_data["queue_count"] >= 1)

        # 2. Submit expert label
        label_res = self.client.post(
            f"/api/expert/label/{self.rep_id}",
            headers={"Authorization": f"Bearer {self.expert_token}"},
            json={
                "true_risk_level": "HIGH",
                "true_defect_type": "knee_valgus",
                "notes": "Severe inward shift at bottom of squat.",
            },
        )
        self.assertEqual(label_res.status_code, 200)
        label_data = label_res.json()
        self.assertEqual(label_data["expert_label"], "HIGH")
        self.assertEqual(label_data["status"], "confirmed_for_retraining")

        # Verify DB updated
        saved = self.db.reps_data[self.rep_id]
        self.assertEqual(saved["expert_reviewed"], True)
        self.assertEqual(saved["expert_label"], "HIGH")
        self.assertEqual(saved["disputed"], False)

    def test_retraining_pipeline_execution(self):
        """Safe retraining executes person-wise splits and registers version."""
        expert_reps = [self.db.reps_data[self.rep_id]]
        result = execute_retraining_pipeline(expert_reps=expert_reps, exercise="squat")
        
        self.assertIn(result["status"], ("activated", "rejected", "skipped"))
        self.assertIn("metrics", result)

        registry = load_model_registry()
        self.assertTrue(len(registry) >= 1)
        latest = registry[-1]
        self.assertIn("version", latest)
        self.assertIn("metrics", latest)


if __name__ == "__main__":
    unittest.main()
