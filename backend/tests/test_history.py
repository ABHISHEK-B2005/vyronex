"""Unit and integration tests for History & Biomechanical Progress (Chunk 11)."""

import unittest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient

from app import app
from core.security import create_token


class MockHistoryDB:
    def __init__(self):
        self.users_data = {}
        self.sessions_data = []
        self.reports_data = {}

        self.users = MagicMock()
        self.users.find_one = AsyncMock(side_effect=lambda q: self.users_data.get(q.get("_id")))

        self.sessions = MagicMock()
        self.sessions.find = MagicMock(side_effect=self._find_sessions)

        self.reports = MagicMock()
        self.reports.find_one = AsyncMock(side_effect=lambda q: self.reports_data.get(q.get("session_id")))

    def _find_sessions(self, query):
        uid = query.get("user_id")
        matched = [s for s in self.sessions_data if s.get("user_id") == uid]
        if "activity" in query:
            matched = [s for s in matched if s.get("activity") == query["activity"]]
        cursor = MagicMock()
        cursor.sort = MagicMock(return_value=cursor)
        cursor.limit = MagicMock(return_value=cursor)
        cursor.to_list = AsyncMock(return_value=matched)
        return cursor


class HistoryRouteTests(unittest.TestCase):
    def setUp(self):
        self.db = MockHistoryDB()
        app.state.database = self.db
        self.client = TestClient(app)

        self.user_id = "athlete_history_01"
        self.token = create_token(self.user_id, "athlete")
        self.db.users_data[self.user_id] = {
            "_id": self.user_id,
            "name": "Morgan Progress",
            "email": "morgan@example.com",
            "role": "athlete",
        }

        # Seed 4 sessions across days with recurring knee weakness
        for i in range(1, 5):
            sid = f"sess_hist_{i}"
            sess = {
                "_id": sid,
                "user_id": self.user_id,
                "activity": "squat",
                "source": "live",
                "started_at": datetime(2026, 10, i, 10, 0, tzinfo=timezone.utc),
                "avg_form_score": 75.0 + i * 2,
                "overall_risk": "MODERATE",
            }
            self.db.sessions_data.append(sess)
            self.db.reports_data[sid] = {
                "session_id": sid,
                "user_id": self.user_id,
                "activity": "squat",
                "accuracy_pct": 70.0 + i * 5,
                "avg_form_score": 75.0 + i * 2,
                "totals": {"reps": 15, "correct": 10, "defective": 4, "high_risk": 1},
                "body_area_breakdown": {"knee": 75, "lower_back": 25},
                "fatigue": {"drop_pct": 14.0},
            }

    def test_history_endpoint_analytics_and_recurring_weakness(self):
        """History returns session list, charts, recurring weakness, badges, and risk prediction."""
        res = self.client.get(
            "/api/history",
            headers={"Authorization": f"Bearer {self.token}"},
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()

        self.assertEqual(data["count"], 4)
        self.assertTrue(data["streak_days"] >= 1)

        # Recurring weakness check
        rec = data["recurring_weakness"]
        self.assertIsNotNone(rec)
        self.assertEqual(rec["area"], "knee")
        self.assertIn("Knee flagged in", rec["message"])

        # Predicted future risk check
        future_risk = data["predicted_future_risk"]
        self.assertIn(future_risk["level"], ("LOW", "MODERATE", "ELEVATED"))
        self.assertIn("Movement-risk screening", future_risk["disclaimer"])

        # Badges
        badges = data["badges"]
        self.assertTrue(len(badges) >= 1)
        self.assertTrue(any(b["id"] == "first_step" for b in badges))

        # Charts data
        charts = data["charts"]
        self.assertEqual(len(charts["accuracy_over_time"]), 4)
        self.assertEqual(len(charts["form_score_over_time"]), 4)


if __name__ == "__main__":
    unittest.main()
