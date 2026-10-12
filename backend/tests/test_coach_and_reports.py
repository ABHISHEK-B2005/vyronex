"""Unit and integration tests for AI Coach, Report Builder, and PDF Export (Chunks 9 & 10)."""

import asyncio
import io
import unittest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from app import app
from core.security import create_token
from services.coach import generate_post_workout_advice
from services.pdf_report import generate_pdf_report
from services.report_builder import build_report


class MockDB:
    def __init__(self):
        self.users_data = {}
        self.sessions_data = {}
        self.reps_data = {}
        self.reports_data = {}
        self.chat_messages = []

        self.users = MagicMock()
        self.users.find_one = AsyncMock(side_effect=lambda q: self.users_data.get(q.get("_id")))

        self.sessions = MagicMock()
        self.sessions.find_one = AsyncMock(side_effect=lambda q: self.sessions_data.get(q.get("_id")))

        self.reps = MagicMock()
        self.reps.find = MagicMock(side_effect=self._find_reps)

        self.reports = MagicMock()
        self.reports.find_one = AsyncMock(side_effect=lambda q: self.reports_data.get(q.get("session_id")))
        self.reports.find = MagicMock(side_effect=self._find_reports)
        self.reports.update_one = AsyncMock(side_effect=self._update_report)

        self.chat_messages_coll = MagicMock()
        self.chat_messages_coll.insert_one = AsyncMock()
        self.chat_messages = self.chat_messages_coll

    def _find_reps(self, query):
        sid = query.get("session_id")
        reps = [r for r in self.reps_data.values() if r.get("session_id") == sid]
        cursor = MagicMock()
        cursor.sort = MagicMock(return_value=cursor)
        cursor.to_list = AsyncMock(return_value=reps)
        return cursor

    def _find_reports(self, query):
        uid = query.get("user_id")
        reps = [r for r in self.reports_data.values() if r.get("user_id") == uid]
        cursor = MagicMock()
        cursor.sort = MagicMock(return_value=cursor)
        cursor.limit = MagicMock(return_value=cursor)
        cursor.to_list = AsyncMock(return_value=reps)
        return cursor

    async def _update_report(self, query, update, upsert=False):
        sid = query.get("session_id")
        self.reports_data[sid] = update.get("$set", {})


class CoachAndReportsTests(unittest.TestCase):
    def setUp(self):
        self.db = MockDB()
        app.state.database = self.db
        self.client = TestClient(app)

        self.user_id = "user_coach_01"
        self.token = create_token(self.user_id, "athlete")
        self.db.users_data[self.user_id] = {
            "_id": self.user_id,
            "name": "Jordan Athlete",
            "email": "jordan@example.com",
            "role": "athlete",
            "injury_history": ["lower_back"],
        }

        self.session_id = "sess_report_01"
        self.db.sessions_data[self.session_id] = {
            "_id": self.session_id,
            "user_id": self.user_id,
            "activity": "squat",
            "source": "live",
            "started_at": datetime.now(timezone.utc),
            "ended_at": datetime.now(timezone.utc),
        }

        # Seed 10 reps: 6 correct, 3 defective (knee/back), 1 high_risk
        for i in range(1, 11):
            rid = f"rep_{i}"
            if i <= 6:
                status, risk, fs, defect, area = "correct", "LOW", 88.0, None, None
            elif i <= 9:
                status, risk, fs, defect, area = "defective", "MODERATE", 68.0, "back_posture", "lower_back"
            else:
                status, risk, fs, defect, area = "high_risk", "HIGH", 45.0, "knee_valgus", "knee"

            self.db.reps_data[rid] = {
                "_id": rid,
                "session_id": self.session_id,
                "user_id": self.user_id,
                "rep_number": i,
                "timestamp_sec": float(i * 3.0),
                "status": status,
                "defect_type": defect,
                "body_area": area,
                "risk_level": risk,
                "form_scores": {"overall": fs},
                "feedback": f"Rep {i} feedback",
            }

    def test_report_builder_non_overlapping_totals_and_fatigue(self):
        """Confirm totals add up exactly: correct + defective + high_risk == reps."""
        report = self.client.get(
            f"/api/reports/{self.session_id}",
            headers={"Authorization": f"Bearer {self.token}"},
        ).json()

        totals = report["totals"]
        self.assertEqual(totals["reps"], 10)
        self.assertEqual(totals["correct"], 6)
        self.assertEqual(totals["defective"], 3)
        self.assertEqual(totals["high_risk"], 1)
        self.assertEqual(totals["correct"] + totals["defective"] + totals["high_risk"], totals["reps"])

        # Accuracy
        self.assertEqual(report["accuracy_pct"], 60.0)
        self.assertEqual(report["overall_risk"], "HIGH")

        # Body area breakdown: 3 lower_back (75%), 1 knee (25%)
        self.assertIn("lower_back", report["body_area_breakdown"])
        self.assertTrue(report["body_area_breakdown"]["lower_back"] >= 50)

        # Fatigue: 1st half score vs 2nd half score
        fatigue = report["fatigue"]
        self.assertTrue("first_half_score" in fatigue)
        self.assertTrue("second_half_score" in fatigue)
        self.assertTrue("drop_pct" in fatigue)

    def test_report_uses_persisted_risk_and_video_duration(self):
        self.db.sessions_data[self.session_id]["overall_risk"] = "moderate"
        self.db.sessions_data[self.session_id]["duration_sec"] = 3.4
        self.db.reps_data = {
            "rep_1": {
                "_id": "rep_1",
                "session_id": self.session_id,
                "user_id": self.user_id,
                "rep_number": 1,
                "timestamp_sec": 2.58,
                "status": "defective",
                "defect_type": "knee_alignment",
                "body_area": "knee",
                "risk_level": "MODERATE",
                "form_scores": {"overall": 55.4},
                "feedback": "Keep the knee tracking in line with the foot.",
            },
        }

        report = asyncio.run(
            build_report(self.db, self.session_id, self.user_id)
        )

        self.assertEqual(report["overall_risk"], "MODERATE")
        self.assertEqual(report["duration_sec"], 3.4)

    def test_report_uses_rep_form_score_threshold(self):
        self.db.reps_data = {
            "rep_1": {
                "_id": "rep_1",
                "session_id": self.session_id,
                "user_id": self.user_id,
                "rep_number": 1,
                "timestamp_sec": 1.0,
                "status": "correct",
                "risk_level": "LOW",
                "form_scores": {"overall": 72.0},
            },
        }

        report = asyncio.run(build_report(self.db, self.session_id, self.user_id))

        self.assertEqual(report["totals"]["correct"], 1)
        self.assertEqual(report["totals"]["defective"], 0)

    def test_report_keeps_unassessed_metrics_null(self):
        self.db.sessions_data[self.session_id]["camera_view"] = "angled"
        self.db.reps_data = {
            "rep_1": {
                "_id": "rep_1",
                "session_id": self.session_id,
                "user_id": self.user_id,
                "rep_number": 1,
                "timestamp_sec": 1.0,
                "status": "correct",
                "risk_level": "LOW",
                "camera_view": "angled",
                "features": {
                    "knee_alignment": None,
                    "knee_symmetry": None,
                    "hip_symmetry": 3.5,
                },
                "form_scores": {
                    "overall": 90.0,
                    "knee_alignment": None,
                    "knee_depth": 90.0,
                    "back_posture": None,
                    "movement_control": None,
                    "symmetry": None,
                },
            },
        }

        report = asyncio.run(build_report(self.db, self.session_id, self.user_id))

        self.assertEqual(report["camera_view"], "angled")
        self.assertIsNone(report["rep_table"][0]["form_scores"]["knee_alignment"])
        self.assertEqual(
            report["rep_table"][0]["features"],
            {
                "knee_alignment": None,
                "knee_symmetry": None,
                "hip_symmetry": 3.5,
            },
        )
        self.assertNotIn("knee_alignment", report["defect_counts"])

    def test_report_includes_alignment_summary_and_unassessed_metrics(self):
        self.db.reps_data = {
            "rep_1": {
                "_id": "rep_1",
                "session_id": self.session_id,
                "user_id": self.user_id,
                "rep_number": 1,
                "timestamp_sec": 1.0,
                "status": "correct",
                "risk_level": "LOW",
                "camera_view": "front",
                "alignment": {
                    "overall": 90.0,
                    "status": "Good",
                    "view": "front",
                    "metrics": [
                        {
                            "key": "knee_alignment",
                            "label": "Knee alignment",
                            "value": 1.0,
                            "unit": "% leg length",
                            "score": 90.0,
                            "status": "Good",
                            "assessed": True,
                            "body_area": "knee",
                        },
                        {
                            "key": "torso_lean",
                            "label": "Torso lean",
                            "value": None,
                            "unit": "degrees",
                            "score": None,
                            "status": "Not assessed",
                            "assessed": False,
                            "body_area": "lower_back",
                        },
                    ],
                    "issues": [],
                },
                "form_scores": {"overall": 90.0},
            },
        }

        report = asyncio.run(build_report(self.db, self.session_id, self.user_id))

        self.assertEqual(report["alignment"]["overall"], 90.0)
        metrics = {metric["key"]: metric for metric in report["alignment"]["metrics"]}
        self.assertEqual(metrics["knee_alignment"]["score"], 90.0)
        self.assertIsNone(metrics["torso_lean"]["score"])
        self.assertFalse(metrics["torso_lean"]["assessed"])

    def test_pdf_report_generation(self):
        """Confirm PDF generation produces valid PDF bytes."""
        res = self.client.get(
            f"/api/reports/{self.session_id}/pdf",
            headers={"Authorization": f"Bearer {self.token}"},
        )
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.headers["content-type"], "application/pdf")
        self.assertTrue(len(res.content) > 1000)
        self.assertTrue(res.content.startswith(b"%PDF"))

    def test_coach_chat_and_recommendations(self):
        """Test chat assistant follows safety rules and mentions disclaimer."""
        # 1. Ask about back pain -> should recommend medical professional
        res = self.client.post(
            "/api/coach/chat",
            headers={"Authorization": f"Bearer {self.token}"},
            json={"message": "Why does my back hurt after squats?"},
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(any(w in data["reply"].lower() for w in ["physio", "doctor", "professional", "medical"]))
        self.assertIn("Movement-risk screening only", data["disclaimer"])

        # 2. Recommendations endpoint
        rec_res = self.client.get(
            f"/api/coach/recommendations/{self.session_id}",
            headers={"Authorization": f"Bearer {self.token}"},
        )
        self.assertEqual(rec_res.status_code, 200)
        rec_data = rec_res.json()
        self.assertTrue(len(rec_data["recommendations"]) > 0)
        self.assertTrue(len(rec_data["corrective_exercises"]) > 0)


if __name__ == "__main__":
    unittest.main()
