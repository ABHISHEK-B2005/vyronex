import asyncio
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace

from services.session_service import (
    classify_rep_status,
    create_session,
    finish_session,
    get_body_area,
    get_session_summary,
    record_completed_rep,
)


class FakeCollection:
    def __init__(self, documents=None):
        self.documents = list(documents or [])
        self.updates = []

    async def insert_one(self, document):
        self.documents.append(document)

    async def update_one(self, query, update):
        self.updates.append((query, update))

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
        matched = []
        for doc in self.documents:
            match = True
            for k, v in query.items():
                if doc.get(k) != v:
                    match = False
                    break
            if match:
                matched.append(doc)

        class FakeCursor:
            def __init__(self, items):
                self.items = items

            async def to_list(self, length=1000):
                return list(self.items[:length])

        return FakeCursor(matched)


class SessionServiceTests(unittest.TestCase):
    def setUp(self):
        self.database = SimpleNamespace(sessions=FakeCollection(), reps=FakeCollection())

    def test_create_session_persists_owner_activity_and_source(self):
        session_id = asyncio.run(create_session(self.database, "user-1", "SQUAT", "live"))

        stored = self.database.sessions.documents[0]
        self.assertEqual(stored["_id"], session_id)
        self.assertEqual(stored["user_id"], "user-1")
        self.assertEqual(stored["activity"], "SQUAT")
        self.assertEqual(stored["source"], "live")
        self.assertIsNone(stored["ended_at"])

    def test_classify_rep_status_rules(self):
        # Rule: high_risk -> risk HIGH
        self.assertEqual(
            classify_rep_status("HIGH", {"knee_alignment": 82, "back_posture": 31}),
            "high_risk",
        )
        self.assertEqual(
            classify_rep_status("high", {"knee_alignment": 90, "back_posture": 90}),
            "high_risk",
        )

        # Rule: defective -> risk MODERATE, or any score < 70
        self.assertEqual(
            classify_rep_status("MODERATE", {"knee_alignment": 80, "back_posture": 80}),
            "defective",
        )
        self.assertEqual(
            classify_rep_status("LOW", {"knee_alignment": 82, "back_posture": 65}),
            "defective",
        )

        # Rule: correct -> risk LOW and every form score >= 70
        self.assertEqual(
            classify_rep_status("LOW", {"knee_alignment": 82, "back_posture": 75}),
            "correct",
        )
        self.assertEqual(
            classify_rep_status("low", {"knee_alignment": 70, "knee_depth": 70}),
            "correct",
        )

    def test_defect_to_body_area_mapping(self):
        self.assertEqual(get_body_area("knee_alignment"), "knee")
        self.assertEqual(get_body_area("knee_depth"), "knee, hip")
        self.assertEqual(get_body_area("back_posture"), "lower_back")
        self.assertEqual(get_body_area("symmetry"), "hip")
        self.assertEqual(get_body_area("movement_control"), "full_chain")
        self.assertEqual(get_body_area("elbow flare (push-up)"), "shoulder")
        self.assertEqual(get_body_area("elbow flare"), "shoulder")
        self.assertEqual(get_body_area("landing (long jump)"), "ankle, knee")
        self.assertEqual(get_body_area("landing"), "ankle, knee")
        self.assertIsNone(get_body_area(None))

    def test_record_completed_rep_persists_full_document(self):
        result = {
            "rep_number": 3,
            "timestamp_sec": 14.2,
            "risk": {"risk_level": "HIGH", "primary_issue": "back_posture"},
            "form_scores": {"knee_alignment": 82, "back_posture": 31},
            "features": {"min_knee_angle": 88, "forward_lean": 38},
            "feedback": {"recommendation": "Torso leaned forward 38 degrees..."},
        }
        stored_rep = asyncio.run(
            record_completed_rep(self.database, "user-1", "session-1", 0.0, result, timestamp_sec=14.2)
        )

        stored = self.database.reps.documents[0]
        self.assertEqual(stored["session_id"], "session-1")
        self.assertEqual(stored["rep_number"], 3)
        self.assertEqual(stored["timestamp_sec"], 14.2)
        self.assertEqual(stored["status"], "high_risk")
        self.assertEqual(stored["defect_type"], "back_posture")
        self.assertEqual(stored["body_area"], "lower_back")
        self.assertEqual(stored["risk_level"], "HIGH")
        self.assertEqual(stored["form_scores"], {"knee_alignment": 82, "back_posture": 31})
        self.assertEqual(stored["features"], {"min_knee_angle": 88, "forward_lean": 38})
        self.assertEqual(stored["feedback"], "Torso leaned forward 38 degrees...")

    def test_finish_session_updates_owner_scoped_summary(self):
        summary = {
            "total_reps": 3,
            "average_form_score": 71.5,
            "low_risk_reps": 2,
            "moderate_risk_reps": 1,
            "high_risk_reps": 0,
        }
        asyncio.run(finish_session(self.database, "user-1", "session-1", summary))

        query, update = self.database.sessions.updates[0]
        self.assertEqual(query, {"_id": "session-1", "user_id": "user-1"})
        self.assertEqual(update["$set"]["avg_form_score"], 71.5)
        self.assertEqual(update["$set"]["overall_risk"], "low")
        self.assertIsNotNone(update["$set"]["ended_at"])

    def test_get_session_summary_from_database(self):
        self.database.sessions.documents.append({
            "_id": "sess-abc",
            "user_id": "user-1",
            "activity": "SQUAT",
            "started_at": datetime(2026, 10, 5, 12, 0, 0, tzinfo=timezone.utc),
            "ended_at": datetime(2026, 10, 5, 12, 0, 30, tzinfo=timezone.utc),
            "avg_form_score": 85.0,
            "overall_risk": "low",
        })
        self.database.reps.documents.append({
            "_id": "rep-1",
            "session_id": "sess-abc",
            "user_id": "user-1",
            "rep_number": 1,
            "risk_level": "LOW",
            "defect_type": None,
            "form_scores": {"overall": 85.0},
        })

        summary = asyncio.run(get_session_summary(self.database, "user-1", "sess-abc"))
        self.assertIsNotNone(summary)
        self.assertEqual(summary["total_reps"], 1)
        self.assertEqual(summary["average_form_score"], 85.0)
        self.assertEqual(summary["duration_seconds"], 30.0)
        self.assertEqual(summary["low_risk_reps"], 1)


if __name__ == "__main__":
    unittest.main()