import asyncio
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace

from routes.alignment import alignment_summary


class FakeCursor:
    def __init__(self, documents):
        self.documents = list(documents)

    def sort(self, field, direction=1):
        self.documents.sort(
            key=lambda document: document.get(field) or datetime.min.replace(tzinfo=timezone.utc),
            reverse=direction < 0,
        )
        return self

    async def to_list(self, length=None):
        return list(self.documents if length is None else self.documents[:length])


class FakeCollection:
    def __init__(self, documents):
        self.documents = documents

    def find(self, query):
        found = []
        for document in self.documents:
            matches = True
            for key, expected in query.items():
                actual = document.get(key)
                if isinstance(expected, dict) and "$in" in expected:
                    matches = actual in expected["$in"]
                elif actual != expected:
                    matches = False
            if matches:
                found.append(document)
        return FakeCursor(found)


class AlignmentSummaryRouteTests(unittest.TestCase):
    def test_returns_recent_alignment_trend_and_fault_areas(self):
        started_at = datetime.now(timezone.utc)
        alignment = {
            "overall": 55.0,
            "status": "Fix",
            "metrics": [
                {
                    "key": "knee_alignment",
                    "label": "Knee alignment",
                    "score": 55.0,
                    "status": "Fix",
                    "assessed": True,
                    "body_area": "knee",
                },
                {
                    "key": "torso_lean",
                    "label": "Torso lean",
                    "score": None,
                    "status": "Not assessed",
                    "assessed": False,
                    "body_area": "lower_back",
                },
            ],
        }
        database = SimpleNamespace(
            sessions=FakeCollection([
                {"_id": "s1", "user_id": "u1", "activity": "SQUAT", "started_at": started_at},
            ]),
            reps=FakeCollection([
                {
                    "session_id": "s1",
                    "user_id": "u1",
                    "rep_number": 1,
                    "alignment": alignment,
                },
            ]),
        )
        request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(database=database)))

        result = asyncio.run(alignment_summary(request, activity="squat", days=30, user={"_id": "u1"}))

        self.assertEqual(result["latest_overall_score"], 55.0)
        self.assertEqual(result["average_score"], 55.0)
        self.assertEqual(result["trend"][0]["score"], 55.0)
        self.assertEqual(result["most_frequent_misalignment"]["key"], "knee_alignment")
        self.assertEqual(result["body_area_counts"], {"knee": 1})

    def test_empty_summary_has_no_fake_zero_score(self):
        database = SimpleNamespace(
            sessions=FakeCollection([]),
            reps=FakeCollection([]),
        )
        request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(database=database)))

        result = asyncio.run(alignment_summary(request, activity=None, days=30, user={"_id": "u1"}))

        self.assertIsNone(result["latest_overall_score"])
        self.assertEqual(result["trend"], [])


if __name__ == "__main__":
    unittest.main()
