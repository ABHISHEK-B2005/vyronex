import asyncio
import unittest
from types import SimpleNamespace

from services.session_report_service import create_session_report


class FakeReports:
    def __init__(self):
        self.documents = []

    async def insert_one(self, document):
        self.documents.append(document)


class SessionReportTests(unittest.TestCase):
    def test_report_rolls_up_completed_rep_risk_and_recommendations(self):
        reports = FakeReports()
        database = SimpleNamespace(reports=reports)
        summary = {
            "total_reps": 1,
            "duration_seconds": 10.0,
            "average_form_score": 72.0,
            "average_risk_score": 28.0,
        }
        reps = [{
            "rep_number": 1,
            "risk": {
                "risk_score": 28.0,
                "risk_level": "LOW",
                "primary_issue": "knee_alignment",
            },
            "feedback": {"recommendation": "Track the knee over the foot."},
        }]

        report = asyncio.run(create_session_report(
            database,
            "user-1",
            "session-1",
            summary,
            reps,
        ))

        self.assertEqual(report["session_id"], "session-1")
        self.assertNotIn("accuracy_pct", report)
        self.assertEqual(report["totals"]["average_form_score"], 72.0)
        self.assertEqual(report["defect_counts"], {"knee_alignment": 1})
        self.assertEqual(report["worst_rep"]["rep_number"], 1)
        self.assertEqual(len(reports.documents), 1)


if __name__ == "__main__":
    unittest.main()
