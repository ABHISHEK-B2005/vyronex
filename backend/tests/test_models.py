import unittest

from pydantic import ValidationError

from models.report import ReportRecord
from models.session import RepRecord, SessionRecord
from models.user import UserCreate, UserRecord


class UserModelTests(unittest.TestCase):
    def test_create_normalizes_email_and_defaults_injury_history(self):
        user = UserCreate(
            name="Morgan",
            email=" Morgan@Example.com ",
            password="long-enough-password",
            role="gym",
        )

        self.assertEqual(user.email, "morgan@example.com")
        self.assertEqual(user.injury_history, [])

    def test_create_rejects_invalid_role_and_short_password(self):
        with self.assertRaises(ValidationError):
            UserCreate(name="Morgan", email="m@example.com", password="short", role="coach")

    def test_record_accepts_mongo_id_alias(self):
        user = UserRecord(
            _id="user-1",
            name="Morgan",
            email="morgan@example.com",
            password_hash="hash",
            role="athlete",
        )

        self.assertEqual(user.id, "user-1")


class ActivityModelTests(unittest.TestCase):
    def test_session_and_rep_validate_requested_fields(self):
        session = SessionRecord(
            user_id="user-1", activity="squat", source="live", avg_form_score=85,
        )
        rep = RepRecord(
            user_id="user-1",
            session_id="session-1",
            rep_number=1,
            timestamp_sec=2.5,
            status="correct",
            risk_level="LOW",
        )

        self.assertEqual(session.source, "live")
        self.assertEqual(rep.rep_number, 1)

    def test_rep_requires_classification_and_owner(self):
        with self.assertRaises(ValidationError):
            RepRecord(
                session_id="session-1",
                rep_number=1,
                timestamp_sec=2.5,
                status="complete",
                risk_level="low",
            )

    def test_report_rejects_accuracy_outside_percentage_range(self):
        with self.assertRaises(ValidationError):
            ReportRecord(session_id="s1", user_id="u1", accuracy_pct=120)


if __name__ == "__main__":
    unittest.main()