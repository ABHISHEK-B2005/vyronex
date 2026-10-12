import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from core.deps import get_current_user
from routes.athlete import router as athlete_router
from routes.gym import router as gym_router


class ActivityRouteTests(unittest.TestCase):
    def setUp(self):
        self.app = FastAPI()
        self.app.include_router(gym_router)
        self.app.include_router(athlete_router)
        self.current_user = {"role": "gym"}
        self.app.dependency_overrides[get_current_user] = lambda: self.current_user
        self.client = TestClient(self.app)

    def test_gym_user_receives_only_gym_activities(self):
        response = self.client.get("/api/gym/activities")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["activities"],
            ["squat", "push-up", "walking-lunge", "plank"],
        )

    def test_gym_user_cannot_access_athlete_activities(self):
        response = self.client.get("/api/athlete/activities")

        self.assertEqual(response.status_code, 403)

    def test_athlete_user_receives_only_athlete_activities(self):
        self.current_user["role"] = "athlete"
        response = self.client.get("/api/athlete/activities")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["activities"],
            ["long-jump", "shot-put", "javelin-throw"],
        )


if __name__ == "__main__":
    unittest.main()
