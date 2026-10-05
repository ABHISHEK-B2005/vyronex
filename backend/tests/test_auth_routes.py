import asyncio
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from pymongo.errors import DuplicateKeyError

from core.deps import get_current_user, require_role
from core.security import create_access_token, verify_password
from models.user import ProfileUpdateIn, UserCreate
from routes.auth import LoginRequest, get_me, login, signup, update_profile


TEST_PASSWORD = "long-enough-password"


class FakeUsers:
    def __init__(self):
        self.documents = {}
        self.indexes = []

    async def create_index(self, field, unique=False):
        self.indexes.append((field, unique))

    async def find_one(self, query):
        if "email" in query:
            return next((doc for doc in self.documents.values() if doc["email"] == query["email"]), None)
        return self.documents.get(query.get("_id"))

    async def insert_one(self, document):
        if any(doc["email"] == document["email"] for doc in self.documents.values()):
            raise DuplicateKeyError("duplicate email")
        self.documents[document["_id"]] = dict(document)

    async def update_one(self, query, update):
        document = await self.find_one(query)
        if document is not None:
            document.update(update["$set"])


def make_request(users):
    return SimpleNamespace(app=SimpleNamespace(
        state=SimpleNamespace(database=SimpleNamespace(users=users)),
    ))


class AuthRouteTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.users = FakeUsers()
        self.request = make_request(self.users)
        self.secret_patch = patch.dict(os.environ, {
            "JWT_SECRET": "test-secret-that-is-long-enough-for-hmac",
        })
        self.secret_patch.start()
        self.addCleanup(self.secret_patch.stop)

    async def test_signup_stores_hash_and_returns_token_without_password(self):
        response = await signup(UserCreate(
            name="Morgan",
            email="morgan@example.com",
            password="long-enough-password",
            role="athlete",
        ), self.request)

        stored = next(iter(self.users.documents.values()))
        self.assertTrue(verify_password("long-enough-password", stored["password_hash"]))
        self.assertNotIn("password", stored)
        self.assertNotIn("password_hash", response["user"])
        self.assertEqual(response["role"], "athlete")
        self.assertEqual(response["user"]["role"], "athlete")
        self.assertEqual(self.users.indexes, [("email", True)])

    async def test_duplicate_signup_and_invalid_login_are_rejected(self):
        body = UserCreate(
            name="Morgan",
            email="morgan@example.com",
            password="long-enough-password",
            role="gym",
        )
        await signup(body, self.request)

        with self.assertRaises(HTTPException) as duplicate:
            await signup(body, self.request)
        self.assertEqual(duplicate.exception.status_code, 409)

        with self.assertRaises(HTTPException) as invalid:
            await login(LoginRequest(email="morgan@example.com", password="not-the-password"), self.request)
        self.assertEqual(invalid.exception.status_code, 401)

    async def test_login_and_current_user_use_persisted_account(self):
        await signup(UserCreate(
            name="Morgan",
            email="morgan@example.com",
            password="long-enough-password",
            role="gym",
        ), self.request)
        response = await login(
            LoginRequest(email="MORGAN@example.com", password="long-enough-password"),
            self.request,
        )
        credentials = HTTPAuthorizationCredentials(
            scheme="Bearer", credentials=response["access_token"],
        )

        current_user = await get_current_user(self.request, credentials)
        self.assertEqual(current_user["email"], "morgan@example.com")

        athlete_guard = require_role("athlete")
        with self.assertRaises(HTTPException) as forbidden:
            await athlete_guard(current_user)
        self.assertEqual(forbidden.exception.status_code, 403)

    async def test_me_profile_update_and_missing_token(self):
        with self.assertRaises(HTTPException) as missing:
            await get_current_user(self.request, None)
        self.assertEqual(missing.exception.status_code, 401)

        await signup(UserCreate(
            name="Morgan",
            email="morgan@example.com",
            password=TEST_PASSWORD,
            role="athlete",
        ), self.request)
        response = await login(
            LoginRequest(email="morgan@example.com", password=TEST_PASSWORD),
            self.request,
        )
        credentials = HTTPAuthorizationCredentials(
            scheme="Bearer", credentials=response["access_token"],
        )
        current_user = await get_current_user(self.request, credentials)

        self.assertEqual((await get_me(current_user)).email, "morgan@example.com")
        updated = await update_profile(
            ProfileUpdateIn(age=25, height_cm=180, weight_kg=75, injury_history=["knee"]),
            self.request,
            current_user,
        )
        self.assertEqual(updated.age, 25)
        self.assertEqual(updated.injury_history, ["knee"])
        self.assertNotIn("password_hash", updated.model_dump())


if __name__ == "__main__":
    unittest.main()