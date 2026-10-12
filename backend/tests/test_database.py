import asyncio
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from core.config import Settings
from core.database import close_mongo_connection, connect_to_mongo


class FakeMongoClient:
    def __init__(self):
        self.database_names = []
        self.closed = False
        self.admin = SimpleNamespace(command=self.command)
        self.database = FakeDatabase()

    def __getitem__(self, database_name):
        self.database_names.append(database_name)
        return self.database

    async def command(self, name):
        return {"ok": 1}

    def close(self):
        self.closed = True


class FakeCollection:
    def __init__(self):
        self.indexes = []

    async def create_index(self, keys, unique=False):
        self.indexes.append((keys, unique))


class FakeDatabase:
    def __init__(self):
        self.users = FakeCollection()
        self.sessions = FakeCollection()
        self.reps = FakeCollection()
        self.reports = FakeCollection()
        self.chat_messages = FakeCollection()

    async def command(self, name):
        return {"ok": 1}


class DatabaseLifecycleTests(unittest.TestCase):
    def test_connect_selects_configured_database_and_close_releases_client(self):
        client = FakeMongoClient()
        app = SimpleNamespace(state=SimpleNamespace())
        settings = Settings(mongodb_uri="mongodb://example", mongodb_database="test_db")

        with patch("core.database.create_mongo_client", return_value=client) as factory:
            asyncio.run(connect_to_mongo(app, settings))
            factory.assert_called_once_with(settings)

        self.assertIs(app.state.mongo_client, client)
        self.assertIs(app.state.database, client.database)
        self.assertIs(app.state.db, client.database)
        self.assertEqual(client.database_names, ["test_db"])
        self.assertEqual(client.database.users.indexes, [("email", True)])
        self.assertEqual(client.database.reports.indexes, [("session_id", True), ("user_id", False)])

        asyncio.run(close_mongo_connection(app))
        self.assertTrue(client.closed)

    def test_settings_read_mongodb_environment(self):
        with patch.dict(os.environ, {
            "MONGO_URI": "mongodb://dbhost:27017",
            "DB_NAME": "vyronex_test",
            "JWT_EXPIRE_MINUTES": "720",
        }):
            settings = Settings.from_env()

        self.assertEqual(settings.mongodb_uri, "mongodb://dbhost:27017")
        self.assertEqual(settings.mongodb_database, "vyronex_test")
        self.assertEqual(settings.access_token_minutes, 720)


if __name__ == "__main__":
    unittest.main()