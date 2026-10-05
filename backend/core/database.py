from __future__ import annotations

from typing import Any

from .config import Settings


def create_mongo_client(settings: Settings) -> Any:
    from motor.motor_asyncio import AsyncIOMotorClient

    return AsyncIOMotorClient(
        settings.mongodb_uri,
        serverSelectionTimeoutMS=5000,
    )


async def ensure_indexes(database: Any) -> None:
    await database.users.create_index("email", unique=True)
    await database.sessions.create_index([("user_id", 1), ("started_at", -1)])
    await database.reps.create_index("session_id")
    await database.reports.create_index("session_id", unique=True)
    await database.reports.create_index("user_id")
    await database.chat_messages.create_index([("user_id", 1), ("created_at", -1)])


async def connect_to_mongo(app: Any, settings: Settings | None = None) -> None:
    settings = settings or Settings.from_env()
    client = create_mongo_client(settings)
    app.state.mongo_client = client
    app.state.db = client[settings.mongodb_database]
    app.state.database = app.state.db
    try:
        await client.admin.command("ping")
        await ensure_indexes(app.state.db)
    except Exception:
        client.close()
        del app.state.mongo_client
        del app.state.db
        del app.state.database
        raise


async def close_mongo_connection(app: Any) -> None:
    client = getattr(app.state, "mongo_client", None)
    if client is not None:
        client.close()