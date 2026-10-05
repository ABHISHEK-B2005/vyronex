from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


BACKEND_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BACKEND_DIR / ".env")


@dataclass(frozen=True)
class Settings:
    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_database: str = "vyronex"
    jwt_secret: str = ""
    access_token_minutes: int = 60

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            mongodb_uri=os.getenv("MONGO_URI", os.getenv("MONGODB_URI", cls.mongodb_uri)),
            mongodb_database=os.getenv("DB_NAME", os.getenv("MONGODB_DATABASE", cls.mongodb_database)),
            jwt_secret=os.getenv("JWT_SECRET", cls.jwt_secret),
            access_token_minutes=int(os.getenv(
                "JWT_EXPIRE_MINUTES",
                os.getenv("ACCESS_TOKEN_MINUTES", str(cls.access_token_minutes)),
            )),
        )