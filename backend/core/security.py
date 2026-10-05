from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone

import bcrypt
from jose import jwt

from .config import Settings


def hash_password(password: str) -> str:
    password_bytes = password.encode("utf-8")
    # Pre-hashing avoids bcrypt's 72-byte input limit without truncating passwords.
    bcrypt_input = hashlib.sha256(password_bytes).digest()
    return bcrypt.hashpw(bcrypt_input, bcrypt.gensalt()).decode("ascii")


def verify_password(password: str, encoded: str) -> bool:
    try:
        bcrypt_input = hashlib.sha256(password.encode("utf-8")).digest()
        return bcrypt.checkpw(bcrypt_input, encoded.encode("ascii"))
    except (AttributeError, ValueError, TypeError, UnicodeEncodeError):
        return False


def create_token(user_id: str, role: str) -> str:
    settings = Settings.from_env()
    return create_access_token(
        user_id,
        role,
        settings.jwt_secret,
        settings.access_token_minutes,
    )


def create_access_token(
    user_id: str,
    role: str,
    secret: str,
    expires_minutes: int = 60,
) -> str:
    if len(secret) < 32:
        raise ValueError("JWT_SECRET must contain at least 32 characters")
    now = datetime.now(timezone.utc)
    return jwt.encode(
        {"sub": user_id, "role": role, "iat": now, "exp": now + timedelta(minutes=expires_minutes)},
        secret,
        algorithm="HS256",
    )


def decode_access_token(token: str, secret: str) -> dict:
    if len(secret) < 32:
        raise ValueError("JWT_SECRET must contain at least 32 characters")
    return jwt.decode(token, secret, algorithms=["HS256"])