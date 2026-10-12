from __future__ import annotations

from typing import Any

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError

from .config import Settings
from .security import decode_access_token


_bearer = HTTPBearer(auto_error=False)
_USER_ROLES = {"gym", "athlete", "expert"}


async def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> dict[str, Any]:
    if credentials is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Authentication required")
    try:
        claims = decode_access_token(credentials.credentials, Settings.from_env().jwt_secret)
        user_id = claims.get("sub")
        role = claims.get("role")
        if (
            not isinstance(user_id, str)
            or not user_id
            or not isinstance(role, str)
            or role not in _USER_ROLES
        ):
            raise ValueError("Token subject or role is missing")
    except (JWTError, ValueError):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or expired access token") from None

    user = await request.app.state.database.users.find_one({"_id": user_id})
    if user is None or user.get("role") != claims["role"]:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "User no longer exists")
    return user


def require_role(*roles: str):
    async def role_guard(user: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
        if user.get("role") not in roles:
            raise HTTPException(status.HTTP_403_FORBIDDEN, "This account cannot access this resource")
        return user

    return role_guard


async def get_current_user_optional(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> dict[str, Any] | None:
    if credentials is None:
        return None
    try:
        claims = decode_access_token(credentials.credentials, Settings.from_env().jwt_secret)
        user_id = claims.get("sub")
        role = claims.get("role")
        if (
            not isinstance(user_id, str)
            or not user_id
            or not isinstance(role, str)
            or role not in _USER_ROLES
        ):
            return None
    except (JWTError, ValueError):
        return None

    if not hasattr(request.app.state, "database") or request.app.state.database is None:
        return {"_id": user_id, "role": role}

    user = await request.app.state.database.users.find_one({"_id": user_id})
    if user is None or user.get("role") != claims["role"]:
        return None
    return user