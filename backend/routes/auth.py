from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pymongo.errors import DuplicateKeyError

from core.deps import get_current_user
from core.security import create_token, hash_password, verify_password
from models.user import LoginIn, ProfileUpdateIn, SignupIn, UserOut


router = APIRouter(prefix="/api/auth", tags=["auth"])
LoginRequest = LoginIn


def _token(user_id: str, role: str) -> str:
    try:
        return create_token(user_id, role)
    except ValueError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc


def _user_out(user: dict) -> UserOut:
    return UserOut(
        id=str(user["_id"]),
        name=user["name"],
        email=user["email"],
        role=user["role"],
        age=user.get("age"),
        height_cm=user.get("height_cm"),
        weight_kg=user.get("weight_kg"),
        injury_history=user.get("injury_history", []),
        data_consent=bool(user.get("data_consent", False)),
        created_at=user.get("created_at"),
    )


def _auth_response(user: dict, token: str) -> dict:
    return {
        "access_token": token,
        "token_type": "bearer",
        "role": user["role"],
        "user": _user_out(user).model_dump(mode="json"),
    }


@router.post("/signup", status_code=status.HTTP_201_CREATED)
async def signup(body: SignupIn, request: Request) -> dict:
    database = request.app.state.database
    users = database.users
    await users.create_index("email", unique=True)

    document = body.model_dump(exclude={"password"})
    document.update({
        "_id": uuid.uuid4().hex,
        "password_hash": hash_password(body.password),
        "created_at": datetime.now(timezone.utc),
    })
    token = _token(document["_id"], document["role"])
    try:
        await users.insert_one(document)
    except DuplicateKeyError:
        raise HTTPException(status.HTTP_409_CONFLICT, "An account with this email already exists") from None
    return _auth_response(document, token)


@router.post("/login")
async def login(body: LoginIn, request: Request) -> dict:
    user = await request.app.state.database.users.find_one({"email": body.email})
    if user is None or not verify_password(body.password, user.get("password_hash", "")):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Incorrect email or password")
    return _auth_response(user, _token(str(user["_id"]), user["role"]))


@router.get("/me", response_model=UserOut)
async def get_me(user: dict = Depends(get_current_user)) -> UserOut:
    return _user_out(user)


@router.put("/profile", response_model=UserOut)
async def update_profile(
    body: ProfileUpdateIn,
    request: Request,
    user: dict = Depends(get_current_user),
) -> UserOut:
    updates = body.model_dump(exclude_unset=True)
    if not updates:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "Provide at least one profile field")
    
    # If user turns off data_consent, respect withdrawal by updating database
    await request.app.state.database.users.update_one({"_id": user["_id"]}, {"$set": updates})
    return _user_out({**user, **updates})


@router.delete("/data", status_code=status.HTTP_200_OK)
async def delete_user_movement_data(
    request: Request,
    user: dict = Depends(get_current_user),
) -> dict:
    """Delete anonymized movement features and withdraw training consent."""
    db = request.app.state.database
    user_id = user["_id"]
    # Turn off consent
    await db.users.update_one({"_id": user_id}, {"$set": {"data_consent": False}})
    # Clear features from reps to guarantee privacy
    await db.reps.update_many({"user_id": user_id}, {"$set": {"features": {}}})
    return {"message": "Movement data cleared and training consent withdrawn successfully"}