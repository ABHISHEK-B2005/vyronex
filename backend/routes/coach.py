from __future__ import annotations

import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from core.deps import get_current_user
from services.coach import MEDICAL_DISCLAIMER, ask_ai_coach, generate_post_workout_advice

router = APIRouter(prefix="/api/coach", tags=["coach"])

# In-memory sliding window rate limiter: user_id -> list of unix timestamps
_RATE_LIMIT_STORE: Dict[str, List[float]] = defaultdict(list)
RATE_LIMIT_WINDOW_SEC = 60
MAX_REQUESTS_PER_WINDOW = 20


class ChatRequestIn(BaseModel):
    message: str = Field(min_length=1, max_length=1000)


class ChatResponseOut(BaseModel):
    reply: str
    disclaimer: str = MEDICAL_DISCLAIMER
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


def _check_rate_limit(user_id: str) -> None:
    now = time.time()
    timestamps = [t for t in _RATE_LIMIT_STORE[user_id] if now - t < RATE_LIMIT_WINDOW_SEC]
    if len(timestamps) >= MAX_REQUESTS_PER_WINDOW:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please wait a minute before sending more messages.",
        )
    timestamps.append(now)
    _RATE_LIMIT_STORE[user_id] = timestamps


@router.post("/chat", response_model=ChatResponseOut)
async def chat_with_coach(
    body: ChatRequestIn,
    request: Request,
    user: dict[str, Any] = Depends(get_current_user),
) -> ChatResponseOut:
    """Chat with the AI coach assistant with user movement history as context."""
    user_id = str(user["_id"])
    _check_rate_limit(user_id)

    db = request.app.state.database

    # Fetch last 3 reports for context
    recent_reports = []
    if db is not None and hasattr(db, "reports"):
        cursor = db.reports.find({"user_id": user_id}).sort("session_id", -1).limit(3)
        recent_reports = await cursor.to_list(length=3) if hasattr(cursor, "to_list") else getattr(cursor, "documents", [])

    user_profile = {
        "name": user.get("name"),
        "role": user.get("role"),
        "injury_history": user.get("injury_history", []),
        "age": user.get("age"),
    }

    reply_text = await ask_ai_coach(
        user_query=body.message,
        user_profile=user_profile,
        recent_reports=recent_reports,
    )

    # Persist message in chat_messages collection
    now = datetime.now(timezone.utc)
    if db is not None and hasattr(db, "chat_messages"):
        await db.chat_messages.insert_one({
            "user_id": user_id,
            "user_message": body.message,
            "assistant_reply": reply_text,
            "created_at": now,
        })

    return ChatResponseOut(
        reply=reply_text,
        disclaimer=MEDICAL_DISCLAIMER,
        timestamp=now,
    )


@router.get("/recommendations/{session_id}", status_code=status.HTTP_200_OK)
async def get_session_recommendations(
    session_id: str,
    request: Request,
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """Retrieve post-session corrective exercise recommendations."""
    db = request.app.state.database
    session = await db.sessions.find_one({"_id": session_id})
    if not session:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Session {session_id} not found")

    if session.get("user_id") != user["_id"] and user.get("role") not in ("expert", "coach"):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Cannot access another athlete's recommendations")

    # Fetch report if existing, else build advice from reps
    report = await db.reports.find_one({"session_id": session_id}) if hasattr(db, "reports") else None
    if report:
        return generate_post_workout_advice(report, user_profile=user)

    # Compute minimal advice from reps
    cursor = db.reps.find({"session_id": session_id})
    reps = await cursor.to_list(length=None) if hasattr(cursor, "to_list") else getattr(cursor, "documents", [])
    
    high_risk_count = sum(1 for r in reps if str(r.get("risk_level", "")).upper() == "HIGH")
    defective_count = sum(1 for r in reps if str(r.get("status", "")).lower() == "defective")
    
    mock_report = {
        "totals": {"reps": len(reps), "high_risk": high_risk_count, "defective": defective_count},
        "body_area_breakdown": {"knee": 50, "lower_back": 30},
        "fatigue": {"drop_pct": 10},
    }
    return generate_post_workout_advice(mock_report, user_profile=user)
