from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from core.deps import get_current_user

router = APIRouter(prefix="/api/feedback", tags=["feedback"])


class RepFeedbackIn(BaseModel):
    is_correct: bool = Field(description="Was this rating correct? True for Yes, False for No")
    comment: Optional[str] = Field(default=None, max_length=500)


@router.post("/rep/{rep_id}", status_code=status.HTTP_200_OK)
async def submit_rep_feedback(
    rep_id: str,
    body: RepFeedbackIn,
    request: Request,
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """Submit user feedback on a rep's form classification.
    
    If the user votes 'No' (is_correct=False), the rep is marked as disputed
    and enqueued for expert coach/physio review.
    """
    database = request.app.state.database
    rep = await database.reps.find_one({"_id": rep_id})
    if not rep:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Rep {rep_id} not found")

    # Enforce ownership: user can only rate reps from their own sessions
    if rep.get("user_id") != user["_id"]:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Cannot submit feedback for another user's rep")

    user_feedback_val = "correct" if body.is_correct else "incorrect"
    disputed_val = not body.is_correct

    update_fields: dict[str, Any] = {
        "user_feedback": user_feedback_val,
        "disputed": disputed_val,
    }
    if body.comment:
        update_fields["user_feedback_comment"] = body.comment

    await database.reps.update_one(
        {"_id": rep_id},
        {"$set": update_fields},
    )

    return {
        "rep_id": rep_id,
        "user_feedback": user_feedback_val,
        "disputed": disputed_val,
        "enqueued_for_expert_review": disputed_val,
        "message": "Feedback recorded. Disputed reps are routed to our expert review queue.",
    }
