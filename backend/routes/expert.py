from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from core.deps import require_role

router = APIRouter(prefix="/api/expert", tags=["expert"])

CONFIDENCE_THRESHOLD = 0.60


class ExpertLabelIn(BaseModel):
    true_risk_level: str = Field(pattern="^(LOW|MODERATE|HIGH)$", description="Expert verified true risk level")
    true_defect_type: Optional[str] = Field(default=None, max_length=100)
    notes: Optional[str] = Field(default=None, max_length=500)


@router.get("/queue", status_code=status.HTTP_200_OK)
async def get_expert_review_queue(
    request: Request,
    limit: int = 50,
    user: dict[str, Any] = Depends(require_role("expert")),
) -> dict[str, Any]:
    """Retrieve queue of disputed reps and low-confidence predictions needing expert review."""
    database = request.app.state.database
    
    # Query reps that are disputed by user OR have model_confidence < 0.60, and haven't been expert-reviewed yet
    query = {
        "expert_reviewed": {"$ne": True},
        "$or": [
            {"disputed": True},
            {"model_confidence": {"$lt": CONFIDENCE_THRESHOLD}},
            {"model_confidence": {"$exists": False}},
        ],
    }

    cursor = database.reps.find(query).sort("timestamp_sec", -1).limit(limit)
    raw_reps = await cursor.to_list(length=limit) if hasattr(cursor, "to_list") else getattr(cursor, "documents", [])

    queue_items = []
    for r in raw_reps:
        queue_items.append({
            "rep_id": str(r.get("_id")),
            "session_id": str(r.get("session_id")),
            "rep_number": r.get("rep_number", 1),
            "predicted_risk_level": r.get("risk_level"),
            "predicted_defect_type": r.get("defect_type"),
            "body_area": r.get("body_area"),
            "model_version": r.get("model_version", "v1.0.0"),
            "model_confidence": r.get("model_confidence", 0.85),
            "disputed": bool(r.get("disputed", False)),
            "user_feedback": r.get("user_feedback"),
            "user_comment": r.get("user_feedback_comment"),
            "features": r.get("features", {}),
            "form_scores": r.get("form_scores", {}),
            "feedback": r.get("feedback", ""),
        })

    return {
        "queue_count": len(queue_items),
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "items": queue_items,
    }


@router.post("/label/{rep_id}", status_code=status.HTTP_200_OK)
async def label_rep_as_expert(
    rep_id: str,
    body: ExpertLabelIn,
    request: Request,
    user: dict[str, Any] = Depends(require_role("expert")),
) -> dict[str, Any]:
    """Expert (coach/physio) sets the verified ground-truth label for a rep.
    
    Only expert-confirmed labels enter the active retraining dataset.
    """
    database = request.app.state.database
    rep = await database.reps.find_one({"_id": rep_id})
    if not rep:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Rep {rep_id} not found")

    now = datetime.now(timezone.utc)
    update_doc = {
        "expert_reviewed": True,
        "expert_label": body.true_risk_level.upper(),
        "expert_defect_type": body.true_defect_type,
        "expert_notes": body.notes,
        "expert_id": str(user["_id"]),
        "expert_name": user.get("name", "Expert Reviewer"),
        "expert_reviewed_at": now,
        "disputed": False,  # resolved
    }

    await database.reps.update_one({"_id": rep_id}, {"$set": update_doc})

    return {
        "rep_id": rep_id,
        "expert_label": body.true_risk_level.upper(),
        "expert_defect_type": body.true_defect_type,
        "expert_id": str(user["_id"]),
        "reviewed_at": now.isoformat(),
        "status": "confirmed_for_retraining",
        "message": "Ground-truth expert label saved. Eligible for safe model retraining.",
    }


@router.post("/retrain", status_code=status.HTTP_200_OK)
async def trigger_retrain(
    request: Request,
    exercise: str = "squat",
    force_activate: bool = False,
    user: dict[str, Any] = Depends(require_role("expert")),
) -> dict[str, Any]:
    """Trigger retraining pipeline from database expert-labeled examples."""
    from services.retrain import trigger_model_retrain
    db = request.app.state.database
    result = await trigger_model_retrain(db, exercise=exercise, force_activate=force_activate)
    return result


@router.get("/versions", status_code=status.HTTP_200_OK)
async def get_model_versions(
    user: dict[str, Any] = Depends(require_role("expert")),
) -> dict[str, Any]:
    """List model version history and active statuses."""
    from ml.retrain import load_model_registry
    registry = load_model_registry()
    return {"versions": registry}
