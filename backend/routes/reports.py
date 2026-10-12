from __future__ import annotations

import io
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.responses import StreamingResponse

from core.deps import get_current_user
from services.report_builder import build_report

router = APIRouter(prefix="/api/reports", tags=["reports"])


@router.get("/latest", status_code=status.HTTP_200_OK)
async def get_latest_report(
    request: Request,
    user: dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """Retrieve the most recent session report for the authenticated user."""
    db = request.app.state.database
    user_id = str(user["_id"])

    session = await db.sessions.find_one({"user_id": user_id}, sort=[("started_at", -1)])
    if not session:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No session found for this user")

    report = await build_report(db, session_id=str(session["_id"]), user_id=user_id)
    return report


@router.get("/user/list", status_code=status.HTTP_200_OK)
async def list_user_reports(
    request: Request,
    activity: Optional[str] = None,
    limit: int = 50,
    user: dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """List session reports for the authenticated user, optionally filtered by activity."""
    db = request.app.state.database
    user_id = str(user["_id"])

    query: Dict[str, Any] = {"user_id": user_id}
    if activity:
        query["activity"] = activity.lower()

    cursor = db.sessions.find(query).sort("started_at", -1).limit(limit)
    sessions = await cursor.to_list(length=limit) if hasattr(cursor, "to_list") else getattr(cursor, "documents", [])

    items = []
    for s in sessions:
        sid = str(s["_id"])
        # Check if cached report exists
        rep = await db.reports.find_one({"session_id": sid}) if hasattr(db, "reports") else None
        if not rep:
            try:
                rep = await build_report(db, session_id=sid, user_id=user_id)
            except Exception:
                rep = None

        if rep:
            items.append({
                "session_id": sid,
                "activity": s.get("activity", "squat"),
                "source": s.get("source", "live"),
                "started_at": s.get("started_at"),
                "ended_at": s.get("ended_at"),
                "accuracy_pct": rep.get("accuracy_pct", 100),
                "avg_form_score": rep.get("avg_form_score", 85),
                "overall_risk": rep.get("overall_risk", "LOW"),
                "total_reps": rep.get("totals", {}).get("reps", 0),
                "most_common_defect": rep.get("most_common_defect"),
            })

    return {
        "count": len(items),
        "reports": items,
    }


@router.get("/{session_id}", status_code=status.HTTP_200_OK)
async def get_session_report(
    session_id: str,
    request: Request,
    user: dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """Generate or retrieve a full biomechanical report for a session."""
    db = request.app.state.database
    session = await db.sessions.find_one({"_id": session_id})
    if not session:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Session '{session_id}' not found")

    # Enforce ownership or expert role
    if session.get("user_id") != user["_id"] and user.get("role") not in ("expert", "admin"):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Access denied to this session report")

    report = await build_report(db, session_id=session_id, user_id=session.get("user_id"))
    return report


@router.get("/{session_id}/pdf")
async def download_session_pdf(
    session_id: str,
    request: Request,
    user: dict[str, Any] = Depends(get_current_user),
):
    """Generate and stream a downloadable PDF report."""
    from services.pdf_report import generate_pdf_report

    db = request.app.state.database
    session = await db.sessions.find_one({"_id": session_id})
    if not session:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Session '{session_id}' not found")

    if session.get("user_id") != user["_id"] and user.get("role") not in ("expert", "admin"):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Access denied to this session report")

    report_data = await build_report(db, session_id=session_id, user_id=session.get("user_id"))
    user_name = user.get("name", "Athlete")
    
    pdf_bytes = generate_pdf_report(report_data, user_name=user_name)

    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={
            "Content-Disposition": f"attachment; filename=Vyronex_Report_{session_id[:8]}.pdf"
        },
    )
