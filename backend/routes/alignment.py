from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import APIRouter, Depends, Query, Request

from core.deps import get_current_user

router = APIRouter(prefix="/api/alignment", tags=["alignment"])


@router.get("/summary")
async def alignment_summary(
    request: Request,
    activity: str | None = None,
    days: int = Query(default=30, ge=1, le=365),
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    database = request.app.state.database
    user_id = str(user["_id"])
    query: dict[str, Any] = {"user_id": user_id}

    cursor = database.sessions.find(query).sort("started_at", -1)
    sessions = await cursor.to_list(length=None) if hasattr(cursor, "to_list") else getattr(cursor, "documents", [])
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    selected = []
    for session in sessions:
        if activity and str(session.get("activity", "")).lower() != activity.lower():
            continue
        started = session.get("started_at")
        if not isinstance(started, datetime):
            continue
        if started.tzinfo is None:
            started = started.replace(tzinfo=timezone.utc)
        if started >= cutoff:
            selected.append(session)

    session_ids = [str(session["_id"]) for session in selected]
    if not session_ids:
        return {
            "latest_overall_score": None,
            "average_score": None,
            "change_since_last": None,
            "trend": [],
            "most_frequent_misalignment": None,
            "body_area_counts": {},
            "sessions_count": 0,
        }

    rep_cursor = database.reps.find({"user_id": user_id, "session_id": {"$in": session_ids}})
    reps = await rep_cursor.to_list(length=None) if hasattr(rep_cursor, "to_list") else getattr(rep_cursor, "documents", [])
    session_dates = {}
    for session in selected:
        started = session.get("started_at")
        if started.tzinfo is None:
            started = started.replace(tzinfo=timezone.utc)
        session_dates[str(session["_id"])] = started.date().isoformat()

    daily_scores: dict[str, list[float]] = defaultdict(list)
    misalignments: Counter[str] = Counter()
    body_areas: Counter[str] = Counter()
    metric_labels: dict[str, str] = {}
    latest_scores_by_session: dict[str, list[float]] = defaultdict(list)
    latest_session_alignment: dict[str, tuple[int, dict[str, Any]]] = {}
    for rep in reps:
        session_id = str(rep.get("session_id", ""))
        features = rep.get("features", {})
        alignment = rep.get("alignment") or (features.get("alignment") if isinstance(features, dict) else {}) or {}
        if not isinstance(alignment, dict):
            continue
        rep_number = int(rep.get("rep_number", 0))
        if (
            session_id not in latest_session_alignment
            or rep_number >= latest_session_alignment[session_id][0]
        ):
            latest_session_alignment[session_id] = (rep_number, alignment)
        score = alignment.get("overall")
        if isinstance(score, (int, float)):
            score = float(score)
            latest_scores_by_session[session_id].append(score)
            daily_scores[session_dates[session_id]].append(score)
        for metric in alignment.get("metrics", []):
            if not isinstance(metric, dict) or not metric.get("assessed"):
                continue
            key = metric.get("key")
            status = metric.get("status")
            if not key or status not in {"Watch", "Fix"}:
                continue
            misalignments[key] += 1
            metric_labels[key] = metric.get("label", str(key).replace("_", " ").title())
            area = metric.get("body_area")
            if area:
                body_areas[area] += 1

    trend = [
        {"date": date, "score": round(sum(scores) / len(scores), 1)}
        for date, scores in sorted(daily_scores.items())
        if scores
    ]
    session_order = [
        str(session["_id"]) for session in sorted(
            selected,
            key=lambda entry: entry.get("started_at") or datetime.min.replace(tzinfo=timezone.utc),
        )
    ]
    session_scores = [
        round(sum(latest_scores_by_session[sid]) / len(latest_scores_by_session[sid]), 1)
        for sid in session_order if latest_scores_by_session.get(sid)
    ]
    frequent_key = misalignments.most_common(1)[0][0] if misalignments else None
    average_score = round(sum(point["score"] for point in trend) / len(trend), 1) if trend else None
    latest_session_id = session_order[-1] if session_order else None
    return {
        "latest_overall_score": session_scores[-1] if session_scores else None,
        "average_score": average_score,
        "change_since_last": round(session_scores[-1] - session_scores[-2], 1) if len(session_scores) > 1 else None,
        "trend": trend,
        "latest_metrics": (
            latest_session_alignment.get(latest_session_id, (0, {}))[1].get("metrics", [])
            if latest_session_id else []
        ),
        "most_frequent_misalignment": (
            {
                "key": frequent_key,
                "label": metric_labels.get(frequent_key, frequent_key.replace("_", " ").title()),
                "count": misalignments[frequent_key],
                "body_area": next(
                    (
                        metric.get("body_area")
                        for rep in reps
                        for metric in _metrics_from_rep(rep)
                        if metric.get("key") == frequent_key
                    ),
                    None,
                ),
            }
            if frequent_key else None
        ),
        "body_area_counts": dict(body_areas),
        "sessions_count": len(selected),
    }


def _metrics_from_rep(rep: dict[str, Any]) -> list[dict[str, Any]]:
    alignment = rep.get("alignment")
    if not isinstance(alignment, dict):
        features = rep.get("features", {})
        alignment = features.get("alignment", {}) if isinstance(features, dict) else {}
    metrics = alignment.get("metrics", []) if isinstance(alignment, dict) else []
    return [metric for metric in metrics if isinstance(metric, dict)]
