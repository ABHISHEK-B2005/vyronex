from __future__ import annotations

import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from src.config import DEFECT_BODY_AREA_MAP, REP_FORM_SCORE_CORRECT_MIN


def get_body_area(defect_type: str | None) -> str | None:
    if not defect_type:
        return None
    raw = str(defect_type).strip().lower()
    if raw in DEFECT_BODY_AREA_MAP:
        return DEFECT_BODY_AREA_MAP[raw]

    under = raw.replace(" ", "_")
    if under in DEFECT_BODY_AREA_MAP:
        return DEFECT_BODY_AREA_MAP[under]

    space = raw.replace("_", " ")
    if space in DEFECT_BODY_AREA_MAP:
        return DEFECT_BODY_AREA_MAP[space]

    base = raw.split("(")[0].strip()
    if base in DEFECT_BODY_AREA_MAP:
        return DEFECT_BODY_AREA_MAP[base]
    if base.replace(" ", "_") in DEFECT_BODY_AREA_MAP:
        return DEFECT_BODY_AREA_MAP[base.replace(" ", "_")]

    return None


def classify_rep_status(risk_level: str | None, form_scores: dict[str, Any] | None) -> str:
    norm_risk = str(risk_level or "LOW").strip().upper()
    if norm_risk == "HIGH":
        return "high_risk"

    form_scores = form_scores or {}
    category_scores = [
        float(v) for k, v in form_scores.items()
        if k != "overall" and isinstance(v, (int, float))
    ]
    if not category_scores and "overall" in form_scores and isinstance(form_scores["overall"], (int, float)):
        category_scores = [float(form_scores["overall"])]

    any_score_below_threshold = any(s < REP_FORM_SCORE_CORRECT_MIN for s in category_scores)

    if norm_risk == "MODERATE" or any_score_below_threshold:
        return "defective"

    if norm_risk == "LOW" and not any_score_below_threshold:
        return "correct"

    return "defective" if any_score_below_threshold else "correct"


async def create_session(
    database: Any,
    user_id: str,
    activity: str,
    source: str = "live",
) -> str:
    session_id = uuid.uuid4().hex
    await database.sessions.insert_one({
        "_id": session_id,
        "user_id": user_id,
        "activity": activity,
        "source": source,
        "started_at": datetime.now(timezone.utc),
        "ended_at": None,
        "avg_form_score": None,
        "overall_risk": None,
    })
    return session_id


async def record_completed_rep(
    database: Any,
    user_id: str,
    session_id: str,
    started_at: float,
    result: dict,
    timestamp_sec: float | None = None,
) -> dict:
    risk = result.get("risk", {})
    if isinstance(risk, dict):
        risk_level = risk.get("risk_level", "LOW")
        defect_type = risk.get("primary_issue")
    else:
        risk_level = str(risk)
        defect_type = result.get("defect_type")

    if not defect_type:
        defect_type = result.get("defect_type")

    body_area = result.get("body_area") or get_body_area(defect_type)
    form_scores = result.get("form_scores", {})
    features = result.get("features", {})

    status = result.get("status")
    if not status:
        status = classify_rep_status(risk_level, form_scores)

    feedback_raw = result.get("feedback", "")
    if isinstance(feedback_raw, dict):
        feedback_text = feedback_raw.get("recommendation") or feedback_raw.get("reason") or ""
    else:
        feedback_text = str(feedback_raw or "")

    if timestamp_sec is not None:
        calc_timestamp = round(float(timestamp_sec), 2)
    elif "timestamp_sec" in result:
        calc_timestamp = round(float(result["timestamp_sec"]), 2)
    elif started_at and started_at > 0:
        calc_timestamp = round(max(0.0, time.time() - started_at), 2)
    else:
        calc_timestamp = 0.0

    rep_doc = {
        "_id": uuid.uuid4().hex,
        "user_id": user_id,
        "session_id": session_id,
        "rep_number": int(result.get("rep_number", 1)),
        "timestamp_sec": calc_timestamp,
        "status": status,
        "defect_type": defect_type,
        "body_area": body_area,
        "risk_level": str(risk_level).upper(),
        "form_scores": form_scores,
        "features": features,
        "feedback": feedback_text,
    }

    await database.reps.insert_one(rep_doc)
    return rep_doc


async def finish_session(
    database: Any,
    user_id: str,
    session_id: str,
    summary: dict | None = None,
) -> dict:
    if summary is not None:
        avg_form_score = summary.get("average_form_score")
        risk_counts = {
            "low": summary.get("low_risk_reps", 0),
            "moderate": summary.get("moderate_risk_reps", 0),
            "high": summary.get("high_risk_reps", 0),
        }
        total_reps = summary.get("total_reps", 0)
        overall_risk = max(risk_counts, key=risk_counts.get) if total_reps else None
    else:
        reps_cursor = database.reps.find({"session_id": session_id})
        reps = await reps_cursor.to_list(length=1000) if hasattr(reps_cursor, "to_list") else getattr(reps_cursor, "documents", [])
        total_reps = len(reps)
        form_scores = []
        risk_counts = {"low": 0, "moderate": 0, "high": 0}
        for r in reps:
            fs = r.get("form_scores", {})
            if "overall" in fs and isinstance(fs["overall"], (int, float)):
                form_scores.append(fs["overall"])
            rl = str(r.get("risk_level", "")).lower()
            if rl in risk_counts:
                risk_counts[rl] += 1
        avg_form_score = round(sum(form_scores) / len(form_scores), 1) if form_scores else None
        overall_risk = max(risk_counts, key=risk_counts.get) if total_reps else None

    ended_at = datetime.now(timezone.utc)
    overall_risk_val = overall_risk.lower() if overall_risk else None

    await database.sessions.update_one(
        {"_id": session_id, "user_id": user_id},
        {"$set": {
            "ended_at": ended_at,
            "avg_form_score": avg_form_score,
            "overall_risk": overall_risk_val,
        }},
    )

    return {
        "session_id": session_id,
        "ended_at": ended_at,
        "avg_form_score": avg_form_score,
        "overall_risk": overall_risk_val,
        "total_reps": total_reps,
    }


end_session = finish_session


async def get_session_summary(database: Any, user_id: str, session_id: str) -> dict | None:
    session = await database.sessions.find_one({"_id": session_id, "user_id": user_id})
    if not session:
        return None

    reps_cursor = database.reps.find({"session_id": session_id})
    reps = await reps_cursor.to_list(length=1000) if hasattr(reps_cursor, "to_list") else getattr(reps_cursor, "documents", [])
    total_reps = len(reps)

    form_scores = []
    risk_scores = []
    risk_counts = {"LOW": 0, "MODERATE": 0, "HIGH": 0}
    issues = []

    for r in reps:
        fs = r.get("form_scores", {})
        if "overall" in fs and isinstance(fs["overall"], (int, float)):
            form_scores.append(fs["overall"])
        rl = str(r.get("risk_level", "")).upper()
        if rl in risk_counts:
            risk_counts[rl] += 1
        defect = r.get("defect_type")
        if defect:
            issues.append(defect)

    avg_form_score = session.get("avg_form_score")
    if avg_form_score is None:
        avg_form_score = round(sum(form_scores) / len(form_scores), 1) if form_scores else 0.0

    most_common_issue = Counter(issues).most_common(1)[0][0] if issues else None

    started_at = session.get("started_at")
    ended_at = session.get("ended_at") or datetime.now(timezone.utc)
    duration_seconds = 0.0
    start_time_ts = 0.0
    if started_at and hasattr(started_at, "timestamp"):
        start_time_ts = started_at.timestamp()
        if ended_at and hasattr(ended_at, "timestamp"):
            duration_seconds = round(max(0.0, ended_at.timestamp() - start_time_ts), 1)

    return {
        "exercise": session.get("activity", "SQUAT"),
        "start_time": start_time_ts,
        "duration_seconds": duration_seconds,
        "total_reps": total_reps,
        "average_form_score": avg_form_score,
        "average_risk_score": 0.0,
        "low_risk_reps": risk_counts["LOW"],
        "moderate_risk_reps": risk_counts["MODERATE"],
        "high_risk_reps": risk_counts["HIGH"],
        "most_common_issue": most_common_issue,
    }