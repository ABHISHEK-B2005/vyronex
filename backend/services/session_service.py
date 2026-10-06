from __future__ import annotations

import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from math import isfinite
from typing import Any

from models.session import RepRecord, SessionRecord
from src.config import DEFECT_BODY_AREA_MAP, REP_FORM_SCORE_CORRECT_MIN


_RISK_SEVERITY = {"low": 0, "moderate": 1, "high": 2}


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
    norm_risk = str(risk_level or "").strip().upper()
    if norm_risk == "HIGH":
        return "high_risk"

    if norm_risk not in {"LOW", "MODERATE"}:
        return "defective"

    form_scores = form_scores if isinstance(form_scores, dict) else {}
    category_scores = [
        float(value)
        for key, value in form_scores.items()
        if key != "overall"
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
        and isfinite(value)
    ]
    if (
        not category_scores
        and isinstance(form_scores.get("overall"), (int, float))
        and not isinstance(form_scores["overall"], bool)
        and isfinite(form_scores["overall"])
    ):
        category_scores = [float(form_scores["overall"])]

    any_score_below_threshold = any(s < REP_FORM_SCORE_CORRECT_MIN for s in category_scores)

    if norm_risk == "MODERATE" or any_score_below_threshold:
        return "defective"

    if norm_risk == "LOW" and category_scores:
        return "correct"

    return "defective"


async def create_session(
    database: Any,
    user_id: str,
    activity: str,
    source: str = "live",
) -> str:
    session_id = uuid.uuid4().hex
    session = SessionRecord(
        id=session_id,
        user_id=user_id,
        activity=activity,
        source=source,
        started_at=datetime.now(timezone.utc),
    )
    await database.sessions.insert_one(
        session.model_dump(by_alias=True),
    )
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

    body_area = get_body_area(defect_type)
    form_scores = result.get("form_scores", {})
    if not isinstance(form_scores, dict):
        form_scores = {}
    features = result.get("features", {})
    if not isinstance(features, dict):
        features = {}
    status = classify_rep_status(risk_level, form_scores)

    feedback_raw = result.get("feedback", "")
    if isinstance(feedback_raw, dict):
        feedback_text = feedback_raw.get("recommendation") or feedback_raw.get("reason") or ""
    else:
        feedback_text = str(feedback_raw or "")

    if timestamp_sec is not None:
        calc_timestamp = float(timestamp_sec)
    elif "timestamp_sec" in result:
        calc_timestamp = float(result["timestamp_sec"])
    elif started_at and started_at > 0:
        calc_timestamp = max(0.0, time.time() - started_at)
    else:
        calc_timestamp = 0.0
    if not isfinite(calc_timestamp) or calc_timestamp < 0:
        raise ValueError("timestamp_sec must be a finite, non-negative number")

    # Check if user has consented to storing movement features for model training
    user = await database.users.find_one({"_id": user_id}) if hasattr(database, "users") else None
    has_opted_out = (user is not None and user.get("data_consent") is False)
    stored_features = {} if has_opted_out else features

    model_version = result.get("model_version") or (risk.get("model_version") if isinstance(risk, dict) else "v1.0.0")
    model_confidence = result.get("model_confidence") or (risk.get("model_confidence") if isinstance(risk, dict) else 0.85)

    rep = RepRecord(
        id=uuid.uuid4().hex,
        user_id=user_id,
        session_id=session_id,
        rep_number=result.get("rep_number", 1),
        timestamp_sec=round(calc_timestamp, 2),
        status=status,
        defect_type=defect_type,
        body_area=body_area,
        risk_level=str(risk_level).upper(),
        form_scores=form_scores,
        features=stored_features,
        feedback=feedback_text,
        model_version=str(model_version) if model_version else "v1.0.0",
        model_confidence=float(model_confidence) if model_confidence is not None else 0.85,
    )
    rep_doc = rep.model_dump(by_alias=True)

    await database.reps.insert_one(rep_doc)
    return rep_doc


async def finish_session(
    database: Any,
    user_id: str,
    session_id: str,
    summary: dict | None = None,
) -> dict:
    reps_cursor = database.reps.find({
        "session_id": session_id,
        "user_id": user_id,
    })
    reps = await reps_cursor.to_list(length=None) if hasattr(reps_cursor, "to_list") else getattr(reps_cursor, "documents", [])
    total_reps = len(reps)
    form_scores = []
    risk_counts = {"low": 0, "moderate": 0, "high": 0}
    for rep in reps:
        rep_form_scores = rep.get("form_scores", {})
        overall_score = rep_form_scores.get("overall")
        if isinstance(overall_score, (int, float)) and isfinite(overall_score):
            form_scores.append(float(overall_score))
        risk_level = str(rep.get("risk_level", "")).lower()
        if risk_level in risk_counts:
            risk_counts[risk_level] += 1

    avg_form_score = round(sum(form_scores) / len(form_scores), 1) if form_scores else None
    overall_risk = max(
        risk_counts,
        key=lambda level: (risk_counts[level], _RISK_SEVERITY[level]),
    ) if total_reps else None

    ended_at = datetime.now(timezone.utc)
    overall_risk_val = overall_risk.lower() if overall_risk else None
    finalized_summary = dict(summary or {})
    finalized_summary.update({
        "total_reps": total_reps,
        "average_form_score": avg_form_score,
        "low_risk_reps": risk_counts["low"],
        "moderate_risk_reps": risk_counts["moderate"],
        "high_risk_reps": risk_counts["high"],
    })

    await database.sessions.update_one(
        {"_id": session_id, "user_id": user_id},
        {"$set": {
            "ended_at": ended_at,
            "avg_form_score": avg_form_score,
            "overall_risk": overall_risk_val,
        }},
    )

    return {
        **finalized_summary,
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

    reps_cursor = database.reps.find({
        "session_id": session_id,
        "user_id": user_id,
    })
    reps = await reps_cursor.to_list(length=None) if hasattr(reps_cursor, "to_list") else getattr(reps_cursor, "documents", [])
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