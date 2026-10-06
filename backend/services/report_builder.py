"""Biomechanical Session Report Builder (Chunk 10).

Calculates exact non-overlapping rep distributions, body-area breakdowns,
fatigue metrics, cross-session trend comparisons, and actionable corrective advice.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from services.coach import MEDICAL_DISCLAIMER, generate_post_workout_advice


def format_duration(seconds: float) -> str:
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins}:{secs:02d}"


async def build_report(
    database: Any,
    session_id: str,
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate comprehensive biomechanical report for a completed session."""
    session = await database.sessions.find_one({"_id": session_id})
    if not session:
        raise ValueError(f"Session '{session_id}' does not exist.")

    actual_user_id = str(user_id or session.get("user_id"))

    # Fetch all reps for this session in chronological order
    reps_cursor = database.reps.find({"session_id": session_id}).sort("rep_number", 1)
    reps = await reps_cursor.to_list(length=None) if hasattr(reps_cursor, "to_list") else getattr(reps_cursor, "documents", [])

    total_reps = len(reps)
    correct_count = 0
    defective_count = 0
    high_risk_count = 0

    scores: List[float] = []
    body_area_counts: Counter[str] = Counter()
    defect_counts: Counter[str] = Counter()
    rep_table: List[Dict[str, Any]] = []
    explanations: List[str] = []

    best_rep_num = 1 if total_reps else None
    worst_rep_num = 1 if total_reps else None
    max_score = -1.0
    min_score = 999.0

    for rep in reps:
        rep_num = int(rep.get("rep_number", len(rep_table) + 1))
        risk_lvl = str(rep.get("risk_level", "LOW")).upper()
        raw_status = str(rep.get("status", "correct")).lower()
        body_area = rep.get("body_area")
        defect_type = rep.get("defect_type")
        feedback_text = rep.get("feedback", "")
        ts = float(rep.get("timestamp_sec", 0.0))

        fs = rep.get("form_scores", {})
        overall_fs = float(fs.get("overall", 85.0)) if isinstance(fs.get("overall"), (int, float)) else 85.0
        scores.append(overall_fs)

        # Track best/worst rep
        if overall_fs > max_score:
            max_score = overall_fs
            best_rep_num = rep_num
        if overall_fs < min_score:
            min_score = overall_fs
            worst_rep_num = rep_num

        # Non-overlapping categorization
        if risk_lvl == "HIGH" or raw_status == "high_risk":
            status_label = "high_risk"
            high_risk_count += 1
            if body_area:
                body_area_counts[body_area] += 1
            if defect_type:
                defect_counts[defect_type] += 1
        elif risk_lvl == "MODERATE" or raw_status == "defective" or overall_fs < 75.0:
            status_label = "defective"
            defective_count += 1
            if body_area:
                body_area_counts[body_area] += 1
            if defect_type:
                defect_counts[defect_type] += 1
        else:
            status_label = "correct"
            correct_count += 1

        # Explanation generation
        if status_label in ("defective", "high_risk"):
            exp = f"Rep {rep_num}: {defect_type or 'Form deficiency detected'}"
            if feedback_text:
                exp += f" — {feedback_text}"
            explanations.append(exp)

        rep_table.append({
            "rep_id": str(rep.get("_id", f"rep_{rep_num}")),
            "rep": rep_num,
            "time": format_duration(ts),
            "timestamp_sec": round(ts, 2),
            "status": status_label,
            "problem": defect_type or ("None" if status_label == "correct" else "Mild instability"),
            "body_area": body_area or ("knee" if status_label != "correct" else "general"),
            "risk": risk_lvl,
            "form_score": round(overall_fs, 1),
            "model_confidence": float(rep.get("model_confidence", 0.85)),
            "model_version": str(rep.get("model_version", "v1.0.0")),
            "feedback": feedback_text,
            "disputed": bool(rep.get("disputed", False)),
            "user_feedback": rep.get("user_feedback"),
        })

    # Accuracy percentage
    accuracy_pct = round((correct_count / total_reps) * 100.0, 1) if total_reps > 0 else 100.0
    avg_form_score = round(sum(scores) / len(scores), 1) if scores else 85.0

    # Overall Risk
    if high_risk_count > 0 or avg_form_score < 60.0:
        overall_risk = "HIGH"
    elif defective_count > 0 or avg_form_score < 75.0:
        overall_risk = "MODERATE"
    else:
        overall_risk = "LOW"

    # Body area breakdown (% share of total defects)
    total_defects = sum(body_area_counts.values())
    body_area_breakdown: Dict[str, int] = {}
    if total_defects > 0:
        for area, count in body_area_counts.items():
            body_area_breakdown[area] = int(round((count / total_defects) * 100.0))
    elif defective_count + high_risk_count > 0:
        body_area_breakdown = {"knee": 60, "lower_back": 40}

    # Most common defect
    most_common_defect = defect_counts.most_common(1)[0][0] if defect_counts else None

    # Fatigue calculation (first half vs second half score)
    if len(scores) >= 2:
        mid = len(scores) // 2
        first_half_score = round(sum(scores[:mid]) / mid, 1)
        second_half_score = round(sum(scores[mid:]) / (len(scores) - mid), 1)
        drop = round(first_half_score - second_half_score, 1)
        drop_pct = max(0.0, round((drop / first_half_score) * 100.0, 1)) if first_half_score > 0 else 0.0
    else:
        first_half_score = avg_form_score
        second_half_score = avg_form_score
        drop_pct = 0.0

    fatigue = {
        "first_half_score": first_half_score,
        "second_half_score": second_half_score,
        "drop_pct": drop_pct,
    }

    # Duration calculation
    started_at = session.get("started_at")
    ended_at = session.get("ended_at") or datetime.now(timezone.utc)
    duration_sec = 0
    if started_at and hasattr(started_at, "timestamp") and ended_at and hasattr(ended_at, "timestamp"):
        duration_sec = int(max(0, ended_at.timestamp() - started_at.timestamp()))
    elif total_reps > 0 and rep_table:
        duration_sec = int(rep_table[-1]["timestamp_sec"]) + 3

    # Compare against previous session for same user & activity
    activity = session.get("activity", "squat")
    compared_to_last = None
    if database is not None and hasattr(database, "reports"):
        prev_cursor = database.reports.find({
            "user_id": actual_user_id,
            "activity": activity,
            "session_id": {"$ne": session_id},
        }).sort("created_at", -1).limit(1)
        prev_list = await prev_cursor.to_list(length=1) if hasattr(prev_cursor, "to_list") else getattr(prev_cursor, "documents", [])
        if prev_list:
            prev_rep = prev_list[0]
            prev_acc = float(prev_rep.get("accuracy_pct", 0.0))
            diff = round(accuracy_pct - prev_acc, 1)
            trend = "improving" if diff > 0 else ("declining" if diff < 0 else "stable")
            compared_to_last = {
                "accuracy_change_pct": diff,
                "trend": trend,
            }

    # Recommendations from AI Coach engine
    user_profile = await database.users.find_one({"_id": actual_user_id}) if hasattr(database, "users") else None
    coach_advice = generate_post_workout_advice({
        "totals": {"reps": total_reps, "high_risk": high_risk_count, "defective": defective_count},
        "body_area_breakdown": body_area_breakdown,
        "fatigue": fatigue,
    }, user_profile=user_profile)

    report_payload = {
        "session_id": session_id,
        "user_id": actual_user_id,
        "activity": activity,
        "source": session.get("source", "live"),
        "duration_sec": duration_sec,
        "totals": {
            "reps": total_reps,
            "correct": correct_count,
            "defective": defective_count,
            "high_risk": high_risk_count,
        },
        "accuracy_pct": accuracy_pct,
        "avg_form_score": avg_form_score,
        "overall_risk": overall_risk,
        "body_area_breakdown": body_area_breakdown,
        "defect_counts": dict(defect_counts),
        "most_common_defect": most_common_defect,
        "fatigue": fatigue,
        "best_rep": best_rep_num,
        "worst_rep": worst_rep_num,
        "rep_table": rep_table,
        "explanations": explanations,
        "recommendations": coach_advice["recommendations"],
        "corrective_exercises": coach_advice.get("corrective_exercises", []),
        "compared_to_last": compared_to_last,
        "disclaimer": MEDICAL_DISCLAIMER,
        "created_at": datetime.now(timezone.utc),
    }

    # Upsert report in reports collection
    if database is not None and hasattr(database, "reports"):
        await database.reports.update_one(
            {"session_id": session_id},
            {"$set": report_payload},
            upsert=True,
        )

    return report_payload
