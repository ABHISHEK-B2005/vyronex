from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status

from core.deps import get_current_user
from services.coach import MEDICAL_DISCLAIMER

router = APIRouter(prefix="/api/history", tags=["history"])


def _calculate_streak_days(session_dates: List[datetime]) -> int:
    """Calculate consecutive workout days streak."""
    if not session_dates:
        return 0
    
    unique_days = sorted(list({d.date() for d in session_dates}), reverse=True)
    if not unique_days:
        return 0

    streak = 1
    for i in range(len(unique_days) - 1):
        delta = (unique_days[i] - unique_days[i + 1]).days
        if delta == 1:
            streak += 1
        else:
            break
    return streak


def _compute_recurring_weakness(reports: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Detect if a body area has been flagged repeatedly in recent sessions."""
    recent = reports[:5]
    if not recent:
        return None

    area_counts: Counter[str] = Counter()
    total_recent = len(recent)

    for rep in recent:
        breakdown = rep.get("body_area_breakdown", {})
        for area, pct in breakdown.items():
            if pct >= 25:
                area_counts[area] += 1

    if not area_counts:
        return None

    most_flagged_area, count = area_counts.most_common(1)[0]
    if count >= 2:
        return {
            "area": most_flagged_area,
            "flagged_sessions": count,
            "total_checked_sessions": total_recent,
            "message": f"{most_flagged_area.replace('_', ' ').title()} flagged in {count} of your last {total_recent} sessions.",
            "recommendation": f"Incorporate targeted {most_flagged_area.replace('_', ' ')} corrective activation before training.",
        }
    return None


def _predict_future_risk(reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Screening estimate combining repeated defects, rising fatigue drop, and frequency."""
    if not reports:
        return {
            "level": "LOW",
            "title": "Optimal Baseline",
            "message": "Complete more sessions to calibrate your ongoing movement risk profile.",
            "disclaimer": MEDICAL_DISCLAIMER,
        }

    recent = reports[:4]
    high_risk_reps_total = sum(r.get("totals", {}).get("high_risk", 0) for r in recent)
    avg_fatigue_drop = sum(r.get("fatigue", {}).get("drop_pct", 0) for r in recent) / len(recent)
    avg_accuracy = sum(r.get("accuracy_pct", 85) for r in recent) / len(recent)

    if high_risk_reps_total >= 3 or avg_fatigue_drop >= 20.0 or avg_accuracy < 65.0:
        return {
            "level": "ELEVATED",
            "title": "Elevated Risk This Week",
            "message": (
                "Multiple high-risk deviations and high fatigue drops detected across recent sessions. "
                "Deload weight by 15% and focus on movement tempo to protect joint integrity."
            ),
            "factors": ["High fatigue drop rate", "Repeated kinematic errors"],
            "disclaimer": MEDICAL_DISCLAIMER,
        }
    elif high_risk_reps_total >= 1 or avg_fatigue_drop >= 12.0:
        return {
            "level": "MODERATE",
            "title": "Moderate Fatigue Accumulation",
            "message": "Form is mostly solid but shows mild degradation as reps progress. Maintain adequate hydration and rest.",
            "factors": ["Mild fatigue drop"],
            "disclaimer": MEDICAL_DISCLAIMER,
        }

    return {
        "level": "LOW",
        "title": "Low Injury Risk",
        "message": "Technique stability and neuromuscular control are consistent across recent sessions. Keep up the great form!",
        "factors": ["High form stability", "Low fatigue decay"],
        "disclaimer": MEDICAL_DISCLAIMER,
    }


def _evaluate_badges(sessions: List[Dict[str, Any]], reports: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Determine unlocked achievement badges."""
    badges = []
    
    total_sessions = len(sessions)
    if total_sessions >= 1:
        badges.append({
            "id": "first_step",
            "title": "First Session",
            "icon": "🎯",
            "description": "Completed your first biomechanical movement screening.",
            "unlocked": True,
        })

    total_reps = sum(r.get("totals", {}).get("reps", 0) for r in reports)
    if total_reps >= 50:
        badges.append({
            "id": "rep_50",
            "title": "50 Rep Club",
            "icon": "⚡",
            "description": "Analyzed over 50 completed repetitions with computer vision.",
            "unlocked": True,
        })

    has_90_accuracy = any(r.get("accuracy_pct", 0) >= 90 for r in reports)
    if has_90_accuracy:
        badges.append({
            "id": "form_master",
            "title": "Form Master (90%+)",
            "icon": "🏆",
            "description": "Achieved 90%+ flawless form accuracy in a workout session.",
            "unlocked": True,
        })

    session_dates = [s.get("started_at") for s in sessions if s.get("started_at")]
    streak = _calculate_streak_days(session_dates)
    if streak >= 3:
        badges.append({
            "id": "streak_champ",
            "title": f"{streak}-Day Streak",
            "icon": "🔥",
            "description": f"Maintained consistent training discipline across {streak} consecutive days.",
            "unlocked": True,
        })

    return badges


@router.get("", status_code=status.HTTP_200_OK)
async def get_user_history(
    request: Request,
    activity: Optional[str] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    limit: int = 100,
    user: dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """Retrieve filtered history, trends, recurring weakness, badges, and future risk prediction."""
    db = request.app.state.database
    user_id = str(user["_id"])

    session_query: Dict[str, Any] = {"user_id": user_id}
    if activity:
        session_query["activity"] = activity.lower()

    if from_date or to_date:
        date_filter: Dict[str, Any] = {}
        if from_date:
            try:
                date_filter["$gte"] = datetime.fromisoformat(from_date)
            except Exception:
                pass
        if to_date:
            try:
                date_filter["$lte"] = datetime.fromisoformat(to_date)
            except Exception:
                pass
        if date_filter:
            session_query["started_at"] = date_filter

    cursor = db.sessions.find(session_query).sort("started_at", -1).limit(limit)
    sessions = await cursor.to_list(length=limit) if hasattr(cursor, "to_list") else getattr(cursor, "documents", [])

    # Fetch corresponding reports
    reports: List[Dict[str, Any]] = []
    history_items: List[Dict[str, Any]] = []

    accuracy_over_time: List[Dict[str, Any]] = []
    form_score_over_time: List[Dict[str, Any]] = []
    risk_weekly_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: {"LOW": 0, "MODERATE": 0, "HIGH": 0})

    for s in sessions:
        sid = str(s["_id"])
        rep = await db.reports.find_one({"session_id": sid}) if hasattr(db, "reports") else None
        
        started = s.get("started_at")
        date_str = started.strftime("%Y-%m-%d") if started and hasattr(started, "strftime") else "2026-10-06"
        
        acc = float(rep.get("accuracy_pct", 80.0)) if rep else (
            85.0 if s.get("overall_risk") == "low" else 65.0
        )
        score = float(s.get("avg_form_score") or (rep.get("avg_form_score") if rep else 80.0))
        risk_lvl = str(s.get("overall_risk") or (rep.get("overall_risk") if rep else "LOW")).upper()

        if rep:
            reports.append(rep)

        history_items.append({
            "session_id": sid,
            "activity": s.get("activity", "squat"),
            "source": s.get("source", "live"),
            "date": date_str,
            "started_at": started,
            "total_reps": rep.get("totals", {}).get("reps", 0) if rep else len(s.get("reps", [])),
            "accuracy_pct": acc,
            "avg_form_score": score,
            "overall_risk": risk_lvl,
            "most_common_defect": rep.get("most_common_defect") if rep else None,
        })

        # Chart time-series points (in chronological order when reversed)
        accuracy_over_time.append({"date": date_str, "value": acc, "session_id": sid})
        form_score_over_time.append({"date": date_str, "value": score, "session_id": sid})

        # Group by week
        week_key = started.strftime("%Y-W%W") if started and hasattr(started, "strftime") else "2026-W40"
        if risk_lvl in risk_weekly_counts[week_key]:
            risk_weekly_counts[week_key][risk_lvl] += 1

    recurring_weakness = _compute_recurring_weakness(reports)
    future_risk = _predict_future_risk(reports)
    badges = _evaluate_badges(sessions, reports)
    streak_days = _calculate_streak_days([s.get("started_at") for s in sessions if s.get("started_at")])

    return {
        "count": len(history_items),
        "streak_days": streak_days,
        "sessions": history_items,
        "charts": {
            "accuracy_over_time": list(reversed(accuracy_over_time)),
            "form_score_over_time": list(reversed(form_score_over_time)),
            "risk_weekly_counts": dict(risk_weekly_counts),
        },
        "recurring_weakness": recurring_weakness,
        "predicted_future_risk": future_risk,
        "badges": badges,
        "disclaimer": MEDICAL_DISCLAIMER,
    }
