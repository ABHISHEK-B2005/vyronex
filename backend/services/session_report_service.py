from __future__ import annotations

import uuid
from typing import Any

from models.report import ReportRecord


async def create_session_report(
    database: Any,
    user_id: str,
    session_id: str,
    summary: dict,
    completed_reps: list[dict],
) -> dict:
    defect_counts: dict[str, int] = {}
    recommendations = []
    worst_rep = None
    for rep in completed_reps:
        issue = rep["risk"].get("primary_issue")
        if issue:
            defect_counts[issue] = defect_counts.get(issue, 0) + 1
        recommendation = rep["feedback"].get("recommendation")
        if recommendation and recommendation not in recommendations:
            recommendations.append(recommendation)
        if worst_rep is None or rep["risk"]["risk_score"] > worst_rep["risk_score"]:
            worst_rep = {
                "rep_number": rep["rep_number"],
                "risk_score": rep["risk"]["risk_score"],
                "risk_level": rep["risk"]["risk_level"],
                "primary_issue": issue,
            }

    report = ReportRecord(
        id=uuid.uuid4().hex,
        session_id=session_id,
        user_id=user_id,
        totals={
            "total_reps": summary["total_reps"],
            "duration_seconds": summary["duration_seconds"],
            "average_form_score": summary["average_form_score"],
            "average_risk_score": summary["average_risk_score"],
        },
        defect_counts=defect_counts,
        recommendations=recommendations,
        worst_rep=worst_rep,
    ).model_dump(by_alias=True, exclude_none=True)
    await database.reports.insert_one(report)
    return report
