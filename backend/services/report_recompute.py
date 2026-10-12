from __future__ import annotations

from typing import Any

from services.report_builder import build_report
from services.session_service import finish_session, record_completed_rep
from services.video_processor import analyze_pose_sequence
from src.landmarks import BodyLandmarks, LandmarkPoint


def _restore_landmarks(raw: dict | None, low_confidence: bool) -> BodyLandmarks | None:
    if not raw:
        return None
    points = {
        name: LandmarkPoint(
            x=float(values["x"]),
            y=float(values["y"]),
            z=float(values.get("z", 0.0)),
            visibility=float(values.get("visibility", 0.0)),
        )
        for name, values in raw.items()
    }
    return BodyLandmarks(points, low_confidence=low_confidence)


async def recompute_report(
    session_id: str,
    database: Any,
    user_id: str | None = None,
) -> dict:
    """Rebuild reps and report using the numeric pose samples saved for a session."""
    session_query = {"_id": session_id}
    if user_id is not None:
        session_query["user_id"] = user_id
    session = await database.sessions.find_one(session_query)
    if not session:
        raise ValueError(f"Session '{session_id}' does not exist.")

    cursor = database.pose_frames.find({"session_id": session_id}).sort("timestamp", 1)
    stored_frames = (
        await cursor.to_list(length=None)
        if hasattr(cursor, "to_list")
        else getattr(cursor, "documents", [])
    )
    if not stored_frames:
        raise ValueError("This session has no saved pose frames and cannot be recalculated.")
    frames = [
        {
            "timestamp": float(frame["timestamp"]),
            "landmarks": _restore_landmarks(
                frame.get("landmarks"),
                bool(frame.get("low_confidence", False)),
            ),
        }
        for frame in stored_frames
    ]
    reps, debug_frames, summary = analyze_pose_sequence(frames)

    await database.reps.delete_many({"session_id": session_id})
    for rep in reps:
        await record_completed_rep(
            database,
            str(session["user_id"]),
            session_id,
            started_at=0.0,
            result=rep,
            timestamp_sec=rep["timestamp_sec"],
        )
    summary["duration_seconds"] = float(session.get("duration_sec") or 0.0)
    await finish_session(database, str(session["user_id"]), session_id, summary)

    await database.pose_frames.delete_many({"session_id": session_id})
    for frame in debug_frames:
        await database.pose_frames.insert_one({"session_id": session_id, **frame})

    return await build_report(database, session_id, str(session["user_id"]))
