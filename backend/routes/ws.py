from __future__ import annotations

import logging
import math
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from jose import JWTError

from core.config import Settings
from core.security import decode_access_token
from services.session_report_service import create_session_report
from services.session_service import (
    create_session,
    finish_session,
    record_completed_rep,
)
from src.landmarks import BodyLandmarks, LandmarkPoint, MEDIAPIPE_INDEX_MAP
from src.pipeline import AnalysisPipeline, FrameStatus, RepResult
from src.session_manager import SessionManager


logger = logging.getLogger(__name__)
router = APIRouter(tags=["live"])


def _serialize_rep(result: RepResult, rep_number: int) -> dict[str, Any]:
    return {
        "rep_number": rep_number,
        "duration_seconds": round(result.rep.duration_seconds, 2),
        "form_scores": result.form_scores.to_dict(),
        "risk": result.risk_result.to_dict(),
        "feedback": {
            "warning_tag": result.feedback.warning_tag,
            "reason": result.feedback.reason,
            "recommendation": result.feedback.recommendation,
            "all_issues": result.feedback.all_issues,
        },
        "features": result.features.to_dict(),
        "alignment": result.features.alignment,
    }


def _parse_frame(payload: Any) -> tuple[BodyLandmarks | None, float] | None:
    if not isinstance(payload, dict):
        return None
    timestamp = payload.get("t")
    if (
        isinstance(timestamp, bool)
        or not isinstance(timestamp, (int, float))
        or not math.isfinite(timestamp)
    ):
        return None

    raw_landmarks = payload.get("landmarks")
    if raw_landmarks is None:
        return None, float(timestamp)
    if not isinstance(raw_landmarks, dict):
        return None

    points: dict[str, LandmarkPoint] = {}
    optional_landmarks = {"left_foot_index", "right_foot_index"}
    for name in MEDIAPIPE_INDEX_MAP:
        raw_point = raw_landmarks.get(name)
        if raw_point is None and name in optional_landmarks:
            continue
        if not isinstance(raw_point, dict):
            return None
        values = (
            raw_point.get("x"),
            raw_point.get("y"),
            raw_point.get("visibility", 1.0),
            raw_point.get("z"),
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in values[:3]
        ):
            return None
        x, y, visibility, z = values
        if not 0.0 <= visibility <= 1.0:
            return None
        if z is not None and (
            isinstance(z, bool) or not isinstance(z, (int, float)) or not math.isfinite(z)
        ):
            return None
        points[name] = LandmarkPoint(
            float(x), float(y), float(visibility), float(z or 0.0),
            raw_point.get("image_x"), raw_point.get("image_y"),
        )
    low_confidence = any(raw_landmarks[name].get("z") is None for name in MEDIAPIPE_INDEX_MAP if name not in optional_landmarks)
    return BodyLandmarks(points, low_confidence=low_confidence), float(timestamp)


async def _authenticate(websocket: WebSocket) -> dict | None:
    token = websocket.query_params.get("token")
    if not token:
        return None
    try:
        claims = decode_access_token(token, Settings.from_env().jwt_secret)
    except (JWTError, ValueError):
        return None

    user_id = claims.get("sub")
    role = claims.get("role")
    if not isinstance(user_id, str) or not user_id or role not in {"gym", "athlete"}:
        return None
    user = await websocket.app.state.database.users.find_one({"_id": user_id})
    if user is None or user.get("role") != role:
        return None
    return user


@router.websocket("/ws/live")
async def live_analysis(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        user = await _authenticate(websocket)
    except Exception:
        logger.exception("Could not authenticate live WebSocket")
        await websocket.close(code=1011, reason="Authentication service unavailable")
        return
    if user is None:
        await websocket.close(code=4401, reason="Invalid or expired access token")
        return

    activity = websocket.query_params.get("activity")
    if activity != "squat":
        await websocket.close(code=4400, reason="Unsupported live activity")
        return
    if user["role"] != "gym":
        await websocket.close(code=4403, reason="Gym account required")
        return

    try:
        rep_offset = int(websocket.query_params.get("rep_offset", "0"))
    except ValueError:
        await websocket.close(code=4400, reason="Invalid rep offset")
        return
    if not 0 <= rep_offset <= 1_000_000:
        await websocket.close(code=4400, reason="Invalid rep offset")
        return

    database = websocket.app.state.database
    user_id = str(user["_id"])
    pipeline = AnalysisPipeline()
    manager = SessionManager(exercise_name="SQUAT")
    completed_reps: list[dict] = []
    latest_risk_level = "LOW"
    latest_risk_score = 0.0
    latest_cue = "Stand tall to begin your squat."
    close_code = 1000
    end_requested = False
    session_id: str | None = None

    try:
        session_id = await create_session(database, user_id, "SQUAT")
        await websocket.send_json({
            "type": "ready",
            "session_id": session_id,
            "rep_count": rep_offset,
            "phase": "Standing",
            "risk_level": latest_risk_level,
            "risk_score": latest_risk_score,
            "cue": latest_cue,
            "view": "unknown",
            "alignment": None,
        })

        while True:
            try:
                payload = await websocket.receive_json()
            except ValueError:
                close_code = 4400
                break
            if isinstance(payload, dict) and payload.get("type") == "end":
                end_requested = True
                break

            frame = _parse_frame(payload)
            if frame is None:
                close_code = 4400
                break
            landmarks, timestamp = frame
            status: FrameStatus = pipeline.process_frame(landmarks, timestamp)
            rep_payload = None
            if status.completed_rep is not None:
                result = status.completed_rep
                rep_number = rep_offset + result.rep.rep_number
                rep_payload = _serialize_rep(result, rep_number)
                manager.record_rep(
                    rep_number,
                    result.form_scores,
                    result.risk_result,
                    result.feedback,
                )
                await record_completed_rep(
                    database,
                    user_id,
                    session_id,
                    manager.start_time,
                    rep_payload,
                )
                completed_reps.append(rep_payload)
                latest_risk_level = result.risk_result.risk_level
                latest_risk_score = result.risk_result.risk_score
                latest_cue = result.feedback.recommendation

            visible = status.landmarks_ok
            await websocket.send_json({
                "type": "frame",
                "phase": status.phase.value,
                "rep_count": rep_offset + status.rep_count,
                "risk_level": latest_risk_level,
                "risk_score": latest_risk_score,
                "cue": latest_cue if visible else "Step back so your full body is visible.",
                "message": None if visible else "Step back so your full body is visible",
                "landmarks_ok": visible,
                "completed_rep": rep_payload,
                "view": status.view,
                "alignment": status.alignment,
            })
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("Live analysis failed for user %s", user_id)
        close_code = 1011
    finally:
        if session_id is not None:
            try:
                manager.end_session()
                summary = manager.summary()
                await finish_session(database, user_id, session_id, summary)
                report = await create_session_report(
                    database, user_id, session_id, summary, completed_reps,
                )
                if end_requested:
                    await websocket.send_json({
                        "type": "ended",
                        "session_id": session_id,
                        "summary": summary,
                        "report_id": report["_id"],
                    })
            except Exception:
                logger.exception("Could not finalize live session %s", session_id)
                close_code = 1011
        try:
            await websocket.close(code=close_code)
        except (RuntimeError, WebSocketDisconnect):
            pass
