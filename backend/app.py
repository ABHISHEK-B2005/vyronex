"""VYRONEX backend API.

Thin FastAPI wrapper around the analysis pipeline from the
injury-risk-detection-prototype (src/). Pose detection itself happens in
the browser (MediaPipe Pose JS) — this backend receives the extracted
body-landmark coordinates for each frame and runs the *real* rule-based
squat-detection, form-scoring, risk-engine and feedback-generation code
that lives in src/, exactly as it would in the original Streamlit app.

Run with:
    uvicorn app:app --reload --port 8000

PROTOTYPE DISCLAIMER: movement-risk screening only, not a medical
diagnostic system. See src/config.py DISCLAIMER.
"""
from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from typing import Dict, Optional

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.landmarks import BodyLandmarks, LandmarkPoint
from src.pipeline import AnalysisPipeline
from src.session_manager import SessionManager
from src.demo_mode import build_demo_sequence
from src.ring_session import RingReading, RingSessionManager
from src.config import DISCLAIMER
from core.database import close_mongo_connection, connect_to_mongo
from core.deps import get_current_user
from routes.auth import router as auth_router
from routes.athlete import router as athlete_router
from routes.gym import router as gym_router
from routes.upload import router as upload_router
from routes.ws import router as live_router
from routes.feedback import router as feedback_router
from routes.expert import router as expert_router
from routes.coach import router as coach_router
from routes.reports import router as reports_router
from routes.history import router as history_router
from routes.alignment import router as alignment_router
from services.session_service import (
    create_session,
    finish_session,
    get_session_summary,
    record_completed_rep,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await connect_to_mongo(app)
    try:
        yield
    finally:
        await close_mongo_connection(app)


app = FastAPI(title="VYRONEX Backend", version="0.2.0", lifespan=lifespan)
app.include_router(auth_router)
app.include_router(gym_router)
app.include_router(athlete_router)
app.include_router(upload_router)
app.include_router(live_router)
app.include_router(feedback_router)
app.include_router(expert_router)
app.include_router(coach_router)
app.include_router(reports_router)
app.include_router(history_router)
app.include_router(alignment_router)

# Wide-open CORS: this is a local prototype meant to be hit from a
# frontend served on a different origin/port (e.g. a local static file
# server or file:// page during development).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store: session_id -> {pipeline, manager}
# NOTE: prototype only. Restarting the server clears all sessions.
_sessions: Dict[str, dict] = {}

# In-memory Smart Ring session store: session_id -> RingSessionManager.
# Kept in a completely separate dict from _sessions above — Smart Ring
# sessions never touch the squat pipeline/landmarks code path.
_ring_sessions: Dict[str, RingSessionManager] = {}


# ---------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------
class LandmarkIn(BaseModel):
    x: float
    y: float
    visibility: float = 1.0
    z: Optional[float] = None
    image_x: Optional[float] = None
    image_y: Optional[float] = None


class StartSessionIn(BaseModel):
    exercise: str = "SQUAT"
    risk_mode: str = "rule_based"  # "rule_based" | "ml"


class FrameIn(BaseModel):
    session_id: str
    landmarks: Dict[str, LandmarkIn]
    t: Optional[float] = None


class StartRingSessionIn(BaseModel):
    mode: str = "demo"  # "demo" today; "device" reserved for a future real Smart Ring integration


class RingReadingIn(BaseModel):
    session_id: str
    heart_rate: int
    hrv: int
    accel_magnitude: float
    source: str = "simulated_demo"  # "simulated_demo" | "smart_ring_device"
    t: Optional[float] = None


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _rep_result_to_dict(rr) -> dict:
    return {
        "rep_number": rr.rep.rep_number,
        "duration_seconds": round(rr.rep.duration_seconds, 2),
        "form_scores": rr.form_scores.to_dict(),
        "risk": rr.risk_result.to_dict(),
        "feedback": {
            "warning_tag": rr.feedback.warning_tag,
            "reason": rr.feedback.reason,
            "recommendation": rr.feedback.recommendation,
            "all_issues": rr.feedback.all_issues,
        },
        "features": rr.features.to_dict(),
        "checklist": build_checklist(rr.form_scores, rr.features),
        "alignment": rr.features.alignment,
    }


# ---------------------------------------------------------------------
# "What's correct / what's wrong" checklist
#
# The rule-based risk engine only ever names ONE primary issue per rep
# (the biggest contributor). For a live coaching view we want a full
# per-factor breakdown — every category, labeled GOOD / BORDERLINE /
# NEEDS WORK, each sentence grounded in the actual measured feature
# value for that rep (never a generic line). Thresholds mirror the
# same 0-100 category scores the pipeline already computes.
# ---------------------------------------------------------------------
GOOD_MIN = 70
FAIR_MIN = 45

_CHECKLIST_COPY = {
    "knee_alignment": {
        "label": "Knee Alignment",
        "good": lambda f: f"Knees tracked well over your feet — no meaningful inward/outward shift (measured shift {f.knee_alignment:.1%}).",
        "fair": lambda f: f"Knees drifted slightly out of line during the rep (shift {f.knee_alignment:.1%}) — worth watching.",
        "poor": lambda f: f"Knees moved noticeably inward/outward during the lowering phase (shift {f.knee_alignment:.1%}).",
    },
    "knee_depth": {
        "label": "Squat Depth",
        "good": lambda f: f"Depth was in a solid, controlled range (lowest knee angle {f.min_knee_angle:.0f}°).",
        "fair": lambda f: f"Depth was a little outside the target range (lowest knee angle {f.min_knee_angle:.0f}°).",
        "poor": lambda f: f"Depth (lowest knee angle {f.min_knee_angle:.0f}°) was well outside a comfortable squat range — either too shallow or too deep.",
    },
    "back_posture": {
        "label": "Back / Torso Posture",
        # NOTE: src's raw `torso_angle` feature is an "uprightness" angle
        # (90 = fully vertical, 0 = bent horizontal) per config.py, which
        # reads confusingly as a lean amount. `forward_lean` (= 90 -
        # torso_angle) is the intuitive "how far forward are you leaning"
        # number, so the checklist uses that instead for clarity.
        "good": lambda f: f"Torso stayed upright — forward lean averaged {f.forward_lean:.0f}°, within a comfortable range.",
        "fair": lambda f: f"Torso leaned forward a bit more than ideal (averaged {f.forward_lean:.0f}° of forward lean).",
        "poor": lambda f: f"Torso leaned forward significantly (averaged {f.forward_lean:.0f}° of forward lean) — more than the target range for this movement.",
    },
    "movement_control": {
        "label": "Movement Control / Tempo",
        "good": lambda f: f"Tempo was smooth and controlled (rep took {f.rep_duration:.1f}s).",
        "fair": lambda f: f"Movement was a little rushed or uneven (rep took {f.rep_duration:.1f}s).",
        "poor": lambda f: f"Movement speed/smoothness suggests the rep was rushed or not fully controlled (rep took {f.rep_duration:.1f}s).",
    },
    "symmetry": {
        "label": "Left / Right Symmetry",
        # The underlying score is the WORSE of knee-angle diff and
        # hip-angle diff (see src/form_analysis.py::_score_symmetry), so
        # the message must report both — showing only knee_symmetry can
        # read as "0.0° but scored POOR" if the hip side is what's
        # actually driving the low score.
        "good": lambda f: f"Both legs moved symmetrically (knee diff {f.knee_symmetry:.1f}°, hip diff {f.hip_symmetry:.1f}°).",
        "fair": lambda f: f"Slight left/right imbalance detected (knee diff {f.knee_symmetry:.1f}°, hip diff {f.hip_symmetry:.1f}°).",
        "poor": lambda f: f"Noticeable left/right asymmetry (knee diff {f.knee_symmetry:.1f}°, hip diff {f.hip_symmetry:.1f}°) — one side is doing more work.",
    },
}


def build_checklist(form_scores, features) -> list:
    items = []
    for category, copy in _CHECKLIST_COPY.items():
        score = getattr(form_scores, category)
        if score >= GOOD_MIN:
            status = "good"
        elif score >= FAIR_MIN:
            status = "fair"
        else:
            status = "poor"
        items.append({
            "category": category,
            "label": copy["label"],
            "score": score,
            "status": status,
            "message": copy[status](features),
        })
    return items


# ---------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------
@app.get("/api/health")
async def health(request: Request):
    try:
        await request.app.state.db.command("ping")
    except Exception as exc:
        raise HTTPException(503, "MongoDB is unavailable") from exc
    return {"status": "ok", "db": "connected"}


@app.post("/api/session/start")
async def start_session(
    body: StartSessionIn,
    request: Request,
    current_user: dict = Depends(get_current_user),
):
    try:
        pipeline = AnalysisPipeline(risk_mode=body.risk_mode)
    except Exception as exc:  # e.g. ML mode requested but no trained model
        raise HTTPException(400, f"Could not start pipeline: {exc}")
    user_id = str(current_user["_id"])
    try:
        session_id = await create_session(request.app.state.database, user_id, body.exercise)
    except Exception as exc:
        raise HTTPException(503, f"Could not create session record: {exc}") from exc
    manager = SessionManager(exercise_name=body.exercise)
    _sessions[session_id] = {
        "pipeline": pipeline,
        "manager": manager,
        "user_id": user_id,
    }
    return {"session_id": session_id, "exercise": body.exercise, "risk_mode": body.risk_mode}


@app.post("/api/frame")
async def process_frame(
    body: FrameIn,
    request: Request,
    current_user: dict = Depends(get_current_user),
):
    sess = _sessions.get(body.session_id)
    if not sess or sess["user_id"] != str(current_user["_id"]):
        raise HTTPException(404, "Unknown session_id — call /api/session/start first.")

    points = {
        name: LandmarkPoint(
            x=lm.x,
            y=lm.y,
            z=lm.z or 0.0,
            visibility=lm.visibility,
            image_x=lm.image_x,
            image_y=lm.image_y,
        )
        for name, lm in body.landmarks.items()
    }
    body_landmarks = BodyLandmarks(
        points,
        low_confidence=not all(lm.z is not None for lm in body.landmarks.values()),
    )

    status = sess["pipeline"].process_frame(body_landmarks, body.t)

    result: dict = {
        "landmarks_ok": status.landmarks_ok,
        "phase": status.phase.value if hasattr(status.phase, "value") else str(status.phase),
        "rep_count": status.rep_count,
        "avg_knee_angle": status.avg_knee_angle,
        "message": status.message,
        "completed_rep": None,
        "view": status.view,
        "alignment": status.alignment,
    }

    if status.completed_rep is not None:
        rr = status.completed_rep
        sess["manager"].record_rep(
            rr.rep.rep_number, rr.form_scores, rr.risk_result, rr.feedback
        )
        result["completed_rep"] = _rep_result_to_dict(rr)
        await record_completed_rep(
            request.app.state.database,
            sess["user_id"],
            body.session_id,
            sess["manager"].start_time,
            result["completed_rep"],
        )

    return result


@app.post("/api/session/{session_id}/end")
async def end_session(
    session_id: str,
    request: Request,
    current_user: dict = Depends(get_current_user),
):
    user_id = str(current_user["_id"])
    sess = _sessions.pop(session_id, None)
    if sess and sess["user_id"] == user_id:
        sess["manager"].end_session()
        summary = sess["manager"].summary()
        await finish_session(request.app.state.database, user_id, session_id, summary)
        return summary

    session = await request.app.state.database.sessions.find_one({"_id": session_id, "user_id": user_id})
    if not session:
        raise HTTPException(404, "Unknown session_id")

    summary = await get_session_summary(request.app.state.database, user_id, session_id)
    await finish_session(request.app.state.database, user_id, session_id, summary)
    return summary or {}


@app.get("/api/session/{session_id}/summary")
async def get_summary(
    session_id: str,
    request: Request,
    current_user: dict = Depends(get_current_user),
):
    user_id = str(current_user["_id"])
    sess = _sessions.get(session_id)
    if sess and sess.get("user_id") == user_id:
        return sess["manager"].summary()

    db_summary = await get_session_summary(request.app.state.database, user_id, session_id)
    if db_summary is not None:
        return db_summary

    raise HTTPException(404, "Unknown session_id")


@app.get("/api/demo/run")
def run_demo():
    """Runs the built-in SYNTHETIC squat sequence (no camera needed)
    entirely server-side, through the real pipeline, and returns every
    completed rep's full result plus a session summary. Useful as a
    guaranteed-working fallback demo if a camera isn't available.
    """
    pipeline = AnalysisPipeline(risk_mode="rule_based")
    manager = SessionManager(exercise_name="SQUAT (SERVER DEMO)")
    sequence = build_demo_sequence()

    reps_out = []
    for step in sequence:
        # The synthetic sequence has no built-in timestamps; without a
        # small delay here, frames get processed back-to-back in a few
        # milliseconds, which makes rep_duration collapse toward 0 and
        # any speed-based feature explode to meaningless values. Sleeping
        # ~1/15s per frame approximates a realistic webcam frame rate so
        # rep_duration (and therefore movement_control scoring) comes out
        # sensible, matching how the original Streamlit demo loop paced
        # itself with time.sleep(0.05) between frames.
        time.sleep(1 / 15)
        status = pipeline.process_frame(step.landmarks)
        if status.completed_rep is not None:
            rr = status.completed_rep
            manager.record_rep(rr.rep.rep_number, rr.form_scores, rr.risk_result, rr.feedback)
            entry = _rep_result_to_dict(rr)
            entry["segment_label"] = step.label
            reps_out.append(entry)

    manager.end_session()
    return {
        "reps": reps_out,
        "summary": manager.summary(),
        "note": "SYNTHETIC / SAMPLE DATA — generated by src/demo_mode.py, not a live camera feed.",
    }


# ---------------------------------------------------------------------
# SMART RING — isolated from the squat/pose pipeline above.
#
# No physical Smart Ring is connected: there is no Bluetooth pairing and
# no manufacturer SDK in this codebase. These endpoints exist so the
# frontend's Demo Mode (which generates clearly-labeled simulated
# readings) has something real to talk to, and so a future real-device
# integration can POST real readings through the exact same three calls
# (start -> reading x N -> end) without any API changes.
# ---------------------------------------------------------------------
@app.post("/api/ring/session/start")
def start_ring_session(body: StartRingSessionIn):
    if body.mode not in ("demo", "device"):
        raise HTTPException(400, "mode must be 'demo' or 'device'")
    if body.mode == "device":
        # Honest failure — no real hardware/SDK is wired up in this build.
        raise HTTPException(
            501,
            "Real Smart Ring hardware is not implemented in this backend yet. "
            "Start a session with mode='demo' instead.",
        )
    session_id = uuid.uuid4().hex[:12]
    _ring_sessions[session_id] = RingSessionManager(mode=body.mode)
    return {"session_id": session_id, "mode": body.mode}


@app.post("/api/ring/reading")
def post_ring_reading(body: RingReadingIn):
    sess = _ring_sessions.get(body.session_id)
    if not sess:
        raise HTTPException(404, "Unknown ring session_id — call /api/ring/session/start first.")
    if body.source not in ("simulated_demo", "smart_ring_device"):
        raise HTTPException(400, "source must be 'simulated_demo' or 'smart_ring_device'")

    reading = RingReading(
        t=body.t if body.t is not None else sess.duration_seconds,
        heart_rate=body.heart_rate,
        hrv=body.hrv,
        accel_magnitude=body.accel_magnitude,
        source=body.source,
    )
    sess.add_reading(reading)
    return {"sample_count": len(sess.readings), "summary_so_far": sess.summary()}


@app.post("/api/ring/session/{session_id}/end")
def end_ring_session(session_id: str):
    sess = _ring_sessions.pop(session_id, None)
    if not sess:
        raise HTTPException(404, "Unknown ring session_id")
    sess.end_session()
    return sess.summary()


@app.get("/api/ring/session/{session_id}/summary")
def get_ring_summary(session_id: str):
    sess = _ring_sessions.get(session_id)
    if not sess:
        raise HTTPException(404, "Unknown ring session_id")
    return sess.summary()
