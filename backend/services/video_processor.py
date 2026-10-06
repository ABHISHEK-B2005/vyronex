"""Video processor service for analyzing recorded workout clips.

Processes uploaded video files frame-by-frame:
1. Opens video using OpenCV VideoCapture.
2. Extracts frame rate (FPS) and video dimensions.
3. Steps through frames (every 2nd frame for speed optimization).
4. Runs MediaPipe Pose estimation to extract BodyLandmarks.
5. Feeds BodyLandmarks with exact timestamps (frame_index / fps) into AnalysisPipeline.
6. Collects completed reps with timestamp_sec.
7. Persists the session and rep data to the database (Chunk 7).
"""

from __future__ import annotations

import logging
import math
import os
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import cv2

from services.session_report_service import create_session_report
from services.session_service import (
    create_session,
    finish_session,
    record_completed_rep,
)
from src.landmarks import BodyLandmarks
from src.pipeline import AnalysisPipeline, FrameStatus, RepResult
from src.pose_estimator import PoseEstimator, PoseEstimatorUnavailable
from src.session_manager import SessionManager

logger = logging.getLogger(__name__)

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
        "good": lambda f: f"Both legs moved symmetrically (knee diff {f.knee_symmetry:.1f}°, hip diff {f.hip_symmetry:.1f}°).",
        "fair": lambda f: f"Slight left/right imbalance detected (knee diff {f.knee_symmetry:.1f}°, hip diff {f.hip_symmetry:.1f}°).",
        "poor": lambda f: f"Noticeable left/right asymmetry (knee diff {f.knee_symmetry:.1f}°, hip diff {f.hip_symmetry:.1f}°) — one side is doing more work.",
    },
}


def build_checklist(form_scores: Any, features: Any) -> list[dict]:
    items = []
    for category, copy in _CHECKLIST_COPY.items():
        score = getattr(form_scores, category, 0.0)
        if score >= GOOD_MIN:
            status = "good"
        elif score >= FAIR_MIN:
            status = "fair"
        else:
            status = "poor"
        msg = copy[status](features) if category in _CHECKLIST_COPY else ""
        items.append({
            "category": category,
            "label": copy["label"],
            "score": score,
            "status": status,
            "message": msg,
        })
    return items


class VideoProcessingError(Exception):
    """Base exception for video processing errors."""


class NoPersonDetectedError(VideoProcessingError):
    """Raised when no human body was detected in the video clip."""


class InvalidVideoFileError(VideoProcessingError):
    """Raised when the video file cannot be opened or decoded."""


def serialize_rep(rr: RepResult, rep_number: int, timestamp_sec: float) -> dict[str, Any]:
    """Serialize a completed rep result to match live mode data format with timestamps."""
    return {
        "rep_number": rep_number,
        "duration_seconds": round(rr.rep.duration_seconds, 2),
        "timestamp_sec": round(timestamp_sec, 2),
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
    }


class VideoProcessor:
    """Processes video clips using OpenCV and MediaPipe Pose."""

    def __init__(self, frame_step: int = 2) -> None:
        self.frame_step = max(1, frame_step)

    async def process_video(
        self,
        video_path: str | Path,
        activity: str = "squat",
        user_id: str = "anonymous",
        database: Any = None,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> dict[str, Any]:
        """Process a video file and return full session results and completed reps."""
        path_str = str(video_path)
        if not os.path.isfile(path_str):
            raise InvalidVideoFileError(f"Video file does not exist: {path_str}")

        cap = cv2.VideoCapture(path_str)
        if not cap.isOpened():
            raise InvalidVideoFileError("Could not open video file for decoding.")

        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            if not fps or fps <= 0 or math.isnan(fps):
                fps = 30.0

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                total_frames = 1

            try:
                estimator = PoseEstimator()
            except PoseEstimatorUnavailable:
                estimator = None

            pipeline = AnalysisPipeline(risk_mode="rule_based")
            manager = SessionManager(exercise_name=activity.upper())

            completed_reps: List[dict[str, Any]] = []
            frame_index = 0
            frames_with_person = 0

            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                if frame_index % self.frame_step == 0:
                    timestamp = frame_index / fps
                    if estimator is not None:
                        landmarks, _results, person_detected = estimator.process(frame)
                    else:
                        landmarks, _results, person_detected = None, None, False

                    if person_detected and landmarks is not None:
                        frames_with_person += 1


                    status: FrameStatus = pipeline.process_frame(landmarks, timestamp=timestamp)

                    if status.completed_rep is not None:
                        rr = status.completed_rep
                        rep_number = rr.rep.rep_number
                        manager.record_rep(
                            rep_number,
                            rr.form_scores,
                            rr.risk_result,
                            rr.feedback,
                        )
                        rep_data = serialize_rep(
                            rr,
                            rep_number=rep_number,
                            timestamp_sec=rr.rep.end_time,
                        )
                        completed_reps.append(rep_data)

                    if progress_callback is not None:
                        progress = min(99.0, (frame_index / max(1, total_frames)) * 100.0)
                        progress_callback(round(progress, 1))

                frame_index += 1

            if estimator is not None:
                estimator.close()
        finally:
            cap.release()

        if frames_with_person == 0:
            raise NoPersonDetectedError("No person detected in the uploaded video.")

        manager.end_session()
        summary = manager.summary()

        video_duration_sec = round(frame_index / fps, 2) if frame_index > 0 else 0.0
        if summary.get("duration_seconds", 0.0) == 0.0 and video_duration_sec > 0:
            summary["duration_seconds"] = video_duration_sec

        session_id = uuid.uuid4().hex
        report_id = uuid.uuid4().hex

        if database is not None:
            try:
                session_id = await create_session(
                    database=database,
                    user_id=user_id,
                    activity=activity.upper(),
                    source="upload",
                )

                for rep in completed_reps:
                    await record_completed_rep(
                        database=database,
                        user_id=user_id,
                        session_id=session_id,
                        started_at=0.0,
                        result=rep,
                        timestamp_sec=rep["timestamp_sec"],
                    )

                await finish_session(
                    database=database,
                    user_id=user_id,
                    session_id=session_id,
                    summary=summary,
                )

                report = await create_session_report(
                    database=database,
                    user_id=user_id,
                    session_id=session_id,
                    summary=summary,
                    completed_reps=completed_reps,
                )
                if report and "_id" in report:
                    report_id = str(report["_id"])
            except Exception as exc:
                logger.exception("Failed to persist uploaded session: %s", exc)

        if progress_callback is not None:
            progress_callback(100.0)

        return {
            "session_id": session_id,
            "report_id": report_id,
            "activity": activity,
            "summary": summary,
            "reps": completed_reps,
            "total_frames": frame_index,
            "frames_with_person": frames_with_person,
            "fps": fps,
            "duration_seconds": summary.get("duration_seconds", video_duration_sec),
        }
