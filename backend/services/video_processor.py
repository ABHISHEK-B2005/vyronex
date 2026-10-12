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

import asyncio
import logging
import math
import os
import statistics
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
from src.config import (
    MAX_FRAME_JUMP_TORSO_RATIO,
    MAX_UNUSABLE_FRAME_SHARE,
    VIEW_FRONT_MIN_RATIO,
    VIEW_SIDE_MAX_RATIO,
)
from src.feature_engineering import extract_rep_features
from src.feedback_engine import generate_feedback
from src.form_analysis import analyze_alignment, analyze_form
from src.geometry import view_width_ratio
from src.landmarks import BodyLandmarks
from src.pipeline import AnalysisPipeline, FrameStatus, RepResult, build_frame_sample
from src.pose_estimator import PoseEstimator, PoseEstimatorUnavailable
from src.pose_smoothing import PoseSmoother, landmark_visibility, pose_jump_ratio
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
        score = getattr(form_scores, category, None)
        if score is None:
            status = "not_assessed"
            msg = "Not assessed from this angle."
        elif score >= GOOD_MIN:
            status = "good"
            msg = copy["good"](features)
        elif score >= FAIR_MIN:
            status = "fair"
            msg = copy["fair"](features)
        else:
            status = "poor"
            msg = copy["poor"](features)
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


TRACKING_FAILURE_MESSAGE = (
    "We could not track you reliably. Use better light and keep your whole body in view."
)


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
        "alignment": rr.features.alignment,
        "checklist": build_checklist(rr.form_scores, rr.features),
        "camera_view": rr.features.camera_view,
        "low_confidence": rr.features.low_confidence,
        "view_warning": (
            "Film from the front or the side for a full check."
            if rr.features.camera_view in {"angled", "unknown"}
            else None
        ),
    }


def analyze_pose_sequence(
    frames: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Analyze sampled pose data through the same path used by uploads and recompute."""
    smoother = PoseSmoother()
    smoothed = [smoother.smooth(row.get("landmarks")) for row in frames]
    ratios = [
        view_width_ratio(landmarks)
        for landmarks in smoothed
        if landmarks is not None and landmark_visibility(landmarks) >= 0.5
    ]
    ratios = [ratio for ratio in ratios if ratio is not None]
    median_ratio = statistics.median(ratios) if ratios else None
    if median_ratio is None:
        camera_view = "angled"
    elif median_ratio > VIEW_FRONT_MIN_RATIO:
        camera_view = "front"
    elif median_ratio < VIEW_SIDE_MAX_RATIO:
        camera_view = "side"
    else:
        camera_view = "angled"

    pipeline = AnalysisPipeline(risk_mode="rule_based", smooth_signals=True)
    manager = SessionManager(exercise_name="SQUAT")
    completed_reps = []
    debug_frames = []
    unusable = 0
    previous = None

    for row, landmarks in zip(frames, smoothed):
        timestamp = float(row["timestamp"])
        visibility = landmark_visibility(landmarks)
        jump_ratio = pose_jump_ratio(previous, row.get("landmarks"))
        tracking_error = False
        if landmarks is None:
            unusable += 1
            previous = None
            status = pipeline.process_frame(None, timestamp, camera_view)
            sample = None
        elif jump_ratio > MAX_FRAME_JUMP_TORSO_RATIO:
            unusable += 1
            tracking_error = True
            previous = row["landmarks"]
            status = FrameStatus(
                False,
                pipeline.squat_detector.phase,
                pipeline.squat_detector.rep_count,
                message="Tracking error: joint position jumped between frames.",
                tracking_error=True,
            )
            sample = None
        else:
            previous = row["landmarks"]
            sample = None
            status = pipeline.process_frame(landmarks, timestamp, camera_view)
            if not status.landmarks_ok:
                unusable += 1
                tracking_error = status.tracking_error
            else:
                sample = build_frame_sample(landmarks, timestamp, camera_view)

        if status.completed_rep is not None:
            rr = status.completed_rep
            manager.record_rep(rr.rep.rep_number, rr.form_scores, rr.risk_result, rr.feedback)
            completed_reps.append(
                serialize_rep(rr, rr.rep.rep_number, rr.rep.end_time)
            )

        frame_debug = {
            "timestamp": round(timestamp, 4),
            "knee_angle": status.avg_knee_angle if status.landmarks_ok else (
                round(sample.avg_knee_angle, 2) if sample else None
            ),
            "torso_angle": sample.torso_angle if sample else None,
            "view": camera_view,
            "visibility": round(visibility, 3),
            "usable": bool(status.landmarks_ok),
            "tracking_error": tracking_error,
            "rep_phase": str(status.phase.value if hasattr(status.phase, "value") else status.phase),
            "landmarks": row["landmarks"].numeric_dict() if row.get("landmarks") else None,
            "low_confidence": bool(row.get("landmarks") and row["landmarks"].low_confidence),
        }
        debug_frames.append(frame_debug)

    final_rep = pipeline.squat_detector.finalize_current_rep(frames[-1]["timestamp"] if frames else 0.0)
    if final_rep is not None:
        features = extract_rep_features(final_rep)
        features.alignment = analyze_alignment(features)
        form_scores = analyze_form(features)
        risk_result = pipeline._risk_engine.assess(form_scores, features)
        feedback = generate_feedback(risk_result, form_scores, features)
        rr = RepResult(
            rep=final_rep,
            features=features,
            form_scores=form_scores,
            risk_result=risk_result,
            feedback=feedback,
        )
        manager.record_rep(rr.rep.rep_number, rr.form_scores, rr.risk_result, rr.feedback)
        completed_reps.append(serialize_rep(rr, rr.rep.rep_number, rr.rep.end_time))

    frame_count = max(1, len(frames))
    unusable_share = unusable / frame_count
    if unusable_share > MAX_UNUSABLE_FRAME_SHARE:
        if not any(row.get("landmarks") is not None for row in frames):
            raise NoPersonDetectedError(TRACKING_FAILURE_MESSAGE)
        raise VideoProcessingError(TRACKING_FAILURE_MESSAGE)

    manager.end_session()
    summary = manager.summary()
    low_confidence_frames = sum(
        bool(row.get("landmarks") and row["landmarks"].low_confidence)
        for row in frames
    )
    warnings = []
    if camera_view in {"angled", "unknown"}:
        warnings.append("Film from the front or the side for a full check.")
    if low_confidence_frames:
        warnings.append("Some frames used image-only landmarks; scores may be less reliable.")
    if not completed_reps:
        warnings.append("No complete squat repetitions were detected.")
    summary.update({
        "camera_view": camera_view,
        "tracking_quality": round(1 - unusable_share, 3),
        "unusable_frames": unusable,
        "low_confidence_frames": low_confidence_frames,
        "frame_count": len(frames),
        "analysis_warning": " ".join(warnings) or None,
    })
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "Upload knee-angle time series (seconds, degrees): %s",
            [(frame["timestamp"], frame["knee_angle"]) for frame in debug_frames],
        )
    return completed_reps, debug_frames, summary


class VideoProcessor:
    """Processes video clips using OpenCV and MediaPipe Pose."""

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
            except PoseEstimatorUnavailable as exc:
                raise VideoProcessingError(f"Pose analysis is unavailable: {exc}") from exc

            pose_frames: list[dict[str, Any]] = []
            frame_index = 0

            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                timestamp = frame_index / fps
                landmarks, _results, _person_detected = estimator.process(frame)
                pose_frames.append({"timestamp": timestamp, "landmarks": landmarks})
                frame_index += 1
                if progress_callback is not None:
                    progress = min(65.0, frame_index / max(1, total_frames) * 65.0)
                    progress_callback(round(progress, 1))
                if frame_index % 30 == 0:
                    await asyncio.sleep(0)

        finally:
            if "estimator" in locals():
                estimator.close()
            cap.release()

        completed_reps, frame_debug, summary = analyze_pose_sequence(pose_frames)

        video_duration_sec = round(frame_index / fps, 2) if frame_index > 0 else 0.0
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

                pose_collection = database.pose_frames
                stored_frames = [
                    {"session_id": session_id, **debug_frame}
                    for debug_frame in frame_debug
                ]
                if hasattr(pose_collection, "insert_many"):
                    for offset in range(0, len(stored_frames), 500):
                        await pose_collection.insert_many(stored_frames[offset:offset + 500])
                else:
                    for stored_frame in stored_frames:
                        await pose_collection.insert_one(stored_frame)

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
                raise VideoProcessingError(
                    f"Video analysis completed but could not be saved: {exc}"
                ) from exc

        if progress_callback is not None:
            progress_callback(100.0)

        return {
            "session_id": session_id,
            "report_id": report_id,
            "activity": activity,
            "summary": summary,
            "reps": completed_reps,
            "total_frames": frame_index,
            "frames_with_person": sum(frame["visibility"] > 0 for frame in frame_debug),
            "camera_view": summary["camera_view"],
            "tracking_quality": summary["tracking_quality"],
            "analysis_warning": summary["analysis_warning"],
            "fps": fps,
            "duration_seconds": summary.get("duration_seconds", video_duration_sec),
        }
