"""Upload routes for video analysis.

Endpoints:
- POST /api/upload/analyze: accepts video file (mp4, mov, webm), processes in background, returns job_id.
- GET /api/upload/status/{job_id}: returns job status and progress percentage, and on completion returns session_id.
"""

from __future__ import annotations

import asyncio
import csv
import io
import logging
import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    Response,
    UploadFile,
    status,
)

from core.deps import get_current_user, get_current_user_optional
from services.report_recompute import recompute_report
from services.video_processor import (
    InvalidVideoFileError,
    NoPersonDetectedError,
    VideoProcessingError,
    VideoProcessor,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/upload", tags=["upload"])

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".webm"}
ALLOWED_MIME_TYPES = {
    "video/mp4",
    "video/quicktime",
    "video/webm",
    "video/x-matroska",
    "application/octet-stream",
}

# Max upload size: 100 MB
MAX_UPLOAD_SIZE = 100 * 1024 * 1024
CHUNK_SIZE = 1024 * 1024  # 1 MB chunk

# In-memory background jobs registry: job_id -> job_dict
_jobs: Dict[str, dict[str, Any]] = {}


def get_job(job_id: str) -> Optional[dict[str, Any]]:
    return _jobs.get(job_id)


async def _run_video_job(
    job_id: str,
    temp_path: str,
    activity: str,
    user_id: str,
    database: Any,
) -> None:
    """Worker function executed as a background task."""
    processor = VideoProcessor()

    def on_progress(percent: float) -> None:
        if job_id in _jobs:
            _jobs[job_id]["progress"] = percent
            _jobs[job_id]["message"] = f"Analyzing video frames ({round(percent)}%)."

    try:
        result = await processor.process_video(
            video_path=temp_path,
            activity=activity,
            user_id=user_id,
            database=database,
            progress_callback=on_progress,
        )

        if job_id in _jobs:
            _jobs[job_id].update({
                "status": "completed",
                "progress": 100.0,
                "message": "Video analysis complete.",
                "session_id": result["session_id"],
                "report_id": result["report_id"],
                "summary": result["summary"],
                "reps": result["reps"],
                "error": None,
                "completed_at": time.time(),
            })
    except NoPersonDetectedError as exc:
        logger.warning("Video analysis detected no person: %s", exc)
        if job_id in _jobs:
            _jobs[job_id].update({
                "status": "failed",
                "progress": 100.0,
                "message": str(exc),
                "error": str(exc),
                "completed_at": time.time(),
            })
    except (InvalidVideoFileError, VideoProcessingError, Exception) as exc:
        logger.exception("Video processing failed for job %s: %s", job_id, exc)
        if job_id in _jobs:
            _jobs[job_id].update({
                "status": "failed",
                "progress": 100.0,
                "message": f"Video analysis failed: {exc}",
                "error": str(exc),
                "completed_at": time.time(),
            })
    finally:
        # Delete the temp file immediately afterwards for privacy and disk hygiene
        if os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception as exc:
                logger.warning("Could not remove temp video %s: %s", temp_path, exc)


@router.post("/analyze", status_code=status.HTTP_200_OK)
async def analyze_video(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    activity: str = Form("squat"),
    current_user: Optional[dict] = Depends(get_current_user_optional),
) -> dict[str, Any]:
    """Upload and analyze a recorded workout video clip."""
    filename = file.filename or ""
    ext = Path(filename).suffix.lower()
    content_type = (file.content_type or "").lower()
    normalized_activity = activity.strip().lower()

    if normalized_activity != "squat":
        raise HTTPException(
            status_code=422,
            detail="Reliable uploaded-video analysis currently supports squats only. Other activity analyzers are not available yet.",
        )

    # Reject non-video formats immediately with 415
    if ext not in ALLOWED_EXTENSIONS or (content_type and content_type not in ALLOWED_MIME_TYPES and not content_type.startswith("video/")):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Unsupported media type. Only MP4, MOV, and WebM videos are supported.",
        )

    # Prepare temp file
    suffix = ext if ext in ALLOWED_EXTENSIONS else ".mp4"
    try:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        temp_path = temp_file.name
    except Exception as exc:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Could not create temporary file") from exc

    # Stream chunks and enforce size limit
    total_size = 0
    try:
        while True:
            chunk = await file.read(CHUNK_SIZE)
            if not chunk:
                break
            total_size += len(chunk)
            if total_size > MAX_UPLOAD_SIZE:
                temp_file.close()
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail="File too large. Maximum supported video size is 100 MB.",
                )
            temp_file.write(chunk)
    except HTTPException:
        raise
    except Exception as exc:
        temp_file.close()
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, f"Failed to save upload: {exc}") from exc
    finally:
        temp_file.close()

    user_id = str(current_user["_id"]) if current_user and "_id" in current_user else "anonymous"
    database = getattr(request.app.state, "database", None)
    job_id = uuid.uuid4().hex

    _jobs[job_id] = {
        "job_id": job_id,
        "status": "processing",
        "progress": 0.0,
        "message": "Upload received. Preparing video analysis.",
        "session_id": None,
        "report_id": None,
        "summary": None,
        "reps": None,
        "error": None,
        "activity": normalized_activity,
        "created_at": time.time(),
    }

    # Dispatch background task
    background_tasks.add_task(
        _run_video_job,
        job_id=job_id,
        temp_path=temp_path,
        activity=normalized_activity,
        user_id=user_id,
        database=database,
    )

    return {
        "job_id": job_id,
        "status": "processing",
        "progress": 0.0,
        "message": "Video upload accepted and analysis started.",
    }


@router.get("/status/{job_id}")
async def get_upload_status(job_id: str) -> dict[str, Any]:
    """Poll the status and progress of a video analysis job."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Upload job '{job_id}' not found.",
        )

    response: dict[str, Any] = {
        "job_id": job_id,
        "status": job["status"],
        "progress": job["progress"],
        "message": job["message"],
    }

    if job["status"] == "completed":
        response.update({
            "session_id": job.get("session_id"),
            "report_id": job.get("report_id"),
            "summary": job.get("summary"),
            "reps": job.get("reps"),
        })
    elif job["status"] == "failed":
        response.update({
            "error": job.get("error"),
        })

    return response


async def _owned_session(request: Request, session_id: str, user: dict) -> dict:
    database = request.app.state.database
    session = await database.sessions.find_one({"_id": session_id})
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    if session.get("user_id") != str(user["_id"]) and user.get("role") not in ("expert", "admin"):
        raise HTTPException(status_code=403, detail="Access denied to this session.")
    return session


@router.get("/debug/{session_id}")
async def get_upload_debug(
    session_id: str,
    request: Request,
    format: str = "json",
    user: dict = Depends(get_current_user),
):
    await _owned_session(request, session_id, user)
    cursor = request.app.state.database.pose_frames.find(
        {"session_id": session_id},
        {"_id": 0, "timestamp": 1, "knee_angle": 1, "torso_angle": 1,
         "view": 1, "visibility": 1, "usable": 1, "tracking_error": 1,
         "rep_phase": 1},
    ).sort("timestamp", 1)
    frames = await cursor.to_list(length=None)
    if format.lower() == "csv":
        output = io.StringIO()
        fields = ("timestamp", "knee_angle", "torso_angle", "view", "visibility",
                  "usable", "tracking_error", "rep_phase")
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(frames)
        return Response(
            output.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="vyronex-{session_id}-debug.csv"'},
        )
    if format.lower() != "json":
        raise HTTPException(status_code=400, detail="format must be json or csv.")
    return {"session_id": session_id, "frames": frames}


@router.post("/recompute/{session_id}")
async def recompute_upload_report(
    session_id: str,
    request: Request,
    user: dict = Depends(get_current_user),
) -> dict:
    await _owned_session(request, session_id, user)
    try:
        return await recompute_report(
            session_id,
            request.app.state.database,
            str(user["_id"]),
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except VideoProcessingError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
