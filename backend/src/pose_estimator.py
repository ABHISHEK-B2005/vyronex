"""MediaPipe pose estimation with support for legacy and Tasks APIs."""

from __future__ import annotations

import hashlib
import os
import tempfile
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Tuple

import cv2
import numpy as np

from src.config import POSE_MIN_DETECTION_CONFIDENCE, POSE_MIN_TRACKING_CONFIDENCE
from src.landmarks import BodyLandmarks

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
MODEL_MD5 = "04a75ddf7c811ac7a1a4523266dd7d88"

try:
    import mediapipe as mp
except Exception as exc:  # pragma: no cover - environment dependent
    mp = None  # type: ignore[assignment]
    _mediapipe_import_error = exc
else:
    _mediapipe_import_error = None

_solutions_pose = getattr(getattr(mp, "solutions", None), "pose", None) if mp else None
try:
    from mediapipe.tasks.python import vision as _tasks_vision
    from mediapipe.tasks.python.core.base_options import BaseOptions as _BaseOptions
except (ImportError, AttributeError):  # pragma: no cover - depends on installed MediaPipe version
    _tasks_vision = None
    _BaseOptions = None


class PoseEstimatorUnavailable(RuntimeError):
    """Raised when MediaPipe pose estimation cannot be initialized."""


def _get_model_path() -> Path:
    cache_dir = Path(tempfile.gettempdir()) / "vyronex-mediapipe"
    model_path = cache_dir / "pose_landmarker_lite.task"
    if model_path.is_file() and hashlib.md5(model_path.read_bytes()).hexdigest() == MODEL_MD5:
        return model_path

    cache_dir.mkdir(parents=True, exist_ok=True)
    temp_path: Optional[Path] = None
    try:
        with urllib.request.urlopen(MODEL_URL, timeout=30) as response:
            content = response.read()
        if hashlib.md5(content).hexdigest() != MODEL_MD5:
            raise PoseEstimatorUnavailable("Downloaded MediaPipe pose model failed its integrity check.")
        with tempfile.NamedTemporaryFile(dir=cache_dir, suffix=".task", delete=False) as temp_file:
            temp_file.write(content)
            temp_path = Path(temp_file.name)
        os.replace(temp_path, model_path)
        return model_path
    except PoseEstimatorUnavailable:
        raise
    except Exception as exc:
        raise PoseEstimatorUnavailable(
            f"Could not download the MediaPipe pose model: {exc}. Check the backend internet connection and retry."
        ) from exc
    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink()


class PoseEstimator:
    """Reuse one MediaPipe pose model while processing a video."""

    def __init__(
        self,
        min_detection_confidence: float = POSE_MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence: float = POSE_MIN_TRACKING_CONFIDENCE,
        model_complexity: int = 1,
    ) -> None:
        self._pose = None
        self._landmarker = None
        if _solutions_pose is not None:
            try:
                self._pose = _solutions_pose.Pose(
                    static_image_mode=False,
                    model_complexity=model_complexity,
                    smooth_landmarks=True,
                    min_detection_confidence=min_detection_confidence,
                    min_tracking_confidence=min_tracking_confidence,
                )
            except Exception as exc:
                raise PoseEstimatorUnavailable(f"Could not initialize MediaPipe Pose: {exc}") from exc
            return

        if mp is None:
            raise PoseEstimatorUnavailable(
                f"MediaPipe could not be imported: {_mediapipe_import_error}. Install backend requirements."
            )
        if _tasks_vision is None or _BaseOptions is None:
            raise PoseEstimatorUnavailable(
                "This MediaPipe build provides neither the legacy Pose API nor the Tasks Pose Landmarker API."
            )

        try:
            options = _tasks_vision.PoseLandmarkerOptions(
                base_options=_BaseOptions(model_asset_path=str(_get_model_path())),
                running_mode=_tasks_vision.RunningMode.IMAGE,
                num_poses=1,
                min_pose_detection_confidence=min_detection_confidence,
                min_pose_presence_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            self._landmarker = _tasks_vision.PoseLandmarker.create_from_options(options)
        except PoseEstimatorUnavailable:
            raise
        except Exception as exc:
            raise PoseEstimatorUnavailable(f"Could not initialize MediaPipe Pose Landmarker: {exc}") from exc

    def process(self, frame_bgr: np.ndarray) -> Tuple[Optional[BodyLandmarks], object, bool]:
        """Estimate a person pose from a BGR video frame."""
        if frame_bgr is None or frame_bgr.size == 0:
            return None, None, False

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if self._pose is not None:
            frame_rgb.flags.writeable = False
            results = self._pose.process(frame_rgb)
            pose_landmarks = getattr(results, "pose_landmarks", None)
            if not pose_landmarks:
                return None, results, False
            height, width = frame_bgr.shape[:2]
            world_landmarks = getattr(results, "pose_world_landmarks", None)
            return BodyLandmarks.from_mediapipe(
                pose_landmarks, width, height, world_landmarks
            ), results, True

        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        results = self._landmarker.detect(image)
        poses = getattr(results, "pose_landmarks", None)
        if not poses:
            return None, results, False

        height, width = frame_bgr.shape[:2]
        compatible_landmarks = SimpleNamespace(landmark=poses[0])
        world_poses = getattr(results, "pose_world_landmarks", None)
        world_landmarks = world_poses[0] if world_poses else None
        return BodyLandmarks.from_mediapipe(
            compatible_landmarks, width, height, world_landmarks
        ), results, True

    def draw_skeleton(self, frame_bgr: np.ndarray, results, color_bgr: Tuple[int, int, int]) -> np.ndarray:
        """Draw legacy MediaPipe skeletons when drawing utilities are available."""
        if self._pose is None or not getattr(results, "pose_landmarks", None) or not mp.solutions.drawing_utils:
            return frame_bgr
        annotated = frame_bgr.copy()
        spec = mp.solutions.drawing_utils.DrawingSpec(color=color_bgr, thickness=3, circle_radius=4)
        mp.solutions.drawing_utils.draw_landmarks(
            annotated,
            results.pose_landmarks,
            self._pose.POSE_CONNECTIONS,
            landmark_drawing_spec=spec,
            connection_drawing_spec=spec,
        )
        return annotated

    def close(self) -> None:
        pose = self._pose or self._landmarker
        if pose is not None:
            pose.close()
        self._pose = None
        self._landmarker = None
