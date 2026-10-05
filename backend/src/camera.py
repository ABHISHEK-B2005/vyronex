"""Webcam capture module built on OpenCV.

Handles camera-not-found errors gracefully, supports selecting a camera
index, and never lets a single bad frame crash the application.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

try:
    import cv2

    OPENCV_AVAILABLE = True
except Exception:  # pragma: no cover
    OPENCV_AVAILABLE = False
    cv2 = None  # type: ignore

from src.config import DEFAULT_CAMERA_INDEX, TARGET_FRAME_HEIGHT, TARGET_FRAME_WIDTH


class CameraUnavailable(RuntimeError):
    """Raised when the requested webcam cannot be opened."""


class Camera:
    """Wraps ``cv2.VideoCapture`` with safe start/stop/read semantics."""

    def __init__(self, camera_index: int = DEFAULT_CAMERA_INDEX):
        if not OPENCV_AVAILABLE:
            raise CameraUnavailable(
                "OpenCV is not installed. Install it with `pip install opencv-python`."
            )
        self.camera_index = camera_index
        self._cap: Optional["cv2.VideoCapture"] = None
        self._consecutive_failures = 0
        self._max_consecutive_failures = 30

    def start(self) -> None:
        """Open the webcam. Raises CameraUnavailable on failure."""
        self._cap = cv2.VideoCapture(self.camera_index)
        if not self._cap or not self._cap.isOpened():
            self._cap = None
            raise CameraUnavailable(
                f"Could not open camera index {self.camera_index}. "
                "Check that a webcam is connected, not in use by another "
                "app, and that camera permissions are granted."
            )
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_FRAME_WIDTH)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_FRAME_HEIGHT)
        self._consecutive_failures = 0

    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a single frame. Returns (success, frame_bgr_or_None).

        Tolerates transient frame-read failures (returns False rather than
        raising) but flags persistent failure via CameraUnavailable so the
        caller can surface a user-friendly message.
        """
        if not self.is_open():
            return False, None

        ok, frame = self._cap.read()
        if not ok or frame is None:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self._max_consecutive_failures:
                raise CameraUnavailable(
                    "Camera stopped returning frames. It may have been "
                    "disconnected or is being used by another application."
                )
            return False, None

        self._consecutive_failures = 0
        return True, frame

    def stop(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    @staticmethod
    def list_available_indices(max_index: int = 4) -> list:
        """Best-effort probe of available camera indices (0..max_index-1).
        Used to populate a camera-selection dropdown. This can be slow, so
        callers should cache the result.
        """
        if not OPENCV_AVAILABLE:
            return []
        available = []
        for idx in range(max_index):
            cap = cv2.VideoCapture(idx)
            if cap is not None and cap.isOpened():
                available.append(idx)
            if cap is not None:
                cap.release()
        return available or [0]

    def __enter__(self) -> "Camera":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()
