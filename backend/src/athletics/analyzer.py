"""Unified Athletics Analysis Engine (Chunk 8D).

Loads event reference profiles from JSON and runs technique scoring and injury-risk assessment.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional

from src.athletics.base import AttemptResult
from src.athletics.javelin import JavelinAnalyzer
from src.athletics.long_jump import LongJumpAnalyzer
from src.athletics.shot_put import ShotPutAnalyzer
from src.landmarks import BodyLandmarks

logger = logging.getLogger(__name__)

_BACKEND_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_PROFILES_PATH = os.path.join(_BACKEND_ROOT, "data", "reference_profiles.json")


class AthleticsAnalyzer:
    """Unified coordinator for Athletics event analysis and reference profile comparison."""

    def __init__(self, profiles_path: str = DEFAULT_PROFILES_PATH) -> None:
        self.profiles_path = profiles_path
        self.profiles: Dict[str, Any] = {}
        self.load_profiles()

        self.long_jump = LongJumpAnalyzer(self.profiles.get("long_jump"))
        self.shot_put = ShotPutAnalyzer(self.profiles.get("shot_put"))
        self.javelin = JavelinAnalyzer(self.profiles.get("javelin_throw") or self.profiles.get("javelin"))

    def load_profiles(self) -> None:
        if os.path.exists(self.profiles_path):
            try:
                with open(self.profiles_path, "r", encoding="utf-8") as fh:
                    self.profiles = json.load(fh)
            except Exception as exc:
                logger.warning("Could not load reference profiles from %s: %s", self.profiles_path, exc)
                self.profiles = {}

    def analyze(
        self,
        event: str,
        phase1_landmarks: BodyLandmarks,
        phase2_landmarks: BodyLandmarks,
        attempt_number: int = 1,
    ) -> AttemptResult:
        norm_event = event.lower().replace("-", "_")

        if "long_jump" in norm_event or norm_event == "long_jump":
            return self.long_jump.analyze_attempt(phase1_landmarks, phase2_landmarks, attempt_number=attempt_number)
        elif "shot_put" in norm_event or norm_event == "shot_put":
            return self.shot_put.analyze_attempt(phase1_landmarks, phase2_landmarks, attempt_number=attempt_number)
        elif "javelin" in norm_event:
            return self.javelin.analyze_attempt(phase1_landmarks, phase2_landmarks, attempt_number=attempt_number)
        else:
            # Default to long jump handler for generic field events
            return self.long_jump.analyze_attempt(phase1_landmarks, phase2_landmarks, attempt_number=attempt_number)
