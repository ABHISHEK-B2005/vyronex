"""Squat repetition detector implemented as a state machine.

STANDING -> DESCENDING -> BOTTOM -> ASCENDING -> STANDING (rep counted)

The detector is intentionally simple and rule-based (angle-threshold +
frame-debounce) so it is transparent and easy to explain during a demo,
while being architected so other exercises can be added later via the
same ``ExerciseDetector`` interface (see ``exercise_detector_base``).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from src.config import (
    ASCEND_TRIGGER_DELTA,
    BOTTOM_KNEE_ANGLE,
    DESCEND_TRIGGER_DELTA,
    MIN_FRAMES_IN_PHASE,
    STANDING_KNEE_ANGLE,
)


class SquatPhase(str, Enum):
    STANDING = "Standing"
    DESCENDING = "Descending"
    BOTTOM = "Bottom"
    ASCENDING = "Ascending"
    UNKNOWN = "Unknown"


@dataclass
class RepFrameSample:
    """One frame's worth of raw measurements collected during a rep,
    used afterward to compute biomechanical features for that rep.
    """
    timestamp: float
    avg_knee_angle: float
    left_knee_angle: float
    right_knee_angle: float
    left_hip_angle: float
    right_hip_angle: float
    torso_angle: float
    knee_alignment_shift: float  # normalized inward/outward knee shift
    hip_midpoint: tuple


@dataclass
class CompletedRep:
    rep_number: int
    samples: List[RepFrameSample] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0

    @property
    def duration_seconds(self) -> float:
        return max(0.0, self.end_time - self.start_time)


class SquatDetector:
    """Rule-based squat phase & repetition detector.

    Call ``update(avg_knee_angle, ...)`` once per frame with the current
    measurements. The detector tracks phase transitions internally and
    returns a ``CompletedRep`` whenever a full rep finishes (i.e. the
    person returns to STANDING after passing through BOTTOM).
    """

    def __init__(self) -> None:
        self.phase: SquatPhase = SquatPhase.STANDING
        self.rep_count: int = 0
        self._frames_in_candidate_phase: int = 0
        self._candidate_phase: Optional[SquatPhase] = None
        self._current_rep_samples: List[RepFrameSample] = []
        self._current_rep_start_time: Optional[float] = None
        self._reached_bottom_this_rep: bool = False
        self._peak_standing_angle: float = STANDING_KNEE_ANGLE

    def reset(self) -> None:
        self.__init__()

    def _set_phase(self, new_phase: SquatPhase) -> None:
        """Debounced phase transition: requires MIN_FRAMES_IN_PHASE
        consecutive frames agreeing on the new phase before switching,
        to avoid flicker from noisy landmark detection.
        """
        if new_phase == self.phase:
            self._candidate_phase = None
            self._frames_in_candidate_phase = 0
            return

        if new_phase == self._candidate_phase:
            self._frames_in_candidate_phase += 1
        else:
            self._candidate_phase = new_phase
            self._frames_in_candidate_phase = 1

        if self._frames_in_candidate_phase >= MIN_FRAMES_IN_PHASE:
            self.phase = new_phase
            self._candidate_phase = None
            self._frames_in_candidate_phase = 0

    def update(self, sample: RepFrameSample) -> Optional[CompletedRep]:
        """Feed one frame of measurements. Returns a CompletedRep if this
        frame completed a repetition, otherwise None.
        """
        angle = sample.avg_knee_angle
        completed_rep: Optional[CompletedRep] = None

        # Determine the "raw" phase suggested purely by the current angle.
        if angle >= STANDING_KNEE_ANGLE:
            raw_phase = SquatPhase.STANDING
        elif angle <= BOTTOM_KNEE_ANGLE:
            raw_phase = SquatPhase.BOTTOM
        elif self.phase in (SquatPhase.STANDING, SquatPhase.DESCENDING):
            raw_phase = SquatPhase.DESCENDING
        else:
            raw_phase = SquatPhase.ASCENDING

        previous_phase = self.phase
        self._set_phase(raw_phase)

        # Track rep lifecycle
        if self.phase == SquatPhase.DESCENDING and previous_phase == SquatPhase.STANDING:
            self._current_rep_samples = []
            self._current_rep_start_time = sample.timestamp
            self._reached_bottom_this_rep = False

        if self.phase in (SquatPhase.DESCENDING, SquatPhase.BOTTOM, SquatPhase.ASCENDING):
            self._current_rep_samples.append(sample)

        if self.phase == SquatPhase.BOTTOM:
            self._reached_bottom_this_rep = True

        if (
            self.phase == SquatPhase.STANDING
            and previous_phase == SquatPhase.ASCENDING
            and self._reached_bottom_this_rep
            and self._current_rep_start_time is not None
        ):
            self.rep_count += 1
            completed_rep = CompletedRep(
                rep_number=self.rep_count,
                samples=list(self._current_rep_samples),
                start_time=self._current_rep_start_time,
                end_time=sample.timestamp,
            )
            self._current_rep_samples = []
            self._current_rep_start_time = None
            self._reached_bottom_this_rep = False

        return completed_rep
