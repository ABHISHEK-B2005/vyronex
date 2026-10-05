"""Personal baseline system.

Lets a user record a handful of "controlled" squats to establish their
own normal ranges, then compares future reps against BOTH:
  1. the general prototype rules (risk_engine / form_analysis), and
  2. their personal baseline.

Explicitly labeled as "Prototype movement indicators" — not a medical
metric — per the project's terminology requirements.
"""

from __future__ import annotations

import json
import os
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.feature_engineering import RepFeatures

BASELINE_PATH = "data/personal_baseline.json"


@dataclass
class PersonalBaseline:
    sample_count: int
    knee_angle_min: float
    knee_angle_max: float
    torso_angle_mean: float
    movement_speed_mean: float
    knee_symmetry_mean: float

    def to_dict(self) -> Dict:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, data: Dict) -> "PersonalBaseline":
        return cls(**data)


class BaselineBuilder:
    """Accumulates RepFeatures across several "controlled" reps performed
    during a "Create Personal Baseline" flow, then produces a
    PersonalBaseline summary.
    """

    def __init__(self) -> None:
        self._reps: List[RepFeatures] = []

    def add_rep(self, features: RepFeatures) -> None:
        self._reps.append(features)

    @property
    def collected_count(self) -> int:
        return len(self._reps)

    def build(self) -> Optional[PersonalBaseline]:
        if not self._reps:
            return None
        knee_mins = [r.min_knee_angle for r in self._reps]
        knee_maxs = [r.max_knee_angle for r in self._reps]
        torso = [r.torso_angle for r in self._reps]
        speed = [r.movement_speed for r in self._reps]
        knee_sym = [r.knee_symmetry for r in self._reps]

        return PersonalBaseline(
            sample_count=len(self._reps),
            knee_angle_min=round(statistics.mean(knee_mins), 1),
            knee_angle_max=round(statistics.mean(knee_maxs), 1),
            torso_angle_mean=round(statistics.mean(torso), 1),
            movement_speed_mean=round(statistics.mean(speed), 1),
            knee_symmetry_mean=round(statistics.mean(knee_sym), 1),
        )


def save_baseline(baseline: PersonalBaseline, path: str = BASELINE_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(baseline.to_dict(), fh, indent=2)


def load_baseline(path: str = BASELINE_PATH) -> Optional[PersonalBaseline]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return PersonalBaseline.from_dict(data)


@dataclass
class PersonalComparison:
    knee_stability: float   # 0-100: how close current rep's depth is to baseline
    symmetry: float         # 0-100
    movement_control: float # 0-100


def compare_to_baseline(features: RepFeatures, baseline: PersonalBaseline) -> PersonalComparison:
    """Compare a single rep's features to the stored personal baseline.
    Produces simple 0-100 "prototype movement indicators" (NOT medical
    metrics).
    """
    knee_deviation = abs(features.min_knee_angle - baseline.knee_angle_min)
    knee_stability = max(0.0, 100.0 - knee_deviation * 2.0)

    symmetry_deviation = abs(features.knee_symmetry - baseline.knee_symmetry_mean)
    symmetry_score = max(0.0, 100.0 - symmetry_deviation * 5.0)

    speed_deviation = abs(features.movement_speed - baseline.movement_speed_mean)
    control_score = max(0.0, 100.0 - speed_deviation * 2.0)

    return PersonalComparison(
        knee_stability=round(min(100.0, knee_stability), 1),
        symmetry=round(min(100.0, symmetry_score), 1),
        movement_control=round(min(100.0, control_score), 1),
    )
