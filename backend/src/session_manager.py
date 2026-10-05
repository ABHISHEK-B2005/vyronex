"""Session tracking & summary statistics.

Keeps an in-memory record of everything that happened during a live
session (or demo run) and can persist a summary to disk as JSON.
"""

from __future__ import annotations

import json
import os
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.config import SESSIONS_DIR
from src.feedback_engine import Feedback
from src.form_analysis import FormScores
from src.risk_engine import RiskResult


@dataclass
class RepRecord:
    rep_number: int
    form_score: float
    risk_score: float
    risk_level: str
    primary_issue: Optional[str]
    timestamp: float = field(default_factory=time.time)


class SessionManager:
    """Tracks statistics for a single exercise session."""

    def __init__(self, exercise_name: str = "SQUAT") -> None:
        self.exercise_name = exercise_name
        self.start_time = time.time()
        self.end_time: Optional[float] = None
        self.reps: List[RepRecord] = []

    def record_rep(
        self,
        rep_number: int,
        form_scores: FormScores,
        risk_result: RiskResult,
        feedback: Feedback,
    ) -> None:
        self.reps.append(
            RepRecord(
                rep_number=rep_number,
                form_score=form_scores.overall,
                risk_score=risk_result.risk_score,
                risk_level=risk_result.risk_level,
                primary_issue=risk_result.primary_issue,
            )
        )

    def end_session(self) -> None:
        self.end_time = time.time()

    @property
    def duration_seconds(self) -> float:
        end = self.end_time or time.time()
        return round(end - self.start_time, 1)

    @property
    def total_reps(self) -> int:
        return len(self.reps)

    @property
    def average_form_score(self) -> float:
        if not self.reps:
            return 0.0
        return round(sum(r.form_score for r in self.reps) / len(self.reps), 1)

    @property
    def average_risk_score(self) -> float:
        if not self.reps:
            return 0.0
        return round(sum(r.risk_score for r in self.reps) / len(self.reps), 1)

    def count_by_risk_level(self) -> Dict[str, int]:
        counts = {"LOW": 0, "MODERATE": 0, "HIGH": 0}
        for rep in self.reps:
            counts[rep.risk_level] = counts.get(rep.risk_level, 0) + 1
        return counts

    def most_common_issue(self) -> Optional[str]:
        issues = [r.primary_issue for r in self.reps if r.primary_issue]
        if not issues:
            return None
        return Counter(issues).most_common(1)[0][0]

    def summary(self) -> Dict:
        counts = self.count_by_risk_level()
        return {
            "exercise": self.exercise_name,
            "start_time": self.start_time,
            "duration_seconds": self.duration_seconds,
            "total_reps": self.total_reps,
            "average_form_score": self.average_form_score,
            "average_risk_score": self.average_risk_score,
            "low_risk_reps": counts["LOW"],
            "moderate_risk_reps": counts["MODERATE"],
            "high_risk_reps": counts["HIGH"],
            "most_common_issue": self.most_common_issue(),
        }

    def save(self, directory: str = SESSIONS_DIR) -> str:
        """Persist the session summary (numbers only, no video/images) to
        a timestamped JSON file. Returns the path written.
        """
        os.makedirs(directory, exist_ok=True)
        filename = f"session_{int(self.start_time)}.json"
        path = os.path.join(directory, filename)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.summary(), fh, indent=2)
        return path
