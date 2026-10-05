"""Smart Ring session tracking.

Mirrors the pattern of ``session_manager.py`` (accumulate readings ->
compute a summary), but is intentionally kept completely separate from
the squat/pose pipeline: it has no dependency on landmarks, geometry,
squat_detector, form_analysis or risk_engine. A Smart Ring reading is a
physiological sample (heart rate / HRV / motion), not a body landmark
frame, so it gets its own small, self-contained data path.

HONESTY ABOUT DATA: this module does not talk to any physical Smart Ring
and has no Bluetooth/manufacturer-SDK code. It only stores and aggregates
whatever readings the caller sends it. Today the only caller is the
frontend's Demo Mode, which generates readings tagged
``source="simulated_demo"``. A future real-device integration would POST
readings tagged ``source="smart_ring_device"`` through the exact same
endpoints — nothing here needs to change for that to work.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class RingReading:
    t: float
    heart_rate: int
    hrv: int
    accel_magnitude: float
    source: str = "simulated_demo"
    timestamp: float = field(default_factory=time.time)


class RingSessionManager:
    """Accumulates Smart Ring readings for one session and computes a
    summary on demand — analogous to SessionManager, but for sensor
    readings instead of squat reps.
    """

    def __init__(self, mode: str = "demo") -> None:
        self.mode = mode  # "demo" today; "device" reserved for future real hardware
        self.start_time = time.time()
        self.end_time: Optional[float] = None
        self.readings: List[RingReading] = []

    def add_reading(self, reading: RingReading) -> None:
        self.readings.append(reading)

    def end_session(self) -> None:
        self.end_time = time.time()

    @property
    def duration_seconds(self) -> float:
        end = self.end_time or time.time()
        return round(end - self.start_time, 1)

    def summary(self) -> Dict:
        if not self.readings:
            return {
                "source": "simulated_demo" if self.mode == "demo" else "smart_ring_device",
                "available": False,
                "reason": "No sensor samples were received for this session.",
                "sample_count": 0,
            }

        hrs = [r.heart_rate for r in self.readings]
        hrvs = [r.hrv for r in self.readings]
        accels = [r.accel_magnitude for r in self.readings]
        avg_hrv = round(sum(hrvs) / len(hrvs))
        avg_accel = round(sum(accels) / len(accels), 2)
        intensity = "HIGH" if avg_accel > 1.1 else "MODERATE" if avg_accel > 0.5 else "LOW"
        # Simple HRV-based readiness proxy — a demo heuristic for this
        # prototype, not a clinical or manufacturer recovery algorithm.
        recovery_score = round(max(0, min(100, avg_hrv * 1.6)))
        # Every stored reading's source should already agree (the same demo
        # or device session shouldn't mix), but fall back honestly if not.
        sources = {r.source for r in self.readings}
        source = sources.pop() if len(sources) == 1 else "mixed"

        return {
            "source": source,
            "available": True,
            "duration_sec": self.duration_seconds,
            "avg_hr": round(sum(hrs) / len(hrs)),
            "max_hr": max(hrs),
            "min_hr": min(hrs),
            "avg_hrv": avg_hrv,
            "avg_accel": avg_accel,
            "intensity": intensity,
            "recovery_score": recovery_score,
            "sample_count": len(self.readings),
        }
