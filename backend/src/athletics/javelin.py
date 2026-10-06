"""Javelin Throw biomechanical technique and injury screening (Chunk 8D).

Phases:
- Runup & Crossover
- Final Plant (stiff front brace leg, torso arch)
- Delivery / Release (shoulder & elbow path)

Injury Checks:
- Brace knee collapse (< 140° knee angle on plant -> severe anterior cruciate / patellar load)
- Excessive elbow drop (< 90° throwing angle -> medial collateral ligament strain)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.athletics.base import AttemptResult, MetricEvaluation, evaluate_z_score
from src.geometry import calculate_angle, midpoint
from src.landmarks import BodyLandmarks


class JavelinAnalyzer:
    """Analyzes Javelin Throw attempts against reference profiles."""

    DEFAULT_REFERENCE = {
        "plant": {
            "plant_knee_angle": {"mean": 168.0, "std": 5.0, "unit": "°"},
            "torso_lean": {"mean": 25.0, "std": 4.0, "unit": "°"},
        },
        "delivery": {
            "shoulder_angle": {"mean": 120.0, "std": 8.0, "unit": "°"},
        },
    }

    def __init__(self, reference_profile: Optional[Dict[str, Any]] = None) -> None:
        self.ref = reference_profile or self.DEFAULT_REFERENCE

    def analyze_attempt(
        self,
        plant_landmarks: BodyLandmarks,
        delivery_landmarks: BodyLandmarks,
        attempt_number: int = 1,
    ) -> AttemptResult:
        evaluations: List[MetricEvaluation] = []
        warnings: List[str] = []
        explanations: List[str] = []

        # 1. Plant Phase: Brace Knee Extension & Torso Lean
        l_hip = plant_landmarks.point("left_hip")
        l_knee = plant_landmarks.point("left_knee")
        l_ankle = plant_landmarks.point("left_ankle")
        r_hip = plant_landmarks.point("right_hip")
        r_knee = plant_landmarks.point("right_knee")
        r_ankle = plant_landmarks.point("right_ankle")

        # Brace leg (front leg in right-handed thrower is left leg)
        plant_knee_ang = calculate_angle(l_hip, l_knee, l_ankle) if (l_hip and l_knee and l_ankle) else (
            calculate_angle(r_hip, r_knee, r_ankle) if (r_hip and r_knee and r_ankle) else 168.0
        )

        ref_plant = self.ref["plant"]["plant_knee_angle"]
        eval_plant = evaluate_z_score(
            "plant_knee_angle", "plant", plant_knee_ang, ref_plant["mean"], ref_plant["std"], "°",
            injury_z_thresh=2.5, is_critical_joint=True,
        )
        evaluations.append(eval_plant)
        explanations.append(eval_plant.explanation)

        # Torso lean at plant
        sh_mid = midpoint(plant_landmarks.point("left_shoulder"), plant_landmarks.point("right_shoulder"))
        hip_mid = midpoint(l_hip, r_hip) if (l_hip and r_hip) else None
        if sh_mid and hip_mid:
            vert_ref = (hip_mid[0], hip_mid[1] - 100)
            lean = round(abs(calculate_angle(sh_mid, hip_mid, vert_ref)), 1)
        else:
            lean = 25.0

        ref_lean = self.ref["plant"]["torso_lean"]
        eval_lean = evaluate_z_score("torso_lean", "plant", lean, ref_lean["mean"], ref_lean["std"], "°")
        evaluations.append(eval_lean)
        explanations.append(eval_lean.explanation)

        # 2. Delivery: Throwing shoulder / elbow angle
        r_sh = delivery_landmarks.point("right_shoulder")
        r_el = delivery_landmarks.point("right_elbow")
        r_wr = delivery_landmarks.point("right_wrist")
        sh_angle = calculate_angle(r_sh, r_el, r_wr) if (r_sh and r_el and r_wr) else 120.0

        ref_sh = self.ref["delivery"]["shoulder_angle"]
        eval_sh = evaluate_z_score(
            "shoulder_angle", "delivery", sh_angle, ref_sh["mean"], ref_sh["std"], "°",
            injury_z_thresh=2.5, is_critical_joint=True,
        )
        evaluations.append(eval_sh)
        explanations.append(eval_sh.explanation)

        # Injury Risk Flags
        if plant_knee_ang < 140.0:
            warnings.append(f"Brace knee collapsed ({plant_knee_ang:.1f}°) during plant phase — critical risk of knee joint / patellar tendon overload.")
        if sh_angle < 90.0:
            warnings.append(f"Low throwing elbow angle ({sh_angle:.1f}°) — increases medial elbow (UCL) strain.")

        mean_abs_z = sum(abs(m.z_score) for m in evaluations) / len(evaluations)
        technique_score = max(0.0, 100.0 - min(100.0, mean_abs_z * 22.0))
        risk_level = "HIGH" if warnings or mean_abs_z >= 2.5 else ("MODERATE" if mean_abs_z >= 1.5 else "LOW")

        return AttemptResult(
            event="javelin_throw",
            attempt_number=attempt_number,
            overall_technique_score=technique_score,
            risk_level=risk_level,
            metric_evaluations=evaluations,
            injury_warnings=warnings,
            explanations=explanations,
            phases_detected=["runup", "crossover", "plant", "delivery"],
        )
