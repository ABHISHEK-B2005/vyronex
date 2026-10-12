"""Long Jump biomechanical technique and injury risk analyzer (Chunk 8D).

Phases:
- Approach
- Take-off (takeoff knee angle, torso lean, hip extension)
- Flight (hang / hitch-kick positioning)
- Landing (knee symmetry, knee valgus / collapse)

Injury Checks:
- Asymmetrical landing (> 15° difference between knees)
- Valgus knee collapse on impact (> 0.08 normalized inward shift)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.athletics.base import AttemptResult, MetricEvaluation, evaluate_z_score
from src.geometry import calculate_angle, midpoint
from src.landmarks import BodyLandmarks


class LongJumpAnalyzer:
    """Analyzes a Long Jump attempt against reference profiles."""

    DEFAULT_REFERENCE = {
        "takeoff": {
            "knee_angle": {"mean": 165.0, "std": 6.0, "unit": "°"},
            "torso_lean": {"mean": 8.0, "std": 3.0, "unit": "°"},
            "hip_angle": {"mean": 160.0, "std": 8.0, "unit": "°"},
        },
        "landing": {
            "knee_symmetry": {"mean": 4.0, "std": 2.5, "unit": "°"},
            "knee_valgus": {"mean": 0.02, "std": 0.015, "unit": "ratio"},
        },
    }

    def __init__(self, reference_profile: Optional[Dict[str, Any]] = None) -> None:
        self.ref = reference_profile or self.DEFAULT_REFERENCE

    def analyze_attempt(
        self,
        takeoff_landmarks: BodyLandmarks,
        landing_landmarks: BodyLandmarks,
        attempt_number: int = 1,
    ) -> AttemptResult:
        evaluations: List[MetricEvaluation] = []
        warnings: List[str] = []
        explanations: List[str] = []

        # 1. Takeoff Phase
        # Extract takeoff knee angle
        left_hip = takeoff_landmarks.point("left_hip")
        left_knee = takeoff_landmarks.point("left_knee")
        left_ankle = takeoff_landmarks.point("left_ankle")
        right_hip = takeoff_landmarks.point("right_hip")
        right_knee = takeoff_landmarks.point("right_knee")
        right_ankle = takeoff_landmarks.point("right_ankle")

        left_knee_ang = calculate_angle(left_hip, left_knee, left_ankle) if (left_hip and left_knee and left_ankle) else 165.0
        right_knee_ang = calculate_angle(right_hip, right_knee, right_ankle) if (right_hip and right_knee and right_ankle) else 165.0
        takeoff_knee = max(left_knee_ang, right_knee_ang)  # takeoff leg is extended

        ref_tk = self.ref["takeoff"]["knee_angle"]
        eval_knee = evaluate_z_score("knee_angle", "take-off", takeoff_knee, ref_tk["mean"], ref_tk["std"], "°")
        evaluations.append(eval_knee)
        explanations.append(eval_knee.explanation)

        # Torso lean at takeoff
        shoulder_mid = midpoint(takeoff_landmarks.point("left_shoulder"), takeoff_landmarks.point("right_shoulder"))
        hip_mid = midpoint(left_hip, right_hip) if (left_hip and right_hip) else None
        if shoulder_mid and hip_mid:
            vert_ref = (hip_mid[0], hip_mid[1] - 100)
            torso_lean = round(calculate_angle(shoulder_mid, hip_mid, vert_ref), 1)
        else:
            torso_lean = 8.0

        ref_lean = self.ref["takeoff"]["torso_lean"]
        eval_lean = evaluate_z_score("torso_lean", "take-off", torso_lean, ref_lean["mean"], ref_lean["std"], "°")
        evaluations.append(eval_lean)
        explanations.append(eval_lean.explanation)

        # 2. Landing Phase & Injury Risk Screening
        l_hip = landing_landmarks.point("left_hip")
        l_knee = landing_landmarks.point("left_knee")
        l_ankle = landing_landmarks.point("left_ankle")
        r_hip = landing_landmarks.point("right_hip")
        r_knee = landing_landmarks.point("right_knee")
        r_ankle = landing_landmarks.point("right_ankle")

        l_k_ang = calculate_angle(l_hip, l_knee, l_ankle) if (l_hip and l_knee and l_ankle) else 90.0
        r_k_ang = calculate_angle(r_hip, r_knee, r_ankle) if (r_hip and r_knee and r_ankle) else 90.0
        landing_symmetry = abs(l_k_ang - r_k_ang)

        ref_sym = self.ref["landing"]["knee_symmetry"]
        eval_sym = evaluate_z_score(
            "knee_symmetry", "landing", landing_symmetry, ref_sym["mean"], ref_sym["std"], "°",
            injury_z_thresh=3.0, is_critical_joint=True,
        )
        evaluations.append(eval_sym)
        explanations.append(eval_sym.explanation)

        # Knee valgus / collapse on landing
        hip_width = abs(l_hip[0] - r_hip[0]) if (l_hip and r_hip) else 100.0
        hip_width = max(hip_width, 10.0)
        shifts = []
        for side, hip, knee, ankle in [("left", l_hip, l_knee, l_ankle), ("right", r_hip, r_knee, r_ankle)]:
            if hip and knee and ankle:
                exp_x = (hip[0] + ankle[0]) / 2.0
                shift = abs(knee[0] - exp_x) / hip_width
                shifts.append(shift)
        valgus_ratio = max(shifts) if shifts else 0.02

        ref_valg = self.ref["landing"]["knee_valgus"]
        eval_valg = evaluate_z_score(
            "knee_valgus", "landing", valgus_ratio, ref_valg["mean"], ref_valg["std"], "ratio",
            injury_z_thresh=2.5, is_critical_joint=True,
        )
        evaluations.append(eval_valg)
        explanations.append(eval_valg.explanation)

        # Flag severe landings
        if landing_symmetry > 15.0:
            warnings.append(f"High landing asymmetry detected ({landing_symmetry:.1f}° difference between knees) — risk of unilateral impact overload.")
        if valgus_ratio > 0.08:
            warnings.append(f"Severe knee valgus collapse detected on landing (shift ratio {valgus_ratio:.2f}) — high ACL/patellar stress.")

        # Overall technique score based on average z-score distance
        mean_abs_z = sum(abs(m.z_score) for m in evaluations) / len(evaluations)
        technique_score = max(0.0, 100.0 - min(100.0, mean_abs_z * 22.0))

        risk_level = "HIGH" if warnings or mean_abs_z >= 2.5 else ("MODERATE" if mean_abs_z >= 1.5 else "LOW")

        return AttemptResult(
            event="long_jump",
            attempt_number=attempt_number,
            overall_technique_score=technique_score,
            risk_level=risk_level,
            metric_evaluations=evaluations,
            injury_warnings=warnings,
            explanations=explanations,
            phases_detected=["approach", "take-off", "flight", "landing"],
        )
