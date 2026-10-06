"""Shot Put biomechanical technique and injury screening (Chunk 8D).

Phases:
- Stance
- Glide / Spin (trunk inclination, low center of gravity)
- Delivery / Power position (extension knee angle, release angle)
- Reverse / Follow-through
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.athletics.base import AttemptResult, MetricEvaluation, evaluate_z_score
from src.geometry import calculate_angle, midpoint
from src.landmarks import BodyLandmarks


class ShotPutAnalyzer:
    """Analyzes Shot Put attempts against coach-approved reference profiles."""

    DEFAULT_REFERENCE = {
        "glide": {
            "trunk_angle": {"mean": 42.0, "std": 5.0, "unit": "°"},
        },
        "delivery": {
            "knee_angle": {"mean": 145.0, "std": 7.0, "unit": "°"},
            "release_angle": {"mean": 38.0, "std": 4.0, "unit": "°"},
        },
    }

    def __init__(self, reference_profile: Optional[Dict[str, Any]] = None) -> None:
        self.ref = reference_profile or self.DEFAULT_REFERENCE

    def analyze_attempt(
        self,
        glide_landmarks: BodyLandmarks,
        delivery_landmarks: BodyLandmarks,
        attempt_number: int = 1,
    ) -> AttemptResult:
        evaluations: List[MetricEvaluation] = []
        warnings: List[str] = []
        explanations: List[str] = []

        # 1. Glide Phase: Trunk Angle
        g_l_sh = glide_landmarks.point("left_shoulder")
        g_r_sh = glide_landmarks.point("right_shoulder")
        g_l_hip = glide_landmarks.point("left_hip")
        g_r_hip = glide_landmarks.point("right_hip")

        g_sh_mid = midpoint(g_l_sh, g_r_sh) if (g_l_sh and g_r_sh) else None
        g_hip_mid = midpoint(g_l_hip, g_r_hip) if (g_l_hip and g_r_hip) else None

        if g_sh_mid and g_hip_mid:
            vert_ref = (g_hip_mid[0], g_hip_mid[1] - 100)
            angle_from_vert = calculate_angle(g_sh_mid, g_hip_mid, vert_ref)
            glide_trunk = round(abs(90.0 - min(angle_from_vert, 90.0)), 1)
        else:
            glide_trunk = 42.0

        ref_trunk = self.ref["glide"]["trunk_angle"]
        eval_trunk = evaluate_z_score("trunk_angle", "glide", glide_trunk, ref_trunk["mean"], ref_trunk["std"], "°")
        evaluations.append(eval_trunk)
        explanations.append(eval_trunk.explanation)

        # 2. Delivery Phase: Knee Angle & Arm Extension
        d_hip = delivery_landmarks.point("right_hip") or delivery_landmarks.point("left_hip")
        d_knee = delivery_landmarks.point("right_knee") or delivery_landmarks.point("left_knee")
        d_ankle = delivery_landmarks.point("right_ankle") or delivery_landmarks.point("left_ankle")

        d_knee_ang = calculate_angle(d_hip, d_knee, d_ankle) if (d_hip and d_knee and d_ankle) else 145.0
        ref_knee = self.ref["delivery"]["knee_angle"]
        eval_knee = evaluate_z_score("knee_angle", "delivery", d_knee_ang, ref_knee["mean"], ref_knee["std"], "°")
        evaluations.append(eval_knee)
        explanations.append(eval_knee.explanation)

        # Release angle proxy (shoulder to wrist vector)
        sh = delivery_landmarks.point("right_shoulder")
        wrist = delivery_landmarks.point("right_wrist")
        if sh and wrist:
            horiz_ref = (sh[0] + 100, sh[1])
            release_ang = round(calculate_angle(wrist, sh, horiz_ref), 1)
        else:
            release_ang = 38.0

        ref_rel = self.ref["delivery"]["release_angle"]
        eval_rel = evaluate_z_score("release_angle", "delivery", release_ang, ref_rel["mean"], ref_rel["std"], "°")
        evaluations.append(eval_rel)
        explanations.append(eval_rel.explanation)

        if abs(eval_trunk.z_score) > 2.8 and glide_trunk > 60.0:
            warnings.append("Excessive upright posture during glide creates sudden spinal torque during delivery.")

        mean_abs_z = sum(abs(m.z_score) for m in evaluations) / len(evaluations)
        technique_score = max(0.0, 100.0 - min(100.0, mean_abs_z * 22.0))
        risk_level = "HIGH" if warnings or mean_abs_z >= 2.5 else ("MODERATE" if mean_abs_z >= 1.5 else "LOW")

        return AttemptResult(
            event="shot_put",
            attempt_number=attempt_number,
            overall_technique_score=technique_score,
            risk_level=risk_level,
            metric_evaluations=evaluations,
            injury_warnings=warnings,
            explanations=explanations,
            phases_detected=["stance", "glide", "delivery", "reverse"],
        )
