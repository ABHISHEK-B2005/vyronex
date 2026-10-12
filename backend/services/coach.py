"""AI Biomechanical Coach and Recommendation Engine (Chunk 9).

Provides:
- Corrective exercise library mapped by body area
- Rule-based post-workout advice generation
- Biomechanical chat assistant with medical disclaimers and safety boundaries
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

MEDICAL_DISCLAIMER = "Movement-risk screening only, not a medical diagnosis."

CORRECTIVE_LIBRARY: Dict[str, List[Dict[str, str]]] = {
    "lower_back": [
        {"name": "Bird-Dog", "sets_reps": "3 sets x 8 reps/side", "benefit": "Core anti-rotation and lumbar stabilization."},
        {"name": "Dead Bug", "sets_reps": "3 sets x 10 reps/side", "benefit": "Anterior core control without spinal shear."},
        {"name": "Hip-Hinge Drills with Dowel", "sets_reps": "2 sets x 12 reps", "benefit": "Retrains neutral spine during flexion."},
        {"name": "Kneeling Hamstring Stretch", "sets_reps": "2 sets x 30s/side", "benefit": "Reduces posterior pelvic pull on lumbar spine."},
    ],
    "knee": [
        {"name": "Glute Bridge with Squeeze", "sets_reps": "3 sets x 12 reps", "benefit": "Activates gluteus medius/maximus to prevent knee valgus."},
        {"name": "Banded Lateral Squats", "sets_reps": "3 sets x 10 reps/side", "benefit": "Enforces external hip rotation and knee alignment."},
        {"name": "Eccentric Step-Downs", "sets_reps": "2 sets x 8 reps/side", "benefit": "Builds quad/patellar tendon deceleration strength."},
        {"name": "Half-Kneeling Quad & Hip Stretch", "sets_reps": "2 sets x 30s/side", "benefit": "Alleviates anterior knee joint tension."},
    ],
    "shoulder": [
        {"name": "Band Pull-Aparts", "sets_reps": "3 sets x 15 reps", "benefit": "Strengthens rhomboids and rear deltoids for thoracic stability."},
        {"name": "Scapular Push-Ups", "sets_reps": "3 sets x 10 reps", "benefit": "Recruits serratus anterior to stabilize the shoulder blade."},
        {"name": "Thoracic Spine Foam Roller Extensions", "sets_reps": "2 sets x 10 passes", "benefit": "Improves overhead mobility and spinal extension."},
    ],
    "ankle": [
        {"name": "Standing Calf & Soleus Raises", "sets_reps": "3 sets x 15 reps", "benefit": "Achilles tendon resilience and plantarflexor power."},
        {"name": "Wall Ankle Mobilization Drill", "sets_reps": "2 sets x 10 reps/side", "benefit": "Increases dorsiflexion range to allow proper squat depth."},
        {"name": "Single-Leg Balance on Foam Pad", "sets_reps": "3 sets x 30s/side", "benefit": "Proprioceptive stabilization of the subtalar joint."},
    ],
    "hip": [
        {"name": "Half-Kneeling Hip Flexor Stretch", "sets_reps": "2 sets x 30s/side", "benefit": "Lengthens psoas to prevent anterior pelvic tilt."},
        {"name": "Side-Lying Banded Clamshells", "sets_reps": "3 sets x 15 reps/side", "benefit": "Isolates gluteus medius for pelvic levelness."},
        {"name": "Pigeon Pose / 90-90 Stretch", "sets_reps": "2 sets x 40s/side", "benefit": "Hip capsule decompression and rotatory mobility."},
    ],
    "elbow": [
        {"name": "Wrist Extensor & Flexor Stretches", "sets_reps": "2 sets x 30s", "benefit": "Relieves medial/lateral epicondyle tension."},
        {"name": "Forearm Pronation / Supination with Light Weight", "sets_reps": "2 sets x 12 reps", "benefit": "Strengthens elbow stabilizers for throwing mechanics."},
    ],
}


def generate_post_workout_advice(
    report_data: Dict[str, Any],
    user_profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Generate structured rule-based post-workout recommendations."""
    totals = report_data.get("totals", {})
    high_risk_reps = totals.get("high_risk", 0)
    defective_reps = totals.get("defective", 0)
    total_reps = totals.get("reps", 0)
    
    fatigue = report_data.get("fatigue", {})
    fatigue_drop = fatigue.get("drop_pct", 0)
    
    body_areas = report_data.get("body_area_breakdown", {})
    worst_area = max(body_areas, key=body_areas.get) if body_areas else "knee"

    recommendations: List[str] = []
    corrective_drills: List[Dict[str, str]] = []

    # 1. Corrective Exercises for Primary Area
    if worst_area in CORRECTIVE_LIBRARY:
        drills = CORRECTIVE_LIBRARY[worst_area]
        corrective_drills = drills[:3]
        drill_names = ", ".join(d["name"] for d in drills[:2])
        recommendations.append(
            f"Target {worst_area.replace('_', ' ')} stability with corrective drills: {drill_names} before your next session."
        )

    # 2. Fatigue Analysis
    if fatigue_drop >= 15:
        recommendations.append(
            f"Significant fatigue drop ({fatigue_drop}% score decline in 2nd half) indicates form breakdown under volume. "
            "Consider reducing working sets by 20% or extending rest intervals by 45 seconds."
        )
    elif fatigue_drop < 5 and total_reps >= 10:
        recommendations.append(
            "Excellent stamina: form remained consistent throughout the entire set."
        )

    # 3. High Risk & Safety Check
    if high_risk_reps > 0:
        recommendations.append(
            f"{high_risk_reps} reps exhibited high movement risk. Focus on lighter loading and tempo control (3-sec eccentric phase) to rebuild motor patterns."
        )

    # 4. Injury History Context
    if user_profile and user_profile.get("injury_history"):
        hist = ", ".join(user_profile["injury_history"])
        recommendations.append(
            f"Noted previous history ({hist}): prioritize full dynamic warm-ups and avoid training through sharp joint discomfort."
        )

    return {
        "primary_area": worst_area,
        "recommendations": recommendations,
        "corrective_exercises": corrective_drills,
        "disclaimer": MEDICAL_DISCLAIMER,
    }


def chat_response_fallback(
    user_query: str,
    user_profile: Optional[Dict[str, Any]] = None,
    recent_reports: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Intelligent fallback chat assistant following biomechanical safety guidelines."""
    query_lower = user_query.lower()

    # Medical / pain red flags
    if any(w in query_lower for w in ["hurt", "pain", "tore", "injury", "injured", "sharp", "swollen", "doctor", "ache"]):
        return (
            "I hear you, but as an AI biomechanical screening tool, I cannot diagnose injuries or prescribe medical treatment. "
            "If you are experiencing persistent or sharp pain, please stop the movement immediately and consult a certified physiotherapist, "
            "sports doctor, or medical professional. In the meantime, avoid loading that joint and apply active rest."
        )

    # Squat depth & knee caves
    if "squat" in query_lower and ("knee" in query_lower or "cave" in query_lower or "valgus" in query_lower):
        return (
            "Knee cave (valgus) during squats usually stems from underactive gluteus medius muscles or restricted ankle dorsiflexion. "
            "Try adding Banded Lateral Squats and Glute Bridges to your warm-up, and cue 'screwing your feet into the floor' to maintain external hip torque."
        )

    # Back rounding / hip hinge
    if "back" in query_lower or "spine" in query_lower:
        return (
            "Maintaining a neutral spine requires bracing your core 360 degrees and initiating the movement from your hips rather than lumbar flexion. "
            "Incorporate Bird-Dogs and Dead Bugs to strengthen anti-rotational core stability. If back soreness persists, have an expert coach or physio evaluate your spinal posture."
        )

    # Fatigue / volume advice
    if "fatigue" in query_lower or "tired" in query_lower or "sore" in query_lower:
        return (
            "When fatigue sets in, joint stabilizers tire first, which increases injury risk. "
            "If your rep scores drop in the second half of sets, consider capping your reps 2 shy of technical failure (RPE 7-8) to keep form pristine."
        )

    # General technique assistant
    user_name = user_profile.get("name", "Athlete") if user_profile else "Athlete"
    return (
        f"Hi {user_name}! I can analyze your rep kinematics, explain defect root causes, and recommend corrective mobility drills. "
        "How can I help you optimize your movement technique today?"
    )


async def ask_ai_coach(
    user_query: str,
    user_profile: Optional[Dict[str, Any]] = None,
    recent_reports: Optional[List[Dict[str, Any]]] = None,
    api_key: Optional[str] = None,
) -> str:
    """Query AI Coach with full athlete biomechanical context."""
    claude_key = api_key or os.getenv("ANTHROPIC_API_KEY")

    if claude_key:
        try:
            import httpx
            
            system_prompt = (
                "You are Vyronex AI Coach, an elite biomechanical and fitness movement assistant. "
                "CRITICAL RULES:\n"
                "1. Strictly provide movement technique, joint kinematics, and corrective exercise advice.\n"
                "2. NEVER diagnose medical conditions or prescribe rehabilitation treatments.\n"
                "3. If the user mentions sharp pain, swelling, or repeated HIGH injury risk, immediately advise consulting a licensed physician or physiotherapist.\n"
                "4. Keep responses direct, encouraging, actionable, and under 150 words.\n"
                f"5. Always append the disclaimer: '{MEDICAL_DISCLAIMER}'"
            )

            context_str = f"Athlete Profile: {json.dumps(user_profile or {})}\nRecent Workout Summaries: {json.dumps(recent_reports or [])}"
            messages = [
                {"role": "user", "content": f"Context:\n{context_str}\n\nQuestion: {user_query}"}
            ]

            async with httpx.AsyncClient(timeout=15.0) as client:
                res = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": claude_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json={
                        "model": "claude-3-haiku-20240307",
                        "max_tokens": 300,
                        "system": system_prompt,
                        "messages": messages,
                    },
                )
                if res.status_code == 200:
                    data = res.json()
                    reply = data["content"][0]["text"]
                    return reply
        except Exception as exc:
            logger.warning("Claude API query failed (%s); using intelligent fallback.", exc)

    return chat_response_fallback(user_query, user_profile, recent_reports)
