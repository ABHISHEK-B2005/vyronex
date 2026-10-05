"""Central configuration for the Injury Risk Detection prototype.

All "magic numbers" (thresholds, weights, colors, angle targets) live here
so they are easy to tune from one place instead of being scattered
throughout the codebase.

NOTE ON VALIDITY:
All thresholds below are ENGINEERING ASSUMPTIONS made for a prototype
demo. They are NOT derived from clinical research and must not be
presented as medically validated cut-offs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple


# --------------------------------------------------------------------------
# Landmark confidence
# --------------------------------------------------------------------------
MIN_LANDMARK_VISIBILITY = 0.5  # below this, a landmark is treated as missing

# --------------------------------------------------------------------------
# Squat state-machine thresholds (knee angle in degrees)
# --------------------------------------------------------------------------
STANDING_KNEE_ANGLE = 165.0       # knee considered "straight" above this
BOTTOM_KNEE_ANGLE = 110.0         # knee considered "deep" below this
DESCEND_TRIGGER_DELTA = 8.0       # degrees dropped from standing to trigger descent
ASCEND_TRIGGER_DELTA = 8.0        # degrees regained from bottom to trigger ascent
MIN_FRAMES_IN_PHASE = 3           # smoothing: min consecutive frames before phase switch

# --------------------------------------------------------------------------
# "Ideal" biomechanical reference ranges (prototype assumptions)
# --------------------------------------------------------------------------
IDEAL_MIN_KNEE_ANGLE: Tuple[float, float] = (80.0, 110.0)   # good squat depth range
IDEAL_TORSO_ANGLE: Tuple[float, float] = (45.0, 90.0)       # degrees from horizontal; closer to 90 = upright
MAX_KNEE_VALGUS_SHIFT_PX_RATIO = 0.08  # normalized inward-knee shift considered risky
MAX_KNEE_SYMMETRY_DIFF = 12.0          # degrees difference between left/right knee angle
MAX_HIP_SYMMETRY_DIFF = 12.0
IDEAL_REP_DURATION: Tuple[float, float] = (1.2, 4.0)  # seconds; too fast/slow both penalized

# --------------------------------------------------------------------------
# Form-analysis category weights (must sum to 1.0) -> used for Overall Form Score
# --------------------------------------------------------------------------
FORM_WEIGHTS: Dict[str, float] = {
    "knee_alignment": 0.30,
    "knee_depth": 0.20,
    "back_posture": 0.25,
    "movement_control": 0.15,
    "symmetry": 0.10,
}

# --------------------------------------------------------------------------
# Risk engine weights (must sum to 1.0)
# --------------------------------------------------------------------------
RISK_WEIGHTS: Dict[str, float] = {
    "knee_alignment": 0.30,
    "back_posture": 0.25,
    "knee_depth": 0.20,
    "movement_control": 0.15,
    "symmetry": 0.10,
}

# --------------------------------------------------------------------------
# Risk classification thresholds (risk score 0-100, higher = riskier)
# --------------------------------------------------------------------------
RISK_LOW_MAX = 30
RISK_MODERATE_MAX = 60
# > RISK_MODERATE_MAX => HIGH

RISK_LABELS = {
    "LOW": "LOW MOVEMENT RISK",
    "MODERATE": "MODERATE MOVEMENT RISK",
    "HIGH": "HIGH MOVEMENT RISK",
}

# --------------------------------------------------------------------------
# UI Colors (BGR for OpenCV, hex for Streamlit)
# --------------------------------------------------------------------------
COLOR_LOW_BGR = (0, 200, 0)
COLOR_MODERATE_BGR = (0, 165, 255)
COLOR_HIGH_BGR = (0, 0, 255)
COLOR_NEUTRAL_BGR = (255, 200, 100)

COLOR_LOW_HEX = "#22c55e"
COLOR_MODERATE_HEX = "#f59e0b"
COLOR_HIGH_HEX = "#ef4444"
COLOR_PRIMARY_HEX = "#0b1f3a"      # dark navy
COLOR_PRIMARY_LIGHT_HEX = "#132a4d"

# --------------------------------------------------------------------------
# Camera / performance
# --------------------------------------------------------------------------
DEFAULT_CAMERA_INDEX = 0
TARGET_FRAME_WIDTH = 640
TARGET_FRAME_HEIGHT = 480
POSE_MIN_DETECTION_CONFIDENCE = 0.5
POSE_MIN_TRACKING_CONFIDENCE = 0.5

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
SAMPLE_DATA_PATH = "data/sample_features.csv"
COLLECTED_DATA_PATH = "data/collected_features.csv"
SESSIONS_DIR = "data/sessions"
MODEL_PATH = "ml/model/risk_model.joblib"
MODEL_COLUMNS_PATH = "ml/model/feature_columns.json"

# --------------------------------------------------------------------------
# ML labels
# --------------------------------------------------------------------------
ML_LABELS = ["SAFE", "MODERATE_RISK", "HIGH_RISK"]

FEATURE_COLUMNS = [
    "knee_angle",
    "hip_angle",
    "torso_angle",
    "knee_alignment",
    "symmetry",
    "movement_speed",
    "rep_duration",
]


@dataclass(frozen=True)
class Disclaimer:
    short: str = (
        "Prototype movement-risk screening — not a medical diagnostic system."
    )
    full: str = (
        "This prototype provides movement-risk screening and educational "
        "feedback. It is not a medical diagnostic system and should not "
        "replace a qualified healthcare professional."
    )
    privacy: str = (
        "Camera processing is performed locally. Video is not stored by "
        "default."
    )


DISCLAIMER = Disclaimer()
