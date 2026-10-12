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
SQUAT_DOWN_ANGLE = 110.0
SQUAT_UP_ANGLE = 160.0
SQUAT_DESCENT_START_ANGLE = 160.0
MIN_REP_DURATION_SECONDS = 0.8
MIN_REP_SAMPLES = 8
VIEW_FRONT_MIN_RATIO = 0.55
VIEW_SIDE_MAX_RATIO = 0.30
MAX_KNEE_DRIFT_LEG_RATIO = 0.25
MAX_FRAME_JUMP_TORSO_RATIO = 0.45
MAX_UNUSABLE_FRAME_SHARE = 0.30
POSITION_SMOOTHING_ALPHA = 0.35
POSITION_MEDIAN_WINDOW = 5

# Alignment scores are prototype heuristics, not clinically validated.
ALIGNMENT_GOOD_SCORE = 80.0
ALIGNMENT_WATCH_SCORE = 60.0
ALIGNMENT_MAX_KNEE_DRIFT_RATIO = 0.08
ALIGNMENT_MAX_LEVEL_TILT_DEGREES = 8.0
ALIGNMENT_MAX_WEIGHT_SHIFT_DEGREES = 12.0
ALIGNMENT_IDEAL_ANKLE_HIP_WIDTH_RATIO = 1.0
ALIGNMENT_MAX_ANKLE_HIP_WIDTH_DEVIATION = 0.60
ALIGNMENT_MAX_TORSO_LEAN_DEGREES = 40.0
ALIGNMENT_MAX_NECK_TORSO_DEGREES = 20.0
ALIGNMENT_MAX_KNEE_OVER_TOE_RATIO = 0.15
ALIGNMENT_IDEAL_BOTTOM_KNEE_ANGLE = 90.0
ALIGNMENT_MAX_BOTTOM_KNEE_ANGLE_DEVIATION = 60.0

# --------------------------------------------------------------------------
# "Ideal" biomechanical reference ranges (prototype assumptions)
# --------------------------------------------------------------------------
IDEAL_MIN_KNEE_ANGLE: Tuple[float, float] = (80.0, 110.0)   # good squat depth range
IDEAL_TORSO_ANGLE: Tuple[float, float] = (70.0, 90.0)       # degrees from horizontal; closer to 90 = upright
MAX_KNEE_VALGUS_SHIFT_LEG_RATIO = 0.08  # inward knee shift as a fraction of hip-to-ankle length
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

# --------------------------------------------------------------------------
# Rep Classification & Defect mapping
# --------------------------------------------------------------------------
REP_FORM_SCORE_CORRECT_MIN = 70.0

DEFECT_BODY_AREA_MAP: Dict[str, str] = {
    "knee_alignment": "knee",
    "knee_depth": "knee, hip",
    "back_posture": "lower_back",
    "symmetry": "hip",
    "movement_control": "full_chain",
    "elbow flare (push-up)": "shoulder",
    "elbow flare": "shoulder",
    "elbow_flare": "shoulder",
    "landing (long jump)": "ankle, knee",
    "landing": "ankle, knee",
    "landing_mechanics": "ankle, knee",
}

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
