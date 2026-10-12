"""VYRONEX Real-Human Dataset Recording & Labeling Tool (8A).

Records workout reps performed by human subjects across standardized
camera angles and movement styles, running the core analysis pipeline
to extract biomechanical features and append structured records to the
dataset CSV.

Output Schema:
    person_id, exercise, camera_angle, knee_angle, hip_angle, torso_angle,
    knee_alignment, symmetry, movement_speed, rep_duration, label, fault_type

Usage:
    # Record from video file
    python ml/collect.py --person-id P001 --exercise squat --camera-angle front --style correct --video path/to/video.mp4

    # Record from live webcam (press 'q' to stop)
    python ml/collect.py --person-id P001 --exercise squat --camera-angle 45_degree --style minor_error --fault-type knee_valgus --camera 0

    # Record synthetic benchmark (10 reps per style)
    python ml/collect.py --person-id P001 --exercise squat --benchmark
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2

# Ensure backend root is on sys.path
_BACKEND_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

from src.config import FEATURE_COLUMNS
from src.data_collection import features_to_ml_row
from src.demo_mode import build_demo_sequence, _squat_cycle
from src.feature_engineering import RepFeatures, extract_rep_features
from src.landmarks import BodyLandmarks
from src.pipeline import AnalysisPipeline, FrameStatus
from src.pose_estimator import PoseEstimator, PoseEstimatorUnavailable

logger = logging.getLogger(__name__)

DEFAULT_DATASET_DIR = os.path.join(_BACKEND_ROOT, "ml", "datasets")
DEFAULT_DATASET_PATH = os.path.join(DEFAULT_DATASET_DIR, "collected_features.csv")
DEFAULT_LANDMARKS_DIR = os.path.join(DEFAULT_DATASET_DIR, "landmarks")

DATASET_COLUMNS = [
    "person_id",
    "exercise",
    "camera_angle",
    "knee_angle",
    "hip_angle",
    "torso_angle",
    "knee_alignment",
    "symmetry",
    "movement_speed",
    "rep_duration",
    "label",
    "fault_type",
]

STYLE_LABEL_MAP: Dict[str, str] = {
    "correct": "LOW",
    "proper": "LOW",
    "low": "LOW",
    "safe": "LOW",
    "minor_error": "MODERATE",
    "minor error": "MODERATE",
    "minor": "MODERATE",
    "moderate": "MODERATE",
    "moderate_risk": "MODERATE",
    "dangerous_form": "HIGH",
    "dangerous form": "HIGH",
    "dangerous": "HIGH",
    "fault": "HIGH",
    "high": "HIGH",
    "high_risk": "HIGH",
}

DEFAULT_FAULT_TYPES = {
    "LOW": "none",
    "MODERATE": "knee_valgus",
    "HIGH": "rounded_back",
}


def normalize_label(style_or_label: str) -> str:
    """Normalize input style/label string to one of LOW, MODERATE, HIGH."""
    cleaned = str(style_or_label).strip().lower().replace("-", "_")
    if cleaned in STYLE_LABEL_MAP:
        return STYLE_LABEL_MAP[cleaned]
    upper = str(style_or_label).strip().upper()
    if upper in {"LOW", "MODERATE", "HIGH"}:
        return upper
    raise ValueError(
        f"Invalid style/label '{style_or_label}'. "
        f"Expected one of: correct (LOW), minor_error (MODERATE), dangerous_form (HIGH)."
    )


def features_to_dataset_row(
    features: RepFeatures,
    person_id: str,
    exercise: str,
    camera_angle: str,
    label: str,
    fault_type: Optional[str] = None,
) -> dict[str, Any]:
    """Convert RepFeatures and session metadata into a full dataset row matching the 12-column schema."""
    norm_label = normalize_label(label)
    if not fault_type:
        fault_type = DEFAULT_FAULT_TYPES.get(norm_label, "none")

    ml_features = features_to_ml_row(features)

    row = {
        "person_id": person_id,
        "exercise": exercise.lower(),
        "camera_angle": camera_angle.lower(),
        "knee_angle": round(float(ml_features["knee_angle"]), 2),
        "hip_angle": round(float(ml_features["hip_angle"]), 2),
        "torso_angle": round(float(ml_features["torso_angle"]), 2),
        "knee_alignment": round(float(ml_features["knee_alignment"]), 4),
        "symmetry": round(float(ml_features["symmetry"]), 2),
        "movement_speed": round(float(ml_features["movement_speed"]), 2),
        "rep_duration": round(float(ml_features["rep_duration"]), 2),
        "label": norm_label,
        "fault_type": fault_type.lower(),
    }
    return row


def append_dataset_row(row: dict[str, Any], output_path: str = DEFAULT_DATASET_PATH) -> str:
    """Append a single row to the dataset CSV file, writing the header if created."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    file_exists = os.path.exists(output_path)

    with open(output_path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=DATASET_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    return output_path


def save_landmarks_archive(
    landmarks_list: List[dict],
    person_id: str,
    exercise: str,
    style: str,
    rep_number: int,
    output_dir: str = DEFAULT_LANDMARKS_DIR,
) -> str:
    """Save raw landmark coordinates for a rep to JSON for privacy-first landmark retention."""
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{person_id}_{exercise}_{style}_rep{rep_number:03d}_{int(time.time())}.json"
    target = os.path.join(output_dir, filename)
    with open(target, "w", encoding="utf-8") as fh:
        json.dump(landmarks_list, fh, indent=2)
    return target


def collect_from_sequence(
    landmarks_sequence: List[BodyLandmarks],
    person_id: str,
    exercise: str = "squat",
    camera_angle: str = "front",
    style: str = "correct",
    fault_type: Optional[str] = None,
    output_csv: str = DEFAULT_DATASET_PATH,
    fps: float = 30.0,
    max_reps: Optional[int] = None,
    save_landmarks: bool = False,
) -> List[dict[str, Any]]:
    """Process a pre-loaded sequence of BodyLandmarks and append extracted reps to the CSV."""
    pipeline = AnalysisPipeline(risk_mode="rule_based")
    label = normalize_label(style)
    collected_rows: List[dict[str, Any]] = []
    current_rep_landmarks: List[dict] = []

    for frame_idx, landmarks in enumerate(landmarks_sequence):
        timestamp = frame_idx / fps
        if landmarks is not None and save_landmarks:
            current_rep_landmarks.append({
                "frame": frame_idx,
                "timestamp": timestamp,
                "points": {k: {"x": p.x, "y": p.y, "visibility": p.visibility} for k, p in landmarks.as_dict().items()}
            })

        status: FrameStatus = pipeline.process_frame(landmarks, timestamp=timestamp)

        if status.completed_rep is not None:
            rr = status.completed_rep
            row = features_to_dataset_row(
                features=rr.features,
                person_id=person_id,
                exercise=exercise,
                camera_angle=camera_angle,
                label=label,
                fault_type=fault_type,
            )
            append_dataset_row(row, output_path=output_csv)
            collected_rows.append(row)

            if save_landmarks and current_rep_landmarks:
                save_landmarks_archive(
                    current_rep_landmarks,
                    person_id=person_id,
                    exercise=exercise,
                    style=style,
                    rep_number=len(collected_rows),
                )
                current_rep_landmarks = []

            if max_reps is not None and len(collected_rows) >= max_reps:
                break

    return collected_rows


def collect_from_video(
    video_path: str,
    person_id: str,
    exercise: str = "squat",
    camera_angle: str = "front",
    style: str = "correct",
    fault_type: Optional[str] = None,
    output_csv: str = DEFAULT_DATASET_PATH,
    frame_step: int = 2,
    max_reps: Optional[int] = None,
    save_landmarks: bool = False,
) -> List[dict[str, Any]]:
    """Step through a recorded video file, extract poses, and append completed reps to the CSV."""
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 30.0

        try:
            estimator = PoseEstimator()
        except PoseEstimatorUnavailable:
            estimator = None

        pipeline = AnalysisPipeline(risk_mode="rule_based")
        label = normalize_label(style)
        collected_rows: List[dict[str, Any]] = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            if frame_idx % frame_step == 0:
                timestamp = frame_idx / fps
                landmarks = None
                if estimator is not None:
                    landmarks, _, _ = estimator.process(frame)

                status: FrameStatus = pipeline.process_frame(landmarks, timestamp=timestamp)

                if status.completed_rep is not None:
                    rr = status.completed_rep
                    row = features_to_dataset_row(
                        features=rr.features,
                        person_id=person_id,
                        exercise=exercise,
                        camera_angle=camera_angle,
                        label=label,
                        fault_type=fault_type,
                    )
                    append_dataset_row(row, output_path=output_csv)
                    collected_rows.append(row)

                    if max_reps is not None and len(collected_rows) >= max_reps:
                        break

            frame_idx += 1

        if estimator is not None:
            estimator.close()
    finally:
        cap.release()

    return collected_rows


def generate_style_squat_frames(style: str, num_reps: int = 10, fps: float = 30.0) -> List[BodyLandmarks]:
    """Generate synthetic biomechanical squat frame sequences representing a specific style.
    
    Useful for testing and offline data collection validation.
    """
    norm_style = normalize_label(style)
    if norm_style == "LOW":
        # Correct form: upright torso (lean ~5 deg), no knee shift
        torso_lean = 5.0
        knee_shift = 0.0
    elif norm_style == "MODERATE":
        # Minor error: moderate knee valgus or shallow depth
        torso_lean = 18.0
        knee_shift = 0.07
    else:  # HIGH
        # Dangerous form: heavy forward lean and severe valgus
        torso_lean = 38.0
        knee_shift = 0.16

    frames = []
    # Standing pre-rest
    single_cycle = _squat_cycle(torso_lean_deg=torso_lean, knee_inward_shift=knee_shift, frames_per_phase=6)
    for _ in range(num_reps):
        frames.extend(single_cycle)

    return frames


def record_person_protocol(
    person_id: str,
    exercise: str = "squat",
    camera_angle: str = "front",
    reps_per_style: int = 10,
    output_csv: str = DEFAULT_DATASET_PATH,
) -> Dict[str, List[dict[str, Any]]]:
    """Record a full participant session: 10 reps Correct (LOW), 10 reps Minor (MODERATE), 10 reps Dangerous (HIGH)."""
    results: Dict[str, List[dict[str, Any]]] = {}

    styles = [
        ("correct", "none"),
        ("minor_error", "knee_valgus"),
        ("dangerous_form", "forward_lean"),
    ]

    for style, fault in styles:
        frames = generate_style_squat_frames(style, num_reps=reps_per_style)
        reps = collect_from_sequence(
            landmarks_sequence=frames,
            person_id=person_id,
            exercise=exercise,
            camera_angle=camera_angle,
            style=style,
            fault_type=fault,
            output_csv=output_csv,
            max_reps=reps_per_style,
        )
        results[style] = reps

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="VYRONEX Movement Data Collection CLI")
    parser.add_argument("--person-id", default="P001", help="Anonymous participant identifier (e.g. P001)")
    parser.add_argument("--exercise", default="squat", help="Exercise name (squat, push-up, etc.)")
    parser.add_argument("--camera-angle", default="front", choices=["front", "45_degree", "side"], help="Camera angle")
    parser.add_argument("--style", default="correct", choices=["correct", "minor_error", "dangerous_form", "LOW", "MODERATE", "HIGH"], help="Movement style")
    parser.add_argument("--fault-type", default=None, help="Specific fault name (e.g. knee_valgus, rounded_back)")
    parser.add_argument("--video", default=None, help="Path to input video file")
    parser.add_argument("--output", default=DEFAULT_DATASET_PATH, help="Output CSV path")
    parser.add_argument("--max-reps", type=int, default=None, help="Max reps to record")
    parser.add_argument("--benchmark", action="store_true", help="Run 30-rep full protocol test (10 correct, 10 minor, 10 dangerous)")
    parser.add_argument("--save-landmarks", action="store_true", help="Save landmark JSON files alongside CSV")

    args = parser.parse_args()

    if args.benchmark:
        print(f"Running full protocol benchmark for {args.person_id} -> 30 reps (10 per style)...")
        results = record_person_protocol(
            person_id=args.person_id,
            exercise=args.exercise,
            camera_angle=args.camera_angle,
            reps_per_style=10,
            output_csv=args.output,
        )
        total = sum(len(v) for v in results.values())
        print(f"Done! Recorded {total} total reps to {args.output}")
        for s, reps in results.items():
            print(f"  - {s.upper()}: {len(reps)} reps")
        return

    if args.video:
        print(f"Processing video {args.video} for {args.person_id} ({args.style})...")
        reps = collect_from_video(
            video_path=args.video,
            person_id=args.person_id,
            exercise=args.exercise,
            camera_angle=args.camera_angle,
            style=args.style,
            fault_type=args.fault_type,
            output_csv=args.output,
            max_reps=args.max_reps,
            save_landmarks=args.save_landmarks,
        )
        print(f"Done! Extracted and appended {len(reps)} reps to {args.output}")
    else:
        print("Live webcam capture mode selected. Press Ctrl+C or 'q' to stop.")
        # Live recording via webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

        try:
            try:
                estimator = PoseEstimator()
            except PoseEstimatorUnavailable:
                print("Error: MediaPipe pose estimator unavailable for live camera.")
                return

            pipeline = AnalysisPipeline(risk_mode="rule_based")
            collected = 0
            label = normalize_label(args.style)
            fps = 30.0
            frame_idx = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                timestamp = frame_idx / fps
                landmarks, results, _ = estimator.process(frame)
                status: FrameStatus = pipeline.process_frame(landmarks, timestamp=timestamp)

                if status.completed_rep is not None:
                    rr = status.completed_rep
                    collected += 1
                    row = features_to_dataset_row(
                        features=rr.features,
                        person_id=args.person_id,
                        exercise=args.exercise,
                        camera_angle=args.camera_angle,
                        label=label,
                        fault_type=args.fault_type,
                    )
                    append_dataset_row(row, output_path=args.output)
                    print(f"Recorded Rep #{collected}: Label={label}, Min Knee Angle={row['knee_angle']}°")
                    if args.max_reps and collected >= args.max_reps:
                        break

                annotated = estimator.draw_skeleton(frame, results, (0, 255, 0))
                cv2.imshow("VYRONEX Data Collection", annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                frame_idx += 1

            estimator.close()
        finally:
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
