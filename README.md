# VYRONEX — Biomechanical Screening & Movement Risk Platform

VYRONEX is an end-to-end computer-vision and machine-learning platform for biomechanical movement-risk screening, kinematics tracking, coach supervision, and longitudinal athletic analytics.

---

## 🏗 System Architecture

```
                                +-----------------------------------+
                                |        Athlete / Gym User         |
                                |       Browser / Mobile Client     |
                                +-----------------+-----------------+
                                                  |
                         WebSockets / REST API    |   MediaPipe Pose (3D World + 2D Image Landmarks)
                                                  v
+---------------------------------------------------------------------------------------------------+
|                                    VYRONEX BACKEND (FastAPI)                                      |
|                                                                                                   |
|  +---------------------+   +---------------------+   +--------------------+   +----------------+  |
|  | Auth & Consent      |   | Video Processor     |   | Risk Engine        |   | Report Builder |  |
|  | (JWT, RBAC, Data Del|   | (OpenCV, MediaPipe) |   | (Rule + ML + SHAP) |   | & PDF Exporter |  |
|  +---------------------+   +---------------------+   +--------------------+   +----------------+  |
|                                                                                                   |
|  +---------------------+   +---------------------+   +--------------------+   +----------------+  |
|  | AI Coach Assistant  |   | Athletics Reference |   | Expert Review      |   | Safe Retrain   |  |
|  | (Anthropic Claude)  |   | Profiles (Z-Scores) |   | Queue (Physio/Coach|   | (GroupKFold)   |  |
|  +---------------------+   +---------------------+   +--------------------+   +----------------+  |
+-------------------------------------------------+-------------------------------------------------+
                                                  |
                                                  v
                                      +-----------------------+
                                      |   MongoDB Database    |
                                      | (Sessions, Reps, User |
                                      +-----------------------+
```

---

## 🌟 Key Capabilities & Module Overview

### 1. Multi-Modal Analysis
- **Live Stream (`/ws/live`)**: Sub-second feedback using in-browser MediaPipe pose estimation streamed over WebSockets.
- **Video Upload (`/api/upload/analyze`)**: Processes every decoded frame with exact `frame_index / fps` timestamps, smooths MediaPipe world landmarks, detects camera view, and deletes the uploaded video after analysis. Joint coordinates and per-frame diagnostic metadata are retained for report recalculation.
- **Upload Diagnostics**: `/api/upload/debug/{session_id}` provides per-frame JSON or `?format=csv`; `POST /api/upload/recompute/{session_id}` rebuilds reps and reports from saved pose landmarks.
- **View-aware checks**: Front view scores knee alignment and symmetry; side view scores squat depth and torso lean; angled view scores depth only. Tune view thresholds (`0.55` front, `0.30` side), 25% maximum knee drift, and the 30% unusable-frame limit in `backend/src/config.py`.
- **Body alignment**: Smoothed 3D pose frames produce front- or side-view alignment metrics. Unavailable metrics remain `None`/“Not assessed”; alignment scores average assessed metrics only. Tune score bands and per-metric limits in `backend/src/config.py`. Live responses include `view` and `alignment`; each stored rep and report includes its alignment summary.
  - Tunable defaults: front/side view ratios `0.55`/`0.30`; status bands Good `>=80`, Watch `>=60`; knee drift `0.08` leg length; shoulder/hip tilt `8°`; knee-angle difference `12°`; torso lean `40°`; neck-torso deviation `20°`; knee-over-toe `0.15` leg length; ankle/hip width target `1.0` (tolerance `0.60`); bottom knee angle target `90°` (tolerance `60°`).

### 2. Biomechanical Risk & SHAP Explanations
- **Person-Wise Trained Classifiers**: Cross-validated with `GroupKFold` on participant IDs (`person_id`) to prevent data leakage.
- **Interpretability**: Real-time SHAP feature attributions explaining the top 3 kinematic factors driving elevated risk predictions.
- **Athletics Reference Profiles**: Z-score deviation scoring against coach-approved biomechanical standards for **Long Jump**, **Shot Put**, and **Javelin Throw**.

### 3. Human-in-the-Loop Supervision & Safe Retraining
- **Athlete Rating Feedback**: Athletes can flag incorrect ratings (`POST /api/feedback/rep/{rep_id}`).
- **Expert Review Queue (`expert.html`)**: Coaches and physiotherapists review disputed reps and low-confidence predictions (`< 60%`), assigning verified ground-truth labels.
- **Champion-Challenger Safe Retraining**: Challenger models are evaluated against the active champion on held-out benchmarks. If high-risk recall regresses, the candidate is automatically rejected and logged.

### 4. AI Biomechanical Coach (`coach.html`)
- **Interactive Chat Assistant**: Biomechanical assistant with full athlete history context.
- **Corrective Exercise Library**: Targeted drills mapped by joint area (`knee`, `lower_back`, `shoulder`, `ankle`, `hip`, `elbow`).
- **Safety Boundaries**: Strictly enforces medical disclaimers, never diagnoses injuries, and refers users to licensed professionals for joint pain.

### 5. Comprehensive Clinical & Coach Reports (`report.html`)
- **Non-Overlapping Metric Accounting**: Exact distribution: `correct + defective + high_risk = total_reps`.
- **Fatigue Progression**: First half vs. second half form score drop percentage.
- **Interactive Body Map**: Dynamic SVG mapping high-risk stress points across joint chains.
- **PDF Export**: Downloadable reports generated with ReportLab and Matplotlib (`GET /api/reports/{session_id}/pdf`).

### 6. Longitudinal Progress & Trend Tracking (`history.html`)
- **Recurring Weakness Detection**: Automatically surfaces chronic joint loading issues across sessions.
- **Screening Risk Forecast**: Early indicator combining fatigue decay, frequency, and defect consistency.
- **Gamification**: Streaks and milestone achievement badges.

---

## 🚀 Quick Start Guide

### Prerequisites
- Python 3.10+
- Docker & Docker Compose (optional for containerized deployment)
- Node.js or local HTTP server for static frontend

### Option A: Local Development Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ABHISHEK-B2005/vyronex.git
   cd vyronex
   ```

2. **Set up backend:**
   ```bash
   cd backend
   python -m venv venv
   # Windows
   .\venv\Scripts\activate
   # Linux/macOS
   source venv/bin/activate

   pip install -r requirements.txt
   uvicorn app:app --reload --port 8000
   ```

3. **Open Frontend:**
   - In a separate terminal:
     ```bash
     cd frontend
     python -m http.server 8080
     ```
   - Open your browser at `http://localhost:8080/login.html`

---

### Option B: Docker Compose Deployment

```bash
docker-compose up --build
```
The FastAPI backend will be available at `http://localhost:8000` connected to MongoDB on port 27017.

---

## 📡 Core API Reference

| Method | Endpoint | Role / Auth | Description |
|---|---|---|---|
| `POST` | `/api/auth/signup` | Public | Create account with role (`gym`, `athlete`, `expert`) & opt-in consent |
| `POST` | `/api/auth/login` | Public | Authenticate and retrieve JWT access token |
| `DELETE`| `/api/auth/data` | Authenticated | Withdraw consent and wipe stored movement features |
| `POST` | `/api/upload/analyze` | Authenticated | Upload video for background computer vision analysis |
| `GET`  | `/api/upload/status/{job_id}` | Authenticated | Poll video analysis job progress |
| `GET`  | `/api/upload/debug/{session_id}` | Authenticated | Inspect per-frame pose metrics; add `?format=csv` to export |
| `POST` | `/api/upload/recompute/{session_id}` | Authenticated | Rebuild rep results and report from saved pose landmarks |
| `POST` | `/api/feedback/rep/{rep_id}` | Authenticated | Submit rep rating validation ("Was this rating correct?") |
| `GET`  | `/api/expert/queue` | `expert` | Fetch review queue of disputed and low-confidence reps |
| `POST` | `/api/expert/label/{rep_id}`| `expert` | Coach/physio sets true ground-truth label |
| `POST` | `/api/expert/retrain` | `expert` | Execute safe champion-challenger model retraining |
| `GET`  | `/api/expert/versions` | `expert` | List model version registry and activation history |
| `POST` | `/api/coach/chat` | Authenticated | Biomechanical chat assistant (rate-limited) |
| `GET`  | `/api/coach/recommendations/{session_id}` | Authenticated | Post-workout corrective drills |
| `GET`  | `/api/reports/{session_id}` | Authenticated | Fetch full session report & analytics |
| `GET`  | `/api/reports/{session_id}/pdf` | Authenticated | Download clinical PDF report |
| `GET`  | `/api/alignment/summary?activity=&days=30` | Authenticated | Alignment score trend and recurring body-area issues |
| `GET`  | `/api/history` | Authenticated | Filterable session history, trends, and risk forecast |

---

## 🧪 Testing

Run the comprehensive test suite with Python unittest:

```bash
cd backend
python -m unittest discover tests
```

For targeted alignment regressions, run `python -m pytest tests/test_video_processor.py tests/test_coach_and_reports.py`.

### Real-clip alignment check

Use four squat clips with a fixed camera, good light, and the whole body visible:

1. **Front, good form:** face the camera and keep knees and shoulders balanced; expect front metrics assessed and mostly Good.
2. **Front, knees-in:** perform controlled reps with a visible inward knee drift; expect knee alignment to change to Watch/Fix, not a tracking error.
3. **Side, good form:** turn 90 degrees to the camera and squat with a comfortable upright torso; expect side metrics assessed and front-only metrics shown as not assessed.
4. **Side, leaning:** repeat from the side with an obvious forward torso lean; expect torso alignment to be flagged.

Compare the live panel and saved report. Repeat clips with feet/knees/hips occluded only as a tracking test; those metrics should be not assessed rather than shown as zero.

---

## ⚠️ Honest Technical & Clinical Limitations

1. **Movement-Risk Screening Only**: VYRONEX is a biomechanical coaching and injury-risk screening aid. It is **NOT** a medical diagnostic tool or medical device. All thresholds and risk indicators are prototype engineering approximations.
2. **2D Single-Camera Perspective**: Monocular 2D pose estimation is sensitive to camera positioning, parallax, loose clothing, and occlusions. Standardized camera angles (45° oblique or sagittal) yield the most reliable kinematic measurements.
3. **Hardware Independence**: Smart Ring readings are currently simulated in demo mode; physical Bluetooth Smart Ring device pairing requires manufacturer SDK integration.
