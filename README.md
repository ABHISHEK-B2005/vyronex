# VYRONEX — AI Fitness & Athletics Performance Platform

Two real parts:

- **frontend/** — a static, multi-page site. Runs in the browser, captures
  webcam video or an uploaded file, runs MediaPipe Pose to get a 33-point
  skeleton, and either calls the backend (Squat) or a local heuristic
  engine (everything else) to turn that into scores and feedback.
- **backend/** — a real Python (FastAPI) server, unchanged from the
  original prototype's `src/`: squat rep/phase detection, 5-factor
  scoring, risk engine, feedback generator.

## Mandatory flow

**Category → Activity → Live Feed / Video Upload → Analysis → Results**

Only three pages live in the main navigation — **Home, Gym, Athletics**.
Live Feed, Video Upload and Results are intentionally *not* separate nav
pages: they're steps that unlock inside `activity.html` only after a
specific workout or event has been selected from Gym or Athletics, and
the Result is shown inline at the end of that same flow. `activity.html`
checks for a `?activity=` query param; if it's missing or invalid it
shows a "Select an Activity First" screen with links back into Gym /
Athletics — it never falls through to a camera or upload UI.

There is no visible backend-URL/connection control in the UI — the
Squat engine talks to the backend at a fixed `http://localhost:8000`
internally.

## Pages

| Page | File | Purpose |
|---|---|---|
| Home | `index.html` | Welcome + GYM / ATHLETICS category picker |
| Gym | `gym.html` | Choose Your Workout: Push Up · Squat · Walking Lunges |
| Athletics | `athletics.html` | Choose Your Event: Long Jump · Shot Put · Javelin Throw |
| *(no nav link)* | `activity.html?activity=<id>` | Instructions → method choice → Live Feed / Video Upload → Performance Results, all scoped to that one activity — reached only via a card on Gym or Athletics |

## Scalable architecture

Every workout/event is **one entry** in `frontend/assets/activities.js`
(`VYRONEX_ACTIVITIES`) — label, icon, instructions, which engine it uses,
and which metrics/measurements it reports. Nothing else in the app
hard-codes an activity name; `gym.html`/`athletics.html` render their
cards by looping the registry, and `activity.html` reads the chosen
activity's config to decide what UI, engine and result fields to show.
Adding a new workout or event means adding one object to the registry
(and, if it needs bespoke detection logic, one small method on
`HeuristicAnalyzer`) — not rebuilding pages.

## Two analysis engines

- **`backend-squat`** (Squat only) — real Python pipeline via
  `/api/session/*` and `/api/frame`, unchanged from the original
  prototype.
- **`heuristic`** (Push Up, Walking Lunges, Long Jump, Shot Put, Javelin
  Throw) — `frontend/assets/heuristic-engine.js`, a client-side
  pose-geometry engine. It computes real joint angles and velocities
  from tracked landmarks (elbow/knee angles for rep counting, wrist
  velocity for release speed and angle, hip trajectory for jump
  distance/flight time, etc.) and turns them into the same
  score/risk/checklist/feedback shape the backend uses — so the UI can
  render any activity generically. It is explicitly labeled a
  **heuristic prototype engine** in the UI, not a certified or
  clinically validated measurement tool, same spirit as the backend's
  own disclaimer. Distance and speed for athletics events are reported
  in relative/normalized units (body-lengths, rel. units/s) since there
  is no calibrated real-world reference scale from a single camera.

## How to run it

```
cd backend
python3 -m venv venv && source venv/bin/activate   # optional
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```
Then open `frontend/index.html` in a browser (or `cd frontend && python3
-m http.server 8080` if your browser restricts camera access on
`file://`). The backend is only required for **Squat** — every other
activity works with the backend offline (Live Feed still needs a
webcam; Video Upload needs neither backend nor camera).

## Project structure
```
VYRONEX/
├── README.md
├── frontend/
│   ├── index.html        ← Home (GYM | ATHLETICS)
│   ├── gym.html            ← Choose Your Workout
│   ├── athletics.html      ← Choose Your Event
│   ├── activity.html       ← gated dynamic flow for one activity (not in main nav)
│   └── assets/
│       ├── style.css          ← shared VYRONEX design system
│       ├── logo.svg           ← VYRONEX logo / favicon
│       ├── activities.js      ← activity registry (single source of truth)
│       ├── heuristic-engine.js← client-side pose-geometry engine
│       └── results-render.js  ← shared Performance Results renderers
└── backend/                 ← unchanged Python analysis pipeline (Squat)
```

## Honest limitations

- Only **Squat** is backed by the validated Python pipeline. Every other
  activity's numbers come from a client-side heuristic prototype — real
  measurements from real tracked landmarks, but not clinically validated
  and clearly labeled as such in the UI.
- Athletics distance/speed figures are relative, uncalibrated estimates
  (no reference object or multi-camera setup to convert to real units).
- Single-camera 2D pose tracking loses depth information, so any motion
  toward/away from the camera is undercounted.
- The backend URL is fixed to `http://localhost:8000` in the frontend
  code (no in-UI connection control) — edit `backendUrl()` in
  `activity.html` if you need to point it elsewhere.
- Movement-risk screening only, not a medical, coaching-certification,
  or athletics-officiating tool.
