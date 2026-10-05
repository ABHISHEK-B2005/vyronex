# VYRONEX — AI Fitness & Athletics Performance Platform

Two parts:

- **frontend/** — a static site with signup/login, role-specific
  dashboards, and registry-backed activity selection.
- **backend/** — a Python FastAPI server with authentication,
  role-protected activity catalogs, and the original squat analysis
  pipeline.

## Mandatory flow

**Role-specific dashboard → Activity**

The gym and athlete dashboards are role-protected. Selecting a card opens
`activity.html?activity=<id>`; if the query is missing or invalid,
`activity.html` shows a "Select an Activity First" screen with links back
to the dashboards. Squat includes a live camera session: MediaPipe Pose
tracks the browser camera, then streams landmarks to `/ws/live` for
rep counting, form-risk feedback, and coaching cues. Ending the session
or disconnecting saves its session and rep records and creates a report.

There is no visible backend-URL/connection control in the UI. The
frontend API client uses `http://localhost:8000`.

## Pages

| Page | File | Purpose |
|---|---|---|
| Home | `index.html` | Welcome + GYM / ATHLETICS category picker |
| Gym | `gym.html` | Role-protected gym dashboard: Squat, Push-up, Walking Lunge, Plank |
| Athlete | `athlete.html` | Role-protected athlete dashboard: Long Jump, Shot Put, Javelin Throw |
| *(no nav link)* | `activity.html?activity=<id>` | Selected activity instructions and metrics, reached via a dashboard card |

## Scalable architecture

Every workout/event is **one entry** in `frontend/assets/activities.js`
(`VYRONEX_ACTIVITIES`) — label, category, instructions, which engine it
uses, and which metrics/measurements it reports. Gym and athlete
dashboards filter this registry using the IDs returned by their
role-protected `/api/gym/activities` and `/api/athlete/activities`
endpoints. `activity.html` reads the chosen activity's config and shows
the "Select an Activity First" screen when the query is missing or invalid.
Adding a new workout or event means adding one object to the registry
(and, if it needs bespoke detection logic, corresponding engine support).

## Activity engines

- **`backend-squat`** (Squat only) — real Python pipeline via the
  authenticated `/ws/live` WebSocket.
- **`heuristic`** — registry metadata for other activities; their
  current activity page presents instructions and metrics.

## How to run it

```
cd backend
python3 -m venv venv && source venv/bin/activate   # optional
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```
Then open `frontend/login.html` in a browser (or `cd frontend && python3
-m http.server 8080` if your browser restricts camera access on
`file://`). The backend is required for sign-in and role-specific activity
lists.

## Project structure
```
VYRONEX/
├── README.md
├── frontend/
│   ├── index.html        ← Home (GYM | ATHLETICS)
│   ├── gym.html            ← Gym dashboard
│   ├── athlete.html        ← Athlete dashboard
│   ├── activity.html       ← selected activity and invalid-selection guard
│   └── assets/
│       ├── style.css       ← shared VYRONEX design system
│       ├── auth.js         ← authentication and API client
│       ├── activities.js   ← activity registry (single source of truth)
│       ├── dashboard.js    ← role-specific dashboard rendering
│       └── live.js         ← selected activity page and live camera client
└── backend/                 ← Python API and analysis pipeline
    ├── routes/ws.py         ← authenticated live analysis WebSocket
    └── services/session_report_service.py
                             ← completed-session report summaries
```

## Honest limitations

- Only **Squat** is backed by the Python analysis pipeline. Other listed
  activities currently provide their instructions and metrics as
  registry metadata; their analysis flows are not implemented here.
- The backend URL is fixed to `http://localhost:8000` in
  `frontend/assets/auth.js` and `frontend/assets/live.js` (no in-UI
  connection control).
- Movement-risk screening only, not a medical, coaching-certification,
  or athletics-officiating tool.
