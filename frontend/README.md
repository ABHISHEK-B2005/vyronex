# VYRONEX Frontend

Plain HTML, CSS, and JavaScript frontend for home-workout users and athletes. The
light theme uses Bricolage Grotesque and Figtree with system font fallbacks.

## Run locally

1. Start the backend in one terminal:

   ```powershell
   cd backend
   uvicorn app:app --reload --port 8000
   ```

2. Serve these files in another terminal:

   ```powershell
   cd frontend
   python -m http.server 5500
   ```

3. Open `http://localhost:5500/index.html`. Sign in or create an account.

The pages check `/api/health` and show a backend-offline banner when the API cannot
be reached. API requests use the JWT in `sessionStorage` as a Bearer token.

## Pages and modules

- `gym.html` and `athlete.html`: role-protected activity dashboards.
- `activity.html`: camera screening and video-upload tabs. Current analysis supports
  squats only; other activity uploads are rejected rather than analyzed as squats.
- `report.html`, `history.html`, `coach.html`, and `profile.html`: reports, session
  history, coaching, and profile preferences.
- `assets/js/activities.js`: activity registry.
- `assets/js/live.js`: camera stream, MediaPipe Pose landmarks, and frame analysis.
- `assets/js/debug.js`: expandable camera and frame diagnostics.

## API endpoints used

- `GET /api/health`
- `POST /api/auth/signup`, `POST /api/auth/login`, `GET /api/auth/me`
- `PUT /api/auth/profile`, `DELETE /api/auth/data`
- `POST /api/session/start`, `POST /api/frame`, `POST /api/session/{session_id}/end`
- `POST /api/upload/analyze`, `GET /api/upload/status/{job_id}`
- `GET /api/reports/latest`, `GET /api/reports/{session_id}`,
  `GET /api/reports/{session_id}/pdf`
- `GET /api/history`
- `POST /api/coach/chat`

## Troubleshooting

- Camera access requires `http://localhost` or HTTPS. Allow camera permission when
  prompted.
- If the backend is unreachable, start it on port 8000 and allow the frontend origin
  (`http://localhost:5500`) in FastAPI CORS.
- Use Python 3.11 or 3.12 for the broadest MediaPipe compatibility. The backend
  supports both MediaPipe's legacy Pose API and the current Tasks Pose Landmarker;
  its first upload downloads and caches the official 5.8 MB pose model.
- Stand 2–3 m from the camera in good light and keep your hips, knees, and ankles
  visible.
- If pose assets fail to load, check the internet connection to `cdn.jsdelivr.net`.
- Use the Camera Check panel on the activity page to inspect pose detection, visible
  landmarks, frame count, HTTP status, and the latest server response.
