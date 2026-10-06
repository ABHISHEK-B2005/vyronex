# VYRONEX — Frontend Portal

Plain HTML5, CSS, and vanilla JavaScript (no framework) frontend for the VYRONEX Biomechanical Screening platform.

---

## 🎨 Color System & Design Rules

- **White Dominant Theme**: Crisp white cards (`#FFFFFF`) on light paper canvas (`#FAF9F6`) with soft subtle shadows.
- **Color Variables**:
  - `--red`: `#D7263D` (Primary buttons, active tabs, risk cues)
  - `--red-deep`: `#9B1226` (Solid High-Risk badges)
  - `--yellow`: `#FFC20E` (Highlights, title underlines, top-bar stripe, `#FFE27A` "Fix" badges with `#4A3500` dark text, selected role tint `#FFF8DC`)
  - `--blue`: `#2B8BE0` (Headings, table headers, secondary buttons, Download PDF button)
  - `--blue-deep`: `#1B6FBF` (Logo and branding)
  - `--green`: `#2A9D68` (Correct reps and accuracy stats ONLY)
- **Typography**: Google Fonts `Bricolage Grotesque` (Headings) and `Figtree` (Body) with system-ui fallbacks.
- **Labels with Words**: Never uses color alone: every state includes textual labels ("Good", "Fix", "Risky").

---

## 📂 Page & Directory Map

```
frontend/
├── index.html          # Landing page with pathway selection (Gym vs Athlete)
├── signup.html         # Signup with role cards, biometrics, opt-in consent toggle, privacy note
├── login.html          # Clean sign-in page with session storage token handling
├── gym.html            # Role-protected dashboard (Squat, Push-up, Lunge, Plank) + latest session
├── athlete.html        # Role-protected dashboard (Long jump, Shot put, Javelin) + latest session
├── activity.html       # Camera Live screening + Upload video analysis tabs
├── report.html         # Detailed report: Doughnut, Body Area, Fatigue line chart, rep table, PDF export
├── history.html        # Longitudinal trend chart, recurring weakness card, filtered session list
├── coach.html          # Biomechanical AI chat assistant with medical disclaimer & corrective library
├── profile.html        # Edit biometrics, toggle training consent, delete movement data
└── assets/
    ├── css/
    │   └── style.css   # Full design system tokens and responsive components
    └── js/
        ├── auth.js         # JWT token in sessionStorage, apiFetch, escapeHtml
        ├── guard.js        # Route protection, header navigation, offline banner
        ├── activities.js   # Single source of truth activity registry
        ├── live.js         # Camera stream, MediaPipe Pose skeleton, WebSocket cues
        ├── upload.js       # Multipart video upload & status polling
        ├── voice.js        # Web Speech API real-time spoken cues
        ├── charts.js       # Chart.js visualizers (Doughnut, Bar, Fatigue line, Trends)
        ├── report.js       # Report loading, rep table rendering, PDF export
        ├── history.js      # History fetching and analytics filters
        └── coach.js        # AI Coach chat client with typing indicators
```

---

## 🚀 How to Run the Frontend

1. **Start the FastAPI Backend** (in a separate terminal):
   ```bash
   cd backend
   uvicorn app:app --reload --port 8000
   ```

2. **Serve the Frontend**:
   ```bash
   cd frontend
   python -m http.server 5500
   ```

3. **Open in Browser**:
   Navigate to: `http://localhost:5500/index.html` (or `http://localhost:5500/login.html`)

---

## 📡 Backend API Endpoints Expected

- `GET  /api/health` — Backend liveness check
- `POST /api/auth/signup` — Account registration (`{name, email, password, role, age, height_cm, weight_kg, data_consent}`)
- `POST /api/auth/login` — Authentication (`{email, password}`) -> returns `{access_token, role, user}`
- `GET  /api/auth/me` — Authenticated user profile verification
- `PUT  /api/auth/profile` — Update biometrics and consent settings
- `DELETE /api/auth/data` — Delete movement data and withdraw training consent
- `WS   /ws/live?token=...&activity=...` — Real-time MediaPipe landmarks stream & live coaching cues
- `POST /api/upload/analyze` — Multipart video file upload
- `GET  /api/upload/status/{job_id}` — Video processing status polling
- `GET  /api/reports/{session_id}` — Biomechanical report details & metrics
- `GET  /api/reports/latest` — Most recent session report
- `GET  /api/reports/{session_id}/pdf` — Downloadable PDF report export
- `GET  /api/history?activity=&from_date=&to_date=` — Filtered session history & trend data
- `POST /api/coach/chat` — Biomechanical chat query (`{message}`) -> returns `{reply, disclaimer}`
- `GET  /api/coach/recommendations/{session_id}` — Post-workout corrective exercises
- `POST /api/feedback/rep/{rep_id}` — Rep rating feedback validation
