import { VYRONEX_ACTIVITIES } from "./activities.js";
import { getToken, logout, requireAuth } from "./auth.js";

const page = document.querySelector("[data-activity-page]");
const query = new URLSearchParams(window.location.search);
const activityId = query.get("activity");
const activity = VYRONEX_ACTIVITIES.find((item) => item.id === activityId);

if (page) {
  initializeActivityPage();
}

async function initializeActivityPage() {
  const user = await requireAuth(null, page);
  if (!user) return;

  page.hidden = false;
  if (activity && activity.category !== user.role) {
    window.location.replace(user.role === "gym" ? "gym.html" : "athlete.html");
    return;
  }

  if (!activity) {
    page.querySelector("[data-select-activity]").hidden = false;
    return;
  }

  const details = page.querySelector("[data-selected-activity]");
  details.hidden = false;
  details.querySelector("[data-activity-category]").textContent =
    activity.category === "gym" ? "Gym workout" : "Athletics event";
  details.querySelector("[data-activity-label]").textContent = activity.label;
  details.querySelector("[data-activity-instructions]").textContent = activity.instructions;
  details.querySelector("[data-dashboard-link]").href =
    activity.category === "gym" ? "gym.html" : "athlete.html";

  const metrics = details.querySelector("[data-activity-metrics]");
  for (const metric of activity.metrics) {
    const item = document.createElement("li");
    item.textContent = metric;
    metrics.append(item);
  }
  if (activity.id === "squat") {
    const livePanel = details.querySelector("[data-live-panel]");
    livePanel.hidden = false;
    initializeLivePanel(livePanel);
  }
}

function initializeLivePanel(panel) {
  const video = panel.querySelector("[data-camera]");
  const canvas = panel.querySelector("[data-skeleton]");
  const context = canvas.getContext("2d");
  const cameraHint = panel.querySelector("[data-camera-hint]");
  const connectionStatus = panel.querySelector("[data-live-connection]");
  const repCountElement = panel.querySelector("[data-rep-count]");
  const phaseElement = panel.querySelector("[data-phase]");
  const riskCard = panel.querySelector("[data-risk]");
  const riskLabel = panel.querySelector("[data-risk-label]");
  const riskMeter = panel.querySelector("[data-risk-meter]");
  const cueElement = panel.querySelector("[data-live-cue]");
  const errorElement = panel.querySelector("[data-live-error]");
  const summaryElement = panel.querySelector("[data-session-summary]");
  const voiceToggle = panel.querySelector("[data-voice-cues]");
  const startButton = panel.querySelector("[data-start-live]");
  const endButton = panel.querySelector("[data-end-live]");
  const landmarkNames = {
    0: "nose",
    11: "left_shoulder",
    12: "right_shoulder",
    13: "left_elbow",
    14: "right_elbow",
    15: "left_wrist",
    16: "right_wrist",
    23: "left_hip",
    24: "right_hip",
    25: "left_knee",
    26: "right_knee",
    27: "left_ankle",
    28: "right_ankle",
  };
  const poseConnections = [
    [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
    [11, 23], [12, 24], [23, 24], [23, 25], [25, 27],
    [24, 26], [26, 28], [0, 11], [0, 12],
  ];
  let mediaStream = null;
  let pose = null;
  let socket = null;
  let stopped = true;
  let sending = false;
  let lastFrameAt = 0;
  let lastCue = cueElement.textContent;
  let reconnectTimer = null;
  let reconnectDelay = 500;
  let displayedRepOffset = 0;

  function setConnectionStatus(message) {
    connectionStatus.textContent = message;
  }

  function setCue(message) {
    if (!message) return;
    cueElement.textContent = message;
    if (message !== lastCue && voiceToggle.checked && "speechSynthesis" in window) {
      window.speechSynthesis.cancel();
      window.speechSynthesis.speak(new SpeechSynthesisUtterance(message));
    }
    lastCue = message;
  }

  function setError(message) {
    errorElement.textContent = message;
    errorElement.hidden = !message;
  }

  function drawPose(results) {
    const width = video.videoWidth;
    const height = video.videoHeight;
    if (!width || !height) return;
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    context.clearRect(0, 0, width, height);
    const landmarks = results.poseLandmarks;
    if (!landmarks) {
      cameraHint.textContent = "Step back so your full body is visible";
      cameraHint.hidden = false;
      sendFrame(null);
      return;
    }

    cameraHint.hidden = true;
    context.strokeStyle = "#c8f169";
    context.lineWidth = Math.max(3, width / 220);
    context.lineCap = "round";
    for (const [from, to] of poseConnections) {
      const start = landmarks[from];
      const end = landmarks[to];
      if (start.visibility < 0.5 || end.visibility < 0.5) continue;
      context.beginPath();
      context.moveTo(start.x * width, start.y * height);
      context.lineTo(end.x * width, end.y * height);
      context.stroke();
    }
    context.fillStyle = "#ffffff";
    for (const landmark of landmarks) {
      if (landmark.visibility < 0.5) continue;
      context.beginPath();
      context.arc(landmark.x * width, landmark.y * height, Math.max(3, width / 160), 0, Math.PI * 2);
      context.fill();
    }

    const namedLandmarks = {};
    for (const [index, name] of Object.entries(landmarkNames)) {
      const landmark = landmarks[Number(index)];
      namedLandmarks[name] = {
        x: landmark.x * width,
        y: landmark.y * height,
        visibility: landmark.visibility ?? 1,
      };
    }
    sendFrame(namedLandmarks);
  }

  function sendFrame(landmarks) {
    if (socket?.readyState !== WebSocket.OPEN) return;
    socket.send(JSON.stringify({
      landmarks,
      t: performance.now() / 1000,
    }));
  }

  function handleServerMessage(event) {
    let message;
    try {
      message = JSON.parse(event.data);
    } catch {
      setError("The live service sent an invalid response.");
      return;
    }
    if (message.type === "ended") {
      const summary = message.summary;
      summaryElement.textContent =
        `Session saved: ${summary.total_reps} reps in ${summary.duration_seconds}s.`;
      summaryElement.hidden = false;
      setConnectionStatus("Session saved");
      return;
    }
    if (message.type === "ready") {
      displayedRepOffset = message.rep_count || displayedRepOffset;
      repCountElement.textContent = String(displayedRepOffset);
      setConnectionStatus("Connected");
      reconnectDelay = 500;
      return;
    }
    if (message.type !== "frame") return;

    repCountElement.textContent = String(message.rep_count);
    phaseElement.textContent = message.phase;
    riskCard.dataset.risk = message.risk_level;
    riskLabel.textContent = `${message.risk_level} · ${Math.round(message.risk_score)}`;
    riskMeter.value = message.risk_score;
    const completedCue = message.completed_rep?.feedback;
    setCue(
      completedCue?.warning_tag
        ? `${completedCue.warning_tag}: ${completedCue.recommendation}`
        : message.cue,
    );
    if (message.landmarks_ok) {
      cameraHint.hidden = true;
    } else {
      cameraHint.textContent = "Step back so your full body is visible";
      cameraHint.hidden = false;
    }
  }

  function connectSocket() {
    if (stopped) return;
    const scheme = window.location.protocol === "https:" ? "wss:" : "ws:";
    const query = new URLSearchParams({
      token: getToken(),
      activity: "squat",
      rep_offset: String(displayedRepOffset),
    });
    socket = new WebSocket(`${scheme}//localhost:8000/ws/live?${query}`);
    setConnectionStatus("Connecting…");
    socket.addEventListener("open", () => {
      if (!stopped) setConnectionStatus("Connected");
    });
    socket.addEventListener("message", handleServerMessage);
    socket.addEventListener("error", () => {
      setConnectionStatus("Connection interrupted");
    });
    socket.addEventListener("close", (event) => {
      if (stopped) return;
      displayedRepOffset = Number(repCountElement.textContent) || displayedRepOffset;
      if (event.code === 4401) {
        logout();
        return;
      }
      if (event.code === 4403 || event.code === 4400) {
        stopped = true;
        setError(event.reason || "This account cannot start the live session.");
        setConnectionStatus("Session unavailable");
        stopMedia();
        startButton.disabled = false;
        endButton.disabled = true;
        return;
      }
      setConnectionStatus("Connection dropped — reconnecting…");
      reconnectTimer = window.setTimeout(connectSocket, reconnectDelay);
      reconnectDelay = Math.min(reconnectDelay * 2, 8000);
    });
  }

  async function startLiveSession() {
    startButton.disabled = true;
    setError("");
    summaryElement.hidden = true;
    try {
      if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error("Camera access is unavailable in this browser. Open the page on localhost or HTTPS.");
      }
      if (typeof window.Pose !== "function") {
        throw new Error("MediaPipe Pose could not be loaded. Check your internet connection and reload.");
      }
      mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
      });
      video.srcObject = mediaStream;
      await video.play();
      pose = new window.Pose({
        locateFile: (file) =>
          `https://cdn.jsdelivr.net/npm/@mediapipe/pose@0.5.1675469404/${file}`,
      });
      pose.setOptions({
        modelComplexity: 1,
        smoothLandmarks: true,
        enableSegmentation: false,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });
      pose.onResults(drawPose);
      stopped = false;
      endButton.disabled = false;
      setConnectionStatus("Connecting…");
      connectSocket();
      runPoseLoop();
    } catch (error) {
      stopMedia();
      startButton.disabled = false;
      setConnectionStatus("Camera unavailable");
      setError(
        error.name === "NotAllowedError"
          ? "Camera permission was denied. Allow camera access in your browser settings and try again."
          : error.message,
      );
    }
  }

  async function runPoseLoop() {
    if (stopped) return;
    const frame = async (now) => {
      if (stopped) return;
      if (!sending && now - lastFrameAt >= 1000 / 15 && video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
        sending = true;
        lastFrameAt = now;
        try {
          await pose.send({ image: video });
        } catch (error) {
          setError(`Pose tracking failed: ${error.message}`);
        } finally {
          sending = false;
        }
      }
      if (!stopped) window.requestAnimationFrame(frame);
    };
    window.requestAnimationFrame(frame);
  }

  function stopMedia() {
    if (mediaStream) {
      for (const track of mediaStream.getTracks()) track.stop();
      mediaStream = null;
    }
    if (video) video.srcObject = null;
    if (pose) {
      pose.close();
      pose = null;
    }
    context.clearRect(0, 0, canvas.width, canvas.height);
  }

  startButton.addEventListener("click", startLiveSession);
  endButton.addEventListener("click", () => {
    if (stopped) return;
    stopped = true;
    window.clearTimeout(reconnectTimer);
    endButton.disabled = true;
    startButton.disabled = false;
    setConnectionStatus("Saving session…");
    if (socket?.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({ type: "end" }));
      window.setTimeout(() => {
        if (socket?.readyState === WebSocket.OPEN) socket.close(1000, "Session ended");
      }, 4000);
    } else if (socket) {
      socket.close(1000, "Session ended");
    }
    stopMedia();
    if ("speechSynthesis" in window) window.speechSynthesis.cancel();
  });
}
