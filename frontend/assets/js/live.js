/**
 * Vyronex Live Camera & MediaPipe Pose WebSocket Client
 */

import { getToken, API_BASE_URL } from "./auth.js";
import { speakCue, cancelVoice } from "./voice.js";

let videoEl = null;
let canvasEl = null;
let ctx = null;
let stream = null;
let ws = null;
let isRunning = false;
let currentSessionId = null;
let lastFrameSent = 0;
const TARGET_FPS = 15;
const FRAME_INTERVAL_MS = 1000 / TARGET_FPS;

const POSE_CONNECTIONS = [
  ["left_shoulder", "right_shoulder"],
  ["left_shoulder", "left_elbow"],
  ["left_elbow", "left_wrist"],
  ["right_shoulder", "right_elbow"],
  ["right_elbow", "right_wrist"],
  ["left_shoulder", "left_hip"],
  ["right_shoulder", "right_hip"],
  ["left_hip", "right_hip"],
  ["left_hip", "left_knee"],
  ["left_knee", "left_ankle"],
  ["right_hip", "right_knee"],
  ["right_knee", "right_ankle"],
];

const MEDIAPIPE_INDEX_MAP = {
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

export async function startLiveSession(activityId, onUpdate) {
  videoEl = document.getElementById("live-video");
  canvasEl = document.getElementById("live-canvas");
  if (!videoEl || !canvasEl) return;
  ctx = canvasEl.getContext("2d");

  const token = getToken();
  if (!token) {
    window.location.assign("login.html");
    return;
  }

  // 1. Request Camera
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
      audio: false,
    });
    videoEl.srcObject = stream;
    await videoEl.play();
  } catch (err) {
    onUpdate({ error: "Camera access was denied. Please allow camera permissions to start movement screening." });
    return;
  }

  isRunning = true;

  // 2. Connect WebSocket
  const wsProto = window.location.protocol === "https:" ? "wss:" : "ws:";
  const wsHost = API_BASE_URL.replace(/^https?:\/\//, "");
  const wsUrl = `${wsProto}//${wsHost}/ws/live?token=${encodeURIComponent(token)}&activity=${encodeURIComponent(activityId)}`;

  connectWebSocket(wsUrl, onUpdate);

  // 3. Start Camera Frame & MediaPipe Loop
  requestAnimationFrame(processVideoFrame);
}

function connectWebSocket(wsUrl, onUpdate) {
  try {
    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      onUpdate({ status: "connected", message: "Connected. Position yourself in camera frame." });
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.session_id) currentSessionId = data.session_id;

        if (data.type === "rep_completed" || data.completed_rep) {
          const rep = data.completed_rep || data;
          const cue = rep.feedback?.recommendation || rep.feedback || "Good rep!";
          speakCue(cue);
        } else if (data.message && data.message !== "Keep moving") {
          speakCue(data.message);
        }

        onUpdate({
          rep_count: data.rep_count || 0,
          phase: data.phase || "ready",
          risk_level: data.risk_level || "LOW",
          cue: data.message || "Ready. Begin movement.",
          landmarks_ok: data.landmarks_ok !== false,
          session_id: currentSessionId,
        });
      } catch (e) {
        console.error("WS Parse error:", e);
      }
    };

    ws.onerror = () => {
      onUpdate({ error: "Live connection interrupted. Reconnecting..." });
    };

    ws.onclose = () => {
      if (isRunning) {
        setTimeout(() => {
          if (isRunning) connectWebSocket(wsUrl, onUpdate);
        }, 2000);
      }
    };
  } catch (err) {
    onUpdate({ error: "Failed to establish WebSocket connection." });
  }
}

function processVideoFrame(timestamp) {
  if (!isRunning || !videoEl || !canvasEl) return;

  if (canvasEl.width !== videoEl.videoWidth && videoEl.videoWidth > 0) {
    canvasEl.width = videoEl.videoWidth;
    canvasEl.height = videoEl.videoHeight;
  }

  ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);

  // Throttle frame transmission
  if (timestamp - lastFrameSent >= FRAME_INTERVAL_MS && ws && ws.readyState === WebSocket.OPEN) {
    lastFrameSent = timestamp;
    extractAndSendLandmarks();
  }

  requestAnimationFrame(processVideoFrame);
}

function extractAndSendLandmarks() {
  // Use client-side MediaPipe if loaded or send video timestamp signal
  if (window.currentMediaPipeLandmarks) {
    const rawLandmarks = window.currentMediaPipeLandmarks;
    const bodyLandmarks = {};
    for (const [idxStr, name] of Object.entries(MEDIAPIPE_INDEX_MAP)) {
      const lm = rawLandmarks[parseInt(idxStr, 10)];
      if (lm) {
        bodyLandmarks[name] = {
          x: lm.x * canvasEl.width,
          y: lm.y * canvasEl.height,
          visibility: lm.visibility || 0.9,
        };
      }
    }

    drawSkeleton(bodyLandmarks);

    if (Object.keys(bodyLandmarks).length >= 6) {
      ws.send(JSON.stringify({
        landmarks: bodyLandmarks,
        timestamp_sec: videoEl.currentTime,
      }));
    }
  }
}

function drawSkeleton(landmarks) {
  ctx.strokeStyle = "#2B8BE0";
  ctx.lineWidth = 3;
  ctx.fillStyle = "#FFC20E";

  // Draw Lines
  for (const [p1, p2] of POSE_CONNECTIONS) {
    if (landmarks[p1] && landmarks[p2]) {
      ctx.beginPath();
      ctx.moveTo(landmarks[p1].x, landmarks[p1].y);
      ctx.lineTo(landmarks[p2].x, landmarks[p2].y);
      ctx.stroke();
    }
  }

  // Draw Joint Dots
  for (const pt of Object.values(landmarks)) {
    ctx.beginPath();
    ctx.arc(pt.x, pt.y, 5, 0, 2 * Math.PI);
    ctx.fill();
  }
}

export function stopLiveSession() {
  isRunning = false;
  cancelVoice();
  if (ws) {
    ws.close();
    ws = null;
  }
  if (stream) {
    stream.getTracks().forEach(t => t.stop());
    stream = null;
  }
  return currentSessionId;
}
