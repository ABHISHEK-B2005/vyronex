/**
 * Browser camera and MediaPipe pose client.
 */

import { apiFetch } from "./auth.js";
import { speakCue, cancelVoice } from "./voice.js";
import { updateDebugStatus } from "./debug.js";
import { POSE_VERSION, loadPoseScript } from "./pose-loader.js";
import {
  cameraError,
  drawAlignmentGuides,
  drawSkeleton,
  getAlignmentCue,
  renderLiveAlignment,
} from "./live-ui.js";

const FRAME_INTERVAL_MS = 75;
const LANDMARK_INDEX = { nose: 0, left_shoulder: 11, right_shoulder: 12, left_elbow: 13, right_elbow: 14, left_wrist: 15, right_wrist: 16, left_hip: 23, right_hip: 24, left_knee: 25, right_knee: 26, left_ankle: 27, right_ankle: 28, left_foot_index: 31, right_foot_index: 32 };
const LOWER_BODY = ["left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"];
let videoEl, canvasEl, ctx, stream, poseInstance;
let isRunning = false, isStarting = false, isProcessingPose = false, isSendingFrame = false;
let currentSessionId = null, currentActivity = "squat";
let lastPoseTime = 0, nextFrameAt = 0, framesSentCount = 0;
let lastSpokenCue = "", hasRestartedSession = false;
let pendingPose = null, pendingFrame = null;
let currentAlignment = null, currentView = "unknown", sessionStartedAt = 0;

export async function startLiveSession(activityId, onUpdate) {
  if (isRunning || isStarting) return;
  isStarting = true;
  currentActivity = activityId || "squat";
  videoEl = document.getElementById("live-video");
  canvasEl = document.getElementById("live-canvas");

  try {
    if (!videoEl || !canvasEl) throw new Error("Camera view is unavailable.");
    if (currentActivity !== "squat") {
      throw new Error("Live camera rep analysis currently supports squats. You can upload a video for other activities.");
    }
    const { hostname, protocol } = window.location;
    if (protocol !== "https:" && hostname !== "localhost" && hostname !== "127.0.0.1") {
      throw new Error("Open this page through a local server (http://localhost) so the camera is allowed.");
    }
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error("This browser cannot access a camera. Open the page through localhost or HTTPS.");
    }
    if (currentSessionId) await endSession();

    const startRes = await apiFetch("/api/session/start", {
      method: "POST",
      body: JSON.stringify({ exercise: currentActivity.toUpperCase(), risk_mode: "rule_based" }),
    });
    const sessionData = await startRes.json();
    if (!sessionData.session_id) throw new Error("The backend did not return a session ID.");
    currentSessionId = sessionData.session_id;
    sessionStartedAt = performance.now();
    currentAlignment = null;
    currentView = "unknown";
    updateDebugStatus({
      lastHttpStatus: startRes.status,
      lastResponseMsg: "Session started",
      framesSent: 0,
      poseDetected: "No",
    });

    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: 640, height: 480 },
        audio: false,
      });
      videoEl.srcObject = stream;
      await videoEl.play();
      updateDebugStatus({ camera: "On" });
    } catch (error) {
      throw cameraError(error);
    }

    ctx = canvasEl.getContext("2d");
    if (!ctx) throw new Error("This browser cannot draw the camera preview.");
    framesSentCount = 0;
    hasRestartedSession = false;
    lastSpokenCue = "";
    lastPoseTime = 0;
    nextFrameAt = 0;
    isRunning = true;
    await initMediaPipe(onUpdate);
    onUpdate({ started: true });
  } catch (error) {
    let message = error.message;
    await releaseCamera();
    try {
      await closePose();
      await endSession();
    } catch (endError) {
      message = `${message} The session could not be finalized: ${endError.message}`;
    }
    isRunning = false;
    onUpdate({ error: message });
    updateDebugStatus({ camera: "Off", lastResponseMsg: message });
  } finally {
    isStarting = false;
  }
}

async function initMediaPipe(onUpdate) {
  try {
    await loadPoseScript();
    if (typeof window.Pose !== "function") throw new Error("MediaPipe Pose did not initialize.");
    poseInstance = new window.Pose({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose@${POSE_VERSION}/${file}`,
    });
    poseInstance.setOptions({ modelComplexity: 1, smoothLandmarks: true, minDetectionConfidence: 0.5, minTrackingConfidence: 0.5 });
    poseInstance.onResults((results) => handlePoseResults(results, onUpdate));
    startCameraProcessingLoop(onUpdate);
  } catch (error) {
    throw new Error(`Could not load MediaPipe Pose. Check your internet connection and retry. (${error.message})`);
  }
}

function startCameraProcessingLoop(onUpdate) {
  const step = async (now) => {
    if (!isRunning || !videoEl || !poseInstance) return;
    if (videoEl.readyState >= 2 && !isProcessingPose && now - lastPoseTime >= FRAME_INTERVAL_MS) {
      isProcessingPose = true;
      lastPoseTime = now;
      pendingPose = (async () => {
        try { await poseInstance.send({ image: videoEl }); }
        catch (error) {
          onUpdate({ error: `Pose analysis stopped: ${error.message}` });
          updateDebugStatus({ lastResponseMsg: "Pose inference failed" });
          isRunning = false;
          await releaseCamera();
          try { await endSession(); } catch (endError) {
            onUpdate({ error: `Pose analysis stopped and the session could not be finalized: ${endError.message}` });
          }
        }
      })();
      try { await pendingPose; }
      finally {
        pendingPose = null;
        isProcessingPose = false;
      }
    }
    if (isRunning) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

function handlePoseResults(results, onUpdate) {
  if (!isRunning || !canvasEl || !videoEl) return;
  if (videoEl.videoWidth > 0 && canvasEl.width !== videoEl.videoWidth) [canvasEl.width, canvasEl.height] = [videoEl.videoWidth, videoEl.videoHeight];
  ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
  if (videoEl.readyState >= 2) ctx.drawImage(videoEl, 0, 0, canvasEl.width, canvasEl.height);

  const landmarks = results.poseLandmarks;
  if (!landmarks?.length) {
    updateDebugStatus({ poseDetected: "No", visibleLandmarks: 0 });
    onUpdate({ landmarks_ok: false, cue: "Step back so your whole body is visible" });
    return;
  }

  drawSkeleton(ctx, canvasEl, landmarks);
  drawAlignmentGuides(ctx, canvasEl, landmarks, currentAlignment, currentView);
  const visibleCount = landmarks.filter((point) => (point.visibility ?? 0) >= 0.5).length;
  updateDebugStatus({ poseDetected: "Yes", visibleLandmarks: visibleCount });

  const payload = {};
  const worldLandmarks = results.poseWorldLandmarks;
  for (const [name, index] of Object.entries(LANDMARK_INDEX)) {
    const point = landmarks[index];
    if (!point) continue;
    const world = worldLandmarks?.[index];
    payload[name] = world ? {
      x: world.x, y: world.y, z: world.z ?? 0,
      image_x: point.x, image_y: point.y,
      visibility: point.visibility ?? 1,
    } : {
      x: Math.min(1, Math.max(0, point.x ?? 0)),
      y: Math.min(1, Math.max(0, point.y ?? 0)),
      image_x: point.x, image_y: point.y,
      visibility: point.visibility ?? 1,
    };
  }
  if (LOWER_BODY.some((name) => payload[name].visibility < 0.5)) {
    const cue = "Move back so your hips, knees and ankles are visible";
    onUpdate({ landmarks_ok: false, cue, guidance: cue });
    return;
  }
  if (!isSendingFrame && Date.now() >= nextFrameAt) {
    pendingFrame = sendFrame(payload, onUpdate).finally(() => { pendingFrame = null; });
  }
}

async function sendFrame(landmarks, onUpdate) {
  if (!currentSessionId || isSendingFrame) return;
  isSendingFrame = true;
  framesSentCount += 1;
  updateDebugStatus({ framesSent: framesSentCount });
  try {
    const response = await apiFetch("/api/frame", {
      method: "POST",
      body: JSON.stringify({
        session_id: currentSessionId,
        landmarks,
        t: (performance.now() - sessionStartedAt) / 1000,
      }),
    });
    const data = await response.json();
    nextFrameAt = 0;
    updateDebugStatus({ framesSent: framesSentCount, lastHttpStatus: response.status, lastResponseMsg: data.message || "Frame analyzed" });
    if (data.landmarks_ok === false) {
      onUpdate({ landmarks_ok: false, cue: data.message || "Adjust your position" });
      return;
    }

    currentAlignment = data.alignment || null;
    currentView = data.view || "unknown";
    renderLiveAlignment(currentAlignment, currentView);
    const completedRep = data.completed_rep;
    const alignmentCue = getAlignmentCue(currentAlignment);
    const cue = alignmentCue || completedRep?.feedback?.recommendation || data.message || "";
    if (completedRep) {
      const riskLevel = (completedRep.risk?.risk_level || "LOW").toUpperCase();
      onUpdate({ risk_level: riskLevel });
      if (cue && cue !== lastSpokenCue) {
        lastSpokenCue = cue;
        speakCue(cue);
      }
    }
    if (cue && cue !== lastSpokenCue) {
      lastSpokenCue = cue;
      speakCue(cue);
    }
    onUpdate({ landmarks_ok: true, rep_count: data.rep_count ?? 0, phase: data.phase || "ready", cue: cue || undefined, completed_rep: completedRep, avg_knee_angle: data.avg_knee_angle, alignment: currentAlignment, view: currentView });
  } catch (error) {
    updateDebugStatus({ lastHttpStatus: error.status || "Error", lastResponseMsg: error.message });
    if (error.status === 404 && !hasRestartedSession) {
      hasRestartedSession = true;
      nextFrameAt = Date.now() + 250;
      try {
        const staleSessionId = currentSessionId;
        if (staleSessionId) {
          try {
            await apiFetch(`/api/session/${staleSessionId}/end`, { method: "POST" });
          } catch (endError) {
            if (endError.status !== 404) throw endError;
          }
        }
        const response = await apiFetch("/api/session/start", { method: "POST", body: JSON.stringify({ exercise: currentActivity.toUpperCase(), risk_mode: "rule_based" }) });
        const data = await response.json();
        if (!data.session_id) throw new Error("The backend did not return a session ID.");
        currentSessionId = data.session_id;
        updateDebugStatus({ lastHttpStatus: response.status, lastResponseMsg: "Session restarted" });
      } catch (restartError) {
        onUpdate({ error: `Could not restart the session: ${restartError.message}` });
      }
    } else {
      nextFrameAt = Date.now() + 1000;
      onUpdate({ reconnecting: true, cue: "Connection lost, reconnecting" });
    }
  } finally {
    isSendingFrame = false;
  }
}

async function releaseCamera() {
  if (stream) stream.getTracks().forEach((track) => track.stop());
  stream = null;
  if (videoEl) videoEl.srcObject = null;
  if (canvasEl && ctx) ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
  updateDebugStatus({ camera: "Off", poseDetected: "No" });
}

async function closePose() {
  if (poseInstance) await poseInstance.close();
  poseInstance = null;
}

async function endSession() {
  const sessionId = currentSessionId;
  if (sessionId) await apiFetch(`/api/session/${sessionId}/end`, { method: "POST" });
  currentSessionId = null;
  return sessionId;
}

export async function stopLiveSession() {
  isRunning = false;
  cancelVoice();
  if (pendingPose) await pendingPose;
  await releaseCamera();
  if (pendingFrame) await pendingFrame;
  try {
    await closePose();
  } finally {
    isProcessingPose = false;
  }
  return endSession();
}
