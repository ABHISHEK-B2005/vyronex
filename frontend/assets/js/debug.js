/**
 * Vyronex Live Camera & Pose Estimation Debug Panel (debug.js)
 */

let debugState = {
  camera: "Off",
  poseDetected: "No",
  visibleLandmarks: 0,
  framesSent: 0,
  lastHttpStatus: "None",
  lastResponseMsg: "None",
};

export function initDebugPanel(containerId = "debug-panel-container") {
  const container = document.getElementById(containerId);
  if (!container) return;

  container.innerHTML = `
    <div class="debug-panel">
      <button type="button" class="debug-header" id="debug-toggle-btn" aria-expanded="false" aria-controls="debug-content-body">
        <span>🔍 Camera Check & Diagnostic HUD</span>
        <span id="debug-toggle-icon">▼</span>
      </button>
      <div class="debug-content" id="debug-content-body" hidden>
        <div><strong>Camera:</strong> <span id="dbg-camera">Off</span></div>
        <div><strong>Pose Detected:</strong> <span id="dbg-pose">No</span></div>
        <div><strong>Visible Points:</strong> <span id="dbg-points">0 / 33</span></div>
        <div><strong>Frames Sent:</strong> <span id="dbg-frames">0</span></div>
        <div><strong>Last HTTP Status:</strong> <span id="dbg-http">None</span></div>
        <div style="grid-column: 1 / -1;"><strong>Last Response:</strong> <span id="dbg-msg">None</span></div>
      </div>
    </div>
  `;

  const toggleBtn = document.getElementById("debug-toggle-btn");
  const body = document.getElementById("debug-content-body");
  const icon = document.getElementById("debug-toggle-icon");

  toggleBtn.addEventListener("click", () => {
    const isHidden = body.hidden;
    body.hidden = !isHidden;
    toggleBtn.setAttribute("aria-expanded", String(isHidden));
    icon.textContent = isHidden ? "▲" : "▼";
  });
}

export function updateDebugStatus(partial) {
  debugState = { ...debugState, ...partial };

  const setEl = (id, val) => {
    const el = document.getElementById(id);
    if (el) el.textContent = val;
  };

  if (partial.camera !== undefined) setEl("dbg-camera", debugState.camera);
  if (partial.poseDetected !== undefined) setEl("dbg-pose", debugState.poseDetected);
  if (partial.visibleLandmarks !== undefined) setEl("dbg-points", `${debugState.visibleLandmarks} / 33`);
  if (partial.framesSent !== undefined) setEl("dbg-frames", String(debugState.framesSent));
  if (partial.lastHttpStatus !== undefined) setEl("dbg-http", String(debugState.lastHttpStatus));
  if (partial.lastResponseMsg !== undefined) setEl("dbg-msg", String(debugState.lastResponseMsg));
}
