const SKELETON_CONNECTIONS = [[11, 12], [11, 13], [13, 15], [12, 14], [14, 16], [11, 23], [12, 24], [23, 24], [23, 25], [25, 27], [24, 26], [26, 28]];
const LANDMARK_DOTS = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 31, 32];

export function cameraError(error) {
  const messages = {
    NotAllowedError: "Camera permission denied. Allow camera access in your browser settings and try again.",
    PermissionDeniedError: "Camera permission denied. Allow camera access in your browser settings and try again.",
    NotFoundError: "No camera was found on this device.",
    DevicesNotFoundError: "No camera was found on this device.",
    NotReadableError: "The camera is in use by another app. Close that app and try again.",
    TrackStartError: "The camera is in use by another app. Close that app and try again.",
  };
  return new Error(messages[error.name] || `Could not start the camera: ${error.message || "Please check your camera settings."}`);
}

export function getAlignmentCue(alignment) {
  const flagged = alignment?.metrics?.find(metric => metric.assessed && metric.status !== "Good");
  if (!flagged) return "";
  const cues = {
    knee_alignment: "Push your knees out and track them over your feet.",
    shoulder_level: "Keep your shoulders level.",
    hip_level: "Keep your hips level.",
    weight_shift: "Share your weight evenly between both legs.",
    foot_ankle_alignment: "Keep your feet evenly spaced.",
    torso_lean: "Keep your back tall and brace your core.",
    neck_torso_alignment: "Keep your head in line with your torso.",
    knee_over_toe: "Keep your knees moving in line with your toes.",
    hip_knee_ankle_stacking: "Keep your hips, knees and ankles aligned.",
  };
  return cues[flagged.key] || flagged.label;
}

export function renderLiveAlignment(alignment, view) {
  const score = document.getElementById("live-alignment-score");
  const status = document.getElementById("live-alignment-status");
  const gauge = document.getElementById("live-alignment-gauge");
  const guide = document.getElementById("live-view-guide");
  const metrics = document.getElementById("live-alignment-metrics");
  if (!metrics) return;
  if (!alignment) {
    if (score) score.textContent = "—";
    if (status) {
      status.textContent = "Waiting for pose";
      status.className = "alignment-status is-unassessed";
    }
    if (gauge) gauge.dataset.status = "Not assessed";
    if (guide) guide.textContent = "Camera view will appear when a pose is detected.";
    metrics.innerHTML = `<p class="alignment-unassessed">Not assessed from this angle.</p>`;
    return;
  }
  const alignmentStatus = alignment.status || "Not assessed";
  if (score) score.textContent = Number.isFinite(alignment.overall) ? `${Math.round(alignment.overall)}` : "—";
  if (status) {
    status.textContent = alignmentStatus;
    status.className = `alignment-status ${statusClass(alignmentStatus)}`;
  }
  if (gauge) {
    gauge.dataset.status = alignmentStatus;
    if (Number.isFinite(alignment.overall)) {
      const color = alignmentStatus === "Good" ? "#22A06B" : alignmentStatus === "Watch" ? "#E8A317" : "#D64545";
      gauge.style.background = `conic-gradient(${color} ${alignment.overall * 3.6}deg, #e8edf4 0deg)`;
    }
  }
  if (guide) {
    const viewName = view === "front" ? "Front" : view === "side" ? "Side" : "Angled";
    const hint = view === "side"
      ? "Front view helps assess knee alignment."
      : "Side view helps assess back alignment.";
    guide.textContent = `${viewName} view detected. ${hint}`;
  }
  metrics.innerHTML = (alignment.metrics || []).map(metric => {
    if (!metric.assessed || !Number.isFinite(metric.score)) {
      return `<div class="alignment-metric is-unassessed"><span>${escapeText(metric.label)}</span><small>Not assessed from this angle</small></div>`;
    }
    return `<div class="alignment-metric"><div><span>${escapeText(metric.label)}</span><strong class="alignment-status ${statusClass(metric.status)}">${escapeText(metric.status)}</strong></div><small>${Number(metric.value).toFixed(1)} ${escapeText(metric.unit || "")}</small><div class="alignment-meter"><i class="${statusClass(metric.status)}" style="width:${Math.max(0, Math.min(100, metric.score))}%"></i></div></div>`;
  }).join("");
}

function statusClass(status) {
  return status === "Good" ? "is-good" : status === "Watch" ? "is-watch" : status === "Fix" ? "is-fix" : "is-unassessed";
}

function escapeText(value) {
  const span = document.createElement("span");
  span.textContent = value ?? "";
  return span.innerHTML;
}

export function drawAlignmentGuides(ctx, canvas, landmarks, alignment, view) {
  if (!alignment) return;
  const getMetric = (key) => alignment.metrics?.find(metric => metric.key === key);
  const colorFor = (metric) => metric?.status === "Good" ? "#22A06B" : metric?.status === "Watch" ? "#E8A317" : "#D64545";
  const point = (index) => ({ x: landmarks[index].x * canvas.width, y: landmarks[index].y * canvas.height });
  ctx.save();
  ctx.lineWidth = 2.5;
  if (view === "front") {
    for (const [hip, knee, ankle] of [[23, 25, 27], [24, 26, 28]]) {
      const h = point(hip), k = point(knee), a = point(ankle);
      ctx.strokeStyle = colorFor(getMetric("knee_alignment"));
      ctx.beginPath(); ctx.moveTo(h.x, h.y); ctx.lineTo(a.x, a.y); ctx.stroke();
      ctx.fillStyle = colorFor(getMetric("knee_alignment"));
      ctx.beginPath(); ctx.arc(k.x, k.y, 7, 0, 2 * Math.PI); ctx.fill();
    }
    for (const [left, right, key] of [[11, 12, "shoulder_level"], [23, 24, "hip_level"]]) {
      const a = point(left), b = point(right);
      ctx.strokeStyle = colorFor(getMetric(key));
      ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
    }
  } else if (view === "side") {
    const ankle = point(27), hip = point(23), shoulder = point(11);
    ctx.strokeStyle = "#5BB8F5";
    ctx.setLineDash([6, 5]);
    ctx.beginPath(); ctx.moveTo(ankle.x, 0); ctx.lineTo(ankle.x, canvas.height); ctx.stroke();
    ctx.setLineDash([]);
    ctx.strokeStyle = colorFor(getMetric("torso_lean"));
    ctx.beginPath(); ctx.moveTo(hip.x, hip.y); ctx.lineTo(shoulder.x, shoulder.y); ctx.stroke();
    const lean = getMetric("torso_lean");
    if (lean?.assessed) {
      ctx.fillStyle = colorFor(lean);
      ctx.font = "600 16px Figtree, sans-serif";
      ctx.fillText(`${lean.value.toFixed(1)}° torso lean`, shoulder.x + 8, shoulder.y);
    }
  }
  ctx.restore();
}

export function drawSkeleton(ctx, canvas, landmarks) {
  const { width, height } = canvas;
  ctx.strokeStyle = "#5BB8F5";
  ctx.lineWidth = 3.5;
  for (const [first, second] of SKELETON_CONNECTIONS) {
    const a = landmarks[first];
    const b = landmarks[second];
    if (a && b && (a.visibility ?? 0) >= 0.4 && (b.visibility ?? 0) >= 0.4) {
      ctx.beginPath(); ctx.moveTo(a.x * width, a.y * height); ctx.lineTo(b.x * width, b.y * height); ctx.stroke();
    }
  }
  for (const index of LANDMARK_DOTS) {
    const point = landmarks[index];
    if (!point || (point.visibility ?? 0) < 0.4) continue;
    ctx.beginPath(); ctx.arc(point.x * width, point.y * height, 5.5, 0, 2 * Math.PI);
    ctx.fillStyle = "#14A3A8"; ctx.fill(); ctx.strokeStyle = "#FFFFFF"; ctx.lineWidth = 2; ctx.stroke();
  }
}
