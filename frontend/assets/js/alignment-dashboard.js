import { apiFetch } from "./auth.js";

const FOCUS_TIPS = {
  knee_alignment: "Try glute bridges and banded squats to practise steady knee tracking.",
  shoulder_level: "Use slow, even repetitions and keep both shoulders relaxed and level.",
  hip_level: "Practise controlled bodyweight squats while keeping your hips level.",
  weight_shift: "Use a chair squat and share pressure evenly between both feet.",
  foot_ankle_alignment: "Set your feet at a comfortable, even width before each repetition.",
  torso_lean: "Practise hip hinges and brace your core to keep your back tall.",
  neck_torso_alignment: "Keep your head in line with your torso; look naturally ahead.",
  knee_over_toe: "Practise a shallow squat, letting each knee track in line with its toes.",
  hip_knee_ankle_stacking: "Use slow squats and focus on keeping hip, knee and ankle aligned.",
};

export async function loadAlignmentDashboard() {
  const loading = document.getElementById("alignment-loading");
  const error = document.getElementById("alignment-error");
  const empty = document.getElementById("alignment-empty");
  const content = document.getElementById("alignment-content");
  const notAssessed = document.getElementById("alignment-not-assessed-note");
  if (!content) return;
  if (loading) loading.hidden = false;
  if (error) error.hidden = true;
  if (empty) empty.hidden = true;
  content.hidden = true;
  try {
    const response = await apiFetch("/api/alignment/summary?days=30");
    const data = await response.json();
    renderAlignmentSummary(data);
  } catch (err) {
    if (error) {
      error.textContent = `Could not load alignment summary: ${err.message}`;
      error.hidden = false;
    }
  } finally {
    if (loading) loading.hidden = true;
  }
}

export function renderAlignmentSummary(data) {
  const empty = document.getElementById("alignment-empty");
  const content = document.getElementById("alignment-content");
  const notAssessed = document.getElementById("alignment-not-assessed-note");
  if (!content) return;
  if (!data.sessions_count) {
    content.hidden = true;
    if (notAssessed) notAssessed.hidden = true;
    if (empty) empty.hidden = false;
    return;
  }
  if (empty) empty.hidden = true;
  content.hidden = false;
  if (notAssessed) notAssessed.hidden = Number.isFinite(data.latest_overall_score);
  renderGauge(data);
  renderBodyMap(data.latest_metrics || []);
  renderTrend(data.trend || []);
  renderAreaBars(data.body_area_counts || {});
  renderFocus(data.most_frequent_misalignment);
}

function renderGauge(data) {
  const gauge = document.getElementById("dashboard-alignment-gauge");
  const score = document.getElementById("dashboard-alignment-score");
  const status = document.getElementById("dashboard-alignment-status");
  const change = document.getElementById("dashboard-alignment-change");
  if (!gauge || !score || !status || !change) return;
  const value = data.latest_overall_score;
  const statusWord = !Number.isFinite(value) ? "Not assessed" : value >= 80 ? "Good" : value >= 60 ? "Watch" : "Fix";
  gauge.style.background = Number.isFinite(value)
    ? `conic-gradient(${statusColor(statusWord)} ${value * 3.6}deg, #e8edf4 0deg)`
    : "conic-gradient(#aeb8c2 0deg, #e8edf4 0deg)";
  score.textContent = Number.isFinite(value) ? `${Math.round(value)}` : "—";
  status.textContent = statusWord;
  status.className = `alignment-status ${statusClass(statusWord)}`;
  const difference = data.change_since_last;
  change.textContent = difference === null || difference === undefined
    ? "No previous session to compare."
    : `${difference > 0 ? "+" : ""}${difference} points since your last session.`;
}

function renderBodyMap(metrics) {
  const map = document.getElementById("alignment-body-map");
  if (!map) return;
  const areaScores = {};
  metrics.forEach(metric => {
    if (!metric.assessed || !metric.body_area || !Number.isFinite(metric.score)) return;
    areaScores[metric.body_area] = Math.min(areaScores[metric.body_area] ?? 100, metric.score);
  });
  map.querySelectorAll("[data-area]").forEach(part => {
    const score = areaScores[part.dataset.area];
    const word = score === undefined ? "Not assessed" : score >= 80 ? "Good" : score >= 60 ? "Watch" : "Fix";
    part.setAttribute("class", `body-map-part ${statusClass(word)}`);
    part.setAttribute("aria-label", `${part.dataset.area.replaceAll("_", " ")}: ${word}`);
  });
}

function renderTrend(points) {
  const canvas = document.getElementById("alignment-trend-chart");
  const empty = document.getElementById("alignment-trend-empty");
  if (!canvas) return;
  if (!points.length || typeof Chart === "undefined") {
    if (empty) {
      empty.textContent = points.length ? "The chart library could not load." : "No alignment trend yet.";
      empty.hidden = false;
    }
    return;
  }
  if (empty) empty.hidden = true;
  if (canvas._chartInstance) canvas._chartInstance.destroy();
  canvas._chartInstance = new Chart(canvas, {
    type: "line",
    data: {
      labels: points.map(point => point.date),
      datasets: [{
        label: "Alignment score",
        data: points.map(point => point.score),
        borderColor: "#14A3A8",
        backgroundColor: "rgba(20,163,168,.12)",
        pointBackgroundColor: points.map(point => statusColor(point.score >= 80 ? "Good" : point.score >= 60 ? "Watch" : "Fix")),
        fill: true,
        tension: 0.25,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: { y: { min: 0, max: 100 } },
      plugins: { legend: { display: false } },
    },
  });
}

function renderAreaBars(counts) {
  const canvas = document.getElementById("alignment-area-chart");
  const empty = document.getElementById("alignment-area-empty");
  const entries = Object.entries(counts);
  if (!canvas) return;
  if (!entries.length || typeof Chart === "undefined") {
    if (empty) {
      empty.textContent = entries.length ? "The chart library could not load." : "No flagged areas yet.";
      empty.hidden = false;
    }
    return;
  }
  if (empty) empty.hidden = true;
  if (canvas._chartInstance) canvas._chartInstance.destroy();
  canvas._chartInstance = new Chart(canvas, {
    type: "bar",
    data: {
      labels: entries.map(([area]) => area.replaceAll("_", " ")),
      datasets: [{
        data: entries.map(([, count]) => count),
        backgroundColor: "#5BB8F5",
        borderRadius: 6,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: { y: { beginAtZero: true, ticks: { precision: 0 } } },
    },
  });
}

function renderFocus(fault) {
  const label = document.getElementById("alignment-focus-fault");
  const tip = document.getElementById("alignment-focus-tip");
  if (!label || !tip) return;
  label.textContent = fault?.label || "No recurring misalignment";
  tip.textContent = fault ? (FOCUS_TIPS[fault.key] || "Use slow, controlled repetitions and focus on balanced movement.") : "Keep practising with clear camera framing.";
}

function statusClass(status) {
  return status === "Good" ? "is-good" : status === "Watch" ? "is-watch" : status === "Fix" ? "is-fix" : "is-unassessed";
}

function statusColor(status) {
  return status === "Good" ? "#22A06B" : status === "Watch" ? "#E8A317" : "#D64545";
}
