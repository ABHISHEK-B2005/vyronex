/**
 * Vyronex Session Report Logic
 */

import { apiFetch, escapeHtml } from "./auth.js";
import { renderQualityDoughnut, renderBodyAreaBarChart, renderFatigueLineChart } from "./charts.js";

function displayValue(value, suffix = "") {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "—";
  }
  return `${value}${suffix}`;
}

export async function loadReport() {
  const params = new URLSearchParams(window.location.search);
  const sessionId = params.get("session") || params.get("session_id");

  const url = sessionId ? `/api/reports/${encodeURIComponent(sessionId)}` : `/api/reports/latest`;
  const loadingEl = document.getElementById("report-loading");
  const errorEl = document.getElementById("report-error");
  if (loadingEl) loadingEl.hidden = false;
  if (errorEl) errorEl.hidden = true;

  try {
    const res = await apiFetch(url);
    const data = await res.json();
    renderReportData(data);
  } catch (err) {
    if (errorEl) {
      errorEl.textContent = `Unable to load report: ${err.message}`;
      errorEl.hidden = false;
    }
  } finally {
    if (loadingEl) loadingEl.hidden = true;
  }
}

function renderReportData(data) {
  // Activity title & metadata
  const titleEl = document.getElementById("report-activity-title");
  if (titleEl) titleEl.textContent = (data.activity || "Exercise").toUpperCase() + " REPORT";

  const dateEl = document.getElementById("report-date");
  if (dateEl) dateEl.textContent = new Date(data.created_at || Date.now()).toLocaleDateString();

  const durEl = document.getElementById("report-duration");
  if (durEl) durEl.textContent = displayValue(data.duration_sec, "s");

  const viewSummary = document.getElementById("view-summary");
  if (viewSummary) {
    const view = String(data.camera_view || "unknown").toLowerCase();
    const labels = { front: "Front", side: "Side", angled: "Angled", unknown: "Unknown" };
    const warning = data.analysis_warning ? ` ${data.analysis_warning}` : "";
    const lowConfidence = data.low_confidence_frames
      ? ` ${data.low_confidence_frames} frame(s) used image-only landmarks.`
      : "";
    viewSummary.textContent = `Camera view: ${labels[view] || "Unknown"}.${warning}${lowConfidence}`;
    viewSummary.hidden = false;
  }

  // Summary Metrics
  const totals = data.totals || {};
  document.getElementById("stat-total").textContent = displayValue(totals.reps, "");
  document.getElementById("stat-correct").textContent = displayValue(totals.correct, "");
  document.getElementById("stat-defective").textContent = displayValue(totals.defective, "");
  document.getElementById("stat-high-risk").textContent = displayValue(totals.high_risk, "");
  document.getElementById("stat-accuracy").textContent = displayValue(data.accuracy_pct, "%");

  // Risk Badge
  const riskBadge = document.getElementById("stat-risk-badge");
  const overallRisk = (data.overall_risk || "LOW").toUpperCase();
  if (riskBadge) {
    riskBadge.textContent = overallRisk === "HIGH" ? "High risk" : (overallRisk === "MODERATE" ? "Watch your form" : "Good");
    riskBadge.className = `badge ${overallRisk === "HIGH" ? "badge-risky" : (overallRisk === "MODERATE" ? "badge-fix" : "badge-good")}`;
  }

  // Visuals (Doughnut, Bar, Line)
  renderQualityDoughnut("doughnut-chart", {
    correct: totals.correct || 0,
    defective: totals.defective || 0,
    highRisk: totals.high_risk || 0,
  });

  renderBodyAreaBarChart("body-area-chart", data.body_area_breakdown || {});

  const repScores = (data.rep_table || [])
    .map(r => r.form_score)
    .filter(score => typeof score === "number" && Number.isFinite(score));
  renderFatigueLineChart("fatigue-chart", repScores.length ? repScores : [0]);

  // Fatigue Note
  const fatigueNoteEl = document.getElementById("fatigue-note");
  const fatigue = data.fatigue || {};
  if (fatigueNoteEl) {
    if (fatigue.drop_pct >= 15 && repScores.length >= 6) {
      const dropRepIndex = Math.floor(repScores.length / 2) + 1;
      fatigueNoteEl.textContent = `⚠️ Your form dropped noticeably after rep ${dropRepIndex} (${fatigue.drop_pct}% score decline in second half).`;
      fatigueNoteEl.style.color = "var(--warn)";
    } else {
      fatigueNoteEl.textContent = `✓ Consistent form maintained throughout the session (${fatigue.drop_pct || 0}% drop).`;
      fatigueNoteEl.style.color = "var(--green)";
    }
  }

  // Comparison vs last session
  const compEl = document.getElementById("comparison-banner");
  if (compEl && data.compared_to_last) {
    const diff = data.compared_to_last.accuracy_change_pct;
    compEl.textContent = `Session comparison: ${diff >= 0 ? '+' : ''}${diff}% accuracy compared to your previous workout (${data.compared_to_last.trend}).`;
    compEl.hidden = false;
  }

  // Reps Table
  renderRepTable(data.rep_table || []);
  renderAlignmentReport(data.alignment || {}, data.rep_table || []);

  // Corrective Exercises & Next Steps
  const recsList = document.getElementById("recommendations-list");
  if (recsList) {
    recsList.innerHTML = (data.recommendations || []).map(r => `<li>${escapeHtml(r)}</li>`).join("");
  }

  // PDF Export
  const pdfBtn = document.getElementById("download-pdf-btn");
  if (pdfBtn && data.session_id) {
    pdfBtn.addEventListener("click", async () => {
      const errorEl = document.getElementById("report-error");
      pdfBtn.disabled = true;
      pdfBtn.textContent = "Preparing PDF...";
      if (errorEl) errorEl.hidden = true;
      try {
        const response = await apiFetch(`/api/reports/${encodeURIComponent(data.session_id)}/pdf`);
        const fileUrl = URL.createObjectURL(await response.blob());
        const link = document.createElement("a");
        link.href = fileUrl;
        link.download = "vyronex-session-report.pdf";
        link.click();
        window.setTimeout(() => URL.revokeObjectURL(fileUrl), 1000);
      } catch (error) {
        if (errorEl) {
          errorEl.textContent = `Unable to download the report: ${error.message}`;
          errorEl.hidden = false;
        }
      } finally {
        pdfBtn.disabled = false;
        pdfBtn.textContent = "Download PDF";
      }
    });
  }
}

function renderRepTable(reps) {
  const tbody = document.getElementById("rep-table-tbody");
  if (!tbody) return;
  tbody.addEventListener("click", (event) => {
    const button = event.target.closest("[data-rep-detail]");
    if (button) toggleRepExplanation(button.dataset.repDetail, button);
  });

  if (reps.length === 0) {
    tbody.innerHTML = `<tr><td colspan="5" style="text-align: center; color: var(--muted); padding: 20px;">No reps recorded.</td></tr>`;
    return;
  }

  tbody.innerHTML = reps.map((r, idx) => {
    const status = (r.status || "correct").toLowerCase();
    const isHigh = status === "high_risk" || (r.risk || "").toUpperCase() === "HIGH";
    const isDefective = status === "defective" || (r.risk || "").toUpperCase() === "MODERATE";

    const badgeClass = isHigh ? "badge-risky" : (isDefective ? "badge-fix" : "badge-good");
    const badgeWord = isHigh ? "High risk" : (isDefective ? "Watch your form" : "Good");

    return `
      <tr class="rep-row">
        <td><strong>#${escapeHtml(r.rep)}</strong><br><button type="button" class="rep-detail-button" data-rep-detail="${idx}" aria-expanded="false" aria-controls="rep-exp-${idx}">Details</button></td>
        <td>${escapeHtml(r.time)}</td>
        <td><span class="badge ${badgeClass}">${badgeWord}</span></td>
        <td>${escapeHtml(r.problem || "Good")}</td>
        <td>${escapeHtml(String(r.body_area || "general").replaceAll("_", " "))}</td>
      </tr>
      <tr id="rep-exp-${idx}" class="rep-explanation-row" hidden>
        <td colspan="5" style="padding: 10px 16px; font-size: 0.88rem; color: var(--muted); border-left: 3px solid var(--blue);">
          <strong>Movement details:</strong> Form score: ${r.form_score === null ? "Not assessed from this angle" : `${escapeHtml(r.form_score)}/100`}. ${escapeHtml(r.feedback || "No additional movement notes for this rep.")}<br>
          ${renderAssessedMetrics(r.form_scores)}
          <br><strong>Alignment measurements:</strong> ${renderAlignmentMeasurements(r.features)}
        </td>
      </tr>
    `;
  }).join("");
}

function renderAssessedMetrics(scores) {
  const labels = {
    knee_alignment: "Knee alignment",
    knee_depth: "Squat depth",
    back_posture: "Torso posture",
    movement_control: "Movement control",
    symmetry: "Left/right symmetry",
  };
  return Object.entries(labels).map(([key, label]) => {
    const value = scores?.[key];
    const display = typeof value === "number" ? `${escapeHtml(value)}/100` : "Not assessed from this angle";
    return `${escapeHtml(label)}: ${display}`;
  }).join(" · ");
}

function renderAlignmentReport(alignment, reps) {
  const scoreEl = document.getElementById("report-alignment-score");
  const gauge = document.getElementById("report-alignment-gauge");
  const statusEl = document.getElementById("report-alignment-status");
  const explanation = document.getElementById("report-alignment-explanation");
  const metricsBody = document.getElementById("report-alignment-metrics");
  const strip = document.getElementById("report-alignment-strip");
  if (!scoreEl || !metricsBody || !strip) return;

  const overall = alignment.overall;
  const status = alignment.status || "Not assessed";
  scoreEl.textContent = typeof overall === "number" && Number.isFinite(overall) ? `${Math.round(overall)}` : "—";
  if (gauge && typeof overall === "number" && Number.isFinite(overall)) {
    const color = status === "Good" ? "var(--success)" : status === "Watch" ? "var(--warn)" : "var(--danger)";
    gauge.style.background = `conic-gradient(${color} ${overall * 3.6}deg, #e8edf4 0deg)`;
  }
  statusEl.textContent = status;
  statusEl.className = `alignment-status ${alignmentStatusClass(status)}`;
  explanation.textContent = overall === null || overall === undefined
    ? "No alignment metrics were assessed from this camera view."
    : `${reps.length} rep${reps.length === 1 ? "" : "s"} included in the alignment summary.`;
  metricsBody.innerHTML = (alignment.metrics || []).map(metric => {
    const assessed = metric.assessed && typeof metric.score === "number";
    const value = assessed && metric.value !== null && metric.value !== undefined
      ? `${escapeHtml(Number(metric.value).toFixed(1))} ${escapeHtml(metric.unit || "")}`
      : "Not assessed from this angle";
    const score = assessed ? `${escapeHtml(metric.score)}/100` : "—";
    const metricStatus = assessed ? metric.status : "Not assessed";
    return `<tr><td>${escapeHtml(metric.label || metric.key)}</td><td>${value}</td><td>${score}</td><td><span class="alignment-status ${alignmentStatusClass(metricStatus)}">${escapeHtml(metricStatus)}</span></td></tr>`;
  }).join("") || `<tr><td colspan="4">No alignment metrics were assessed.</td></tr>`;
  strip.innerHTML = (alignment.rep_scores || []).map(entry => {
    const word = entry.status || "Not assessed";
    const title = `Rep ${entry.rep}: ${entry.score ?? "Not assessed"} ${entry.score === null ? "" : "/100"} — ${word}`;
    return `<span role="listitem" class="alignment-strip-item ${alignmentStatusClass(word)}" title="${escapeHtml(title)}" aria-label="${escapeHtml(title)}"></span>`;
  }).join("");
  document.getElementById("report-alignment-best").textContent = alignment.best_rep
    ? `Best rep #${alignment.best_rep.rep}: ${alignment.best_rep.explanation}`
    : "Best rep: Not assessed";
  document.getElementById("report-alignment-worst").textContent = alignment.worst_rep
    ? `Rep to review #${alignment.worst_rep.rep}: ${alignment.worst_rep.explanation}`
    : "Rep to review: Not assessed";
}

function alignmentStatusClass(status) {
  return status === "Good" ? "is-good" : status === "Watch" ? "is-watch" : status === "Fix" ? "is-fix" : "is-unassessed";
}

function renderAlignmentMeasurements(features) {
  const measurements = [
    ["knee_alignment", "Knee drift", value => `${formatMetric(value * 100, "% of leg length")}`],
    ["knee_symmetry", "Knee angle difference", value => formatMetric(value, "°")],
    ["hip_symmetry", "Hip angle difference", value => formatMetric(value, "°")],
  ];

  return measurements.map(([key, label, format]) => {
    const value = features?.[key];
    const display = typeof value === "number" && Number.isFinite(value)
      ? escapeHtml(format(value))
      : "Not assessed from this angle";
    return `${escapeHtml(label)}: ${display}`;
  }).join(" · ");
}

function formatMetric(value, suffix) {
  return `${Number(value).toFixed(1)}${suffix}`;
}

function toggleRepExplanation(idx, button) {
  const row = document.getElementById(`rep-exp-${idx}`);
  if (!row) return;
  row.hidden = !row.hidden;
  button.setAttribute("aria-expanded", String(!row.hidden));
}
