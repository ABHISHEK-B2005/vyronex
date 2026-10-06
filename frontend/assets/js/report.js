/**
 * Vyronex Session Report Logic
 */

import { apiFetch, escapeHtml, API_BASE_URL, getToken } from "./auth.js";
import { renderQualityDoughnut, renderBodyAreaBarChart, renderFatigueLineChart } from "./charts.js";

export async function loadReport() {
  const params = new URLSearchParams(window.location.search);
  const sessionId = params.get("session") || params.get("session_id");

  const url = sessionId ? `/api/reports/${sessionId}` : `/api/reports/latest`;

  try {
    const res = await apiFetch(url);
    const data = await res.json();
    renderReportData(data);
  } catch (err) {
    const errEl = document.getElementById("report-error");
    if (errEl) {
      errEl.textContent = `Unable to load report: ${err.message}`;
      errEl.hidden = false;
    }
  }
}

function renderReportData(data) {
  // Activity title & metadata
  const titleEl = document.getElementById("report-activity-title");
  if (titleEl) titleEl.textContent = (data.activity || "Exercise").toUpperCase() + " REPORT";

  const dateEl = document.getElementById("report-date");
  if (dateEl) dateEl.textContent = new Date(data.created_at || Date.now()).toLocaleDateString();

  const durEl = document.getElementById("report-duration");
  if (durEl) durEl.textContent = `${data.duration_sec || 0}s`;

  // Summary Metrics
  const totals = data.totals || {};
  document.getElementById("stat-total").textContent = totals.reps || 0;
  document.getElementById("stat-correct").textContent = totals.correct || 0;
  document.getElementById("stat-defective").textContent = totals.defective || 0;
  document.getElementById("stat-high-risk").textContent = totals.high_risk || 0;
  document.getElementById("stat-accuracy").textContent = `${data.accuracy_pct || 0}%`;

  // Risk Badge
  const riskBadge = document.getElementById("stat-risk-badge");
  const overallRisk = (data.overall_risk || "LOW").toUpperCase();
  if (riskBadge) {
    riskBadge.textContent = overallRisk === "HIGH" ? "Risky" : (overallRisk === "MODERATE" ? "Fix Needed" : "Good Form");
    riskBadge.className = `badge ${overallRisk === "HIGH" ? "badge-risky" : (overallRisk === "MODERATE" ? "badge-fix" : "badge-good")}`;
  }

  // Visuals (Doughnut, Bar, Line)
  renderQualityDoughnut("doughnut-chart", {
    correct: totals.correct || 0,
    defective: totals.defective || 0,
    highRisk: totals.high_risk || 0,
  });

  renderBodyAreaBarChart("body-area-chart", data.body_area_breakdown || {});

  const repScores = (data.rep_table || []).map(r => r.form_score || 80);
  renderFatigueLineChart("fatigue-chart", repScores);

  // Fatigue Note
  const fatigueNoteEl = document.getElementById("fatigue-note");
  const fatigue = data.fatigue || {};
  if (fatigueNoteEl) {
    if (fatigue.drop_pct >= 15 && repScores.length >= 6) {
      const dropRepIndex = Math.floor(repScores.length / 2) + 1;
      fatigueNoteEl.textContent = `⚠️ Your form dropped noticeably after rep ${dropRepIndex} (${fatigue.drop_pct}% score decline in second half).`;
      fatigueNoteEl.style.color = "var(--red)";
    } else {
      fatigueNoteEl.textContent = `✓ Consistent form maintained throughout the session (${fatigue.drop_pct || 0}% drop).`;
      fatigueNoteEl.style.color = "var(--green)";
    }
  }

  // Comparison vs last session
  const compEl = document.getElementById("comparison-banner");
  if (compEl && data.compared_to_last) {
    const diff = data.compared_to_last.accuracy_change_pct;
    compEl.innerHTML = `<strong>Session Comparison:</strong> ${diff >= 0 ? '+' : ''}${diff}% accuracy compared to your previous workout (${data.compared_to_last.trend}).`;
    compEl.hidden = false;
  }

  // Reps Table
  renderRepTable(data.rep_table || []);

  // Corrective Exercises & Next Steps
  const recsList = document.getElementById("recommendations-list");
  if (recsList) {
    recsList.innerHTML = (data.recommendations || []).map(r => `<li>${escapeHtml(r)}</li>`).join("");
  }

  // PDF Export
  const pdfBtn = document.getElementById("download-pdf-btn");
  if (pdfBtn && data.session_id) {
    pdfBtn.addEventListener("click", () => {
      const token = getToken();
      window.location.href = `${API_BASE_URL}/api/reports/${data.session_id}/pdf?token=${encodeURIComponent(token || "")}`;
    });
  }
}

function renderRepTable(reps) {
  const tbody = document.getElementById("rep-table-tbody");
  if (!tbody) return;

  if (reps.length === 0) {
    tbody.innerHTML = `<tr><td colspan="5" style="text-align: center; color: var(--muted); padding: 20px;">No reps recorded.</td></tr>`;
    return;
  }

  tbody.innerHTML = reps.map((r, idx) => {
    const status = (r.status || "correct").toLowerCase();
    const isHigh = status === "high_risk" || (r.risk || "").toUpperCase() === "HIGH";
    const isDefective = status === "defective" || (r.risk || "").toUpperCase() === "MODERATE";

    const badgeClass = isHigh ? "badge-risky" : (isDefective ? "badge-fix" : "badge-good");
    const badgeWord = isHigh ? "Risky" : (isDefective ? "Fix" : "Good");

    return `
      <tr class="rep-row" onclick="window.toggleRepExplanation(${idx})">
        <td><strong>#${r.rep}</strong></td>
        <td>${r.time}</td>
        <td><span class="badge ${badgeClass}">${badgeWord}</span></td>
        <td>${escapeHtml(r.problem || "Good Form")}</td>
        <td>${escapeHtml((r.body_area || "general").replace("_", " "))}</td>
      </tr>
      <tr id="rep-exp-${idx}" class="rep-explanation-row" style="display: none; background: #fdfaf6;">
        <td colspan="5" style="padding: 10px 16px; font-size: 0.88rem; color: var(--muted); border-left: 3px solid var(--blue);">
          <strong>Kinematic Breakdown:</strong> Form Score: ${r.form_score}/100. ${escapeHtml(r.feedback || "Standard execution parameters met.")}
        </td>
      </tr>
    `;
  }).join("");
}

window.toggleRepExplanation = function(idx) {
  const row = document.getElementById(`rep-exp-${idx}`);
  if (row) {
    row.style.display = row.style.display === "none" ? "table-row" : "none";
  }
};
