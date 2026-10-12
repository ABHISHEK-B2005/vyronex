import { apiFetch, requireAuth, logout } from "./auth.js";
import { renderDoughnutChart, renderFatigueLineChart, renderBodyMapSVG } from "./charts.js";

let currentReport = null;

export async function initReportPage() {
  const user = await requireAuth();
  if (!user) return;

  document.getElementById("logout-btn")?.addEventListener("click", logout);

  const params = new URLSearchParams(window.location.search);
  const sessionId = params.get("session_id");

  try {
    const url = sessionId ? `/api/reports/${sessionId}` : `/api/reports/latest`;
    const res = await apiFetch(url);
    currentReport = await res.json();
    renderReport(currentReport);
  } catch (err) {
    document.getElementById("report-error").textContent = `Could not load report: ${err.message}`;
    document.getElementById("report-error").hidden = false;
  }
}

function renderReport(data) {
  // Title & Metadata
  document.getElementById("session-activity").textContent = (data.activity || "Exercise").toUpperCase();
  document.getElementById("session-date").textContent = new Date(data.created_at || Date.now()).toLocaleDateString();
  document.getElementById("session-duration").textContent = `${data.duration_sec || 0}s`;

  // Summary Cards
  const totals = data.totals || {};
  document.getElementById("metric-total").textContent = totals.reps || 0;
  document.getElementById("metric-correct").textContent = totals.correct || 0;
  document.getElementById("metric-defective").textContent = totals.defective || 0;
  document.getElementById("metric-high-risk").textContent = totals.high_risk || 0;
  document.getElementById("metric-accuracy").textContent = `${data.accuracy_pct || 0}%`;
  document.getElementById("metric-form-score").textContent = `${data.avg_form_score || 0}/100`;

  const riskBadge = document.getElementById("metric-risk-badge");
  const risk = (data.overall_risk || "LOW").toUpperCase();
  riskBadge.textContent = risk;
  riskBadge.className = `badge badge-${risk === "HIGH" ? "high" : (risk === "MODERATE" ? "mod" : "low")}`;

  // Doughnut Chart
  const doughnutCanvas = document.getElementById("doughnut-canvas");
  renderDoughnutChart(doughnutCanvas, {
    correct: totals.correct || 0,
    defective: totals.defective || 0,
    highRisk: totals.high_risk || 0,
  });

  // Body Map & Breakdown
  const bodyMapEl = document.getElementById("body-map-container");
  const breakdown = data.body_area_breakdown || {};
  renderBodyMapSVG(bodyMapEl, breakdown);

  const breakdownBars = document.getElementById("breakdown-bars");
  if (Object.keys(breakdown).length === 0) {
    breakdownBars.innerHTML = `<div style="color: var(--accent); padding: 12px 0;">✓ No recurring joint deficiencies detected.</div>`;
  } else {
    breakdownBars.innerHTML = Object.entries(breakdown).map(([area, pct]) => `
      <div style="margin-bottom: 10px;">
        <div style="display: flex; justify-content: space-between; font-size: 0.85rem; margin-bottom: 4px;">
          <span>${area.replace("_", " ").toUpperCase()}</span>
          <strong>${pct}%</strong>
        </div>
        <div style="background: var(--surface-raised); height: 8px; border-radius: 4px; overflow: hidden;">
          <div style="width: ${pct}%; height: 100%; background: ${pct >= 40 ? '#f87171' : (pct >= 20 ? '#fbbf24' : '#60a5fa')};"></div>
        </div>
      </div>
    `).join("");
  }

  // Fatigue & Defects
  const repScores = (data.rep_table || []).map(r => r.form_score || 80);
  const fatigueCanvas = document.getElementById("fatigue-canvas");
  renderFatigueLineChart(fatigueCanvas, repScores);

  const fatigue = data.fatigue || {};
  document.getElementById("fatigue-text").innerHTML = `
    1st Half Avg: <b>${fatigue.first_half_score || 0}</b> | 2nd Half Avg: <b>${fatigue.second_half_score || 0}</b> 
    (Drop: <b style="color: ${(fatigue.drop_pct || 0) >= 15 ? '#f87171' : 'var(--accent)'};">${fatigue.drop_pct || 0}%</b>)
  `;
  document.getElementById("best-worst-text").innerHTML = `Best Rep: <b>#${data.best_rep || 1}</b> | Worst Rep: <b>#${data.worst_rep || 1}</b>`;

  // Compared to Last Session
  const compCard = document.getElementById("comparison-card");
  if (data.compared_to_last) {
    const diff = data.compared_to_last.accuracy_change_pct;
    const trend = data.compared_to_last.trend;
    compCard.innerHTML = `
      <strong>Progress vs Last Session:</strong> 
      <span style="color: ${diff >= 0 ? '#4ade80' : '#f87171'}; font-weight: bold;">
        ${diff >= 0 ? '+' : ''}${diff}% accuracy (${trend})
      </span>
    `;
    compCard.hidden = false;
  } else {
    compCard.hidden = true;
  }

  // Reps Table
  const tbody = document.getElementById("rep-table-tbody");
  tbody.innerHTML = (data.rep_table || []).map(r => {
    const statusClass = r.status === "high_risk" ? "badge-high" : (r.status === "defective" ? "badge-mod" : "badge-low");
    const feedbackStatus = r.user_feedback ? `<span style="color: var(--muted); font-size: 0.8rem;">Rated: ${r.user_feedback}</span>` : `
      <button class="action-btn-sm" style="background: rgba(74, 222, 128, 0.2); color: #4ade80; border: 1px solid rgba(74, 222, 128, 0.4);" onclick="window.submitFeedback('${r.rep_id}', true)">Yes</button>
      <button class="action-btn-sm" style="background: rgba(248, 113, 113, 0.2); color: #f87171; border: 1px solid rgba(248, 113, 113, 0.4);" onclick="window.submitFeedback('${r.rep_id}', false)">No</button>
    `;

    return `
      <tr>
        <td><b>#${r.rep}</b></td>
        <td>${r.time}</td>
        <td><span class="badge ${statusClass}">${r.status.replace("_", " ")}</span></td>
        <td>${r.problem || "None"}</td>
        <td>${(r.body_area || "general").replace("_", " ")}</td>
        <td><b>${r.form_score}</b></td>
        <td>${feedbackStatus}</td>
      </tr>
      ${r.feedback ? `<tr style="background: var(--surface-raised);"><td colspan="7" style="font-size: 0.82rem; color: var(--muted); padding: 6px 14px;">💬 ${r.feedback}</td></tr>` : ''}
    `;
  }).join("");

  // Recommendations
  const recsList = document.getElementById("recommendations-list");
  recsList.innerHTML = (data.recommendations || []).map(rec => `
    <li style="margin-bottom: 8px; color: #e5e7eb; font-size: 0.9rem;">${rec}</li>
  `).join("");

  // Corrective Drills
  const drillsList = document.getElementById("corrective-drills-container");
  if (data.corrective_exercises && data.corrective_exercises.length > 0) {
    drillsList.innerHTML = data.corrective_exercises.map(d => `
      <div style="background: var(--surface-raised); border: 1px solid var(--border); border-radius: 8px; padding: 12px; margin-bottom: 8px;">
        <div style="display: flex; justify-content: space-between;">
          <strong style="color: var(--accent);">${d.name}</strong>
          <small style="color: var(--muted);">${d.sets_reps || ''}</small>
        </div>
        <div style="font-size: 0.83rem; color: #d1d5db; margin-top: 4px;">${d.benefit || ''}</div>
      </div>
    `).join("");
  } else {
    drillsList.innerHTML = `<div style="color: var(--muted);">No corrective exercises required for this session.</div>`;
  }

  // PDF Export Button
  const pdfBtn = document.getElementById("download-pdf-btn");
  if (pdfBtn) {
    pdfBtn.addEventListener("click", () => {
      window.location.href = `/api/reports/${data.session_id}/pdf`;
    });
  }
}

window.submitFeedback = async function(repId, isCorrect) {
  try {
    await apiFetch(`/api/feedback/rep/${repId}`, {
      method: "POST",
      body: JSON.stringify({ is_correct: isCorrect }),
    });
    alert("Thank you! Feedback recorded.");
    if (currentReport) {
      initReportPage();
    }
  } catch (err) {
    alert("Feedback error: " + err.message);
  }
};
