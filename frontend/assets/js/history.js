/**
 * Vyronex Longitudinal History & Analytics
 */

import { apiFetch, escapeHtml } from "./auth.js";
import { renderProgressTrendChart } from "./charts.js";

export async function initHistoryPage() {
  await fetchHistory();

  const filterForm = document.getElementById("history-filter-form");
  if (filterForm) {
    filterForm.addEventListener("submit", (e) => {
      e.preventDefault();
      fetchHistory();
    });
  }
}

async function fetchHistory() {
  const errorEl = document.getElementById("history-error");
  const tbody = document.getElementById("history-tbody");
  if (errorEl) errorEl.hidden = true;
  if (tbody) tbody.innerHTML = `<tr><td colspan="6" role="status" style="text-align: center; padding: 24px;"><span class="spinner" aria-hidden="true"></span> Loading sessions...</td></tr>`;
  const activity = document.getElementById("filter-activity")?.value || "";
  const fromDate = document.getElementById("filter-from")?.value || "";
  const toDate = document.getElementById("filter-to")?.value || "";

  const qs = new URLSearchParams();
  if (activity) qs.append("activity", activity);
  if (fromDate) qs.append("from_date", fromDate);
  if (toDate) qs.append("to_date", toDate);

  const url = `/api/history${qs.toString() ? '?' + qs.toString() : ''}`;

  try {
    const res = await apiFetch(url);
    const data = await res.json();
    renderHistoryData(data);
  } catch (err) {
    if (errorEl) {
      errorEl.textContent = `Error loading history: ${err.message}`;
      errorEl.hidden = false;
    }
    if (tbody) tbody.innerHTML = `<tr><td colspan="6" style="text-align: center; padding: 24px;">Your session history could not be loaded.</td></tr>`;
  }
}

function renderHistoryData(data) {
  // Recurring Weakness Card
  const weaknessCard = document.getElementById("recurring-weakness-card");
  const weaknessText = document.getElementById("recurring-weakness-text");
  if (data.recurring_weakness) {
    weaknessText.textContent = data.recurring_weakness.message;
    weaknessCard.hidden = false;
  } else if (weaknessCard) {
    weaknessCard.hidden = true;
  }

  // Progress Trend Chart
  const trendPoints = (data.charts?.accuracy_over_time || []).map(pt => ({
    date: pt.date,
    value: pt.value,
  }));
  renderProgressTrendChart("accuracy-trend-chart", trendPoints);

  // Sessions Table
  const tbody = document.getElementById("history-tbody");
  if (!tbody) return;

  const sessions = data.sessions || [];
  if (sessions.length === 0) {
    tbody.innerHTML = `<tr><td colspan="6" style="text-align: center; color: var(--muted); padding: 24px;">No workout sessions match your filter.</td></tr>`;
    return;
  }

  tbody.innerHTML = sessions.map(s => {
    const risk = String(s.overall_risk || "LOW").toUpperCase();
    const isRisky = risk === "HIGH";
    const isFix = risk === "MODERATE";
    const badgeClass = isRisky ? "badge-risky" : (isFix ? "badge-fix" : "badge-good");
    const badgeWord = isRisky ? "High risk" : (isFix ? "Watch your form" : "Good");
    const sessionId = encodeURIComponent(s.session_id || "");

    return `
      <tr>
        <td>${escapeHtml(s.date)}</td>
        <td><strong>${escapeHtml(String(s.activity || "Activity").toUpperCase())}</strong></td>
        <td>${escapeHtml(s.total_reps)} reps</td>
        <td><strong>${escapeHtml(s.accuracy_pct)}%</strong></td>
        <td><span class="badge ${badgeClass}">${badgeWord}</span></td>
        <td><a href="report.html?session=${sessionId}" class="btn btn-secondary btn-sm">View Report</a></td>
      </tr>
    `;
  }).join("");
}
