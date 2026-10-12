/**
 * Vyronex Chart.js Visualizer Module
 */

export function renderQualityDoughnut(canvasId, { correct, defective, highRisk }) {
  const canvas = document.getElementById(canvasId);
  if (!canvas || typeof Chart === "undefined") return;

  const total = (correct + defective + highRisk) || 1;

  if (canvas._chartInstance) canvas._chartInstance.destroy();

  canvas._chartInstance = new Chart(canvas, {
    type: "doughnut",
    data: {
      labels: ["Correct", "Fix Required", "High Risk"],
      datasets: [{
        data: [correct, defective, highRisk],
        backgroundColor: ["#2A9D68", "#FFC20E", "#9B1226"],
        borderWidth: 2,
        borderColor: "#FFFFFF",
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: "bottom",
          labels: { font: { family: "Figtree", size: 12 }, color: "#241112" },
        },
      },
      cutout: "68%",
    },
  });
}

export function renderBodyAreaBarChart(canvasId, bodyAreaBreakdown) {
  const canvas = document.getElementById(canvasId);
  if (!canvas || typeof Chart === "undefined") return;

  const labels = Object.keys(bodyAreaBreakdown || {}).map(k => k.replace("_", " ").toUpperCase());
  const data = Object.values(bodyAreaBreakdown || {});

  if (labels.length === 0) {
    labels.push("NO FLAWS");
    data.push(100);
  }

  if (canvas._chartInstance) canvas._chartInstance.destroy();

  canvas._chartInstance = new Chart(canvas, {
    type: "bar",
    data: {
      labels,
      datasets: [{
        label: "% of Defective Reps",
        data,
        backgroundColor: data.map(v => v >= 40 ? "#9B1226" : (v >= 20 ? "#FFC20E" : "#2B8BE0")),
        borderRadius: 6,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        y: { beginAtZero: true, max: 100, ticks: { callback: v => v + "%" } },
      },
    },
  });
}

export function renderFatigueLineChart(canvasId, repScores) {
  const canvas = document.getElementById(canvasId);
  if (!canvas || typeof Chart === "undefined" || !repScores || repScores.length === 0) return;

  const labels = repScores.map((_, i) => `#${i + 1}`);

  if (canvas._chartInstance) canvas._chartInstance.destroy();

  canvas._chartInstance = new Chart(canvas, {
    type: "line",
    data: {
      labels,
      datasets: [{
        label: "Form Score (0-100)",
        data: repScores,
        borderColor: "#2B8BE0",
        backgroundColor: "rgba(43, 139, 224, 0.1)",
        fill: true,
        tension: 0.3,
        pointBackgroundColor: repScores.map(s => s >= 75 ? "#2A9D68" : (s >= 50 ? "#FFC20E" : "#9B1226")),
        pointRadius: 5,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: { min: 0, max: 100 },
      },
      plugins: {
        legend: { display: false },
      },
    },
  });
}

export function renderProgressTrendChart(canvasId, trendPoints) {
  const canvas = document.getElementById(canvasId);
  if (!canvas || typeof Chart === "undefined" || !trendPoints || trendPoints.length === 0) return;

  const labels = trendPoints.map(p => p.date);
  const data = trendPoints.map(p => p.value);

  if (canvas._chartInstance) canvas._chartInstance.destroy();

  canvas._chartInstance = new Chart(canvas, {
    type: "line",
    data: {
      labels,
      datasets: [{
        label: "Accuracy %",
        data,
        borderColor: "#2B8BE0",
        backgroundColor: "rgba(43, 139, 224, 0.08)",
        fill: true,
        tension: 0.25,
        pointBackgroundColor: "#2B8BE0",
        pointRadius: 6,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: { min: 0, max: 100, ticks: { callback: v => v + "%" } },
      },
    },
  });
}
