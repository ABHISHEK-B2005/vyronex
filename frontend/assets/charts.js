/**
 * VYRONEX Biomechanical Visuals & Chart Helpers (Vanilla Canvas + SVG)
 */

export function renderDoughnutChart(canvas, { correct, defective, highRisk }) {
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  const centerX = width / 2;
  const centerY = height / 2;
  const radius = Math.min(centerX, centerY) - 15;
  const innerRadius = radius * 0.65;

  ctx.clearRect(0, 0, width, height);

  const total = (correct + defective + highRisk) || 1;
  const slices = [
    { value: correct, color: "#4ade80", label: "Correct" },
    { value: defective, color: "#fbbf24", label: "Defective" },
    { value: highRisk, color: "#f87171", label: "High Risk" },
  ];

  let currentAngle = -Math.PI / 2;

  slices.forEach(slice => {
    const sliceAngle = (slice.value / total) * 2 * Math.PI;
    if (sliceAngle > 0) {
      ctx.beginPath();
      ctx.arc(centerX, centerY, radius, currentAngle, currentAngle + sliceAngle);
      ctx.arc(centerX, centerY, innerRadius, currentAngle + sliceAngle, currentAngle, true);
      ctx.closePath();
      ctx.fillStyle = slice.color;
      ctx.fill();
    }
    currentAngle += sliceAngle;
  });

  // Center text
  ctx.fillStyle = "#ffffff";
  ctx.font = "bold 16px Inter, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(`${correct}/${total}`, centerX, centerY - 6);

  ctx.fillStyle = "#a4aa9f";
  ctx.font = "10px Inter, sans-serif";
  ctx.fillText("CORRECT", centerX, centerY + 12);
}

export function renderFatigueLineChart(canvas, repScores) {
  if (!canvas || !repScores || repScores.length === 0) return;
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  const padding = 30;

  ctx.clearRect(0, 0, width, height);

  const usableWidth = width - padding * 2;
  const usableHeight = height - padding * 2;

  // Grid lines
  ctx.strokeStyle = "#343a31";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = padding + (usableHeight / 4) * i;
    ctx.beginPath();
    ctx.moveTo(padding, y);
    ctx.lineTo(width - padding, y);
    ctx.stroke();

    const scoreVal = 100 - i * 25;
    ctx.fillStyle = "#a4aa9f";
    ctx.font = "9px Inter, sans-serif";
    ctx.textAlign = "right";
    ctx.fillText(scoreVal, padding - 6, y + 3);
  }

  // Draw trend line
  const stepX = repScores.length > 1 ? usableWidth / (repScores.length - 1) : usableWidth / 2;
  ctx.beginPath();
  ctx.strokeStyle = "#c8f169";
  ctx.lineWidth = 2.5;

  repScores.forEach((score, idx) => {
    const x = padding + idx * stepX;
    const y = padding + usableHeight - (score / 100) * usableHeight;
    if (idx === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // Draw data dots
  repScores.forEach((score, idx) => {
    const x = padding + idx * stepX;
    const y = padding + usableHeight - (score / 100) * usableHeight;
    ctx.beginPath();
    ctx.arc(x, y, 4, 0, 2 * Math.PI);
    ctx.fillStyle = score >= 75 ? "#4ade80" : (score >= 50 ? "#fbbf24" : "#f87171");
    ctx.fill();
    ctx.strokeStyle = "#10120f";
    ctx.lineWidth = 1.5;
    ctx.stroke();
  });
}

export function renderBodyMapSVG(container, bodyAreaBreakdown) {
  if (!container) return;

  const kneeColor = bodyAreaBreakdown.knee ? (bodyAreaBreakdown.knee > 30 ? "#f87171" : "#fbbf24") : "#4ade80";
  const backColor = bodyAreaBreakdown.lower_back ? (bodyAreaBreakdown.lower_back > 30 ? "#f87171" : "#fbbf24") : "#4ade80";
  const shoulderColor = bodyAreaBreakdown.shoulder ? (bodyAreaBreakdown.shoulder > 30 ? "#f87171" : "#fbbf24") : "#4ade80";
  const hipColor = bodyAreaBreakdown.hip ? (bodyAreaBreakdown.hip > 30 ? "#f87171" : "#fbbf24") : "#4ade80";
  const ankleColor = bodyAreaBreakdown.ankle ? (bodyAreaBreakdown.ankle > 30 ? "#f87171" : "#fbbf24") : "#4ade80";

  container.innerHTML = `
    <svg viewBox="0 0 200 300" style="width: 100%; max-width: 180px; height: auto;">
      <!-- Head -->
      <circle cx="100" cy="30" r="16" fill="#343a31" stroke="#a4aa9f" stroke-width="1.5"/>
      <!-- Neck -->
      <line x1="100" y1="46" x2="100" y2="58" stroke="#a4aa9f" stroke-width="3"/>
      <!-- Shoulders -->
      <line x1="60" y1="62" x2="140" y2="62" stroke="${shoulderColor}" stroke-width="6" stroke-linecap="round"/>
      <!-- Spine / Upper Torso -->
      <line x1="100" y1="62" x2="100" y2="95" stroke="#343a31" stroke-width="8"/>
      <!-- Lower Back / Lumbar -->
      <line x1="100" y1="95" x2="100" y2="120" stroke="${backColor}" stroke-width="10" stroke-linecap="round"/>
      <!-- Pelvis / Hip -->
      <line x1="75" y1="125" x2="125" y2="125" stroke="${hipColor}" stroke-width="7" stroke-linecap="round"/>
      <!-- Left Leg (Hip to Knee) -->
      <line x1="80" y1="128" x2="76" y2="185" stroke="#343a31" stroke-width="5"/>
      <!-- Right Leg (Hip to Knee) -->
      <line x1="120" y1="128" x2="124" y2="185" stroke="#343a31" stroke-width="5"/>
      <!-- Knees -->
      <circle cx="76" cy="188" r="7" fill="${kneeColor}"/>
      <circle cx="124" cy="188" r="7" fill="${kneeColor}"/>
      <!-- Lower Legs (Knee to Ankle) -->
      <line x1="76" y1="195" x2="74" y2="255" stroke="#343a31" stroke-width="4"/>
      <line x1="124" y1="195" x2="126" y2="255" stroke="#343a31" stroke-width="4"/>
      <!-- Ankles -->
      <circle cx="74" cy="258" r="6" fill="${ankleColor}"/>
      <circle cx="126" cy="258" r="6" fill="${ankleColor}"/>
      <!-- Feet -->
      <line x1="74" y1="262" x2="60" y2="264" stroke="#a4aa9f" stroke-width="3" stroke-linecap="round"/>
      <line x1="126" y1="262" x2="140" y2="264" stroke="#a4aa9f" stroke-width="3" stroke-linecap="round"/>
    </svg>
  `;
}
