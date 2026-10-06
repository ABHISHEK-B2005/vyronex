"""PDF Report Generator for Vyronex Biomechanical Sessions (Chunk 10c).

Uses ReportLab and Matplotlib to compile beautiful, exportable clinical/coach-grade PDFs.
"""

from __future__ import annotations

import io
from datetime import datetime
from typing import Any, Dict, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    HRFlowable,
    Image,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def _render_body_area_chart(body_areas: Dict[str, int]) -> io.BytesIO:
    """Generate a clean horizontal bar chart image of body area defect breakdown."""
    fig, ax = plt.subplots(figsize=(6.5, 2.0), dpi=180)
    fig.patch.set_facecolor("#111827")
    ax.set_facecolor("#111827")

    if not body_areas:
        body_areas = {"No defects": 100}

    labels = [k.replace("_", " ").title() for k in body_areas.keys()]
    values = list(body_areas.values())
    y_pos = range(len(labels))

    chart_colors = ["#ef4444" if v >= 40 else ("#f59e0b" if v >= 20 else "#3b82f6") for v in values]
    bars = ax.barh(y_pos, values, color=chart_colors, height=0.55, edgecolor="none")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, color="#e5e7eb", fontsize=9, fontweight="bold")
    ax.set_xlabel("Defect Distribution Share (%)", color="#9ca3af", fontsize=8)
    ax.set_xlim(0, max(max(values) * 1.25, 100))
    ax.tick_params(colors="#9ca3af", labelsize=8)

    for spine in ax.spines.values():
        spine.set_color("#374151")

    # Value annotations
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + 2,
            bar.get_y() + bar.get_height() / 2,
            f"{val}%",
            va="center",
            ha="left",
            color="#f3f4f6",
            fontsize=8,
            fontweight="bold",
        )

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor(), dpi=180)
    plt.close(fig)
    buf.seek(0)
    return buf


def generate_pdf_report(
    report_data: Dict[str, Any],
    user_name: str = "Athlete",
) -> bytes:
    """Build PDF byte stream matching all specifications in Chunk 10c."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        "VyronexTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#1e293b"),
    )
    subtitle_style = ParagraphStyle(
        "VyronexSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=13,
        textColor=colors.HexColor("#64748b"),
    )
    section_heading = ParagraphStyle(
        "VyronexSection",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#0f172a"),
        spaceBefore=8,
        spaceAfter=4,
    )
    body_text = ParagraphStyle(
        "VyronexBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#334155"),
    )
    bold_text = ParagraphStyle(
        "VyronexBold",
        parent=body_text,
        fontName="Helvetica-Bold",
    )
    disclaimer_style = ParagraphStyle(
        "VyronexDisclaimer",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#dc2626"),
        alignment=1,  # Center
    )

    story = []

    # 1. Header Banner
    story.append(Paragraph("<b>VYRONEX</b> Biomechanical Performance & Risk Report", title_style))
    created_str = datetime.now().strftime("%B %d, %Y - %H:%M UTC")
    session_id = str(report_data.get("session_id", "N/A"))[:12]
    activity = str(report_data.get("activity", "Exercise")).upper()
    duration = f"{report_data.get('duration_sec', 0)} seconds"

    story.append(
        Paragraph(
            f"<b>Athlete:</b> {user_name} &nbsp;|&nbsp; <b>Activity:</b> {activity} &nbsp;|&nbsp; "
            f"<b>Duration:</b> {duration} &nbsp;|&nbsp; <b>Session ID:</b> {session_id} &nbsp;|&nbsp; <b>Date:</b> {created_str}",
            subtitle_style,
        )
    )
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#cbd5e1"), spaceAfter=12))

    # 2. Executive Summary Metrics Table
    totals = report_data.get("totals", {})
    acc_pct = report_data.get("accuracy_pct", 100)
    avg_score = report_data.get("avg_form_score", 85)
    overall_risk = str(report_data.get("overall_risk", "LOW")).upper()

    risk_bg = colors.HexColor("#dcfce7") if overall_risk == "LOW" else (
        colors.HexColor("#fef3c7") if overall_risk == "MODERATE" else colors.HexColor("#fee2e2")
    )
    risk_tc = colors.HexColor("#15803d") if overall_risk == "LOW" else (
        colors.HexColor("#b45309") if overall_risk == "MODERATE" else colors.HexColor("#b91c1c")
    )

    summary_data = [
        [
            Paragraph("<b>Total Reps</b>", bold_text),
            Paragraph("<b>Correct (Low)</b>", bold_text),
            Paragraph("<b>Defective (Mod)</b>", bold_text),
            Paragraph("<b>High Risk</b>", bold_text),
            Paragraph("<b>Accuracy</b>", bold_text),
            Paragraph("<b>Form Score</b>", bold_text),
            Paragraph("<b>Overall Risk</b>", bold_text),
        ],
        [
            Paragraph(f"<b>{totals.get('reps', 0)}</b>", body_text),
            Paragraph(f"<font color='#16a34a'><b>{totals.get('correct', 0)}</b></font>", body_text),
            Paragraph(f"<font color='#d97706'><b>{totals.get('defective', 0)}</b></font>", body_text),
            Paragraph(f"<font color='#dc2626'><b>{totals.get('high_risk', 0)}</b></font>", body_text),
            Paragraph(f"<b>{acc_pct}%</b>", body_text),
            Paragraph(f"<b>{avg_score} / 100</b>", body_text),
            Paragraph(f"<b><font color='{risk_tc.hexval()}'>{overall_risk}</font></b>", bold_text),
        ],
    ]

    t_summary = Table(summary_data, colWidths=[70, 80, 85, 70, 65, 75, 80])
    t_summary.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("BACKGROUND", (6, 1), (6, 1), risk_bg),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(t_summary)
    story.append(Spacer(1, 12))

    # 3. Body-Area Defect Breakdown Chart
    story.append(Paragraph("Injury Risk & Body Area Breakdown", section_heading))
    body_areas = report_data.get("body_area_breakdown", {})
    chart_buf = _render_body_area_chart(body_areas)
    story.append(Image(chart_buf, width=480, height=135))
    story.append(Spacer(1, 10))

    # 4. Fatigue & Trend Analysis
    fatigue = report_data.get("fatigue", {})
    compared = report_data.get("compared_to_last")
    
    fatigue_text = (
        f"<b>Fatigue Analysis:</b> First half average score was <b>{fatigue.get('first_half_score', avg_score)}</b> vs "
        f"<b>{fatigue.get('second_half_score', avg_score)}</b> in the second half "
        f"(Drop: <b>{fatigue.get('drop_pct', 0)}%</b>). "
        f"Best Rep: #{report_data.get('best_rep', 1)} | Worst Rep: #{report_data.get('worst_rep', 1)}."
    )
    if compared:
        change = compared.get("accuracy_change_pct", 0)
        trend = compared.get("trend", "stable")
        fatigue_text += f"<br/><b>Progress vs Last Session:</b> {change:+}% accuracy ({trend})."

    story.append(Paragraph(fatigue_text, body_text))
    story.append(Spacer(1, 12))

    # 5. Rep-by-Rep Detail Table
    story.append(Paragraph("Rep-by-Rep Kinematic Log", section_heading))
    rep_table_data = report_data.get("rep_table", [])
    
    table_rows = [
        [
            Paragraph("<b>Rep #</b>", bold_text),
            Paragraph("<b>Time</b>", bold_text),
            Paragraph("<b>Status</b>", bold_text),
            Paragraph("<b>Primary Defect</b>", bold_text),
            Paragraph("<b>Body Area</b>", bold_text),
            Paragraph("<b>Risk</b>", bold_text),
            Paragraph("<b>Score</b>", bold_text),
        ]
    ]

    # Show up to 15 reps cleanly
    for r in rep_table_data[:15]:
        r_status = str(r.get("status", "")).lower()
        r_risk = str(r.get("risk", "LOW")).upper()
        status_color = "#16a34a" if r_status == "correct" else ("#d97706" if r_status == "defective" else "#dc2626")

        table_rows.append([
            Paragraph(str(r.get("rep")), body_text),
            Paragraph(str(r.get("time")), body_text),
            Paragraph(f"<font color='{status_color}'><b>{r_status.replace('_', ' ').title()}</b></font>", body_text),
            Paragraph(str(r.get("problem", "None"))[:28], body_text),
            Paragraph(str(r.get("body_area", "general")).replace("_", " ").title(), body_text),
            Paragraph(r_risk, bold_text),
            Paragraph(f"{r.get('form_score', 80)}", body_text),
        ])

    t_reps = Table(table_rows, colWidths=[40, 50, 80, 150, 80, 60, 60])
    t_reps.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f8fafc")),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("ALIGN", (0, 0), (1, -1), "CENTER"),
        ("ALIGN", (5, 0), (6, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(t_reps)
    story.append(Spacer(1, 12))

    # 6. AI Coach Recommendations
    recs = report_data.get("recommendations", [])
    if recs:
        story.append(Paragraph("AI Coach Corrective Guidance", section_heading))
        for rec in recs:
            story.append(Paragraph(f"• {rec}", body_text))
            story.append(Spacer(1, 2))
        story.append(Spacer(1, 10))

    # 7. Medical Disclaimer Footer
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e2e8f0"), spaceBefore=6, spaceAfter=8))
    story.append(Paragraph(f"<b>DISCLAIMER:</b> {report_data.get('disclaimer', 'Movement-risk screening only, not a medical diagnosis.')}", disclaimer_style))

    doc.build(story)
    pdf_bytes = buf.getvalue()
    buf.close()
    return pdf_bytes
