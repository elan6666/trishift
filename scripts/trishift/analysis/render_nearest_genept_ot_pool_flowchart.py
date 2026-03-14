from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph
from reportlab.pdfgen import canvas


REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = REPO_ROOT / "output" / "pdf"
OUT_PATH = OUT_DIR / "trishift_nearest_genept_ot_pool_flowchart.pdf"


def _draw_round_box(c: canvas.Canvas, x: float, y: float, w: float, h: float, text: str, fill_color) -> None:
    c.setStrokeColor(colors.HexColor("#3A4A5A"))
    c.setFillColor(fill_color)
    c.roundRect(x, y, w, h, 6 * mm, stroke=1, fill=1)
    styles = getSampleStyleSheet()
    style = ParagraphStyle(
        "flow",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=10,
        leading=12,
        alignment=1,
        textColor=colors.HexColor("#13202B"),
    )
    p = Paragraph(text, style=style)
    tw, th = p.wrap(w - 10 * mm, h - 8 * mm)
    p.drawOn(c, x + (w - tw) / 2, y + (h - th) / 2)


def _arrow(c: canvas.Canvas, x1: float, y1: float, x2: float, y2: float) -> None:
    c.setStrokeColor(colors.HexColor("#5A6D7F"))
    c.setLineWidth(1.5)
    c.line(x1, y1, x2, y2)
    angle = 0.0
    import math

    angle = math.atan2(y2 - y1, x2 - x1)
    ah = 5 * mm
    aw = 2.6 * mm
    x3 = x2 - ah * math.cos(angle) + aw * math.sin(angle)
    y3 = y2 - ah * math.sin(angle) - aw * math.cos(angle)
    x4 = x2 - ah * math.cos(angle) - aw * math.sin(angle)
    y4 = y2 - ah * math.sin(angle) + aw * math.cos(angle)
    c.setFillColor(colors.HexColor("#5A6D7F"))
    c.setStrokeColor(colors.HexColor("#5A6D7F"))
    c.line(x2, y2, x3, y3)
    c.line(x2, y2, x4, y4)


def render() -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(OUT_PATH), pagesize=landscape(A4))
    width, height = landscape(A4)

    c.setTitle("TriShift nearest_genept_ot_pool flowchart")
    c.setAuthor("OpenAI Codex")

    c.setFont("Helvetica-Bold", 20)
    c.drawString(24 * mm, height - 18 * mm, "TriShift inference innovation: nearest_genept_ot_pool")
    c.setFont("Helvetica", 10)
    c.setFillColor(colors.HexColor("#334455"))
    c.drawString(
        24 * mm,
        height - 25 * mm,
        "Condition-aware retrieval of OT control pools guided by GenePT condition similarity",
    )

    box_w = 40 * mm
    box_h = 24 * mm
    y_top = height - 60 * mm
    y_mid = height - 102 * mm
    y_bottom = height - 145 * mm

    xs = [14 * mm, 59 * mm, 104 * mm, 149 * mm, 194 * mm]

    _draw_round_box(
        c,
        xs[0],
        y_top,
        box_w,
        box_h,
        "Train split perturbation cells\n+ train control pool",
        colors.HexColor("#E8F1FB"),
    )
    _draw_round_box(
        c,
        xs[1],
        y_top,
        box_w,
        box_h,
        "Build train-side top-k OT map\nfor each train perturbation condition",
        colors.HexColor("#EAF7EE"),
    )
    _draw_round_box(
        c,
        xs[2],
        y_top,
        box_w,
        box_h,
        "Encode train/test conditions\nin GenePT space",
        colors.HexColor("#FFF3E3"),
    )
    _draw_round_box(
        c,
        xs[3],
        y_top,
        box_w,
        box_h,
        "Find nearest train condition(s)\nfor each test condition",
        colors.HexColor("#F6EAF7"),
    )
    _draw_round_box(
        c,
        xs[4],
        y_top,
        box_w,
        box_h,
        "Reuse the matched train OT pool\nas the evaluation control pool",
        colors.HexColor("#FDECEC"),
    )

    for i in range(4):
        _arrow(c, xs[i] + box_w, y_top + box_h / 2, xs[i + 1], y_top + box_h / 2)

    _draw_round_box(
        c,
        43 * mm,
        y_mid,
        58 * mm,
        box_h,
        "aggregate_cond\nCompare pooled condition embeddings",
        colors.HexColor("#EAF3FF"),
    )
    _draw_round_box(
        c,
        125 * mm,
        y_mid,
        72 * mm,
        box_h,
        "per_gene_nearest_cond\nEach perturbed gene token retrieves\nits own nearest train condition",
        colors.HexColor("#FFF8E8"),
    )

    _arrow(c, xs[3] + box_w / 2, y_top, 72 * mm, y_mid + box_h)
    _arrow(c, xs[3] + box_w / 2, y_top, 161 * mm, y_mid + box_h)

    _draw_round_box(
        c,
        28 * mm,
        y_bottom,
        72 * mm,
        box_h,
        "Candidate train conditions can be filtered\nfor example Norman train-single only",
        colors.HexColor("#EEF7F9"),
    )
    _draw_round_box(
        c,
        117 * mm,
        y_bottom,
        66 * mm,
        box_h,
        "Distance metric\ncosine or L2 in GenePT space",
        colors.HexColor("#EEF7F9"),
    )
    _draw_round_box(
        c,
        196 * mm,
        y_bottom,
        52 * mm,
        box_h,
        "Sample control expressions from the\nretrieved pool and run ensemble inference",
        colors.HexColor("#EAF7EE"),
    )

    _arrow(c, 64 * mm, y_mid, 64 * mm, y_bottom + box_h)
    _arrow(c, 161 * mm, y_mid, 150 * mm, y_bottom + box_h)
    _arrow(c, xs[4] + box_w / 2, y_top, 222 * mm, y_bottom + box_h)

    c.setFont("Helvetica-Oblique", 9)
    c.setFillColor(colors.HexColor("#556677"))
    c.drawString(
        24 * mm,
        12 * mm,
        "This is an inference-stage innovation: model parameters stay fixed; only the evaluation control pool changes.",
    )

    c.save()
    return OUT_PATH


if __name__ == "__main__":
    path = render()
    print(path)
