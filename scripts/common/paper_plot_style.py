from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt


_GEARS_STYLE_APPLIED = False


MODEL_COLORS = {
    "TriShift": "#4C72B0",
    "TriShift nearest": "#4C72B0",
    "TriShift random": "#9ECAE1",
    "GEARS": "#55A868",
    "biolord": "#C44E52",
    "GenePert": "#8172B2",
    "scGPT": "#CCB974",
    "Scouter": "#64B5CD",
    "Systema nonctl-mean": "#7F7F7F",
    "Systema matching-mean": "#A9A9A9",
    "Mean perturbation": "#8C8C8C",
    "Experimental data": "#2F2F2F",
}

CLUSTER_COLORS = ["#C9CDD1", "#F28E2B", "#4E9F50", "#E15759", "#5E81AC", "#B07AA1"]


def apply_gears_paper_style(font_scale: float = 1.0) -> None:
    global _GEARS_STYLE_APPLIED
    if _GEARS_STYLE_APPLIED:
        return
    base_font = max(8.0, 10.0 * float(font_scale))
    plt.rcParams.update(
        {
            "figure.dpi": 220,
            "savefig.dpi": 220,
            "savefig.bbox": "tight",
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": base_font,
            "axes.labelsize": base_font,
            "axes.titlesize": base_font + 0.5,
            "axes.titleweight": "regular",
            "axes.facecolor": (0, 0, 0, 0),
            "figure.facecolor": "white",
            "axes.edgecolor": "#4A4A4A",
            "axes.linewidth": 0.8,
            "axes.grid": False,
            "xtick.labelsize": base_font - 0.5,
            "ytick.labelsize": base_font - 0.5,
            "legend.frameon": False,
            "lines.linewidth": 1.6,
            "patch.linewidth": 0.8,
        }
    )
    _GEARS_STYLE_APPLIED = True


def model_color_map(model_names: list[str]) -> dict[str, Any]:
    apply_gears_paper_style()
    fallback = plt.get_cmap("tab10")
    out: dict[str, Any] = {}
    for idx, name in enumerate(model_names):
        out[name] = MODEL_COLORS.get(name, fallback(idx % 10))
    return out


def style_axis(ax, *, grid_axis: str | None = None) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#4A4A4A")
    ax.spines["bottom"].set_color("#4A4A4A")
    ax.tick_params(length=3.0, width=0.8, colors="#3A3A3A")
    if grid_axis:
        ax.grid(axis=grid_axis, color="#D5D8DC", linewidth=0.7, alpha=0.35)

