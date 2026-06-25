#!/usr/bin/env python3
"""Render editable Fig. 1 notation snippets as standalone SVG files.

These SVGs are intended for manual schematic editing in tools such as
Illustrator, Figma or Keynote. Text is converted to paths to avoid font
substitution issues when the figure is opened on another machine.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "artifacts/paper_figures/main/Fig1_MethodOverview/symbol_svgs"


SYMBOLS: list[tuple[str, str, str]] = [
    ("control_cell", r"$x_i^0$", "control-cell expression"),
    ("perturbed_response", r"$y_j$", "observed perturbed response"),
    ("perturbation_label", r"$p_j$", "perturbation label"),
    ("perturbation_prior", r"$e_j$", "fixed perturbation-prior vector"),
    ("retrieval_repr_generic", r"$r=R_\phi(x)$", "generic retrieval representation"),
    ("retrieval_repr_control", r"$r_i^0$", "control retrieval representation"),
    ("retrieval_repr_perturbed", r"$r_j^1$", "perturbed/query retrieval representation"),
    ("reference_pool", r"$\mathcal{M}_j$", "top-k reference pool"),
    ("reference_index", r"$m_j$", "sampled reference index"),
    ("sampled_reference", r"$x_{m_j}^{0}$", "sampled reference-control expression"),
    ("state_encoding", r"$z_{\mathrm{state},m_j}^{0}$", "generator-side reference state encoding"),
    ("shift_repr", r"$h_j$", "conditional shift representation"),
    ("predicted_response", r"$\hat{y}_j$", "predicted perturbed response"),
    ("ot_plan", r"$P^\ast$", "optimal-transport coupling plan"),
    ("encoder_posterior", r"$q_\phi(z\mid\tilde{x})$", "denoising VAE posterior"),
    (
        "generation_formula",
        r"$\hat{y}_j=x_{m_j}^{0}+f_{\mathrm{gen}}\!\left(z_{\mathrm{state},m_j}^{0},e_j,h_j\right)$",
        "reference-conditioned residual generation formula",
    ),
]


def render_symbol(name: str, tex: str) -> None:
    fig = plt.figure(figsize=(4.8, 1.05), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.text(0.5, 0.5, tex, ha="center", va="center", fontsize=32, color="#273043")
    out = OUT_DIR / f"{name}.svg"
    fig.savefig(out, format="svg", transparent=True, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def render_contact_sheet() -> None:
    rows = len(SYMBOLS)
    fig, axes = plt.subplots(rows, 1, figsize=(12.0, 0.52 * rows), dpi=220)
    if rows == 1:
        axes = [axes]
    for ax, (name, tex, desc) in zip(axes, SYMBOLS):
        ax.axis("off")
        ax.text(0.02, 0.5, name, ha="left", va="center", fontsize=7.5, color="#667085")
        size = 16 if name != "generation_formula" else 12
        ax.text(0.42, 0.5, tex, ha="center", va="center", fontsize=size, color="#273043")
        ax.text(0.70, 0.5, desc, ha="left", va="center", fontsize=7.5, color="#667085")
    fig.savefig(OUT_DIR / "fig1_symbol_contact_sheet.png", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def main() -> None:
    mpl.rcParams["svg.fonttype"] = "path"
    mpl.rcParams["mathtext.fontset"] = "dejavusans"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, tex, _ in SYMBOLS:
        render_symbol(name, tex)
    with (OUT_DIR / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "tex", "description"])
        for name, tex, desc in SYMBOLS:
            writer.writerow([f"{name}.svg", tex, desc])
    render_contact_sheet()
    print(OUT_DIR)


if __name__ == "__main__":
    main()
