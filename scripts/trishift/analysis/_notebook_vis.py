from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def render_metric_barplot(
    summary_df: pd.DataFrame,
    *,
    id_col: str,
    metric_cols: list[str],
    out_path: str | Path,
    title: str,
) -> Path:
    out = Path(out_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, max(4, 0.85 * max(1, len(summary_df)))), dpi=220)
    if summary_df.empty or id_col not in summary_df.columns:
        plt.text(0.5, 0.5, "No summary rows", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out)
        plt.close()
        return out

    keep = [c for c in metric_cols if c in summary_df.columns]
    if not keep:
        plt.text(0.5, 0.5, "No metric columns", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out)
        plt.close()
        return out

    plot_df = summary_df[[id_col, *keep]].copy()
    plot_df = plot_df.set_index(id_col)
    plot_df = plot_df.apply(pd.to_numeric, errors="coerce")
    ax = plt.gca()
    plot_df.T.plot(kind="bar", ax=ax, width=0.82)
    plt.ylabel("mean score")
    plt.title(title)
    plt.grid(axis="y", alpha=0.2)
    plt.xticks(rotation=30, ha="right")
    plt.legend(title=id_col, frameon=False, ncol=min(4, len(plot_df.index)))
    for container in getattr(ax, "containers", []):
        labels = []
        for bar in container:
            height = float(bar.get_height())
            labels.append("" if pd.isna(height) else f"{height:.3f}")
        ax.bar_label(container, labels=labels, fontsize=8, rotation=0, padding=3)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    return out


def render_image_panel(
    image_items: list[dict[str, str]],
    *,
    out_path: str | Path,
    title: str,
    ncols: int = 2,
) -> Path:
    out = Path(out_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    valid = []
    for item in image_items:
        path = Path(str(item.get("path", ""))).resolve()
        if path.exists():
            valid.append({"label": str(item.get("label", path.stem)), "path": path})

    if not valid:
        plt.figure(figsize=(8, 4), dpi=220)
        plt.text(0.5, 0.5, "No panel images available", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out)
        plt.close()
        return out

    ncols = max(1, int(ncols))
    nrows = int(math.ceil(len(valid) / float(ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5.2, nrows * 4.2), dpi=220)
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
    for ax, item in zip(axes, valid):
        img = plt.imread(item["path"])
        ax.imshow(img)
        ax.set_title(item["label"], fontsize=10)
        ax.axis("off")
    for ax in axes[len(valid):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    return out
