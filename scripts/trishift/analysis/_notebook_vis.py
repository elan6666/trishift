from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from scripts.common.paper_plot_style import apply_gears_paper_style, model_color_map, style_axis


DEFAULT_MODEL_ORDER = [
    "TriShift",
    "CellOT",
    "GEARS",
    "BioLORD",
    "GenePert",
    "scGPT",
    "Scouter",
    "Systema nonctl-mean",
    "Systema matching-mean",
]


def read_metric_long(paths: list[str | Path] | str | Path) -> pd.DataFrame:
    """Read one or more long-format metric CSVs and keep the paper-facing schema stable."""
    if isinstance(paths, (str, Path)):
        path_list = [paths]
    else:
        path_list = list(paths)
    frames = []
    required = {"dataset", "setting", "split", "condition", "model", "metric", "value"}
    for path in path_list:
        p = Path(path)
        if not p.exists() or p.stat().st_size <= 0:
            continue
        df = pd.read_csv(p)
        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(f"{p} is missing long metric columns: {missing}")
        df = df.copy()
        df["source_file"] = str(p)
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=[*sorted(required), "source_file"])
    out = pd.concat(frames, ignore_index=True)
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    return out


def ordered_models(models: list[str] | pd.Series, preferred: list[str] | None = None) -> list[str]:
    values = [str(x) for x in pd.Series(models).dropna().unique().tolist()]
    order = preferred or DEFAULT_MODEL_ORDER
    return [m for m in order if m in values] + sorted([m for m in values if m not in order])


def summarize_metric_long(df: pd.DataFrame, *, group_cols: list[str] | None = None) -> pd.DataFrame:
    """Summarize long metrics while preserving split/condition-level provenance."""
    if df.empty:
        return pd.DataFrame()
    group_cols = group_cols or ["dataset", "setting", "model", "metric"]
    out = (
        df.groupby(group_cols, as_index=False)
        .agg(
            value_mean=("value", "mean"),
            value_median=("value", "median"),
            value_sem=("value", "sem"),
            n=("value", "count"),
        )
        .reset_index(drop=True)
    )
    return out


def write_panel_source(df: pd.DataFrame, out_path: str | Path) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return out


def render_compact_metric_points(
    long_df: pd.DataFrame,
    *,
    metric: str,
    out_path: str | Path,
    title: str,
    dataset_order: list[str] | None = None,
    model_order: list[str] | None = None,
    ylabel: str | None = None,
) -> Path:
    """Render a compact paper-style point plot with split/condition points and mean markers."""
    apply_gears_paper_style(font_scale=0.9)
    out = Path(out_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    df = long_df[long_df["metric"].astype(str).eq(str(metric))].copy()
    if df.empty:
        plt.figure(figsize=(4.2, 2.4), dpi=220)
        plt.text(0.5, 0.5, f"No rows for {metric}", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out)
        plt.close()
        return out
    dataset_vals = dataset_order or sorted(df["dataset"].astype(str).unique().tolist())
    model_vals = model_order or ordered_models(df["model"])
    colors = model_color_map(model_vals)
    fig, ax = plt.subplots(figsize=(max(4.0, 1.15 * len(dataset_vals)), 2.8), dpi=220)
    width = 0.7 / max(1, len(model_vals))
    x_lookup = {d: i for i, d in enumerate(dataset_vals)}
    for midx, model in enumerate(model_vals):
        sub = df[df["model"].astype(str).eq(model)]
        offset = (midx - (len(model_vals) - 1) / 2.0) * width
        xs = sub["dataset"].astype(str).map(x_lookup).astype(float) + offset
        ax.scatter(
            xs,
            sub["value"],
            s=9,
            alpha=0.32,
            color=colors.get(model),
            linewidths=0,
        )
        means = sub.groupby("dataset")["value"].mean()
        sems = sub.groupby("dataset")["value"].sem()
        for dataset in dataset_vals:
            if dataset not in means:
                continue
            ax.errorbar(
                x_lookup[dataset] + offset,
                float(means[dataset]),
                yerr=float(sems.get(dataset, 0.0)) if pd.notna(sems.get(dataset, 0.0)) else 0.0,
                marker="o",
                markersize=3.8,
                color=colors.get(model),
                capsize=2,
                linewidth=0.8,
                label=model if dataset == dataset_vals[0] else None,
            )
    ax.set_xticks(range(len(dataset_vals)))
    ax.set_xticklabels(dataset_vals, rotation=25, ha="right")
    ax.set_ylabel(ylabel or metric)
    ax.set_title(title, pad=3)
    style_axis(ax, grid_axis="y")
    ax.legend(frameon=False, fontsize=7, ncol=min(3, len(model_vals)), handlelength=1.2)
    fig.tight_layout(pad=0.4)
    fig.savefig(out)
    plt.close(fig)
    return out


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
