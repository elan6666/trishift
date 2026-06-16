from __future__ import annotations

import argparse
import math
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageChops, ImageDraw, ImageFont

try:
    import seaborn as sns  # noqa: F401
except Exception:  # pragma: no cover - plotting must also work in lean local envs
    sns = None

from scripts.common.paper_plot_style import apply_gears_paper_style, model_color_map, style_axis

try:
    from scripts.trishift.analysis._result_adapter import load_payload_item
except Exception:  # pragma: no cover - optional during lightweight artifact QA
    load_payload_item = None


FIG_ROOT = ROOT / "artifacts" / "paper_figures"
COMP_ROOT = FIG_ROOT / "composites"
SERVER_REFRESH_ROOT = ROOT / "artifacts" / "server_refresh" / "extracted"

DATASET_ORDER = ["Adamson", "Dixit", "Norman", "PBMC"]
MODEL_ORDER = ["TriShift", "CellOT", "GEARS", "BioLORD", "GenePert", "scGPT"]
TRISHIFT_MODEL_ORDER = ["TriShift", "kNN", "OT", "no reference", "no prior", "full"]
SUBGROUP_ORDER = ["single", "seen2", "seen1", "seen0"]
DISPLAY_COLORS = {
    "TriShift": "#9FD9D3",
    "CellOT": "#DD8452",
    "GEARS": "#F2B56B",
    "BioLORD": "#F0806A",
    "GenePert": "#87A8D8",
    "scGPT": "#DDD3C8",
    "Truth": "#7F7F7F",
    "Control": "#CFCFCF",
    "Perturbed": "#7F7F7F",
    "random": "#C8DCEB",
    "kNN": "#B7C8A6",
    "OT": "#9FD9D3",
    "no reference": "#C9CDD1",
    "no prior": "#D8BFD8",
    "full": "#9FD9D3",
}
RESULT_MODE = "unseen_ctrl"
SUPP_DIRS = {
    "figs1": "FigS1_ReferenceTransferFull",
    "figs2": "FigS2_DixitStability",
    "figs3": "FigS3_AdditionalCases",
    "figs4": "FigS4_DixitRobustness",
    "figs5": "FigS5_Module1LatentState",
}


def _display_dataset(name: str) -> str:
    lookup = {
        "adamson": "Adamson",
        "dixit": "Dixit",
        "norman": "Norman",
        "scgen_pbmc_celltype": "PBMC",
        "pbmc": "PBMC",
    }
    return lookup.get(str(name).lower(), str(name))


def _display_model(name: str) -> str:
    lookup = {
        "trishift": "TriShift",
        "trishift_nearest": "TriShift",
        "trishift_random": "TriShift random",
        "systema_nonctl_mean": "Perturbed mean",
        "perturbed mean": "Perturbed mean",
        "cellot": "CellOT",
        "gears": "GEARS",
        "biolord": "BioLORD",
        "genepert": "GenePert",
        "scgpt": "scGPT",
    }
    return lookup.get(str(name).lower(), str(name))


def _relative_artifact_path(path: Path) -> Path | None:
    path = Path(path)
    try:
        return path.relative_to(ROOT)
    except ValueError:
        pass
    parts = list(path.parts)
    if "trishift" in parts:
        idx = len(parts) - 1 - parts[::-1].index("trishift")
        if idx + 1 < len(parts):
            return Path(*parts[idx + 1 :])
    if "artifacts" in parts:
        idx = parts.index("artifacts")
        return Path(*parts[idx:])
    return path if not path.is_absolute() else None


def _candidate_paths(path: Path) -> list[Path]:
    path = Path(path)
    candidates = [path]
    rel = _relative_artifact_path(path)
    if rel is not None:
        candidates.append(ROOT / rel)
        candidates.append(SERVER_REFRESH_ROOT / rel)
    unique: list[Path] = []
    seen: set[str] = set()
    for item in candidates:
        key = str(item)
        if key not in seen:
            unique.append(item)
            seen.add(key)
    return unique


def _read_csv(path: Path) -> pd.DataFrame:
    for candidate in _candidate_paths(path):
        if not candidate.exists() or candidate.stat().st_size == 0:
            continue
        try:
            return pd.read_csv(candidate)
        except Exception:
            continue
    return pd.DataFrame()


def _write_source(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def collect_prediction_metrics(*, heldout: bool = True) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    suffix = "metrics_unseen_ctrl.csv" if heldout else "metrics.csv"
    specs = []
    for ds in ["adamson", "dixit", "norman"]:
        specs.append((ROOT / "artifacts" / "results" / ds / suffix, "TriShift", ds))
        for model in ["gears", "biolord", "genepert", "scgpt"]:
            specs.append((ROOT / "artifacts" / "results" / model / ds / suffix, _display_model(model), ds))
    # PBMC uses a single stimulation condition and has model-specific paths.
    specs.extend(
        [
            (ROOT / "artifacts" / "results" / "scgen_pbmc_celltype" / "metrics.csv", "TriShift", "PBMC"),
            (
                ROOT
                / "artifacts"
                / "results"
                / "biolord"
                / "scgen_pbmc_celltype"
                / "emb_scgen_ifnb1_zenodo_prott5"
                / "metrics.csv",
                "BioLORD",
                "PBMC",
            ),
            (ROOT / "artifacts" / "results" / "scgpt" / "scgen_pbmc_celltype" / "metrics.csv", "scGPT", "PBMC"),
        ]
    )
    for path, model, dataset in specs:
        df = _read_csv(path)
        if df.empty:
            continue
        df = df.copy()
        df["model"] = model
        df["dataset"] = _display_dataset(dataset)
        df["source_file"] = str(path.relative_to(ROOT))
        rows.append(df)

    if heldout:
        for ds in ["adamson", "dixit", "norman", "scgen_pbmc_celltype"]:
            cellot_path = ROOT / "artifacts" / "results" / "cellot" / ds / "metrics_unseen_ctrl.csv"
            cellot = _read_csv(cellot_path)
            if cellot.empty:
                continue
            cellot = cellot.copy()
            cellot["model"] = "CellOT"
            if "dataset" not in cellot.columns:
                cellot["dataset"] = ds
            cellot["dataset"] = cellot["dataset"].map(_display_dataset)
            cellot["source_file"] = str(cellot_path.relative_to(ROOT))
            rows.append(cellot)
        legacy_cellot = _read_csv(ROOT / "artifacts" / "results" / "cellot" / "metrics_unseen_ctrl.csv")
        if not legacy_cellot.empty and not any(
            (ROOT / "artifacts" / "results" / "cellot" / ds / "metrics_unseen_ctrl.csv").exists()
            for ds in ["adamson", "dixit", "norman", "scgen_pbmc_celltype"]
        ):
            legacy_cellot = legacy_cellot.copy()
            legacy_cellot["model"] = "CellOT"
            legacy_cellot["dataset"] = legacy_cellot["dataset"].map(_display_dataset)
            legacy_cellot["source_file"] = "artifacts/results/cellot/metrics_unseen_ctrl.csv"
            rows.append(legacy_cellot)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    if {"dataset", "condition", "subgroup"}.issubset(out.columns):
        keyed = out.dropna(subset=["dataset", "condition", "subgroup"]).copy()
        if not keyed.empty:
            subgroup_map = keyed.drop_duplicates(["dataset", "condition"]).set_index(["dataset", "condition"])["subgroup"].to_dict()
            missing = out["subgroup"].isna()
            out.loc[missing, "subgroup"] = [
                subgroup_map.get((dataset, condition))
                for dataset, condition in zip(out.loc[missing, "dataset"], out.loc[missing, "condition"])
            ]
    for col in ["pearson", "nmse", "mse_pred", "systema_corr_20de_allpert", "scpram_wasserstein_degs_sum"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def collect_long(path: Path) -> pd.DataFrame:
    df = _read_csv(path)
    if df.empty:
        return df
    df = df.copy()
    if "dataset" in df.columns:
        df["dataset"] = df["dataset"].map(_display_dataset)
    if "model" in df.columns:
        df["model"] = df["model"].map(_display_model)
    if "value" in df.columns:
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df


def collect_ablation_metrics() -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    seen: set[tuple[str, str]] = set()
    ablation_roots = [
        ROOT / "artifacts" / "results" / "ablation",
        SERVER_REFRESH_ROOT / "artifacts" / "results" / "ablation",
    ]
    for ablation_root in ablation_roots:
        if not ablation_root.exists():
            continue
        for preset_dir in sorted(p for p in ablation_root.glob("*/*") if p.is_dir()):
            parts = preset_dir.relative_to(ablation_root).parts
            if len(parts) < 2 or parts[0] == "plan":
                continue
            dataset, preset = parts[0], parts[1]
            key = (dataset, preset)
            if key in seen:
                continue
            p = preset_dir / "metrics_unseen_ctrl.csv"
            if not p.exists():
                p = preset_dir / "metrics.csv"
            if not p.exists():
                continue
            seen.add(key)
            df = _read_csv(p)
            if df.empty:
                continue
            df = df.copy()
            df["dataset"] = _display_dataset(dataset)
            df["preset"] = preset
            rel = _relative_artifact_path(p)
            df["source_file"] = str(rel if rel is not None else p)
            rows.append(df)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    for col in ["pearson", "nmse", "systema_corr_20de_allpert", "mse_pred"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def available_order(values: pd.Series, preferred: list[str]) -> list[str]:
    present = [str(x) for x in values.dropna().unique()]
    return [x for x in preferred if x in present] + sorted([x for x in present if x not in preferred])


def no_data_panel(out: Path, title: str, message: str = "Required source table is unavailable") -> Path:
    apply_gears_paper_style(font_scale=0.85)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.4, 2.8), dpi=240)
    ax.text(0.5, 0.58, title, ha="center", va="center", fontsize=10, weight="bold")
    ax.text(0.5, 0.42, message, ha="center", va="center", fontsize=8, color="#555555")
    ax.axis("off")
    fig.tight_layout(pad=0.4)
    fig.savefig(out)
    plt.close(fig)
    return out


def _color_map(names: list[str]) -> dict[str, object]:
    cmap = model_color_map(names)
    for name in names:
        if name in DISPLAY_COLORS:
            cmap[name] = DISPLAY_COLORS[name]
    return cmap


def _legend_above(
    ax,
    handles=None,
    labels=None,
    *,
    ncol: int | None = None,
    fontsize: float = 6.2,
    y: float = 1.18,
) -> None:
    if handles is None or labels is None:
        handles, labels = ax.get_legend_handles_labels()
    labels = list(labels)
    if not labels:
        return
    ncol = ncol or min(4, max(1, len(labels)))
    ax.legend(
        handles,
        labels,
        frameon=False,
        fontsize=fontsize,
        ncol=ncol,
        loc="upper center",
        bbox_to_anchor=(0.5, y),
        borderaxespad=0.0,
        handlelength=0.95,
        columnspacing=0.75,
        handletextpad=0.35,
    )


def _metric_plot_frame(
    df: pd.DataFrame,
    metric_col: str,
    *,
    x_col: str = "dataset",
    hue_col: str = "model",
) -> pd.DataFrame:
    if df.empty or metric_col not in df.columns or x_col not in df.columns or hue_col not in df.columns:
        return pd.DataFrame()
    plot = df[[x_col, hue_col, metric_col]].copy()
    plot[metric_col] = pd.to_numeric(plot[metric_col], errors="coerce")
    plot = plot.dropna(subset=[x_col, hue_col, metric_col])
    plot[x_col] = plot[x_col].astype(str)
    plot[hue_col] = plot[hue_col].astype(str).map(_display_model)
    return plot


def _visible_limit(values: pd.Series) -> float | None:
    vals = pd.to_numeric(values, errors="coerce").dropna()
    if vals.empty:
        return None
    finite = vals[np.isfinite(vals)]
    if finite.empty:
        return None
    positive = finite[finite > 0]
    if len(positive) < 3:
        return None
    q1, q3 = np.percentile(positive, [25, 75])
    iqr = max(q3 - q1, 1e-12)
    robust = q3 + 5.0 * iqr
    normal = positive[positive <= robust]
    if normal.empty:
        normal = positive
    limit = float(normal.max() * 1.22)
    if float(positive.max()) > max(limit * 2.2, limit + 1e-9):
        return limit
    return None


def compact_bar_panel(
    df: pd.DataFrame,
    metric_col: str,
    out: Path,
    title: str,
    ylabel: str,
    *,
    x_col: str = "dataset",
    hue_col: str = "model",
    x_order: list[str] | None = None,
    hue_order: list[str] | None = None,
    cap_extreme: bool = False,
    figsize: tuple[float, float] = (5.2, 3.9),
) -> Path:
    plot = _metric_plot_frame(df, metric_col, x_col=x_col, hue_col=hue_col)
    if plot.empty:
        return no_data_panel(out, title)
    summary = plot.groupby([x_col, hue_col], as_index=False).agg(mean=(metric_col, "mean"), sem=(metric_col, "sem"), n=(metric_col, "size"))
    xs = x_order or available_order(summary[x_col], DATASET_ORDER if x_col == "dataset" else SUBGROUP_ORDER)
    hues = hue_order or available_order(summary[hue_col], MODEL_ORDER)
    xs = [x for x in xs if x in set(summary[x_col])]
    hues = [h for h in hues if h in set(summary[hue_col])]
    if not xs or not hues:
        return no_data_panel(out, title, "No plottable groups")
    apply_gears_paper_style(font_scale=0.82)
    colors = _color_map(hues)
    fig, ax = plt.subplots(figsize=figsize, dpi=240)
    group_width = 0.72
    base_width = min(0.13, group_width / max(len(hues), 1))
    bar_width = base_width * 0.82
    offsets = [0.0] if len(hues) == 1 else np.linspace(
        -base_width * (len(hues) - 1) / 2,
        base_width * (len(hues) - 1) / 2,
        len(hues),
    )
    y_cap = _visible_limit(summary["mean"]) if cap_extreme else None
    for hidx, hue in enumerate(hues):
        hue_df = summary[summary[hue_col].eq(hue)]
        for xidx, xval in enumerate(xs):
            row = hue_df[hue_df[x_col].eq(xval)]
            if row.empty:
                continue
            value = float(row["mean"].iloc[0])
            sem = row["sem"].iloc[0]
            draw_value = min(value, y_cap) if y_cap is not None else value
            ax.bar(
                xidx + float(offsets[hidx]),
                draw_value,
                width=bar_width,
                color=colors.get(hue),
                edgecolor="black",
                linewidth=0.5,
                label=hue if xidx == 0 else None,
            )
            if pd.notna(sem) and not (y_cap is not None and value > y_cap):
                ax.errorbar(
                    xidx + float(offsets[hidx]),
                    draw_value,
                    yerr=float(sem),
                    color="#333333",
                    linewidth=0.5,
                    capsize=1.4,
                    capthick=0.5,
                )
            if y_cap is not None and value > y_cap:
                ax.text(xidx + float(offsets[hidx]), y_cap, ">", ha="center", va="bottom", fontsize=7, color="#222222")
    ax.set_xticks(np.arange(len(xs)))
    ax.set_xticklabels(xs, rotation=32, ha="right")
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=13)
    if y_cap is not None:
        ax.set_ylim(top=y_cap * 1.18)
    else:
        vals = summary["mean"].dropna()
        if not vals.empty and vals.min() >= 0:
            ax.set_ylim(bottom=0)
    style_axis(ax, grid_axis="y")
    _legend_above(ax, ncol=min(4, max(1, len(hues))), y=1.32)
    fig.tight_layout(pad=0.35, rect=(0, 0, 1, 0.78))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    _write_source(summary, out.with_suffix(".csv"))
    return out


def boxplot_panel(
    df: pd.DataFrame,
    metric_col: str,
    out: Path,
    title: str,
    ylabel: str,
    *,
    x_col: str = "dataset",
    hue_col: str = "model",
    x_order: list[str] | None = None,
    hue_order: list[str] | None = None,
    cap_extreme: bool = False,
) -> Path:
    plot = _metric_plot_frame(df, metric_col, x_col=x_col, hue_col=hue_col)
    if plot.empty:
        return no_data_panel(out, title)
    xs = x_order or available_order(plot[x_col], DATASET_ORDER if x_col == "dataset" else SUBGROUP_ORDER)
    hues = hue_order or available_order(plot[hue_col], MODEL_ORDER)
    xs = [x for x in xs if x in set(plot[x_col])]
    hues = [h for h in hues if h in set(plot[hue_col])]
    if not xs or not hues:
        return no_data_panel(out, title, "No plottable groups")
    apply_gears_paper_style(font_scale=0.78)
    colors = _color_map(hues)
    fig, ax = plt.subplots(figsize=(5.4, 3.6), dpi=240)
    base_width = min(0.13, 0.76 / max(len(hues), 1))
    box_width = base_width * 0.85
    offsets = [0.0] if len(hues) == 1 else np.linspace(
        -base_width * (len(hues) - 1) / 2,
        base_width * (len(hues) - 1) / 2,
        len(hues),
    )
    y_cap = _visible_limit(plot[metric_col]) if cap_extreme else None
    for hidx, hue in enumerate(hues):
        series = []
        positions = []
        for xidx, xval in enumerate(xs):
            vals = plot[(plot[x_col].eq(xval)) & (plot[hue_col].eq(hue))][metric_col].astype(float).dropna()
            if vals.empty:
                continue
            if y_cap is not None:
                vals = vals.clip(upper=y_cap)
            series.append(vals.values)
            positions.append(xidx + float(offsets[hidx]))
        if not series:
            continue
        bp = ax.boxplot(
            series,
            positions=positions,
            widths=box_width,
            patch_artist=True,
            manage_ticks=False,
            boxprops={"linewidth": 0.65, "edgecolor": "#444444"},
            medianprops={"linewidth": 0.8, "color": "#333333"},
            whiskerprops={"linewidth": 0.65, "color": "#555555"},
            capprops={"linewidth": 0.65, "color": "#555555"},
            flierprops={"marker": ".", "markersize": 2.0, "markerfacecolor": "#555555", "markeredgewidth": 0, "alpha": 0.55},
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(colors.get(hue))
            patch.set_alpha(0.95)
    ax.set_xticks(np.arange(len(xs)))
    ax.set_xticklabels(xs, rotation=28, ha="right")
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=12)
    if y_cap is not None:
        ax.set_ylim(top=y_cap * 1.18)
    style_axis(ax, grid_axis="y")
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=colors[h], edgecolor="#444444", linewidth=0.65) for h in hues]
    _legend_above(ax, handles, hues, ncol=min(4, len(hues)), y=1.32)
    fig.tight_layout(pad=0.35, rect=(0, 0, 1, 0.78))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    _write_source(plot, out.with_suffix(".csv"))
    return out


def long_metric_frame(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    if df.empty or "metric" not in df.columns:
        return pd.DataFrame()
    sub = df[df["metric"].astype(str).eq(metric)].copy()
    if sub.empty:
        return pd.DataFrame()
    return sub.rename(columns={"value": metric})


def long_metric_bar_panel(df: pd.DataFrame, metric: str, out: Path, title: str, ylabel: str, **kwargs) -> Path:
    sub = long_metric_frame(df, metric)
    if sub.empty:
        return no_data_panel(out, title, f"No rows for {metric}")
    return compact_bar_panel(sub, metric, out, title, ylabel, **kwargs)


def long_metric_box_panel(df: pd.DataFrame, metric: str, out: Path, title: str, ylabel: str, **kwargs) -> Path:
    sub = long_metric_frame(df, metric)
    if sub.empty:
        return no_data_panel(out, title, f"No rows for {metric}")
    return boxplot_panel(sub, metric, out, title, ylabel, **kwargs)


def ablation_metric_panel(
    df: pd.DataFrame,
    presets: list[str],
    labels: dict[str, str],
    metric_col: str,
    out: Path,
    title: str,
    ylabel: str,
) -> Path:
    if df.empty:
        return no_data_panel(out, title)
    sub = df[df["preset"].isin(presets)].copy()
    if sub.empty or metric_col not in sub.columns:
        return no_data_panel(out, title, "No completed ablation metrics")
    sub["variant"] = sub["preset"].map(labels).fillna(sub["preset"])
    order = [labels[p] for p in presets if p in labels and labels[p] in set(sub["variant"])]
    return compact_bar_panel(
        sub,
        metric_col,
        out,
        title,
        ylabel,
        x_col="variant",
        hue_col="dataset",
        x_order=order,
        hue_order=DATASET_ORDER,
        cap_extreme=metric_col == "nmse",
        figsize=(4.5, 3.2),
    )


def heatmap_panel(df: pd.DataFrame, out: Path, title: str, metric: str = "pearson") -> Path:
    if df.empty or "subgroup" not in df.columns or metric not in df.columns:
        return no_data_panel(out, title)
    sub = df.dropna(subset=["subgroup", metric]).copy()
    if sub.empty:
        return no_data_panel(out, title, "No subgroup rows")
    sub["model"] = sub["model"].map(_display_model)
    piv = sub.pivot_table(index="model", columns="subgroup", values=metric, aggfunc="mean")
    piv = piv.reindex(index=available_order(pd.Series(piv.index), MODEL_ORDER), columns=[c for c in SUBGROUP_ORDER if c in piv.columns])
    apply_gears_paper_style(font_scale=0.78)
    fig, ax = plt.subplots(figsize=(4.4, 2.9), dpi=240)
    im = ax.imshow(piv.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels(piv.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            val = piv.iloc[i, j]
            if pd.notna(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6, color="white")
    ax.set_title(title, pad=3)
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    fig.tight_layout(pad=0.35)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    _write_source(piv.reset_index(), out.with_suffix(".csv"))
    return out


def cellot_alignment_panel(out: Path) -> Path:
    frames = []
    for ds in ["adamson", "dixit", "norman", "scgen_pbmc_celltype"]:
        path = ROOT / "artifacts" / "results" / "cellot" / ds / "metrics_unseen_ctrl.csv"
        df = _read_csv(path)
        if df.empty:
            continue
        df = df.copy()
        if "dataset" not in df.columns:
            df["dataset"] = ds
        frames.append(df)
    if frames:
        cellot = pd.concat(frames, ignore_index=True)
    else:
        cellot = _read_csv(ROOT / "artifacts" / "results" / "cellot" / "metrics_unseen_ctrl.csv")
    if cellot.empty:
        return no_data_panel(out, "CellOT aligned runs")
    status = (
        cellot.assign(dataset=cellot["dataset"].map(_display_dataset))
        .groupby(["dataset", "status"], as_index=False)
        .size()
        .sort_values(["dataset", "status"])
    )
    apply_gears_paper_style(font_scale=0.78)
    fig, ax = plt.subplots(figsize=(4.5, 2.5), dpi=240)
    ax.axis("off")
    lines = ["CellOT aligned OT baseline"]
    for _, row in status.iterrows():
        lines.append(f"{row['dataset']}: {row['status']} (n={row['size']})")
    ax.text(0.02, 0.95, "\n".join(lines), ha="left", va="top", fontsize=8)
    fig.tight_layout(pad=0.4)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    _write_source(status, out.with_suffix(".csv"))
    return out


def protocol_panel(out: Path) -> Path:
    return cellot_alignment_panel(out)


def copy_panel(candidates: list[Path], out: Path, title: str, message: str = "Source panel is unavailable") -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    for candidate in candidates:
        for path in _candidate_paths(candidate):
            if path.exists() and path.stat().st_size > 0:
                shutil.copy2(path, out)
                csv_path = path.with_suffix(".csv")
                if csv_path.exists():
                    shutil.copy2(csv_path, out.with_suffix(".csv"))
                value_csv = path.with_name(path.stem + "_values.csv")
                if value_csv.exists():
                    shutil.copy2(value_csv, out.with_name(out.stem + "_values.csv"))
                return out
    return no_data_panel(out, title, message)


def _load_case_item(dataset: str, split_id: int, condition: str, model_name: str) -> dict[str, object]:
    if load_payload_item is None:
        raise RuntimeError("load_payload_item is unavailable")
    return load_payload_item(dataset=dataset, model_name=model_name, split_id=split_id, condition=condition, result_mode=RESULT_MODE)


def _top_genes_by_truth_delta(item: dict[str, object], top_k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    genes = np.asarray(item["gene_name_full"]).astype(str)
    truth = np.asarray(item["Truth_full"], dtype=float)
    ctrl = np.asarray(item["Ctrl_full"], dtype=float)
    delta = truth.mean(axis=0) - ctrl.mean(axis=0)
    top_idx = np.argsort(-np.abs(delta))[:top_k]
    return genes[top_idx], delta[top_idx], top_idx


def case_bar_panel(
    *,
    dataset: str,
    split_id: int,
    condition: str,
    out: Path,
    title: str,
    models: list[str] | None = None,
    top_k: int = 12,
    fallback: list[Path] | None = None,
) -> Path:
    models = models or ["trishift_nearest", "biolord", "gears", "genepert", "scgpt"]
    labels = {
        "trishift_nearest": "TriShift",
        "biolord": "BioLORD",
        "gears": "GEARS",
        "genepert": "GenePert",
        "scgpt": "scGPT",
    }
    try:
        items = {model: _load_case_item(dataset, split_id, condition, model) for model in models}
    except Exception:
        if fallback:
            return copy_panel(fallback, out, title)
        return no_data_panel(out, title, "Case payload is unavailable")
    ref_item = items[models[0]]
    genes, truth_delta, top_idx = _top_genes_by_truth_delta(ref_item, top_k)
    rows: list[dict[str, object]] = []
    for gene, idx, delta in zip(genes, top_idx, truth_delta):
        rows.append({"gene": gene, "model": "Truth", "delta": float(delta)})
        for model_name, item in items.items():
            model_genes = np.asarray(item["gene_name_full"]).astype(str)
            match = np.where(model_genes == gene)[0]
            if len(match) == 0:
                continue
            pred = np.asarray(item["Pred_full"], dtype=float)
            ctrl = np.asarray(item["Ctrl_full"], dtype=float)
            pred_delta = pred[:, int(match[0])].mean() - ctrl[:, int(match[0])].mean()
            rows.append({"gene": gene, "model": labels.get(model_name, model_name), "delta": float(pred_delta)})
    plot_df = pd.DataFrame(rows)
    hue_order = ["Truth"] + [labels[m] for m in models if m in labels]
    apply_gears_paper_style(font_scale=0.82)
    fig, ax = plt.subplots(figsize=(9.0, 5.2), dpi=240)
    genes_order = list(dict.fromkeys(plot_df["gene"].astype(str).tolist()))
    base_width = min(0.14, 0.78 / max(len(hue_order), 1))
    bar_width = base_width * 0.82
    offsets = [0.0] if len(hue_order) == 1 else np.linspace(
        -base_width * (len(hue_order) - 1) / 2,
        base_width * (len(hue_order) - 1) / 2,
        len(hue_order),
    )
    for midx, model in enumerate(hue_order):
        for gidx, gene in enumerate(genes_order):
            row = plot_df[(plot_df["gene"].astype(str).eq(gene)) & (plot_df["model"].astype(str).eq(model))]
            if row.empty:
                continue
            ax.bar(
                gidx + float(offsets[midx]),
                float(row["delta"].iloc[0]),
                width=bar_width,
                color=DISPLAY_COLORS.get(model, "#BBBBBB"),
                edgecolor="black",
                linewidth=0.5,
                label=model if gidx == 0 else None,
            )
    ax.set_xticks(np.arange(len(genes_order)))
    ax.set_xticklabels(genes_order)
    ax.set_xlabel("")
    ax.set_ylabel("Change over control")
    ax.set_title(title, pad=16)
    ax.tick_params(axis="x", rotation=38)
    style_axis(ax, grid_axis="y")
    _legend_above(ax, ncol=min(6, len(hue_order)), fontsize=6.5, y=1.17)
    fig.subplots_adjust(top=0.78, bottom=0.26, left=0.08, right=0.99)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    _write_source(plot_df, out.with_suffix(".csv"))
    return out


def violin_case_panel(
    *,
    dataset: str,
    split_id: int,
    condition: str,
    gene: str,
    out: Path,
    title: str,
    fallback: list[Path] | None = None,
) -> Path:
    labels = {"trishift_nearest": "TriShift", "scgpt": "scGPT"}
    try:
        base_item = _load_case_item(dataset, split_id, condition, "trishift_nearest")
        genes = np.asarray(base_item["gene_name_full"]).astype(str)
        gene_idx = int(np.where(genes == gene)[0][0])
        rows: list[dict[str, object]] = []
        for value in np.asarray(base_item["Ctrl_full"], dtype=float)[:, gene_idx]:
            rows.append({"group": "Control", "expression": float(value)})
        for value in np.asarray(base_item["Truth_full"], dtype=float)[:, gene_idx]:
            rows.append({"group": "Perturbed", "expression": float(value)})
        for model_name, label in labels.items():
            item = _load_case_item(dataset, split_id, condition, model_name)
            model_genes = np.asarray(item["gene_name_full"]).astype(str)
            model_idx = int(np.where(model_genes == gene)[0][0])
            for value in np.asarray(item["Pred_full"], dtype=float)[:, model_idx]:
                rows.append({"group": label, "expression": float(value)})
        plot_df = pd.DataFrame(rows)
    except Exception:
        if fallback:
            return copy_panel(fallback, out, title)
        return no_data_panel(out, title, "Case payload is unavailable")
    order = ["Control", "Perturbed", "TriShift", "scGPT"]
    apply_gears_paper_style(font_scale=0.85)
    fig, ax = plt.subplots(figsize=(6.6, 4.4), dpi=240)
    values = [plot_df[plot_df["group"].eq(group)]["expression"].astype(float).dropna().values for group in order]
    parts = ax.violinplot(values, positions=np.arange(len(order)), widths=0.82, showmeans=False, showmedians=False, showextrema=False)
    for body, group in zip(parts["bodies"], order):
        body.set_facecolor(DISPLAY_COLORS.get(group, "#BBBBBB"))
        body.set_edgecolor("black")
        body.set_linewidth(0.8)
        body.set_alpha(1.0)
    for idx, vals in enumerate(values):
        if len(vals) == 0:
            continue
        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        ax.plot([idx - 0.18, idx + 0.18], [med, med], color="#222222", linewidth=0.8)
        ax.plot([idx, idx], [q1, q3], color="#222222", linewidth=0.8)
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(order)
    ax.set_xlabel("")
    ax.set_ylabel(f"{gene} expression")
    ax.set_title(title, pad=10)
    ax.tick_params(axis="x", labelrotation=28)
    style_axis(ax, grid_axis="y")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    _write_source(plot_df, out.with_suffix(".csv"))
    return out


def _merge_with_difficulty(df: pd.DataFrame, value_col: str, *, split_col: str = "split_id") -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    meta = _read_csv(ROOT / "artifacts" / "analysis" / "systema_mechanism" / "dixit" / "condition_metadata.csv")
    if meta.empty:
        return pd.DataFrame()
    work = df.copy()
    if split_col not in work.columns and "split" in work.columns:
        work = work.rename(columns={"split": split_col})
    if split_col not in work.columns or "condition" not in work.columns:
        return pd.DataFrame()
    work[split_col] = pd.to_numeric(work[split_col], errors="coerce").astype("Int64")
    meta = meta[["split_id", "condition", "train_distance_bin"]].copy()
    meta["split_id"] = pd.to_numeric(meta["split_id"], errors="coerce").astype("Int64")
    merged = work.merge(meta, left_on=[split_col, "condition"], right_on=["split_id", "condition"], how="inner")
    if value_col in merged.columns:
        merged[value_col] = pd.to_numeric(merged[value_col], errors="coerce")
    if "model" in merged.columns:
        merged["model"] = merged["model"].map(_display_model)
    return merged.dropna(subset=["train_distance_bin", value_col])


def dixit_metric_by_distance(metric_col: str) -> pd.DataFrame:
    metrics = collect_prediction_metrics(heldout=True)
    if metrics.empty or metric_col not in metrics.columns:
        return pd.DataFrame()
    dixit = metrics[metrics["dataset"].eq("Dixit")].copy()
    return _merge_with_difficulty(dixit, metric_col, split_col="split_id")


def dixit_long_by_distance(metric: str) -> pd.DataFrame:
    long_df = collect_long(ROOT / "artifacts" / "analysis" / "deg_prediction" / "deg_prediction_all_long.csv")
    sub = long_metric_frame(long_df, metric)
    if sub.empty:
        return pd.DataFrame()
    sub = sub[sub["dataset"].eq("Dixit")].copy()
    return _merge_with_difficulty(sub, metric, split_col="split")


def dixit_residualized_by_distance() -> pd.DataFrame:
    path = ROOT / "artifacts" / "analysis" / "systema_mechanism" / "dixit" / "difficulty_bin_generic_shift_summary.csv"
    df = _read_csv(path)
    if df.empty or "train_distance_bin" not in df.columns:
        return pd.DataFrame()
    df = df.copy()
    if "model_name" in df.columns:
        df["model"] = df["model_name"].map(_display_model)
    for col in ["systema_corr_20de_allpert", "residualized_systema_corr_20de_allpert", "generic_projection_ratio"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def line_panel(
    df: pd.DataFrame,
    metric_col: str,
    out: Path,
    title: str,
    ylabel: str,
    *,
    bin_col: str = "train_distance_bin",
    model_col: str = "model",
) -> Path:
    if df.empty or metric_col not in df.columns or bin_col not in df.columns or model_col not in df.columns:
        return no_data_panel(out, title)
    plot = df[[bin_col, model_col, metric_col]].copy()
    plot[metric_col] = pd.to_numeric(plot[metric_col], errors="coerce")
    plot = plot.dropna(subset=[bin_col, model_col, metric_col])
    if plot.empty:
        return no_data_panel(out, title, "No finite values")
    order = [b for b in ["near", "medium", "far"] if b in set(plot[bin_col].astype(str))]
    models = available_order(plot[model_col].map(_display_model), MODEL_ORDER)
    plot[model_col] = plot[model_col].map(_display_model)
    summary = plot.groupby([bin_col, model_col], as_index=False).agg(mean=(metric_col, "mean"), sem=(metric_col, "sem"), n=(metric_col, "size"))
    apply_gears_paper_style(font_scale=0.78)
    colors = _color_map(models)
    fig, ax = plt.subplots(figsize=(4.6, 3.0), dpi=240)
    x = np.arange(len(order))
    for model in models:
        sub = summary[summary[model_col].eq(model)].set_index(bin_col)
        vals = [sub.loc[b, "mean"] if b in sub.index else np.nan for b in order]
        errs = [sub.loc[b, "sem"] if b in sub.index and pd.notna(sub.loc[b, "sem"]) else 0.0 for b in order]
        if all(pd.isna(v) for v in vals):
            continue
        ax.errorbar(x, vals, yerr=errs, marker="o", markersize=3.2, linewidth=1.1, capsize=1.6, color=colors.get(model), label=model)
    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_xlabel("Train-distance bin")
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=11)
    style_axis(ax, grid_axis="y")
    _legend_above(ax, ncol=min(3, max(1, len(models))), y=1.30)
    fig.tight_layout(pad=0.35, rect=(0, 0, 1, 0.78))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    _write_source(summary, out.with_suffix(".csv"))
    return out


def dixit_distance_scatter_panel(out: Path) -> Path:
    meta = _read_csv(ROOT / "artifacts" / "analysis" / "systema_mechanism" / "dixit" / "condition_metadata.csv")
    if meta.empty or "train_test_distance" not in meta.columns:
        return no_data_panel(out, "Dixit difficulty structure")
    meta = meta.copy()
    meta["train_test_distance"] = pd.to_numeric(meta["train_test_distance"], errors="coerce")
    meta["truth_ctrl_shift_norm"] = pd.to_numeric(meta["truth_ctrl_shift_norm"], errors="coerce")
    meta = meta.dropna(subset=["train_test_distance", "truth_ctrl_shift_norm", "train_distance_bin"])
    apply_gears_paper_style(font_scale=0.78)
    fig, ax = plt.subplots(figsize=(4.6, 3.0), dpi=240)
    palette = {"near": "#9FD9D3", "medium": "#B9AEDC", "far": "#F2B56B"}
    for bin_name in ["near", "medium", "far"]:
        sub = meta[meta["train_distance_bin"].astype(str).eq(bin_name)]
        if sub.empty:
            continue
        ax.scatter(sub["train_test_distance"], sub["truth_ctrl_shift_norm"], s=12, alpha=0.65, color=palette[bin_name], edgecolors="none", label=bin_name)
    ax.set_xlabel("Nearest-training-condition distance")
    ax.set_ylabel("Observed response norm")
    ax.set_title("Dixit difficulty structure", pad=4)
    style_axis(ax, grid_axis="y")
    ax.legend(frameon=False, fontsize=6.2, title="", handlelength=1.0)
    fig.tight_layout(pad=0.35)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    _write_source(meta, out.with_suffix(".csv"))
    return out


def _font(size: int, bold: bool = False):
    for candidate in [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def crop_white(im: Image.Image, pad: int = 10) -> Image.Image:
    rgba = im.convert("RGBA")
    diff = ImageChops.difference(rgba, Image.new("RGBA", rgba.size, (255, 255, 255, 255)))
    bbox = diff.getbbox()
    if bbox is None:
        return rgba
    l, t, r, b = bbox
    return rgba.crop((max(0, l - pad), max(0, t - pad), min(rgba.width, r + pad), min(rgba.height, b + pad)))


def fit(im: Image.Image, max_w: int, max_h: int) -> Image.Image:
    scale = min(max_w / max(1, im.width), max_h / max(1, im.height))
    return im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))), Image.Resampling.LANCZOS)


def compose_grid(panels: list[tuple[str, Path]], out: Path, *, cols: int, cell_w: int = 980, cell_h: int = 620) -> Path:
    rows = math.ceil(len(panels) / cols)
    margin = 90
    label_pad = 58
    gap_x = 60
    gap_y = 80
    width = margin * 2 + cols * cell_w + (cols - 1) * gap_x
    height = margin * 2 + rows * (cell_h + label_pad) + (rows - 1) * gap_y
    canvas = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    label_font = _font(44, bold=True)
    for idx, (label, src) in enumerate(panels):
        row, col = divmod(idx, cols)
        x = margin + col * (cell_w + gap_x)
        y = margin + row * (cell_h + label_pad + gap_y)
        draw.text((x, y), label, fill=(0, 0, 0), font=label_font)
        if not src.exists():
            panel = no_data_panel(src, src.stem)
            src = panel
        im = fit(crop_white(Image.open(src), pad=12), cell_w, cell_h)
        px = x + (cell_w - im.width) // 2
        py = y + label_pad + (cell_h - im.height) // 2
        canvas.alpha_composite(im, (px, py))
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(out, quality=95)
    return out


def compose_rows(
    rows: list[list[tuple[str, Path]]],
    out: Path,
    *,
    max_cols: int = 3,
    cell_w: int = 760,
    cell_h: int = 520,
    wide_h: int = 620,
) -> Path:
    margin = 90
    label_pad = 58
    gap_x = 60
    gap_y = 80
    width = margin * 2 + max_cols * cell_w + (max_cols - 1) * gap_x
    row_heights = [wide_h if len(row) == 1 else cell_h for row in rows]
    height = margin * 2 + sum(h + label_pad for h in row_heights) + (len(rows) - 1) * gap_y
    canvas = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    label_font = _font(44, bold=True)
    y = margin
    for row, row_h in zip(rows, row_heights):
        row_cols = len(row)
        if row_cols == 1:
            slots = [(margin, width - 2 * margin)]
        else:
            slots = [(margin + idx * (cell_w + gap_x), cell_w) for idx in range(row_cols)]
        for (label, src), (x, slot_w) in zip(row, slots):
            draw.text((x, y), label, fill=(0, 0, 0), font=label_font)
            if not src.exists():
                src = no_data_panel(src, src.stem)
            im = fit(crop_white(Image.open(src), pad=12), slot_w, row_h)
            px = x + (slot_w - im.width) // 2
            py = y + label_pad + (row_h - im.height) // 2
            canvas.alpha_composite(im, (px, py))
        y += row_h + label_pad + gap_y
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(out, quality=95)
    return out


def render_fig2() -> Path:
    out_dir = FIG_ROOT / "main" / "Fig2_ReferenceTransfer"
    metrics = collect_prediction_metrics(heldout=True)
    systema = collect_long(ROOT / "artifacts" / "analysis" / "systema_mechanism" / "systema_mechanism_all_long.csv")
    deg = collect_long(ROOT / "artifacts" / "analysis" / "deg_prediction" / "deg_prediction_all_long.csv")
    panels = [
        ("a", compact_bar_panel(metrics, "pearson", out_dir / "fig2a_pearson.png", "Held-out/reference transfer Pearson", "Pearson")),
        ("b", compact_bar_panel(metrics, "nmse", out_dir / "fig2b_nmse.png", "Held-out/reference transfer nMSE", "nMSE", cap_extreme=True)),
        ("c", compact_bar_panel(metrics, "systema_corr_20de_allpert", out_dir / "fig2c_systema_pearson.png", "Reference-centered Systema Pearson", "Systema Pearson")),
        ("d", long_metric_bar_panel(systema, "residualized_systema_corr_20de_allpert", out_dir / "fig2d_residualized_systema.png", "Residualized Systema Pearson", "residualized Pearson")),
        ("e", long_metric_bar_panel(systema, "centroid_accuracy", out_dir / "fig2e_centroid_accuracy.png", "Centroid accuracy", "accuracy")),
        ("f", long_metric_bar_panel(systema, "generic_projection_ratio", out_dir / "fig2f_generic_shift.png", "Generic-shift dependence", "projection ratio")),
        ("g", long_metric_bar_panel(deg, "deg_auroc", out_dir / "fig2g_deg_auroc.png", "Post-perturbation DEG prediction", "DEG AUROC")),
        (
            "h",
            case_bar_panel(
                dataset="norman",
                split_id=1,
                condition="ELMSAN1+ctrl",
                out=out_dir / "fig2h_elmsan1_case.png",
                title="ELMSAN1+ctrl response case",
                fallback=[
                    SERVER_REFRESH_ROOT
                    / "artifacts"
                    / "analysis"
                    / "case_selection"
                    / "unseen_ctrl"
                    / "preview"
                    / "norman_split1_ELMSAN1-ctrl_unseen_ctrl.png"
                ],
            ),
        ),
    ]
    return compose_grid(panels, COMP_ROOT / "fig2_main_composite.png", cols=4, cell_w=720, cell_h=470)


def render_fig3() -> Path:
    out_dir = FIG_ROOT / "main" / "Fig3_Ablation"
    ab = collect_ablation_metrics()
    panels = [
        ("a", ablation_metric_panel(ab, ["ref_knn", "ref_ot"], {"ref_knn": "kNN", "ref_ot": "OT"}, "pearson", out_dir / "fig3a_reference_pearson.png", "Reference construction: Pearson", "Pearson")),
        ("b", ablation_metric_panel(ab, ["ref_knn", "ref_ot"], {"ref_knn": "kNN", "ref_ot": "OT"}, "nmse", out_dir / "fig3b_reference_nmse.png", "Reference construction: nMSE", "nMSE")),
        ("c", ablation_metric_panel(ab, ["ref_knn", "ref_ot"], {"ref_knn": "kNN", "ref_ot": "OT"}, "systema_corr_20de_allpert", out_dir / "fig3c_reference_systema.png", "Reference construction: Systema", "Systema Pearson")),
        ("d", ablation_metric_panel(ab, ["cond_no_reference", "cond_no_prior", "cond_full"], {"cond_no_reference": "no reference", "cond_no_prior": "no prior", "cond_full": "full"}, "pearson", out_dir / "fig3d_conditioning_pearson.png", "Conditioning input: Pearson", "Pearson")),
        ("e", ablation_metric_panel(ab, ["cond_no_reference", "cond_no_prior", "cond_full"], {"cond_no_reference": "no reference", "cond_no_prior": "no prior", "cond_full": "full"}, "nmse", out_dir / "fig3e_conditioning_nmse.png", "Conditioning input: nMSE", "nMSE")),
        ("f", ablation_metric_panel(ab, ["cond_no_reference", "cond_no_prior", "cond_full"], {"cond_no_reference": "no reference", "cond_no_prior": "no prior", "cond_full": "full"}, "systema_corr_20de_allpert", out_dir / "fig3f_conditioning_systema.png", "Conditioning input: Systema", "Systema Pearson")),
    ]
    return compose_grid(panels, COMP_ROOT / "fig3_main_composite.png", cols=3, cell_w=760, cell_h=520)


def render_fig4() -> Path:
    out_dir = FIG_ROOT / "main" / "Fig4_NormanGeneralization"
    metrics = collect_prediction_metrics(heldout=True)
    norman = metrics[metrics["dataset"].eq("Norman")].copy() if not metrics.empty else pd.DataFrame()
    deg = collect_long(ROOT / "artifacts" / "analysis" / "deg_prediction" / "deg_prediction_all_long.csv")
    norman_deg = deg[deg["dataset"].eq("Norman")].copy() if not deg.empty else pd.DataFrame()
    panels = [
        ("a", heatmap_panel(norman, out_dir / "fig4a_norman_subgroup_heatmap.png", "Norman subgroup Pearson", "pearson")),
        ("b", compact_bar_panel(norman, "pearson", out_dir / "fig4b_subgroup_pearson.png", "Norman subgroup Pearson", "Pearson", x_col="subgroup", x_order=SUBGROUP_ORDER)),
        ("c", compact_bar_panel(norman, "nmse", out_dir / "fig4c_subgroup_nmse.png", "Norman subgroup nMSE", "nMSE", x_col="subgroup", x_order=SUBGROUP_ORDER, cap_extreme=True)),
        ("d", compact_bar_panel(norman, "systema_corr_20de_allpert", out_dir / "fig4d_subgroup_systema.png", "Norman subgroup Systema", "Systema Pearson", x_col="subgroup", x_order=SUBGROUP_ORDER)),
        ("e", long_metric_bar_panel(norman_deg, "deg_auroc", out_dir / "fig4e_norman_deg_auroc.png", "Norman DEG prediction", "DEG AUROC")),
        ("f", long_metric_bar_panel(norman_deg, "direction_agreement_overlap_at_20", out_dir / "fig4f_direction_agreement.png", "Norman direction agreement", "agreement")),
        (
            "g",
            case_bar_panel(
                dataset="norman",
                split_id=3,
                condition="CNN1+MAPK1",
                out=out_dir / "fig4g_cnn1_mapk1_case.png",
                title="CNN1+MAPK1 response case",
                fallback=[
                    SERVER_REFRESH_ROOT
                    / "artifacts"
                    / "analysis"
                    / "case_selection"
                    / "unseen_ctrl"
                    / "preview"
                    / "norman_split3_CNN1-MAPK1_unseen_ctrl.png"
                ],
            ),
        ),
    ]
    return compose_rows([panels[:3], panels[3:6], panels[6:]], COMP_ROOT / "fig4_main_composite.png", max_cols=3, cell_w=760, cell_h=520, wide_h=620)


def render_fig5() -> Path:
    out_dir = FIG_ROOT / "main" / "Fig5_DistributionRecovery"
    metrics = collect_prediction_metrics(heldout=True)
    panels = [
        ("a", boxplot_panel(metrics, "scpram_wasserstein_degs_sum", out_dir / "fig5a_wasserstein.png", "Wasserstein distance", "Wasserstein", cap_extreme=True)),
        ("b", compact_bar_panel(metrics, "scpram_r2_all_mean_mean", out_dir / "fig5b_mean_rho2.png", "Mean recovery", r"mean $\rho^2$")),
        ("c", compact_bar_panel(metrics, "scpram_r2_all_var_mean", out_dir / "fig5c_variance_rho2.png", "Variance recovery", r"variance $\rho^2$")),
        (
            "d",
            violin_case_panel(
                dataset="adamson",
                split_id=4,
                condition="PTDSS1+ctrl",
                gene="RPS29",
                out=out_dir / "fig5d_rps29_violin.png",
                title="RPS29 | PTDSS1+ctrl",
                fallback=[
                    SERVER_REFRESH_ROOT
                    / "artifacts"
                    / "paper_figures"
                    / "supp"
                    / "FigS2_AdditionalCases"
                    / "figs2b_single_case_violin.png"
                ],
            ),
        ),
    ]
    return compose_grid(panels, COMP_ROOT / "fig5_main_composite.png", cols=4, cell_w=720, cell_h=500)


def render_supp(name: str) -> Path:
    fig = name.lower()
    out_dir = FIG_ROOT / "supp" / SUPP_DIRS[fig]
    if fig == "figs1":
        metrics = collect_prediction_metrics(heldout=True)
        systema = collect_long(ROOT / "artifacts" / "analysis" / "systema_mechanism" / "systema_mechanism_all_long.csv")
        deg = collect_long(ROOT / "artifacts" / "analysis" / "deg_prediction" / "deg_prediction_all_long.csv")
        panels = [
            ("a", boxplot_panel(metrics, "pearson", out_dir / "figs1a_pearson.png", "Reference-transfer Pearson", "Pearson")),
            ("b", boxplot_panel(metrics, "nmse", out_dir / "figs1b_nmse.png", "Reference-transfer nMSE", "nMSE", cap_extreme=True)),
            ("c", boxplot_panel(metrics, "systema_corr_20de_allpert", out_dir / "figs1c_systema.png", "Systema Pearson", "Systema Pearson")),
            ("d", long_metric_box_panel(systema, "residualized_systema_corr_20de_allpert", out_dir / "figs1d_residualized_systema.png", "Residualized Systema Pearson", "residualized Pearson")),
            ("e", long_metric_box_panel(systema, "centroid_accuracy", out_dir / "figs1e_centroid_accuracy.png", "Centroid accuracy", "accuracy")),
            ("f", long_metric_box_panel(systema, "generic_projection_ratio", out_dir / "figs1f_generic_shift.png", "Generic-shift dependence", "projection ratio")),
            ("g", long_metric_box_panel(deg, "deg_auroc", out_dir / "figs1g_deg_auroc.png", "DEG AUROC", "AUROC")),
        ]
        return compose_grid(panels, COMP_ROOT / "figs1_composite.png", cols=3, cell_w=760, cell_h=520)
    if fig == "figs2":
        metrics = collect_prediction_metrics(heldout=True)
        dixit = metrics[metrics["dataset"].eq("Dixit")].copy() if not metrics.empty else pd.DataFrame()
        deg = collect_long(ROOT / "artifacts" / "analysis" / "deg_prediction" / "deg_prediction_all_long.csv")
        dixit_deg = deg[deg["dataset"].eq("Dixit")].copy() if not deg.empty else pd.DataFrame()
        panels = [
            ("a", compact_bar_panel(dixit, "pearson", out_dir / "figs2a_dixit_pearson.png", "Dixit Pearson", "Pearson")),
            ("b", compact_bar_panel(dixit, "nmse", out_dir / "figs2b_dixit_nmse.png", "Dixit nMSE", "nMSE", cap_extreme=True)),
            ("c", compact_bar_panel(dixit, "systema_corr_20de_allpert", out_dir / "figs2c_dixit_systema.png", "Dixit Systema Pearson", "Systema")),
            ("d", long_metric_bar_panel(dixit_deg, "deg_auroc", out_dir / "figs2d_dixit_deg_auroc.png", "Dixit DEG AUROC", "AUROC")),
        ]
        return compose_grid(panels, COMP_ROOT / "figs2_composite.png", cols=2, cell_w=1080, cell_h=570)
    if fig == "figs3":
        panels = [
            ("a", copy_panel([SERVER_REFRESH_ROOT / "artifacts" / "paper_figures" / "supp" / "FigS2_AdditionalCases" / "figs2a_single_case_barplot.png"], out_dir / "figs3a_single_case_barplot.png", "PTDSS1+ctrl case")),
            ("b", copy_panel([SERVER_REFRESH_ROOT / "artifacts" / "paper_figures" / "supp" / "FigS2_AdditionalCases" / "figs2b_single_case_violin.png"], out_dir / "figs3b_single_case_violin.png", "RPS29 distribution")),
            ("c", copy_panel([SERVER_REFRESH_ROOT / "artifacts" / "paper_figures" / "supp" / "FigS2_AdditionalCases" / "figs2c_combo_case_barplot.png"], out_dir / "figs3c_combo_case_barplot.png", "UBASH3A+UBASH3B case")),
            ("d", copy_panel([SERVER_REFRESH_ROOT / "artifacts" / "paper_figures" / "supp" / "FigS2_AdditionalCases" / "figs2d_combo_case_violin.png"], out_dir / "figs3d_combo_case_violin.png", "HBZ distribution")),
            ("e", copy_panel([SERVER_REFRESH_ROOT / "artifacts" / "paper_figures" / "supp" / "FigS2_AdditionalCases" / "figs2e_moved_fig5d_violin.png"], out_dir / "figs3e_spcs2_ddit4_violin.png", "DDIT4 distribution")),
        ]
        return compose_grid(panels, COMP_ROOT / "figs3_composite.png", cols=3, cell_w=760, cell_h=520)
    if fig == "figs4":
        residual = dixit_residualized_by_distance()
        panels = [
            ("a", dixit_distance_scatter_panel(out_dir / "figs4a_dixit_difficulty_scatter.png")),
            ("b", line_panel(dixit_metric_by_distance("pearson"), "pearson", out_dir / "figs4b_dixit_pearson_line.png", "Dixit Pearson by distance", "Pearson")),
            ("c", line_panel(dixit_metric_by_distance("nmse"), "nmse", out_dir / "figs4c_dixit_nmse_line.png", "Dixit nMSE by distance", "nMSE")),
            ("d", line_panel(residual, "systema_corr_20de_allpert", out_dir / "figs4d_dixit_systema_line.png", "Dixit Systema by distance", "Systema Pearson")),
            ("e", line_panel(residual, "residualized_systema_corr_20de_allpert", out_dir / "figs4e_dixit_residualized_line.png", "Residualized Systema by distance", "residualized Pearson")),
            ("f", line_panel(dixit_long_by_distance("deg_auroc"), "deg_auroc", out_dir / "figs4f_dixit_deg_auroc_line.png", "Dixit DEG AUROC by distance", "DEG AUROC")),
        ]
        return compose_grid(panels, COMP_ROOT / "figs4_composite.png", cols=3, cell_w=760, cell_h=500)
    if fig == "figs5":
        src = FIG_ROOT / "supp" / "FigS6_Module1LatentState"
        src_refresh = SERVER_REFRESH_ROOT / "artifacts" / "paper_figures" / "supp" / "FigS6_Module1LatentState"
        panels = [
            ("a", copy_panel([src_refresh / "figs6a_umap_by_cluster.png", src / "figs6a_umap_by_cluster.png"], out_dir / "figs5a_umap_by_cluster.png", "Module 1 latent clusters")),
            ("b", copy_panel([src_refresh / "figs6b_umap_by_cell_type.png", src / "figs6b_umap_by_cell_type.png"], out_dir / "figs5b_umap_by_cell_type.png", "Module 1 cell-type structure")),
            ("c", copy_panel([src_refresh / "figs6c_cluster_vs_cell_type.png", src / "figs6c_cluster_vs_cell_type.png"], out_dir / "figs5c_cluster_vs_cell_type.png", "Cluster-cell-type alignment")),
            ("d", copy_panel([src_refresh / "figs6d_cluster_metrics.png", src / "figs6d_cluster_metrics.png"], out_dir / "figs5d_cluster_metrics.png", "Latent cluster metrics")),
            ("e", copy_panel([src_refresh / "figs6e_trishift_scgen_unseen_umap.png", src / "figs6e_trishift_scgen_unseen_umap.png"], out_dir / "figs5e_trishift_scgen_unseen_umap.png", "PBMC target-domain prediction")),
        ]
        return compose_grid(panels, COMP_ROOT / "figs5_composite.png", cols=3, cell_w=760, cell_h=500)
    raise ValueError(f"Unknown supplementary figure: {name}")


def write_manifest() -> Path:
    rows = [
        ("main", "fig1", "TriShift overview", "assets/biorender/trishift_overview_biorender.png"),
        ("main", "fig2", "Held-out control reference transfer and response-specific diagnostics", "artifacts/paper_figures/composites/fig2_main_composite.png"),
        ("main", "fig3", "TriShift component ablations", "artifacts/paper_figures/composites/fig3_main_composite.png"),
        ("main", "fig4", "Norman unseen combinatorial perturbation generalization", "artifacts/paper_figures/composites/fig4_main_composite.png"),
        ("main", "fig5", "Cell-level distribution recovery", "artifacts/paper_figures/composites/fig5_main_composite.png"),
        ("supp", "figs1", "Split-level reference-transfer boxplots corresponding to Fig. 2a-g", "artifacts/paper_figures/composites/figs1_composite.png"),
        ("supp", "figs2", "Dixit scarce-condition stability and DEG recovery", "artifacts/paper_figures/composites/figs2_composite.png"),
        ("supp", "figs3", "Additional fixed-display response cases", "artifacts/paper_figures/composites/figs3_composite.png"),
        ("supp", "figs4", "Dixit distance-stratified robustness diagnostics", "artifacts/paper_figures/composites/figs4_composite.png"),
        ("supp", "figs5", "Module 1 latent-state and PBMC target-domain diagnostics", "artifacts/paper_figures/composites/figs5_composite.png"),
    ]
    manifest = pd.DataFrame(rows, columns=["section", "figure_id", "description", "composite_path"])
    out = FIG_ROOT / "figure_manifest.csv"
    _write_source(manifest, out)
    return out


def render(figure: str) -> Path:
    key = figure.lower().replace(".", "")
    if key == "fig2":
        return render_fig2()
    if key == "fig3":
        return render_fig3()
    if key == "fig4":
        return render_fig4()
    if key == "fig5":
        return render_fig5()
    if key in {"figs1", "figs2", "figs3", "figs4", "figs5"}:
        return render_supp(key)
    if key == "figs6":
        return render_supp("figs5")
    if key == "all":
        paths = [render_fig2(), render_fig3(), render_fig4(), render_fig5()]
        paths.extend(render_supp(f"figs{i}") for i in range(1, 6))
        write_manifest()
        return paths[-1]
    raise ValueError(f"Unknown figure: {figure}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Render refreshed TriShift paper figures from server-side result tables.")
    ap.add_argument("--figure", default="all", help="fig2, fig3, fig4, fig5, figs1-figs5, or all")
    args = ap.parse_args()
    out = render(args.figure)
    print(out)


if __name__ == "__main__":
    main()
