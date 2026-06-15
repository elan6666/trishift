from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageChops, ImageDraw, ImageFont

from scripts.common.paper_plot_style import apply_gears_paper_style, model_color_map, style_axis


FIG_ROOT = ROOT / "artifacts" / "paper_figures"
COMP_ROOT = FIG_ROOT / "composites"

DATASET_ORDER = ["Adamson", "Dixit", "Norman", "PBMC"]
MODEL_ORDER = ["TriShift", "CellOT", "GEARS", "BioLORD", "GenePert", "scGPT"]
SUBGROUP_ORDER = ["single", "seen2", "seen1", "seen0"]
SUPP_DIRS = {
    "figs1": "FigS1_ReferenceTransferFull",
    "figs2": "FigS2_DixitStability",
    "figs3": "FigS3_UnseenBenchmarkBackup",
    "figs4": "FigS4_AblationAndSystemaDiagnostics",
    "figs5": "FigS5_CombinationDistributionExtension",
    "figs6": "FigS6_Module1LatentState",
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
        "cellot": "CellOT",
        "gears": "GEARS",
        "biolord": "BioLORD",
        "genepert": "GenePert",
        "scgpt": "scGPT",
    }
    return lookup.get(str(name).lower(), str(name))


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
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
    ablation_root = ROOT / "artifacts" / "results" / "ablation"
    for preset_dir in sorted(p for p in ablation_root.glob("*/*") if p.is_dir()):
        parts = preset_dir.relative_to(ablation_root).parts
        if len(parts) < 2 or parts[0] == "plan":
            continue
        dataset, preset = parts[0], parts[1]
        p = preset_dir / "metrics_unseen_ctrl.csv"
        if not p.exists():
            p = preset_dir / "metrics.csv"
        if not p.exists():
            continue
        df = _read_csv(p)
        if df.empty:
            continue
        df = df.copy()
        df["dataset"] = _display_dataset(dataset)
        df["preset"] = preset
        df["source_file"] = str(p.relative_to(ROOT))
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


def point_panel(
    df: pd.DataFrame,
    metric_col: str,
    out: Path,
    title: str,
    ylabel: str,
    *,
    dataset_col: str = "dataset",
    model_col: str = "model",
    smaller: bool = True,
) -> Path:
    if df.empty or metric_col not in df.columns:
        return no_data_panel(out, title)
    plot = df[[dataset_col, model_col, metric_col]].copy()
    plot[metric_col] = pd.to_numeric(plot[metric_col], errors="coerce")
    plot = plot.dropna(subset=[metric_col])
    if plot.empty:
        return no_data_panel(out, title, "No finite values")
    apply_gears_paper_style(font_scale=0.78 if smaller else 0.9)
    datasets = available_order(plot[dataset_col], DATASET_ORDER)
    models = available_order(plot[model_col], MODEL_ORDER)
    colors = model_color_map(models)
    width = 0.72 / max(1, len(models))
    fig, ax = plt.subplots(figsize=(4.6, 2.8), dpi=240)
    x_lookup = {d: i for i, d in enumerate(datasets)}
    for midx, model in enumerate(models):
        sub = plot[plot[model_col].eq(model)]
        offset = (midx - (len(models) - 1) / 2.0) * width
        xs = sub[dataset_col].map(x_lookup).astype(float) + offset
        ax.scatter(xs, sub[metric_col], s=8, alpha=0.24, color=colors.get(model), linewidths=0)
        means = sub.groupby(dataset_col)[metric_col].mean()
        sems = sub.groupby(dataset_col)[metric_col].sem()
        for dataset in datasets:
            if dataset not in means:
                continue
            yerr = sems.get(dataset, 0.0)
            ax.errorbar(
                x_lookup[dataset] + offset,
                means[dataset],
                yerr=0 if pd.isna(yerr) else yerr,
                marker="o",
                markersize=3.2,
                linewidth=0.7,
                capsize=1.8,
                color=colors.get(model),
                label=model if dataset == datasets[0] else None,
            )
    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels(datasets, rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=3)
    style_axis(ax, grid_axis="y")
    ax.legend(frameon=False, fontsize=6.2, ncol=2, handlelength=1.1)
    fig.tight_layout(pad=0.35)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    _write_source(plot, out.with_suffix(".csv"))
    return out


def long_metric_panel(df: pd.DataFrame, metric: str, out: Path, title: str, ylabel: str) -> Path:
    if df.empty or "metric" not in df.columns:
        return no_data_panel(out, title)
    sub = df[df["metric"].astype(str).eq(metric)].copy()
    if sub.empty:
        return no_data_panel(out, title, f"No rows for {metric}")
    sub = sub.rename(columns={"value": metric})
    return point_panel(sub, metric, out, title, ylabel)


def ablation_panel(df: pd.DataFrame, presets: list[str], labels: dict[str, str], out: Path, title: str) -> Path:
    if df.empty:
        return no_data_panel(out, title)
    sub = df[df["preset"].isin(presets)].copy()
    if sub.empty:
        return no_data_panel(out, title, "No completed ablation metrics")
    summary = (
        sub.groupby(["dataset", "preset"], as_index=False)
        .agg(pearson=("pearson", "mean"), nmse=("nmse", "mean"), systema=("systema_corr_20de_allpert", "mean"))
        .melt(id_vars=["dataset", "preset"], var_name="metric", value_name="value")
    )
    summary["variant"] = summary["preset"].map(labels).fillna(summary["preset"])
    apply_gears_paper_style(font_scale=0.78)
    metrics = ["pearson", "nmse", "systema"]
    variants = [labels[p] for p in presets if p in labels and p in sub["preset"].unique()]
    colors = model_color_map(variants)
    fig, axes = plt.subplots(1, 3, figsize=(8.2, 2.6), dpi=240, sharex=False)
    for ax, metric in zip(axes, metrics):
        mdf = summary[summary["metric"].eq(metric)]
        x = range(len(variants))
        vals = [mdf[mdf["variant"].eq(v)]["value"].mean() for v in variants]
        ax.bar(x, vals, color=[colors.get(v) for v in variants], width=0.72)
        ax.set_xticks(list(x))
        ax.set_xticklabels(variants, rotation=30, ha="right", fontsize=6.2)
        ax.set_title(metric, pad=2)
        style_axis(ax, grid_axis="y")
    axes[0].set_ylabel("mean score")
    fig.suptitle(title, y=1.02, fontsize=9.2)
    fig.tight_layout(pad=0.3)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    _write_source(summary, out.with_suffix(".csv"))
    return out


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


def render_fig2() -> Path:
    out_dir = FIG_ROOT / "main" / "Fig2_ReferenceTransfer"
    metrics = collect_prediction_metrics(heldout=True)
    systema = collect_long(ROOT / "artifacts" / "analysis" / "systema_mechanism" / "systema_mechanism_all_long.csv")
    deg = collect_long(ROOT / "artifacts" / "analysis" / "deg_prediction" / "deg_prediction_all_long.csv")
    panels = [
        ("a", point_panel(metrics, "pearson", out_dir / "fig2a_pearson.png", "Held-out/reference transfer Pearson", "Pearson")),
        ("b", point_panel(metrics, "nmse", out_dir / "fig2b_nmse.png", "Held-out/reference transfer nMSE", "nMSE")),
        ("c", point_panel(metrics, "systema_corr_20de_allpert", out_dir / "fig2c_systema_pearson.png", "Reference-centered Systema Pearson", "Systema Pearson")),
        ("d", long_metric_panel(systema, "residualized_systema_corr_20de_allpert", out_dir / "fig2d_residualized_systema.png", "Residualized Systema Pearson", "residualized Pearson")),
        ("e", long_metric_panel(systema, "centroid_accuracy", out_dir / "fig2e_centroid_accuracy.png", "Centroid accuracy", "accuracy")),
        ("f", long_metric_panel(systema, "generic_projection_ratio", out_dir / "fig2f_generic_shift.png", "Generic-shift dependence", "projection ratio")),
        ("g", long_metric_panel(deg, "deg_auroc", out_dir / "fig2g_deg_auroc.png", "Post-perturbation DEG prediction", "DEG AUROC")),
        ("h", cellot_alignment_panel(out_dir / "fig2h_cellot_aligned.png")),
    ]
    return compose_grid(panels, COMP_ROOT / "fig2_main_composite.png", cols=4, cell_w=720, cell_h=470)


def render_fig3() -> Path:
    out_dir = FIG_ROOT / "main" / "Fig3_Ablation"
    ab = collect_ablation_metrics()
    panels = [
        ("a", ablation_panel(ab, ["ref_random", "ref_knn", "ref_ot"], {"ref_random": "random", "ref_knn": "kNN", "ref_ot": "OT"}, out_dir / "fig3a_reference_construction.png", "Reference construction")),
        ("b", ablation_panel(ab, ["cond_no_reference", "cond_no_prior", "cond_full"], {"cond_no_reference": "no ref", "cond_no_prior": "no prior", "cond_full": "full"}, out_dir / "fig3b_conditioning_inputs.png", "Conditioning inputs")),
        ("c", ablation_panel(ab, ["fusion_mlp", "fusion_transformer", "fusion_full"], {"fusion_mlp": "MLP", "fusion_transformer": "Transformer", "fusion_full": "full"}, out_dir / "fig3c_fusion_module.png", "Fusion module")),
        ("d", ablation_panel(ab, ["loss_mse", "loss_gears", "loss_gears_neg"], {"loss_mse": "L_mse", "loss_gears": "L_gears", "loss_gears_neg": "L_gears+L_neg"}, out_dir / "fig3d_expression_objective.png", "Expression objective")),
    ]
    return compose_grid(panels, COMP_ROOT / "fig3_main_composite.png", cols=2, cell_w=1080, cell_h=570)


def render_fig4() -> Path:
    out_dir = FIG_ROOT / "main" / "Fig4_NormanGeneralization"
    metrics = collect_prediction_metrics(heldout=True)
    norman = metrics[metrics["dataset"].eq("Norman")].copy() if not metrics.empty else pd.DataFrame()
    deg = collect_long(ROOT / "artifacts" / "analysis" / "deg_prediction" / "deg_prediction_all_long.csv")
    norman_deg = deg[deg["dataset"].eq("Norman")].copy() if not deg.empty else pd.DataFrame()
    panels = [
        ("a", heatmap_panel(norman, out_dir / "fig4a_norman_subgroup_heatmap.png", "Norman subgroup Pearson", "pearson")),
        ("b", point_panel(norman[norman.get("subgroup", pd.Series(dtype=str)).eq("seen0")] if "subgroup" in norman.columns else pd.DataFrame(), "pearson", out_dir / "fig4b_seen0_pearson.png", "seen0 Pearson", "Pearson", dataset_col="subgroup")),
        ("c", long_metric_panel(norman_deg, "deg_auroc", out_dir / "fig4c_norman_deg_auroc.png", "Norman DEG prediction", "DEG AUROC")),
        ("d", long_metric_panel(norman_deg, "direction_agreement_overlap_at_20", out_dir / "fig4d_direction_agreement.png", "Norman direction agreement", "agreement")),
    ]
    return compose_grid(panels, COMP_ROOT / "fig4_main_composite.png", cols=2, cell_w=1080, cell_h=570)


def render_fig5() -> Path:
    out_dir = FIG_ROOT / "main" / "Fig5_DistributionRecovery"
    metrics = collect_prediction_metrics(heldout=True)
    panels = [
        ("a", point_panel(metrics, "scpram_wasserstein_degs_sum", out_dir / "fig5a_wasserstein.png", "Wasserstein distance", "Wasserstein")),
        ("b", point_panel(metrics, "scpram_r2_all_mean_mean", out_dir / "fig5b_mean_rho2.png", "Mean recovery", r"mean $\rho^2$")),
        ("c", point_panel(metrics, "scpram_r2_all_var_mean", out_dir / "fig5c_variance_rho2.png", "Variance recovery", r"variance $\rho^2$")),
        ("d", point_panel(metrics, "r2_degs_var_mean", out_dir / "fig5d_deg_variance.png", "DEG variance recovery", r"DEG variance $\rho^2$")),
    ]
    return compose_grid(panels, COMP_ROOT / "fig5_main_composite.png", cols=4, cell_w=720, cell_h=500)


def render_supp(name: str) -> Path:
    fig = name.lower()
    out_dir = FIG_ROOT / "supp" / SUPP_DIRS[fig]
    if fig == "figs1":
        metrics = collect_prediction_metrics(heldout=True)
        deg = collect_long(ROOT / "artifacts" / "analysis" / "deg_prediction" / "deg_prediction_all_long.csv")
        panels = [
            ("a", point_panel(metrics, "pearson", out_dir / "figs1a_pearson.png", "Reference-transfer Pearson", "Pearson")),
            ("b", point_panel(metrics, "nmse", out_dir / "figs1b_nmse.png", "Reference-transfer nMSE", "nMSE")),
            ("c", point_panel(metrics, "systema_corr_20de_allpert", out_dir / "figs1c_systema.png", "Systema Pearson", "Systema Pearson")),
            ("d", long_metric_panel(deg, "deg_auprc", out_dir / "figs1d_deg_auprc.png", "DEG AUPRC", "AUPRC")),
        ]
        return compose_grid(panels, COMP_ROOT / "figs1_composite.png", cols=2, cell_w=1080, cell_h=570)
    if fig == "figs2":
        metrics = collect_prediction_metrics(heldout=True)
        dixit = metrics[metrics["dataset"].eq("Dixit")].copy() if not metrics.empty else pd.DataFrame()
        deg = collect_long(ROOT / "artifacts" / "analysis" / "deg_prediction" / "deg_prediction_all_long.csv")
        dixit_deg = deg[deg["dataset"].eq("Dixit")].copy() if not deg.empty else pd.DataFrame()
        panels = [
            ("a", point_panel(dixit, "pearson", out_dir / "figs2a_dixit_pearson.png", "Dixit Pearson", "Pearson")),
            ("b", point_panel(dixit, "nmse", out_dir / "figs2b_dixit_nmse.png", "Dixit nMSE", "nMSE")),
            ("c", point_panel(dixit, "systema_corr_20de_allpert", out_dir / "figs2c_dixit_systema.png", "Dixit Systema Pearson", "Systema")),
            ("d", long_metric_panel(dixit_deg, "deg_auroc", out_dir / "figs2d_dixit_deg_auroc.png", "Dixit DEG AUROC", "AUROC")),
        ]
        return compose_grid(panels, COMP_ROOT / "figs2_composite.png", cols=2, cell_w=1080, cell_h=570)
    if fig == "figs3":
        metrics = collect_prediction_metrics(heldout=True)
        panels = [
            ("a", point_panel(metrics, "pearson", out_dir / "figs3a_heldout_pearson.png", "Held-out-control Pearson", "Pearson")),
            ("b", point_panel(metrics, "nmse", out_dir / "figs3b_heldout_nmse.png", "Held-out-control nMSE", "nMSE")),
            ("c", point_panel(metrics, "systema_corr_20de_allpert", out_dir / "figs3c_heldout_systema.png", "Held-out-control Systema", "Systema")),
            ("d", cellot_alignment_panel(out_dir / "figs3d_cellot_aligned.png")),
        ]
        return compose_grid(panels, COMP_ROOT / "figs3_composite.png", cols=2, cell_w=1080, cell_h=570)
    if fig == "figs4":
        ab = collect_ablation_metrics()
        systema = collect_long(ROOT / "artifacts" / "analysis" / "systema_mechanism" / "systema_mechanism_all_long.csv")
        panels = [
            ("a", ablation_panel(ab, ["ref_random", "ref_knn", "ref_ot"], {"ref_random": "random", "ref_knn": "kNN", "ref_ot": "OT"}, out_dir / "figs4a_reference.png", "Reference construction")),
            ("b", ablation_panel(ab, ["cond_no_reference", "cond_no_prior", "cond_full"], {"cond_no_reference": "no ref", "cond_no_prior": "no prior", "cond_full": "full"}, out_dir / "figs4b_conditioning.png", "Conditioning")),
            ("c", ablation_panel(ab, ["loss_mse", "loss_gears", "loss_gears_neg"], {"loss_mse": "L_mse", "loss_gears": "L_gears", "loss_gears_neg": "L_gears+L_neg"}, out_dir / "figs4c_loss.png", "Expression objective")),
            ("d", long_metric_panel(systema, "residualized_systema_corr_20de_allpert", out_dir / "figs4d_residualized.png", "Residualized Systema", "Pearson")),
            ("e", long_metric_panel(systema, "centroid_accuracy", out_dir / "figs4e_centroid.png", "Centroid accuracy", "accuracy")),
            ("f", long_metric_panel(systema, "generic_projection_ratio", out_dir / "figs4f_generic.png", "Generic-shift ratio", "ratio")),
        ]
        return compose_grid(panels, COMP_ROOT / "figs4_composite.png", cols=3, cell_w=760, cell_h=500)
    if fig == "figs5":
        metrics = collect_prediction_metrics(heldout=True)
        norman = metrics[metrics["dataset"].eq("Norman")].copy() if not metrics.empty else pd.DataFrame()
        panels = [
            ("a", heatmap_panel(norman, out_dir / "figs5a_norman_heatmap.png", "Norman subgroup Pearson", "pearson")),
            ("b", heatmap_panel(norman, out_dir / "figs5b_norman_nmse.png", "Norman subgroup nMSE", "nmse")),
            ("c", point_panel(metrics, "scpram_wasserstein_degs_sum", out_dir / "figs5c_wasserstein.png", "Distribution Wasserstein", "Wasserstein")),
            ("d", point_panel(metrics, "scpram_r2_all_var_mean", out_dir / "figs5d_variance.png", "Distribution variance recovery", r"variance $\rho^2$")),
        ]
        return compose_grid(panels, COMP_ROOT / "figs5_composite.png", cols=2, cell_w=1080, cell_h=570)
    if fig == "figs6":
        src = FIG_ROOT / "supp" / "FigS6_Module1LatentState"
        candidate_panels = [
            ("a", src / "figs6a_umap_by_cluster.png"),
            ("b", src / "figs6b_umap_by_cell_type.png"),
            ("c", src / "figs6c_cluster_vs_cell_type.png"),
            ("d", src / "figs6d_cluster_metrics.png"),
            ("e", src / "figs6e_trishift_scgen_unseen_umap.png"),
        ]
        for label, path in candidate_panels:
            if not path.exists():
                no_data_panel(path, f"FigS6{label}", "Run FigS6_Module1LatentState first")
        return compose_grid(candidate_panels, COMP_ROOT / "figs6_composite.png", cols=3, cell_w=760, cell_h=500)
    raise ValueError(f"Unknown supplementary figure: {name}")


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
    if key in {"figs1", "figs2", "figs3", "figs4", "figs5", "figs6"}:
        return render_supp(key)
    if key == "all":
        paths = [render_fig2(), render_fig3(), render_fig4(), render_fig5()]
        paths.extend(render_supp(f"figs{i}") for i in range(1, 7))
        return paths[-1]
    raise ValueError(f"Unknown figure: {figure}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Render refreshed TriShift paper figures from server-side result tables.")
    ap.add_argument("--figure", default="all", help="fig2, fig3, fig4, fig5, figs1-figs6, or all")
    args = ap.parse_args()
    out = render(args.figure)
    print(out)


if __name__ == "__main__":
    main()
