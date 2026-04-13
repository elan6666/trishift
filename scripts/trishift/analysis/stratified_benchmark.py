from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts.trishift.analysis._result_adapter import (
    DATASET_EMBEDDING_KEYS,
    condition_embedding,
    load_metrics_df,
    load_payload_item,
    parse_models,
    parse_split_ids,
    resolve_model_spec,
    resolve_result,
    ts_local,
    warn_skip,
    write_run_meta,
)
from scripts.common.paper_plot_style import apply_gears_paper_style, model_color_map, style_axis
from scripts.common.split_utils import split_by_dataset_policy
from trishift.TriShiftData import TriShiftData
from trishift._utils import apply_alias_mapping, load_adata, load_embedding_df, load_yaml, normalize_condition


SUMMARY_METRICS = [
    "pearson",
    "nmse",
    "deg_mean_r2",
    "systema_corr_20de_allpert",
    "systema_corr_deg_r2",
    "scpram_r2_degs_var_mean",
]
DIFFICULTY_METRICS = [
    "truth_ctrl_shift_norm",
    "train_test_distance",
    "deg_count",
]

LOWER_IS_BETTER = {"nmse"}
METADATA_COLUMNS = [
    "dataset",
    "split_id",
    "condition",
    "subgroup",
    "truth_ctrl_shift_norm",
    "deg_count",
    "train_test_distance",
]


def _resolve_paths_yaml(paths_path: str | Path) -> Path:
    p = Path(paths_path)
    candidates = []
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.extend(
            [
                Path.cwd() / p,
                REPO_ROOT / p,
                REPO_ROOT / "configs" / p.name,
            ]
        )
    for cand in candidates:
        cand = cand.resolve()
        if cand.exists():
            return cand
    raise FileNotFoundError(f"Could not resolve paths yaml: {paths_path}")


def _ordered_labels(values: list[str], preferred: list[str]) -> list[str]:
    present = [label for label in preferred if label in values]
    remaining = sorted([label for label in values if label not in preferred])
    return present + remaining


def _model_color_map(model_names: list[str]) -> dict[str, Any]:
    return model_color_map(model_names)


def _available_summary_metrics(df: pd.DataFrame) -> list[str]:
    if df.empty:
        return []
    metrics: list[str] = []
    for metric in SUMMARY_METRICS:
        if metric not in df.columns:
            continue
        series = pd.to_numeric(df[metric], errors="coerce")
        if series.notna().any():
            metrics.append(metric)
    return metrics


def _available_difficulty_metrics(df: pd.DataFrame) -> list[str]:
    if df.empty:
        return []
    metrics: list[str] = []
    for metric in DIFFICULTY_METRICS:
        if metric not in df.columns:
            continue
        series = pd.to_numeric(df[metric], errors="coerce")
        if series.notna().any():
            metrics.append(metric)
    return metrics


def _qcut_labels(series: pd.Series, labels: list[str]) -> tuple[pd.Series, list[float]]:
    s = pd.to_numeric(series, errors="coerce")
    valid = s.dropna()
    if valid.empty:
        return pd.Series([np.nan] * len(series), index=series.index, dtype=object), []
    try:
        bins = pd.qcut(valid.rank(method="first"), q=len(labels), labels=labels, duplicates="drop")
        q = list(np.quantile(valid.to_numpy(), np.linspace(0.0, 1.0, len(labels) + 1)))
    except Exception:
        bins = pd.Series([labels[len(labels) // 2]] * len(valid), index=valid.index, dtype=object)
        q = [float(valid.min()), float(valid.max())]
    out = pd.Series([np.nan] * len(series), index=series.index, dtype=object)
    out.loc[valid.index] = bins.astype(str)
    return out, q


def _load_dataset_split(data_name: str, split_id: int, paths_path: str | Path) -> dict[str, Any]:
    cfg = load_yaml(str(_resolve_paths_yaml(paths_path)))
    adata = load_adata(cfg["datasets"][data_name])
    emb_key = DATASET_EMBEDDING_KEYS[data_name]
    embd_df = load_embedding_df(cfg["embeddings"][emb_key])
    embd_df = apply_alias_mapping(embd_df, data_name)
    data = TriShiftData(adata, embd_df)
    return split_by_dataset_policy(data, data_name, seed=int(split_id))


def _build_metadata_from_payload(
    dataset: str,
    payload: dict[str, Any],
    split_id: int,
    train_conds: list[str],
    paths_path: str | Path,
) -> pd.DataFrame:
    subgroup_lookup: dict[str, str] = {}
    if dataset == "norman":
        train_single_set = {
            normalize_condition(cond)
            for cond in train_conds
            if "ctrl" in normalize_condition(cond).split("+")
        }
        for key in payload.keys():
            cond_key = normalize_condition(str(key))
            parts = cond_key.split("+")
            if "ctrl" in parts:
                subgroup_lookup[cond_key] = "single"
                continue
            seen_count = sum(
                1 for part in parts if normalize_condition(f"{part}+ctrl") in train_single_set
            )
            subgroup_lookup[cond_key] = f"seen{seen_count}"

    train_embs = {
        normalize_condition(c): condition_embedding(dataset, c, paths_path=paths_path)
        for c in train_conds
    }
    rows: list[dict[str, Any]] = []
    for condition, item in payload.items():
        cond_key = normalize_condition(str(condition))
        truth_full = np.asarray(item["Truth_full"] if "Truth_full" in item else item["Truth"], dtype=np.float32)
        ctrl_full = np.asarray(item["Ctrl_full"] if "Ctrl_full" in item else item["Ctrl"], dtype=np.float32)
        truth_delta = truth_full.mean(axis=0) - ctrl_full.mean(axis=0)
        cond_emb = condition_embedding(dataset, cond_key, paths_path=paths_path)
        dists = [float(np.linalg.norm(cond_emb - emb)) for emb in train_embs.values()] if train_embs else [float("nan")]
        rows.append(
            {
                "dataset": str(dataset),
                "split_id": int(split_id),
                "condition": cond_key,
                "subgroup": subgroup_lookup.get(cond_key, ""),
                "truth_ctrl_shift_norm": float(np.linalg.norm(truth_delta)),
                "deg_count": int(len(np.asarray(item.get("DE_name", [])))),
                "train_test_distance": float(np.nanmin(np.asarray(dists, dtype=float))),
            }
        )
    return pd.DataFrame(rows)


def _choose_reference_payload_model(dataset: str, models: list[str], split_id: int) -> str | None:
    for model_name in models:
        spec = resolve_model_spec(model_name)
        if spec.kind != "payload":
            continue
        try:
            load_payload_item(dataset=dataset, model_name=model_name, split_id=int(split_id), condition=None)
            return model_name
        except Exception:
            continue
    return None


def _render_boxplot(df: pd.DataFrame, out_path: Path, metric: str = "pearson") -> None:
    apply_gears_paper_style()
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=220)
    required_cols = {"effect_strength_bin", "model_name", metric}
    if df.empty or not required_cols.issubset(df.columns):
        ax.text(0.5, 0.5, "No stratified rows", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        return

    work = df.copy()
    work[metric] = pd.to_numeric(work[metric], errors="coerce")
    work = work.dropna(subset=["effect_strength_bin", "model_name", metric])
    if work.empty:
        ax.text(0.5, 0.5, "No valid boxplot rows", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        return

    stratum_labels = _ordered_labels(
        work["effect_strength_bin"].astype(str).dropna().unique().tolist(),
        ["weak", "medium", "strong"],
    )
    model_names = sorted(work["model_name"].astype(str).dropna().unique().tolist())
    color_map = _model_color_map(model_names)

    base_positions = np.arange(len(stratum_labels)) * (len(model_names) + 1.5)
    any_box = False
    for model_idx, model_name in enumerate(model_names):
        arrays: list[np.ndarray] = []
        positions: list[float] = []
        for stratum_idx, stratum_value in enumerate(stratum_labels):
            vals = pd.to_numeric(
                work[
                    (work["effect_strength_bin"].astype(str) == stratum_value)
                    & (work["model_name"].astype(str) == model_name)
                ][metric],
                errors="coerce",
            ).dropna().to_numpy(dtype=float)
            if vals.size == 0:
                continue
            arrays.append(vals)
            positions.append(float(base_positions[stratum_idx] + model_idx))
        if not arrays:
            continue
        any_box = True
        bp = ax.boxplot(
            arrays,
            positions=positions,
            widths=0.7,
            patch_artist=True,
            manage_ticks=False,
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(color_map[model_name])
            patch.set_alpha(0.45)
        for median in bp["medians"]:
            median.set_color(color_map[model_name])
            median.set_linewidth(1.5)
        for whisker in bp["whiskers"]:
            whisker.set_color(color_map[model_name])
        for cap in bp["caps"]:
            cap.set_color(color_map[model_name])

    if not any_box:
        ax.text(0.5, 0.5, "No valid boxplot rows", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        return

    centers = base_positions + (len(model_names) - 1) / 2.0
    ax.set_xticks(centers, stratum_labels)
    ax.set_ylabel(metric)
    ax.set_xlabel("effect_strength_bin")
    ax.set_title(f"Stratified performance ({metric})")
    style_axis(ax, grid_axis="y")
    legend_handles = [
        plt.Line2D([0], [0], color=color_map[name], lw=6, alpha=0.6, label=name)
        for name in model_names
    ]
    ax.legend(handles=legend_handles, frameon=False, ncol=min(4, len(model_names)))
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _render_scatter(df: pd.DataFrame, out_path: Path, metric: str = "pearson") -> None:
    apply_gears_paper_style()
    fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=220)
    required_cols = {"train_test_distance", "model_name", metric}
    if df.empty or not required_cols.issubset(df.columns):
        ax.text(0.5, 0.5, "No difficulty rows", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        return

    work = df.copy()
    work["train_test_distance"] = pd.to_numeric(work["train_test_distance"], errors="coerce")
    work[metric] = pd.to_numeric(work[metric], errors="coerce")
    work = work.dropna(subset=["train_test_distance", metric, "model_name"])
    if work.empty:
        ax.text(0.5, 0.5, "No valid difficulty rows", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        return

    model_names = sorted(work["model_name"].astype(str).dropna().unique().tolist())
    color_map = _model_color_map(model_names)
    for model_name in model_names:
        g = work[work["model_name"].astype(str) == model_name]
        ax.scatter(
            g["train_test_distance"].to_numpy(dtype=float),
            g[metric].to_numpy(dtype=float),
            alpha=0.7,
            s=28,
            label=model_name,
            color=color_map[model_name],
        )
    ax.set_xlabel("train_test_distance")
    ax.set_ylabel(metric)
    ax.set_title(f"Difficulty vs performance ({metric})")
    style_axis(ax, grid_axis="both")
    ax.legend(frameon=False, ncol=min(2, len(model_names)))
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _render_summary_barplot(summary_df: pd.DataFrame, out_path: Path, metric: str = "pearson") -> None:
    apply_gears_paper_style()
    stratum_specs = [
        ("effect_strength_bin", ["weak", "medium", "strong"]),
        ("train_distance_bin", ["near", "medium", "far"]),
        ("deg_difficulty_bin", ["easy", "medium", "hard"]),
    ]
    required_cols = {"model_name", "stratum_name", "stratum_value", metric}
    available_specs = [spec for spec in stratum_specs if spec[0] in summary_df.get("stratum_name", pd.Series(dtype=str)).astype(str).unique().tolist()]
    if summary_df.empty or not required_cols.issubset(summary_df.columns) or not available_specs:
        fig, ax = plt.subplots(figsize=(10, 4), dpi=220)
        ax.text(0.5, 0.5, "No summary rows", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        return

    work = summary_df[["model_name", "stratum_name", "stratum_value", metric]].copy()
    work[metric] = pd.to_numeric(work[metric], errors="coerce")
    work = work.dropna(subset=["model_name", "stratum_name", "stratum_value", metric])
    if work.empty:
        fig, ax = plt.subplots(figsize=(10, 4), dpi=220)
        ax.text(0.5, 0.5, f"No valid rows for {metric}", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        return

    fig, axes = plt.subplots(1, len(available_specs), figsize=(6.6 * len(available_specs), 4.7), dpi=220)
    axes_arr = np.atleast_1d(axes).ravel()
    model_names = sorted(work["model_name"].astype(str).dropna().unique().tolist())
    color_map = _model_color_map(model_names)
    legend_handles: list[Any] = []

    for ax, (stratum_name, preferred_order) in zip(axes_arr, available_specs):
        sub = work[work["stratum_name"].astype(str) == stratum_name].copy()
        if sub.empty:
            ax.text(0.5, 0.5, f"No rows for {stratum_name}", ha="center", va="center")
            ax.axis("off")
            continue
        order = _ordered_labels(sub["stratum_value"].astype(str).unique().tolist(), preferred_order)
        x = np.arange(len(order))
        n_models = max(1, len(model_names))
        width = min(0.8 / n_models, 0.22)
        ymax = float(pd.to_numeric(sub[metric], errors="coerce").max()) if len(sub) else 0.0
        ypad = max(abs(ymax) * 0.10, 0.06)

        any_bar = False
        for model_idx, model_name in enumerate(model_names):
            sub_model = sub[sub["model_name"].astype(str) == model_name].copy()
            if sub_model.empty:
                continue
            any_bar = True
            sub_map = {str(row["stratum_value"]): float(row[metric]) for _, row in sub_model.iterrows()}
            vals = [sub_map.get(bin_name, np.nan) for bin_name in order]
            pos = x + (model_idx - (n_models - 1) / 2.0) * width
            bars = ax.bar(
                pos,
                vals,
                width=width,
                color=color_map[model_name],
                alpha=0.92,
                label=model_name,
                edgecolor="white",
                linewidth=0.8,
            )
            if not legend_handles:
                legend_handles = [
                    plt.Line2D([0], [0], color=color_map[name], lw=6, alpha=0.85, label=name)
                    for name in model_names
                ]
            for bar, val in zip(bars, vals):
                if np.isnan(val):
                    continue
                label_y = float(val) - max(abs(float(val)) * 0.08, 0.035)
                va = "top"
                if label_y <= 0.03:
                    label_y = float(val) + max(abs(float(val)) * 0.03, 0.02)
                    va = "bottom"
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    label_y,
                    f"{float(val):.2f}",
                    ha="center",
                    va=va,
                    fontsize=6,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.65, "pad": 0.2},
                    clip_on=False,
                )
        if not any_bar:
            ax.text(0.5, 0.5, f"No grouped rows for {stratum_name}", ha="center", va="center")
            ax.axis("off")
            continue
        ax.set_xticks(x, order)
        ax.set_title(stratum_name.replace("_", " "))
        ax.set_ylabel(metric.replace("_", " "))
        ax.set_xlabel("")
        ax.set_ylim(top=ymax + ypad)
        ax.set_axisbelow(True)
        style_axis(ax, grid_axis="y")

    fig.suptitle(f"Stratified {metric.replace('_', ' ')}", y=0.98, fontsize=12, fontweight="regular")
    if legend_handles:
        fig.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.99),
            frameon=False,
            ncol=min(3, len(model_names)),
        )
    fig.tight_layout(rect=[0.01, 0.01, 0.995, 0.90])
    fig.savefig(out_path)
    plt.close(fig)


def run_stratified_benchmark(
    *,
    dataset: str,
    models: str | list[str] | tuple[str, ...] | None = None,
    split_ids: int | str | list[int] | tuple[int, ...] = "1",
    out_root: str | Path | None = None,
    paths_path: str | Path = "configs/paths.yaml",
    systema_root: str | Path | None = None,
) -> dict[str, Any]:
    dataset_key = str(dataset).strip()
    model_requests = parse_models(models)
    split_list = parse_split_ids(split_ids)
    out_dir = Path(out_root).resolve() if out_root else (Path("artifacts/analysis") / f"{ts_local()}_stratified_{dataset_key}").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata_frames: list[pd.DataFrame] = []
    metadata_messages: list[str] = []
    thresholds: dict[str, list[float]] = {}
    skipped_models: list[dict[str, Any]] = []

    for split_id in split_list:
        ref_model = _choose_reference_payload_model(dataset_key, model_requests, int(split_id))
        if ref_model is None:
            message = f"[stratified] no payload-backed reference model for dataset={dataset_key} split={split_id}"
            metadata_messages.append(message)
            warn_skip(message)
            continue
        try:
            split_dict = _load_dataset_split(dataset_key, int(split_id), paths_path)
            _, payload = load_payload_item(dataset=dataset_key, model_name=ref_model, split_id=int(split_id), condition=None)
            metadata_frames.append(
                _build_metadata_from_payload(
                    dataset_key,
                    payload,
                    int(split_id),
                    [normalize_condition(c) for c in split_dict.get("train_conds", [])],
                    paths_path,
                )
            )
        except Exception as exc:
            message = f"[stratified] skip split={split_id}: {exc}"
            metadata_messages.append(message)
            warn_skip(message)

    metadata_df = (
        pd.concat(metadata_frames, ignore_index=True).drop_duplicates(["split_id", "condition"])
        if metadata_frames
        else pd.DataFrame(columns=METADATA_COLUMNS)
    )
    if not metadata_df.empty:
        metadata_df["effect_strength_bin"], thresholds["effect_strength_bin"] = _qcut_labels(
            metadata_df["truth_ctrl_shift_norm"], ["weak", "medium", "strong"]
        )
        metadata_df["train_distance_bin"], thresholds["train_distance_bin"] = _qcut_labels(
            metadata_df["train_test_distance"], ["near", "medium", "far"]
        )
        metadata_df["deg_difficulty_bin"], thresholds["deg_difficulty_bin"] = _qcut_labels(
            metadata_df["deg_count"], ["easy", "medium", "hard"]
        )
    elif split_list:
        message = (
            f"[stratified] metadata_df is empty for dataset={dataset_key}; "
            "difficulty scatter and stratified summary outputs will be empty"
        )
        metadata_messages.append(message)
        warn_skip(message)

    merged_frames: list[pd.DataFrame] = []
    for model_name in model_requests:
        try:
            resolved = resolve_result(dataset=dataset_key, model_name=model_name, systema_root=systema_root)
            metrics_df = load_metrics_df(resolved)
        except Exception as exc:
            warn_skip(f"[stratified] skip model={model_name}: {exc}")
            skipped_models.append({"model_name": str(model_name), "reason": str(exc)})
            continue
        if "split_id" in metrics_df.columns:
            metrics_df = metrics_df[metrics_df["split_id"].isin(split_list)].copy()
        if metrics_df.empty:
            skipped_models.append({"model_name": str(model_name), "reason": "no rows after split filter"})
            continue
        metrics_df["condition"] = metrics_df["condition"].astype(str).map(normalize_condition)
        metrics_df["model_name"] = str(model_name)
        if not {"split_id", "condition"}.issubset(metadata_df.columns):
            warn_skip(
                f"[stratified] metadata unavailable for dataset={dataset_key}; "
                f"writing metrics without strata for model={model_name}"
            )
            metadata_df = pd.DataFrame(columns=METADATA_COLUMNS)
        merged = metrics_df.merge(metadata_df, on=["split_id", "condition"], how="left")
        merged_frames.append(merged)

    merged_df = pd.concat(merged_frames, ignore_index=True) if merged_frames else pd.DataFrame()
    strata_rows: list[dict[str, Any]] = []
    if not merged_df.empty:
        for stratum_name in ["subgroup", "effect_strength_bin", "train_distance_bin", "deg_difficulty_bin"]:
            if stratum_name not in merged_df.columns:
                continue
            work = merged_df[merged_df[stratum_name].astype(str).ne("") & merged_df[stratum_name].notna()].copy()
            for keys, g in work.groupby(["model_name", stratum_name], sort=True):
                model_name, stratum_value = keys
                row = {
                    "dataset": dataset_key,
                    "model_name": str(model_name),
                    "stratum_name": str(stratum_name),
                    "stratum_value": str(stratum_value),
                    "n_rows": int(len(g)),
                }
                for metric in SUMMARY_METRICS:
                    if metric in g.columns:
                        row[metric] = float(pd.to_numeric(g[metric], errors="coerce").mean())
                strata_rows.append(row)
    stratified_df = pd.DataFrame(strata_rows)

    summary_rows: list[dict[str, Any]] = []
    win_rows: list[dict[str, Any]] = []
    if not stratified_df.empty:
        for (stratum_name, stratum_value), g in stratified_df.groupby(["stratum_name", "stratum_value"], sort=True):
            best_metric_rows = {}
            for metric in SUMMARY_METRICS:
                if metric not in g.columns:
                    continue
                series = pd.to_numeric(g[metric], errors="coerce")
                if series.notna().sum() == 0:
                    continue
                idx = series.idxmin() if metric in LOWER_IS_BETTER else series.idxmax()
                best_metric_rows[metric] = str(g.loc[idx, "model_name"])
            win_row = {"dataset": dataset_key, "stratum_name": str(stratum_name), "stratum_value": str(stratum_value)}
            for metric, model_name in best_metric_rows.items():
                win_row[f"best_{metric}"] = model_name
            win_rows.append(win_row)
        summary_rows = (
            stratified_df.groupby(["model_name", "stratum_name", "stratum_value"], as_index=False)[SUMMARY_METRICS]
            .mean()
            .to_dict(orient="records")
        )
    summary_df = pd.DataFrame(summary_rows)
    win_df = pd.DataFrame(win_rows)
    metric_summary_df = (
        stratified_df.groupby("model_name", as_index=False)[SUMMARY_METRICS].mean()
        if not stratified_df.empty
        else pd.DataFrame(columns=["model_name", *SUMMARY_METRICS])
    )
    rendered_metrics = ["pearson"] if "pearson" in summary_df.columns else []

    out_dir.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(out_dir / "difficulty_scatter.csv", index=False, encoding="utf-8-sig")
    stratified_df.to_csv(out_dir / "stratified_metrics.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(out_dir / "stratified_summary.csv", index=False, encoding="utf-8-sig")
    win_df.to_csv(out_dir / "model_winrate_by_strata.csv", index=False, encoding="utf-8-sig")
    metric_summary_df.to_csv(out_dir / "stratified_metric_means.csv", index=False, encoding="utf-8-sig")
    metadata_df.to_csv(out_dir / "condition_difficulty_metadata.csv", index=False, encoding="utf-8-sig")
    _render_boxplot(merged_df, out_dir / "stratified_boxplot.png")
    _render_scatter(merged_df, out_dir / "difficulty_scatter.png")
    _render_summary_barplot(summary_df, out_dir / "stratified_summary_barplot.png", metric="pearson")

    write_run_meta(
        out_dir / "run_meta.json",
        {
            "dataset": dataset_key,
            "models": model_requests,
            "split_ids": split_list,
            "paths_path": str(Path(paths_path).resolve()),
            "thresholds": thresholds,
            "rendered_metrics": rendered_metrics,
            "metadata_messages": metadata_messages,
            "skipped_models": skipped_models,
            "out_dir": str(out_dir),
        },
    )
    return {
        "out_dir": out_dir,
        "metadata_df": metadata_df,
        "merged_df": merged_df,
        "stratified_df": stratified_df,
        "summary_df": summary_df,
        "win_df": win_df,
        "metric_summary_df": metric_summary_df,
        "rendered_metrics": rendered_metrics,
        "metadata_messages": metadata_messages,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run stratified robustness benchmark on exported metrics.")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--models", default="trishift_nearest,scouter,gears,genepert,systema_nonctl_mean,systema_matching_mean")
    ap.add_argument("--split_ids", default="1")
    ap.add_argument("--out_root", default="")
    ap.add_argument("--paths_path", default="configs/paths.yaml")
    ap.add_argument("--systema_root", default="")
    args = ap.parse_args(argv)

    result = run_stratified_benchmark(
        dataset=str(args.dataset).strip(),
        models=str(args.models).strip(),
        split_ids=str(args.split_ids).strip(),
        out_root=str(args.out_root).strip() or None,
        paths_path=str(args.paths_path).strip(),
        systema_root=str(args.systema_root).strip() or None,
    )
    print(f"out_dir: {result['out_dir']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
