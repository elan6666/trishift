from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dcor import distance_correlation
from sklearn.linear_model import TheilSenRegressor

import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts.common.split_utils import norman_subgroup
from scripts.trishift.analysis._result_adapter import (
    normalize_condition,
    parse_models,
    parse_split_ids,
    load_payload_item,
    ts_local,
    warn_skip,
    write_run_meta,
)

GI_METRICS_ALL = [
    "c1",
    "c2",
    "mag",
    "corr_fit",
    "dominance",
    "eq_contr",
    "dcor",
    "dcor_singles",
    "dcor_first",
    "dcor_second",
]
GI_METRICS_CORE = ["c1", "c2", "mag", "corr_fit", "dominance", "eq_contr"]
DEFAULT_MODELS = ["trishift_nearest", "scouter", "gears", "genepert", "scgpt"]


def _normalize_space(space: str) -> str:
    key = normalize_condition(str(space))
    aliases = {
        "full": "full_gene",
        "fullgene": "full_gene",
        "full_gene": "full_gene",
    }
    if key not in aliases:
        raise ValueError("space must be full_gene")
    return aliases[key]


def _gene_name_array(item: dict[str, Any], key: str) -> np.ndarray:
    if key not in item:
        raise KeyError(f"Missing {key}")
    arr = np.asarray(item[key], dtype=object).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{key} is empty")
    return arr


def _full_delta(item: dict[str, Any], expr_key: str) -> tuple[np.ndarray, np.ndarray]:
    genes = _gene_name_array(item, "gene_name_full")
    expr = np.asarray(item[expr_key], dtype=np.float32)
    ctrl = np.asarray(item["Ctrl_full"], dtype=np.float32)
    if expr.ndim != 2 or ctrl.ndim != 2:
        raise ValueError(f"{expr_key} and Ctrl_full must be 2D arrays")
    if expr.shape[1] != genes.size or ctrl.shape[1] != genes.size:
        raise ValueError(f"{expr_key} / Ctrl_full width does not match gene_name_full")
    delta = expr.mean(axis=0).astype(np.float32, copy=False) - ctrl.mean(axis=0).astype(np.float32, copy=False)
    return genes, delta


def _aligned_triplet_deltas(
    *,
    item_ab: dict[str, Any],
    item_a: dict[str, Any],
    item_b: dict[str, Any],
    expr_key: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    genes_ab, delta_ab = _full_delta(item_ab, expr_key)
    genes_a, delta_a = _full_delta(item_a, expr_key)
    genes_b, delta_b = _full_delta(item_b, expr_key)
    index_a = {str(g): i for i, g in enumerate(genes_a.tolist())}
    index_b = {str(g): i for i, g in enumerate(genes_b.tolist())}
    common_genes = [str(g) for g in genes_ab.tolist() if str(g) in index_a and str(g) in index_b]
    if len(common_genes) < 2:
        raise ValueError("Need at least 2 shared genes across combo and both singles in full_gene space")
    idx_ab = np.asarray([i for i, g in enumerate(genes_ab.tolist()) if str(g) in set(common_genes)], dtype=int)
    idx_a = np.asarray([index_a[g] for g in common_genes], dtype=int)
    idx_b = np.asarray([index_b[g] for g in common_genes], dtype=int)
    return (
        delta_a[idx_a].astype(np.float64, copy=False),
        delta_b[idx_b].astype(np.float64, copy=False),
        delta_ab[idx_ab].astype(np.float64, copy=False),
    )


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    a = np.asarray(x, dtype=float).reshape(-1)
    b = np.asarray(y, dtype=float).reshape(-1)
    if a.size == 0 or b.size == 0:
        return float("nan")
    if np.allclose(np.std(a), 0.0) or np.allclose(np.std(b), 0.0):
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _safe_distance_correlation(x: np.ndarray, y: np.ndarray) -> float:
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    if a.size == 0 or b.size == 0:
        return float("nan")
    try:
        return float(distance_correlation(a, b))
    except Exception:
        return float("nan")


def _safe_ratio(numer: float, denom: float) -> float:
    if np.isnan(numer) or np.isnan(denom) or np.isclose(denom, 0.0):
        return float("nan")
    return float(numer / denom)


def _safe_log_ratio(c1: float, c2: float) -> float:
    if np.isnan(c1) or np.isnan(c2) or np.isclose(c1, 0.0) or np.isclose(c2, 0.0):
        return float("nan")
    with np.errstate(divide="ignore", invalid="ignore"):
        value = np.abs(np.log10(c1 / c2))
    return float(value) if np.isfinite(value) else float("nan")


def _compute_gi_params(first_expr: np.ndarray, second_expr: np.ndarray, double_expr: np.ndarray) -> dict[str, float]:
    x = np.column_stack([np.asarray(first_expr, dtype=float), np.asarray(second_expr, dtype=float)])
    y = np.asarray(double_expr, dtype=float).reshape(-1)
    if x.shape[0] < 2:
        raise ValueError("Need at least 2 genes to fit GI parameters")
    model = TheilSenRegressor(
        fit_intercept=False,
        max_subpopulation=int(1e5),
        max_iter=1000,
        random_state=1000,
    )
    model.fit(x, y)
    fit = np.asarray(model.predict(x), dtype=float).reshape(-1)
    c1 = float(model.coef_[0])
    c2 = float(model.coef_[1])
    dcor_first = _safe_distance_correlation(first_expr, y)
    dcor_second = _safe_distance_correlation(second_expr, y)
    eq_denom = np.nanmax([dcor_first, dcor_second])
    eq_numer = np.nanmin([dcor_first, dcor_second])
    return {
        "c1": c1,
        "c2": c2,
        "mag": float(math.sqrt(c1**2 + c2**2)),
        "corr_fit": _safe_corr(fit, y),
        "dominance": _safe_log_ratio(c1, c2),
        "eq_contr": _safe_ratio(eq_numer, eq_denom),
        "dcor": _safe_distance_correlation(x, y),
        "dcor_singles": _safe_distance_correlation(first_expr, second_expr),
        "dcor_first": dcor_first,
        "dcor_second": dcor_second,
    }


def _combo_single_keys(condition: str) -> tuple[str, str]:
    tokens = [x for x in normalize_condition(str(condition)).split("+") if x and x != "ctrl"]
    if len(tokens) != 2:
        raise ValueError("Only 2-gene combo conditions are supported")
    a, b = sorted(tokens)
    return normalize_condition(f"{a}+ctrl"), normalize_condition(f"{b}+ctrl")


def _build_subgroup_lookup(conditions: list[str], split_id: int) -> dict[str, str]:
    pert_list = [normalize_condition(x) for x in conditions]
    if "ctrl" not in pert_list:
        pert_list = ["ctrl"] + pert_list
    try:
        df = norman_subgroup(pert_list, seed=int(split_id))
        return {normalize_condition(str(idx)): str(row["subgroup"]) for idx, row in df.iterrows()}
    except Exception:
        out: dict[str, str] = {}
        for cond in pert_list:
            key = normalize_condition(cond)
            tokens = [x for x in key.split("+") if x and x != "ctrl"]
            if key == "ctrl":
                out[key] = "unknown"
            elif len(tokens) == 1:
                out[key] = "single"
            else:
                out[key] = "seen0"
        return out


def _compute_condition_gi(
    *,
    item_ab: dict[str, Any],
    item_a: dict[str, Any],
    item_b: dict[str, Any],
) -> dict[str, float]:
    truth_a, truth_b, truth_ab = _aligned_triplet_deltas(item_ab=item_ab, item_a=item_a, item_b=item_b, expr_key="Truth_full")
    pred_a, pred_b, pred_ab = _aligned_triplet_deltas(item_ab=item_ab, item_a=item_a, item_b=item_b, expr_key="Pred_full")
    truth_metrics = _compute_gi_params(truth_a, truth_b, truth_ab)
    pred_metrics = _compute_gi_params(pred_a, pred_b, pred_ab)
    row: dict[str, float] = {}
    for metric in GI_METRICS_ALL:
        truth_value = float(truth_metrics.get(metric, float("nan")))
        pred_value = float(pred_metrics.get(metric, float("nan")))
        row[f"truth_{metric}"] = truth_value
        row[f"pred_{metric}"] = pred_value
        row[f"signed_err_{metric}"] = float(pred_value - truth_value) if np.isfinite(truth_value) and np.isfinite(pred_value) else float("nan")
        row[f"abs_err_{metric}"] = float(abs(pred_value - truth_value)) if np.isfinite(truth_value) and np.isfinite(pred_value) else float("nan")
    return row


def _render_metric_panel(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.5), dpi=220)
    axes_flat = axes.flatten()
    if df.empty:
        for ax in axes_flat:
            ax.axis("off")
        axes_flat[0].text(0.5, 0.5, "No GI summary rows", ha="center", va="center")
    else:
        work = df.copy()
        labels = work["model_name"].astype(str).tolist()
        x = np.arange(len(labels))
        for ax, metric in zip(axes_flat, GI_METRICS_CORE):
            y = pd.to_numeric(work.get(f"abs_err_{metric}_mean"), errors="coerce")
            ax.bar(x, y.fillna(0.0), color="#4C72B0")
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=20, ha="right")
            ax.set_ylabel(f"mean abs_err_{metric}")
            ax.set_title(metric)
            ax.grid(alpha=0.2, axis="y")
        for ax in axes_flat[len(GI_METRICS_CORE) :]:
            ax.axis("off")
    fig.suptitle("Norman GI metric panel", y=0.995)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def _render_truth_pred_scatter(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.5), dpi=220)
    axes_flat = axes.flatten()
    if df.empty:
        for ax in axes_flat:
            ax.axis("off")
        axes_flat[0].text(0.5, 0.5, "No valid GI conditions", ha="center", va="center")
    else:
        model_names = [str(x) for x in sorted(df["model_name"].astype(str).unique().tolist())]
        cmap = plt.get_cmap("tab10")
        colors = {name: cmap(i % 10) for i, name in enumerate(model_names)}
        for ax, metric in zip(axes_flat, GI_METRICS_CORE):
            for model_name in model_names:
                sub = df[df["model_name"].astype(str) == model_name]
                x = pd.to_numeric(sub.get(f"truth_{metric}"), errors="coerce")
                y = pd.to_numeric(sub.get(f"pred_{metric}"), errors="coerce")
                ax.scatter(x, y, s=28, alpha=0.8, color=colors[model_name], label=model_name)
            ax.set_xlabel(f"truth_{metric}")
            ax.set_ylabel(f"pred_{metric}")
            ax.set_title(metric)
            ax.grid(alpha=0.2)
        handles, labels = axes_flat[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=min(5, len(labels)))
        for ax in axes_flat[len(GI_METRICS_CORE) :]:
            ax.axis("off")
    fig.suptitle("Norman GI truth vs prediction", y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_path)
    plt.close(fig)


def _render_subgroup_barplot(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.5), dpi=220)
    axes_flat = axes.flatten()
    if df.empty:
        for ax in axes_flat:
            ax.axis("off")
        axes_flat[0].text(0.5, 0.5, "No subgroup GI results", ha="center", va="center")
    else:
        work = df.copy()
        subgroup_order = ["seen0", "seen1", "seen2"]
        model_order = [str(x) for x in sorted(work["model_name"].astype(str).unique().tolist())]
        work["subgroup"] = pd.Categorical(work["subgroup"], categories=subgroup_order, ordered=True)
        work = work.sort_values(["subgroup", "model_name"])
        x = np.arange(len(subgroup_order))
        width = 0.8 / max(1, len(model_order))
        cmap = plt.get_cmap("tab10")
        for ax, metric in zip(axes_flat, GI_METRICS_CORE):
            for idx, model_name in enumerate(model_order):
                sub = work[work["model_name"].astype(str) == model_name]
                sub = sub.set_index("subgroup").reindex(subgroup_order)
                y = pd.to_numeric(sub.get(f"abs_err_{metric}_mean"), errors="coerce").fillna(0.0)
                ax.bar(x + (idx - (len(model_order) - 1) / 2.0) * width, y, width=width, label=model_name, color=cmap(idx % 10))
            ax.set_xticks(x)
            ax.set_xticklabels(subgroup_order)
            ax.set_ylabel(f"mean abs_err_{metric}")
            ax.set_title(metric)
            ax.grid(alpha=0.2, axis="y")
        handles, labels = axes_flat[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=min(5, len(labels)))
        for ax in axes_flat[len(GI_METRICS_CORE) :]:
            ax.axis("off")
    fig.suptitle("Norman GI by subgroup", y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_path)
    plt.close(fig)


def run_norman_nonadd_experiment(
    *,
    dataset: str,
    models: str | list[str] | tuple[str, ...] | None = None,
    split_ids: int | str | list[int] | tuple[int, ...] = "1",
    out_root: str | Path | None = None,
    space: str = "full_gene",
) -> dict[str, Any]:
    dataset_key = normalize_condition(str(dataset))
    if dataset_key != "norman":
        raise ValueError("norman_nonadd_experiment only supports dataset=norman")
    space_key = _normalize_space(space)
    model_requests = parse_models(models or DEFAULT_MODELS)
    split_list = parse_split_ids(split_ids)
    out_dir = Path(out_root).resolve() if out_root else (Path("artifacts/analysis") / f"{ts_local()}_norman_nonadd").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []
    used_models: list[str] = []
    for model_name in model_requests:
        try:
            _, payload = load_payload_item(dataset="norman", model_name=model_name, split_id=int(split_list[0]), condition=None)
        except Exception:
            payload = None
        if payload is None:
            warn_skip(f"[norman_nonadd] skip model={model_name}: payload unavailable")
            skipped_rows.append({"model_name": str(model_name), "split_id": "", "condition": "", "skip_reason": "payload unavailable"})
            continue
        used_models.append(str(model_name))

        for split_id in split_list:
            try:
                _, payload = load_payload_item(dataset="norman", model_name=model_name, split_id=int(split_id), condition=None)
            except Exception as exc:
                warn_skip(f"[norman_nonadd] skip model={model_name} split={split_id}: {exc}")
                skipped_rows.append({"model_name": str(model_name), "split_id": int(split_id), "condition": "", "skip_reason": str(exc)})
                continue
            normalized_payload = {normalize_condition(str(k)): v for k, v in payload.items() if isinstance(v, dict)}
            subgroup_lookup = _build_subgroup_lookup(list(normalized_payload.keys()), int(split_id))
            for condition, item_ab in sorted(normalized_payload.items()):
                tokens = [x for x in condition.split("+") if x and x != "ctrl"]
                if len(tokens) != 2:
                    continue
                try:
                    single_a, single_b = _combo_single_keys(condition)
                except Exception as exc:
                    skipped_rows.append({"model_name": str(model_name), "split_id": int(split_id), "condition": str(condition), "skip_reason": str(exc)})
                    continue
                if single_a not in normalized_payload or single_b not in normalized_payload:
                    missing = single_a if single_a not in normalized_payload else single_b
                    skipped_rows.append(
                        {
                            "model_name": str(model_name),
                            "split_id": int(split_id),
                            "condition": str(condition),
                            "skip_reason": f"missing single perturbation: {missing}",
                        }
                    )
                    continue
                try:
                    gi_row = _compute_condition_gi(
                        item_ab=item_ab,
                        item_a=normalized_payload[single_a],
                        item_b=normalized_payload[single_b],
                    )
                except Exception as exc:
                    skipped_rows.append({"model_name": str(model_name), "split_id": int(split_id), "condition": str(condition), "skip_reason": str(exc)})
                    continue
                rows.append(
                    {
                        "dataset": "norman",
                        "model_name": str(model_name),
                        "split_id": int(split_id),
                        "condition": str(condition),
                        "subgroup": subgroup_lookup.get(condition, "unknown"),
                        "space_used": str(space_key),
                        **gi_row,
                    }
                )

    per_condition_df = pd.DataFrame(rows).sort_values(["model_name", "split_id", "condition"]).reset_index(drop=True) if rows else pd.DataFrame()
    skipped_df = pd.DataFrame(skipped_rows)

    agg_cols = [f"{prefix}_{metric}" for prefix in ("truth", "pred", "abs_err", "signed_err") for metric in GI_METRICS_ALL]
    summary_df = (
        per_condition_df.groupby(["model_name"], as_index=False)[agg_cols]
        .mean()
        .rename(columns={c: f"{c}_mean" for c in agg_cols})
        if not per_condition_df.empty
        else pd.DataFrame()
    )
    subgroup_df = (
        per_condition_df.groupby(["model_name", "subgroup"], as_index=False)[agg_cols]
        .mean()
        .rename(columns={c: f"{c}_mean" for c in agg_cols})
        if not per_condition_df.empty
        else pd.DataFrame()
    )

    per_condition_df.to_csv(out_dir / "norman_gi_per_condition.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(out_dir / "norman_gi_summary.csv", index=False, encoding="utf-8-sig")
    subgroup_df.to_csv(out_dir / "norman_gi_by_subgroup.csv", index=False, encoding="utf-8-sig")
    skipped_df.to_csv(out_dir / "norman_gi_skipped.csv", index=False, encoding="utf-8-sig")
    _render_metric_panel(summary_df, out_dir / "norman_gi_metric_panel.png")
    _render_truth_pred_scatter(per_condition_df, out_dir / "norman_gi_truth_vs_pred_scatter.png")
    _render_subgroup_barplot(subgroup_df, out_dir / "norman_gi_subgroup_barplot.png")

    write_run_meta(
        out_dir / "run_meta.json",
        {
            "dataset": "norman",
            "models_requested": model_requests,
            "models_used": used_models,
            "split_ids": split_list,
            "space": str(space_key),
            "metrics_rendered": GI_METRICS_CORE,
            "metrics_all": GI_METRICS_ALL,
            "skipped_rows": skipped_rows,
            "out_dir": str(out_dir),
        },
    )
    return {
        "out_dir": out_dir,
        "per_condition_df": per_condition_df,
        "summary_df": summary_df,
        "subgroup_df": subgroup_df,
        "skipped_df": skipped_df,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run GEARS-style Norman GI analysis from exported PKLs.")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--models", default="trishift_nearest,scouter,gears,genepert,scgpt")
    ap.add_argument("--split_ids", default="1")
    ap.add_argument("--out_root", default="")
    ap.add_argument("--space", default="full_gene", choices=["full_gene"])
    args = ap.parse_args(argv)

    result = run_norman_nonadd_experiment(
        dataset=str(args.dataset).strip(),
        models=str(args.models).strip(),
        split_ids=str(args.split_ids).strip(),
        out_root=str(args.out_root).strip() or None,
        space=str(args.space).strip(),
    )
    print(f"out_dir: {result['out_dir']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
