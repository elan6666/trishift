from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.common.paper_plot_style import apply_gears_paper_style, style_axis
from scripts.trishift.analysis._result_adapter import (
    load_metrics_df,
    load_payload_item,
    normalize_condition,
    parse_split_ids,
    resolve_result,
)


MODEL_LABELS = {
    "trishift_nearest": "TriShift",
    "gears": "GEARS",
    "genepert": "GenePert",
    "scgpt": "scGPT",
    "biolord": "biolord",
}

MODEL_COLORS = {
    "Truth": "#7F7F7F",
    "TriShift": "#9FD9D3",
    "GEARS": "#F2B56B",
    "GenePert": "#87A8D8",
    "scGPT": "#DDD3C8",
    "biolord": "#F0806A",
}


def _mean_1d(x: Any) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        return arr.reshape(-1)
    return arr.mean(axis=0).reshape(-1)


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.size < 2 or y.size != x.size:
        return float("nan")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        return float("nan")
    if float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _sign_agreement(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.size == 0 or y.size != x.size:
        return float("nan")
    return float(np.mean(np.sign(x) == np.sign(y)))


def _truth_delta(item: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    genes = np.asarray(item["gene_name_full"]).astype(str)
    truth = _mean_1d(item["Truth_full"])
    ctrl = _mean_1d(item["Ctrl_full"])
    return genes, truth - ctrl, ctrl


def _pred_delta(item: dict[str, Any]) -> np.ndarray:
    pred = _mean_1d(item["Pred_full"])
    ctrl = _mean_1d(item["Ctrl_full"])
    return pred - ctrl


def _load_payload_map(dataset: str, model: str, split_id: int, result_mode: str) -> dict[str, dict[str, Any]]:
    pkl_path, payload = load_payload_item(
        dataset=dataset,
        model_name=model,
        split_id=int(split_id),
        condition=None,
        result_mode=result_mode,
    )
    return {normalize_condition(str(k)): v for k, v in payload.items() if isinstance(v, dict)}


def _score_case(
    *,
    dataset: str,
    split_id: int,
    condition: str,
    result_mode: str,
    models: list[str],
    payloads: dict[str, dict[str, dict[str, Any]]] | None,
    top_k: int,
    min_effect: float,
) -> tuple[dict[str, Any] | None, pd.DataFrame | None]:
    items: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for model in models:
        try:
            if payloads is not None:
                items[model] = payloads[model][normalize_condition(condition)]
            else:
                items[model] = load_payload_item(
                    dataset=dataset,
                    model_name=model,
                    split_id=int(split_id),
                    condition=condition,
                    result_mode=result_mode,
                )
        except Exception as exc:
            errors.append(f"{model}: {exc}")
    if "trishift_nearest" not in items:
        return None, None
    if any(model not in items for model in models):
        return None, None

    genes, true_delta_all, _ctrl_mean = _truth_delta(items["trishift_nearest"])
    if genes.size != true_delta_all.size:
        return None, None
    top_idx = np.argsort(-np.abs(true_delta_all))[: int(top_k)]
    if top_idx.size < max(4, int(top_k) // 2):
        return None, None
    top_genes = genes[top_idx]
    truth_delta = true_delta_all[top_idx]
    truth_effect = float(np.mean(np.abs(truth_delta)))
    truth_spread = float(np.max(truth_delta) - np.min(truth_delta))
    if not np.isfinite(truth_effect) or truth_effect < float(min_effect):
        return None, None

    rows = []
    for model, item in items.items():
        pred_delta_all = _pred_delta(item)
        if pred_delta_all.size != true_delta_all.size:
            return None, None
        pred_delta = pred_delta_all[top_idx]
        mae = float(np.mean(np.abs(pred_delta - truth_delta)))
        rows.append(
            {
                "dataset": dataset,
                "split_id": int(split_id),
                "condition": condition,
                "model_name": model,
                "label": MODEL_LABELS.get(model, model),
                "pearson_to_truth": _safe_corr(truth_delta, pred_delta),
                "mae_to_truth": mae,
                "sign_agreement": _sign_agreement(truth_delta, pred_delta),
                "truth_abs_mean": truth_effect,
                "truth_spread": truth_spread,
                "top_genes": ";".join(map(str, top_genes.tolist())),
            }
        )
    metric_df = pd.DataFrame(rows)
    tri = metric_df[metric_df["model_name"] == "trishift_nearest"].iloc[0]
    base = metric_df[metric_df["model_name"] != "trishift_nearest"].copy()
    if base.empty:
        return None, None
    best_base_corr = float(base["pearson_to_truth"].max())
    best_base_sign = float(base["sign_agreement"].max())
    best_base_mae = float(base["mae_to_truth"].min())
    median_base_corr = float(base["pearson_to_truth"].median())
    median_base_mae = float(base["mae_to_truth"].median())

    tri_corr = float(tri["pearson_to_truth"])
    tri_sign = float(tri["sign_agreement"])
    tri_mae = float(tri["mae_to_truth"])
    if not np.isfinite(tri_corr) or not np.isfinite(tri_mae):
        return None, None

    corr_gain = tri_corr - best_base_corr
    sign_gain = tri_sign - best_base_sign
    mae_gain = best_base_mae - tri_mae
    median_corr_gain = tri_corr - median_base_corr
    median_mae_gain = median_base_mae - tri_mae
    visible_score = (
        1.8 * corr_gain
        + 1.2 * sign_gain
        + 1.4 * (mae_gain / max(truth_effect, 1e-8))
        + 0.5 * median_corr_gain
        + 0.8 * (median_mae_gain / max(truth_effect, 1e-8))
        + 0.2 * math.log1p(max(truth_effect, 0.0))
    )
    summary = {
        "dataset": dataset,
        "result_mode": result_mode,
        "split_id": int(split_id),
        "condition": condition,
        "top_k": int(top_k),
        "truth_abs_mean": truth_effect,
        "truth_spread": truth_spread,
        "tri_pearson": tri_corr,
        "tri_sign_agreement": tri_sign,
        "tri_mae": tri_mae,
        "best_baseline_pearson": best_base_corr,
        "best_baseline_sign_agreement": best_base_sign,
        "best_baseline_mae": best_base_mae,
        "median_baseline_pearson": median_base_corr,
        "median_baseline_mae": median_base_mae,
        "tri_corr_gain_vs_best": corr_gain,
        "tri_sign_gain_vs_best": sign_gain,
        "tri_mae_gain_vs_best": mae_gain,
        "tri_corr_gain_vs_median": median_corr_gain,
        "tri_mae_gain_vs_median": median_mae_gain,
        "visible_score": float(visible_score),
        "top_genes": ";".join(map(str, top_genes.tolist())),
        "missing_errors": " | ".join(errors[:4]),
    }
    return summary, metric_df


def _plot_case(row: pd.Series, *, models: list[str], out_dir: Path, top_k: int) -> None:
    dataset = str(row["dataset"])
    split_id = int(row["split_id"])
    condition = str(row["condition"])
    result_mode = str(row["result_mode"])
    base_item = load_payload_item(
        dataset=dataset,
        model_name="trishift_nearest",
        split_id=split_id,
        condition=condition,
        result_mode=result_mode,
    )
    genes, true_delta_all, _ctrl_mean = _truth_delta(base_item)
    top_idx = np.argsort(-np.abs(true_delta_all))[: int(top_k)]
    top_genes = genes[top_idx]
    rows = [
        pd.DataFrame(
            {
                "Gene": top_genes,
                "Expression": true_delta_all[top_idx],
                "Group": "Truth",
            }
        )
    ]
    for model in models:
        item = load_payload_item(
            dataset=dataset,
            model_name=model,
            split_id=split_id,
            condition=condition,
            result_mode=result_mode,
        )
        pred_delta = _pred_delta(item)[top_idx]
        rows.append(
            pd.DataFrame(
                {
                    "Gene": top_genes,
                    "Expression": pred_delta,
                    "Group": MODEL_LABELS.get(model, model),
                }
            )
        )
    plot_df = pd.concat(rows, ignore_index=True)
    apply_gears_paper_style(font_scale=1.0)
    fig, ax = plt.subplots(figsize=(12.8, 4.8), dpi=180)
    sns.barplot(
        data=plot_df,
        x="Gene",
        y="Expression",
        hue="Group",
        palette=MODEL_COLORS,
        errorbar=None,
        saturation=0.95,
        ax=ax,
    )
    for patch in ax.patches:
        patch.set_edgecolor("black")
        patch.set_linewidth(0.6)
    ax.axhline(0, color="#4A4A4A", linewidth=0.8)
    ax.set_xlabel("")
    ax.set_ylabel("Expression change over control")
    ax.set_title(f"{dataset} | split {split_id} | {condition} | score={float(row['visible_score']):.3f}")
    ax.tick_params(axis="x", rotation=32)
    ax.legend(title="", frameon=False, ncol=min(3, plot_df["Group"].nunique()), loc="upper center", bbox_to_anchor=(0.5, 1.18))
    style_axis(ax, grid_axis="y")
    fig.tight_layout()
    out_name = f"{dataset}_split{split_id}_{condition.replace('+', '-')}_{result_mode}.png"
    fig.savefig(out_dir / out_name, bbox_inches="tight")
    plt.close(fig)
    plot_df.to_csv(out_dir / out_name.replace(".png", ".csv"), index=False, encoding="utf-8-sig")


def run_selection(
    *,
    datasets: list[str],
    result_mode: str,
    split_ids: list[int],
    models: list[str],
    top_k: int,
    min_effect: float,
    out_dir: Path,
    render_top: int,
    preselect_per_dataset: int,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    candidate_rows: list[dict[str, Any]] = []
    detail_frames: list[pd.DataFrame] = []
    preselected = _preselect_from_metrics(
        datasets=datasets,
        result_mode=result_mode,
        split_ids=split_ids,
        models=models,
        limit_per_dataset=int(preselect_per_dataset),
    )
    payload_cache: dict[tuple[str, int, str], dict[str, dict[str, Any]]] = {}
    for dataset in datasets:
        candidate_pairs = preselected.get(dataset, [])
        if not candidate_pairs:
            candidate_pairs = [(int(split_id), None) for split_id in split_ids]
        for split_id, condition_override in candidate_pairs:
            payloads: dict[str, dict[str, dict[str, Any]]] = {}
            try:
                for model in models:
                    key = (dataset, int(split_id), model)
                    if key not in payload_cache:
                        payload_cache[key] = _load_payload_map(dataset, model, int(split_id), result_mode)
                    payloads[model] = payload_cache[key]
                if condition_override is None:
                    condition_sets = [set(payload.keys()) for payload in payloads.values()]
                    conditions = sorted(set.intersection(*condition_sets)) if condition_sets else []
                else:
                    conditions = [normalize_condition(str(condition_override))]
            except Exception as exc:
                print(f"[warn] skip {dataset} split {split_id}: {exc}")
                continue
            for condition in conditions:
                summary, detail = _score_case(
                    dataset=dataset,
                    split_id=int(split_id),
                    condition=condition,
                    result_mode=result_mode,
                    models=models,
                    payloads=payloads,
                    top_k=int(top_k),
                    min_effect=float(min_effect),
                )
                if summary is None or detail is None:
                    continue
                candidate_rows.append(summary)
                detail_frames.append(detail)

    candidates = pd.DataFrame(candidate_rows)
    if not candidates.empty:
        candidates = candidates.sort_values(
            ["visible_score", "tri_corr_gain_vs_best", "truth_abs_mean"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
        candidates.insert(0, "rank", np.arange(1, len(candidates) + 1))
    details = pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame()
    candidates.to_csv(out_dir / "case_candidates.csv", index=False, encoding="utf-8-sig")
    details.to_csv(out_dir / "case_candidate_model_metrics.csv", index=False, encoding="utf-8-sig")

    recommendations = {}
    if not candidates.empty:
        top = candidates.iloc[0].to_dict()
        recommendations["top"] = {
            k: (v.item() if hasattr(v, "item") else v)
            for k, v in top.items()
            if k not in {"top_genes", "missing_errors"}
        }
        recommendations["top_genes"] = str(top.get("top_genes", "")).split(";")
    (out_dir / "recommended_cases.json").write_text(
        json.dumps(recommendations, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if int(render_top) > 0 and not candidates.empty:
        preview_dir = out_dir / "preview"
        preview_dir.mkdir(parents=True, exist_ok=True)
        for _, row in candidates.head(int(render_top)).iterrows():
            _plot_case(row, models=models, out_dir=preview_dir, top_k=int(top_k))

    return {
        "n_candidates": int(len(candidates)),
        "out_dir": str(out_dir),
        "top": recommendations.get("top", {}),
    }


def _preselect_from_metrics(
    *,
    datasets: list[str],
    result_mode: str,
    split_ids: list[int],
    models: list[str],
    limit_per_dataset: int,
) -> dict[str, list[tuple[int, str]]]:
    if int(limit_per_dataset) <= 0:
        return {}
    selected: dict[str, list[tuple[int, str]]] = {}
    for dataset in datasets:
        frames = []
        for model in models:
            try:
                resolved = resolve_result(dataset=dataset, model_name=model, result_mode=result_mode)
                df = load_metrics_df(resolved).copy()
            except Exception as exc:
                print(f"[warn] metrics unavailable for {dataset}/{model}/{result_mode}: {exc}")
                continue
            if not {"split_id", "condition", "pearson", "nmse"}.issubset(df.columns):
                continue
            sub = df[df["split_id"].astype(int).isin([int(x) for x in split_ids])].copy()
            sub["condition"] = sub["condition"].astype(str).map(normalize_condition)
            sub = sub[["split_id", "condition", "pearson", "nmse"]].copy()
            sub = sub.rename(columns={"pearson": f"pearson__{model}", "nmse": f"nmse__{model}"})
            frames.append(sub)
        if not frames:
            continue
        merged = frames[0]
        for frame in frames[1:]:
            merged = merged.merge(frame, on=["split_id", "condition"], how="inner")
        if merged.empty or "pearson__trishift_nearest" not in merged.columns:
            continue
        baseline_cols = [c for c in merged.columns if c.startswith("pearson__") and c != "pearson__trishift_nearest"]
        if not baseline_cols:
            continue
        nmse_base_cols = [c for c in merged.columns if c.startswith("nmse__") and c != "nmse__trishift_nearest"]
        merged["best_baseline_pearson"] = merged[baseline_cols].max(axis=1)
        merged["median_baseline_pearson"] = merged[baseline_cols].median(axis=1)
        merged["corr_gain"] = merged["pearson__trishift_nearest"] - merged["best_baseline_pearson"]
        if "nmse__trishift_nearest" in merged.columns and nmse_base_cols:
            merged["best_baseline_nmse"] = merged[nmse_base_cols].min(axis=1)
            merged["nmse_gain"] = merged["best_baseline_nmse"] - merged["nmse__trishift_nearest"]
        else:
            merged["nmse_gain"] = 0.0
        merged["preselect_score"] = (
            merged["corr_gain"].fillna(-9.0)
            + 0.25 * (merged["pearson__trishift_nearest"].fillna(0.0))
            + 0.35 * merged["nmse_gain"].fillna(0.0)
        )
        merged = merged.sort_values("preselect_score", ascending=False)
        selected[dataset] = [
            (int(row["split_id"]), str(row["condition"]))
            for _, row in merged.head(int(limit_per_dataset)).iterrows()
        ]
        out_preview = merged.head(int(limit_per_dataset)).copy()
        preview_dir = ROOT / "artifacts" / "analysis" / "case_selection" / str(result_mode)
        preview_dir.mkdir(parents=True, exist_ok=True)
        out_preview.to_csv(preview_dir / f"{dataset}_metric_preselection.csv", index=False, encoding="utf-8-sig")
    return selected


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Select visible qualitative cases for TriShift figures.")
    ap.add_argument("--datasets", default="adamson,norman,dixit")
    ap.add_argument("--result_mode", default="default", choices=["default", "unseen_ctrl"])
    ap.add_argument("--split_ids", default="1,2,3,4,5")
    ap.add_argument("--models", default="trishift_nearest,gears,genepert,scgpt,biolord")
    ap.add_argument("--top_k", type=int, default=12)
    ap.add_argument("--min_effect", type=float, default=0.03)
    ap.add_argument("--out_dir", default="")
    ap.add_argument("--render_top", type=int, default=12)
    ap.add_argument("--preselect_per_dataset", type=int, default=60)
    args = ap.parse_args(argv)

    datasets = [x.strip() for x in str(args.datasets).split(",") if x.strip()]
    models = [x.strip() for x in str(args.models).split(",") if x.strip()]
    out_dir = Path(args.out_dir).resolve() if str(args.out_dir).strip() else (
        ROOT / "artifacts" / "analysis" / "case_selection" / str(args.result_mode)
    ).resolve()
    result = run_selection(
        datasets=datasets,
        result_mode=str(args.result_mode),
        split_ids=parse_split_ids(str(args.split_ids)),
        models=models,
        top_k=int(args.top_k),
        min_effect=float(args.min_effect),
        out_dir=out_dir,
        render_top=int(args.render_top),
        preselect_per_dataset=int(args.preselect_per_dataset),
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
