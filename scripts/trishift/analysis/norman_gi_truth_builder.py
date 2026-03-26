from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd

import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from trishift._utils import load_yaml
from scripts.trishift.analysis._result_adapter import normalize_condition, write_run_meta
from scripts.trishift.analysis.norman_nonadd_experiment import GI_METRICS_ALL, _combo_single_keys, _compute_gi_params

GI_TYPE_ORDER = [
    "synergy",
    "suppressor",
    "neomorphic",
    "redundancy",
    "epistasis",
    "strong_interaction",
    "balanced_synergy",
    "dominant_epistasis",
]
GI_TYPE_LABELS = {
    "synergy": "Synergy",
    "suppressor": "Suppression",
    "neomorphic": "Neomorphism",
    "redundancy": "Redundancy",
    "epistasis": "Epistasis",
    "strong_interaction": "Strong Interaction",
    "balanced_synergy": "Balanced Synergy",
    "dominant_epistasis": "Dominant Epistasis",
}
GI_TYPE_RULES = {
    "synergy": {
        "logic": "all",
        "criteria": [{"metric_name": "mag", "direction": "high"}],
    },
    "suppressor": {
        "logic": "all",
        "criteria": [{"metric_name": "mag", "direction": "low"}],
    },
    "neomorphic": {
        "logic": "all",
        "criteria": [{"metric_name": "corr_fit", "direction": "low"}],
    },
    "redundancy": {
        "logic": "all",
        "criteria": [{"metric_name": "dcor", "direction": "high"}],
    },
    "epistasis": {
        "logic": "all",
        "criteria": [{"metric_name": "eq_contr", "direction": "low"}],
    },
    "strong_interaction": {
        "logic": "all",
        "criteria": [
            {"metric_name": "mag", "direction": "high"},
            {"metric_name": "corr_fit", "direction": "low"},
        ],
    },
    "balanced_synergy": {
        "logic": "all",
        "criteria": [
            {"metric_name": "mag", "direction": "high"},
            {"metric_name": "eq_contr", "direction": "high"},
        ],
    },
    "dominant_epistasis": {
        "logic": "all",
        "criteria": [
            {"metric_name": "eq_contr", "direction": "low"},
            {"metric_name": "dominance", "direction": "high"},
        ],
    },
}


def _resolve_norman_truth_h5ad_path() -> Path:
    default_path = (REPO_ROOT / "src" / "data" / "norman" / "perturb_processed.h5ad").resolve()
    if default_path.exists():
        return default_path
    paths_cfg_path = REPO_ROOT / "configs" / "paths.yaml"
    if paths_cfg_path.exists():
        paths_cfg = load_yaml(str(paths_cfg_path))
        datasets = paths_cfg.get("datasets", {}) or {}
        raw = str(datasets.get("norman", "")).strip()
        if raw:
            alt = Path(raw).resolve()
            if alt.exists():
                return alt
    raise FileNotFoundError(f"Norman truth h5ad not found. Tried {default_path} and configs/paths.yaml.")


def _mean_vector(mat) -> np.ndarray:
    return np.asarray(mat.mean(axis=0)).reshape(-1).astype(np.float64, copy=False)


def _compute_condition_means(adata: ad.AnnData) -> tuple[dict[str, np.ndarray], dict[str, int], np.ndarray]:
    if "condition" not in adata.obs.columns:
        raise ValueError("Norman h5ad is missing obs['condition']")
    conditions = adata.obs["condition"].astype(str).map(normalize_condition)
    unique_conditions = sorted(set(conditions.tolist()))
    if "ctrl" not in unique_conditions:
        raise ValueError("Norman h5ad is missing ctrl condition")
    cond_means: dict[str, np.ndarray] = {}
    cond_counts: dict[str, int] = {}
    cond_arr = conditions.to_numpy()
    for condition in unique_conditions:
        mask = cond_arr == condition
        if not np.any(mask):
            continue
        cond_means[str(condition)] = _mean_vector(adata[mask].X)
        cond_counts[str(condition)] = int(mask.sum())
    genes = adata.var["gene_name"].astype(str).to_numpy()
    return cond_means, cond_counts, genes


def _collect_truth_rows_from_raw_h5ad(h5ad_path: Path) -> pd.DataFrame:
    adata = ad.read_h5ad(h5ad_path)
    cond_means, cond_counts, genes = _compute_condition_means(adata)
    ctrl_mean = cond_means["ctrl"]
    rows: list[dict[str, Any]] = []
    for condition in sorted(cond_means.keys()):
        if condition == "ctrl":
            continue
        tokens = [x for x in condition.split("+") if x and x != "ctrl"]
        if len(tokens) != 2:
            continue
        single_a, single_b = _combo_single_keys(condition)
        if single_a not in cond_means or single_b not in cond_means:
            continue
        delta_a = cond_means[single_a] - ctrl_mean
        delta_b = cond_means[single_b] - ctrl_mean
        delta_ab = cond_means[condition] - ctrl_mean
        truth_metrics = _compute_gi_params(delta_a, delta_b, delta_ab)
        rows.append(
            {
                "dataset": "norman",
                "condition": str(condition),
                "single_a": str(single_a),
                "single_b": str(single_b),
                "n_cells_combo": int(cond_counts.get(condition, 0)),
                "n_cells_single_a": int(cond_counts.get(single_a, 0)),
                "n_cells_single_b": int(cond_counts.get(single_b, 0)),
                "n_genes": int(len(genes)),
                **{f"truth_{metric}": float(truth_metrics.get(metric, float("nan"))) for metric in GI_METRICS_ALL},
            }
        )
    return pd.DataFrame(rows)


def _safe_quantile(values: pd.Series, q: float) -> float:
    work = pd.to_numeric(values, errors="coerce").dropna()
    if work.empty:
        return float("nan")
    return float(work.quantile(float(q)))


def _criterion_mask(work: pd.DataFrame, *, metric_name: str, direction: str, low_quantile: float, high_quantile: float) -> tuple[pd.Series, dict[str, Any]]:
    metric_col = f"truth_{metric_name}"
    quantile = float(high_quantile if direction == "high" else low_quantile)
    threshold = _safe_quantile(work[metric_col], quantile)
    if np.isnan(threshold):
        mask = pd.Series(False, index=work.index, dtype=bool)
    elif direction == "high":
        mask = pd.to_numeric(work[metric_col], errors="coerce") >= float(threshold)
    else:
        mask = pd.to_numeric(work[metric_col], errors="coerce") <= float(threshold)
    return mask.fillna(False).astype(bool), {
        "metric_name": metric_name,
        "direction": direction,
        "quantile": quantile,
        "threshold": threshold,
    }


def _apply_truth_labels(
    truth_df: pd.DataFrame,
    *,
    low_quantile: float,
    high_quantile: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, Any]]]:
    work = truth_df.copy()
    summary_rows: list[dict[str, Any]] = []
    thresholds: dict[str, dict[str, Any]] = {}
    for gi_type in GI_TYPE_ORDER:
        spec = GI_TYPE_RULES[gi_type]
        criterion_masks: list[pd.Series] = []
        criterion_meta: list[dict[str, Any]] = []
        for criterion in spec["criteria"]:
            mask, meta = _criterion_mask(
                work,
                metric_name=str(criterion["metric_name"]),
                direction=str(criterion["direction"]),
                low_quantile=float(low_quantile),
                high_quantile=float(high_quantile),
            )
            criterion_masks.append(mask)
            criterion_meta.append(meta)
        if not criterion_masks:
            labels = pd.Series(False, index=work.index, dtype=bool)
        elif str(spec.get("logic", "all")) == "any":
            labels = pd.concat(criterion_masks, axis=1).any(axis=1)
        else:
            labels = pd.concat(criterion_masks, axis=1).all(axis=1)
        label_col = f"is_{gi_type}"
        work[label_col] = labels.fillna(False).astype(bool)
        positive_n = int(work[label_col].sum())
        thresholds[gi_type] = {
            "logic": str(spec.get("logic", "all")),
            "criteria": criterion_meta,
            "positive_n": positive_n,
        }
        summary_rows.append(
            {
                "gi_type": gi_type,
                "gi_label": GI_TYPE_LABELS[gi_type],
                "logic": str(spec.get("logic", "all")),
                "criteria": "; ".join(
                    f"{m['metric_name']}:{m['direction']}@q={m['quantile']:.2f} thr={m['threshold']:.6g}"
                    for m in criterion_meta
                ),
                "positive_n": positive_n,
                "positive_fraction": float(positive_n / len(work)) if len(work) else float("nan"),
            }
        )
    label_cols = [f"is_{gi_type}" for gi_type in GI_TYPE_ORDER]
    work["is_any_gi"] = work[label_cols].any(axis=1) if label_cols else False
    return work, pd.DataFrame(summary_rows), thresholds


def run_norman_gi_truth_builder(
    *,
    dataset: str,
    models: str | list[str] | tuple[str, ...] | None = None,
    split_ids: int | str | list[int] | tuple[int, ...] = "1,2,3,4,5",
    out_root: str | Path | None = None,
    low_quantile: float = 0.2,
    high_quantile: float = 0.8,
) -> dict[str, Any]:
    dataset_key = normalize_condition(str(dataset))
    if dataset_key != "norman":
        raise ValueError("norman_gi_truth_builder only supports dataset=norman")
    if not (0.0 < float(low_quantile) < 1.0 and 0.0 < float(high_quantile) < 1.0):
        raise ValueError("low_quantile and high_quantile must be between 0 and 1")
    if float(low_quantile) >= float(high_quantile):
        raise ValueError("low_quantile must be smaller than high_quantile")

    out_dir = Path(out_root).resolve() if out_root else (REPO_ROOT / "artifacts" / "analysis" / "norman_gi_truth").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    h5ad_path = _resolve_norman_truth_h5ad_path()
    raw_truth_df = _collect_truth_rows_from_raw_h5ad(h5ad_path)
    truth_df, summary_df, thresholds = _apply_truth_labels(
        raw_truth_df,
        low_quantile=float(low_quantile),
        high_quantile=float(high_quantile),
    ) if not raw_truth_df.empty else (pd.DataFrame(), pd.DataFrame(), {})
    skipped_df = pd.DataFrame(columns=["condition", "skip_reason"])

    raw_truth_df.to_csv(out_dir / "norman_gi_truth_sources.csv", index=False, encoding="utf-8-sig")
    truth_df.to_csv(out_dir / "norman_gi_truth_labels.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(out_dir / "norman_gi_truth_summary.csv", index=False, encoding="utf-8-sig")
    skipped_df.to_csv(out_dir / "norman_gi_truth_skipped.csv", index=False, encoding="utf-8-sig")

    write_run_meta(
        out_dir / "run_meta.json",
        {
            "dataset": "norman",
            "source_mode": "raw_truth_h5ad",
            "h5ad_path": str(h5ad_path),
            "low_quantile": float(low_quantile),
            "high_quantile": float(high_quantile),
            "thresholds": thresholds,
            "condition_count": int(len(truth_df)),
            "source_row_count": int(len(raw_truth_df)),
            "label_columns": [f"is_{gi_type}" for gi_type in GI_TYPE_ORDER] + ["is_any_gi"],
            "out_dir": str(out_dir),
            "ignored_models_arg": models,
            "ignored_split_ids_arg": split_ids,
        },
    )
    return {
        "out_dir": out_dir,
        "per_source_df": raw_truth_df,
        "truth_df": truth_df,
        "summary_df": summary_df,
        "skipped_df": skipped_df,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build Norman GI truth labels from raw Norman truth h5ad.")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--models", default="")
    ap.add_argument("--split_ids", default="")
    ap.add_argument("--out_root", default="")
    ap.add_argument("--low_quantile", type=float, default=0.2)
    ap.add_argument("--high_quantile", type=float, default=0.8)
    args = ap.parse_args(argv)

    result = run_norman_gi_truth_builder(
        dataset=str(args.dataset).strip(),
        models=str(args.models).strip() or None,
        split_ids=str(args.split_ids).strip() or "1,2,3,4,5",
        out_root=str(args.out_root).strip() or None,
        low_quantile=float(args.low_quantile),
        high_quantile=float(args.high_quantile),
    )
    print(f"out_dir: {result['out_dir']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
