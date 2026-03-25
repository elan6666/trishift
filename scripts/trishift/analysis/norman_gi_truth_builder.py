from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts.trishift.analysis._result_adapter import (
    load_payload_item,
    normalize_condition,
    parse_models,
    parse_split_ids,
    warn_skip,
    write_run_meta,
)
from scripts.trishift.analysis.norman_nonadd_experiment import (
    DEFAULT_MODELS,
    GI_METRICS_ALL,
    _aligned_triplet_deltas,
    _combo_single_keys,
    _compute_gi_params,
)

GI_TYPE_ORDER = ["synergy", "suppressor", "neomorphic", "redundancy", "epistasis"]
GI_TYPE_LABELS = {
    "synergy": "Synergy",
    "suppressor": "Suppression",
    "neomorphic": "Neomorphism",
    "redundancy": "Redundancy",
    "epistasis": "Epistasis",
}
GI_TYPE_RULES = {
    "synergy": {"metric_name": "mag", "direction": "high"},
    "suppressor": {"metric_name": "mag", "direction": "low"},
    "neomorphic": {"metric_name": "corr_fit", "direction": "low"},
    "redundancy": {"metric_name": "dcor", "direction": "high"},
    "epistasis": {"metric_name": "eq_contr", "direction": "low"},
}


def _compute_truth_condition_gi(*, item_ab: dict[str, Any], item_a: dict[str, Any], item_b: dict[str, Any]) -> dict[str, float]:
    truth_a, truth_b, truth_ab = _aligned_triplet_deltas(item_ab=item_ab, item_a=item_a, item_b=item_b, expr_key="Truth_full")
    truth_metrics = _compute_gi_params(truth_a, truth_b, truth_ab)
    return {f"truth_{metric}": float(truth_metrics.get(metric, float("nan"))) for metric in GI_METRICS_ALL}


def _collect_truth_rows(
    *,
    model_name: str,
    split_id: int,
    payload: dict[str, Any],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    normalized_payload = {normalize_condition(str(k)): v for k, v in payload.items() if isinstance(v, dict)}
    rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []
    for condition, item_ab in sorted(normalized_payload.items()):
        tokens = [x for x in condition.split("+") if x and x != "ctrl"]
        if len(tokens) != 2:
            continue
        try:
            single_a, single_b = _combo_single_keys(condition)
        except Exception as exc:
            skipped_rows.append(
                {
                    "source_model": str(model_name),
                    "split_id": int(split_id),
                    "condition": str(condition),
                    "skip_reason": str(exc),
                }
            )
            continue
        if single_a not in normalized_payload or single_b not in normalized_payload:
            missing = single_a if single_a not in normalized_payload else single_b
            skipped_rows.append(
                {
                    "source_model": str(model_name),
                    "split_id": int(split_id),
                    "condition": str(condition),
                    "skip_reason": f"missing single perturbation: {missing}",
                }
            )
            continue
        try:
            truth_row = _compute_truth_condition_gi(item_ab=item_ab, item_a=normalized_payload[single_a], item_b=normalized_payload[single_b])
        except Exception as exc:
            skipped_rows.append(
                {
                    "source_model": str(model_name),
                    "split_id": int(split_id),
                    "condition": str(condition),
                    "skip_reason": str(exc),
                }
            )
            continue
        rows.append(
            {
                "dataset": "norman",
                "condition": str(condition),
                "source_model": str(model_name),
                "split_id": int(split_id),
                **truth_row,
            }
        )
    return pd.DataFrame(rows), skipped_rows


def _aggregate_truth_rows(per_source_df: pd.DataFrame) -> pd.DataFrame:
    if per_source_df.empty:
        return pd.DataFrame()
    truth_cols = [col for col in per_source_df.columns if col.startswith("truth_")]
    truth_df = per_source_df.groupby("condition", as_index=False)[truth_cols].mean(numeric_only=True)
    source_summary = (
        per_source_df.groupby("condition", as_index=False)
        .agg(
            source_count=("condition", "size"),
            source_models=("source_model", lambda x: ",".join(sorted({str(v) for v in x}))),
            source_splits=("split_id", lambda x: ",".join(str(int(v)) for v in sorted({int(v) for v in x}))),
        )
    )
    return truth_df.merge(source_summary, on="condition", how="left")


def _safe_quantile(values: pd.Series, q: float) -> float:
    work = pd.to_numeric(values, errors="coerce").dropna()
    if work.empty:
        return float("nan")
    return float(work.quantile(float(q)))


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
        metric_name = str(spec["metric_name"])
        metric_col = f"truth_{metric_name}"
        direction = str(spec["direction"])
        quantile = float(high_quantile if direction == "high" else low_quantile)
        threshold = _safe_quantile(work[metric_col], quantile)
        if np.isnan(threshold):
            labels = pd.Series(False, index=work.index, dtype=bool)
        elif direction == "high":
            labels = pd.to_numeric(work[metric_col], errors="coerce") >= float(threshold)
        else:
            labels = pd.to_numeric(work[metric_col], errors="coerce") <= float(threshold)
        label_col = f"is_{gi_type}"
        work[label_col] = labels.fillna(False).astype(bool)
        positive_n = int(work[label_col].sum())
        thresholds[gi_type] = {
            "metric_name": metric_name,
            "direction": direction,
            "quantile": quantile,
            "threshold": threshold,
            "positive_n": positive_n,
        }
        summary_rows.append(
            {
                "gi_type": gi_type,
                "gi_label": GI_TYPE_LABELS[gi_type],
                "metric_name": metric_name,
                "direction": direction,
                "quantile": quantile,
                "threshold": threshold,
                "positive_n": positive_n,
                "positive_fraction": float(positive_n / len(work)) if len(work) else float("nan"),
            }
        )
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

    model_requests = parse_models(models or DEFAULT_MODELS)
    split_list = parse_split_ids(split_ids)
    out_dir = Path(out_root).resolve() if out_root else (REPO_ROOT / "artifacts" / "analysis" / "norman_gi_truth").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    used_models: list[str] = []
    per_source_rows: list[pd.DataFrame] = []
    skipped_rows: list[dict[str, Any]] = []
    for model_name in model_requests:
        model_used = False
        for split_id in split_list:
            try:
                _, payload = load_payload_item(dataset="norman", model_name=model_name, split_id=int(split_id), condition=None)
            except Exception as exc:
                skipped_rows.append(
                    {
                        "source_model": str(model_name),
                        "split_id": int(split_id),
                        "condition": "",
                        "skip_reason": str(exc),
                    }
                )
                continue
            rows_df, skip_rows = _collect_truth_rows(model_name=str(model_name), split_id=int(split_id), payload=payload)
            if not rows_df.empty:
                model_used = True
                per_source_rows.append(rows_df)
            skipped_rows.extend(skip_rows)
        if model_used:
            used_models.append(str(model_name))
        else:
            warn_skip(f"[norman_gi_truth_builder] no usable payload rows for model={model_name}")

    per_source_df = pd.concat(per_source_rows, ignore_index=True) if per_source_rows else pd.DataFrame()
    truth_df = _aggregate_truth_rows(per_source_df)
    truth_df, summary_df, thresholds = _apply_truth_labels(
        truth_df,
        low_quantile=float(low_quantile),
        high_quantile=float(high_quantile),
    ) if not truth_df.empty else (pd.DataFrame(), pd.DataFrame(), {})
    skipped_df = pd.DataFrame(skipped_rows)

    per_source_df.to_csv(out_dir / "norman_gi_truth_sources.csv", index=False, encoding="utf-8-sig")
    truth_df.to_csv(out_dir / "norman_gi_truth_labels.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(out_dir / "norman_gi_truth_summary.csv", index=False, encoding="utf-8-sig")
    skipped_df.to_csv(out_dir / "norman_gi_truth_skipped.csv", index=False, encoding="utf-8-sig")

    write_run_meta(
        out_dir / "run_meta.json",
        {
            "dataset": "norman",
            "models_requested": model_requests,
            "models_used": used_models,
            "split_ids": split_list,
            "low_quantile": float(low_quantile),
            "high_quantile": float(high_quantile),
            "thresholds": thresholds,
            "condition_count": int(len(truth_df)),
            "source_row_count": int(len(per_source_df)),
            "label_columns": [f"is_{gi_type}" for gi_type in GI_TYPE_ORDER],
            "out_dir": str(out_dir),
            "skipped_rows": skipped_rows,
        },
    )
    return {
        "out_dir": out_dir,
        "per_source_df": per_source_df,
        "truth_df": truth_df,
        "summary_df": summary_df,
        "skipped_df": skipped_df,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build Norman GI truth labels from truth_full payload data.")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--models", default="trishift_nearest,scouter,gears,genepert,scgpt")
    ap.add_argument("--split_ids", default="1,2,3,4,5")
    ap.add_argument("--out_root", default="")
    ap.add_argument("--low_quantile", type=float, default=0.2)
    ap.add_argument("--high_quantile", type=float, default=0.8)
    args = ap.parse_args(argv)

    result = run_norman_gi_truth_builder(
        dataset=str(args.dataset).strip(),
        models=str(args.models).strip(),
        split_ids=str(args.split_ids).strip(),
        out_root=str(args.out_root).strip() or None,
        low_quantile=float(args.low_quantile),
        high_quantile=float(args.high_quantile),
    )
    print(f"out_dir: {result['out_dir']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
