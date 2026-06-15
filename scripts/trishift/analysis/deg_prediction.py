from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from scripts.common.time_utils import ts_local


DEFAULT_RESULT_ROOTS = {
    "trishift": REPO_ROOT / "artifacts" / "results",
    "trishift_nearest": REPO_ROOT / "artifacts" / "results",
    "trishift_random": REPO_ROOT / "artifacts" / "results",
    "gears": REPO_ROOT / "artifacts" / "results" / "gears",
    "genepert": REPO_ROOT / "artifacts" / "results" / "genepert",
    "scgpt": REPO_ROOT / "artifacts" / "results" / "scgpt",
    "biolord": REPO_ROOT / "artifacts" / "results" / "biolord",
    "cellot": REPO_ROOT / "artifacts" / "results" / "cellot",
}


def _parse_split_ids(value: str) -> list[int]:
    out = [int(x.strip()) for x in str(value).split(",") if x.strip()]
    if not out:
        raise ValueError("split_ids cannot be empty")
    return out


def _parse_int_list(value: str) -> list[int]:
    out = [int(x.strip()) for x in str(value).split(",") if x.strip()]
    if not out:
        raise ValueError("at least one integer is required")
    return out


def _parse_condition_tokens(condition: str) -> set[str]:
    return {
        token.strip()
        for token in str(condition).split("+")
        if token.strip() and token.strip().lower() != "ctrl"
    }


def _payload_path(
    *,
    model_name: str,
    dataset: str,
    split_id: int,
    result_root: Path,
    variant_tag: str | None,
    result_mode: str,
) -> Path:
    key = str(model_name).strip().lower()
    mode = str(result_mode or "default").strip().lower()
    if key == "trishift":
        if mode in {"unseen", "unseen_ctrl", "unseen-control"}:
            return result_root / f"trishift_{dataset}_{int(split_id)}_unseen_ctrl.pkl"
        suffix = f"_{variant_tag}" if variant_tag else ""
        return result_root / f"trishift_{dataset}_{int(split_id)}{suffix}.pkl"
    if key in {"trishift_nearest", "trishift_random"}:
        if mode in {"unseen", "unseen_ctrl", "unseen-control"}:
            return result_root / f"trishift_{dataset}_{int(split_id)}_unseen_ctrl.pkl"
        suffix = "_nearest" if key == "trishift_nearest" else "_random"
        return result_root / f"trishift_{dataset}_{int(split_id)}{suffix}.pkl"
    if mode in {"unseen", "unseen_ctrl", "unseen-control"}:
        return result_root / f"{key}_{dataset}_{int(split_id)}_unseen_ctrl.pkl"
    return result_root / f"{key}_{dataset}_{int(split_id)}.pkl"


def _load_payload(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected dict payload at {path}, got {type(payload)}")
    return payload


def _condition_arrays(item: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if all(k in item for k in ("Pred_full", "Ctrl_full", "Truth_full", "gene_name_full")):
        pred = np.asarray(item["Pred_full"], dtype=np.float64)
        ctrl = np.asarray(item["Ctrl_full"], dtype=np.float64)
        truth = np.asarray(item["Truth_full"], dtype=np.float64)
        genes = np.asarray(item["gene_name_full"]).astype(str)
    else:
        pred = np.asarray(item["Pred"], dtype=np.float64)
        ctrl = np.asarray(item["Ctrl"], dtype=np.float64)
        truth = np.asarray(item["Truth"], dtype=np.float64)
        if "DE_name" in item and len(item["DE_name"]) == pred.shape[1]:
            genes = np.asarray(item["DE_name"]).astype(str)
        else:
            genes = np.asarray([f"gene_{i}" for i in range(pred.shape[1])], dtype=str)
    if pred.ndim != 2 or ctrl.ndim != 2 or truth.ndim != 2:
        raise ValueError("Pred/Ctrl/Truth arrays must be 2D")
    if pred.shape[1] != ctrl.shape[1] or pred.shape[1] != truth.shape[1]:
        raise ValueError("Pred/Ctrl/Truth arrays must share gene dimension")
    if genes.shape[0] != pred.shape[1]:
        raise ValueError("gene list length must match gene dimension")
    return pred, ctrl, truth, genes


def _rank_desc(values: np.ndarray, candidate_mask: np.ndarray) -> np.ndarray:
    scores = np.asarray(values, dtype=np.float64).reshape(-1)
    mask = np.asarray(candidate_mask, dtype=bool).reshape(-1)
    idx = np.where(mask & np.isfinite(scores))[0]
    if idx.size == 0:
        return idx
    order = np.argsort(-scores[idx], kind="mergesort")
    return idx[order]


def _binary_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=int).reshape(-1)
    s = np.asarray(score, dtype=np.float64).reshape(-1)
    mask = np.isfinite(s)
    y = y[mask]
    s = s[mask]
    n_pos = int(y.sum())
    n_neg = int(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    sorted_s = s[order]
    start = 0
    while start < len(sorted_s):
        end = start + 1
        while end < len(sorted_s) and sorted_s[end] == sorted_s[start]:
            end += 1
        avg_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = avg_rank
        start = end
    pos_rank_sum = float(ranks[y == 1].sum())
    return float((pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _average_precision(y_true: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=int).reshape(-1)
    s = np.asarray(score, dtype=np.float64).reshape(-1)
    mask = np.isfinite(s)
    y = y[mask]
    s = s[mask]
    n_pos = int(y.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    ranks = np.arange(1, y_sorted.size + 1, dtype=np.float64)
    precision = tp / ranks
    return float((precision * y_sorted).sum() / n_pos)


def _direction_agreement(
    truth_delta: np.ndarray,
    pred_delta: np.ndarray,
    idx: np.ndarray,
) -> float:
    idx = np.asarray(idx, dtype=int).reshape(-1)
    if idx.size == 0:
        return float("nan")
    truth_sign = np.sign(np.asarray(truth_delta, dtype=np.float64).reshape(-1)[idx])
    pred_sign = np.sign(np.asarray(pred_delta, dtype=np.float64).reshape(-1)[idx])
    informative = truth_sign != 0
    if not np.any(informative):
        return float("nan")
    return float(np.mean(truth_sign[informative] == pred_sign[informative]))


def _condition_metrics(
    *,
    model_name: str,
    dataset: str,
    split_id: int,
    condition: str,
    item: dict[str, Any],
    label_top_k: int,
    overlap_ks: list[int],
    remove_perturbed_genes: bool,
    pkl_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    pred, ctrl, truth, genes = _condition_arrays(item)
    pred_delta = pred.mean(axis=0) - ctrl.mean(axis=0)
    truth_delta = truth.mean(axis=0) - ctrl.mean(axis=0)
    pred_score = np.abs(pred_delta)
    truth_score = np.abs(truth_delta)
    candidate_mask = np.isfinite(pred_score) & np.isfinite(truth_score)
    if remove_perturbed_genes:
        perturbed = _parse_condition_tokens(condition)
        candidate_mask &= np.asarray([str(g) not in perturbed for g in genes], dtype=bool)

    truth_rank = _rank_desc(truth_score, candidate_mask)
    pred_rank = _rank_desc(pred_score, candidate_mask)
    truth_pos = truth_rank[: min(int(label_top_k), truth_rank.size)]
    y_true = np.zeros(genes.shape[0], dtype=int)
    y_true[truth_pos] = 1
    eval_idx = np.where(candidate_mask)[0]

    row: dict[str, Any] = {
        "model_name": str(model_name),
        "dataset": str(dataset),
        "setting": "deg_prediction",
        "split": int(split_id),
        "split_id": int(split_id),
        "condition": str(condition),
        "pkl_path": str(pkl_path),
        "label_top_k": int(label_top_k),
        "n_candidate_genes": int(eval_idx.size),
        "n_positive_genes": int(y_true[eval_idx].sum()),
        "deg_auroc": _binary_auc(y_true[eval_idx], pred_score[eval_idx]),
        "deg_auprc": _average_precision(y_true[eval_idx], pred_score[eval_idx]),
        "direction_agreement_label_top_k": _direction_agreement(truth_delta, pred_delta, truth_pos),
    }
    gene_rows: list[dict[str, Any]] = []
    truth_set = set(int(i) for i in truth_pos.tolist())
    for k in overlap_ks:
        pred_top = pred_rank[: min(int(k), pred_rank.size)]
        truth_top = truth_rank[: min(int(k), truth_rank.size)]
        pred_set = set(int(i) for i in pred_top.tolist())
        truth_k_set = set(int(i) for i in truth_top.tolist())
        common = np.asarray(sorted(pred_set & truth_k_set), dtype=int)
        row[f"overlap_at_{int(k)}"] = int(common.size)
        row[f"precision_at_{int(k)}"] = float(common.size / max(len(pred_set), 1))
        row[f"recall_at_{int(k)}"] = float(common.size / max(len(truth_k_set), 1))
        row[f"jaccard_at_{int(k)}"] = float(common.size / len(pred_set | truth_k_set)) if (pred_set | truth_k_set) else float("nan")
        row[f"direction_agreement_overlap_at_{int(k)}"] = _direction_agreement(truth_delta, pred_delta, common)
    for rank, idx in enumerate(pred_rank[: max(overlap_ks + [label_top_k])], start=1):
        gene_rows.append(
            {
                "model_name": str(model_name),
                "dataset": str(dataset),
                "split": int(split_id),
                "condition": str(condition),
                "rank": int(rank),
                "gene": str(genes[int(idx)]),
                "pred_score": float(pred_score[int(idx)]),
                "truth_score": float(truth_score[int(idx)]),
                "is_truth_deg_label": int(int(idx) in truth_set),
                "truth_direction": float(np.sign(truth_delta[int(idx)])),
                "pred_direction": float(np.sign(pred_delta[int(idx)])),
            }
        )
    return row, gene_rows


def run_deg_prediction(
    *,
    dataset: str,
    model_name: str,
    split_ids: list[int],
    result_dir: str | Path | None = None,
    out_root: str | Path | None = None,
    result_mode: str = "default",
    variant_tag: str | None = None,
    payload_path: str | Path | None = None,
    label_top_k: int = 100,
    overlap_ks: list[int] | None = None,
    remove_perturbed_genes: bool = True,
) -> dict[str, Any]:
    model_key = str(model_name).strip().lower()
    if model_key not in DEFAULT_RESULT_ROOTS:
        raise ValueError(f"Unsupported model_name={model_name}")
    dataset_key = str(dataset).strip()
    overlap_ks = overlap_ks or [20, 50]
    result_root = (
        Path(result_dir).resolve()
        if result_dir
        else (DEFAULT_RESULT_ROOTS[model_key] / dataset_key).resolve()
    )
    out_dir = (
        Path(out_root).resolve()
        if out_root
        else (REPO_ROOT / "artifacts" / "analysis" / "deg_prediction" / f"{ts_local()}_{dataset_key}_{model_key}").resolve()
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    gene_rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for split_id in split_ids:
        pkl_path = (
            Path(payload_path).resolve()
            if payload_path and len(split_ids) == 1
            else _payload_path(
                model_name=model_key,
                dataset=dataset_key,
                split_id=int(split_id),
                result_root=result_root,
                variant_tag=variant_tag,
                result_mode=result_mode,
            )
        )
        if not pkl_path.exists():
            missing.append({"split": int(split_id), "path": str(pkl_path), "reason": "missing_payload"})
            continue
        payload = _load_payload(pkl_path)
        for condition, item in payload.items():
            if not isinstance(item, dict):
                continue
            row, gene_row = _condition_metrics(
                model_name=model_key,
                dataset=dataset_key,
                split_id=int(split_id),
                condition=str(condition),
                item=item,
                label_top_k=int(label_top_k),
                overlap_ks=overlap_ks,
                remove_perturbed_genes=bool(remove_perturbed_genes),
                pkl_path=pkl_path,
            )
            rows.append(row)
            gene_rows.extend(gene_row)

    per_condition = pd.DataFrame(rows)
    if per_condition.empty:
        summary = pd.DataFrame(columns=["dataset", "model_name", "metric", "value"])
    else:
        metric_cols = [
            c
            for c in per_condition.columns
            if c.startswith("deg_")
            or c.startswith("overlap_at_")
            or c.startswith("precision_at_")
            or c.startswith("recall_at_")
            or c.startswith("jaccard_at_")
            or c.startswith("direction_agreement")
        ]
        summary_rows = []
        for (dataset_val, model_val), df in per_condition.groupby(["dataset", "model_name"], sort=True):
            for col in metric_cols:
                vals = pd.to_numeric(df[col], errors="coerce")
                summary_rows.append(
                    {
                        "dataset": dataset_val,
                        "model_name": model_val,
                        "metric": col,
                        "value": float(vals.mean()),
                        "n_conditions": int(vals.notna().sum()),
                    }
                )
        summary = pd.DataFrame(summary_rows)

    long_rows = []
    if not per_condition.empty:
        metric_cols = [c for c in per_condition.columns if c not in {"model_name", "dataset", "setting", "split", "split_id", "condition", "pkl_path"} and pd.api.types.is_numeric_dtype(per_condition[c])]
        for _, row in per_condition.iterrows():
            for col in metric_cols:
                long_rows.append(
                    {
                        "dataset": row["dataset"],
                        "setting": row["setting"],
                        "split": int(row["split"]),
                        "condition": row["condition"],
                        "model": row["model_name"],
                        "metric": col,
                        "value": float(row[col]) if pd.notna(row[col]) else float("nan"),
                    }
                )
    long_df = pd.DataFrame(long_rows)
    gene_scores = pd.DataFrame(gene_rows)

    per_condition.to_csv(out_dir / "deg_prediction_per_condition.csv", index=False)
    long_df.to_csv(out_dir / "deg_prediction_long.csv", index=False)
    summary.to_csv(out_dir / "deg_prediction_summary.csv", index=False)
    gene_scores.to_csv(out_dir / "deg_prediction_gene_scores.csv", index=False)
    provenance = {
        "dataset": dataset_key,
        "model_name": model_key,
        "split_ids": [int(x) for x in split_ids],
        "result_root": str(result_root),
        "result_mode": str(result_mode),
        "variant_tag": str(variant_tag or ""),
        "payload_path": str(payload_path or ""),
        "label_top_k": int(label_top_k),
        "overlap_ks": [int(x) for x in overlap_ks],
        "remove_perturbed_genes": bool(remove_perturbed_genes),
        "score_definition": "abs(mean(predicted perturbed expression) - mean(control expression))",
        "label_definition": "top-k genes by abs(mean(observed perturbed expression) - mean(control expression)); evaluation-only labels",
        "claim_boundary": "DEG identity metrics are evaluation-only and not causal mechanism discovery; Systema-style diagnostics reduce but do not eliminate confounding concerns.",
        "missing_payloads": missing,
    }
    (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return {
        "out_dir": out_dir,
        "per_condition": per_condition,
        "long": long_df,
        "summary": summary,
        "gene_scores": gene_scores,
        "missing_payloads": missing,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate post-perturbation DEG identity prediction from exported payloads.")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model-name", required=True, choices=sorted(DEFAULT_RESULT_ROOTS))
    ap.add_argument("--split-ids", default="1")
    ap.add_argument("--result-dir", default="")
    ap.add_argument("--out-root", default="")
    ap.add_argument("--result-mode", default="unseen_ctrl")
    ap.add_argument("--variant-tag", default="")
    ap.add_argument("--payload", default="", help="direct payload path; only valid with a single split id")
    ap.add_argument("--label-top-k", type=int, default=100)
    ap.add_argument("--overlap-ks", default="20,50")
    ap.add_argument("--keep-perturbed-genes", action="store_true")
    args = ap.parse_args(argv)
    result = run_deg_prediction(
        dataset=str(args.dataset).strip(),
        model_name=str(args.model_name).strip(),
        split_ids=_parse_split_ids(str(args.split_ids)),
        result_dir=str(args.result_dir).strip() or None,
        out_root=str(args.out_root).strip() or None,
        result_mode=str(args.result_mode).strip() or "unseen_ctrl",
        variant_tag=str(args.variant_tag).strip() or None,
        payload_path=str(args.payload).strip() or None,
        label_top_k=int(args.label_top_k),
        overlap_ks=_parse_int_list(str(args.overlap_ks)),
        remove_perturbed_genes=not bool(args.keep_perturbed_genes),
    )
    print(f"out_dir: {result['out_dir']}")
    print(result["summary"].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
