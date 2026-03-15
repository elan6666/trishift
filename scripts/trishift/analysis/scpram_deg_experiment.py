from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from trishift._external_metrics import compute_scpram_metrics_from_arrays


DEFAULT_RESULT_ROOTS = {
    "trishift": REPO_ROOT / "artifacts" / "results",
    "gears": REPO_ROOT / "artifacts" / "results" / "gears",
    "genepert": REPO_ROOT / "artifacts" / "results" / "genepert",
    "scouter": REPO_ROOT / "artifacts" / "results" / "scouter",
}

REQUIRED_FULL_GENE_FIELDS = ("Pred_full", "Ctrl_full", "Truth_full", "gene_name_full")


@dataclass
class ScPRAMDEGExperimentResult:
    out_dir: Path
    per_condition_df: pd.DataFrame
    split_summary_df: pd.DataFrame
    dataset_summary_df: pd.DataFrame
    gene_lists_df: pd.DataFrame
    representative_df: pd.DataFrame


def _ts_local() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _make_default_out_dir(dataset: str) -> Path:
    base = (REPO_ROOT / "artifacts" / "results" / dataset / f"scpram_deg_downstream_{_ts_local()}").resolve()
    if not base.exists():
        return base
    idx = 2
    while True:
        candidate = base.with_name(f"{base.name}_{idx}")
        if not candidate.exists():
            return candidate
        idx += 1


def _normalize_model_name(model_name: str) -> str:
    key = str(model_name).strip().lower()
    if key not in DEFAULT_RESULT_ROOTS:
        raise ValueError(f"Unsupported model_name={model_name}")
    return key


def _split_ids_from_value(split_ids: int | str | list[int] | tuple[int, ...]) -> list[int]:
    if isinstance(split_ids, int):
        return [int(split_ids)]
    if isinstance(split_ids, (list, tuple)):
        out = [int(x) for x in split_ids]
        if not out:
            raise ValueError("split_ids cannot be empty")
        return out
    text = str(split_ids).strip()
    if not text:
        raise ValueError("split_ids cannot be empty")
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _result_root(model_name: str, dataset: str, result_dir: str | Path | None) -> Path:
    if result_dir is not None and str(result_dir).strip():
        return Path(result_dir).resolve()
    return (DEFAULT_RESULT_ROOTS[_normalize_model_name(model_name)] / dataset).resolve()


def _pkl_path(
    *,
    model_name: str,
    dataset: str,
    split_id: int,
    result_root: Path,
    variant_tag: str | None = None,
) -> Path:
    model_key = _normalize_model_name(model_name)
    if model_key == "trishift":
        suffix = f"_{variant_tag}" if variant_tag else ""
        return result_root / f"trishift_{dataset}_{split_id}{suffix}.pkl"
    return result_root / f"{model_key}_{dataset}_{split_id}.pkl"


def _parse_condition_tokens(condition: str) -> list[str]:
    tokens: list[str] = []
    for token in str(condition).split("+"):
        tok = str(token).strip()
        if not tok or tok.lower() == "ctrl":
            continue
        tokens.append(tok)
    return tokens


def _remove_perturbed_genes(genes: list[str], condition: str) -> list[str]:
    perturbed = set(_parse_condition_tokens(condition))
    return [g for g in genes if str(g) not in perturbed]


def _full_payload_item_arrays_strict(
    obj: dict[str, Any],
    *,
    pkl_path: Path,
    condition: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    missing = [key for key in REQUIRED_FULL_GENE_FIELDS if key not in obj]
    if missing:
        raise ValueError(
            "scPRAM-style DEG evaluation requires full-gene payload fields "
            f"{list(REQUIRED_FULL_GENE_FIELDS)}; missing {missing} for condition "
            f"{condition!r} in {pkl_path}. Please re-export a full-gene pkl and rerun."
        )
    pred = np.asarray(obj["Pred_full"], dtype=np.float32)
    ctrl = np.asarray(obj["Ctrl_full"], dtype=np.float32)
    truth = np.asarray(obj["Truth_full"], dtype=np.float32)
    gene_names = np.asarray(obj["gene_name_full"]).astype(str)
    if pred.ndim != 2 or ctrl.ndim != 2 or truth.ndim != 2:
        raise ValueError(f"Expected 2D arrays in {pkl_path} for condition {condition!r}")
    if pred.shape[1] != ctrl.shape[1] or pred.shape[1] != truth.shape[1]:
        raise ValueError(
            f"Pred/Ctrl/Truth gene dimensions do not match for condition {condition!r} in {pkl_path}"
        )
    if gene_names.shape[0] != pred.shape[1]:
        raise ValueError(
            f"gene_name_full length does not match gene dimension for condition {condition!r} in {pkl_path}"
        )
    return pred, ctrl, truth, gene_names


def _build_eval_adata(
    *,
    pred: np.ndarray,
    ctrl: np.ndarray,
    truth: np.ndarray,
    gene_names: np.ndarray,
) -> ad.AnnData:
    x = np.vstack([ctrl, truth, pred]).astype(np.float32, copy=False)
    obs = pd.DataFrame(
        {
            "condition": (["control"] * int(ctrl.shape[0]))
            + (["stimulated"] * int(truth.shape[0]))
            + (["predicted"] * int(pred.shape[0]))
        }
    )
    var = pd.DataFrame(index=pd.Index(gene_names.astype(str), name="gene_name"))
    return ad.AnnData(X=x, obs=obs, var=var)


def _rank_eval_deg_lists(
    *,
    pred: np.ndarray,
    ctrl: np.ndarray,
    truth: np.ndarray,
    gene_names: np.ndarray,
    top_k: int,
    condition: str,
    remove_perturbed_genes: bool,
) -> tuple[list[str], list[str], np.ndarray]:
    eval_adata = _build_eval_adata(pred=pred, ctrl=ctrl, truth=truth, gene_names=gene_names)
    sc.tl.rank_genes_groups(
        eval_adata,
        groupby="condition",
        reference="control",
        method="wilcoxon",
        n_genes=eval_adata.n_vars,
    )
    names = eval_adata.uns["rank_genes_groups"]["names"]
    truth_genes = [str(g) for g in list(names["stimulated"])]
    pred_genes = [str(g) for g in list(names["predicted"])]
    if remove_perturbed_genes:
        truth_genes = _remove_perturbed_genes(truth_genes, condition)
        pred_genes = _remove_perturbed_genes(pred_genes, condition)
    truth_genes = truth_genes[: int(top_k)]
    pred_genes = pred_genes[: int(top_k)]
    name_to_idx = {str(g): i for i, g in enumerate(gene_names.astype(str).tolist())}
    truth_idx = np.asarray([name_to_idx[g] for g in truth_genes if g in name_to_idx], dtype=int)
    return truth_genes, pred_genes, truth_idx


def _overlap_metrics(truth_genes: list[str], pred_genes: list[str], top_k: int) -> dict[str, float | int]:
    truth = list(dict.fromkeys([str(g) for g in truth_genes]))
    pred = list(dict.fromkeys([str(g) for g in pred_genes]))
    truth_set = set(truth)
    pred_set = set(pred)
    common = truth_set & pred_set
    union = truth_set | pred_set
    return {
        "deg_top_k": int(top_k),
        "common_degs_at_k": int(len(common)),
        "jaccard_at_k": float(len(common) / len(union)) if union else float("nan"),
        "precision_at_k": float(len(common) / len(pred_set)) if pred_set else float("nan"),
        "recall_at_k": float(len(common) / len(truth_set)) if truth_set else float("nan"),
        "truth_deg_count": int(len(truth)),
        "pred_deg_count": int(len(pred)),
    }


def _condition_rows_from_payload(
    *,
    model_name: str,
    dataset: str,
    split_id: int,
    pkl_path: Path,
    payload: dict[str, Any],
    top_k: int,
    remove_perturbed_genes: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    gene_rows: list[dict[str, Any]] = []
    for condition, obj in payload.items():
        pred, ctrl, truth, gene_names = _full_payload_item_arrays_strict(
            obj,
            pkl_path=pkl_path,
            condition=str(condition),
        )
        truth_deg_topk, pred_deg_topk, truth_deg_idx = _rank_eval_deg_lists(
            pred=pred,
            ctrl=ctrl,
            truth=truth,
            gene_names=gene_names,
            top_k=top_k,
            condition=str(condition),
            remove_perturbed_genes=remove_perturbed_genes,
        )
        overlap = _overlap_metrics(truth_deg_topk, pred_deg_topk, top_k)
        common_deg_topk = [g for g in truth_deg_topk if g in set(pred_deg_topk)]
        scpram = compute_scpram_metrics_from_arrays(
            X_true=truth,
            X_pred=pred,
            deg_idx=truth_deg_idx,
            n_degs=int(top_k),
            sample_ratio=0.8,
            times=100,
        )
        rows.append(
            {
                "model_name": str(model_name),
                "dataset": str(dataset),
                "split_id": int(split_id),
                "condition": str(condition),
                "focus_key": f"{int(split_id)}:{str(condition)}",
                "pkl_path": str(pkl_path),
                **overlap,
                "deg_flow": "scpram_eval_wilcoxon",
                "scpram_r2_degs_mean_mean": float(scpram["scpram_r2_degs_mean_mean"]),
                "scpram_r2_degs_var_mean": float(scpram["scpram_r2_degs_var_mean"]),
                "scpram_wasserstein_degs_sum": float(scpram["scpram_wasserstein_degs_sum"]),
            }
        )
        for list_type, genes in (
            ("truth_deg_topk", truth_deg_topk),
            ("pred_deg_topk", pred_deg_topk),
            ("common_deg_topk", common_deg_topk),
        ):
            for rank, gene in enumerate(genes, start=1):
                gene_rows.append(
                    {
                        "model_name": str(model_name),
                        "dataset": str(dataset),
                        "split_id": int(split_id),
                        "condition": str(condition),
                        "focus_key": f"{int(split_id)}:{str(condition)}",
                        "list_type": str(list_type),
                        "rank": int(rank),
                        "gene": str(gene),
                        "deg_top_k": int(top_k),
                    }
                )
    return rows, gene_rows


def _summarize_by_split(per_condition_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "common_degs_at_k",
        "jaccard_at_k",
        "precision_at_k",
        "recall_at_k",
        "truth_deg_count",
        "pred_deg_count",
        "scpram_r2_degs_mean_mean",
        "scpram_r2_degs_var_mean",
        "scpram_wasserstein_degs_sum",
    ]
    rows: list[dict[str, Any]] = []
    for split_id, split_df in per_condition_df.groupby("split_id", sort=True):
        row: dict[str, Any] = {
            "model_name": str(split_df["model_name"].iloc[0]),
            "dataset": str(split_df["dataset"].iloc[0]),
            "split_id": int(split_id),
            "deg_top_k": int(pd.to_numeric(split_df["deg_top_k"], errors="coerce").iloc[0]),
            "n_conditions": int(len(split_df)),
        }
        for col in metric_cols:
            vals = pd.to_numeric(split_df[col], errors="coerce")
            row[f"{col}_mean"] = float(vals.mean())
            row[f"{col}_median"] = float(vals.median())
        rows.append(row)
    return pd.DataFrame(rows)


def _summarize_dataset(split_summary_df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [c for c in split_summary_df.columns if c.endswith("_mean") or c.endswith("_median")]
    row: dict[str, Any] = {
        "model_name": str(split_summary_df["model_name"].iloc[0]),
        "dataset": str(split_summary_df["dataset"].iloc[0]),
        "deg_top_k": int(pd.to_numeric(split_summary_df["deg_top_k"], errors="coerce").iloc[0]),
        "n_splits": int(split_summary_df["split_id"].nunique()),
    }
    for col in numeric_cols:
        vals = pd.to_numeric(split_summary_df[col], errors="coerce")
        row[col] = float(vals.mean())
    return pd.DataFrame([row])


def select_representative_conditions(
    per_condition_df: pd.DataFrame,
    *,
    focus_conditions: list[str] | None = None,
) -> pd.DataFrame:
    if focus_conditions:
        focus_set = {str(x) for x in focus_conditions}
        picked = per_condition_df[per_condition_df["condition"].astype(str).isin(focus_set)].copy()
        return picked.sort_values(by=["split_id", "condition"]).reset_index(drop=True)
    ranked = per_condition_df.sort_values(
        by=["common_degs_at_k", "jaccard_at_k", "scpram_r2_degs_mean_mean"],
        ascending=[False, False, False],
        na_position="last",
    ).reset_index(drop=True)
    if ranked.empty:
        return ranked
    top_row = ranked.iloc[[0]]
    mid_row = ranked.iloc[[len(ranked) // 2]]
    worst_row = ranked.iloc[[-1]]
    return pd.concat([top_row, mid_row, worst_row], ignore_index=True).drop_duplicates(
        subset=["focus_key"]
    )


def load_condition_payload(
    *,
    model_name: str,
    dataset: str,
    split_id: int,
    condition: str,
    result_dir: str | Path | None = None,
    variant_tag: str | None = None,
) -> dict[str, Any]:
    result_root = _result_root(model_name, dataset, result_dir)
    pkl_path = _pkl_path(
        model_name=model_name,
        dataset=dataset,
        split_id=int(split_id),
        result_root=result_root,
        variant_tag=variant_tag,
    )
    with pkl_path.open("rb") as fh:
        payload = pickle.load(fh)
    if str(condition) not in payload:
        raise KeyError(f"Condition not found in payload: {condition}")
    return payload[str(condition)]


def summarize_condition_payload(
    *,
    payload_item: dict[str, Any],
    condition: str,
    pkl_path: str | Path,
    top_k: int = 100,
    remove_perturbed_genes: bool = False,
) -> dict[str, Any]:
    pred, ctrl, truth, gene_names = _full_payload_item_arrays_strict(
        payload_item,
        pkl_path=Path(pkl_path),
        condition=str(condition),
    )
    truth_deg_topk, pred_deg_topk, truth_deg_idx = _rank_eval_deg_lists(
        pred=pred,
        ctrl=ctrl,
        truth=truth,
        gene_names=gene_names,
        top_k=top_k,
        condition=str(condition),
        remove_perturbed_genes=remove_perturbed_genes,
    )
    common_deg_topk = [g for g in truth_deg_topk if g in set(pred_deg_topk)]
    return {
        "condition": str(condition),
        "pred": pred,
        "ctrl": ctrl,
        "truth": truth,
        "gene_name_full": gene_names,
        "truth_deg_topk": truth_deg_topk,
        "pred_deg_topk": pred_deg_topk,
        "common_deg_topk": common_deg_topk,
        "truth_deg_idx": truth_deg_idx,
        "deg_top_k": int(top_k),
    }


def build_mean_var_scatter(
    *,
    payload_item: dict[str, Any],
    condition: str,
    pkl_path: str | Path,
    title_prefix: str,
    top_k: int = 100,
    remove_perturbed_genes: bool = False,
) -> tuple[plt.Figure, plt.Figure]:
    summary = summarize_condition_payload(
        payload_item=payload_item,
        condition=condition,
        pkl_path=pkl_path,
        top_k=top_k,
        remove_perturbed_genes=remove_perturbed_genes,
    )
    pred = np.asarray(summary["pred"], dtype=np.float32)
    truth = np.asarray(summary["truth"], dtype=np.float32)
    gene_names = np.asarray(summary["gene_name_full"]).astype(str)
    degs = np.asarray(summary["truth_deg_idx"], dtype=int).reshape(-1)
    degs = degs[(degs >= 0) & (degs < pred.shape[1])]
    pred_mean = pred.mean(axis=0)[degs]
    truth_mean = truth.mean(axis=0)[degs]
    pred_var = pred.var(axis=0)[degs]
    truth_var = truth.var(axis=0)[degs]
    labels = gene_names[degs] if degs.size > 0 else np.array([], dtype=str)

    fig_mean, ax_mean = plt.subplots(figsize=(5, 4))
    ax_mean.scatter(truth_mean, pred_mean, s=18, alpha=0.8)
    ax_mean.set_xlabel("Truth mean")
    ax_mean.set_ylabel("Pred mean")
    ax_mean.set_title(f"{title_prefix}: top{int(top_k)} mean")
    for x, y, label in zip(truth_mean[:10], pred_mean[:10], labels[:10]):
        ax_mean.text(float(x), float(y), str(label), fontsize=8)

    fig_var, ax_var = plt.subplots(figsize=(5, 4))
    ax_var.scatter(truth_var, pred_var, s=18, alpha=0.8)
    ax_var.set_xlabel("Truth var")
    ax_var.set_ylabel("Pred var")
    ax_var.set_title(f"{title_prefix}: top{int(top_k)} variance")
    for x, y, label in zip(truth_var[:10], pred_var[:10], labels[:10]):
        ax_var.text(float(x), float(y), str(label), fontsize=8)
    return fig_mean, fig_var


def build_violin_plot(
    *,
    payload_item: dict[str, Any],
    condition: str,
    pkl_path: str | Path,
    gene: str,
) -> plt.Figure:
    pred, ctrl, truth, gene_names = _full_payload_item_arrays_strict(
        payload_item,
        pkl_path=Path(pkl_path),
        condition=str(condition),
    )
    name_to_idx = {str(g): i for i, g in enumerate(gene_names.astype(str).tolist())}
    if str(gene) not in name_to_idx:
        raise KeyError(f"Gene {gene!r} not found for condition {condition!r}")
    idx = int(name_to_idx[str(gene)])
    data = [ctrl[:, idx], truth[:, idx], pred[:, idx]]
    fig, ax = plt.subplots(figsize=(5, 4))
    parts = ax.violinplot(data, showmeans=True, showextrema=False)
    for body in parts["bodies"]:
        body.set_alpha(0.55)
    ax.set_xticks([1, 2, 3], ["control", "stimulated", "predicted"])
    ax.set_ylabel("Expression")
    ax.set_title(f"{condition}: {gene}")
    return fig


def run_scpram_deg_experiment(
    *,
    dataset: str,
    model_name: str,
    split_ids: int | str | list[int] | tuple[int, ...],
    result_dir: str | Path | None = None,
    out_root: str | Path | None = None,
    variant_tag: str | None = None,
    focus_conditions: list[str] | None = None,
    top_k: int = 100,
    remove_perturbed_genes: bool = False,
) -> ScPRAMDEGExperimentResult:
    split_list = _split_ids_from_value(split_ids)
    result_root = _result_root(model_name, dataset, result_dir)
    if out_root is not None and str(out_root).strip():
        out_dir = Path(out_root).resolve()
    else:
        out_dir = _make_default_out_dir(dataset)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_condition_rows: list[dict[str, Any]] = []
    gene_rows: list[dict[str, Any]] = []
    for split_id in split_list:
        pkl_path = _pkl_path(
            model_name=model_name,
            dataset=dataset,
            split_id=int(split_id),
            result_root=result_root,
            variant_tag=variant_tag,
        )
        if not pkl_path.exists():
            raise FileNotFoundError(f"Missing pkl for split {split_id}: {pkl_path}")
        with pkl_path.open("rb") as fh:
            payload = pickle.load(fh)
        if not isinstance(payload, dict):
            raise TypeError(f"Unexpected payload type at {pkl_path}: {type(payload)}")
        rows, list_rows = _condition_rows_from_payload(
            model_name=model_name,
            dataset=dataset,
            split_id=int(split_id),
            pkl_path=pkl_path,
            payload=payload,
            top_k=int(top_k),
            remove_perturbed_genes=bool(remove_perturbed_genes),
        )
        per_condition_rows.extend(rows)
        gene_rows.extend(list_rows)

    per_condition_df = pd.DataFrame(per_condition_rows).sort_values(
        by=["split_id", "condition"]
    ).reset_index(drop=True)
    gene_lists_df = pd.DataFrame(gene_rows).sort_values(
        by=["split_id", "condition", "list_type", "rank"]
    ).reset_index(drop=True)
    split_summary_df = _summarize_by_split(per_condition_df)
    dataset_summary_df = _summarize_dataset(split_summary_df)
    representative_df = select_representative_conditions(
        per_condition_df,
        focus_conditions=focus_conditions,
    )

    per_condition_df.to_csv(out_dir / "per_condition_metrics.csv", index=False)
    split_summary_df.to_csv(out_dir / "split_summary.csv", index=False)
    dataset_summary_df.to_csv(out_dir / "dataset_summary.csv", index=False)
    gene_lists_df.to_csv(out_dir / "deg_gene_lists_long.csv", index=False)
    representative_df.to_csv(out_dir / "representative_conditions.csv", index=False)
    meta = {
        "dataset": str(dataset),
        "model_name": str(model_name),
        "split_ids": [int(x) for x in split_list],
        "variant_tag": None if variant_tag is None else str(variant_tag),
        "top_k": int(top_k),
        "remove_perturbed_genes": bool(remove_perturbed_genes),
        "deg_flow": "scpram_eval_wilcoxon",
        "result_dir": str(result_root),
        "out_dir": str(out_dir),
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return ScPRAMDEGExperimentResult(
        out_dir=out_dir,
        per_condition_df=per_condition_df,
        split_summary_df=split_summary_df,
        dataset_summary_df=dataset_summary_df,
        gene_lists_df=gene_lists_df,
        representative_df=representative_df,
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run scPRAM-style eval-derived DEG analysis on exported pkls.")
    ap.add_argument("--dataset", default="adamson")
    ap.add_argument("--model_name", default="trishift")
    ap.add_argument("--split_ids", default="1,2,3,4,5")
    ap.add_argument("--result_dir", default="")
    ap.add_argument("--out_root", default="")
    ap.add_argument("--variant_tag", default="nearest")
    ap.add_argument("--focus_conditions", default="")
    ap.add_argument("--top_k", type=int, default=100)
    ap.add_argument("--remove_perturbed_genes", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    focus_conditions = [x.strip() for x in str(args.focus_conditions).split(",") if x.strip()]
    result = run_scpram_deg_experiment(
        dataset=args.dataset,
        model_name=args.model_name,
        split_ids=args.split_ids,
        result_dir=args.result_dir or None,
        out_root=args.out_root or None,
        variant_tag=(str(args.variant_tag).strip() or None),
        focus_conditions=focus_conditions or None,
        top_k=int(args.top_k),
        remove_perturbed_genes=bool(args.remove_perturbed_genes),
    )
    print(f"[done] out_dir={result.out_dir}")


if __name__ == "__main__":
    main()
