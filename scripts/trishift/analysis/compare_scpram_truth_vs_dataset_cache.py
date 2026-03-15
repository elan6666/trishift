from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts.trishift.analysis.compare_eval_deg_vs_dataset_cache import (
    DATASET_EMBEDDING_KEYS,
    load_yaml,
    overlap_metrics,
)
from scripts.trishift.analysis.scpram_deg_experiment import (
    _full_payload_item_arrays_strict,
    _pkl_path,
    _result_root,
    _rank_eval_deg_lists,
)
from trishift.TriShiftData import TriShiftData
from trishift._utils import apply_alias_mapping, load_adata, load_embedding_df


VALID_CACHE_KEYS = ("top20_degs_non_dropout", "top20_degs_non_zero", "top20_degs")


def parse_split_ids(value: str) -> list[int]:
    text = str(value).strip()
    if not text:
        raise ValueError("split_ids cannot be empty")
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def prepare_filtered_dataset(
    *,
    dataset: str,
    paths_path: str | Path,
    var_gene_key: str = "gene_name",
    label_key: str = "condition",
) -> tuple[Any, np.ndarray]:
    cfg = load_yaml(paths_path)
    h5ad_path = cfg["datasets"][dataset]
    emb_key = DATASET_EMBEDDING_KEYS[dataset]
    emb_path = cfg["embeddings"][emb_key]

    adata = load_adata(h5ad_path)
    embd_df = load_embedding_df(emb_path)
    embd_df = apply_alias_mapping(embd_df, dataset)
    data = TriShiftData(adata, embd_df, label_key=label_key, var_gene_key=var_gene_key)
    data.setup_embedding_index()
    if var_gene_key in data.adata_all.var.columns:
        gene_names = data.adata_all.var[var_gene_key].astype(str).values
    else:
        gene_names = data.adata_all.var_names.astype(str).values
    return data.adata_all, gene_names


def cache_idx_to_names(
    *,
    cache_obj: dict[str, Any],
    cache_key: str,
    condition: str,
    gene_names: np.ndarray,
) -> list[str]:
    cond_map = cache_obj.get(cache_key, {}) or {}
    idx = np.asarray(cond_map.get(condition, np.array([], dtype=int)), dtype=int)
    idx = idx[(idx >= 0) & (idx < gene_names.shape[0])]
    return gene_names[idx].astype(str).tolist()


def build_comparison_rows(
    *,
    dataset: str,
    model_name: str,
    split_ids: list[int],
    result_root: Path,
    variant_tag: str | None,
    cache_key: str,
    cache_obj: dict[str, Any],
    filtered_gene_names: np.ndarray,
    top_k: int,
    remove_perturbed_genes: bool,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split_id in split_ids:
        pkl_path = _pkl_path(
            model_name=model_name,
            dataset=dataset,
            split_id=split_id,
            result_root=result_root,
            variant_tag=variant_tag,
        )
        with pkl_path.open("rb") as fh:
            payload = pickle.load(fh)
        if not isinstance(payload, dict):
            raise TypeError(f"Unexpected payload type at {pkl_path}: {type(payload)}")
        for condition, obj in payload.items():
            pred, ctrl, truth, gene_names = _full_payload_item_arrays_strict(
                obj,
                pkl_path=pkl_path,
                condition=str(condition),
            )
            eval_truth_genes, _, _ = _rank_eval_deg_lists(
                pred=pred,
                ctrl=ctrl,
                truth=truth,
                gene_names=gene_names,
                top_k=top_k,
                condition=str(condition),
                remove_perturbed_genes=remove_perturbed_genes,
            )
            cache_genes = cache_idx_to_names(
                cache_obj=cache_obj,
                cache_key=cache_key,
                condition=str(condition),
                gene_names=filtered_gene_names,
            )
            if remove_perturbed_genes:
                perturbed = {tok for tok in str(condition).split("+") if tok and tok != "ctrl"}
                cache_genes = [g for g in cache_genes if g not in perturbed]
            cache_genes = cache_genes[: int(top_k)]
            metrics = overlap_metrics(eval_truth_genes, cache_genes)
            rows.append(
                {
                    "dataset": dataset,
                    "model_name": model_name,
                    "split_id": int(split_id),
                    "condition": str(condition),
                    "deg_flow": "scpram_eval_wilcoxon",
                    "cache_key": str(cache_key),
                    "top_k": int(top_k),
                    "remove_perturbed_genes": bool(remove_perturbed_genes),
                    "eval_truth_genes": "|".join(eval_truth_genes),
                    "cache_genes": "|".join(cache_genes),
                    **metrics,
                }
            )
    return pd.DataFrame(rows).sort_values(by=["split_id", "condition"]).reset_index(drop=True)


def summarize(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_rows: list[dict[str, Any]] = []
    for split_id, sub in df.groupby("split_id", sort=True):
        split_rows.append(
            {
                "split_id": int(split_id),
                "n_conditions": int(len(sub)),
                "mean_common_n": float(sub["common_n"].mean()),
                "mean_jaccard": float(sub["jaccard"].mean()),
                "mean_precision_vs_cache": float(sub["precision_vs_cache"].mean()),
                "mean_recall_vs_eval": float(sub["recall_vs_eval"].mean()),
                "exact_set_match_count": int((sub["jaccard"] == 1.0).sum()),
            }
        )
    split_df = pd.DataFrame(split_rows)
    dataset_df = pd.DataFrame(
        [
            {
                "n_splits": int(df["split_id"].nunique()),
                "n_conditions_total": int(len(df)),
                "mean_common_n": float(df["common_n"].mean()),
                "mean_jaccard": float(df["jaccard"].mean()),
                "mean_precision_vs_cache": float(df["precision_vs_cache"].mean()),
                "mean_recall_vs_eval": float(df["recall_vs_eval"].mean()),
                "exact_set_match_count": int((df["jaccard"] == 1.0).sum()),
            }
        ]
    )
    return split_df, dataset_df


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Compare scPRAM-style eval-derived truth DEG with full-dataset DEG cache."
    )
    ap.add_argument("--dataset", default="adamson")
    ap.add_argument("--model_name", default="trishift")
    ap.add_argument("--split_ids", default="1,2,3,4,5")
    ap.add_argument("--paths_path", default="configs/paths.yaml")
    ap.add_argument("--result_dir", default="")
    ap.add_argument("--variant_tag", default="nearest")
    ap.add_argument("--cache_key", default="top20_degs_non_dropout", choices=VALID_CACHE_KEYS)
    ap.add_argument("--cache_path", default="")
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--remove_perturbed_genes", action="store_true")
    ap.add_argument("--out_dir", default="")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    dataset = str(args.dataset).strip()
    model_name = str(args.model_name).strip()
    split_ids = parse_split_ids(args.split_ids)
    result_root = _result_root(model_name, dataset, args.result_dir or None)
    variant_tag = str(args.variant_tag).strip() or None
    cache_key = str(args.cache_key).strip()
    top_k = int(args.top_k)

    cache_path = (
        Path(args.cache_path).resolve()
        if str(args.cache_path).strip()
        else (REPO_ROOT / "artifacts" / "cache" / "degs" / f"{dataset}_degs.pkl").resolve()
    )
    out_dir = (
        Path(args.out_dir).resolve()
        if str(args.out_dir).strip()
        else (
            REPO_ROOT
            / "artifacts"
            / "cache"
            / "degs"
            / f"{dataset}_scpram_truth_vs_dataset_cache__top{top_k}__{cache_key}"
        ).resolve()
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] cache={cache_path}")
    with cache_path.open("rb") as fh:
        cache_obj = pickle.load(fh)

    print("[load] preparing embedding-filtered dataset")
    _, filtered_gene_names = prepare_filtered_dataset(dataset=dataset, paths_path=args.paths_path)

    print("[compute] comparing scPRAM-style eval truth with dataset cache")
    per_condition_df = build_comparison_rows(
        dataset=dataset,
        model_name=model_name,
        split_ids=split_ids,
        result_root=result_root,
        variant_tag=variant_tag,
        cache_key=cache_key,
        cache_obj=cache_obj,
        filtered_gene_names=filtered_gene_names,
        top_k=top_k,
        remove_perturbed_genes=bool(args.remove_perturbed_genes),
    )
    split_df, dataset_df = summarize(per_condition_df)

    per_condition_df.to_csv(out_dir / "per_condition_comparison.csv", index=False)
    split_df.to_csv(out_dir / "split_summary.csv", index=False)
    dataset_df.to_csv(out_dir / "dataset_summary.csv", index=False)
    meta = {
        "dataset": dataset,
        "model_name": model_name,
        "split_ids": split_ids,
        "variant_tag": variant_tag,
        "cache_key": cache_key,
        "top_k": top_k,
        "remove_perturbed_genes": bool(args.remove_perturbed_genes),
        "deg_flow": "scpram_eval_wilcoxon",
        "cache_path": str(cache_path),
        "out_dir": str(out_dir),
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("[done]")
    print(dataset_df.to_string(index=False))
    print(f"[saved] {out_dir}")


if __name__ == "__main__":
    main()
