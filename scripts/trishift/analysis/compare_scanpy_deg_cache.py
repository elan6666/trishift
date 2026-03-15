from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from trishift._utils import apply_alias_mapping, load_adata, load_embedding_df
from trishift.TriShiftData import TriShiftData


VALID_COMPARE_KEYS = (
    "top20_degs_non_dropout",
    "top20_degs_non_zero",
    "top20_degs",
)

DATASET_EMBEDDING_KEYS = {
    "adamson": "emb_b",
    "dixit": "emb_c",
    "norman": "emb_a",
    "replogle_k562_essential": "emb_c",
    "replogle_rpe1_essential": "emb_b",
}


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def get_gene_names(adata, var_gene_key: str = "gene_name") -> np.ndarray:
    if var_gene_key in adata.var.columns:
        return adata.var[var_gene_key].astype(str).values
    return adata.var_names.astype(str).values


def normalize_condition(cond: str) -> str:
    return str(cond).replace(" ", "")


def condition_to_gene_names(cond: str, ctrl_label: str = "ctrl") -> list[str]:
    cond_str = str(cond)
    if cond_str == ctrl_label:
        return []
    return normalize_condition(cond_str).split("+")


def compute_nonzero_non_dropout(
    pert_mean: np.ndarray,
    ctrl_mean: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    non_zero = np.where(pert_mean != 0)[0]
    zero = np.where(pert_mean == 0)[0]
    true_zeros = np.intersect1d(zero, np.where(ctrl_mean == 0)[0])
    non_dropouts = np.concatenate((non_zero, true_zeros))
    return non_zero, non_dropouts


def compute_scanpy_deg_cache_like(
    *,
    adata,
    label_key: str = "condition",
    ctrl_label: str = "ctrl",
    var_gene_key: str = "gene_name",
) -> dict[str, dict[str, np.ndarray]]:
    adata = adata.copy()
    cond_series = adata.obs[label_key].astype(str).values
    conditions_all = pd.unique(cond_series).tolist()
    conditions_pert = [c for c in conditions_all if c != ctrl_label]

    var_names_backup = None
    if var_gene_key in adata.var.columns:
        var_names_backup = adata.var_names.copy()
        adata.var_names = adata.var[var_gene_key].astype(str).values

    sc.tl.rank_genes_groups(
        adata,
        groupby=label_key,
        reference=ctrl_label,
        rankby_abs=True,
        n_genes=adata.n_vars,
        method="t-test",
    )
    names_df = pd.DataFrame(adata.uns["rank_genes_groups"]["names"])
    rank_genes_by_cond = {g: names_df[g].tolist() for g in names_df.columns}

    if var_names_backup is not None:
        adata.var_names = var_names_backup

    ctrl_mask = cond_series == ctrl_label
    ctrl_mean = np.asarray(adata[ctrl_mask].X.mean(axis=0)).ravel()
    gene_names = get_gene_names(adata, var_gene_key=var_gene_key)
    gene_id2idx = {g: i for i, g in enumerate(gene_names)}
    gene_to_var_index = {g: i for i, g in enumerate(gene_names)}

    top_de_20: dict[str, np.ndarray] = {}
    top_non_zero_de_20: dict[str, np.ndarray] = {}
    top_non_dropout_de_20: dict[str, np.ndarray] = {}
    top20_degs_final: dict[str, np.ndarray] = {}

    for cond in conditions_pert:
        pert_mask = cond_series == cond
        if not np.any(pert_mask):
            continue
        pert_mean = np.asarray(adata[pert_mask].X.mean(axis=0)).ravel()
        non_zero, non_dropouts = compute_nonzero_non_dropout(pert_mean, ctrl_mean)

        rank_genes = rank_genes_by_cond.get(cond, [])
        gene_idx_top = [gene_id2idx[g] for g in rank_genes if g in gene_id2idx]

        de_20 = np.asarray(gene_idx_top[:20], dtype=int)
        non_zero_20 = np.asarray(
            [i for i in gene_idx_top if i in non_zero][:20],
            dtype=int,
        )
        non_dropout_20 = np.asarray(
            [i for i in gene_idx_top if i in non_dropouts][:20],
            dtype=int,
        )

        top_de_20[cond] = de_20
        top_non_zero_de_20[cond] = non_zero_20
        top_non_dropout_de_20[cond] = non_dropout_20

        remove_idx = np.asarray(
            [
                gene_to_var_index[g]
                for g in condition_to_gene_names(cond, ctrl_label=ctrl_label)
                if g in gene_to_var_index
            ],
            dtype=int,
        )
        final_idx = non_dropout_20
        if remove_idx.size > 0:
            final_idx = np.setdiff1d(final_idx, remove_idx)
        top20_degs_final[cond] = np.asarray(final_idx, dtype=int)

    return {
        "top20_degs": top_de_20,
        "top20_degs_non_zero": top_non_zero_de_20,
        "top20_degs_non_dropout": top_non_dropout_de_20,
        "top20_degs_final": top20_degs_final,
    }


def load_deg_cache(path: str | Path) -> dict:
    with open(path, "rb") as fh:
        return pickle.load(fh)


def compare_deg_maps(
    *,
    computed: dict[str, np.ndarray],
    cached: dict[str, np.ndarray],
    gene_names: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    rows: list[dict[str, object]] = []
    all_conds = sorted(set(computed.keys()) | set(cached.keys()))

    exact_match_count = 0
    identical_set_count = 0

    for cond in all_conds:
        comp_idx = np.asarray(computed.get(cond, np.array([], dtype=int)), dtype=int)
        cache_idx = np.asarray(cached.get(cond, np.array([], dtype=int)), dtype=int)

        comp_names = gene_names[comp_idx].astype(str).tolist() if comp_idx.size else []
        cache_names = gene_names[cache_idx].astype(str).tolist() if cache_idx.size else []

        comp_set = set(comp_names)
        cache_set = set(cache_names)
        common = sorted(comp_set & cache_set)
        union = sorted(comp_set | cache_set)

        exact_match = list(comp_idx.tolist()) == list(cache_idx.tolist())
        identical_set = comp_set == cache_set
        if exact_match:
            exact_match_count += 1
        if identical_set:
            identical_set_count += 1

        rows.append(
            {
                "condition": cond,
                "computed_n": int(len(comp_names)),
                "cached_n": int(len(cache_names)),
                "common_n": int(len(common)),
                "jaccard": float(len(common) / len(union)) if union else 1.0,
                "exact_match": bool(exact_match),
                "identical_set": bool(identical_set),
                "computed_names": "|".join(comp_names),
                "cached_names": "|".join(cache_names),
                "common_names": "|".join(common),
            }
        )

    df = pd.DataFrame(rows).sort_values(
        by=["exact_match", "identical_set", "jaccard", "condition"],
        ascending=[True, True, True, True],
    )
    summary = {
        "n_conditions": int(len(all_conds)),
        "exact_match_count": int(exact_match_count),
        "identical_set_count": int(identical_set_count),
        "exact_match_ratio": float(exact_match_count / len(all_conds)) if all_conds else 0.0,
        "identical_set_ratio": float(identical_set_count / len(all_conds)) if all_conds else 0.0,
        "mean_jaccard": float(df["jaccard"].mean()) if not df.empty else 0.0,
        "min_jaccard": float(df["jaccard"].min()) if not df.empty else 0.0,
    }
    return df, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute Scanpy DEG20 and compare with cached degs.pkl."
    )
    parser.add_argument("--dataset", default="adamson")
    parser.add_argument("--paths_path", default="configs/paths.yaml")
    parser.add_argument("--apply_embedding_filter", action="store_true", default=True)
    parser.add_argument("--cache_path", default=None)
    parser.add_argument("--compare_key", default="top20_degs_non_dropout", choices=VALID_COMPARE_KEYS)
    parser.add_argument("--label_key", default="condition")
    parser.add_argument("--ctrl_label", default="ctrl")
    parser.add_argument("--var_gene_key", default="gene_name")
    parser.add_argument("--output_csv", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = load_yaml(args.paths_path)
    h5ad_path = Path(cfg["datasets"][args.dataset])
    cache_path = (
        Path(args.cache_path)
        if args.cache_path is not None
        else Path("artifacts") / "cache" / "degs" / f"{args.dataset}_degs.pkl"
    )

    print(f"[load] h5ad: {h5ad_path}")
    print(f"[load] cache: {cache_path}")
    adata = load_adata(str(h5ad_path))
    if args.apply_embedding_filter:
        emb_key = DATASET_EMBEDDING_KEYS.get(args.dataset)
        if emb_key is None:
            raise ValueError(f"No embedding key mapping for dataset: {args.dataset}")
        emb_path = cfg["embeddings"][emb_key]
        embd_df = load_embedding_df(emb_path)
        embd_df = apply_alias_mapping(embd_df, args.dataset)
        data = TriShiftData(adata, embd_df, label_key=args.label_key, var_gene_key=args.var_gene_key)
        data.setup_embedding_index()
        adata = data.adata_all
        print(f"[filter] embedding-filtered conditions: {len(data.conditions_pert)} pert + ctrl")
    cache = load_deg_cache(cache_path)
    gene_names = get_gene_names(adata, var_gene_key=args.var_gene_key)

    print("[compute] recomputing scanpy t-test DEG cache-like outputs")
    computed_all = compute_scanpy_deg_cache_like(
        adata=adata,
        label_key=args.label_key,
        ctrl_label=args.ctrl_label,
        var_gene_key=args.var_gene_key,
    )
    computed = computed_all[args.compare_key]
    cached = cache.get(args.compare_key, {})

    print(f"[compare] key={args.compare_key}")
    compare_df, summary = compare_deg_maps(
        computed=computed,
        cached=cached,
        gene_names=gene_names,
    )

    for k, v in summary.items():
        print(f"{k}: {v}")

    mismatches = compare_df.loc[~compare_df["exact_match"]]
    print(f"[compare] mismatched conditions: {len(mismatches)}")
    if not mismatches.empty:
        print("[compare] first mismatches:")
        preview = mismatches.loc[:, ["condition", "common_n", "jaccard", "computed_names", "cached_names"]].head(10)
        with pd.option_context("display.max_colwidth", 200):
            print(preview.to_string(index=False))

    output_csv = Path(args.output_csv) if args.output_csv else (
        Path("artifacts")
        / "cache"
        / "degs"
        / f"{args.dataset}_{args.compare_key}_scanpy_compare.csv"
    )
    compare_df.to_csv(output_csv, index=False)
    print(f"[save] comparison csv: {output_csv}")


if __name__ == "__main__":
    main()
