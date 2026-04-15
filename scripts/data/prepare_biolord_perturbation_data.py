from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DATA_ROOT = REPO_ROOT / "src" / "data"
GEARS_DATA_ROOT = SRC_DATA_ROOT / "Data_GEARS"
ESSENTIAL_GENES_PATH = GEARS_DATA_ROOT / "essential_all_data_pert_genes.pkl"


@dataclass(frozen=True)
class BiolordPrepConfig:
    dataset_name: str
    split_ids: tuple[int, ...]
    split_suffix: str
    ordered_attribute_key: str
    norman_style: bool = False


DEFAULT_CONFIGS: dict[str, BiolordPrepConfig] = {
    "adamson": BiolordPrepConfig(
        dataset_name="adamson",
        split_ids=(1, 2, 3, 4, 5),
        split_suffix="0.75",
        ordered_attribute_key="perturbation_neighbors",
    ),
    "norman": BiolordPrepConfig(
        dataset_name="norman",
        split_ids=(1, 2, 3, 4, 5),
        split_suffix="0.75",
        ordered_attribute_key="perturbation_neighbors1",
        norman_style=True,
    ),
    "dixit": BiolordPrepConfig(
        dataset_name="dixit",
        split_ids=(1, 2, 3, 4, 5),
        split_suffix="0.9",
        ordered_attribute_key="perturbation_neighbors",
    ),
}


def _load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def _dataset_paths(dataset_name: str) -> dict[str, Path]:
    return {
        "full_input": SRC_DATA_ROOT / dataset_name / "perturb_processed.h5ad",
        "go_csv": GEARS_DATA_ROOT / dataset_name / "go.csv",
        "split_dir": GEARS_DATA_ROOT / dataset_name / "splits",
        "full_output": SRC_DATA_ROOT / dataset_name / f"{dataset_name}_biolord.h5ad",
        "single_output": SRC_DATA_ROOT / dataset_name / f"{dataset_name}_single_biolord.h5ad",
        "summary_output": SRC_DATA_ROOT / dataset_name / f"{dataset_name}_biolord_prep_summary.json",
    }


def _require_files(paths: dict[str, Path]) -> None:
    required = ["full_input", "go_csv", "split_dir"]
    for key in required:
        if not paths[key].exists():
            raise FileNotFoundError(f"Missing required input for BioLORD preprocessing: {paths[key]}")
    if not ESSENTIAL_GENES_PATH.exists():
        raise FileNotFoundError(f"Missing essential gene whitelist: {ESSENTIAL_GENES_PATH}")


def _read_split_files(split_dir: Path, dataset_name: str, split_id: int, split_suffix: str) -> tuple[dict, dict]:
    split_path = split_dir / f"{dataset_name}_simulation_{int(split_id)}_{split_suffix}.pkl"
    subgroup_path = split_dir / f"{dataset_name}_simulation_{int(split_id)}_{split_suffix}_subgroup.pkl"
    if not split_path.exists():
        raise FileNotFoundError(f"Missing split file: {split_path}")
    if not subgroup_path.exists():
        raise FileNotFoundError(f"Missing split subgroup file: {subgroup_path}")
    split_obj = _load_pickle(split_path)
    subgroup_obj = _load_pickle(subgroup_path)
    if not isinstance(split_obj, dict):
        raise TypeError(f"Expected split file to be a dict: {split_path}")
    if not isinstance(subgroup_obj, dict):
        raise TypeError(f"Expected subgroup file to be a dict: {subgroup_path}")
    return split_obj, subgroup_obj


def _add_split_columns(
    adata: ad.AnnData,
    *,
    dataset_name: str,
    split_ids: tuple[int, ...],
    split_suffix: str,
    split_dir: Path,
    norman_style: bool,
) -> ad.AnnData:
    out = adata.copy()
    cond_series = out.obs["condition"].astype(str)
    for split_id in split_ids:
        split_obj, subgroup_obj = _read_split_files(split_dir, dataset_name, int(split_id), split_suffix)
        cond_to_split: dict[str, str] = {}
        for src_label, dst_label in {"train": "train", "val": "test", "test": "ood"}.items():
            for cond in split_obj.get(src_label, []):
                cond_to_split[str(cond)] = dst_label
        split_vals = [cond_to_split.get(str(cond), "ignore") for cond in cond_series]
        out.obs[f"split{int(split_id)}"] = pd.Categorical(split_vals, categories=["train", "test", "ood", "ignore"])

        subgroup_vals = np.full(out.n_obs, "Train/Val", dtype=object)
        subgroup_dict = subgroup_obj.get("test_subgroup", {}) or {}
        subgroup_lookup: dict[str, str] = {}
        for subgroup_name, conds in subgroup_dict.items():
            for cond in conds:
                subgroup_lookup[str(cond)] = str(subgroup_name)
        for idx, cond in enumerate(cond_series):
            subgroup_vals[idx] = subgroup_lookup.get(str(cond), "Train/Val")
        if not norman_style:
            subgroup_vals = np.where(np.asarray(split_vals) == "ood", "unseen_single", subgroup_vals)
        out.obs[f"subgroup{int(split_id)}"] = pd.Categorical(
            subgroup_vals,
            categories=[
                "Train/Val",
                "unseen_single",
                "combo_seen0",
                "combo_seen1",
                "combo_seen2",
            ],
        )
    return out


def _prepare_perturbation_columns(adata: ad.AnnData, norman_style: bool) -> ad.AnnData:
    out = adata.copy()
    conds = out.obs["condition"].astype(str)
    if norman_style:
        perturbation = pd.Series([cond.split("+")[0] for cond in conds], index=out.obs_names, dtype="object").astype("category")
        perturbation_rep = pd.Series(
            [cond.split("+")[1] if len(cond.split("+")) > 1 else "ctrl" for cond in conds],
            index=out.obs_names,
            dtype="object",
        ).astype("category")
        new_cats = perturbation_rep.cat.categories[~perturbation_rep.cat.categories.isin(perturbation.cat.categories)]
        perturbation = perturbation.cat.add_categories(new_cats)
        new_cats_rep = perturbation.cat.categories[~perturbation.cat.categories.isin(perturbation_rep.cat.categories)]
        perturbation_rep = perturbation_rep.cat.add_categories(new_cats_rep)
        perturbation = perturbation.cat.reorder_categories(perturbation_rep.cat.categories)
        out.obs["perturbation"] = perturbation
        out.obs["perturbation_rep"] = perturbation_rep
    else:
        out.obs["perturbation"] = pd.Series([cond.split("+")[0] for cond in conds], index=out.obs_names, dtype="object").astype(
            "category"
        )
    return out


def _build_go_neighbors(
    adata: ad.AnnData,
    *,
    go_csv_path: Path,
    require_go_coverage: bool,
) -> tuple[dict[str, np.ndarray], np.ndarray, list[str]]:
    go_df = pd.read_csv(go_csv_path)
    required_cols = {"source", "target", "importance"}
    if not required_cols.issubset(set(go_df.columns)):
        raise ValueError(f"{go_csv_path} must contain columns: {sorted(required_cols)}")
    go_df = (
        go_df.sort_values(["target", "importance"], ascending=[True, False])
        .groupby("target", as_index=False, group_keys=False)
        .head(21)
        .reset_index(drop=True)
    )

    gene_list = np.asarray(_load_pickle(ESSENTIAL_GENES_PATH), dtype=object)
    gene_set = set(map(str, gene_list.tolist()))
    go_df = go_df[go_df["source"].astype(str).isin(gene_set)].copy()

    basis = pd.Index([str(g) for g in gene_list.tolist()])
    basis_to_idx = {gene: idx for idx, gene in enumerate(basis)}
    pert_categories = [str(val) for val in adata.obs["perturbation"].cat.categories]

    missing: list[str] = []
    pert2neighbor: dict[str, np.ndarray] = {}
    grouped = go_df.groupby("target")
    for pert in pert_categories:
        vec = np.zeros(len(basis), dtype=np.float32)
        if pert in grouped.groups:
            sub = grouped.get_group(pert)
            for row in sub.itertuples(index=False):
                src = str(row.source)
                if src in basis_to_idx:
                    vec[basis_to_idx[src]] = float(row.importance)
        if np.count_nonzero(vec) == 0 and pert != "ctrl":
            missing.append(pert)
        pert2neighbor[pert] = vec

    if require_go_coverage and missing:
        preview = ", ".join(missing[:20])
        suffix = " ..." if len(missing) > 20 else ""
        raise RuntimeError(
            f"These perturbations are not in the GO graph and cannot be encoded for BioLORD: {preview}{suffix}"
        )
    return pert2neighbor, basis.values.astype(object), missing


def _condition_mean_matrix(adata: ad.AnnData) -> pd.DataFrame:
    dense_x = np.asarray(adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X, dtype=np.float32)
    df = pd.DataFrame(dense_x, index=adata.obs_names, columns=adata.var_names)
    df["condition"] = adata.obs["condition"].astype(str).values
    return df.groupby("condition", observed=False).mean()


def _build_single_adata(
    full_adata: ad.AnnData,
    *,
    pert2neighbor: dict[str, np.ndarray],
    ordered_attribute_key: str,
    split_ids: tuple[int, ...],
    norman_style: bool,
) -> ad.AnnData:
    cond_categories = list(dict.fromkeys(full_adata.obs["condition"].astype(str).tolist()))
    single_conditions: list[str] = []
    single_perts_name: list[str] = []
    for cond in cond_categories:
        parts = cond.split("+")
        if len(parts) == 1:
            continue
        if "ctrl" in parts:
            single_conditions.append(cond)
            p1, p2 = parts
            single_perts_name.append(p1 if p2 == "ctrl" else p2)
    if "ctrl" not in single_conditions:
        single_conditions.append("ctrl")
        single_perts_name.append("ctrl")

    cond_means = _condition_mean_matrix(full_adata)
    single_matrix = cond_means.loc[single_conditions].values.astype(np.float32)
    single_var = full_adata.var.copy()
    single_adata = ad.AnnData(X=single_matrix, var=single_var, dtype=single_matrix.dtype)
    single_adata.obs_names = pd.Index(single_conditions, dtype="object")
    single_adata.obs["condition"] = pd.Categorical(single_conditions, categories=single_conditions)
    single_adata.obs["perts_name"] = pd.Categorical(single_perts_name, categories=pd.Index(single_perts_name).unique())

    pert2neighbor_arr = np.asarray([pert2neighbor[str(pert)] for pert in full_adata.obs["perturbation"].cat.categories], dtype=np.float32)
    keep_idx = pert2neighbor_arr.sum(axis=0) > 0
    single_adata.obsm["perturbation_neighbors"] = np.asarray(
        [pert2neighbor[str(pert)][keep_idx] for pert in single_adata.obs["perts_name"].astype(str)],
        dtype=np.float32,
    )

    if norman_style:
        keep_idx1 = pert2neighbor_arr.sum(axis=0) > 1
        keep_idx2 = pert2neighbor_arr.sum(axis=0) > 2
        keep_idx3 = pert2neighbor_arr.sum(axis=0) > 3
        single_adata.obsm["perturbation_neighbors1"] = np.asarray(
            [pert2neighbor[str(pert)][keep_idx1] for pert in single_adata.obs["perts_name"].astype(str)],
            dtype=np.float32,
        )
        single_adata.obsm["perturbation_neighbors2"] = np.asarray(
            [pert2neighbor[str(pert)][keep_idx2] for pert in single_adata.obs["perts_name"].astype(str)],
            dtype=np.float32,
        )
        single_adata.obsm["perturbation_neighbors3"] = np.asarray(
            [pert2neighbor[str(pert)][keep_idx3] for pert in single_adata.obs["perts_name"].astype(str)],
            dtype=np.float32,
        )

    full_condition_series = full_adata.obs["condition"].astype(str)
    for split_id in split_ids:
        split_col = f"split{int(split_id)}"
        subgroup_col = f"subgroup{int(split_id)}"
        single_split_vals = np.full(single_adata.n_obs, "ignore", dtype=object)
        single_subgroup_vals = np.full(single_adata.n_obs, "Train/Val", dtype=object)
        for cat in ["train", "test", "ood"]:
            condition_set = set(
                full_condition_series[full_adata.obs[split_col].astype(str).eq(cat).values].astype(str).tolist()
            )
            mask = single_adata.obs["condition"].astype(str).isin(condition_set).values
            single_split_vals[mask] = cat
            if cat == "ood":
                if norman_style:
                    for ood_set in ["combo_seen0", "combo_seen1", "combo_seen2", "unseen_single"]:
                        subgroup_condition_set = set(
                            full_condition_series[full_adata.obs[subgroup_col].astype(str).eq(ood_set).values]
                            .astype(str)
                            .tolist()
                        )
                        subgroup_mask = single_adata.obs["condition"].astype(str).isin(subgroup_condition_set).values
                        single_subgroup_vals[subgroup_mask] = ood_set
                else:
                    single_subgroup_vals[mask] = "unseen_single"
        single_adata.obs[split_col] = pd.Categorical(single_split_vals, categories=["train", "test", "ood", "ignore"])
        single_adata.obs[subgroup_col] = pd.Categorical(
            single_subgroup_vals,
            categories=["Train/Val", "unseen_single", "combo_seen0", "combo_seen1", "combo_seen2"],
        )

    if ordered_attribute_key not in single_adata.obsm:
        raise KeyError(f"Missing ordered attribute key in single BioLORD adata: {ordered_attribute_key}")
    return single_adata


def _write_outputs(
    *,
    full_adata: ad.AnnData,
    single_adata: ad.AnnData,
    summary: dict,
    paths: dict[str, Path],
    overwrite: bool,
) -> None:
    for key in ["full_output", "single_output", "summary_output"]:
        if paths[key].exists() and not overwrite:
            raise FileExistsError(f"Output already exists. Pass --overwrite to replace: {paths[key]}")
        paths[key].parent.mkdir(parents=True, exist_ok=True)
    full_adata.write(paths["full_output"])
    single_adata.write(paths["single_output"])
    paths["summary_output"].write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def prepare_dataset(
    dataset_name: str,
    *,
    split_ids: tuple[int, ...],
    overwrite: bool,
    require_go_coverage: bool,
) -> dict:
    if dataset_name not in DEFAULT_CONFIGS:
        raise ValueError(f"Unsupported BioLORD dataset: {dataset_name}")
    cfg = DEFAULT_CONFIGS[dataset_name]
    paths = _dataset_paths(dataset_name)
    _require_files(paths)

    full_adata = ad.read_h5ad(paths["full_input"])
    full_adata = _add_split_columns(
        full_adata,
        dataset_name=dataset_name,
        split_ids=split_ids,
        split_suffix=cfg.split_suffix,
        split_dir=paths["split_dir"],
        norman_style=cfg.norman_style,
    )
    full_adata = _prepare_perturbation_columns(full_adata, cfg.norman_style)
    pert2neighbor, basis, missing = _build_go_neighbors(
        full_adata,
        go_csv_path=paths["go_csv"],
        require_go_coverage=require_go_coverage,
    )
    full_adata.uns["pert2neighbor"] = pert2neighbor
    full_adata.uns["biolord_neighbor_basis"] = basis

    single_adata = _build_single_adata(
        full_adata,
        pert2neighbor=pert2neighbor,
        ordered_attribute_key=cfg.ordered_attribute_key,
        split_ids=split_ids,
        norman_style=cfg.norman_style,
    )

    summary = {
        "dataset": dataset_name,
        "split_ids": [int(x) for x in split_ids],
        "ordered_attribute_key": cfg.ordered_attribute_key,
        "full_output": str(paths["full_output"]),
        "single_output": str(paths["single_output"]),
        "n_full_obs": int(full_adata.n_obs),
        "n_full_vars": int(full_adata.n_vars),
        "n_single_obs": int(single_adata.n_obs),
        "n_single_vars": int(single_adata.n_vars),
        "n_perturbation_categories": int(len(full_adata.obs["perturbation"].cat.categories)),
        "n_missing_go_perturbations": int(len(missing)),
        "missing_go_perturbations": missing,
        "single_attribute_shapes": {
            key: [int(v) for v in single_adata.obsm[key].shape]
            for key in sorted(single_adata.obsm.keys())
            if key.startswith("perturbation_neighbors")
        },
    }
    _write_outputs(
        full_adata=full_adata,
        single_adata=single_adata,
        summary=summary,
        paths=paths,
        overwrite=overwrite,
    )
    return summary


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Prepare BioLORD-specific perturbation datasets with GO-graph neighbor embeddings "
            "for Adamson, Norman, and Dixit."
        )
    )
    ap.add_argument(
        "--datasets",
        nargs="+",
        default=["adamson", "norman", "dixit"],
        choices=sorted(DEFAULT_CONFIGS),
        help="Datasets to preprocess for BioLORD.",
    )
    ap.add_argument(
        "--split-ids",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4, 5],
        help="Split ids to materialize into split*/subgroup* columns.",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing *_biolord.h5ad outputs.",
    )
    ap.add_argument(
        "--require-go-coverage",
        action="store_true",
        help="Fail if any perturbation gene has no GO neighbor vector after filtering.",
    )
    return ap


def main() -> None:
    ap = build_argparser()
    args = ap.parse_args()
    split_ids = tuple(int(x) for x in args.split_ids)
    for dataset_name in args.datasets:
        summary = prepare_dataset(
            dataset_name,
            split_ids=split_ids,
            overwrite=bool(args.overwrite),
            require_go_coverage=bool(args.require_go_coverage),
        )
        print(
            f"[biolord-prep] dataset={dataset_name} full={summary['full_output']} "
            f"single={summary['single_output']} missing_go={summary['n_missing_go_perturbations']}"
        )


if __name__ == "__main__":
    main()
