from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import importlib
import pickle
import random
import sys

import anndata as ad
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error as mse

ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = ROOT / "src"
LOCAL_DATA_ROOT = ROOT / "src" / "data"
LOCAL_EMBEDDING_PATH = (
    LOCAL_DATA_ROOT / "Data_GeneEmbd" / "GenePT_gene_embedding_ada_text.pickle"
)

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SRC_ROOT))

from trishift import _utils
from trishift._external_metrics import (
    average_of_perturbation_centroids,
    compute_scpram_metrics_from_arrays,
    pearson_delta_reference_metrics,
    regression_r2_safe,
)
from trishift.TriShiftData import TriShiftData
from scripts.common.split_utils import (
    condition_sort as _shared_condition_sort,
    norman_subgroup as _shared_norman_subgroup,
)
from scripts.common.yaml_utils import load_yaml_file


@dataclass(frozen=True)
class GenePertDatasetConfig:
    data_rel: str
    splits: list[int]
    test_ratio: float
    norman_split: bool = False


PROFILE_DIR = Path(__file__).resolve().parents[1] / "eval" / "configs"
PROFILE_ALLOWED_KEYS = {"dataset", "task_args"}
DEFAULT_ALPHA_GRID = [0.1, 1.0, 10.0]
ALIAS_GENE_TO_EMBEDDING_GENE = {
    "KIAA1804": "MAP3K21",
    "RHOXF2BB": "RHOXF2B",
    "AC118549.1": "ZZZ3",
}


DATASET_CONFIG = {
    "adamson": GenePertDatasetConfig(
        data_rel="data/adamson/perturb_processed.h5ad",
        splits=[1, 2, 3, 4, 5],
        test_ratio=0.2,
    ),
    "dixit": GenePertDatasetConfig(
        data_rel="data/dixit/perturb_processed.h5ad",
        splits=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        test_ratio=0.1,
    ),
    "norman": GenePertDatasetConfig(
        data_rel="data/norman/perturb_processed.h5ad",
        splits=[1, 2, 3, 4, 5],
        test_ratio=0.2,
        norman_split=True,
    ),
    "replogle_k562_essential": GenePertDatasetConfig(
        data_rel="data/replogle_k562_essential/perturb_processed.h5ad",
        splits=[1, 2, 3, 4, 5],
        test_ratio=0.2,
    ),
    "replogle_rpe1_essential": GenePertDatasetConfig(
        data_rel="data/replogle_rpe1_essential/perturb_processed.h5ad",
        splits=[1, 2, 3, 4, 5],
        test_ratio=0.2,
    ),
}


def _load_profile(profile: str) -> dict:
    p = PROFILE_DIR / f"{profile}.yaml"
    if not p.exists():
        raise FileNotFoundError(f"GenePert profile not found: {p}")
    obj = load_yaml_file(p)
    unknown = sorted(set(obj.keys()) - PROFILE_ALLOWED_KEYS)
    if unknown:
        raise ValueError(f"Unknown keys in GenePert profile {p.name}: {unknown}")
    dataset = str(obj.get("dataset", "")).strip()
    if dataset != profile:
        raise ValueError(f"Profile dataset mismatch: file={p.name} dataset={dataset!r}")
    task_args = obj.get("task_args") or {}
    if not isinstance(task_args, dict):
        raise TypeError(f"profile.task_args must be a mapping: {p}")
    return {"dataset": dataset, "task_args": task_args}


def _coerce_alpha_grid(alpha_grid) -> list[float]:
    if alpha_grid is None:
        return [float(x) for x in DEFAULT_ALPHA_GRID]
    if not isinstance(alpha_grid, (list, tuple)):
        raise TypeError("alpha_grid must be a list or tuple of numbers")
    out = [float(x) for x in alpha_grid]
    if not out:
        raise ValueError("alpha_grid must not be empty")
    return out


def run_profile(profile: str) -> None:
    prof = _load_profile(profile)
    task_args = prof["task_args"]
    run_genepert_eval(
        prof["dataset"],
        base_seed=int(task_args.get("seed", 24)),
        export_notebook_pkl=bool(task_args.get("export_notebook_pkl", True)),
        alpha_grid=_coerce_alpha_grid(task_args.get("alpha_grid")),
    )


def _resolve_mean_metric_keys(numeric_means: pd.Series) -> list[str]:
    preferred_order = [
        "pearson",
        "nmse",
        "mse_pred",
        "mse_ctrl",
        "deg_mean_r2",
        "systema_corr_20de_allpert",
        "systema_corr_deg_r2",
        "scpram_r2_degs_mean_mean",
        "scpram_r2_degs_var_mean",
        "scpram_wasserstein_degs_sum",
    ]
    exclude_keys = {"split_id", "n_ensemble"}
    keys = [k for k in preferred_order if k in numeric_means.index and k not in exclude_keys]
    keys.extend([k for k in numeric_means.index if k not in exclude_keys and k not in keys])
    return keys


def _append_subgroup_mean_lines(lines: list[str], metrics_df: pd.DataFrame, keys: list[str]) -> None:
    if "subgroup" not in metrics_df.columns:
        return
    subgroup_series = metrics_df["subgroup"]
    if subgroup_series.isna().all():
        return

    ordered = ["single", "seen0", "seen1", "seen2"]
    present = set(subgroup_series.dropna().astype(str).tolist())
    subgroup_names = [g for g in ordered if g in present]
    if "unknown" in present:
        subgroup_names.append("unknown")
    if not subgroup_names:
        return

    lines.append("# subgroup_means_row_weighted\n")
    lines.append("subgroup_order=single,seen0,seen1,seen2\n")
    lines.append("\n")

    for g in subgroup_names:
        sub_df = metrics_df[metrics_df["subgroup"].astype(str) == g]
        lines.append(f"subgroup_{g}_n_rows={int(len(sub_df))}\n")
        sub_numeric_means = sub_df.mean(numeric_only=True)
        for key in keys:
            if key not in sub_numeric_means.index:
                continue
            lines.append(f"subgroup_{g}_mean_{key}={float(sub_numeric_means[key])}\n")
        lines.append("\n")


def _attach_subgroup_column(metrics_df: pd.DataFrame, subgroup_df: pd.DataFrame | None) -> pd.DataFrame:
    if subgroup_df is None or "condition" not in metrics_df.columns:
        return metrics_df
    if "subgroup" not in subgroup_df.columns:
        return metrics_df

    out = metrics_df.copy()
    if "subgroup" in out.columns:
        out = out.drop(columns=["subgroup"])
    subgroup_map = subgroup_df["subgroup"].astype(str).to_dict()
    subgroup_vals = out["condition"].astype(str).map(subgroup_map).fillna("unknown")
    insert_at = list(out.columns).index("condition") + 1
    out.insert(insert_at, "subgroup", subgroup_vals)
    return out


def _write_mean_metrics(path: Path, metrics_df: pd.DataFrame) -> None:
    numeric_means = metrics_df.mean(numeric_only=True)
    mean_pearson = float(numeric_means.get("pearson", float("nan")))
    keys = _resolve_mean_metric_keys(numeric_means)

    lines = [f"{mean_pearson}\n"]
    for key in keys:
        lines.append(f"mean_{key}={float(numeric_means[key])}\n")
    _append_subgroup_mean_lines(lines, metrics_df, keys)
    path.write_text("".join(lines), encoding="utf-8")


def set_seeds(seed: int = 24) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def condition_sort(x: str) -> str:
    return _shared_condition_sort(x)


def subgroup(pert_list: list[str], seed: int) -> pd.DataFrame:
    return _shared_norman_subgroup(pert_list=pert_list, seed=seed)


def _require_genepert_class():
    try:
        mod = importlib.import_module("GenePertExperiment")
    except ImportError as exc:
        raise ImportError(
            "GenePert is not importable. Install the GenePert package/module into the current "
            "Python environment so `import GenePertExperiment` works."
        ) from exc
    return getattr(mod, "GenePertExperiment")


def _resolve_eval_data_path(name: str, cfg: GenePertDatasetConfig) -> Path:
    data_path_primary = LOCAL_DATA_ROOT / name / "perturb_processed.h5ad"
    if data_path_primary.exists():
        return data_path_primary

    paths_cfg_path = ROOT / "configs" / "paths.yaml"
    if paths_cfg_path.exists():
        paths_cfg = _utils.load_yaml(str(paths_cfg_path))
        datasets_map = paths_cfg.get("datasets", {})
        data_path_alt = Path(datasets_map[name]) if name in datasets_map else data_path_primary
        if data_path_alt.exists():
            return data_path_alt

    raise FileNotFoundError(
        "Missing evaluation h5ad for GenePert. "
        f"Tried primary data={data_path_primary}; fallback via {paths_cfg_path}."
    )


def _resolve_embedding_path() -> Path:
    if LOCAL_EMBEDDING_PATH.exists():
        return LOCAL_EMBEDDING_PATH
    raise FileNotFoundError(f"GenePert embedding file not found: {LOCAL_EMBEDDING_PATH}")


def _load_embedding_dict(emb_path: Path) -> dict[str, np.ndarray]:
    with emb_path.open("rb") as handle:
        raw = pickle.load(handle)
    if not isinstance(raw, dict):
        raise TypeError(f"Expected embedding pickle to load a dict: {emb_path}")
    out: dict[str, np.ndarray] = {}
    for key, value in raw.items():
        out[str(key)] = np.asarray(value, dtype=np.float32).reshape(-1)
    return out


def _prepare_eval_adata(data_path: Path) -> ad.AnnData:
    return _utils.load_adata(str(data_path))


def _condition_tokens_no_ctrl(condition: str) -> list[str]:
    return [tok for tok in str(condition).split("+") if tok and tok != "ctrl"]


def _clean_condition(condition: str) -> str:
    return "+".join(_condition_tokens_no_ctrl(condition))


def _dataset_topgene_prefix(dataset_name: str) -> str:
    if dataset_name == "adamson":
        return "K562(?)_"
    if dataset_name == "norman":
        return "A549_"
    if dataset_name in {"dixit", "replogle_k562_essential"}:
        return "K562_"
    if dataset_name == "replogle_rpe1_essential":
        return "rpe1_"
    raise ValueError(f"Unsupported GenePert dataset for topgene prefix: {dataset_name}")


def _resolve_gene_embeddings_for_adata(
    adata: ad.AnnData,
    embeddings_raw: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], list[str]]:
    needed_genes: set[str] = set()
    for cond in adata.obs["condition"].astype(str).unique():
        if str(cond) == "ctrl":
            continue
        needed_genes.update(_condition_tokens_no_ctrl(str(cond)))

    resolved: dict[str, np.ndarray] = {}
    missing: list[str] = []
    for gene in sorted(needed_genes):
        if gene in embeddings_raw:
            resolved[gene] = embeddings_raw[gene]
            continue
        alias_gene = ALIAS_GENE_TO_EMBEDDING_GENE.get(gene, "")
        if alias_gene and alias_gene in embeddings_raw:
            resolved[gene] = embeddings_raw[alias_gene]
            continue
        missing.append(gene)
    return resolved, missing


def _drop_unresolved_conditions(
    adata: ad.AnnData,
    resolved_embeddings: dict[str, np.ndarray],
) -> tuple[ad.AnnData, list[str]]:
    dropped: list[str] = []
    keep_mask = np.ones(adata.n_obs, dtype=bool)
    cond_arr = adata.obs["condition"].astype(str).values
    for cond in sorted(set(cond_arr.tolist())):
        if cond == "ctrl":
            continue
        genes = _condition_tokens_no_ctrl(cond)
        if any(g not in resolved_embeddings for g in genes):
            dropped.append(cond)
            keep_mask &= cond_arr != cond
    if dropped:
        print(f"[genepert] dropping {len(dropped)} conditions without embeddings: {dropped}")
    return adata[keep_mask].copy(), dropped


def _build_embedding_df(resolved_embeddings: dict[str, np.ndarray]) -> pd.DataFrame:
    embd_df = pd.DataFrame(resolved_embeddings).T
    ctrl_row = pd.DataFrame(
        [np.zeros(embd_df.shape[1], dtype=np.float32)],
        columns=embd_df.columns,
        index=["ctrl"],
    )
    return pd.concat([ctrl_row, embd_df])


def _select_alpha_from_result(result: dict) -> float:
    ridge_metrics = ((result or {}).get("aggregate") or {}).get("ridge") or {}
    if not ridge_metrics:
        raise ValueError("GenePert alpha sweep returned no ridge metrics")

    ranked = []
    for params, metrics in ridge_metrics.items():
        alpha = None
        for key, value in params:
            if str(key) == "alpha":
                alpha = float(value)
                break
        if alpha is None:
            continue
        corr = float(metrics.get("corr", np.nan))
        mse_val = float(metrics.get("mse", np.nan))
        corr_rank = -corr if np.isfinite(corr) else float("inf")
        mse_rank = mse_val if np.isfinite(mse_val) else float("inf")
        ranked.append((corr_rank, mse_rank, alpha))
    if not ranked:
        raise ValueError("GenePert alpha sweep produced no parsable alpha values")
    ranked.sort()
    return float(ranked[0][2])


def _subset_no_ctrl(adata: ad.AnnData) -> ad.AnnData:
    mask = adata.obs["condition"].astype(str).values != "ctrl"
    return adata[mask].copy()


def _combine_splits(adata: ad.AnnData, conds: list[str]) -> ad.AnnData:
    cond_set = set(str(c) for c in conds)
    cond_set.discard("")
    cond_set.add("ctrl")
    mask = adata.obs["condition"].astype(str).isin(cond_set).values
    return adata[mask].copy()


def _topgene_dict_or_raise(eval_adata: ad.AnnData) -> dict:
    topgene_dict = eval_adata.uns.get("top_non_dropout_de_20", {})
    if not isinstance(topgene_dict, dict):
        raise TypeError("Expected eval_adata.uns['top_non_dropout_de_20'] to be a mapping")
    return topgene_dict


def _compute_metrics_and_export_payload(
    *,
    eval_adata: ad.AnnData,
    reference_adata: ad.AnnData,
    dataset_name: str,
    split_id: int,
    result: dict,
) -> tuple[pd.DataFrame, dict]:
    results = []
    export_payload = {}
    eval_conds = eval_adata.obs["condition"].astype(str).values
    ctrl_mean = _utils.densify_X(eval_adata[eval_adata.obs["condition"] == "ctrl"].X).mean(
        axis=0,
        keepdims=True,
    )
    pert_reference = average_of_perturbation_centroids(
        X=_utils.densify_X(reference_adata.X),
        conditions=reference_adata.obs["condition"].astype(str).values,
        ctrl_label="ctrl",
    )
    if "gene_name" in eval_adata.var.columns:
        gene_names = eval_adata.var["gene_name"].astype(str).values
    else:
        gene_names = eval_adata.var_names.astype(str).values
    raw_to_gene_name = dict(zip(eval_adata.var.index.astype(str).values, gene_names))
    gene_name_to_idx = {str(g): i for i, g in enumerate(gene_names)}
    topgene_dict = _topgene_dict_or_raise(eval_adata)
    per_gene = (result or {}).get("per_gene") or {}
    clean_to_raw_test = {}
    for cond in sorted(set(eval_conds.tolist())):
        if cond == "ctrl":
            continue
        clean_to_raw_test[_clean_condition(cond)] = cond

    for clean_cond, model_results in per_gene.items():
        raw_cond = clean_to_raw_test.get(str(clean_cond))
        if not raw_cond:
            continue
        ridge_payload = model_results.get("ridge") or {}
        if len(ridge_payload) != 1:
            raise ValueError(
                f"Expected exactly one final GenePert ridge result for {raw_cond}, got {len(ridge_payload)}"
            )
        _, packed = next(iter(ridge_payload.items()))
        pred_mean = np.asarray(packed[2], dtype=np.float32).reshape(1, -1)
        cond_mask = eval_conds == raw_cond
        if not bool(cond_mask.any()):
            print(f"[genepert] skip condition missing in eval adata: {raw_cond}")
            continue
        true = _utils.densify_X(eval_adata[cond_mask].X)
        topgene_key = f"{_dataset_topgene_prefix(dataset_name)}{raw_cond}_1+1"
        if topgene_key not in topgene_dict:
            print(f"[genepert] skip condition without top_non_dropout_de_20 entry: {raw_cond}")
            continue
        deg_gene_names = []
        for raw_gene in topgene_dict[topgene_key]:
            mapped_name = raw_to_gene_name.get(str(raw_gene))
            if mapped_name is None:
                continue
            deg_gene_names.append(str(mapped_name))
        degs = np.asarray(
            [gene_name_to_idx[g] for g in deg_gene_names if g in gene_name_to_idx],
            dtype=int,
        ).reshape(-1)
        if degs.size == 0:
            print(f"[genepert] skip condition without DEGs: {raw_cond}")
            continue
        pert_genes = np.where(
            np.isin(gene_names, np.asarray(_condition_tokens_no_ctrl(raw_cond), dtype=str))
        )[0]
        degs = np.setdiff1d(degs, pert_genes)
        if degs.size == 0:
            print(f"[genepert] skip condition after removing perturbed genes from DEGs: {raw_cond}")
            continue

        pred = np.repeat(pred_mean, true.shape[0], axis=0)
        ctrl = np.repeat(ctrl_mean, true.shape[0], axis=0)
        pred_vec = pred[:, degs].mean(axis=0)
        ctrl_vec = ctrl[:, degs].mean(axis=0)
        true_vec = true[:, degs].mean(axis=0)

        mse_ctrl_val = float(mse(true_vec, ctrl_vec))
        mse_pred_val = float(mse(true_vec, pred_vec))
        nmse_val = float(mse_pred_val / mse_ctrl_val) if mse_ctrl_val > 0 else np.nan
        pearson_val = float(pearsonr(true_vec - ctrl_vec, pred_vec - ctrl_vec)[0])
        deg_mean_r2_val = regression_r2_safe(true_vec - ctrl_vec, pred_vec - ctrl_vec)
        systema_metrics = pearson_delta_reference_metrics(
            X_true=true.mean(axis=0),
            X_pred=pred.mean(axis=0),
            reference=pert_reference,
            top20_de_idxs=degs,
        )
        scpram_metrics = compute_scpram_metrics_from_arrays(
            X_true=true,
            X_pred=pred,
            deg_idx=degs,
            n_degs=100,
            sample_ratio=0.8,
            times=100,
        )
        results.append(
            {
                "condition": raw_cond,
                "mse_pred": mse_pred_val,
                "mse_ctrl": mse_ctrl_val,
                "nmse": nmse_val,
                "pearson": pearson_val,
                "deg_mean_r2": float(deg_mean_r2_val),
                "systema_corr_20de_allpert": float(systema_metrics["corr_20de_allpert"]),
                "systema_corr_deg_r2": float(systema_metrics["corr_deg_r2"]),
                **scpram_metrics,
                "split_id": int(split_id),
                "n_ensemble": int(pred.shape[0]),
            }
        )
        export_payload[raw_cond] = {
            "Pred": pred[:, degs] if degs.size > 0 else pred[:, :0],
            "Ctrl": ctrl[:, degs] if degs.size > 0 else ctrl[:, :0],
            "Truth": true[:, degs] if degs.size > 0 else true[:, :0],
            "Pred_full": pred,
            "Ctrl_full": ctrl,
            "Truth_full": true,
            "DE_idx": degs,
            "DE_name": gene_names[degs] if degs.size > 0 else np.array([], dtype=gene_names.dtype),
            "gene_name_full": gene_names,
        }

    return pd.DataFrame(results), export_payload


def run_genepert_eval(
    name: str,
    base_seed: int = 24,
    export_notebook_pkl: bool = True,
    alpha_grid: list[float] | None = None,
) -> None:
    if name not in DATASET_CONFIG:
        raise ValueError(f"Unknown dataset: {name}")
    cfg = DATASET_CONFIG[name]
    alpha_grid_eff = _coerce_alpha_grid(alpha_grid)
    GenePertExperiment = _require_genepert_class()

    data_path = _resolve_eval_data_path(name, cfg)
    emb_path = _resolve_embedding_path()

    out_dir = ROOT / "artifacts" / "results" / "genepert" / name
    out_dir.mkdir(parents=True, exist_ok=True)

    adata = _prepare_eval_adata(data_path)
    embeddings_raw = _load_embedding_dict(emb_path)
    resolved_embeddings, _ = _resolve_gene_embeddings_for_adata(adata, embeddings_raw)
    adata, _ = _drop_unresolved_conditions(adata, resolved_embeddings)
    embd_df = _build_embedding_df(resolved_embeddings)
    data = TriShiftData(adata, embd_df)
    data.setup_embedding_index()
    filtered_adata = data.adata_all
    filtered_embeddings, _ = _resolve_gene_embeddings_for_adata(filtered_adata, embeddings_raw)

    metrics_all = []
    for split in cfg.splits:
        print(f"[genepert] dataset={name} split={split}")
        set_seeds(base_seed + int(split))

        subgroup_df = None
        if cfg.norman_split:
            subgroup_df = subgroup(
                list(filtered_adata.obs["condition"].astype(str).unique()),
                seed=int(split),
            )
            test_conds = list(subgroup_df[subgroup_df.group == "test"].index)
            val_conds = list(subgroup_df[subgroup_df.group == "val"].index)
            split_dict = data.split_by_condition(
                seed=int(split),
                test_conds=test_conds,
                val_conds=val_conds,
            )
        else:
            split_dict = data.split_by_condition(
                seed=int(split),
                test_ratio=float(cfg.test_ratio),
            )

        train_no_ctrl = _subset_no_ctrl(split_dict["train"])
        val_no_ctrl = _subset_no_ctrl(split_dict["val"])
        test_no_ctrl = _subset_no_ctrl(split_dict["test"])
        if train_no_ctrl.n_obs == 0:
            raise ValueError(f"GenePert train split is empty for dataset={name} split={split}")
        if val_no_ctrl.n_obs == 0:
            raise ValueError(f"GenePert val split is empty for dataset={name} split={split}")
        if test_no_ctrl.n_obs == 0:
            raise ValueError(f"GenePert test split is empty for dataset={name} split={split}")

        experiment = GenePertExperiment(filtered_embeddings)
        experiment.adata = filtered_adata
        experiment.mean_expression = np.asarray(
            filtered_adata[filtered_adata.obs["condition"] == "ctrl"].to_df().mean(),
            dtype=np.float32,
        )

        ridge_params = [{"alpha": float(alpha)} for alpha in alpha_grid_eff]
        val_result = experiment.run_experiment_with_adata(
            train_no_ctrl,
            val_no_ctrl,
            ridge_params=ridge_params,
            knn_params=[],
        )
        best_alpha = _select_alpha_from_result(val_result)
        print(f"[genepert] dataset={name} split={split} selected_alpha={best_alpha}")

        train_val_conds = split_dict["train_conds"] + split_dict["val_conds"]
        train_val_adata = _combine_splits(filtered_adata, train_val_conds)
        train_val_no_ctrl = _subset_no_ctrl(train_val_adata)
        final_result = experiment.run_experiment_with_adata(
            train_val_no_ctrl,
            test_no_ctrl,
            ridge_params=[{"alpha": float(best_alpha)}],
            knn_params=[],
        )
        metrics_df, export_payload = _compute_metrics_and_export_payload(
            eval_adata=filtered_adata,
            reference_adata=train_val_adata,
            dataset_name=name,
            split_id=int(split),
            result=final_result,
        )
        metrics_df = _attach_subgroup_column(metrics_df, subgroup_df)
        metrics_all.append(metrics_df)
        if export_notebook_pkl:
            out_pkl = out_dir / f"genepert_{name}_{split}.pkl"
            with out_pkl.open("wb") as handle:
                pickle.dump(export_payload, handle)
            print(f"[genepert] saved notebook payload: {out_pkl}")

    metrics_df_all = pd.concat(metrics_all, ignore_index=True)
    metrics_df_all.to_csv(out_dir / "metrics.csv", index=False)
    _write_mean_metrics(out_dir / "mean_pearson.txt", metrics_df_all)
    print(f"[genepert] saved metrics: {out_dir / 'metrics.csv'}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run GenePert eval with TriShift metrics")
    parser.add_argument("--profile", default="", help="dataset profile under scripts/genepert/eval/configs")
    parser.add_argument("--name", default="", help="dataset name (legacy mode)")
    parser.add_argument("--seed", type=int, default=24, help="base seed")
    parser.add_argument(
        "--no_export_notebook_pkl",
        action="store_true",
        help="disable notebook-compatible pickle export",
    )
    args = parser.parse_args(argv)

    profile = str(args.profile).strip()
    if profile:
        prof = _load_profile(profile)
        task_args = prof["task_args"]
        seed = int(task_args.get("seed", 24)) if int(args.seed) == 24 else int(args.seed)
        export_notebook_pkl = bool(task_args.get("export_notebook_pkl", True))
        if bool(args.no_export_notebook_pkl):
            export_notebook_pkl = False
        alpha_grid = _coerce_alpha_grid(task_args.get("alpha_grid"))
        run_genepert_eval(
            prof["dataset"],
            base_seed=seed,
            export_notebook_pkl=export_notebook_pkl,
            alpha_grid=alpha_grid,
        )
        return

    name = str(args.name).strip()
    if not name:
        raise SystemExit("Provide --profile or --name")
    run_genepert_eval(
        name,
        base_seed=int(args.seed),
        export_notebook_pkl=not bool(args.no_export_notebook_pkl),
    )
