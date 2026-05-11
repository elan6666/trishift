from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import inspect
import pickle
import random
import sys
import warnings

import anndata as ad
import numpy as np
import pandas as pd
import torch
import scipy.linalg
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error as mse

ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = ROOT / "src"
LOCAL_DATA_ROOT = ROOT / "src" / "data"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SRC_ROOT))

from trishift import _utils
from trishift._external_metrics import (
    average_of_perturbation_centroids,
    compute_distributional_systema_metrics_from_arrays,
    compute_mean_effect_metrics,
    compute_scpram_metrics_bundle_from_arrays,
    pearson_delta_reference_metrics,
    regression_r2_safe,
)
from trishift.TriShiftData import TriShiftData
from scripts.common.yaml_utils import load_yaml_file


@dataclass(frozen=True)
class BiolordDatasetConfig:
    full_data_rel: str
    splits: list[int]
    single_data_rel: str = ""
    ordered_attribute_key: str = "perturbation_neighbors"
    legacy_multihot_fallback: bool = False
    test_ratio: float | None = None
    split_policy: str = "condition"
    domain_key: str = ""
    train_domain_values: tuple[str, ...] = ()
    test_domain_values: tuple[str, ...] = ()
    domain_test_ratio: float = 0.2
    drug_test_ratio: float = 0.2
    val_ratio: float = 0.1
    valid_split: str = "test"
    test_split: str = "ood"
    aggregate_legacy_condition_means: bool = False
    n_latent: int = 32
    batch_size: int = 32
    max_epochs: int = 75
    early_stopping: bool = True
    early_stopping_patience: int = 10
    check_val_every_n_epoch: int = 5
    attribute_nn_width: int = 64
    attribute_nn_depth: int = 6
    attribute_dropout_rate: float = 0.1
    n_latent_attribute_ordered: int = 512
    n_latent_attribute_categorical: int = 16
    reconstruction_penalty: float = 1000.0
    unknown_attribute_penalty: float = 10000.0
    decoder_width: int = 64
    decoder_depth: int = 1
    decoder_activation: bool = False
    attribute_nn_activation: bool = False
    unknown_attributes: bool = False
    use_batch_norm: bool = False
    use_layer_norm: bool = False
    latent_lr: float = 1e-4
    latent_wd: float = 1e-3
    attribute_nn_lr: float = 1e-3
    attribute_nn_wd: float = 4e-8
    step_size_lr: int = 45
    cosine_scheduler: bool = True
    scheduler_final_lr: float = 1e-5
    decoder_lr: float = 1e-3
    decoder_wd: float = 1e-2
    unknown_attribute_noise_param: float = 0.0
    export_control_pool_size: int = 300


PROFILE_DIR = Path(__file__).resolve().parents[1] / "eval" / "configs"
PROFILE_ALLOWED_KEYS = {"dataset", "task_args"}


DATASET_CONFIG = {
    "adamson": BiolordDatasetConfig(
        full_data_rel="data/adamson/adamson_biolord.h5ad",
        single_data_rel="data/adamson/adamson_single_biolord.h5ad",
        splits=[1, 2, 3, 4, 5],
    ),
    "dixit": BiolordDatasetConfig(
        full_data_rel="data/dixit/dixit_biolord.h5ad",
        single_data_rel="data/dixit/dixit_single_biolord.h5ad",
        splits=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        early_stopping=False,
    ),
    "norman": BiolordDatasetConfig(
        full_data_rel="data/norman/norman_biolord.h5ad",
        single_data_rel="data/norman/norman_single_biolord.h5ad",
        splits=[1, 2, 3, 4, 5],
        ordered_attribute_key="perturbation_neighbors1",
        max_epochs=75,
        early_stopping_patience=200,
        attribute_nn_depth=2,
        attribute_dropout_rate=0.0,
        n_latent_attribute_ordered=32,
        reconstruction_penalty=10000.0,
        decoder_width=32,
        decoder_depth=2,
        decoder_activation=True,
        attribute_nn_activation=True,
        latent_lr=1e-2,
        latent_wd=1e-5,
        decoder_lr=1e-2,
        unknown_attribute_noise_param=0.2,
    ),
    "openproblems_donor": BiolordDatasetConfig(
        full_data_rel="data/openproblems/perturb_processed_hvg5000.h5ad",
        splits=[1, 2, 3, 4, 5],
        ordered_attribute_key="rdkit2d",
        legacy_multihot_fallback=True,
        test_ratio=0.2,
        split_policy="donor_unseen_drug",
        domain_key="donor_id",
        train_domain_values=("donor_0", "donor_1"),
        test_domain_values=("donor_2",),
        drug_test_ratio=0.2,
        val_ratio=0.1,
        valid_split="val",
        test_split="test",
        aggregate_legacy_condition_means=False,
    ),
    "openproblems_celltype": BiolordDatasetConfig(
        full_data_rel="data/openproblems/perturb_processed_hvg5000.h5ad",
        splits=[1, 2, 3, 4, 5],
        ordered_attribute_key="rdkit2d",
        legacy_multihot_fallback=True,
        test_ratio=0.2,
        split_policy="celltype_unseen_drug",
        domain_key="cell_type",
        domain_test_ratio=0.2,
        drug_test_ratio=0.2,
        val_ratio=0.1,
        valid_split="val",
        test_split="test",
        aggregate_legacy_condition_means=False,
    ),
}


def _load_profile(profile: str) -> dict:
    p = PROFILE_DIR / f"{profile}.yaml"
    if not p.exists():
        raise FileNotFoundError(f"biolord profile not found: {p}")
    obj = load_yaml_file(p)
    unknown = sorted(set(obj.keys()) - PROFILE_ALLOWED_KEYS)
    if unknown:
        raise ValueError(f"Unknown keys in biolord profile {p.name}: {unknown}")
    dataset = str(obj.get("dataset", "")).strip()
    if dataset != profile:
        raise ValueError(f"Profile dataset mismatch: file={p.name} dataset={dataset!r}")
    task_args = obj.get("task_args") or {}
    if not isinstance(task_args, dict):
        raise TypeError(f"profile.task_args must be a mapping: {p}")
    return {"dataset": dataset, "task_args": task_args}


def _resolve_mean_metric_keys(numeric_means: pd.Series) -> list[str]:
    preferred_order = [
        "pearson",
        "nmse",
        "mse_pred",
        "mse_ctrl",
        "deg_mean_r2",
        "systema_corr_20de_allpert",
        "systema_corr_deg_r2",
        "systema_corr_20de_allpert_dist",
        "systema_corr_deg_r2_dist",
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


def _export_training_history_artifacts(model, out_dir: Path, dataset_name: str, split_id: int) -> None:
    history_obj = getattr(getattr(model, "training_plan", None), "epoch_history", None)
    if not isinstance(history_obj, dict) or not history_obj:
        return

    history_df = pd.DataFrame.from_dict(history_obj)
    if history_df.empty:
        return

    history_csv = out_dir / f"biolord_{dataset_name}_epoch_hist_{int(split_id)}.csv"
    history_df.to_csv(history_csv, index=False)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - environment-dependent
        warnings.warn(
            f"[biolord] matplotlib unavailable; skipped loss plot export for {dataset_name} split={split_id}: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return

    plot_columns = [
        "reconstruction_loss",
        "unknown_attribute_penalty_loss",
        "biolord_metric",
        "generative_mean_accuracy",
        "generative_var_accuracy",
    ]
    available = [col for col in plot_columns if col in history_df.columns]
    if not available or "mode" not in history_df.columns or "epoch" not in history_df.columns:
        return

    fig, axes = plt.subplots(len(available), 1, figsize=(8, 2.4 * len(available)), sharex=True)
    if len(available) == 1:
        axes = [axes]

    for ax, col in zip(axes, available):
        for mode, color in [("train", "#1f77b4"), ("valid", "#d62728"), ("test", "#2ca02c")]:
            sub = history_df[history_df["mode"].astype(str) == mode]
            if sub.empty:
                continue
            ax.plot(sub["epoch"], sub[col], label=mode, linewidth=1.8, color=color, alpha=0.95)
        ax.set_ylabel(col)
        ax.grid(alpha=0.25, linewidth=0.6)
        ax.legend(frameon=False, fontsize=8, loc="best")

    axes[-1].set_xlabel("epoch")
    fig.suptitle(f"BioLORD {dataset_name} split {int(split_id)} training history", fontsize=12)
    fig.tight_layout()
    plot_path = out_dir / f"biolord_{dataset_name}_loss_{int(split_id)}.png"
    fig.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def set_seeds(seed: int = 24) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _dummy_embedding_df() -> pd.DataFrame:
    return pd.DataFrame([[0.0]], index=["ctrl"], columns=["dummy"])


def _resolve_dataset_path(name: str, data_rel: str) -> Path:
    primary = (LOCAL_DATA_ROOT / Path(data_rel).relative_to("data")).resolve()
    if primary.exists():
        return primary
    paths_cfg_path = ROOT / "configs" / "paths.yaml"
    if paths_cfg_path.exists():
        paths_cfg = _utils.load_yaml(str(paths_cfg_path))
        datasets_map = paths_cfg.get("datasets", {})
        if name in datasets_map:
            alt = Path(datasets_map[name]).resolve()
            if alt.exists():
                return alt
    raise FileNotFoundError(
        f"Missing BioLORD input for dataset={name}. Tried {primary}"
    )


def _resolve_eval_data_paths(name: str, cfg: BiolordDatasetConfig) -> tuple[Path, Path | None]:
    full_path = _resolve_dataset_path(name, cfg.full_data_rel)
    single_path = _resolve_dataset_path(name, cfg.single_data_rel) if str(cfg.single_data_rel).strip() else None
    return full_path, single_path


def _prepare_full_eval_adata(data_path: Path) -> tuple[ad.AnnData, TriShiftData]:
    adata = _utils.load_adata(str(data_path))
    data = TriShiftData(adata, _dummy_embedding_df())
    data.build_or_load_degs()
    return data.adata_all, data


def _prepare_single_train_adata(data_path: Path, ordered_attribute_key: str) -> ad.AnnData:
    adata = _utils.load_adata(str(data_path))
    if ordered_attribute_key not in adata.obsm:
        raise KeyError(
            f"Missing ordered attribute key '{ordered_attribute_key}' in BioLORD single adata: {data_path}"
        )
    return adata


def _build_split_dict_legacy(data: TriShiftData, split_id: int, test_ratio: float) -> dict:
    return data.split_by_condition(seed=int(split_id), test_ratio=float(test_ratio))


def _split_conditions_for_holdout(
    conditions: list[str],
    *,
    ratio: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    conds = np.array([str(c) for c in conditions], dtype=object)
    rng = np.random.RandomState(int(seed))
    rng.shuffle(conds)
    n_test = round(float(ratio) * len(conds))
    test_arr, train_arr = np.split(conds, [n_test])
    return list(test_arr), list(train_arr)


def _domain_values(adata: ad.AnnData, domain_key: str) -> list[str]:
    if domain_key not in adata.obs.columns:
        raise ValueError(f"adata.obs is missing required split domain column: {domain_key}")
    return sorted(adata.obs[domain_key].astype(str).unique().tolist())


def _condition_values(data: TriShiftData) -> list[str]:
    return [str(c) for c in data.conditions_pert if str(c) != data.ctrl_label]


def _ensure_domain_test_perturbation(
    *,
    adata: ad.AnnData,
    label_key: str,
    ctrl_label: str,
    domain_key: str,
    test_domain_values: list[str],
    test_conds: list[str],
    seed: int,
) -> list[str]:
    cond_series = adata.obs[label_key].astype(str)
    domain_series = adata.obs[domain_key].astype(str)
    test_domain_set = {str(x) for x in test_domain_values}
    test_cond_set = {str(x) for x in test_conds}
    domain_mask = domain_series.isin(test_domain_set)
    domain_conds = sorted(
        c
        for c in cond_series[domain_mask].unique().tolist()
        if str(c) != ctrl_label
    )
    if test_cond_set.intersection(domain_conds):
        return list(test_conds)
    candidates = [c for c in domain_conds if c not in test_cond_set]
    if not candidates:
        raise ValueError(
            f"No non-control conditions available in held-out {domain_key}={test_domain_values}"
        )
    replacement = candidates[(int(seed) - 1) % len(candidates)]
    out = list(test_conds)
    if not out:
        return [replacement]
    out[-1] = replacement
    return out


def _split_domain_unseen_drug(
    data: TriShiftData,
    *,
    seed: int,
    domain_key: str,
    train_domain_values: list[str],
    test_domain_values: list[str],
    drug_test_ratio: float,
    val_ratio: float,
    ensure_test_perturbation: bool,
) -> dict:
    adata = data.adata_all
    label_key = data.label_key
    ctrl_label = data.ctrl_label
    if domain_key not in adata.obs.columns:
        raise ValueError(f"adata.obs is missing required split domain column: {domain_key}")

    all_conds = _condition_values(data)
    test_conds, remaining_conds = _split_conditions_for_holdout(
        all_conds,
        ratio=float(drug_test_ratio),
        seed=int(seed),
    )
    if ensure_test_perturbation:
        test_conds = _ensure_domain_test_perturbation(
            adata=adata,
            label_key=label_key,
            ctrl_label=ctrl_label,
            domain_key=domain_key,
            test_domain_values=test_domain_values,
            test_conds=test_conds,
            seed=int(seed),
        )
        remaining_conds = [c for c in all_conds if c not in set(test_conds)]

    val_conds, train_conds = _split_conditions_for_holdout(
        remaining_conds,
        ratio=float(val_ratio),
        seed=int(seed),
    )

    cond_series = adata.obs[label_key].astype(str)
    domain_series = adata.obs[domain_key].astype(str)
    train_domain_set = {str(x) for x in train_domain_values}
    test_domain_set = {str(x) for x in test_domain_values}

    train_mask = domain_series.isin(train_domain_set) & cond_series.isin(
        list(train_conds) + [ctrl_label]
    )
    val_mask = domain_series.isin(train_domain_set) & cond_series.isin(
        list(val_conds) + [ctrl_label]
    )
    test_mask = domain_series.isin(test_domain_set) & cond_series.isin(
        list(test_conds) + [ctrl_label]
    )

    if not np.any(train_mask.values):
        raise ValueError("domain unseen-drug split produced empty train set")
    if not np.any(val_mask.values):
        raise ValueError("domain unseen-drug split produced empty validation set")
    if not np.any(test_mask.values):
        raise ValueError("domain unseen-drug split produced empty test set")
    test_pert_mask = test_mask.values & (cond_series.values != ctrl_label)
    if not np.any(test_pert_mask):
        raise ValueError("domain unseen-drug split produced no test perturbation cells")

    return {
        "train": adata[train_mask.values],
        "val": adata[val_mask.values],
        "test": adata[test_mask.values],
        "train_conds": [str(c) for c in train_conds],
        "val_conds": [str(c) for c in val_conds],
        "test_conds": [str(c) for c in test_conds],
        "split_policy": "domain_unseen_drug",
        "split_domain_key": str(domain_key),
        "train_domain_values": [str(x) for x in train_domain_values],
        "test_domain_values": [str(x) for x in test_domain_values],
    }


def _build_split_dict_for_config(data: TriShiftData, cfg: BiolordDatasetConfig, split_id: int) -> dict:
    policy = str(cfg.split_policy or "condition").strip()
    if policy in {"", "condition"}:
        return _build_split_dict_legacy(data, int(split_id), float(cfg.test_ratio or 0.2))
    if policy == "donor_unseen_drug":
        return _split_domain_unseen_drug(
            data,
            seed=int(split_id),
            domain_key=str(cfg.domain_key or "donor_id"),
            train_domain_values=[str(x) for x in cfg.train_domain_values],
            test_domain_values=[str(x) for x in cfg.test_domain_values],
            drug_test_ratio=float(cfg.drug_test_ratio or cfg.test_ratio or 0.2),
            val_ratio=float(cfg.val_ratio),
            ensure_test_perturbation=True,
        )
    if policy == "celltype_unseen_drug":
        domain_key = str(cfg.domain_key or "cell_type")
        test_domains, train_domains = _split_conditions_for_holdout(
            _domain_values(data.adata_all, domain_key),
            ratio=float(cfg.domain_test_ratio),
            seed=int(split_id),
        )
        return _split_domain_unseen_drug(
            data,
            seed=int(split_id),
            domain_key=domain_key,
            train_domain_values=train_domains,
            test_domain_values=test_domains,
            drug_test_ratio=float(cfg.drug_test_ratio or cfg.test_ratio or 0.2),
            val_ratio=float(cfg.val_ratio),
            ensure_test_perturbation=True,
        )
    raise ValueError(f"Unsupported BioLORD split_policy={policy!r}")


def _assign_split_labels_legacy(adata: ad.AnnData, split_dict: dict, split_key: str) -> ad.AnnData:
    out = adata.copy()
    cond_series = out.obs["condition"].astype(str)
    split_vals = np.full(out.n_obs, "ignore", dtype=object)
    split_vals[cond_series.eq("ctrl").values] = "train"
    split_vals[cond_series.isin(list(map(str, split_dict.get("train_conds", [])))).values] = "train"
    split_vals[cond_series.isin(list(map(str, split_dict.get("val_conds", [])))).values] = "val"
    split_vals[cond_series.isin(list(map(str, split_dict.get("test_conds", [])))).values] = "test"
    out.obs[split_key] = pd.Categorical(split_vals, categories=["train", "val", "test", "ignore"])
    keep_mask = out.obs[split_key].astype(str).isin(["train", "val", "test"]).values
    return out[keep_mask].copy()


def _assign_split_labels_from_split_dict(adata: ad.AnnData, split_dict: dict, split_key: str) -> ad.AnnData:
    out = adata.copy()
    split_vals = np.full(out.n_obs, "ignore", dtype=object)
    obs_names = out.obs_names.astype(str)
    for label in ("val", "test", "train"):
        split_adata = split_dict.get(label)
        if isinstance(split_adata, ad.AnnData):
            split_vals[obs_names.isin(split_adata.obs_names.astype(str))] = label
    out.obs[split_key] = pd.Categorical(split_vals, categories=["train", "val", "test", "ignore"])
    keep_mask = out.obs[split_key].astype(str).isin(["train", "val", "test"]).values
    return out[keep_mask].copy()


def _build_multihot_attributes(adata: ad.AnnData, key: str) -> ad.AnnData:
    out = adata.copy()
    conds = out.obs["condition"].astype(str).tolist()
    pert_basis = sorted(
        {
            token
            for cond in conds
            for token in _condition_tokens_no_ctrl(_utils.normalize_condition(cond))
        }
    )
    if not pert_basis:
        raise ValueError("No perturbation tokens found for BioLORD multihot attribute construction")
    token_to_idx = {tok: i for i, tok in enumerate(pert_basis)}
    emb = np.zeros((out.n_obs, len(pert_basis)), dtype=np.float32)
    for row_idx, cond in enumerate(conds):
        for tok in _condition_tokens_no_ctrl(_utils.normalize_condition(cond)):
            idx = token_to_idx.get(tok)
            if idx is not None:
                emb[row_idx, idx] = 1.0
    out.obsm[key] = emb
    return out


def _as_mean_vector(x) -> np.ndarray:
    return np.asarray(x.mean(axis=0)).reshape(-1).astype(np.float32)


def _aggregate_legacy_condition_means(
    adata: ad.AnnData,
    *,
    split_key: str,
    cfg: BiolordDatasetConfig,
) -> ad.AnnData:
    group_cols = [split_key]
    if cfg.domain_key and cfg.domain_key in adata.obs.columns:
        group_cols.append(str(cfg.domain_key))
    group_cols.append("condition")

    obs = adata.obs.copy()
    grouped = obs.groupby(group_cols, sort=False, observed=True).indices
    if not grouped:
        raise ValueError("Cannot aggregate empty BioLORD training AnnData")

    x_rows: list[np.ndarray] = []
    attr_rows: list[np.ndarray] = []
    obs_rows: list[dict] = []
    attr = np.asarray(adata.obsm[cfg.ordered_attribute_key], dtype=np.float32)
    optional_first_cols = [
        col
        for col in ["condition_name", "SMILES", "sm_name", "donor_id", "cell_type"]
        if col in obs.columns and col not in group_cols
    ]

    for group_key, idx in grouped.items():
        idx_arr = np.asarray(idx, dtype=int)
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        row = {col: str(val) for col, val in zip(group_cols, group_key)}
        for col in optional_first_cols:
            row[col] = str(obs.iloc[idx_arr[0]][col])
        row["n_cells_aggregated"] = int(idx_arr.size)
        obs_rows.append(row)
        x_rows.append(_as_mean_vector(adata.X[idx_arr]))
        attr_rows.append(attr[idx_arr].mean(axis=0).astype(np.float32))

    obs_df = pd.DataFrame(obs_rows)
    obs_names = [
        "|".join(str(row[col]) for col in group_cols)
        for _, row in obs_df.iterrows()
    ]
    obs_df.index = pd.Index(obs_names, dtype="object")
    out = ad.AnnData(
        X=np.vstack(x_rows).astype(np.float32),
        obs=obs_df,
        var=adata.var.copy(),
        dtype=np.float32,
    )
    out.obs[split_key] = pd.Categorical(
        out.obs[split_key].astype(str),
        categories=["train", "val", "test", "ignore"],
    )
    out.obs["condition"] = out.obs["condition"].astype(str).astype("category")
    if cfg.domain_key and cfg.domain_key in out.obs.columns:
        out.obs[str(cfg.domain_key)] = out.obs[str(cfg.domain_key)].astype(str).astype("category")
    out.obsm[cfg.ordered_attribute_key] = np.vstack(attr_rows).astype(np.float32)
    return out


def _prepare_legacy_train_adata(
    *,
    eval_adata: ad.AnnData,
    split_dict: dict,
    split_id: int,
    cfg: BiolordDatasetConfig,
) -> tuple[ad.AnnData, str]:
    split_key = f"split{int(split_id)}"
    if str(cfg.split_policy or "condition").strip() in {"", "condition"}:
        split_adata = _assign_split_labels_legacy(eval_adata, split_dict, split_key)
    else:
        split_adata = _assign_split_labels_from_split_dict(eval_adata, split_dict, split_key)

    if cfg.ordered_attribute_key in split_adata.obsm:
        split_adata.obsm[cfg.ordered_attribute_key] = np.asarray(
            split_adata.obsm[cfg.ordered_attribute_key],
            dtype=np.float32,
        )
    else:
        if str(cfg.ordered_attribute_key) == "rdkit2d":
            raise KeyError("OpenProblems BioLORD requires adata.obsm['rdkit2d']")
        split_adata = _build_multihot_attributes(split_adata, cfg.ordered_attribute_key)
    if bool(cfg.aggregate_legacy_condition_means):
        split_adata = _aggregate_legacy_condition_means(
            split_adata,
            split_key=split_key,
            cfg=cfg,
        )
    return split_adata, split_key


def _gene_names_from_adata(adata: ad.AnnData) -> np.ndarray:
    for col in ["gene_name", "gene_names"]:
        if col in adata.var.columns:
            return adata.var[col].astype(str).values
    return adata.var_names.astype(str).values


def _condition_tokens_no_ctrl(condition: str) -> list[str]:
    return [tok for tok in str(condition).split("+") if tok and tok != "ctrl"]


def _build_name_map(adata: ad.AnnData) -> dict[str, str]:
    out: dict[str, str] = {}
    if "condition_name" in adata.obs.columns:
        pairs = adata.obs[["condition", "condition_name"]].drop_duplicates().values
        for cond, cname in pairs:
            out[_utils.normalize_condition(str(cond))] = str(cname)
    for cond in adata.obs["condition"].astype(str).unique():
        out.setdefault(_utils.normalize_condition(cond), cond)
    return out


def _require_split_columns(adata: ad.AnnData, split_id: int) -> tuple[str, str]:
    split_key = f"split{int(split_id)}"
    subgroup_key = f"subgroup{int(split_id)}"
    missing = [key for key in [split_key, subgroup_key] if key not in adata.obs.columns]
    if missing:
        raise KeyError(
            f"BioLORD preprocessed adata is missing split metadata columns: {missing}. "
            "Run scripts/data/prepare_biolord_perturbation_data.py first."
        )
    return split_key, subgroup_key


def _unique_conditions_for_split(adata: ad.AnnData, split_key: str, split_value: str) -> list[str]:
    mask = adata.obs[split_key].astype(str).eq(split_value).values
    conds = adata.obs.loc[mask, "condition"].astype(str).tolist()
    return sorted({_utils.normalize_condition(cond) for cond in conds if str(cond) != "ctrl"})


def _map_subgroup_label(raw: str) -> str:
    mapping = {
        "unseen_single": "single",
        "combo_seen0": "seen0",
        "combo_seen1": "seen1",
        "combo_seen2": "seen2",
    }
    return mapping.get(str(raw), "unknown")


def _subgroup_frame_from_full_adata(adata: ad.AnnData, split_id: int) -> pd.DataFrame:
    split_key, subgroup_key = _require_split_columns(adata, split_id)
    obs = adata.obs[["condition", split_key, subgroup_key]].copy()
    obs["condition"] = obs["condition"].astype(str).map(_utils.normalize_condition)
    obs = obs[obs[split_key].astype(str) == "ood"]
    if obs.empty:
        return pd.DataFrame(columns=["subgroup"])
    subgroup_map = (
        obs[["condition", subgroup_key]]
        .drop_duplicates()
        .assign(subgroup=lambda df: df[subgroup_key].astype(str).map(_map_subgroup_label))
        .drop(columns=[subgroup_key])
        .set_index("condition")
    )
    return subgroup_map


def _require_biolord_stack():
    # scvi-tools 0.20 / jax expects scipy.linalg.tril|triu, which are missing in
    # newer SciPy builds available on Python 3.12 wheels.
    if not hasattr(scipy.linalg, "tril"):
        scipy.linalg.tril = np.tril  # type: ignore[attr-defined]
    if not hasattr(scipy.linalg, "triu"):
        scipy.linalg.triu = np.triu  # type: ignore[attr-defined]
    try:
        import biolord  # type: ignore
    except Exception as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError(
            "biolord is not installed in the current environment. Run this pipeline in the BioLORD environment."
        ) from exc

    if not getattr(biolord, "_trishift_runtime_patched", False):
        import pytorch_lightning as pl
        from scvi.dataloaders import AnnDataLoader
        from scvi.train import TrainRunner as ScviTrainRunner

        class CompatAnnDataSplitter(pl.LightningDataModule):
            def __init__(
                self,
                adata_manager,
                train_indices,
                valid_indices,
                test_indices,
                pin_memory: bool = False,
                **kwargs,
            ):
                super().__init__()
                self.adata_manager = adata_manager
                self.train_idx = train_indices
                self.val_idx = valid_indices
                self.test_idx = test_indices
                self.pin_memory = pin_memory
                self.data_loader_kwargs = kwargs
                self.n_train = len(train_indices) if train_indices is not None else 0
                self.n_val = len(valid_indices) if valid_indices is not None else 0

            def setup(self, stage=None):
                return

            def train_dataloader(self):
                return AnnDataLoader(
                    self.adata_manager,
                    indices=self.train_idx,
                    shuffle=True,
                    drop_last=False,
                    pin_memory=self.pin_memory,
                    **self.data_loader_kwargs,
                )

            def val_dataloader(self):
                if self.val_idx is None or len(self.val_idx) == 0:
                    return None
                return AnnDataLoader(
                    self.adata_manager,
                    indices=self.val_idx,
                    shuffle=False,
                    drop_last=False,
                    pin_memory=self.pin_memory,
                    **self.data_loader_kwargs,
                )

            def test_dataloader(self):
                if self.test_idx is None or len(self.test_idx) == 0:
                    return None
                return AnnDataLoader(
                    self.adata_manager,
                    indices=self.test_idx,
                    shuffle=False,
                    drop_last=False,
                    pin_memory=self.pin_memory,
                    **self.data_loader_kwargs,
                )

        class CompatTrainRunner(ScviTrainRunner):
            def __init__(self, *args, use_gpu=None, **trainer_kwargs):
                if isinstance(use_gpu, str) and use_gpu.lower() == "auto":
                    use_gpu = None
                trainer_kwargs.pop("checkpointing_monitor", None)
                base_params = inspect.signature(ScviTrainRunner.__init__).parameters
                if "use_gpu" in base_params:
                    super().__init__(*args, use_gpu=use_gpu, **trainer_kwargs)
                    return

                accelerator = trainer_kwargs.pop("accelerator", None)
                devices = trainer_kwargs.pop("devices", None)
                if accelerator is None and devices is None:
                    if use_gpu is False:
                        accelerator = "cpu"
                        devices = "auto"
                    elif use_gpu is None or use_gpu is True:
                        accelerator = "auto"
                        devices = "auto"
                    elif isinstance(use_gpu, int):
                        accelerator = "gpu"
                        devices = [int(use_gpu)]
                    elif isinstance(use_gpu, str):
                        lowered = use_gpu.lower()
                        if lowered.startswith("cuda"):
                            accelerator = "gpu"
                            try:
                                devices = [int(lowered.split(":")[1])]
                            except Exception:
                                devices = 1
                        else:
                            accelerator = "auto"
                            devices = "auto"
                    else:
                        accelerator = "auto"
                        devices = "auto"

                super().__init__(
                    *args,
                    accelerator=accelerator,
                    devices=devices,
                    **trainer_kwargs,
                )

            def __call__(self):
                if hasattr(self.data_splitter, "n_train"):
                    self.training_plan.n_obs_training = self.data_splitter.n_train
                if hasattr(self.data_splitter, "n_val"):
                    self.training_plan.n_obs_validation = self.data_splitter.n_val

                train_dl = self.data_splitter.train_dataloader()
                val_dl = self.data_splitter.val_dataloader()
                self.trainer.fit(
                    self.training_plan,
                    train_dataloaders=train_dl,
                    val_dataloaders=val_dl,
                )
                self._update_history()

                self.model.train_indices = getattr(self.data_splitter, "train_idx", None)
                self.model.test_indices = getattr(self.data_splitter, "test_idx", None)
                self.model.validation_indices = getattr(self.data_splitter, "val_idx", None)

                self.model.module.eval()
                self.model.is_trained_ = True
                self.model.to_device(self.device)
                self.model.trainer = self.trainer

        biolord._data.AnnDataSplitter = CompatAnnDataSplitter
        biolord._model.AnnDataSplitter = CompatAnnDataSplitter
        biolord._model.TrainRunner = CompatTrainRunner
        biolord._trishift_runtime_patched = True
    return biolord


def _build_module_params(cfg: BiolordDatasetConfig, seed: int) -> dict:
    return {
        "attribute_nn_width": int(cfg.attribute_nn_width),
        "attribute_nn_depth": int(cfg.attribute_nn_depth),
        "use_batch_norm": bool(cfg.use_batch_norm),
        "use_layer_norm": bool(cfg.use_layer_norm),
        "attribute_dropout_rate": float(cfg.attribute_dropout_rate),
        "unknown_attribute_noise_param": float(cfg.unknown_attribute_noise_param),
        "seed": int(seed),
        "n_latent_attribute_ordered": int(cfg.n_latent_attribute_ordered),
        "n_latent_attribute_categorical": int(cfg.n_latent_attribute_categorical),
        "reconstruction_penalty": float(cfg.reconstruction_penalty),
        "unknown_attribute_penalty": float(cfg.unknown_attribute_penalty),
        "decoder_width": int(cfg.decoder_width),
        "decoder_depth": int(cfg.decoder_depth),
        "decoder_activation": bool(cfg.decoder_activation),
        "attribute_nn_activation": bool(cfg.attribute_nn_activation),
        "unknown_attributes": bool(cfg.unknown_attributes),
    }


def _build_trainer_params(cfg: BiolordDatasetConfig) -> dict:
    return {
        "n_epochs_warmup": 0,
        "latent_lr": float(cfg.latent_lr),
        "latent_wd": float(cfg.latent_wd),
        "attribute_nn_lr": float(cfg.attribute_nn_lr),
        "attribute_nn_wd": float(cfg.attribute_nn_wd),
        "step_size_lr": int(cfg.step_size_lr),
        "cosine_scheduler": bool(cfg.cosine_scheduler),
        "scheduler_final_lr": float(cfg.scheduler_final_lr),
        "decoder_lr": float(cfg.decoder_lr),
        "decoder_wd": float(cfg.decoder_wd),
    }


def _build_model_and_train(
    *,
    biolord_mod,
    train_adata: ad.AnnData,
    split_key: str,
    cfg: BiolordDatasetConfig,
    seed: int,
):
    biolord_mod.Biolord.setup_anndata(
        train_adata,
        ordered_attributes_keys=[cfg.ordered_attribute_key],
        categorical_attributes_keys=None,
        retrieval_attribute_key=None,
    )
    model = biolord_mod.Biolord(
        adata=train_adata,
        n_latent=int(cfg.n_latent),
        model_name="biolord",
        module_params=_build_module_params(cfg, seed),
        train_classifiers=False,
        split_key=split_key,
        train_split="train",
        valid_split=str(cfg.valid_split),
        test_split=str(cfg.test_split),
    )
    model.train(
        max_epochs=int(cfg.max_epochs),
        batch_size=int(cfg.batch_size),
        plan_kwargs=_build_trainer_params(cfg),
        early_stopping=bool(cfg.early_stopping),
        early_stopping_patience=int(cfg.early_stopping_patience),
        check_val_every_n_epoch=int(cfg.check_val_every_n_epoch),
        num_workers=0,
        enable_checkpointing=False,
    )
    return model


def _get_condition_attribute_tensor(model, adata: ad.AnnData, condition: str, ordered_attribute_key: str) -> torch.Tensor:
    cond_mask = (
        adata.obs["condition"].astype(str).map(_utils.normalize_condition).eq(_utils.normalize_condition(condition)).values
    )
    if not np.any(cond_mask):
        raise KeyError(f"Condition missing in biolord adata: {condition}")
    cond_adata = adata[cond_mask][:1].copy()
    dataset_ref = model.get_dataset(cond_adata)
    return dataset_ref[ordered_attribute_key][0, :]


def _repeat_n(x: torch.Tensor, n: int) -> torch.Tensor:
    return x.to(x.device).view(1, -1).repeat(int(n), 1)


def _stable_condition_seed(base_seed: int, split_id: int, condition: str) -> int:
    raw = f"{int(base_seed)}::{int(split_id)}::{condition}"
    return int(sum((i + 1) * ord(ch) for i, ch in enumerate(raw)) % (2**32 - 1))


def _array_summary(x: np.ndarray) -> dict:
    arr = np.asarray(x, dtype=np.float32)
    return {
        "n_cells": int(arr.shape[0]) if arr.ndim == 2 else 0,
        "mean": arr.mean(axis=0).astype(np.float32) if arr.ndim == 2 and arr.shape[0] > 0 else np.asarray([], dtype=np.float32),
        "var": arr.var(axis=0, ddof=1).astype(np.float32)
        if arr.ndim == 2 and arr.shape[0] > 1
        else (np.zeros(arr.shape[1], dtype=np.float32) if arr.ndim == 2 else np.asarray([], dtype=np.float32)),
    }


def _sample_rows_for_export(x: np.ndarray, sample_size: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(x, dtype=np.float32)
    n_rows = int(arr.shape[0]) if arr.ndim == 2 else 0
    if n_rows <= 0:
        return arr[:0], np.asarray([], dtype=int)
    sample_n = min(n_rows, max(1, int(sample_size)))
    if sample_n < n_rows:
        rng = np.random.default_rng(int(seed))
        idx = np.sort(rng.choice(n_rows, size=sample_n, replace=False))
    else:
        idx = np.arange(n_rows, dtype=int)
    return np.asarray(arr[idx], dtype=np.float32), idx.astype(int)


def _top20_degs_mapping(adata: ad.AnnData) -> dict:
    for key in (
        "top20_degs_final",
        "top20_degs_non_dropout",
        "top_non_dropout_de_20",
        "top_non_zero_de_20",
        "top20_degs",
    ):
        obj = adata.uns.get(key, None)
        if isinstance(obj, dict):
            return obj
    raise TypeError("Expected a top-20 DEG mapping in adata.uns")


def _slice_dataset_batch(dataset: dict, start: int, end: int, n_obs: int) -> dict:
    out = {}
    for key, value in dataset.items():
        if torch.is_tensor(value) and value.ndim > 0 and int(value.shape[0]) == int(n_obs):
            out[key] = value[start:end]
        elif isinstance(value, np.ndarray) and value.ndim > 0 and int(value.shape[0]) == int(n_obs):
            out[key] = value[start:end]
        else:
            out[key] = value
    return out


def _predict_biolord_expr_in_batches(
    *,
    model,
    dataset_control: dict,
    attr_tensor: torch.Tensor,
    ordered_attribute_key: str,
    n_obs: int,
    batch_size: int = 4096,
) -> np.ndarray:
    pieces = []
    step = max(1, int(batch_size))
    with torch.no_grad():
        for start in range(0, int(n_obs), step):
            end = min(start + step, int(n_obs))
            dataset_pred = _slice_dataset_batch(dataset_control, start, end, int(n_obs))
            dataset_pred[ordered_attribute_key] = _repeat_n(attr_tensor, end - start)
            pred_expr, _ = model.module.get_expression(dataset_pred)
            pieces.append(np.asarray(pred_expr.detach().cpu().numpy(), dtype=np.float32))
    if not pieces:
        return np.zeros((0, 0), dtype=np.float32)
    return np.vstack(pieces).astype(np.float32, copy=False)


def _condition_metric_summary(
    *,
    X_true: np.ndarray,
    X_pred: np.ndarray,
    X_ctrl: np.ndarray,
    degs: np.ndarray,
    condition: str,
    split_dict: dict,
    split_id: int,
    base_seed: int,
    systema_reference: np.ndarray,
    eval_ctrl_source: str,
) -> tuple[dict, dict]:
    metric_seed = _stable_condition_seed(base_seed, split_id, condition)
    mean_metrics = compute_mean_effect_metrics(
        X_true=X_true,
        X_pred=X_pred,
        X_ctrl=X_ctrl,
        deg_idx=degs,
    )
    systema_metrics = pearson_delta_reference_metrics(
        X_true=np.asarray(X_true, dtype=np.float32).mean(axis=0),
        X_pred=np.asarray(X_pred, dtype=np.float32).mean(axis=0),
        reference=systema_reference,
        top20_de_idxs=degs,
    )
    scpram_bundle = compute_scpram_metrics_bundle_from_arrays(
        X_true=X_true,
        X_pred=X_pred,
        deg_idx=degs,
        n_degs=100,
        sample_ratio=0.8,
        times=100,
        seed=metric_seed,
    )
    dist_bundle = compute_distributional_systema_metrics_from_arrays(
        X_true=X_true,
        X_pred=X_pred,
        reference=systema_reference,
        deg_idx=degs,
        sample_ratio=0.8,
        times=100,
        seed=metric_seed,
    )
    metrics = {
        **mean_metrics,
        "systema_corr_20de_allpert": float(systema_metrics["corr_20de_allpert"]),
        "systema_corr_deg_r2": float(systema_metrics["corr_deg_r2"]),
        **scpram_bundle["metrics"],
        **dist_bundle["metrics"],
    }
    full_summary = {
        "metrics": dict(metrics),
        "true": _array_summary(X_true),
        "pred": _array_summary(X_pred),
        "ctrl": _array_summary(X_ctrl),
        "scpram_repeats": scpram_bundle["repeats"],
        "scpram_wasserstein_degs_by_gene": scpram_bundle["wasserstein_degs_by_gene"],
        "scpram_degs_used": scpram_bundle["degs_used"],
        "systema_distributional_repeats": dist_bundle["repeats"],
        "systema_distributional_degs_used": dist_bundle["degs_used"],
        "metric_seed": int(metric_seed),
        "sample_ratio": 0.8,
        "times": 100,
        "split_policy": split_dict.get("split_policy", None),
        "split_domain_key": split_dict.get("split_domain_key", None),
        "train_domain_values": split_dict.get("train_domain_values", None),
        "test_domain_values": split_dict.get("test_domain_values", None),
        "eval_ctrl_source": str(eval_ctrl_source),
    }
    return metrics, full_summary


def _nan_scpram_metrics() -> dict[str, float]:
    # BioLORD's current adapter produces control-conditioned samples whose
    # variance structure can collapse, so distribution-level scPRAM metrics are
    # not comparable to models with explicit cell-level stochastic predictions.
    return {
        "scpram_r2_degs_mean_mean": np.nan,
        "scpram_r2_degs_var_mean": np.nan,
        "scpram_wasserstein_degs_sum": np.nan,
    }


def _compute_metrics_and_export_payload_legacy(
    *,
    model,
    eval_adata: ad.AnnData,
    train_adata: ad.AnnData,
    split_dict: dict,
    split_id: int,
    cfg: BiolordDatasetConfig,
    base_seed: int = 24,
) -> tuple[pd.DataFrame, dict]:
    name_map = _build_name_map(eval_adata)
    gene_names = _gene_names_from_adata(eval_adata)
    truth_adata = split_dict.get("test") if isinstance(split_dict.get("test"), ad.AnnData) else eval_adata
    cond_truth = truth_adata.obs["condition"].astype(str).map(_utils.normalize_condition)
    split_key = f"split{int(split_id)}"
    use_target_ctrl = str(split_dict.get("split_policy", "")) == "domain_unseen_drug"
    if use_target_ctrl:
        ctrl_mask = truth_adata.obs["condition"].astype(str).eq("ctrl").values
        if not np.any(ctrl_mask):
            raise ValueError("OpenProblems BioLORD target-domain test split has no ctrl cells")
        adata_control = truth_adata[ctrl_mask].copy()
        eval_ctrl_source = "target_domain_test_ctrl"
    else:
        train_ctrl_mask = train_adata.obs["condition"].astype(str).eq("ctrl").values
        if split_key in train_adata.obs.columns:
            train_ctrl_mask = train_ctrl_mask & train_adata.obs[split_key].astype(str).eq("train").values
        if not np.any(train_ctrl_mask):
            train_ctrl_mask = train_adata.obs["condition"].astype(str).eq("ctrl").values
        adata_control = train_adata[train_ctrl_mask].copy()
        eval_ctrl_source = "train_ctrl"
    ctrl_full = np.asarray(_utils.densify_X(adata_control.X), dtype=np.float32)
    top20_degs_final = _top20_degs_mapping(eval_adata)

    train_ref = split_dict.get("train")
    val_ref = split_dict.get("val")
    if isinstance(train_ref, ad.AnnData) and isinstance(val_ref, ad.AnnData):
        val_ref_non_ctrl = val_ref[val_ref.obs["condition"].astype(str) != "ctrl"]
        reference_adata = ad.concat(
            [train_ref, val_ref_non_ctrl],
            join="outer",
            merge="same",
            index_unique=None,
        ).copy()
    else:
        train_val_conds = list(map(str, split_dict.get("train_conds", []))) + list(map(str, split_dict.get("val_conds", [])))
        ref_mask = eval_adata.obs["condition"].astype(str).isin(set(train_val_conds) | {"ctrl"}).values
        reference_adata = eval_adata[ref_mask].copy()
    if use_target_ctrl:
        pert_reference = ctrl_full.mean(axis=0).reshape(-1)
    else:
        pert_reference = average_of_perturbation_centroids(
            X=_utils.densify_X(reference_adata.X),
            conditions=reference_adata.obs["condition"].astype(str).values,
            ctrl_label="ctrl",
        )

    dataset_control = model.get_dataset(adata_control)

    results = []
    export_payload = {}
    for condition in sorted(set(map(str, split_dict.get("test_conds", [])))):
        cond_norm = _utils.normalize_condition(condition)
        cond_mask_truth = cond_truth.eq(cond_norm).values
        if not np.any(cond_mask_truth):
            warnings.warn(f"[biolord] test condition missing in eval adata: {condition}", RuntimeWarning, stacklevel=2)
            continue

        true_expr = _utils.densify_X(truth_adata[cond_mask_truth].X)
        if true_expr.ndim == 1:
            true_expr = true_expr.reshape(1, -1)
        degs = np.asarray(top20_degs_final.get(cond_norm, []), dtype=int).reshape(-1)
        if degs.size == 0:
            warnings.warn(f"[biolord] skip condition without DEGs: {condition}", RuntimeWarning, stacklevel=2)
            continue

        attr_tensor = _get_condition_attribute_tensor(model, train_adata, cond_norm, cfg.ordered_attribute_key)
        pred_expr = _predict_biolord_expr_in_batches(
            model=model,
            dataset_control=dataset_control,
            attr_tensor=attr_tensor,
            ordered_attribute_key=cfg.ordered_attribute_key,
            n_obs=int(adata_control.n_obs),
            batch_size=4096,
        )
        metrics, full_summary = _condition_metric_summary(
            X_true=true_expr,
            X_pred=pred_expr,
            X_ctrl=ctrl_full,
            degs=degs,
            condition=cond_norm,
            split_dict=split_dict,
            split_id=split_id,
            base_seed=base_seed,
            systema_reference=pert_reference,
            eval_ctrl_source=eval_ctrl_source,
        )

        results.append(
            {
                "condition": cond_norm,
                "condition_name": name_map.get(cond_norm, cond_norm),
                **metrics,
                "split_id": int(split_id),
                "n_ensemble": int(pred_expr.shape[0]),
                "n_eval_ctrl": int(ctrl_full.shape[0]),
                "eval_ctrl_source": str(eval_ctrl_source),
            }
        )
        export_sample_size = max(1, int(cfg.export_control_pool_size))
        seed_base = _stable_condition_seed(base_seed, split_id, str(cond_norm))
        pred_export, pred_idx = _sample_rows_for_export(pred_expr, export_sample_size, seed_base + 11)
        if int(ctrl_full.shape[0]) == int(pred_expr.shape[0]) and pred_idx.size > 0:
            ctrl_export = np.asarray(ctrl_full[pred_idx], dtype=np.float32)
            ctrl_idx = pred_idx
        else:
            ctrl_export, ctrl_idx = _sample_rows_for_export(ctrl_full, export_sample_size, seed_base + 17)
        truth_export, truth_idx = _sample_rows_for_export(true_expr, export_sample_size, seed_base + 23)
        export_payload[cond_norm] = {
            "Pred": pred_export[:, degs],
            "Ctrl": ctrl_export[:, degs],
            "Truth": truth_export[:, degs],
            "Pred_full": pred_export,
            "Ctrl_full": ctrl_export,
            "Truth_full": truth_export,
            "DE_idx": degs,
            "DE_name": gene_names[degs] if degs.size > 0 else np.array([], dtype=gene_names.dtype),
            "gene_name_full": gene_names,
            "export_metadata": {
                "export_is_subset": True,
                "export_sample_size": int(export_sample_size),
                "metrics_computed_on_full": True,
                "eval_ctrl_source": str(eval_ctrl_source),
                "split_id": int(split_id),
                "split_policy": split_dict.get("split_policy", None),
                "split_domain_key": split_dict.get("split_domain_key", None),
                "train_domain_values": split_dict.get("train_domain_values", None),
                "test_domain_values": split_dict.get("test_domain_values", None),
                "pred_sample_idx": pred_idx.astype(int),
                "ctrl_sample_idx": ctrl_idx.astype(int),
                "truth_sample_idx": truth_idx.astype(int),
                "n_pred_full": int(pred_expr.shape[0]),
                "n_ctrl_full": int(ctrl_full.shape[0]),
                "n_truth_full": int(true_expr.shape[0]),
            },
            "full_summary": full_summary,
        }
    return pd.DataFrame(results), export_payload


def _condition_mean_lookup(adata: ad.AnnData) -> dict[str, np.ndarray]:
    dense_x = np.asarray(_utils.densify_X(adata.X), dtype=np.float32)
    conds = adata.obs["condition"].astype(str).map(_utils.normalize_condition).values
    df = pd.DataFrame(dense_x, columns=adata.var_names)
    df["condition"] = conds
    grouped = df.groupby("condition", observed=False).mean()
    return {str(idx): grouped.loc[idx].to_numpy(dtype=np.float32).reshape(1, -1) for idx in grouped.index}


def _single_lookup_by_gene(single_adata: ad.AnnData) -> dict[str, np.ndarray]:
    dense_x = np.asarray(_utils.densify_X(single_adata.X), dtype=np.float32)
    perts = single_adata.obs["perts_name"].astype(str).map(_utils.normalize_condition).values
    return {str(pert): dense_x[idx : idx + 1] for idx, pert in enumerate(perts)}


def _single_lookup_by_condition(single_adata: ad.AnnData) -> dict[str, np.ndarray]:
    dense_x = np.asarray(_utils.densify_X(single_adata.X), dtype=np.float32)
    conds = single_adata.obs["condition"].astype(str).map(_utils.normalize_condition).values
    return {str(cond): dense_x[idx : idx + 1] for idx, cond in enumerate(conds)}


def _predict_single_gene_expression(
    *,
    model,
    single_adata: ad.AnnData,
    dataset_control: dict,
    gene_symbol: str,
    ordered_attribute_key: str,
) -> np.ndarray:
    gene_norm = _utils.normalize_condition(gene_symbol)
    idx_mask = single_adata.obs["perts_name"].astype(str).map(_utils.normalize_condition).eq(gene_norm).values
    if not np.any(idx_mask):
        raise KeyError(f"Single perturbation gene missing in BioLORD single adata: {gene_symbol}")
    dataset_ref = model.get_dataset(single_adata[idx_mask][:1].copy())
    attr_tensor = dataset_ref[ordered_attribute_key][0, :]
    dataset_pred = {
        key: (value.clone() if torch.is_tensor(value) else value)
        for key, value in dataset_control.items()
    }
    dataset_pred[ordered_attribute_key] = _repeat_n(attr_tensor, 1)
    pred_expr, _ = model.module.get_expression(dataset_pred)
    return np.asarray(pred_expr.detach().cpu().numpy(), dtype=np.float32)


def _predict_condition_expression(
    *,
    model,
    condition: str,
    cfg: BiolordDatasetConfig,
    single_adata: ad.AnnData,
    dataset_control: dict,
    train_single_conditions: set[str],
    train_double_conditions: set[str],
    train_single_genes: set[str],
    single_condition_lookup: dict[str, np.ndarray],
    single_gene_lookup: dict[str, np.ndarray],
    full_condition_lookup: dict[str, np.ndarray],
    ctrl_mean: np.ndarray,
) -> np.ndarray:
    cond_norm = _utils.normalize_condition(condition)
    if cond_norm in train_single_conditions:
        return np.asarray(single_condition_lookup[cond_norm], dtype=np.float32)
    if cond_norm in train_double_conditions:
        return np.asarray(full_condition_lookup[cond_norm], dtype=np.float32)

    parts = _condition_tokens_no_ctrl(cond_norm)
    if len(parts) <= 1:
        gene = parts[0] if parts else "ctrl"
        return _predict_single_gene_expression(
            model=model,
            single_adata=single_adata,
            dataset_control=dataset_control,
            gene_symbol=gene,
            ordered_attribute_key=cfg.ordered_attribute_key,
        )

    component_preds: list[np.ndarray] = []
    for gene in parts:
        gene_norm = _utils.normalize_condition(gene)
        if gene_norm in train_single_genes:
            component_preds.append(np.asarray(single_gene_lookup[gene_norm], dtype=np.float32))
        else:
            component_preds.append(
                _predict_single_gene_expression(
                    model=model,
                    single_adata=single_adata,
                    dataset_control=dataset_control,
                    gene_symbol=gene_norm,
                    ordered_attribute_key=cfg.ordered_attribute_key,
                )
            )
    if len(component_preds) != 2:
        raise ValueError(f"Unsupported BioLORD combo condition: {condition}")
    return component_preds[0] + component_preds[1] - ctrl_mean.reshape(1, -1)


def _compute_metrics_and_export_payload(
    *,
    model,
    full_adata: ad.AnnData,
    single_adata: ad.AnnData,
    split_id: int,
    cfg: BiolordDatasetConfig,
) -> tuple[pd.DataFrame, dict]:
    split_key, _ = _require_split_columns(full_adata, split_id)
    name_map = _build_name_map(full_adata)
    gene_names = _gene_names_from_adata(full_adata)
    cond_all = full_adata.obs["condition"].astype(str).map(_utils.normalize_condition)
    ctrl_mask_all = cond_all.eq("ctrl").values
    ctrl_full = np.asarray(_utils.densify_X(full_adata[ctrl_mask_all].X), dtype=np.float32)
    ctrl_mean = ctrl_full.mean(axis=0).reshape(-1)
    top20_degs_final = full_adata.uns.get("top20_degs_final", {})
    if not isinstance(top20_degs_final, dict):
        raise TypeError("Expected full_adata.uns['top20_degs_final'] to be a mapping")

    ref_mask = full_adata.obs[split_key].astype(str).isin(["train", "test"]).values | ctrl_mask_all
    reference_adata = full_adata[ref_mask].copy()
    pert_reference = average_of_perturbation_centroids(
        X=_utils.densify_X(reference_adata.X),
        conditions=reference_adata.obs["condition"].astype(str).values,
        ctrl_label="ctrl",
    )

    adata_control = single_adata[single_adata.obs["condition"].astype(str) == "ctrl"].copy()
    dataset_control = model.get_dataset(adata_control)
    full_condition_lookup = _condition_mean_lookup(full_adata)
    single_condition_lookup = _single_lookup_by_condition(single_adata)
    single_gene_lookup = _single_lookup_by_gene(single_adata)
    train_single_mask = single_adata.obs[split_key].astype(str).eq("train").values
    train_single_conditions = set(
        single_adata.obs.loc[train_single_mask, "condition"].astype(str).map(_utils.normalize_condition).tolist()
    )
    train_single_genes = set(
        single_adata.obs.loc[train_single_mask, "perts_name"].astype(str).map(_utils.normalize_condition).tolist()
    ) - {"ctrl"}
    full_train_conditions = set(_unique_conditions_for_split(full_adata, split_key, "train"))
    train_double_conditions = {cond for cond in full_train_conditions if len(_condition_tokens_no_ctrl(cond)) > 1}
    test_conditions = _unique_conditions_for_split(full_adata, split_key, "ood")

    results = []
    export_payload = {}
    for condition in test_conditions:
        cond_norm = _utils.normalize_condition(condition)
        cond_mask_all = cond_all.eq(cond_norm).values
        if not np.any(cond_mask_all):
            warnings.warn(f"[biolord] test condition missing in full adata: {condition}", RuntimeWarning, stacklevel=2)
            continue

        true_expr = _utils.densify_X(full_adata[cond_mask_all].X)
        if true_expr.ndim == 1:
            true_expr = true_expr.reshape(1, -1)
        degs = np.asarray(top20_degs_final.get(cond_norm, []), dtype=int).reshape(-1)
        if degs.size == 0:
            warnings.warn(f"[biolord] skip condition without DEGs: {condition}", RuntimeWarning, stacklevel=2)
            continue

        pred_expr = _predict_condition_expression(
            model=model,
            condition=cond_norm,
            cfg=cfg,
            single_adata=single_adata,
            dataset_control=dataset_control,
            train_single_conditions=train_single_conditions,
            train_double_conditions=train_double_conditions,
            train_single_genes=train_single_genes,
            single_condition_lookup=single_condition_lookup,
            single_gene_lookup=single_gene_lookup,
            full_condition_lookup=full_condition_lookup,
            ctrl_mean=ctrl_mean,
        )
        pred_expr = np.asarray(pred_expr, dtype=np.float32)
        if pred_expr.ndim == 1:
            pred_expr = pred_expr.reshape(1, -1)

        pred_vec = pred_expr[:, degs].mean(axis=0)
        ctrl_vec = ctrl_full[:, degs].mean(axis=0)
        true_vec = true_expr[:, degs].mean(axis=0)
        mse_ctrl_val = float(mse(true_vec, ctrl_vec))
        mse_pred_val = float(mse(true_vec, pred_vec))
        nmse_val = float(mse_pred_val / mse_ctrl_val) if mse_ctrl_val > 0 else np.nan
        pearson_val = float(pearsonr(true_vec - ctrl_vec, pred_vec - ctrl_vec)[0])
        deg_mean_r2_val = regression_r2_safe(true_vec - ctrl_vec, pred_vec - ctrl_vec)
        systema_metrics = pearson_delta_reference_metrics(
            X_true=true_expr.mean(axis=0),
            X_pred=pred_expr.mean(axis=0),
            reference=pert_reference,
            top20_de_idxs=degs,
        )
        scpram_metrics = _nan_scpram_metrics()

        results.append(
            {
                "condition": cond_norm,
                "condition_name": name_map.get(cond_norm, cond_norm),
                "mse_pred": mse_pred_val,
                "mse_ctrl": mse_ctrl_val,
                "nmse": nmse_val,
                "pearson": pearson_val,
                "deg_mean_r2": float(deg_mean_r2_val),
                "systema_corr_20de_allpert": float(systema_metrics["corr_20de_allpert"]),
                "systema_corr_deg_r2": float(systema_metrics["corr_deg_r2"]),
                **scpram_metrics,
                "split_id": int(split_id),
                "n_ensemble": int(pred_expr.shape[0]),
            }
        )
        export_sample_size = max(1, int(cfg.export_control_pool_size))
        sample_n = min(int(ctrl_full.shape[0]), export_sample_size)
        if sample_n < int(ctrl_full.shape[0]):
            seed_base = (int(split_id) * 1000003) + sum(ord(ch) for ch in str(cond_norm))
            rng = np.random.default_rng(seed_base)
            sample_idx = np.sort(
                rng.choice(
                    int(ctrl_full.shape[0]),
                    size=sample_n,
                    replace=False,
                )
            )
            ctrl_export = np.asarray(ctrl_full[sample_idx], dtype=np.float32)
        else:
            ctrl_export = np.asarray(ctrl_full, dtype=np.float32)
            sample_n = int(ctrl_export.shape[0])
        pred_export = np.repeat(np.asarray(pred_expr[:1], dtype=np.float32), repeats=sample_n, axis=0)
        export_payload[cond_norm] = {
            "Pred": pred_export[:, degs],
            "Ctrl": ctrl_export[:, degs],
            "Truth": true_expr[:, degs],
            "Pred_full": pred_export,
            "Ctrl_full": ctrl_export,
            "Truth_full": true_expr,
            "DE_idx": degs,
            "DE_name": gene_names[degs] if degs.size > 0 else np.array([], dtype=gene_names.dtype),
            "gene_name_full": gene_names,
        }
    return pd.DataFrame(results), export_payload


def run_biolord_eval(
    name: str,
    base_seed: int = 24,
    export_notebook_pkl: bool = True,
    max_epochs: int | None = None,
    batch_size: int | None = None,
    n_latent: int | None = None,
    attribute_width: int | None = None,
    attribute_depth: int | None = None,
) -> None:
    if name not in DATASET_CONFIG:
        raise ValueError(f"Unknown dataset: {name}")
    biolord_mod = _require_biolord_stack()
    print(
        f"[biolord] start dataset={name} seed={base_seed} export_notebook_pkl={bool(export_notebook_pkl)}",
        flush=True,
    )
    base_cfg = DATASET_CONFIG[name]
    cfg = BiolordDatasetConfig(
        full_data_rel=base_cfg.full_data_rel,
        single_data_rel=base_cfg.single_data_rel,
        splits=list(base_cfg.splits),
        ordered_attribute_key=str(base_cfg.ordered_attribute_key),
        legacy_multihot_fallback=bool(base_cfg.legacy_multihot_fallback),
        test_ratio=base_cfg.test_ratio,
        split_policy=str(base_cfg.split_policy),
        domain_key=str(base_cfg.domain_key),
        train_domain_values=tuple(map(str, base_cfg.train_domain_values)),
        test_domain_values=tuple(map(str, base_cfg.test_domain_values)),
        domain_test_ratio=float(base_cfg.domain_test_ratio),
        drug_test_ratio=float(base_cfg.drug_test_ratio),
        val_ratio=float(base_cfg.val_ratio),
        valid_split=str(base_cfg.valid_split),
        test_split=str(base_cfg.test_split),
        aggregate_legacy_condition_means=bool(base_cfg.aggregate_legacy_condition_means),
        n_latent=int(base_cfg.n_latent if n_latent is None else n_latent),
        batch_size=int(base_cfg.batch_size if batch_size is None else batch_size),
        max_epochs=int(base_cfg.max_epochs if max_epochs is None else max_epochs),
        early_stopping=bool(base_cfg.early_stopping),
        early_stopping_patience=int(base_cfg.early_stopping_patience),
        check_val_every_n_epoch=int(base_cfg.check_val_every_n_epoch),
        attribute_nn_width=int(base_cfg.attribute_nn_width if attribute_width is None else attribute_width),
        attribute_nn_depth=int(base_cfg.attribute_nn_depth if attribute_depth is None else attribute_depth),
        attribute_dropout_rate=float(base_cfg.attribute_dropout_rate),
        n_latent_attribute_ordered=int(base_cfg.n_latent_attribute_ordered),
        n_latent_attribute_categorical=int(base_cfg.n_latent_attribute_categorical),
        reconstruction_penalty=float(base_cfg.reconstruction_penalty),
        unknown_attribute_penalty=float(base_cfg.unknown_attribute_penalty),
        decoder_width=int(base_cfg.decoder_width),
        decoder_depth=int(base_cfg.decoder_depth),
        decoder_activation=bool(base_cfg.decoder_activation),
        attribute_nn_activation=bool(base_cfg.attribute_nn_activation),
        unknown_attributes=bool(base_cfg.unknown_attributes),
        use_batch_norm=bool(base_cfg.use_batch_norm),
        use_layer_norm=bool(base_cfg.use_layer_norm),
        latent_lr=float(base_cfg.latent_lr),
        latent_wd=float(base_cfg.latent_wd),
        attribute_nn_lr=float(base_cfg.attribute_nn_lr),
        attribute_nn_wd=float(base_cfg.attribute_nn_wd),
        step_size_lr=int(base_cfg.step_size_lr),
        cosine_scheduler=bool(base_cfg.cosine_scheduler),
        scheduler_final_lr=float(base_cfg.scheduler_final_lr),
        decoder_lr=float(base_cfg.decoder_lr),
        decoder_wd=float(base_cfg.decoder_wd),
        unknown_attribute_noise_param=float(base_cfg.unknown_attribute_noise_param),
        export_control_pool_size=int(base_cfg.export_control_pool_size),
    )

    full_data_path, single_data_path = _resolve_eval_data_paths(name, cfg)
    print(f"[biolord] loading full data: {full_data_path}", flush=True)
    full_adata, _ = _prepare_full_eval_adata(full_data_path)
    out_dir = ROOT / "artifacts" / "results" / "biolord" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[biolord] output dir: {out_dir}", flush=True)

    if bool(cfg.legacy_multihot_fallback):
        print(
            f"[biolord] using full-data attribute path key={cfg.ordered_attribute_key} "
            f"split_policy={cfg.split_policy}",
            flush=True,
        )
        data = TriShiftData(full_adata.copy(), _dummy_embedding_df())
        metrics_all = []
        for split in cfg.splits:
            print(
                f"[biolord] dataset={name} split={split} max_epochs={cfg.max_epochs} "
                f"attribute_key={cfg.ordered_attribute_key}",
                flush=True,
            )
            set_seeds(base_seed + int(split))
            split_dict = _build_split_dict_for_config(data, cfg, int(split))
            train_adata, split_key = _prepare_legacy_train_adata(
                eval_adata=full_adata,
                split_dict=split_dict,
                split_id=int(split),
                cfg=cfg,
            )
            model = _build_model_and_train(
                biolord_mod=biolord_mod,
                train_adata=train_adata,
                split_key=split_key,
                cfg=cfg,
                seed=base_seed + int(split),
            )
            _export_training_history_artifacts(model, out_dir, name, int(split))
            metrics_df, export_payload = _compute_metrics_and_export_payload_legacy(
                model=model,
                eval_adata=full_adata,
                train_adata=train_adata,
                split_dict=split_dict,
                split_id=int(split),
                cfg=cfg,
                base_seed=base_seed,
            )
            metrics_all.append(metrics_df)
            if export_notebook_pkl:
                out_pkl = out_dir / f"biolord_{name}_{split}.pkl"
                with out_pkl.open("wb") as handle:
                    pickle.dump(export_payload, handle)
                print(f"[biolord] saved notebook payload: {out_pkl}", flush=True)

        metrics_df_all = pd.concat(metrics_all, ignore_index=True)
        metrics_df_all.to_csv(out_dir / "metrics.csv", index=False)
        _write_mean_metrics(out_dir / "mean_pearson.txt", metrics_df_all)
        print(f"[biolord] saved metrics: {out_dir / 'metrics.csv'}", flush=True)
        return

    if single_data_path is None:
        raise FileNotFoundError(f"Missing BioLORD single-data path for dataset={name}")
    print(f"[biolord] loading single data: {single_data_path}", flush=True)
    single_adata = _prepare_single_train_adata(single_data_path, cfg.ordered_attribute_key)
    metrics_all = []
    for split in cfg.splits:
        print(
            f"[biolord] dataset={name} split={split} max_epochs={cfg.max_epochs}",
            flush=True,
        )
        set_seeds(base_seed + int(split))
        split_key, _ = _require_split_columns(single_adata, int(split))
        subgroup_df = _subgroup_frame_from_full_adata(full_adata, int(split))
        model = _build_model_and_train(
            biolord_mod=biolord_mod,
            train_adata=single_adata,
            split_key=split_key,
            cfg=cfg,
            seed=base_seed + int(split),
        )
        _export_training_history_artifacts(model, out_dir, name, int(split))
        metrics_df, export_payload = _compute_metrics_and_export_payload(
            model=model,
            full_adata=full_adata,
            single_adata=single_adata,
            split_id=int(split),
            cfg=cfg,
        )
        metrics_df = _attach_subgroup_column(metrics_df, subgroup_df)
        metrics_all.append(metrics_df)
        if export_notebook_pkl:
            out_pkl = out_dir / f"biolord_{name}_{split}.pkl"
            with out_pkl.open("wb") as handle:
                pickle.dump(export_payload, handle)
            print(f"[biolord] saved notebook payload: {out_pkl}", flush=True)

    metrics_df_all = pd.concat(metrics_all, ignore_index=True)
    metrics_df_all.to_csv(out_dir / "metrics.csv", index=False)
    _write_mean_metrics(out_dir / "mean_pearson.txt", metrics_df_all)
    print(f"[biolord] saved metrics: {out_dir / 'metrics.csv'}", flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run BioLORD training + eval with TriShift metrics")
    parser.add_argument("--profile", default="", help="dataset profile under scripts/biolord/eval/configs")
    parser.add_argument("--name", default="", help="dataset name")
    parser.add_argument("--seed", type=int, default=24, help="base seed")
    parser.add_argument("--max_epochs", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=0)
    parser.add_argument("--n_latent", type=int, default=0)
    parser.add_argument("--attribute_width", type=int, default=0)
    parser.add_argument("--attribute_depth", type=int, default=0)
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
        run_biolord_eval(
            prof["dataset"],
            base_seed=int(task_args.get("seed", 24)) if int(args.seed) == 24 else int(args.seed),
            export_notebook_pkl=(
                False if bool(args.no_export_notebook_pkl) else bool(task_args.get("export_notebook_pkl", True))
            ),
            max_epochs=(
                int(task_args.get("max_epochs", 0))
                if int(args.max_epochs) == 0
                else int(args.max_epochs)
            ) or None,
            batch_size=(
                int(task_args.get("batch_size", 0))
                if int(args.batch_size) == 0
                else int(args.batch_size)
            ) or None,
            n_latent=(
                int(task_args.get("n_latent", 0))
                if int(args.n_latent) == 0
                else int(args.n_latent)
            ) or None,
            attribute_width=(
                int(task_args.get("attribute_width", 0))
                if int(args.attribute_width) == 0
                else int(args.attribute_width)
            ) or None,
            attribute_depth=(
                int(task_args.get("attribute_depth", 0))
                if int(args.attribute_depth) == 0
                else int(args.attribute_depth)
            ) or None,
        )
        return

    name = str(args.name).strip()
    if not name:
        raise SystemExit("Provide --profile or --name")
    run_biolord_eval(
        name,
        base_seed=int(args.seed),
        export_notebook_pkl=not bool(args.no_export_notebook_pkl),
        max_epochs=(int(args.max_epochs) or None),
        batch_size=(int(args.batch_size) or None),
        n_latent=(int(args.n_latent) or None),
        attribute_width=(int(args.attribute_width) or None),
        attribute_depth=(int(args.attribute_depth) or None),
    )


if __name__ == "__main__":
    main()
