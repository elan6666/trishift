from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import copy
import pickle
import random
import sys
import warnings

import anndata as ad
import numpy as np
import pandas as pd
import torch
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
class BiolordDatasetConfig:
    data_rel: str
    splits: list[int]
    test_ratio: float
    norman_split: bool = False
    ordered_attribute_key: str = "perturbation_neighbors"
    n_latent: int = 32
    batch_size: int = 32
    max_epochs: int = 75
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
        data_rel="data/adamson/perturb_processed.h5ad",
        splits=[1, 2, 3, 4, 5],
        test_ratio=0.2,
    ),
    "dixit": BiolordDatasetConfig(
        data_rel="data/dixit/perturb_processed.h5ad",
        splits=[1, 2, 3, 4, 5],
        test_ratio=0.2,
    ),
    "norman": BiolordDatasetConfig(
        data_rel="data/norman/perturb_processed.h5ad",
        splits=[1, 2, 3, 4, 5],
        test_ratio=0.2,
        norman_split=True,
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
    "replogle_k562_essential": BiolordDatasetConfig(
        data_rel="data/replogle_k562_essential/perturb_processed.h5ad",
        splits=[1, 2, 3, 4, 5],
        test_ratio=0.2,
    ),
    "replogle_rpe1_essential": BiolordDatasetConfig(
        data_rel="data/replogle_rpe1_essential/perturb_processed.h5ad",
        splits=[1, 2, 3, 4, 5],
        test_ratio=0.2,
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


def _dummy_embedding_df() -> pd.DataFrame:
    return pd.DataFrame([[0.0]], index=["ctrl"], columns=["dummy"])


def _resolve_eval_data_path(name: str, cfg: BiolordDatasetConfig) -> Path:
    primary = (LOCAL_DATA_ROOT / Path(cfg.data_rel).relative_to("data")).resolve()
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
        f"Missing evaluation h5ad for biolord dataset={name}. Tried {primary}"
    )


def _prepare_eval_adata(data_path: Path) -> tuple[ad.AnnData, TriShiftData]:
    adata = _utils.load_adata(str(data_path))
    data = TriShiftData(adata, _dummy_embedding_df())
    data.build_or_load_degs()
    return data.adata_all, data


def _gene_names_from_adata(adata: ad.AnnData) -> np.ndarray:
    for col in ["gene_name", "gene_names"]:
        if col in adata.var.columns:
            return adata.var[col].astype(str).values
    return adata.var_names.astype(str).values


def _condition_tokens_no_ctrl(condition: str) -> list[str]:
    return [tok for tok in str(condition).split("+") if tok and tok != "ctrl"]


def _normalize_pert_symbol(token: str) -> str:
    return str(token).strip()


def _build_name_map(adata: ad.AnnData) -> dict[str, str]:
    out: dict[str, str] = {}
    if "condition_name" in adata.obs.columns:
        pairs = adata.obs[["condition", "condition_name"]].drop_duplicates().values
        for cond, cname in pairs:
            out[_utils.normalize_condition(str(cond))] = str(cname)
    for cond in adata.obs["condition"].astype(str).unique():
        out.setdefault(_utils.normalize_condition(cond), cond)
    return out


def _build_split_dict(
    name: str,
    data: TriShiftData,
    split_id: int,
    test_ratio: float,
) -> tuple[dict, pd.DataFrame | None]:
    subgroup_df = None
    if name == "norman":
        subgroup_df = subgroup(list(data.adata_all.obs["condition"].astype(str).unique()), seed=int(split_id))
        test_conds = list(subgroup_df[subgroup_df.group == "test"].index)
        val_conds = list(subgroup_df[subgroup_df.group == "val"].index)
        split_dict = data.split_by_condition(
            seed=int(split_id),
            test_conds=test_conds,
            val_conds=val_conds,
        )
        return split_dict, subgroup_df
    return data.split_by_condition(seed=int(split_id), test_ratio=float(test_ratio)), None


def _assign_split_labels(adata: ad.AnnData, split_dict: dict, split_key: str) -> ad.AnnData:
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


def _build_multihot_attributes(adata: ad.AnnData, key: str) -> ad.AnnData:
    out = adata.copy()
    conds = out.obs["condition"].astype(str).tolist()
    pert_basis = sorted(
        {
            _normalize_pert_symbol(tok)
            for cond in conds
            for tok in _condition_tokens_no_ctrl(_utils.normalize_condition(cond))
        }
    )
    if not pert_basis:
        raise ValueError("No perturbation tokens found for biolord attribute construction")
    token_to_idx = {tok: i for i, tok in enumerate(pert_basis)}
    emb = np.zeros((out.n_obs, len(pert_basis)), dtype=np.float32)
    for row_idx, cond in enumerate(conds):
        for tok in _condition_tokens_no_ctrl(_utils.normalize_condition(cond)):
            tok_norm = _normalize_pert_symbol(tok)
            if tok_norm in token_to_idx:
                emb[row_idx, token_to_idx[tok_norm]] = 1.0
    out.obsm[key] = emb
    return out


def _prepare_biolord_adata(
    *,
    eval_adata: ad.AnnData,
    split_dict: dict,
    split_id: int,
    cfg: BiolordDatasetConfig,
) -> tuple[ad.AnnData, str]:
    split_key = f"split{int(split_id)}"
    split_adata = _assign_split_labels(eval_adata, split_dict, split_key)
    split_adata = _build_multihot_attributes(split_adata, cfg.ordered_attribute_key)
    return split_adata, split_key


def _require_biolord_stack():
    try:
        import biolord  # type: ignore
    except Exception as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError(
            "biolord is not installed in the current environment. Run this pipeline in the BioLORD environment."
        ) from exc
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
        valid_split="val",
        test_split="test",
    )
    model.train(
        max_epochs=int(cfg.max_epochs),
        batch_size=int(cfg.batch_size),
        plan_kwargs=_build_trainer_params(cfg),
        early_stopping=True,
        early_stopping_patience=int(cfg.early_stopping_patience),
        check_val_every_n_epoch=int(cfg.check_val_every_n_epoch),
        num_workers=0,
        enable_checkpointing=False,
    )
    return model


def _get_condition_attribute_tensor(model, adata: ad.AnnData, condition: str, ordered_attribute_key: str) -> torch.Tensor:
    cond_mask = adata.obs["condition"].astype(str).eq(condition).values
    if not np.any(cond_mask):
        raise KeyError(f"Condition missing in biolord adata: {condition}")
    cond_adata = adata[cond_mask][:1].copy()
    dataset_ref = model.get_dataset(cond_adata)
    return dataset_ref[ordered_attribute_key][0, :]


def _repeat_n(x: torch.Tensor, n: int) -> torch.Tensor:
    return x.to(x.device).view(1, -1).repeat(int(n), 1)


def _safe_scpram_metrics(X_true: np.ndarray, X_pred: np.ndarray, deg_idx: np.ndarray) -> dict[str, float]:
    try:
        return compute_scpram_metrics_from_arrays(
            X_true=X_true,
            X_pred=X_pred,
            deg_idx=deg_idx,
            n_degs=100,
            sample_ratio=0.8,
            times=100,
        )
    except Exception as exc:
        warnings.warn(f"[biolord] scpram metric fallback to NaN: {exc}", RuntimeWarning, stacklevel=2)
        return {
            "scpram_r2_degs_mean_mean": np.nan,
            "scpram_r2_degs_var_mean": np.nan,
            "scpram_wasserstein_degs_sum": np.nan,
        }


def _compute_metrics_and_export_payload(
    *,
    model,
    eval_adata: ad.AnnData,
    train_adata: ad.AnnData,
    split_dict: dict,
    split_id: int,
    cfg: BiolordDatasetConfig,
) -> tuple[pd.DataFrame, dict]:
    name_map = _build_name_map(eval_adata)
    gene_names = _gene_names_from_adata(eval_adata)
    cond_all = eval_adata.obs["condition"].astype(str).map(_utils.normalize_condition)
    ctrl_mask_all = cond_all.eq("ctrl").values
    ctrl_full = np.asarray(_utils.densify_X(eval_adata[ctrl_mask_all].X), dtype=np.float32)
    ctrl_mean = ctrl_full.mean(axis=0).reshape(-1)
    top20_degs_final = eval_adata.uns.get("top20_degs_final", {})
    if not isinstance(top20_degs_final, dict):
        raise TypeError("Expected eval_adata.uns['top20_degs_final'] to be a mapping")

    train_val_conds = list(map(str, split_dict.get("train_conds", []))) + list(map(str, split_dict.get("val_conds", [])))
    ref_mask = eval_adata.obs["condition"].astype(str).isin(set(train_val_conds) | {"ctrl"}).values
    reference_adata = eval_adata[ref_mask].copy()
    pert_reference = average_of_perturbation_centroids(
        X=_utils.densify_X(reference_adata.X),
        conditions=reference_adata.obs["condition"].astype(str).values,
        ctrl_label="ctrl",
    )

    adata_control = train_adata[train_adata.obs["condition"].astype(str) == "ctrl"].copy()
    dataset_control = model.get_dataset(adata_control)
    n_obs = adata_control.n_obs

    results = []
    export_payload = {}
    for condition in sorted(set(map(str, split_dict.get("test_conds", [])))):
        cond_norm = _utils.normalize_condition(condition)
        cond_mask_all = cond_all.eq(cond_norm).values
        if not np.any(cond_mask_all):
            warnings.warn(f"[biolord] test condition missing in eval adata: {condition}", RuntimeWarning, stacklevel=2)
            continue

        true_expr = _utils.densify_X(eval_adata[cond_mask_all].X)
        if true_expr.ndim == 1:
            true_expr = true_expr.reshape(1, -1)
        degs = np.asarray(top20_degs_final.get(cond_norm, []), dtype=int).reshape(-1)
        if degs.size == 0:
            warnings.warn(f"[biolord] skip condition without DEGs: {condition}", RuntimeWarning, stacklevel=2)
            continue

        attr_tensor = _get_condition_attribute_tensor(model, train_adata, cond_norm, cfg.ordered_attribute_key)
        dataset_pred = {
            key: (value.clone() if torch.is_tensor(value) else value)
            for key, value in dataset_control.items()
        }
        dataset_pred[cfg.ordered_attribute_key] = _repeat_n(attr_tensor, n_obs)
        pred_expr, _ = model.module.get_expression(dataset_pred)
        pred_expr = np.asarray(pred_expr.detach().cpu().numpy(), dtype=np.float32)

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
        scpram_metrics = _safe_scpram_metrics(true_expr, pred_expr, degs)

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
        sample_n = min(int(pred_expr.shape[0]), int(ctrl_full.shape[0]), export_sample_size)
        if sample_n < min(int(pred_expr.shape[0]), int(ctrl_full.shape[0])):
            seed_base = (int(split_id) * 1000003) + sum(ord(ch) for ch in str(cond_norm))
            rng = np.random.default_rng(seed_base)
            sample_idx = np.sort(
                rng.choice(
                    min(int(pred_expr.shape[0]), int(ctrl_full.shape[0])),
                    size=sample_n,
                    replace=False,
                )
            )
            pred_export = np.asarray(pred_expr[sample_idx], dtype=np.float32)
            ctrl_export = np.asarray(ctrl_full[sample_idx], dtype=np.float32)
        else:
            pred_export = np.asarray(pred_expr, dtype=np.float32)
            ctrl_export = np.asarray(ctrl_full, dtype=np.float32)
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
        data_rel=base_cfg.data_rel,
        splits=list(base_cfg.splits),
        test_ratio=float(base_cfg.test_ratio),
        norman_split=bool(base_cfg.norman_split),
        ordered_attribute_key=str(base_cfg.ordered_attribute_key),
        n_latent=int(base_cfg.n_latent if n_latent is None else n_latent),
        batch_size=int(base_cfg.batch_size if batch_size is None else batch_size),
        max_epochs=int(base_cfg.max_epochs if max_epochs is None else max_epochs),
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

    data_path = _resolve_eval_data_path(name, cfg)
    print(f"[biolord] loading data: {data_path}", flush=True)
    eval_adata, data = _prepare_eval_adata(data_path)
    out_dir = ROOT / "artifacts" / "results" / "biolord" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[biolord] output dir: {out_dir}", flush=True)

    metrics_all = []
    for split in cfg.splits:
        print(
            f"[biolord] dataset={name} split={split} max_epochs={cfg.max_epochs}",
            flush=True,
        )
        set_seeds(base_seed + int(split))
        split_dict, subgroup_df = _build_split_dict(name, data, int(split), float(cfg.test_ratio))
        train_adata, split_key = _prepare_biolord_adata(
            eval_adata=eval_adata,
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
        metrics_df, export_payload = _compute_metrics_and_export_payload(
            model=model,
            eval_adata=eval_adata,
            train_adata=train_adata,
            split_dict=split_dict,
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
