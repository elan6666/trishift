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
import scipy.linalg
import torch

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
)
from trishift.TriShiftData import TriShiftData
from scripts.common.split_utils import (
    condition_sort,
    split_unseen_ctrl_unseen_perturbation,
)
from scripts.common.payload_subset import subset_payload_item
from scripts.common.yaml_utils import load_yaml_file
from scripts.trishift._core.run_dataset_core import _split_celltype_seen_perturbation


@dataclass(frozen=True)
class BiolordDatasetConfig:
    full_data_rel: str
    single_data_rel: str
    splits: list[int]
    test_ratio: float = 0.2
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
    ctrl_test_ratio: float = 0.2
    ctrl_val_ratio: float = 0.1
    pert_val_ratio: float = 0.1


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
}


def _load_profile(profile: str) -> dict:
    p = PROFILE_DIR / f"{profile}.yaml"
    if not p.exists():
        raise FileNotFoundError(f"BioLORD profile not found: {p}")
    obj = load_yaml_file(p)
    unknown = sorted(set(obj.keys()) - PROFILE_ALLOWED_KEYS)
    if unknown:
        raise ValueError(f"Unknown keys in BioLORD profile {p.name}: {unknown}")
    dataset = str(obj.get("dataset", "")).strip()
    if dataset != profile:
        raise ValueError(f"Profile dataset mismatch: file={p.name} dataset={dataset!r}")
    task_args = obj.get("task_args") or {}
    if not isinstance(task_args, dict):
        raise TypeError(f"profile.task_args must be a mapping: {p}")
    return {"dataset": dataset, "task_args": task_args}


def _resolve_mean_metric_keys(numeric_means: pd.Series) -> list[str]:
    scpram_legacy_aliases = {
        "r2_all_mean_mean": "scpram_r2_all_mean_mean",
        "r2_all_var_mean": "scpram_r2_all_var_mean",
        "r2_degs_mean_mean": "scpram_r2_degs_mean_mean",
        "r2_degs_var_mean": "scpram_r2_degs_var_mean",
    }
    preferred_order = [
        "pearson",
        "nmse",
        "mse_pred",
        "mse_ctrl",
        "deg_mean_r2",
        "systema_corr_20de_allpert",
        "systema_corr_deg_r2",
        "scpram_r2_all_mean_mean",
        "scpram_r2_all_var_mean",
        "scpram_r2_degs_mean_mean",
        "scpram_r2_degs_var_mean",
        "scpram_wasserstein_degs_sum",
    ]
    exclude_keys = {"split_id", "n_ensemble", "n_eval_ctrl"}
    exclude_keys.update(
        alias
        for alias, canonical in scpram_legacy_aliases.items()
        if canonical in numeric_means.index
    )
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
    lines.append("subgroup_order=single,seen0,seen1,seen2\n\n")
    for g in subgroup_names:
        sub_df = metrics_df[metrics_df["subgroup"].astype(str) == g]
        lines.append(f"subgroup_{g}_n_rows={int(len(sub_df))}\n")
        sub_numeric_means = sub_df.mean(numeric_only=True)
        for key in keys:
            if key in sub_numeric_means.index:
                lines.append(f"subgroup_{g}_mean_{key}={float(sub_numeric_means[key])}\n")
        lines.append("\n")


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
    raise FileNotFoundError(f"Missing BioLORD input for dataset={name}. Tried {primary}")


def _resolve_eval_data_paths(name: str, cfg: BiolordDatasetConfig) -> tuple[Path, Path]:
    return _resolve_dataset_path(name, cfg.full_data_rel), _resolve_dataset_path(name, cfg.single_data_rel)


def _normalize_condition_obs(adata: ad.AnnData) -> ad.AnnData:
    out = adata.copy()
    out.obs["condition"] = (
        out.obs["condition"].astype(str).map(condition_sort).astype("category")
    )
    return out


def _prepare_full_eval_adata(data_path: Path) -> tuple[ad.AnnData, TriShiftData]:
    adata = _normalize_condition_obs(ad.read_h5ad(data_path))
    data = TriShiftData(adata, _dummy_embedding_df())
    return data.adata_all, data


def _prepare_single_train_adata(data_path: Path, ordered_attribute_key: str) -> ad.AnnData:
    adata = _normalize_condition_obs(ad.read_h5ad(data_path))
    if ordered_attribute_key not in adata.obsm:
        raise KeyError(
            f"Missing ordered attribute key '{ordered_attribute_key}' in BioLORD single adata: {data_path}"
        )
    return adata


def _gene_names_from_adata(adata: ad.AnnData) -> np.ndarray:
    for col in ["gene_name", "gene_names"]:
        if col in adata.var.columns:
            return adata.var[col].astype(str).values
    return adata.var_names.astype(str).values


def _coerce_gene_indices(values, adata: ad.AnnData) -> np.ndarray:
    arr = np.asarray(values).reshape(-1)
    if arr.size == 0:
        return np.asarray([], dtype=int)
    by_name = {str(g): i for i, g in enumerate(adata.var_names.astype(str))}
    for col in ["gene_name", "gene_names"]:
        if col in adata.var.columns:
            for i, g in enumerate(adata.var[col].astype(str).values):
                by_name.setdefault(str(g), i)

    idx: list[int] = []
    for item in arr.tolist():
        key = str(item)
        if key in by_name:
            idx.append(int(by_name[key]))
            continue
        try:
            parsed = int(item)
        except Exception:
            continue
        if 0 <= parsed < adata.n_vars:
            idx.append(parsed)
    seen: set[int] = set()
    out = []
    for i in idx:
        if i not in seen:
            out.append(i)
            seen.add(i)
    return np.asarray(out, dtype=int)


def _condition_tokens_no_ctrl(condition: str) -> list[str]:
    return [tok for tok in condition_sort(str(condition)).split("+") if tok and tok != "ctrl"]


def _condition_name_candidates(adata: ad.AnnData) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    conds = adata.obs["condition"].astype(str).map(condition_sort)
    if "condition_name" in adata.obs.columns:
        names = adata.obs["condition_name"].astype(str)
        for cond, name in zip(conds.values, names.values):
            out.setdefault(str(cond), [])
            if str(name) not in out[str(cond)]:
                out[str(cond)].append(str(name))
    for cond in conds.unique():
        out.setdefault(str(cond), [])
        if str(cond) not in out[str(cond)]:
            out[str(cond)].append(str(cond))
    return out


def _build_top20_lookup(adata: ad.AnnData) -> dict[str, np.ndarray]:
    top_map = None
    for key in [
        "top20_degs_final",
        "top20_degs_non_dropout",
        "top_non_zero_de_20",
        "top_non_dropout_de_20",
    ]:
        obj = adata.uns.get(key)
        if isinstance(obj, dict):
            top_map = obj
            break
    if not isinstance(top_map, dict):
        raise TypeError("Expected a top-20 DEG mapping in adata.uns")

    name_candidates = _condition_name_candidates(adata)
    out: dict[str, np.ndarray] = {}
    for cond, candidates in name_candidates.items():
        for cand in candidates:
            if cand in top_map:
                out[cond] = _coerce_gene_indices(top_map[cand], adata)
                break
        if cond not in out and cond in top_map:
            out[cond] = _coerce_gene_indices(top_map[cond], adata)
    return out


def _first_condition_name(adata: ad.AnnData) -> dict[str, str]:
    candidates = _condition_name_candidates(adata)
    return {cond: vals[0] if vals else cond for cond, vals in candidates.items()}


def _map_subgroup_label(raw: str) -> str:
    mapping = {
        "unseen_single": "single",
        "combo_seen0": "seen0",
        "combo_seen1": "seen1",
        "combo_seen2": "seen2",
        "single": "single",
        "seen0": "seen0",
        "seen1": "seen1",
        "seen2": "seen2",
    }
    return mapping.get(str(raw), "unknown")


def _subgroup_map_from_df(subgroup_df: pd.DataFrame | None) -> dict[str, str]:
    if subgroup_df is None or "subgroup" not in subgroup_df.columns:
        return {}
    return {
        condition_sort(str(idx)): _map_subgroup_label(str(row["subgroup"]))
        for idx, row in subgroup_df.iterrows()
        if condition_sort(str(idx)) != "ctrl"
    }


def _subgroup_frame_from_full_adata(adata: ad.AnnData, split_id: int) -> pd.DataFrame:
    split_key = f"split{int(split_id)}"
    subgroup_key = f"subgroup{int(split_id)}"
    if split_key not in adata.obs.columns or subgroup_key not in adata.obs.columns:
        return pd.DataFrame(columns=["subgroup"])
    obs = adata.obs[["condition", split_key, subgroup_key]].copy()
    obs["condition"] = obs["condition"].astype(str).map(condition_sort)
    obs = obs[obs[split_key].astype(str) == "ood"]
    if obs.empty:
        return pd.DataFrame(columns=["subgroup"])
    return (
        obs[["condition", subgroup_key]]
        .drop_duplicates()
        .assign(subgroup=lambda df: df[subgroup_key].astype(str).map(_map_subgroup_label))
        .drop(columns=[subgroup_key])
        .set_index("condition")
    )


def _attach_subgroup_column(metrics_df: pd.DataFrame, subgroup_map: dict[str, str] | pd.DataFrame | None) -> pd.DataFrame:
    if subgroup_map is None or "condition" not in metrics_df.columns:
        return metrics_df
    if isinstance(subgroup_map, pd.DataFrame):
        if "subgroup" not in subgroup_map.columns:
            return metrics_df
        map_eff = subgroup_map["subgroup"].astype(str).to_dict()
    else:
        map_eff = {condition_sort(k): str(v) for k, v in subgroup_map.items()}
    out = metrics_df.copy()
    if "subgroup" in out.columns:
        out = out.drop(columns=["subgroup"])
    vals = out["condition"].astype(str).map(lambda x: map_eff.get(condition_sort(x), "unknown"))
    out.insert(list(out.columns).index("condition") + 1, "subgroup", vals)
    return out


def _require_biolord_stack():
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
                    elif isinstance(use_gpu, str) and use_gpu.lower().startswith("cuda"):
                        accelerator = "gpu"
                        try:
                            devices = [int(use_gpu.split(":")[1])]
                        except Exception:
                            devices = 1
                    else:
                        accelerator = "auto"
                        devices = "auto"

                super().__init__(*args, accelerator=accelerator, devices=devices, **trainer_kwargs)

            def __call__(self):
                if hasattr(self.data_splitter, "n_train"):
                    self.training_plan.n_obs_training = self.data_splitter.n_train
                if hasattr(self.data_splitter, "n_val"):
                    self.training_plan.n_obs_validation = self.data_splitter.n_val
                self.trainer.fit(
                    self.training_plan,
                    train_dataloaders=self.data_splitter.train_dataloader(),
                    val_dataloaders=self.data_splitter.val_dataloader(),
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
        valid_split="test",
        test_split="ood",
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


def _repeat_n(x: torch.Tensor, n: int) -> torch.Tensor:
    return x.to(x.device).view(1, -1).repeat(int(n), 1)


def _nan_scpram_metrics() -> dict[str, float]:
    return {
        "scpram_r2_all_mean_mean": np.nan,
        "scpram_r2_all_var_mean": np.nan,
        "scpram_r2_degs_mean_mean": np.nan,
        "scpram_r2_degs_var_mean": np.nan,
        "scpram_wasserstein_degs_sum": np.nan,
    }


def _condition_mean_lookup(adata: ad.AnnData) -> dict[str, np.ndarray]:
    dense_x = np.asarray(_utils.densify_X(adata.X), dtype=np.float32)
    conds = adata.obs["condition"].astype(str).map(condition_sort).values
    df = pd.DataFrame(dense_x, columns=adata.var_names)
    df["condition"] = conds
    grouped = df.groupby("condition", observed=False).mean()
    return {str(idx): grouped.loc[idx].to_numpy(dtype=np.float32).reshape(1, -1) for idx in grouped.index}


def _single_lookup_by_gene(single_adata: ad.AnnData) -> dict[str, np.ndarray]:
    dense_x = np.asarray(_utils.densify_X(single_adata.X), dtype=np.float32)
    perts = single_adata.obs["perts_name"].astype(str).map(condition_sort).values
    return {str(pert): dense_x[idx : idx + 1] for idx, pert in enumerate(perts)}


def _single_lookup_by_condition(single_adata: ad.AnnData) -> dict[str, np.ndarray]:
    dense_x = np.asarray(_utils.densify_X(single_adata.X), dtype=np.float32)
    conds = single_adata.obs["condition"].astype(str).map(condition_sort).values
    return {str(cond): dense_x[idx : idx + 1] for idx, cond in enumerate(conds)}


def _predict_single_gene_expression(
    *,
    model,
    single_adata: ad.AnnData,
    dataset_control: dict,
    gene_symbol: str,
    ordered_attribute_key: str,
) -> np.ndarray:
    gene_norm = condition_sort(gene_symbol)
    idx_mask = single_adata.obs["perts_name"].astype(str).map(condition_sort).eq(gene_norm).values
    if not np.any(idx_mask):
        raise KeyError(f"Single perturbation gene missing in BioLORD single adata: {gene_symbol}")
    dataset_ref = model.get_dataset(single_adata[idx_mask][:1].copy())
    attr_tensor = dataset_ref[ordered_attribute_key][0, :]
    dataset_pred = {
        key: (value.clone() if torch.is_tensor(value) else value)
        for key, value in dataset_control.items()
    }
    dataset_pred[ordered_attribute_key] = _repeat_n(attr_tensor, 1)
    with torch.no_grad():
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
    ctrl_mean_for_combo: np.ndarray,
) -> np.ndarray:
    cond_norm = condition_sort(condition)
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
        gene_norm = condition_sort(gene)
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
    return component_preds[0] + component_preds[1] - ctrl_mean_for_combo.reshape(1, -1)


def _set_single_ctrl_expression(single_adata: ad.AnnData, ctrl_mean: np.ndarray) -> ad.AnnData:
    out = single_adata.copy()
    ctrl_mask = out.obs["condition"].astype(str).map(condition_sort).eq("ctrl").values
    if not np.any(ctrl_mask):
        raise ValueError("BioLORD single adata has no ctrl row")
    X = np.asarray(_utils.densify_X(out.X), dtype=np.float32)
    X[ctrl_mask, :] = np.asarray(ctrl_mean, dtype=np.float32).reshape(1, -1)
    out.X = X
    return out


def _apply_unseen_split_to_single_adata(
    single_adata: ad.AnnData,
    split_dict: dict,
    split_id: int,
) -> tuple[ad.AnnData, str]:
    out = single_adata.copy()
    split_key = f"unseen_ctrl_split{int(split_id)}"
    subgroup_key = f"unseen_ctrl_subgroup{int(split_id)}"
    conds = out.obs["condition"].astype(str).map(condition_sort)
    train_conds = {condition_sort(c) for c in split_dict.get("train_conds", [])}
    val_conds = {condition_sort(c) for c in split_dict.get("val_conds", [])}
    test_conds = {condition_sort(c) for c in split_dict.get("test_conds", [])}

    split_vals = np.full(out.n_obs, "ignore", dtype=object)
    split_vals[conds.isin(train_conds).values] = "train"
    split_vals[conds.isin(val_conds).values] = "test"
    split_vals[conds.isin(test_conds).values] = "ood"
    split_vals[conds.eq("ctrl").values] = "ignore"
    if not np.any(split_vals == "train") or not np.any(split_vals == "test"):
        raise ValueError(
            f"BioLORD single split has empty train/val labels: "
            f"train={int(np.sum(split_vals == 'train'))}, test={int(np.sum(split_vals == 'test'))}"
        )
    out.obs[split_key] = pd.Categorical(split_vals, categories=["train", "test", "ood", "ignore"])

    subgroup_vals = np.full(out.n_obs, "Train/Val", dtype=object)
    subgroup_vals[conds.isin(test_conds).values] = "unseen_single"
    out.obs[subgroup_key] = pd.Categorical(
        subgroup_vals,
        categories=["Train/Val", "unseen_single", "combo_seen0", "combo_seen1", "combo_seen2"],
    )
    return out, split_key


def _conditions_for_split_column(adata: ad.AnnData, split_key: str, value: str) -> list[str]:
    if split_key not in adata.obs.columns:
        raise KeyError(f"BioLORD adata is missing split column: {split_key}")
    mask = adata.obs[split_key].astype(str).eq(value).values
    conds = adata.obs.loc[mask, "condition"].astype(str).map(condition_sort).tolist()
    return sorted({str(c) for c in conds if str(c) != "ctrl"})


def _metric_and_payload_for_conditions(
    *,
    model,
    full_adata: ad.AnnData,
    single_adata: ad.AnnData,
    cfg: BiolordDatasetConfig,
    dataset_name: str,
    split_id: int,
    test_adata: ad.AnnData,
    ctrl_metric_adata: ad.AnnData,
    reference_adata: ad.AnnData,
    test_conditions: list[str],
    train_conditions: list[str],
    split_policy: str,
    eval_ctrl_source: str,
    prediction_ctrl_source: str,
) -> tuple[pd.DataFrame, dict]:
    gene_names = _gene_names_from_adata(full_adata)
    top20 = _build_top20_lookup(full_adata)
    cond_names = _first_condition_name(full_adata)
    cond_all = full_adata.obs["condition"].astype(str).map(condition_sort)
    test_cond_arr = test_adata.obs["condition"].astype(str).map(condition_sort).values

    ctrl_metric = np.asarray(_utils.densify_X(ctrl_metric_adata.X), dtype=np.float32)
    if ctrl_metric.ndim == 1:
        ctrl_metric = ctrl_metric.reshape(1, -1)
    if ctrl_metric.shape[0] == 0:
        raise ValueError("BioLORD evaluation received no ctrl cells")
    ctrl_mean_for_combo = np.asarray(_utils.densify_X(ctrl_metric_adata.X), dtype=np.float32).mean(axis=0)

    reference = average_of_perturbation_centroids(
        X=_utils.densify_X(reference_adata.X),
        conditions=reference_adata.obs["condition"].astype(str).map(condition_sort).values,
        ctrl_label="ctrl",
    )
    adata_control = single_adata[single_adata.obs["condition"].astype(str).map(condition_sort).eq("ctrl").values].copy()
    dataset_control = model.get_dataset(adata_control)

    full_condition_lookup = _condition_mean_lookup(full_adata)
    single_condition_lookup = _single_lookup_by_condition(single_adata)
    single_gene_lookup = _single_lookup_by_gene(single_adata)
    train_single_conditions = {
        c for c in {condition_sort(x) for x in train_conditions} if len(_condition_tokens_no_ctrl(c)) <= 1
    }
    train_double_conditions = {
        c for c in {condition_sort(x) for x in train_conditions} if len(_condition_tokens_no_ctrl(c)) > 1
    }
    train_single_genes = {
        _condition_tokens_no_ctrl(c)[0]
        for c in train_single_conditions
        if _condition_tokens_no_ctrl(c)
    }

    results: list[dict] = []
    export_payload: dict[str, dict] = {}
    for condition in sorted({condition_sort(c) for c in test_conditions if condition_sort(c) != "ctrl"}):
        cond_mask = test_cond_arr == condition
        if not np.any(cond_mask):
            warnings.warn(f"[biolord] skip condition missing in eval split: {condition}", RuntimeWarning, stacklevel=2)
            continue
        degs = np.asarray(top20.get(condition, []), dtype=int).reshape(-1)
        if degs.size == 0:
            warnings.warn(f"[biolord] skip condition without DEGs: {condition}", RuntimeWarning, stacklevel=2)
            continue

        true_expr = np.asarray(_utils.densify_X(test_adata[cond_mask].X), dtype=np.float32)
        if true_expr.ndim == 1:
            true_expr = true_expr.reshape(1, -1)
        pred_expr = _predict_condition_expression(
            model=model,
            condition=condition,
            cfg=cfg,
            single_adata=single_adata,
            dataset_control=dataset_control,
            train_single_conditions=train_single_conditions,
            train_double_conditions=train_double_conditions,
            train_single_genes=train_single_genes,
            single_condition_lookup=single_condition_lookup,
            single_gene_lookup=single_gene_lookup,
            full_condition_lookup=full_condition_lookup,
            ctrl_mean_for_combo=ctrl_mean_for_combo,
        )
        pred_expr = np.asarray(pred_expr, dtype=np.float32)
        if pred_expr.ndim == 1:
            pred_expr = pred_expr.reshape(1, -1)
        if pred_expr.shape[1] != true_expr.shape[1]:
            raise ValueError(f"BioLORD prediction shape mismatch for {condition}: {pred_expr.shape} vs {true_expr.shape}")

        mean_metrics = compute_mean_effect_metrics(
            X_true=true_expr,
            X_pred=pred_expr,
            X_ctrl=ctrl_metric,
            deg_idx=degs,
        )
        systema_metrics = pearson_delta_reference_metrics(
            X_true=true_expr.mean(axis=0),
            X_pred=pred_expr.mean(axis=0),
            reference=reference,
            top20_de_idxs=degs,
        )
        results.append(
            {
                "condition": condition,
                "condition_name": cond_names.get(condition, condition),
                **mean_metrics,
                "systema_corr_20de_allpert": float(systema_metrics["corr_20de_allpert"]),
                "systema_corr_deg_r2": float(systema_metrics["corr_deg_r2"]),
                **_nan_scpram_metrics(),
                "split_id": int(split_id),
                "n_ensemble": int(pred_expr.shape[0]),
                "n_eval_ctrl": int(ctrl_metric.shape[0]),
                "eval_ctrl_source": str(eval_ctrl_source),
                "prediction_ctrl_source": str(prediction_ctrl_source),
            }
        )

        sample_n = min(int(ctrl_metric.shape[0]), max(1, int(cfg.export_control_pool_size)))
        if sample_n < int(ctrl_metric.shape[0]):
            seed_base = (int(split_id) * 1000003) + sum(ord(ch) for ch in str(condition))
            rng = np.random.default_rng(seed_base)
            sample_idx = np.sort(rng.choice(int(ctrl_metric.shape[0]), size=sample_n, replace=False))
            ctrl_export = np.asarray(ctrl_metric[sample_idx], dtype=np.float32)
        else:
            ctrl_export = np.asarray(ctrl_metric, dtype=np.float32)
            sample_n = int(ctrl_export.shape[0])
        pred_export = np.repeat(np.asarray(pred_expr[:1], dtype=np.float32), repeats=sample_n, axis=0)
        export_item = {
            "Pred": pred_export[:, degs],
            "Ctrl": ctrl_export[:, degs],
            "Truth": true_expr[:, degs],
            "Pred_full": pred_export,
            "Ctrl_full": ctrl_export,
            "Truth_full": true_expr,
            "DE_idx": degs,
            "DE_name": gene_names[degs] if degs.size > 0 else np.array([], dtype=gene_names.dtype),
            "gene_name_full": gene_names,
            "export_metadata": {
                "model": "biolord",
                "split_id": int(split_id),
                "split_policy": str(split_policy),
                "eval_ctrl_source": str(eval_ctrl_source),
                "prediction_ctrl_source": str(prediction_ctrl_source),
                "source_payload_reused": False,
                "ordered_attribute_key": str(cfg.ordered_attribute_key),
                "go_attribute_source": "GEARS/PertData GO graph",
            },
        }
        export_payload[condition] = subset_payload_item(
            export_item,
            model_name="biolord",
            dataset=str(dataset_name),
            split_id=int(split_id),
            condition=str(condition),
            sample_size=int(cfg.export_control_pool_size),
        )
    return pd.DataFrame(results), export_payload


SCGEN_BIOLORD_SELF_ATTRIBUTE_KEY = "biolord_self_attribute"
SCGEN_BIOLORD_PRIOR_KEYS = {
    "emb_scgen_ifnb1_uniprot_prott5",
    "emb_scgen_ifnb1_zenodo_prott5",
    "emb_scgen_ifnb1_esm2_15b",
    "emb_scgen_ifnb1_genept",
    SCGEN_BIOLORD_SELF_ATTRIBUTE_KEY,
}


def _array_summary(x: np.ndarray) -> dict:
    arr = np.asarray(x, dtype=np.float32)
    return {
        "n_cells": int(arr.shape[0]) if arr.ndim == 2 else 0,
        "mean": arr.mean(axis=0).astype(np.float32)
        if arr.ndim == 2 and arr.shape[0] > 0
        else np.asarray([], dtype=np.float32),
        "var": arr.var(axis=0, ddof=1).astype(np.float32)
        if arr.ndim == 2 and arr.shape[0] > 1
        else (np.zeros(arr.shape[1], dtype=np.float32) if arr.ndim == 2 else np.asarray([], dtype=np.float32)),
    }


def _stable_condition_seed(base_seed: int, split_id: int, condition: str) -> int:
    raw = f"{int(base_seed)}::{int(split_id)}::{condition}"
    return int(sum((i + 1) * ord(ch) for i, ch in enumerate(raw)) % (2**32 - 1))


def _scgen_metric_summary(
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
        "split_policy": split_dict.get("split_policy"),
        "split_domain_key": split_dict.get("split_domain_key"),
        "train_domain_values": split_dict.get("train_domain_values"),
        "val_domain_values": split_dict.get("val_domain_values"),
        "test_domain_values": split_dict.get("test_domain_values"),
        "eval_ctrl_source": "target_domain_test_ctrl",
    }
    return metrics, full_summary


def _resolve_scgen_prior_path(prior_key: str) -> Path:
    paths_cfg = ROOT / "configs" / "paths.yaml"
    obj = load_yaml_file(paths_cfg) if paths_cfg.exists() else {}
    rel = (obj.get("embeddings") or {}).get(str(prior_key))
    if not rel:
        raise KeyError(f"Unknown scGen BioLORD prior key: {prior_key}")
    p = Path(str(rel))
    return p if p.is_absolute() else (ROOT / p).resolve()


def _load_scgen_biolord_attribute_vector(prior_key: str, condition: str) -> tuple[np.ndarray, str]:
    key = str(prior_key).strip()
    if key == SCGEN_BIOLORD_SELF_ATTRIBUTE_KEY:
        return np.asarray([1.0], dtype=np.float32), "self_generated_binary_condition"
    if key not in SCGEN_BIOLORD_PRIOR_KEYS:
        raise ValueError(f"Unsupported BioLORD scGen prior key={key!r}; choices={sorted(SCGEN_BIOLORD_PRIOR_KEYS)}")
    path = _resolve_scgen_prior_path(key)
    with path.open("rb") as handle:
        obj = pickle.load(handle)
    if str(condition) not in obj:
        raise KeyError(f"{path} does not contain condition key {condition!r}")
    vec = np.asarray(obj[str(condition)], dtype=np.float32).reshape(-1)
    if vec.size == 0 or not np.all(np.isfinite(vec)):
        raise ValueError(f"Invalid BioLORD scGen prior vector in {path}")
    return vec, str(path)


def _attach_scgen_biolord_attribute(
    adata: ad.AnnData,
    *,
    prior_key: str,
    attribute_key: str,
    condition: str,
) -> tuple[ad.AnnData, str]:
    out = _normalize_condition_obs(adata)
    out.obs = out.obs.copy()
    out.obs["perts_name"] = out.obs["condition"].astype(str).map(condition_sort).values
    vec, source = _load_scgen_biolord_attribute_vector(prior_key, condition)
    attr = np.zeros((out.n_obs, int(vec.shape[0])), dtype=np.float32)
    stim_mask = out.obs["condition"].astype(str).map(condition_sort).eq(str(condition)).values
    attr[stim_mask, :] = vec.reshape(1, -1)
    out.obsm[str(attribute_key)] = attr
    return out, source


def _scgen_split_dict(data: TriShiftData, *, split_id: int, condition: str) -> dict:
    return _split_celltype_seen_perturbation(
        data,
        seed=int(split_id),
        domain_key="cell_type",
        domain_test_ratio=0.2,
        val_domain_ratio=0.1,
        perturbation_condition=str(condition),
        include_test_ctrl_in_train=True,
    )


def _scgen_training_adata_from_split(
    full_attr_adata: ad.AnnData,
    split_dict: dict,
    split_id: int,
    attribute_key: str,
) -> tuple[ad.AnnData, str]:
    split_key = f"scgen_split{int(split_id)}"
    frames = [
        split_dict["train"].copy(),
        split_dict["val"].copy(),
        split_dict["test"][
            split_dict["test"].obs["condition"].astype(str).map(condition_sort).ne("ctrl").values
        ].copy(),
    ]
    labels = ["train", "test", "ood"]
    parts: list[ad.AnnData] = []
    attr_by_obs = {
        str(idx): full_attr_adata.obsm[str(attribute_key)][pos]
        for pos, idx in enumerate(full_attr_adata.obs_names.astype(str))
    }
    for frame, label in zip(frames, labels):
        part = frame.copy()
        part.obs[split_key] = label
        part.obsm[str(attribute_key)] = np.vstack([attr_by_obs[str(idx)] for idx in part.obs_names.astype(str)]).astype(np.float32)
        parts.append(part)
    out = ad.concat(parts, join="outer", merge="same", index_unique="-biolord")
    out.obs[split_key] = pd.Categorical(out.obs[split_key].astype(str), categories=["train", "test", "ood"])
    return out, split_key


def _predict_scgen_biolord(
    *,
    model,
    ctrl_adata: ad.AnnData,
    prior_key: str,
    attribute_key: str,
    condition: str,
) -> np.ndarray:
    base = ctrl_adata.copy()
    base.obs = base.obs.copy()
    base.obs["condition"] = str(condition)
    pred_adata, _ = _attach_scgen_biolord_attribute(
        base,
        prior_key=prior_key,
        attribute_key=attribute_key,
        condition=condition,
    )
    dataset_pred = model.get_dataset(pred_adata)
    with torch.no_grad():
        pred_expr, _ = model.module.get_expression(dataset_pred)
    return np.asarray(pred_expr.detach().cpu().numpy(), dtype=np.float32)


def run_biolord_scgen_pbmc_celltype_eval(
    *,
    prior_key: str = "emb_scgen_ifnb1_zenodo_prott5",
    base_seed: int = 24,
    export_notebook_pkl: bool = True,
    split_ids: list[int] | tuple[int, ...] | None = None,
    max_epochs: int | None = None,
    batch_size: int | None = None,
    n_latent: int | None = None,
    attribute_width: int | None = None,
    attribute_depth: int | None = None,
) -> None:
    name = "scgen_pbmc_celltype"
    condition = "stimulated"
    attribute_key = "biolord_attribute"
    splits = [int(x) for x in (split_ids if split_ids is not None else [1, 2, 3, 4, 5])]
    cfg = BiolordDatasetConfig(
        full_data_rel="data/scgen/perturb_processed.h5ad",
        single_data_rel="data/scgen/perturb_processed.h5ad",
        splits=splits,
        ordered_attribute_key=attribute_key,
        n_latent=int(32 if n_latent is None else n_latent),
        batch_size=int(64 if batch_size is None else batch_size),
        max_epochs=int(100 if max_epochs is None else max_epochs),
        attribute_nn_width=int(64 if attribute_width is None else attribute_width),
        attribute_nn_depth=int(6 if attribute_depth is None else attribute_depth),
        n_latent_attribute_ordered=512,
        reconstruction_penalty=1000.0,
        unknown_attribute_penalty=10000.0,
    )
    biolord_mod = _require_biolord_stack()
    full_path = _resolve_dataset_path(name, cfg.full_data_rel)
    full_raw = _normalize_condition_obs(ad.read_h5ad(full_path))
    full_attr_adata, attribute_source = _attach_scgen_biolord_attribute(
        full_raw,
        prior_key=str(prior_key),
        attribute_key=attribute_key,
        condition=condition,
    )
    data = TriShiftData(full_attr_adata, _dummy_embedding_df())
    top20 = _build_top20_lookup(data.adata_all)
    cond_names = _first_condition_name(data.adata_all)
    gene_names = _gene_names_from_adata(data.adata_all)
    out_dir = ROOT / "artifacts" / "results" / "biolord" / name / str(prior_key)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_all: list[pd.DataFrame] = []
    for split_id in cfg.splits:
        print(
            f"[biolord] dataset={name} split={split_id} prior_key={prior_key} max_epochs={cfg.max_epochs}",
            flush=True,
        )
        set_seeds(int(base_seed) + int(split_id))
        split_dict = _scgen_split_dict(data, split_id=int(split_id), condition=condition)
        train_adata, split_key = _scgen_training_adata_from_split(
            full_attr_adata,
            split_dict,
            int(split_id),
            attribute_key,
        )
        model = _build_model_and_train(
            biolord_mod=biolord_mod,
            train_adata=train_adata,
            split_key=split_key,
            cfg=cfg,
            seed=int(base_seed) + int(split_id),
        )
        test_adata = split_dict["test"]
        ctrl_mask = test_adata.obs["condition"].astype(str).map(condition_sort).eq("ctrl").values
        true_mask = test_adata.obs["condition"].astype(str).map(condition_sort).eq(condition).values
        ctrl_adata = test_adata[ctrl_mask].copy()
        true_expr = np.asarray(_utils.densify_X(test_adata[true_mask].X), dtype=np.float32)
        ctrl_expr = np.asarray(_utils.densify_X(ctrl_adata.X), dtype=np.float32)
        pred_expr = _predict_scgen_biolord(
            model=model,
            ctrl_adata=ctrl_adata,
            prior_key=str(prior_key),
            attribute_key=attribute_key,
            condition=condition,
        )
        degs = np.asarray(top20.get(condition, []), dtype=int).reshape(-1)
        if degs.size == 0:
            raise ValueError(f"Missing DEG list for condition={condition}")
        reference = average_of_perturbation_centroids(
            X=_utils.densify_X(split_dict["train"].X),
            conditions=split_dict["train"].obs["condition"].astype(str).map(condition_sort).values,
            ctrl_label="ctrl",
        )
        metrics, full_summary = _scgen_metric_summary(
            X_true=true_expr,
            X_pred=pred_expr,
            X_ctrl=ctrl_expr,
            degs=degs,
            condition=condition,
            split_dict=split_dict,
            split_id=int(split_id),
            base_seed=int(base_seed),
            systema_reference=reference,
        )
        metrics_df = pd.DataFrame(
            [
                {
                    "condition": condition,
                    "condition_name": cond_names.get(condition, condition),
                    **metrics,
                    "split_id": int(split_id),
                    "n_ensemble": int(pred_expr.shape[0]),
                    "n_eval_ctrl": int(ctrl_expr.shape[0]),
                    "eval_ctrl_source": "target_domain_test_ctrl",
                    "prediction_ctrl_source": "target_domain_test_ctrl",
                    "biolord_prior_key": str(prior_key),
                    "biolord_attribute_source": str(attribute_source),
                }
            ]
        )
        metrics_all.append(metrics_df)
        if export_notebook_pkl:
            export_item = {
                "Pred": pred_expr[:, degs],
                "Ctrl": ctrl_expr[:, degs],
                "Truth": true_expr[:, degs],
                "Pred_full": pred_expr,
                "Ctrl_full": ctrl_expr,
                "Truth_full": true_expr,
                "DE_idx": degs,
                "DE_name": gene_names[degs],
                "gene_name_full": gene_names,
                "export_metadata": {
                    "model": "biolord",
                    "dataset": name,
                    "export_is_subset": False,
                    "export_sample_size": None,
                    "metrics_computed_on_full": True,
                    "eval_ctrl_source": "target_domain_test_ctrl",
                    "prediction_ctrl_source": "target_domain_test_ctrl",
                    "split_id": int(split_id),
                    "split_policy": split_dict.get("split_policy"),
                    "split_domain_key": split_dict.get("split_domain_key"),
                    "train_domain_values": split_dict.get("train_domain_values"),
                    "val_domain_values": split_dict.get("val_domain_values"),
                    "test_domain_values": split_dict.get("test_domain_values"),
                    "ordered_attribute_key": attribute_key,
                    "biolord_prior_key": str(prior_key),
                    "biolord_attribute_source": str(attribute_source),
                },
                "full_summary": full_summary,
            }
            payload = {
                condition: subset_payload_item(
                    export_item,
                    model_name="biolord",
                    dataset=name,
                    split_id=int(split_id),
                    condition=condition,
                    sample_size=300,
                )
            }
            out_pkl = out_dir / f"biolord_{name}_{split_id}.pkl"
            with out_pkl.open("wb") as handle:
                pickle.dump(payload, handle)
            print(f"[biolord] saved notebook payload: {out_pkl}", flush=True)

    metrics_df_all = pd.concat(metrics_all, ignore_index=True)
    metrics_df_all.to_csv(out_dir / "metrics.csv", index=False)
    _write_mean_metrics(out_dir / "mean_pearson.txt", metrics_df_all)
    print(f"[biolord] saved metrics: {out_dir / 'metrics.csv'}", flush=True)


def _config_with_overrides(
    name: str,
    *,
    split_ids: list[int] | tuple[int, ...] | None = None,
    max_epochs: int | None = None,
    batch_size: int | None = None,
    n_latent: int | None = None,
    attribute_width: int | None = None,
    attribute_depth: int | None = None,
    export_control_pool_size: int | None = None,
) -> BiolordDatasetConfig:
    base = DATASET_CONFIG[name]
    return BiolordDatasetConfig(
        full_data_rel=base.full_data_rel,
        single_data_rel=base.single_data_rel,
        splits=[int(x) for x in (split_ids if split_ids is not None else base.splits)],
        test_ratio=float(base.test_ratio),
        ordered_attribute_key=str(base.ordered_attribute_key),
        n_latent=int(base.n_latent if n_latent is None else n_latent),
        batch_size=int(base.batch_size if batch_size is None else batch_size),
        max_epochs=int(base.max_epochs if max_epochs is None else max_epochs),
        early_stopping_patience=int(base.early_stopping_patience),
        check_val_every_n_epoch=int(base.check_val_every_n_epoch),
        attribute_nn_width=int(base.attribute_nn_width if attribute_width is None else attribute_width),
        attribute_nn_depth=int(base.attribute_nn_depth if attribute_depth is None else attribute_depth),
        attribute_dropout_rate=float(base.attribute_dropout_rate),
        n_latent_attribute_ordered=int(base.n_latent_attribute_ordered),
        n_latent_attribute_categorical=int(base.n_latent_attribute_categorical),
        reconstruction_penalty=float(base.reconstruction_penalty),
        unknown_attribute_penalty=float(base.unknown_attribute_penalty),
        decoder_width=int(base.decoder_width),
        decoder_depth=int(base.decoder_depth),
        decoder_activation=bool(base.decoder_activation),
        attribute_nn_activation=bool(base.attribute_nn_activation),
        unknown_attributes=bool(base.unknown_attributes),
        use_batch_norm=bool(base.use_batch_norm),
        use_layer_norm=bool(base.use_layer_norm),
        latent_lr=float(base.latent_lr),
        latent_wd=float(base.latent_wd),
        attribute_nn_lr=float(base.attribute_nn_lr),
        attribute_nn_wd=float(base.attribute_nn_wd),
        step_size_lr=int(base.step_size_lr),
        cosine_scheduler=bool(base.cosine_scheduler),
        scheduler_final_lr=float(base.scheduler_final_lr),
        decoder_lr=float(base.decoder_lr),
        decoder_wd=float(base.decoder_wd),
        unknown_attribute_noise_param=float(base.unknown_attribute_noise_param),
        export_control_pool_size=int(
            base.export_control_pool_size if export_control_pool_size is None else export_control_pool_size
        ),
        ctrl_test_ratio=float(base.ctrl_test_ratio),
        ctrl_val_ratio=float(base.ctrl_val_ratio),
        pert_val_ratio=float(base.pert_val_ratio),
    )


def run_biolord_eval(
    name: str,
    *,
    base_seed: int = 24,
    export_notebook_pkl: bool = True,
    split_ids: list[int] | tuple[int, ...] | None = None,
    max_epochs: int | None = None,
    batch_size: int | None = None,
    n_latent: int | None = None,
    attribute_width: int | None = None,
    attribute_depth: int | None = None,
    export_control_pool_size: int | None = None,
) -> None:
    if name not in DATASET_CONFIG:
        raise ValueError(f"Unknown BioLORD dataset: {name}")
    biolord_mod = _require_biolord_stack()
    cfg = _config_with_overrides(
        name,
        split_ids=split_ids,
        max_epochs=max_epochs,
        batch_size=batch_size,
        n_latent=n_latent,
        attribute_width=attribute_width,
        attribute_depth=attribute_depth,
        export_control_pool_size=export_control_pool_size,
    )
    full_path, single_path = _resolve_eval_data_paths(name, cfg)
    full_adata, _ = _prepare_full_eval_adata(full_path)
    single_adata_base = _prepare_single_train_adata(single_path, cfg.ordered_attribute_key)
    out_dir = ROOT / "artifacts" / "results" / "biolord" / name
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_all: list[pd.DataFrame] = []
    for split_id in cfg.splits:
        print(f"[biolord] dataset={name} split={split_id} mode=default max_epochs={cfg.max_epochs}", flush=True)
        set_seeds(int(base_seed) + int(split_id))
        split_key = f"split{int(split_id)}"
        if split_key not in single_adata_base.obs.columns:
            raise KeyError(f"BioLORD single adata is missing split column: {split_key}")
        if split_key not in full_adata.obs.columns:
            raise KeyError(f"BioLORD full adata is missing split column: {split_key}")
        single_adata = single_adata_base.copy()
        model = _build_model_and_train(
            biolord_mod=biolord_mod,
            train_adata=single_adata,
            split_key=split_key,
            cfg=cfg,
            seed=int(base_seed) + int(split_id),
        )
        train_conditions = _conditions_for_split_column(full_adata, split_key, "train")
        val_conditions = _conditions_for_split_column(full_adata, split_key, "test")
        test_conditions = _conditions_for_split_column(full_adata, split_key, "ood")
        ctrl_adata = full_adata[full_adata.obs["condition"].astype(str).map(condition_sort).eq("ctrl").values]
        eval_mask = full_adata.obs["condition"].astype(str).map(condition_sort).isin(set(test_conditions)).values
        eval_adata = full_adata[eval_mask]
        ref_mask = full_adata.obs["condition"].astype(str).map(condition_sort).isin(set(train_conditions + val_conditions)).values
        reference_adata = full_adata[ref_mask | full_adata.obs["condition"].astype(str).map(condition_sort).eq("ctrl").values]
        metrics_df, export_payload = _metric_and_payload_for_conditions(
            model=model,
            full_adata=full_adata,
            single_adata=single_adata,
            cfg=cfg,
            dataset_name=name,
            split_id=int(split_id),
            test_adata=eval_adata,
            ctrl_metric_adata=ctrl_adata,
            reference_adata=reference_adata,
            test_conditions=test_conditions,
            train_conditions=train_conditions,
            split_policy="biolord_preprocessed_condition_split",
            eval_ctrl_source="all_ctrl",
            prediction_ctrl_source="single_ctrl_mean",
        )
        metrics_df = _attach_subgroup_column(metrics_df, _subgroup_frame_from_full_adata(full_adata, int(split_id)))
        metrics_all.append(metrics_df)
        if export_notebook_pkl:
            out_pkl = out_dir / f"biolord_{name}_{split_id}.pkl"
            with out_pkl.open("wb") as handle:
                pickle.dump(export_payload, handle)
            print(f"[biolord] saved notebook payload: {out_pkl}", flush=True)

    metrics_df_all = pd.concat(metrics_all, ignore_index=True)
    metrics_df_all.to_csv(out_dir / "metrics.csv", index=False)
    _write_mean_metrics(out_dir / "mean_pearson.txt", metrics_df_all)
    print(f"[biolord] saved metrics: {out_dir / 'metrics.csv'}", flush=True)


def run_biolord_unseen_ctrl_eval(
    name: str,
    *,
    base_seed: int = 24,
    export_notebook_pkl: bool = True,
    split_ids: list[int] | tuple[int, ...] | None = None,
    max_epochs: int | None = None,
    batch_size: int | None = None,
    n_latent: int | None = None,
    attribute_width: int | None = None,
    attribute_depth: int | None = None,
    export_control_pool_size: int | None = None,
) -> None:
    if name not in DATASET_CONFIG:
        raise ValueError(f"Unknown BioLORD dataset: {name}")
    biolord_mod = _require_biolord_stack()
    cfg = _config_with_overrides(
        name,
        split_ids=split_ids,
        max_epochs=max_epochs,
        batch_size=batch_size,
        n_latent=n_latent,
        attribute_width=attribute_width,
        attribute_depth=attribute_depth,
        export_control_pool_size=export_control_pool_size,
    )
    full_path, single_path = _resolve_eval_data_paths(name, cfg)
    full_adata, data = _prepare_full_eval_adata(full_path)
    single_adata_base = _prepare_single_train_adata(single_path, cfg.ordered_attribute_key)
    out_dir = ROOT / "artifacts" / "results" / "biolord" / name
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_all: list[pd.DataFrame] = []
    for split_id in cfg.splits:
        print(f"[biolord] dataset={name} split={split_id} mode=unseen_ctrl max_epochs={cfg.max_epochs}", flush=True)
        set_seeds(int(base_seed) + int(split_id))
        split_dict, subgroup_df = split_unseen_ctrl_unseen_perturbation(
            data,
            name,
            seed=int(split_id),
            test_ratio=float(cfg.test_ratio),
            pert_val_ratio=float(cfg.pert_val_ratio),
            ctrl_test_ratio=float(cfg.ctrl_test_ratio),
            ctrl_val_ratio=float(cfg.ctrl_val_ratio),
        )
        train_val_adata = ad.concat([split_dict["train"], split_dict["val"]], join="outer", merge="same")
        train_val_ctrl = train_val_adata[
            train_val_adata.obs["condition"].astype(str).map(condition_sort).eq("ctrl").values
        ]
        if train_val_ctrl.n_obs == 0:
            raise ValueError("BioLORD unseen-control split has no train/val ctrl cells")
        ctrl_trainval_mean = np.asarray(_utils.densify_X(train_val_ctrl.X), dtype=np.float32).mean(axis=0)
        single_adata_ctrl = _set_single_ctrl_expression(single_adata_base, ctrl_trainval_mean)
        single_adata, split_key = _apply_unseen_split_to_single_adata(single_adata_ctrl, split_dict, int(split_id))
        model = _build_model_and_train(
            biolord_mod=biolord_mod,
            train_adata=single_adata,
            split_key=split_key,
            cfg=cfg,
            seed=int(base_seed) + int(split_id),
        )
        test_ctrl = split_dict["test"][
            split_dict["test"].obs["condition"].astype(str).map(condition_sort).eq("ctrl").values
        ]
        metrics_df, export_payload = _metric_and_payload_for_conditions(
            model=model,
            full_adata=full_adata,
            single_adata=single_adata,
            cfg=cfg,
            dataset_name=name,
            split_id=int(split_id),
            test_adata=split_dict["test"],
            ctrl_metric_adata=test_ctrl,
            reference_adata=train_val_adata,
            test_conditions=[condition_sort(c) for c in split_dict.get("test_conds", [])],
            train_conditions=[condition_sort(c) for c in split_dict.get("train_conds", [])],
            split_policy=str(split_dict.get("split_policy", "unseen_ctrl_unseen_perturbation")),
            eval_ctrl_source="target_domain_test_ctrl",
            prediction_ctrl_source="train_val_ctrl_mean",
        )
        metrics_df = _attach_subgroup_column(metrics_df, _subgroup_map_from_df(subgroup_df))
        metrics_all.append(metrics_df)
        if export_notebook_pkl:
            out_pkl = out_dir / f"biolord_{name}_{split_id}_unseen_ctrl.pkl"
            with out_pkl.open("wb") as handle:
                pickle.dump(export_payload, handle)
            print(f"[biolord] saved notebook payload: {out_pkl}", flush=True)

    metrics_df_all = pd.concat(metrics_all, ignore_index=True)
    metrics_df_all.to_csv(out_dir / "metrics_unseen_ctrl.csv", index=False)
    _write_mean_metrics(out_dir / "mean_pearson_unseen_ctrl.txt", metrics_df_all)
    print(f"[biolord] saved metrics: {out_dir / 'metrics_unseen_ctrl.csv'}", flush=True)


def _split_ids_from_string(value: str) -> list[int] | None:
    raw = str(value).strip()
    if not raw:
        return None
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run BioLORD training + eval with TriShift metrics")
    parser.add_argument("--profile", default="", help="dataset profile under scripts/biolord/eval/configs")
    parser.add_argument("--name", default="", help="dataset name")
    parser.add_argument("--seed", type=int, default=24, help="base seed")
    parser.add_argument("--splits", default="", help="comma-separated split ids; default uses dataset config")
    parser.add_argument("--max_epochs", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=0)
    parser.add_argument("--n_latent", type=int, default=0)
    parser.add_argument("--attribute_width", type=int, default=0)
    parser.add_argument("--attribute_depth", type=int, default=0)
    parser.add_argument("--export_control_pool_size", type=int, default=0)
    parser.add_argument("--no_export_notebook_pkl", action="store_true")
    parser.add_argument(
        "--unseen_ctrl_eval",
        action="store_true",
        help="run held-out ctrl/unseen perturbation evaluation without overwriting default metrics",
    )
    args = parser.parse_args(argv)

    profile = str(args.profile).strip()
    if profile:
        prof = _load_profile(profile)
        task_args = prof["task_args"]
        name = str(prof["dataset"])
        base_seed = int(task_args.get("seed", 24)) if int(args.seed) == 24 else int(args.seed)
    else:
        name = str(args.name).strip()
        task_args = {}
        base_seed = int(args.seed)
    if not name:
        raise SystemExit("Provide --profile or --name")

    kwargs = {
        "base_seed": base_seed,
        "export_notebook_pkl": False
        if bool(args.no_export_notebook_pkl)
        else bool(task_args.get("export_notebook_pkl", True)),
        "split_ids": _split_ids_from_string(str(args.splits)),
        "max_epochs": (int(task_args.get("max_epochs", 0)) if int(args.max_epochs) == 0 else int(args.max_epochs)) or None,
        "batch_size": (int(task_args.get("batch_size", 0)) if int(args.batch_size) == 0 else int(args.batch_size)) or None,
        "n_latent": (int(task_args.get("n_latent", 0)) if int(args.n_latent) == 0 else int(args.n_latent)) or None,
        "attribute_width": (
            int(task_args.get("attribute_width", 0))
            if int(args.attribute_width) == 0
            else int(args.attribute_width)
        )
        or None,
        "attribute_depth": (
            int(task_args.get("attribute_depth", 0))
            if int(args.attribute_depth) == 0
            else int(args.attribute_depth)
        )
        or None,
        "export_control_pool_size": (
            int(task_args.get("export_control_pool_size", 0))
            if int(args.export_control_pool_size) == 0
            else int(args.export_control_pool_size)
        )
        or None,
    }
    if bool(args.unseen_ctrl_eval):
        run_biolord_unseen_ctrl_eval(name, **kwargs)
    else:
        run_biolord_eval(name, **kwargs)


if __name__ == "__main__":
    main()
