from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import pickle
import random
import sys

import anndata as ad
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = ROOT / "src"
LOCAL_DATA_ROOT = ROOT / "src" / "data"
EXTERNAL_SCPRAM_ROOT = ROOT / "external" / "scPRAM-main"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(0, str(EXTERNAL_SCPRAM_ROOT))

from trishift import _utils
from trishift._external_metrics import (
    average_of_perturbation_centroids,
    compute_distributional_systema_metrics_from_arrays,
    compute_mean_effect_metrics,
    compute_scpram_metrics_bundle_from_arrays,
    pearson_delta_reference_metrics,
)
from trishift.TriShiftData import TriShiftData
from scripts.biolord._core.biolord_eval_core import (
    _build_top20_lookup,
    _first_condition_name,
    _write_mean_metrics,
)
from scripts.common.split_utils import condition_sort
from scripts.common.yaml_utils import load_yaml_file
from scripts.trishift._core.run_dataset_core import _split_celltype_seen_perturbation


@dataclass(frozen=True)
class ScpramDatasetConfig:
    data_rel: str
    splits: list[int]
    perturbation_condition: str = "stimulated"
    domain_key: str = "cell_type"
    domain_test_ratio: float = 0.2
    val_domain_ratio: float = 0.1
    include_test_ctrl_in_train: bool = True
    epochs: int = 100
    batch_size: int = 128
    lr: float = 5e-4
    latent_dim: int = 100
    hidden_dim: int = 1000
    noise_rate: float = 0.1
    kl_weight: float = 5e-4
    export_notebook_pkl: bool = True


DATASET_CONFIG: dict[str, ScpramDatasetConfig] = {
    "scgen_pbmc_celltype": ScpramDatasetConfig(
        data_rel="data/scgen/perturb_processed.h5ad",
        splits=[1, 2, 3, 4, 5],
    ),
}


def _dummy_embedding_df() -> pd.DataFrame:
    return pd.DataFrame([[0.0]], index=["ctrl"], columns=["dummy"])


def _resolve_dataset_path(name: str, data_rel: str) -> Path:
    primary = (LOCAL_DATA_ROOT / Path(data_rel).relative_to("data")).resolve()
    if primary.exists():
        return primary
    paths_cfg = ROOT / "configs" / "paths.yaml"
    if paths_cfg.exists():
        obj = load_yaml_file(paths_cfg)
        alt = obj.get("datasets", {}).get(name)
        if alt:
            p = Path(str(alt))
            p = p if p.is_absolute() else (ROOT / p)
            if p.exists():
                return p.resolve()
    raise FileNotFoundError(f"Missing scPRAM input for dataset={name}. Tried {primary}")


def _normalize_adata(adata: ad.AnnData) -> ad.AnnData:
    out = adata.copy()
    out.obs = out.obs.copy()
    out.obs["condition"] = out.obs["condition"].astype(str).map(condition_sort).astype("category")
    if "cell_type" not in out.obs.columns:
        raise KeyError("scPRAM scGen evaluation requires obs['cell_type']")
    out.obs["cell_type"] = out.obs["cell_type"].astype(str).astype("category")
    return out


def _require_scpram_stack():
    try:
        from scpram.models import SCPRAM  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "scPRAM is not importable. Populate external/scPRAM-main or install the scpram package first."
        ) from exc
    return SCPRAM


def set_seeds(seed: int) -> None:
    np.random.seed(int(seed))
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


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


def _split_scgen(data: TriShiftData, cfg: ScpramDatasetConfig, split_id: int) -> dict:
    return _split_celltype_seen_perturbation(
        data,
        seed=int(split_id),
        domain_key=str(cfg.domain_key),
        domain_test_ratio=float(cfg.domain_test_ratio),
        val_domain_ratio=float(cfg.val_domain_ratio),
        perturbation_condition=str(cfg.perturbation_condition),
        include_test_ctrl_in_train=bool(cfg.include_test_ctrl_in_train),
    )


def _predict_split(
    *,
    model,
    split_dict: dict,
    cfg: ScpramDatasetConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    key_dic = {
        "condition_key": "condition",
        "cell_type_key": str(cfg.domain_key),
        "ctrl_key": "ctrl",
        "stim_key": str(cfg.perturbation_condition),
        "pred_key": "pred",
    }
    pred_rows: list[np.ndarray] = []
    ctrl_rows: list[np.ndarray] = []
    true_rows: list[np.ndarray] = []
    train_adata = split_dict["train"].copy()
    test_adata = split_dict["test"]
    for domain in list(split_dict.get("test_domain_values", [])):
        pred_adata = model.cross_cell_predict(train_adata, str(domain), key_dic)
        pred_rows.append(np.asarray(pred_adata.X, dtype=np.float32))
        ctrl_mask = (
            test_adata.obs[str(cfg.domain_key)].astype(str).eq(str(domain)).values
            & test_adata.obs["condition"].astype(str).map(condition_sort).eq("ctrl").values
        )
        true_mask = (
            test_adata.obs[str(cfg.domain_key)].astype(str).eq(str(domain)).values
            & test_adata.obs["condition"].astype(str).map(condition_sort).eq(str(cfg.perturbation_condition)).values
        )
        ctrl_rows.append(np.asarray(_utils.densify_X(test_adata[ctrl_mask].X), dtype=np.float32))
        true_rows.append(np.asarray(_utils.densify_X(test_adata[true_mask].X), dtype=np.float32))
    if not pred_rows:
        raise ValueError("scPRAM split produced no predictions")
    return np.vstack(pred_rows), np.vstack(ctrl_rows), np.vstack(true_rows)


def _config_with_overrides(
    name: str,
    *,
    split_ids: list[int] | tuple[int, ...] | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    latent_dim: int | None = None,
    hidden_dim: int | None = None,
    lr: float | None = None,
    export_notebook_pkl: bool | None = None,
) -> ScpramDatasetConfig:
    base = DATASET_CONFIG[name]
    return ScpramDatasetConfig(
        data_rel=base.data_rel,
        splits=[int(x) for x in (split_ids if split_ids is not None else base.splits)],
        perturbation_condition=str(base.perturbation_condition),
        domain_key=str(base.domain_key),
        domain_test_ratio=float(base.domain_test_ratio),
        val_domain_ratio=float(base.val_domain_ratio),
        include_test_ctrl_in_train=bool(base.include_test_ctrl_in_train),
        epochs=int(base.epochs if epochs is None else epochs),
        batch_size=int(base.batch_size if batch_size is None else batch_size),
        lr=float(base.lr if lr is None else lr),
        latent_dim=int(base.latent_dim if latent_dim is None else latent_dim),
        hidden_dim=int(base.hidden_dim if hidden_dim is None else hidden_dim),
        noise_rate=float(base.noise_rate),
        kl_weight=float(base.kl_weight),
        export_notebook_pkl=bool(base.export_notebook_pkl if export_notebook_pkl is None else export_notebook_pkl),
    )


def run_scpram_eval(
    name: str,
    *,
    base_seed: int = 24,
    split_ids: list[int] | tuple[int, ...] | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    latent_dim: int | None = None,
    hidden_dim: int | None = None,
    lr: float | None = None,
    export_notebook_pkl: bool | None = None,
) -> None:
    if name not in DATASET_CONFIG:
        raise ValueError(f"Unknown scPRAM dataset: {name}")
    SCPRAM = _require_scpram_stack()
    cfg = _config_with_overrides(
        name,
        split_ids=split_ids,
        epochs=epochs,
        batch_size=batch_size,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        lr=lr,
        export_notebook_pkl=export_notebook_pkl,
    )
    adata = _normalize_adata(ad.read_h5ad(_resolve_dataset_path(name, cfg.data_rel)))
    data = TriShiftData(adata, _dummy_embedding_df())
    top20 = _build_top20_lookup(data.adata_all)
    cond_names = _first_condition_name(data.adata_all)
    gene_names = (
        data.adata_all.var["gene_name"].astype(str).values
        if "gene_name" in data.adata_all.var.columns
        else data.adata_all.var_names.astype(str).values
    )
    out_dir = ROOT / "artifacts" / "results" / "scpram" / name
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_all: list[pd.DataFrame] = []
    for split_id in cfg.splits:
        print(f"[scpram] dataset={name} split={split_id} epochs={cfg.epochs}", flush=True)
        set_seeds(int(base_seed) + int(split_id))
        split_dict = _split_scgen(data, cfg, int(split_id))
        model = SCPRAM(
            input_dim=int(data.adata_all.n_vars),
            latent_dim=int(cfg.latent_dim),
            hidden_dim=int(cfg.hidden_dim),
            noise_rate=float(cfg.noise_rate),
            kl_weight=float(cfg.kl_weight),
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
        model.to(model.device)
        model.train_SCPRAM(
            split_dict["train"].copy(),
            epochs=int(cfg.epochs),
            batch_size=int(cfg.batch_size),
            lr=float(cfg.lr),
        )
        condition = str(cfg.perturbation_condition)
        pred, ctrl, true = _predict_split(model=model, split_dict=split_dict, cfg=cfg)
        degs = np.asarray(top20.get(condition, []), dtype=int).reshape(-1)
        if degs.size == 0:
            raise ValueError(f"Missing DEG list for condition={condition}")
        reference = average_of_perturbation_centroids(
            X=_utils.densify_X(split_dict["train"].X),
            conditions=split_dict["train"].obs["condition"].astype(str).map(condition_sort).values,
            ctrl_label="ctrl",
        )
        metrics, full_summary = _condition_metric_summary(
            X_true=true,
            X_pred=pred,
            X_ctrl=ctrl,
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
                    "n_ensemble": int(pred.shape[0]),
                    "n_eval_ctrl": int(ctrl.shape[0]),
                    "eval_ctrl_source": "target_domain_test_ctrl",
                    "prediction_ctrl_source": "scpram_cross_cell_target_ctrl",
                }
            ]
        )
        metrics_all.append(metrics_df)
        if cfg.export_notebook_pkl:
            payload = {
                condition: {
                    "Pred": pred[:, degs],
                    "Ctrl": ctrl[:, degs],
                    "Truth": true[:, degs],
                    "Pred_full": pred,
                    "Ctrl_full": ctrl,
                    "Truth_full": true,
                    "DE_idx": degs,
                    "DE_name": gene_names[degs],
                    "gene_name_full": gene_names,
                    "export_metadata": {
                        "model": "scpram",
                        "dataset": str(name),
                        "export_is_subset": False,
                        "export_sample_size": None,
                        "metrics_computed_on_full": True,
                        "eval_ctrl_source": "target_domain_test_ctrl",
                        "prediction_ctrl_source": "scpram_cross_cell_target_ctrl",
                        "split_id": int(split_id),
                        "split_policy": split_dict.get("split_policy"),
                        "split_domain_key": split_dict.get("split_domain_key"),
                        "train_domain_values": split_dict.get("train_domain_values"),
                        "val_domain_values": split_dict.get("val_domain_values"),
                        "test_domain_values": split_dict.get("test_domain_values"),
                    },
                    "full_summary": full_summary,
                }
            }
            out_pkl = out_dir / f"scpram_{name}_{split_id}.pkl"
            with out_pkl.open("wb") as handle:
                pickle.dump(payload, handle)
            print(f"[scpram] saved notebook payload: {out_pkl}", flush=True)

    metrics_all_df = pd.concat(metrics_all, ignore_index=True)
    metrics_all_df.to_csv(out_dir / "metrics.csv", index=False)
    _write_mean_metrics(out_dir / "mean_pearson.txt", metrics_all_df)
    print(f"[scpram] saved metrics: {out_dir / 'metrics.csv'}", flush=True)


def _split_ids_from_string(value: str) -> list[int] | None:
    raw = str(value).strip()
    if not raw:
        return None
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run scPRAM training + eval with TriShift metrics")
    parser.add_argument("--name", default="scgen_pbmc_celltype")
    parser.add_argument("--seed", type=int, default=24)
    parser.add_argument("--splits", default="")
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=0)
    parser.add_argument("--latent_dim", type=int, default=0)
    parser.add_argument("--hidden_dim", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.0)
    parser.add_argument("--no_export_notebook_pkl", action="store_true")
    args = parser.parse_args(argv)
    run_scpram_eval(
        str(args.name),
        base_seed=int(args.seed),
        split_ids=_split_ids_from_string(str(args.splits)),
        epochs=int(args.epochs) or None,
        batch_size=int(args.batch_size) or None,
        latent_dim=int(args.latent_dim) or None,
        hidden_dim=int(args.hidden_dim) or None,
        lr=float(args.lr) or None,
        export_notebook_pkl=not bool(args.no_export_notebook_pkl),
    )


if __name__ == "__main__":
    main()
