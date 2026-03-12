from __future__ import annotations

from pathlib import Path
import argparse
from dataclasses import dataclass
import sys
import pickle
import random
import json
import hashlib
import subprocess
from datetime import datetime, timezone

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch
import pandas as pd
import numpy as np
import yaml

from trishift._utils import (
    load_yaml,
    set_seeds,
    load_adata,
    load_embedding_df,
    apply_alias_mapping,
)
from trishift.TriShiftData import TriShiftData
from trishift.TriShift import TriShift
from scripts.common.split_utils import (
    norman_subgroup as _shared_norman_subgroup,
    split_list_by_ratio as _shared_split_list_by_ratio,
)


DATASET_CONFIG = {
    "adamson": {
        "emb_key": "emb_b",
        "defaults": {"stage1": 10, "stage23": 6, "stage2": 3, "stage3": 3},
        "multi_split_default": 5,
        "test_ratio": 0.2,
    },
    "dixit": {
        "emb_key": "emb_c",
        "defaults": {"stage1": 5, "stage23": 10, "stage2": 5, "stage3": 5},
        "multi_split_default": 10,
        # Align split policy with Scouter for Dixit.
        "test_ratio": 0.1,
    },
    "norman": {
        "emb_key": "emb_a",
        "defaults": {"stage1": 5, "stage23": 10, "stage2": 5, "stage3": 5},
        "multi_split_default": 5,
        "test_ratio": 0.2,
    },
    "replogle_k562_essential": {
        "emb_key": "emb_c",
        "defaults": {"stage1": 5, "stage23": 10, "stage2": 5, "stage3": 5},
        "multi_split_default": 5,
        "test_ratio": 0.2,
    },
    "replogle_rpe1_essential": {
        "emb_key": "emb_b",
        "defaults": {"stage1": 5, "stage23": 10, "stage2": 5, "stage3": 5},
        "multi_split_default": 5,
        "test_ratio": 0.2,
    },
}

DEGS_CACHE_KEYS = ("top20_degs_non_dropout", "top20_degs_non_zero", "top20_degs", "degs_meta")
DEGS_CONDITION_KEYS = ("top20_degs_non_dropout", "top20_degs_non_zero", "top20_degs")
DEGS_EXPECTED_METHOD = "t-test"


@dataclass(frozen=True)
class PerfConfig:
    amp: bool
    num_workers: int
    pin_memory: bool
    grad_accum_steps: int


@dataclass(frozen=True)
class LossConfig:
    gamma: float
    lambda_dir: float
    lambda_dir_expr: float
    lambda_dir_z: float
    lambda_z: float
    deg_weight: float
    lambda_expr_mse: float


@dataclass(frozen=True)
class StageSchedule:
    sched_gamma: float
    patience: int
    min_delta: float


def _get_cfg_or(cfg: dict, key: str, default: int) -> int:
    """Fetch integer config value with a default fallback."""
    return int(cfg.get(key, default))


def _extract_perf(defaults: dict) -> PerfConfig:
    """Extract performance-related config values."""
    perf = defaults.get("performance", {})
    return PerfConfig(
        amp=bool(perf.get("amp", True)),
        num_workers=int(perf.get("num_workers", 0)),
        pin_memory=bool(perf.get("pin_memory", True)),
        grad_accum_steps=int(perf.get("grad_accum_steps", 1)),
    )


def _extract_loss(defaults: dict) -> LossConfig:
    """Extract loss-related config values."""
    loss_cfg = defaults.get("loss", {})
    gamma = float(loss_cfg.get("gamma", 1.0))
    lambda_dir = float(loss_cfg.get("lambda_dir", 1.0))
    return LossConfig(
        gamma=gamma,
        lambda_dir=lambda_dir,
        lambda_dir_expr=float(loss_cfg.get("lambda_dir_expr", lambda_dir)),
        lambda_dir_z=float(loss_cfg.get("lambda_dir_z", lambda_dir)),
        lambda_z=float(loss_cfg.get("lambda_z", 1.0)),
        deg_weight=float(loss_cfg.get("deg_weight", 1.0)),
        lambda_expr_mse=float(loss_cfg.get("lambda_expr_mse", 0.0)),
    )


def _extract_schedule(cfg: dict) -> StageSchedule:
    """Extract scheduling/early-stop settings for a training stage."""
    return StageSchedule(
        sched_gamma=float(cfg.get("sched_gamma", 0.9)),
        patience=int(cfg.get("patience", 5)),
        min_delta=float(cfg.get("min_delta", 1e-3)),
    )


def _resolve_mean_metric_keys(numeric_means: pd.Series) -> list[str]:
    preferred_order = [
        "pearson",
        "nmse",
        "mse_pred",
        "mse_ctrl",
        "deg_mean_r2",
        "systema_corr_all_allpert",
        "systema_corr_20de_allpert",
        "systema_corr_all_r2",
        "systema_corr_deg_r2",
        "r2_degs_var_mean",
        "r2_all_var_mean",
        "scpram_r2_all_mean_mean",
        "scpram_r2_all_var_mean",
        "scpram_r2_degs_mean_mean",
        "scpram_r2_degs_var_mean",
        "scpram_r2_all_mean_std",
        "scpram_r2_all_var_std",
        "scpram_r2_degs_mean_std",
        "scpram_r2_degs_var_std",
        "scpram_wasserstein_all_sum",
        "scpram_wasserstein_degs_sum",
    ]
    exclude_keys = {"split_id", "n_ensemble"}
    keys = [k for k in preferred_order if k in numeric_means.index and k not in exclude_keys]
    keys.extend(
        [
            k
            for k in numeric_means.index
            if k not in exclude_keys and k not in keys
        ]
    )
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
            val = float(sub_numeric_means[key])
            lines.append(f"subgroup_{g}_mean_{key}={val}\n")
        lines.append("\n")


def _attach_subgroup_column(
    metrics_df: pd.DataFrame,
    subgroup_df: pd.DataFrame | None,
) -> pd.DataFrame:
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
    """Write mean pearson and other averaged numeric metrics to text file.

    Backward compatibility:
    - First line remains plain mean pearson float for old readers.
    """
    numeric_means = metrics_df.mean(numeric_only=True)
    mean_pearson = float(numeric_means.get("pearson", float("nan")))

    keys = _resolve_mean_metric_keys(numeric_means)

    lines = [f"{mean_pearson}\n"]
    for key in keys:
        val = float(numeric_means[key])
        lines.append(f"mean_{key}={val}\n")
    _append_subgroup_mean_lines(lines, metrics_df, keys)

    path.write_text("".join(lines), encoding="utf-8")


def _safe_git_commit() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        )
    except Exception:
        return None
    return out.decode("utf-8", "replace").strip() or None


def _dump_yaml(path: Path, obj: dict) -> None:
    path.write_text(
        yaml.safe_dump(obj, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )


def _resolve_cond_dims_for_pool_mode(
    data: TriShiftData,
    embd_df: pd.DataFrame,
    cond_pool_mode: str,
) -> int:
    """Resolve condition vector dimension from pool mode."""
    emb_dim = int(embd_df.shape[1])
    mode = str(cond_pool_mode)
    if mode in {"sum", "mean"}:
        return emb_dim
    raise ValueError("cond_pool_mode must be one of: sum, mean")


def _resolve_eval_compare_modes(eval_genept_compare_mode: str) -> list[str]:
    mode = str(eval_genept_compare_mode)
    if mode == "all":
        # compare_mode=all evaluates both compare algorithms.
        return ["aggregate_cond", "per_gene_nearest_cond"]
    return [mode]


def _resolve_eval_candidate_modes(
    eval_genept_compare_mode: str,
    eval_genept_train_candidate_mode: str,
) -> list[str]:
    if str(eval_genept_compare_mode) == "all":
        return [
            "all_train_pert",
            "norman_train_single_only",
            "norman_single_nearest_else_random",
        ]
    return [str(eval_genept_train_candidate_mode)]


def _load_z_mu_cache(cache_path: Path, n_obs: int) -> np.ndarray | None:
    if not cache_path.exists():
        return None
    try:
        cached = np.load(cache_path, allow_pickle=False)
    except Exception as exc:
        print(f"[z_mu] failed to load cache: {cache_path} ({exc})")
        return None
    if "z_mu" not in cached:
        print(f"[z_mu] invalid cache payload (missing z_mu): {cache_path}")
        return None
    z_mu = np.asarray(cached["z_mu"])
    if z_mu.ndim != 2 or z_mu.shape[0] != n_obs:
        print(
            f"[z_mu] cache shape mismatch; rebuilding: path={cache_path}, "
            f"cached={z_mu.shape}, expected=({n_obs}, z_dim)"
        )
        return None
    return z_mu


def _save_z_mu_cache(cache_path: Path, z_mu: np.ndarray) -> None:
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, z_mu=np.asarray(z_mu))
    except Exception as exc:
        print(f"[z_mu] failed to save cache: {cache_path} ({exc})")


def _init_model(
    data: TriShiftData,
    adata,
    embd_df: pd.DataFrame,
    stage1_model_cfg: dict,
    stage2_model_cfg: dict,
    model_cfg: dict,
    *,
    predict_shift: bool = True,
    shift_input_source_override: str | None = None,
    gen_state_source: str = "compressor",
) -> TriShift:
    """Initialize TriShift model and return it."""
    gen_encoder_hidden = model_cfg.get("encoder_hidden")
    gen_state_dim = model_cfg.get("state_dim")
    gen_decoder_hidden = model_cfg.get("decoder_hidden")
    gen_input_mode = str(model_cfg.get("input_mode", "full"))
    gen_use_batch_norm = bool(model_cfg.get("use_batch_norm", True))
    gen_use_layer_norm = bool(model_cfg.get("use_layer_norm", False))
    gen_dropout = float(model_cfg.get("dropout", 0.0))
    gen_use_residual_head = bool(model_cfg.get("use_residual_head", False))
    shift_hidden = [int(h) for h in stage2_model_cfg.get("shift_hidden", [256, 256])]
    shift_predict_delta = bool(stage2_model_cfg.get("predict_delta", True))
    shift_repr_dim_raw = stage2_model_cfg.get("shift_repr_dim", None)
    if shift_repr_dim_raw is None:
        shift_repr_dim = None
    else:
        try:
            shift_repr_dim = int(shift_repr_dim_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "model.stage2.shift_repr_dim must be a positive integer or null"
            ) from exc
        if shift_repr_dim <= 0:
            raise ValueError("model.stage2.shift_repr_dim must be a positive integer")
    if shift_predict_delta and shift_repr_dim is not None:
        raise ValueError(
            "model.stage2.shift_repr_dim is only valid when model.stage2.predict_delta=false"
        )
    shift_use_cross_attention = bool(stage2_model_cfg.get("use_cross_attention", False))
    shift_cross_attn_heads = int(stage2_model_cfg.get("cross_attn_heads", 4))
    shift_cross_attn_dropout = float(stage2_model_cfg.get("cross_attn_dropout", 0.0))
    shift_use_transformer_block = bool(stage2_model_cfg.get("use_transformer_block", False))
    shift_transformer_layers = int(stage2_model_cfg.get("transformer_layers", 1))
    shift_transformer_ff_mult = int(stage2_model_cfg.get("transformer_ff_mult", 4))
    shift_transformer_dropout = float(stage2_model_cfg.get("transformer_dropout", 0.0))
    shift_transformer_readout = str(stage2_model_cfg.get("transformer_readout", "first"))
    if not bool(predict_shift):
        shift_predict_delta = False
        shift_repr_dim = None
    shift_input_source = (
        str(shift_input_source_override)
        if shift_input_source_override is not None
        else str(stage2_model_cfg.get("shift_input_source", "latent_mu"))
    )
    cond_pool_mode = str(stage2_model_cfg.get("cond_pool_mode", "sum"))
    cond_l2_norm = bool(stage2_model_cfg.get("cond_l2_norm", False))
    cond_dim_eff = _resolve_cond_dims_for_pool_mode(
        data=data,
        embd_df=embd_df,
        cond_pool_mode=cond_pool_mode,
    )
    stage1_hidden_dim = int(stage1_model_cfg.get("hidden_dim", 1000))
    stage1_noise_rate = float(stage1_model_cfg.get("noise_rate", 0.1))
    stage1_kl_weight = float(stage1_model_cfg.get("kl_weight", 5e-4))

    model = TriShift(data, device="cuda" if torch.cuda.is_available() else "cpu")
    model.model_init(
        x_dim=adata.shape[1],
        z_dim=int(stage1_model_cfg.get("z_dim", 100)),
        cond_dim=cond_dim_eff,
        vae_enc_hidden=stage1_model_cfg.get("enc_hidden", [512, 256]),
        vae_dec_hidden=stage1_model_cfg.get("dec_hidden", [256, 512]),
        shift_hidden=shift_hidden,
        gen_hidden=[512, 512],
        vae_hidden_dim=stage1_hidden_dim,
        vae_noise_rate=stage1_noise_rate,
        vae_kl_weight=stage1_kl_weight,
        dropout=gen_dropout,
        gen_encoder_hidden=gen_encoder_hidden,
        gen_state_dim=gen_state_dim,
        gen_decoder_hidden=gen_decoder_hidden,
        gen_input_mode=gen_input_mode,
        gen_use_batchnorm=gen_use_batch_norm,
        gen_use_layernorm=gen_use_layer_norm,
        predict_shift=bool(predict_shift),
        shift_predict_delta=shift_predict_delta,
        shift_use_cross_attention=shift_use_cross_attention,
        shift_cross_attn_heads=shift_cross_attn_heads,
        shift_cross_attn_dropout=shift_cross_attn_dropout,
        shift_use_transformer_block=shift_use_transformer_block,
        shift_transformer_layers=shift_transformer_layers,
        shift_transformer_ff_mult=shift_transformer_ff_mult,
        shift_transformer_dropout=shift_transformer_dropout,
        shift_transformer_readout=shift_transformer_readout,
        shift_repr_dim=shift_repr_dim,
        gen_state_source=gen_state_source,
        shift_input_source=shift_input_source,
        cond_pool_mode=cond_pool_mode,
        cond_l2_norm=cond_l2_norm,
        gen_use_residual_head=gen_use_residual_head,
    )
    return model


def _load_degs_cache(cache_path: Path) -> dict | None:
    """Load cached DEGs dict from disk."""
    if not cache_path.exists():
        return None
    try:
        with cache_path.open("rb") as f:
            obj = pickle.load(f)
    except Exception as exc:
        print(f"[degs] failed to load cache: {cache_path} ({exc})")
        return None
    if not isinstance(obj, dict):
        print(f"[degs] invalid cache payload: {cache_path}")
        return None
    return obj


def _save_degs_cache(cache_path: Path, uns: dict) -> None:
    """Persist DEGs dict to disk for reuse."""
    payload = {key: uns[key] for key in DEGS_CACHE_KEYS if key in uns}
    if not payload:
        return
    try:
        with cache_path.open("wb") as f:
            pickle.dump(payload, f)
    except Exception as exc:
        print(f"[degs] failed to save cache: {cache_path} ({exc})")


def _degs_cache_meta_ok(uns: dict) -> tuple[bool, str]:
    """Validate DEG cache metadata against Scouter-aligned settings."""
    meta = uns.get("degs_meta")
    if not isinstance(meta, dict):
        return False, "missing degs_meta"
    method = str(meta.get("rank_genes_groups_method", "")).strip()
    if method != DEGS_EXPECTED_METHOD:
        return False, f"method={method or '<missing>'} != {DEGS_EXPECTED_METHOD}"
    if str(meta.get("rank_genes_groups_reference", "")).strip() != "ctrl":
        return False, "reference!=ctrl"
    if bool(meta.get("rankby_abs", False)) is not True:
        return False, "rankby_abs!=True"
    return True, "ok"


def _logs_to_records(split_id: int, train_logs: dict) -> list[dict]:
    """Flatten nested training logs into tabular records."""
    records: list[dict] = []
    for stage_name, payload in train_logs.items():
        if stage_name in {"split_id", "train_mode"}:
            continue
        if not isinstance(payload, dict):
            continue

        if "epochs" in payload and isinstance(payload["epochs"], list):
            for epoch, item in enumerate(payload["epochs"], start=1):
                if isinstance(item, dict):
                    rec = {"split_id": split_id, "stage": stage_name, "epoch": epoch}
                    rec.update(item)
                    records.append(rec)
            continue

        for sub_stage in ("stage2", "stage3"):
            if sub_stage in payload and isinstance(payload[sub_stage], list):
                for epoch, item in enumerate(payload[sub_stage], start=1):
                    if isinstance(item, dict):
                        rec = {
                            "split_id": split_id,
                            "stage": f"{stage_name}.{sub_stage}",
                            "epoch": epoch,
                        }
                        rec.update(item)
                        records.append(rec)
    return records


def _split_list_by_ratio(lst: list[str], ratios: list[float], seed: int) -> list[list[str]]:
    """Compatibility wrapper around shared split implementation."""
    return _shared_split_list_by_ratio(lst=lst, ratios=ratios, seed=seed)


def _make_stage1_cache_signature(
    *,
    stage1_train_cfg: dict,
    stage1_model_cfg: dict,
    stage1_use_train_split: bool,
    resolved_stage1_epochs: int,
    resolved_stage1_batch_size: int,
    resolved_stage1_lr: float,
    resolved_stage1_beta: float,
    base_seed: int,
    perf: PerfConfig,
) -> str:
    """Build a stable hash for stage1 setup used to generate z_mu."""
    payload = {
        "stage1_train_cfg": stage1_train_cfg,
        "stage1_model_cfg": stage1_model_cfg,
        "stage1_use_train_split": bool(stage1_use_train_split),
        "resolved_stage1_epochs": int(resolved_stage1_epochs),
        "resolved_stage1_batch_size": int(resolved_stage1_batch_size),
        "resolved_stage1_lr": float(resolved_stage1_lr),
        "resolved_stage1_beta": float(resolved_stage1_beta),
        "base_seed": int(base_seed),
        "amp": bool(perf.amp),
        "grad_accum_steps": int(perf.grad_accum_steps),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _norman_subgroup(pert_list: list[str], seed: int) -> pd.DataFrame:
    """Compatibility wrapper around shared split implementation."""
    return _shared_norman_subgroup(pert_list=pert_list, seed=seed)


def run_dataset(name: str, fast: bool = False) -> None:
    """Run a full training/evaluation pipeline for the given dataset name.

    Args:
        name: Dataset key from configs/paths.yaml.
        fast: If True, use minimal epochs/splits for quick validation.
    """
    return run_dataset_with_paths(name=name, fast=fast)


def run_dataset_with_paths(
    *,
    name: str,
    fast: bool = False,
    defaults_path: str = "configs/defaults.yaml",
    paths_path: str = "configs/paths.yaml",
    out_dir: str | None = None,
) -> None:
    """Run pipeline with explicit config paths and output directory.

    Backward compatibility:
    - run_dataset(name, fast) calls this with default paths.
    """
    if name not in DATASET_CONFIG:
        raise ValueError(f"Unknown dataset name: {name}")

    dataset_cfg = DATASET_CONFIG[name]
    print("[run] load configs")
    cfg = load_yaml(paths_path)
    defaults = load_yaml(defaults_path)
    base_seed = int(defaults.get("seed", 24))
    set_seeds(base_seed)

    h5ad_path = cfg["datasets"][name]
    emb_key = dataset_cfg["emb_key"]
    emb_path = cfg["embeddings"][emb_key]

    print("[run] load data")
    adata = load_adata(h5ad_path)
    adata.uns = {}
    embd_df = load_embedding_df(emb_path)
    embd_df = apply_alias_mapping(embd_df, name)

    degs_cache_dir = Path("artifacts") / "cache" / "degs"
    degs_cache_dir.mkdir(parents=True, exist_ok=True)
    degs_cache_path = degs_cache_dir / f"{name}_degs.pkl"
    degs_cache = _load_degs_cache(degs_cache_path)
    if degs_cache:
        adata.uns.update(degs_cache)
        print(f"[degs] loaded cache: {degs_cache_path}")

    print("[run] init data")
    data = TriShiftData(adata, embd_df)
    data.setup_embedding_index()
    if degs_cache:
        meta_ok, meta_msg = _degs_cache_meta_ok(adata.uns)
        if not meta_ok:
            print(f"[degs] cache {meta_msg}; recomputing DEGs")
            for key in (*DEGS_CACHE_KEYS, "rank_genes_groups", "top20_degs_final"):
                adata.uns.pop(key, None)
            degs_cache = None
        else:
            print(f"[degs] cache method={DEGS_EXPECTED_METHOD} (ok)")
    if degs_cache:
        # Guard against stale cache after condition/alias changes (e.g., Norman gene alias updates).
        required_conds = set(data.conditions_pert)
        cached_conds = set()
        for key in DEGS_CONDITION_KEYS:
            obj = adata.uns.get(key, {})
            if isinstance(obj, dict):
                cached_conds |= set(obj.keys())
        missing = sorted(required_conds - cached_conds)
        if missing:
            print(
                "[degs] cache missing conditions "
                f"({len(missing)}); recomputing DEGs. examples={missing[:5]}"
            )
            for key in (*DEGS_CACHE_KEYS, "rank_genes_groups", "top20_degs_final"):
                adata.uns.pop(key, None)
            degs_cache = None
    data.build_or_load_degs()
    if not degs_cache:
        _save_degs_cache(degs_cache_path, adata.uns)
        if degs_cache_path.exists():
            print(f"[degs] saved cache: {degs_cache_path}")

    perf = _extract_perf(defaults)
    train_cfg = defaults.get("train", {})
    run_cfg = defaults.get("run", {})
    ablation_cfg = defaults.get("ablation", {})
    stage1_cfg = train_cfg.get("stage1", {})
    stage23_cfg = train_cfg.get("stage23", {})
    stage2_cfg = train_cfg.get("stage2", {})
    stage3_cfg = train_cfg.get("stage3", {})
    sched_stage1 = _extract_schedule(stage1_cfg)
    sched_stage23 = _extract_schedule(stage23_cfg)
    sched_stage2 = _extract_schedule(stage2_cfg)
    sched_stage3 = _extract_schedule(stage3_cfg)
    loss = _extract_loss(defaults)

    model_root = defaults.get("model", {})
    stage1_model_cfg = model_root.get("stage1", {})
    stage2_model_cfg = dict(model_root.get("stage2", {}))
    model_cfg = dict(model_root.get("stage3", {}))

    emb_table = torch.tensor(embd_df.values, dtype=torch.float32)
    mode = str(defaults.get("matching_mode", "knn"))
    k = int(defaults.get("k_topk", 5))
    n_eval_ensemble = int(defaults.get("n_eval_ensemble", 300))
    eval_ctrl_pool_mode = str(defaults.get("eval_ctrl_pool_mode", "random_train_ctrl"))
    eval_genept_distance = str(defaults.get("eval_genept_distance", "both"))
    eval_genept_compare_mode = str(defaults.get("eval_genept_compare_mode", "aggregate_cond"))
    eval_genept_train_candidate_mode = str(
        defaults.get("eval_genept_train_candidate_mode", "all_train_pert")
    )
    train_mode = defaults.get("train_mode", "joint")
    valid_train_modes = {"joint", "sequential", "stage3_only"}
    if train_mode not in valid_train_modes:
        raise ValueError(
            f"Unsupported train_mode={train_mode}. "
            f"Expected one of: {sorted(valid_train_modes)}"
        )
    removed_seq_keys = [
        k
        for k in (
            "sequential_joint_finetune",
            "joint_finetune_epochs",
            "joint_finetune_lr_scale",
        )
        if k in ablation_cfg
    ]
    if removed_seq_keys:
        raise ValueError(
            "ablation sequential joint finetune has been removed. "
            "Remove these keys from config: "
            + ", ".join(removed_seq_keys)
        )
    stage1_use_train_split = bool(ablation_cfg.get("stage1_use_train_split", False))
    valid_matching_modes = {"knn", "ot", "knn_ot", "soft_ot", "scpram_ot"}
    if mode not in valid_matching_modes:
        raise ValueError(
            f"Unsupported matching_mode={mode}. "
            f"Supported: {sorted(valid_matching_modes)}"
        )
    valid_eval_pool_modes = {"random_train_ctrl", "nearest_genept_ot_pool", "all"}
    if eval_ctrl_pool_mode not in valid_eval_pool_modes:
        raise ValueError(
            f"Unsupported eval_ctrl_pool_mode={eval_ctrl_pool_mode}. "
            f"Supported: {sorted(valid_eval_pool_modes)}"
        )
    valid_eval_distances = {"cosine", "l2", "both"}
    if eval_genept_distance not in valid_eval_distances:
        raise ValueError(
            f"Unsupported eval_genept_distance={eval_genept_distance}. "
            f"Supported: {sorted(valid_eval_distances)}"
        )
    valid_eval_compare_modes = {"aggregate_cond", "per_gene_nearest_cond", "all"}
    if eval_genept_compare_mode not in valid_eval_compare_modes:
        raise ValueError(
            f"Unsupported eval_genept_compare_mode={eval_genept_compare_mode}. "
            f"Supported: {sorted(valid_eval_compare_modes)}"
        )
    valid_eval_train_candidate_modes = {
        "all_train_pert",
        "norman_train_single_only",
        "norman_single_nearest_else_random",
    }
    if eval_genept_train_candidate_mode not in valid_eval_train_candidate_modes:
        raise ValueError(
            f"Unsupported eval_genept_train_candidate_mode={eval_genept_train_candidate_mode}. "
            f"Supported: {sorted(valid_eval_train_candidate_modes)}"
        )
    topk_strategy = str(ablation_cfg.get("topk_strategy", "random"))
    if topk_strategy not in {"random", "weighted_sample"}:
        raise ValueError(
            f"Unsupported ablation.topk_strategy={topk_strategy}. "
            "Supported: random, weighted_sample"
        )
    sample_soft_ctrl = bool(ablation_cfg.get("sample_soft_ctrl", True))
    per_condition_ot = bool(ablation_cfg.get("per_condition_ot", False))
    reuse_ot_cache = bool(ablation_cfg.get("reuse_ot_cache", False))
    reuse_z_mu_cache = bool(ablation_cfg.get("reuse_z_mu_cache", False))
    latent_loss_type = str(ablation_cfg.get("latent_loss_type", "gears"))
    predict_shift_cfg = bool(stage2_model_cfg.get("predict_shift", True))
    predict_shift_effective = bool(predict_shift_cfg)
    predict_shift_auto_overrides: list[str] = []
    if not predict_shift_effective:
        input_mode_before = str(model_cfg.get("input_mode", "full"))
        if input_mode_before != "full":
            model_cfg["input_mode"] = "full"
            msg = (
                "predict_shift=false forces model.stage3.input_mode=full "
                f"(was {input_mode_before})"
            )
            predict_shift_auto_overrides.append(msg)
            print(f"[config] warning: {msg}")
        if "predict_delta" in stage2_model_cfg:
            msg = (
                "model.stage2.predict_delta is ignored when "
                "model.stage2.predict_shift=false"
            )
            predict_shift_auto_overrides.append(msg)
            print(f"[config] warning: {msg}")
        if stage2_model_cfg.get("shift_repr_dim", None) is not None:
            msg = (
                "model.stage2.shift_repr_dim is ignored when "
                "model.stage2.predict_shift=false"
            )
            predict_shift_auto_overrides.append(msg)
            print(f"[config] warning: {msg}")
        if "shift_input_source" in stage2_model_cfg:
            msg = (
                "model.stage2.shift_input_source is ignored when "
                "model.stage2.predict_shift=false"
            )
            predict_shift_auto_overrides.append(msg)
            print(f"[config] warning: {msg}")
        stage2_model_cfg["predict_delta"] = False
        stage2_model_cfg["shift_repr_dim"] = None
        stage2_model_cfg["shift_input_source"] = "latent_mu"

    shift_predict_delta_cfg = bool(stage2_model_cfg.get("predict_delta", True))
    legacy_disable_loss_z = bool(ablation_cfg.get("disable_loss_z_supervision", False))
    if legacy_disable_loss_z and shift_predict_delta_cfg:
        # Backward compatibility: old flag now maps to "no delta prediction".
        print(
            "[config] ablation.disable_loss_z_supervision is deprecated; "
            "forcing model.stage2.predict_delta=False"
        )
        stage2_model_cfg = dict(stage2_model_cfg)
        stage2_model_cfg["predict_delta"] = False
        shift_predict_delta_cfg = False
    disable_loss_z_supervision = (not shift_predict_delta_cfg) or (not predict_shift_effective)
    model_root["stage2"] = stage2_model_cfg
    model_root["stage3"] = model_cfg
    cond_pool_mode_cfg = str(stage2_model_cfg.get("cond_pool_mode", "sum"))
    if cond_pool_mode_cfg not in {"sum", "mean"}:
        raise ValueError(
            f"Unsupported model.stage2.cond_pool_mode={cond_pool_mode_cfg}. "
            "Supported: sum, mean"
        )
    shift_input_source_cfg = str(stage2_model_cfg.get("shift_input_source", "latent_mu"))
    shift_input_source_eff = shift_input_source_cfg
    gen_state_source = "compressor"
    cache_dir = Path("artifacts") / "cache" / "topk"
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_dir_path = Path(out_dir) if out_dir is not None else (Path("artifacts") / "results" / name)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    if mode == "scpram_ot":
        msg = (
            f"[run] matching_mode=scpram_ot (EMD hard-match semantics; "
            f"k_topk={k}, per_condition_ot={per_condition_ot})"
        )
        if k > 1:
            msg += " [top-k extension of scPRAM OT]"
        print(msg)
    print(f"[run] predict_shift={predict_shift_effective}")
    if not predict_shift_effective:
        print("[run] shift-disabled mode enabled: stage23 uses generator forward_no_delta")
    if eval_ctrl_pool_mode == "nearest_genept_ot_pool":
        print(
            "[run] eval_ctrl_pool_mode=nearest_genept_ot_pool "
            f"(sample_size=n_eval_ensemble={n_eval_ensemble}, distance={eval_genept_distance}, "
            f"compare_mode={eval_genept_compare_mode}, "
            f"candidate_mode={eval_genept_train_candidate_mode})"
        )
        if eval_genept_compare_mode == "all":
            print(
                "[run] eval_genept_compare_mode=all => compare algorithms "
                "aggregate_cond + per_gene_nearest_cond, and candidate mode will expand "
                "to all configured modes"
            )
    elif eval_ctrl_pool_mode == "all":
        print(
            "[run] eval_ctrl_pool_mode=all "
            "(run both random_train_ctrl and nearest_genept_ot_pool); "
            f"nearest settings: distance={eval_genept_distance}, "
            f"compare_mode={eval_genept_compare_mode}, "
            f"candidate_mode={eval_genept_train_candidate_mode}, "
            f"sample_size=n_eval_ensemble={n_eval_ensemble}"
        )

    # Snapshot the exact configs used for this run for reproducibility.
    # Keep names stable for downstream scripts.
    try:
        _dump_yaml(out_dir_path / "defaults_used.yaml", defaults)
        _dump_yaml(out_dir_path / "paths_used.yaml", cfg)
    except Exception as exc:
        print(f"[run] warning: failed to write config snapshots ({exc})")

    try:
        if eval_ctrl_pool_mode in {"nearest_genept_ot_pool", "all"}:
            compare_modes_executed = _resolve_eval_compare_modes(eval_genept_compare_mode)
            candidate_modes_executed = _resolve_eval_candidate_modes(
                eval_genept_compare_mode=eval_genept_compare_mode,
                eval_genept_train_candidate_mode=eval_genept_train_candidate_mode,
            )
        else:
            compare_modes_executed = []
            candidate_modes_executed = []
        meta = {
            "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "git_commit": _safe_git_commit(),
            "dataset": name,
            "defaults_path": str(defaults_path),
            "paths_path": str(paths_path),
            "out_dir": str(out_dir_path),
            "fast": bool(fast),
            "seed": int(base_seed),
            "train_mode": str(train_mode),
            "matching_mode": str(mode),
            "k_topk": int(k),
            "predict_shift": bool(predict_shift_cfg),
            "predict_shift_effective": bool(predict_shift_effective),
            "predict_shift_auto_overrides": list(predict_shift_auto_overrides),
            "n_eval_ensemble": int(n_eval_ensemble),
            "eval_ctrl_pool_mode": str(eval_ctrl_pool_mode),
            "eval_ctrl_modes_executed": (
                ["random_train_ctrl", "nearest_genept_ot_pool"]
                if str(eval_ctrl_pool_mode) == "all"
                else [str(eval_ctrl_pool_mode)]
            ),
            "eval_genept_distance": str(eval_genept_distance),
            "eval_genept_compare_mode": str(eval_genept_compare_mode),
            "eval_genept_train_candidate_mode": str(eval_genept_train_candidate_mode),
            "eval_compare_modes_executed": list(compare_modes_executed),
            "eval_candidate_modes_executed": list(candidate_modes_executed),
            "eval_ctrl_sample_size_source": "n_eval_ensemble",
            "reuse_ot_cache": bool(reuse_ot_cache),
            "reuse_z_mu_cache": bool(reuse_z_mu_cache),
        }
        meta_path = out_dir_path / "run_meta.json"
        if meta_path.exists():
            try:
                existing = json.loads(meta_path.read_text(encoding="utf-8"))
                if isinstance(existing, dict):
                    existing.update(meta)
                    meta = existing
            except Exception:
                # Keep the new meta if existing cannot be parsed.
                pass
        meta_path.write_text(json.dumps(meta, indent=2, sort_keys=False), encoding="utf-8")
    except Exception as exc:
        print(f"[run] warning: failed to write run_meta.json ({exc})")

    defaults_epochs = dataset_cfg["defaults"]
    stage1_epochs = _get_cfg_or(stage1_cfg, "epochs", defaults_epochs["stage1"])
    stage23_epochs = _get_cfg_or(stage23_cfg, "epochs", defaults_epochs["stage23"])
    stage2_epochs = _get_cfg_or(stage2_cfg, "epochs", defaults_epochs["stage2"])
    stage3_epochs = _get_cfg_or(stage3_cfg, "epochs", defaults_epochs["stage3"])
    stage1_batch_size = int(stage1_cfg.get("batch_size", 64))
    stage1_lr = float(stage1_cfg.get("lr", 1e-3))
    stage1_beta = float(stage1_cfg.get("beta", 1.0))
    stage1_deg_weight = float(stage1_cfg.get("deg_weight", 1.0))
    stage1_ecs_cfg = dict(stage1_cfg.get("ecs", {}))

    metrics_all_by_tag: dict[str, list[pd.DataFrame]] = {}
    n_splits = (
        int(run_cfg.get("n_splits", dataset_cfg["multi_split_default"]))
        if run_cfg.get("multi_split", False)
        else 1
    )
    if fast:
        stage1_epochs = 1
        stage23_epochs = 1
        stage2_epochs = 1
        stage3_epochs = 1
        n_splits = 1
        n_eval_ensemble = min(n_eval_ensemble, 20)
    eval_modes_to_run = (
        ["random_train_ctrl", "nearest_genept_ot_pool"]
        if eval_ctrl_pool_mode == "all"
        else [eval_ctrl_pool_mode]
    )
    stage1_cache_sig = _make_stage1_cache_signature(
        stage1_train_cfg=stage1_cfg,
        stage1_model_cfg=stage1_model_cfg,
        stage1_use_train_split=stage1_use_train_split,
        resolved_stage1_epochs=stage1_epochs,
        resolved_stage1_batch_size=stage1_batch_size,
        resolved_stage1_lr=stage1_lr,
        resolved_stage1_beta=stage1_beta,
        base_seed=base_seed,
        perf=perf,
    )
    z_mu_cache_dir = Path("artifacts") / "cache" / "z_mu"
    z_mu_cache_dir.mkdir(parents=True, exist_ok=True)
    for split_id in range(1, n_splits + 1):
        print(f"[run] split {split_id}/{n_splits}")
        set_seeds(base_seed)
        subgroup_df = None
        if name == "norman":
            subgroup_df = _norman_subgroup(
                list(adata.obs["condition"].astype(str).unique()),
                seed=split_id,
            )
            test_conds = list(subgroup_df[subgroup_df.group == "test"].index)
            val_conds = list(subgroup_df[subgroup_df.group == "val"].index)
            split_dict = data.split_by_condition(
                seed=split_id, test_conds=test_conds, val_conds=val_conds
            )
        else:
            split_dict = data.split_by_condition(
                seed=split_id,
                test_ratio=float(dataset_cfg.get("test_ratio", 0.2)),
            )
        train_logs: dict = {"split_id": split_id, "train_mode": train_mode}

        print("[run] init model")
        model = _init_model(
            data,
            adata,
            embd_df,
            stage1_model_cfg,
            stage2_model_cfg,
            model_cfg,
            predict_shift=predict_shift_effective,
            shift_input_source_override=shift_input_source_eff,
            gen_state_source=gen_state_source,
        )
        model.set_base_seed(base_seed)

        if train_mode != "stage3_only":
            z_mu_cache_path = z_mu_cache_dir / (
                f"{name}_split{split_id}_s1{stage1_cache_sig[:12]}.npz"
            )
            z_mu_cached = None
            if reuse_z_mu_cache:
                z_mu_cached = _load_z_mu_cache(z_mu_cache_path, data.adata_all.n_obs)
            if z_mu_cached is not None:
                data.set_latent_mu(z_mu_cached, key="z_mu")
                print(f"[z_mu] loaded cache: {z_mu_cache_path}")
            else:
                print("[run] stage1")
                if stage1_use_train_split:
                    # Scouter-like: train only on split train cells (ctrl + pert).
                    stage1_adata = split_dict["train"]
                else:
                    stage1_adata = data.adata_ctrl
                stage1_val_adata = None
                val_split = split_dict.get("val")
                if val_split is not None:
                    if stage1_use_train_split:
                        stage1_val_adata = val_split
                    else:
                        val_ctrl_mask = (
                            val_split.obs[data.label_key].astype(str) == data.ctrl_label
                        )
                        if bool(val_ctrl_mask.any()):
                            stage1_val_adata = val_split[val_ctrl_mask]
                train_logs["stage1"] = model.train_stage1_vae(
                    adata_ctrl_pool=stage1_adata,
                    epochs=stage1_epochs,
                    batch_size=stage1_batch_size,
                    lr=stage1_lr,
                    beta=stage1_beta,
                    deg_weight=stage1_deg_weight,
                    ecs_enable=bool(stage1_ecs_cfg.get("enable", False)),
                    ecs_epochs=int(stage1_ecs_cfg.get("epochs", 10)),
                    ecs_lr=float(stage1_ecs_cfg.get("lr", 1e-4)),
                    ecs_sched_gamma=float(stage1_ecs_cfg.get("sched_gamma", 0.9)),
                    ecs_weight=float(stage1_ecs_cfg.get("weight", 10.0)),
                    ecs_threshold=float(stage1_ecs_cfg.get("threshold", 0.8)),
                    ecs_patience=int(stage1_ecs_cfg.get("patience", 5)),
                    ecs_min_delta=float(stage1_ecs_cfg.get("min_delta", 1e-3)),
                    sched_gamma=sched_stage1.sched_gamma,
                    patience=sched_stage1.patience,
                    min_delta=sched_stage1.min_delta,
                    amp=perf.amp,
                    num_workers=perf.num_workers,
                    pin_memory=perf.pin_memory,
                    grad_accum_steps=perf.grad_accum_steps,
                    adata_val=stage1_val_adata,
                )

                print("[run] encode z_mu")
                model.encode_and_cache_mu(
                    data.adata_all,
                    batch_size=stage1_batch_size,
                    amp=perf.amp,
                    num_workers=perf.num_workers,
                    pin_memory=perf.pin_memory,
                )
                if reuse_z_mu_cache:
                    z_mu_to_save = np.asarray(data.adata_all.obsm.get("z_mu"))
                    if z_mu_to_save is not None and z_mu_to_save.ndim == 2:
                        _save_z_mu_cache(z_mu_cache_path, z_mu_to_save)
                        print(f"[z_mu] saved cache: {z_mu_cache_path}")

        topk_cache_key = (
            f"name={name}|mode={mode}|k={k}|split_id={split_id}|"
            f"stage1_sig={stage1_cache_sig}|per_condition_ot={bool(per_condition_ot)}"
        )
        cache_path = cache_dir / (
            f"{name}_split{split_id}_{mode}_k{k}_s1{stage1_cache_sig[:12]}.npz"
        )

        print(f"[run] stage23 mode={train_mode}")
        if train_mode == "stage3_only":
            if str(model_cfg.get("input_mode", "full")) != "full":
                raise ValueError(
                    "model.stage3.input_mode!=full requires shift representation and "
                    "is incompatible with train_mode=stage3_only"
                )
            train_logs["stage3_only"] = model.train_stage3_only(
                split_dict=split_dict,
                emb_table=emb_table,
                split_id=split_id,
                epochs=stage3_epochs,
                batch_size=int(stage23_cfg.get("batch_size", 64)),
                lr=float(stage3_cfg.get("lr", 1e-3)),
                sched_gamma=sched_stage3.sched_gamma,
                patience=sched_stage3.patience,
                min_delta=sched_stage3.min_delta,
                amp=perf.amp,
                num_workers=perf.num_workers,
                pin_memory=perf.pin_memory,
                grad_accum_steps=perf.grad_accum_steps,
                gamma=loss.gamma,
                lambda_dir=loss.lambda_dir,
                lambda_dir_expr=loss.lambda_dir_expr,
                deg_weight=loss.deg_weight,
                lambda_expr_mse=loss.lambda_expr_mse,
            )
        elif train_mode == "sequential":
            train_logs["stage23_sequential"] = model.train_stage23_sequential(
                split_dict=split_dict,
                emb_table=emb_table,
                mode=mode,
                k=k,
                split_id=split_id,
                epochs_stage2=stage2_epochs,
                epochs_stage3=stage3_epochs,
                batch_size=int(stage23_cfg.get("batch_size", 64)),
                lr_stage2=float(stage2_cfg.get("lr", 1e-3)),
                lr_stage3=float(stage3_cfg.get("lr", 1e-3)),
                sched_gamma_stage2=sched_stage2.sched_gamma,
                sched_gamma_stage3=sched_stage3.sched_gamma,
                patience_stage2=sched_stage2.patience,
                patience_stage3=sched_stage3.patience,
                min_delta_stage2=sched_stage2.min_delta,
                min_delta_stage3=sched_stage3.min_delta,
                amp=perf.amp,
                num_workers=perf.num_workers,
                pin_memory=perf.pin_memory,
                grad_accum_steps=perf.grad_accum_steps,
                cache_topk_path=str(cache_path),
                gamma=loss.gamma,
                lambda_dir=loss.lambda_dir,
                lambda_dir_expr=loss.lambda_dir_expr,
                deg_weight=loss.deg_weight,
                topk_strategy=topk_strategy,
                sample_soft_ctrl=sample_soft_ctrl,
                lambda_expr_mse=loss.lambda_expr_mse,
                per_condition_ot=per_condition_ot,
                disable_loss_z_supervision=disable_loss_z_supervision,
                reuse_ot_cache=reuse_ot_cache,
                topk_cache_key=topk_cache_key,
            )
        else:
            train_logs["stage23_joint"] = model.train_stage23_joint(
                split_dict=split_dict,
                emb_table=emb_table,
                mode=mode,
                k=k,
                split_id=split_id,
                epochs=stage23_epochs,
                batch_size=int(stage23_cfg.get("batch_size", 64)),
                lr=float(stage23_cfg.get("lr", 1e-3)),
                sched_gamma=sched_stage23.sched_gamma,
                patience=sched_stage23.patience,
                min_delta=sched_stage23.min_delta,
                amp=perf.amp,
                num_workers=perf.num_workers,
                pin_memory=perf.pin_memory,
                grad_accum_steps=perf.grad_accum_steps,
                cache_topk_path=str(cache_path),
                gamma=loss.gamma,
                lambda_dir=loss.lambda_dir,
                lambda_dir_expr=loss.lambda_dir_expr,
                lambda_dir_z=loss.lambda_dir_z,
                lambda_z=loss.lambda_z,
                deg_weight=loss.deg_weight,
                topk_strategy=topk_strategy,
                sample_soft_ctrl=sample_soft_ctrl,
                latent_loss_type=latent_loss_type,
                lambda_expr_mse=loss.lambda_expr_mse,
                per_condition_ot=per_condition_ot,
                disable_loss_z_supervision=disable_loss_z_supervision,
                reuse_ot_cache=reuse_ot_cache,
                topk_cache_key=topk_cache_key,
            )

        train_log_pkl = out_dir_path / f"train_logs_split{split_id}.pkl"
        with open(train_log_pkl, "wb") as f:
            pickle.dump(train_logs, f)
        train_records = _logs_to_records(split_id, train_logs)
        if train_records:
            pd.DataFrame(train_records).to_csv(
                out_dir_path / f"train_loss_split{split_id}.csv", index=False
            )

        print("[run] evaluate")
        for eval_mode in eval_modes_to_run:
            if eval_mode == "nearest_genept_ot_pool":
                train_split = split_dict.get("train")
                if train_split is None:
                    raise ValueError("split_dict['train'] is required for nearest_genept_ot_pool")
                split_train_cond = train_split.obs[data.label_key].astype(str).values
                split_train_pert = train_split[split_train_cond != data.ctrl_label]
                _, train_ctrl_global_idx = model._get_ctrl_pool_from_split(train_split)
                eval_topk_map = data.build_or_load_topk_map(
                    split_adata=split_train_pert,
                    mode=mode,
                    k=k,
                    seed=split_id,
                    candidates=100,
                    cache_path=str(cache_path),
                    per_condition_ot=per_condition_ot,
                    reuse_ot_cache=reuse_ot_cache,
                    cache_key=topk_cache_key,
                    ctrl_global_indices=train_ctrl_global_idx,
                )
                eval_distances = (
                    ["cosine", "l2"] if eval_genept_distance == "both" else [eval_genept_distance]
                )
                compare_modes_for_nearest = _resolve_eval_compare_modes(eval_genept_compare_mode)
                candidate_modes_for_nearest = _resolve_eval_candidate_modes(
                    eval_genept_compare_mode=eval_genept_compare_mode,
                    eval_genept_train_candidate_mode=eval_genept_train_candidate_mode,
                )
                nearest_alias_preds = None
                nearest_combo_seen: set[tuple[str, str]] = set()
                split_test_conds = [str(c) for c in split_dict.get("test_conds", [])]

                for compare_mode_eff in compare_modes_for_nearest:
                    for candidate_mode_req in candidate_modes_for_nearest:
                        candidate_mode_eff = str(candidate_mode_req)
                        candidate_conds: list[str] | None = None
                        target_test_conds: list[str] | None = None

                        if candidate_mode_req == "norman_train_single_only":
                            if name != "norman" or subgroup_df is None:
                                print(
                                    "[eval] warning: eval_genept_train_candidate_mode=norman_train_single_only "
                                    f"not applicable for dataset={name}; fallback to all_train_pert"
                                )
                                candidate_mode_eff = "all_train_pert"
                            else:
                                train_pert_conds = {
                                    str(c) for c in split_train_cond.tolist() if str(c) != data.ctrl_label
                                }
                                single_train_conds = {
                                    str(idx)
                                    for idx in subgroup_df.index[
                                        (subgroup_df["group"].astype(str) == "train")
                                        & (subgroup_df["subgroup"].astype(str) == "single")
                                    ].tolist()
                                }
                                filtered = sorted(single_train_conds & train_pert_conds)
                                if len(filtered) == 0:
                                    print(
                                        "[eval] warning: single candidate set is empty; "
                                        "fallback to all_train_pert"
                                    )
                                    candidate_mode_eff = "all_train_pert"
                                else:
                                    candidate_conds = filtered

                        elif candidate_mode_req == "norman_single_nearest_else_random":
                            if name != "norman" or subgroup_df is None:
                                print(
                                    "[eval] warning: eval_genept_train_candidate_mode=norman_single_nearest_else_random "
                                    f"not applicable for dataset={name}; fallback to all_train_pert"
                                )
                                candidate_mode_eff = "all_train_pert"
                            else:
                                single_test_conds = {
                                    str(idx)
                                    for idx in subgroup_df.index[
                                        (subgroup_df["group"].astype(str) == "test")
                                        & (subgroup_df["subgroup"].astype(str) == "single")
                                    ].tolist()
                                }
                                target_test_conds = [c for c in split_test_conds if c in single_test_conds]
                                print(
                                    "[eval] target_test_conds source=norman_test_single "
                                    f"(count={len(target_test_conds)})"
                                )

                        combo_key = (str(compare_mode_eff), str(candidate_mode_eff))
                        if combo_key in nearest_combo_seen:
                            print(
                                f"[eval] skip duplicate nearest combo compare={compare_mode_eff} "
                                f"candidate={candidate_mode_eff}"
                            )
                            continue
                        nearest_combo_seen.add(combo_key)

                        if candidate_conds is None:
                            print(f"[eval] train candidate source={candidate_mode_eff}")
                        else:
                            print(
                                f"[eval] train candidate source={candidate_mode_eff} "
                                f"(count={len(candidate_conds)})"
                            )

                        for dist_tag in eval_distances:
                            eval_strategy = model.build_eval_ctrl_strategy(
                                split_dict=split_dict,
                                emb_table=emb_table,
                                topk_map=eval_topk_map,
                                distance_metric=dist_tag,
                                sample_size=n_eval_ensemble,
                                compare_mode=compare_mode_eff,
                                train_candidate_conds=candidate_conds,
                                target_test_conds=target_test_conds,
                            )
                            metrics_df = model.evaluate(
                                split_dict=split_dict,
                                emb_table=emb_table,
                                split_id=split_id,
                                n_ensemble=n_eval_ensemble,
                                base_seed=base_seed,
                                eval_ctrl_strategy=eval_strategy,
                            )
                            metrics_df = _attach_subgroup_column(metrics_df, subgroup_df)

                            # Keep old tag naming for legacy combos; add explicit tags for compare_mode=all.
                            if eval_genept_compare_mode == "all":
                                if eval_genept_distance == "both":
                                    metrics_tag = (
                                        f"nearest_{compare_mode_eff}_{candidate_mode_eff}_{dist_tag}"
                                    )
                                    pkl_tag = (
                                        f"nearest_{compare_mode_eff}_{candidate_mode_eff}_{dist_tag}"
                                    )
                                else:
                                    metrics_tag = f"nearest_{compare_mode_eff}_{candidate_mode_eff}"
                                    pkl_tag = f"nearest_{compare_mode_eff}_{candidate_mode_eff}"
                            else:
                                if eval_ctrl_pool_mode == "all":
                                    if eval_genept_distance == "both":
                                        metrics_tag = f"nearest_{dist_tag}"
                                        pkl_tag = f"nearest_{dist_tag}"
                                    else:
                                        metrics_tag = "nearest"
                                        pkl_tag = "nearest"
                                else:
                                    if eval_genept_distance == "both":
                                        metrics_tag = dist_tag
                                        pkl_tag = dist_tag
                                    else:
                                        metrics_tag = "main"
                                        pkl_tag = ""

                            metrics_all_by_tag.setdefault(metrics_tag, []).append(metrics_df)

                            if eval_ctrl_pool_mode == "all" or eval_genept_compare_mode == "all":
                                pkl_name = (
                                    f"trishift_{name}_{split_id}_{pkl_tag}.pkl"
                                    if pkl_tag
                                    else f"trishift_{name}_{split_id}.pkl"
                                )
                                out_pkl = out_dir_path / pkl_name
                            elif eval_genept_distance == "both":
                                out_pkl = out_dir_path / f"trishift_{name}_{split_id}_{dist_tag}.pkl"
                            else:
                                out_pkl = out_dir_path / f"trishift_{name}_{split_id}.pkl"

                            preds = model.export_predictions(
                                split_dict=split_dict,
                                emb_table=emb_table,
                                split_id=split_id,
                                n_ensemble=n_eval_ensemble,
                                base_seed=base_seed,
                                out_path=str(out_pkl),
                                eval_ctrl_strategy=eval_strategy,
                            )

                            # Alias policy for nearest-only mode:
                            # - legacy: both -> cosine alias
                            # - compare_mode=all: alias all_train_pert (+cosine when both)
                            if eval_ctrl_pool_mode != "all":
                                if eval_genept_compare_mode == "all":
                                    if (
                                        compare_mode_eff == "aggregate_cond"
                                        and candidate_mode_eff == "all_train_pert"
                                    ):
                                        if eval_genept_distance != "both" or dist_tag == "cosine":
                                            nearest_alias_preds = preds
                                elif eval_genept_distance == "both" and dist_tag == "cosine":
                                    nearest_alias_preds = preds

                if eval_ctrl_pool_mode != "all" and nearest_alias_preds is not None:
                    with open(out_dir_path / f"trishift_{name}_{split_id}.pkl", "wb") as f:
                        pickle.dump(nearest_alias_preds, f)
            else:
                metrics_df = model.evaluate(
                    split_dict=split_dict,
                    emb_table=emb_table,
                    split_id=split_id,
                    n_ensemble=n_eval_ensemble,
                    base_seed=base_seed,
                )
                metrics_df = _attach_subgroup_column(metrics_df, subgroup_df)
                out_default = out_dir_path / f"trishift_{name}_{split_id}.pkl"
                preds = model.export_predictions(
                    split_dict=split_dict,
                    emb_table=emb_table,
                    split_id=split_id,
                    n_ensemble=n_eval_ensemble,
                    base_seed=base_seed,
                    out_path=str(out_default),
                )
                if eval_ctrl_pool_mode == "all":
                    with open(out_dir_path / f"trishift_{name}_{split_id}_random.pkl", "wb") as f:
                        pickle.dump(preds, f)
                    metrics_all_by_tag.setdefault("random", []).append(metrics_df)
                else:
                    metrics_all_by_tag.setdefault("main", []).append(metrics_df)

    if metrics_all_by_tag:
        if eval_ctrl_pool_mode == "all":
            for tag, frames in metrics_all_by_tag.items():
                metrics_df_all = pd.concat(frames, ignore_index=True)
                metrics_df_all.to_csv(out_dir_path / f"metrics_{tag}.csv", index=False)
                _write_mean_metrics(out_dir_path / f"mean_pearson_{tag}.txt", metrics_df_all)
            alias_tag = "random" if "random" in metrics_all_by_tag else next(iter(metrics_all_by_tag.keys()))
            alias_df = pd.concat(metrics_all_by_tag[alias_tag], ignore_index=True)
            alias_df.to_csv(out_dir_path / "metrics.csv", index=False)
            _write_mean_metrics(out_dir_path / "mean_pearson.txt", alias_df)
        elif eval_ctrl_pool_mode == "nearest_genept_ot_pool":
            if eval_genept_compare_mode == "all":
                for tag, frames in metrics_all_by_tag.items():
                    metrics_df_all = pd.concat(frames, ignore_index=True)
                    metrics_df_all.to_csv(out_dir_path / f"metrics_{tag}.csv", index=False)
                    _write_mean_metrics(out_dir_path / f"mean_pearson_{tag}.txt", metrics_df_all)

                if eval_genept_distance == "both":
                    alias_tag = "nearest_aggregate_cond_all_train_pert_cosine"
                else:
                    alias_tag = "nearest_aggregate_cond_all_train_pert"
                if alias_tag not in metrics_all_by_tag:
                    fallbacks = [
                        k0
                        for k0 in metrics_all_by_tag.keys()
                        if k0.startswith("nearest_aggregate_cond_all_train_pert")
                    ]
                    alias_tag = fallbacks[0] if fallbacks else next(iter(metrics_all_by_tag.keys()))
                alias_df = pd.concat(metrics_all_by_tag[alias_tag], ignore_index=True)
                alias_df.to_csv(out_dir_path / "metrics.csv", index=False)
                _write_mean_metrics(out_dir_path / "mean_pearson.txt", alias_df)
            elif eval_genept_distance == "both":
                cosine_df_all = None
                for dist_tag in ("cosine", "l2"):
                    if dist_tag not in metrics_all_by_tag:
                        continue
                    metrics_df_all = pd.concat(metrics_all_by_tag[dist_tag], ignore_index=True)
                    metrics_df_all.to_csv(out_dir_path / f"metrics_{dist_tag}.csv", index=False)
                    _write_mean_metrics(out_dir_path / f"mean_pearson_{dist_tag}.txt", metrics_df_all)
                    if dist_tag == "cosine":
                        cosine_df_all = metrics_df_all
                if cosine_df_all is not None:
                    cosine_df_all.to_csv(out_dir_path / "metrics.csv", index=False)
                    _write_mean_metrics(out_dir_path / "mean_pearson.txt", cosine_df_all)
            else:
                if "main" in metrics_all_by_tag:
                    main_tag = "main"
                else:
                    main_tag = next(iter(metrics_all_by_tag.keys()))
                metrics_df_all = pd.concat(metrics_all_by_tag[main_tag], ignore_index=True)
                metrics_df_all.to_csv(out_dir_path / "metrics.csv", index=False)
                _write_mean_metrics(out_dir_path / "mean_pearson.txt", metrics_df_all)
        else:
            if "main" in metrics_all_by_tag:
                main_tag = "main"
            else:
                main_tag = next(iter(metrics_all_by_tag.keys()))
            metrics_df_all = pd.concat(metrics_all_by_tag[main_tag], ignore_index=True)
            metrics_df_all.to_csv(out_dir_path / "metrics.csv", index=False)
            _write_mean_metrics(out_dir_path / "mean_pearson.txt", metrics_df_all)
    print(f"[run] saved metrics: {out_dir_path / 'metrics.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TriShift on a dataset")
    parser.add_argument("--name", required=True, help="dataset name")
    parser.add_argument("--fast", action="store_true", help="use minimal epochs/splits")
    parser.add_argument(
        "--defaults",
        default="configs/defaults.yaml",
        help="defaults yaml path (default: configs/defaults.yaml)",
    )
    parser.add_argument(
        "--paths",
        default="configs/paths.yaml",
        help="paths yaml path (default: configs/paths.yaml)",
    )
    parser.add_argument(
        "--out_dir",
        default="",
        help="output directory; if empty uses artifacts/results/<dataset>",
    )
    args = parser.parse_args()
    out_dir = args.out_dir.strip() or None
    run_dataset_with_paths(
        name=args.name,
        fast=bool(args.fast),
        defaults_path=str(args.defaults),
        paths_path=str(args.paths),
        out_dir=out_dir,
    )


if __name__ == "__main__":
    main()
