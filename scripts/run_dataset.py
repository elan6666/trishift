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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

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


DATASET_CONFIG = {
    "adamson": {
        "emb_key": "emb_b",
        "defaults": {"stage1": 10, "stage23": 6, "stage2": 3, "stage3": 3},
        "multi_split_default": 5,
    },
    "dixit": {
        "emb_key": "emb_c",
        "defaults": {"stage1": 5, "stage23": 10, "stage2": 5, "stage3": 5},
        "multi_split_default": 10,
    },
    "norman": {
        "emb_key": "emb_a",
        "defaults": {"stage1": 5, "stage23": 10, "stage2": 5, "stage3": 5},
        "multi_split_default": 5,
    },
    "replogle_k562_essential": {
        "emb_key": "emb_c",
        "defaults": {"stage1": 5, "stage23": 10, "stage2": 5, "stage3": 5},
        "multi_split_default": 5,
    },
    "replogle_rpe1_essential": {
        "emb_key": "emb_b",
        "defaults": {"stage1": 5, "stage23": 10, "stage2": 5, "stage3": 5},
        "multi_split_default": 5,
    },
}

DEGS_CACHE_KEYS = ("top20_degs_non_dropout", "top20_degs_non_zero", "top20_degs")


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


def _write_mean_metrics(path: Path, metrics_df: pd.DataFrame) -> None:
    """Write mean pearson and other averaged numeric metrics to text file.

    Backward compatibility:
    - First line remains plain mean pearson float for old readers.
    """
    numeric_means = metrics_df.mean(numeric_only=True)
    mean_pearson = float(numeric_means.get("pearson", float("nan")))

    preferred_order = [
        "pearson",
        "nmse",
        "systema_corr_all_allpert",
        "systema_corr_20de_allpert",
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

    lines = [f"{mean_pearson}\n"]
    for key in keys:
        val = float(numeric_means[key])
        lines.append(f"mean_{key}={val}\n")

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
    shift_use_cross_attention = bool(stage2_model_cfg.get("use_cross_attention", False))
    shift_cross_attn_heads = int(stage2_model_cfg.get("cross_attn_heads", 4))
    shift_cross_attn_dropout = float(stage2_model_cfg.get("cross_attn_dropout", 0.0))
    shift_use_transformer_block = bool(stage2_model_cfg.get("use_transformer_block", False))
    shift_transformer_layers = int(stage2_model_cfg.get("transformer_layers", 1))
    shift_transformer_ff_mult = int(stage2_model_cfg.get("transformer_ff_mult", 4))
    shift_transformer_dropout = float(stage2_model_cfg.get("transformer_dropout", 0.0))
    shift_transformer_readout = str(stage2_model_cfg.get("transformer_readout", "first"))
    shift_input_source = (
        str(shift_input_source_override)
        if shift_input_source_override is not None
        else str(stage2_model_cfg.get("shift_input_source", "latent_mu"))
    )
    cond_pool_mode = str(stage2_model_cfg.get("cond_pool_mode", "sum"))
    cond_l2_norm = bool(stage2_model_cfg.get("cond_l2_norm", False))
    stage1_hidden_dim = int(stage1_model_cfg.get("hidden_dim", 1000))
    stage1_noise_rate = float(stage1_model_cfg.get("noise_rate", 0.1))
    stage1_kl_weight = float(stage1_model_cfg.get("kl_weight", 5e-4))

    model = TriShift(data, device="cuda" if torch.cuda.is_available() else "cpu")
    model.model_init(
        x_dim=adata.shape[1],
        z_dim=int(stage1_model_cfg.get("z_dim", 100)),
        cond_dim=embd_df.shape[1],
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
        shift_predict_delta=shift_predict_delta,
        shift_use_cross_attention=shift_use_cross_attention,
        shift_cross_attn_heads=shift_cross_attn_heads,
        shift_cross_attn_dropout=shift_cross_attn_dropout,
        shift_use_transformer_block=shift_use_transformer_block,
        shift_transformer_layers=shift_transformer_layers,
        shift_transformer_ff_mult=shift_transformer_ff_mult,
        shift_transformer_dropout=shift_transformer_dropout,
        shift_transformer_readout=shift_transformer_readout,
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
    """Scouter-compatible list split by ratios (same implementation style as Scouter eval)."""
    random.seed(seed)
    random.shuffle(lst)

    total_ratio = sum(ratios)
    total_length = len(lst)
    num_splits = len(ratios)

    if total_length < num_splits:
        raise ValueError("The length of the list must be >= number of ratios")

    sizes = [max(1, int(total_length * ratio / total_ratio)) for ratio in ratios]

    while sum(sizes) > total_length:
        sizes[sizes.index(max(sizes))] -= 1

    sizes[-1] += total_length - sum(sizes)

    splits: list[list[str]] = []
    current_index = 0
    for size in sizes:
        splits.append(lst[current_index : current_index + size])
        current_index += size

    return splits


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
    """Scouter-compatible Norman subgroup split builder."""
    uniq_perts = pert_list.copy()
    uniq_perts.remove("ctrl")
    uniq_combos = [p for p in uniq_perts if "ctrl" not in p.split("+")]
    uniq_singles = [p for p in uniq_perts if "ctrl" in p.split("+")]

    test_single, val_single, train_single = _split_list_by_ratio(
        uniq_singles, [0.2, 0.05, 0.75], seed=seed
    )

    combo_seen0 = [
        p
        for p in uniq_combos
        if sum([i + "+ctrl" in train_single for i in p.split("+")]) == 0
    ]
    combo_seen1 = [
        p
        for p in uniq_combos
        if sum([i + "+ctrl" in train_single for i in p.split("+")]) == 1
    ]
    combo_seen2 = [
        p
        for p in uniq_combos
        if sum([i + "+ctrl" in train_single for i in p.split("+")]) == 2
    ]

    test_seen0, val_seen0, train_seen0 = _split_list_by_ratio(
        combo_seen0, [0.2, 0.1, 0.7], seed=seed
    )
    test_seen1, val_seen1, train_seen1 = _split_list_by_ratio(
        combo_seen1, [0.2, 0.1, 0.7], seed=seed
    )
    test_seen2, val_seen2, train_seen2 = _split_list_by_ratio(
        combo_seen2, [0.2, 0.1, 0.7], seed=seed
    )

    test_all = test_single + test_seen0 + test_seen1 + test_seen2
    val_all = val_single + val_seen0 + val_seen1 + val_seen2
    train_all = train_single + train_seen0 + train_seen1 + train_seen2

    group = [
        "train" if p in train_all else "val" if p in val_all else "test"
        for p in uniq_perts
    ]
    subgroup_list = []
    for p in uniq_perts:
        if p in uniq_singles:
            subgroup_list.append("single")
        elif p in combo_seen0:
            subgroup_list.append("seen0")
        elif p in combo_seen1:
            subgroup_list.append("seen1")
        elif p in combo_seen2:
            subgroup_list.append("seen2")

    return pd.DataFrame({"group": group, "subgroup": subgroup_list}, index=uniq_perts)


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
    stage2_model_cfg = model_root.get("stage2", {})
    model_cfg = model_root.get("stage3", {})

    emb_table = torch.tensor(embd_df.values, dtype=torch.float32)
    mode = defaults.get("matching_mode", "knn")
    k = int(defaults.get("k_topk", 5))
    train_mode = defaults.get("train_mode", "joint")
    valid_train_modes = {"joint", "sequential", "stage3_only", "latent_decoder"}
    if train_mode not in valid_train_modes:
        raise ValueError(
            f"Unsupported train_mode={train_mode}. "
            f"Expected one of: {sorted(valid_train_modes)}"
        )
    stage1_use_train_split = bool(ablation_cfg.get("stage1_use_train_split", False))
    topk_strategy = str(ablation_cfg.get("topk_strategy", "random"))
    sample_soft_ctrl = bool(ablation_cfg.get("sample_soft_ctrl", True))
    per_condition_ot = bool(ablation_cfg.get("per_condition_ot", False))
    reuse_ot_cache = bool(ablation_cfg.get("reuse_ot_cache", False))
    reuse_z_mu_cache = bool(ablation_cfg.get("reuse_z_mu_cache", False))
    latent_loss_type = str(ablation_cfg.get("latent_loss_type", "gears"))
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
    disable_loss_z_supervision = not shift_predict_delta_cfg
    seq_joint_enable = bool(ablation_cfg.get("sequential_joint_finetune", False))
    seq_joint_epochs = int(ablation_cfg.get("joint_finetune_epochs", 6))
    seq_joint_lr_scale = float(ablation_cfg.get("joint_finetune_lr_scale", 0.2))
    shift_input_source_cfg = str(stage2_model_cfg.get("shift_input_source", "latent_mu"))
    shift_input_source_eff = shift_input_source_cfg
    gen_state_source = "compressor"
    if train_mode == "latent_decoder":
        gen_state_source = "latent_mu"
        if shift_input_source_cfg != "latent_mu":
            print(
                "[config] train_mode=latent_decoder ignores "
                f"model.stage2.shift_input_source={shift_input_source_cfg}; "
                "forcing latent_mu"
            )
        shift_input_source_eff = "latent_mu"
        if seq_joint_enable:
            print(
                "[config] train_mode=latent_decoder ignores "
                "ablation.sequential_joint_finetune settings"
            )
    cache_dir = Path("artifacts") / "cache" / "topk"
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_dir_path = Path(out_dir) if out_dir is not None else (Path("artifacts") / "results" / name)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    # Snapshot the exact configs used for this run for reproducibility.
    # Keep names stable for downstream scripts.
    try:
        _dump_yaml(out_dir_path / "defaults_used.yaml", defaults)
        _dump_yaml(out_dir_path / "paths_used.yaml", cfg)
    except Exception as exc:
        print(f"[run] warning: failed to write config snapshots ({exc})")

    try:
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
            "n_eval_ensemble": int(defaults.get("n_eval_ensemble", 300)),
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

    metrics_all = []
    n_splits = (
        int(run_cfg.get("n_splits", dataset_cfg["multi_split_default"]))
        if run_cfg.get("multi_split", False)
        else 1
    )
    n_eval_ensemble = int(defaults.get("n_eval_ensemble", 300))
    if fast:
        stage1_epochs = 1
        stage23_epochs = 1
        stage2_epochs = 1
        stage3_epochs = 1
        n_splits = 1
        n_eval_ensemble = min(n_eval_ensemble, 20)
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
            split_dict = data.split_by_condition(seed=split_id)
        train_logs: dict = {"split_id": split_id, "train_mode": train_mode}

        print("[run] init model")
        model = _init_model(
            data,
            adata,
            embd_df,
            stage1_model_cfg,
            stage2_model_cfg,
            model_cfg,
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
        elif train_mode == "latent_decoder":
            train_logs["stage23_latent_decoder_joint"] = model.train_stage23_joint(
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
            if seq_joint_enable and seq_joint_epochs > 0:
                finetune_lr = float(stage23_cfg.get("lr", 1e-3)) * seq_joint_lr_scale
                train_logs["stage23_joint_finetune"] = model.train_stage23_joint(
                    split_dict=split_dict,
                    emb_table=emb_table,
                    mode=mode,
                    k=k,
                    split_id=split_id,
                    epochs=seq_joint_epochs,
                    batch_size=int(stage23_cfg.get("batch_size", 64)),
                    lr=finetune_lr,
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
        metrics_df = model.evaluate(
            split_dict=split_dict,
            emb_table=emb_table,
            split_id=split_id,
            n_ensemble=n_eval_ensemble,
            base_seed=base_seed,
        )
        model.export_predictions(
            split_dict=split_dict,
            emb_table=emb_table,
            split_id=split_id,
            n_ensemble=n_eval_ensemble,
            base_seed=base_seed,
            out_path=str(out_dir_path / f"trishift_{name}_{split_id}.pkl"),
        )
        metrics_all.append(metrics_df)

    if metrics_all:
        metrics_df_all = pd.concat(metrics_all, ignore_index=True)
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
