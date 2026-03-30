from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
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
from scripts.common.split_utils import (
    condition_sort as _shared_condition_sort,
    norman_subgroup as _shared_norman_subgroup,
)
from scripts.common.yaml_utils import load_yaml_file


@dataclass(frozen=True)
class BiolordDatasetConfig:
    data_rel: str
    splits: list[int]
    raw_patterns: list[str]
    norman_split: bool = False


PROFILE_DIR = Path(__file__).resolve().parents[1] / "eval" / "configs"
PROFILE_ALLOWED_KEYS = {"dataset", "task_args"}

DATASET_CONFIG = {
    "adamson": BiolordDatasetConfig(
        data_rel="data/adamson/perturb_processed.h5ad",
        splits=[1, 2, 3, 4, 5],
        raw_patterns=["biolord_adamson_{split}.pkl"],
    ),
    "dixit": BiolordDatasetConfig(
        data_rel="data/dixit/perturb_processed.h5ad",
        splits=[1, 2, 3, 4, 5],
        raw_patterns=["biolord_dixit_{split}.pkl"],
    ),
    "norman": BiolordDatasetConfig(
        data_rel="data/norman/perturb_processed.h5ad",
        splits=[1, 2, 3, 4, 5],
        raw_patterns=[
            "biolord_norman_combo_seen0_{split}.pkl",
            "biolord_norman_combo_seen1_{split}.pkl",
            "biolord_norman_combo_seen2_{split}.pkl",
            "biolord_norman_unseen_single_{split}.pkl",
        ],
        norman_split=True,
    ),
    "replogle_k562_essential": BiolordDatasetConfig(
        data_rel="data/replogle_k562_essential/perturb_processed.h5ad",
        splits=[1, 2, 3, 4, 5],
        raw_patterns=["biolord_k562_{split}.pkl"],
    ),
    "replogle_rpe1_essential": BiolordDatasetConfig(
        data_rel="data/replogle_rpe1_essential/perturb_processed.h5ad",
        splits=[1, 2, 3, 4, 5],
        raw_patterns=["biolord_rpe1_{split}.pkl"],
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


def _resolve_eval_data_path(name: str, cfg: BiolordDatasetConfig) -> Path:
    data_path_primary = LOCAL_DATA_ROOT / name / "perturb_processed.h5ad"
    if data_path_primary.exists():
        return data_path_primary
    paths_cfg_path = ROOT / "configs" / "paths.yaml"
    if paths_cfg_path.exists():
        paths_cfg = _utils.load_yaml(str(paths_cfg_path))
        datasets_map = paths_cfg.get("datasets", {})
        if name in datasets_map:
            data_path_alt = Path(datasets_map[name])
            if data_path_alt.exists():
                return data_path_alt
    raise FileNotFoundError(
        f"Missing evaluation h5ad for biolord dataset={name}. Tried {data_path_primary}"
    )


def _prepare_eval_adata(data_path: Path) -> ad.AnnData:
    adata = ad.read_h5ad(str(data_path))
    adata.obs["condition"] = adata.obs["condition"].astype(str).apply(condition_sort).astype("category")
    return adata


def _gene_names_from_adata(adata: ad.AnnData) -> np.ndarray:
    for col in ["gene_name", "gene_names"]:
        if col in adata.var.columns:
            return adata.var[col].astype(str).values
    return adata.var_names.astype(str).values


def _build_condition_alias_map(adata: ad.AnnData) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for _, row in adata.obs[["condition"]].drop_duplicates().reset_index(drop=True).iterrows():
        cond = condition_sort(str(row["condition"]))
        mapping[_utils.normalize_condition(cond)] = cond
    if "condition_name" in adata.obs.columns:
        for _, row in adata.obs[["condition", "condition_name"]].drop_duplicates().reset_index(drop=True).iterrows():
            cond = condition_sort(str(row["condition"]))
            cname = condition_sort(str(row["condition_name"]))
            mapping[_utils.normalize_condition(cname)] = cond
    return mapping


def _resolve_raw_result_roots(raw_result_root: str | Path | None) -> list[Path]:
    if raw_result_root:
        return [Path(raw_result_root).resolve()]
    return [
        (ROOT / "artifacts" / "raw_results" / "biolord").resolve(),
        (ROOT / "external" / "scouter" / "scouter_misc-main" / "scouter_misc-main" / "results" / "biolord").resolve(),
        (ROOT / "artifacts" / "results" / "biolord" / "raw").resolve(),
    ]


def _find_existing_raw_file(roots: list[Path], pattern: str) -> Path | None:
    for root in roots:
        p = root / pattern
        if p.exists():
            return p
    return None


def _load_raw_split_payload(
    dataset_name: str,
    split_id: int,
    cfg: BiolordDatasetConfig,
    roots: list[Path],
) -> dict[str, dict]:
    merged: dict[str, dict] = {}
    for pattern in cfg.raw_patterns:
        p = _find_existing_raw_file(roots, pattern.format(split=int(split_id)))
        if p is None:
            continue
        with p.open("rb") as handle:
            payload = pickle.load(handle)
        if not isinstance(payload, dict):
            raise TypeError(f"Unexpected biolord raw payload type at {p}: {type(payload)}")
        for cond, item in payload.items():
            merged[str(cond)] = item
    if not merged:
        searched = [str(root / pattern.format(split=int(split_id))) for root in roots for pattern in cfg.raw_patterns]
        raise FileNotFoundError(
            f"No raw biolord results found for dataset={dataset_name} split={split_id}. Searched: {searched}"
        )
    return merged


def _as_2d(arr) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float32)
    if out.ndim == 1:
        out = out.reshape(1, -1)
    return out


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
    adata: ad.AnnData,
    split_id: int,
    raw_payload: dict[str, dict],
) -> tuple[pd.DataFrame, dict]:
    alias_map = _build_condition_alias_map(adata)
    gene_names = _gene_names_from_adata(adata)
    train_mask = None
    split_key = f"split{int(split_id)}"
    if split_key in adata.obs.columns:
        train_mask = adata.obs[split_key].astype(str).eq("train").values
    ref_adata = adata[train_mask].copy() if train_mask is not None and np.any(train_mask) else adata
    pert_reference = average_of_perturbation_centroids(
        X=_utils.densify_X(ref_adata.X),
        conditions=ref_adata.obs["condition"].astype(str).values,
        ctrl_label="ctrl",
    )

    results = []
    export_payload = {}
    for raw_cond, item in raw_payload.items():
        cond = alias_map.get(_utils.normalize_condition(str(raw_cond)), condition_sort(str(raw_cond)))
        if not isinstance(item, dict):
            continue
        degs = np.asarray(item.get("DE_idx", []), dtype=int)
        if degs.size == 0:
            continue
        pred = _as_2d(item.get("Pred"))
        ctrl = _as_2d(item.get("Ctrl"))
        true = _as_2d(item.get("Truth"))
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
        scpram_metrics = _safe_scpram_metrics(true, pred, degs)
        results.append(
            {
                "condition": cond,
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
        export_payload[cond] = {
            "Pred": pred[:, degs],
            "Ctrl": ctrl[:, degs],
            "Truth": true[:, degs],
            "Pred_full": pred,
            "Ctrl_full": ctrl,
            "Truth_full": true,
            "DE_idx": degs,
            "DE_name": gene_names[degs] if degs.size > 0 else np.array([], dtype=gene_names.dtype),
            "gene_name_full": gene_names,
        }
    return pd.DataFrame(results), export_payload


def run_biolord_eval(
    name: str,
    base_seed: int = 24,
    export_notebook_pkl: bool = True,
    raw_result_root: str | Path | None = None,
) -> None:
    if name not in DATASET_CONFIG:
        raise ValueError(f"Unknown dataset: {name}")
    cfg = DATASET_CONFIG[name]
    data_path = _resolve_eval_data_path(name, cfg)
    adata = _prepare_eval_adata(data_path)
    roots = _resolve_raw_result_roots(raw_result_root)
    out_dir = ROOT / "artifacts" / "results" / "biolord" / name
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_all = []
    for split in cfg.splits:
        print(f"[biolord] dataset={name} split={split}")
        set_seeds(base_seed + int(split))
        raw_payload = _load_raw_split_payload(name, int(split), cfg, roots)
        subgroup_df = None
        if cfg.norman_split:
            subgroup_df = subgroup(list(adata.obs["condition"].astype(str).unique()), seed=int(split))
        metrics_df, export_payload = _compute_metrics_and_export_payload(
            adata=adata,
            split_id=int(split),
            raw_payload=raw_payload,
        )
        metrics_df = _attach_subgroup_column(metrics_df, subgroup_df)
        metrics_all.append(metrics_df)
        if export_notebook_pkl:
            out_pkl = out_dir / f"biolord_{name}_{split}.pkl"
            with out_pkl.open("wb") as handle:
                pickle.dump(export_payload, handle)
            print(f"[biolord] saved notebook payload: {out_pkl}")

    metrics_df_all = pd.concat(metrics_all, ignore_index=True)
    metrics_df_all.to_csv(out_dir / "metrics.csv", index=False)
    _write_mean_metrics(out_dir / "mean_pearson.txt", metrics_df_all)
    print(f"[biolord] saved metrics: {out_dir / 'metrics.csv'}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run biolord eval with TriShift metrics")
    parser.add_argument("--profile", default="", help="dataset profile under scripts/biolord/eval/configs")
    parser.add_argument("--name", default="", help="dataset name (legacy mode)")
    parser.add_argument("--seed", type=int, default=24, help="base seed")
    parser.add_argument("--raw_result_root", default="", help="optional directory containing raw biolord PKLs")
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
        raw_result_root = str(args.raw_result_root).strip() or str(task_args.get("raw_result_root", "")).strip() or None
        run_biolord_eval(
            prof["dataset"],
            base_seed=seed,
            export_notebook_pkl=export_notebook_pkl,
            raw_result_root=raw_result_root,
        )
        return

    name = str(args.name).strip()
    if not name:
        raise SystemExit("Provide --profile or --name")
    run_biolord_eval(
        name,
        base_seed=int(args.seed),
        export_notebook_pkl=not bool(args.no_export_notebook_pkl),
        raw_result_root=str(args.raw_result_root).strip() or None,
    )
