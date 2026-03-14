from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error as mse
import yaml

import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
from scripts.common.git_utils import safe_git_commit as _common_safe_git_commit
from scripts.common.time_utils import ts_shanghai as _common_ts_shanghai
from scripts.common.yaml_utils import dump_yaml as _common_dump_yaml, load_yaml_file as _common_load_yaml_file

from trishift.TriShiftData import TriShiftData
from trishift._external_metrics import (
    average_of_perturbation_centroids,
    pearson_delta_reference_metrics,
    regression_r2_safe,
)
from trishift._utils import (
    apply_alias_mapping,
    load_adata,
    load_embedding_df,
    load_yaml,
    normalize_condition,
)


DATASET_CONFIG = {
    "adamson": {"emb_key": "emb_b"},
    "dixit": {"emb_key": "emb_c"},
    "norman": {"emb_key": "emb_a"},
    "replogle_k562_essential": {"emb_key": "emb_c"},
    "replogle_rpe1_essential": {"emb_key": "emb_b"},
}

PROFILE_DIR = Path(__file__).resolve().parents[1] / "baselines" / "configs"
PROFILE_ALLOWED_KEYS = {"dataset", "task_args"}


SYSTEMA_BASELINE_NAME_TO_DIR_SUFFIX = {
    "nonctl-mean": "systema_nonctl_mean",
    "matching-mean": "systema_matching_mean",
}

DEFAULT_METRIC_COLUMNS = [
    "condition",
    "mse_pred",
    "mse_ctrl",
    "nmse",
    "pearson",
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
    "split_id",
    "n_ensemble",
]


@dataclass(frozen=True)
class AlignSpec:
    run_dir: Path
    columns: list[str]
    split_to_conditions: dict[int, list[str]]
    row_values: dict[tuple[int, str], dict[str, object]]


def _now_shanghai_ts() -> str:
    return _common_ts_shanghai()


def _safe_git_commit() -> str | None:
    return _common_safe_git_commit(REPO_ROOT)


def _dump_yaml(path: Path, obj: dict) -> None:
    _common_dump_yaml(path, obj, allow_unicode=False)


def _load_profile(profile: str) -> dict:
    p = PROFILE_DIR / f"{profile}.yaml"
    if not p.exists():
        raise FileNotFoundError(f"Systema baselines profile not found: {p}")
    obj = _common_load_yaml_file(p)
    unknown = sorted(set(obj.keys()) - PROFILE_ALLOWED_KEYS)
    if unknown:
        raise ValueError(f"Unknown keys in Systema baselines profile {p.name}: {unknown}")
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
            val = float(sub_numeric_means[key])
            lines.append(f"subgroup_{g}_mean_{key}={val}\n")
        lines.append("\n")


def _write_mean_metrics(path: Path, metrics_df: pd.DataFrame) -> None:
    """Match scripts/run_dataset.py:_write_mean_metrics."""
    numeric_means = metrics_df.mean(numeric_only=True)
    mean_pearson = float(numeric_means.get("pearson", float("nan")))

    keys = _resolve_mean_metric_keys(numeric_means)

    lines = [f"{mean_pearson}\n"]
    for key in keys:
        val = float(numeric_means[key])
        lines.append(f"mean_{key}={val}\n")
    _append_subgroup_mean_lines(lines, metrics_df, keys)
    path.write_text("".join(lines), encoding="utf-8")


def _discover_max_run_idx(sweep_root: Path) -> int:
    max_idx = 0
    for p in sweep_root.iterdir():
        if not p.is_dir():
            continue
        m = re.match(r"^(\d{2})_", p.name)
        if not m:
            continue
        max_idx = max(max_idx, int(m.group(1)))
    return max_idx


def _choose_align_run(sweep_root: Path, align_to_run: str | None) -> Path:
    if align_to_run:
        p = Path(align_to_run)
        if not p.is_absolute():
            p = (sweep_root / p).resolve()
        if not p.exists():
            raise FileNotFoundError(f"--align_to_run not found: {p}")
        if not (p / "metrics.csv").exists():
            raise FileNotFoundError(f"align run missing metrics.csv: {p}")
        return p

    preferred = sweep_root / "01_B0_baseline"
    if (preferred / "metrics.csv").exists():
        return preferred

    for p in sorted(sweep_root.iterdir()):
        if not p.is_dir():
            continue
        if (p / "metrics.csv").exists():
            return p
    raise FileNotFoundError("No metrics.csv found in sweep; cannot align.")


def _load_align_spec(align_run_dir: Path) -> AlignSpec:
    p = align_run_dir / "metrics.csv"
    df = pd.read_csv(p)
    if "condition" not in df.columns or "split_id" not in df.columns:
        raise ValueError(f"align metrics.csv missing required columns: {p}")

    split_to_conditions: dict[int, list[str]] = {}
    for split_id, g in df.groupby("split_id", sort=True):
        split_to_conditions[int(split_id)] = g["condition"].astype(str).tolist()
    row_values = {
        (int(row["split_id"]), str(row["condition"])): row
        for row in df.to_dict(orient="records")
    }

    return AlignSpec(
        run_dir=align_run_dir,
        columns=[str(c) for c in df.columns.tolist()],
        split_to_conditions=split_to_conditions,
        row_values=row_values,
    )


def _align_spec_from_metrics_csv(metrics_csv: Path) -> AlignSpec:
    df = pd.read_csv(metrics_csv)
    if "condition" not in df.columns or "split_id" not in df.columns:
        raise ValueError(f"align metrics.csv missing required columns: {metrics_csv}")

    split_to_conditions: dict[int, list[str]] = {}
    for split_id, g in df.groupby("split_id", sort=True):
        split_to_conditions[int(split_id)] = g["condition"].astype(str).tolist()
    row_values = {
        (int(row["split_id"]), str(row["condition"])): row
        for row in df.to_dict(orient="records")
    }

    return AlignSpec(
        run_dir=metrics_csv.parent,
        columns=[str(c) for c in df.columns.tolist()],
        split_to_conditions=split_to_conditions,
        row_values=row_values,
    )


def _build_align_spec_from_splits(
    data: TriShiftData,
    splits: list[int],
    columns: list[str] | None = None,
) -> AlignSpec:
    split_to_conditions: dict[int, list[str]] = {}
    for split_id in splits:
        split_dict = data.split_by_condition(seed=int(split_id))
        split_to_conditions[int(split_id)] = [str(c) for c in split_dict.get("test_conds", [])]
    return AlignSpec(
        run_dir=Path("."),
        columns=list(columns or DEFAULT_METRIC_COLUMNS),
        split_to_conditions=split_to_conditions,
        row_values={},
    )


def _infer_dataset_from_run(run_dir: Path) -> str | None:
    meta = run_dir / "run_meta.json"
    if meta.exists():
        try:
            obj = json.loads(meta.read_text(encoding="utf-8"))
            if isinstance(obj, dict) and obj.get("dataset"):
                return str(obj["dataset"])
        except Exception:
            pass

    # Fallback: trishift_<dataset>_<split>.pkl
    for p in run_dir.glob("trishift_*_*.pkl"):
        m = re.match(r"^trishift_(.+)_\d+\.pkl$", p.name)
        if m:
            return m.group(1)
    return None


def _load_degs_cache(cache_path: Path) -> dict | None:
    if not cache_path.exists():
        return None
    try:
        import pickle

        with open(cache_path, "rb") as f:
            obj = pickle.load(f)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _ctrl_pool_from_split(data: TriShiftData, split_adata):
    if split_adata is None:
        return data.adata_ctrl
    cond = split_adata.obs[data.label_key].astype(str).values
    ctrl_mask = cond == data.ctrl_label
    if not np.any(ctrl_mask):
        print("[ctrl] warning: split has no ctrl; falling back to global ctrl pool")
        return data.adata_ctrl
    return split_adata[ctrl_mask]


def _mean_vector(X) -> np.ndarray:
    if sp.issparse(X):
        return np.asarray(X.mean(axis=0), dtype=np.float32).reshape(-1)
    return np.asarray(X, dtype=np.float32).mean(axis=0).reshape(-1)


def _mean_vector_from_adata(adata) -> np.ndarray:
    return _mean_vector(adata.X)


def _systema_nonctl_mean_mu(train_adata) -> np.ndarray:
    # Preserve the original Systema nonctl-mean semantics:
    # pert_adata = train_adata[control==0]; pert_mean = pert_adata.X.mean(axis=0)
    if "control" in train_adata.obs.columns:
        control = train_adata.obs["control"].to_numpy()
        pert_mask = np.asarray(control, dtype=int) == 0
    else:
        cond = train_adata.obs["condition"].astype(str).values
        pert_mask = cond != "ctrl"
    if not np.any(pert_mask):
        return _mean_vector_from_adata(train_adata)
    return _mean_vector_from_adata(train_adata[pert_mask])


def _condition_centroid(train_adata, cond: str) -> np.ndarray | None:
    cond_series = train_adata.obs["condition"].astype(str).values
    mask = cond_series == cond
    if not np.any(mask):
        return None
    return _mean_vector_from_adata(train_adata[mask])


def _systema_matching_mean_mu(
    condition: str,
    train_adata,
    train_cond_set: set[str],
    mu_pert: np.ndarray,
    centroid_cache: dict[str, np.ndarray],
) -> np.ndarray:
    # Preserve the original Systema matching-mean semantics, adapted to
    # TriShift-normalized conditions: ignore 'ctrl' tokens when building gene list.
    tokens = [t for t in str(condition).split("+") if t and t != "ctrl"]
    if len(tokens) == 0:
        return mu_pert

    X_perts = []
    for g in tokens:
        single = normalize_condition(f"{g}+ctrl")
        if single in train_cond_set:
            if single not in centroid_cache:
                c = _condition_centroid(train_adata, single)
                if c is not None:
                    centroid_cache[single] = c
            vec = centroid_cache.get(single)
            if vec is None:
                vec = mu_pert
        else:
            vec = mu_pert
        X_perts.append(vec)
    return np.asarray(np.mean(np.stack(X_perts, axis=0), axis=0), dtype=np.float32).reshape(-1)


def _get_deg_idx(data: TriShiftData, cond: str) -> np.ndarray:
    uns = data.adata_all.uns
    degs_non_dropout = uns.get("top20_degs_non_dropout", {}) or {}
    degs_final = uns.get("top20_degs_final", {}) or {}

    if cond in degs_non_dropout:
        deg_idx = np.asarray(degs_non_dropout.get(cond, []), dtype=int)
    else:
        deg_idx = np.asarray(degs_final.get(cond, []), dtype=int)

    remove_idx = np.asarray(data.cond_to_gene_idx.get(cond, []), dtype=int)
    if remove_idx.size > 0 and deg_idx.size > 0:
        deg_idx = np.setdiff1d(deg_idx, remove_idx)
    return np.asarray(deg_idx, dtype=int).reshape(-1)


def _compute_row(
    *,
    data: TriShiftData,
    cond: str,
    split_id: int,
    n_ensemble: int,
    X_all,
    cond_series: np.ndarray,
    ctrl_mean_all: np.ndarray,
    pred_mean: np.ndarray,
    pert_reference: np.ndarray,
    out_columns: list[str],
    align_row_values: dict[str, object] | None = None,
) -> dict:
    row: dict[str, object] = {c: np.nan for c in out_columns}
    if align_row_values:
        for key, val in align_row_values.items():
            if key in row:
                row[key] = val
    row["condition"] = cond
    row["split_id"] = int(split_id)
    row["n_ensemble"] = int(n_ensemble)

    mask = cond_series == cond
    if not np.any(mask):
        return row

    if sp.issparse(X_all):
        true_mean = np.asarray(X_all[mask].mean(axis=0), dtype=np.float32).reshape(-1)
    else:
        true_mean = np.asarray(X_all[mask], dtype=np.float32).mean(axis=0).reshape(-1)

    deg_idx = _get_deg_idx(data, cond)
    if deg_idx.size == 0:
        return row

    true_vec = true_mean[deg_idx].reshape(-1)
    pred_vec = pred_mean[deg_idx].reshape(-1)
    ctrl_vec = ctrl_mean_all[deg_idx].reshape(-1)

    mse_ctrl_val = float(mse(true_vec, ctrl_vec))
    mse_pred_val = float(mse(true_vec, pred_vec))
    nmse_val = float(mse_pred_val / mse_ctrl_val) if mse_ctrl_val > 0 else np.nan
    try:
        pearson_val = float(pearsonr(true_vec - ctrl_vec, pred_vec - ctrl_vec)[0])
    except Exception:
        pearson_val = np.nan
    deg_mean_r2_val = regression_r2_safe(
        true_vec - ctrl_vec,
        pred_vec - ctrl_vec,
    )

    systema = pearson_delta_reference_metrics(
        X_true=true_mean.reshape(-1),
        X_pred=pred_mean.reshape(-1),
        reference=pert_reference.reshape(-1),
        top20_de_idxs=deg_idx,
    )

    row["mse_pred"] = mse_pred_val
    row["mse_ctrl"] = mse_ctrl_val
    row["nmse"] = nmse_val
    row["pearson"] = pearson_val
    row["deg_mean_r2"] = float(deg_mean_r2_val)
    row["systema_corr_all_allpert"] = float(systema.get("corr_all_allpert", np.nan))
    row["systema_corr_20de_allpert"] = float(systema.get("corr_20de_allpert", np.nan))
    row["systema_corr_all_r2"] = float(systema.get("corr_all_r2", np.nan))
    row["systema_corr_deg_r2"] = float(systema.get("corr_deg_r2", np.nan))

    # scPRAM metrics intentionally not computed for Systema baselines.
    for k in list(row.keys()):
        if str(k).startswith("scpram_"):
            row[k] = np.nan
    return row


def _run_one_method(
    *,
    method: str,
    out_dir: Path,
    data: TriShiftData,
    align: AlignSpec,
    splits: list[int],
    n_ensemble: int,
    defaults: dict,
    paths_cfg: dict,
) -> None:
    if method not in SYSTEMA_BASELINE_NAME_TO_DIR_SUFFIX:
        raise ValueError(f"unknown method: {method}")

    out_dir.mkdir(parents=True, exist_ok=True)
    _dump_yaml(out_dir / "defaults_used.yaml", defaults)
    _dump_yaml(out_dir / "paths_used.yaml", paths_cfg)

    meta = {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "timestamp_shanghai": _now_shanghai_ts(),
        "git_commit": _safe_git_commit(),
        "baseline_method": method,
        "splits": [int(s) for s in splits],
        "n_ensemble": int(n_ensemble),
        "align_run_dir": str(align.run_dir),
        "align_metrics_csv": str(align.run_dir / "metrics.csv"),
    }
    (out_dir / "run_meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=False), encoding="utf-8"
    )

    X_all = data.adata_all.X
    cond_series = data.adata_all.obs[data.label_key].astype(str).values

    rows: list[dict] = []
    for split_id in splits:
        split_dict = data.split_by_condition(seed=int(split_id))
        train_adata = split_dict.get("train")
        if train_adata is None:
            raise ValueError("split_by_condition returned missing train split")

        train_ctrl_adata = _ctrl_pool_from_split(data, train_adata)
        ctrl_mean_all = _mean_vector_from_adata(train_ctrl_adata)

        mu_pert = _systema_nonctl_mean_mu(train_adata)
        train_cond_set = set(train_adata.obs[data.label_key].astype(str).unique().tolist())
        centroid_cache: dict[str, np.ndarray] = {}

        train_cond_arr = train_adata.obs[data.label_key].astype(str).values
        pert_reference = average_of_perturbation_centroids(
            X=train_adata.X,
            conditions=train_cond_arr,
            ctrl_label=data.ctrl_label,
        )

        conds = align.split_to_conditions.get(int(split_id), [])
        if not conds:
            raise ValueError(f"align spec missing split_id={split_id} conditions")

        for cond in conds:
            if method == "nonctl-mean":
                pred_mean = mu_pert
            else:
                pred_mean = _systema_matching_mean_mu(
                    condition=cond,
                    train_adata=train_adata,
                    train_cond_set=train_cond_set,
                    mu_pert=mu_pert,
                    centroid_cache=centroid_cache,
                )

            row = _compute_row(
                data=data,
                cond=str(cond),
                split_id=int(split_id),
                n_ensemble=int(n_ensemble),
                X_all=X_all,
                cond_series=cond_series,
                ctrl_mean_all=ctrl_mean_all,
                pred_mean=pred_mean,
                pert_reference=pert_reference,
                out_columns=align.columns,
                align_row_values=align.row_values.get((int(split_id), str(cond))),
            )
            rows.append(row)

    df = pd.DataFrame(rows, columns=align.columns)
    df.to_csv(out_dir / "metrics.csv", index=False)
    _write_mean_metrics(out_dir / "mean_pearson.txt", df)
    print(f"[systema_baselines] saved: {out_dir / 'metrics.csv'}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Generate Systema baselines (nonctl-mean / matching-mean) in TriShift metrics.csv format."
    )
    ap.add_argument("--profile", default="", help="dataset profile under scripts/systema/baselines/configs")
    ap.add_argument("--dataset", default="", help="dataset name (e.g. adamson/dixit/norman)")
    ap.add_argument("--splits", default="1,2,3", help="comma-separated split ids (default: 1,2,3)")
    ap.add_argument("--n_ensemble", type=int, default=300, help="value to write into metrics.csv (default 300)")
    ap.add_argument("--defaults", default="configs/defaults.yaml", help="defaults yaml path")
    ap.add_argument("--paths", default="configs/paths.yaml", help="paths yaml path")
    ap.add_argument(
        "--out_root",
        default="artifacts/results",
        help="output root dir (default: artifacts/results)",
    )
    ap.add_argument(
        "--align_metrics_csv",
        default="",
        help="optional metrics.csv path for split/condition/columns alignment",
    )
    ap.add_argument(
        "--sweep_root",
        default="",
        help="optional sweep root (legacy fallback align source)",
    )
    ap.add_argument("--align_to_run", default="", help="run dir to align split/condition order to")
    args = ap.parse_args(argv)

    profile = str(args.profile).strip()
    if profile:
        prof = _load_profile(profile)
        task_args = prof["task_args"]
        if not str(args.dataset).strip():
            args.dataset = prof["dataset"]
        if str(args.splits) == "1,2,3" and "splits" in task_args:
            args.splits = ",".join([str(x) for x in list(task_args["splits"])])
        if str(args.out_root) == "artifacts/results" and "out_root" in task_args:
            args.out_root = str(task_args["out_root"])
        if not str(args.align_metrics_csv).strip() and task_args.get("align_metrics_csv") not in (None, ""):
            args.align_metrics_csv = str(task_args["align_metrics_csv"])
        if not str(args.align_to_run).strip() and task_args.get("align_to_run") not in (None, ""):
            args.align_to_run = str(task_args["align_to_run"])

    dataset = args.dataset.strip()
    if not dataset:
        raise ValueError("Either --profile or --dataset must be provided")
    if dataset not in DATASET_CONFIG:
        raise ValueError(f"unsupported dataset={dataset}; add to DATASET_CONFIG in this script")

    splits = [int(s.strip()) for s in str(args.splits).split(",") if s.strip()]
    if not splits:
        raise ValueError("--splits is empty")

    defaults = load_yaml(str(Path(args.defaults)))
    paths_cfg = load_yaml(str(Path(args.paths)))

    h5ad_path = paths_cfg["datasets"][dataset]
    emb_key = DATASET_CONFIG[dataset]["emb_key"]
    emb_path = paths_cfg["embeddings"][emb_key]

    print("[systema_baselines] load data")
    adata = load_adata(h5ad_path)
    adata.uns = {}
    embd_df = load_embedding_df(emb_path)
    embd_df = apply_alias_mapping(embd_df, dataset)

    degs_cache_dir = Path("artifacts") / "cache" / "degs"
    degs_cache_path = degs_cache_dir / f"{dataset}_degs.pkl"
    degs_cache = _load_degs_cache(degs_cache_path)
    if degs_cache:
        adata.uns.update(degs_cache)
        print(f"[degs] loaded cache: {degs_cache_path}")

    print("[systema_baselines] init data")
    data = TriShiftData(adata, embd_df)
    data.setup_embedding_index()
    data.build_or_load_degs()

    align: AlignSpec | None = None
    align_metrics_csv = args.align_metrics_csv.strip()
    if align_metrics_csv:
        align_csv_path = Path(align_metrics_csv).resolve()
        if not align_csv_path.exists():
            raise FileNotFoundError(f"--align_metrics_csv not found: {align_csv_path}")
        align = _align_spec_from_metrics_csv(align_csv_path)
    elif args.align_to_run.strip():
        align_run_dir = Path(args.align_to_run.strip())
        if not align_run_dir.is_absolute():
            align_run_dir = align_run_dir.resolve()
        if not align_run_dir.exists():
            raise FileNotFoundError(f"--align_to_run not found: {align_run_dir}")
        align = _load_align_spec(align_run_dir)
    else:
        # Preferred default: align to artifacts/results/<dataset>/metrics.csv if present.
        default_align_csv = (Path(args.out_root) / dataset / "metrics.csv").resolve()
        if default_align_csv.exists():
            align = _align_spec_from_metrics_csv(default_align_csv)
        elif args.sweep_root.strip():
            sweep_root = Path(args.sweep_root).resolve()
            if not sweep_root.exists():
                raise FileNotFoundError(f"--sweep_root not found: {sweep_root}")
            align_run_dir = _choose_align_run(sweep_root, None)
            align = _load_align_spec(align_run_dir)
        else:
            print(
                "[systema_baselines] no align metrics found; "
                "falling back to split_by_condition test_conds order"
            )
            align = _build_align_spec_from_splits(data=data, splits=splits)

    out_root = (Path(args.out_root) / dataset).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    timestamp = _now_shanghai_ts()
    run_root = out_root / f"systema_baselines_{timestamp}"
    run_root.mkdir(parents=True, exist_ok=True)
    out0 = run_root / SYSTEMA_BASELINE_NAME_TO_DIR_SUFFIX["nonctl-mean"]
    out1 = run_root / SYSTEMA_BASELINE_NAME_TO_DIR_SUFFIX["matching-mean"]

    print(f"[systema_baselines] align source: {align.run_dir}")
    print(f"[systema_baselines] dataset={dataset} splits={splits} writing:\n  {out0}\n  {out1}")

    _run_one_method(
        method="nonctl-mean",
        out_dir=out0,
        data=data,
        align=align,
        splits=splits,
        n_ensemble=int(args.n_ensemble),
        defaults=defaults,
        paths_cfg=paths_cfg,
    )
    _run_one_method(
        method="matching-mean",
        out_dir=out1,
        data=data,
        align=align,
        splits=splits,
        n_ensemble=int(args.n_ensemble),
        defaults=defaults,
        paths_cfg=paths_cfg,
    )


if __name__ == "__main__":
    main()
