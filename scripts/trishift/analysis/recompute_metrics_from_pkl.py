from __future__ import annotations

from pathlib import Path
import argparse
import pickle
import sys
from datetime import datetime

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from trishift._external_metrics import regression_r2_safe
from trishift._utils import load_yaml, load_adata
from scripts.common.split_utils import norman_subgroup


def _safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64).reshape(-1)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    if aa.size < 2 or bb.size < 2:
        return float("nan")
    if (not np.isfinite(aa).all()) or (not np.isfinite(bb).all()):
        return float("nan")
    if float(np.std(aa)) <= 0.0 or float(np.std(bb)) <= 0.0:
        return float("nan")
    return float(pearsonr(aa, bb)[0])


def _resolve_mean_metric_keys(numeric_means: pd.Series) -> list[str]:
    preferred_order = [
        "pearson",
        "nmse",
        "mse_pred",
        "mse_ctrl",
        "deg_mean_r2",
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


def _write_mean_metrics(path: Path, metrics_df: pd.DataFrame) -> None:
    numeric_means = metrics_df.mean(numeric_only=True)
    mean_pearson = float(numeric_means.get("pearson", float("nan")))
    keys = _resolve_mean_metric_keys(numeric_means)

    lines = [f"{mean_pearson}\n"]
    for key in keys:
        lines.append(f"mean_{key}={float(numeric_means[key])}\n")
    _append_subgroup_mean_lines(lines, metrics_df, keys)
    path.write_text("".join(lines), encoding="utf-8")


def _pkl_path(result_dir: Path, dataset: str, split_id: int, tag: str | None, both_mode: bool) -> Path:
    if both_mode and tag is not None:
        return result_dir / f"trishift_{dataset}_{split_id}_{tag}.pkl"
    return result_dir / f"trishift_{dataset}_{split_id}.pkl"


def _ts_local() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _resolve_split_ids(defaults: dict, split_ids_arg: str) -> list[int]:
    if split_ids_arg.strip():
        out = [int(x.strip()) for x in split_ids_arg.split(",") if x.strip()]
        if not out:
            raise ValueError("--split_ids parsed to empty list")
        return out
    run_cfg = defaults.get("run", {})
    if bool(run_cfg.get("multi_split", True)):
        n_splits = int(run_cfg.get("n_splits", 5))
        return list(range(1, n_splits + 1))
    return [1]


def _norman_subgroup_map_by_split(dataset_path: str, split_ids: list[int]) -> dict[int, dict[str, str]]:
    adata = load_adata(dataset_path)
    all_conds = sorted(set(adata.obs["condition"].astype(str).tolist()))
    if "ctrl" not in all_conds:
        all_conds = ["ctrl"] + all_conds
    out: dict[int, dict[str, str]] = {}
    for split_id in split_ids:
        sub_df = norman_subgroup(all_conds.copy(), seed=int(split_id))
        out[int(split_id)] = sub_df["subgroup"].astype(str).to_dict()
    return out


def _compute_rows_from_pkl(
    pkl_path: Path,
    split_id: int,
    subgroup_map: dict[str, str] | None,
) -> list[dict]:
    with pkl_path.open("rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict):
        raise TypeError(f"Unexpected pkl payload: {pkl_path} type={type(payload)}")

    rows: list[dict] = []
    for cond, obj in payload.items():
        pred = np.asarray(obj.get("Pred"), dtype=np.float32)
        ctrl = np.asarray(obj.get("Ctrl"), dtype=np.float32)
        truth = np.asarray(obj.get("Truth"), dtype=np.float32)
        if pred.ndim != 2 or ctrl.ndim != 2 or truth.ndim != 2:
            continue
        if pred.shape[1] == 0 or ctrl.shape[1] == 0 or truth.shape[1] == 0:
            continue

        pred_mean = pred.mean(axis=0)
        ctrl_mean = ctrl.mean(axis=0)
        true_mean = truth.mean(axis=0)
        mse_pred = float(np.mean((true_mean - pred_mean) ** 2))
        mse_ctrl = float(np.mean((true_mean - ctrl_mean) ** 2))
        nmse = float(mse_pred / mse_ctrl) if mse_ctrl > 0 else float("nan")
        pear = _safe_pearson(true_mean - ctrl_mean, pred_mean - ctrl_mean)
        deg_mean_r2 = float(regression_r2_safe(true_mean - ctrl_mean, pred_mean - ctrl_mean))

        row = {
            "condition": str(cond),
            "mse_pred": mse_pred,
            "mse_ctrl": mse_ctrl,
            "nmse": nmse,
            "pearson": pear,
            "deg_mean_r2": deg_mean_r2,
            "split_id": int(split_id),
            "n_ensemble": int(pred.shape[0]),
        }
        if subgroup_map is not None:
            row["subgroup"] = str(subgroup_map.get(str(cond), "unknown"))
        rows.append(row)
    return rows


def _write_outputs_for_tag(
    pkl_dir: Path,
    out_dir: Path,
    dataset: str,
    split_ids: list[int],
    tag: str | None,
    both_mode: bool,
    subgroup_maps: dict[int, dict[str, str]] | None,
) -> pd.DataFrame:
    rows: list[dict] = []
    for split_id in split_ids:
        pkl_path = _pkl_path(pkl_dir, dataset, split_id, tag=tag, both_mode=both_mode)
        if not pkl_path.exists():
            raise FileNotFoundError(f"Missing pkl for split {split_id}: {pkl_path}")
        split_sub = subgroup_maps.get(split_id) if subgroup_maps is not None else None
        rows.extend(_compute_rows_from_pkl(pkl_path, split_id=split_id, subgroup_map=split_sub))

    if not rows:
        raise ValueError("No rows computed from pkl payloads")
    df = pd.DataFrame(rows)
    col_order = [
        "condition",
        "subgroup",
        "mse_pred",
        "mse_ctrl",
        "nmse",
        "pearson",
        "deg_mean_r2",
        "split_id",
        "n_ensemble",
    ]
    keep = [c for c in col_order if c in df.columns]
    tail = [c for c in df.columns if c not in keep]
    df = df[keep + tail]

    if both_mode and tag is not None:
        metrics_path = out_dir / f"metrics_{tag}.csv"
        mean_path = out_dir / f"mean_pearson_{tag}.txt"
    else:
        metrics_path = out_dir / "metrics.csv"
        mean_path = out_dir / "mean_pearson.txt"
    df.to_csv(metrics_path, index=False)
    _write_mean_metrics(mean_path, df)
    print(f"[pkl-eval] wrote {metrics_path}")
    print(f"[pkl-eval] wrote {mean_path}")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Recompute TriShift metrics/mean files directly from exported pkl (no retraining)."
    )
    ap.add_argument("--dataset", required=True, help="dataset name, e.g. norman")
    ap.add_argument(
        "--result_dir",
        default="",
        help="directory containing trishift_<dataset>_<split>.pkl (default: artifacts/results/<dataset>)",
    )
    ap.add_argument(
        "--out_root",
        default="",
        help="output root directory (default: same as --result_dir)",
    )
    ap.add_argument("--defaults", default="configs/defaults.yaml", help="defaults yaml path")
    ap.add_argument("--paths", default="configs/paths.yaml", help="paths yaml path")
    ap.add_argument(
        "--split_ids",
        default="",
        help="optional comma-separated split ids (default: infer from defaults.run)",
    )
    args = ap.parse_args()

    dataset = str(args.dataset).strip()
    pkl_dir = Path(args.result_dir).resolve() if args.result_dir else (REPO_ROOT / "artifacts" / "results" / dataset)
    out_root = Path(args.out_root).resolve() if args.out_root else pkl_dir
    out_dir = out_root / f"{_ts_local()}_pkl_eval"
    defaults = load_yaml(str(Path(args.defaults).resolve()))
    paths_cfg = load_yaml(str(Path(args.paths).resolve()))
    split_ids = _resolve_split_ids(defaults, str(args.split_ids))

    eval_ctrl_pool_mode = str(defaults.get("eval_ctrl_pool_mode", "random_train_ctrl"))
    eval_genept_distance = str(defaults.get("eval_genept_distance", "cosine"))
    both_mode = (
        eval_ctrl_pool_mode == "nearest_genept_ot_pool"
        and eval_genept_distance == "both"
    )
    if both_mode:
        tags: list[str | None] = ["cosine", "l2"]
    else:
        tags = [None]

    subgroup_maps: dict[int, dict[str, str]] | None = None
    if dataset == "norman":
        datasets_map = paths_cfg.get("datasets", {})
        dataset_path = datasets_map.get(dataset)
        if dataset_path is None:
            raise ValueError(f"dataset '{dataset}' missing in paths.yaml datasets")
        subgroup_maps = _norman_subgroup_map_by_split(str(dataset_path), split_ids)

    print(
        f"[pkl-eval] dataset={dataset} "
        f"splits={split_ids} mode={eval_ctrl_pool_mode} distance={eval_genept_distance}"
    )
    print(
        "[pkl-eval] note: pkl payload contains DE-subset arrays only; "
        "recomputed metrics include pearson/nmse/mse_pred/mse_ctrl/deg_mean_r2."
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[pkl-eval] pkl_dir={pkl_dir}")
    print(f"[pkl-eval] out_dir={out_dir}")

    tag_to_df: dict[str, pd.DataFrame] = {}
    for tag in tags:
        df = _write_outputs_for_tag(
            pkl_dir=pkl_dir,
            out_dir=out_dir,
            dataset=dataset,
            split_ids=split_ids,
            tag=tag,
            both_mode=both_mode,
            subgroup_maps=subgroup_maps,
        )
        tag_to_df["main" if tag is None else tag] = df

    if both_mode and "cosine" in tag_to_df:
        cosine_df = tag_to_df["cosine"]
        cosine_df.to_csv(out_dir / "metrics.csv", index=False)
        _write_mean_metrics(out_dir / "mean_pearson.txt", cosine_df)
        print(f"[pkl-eval] wrote {out_dir / 'metrics.csv'} (alias from cosine)")
        print(f"[pkl-eval] wrote {out_dir / 'mean_pearson.txt'} (alias from cosine)")


if __name__ == "__main__":
    main()
