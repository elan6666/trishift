from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial.distance import cdist

import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts.trishift.analysis._result_adapter import (
    load_payload_item,
    parse_models,
    parse_split_ids,
    resolve_model_spec,
    ts_local,
    write_run_meta,
)
from scripts.trishift.analysis.stratified_benchmark import (
    _build_metadata_from_payload,
    _load_dataset_split,
    _qcut_labels,
)
from scripts.systema._core.baselines_core import (
    _ctrl_pool_from_split,
    _get_deg_idx,
    _mean_vector_from_adata,
    _systema_matching_mean_mu,
    _systema_nonctl_mean_mu,
)
from trishift.TriShiftData import TriShiftData
from trishift._external_metrics import average_of_perturbation_centroids
from trishift._utils import apply_alias_mapping, load_adata, load_embedding_df, load_yaml, normalize_condition


DATASET_EMBEDDING_KEYS = {
    "adamson": "emb_b",
    "dixit": "emb_c",
    "norman": "emb_a",
    "replogle_k562_essential": "emb_c",
    "replogle_rpe1_essential": "emb_b",
}


def _resolve_paths_yaml(paths_path: str | Path) -> Path:
    p = Path(paths_path)
    candidates = []
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.extend([Path.cwd() / p, REPO_ROOT / p, REPO_ROOT / "configs" / p.name])
    for cand in candidates:
        cand = cand.resolve()
        if cand.exists():
            return cand
    raise FileNotFoundError(f"Could not resolve paths yaml: {paths_path}")


def _load_data(dataset: str, paths_path: str | Path) -> TriShiftData:
    dataset_key = str(dataset).strip()
    cfg = load_yaml(str(_resolve_paths_yaml(paths_path)))
    adata = load_adata(cfg["datasets"][dataset_key])
    emb_key = DATASET_EMBEDDING_KEYS[dataset_key]
    embd_df = load_embedding_df(cfg["embeddings"][emb_key])
    embd_df = apply_alias_mapping(embd_df, dataset_key)
    data = TriShiftData(adata, embd_df)
    data.setup_embedding_index()
    data.build_or_load_degs()
    return data


def _mean_dense(X: Any) -> np.ndarray:
    if sp.issparse(X):
        return np.asarray(X.mean(axis=0), dtype=np.float32).reshape(-1)
    return np.asarray(X, dtype=np.float32).mean(axis=0).reshape(-1)


def _pearson_safe(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2 or np.allclose(np.std(x), 0.0) or np.allclose(np.std(y), 0.0):
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _project(vec: np.ndarray, axis: np.ndarray) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float64).reshape(-1)
    a = np.asarray(axis, dtype=np.float64).reshape(-1)
    denom = float(np.dot(a, a))
    if denom <= 0 or not np.isfinite(denom):
        return np.zeros_like(v, dtype=np.float64)
    scale = float(np.dot(v, a)) / denom
    return scale * a


def _projection_ratio(vec: np.ndarray, axis: np.ndarray) -> float:
    v = np.asarray(vec, dtype=np.float64).reshape(-1)
    proj = _project(v, axis)
    denom = float(np.linalg.norm(v))
    if denom <= 0 or not np.isfinite(denom):
        return float("nan")
    return float(np.linalg.norm(proj) / denom)


def _residualized_systema_corr(
    truth_mean: np.ndarray,
    pred_mean: np.ndarray,
    reference: np.ndarray,
    generic_axis: np.ndarray,
    deg_idx: np.ndarray,
) -> float:
    delta_true = np.asarray(truth_mean - reference, dtype=np.float64).reshape(-1)
    delta_pred = np.asarray(pred_mean - reference, dtype=np.float64).reshape(-1)
    delta_true_res = delta_true - _project(delta_true, generic_axis)
    delta_pred_res = delta_pred - _project(delta_pred, generic_axis)
    idx = np.asarray(deg_idx, dtype=int).reshape(-1)
    idx = idx[(idx >= 0) & (idx < int(delta_true_res.size))]
    if idx.size == 0:
        return float("nan")
    return _pearson_safe(delta_true_res[idx], delta_pred_res[idx])


def _systema_corr(
    truth_mean: np.ndarray,
    pred_mean: np.ndarray,
    reference: np.ndarray,
    deg_idx: np.ndarray,
) -> float:
    delta_true = np.asarray(truth_mean - reference, dtype=np.float64).reshape(-1)
    delta_pred = np.asarray(pred_mean - reference, dtype=np.float64).reshape(-1)
    idx = np.asarray(deg_idx, dtype=int).reshape(-1)
    idx = idx[(idx >= 0) & (idx < int(delta_true.size))]
    if idx.size == 0:
        return float("nan")
    return _pearson_safe(delta_true[idx], delta_pred[idx])


def _systema_predictions_for_split(
    *,
    data: TriShiftData,
    train_adata,
    test_conds: list[str],
) -> dict[str, dict[str, np.ndarray]]:
    train_ctrl_adata = _ctrl_pool_from_split(data, train_adata)
    ctrl_mean = _mean_vector_from_adata(train_ctrl_adata)
    mu_pert = _systema_nonctl_mean_mu(train_adata)
    train_cond_set = set(train_adata.obs[data.label_key].astype(str).unique().tolist())
    centroid_cache: dict[str, np.ndarray] = {}
    out: dict[str, dict[str, np.ndarray]] = {
        "systema_nonctl_mean": {},
        "systema_matching_mean": {},
    }
    for cond in test_conds:
        out["systema_nonctl_mean"][cond] = np.asarray(mu_pert, dtype=np.float32).reshape(-1)
        out["systema_matching_mean"][cond] = _systema_matching_mean_mu(
            condition=cond,
            train_adata=train_adata,
            train_cond_set=train_cond_set,
            mu_pert=mu_pert,
            centroid_cache=centroid_cache,
        )
    return out


def _payload_predictions_for_split(dataset: str, model_name: str, split_id: int) -> dict[str, np.ndarray]:
    _, payload = load_payload_item(dataset=dataset, model_name=model_name, split_id=int(split_id), condition=None)
    out: dict[str, np.ndarray] = {}
    for cond, item in payload.items():
        cond_key = normalize_condition(str(cond))
        pred_full = np.asarray(item["Pred_full"] if "Pred_full" in item else item["Pred"], dtype=np.float32)
        out[cond_key] = pred_full.mean(axis=0).reshape(-1)
    return out


def _truth_centroids(data: TriShiftData, conditions: list[str]) -> dict[str, np.ndarray]:
    X_all = data.adata_all.X
    cond_series = data.adata_all.obs[data.label_key].astype(str).map(normalize_condition).values
    out: dict[str, np.ndarray] = {}
    for cond in conditions:
        mask = cond_series == cond
        if not np.any(mask):
            continue
        out[cond] = _mean_dense(X_all[mask])
    return out


def _centroid_accuracy_rows(dataset: str, split_id: int, pred_by_model: dict[str, dict[str, np.ndarray]], truth_by_cond: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    truth_conds = sorted(truth_by_cond.keys())
    if not truth_conds:
        return rows
    truth_mat = np.vstack([truth_by_cond[c] for c in truth_conds]).astype(np.float64, copy=False)
    for model_name, pred_map in pred_by_model.items():
        common = [c for c in truth_conds if c in pred_map]
        if not common:
            continue
        pred_mat = np.vstack([pred_map[c] for c in common]).astype(np.float64, copy=False)
        gt_mat = np.vstack([truth_by_cond[c] for c in common]).astype(np.float64, copy=False)
        dmat = cdist(pred_mat, truth_mat, metric="euclidean")
        truth_index_lookup = {cond: idx for idx, cond in enumerate(truth_conds)}
        per_scores = []
        top1 = []
        top3 = []
        ranks = []
        for row_idx, cond in enumerate(common):
            true_col_idx = truth_index_lookup[cond]
            self_dist = float(dmat[row_idx, true_col_idx])
            all_d = np.asarray(dmat[row_idx], dtype=np.float64)
            score = float((all_d > self_dist).sum() / max(1, len(all_d) - 1))
            rank = int(np.argsort(all_d).tolist().index(true_col_idx) + 1)
            per_scores.append(score)
            ranks.append(rank)
            top1.append(float(rank == 1))
            top3.append(float(rank <= 3))
        rows.append(
            {
                "dataset": dataset,
                "split_id": int(split_id),
                "model_name": model_name,
                "centroid_accuracy": float(np.nanmean(per_scores)),
                "centroid_top1": float(np.nanmean(top1)),
                "centroid_top3": float(np.nanmean(top3)),
                "centroid_mean_rank": float(np.nanmean(ranks)),
                "n_conditions": int(len(common)),
            }
        )
    return rows


def run_systema_mechanism_analysis(
    *,
    dataset: str,
    models: str | list[str] | tuple[str, ...],
    split_ids: int | str | list[int] | tuple[int, ...] = "1",
    out_root: str | Path | None = None,
    paths_path: str | Path = "configs/paths.yaml",
) -> dict[str, Any]:
    dataset_key = str(dataset).strip()
    model_requests = parse_models(models)
    split_list = parse_split_ids(split_ids)
    out_dir = Path(out_root).resolve() if out_root else (Path("artifacts/analysis") / f"{ts_local()}_systema_mechanism_{dataset_key}").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    data = _load_data(dataset_key, paths_path=paths_path)
    per_condition_rows: list[dict[str, Any]] = []
    centroid_rows: list[dict[str, Any]] = []
    metadata_frames: list[pd.DataFrame] = []
    skipped_models: list[dict[str, Any]] = []
    used_models: list[str] = []

    for split_id in split_list:
        split_dict = _load_dataset_split(dataset_key, int(split_id), paths_path=paths_path)
        train_adata = split_dict.get("train")
        split_test_conds = [normalize_condition(c) for c in split_dict.get("test_conds", [])]
        if train_adata is None:
            continue

        train_ctrl_adata = _ctrl_pool_from_split(data, train_adata)
        ctrl_mean_train = _mean_vector_from_adata(train_ctrl_adata)
        train_cond_arr = train_adata.obs[data.label_key].astype(str).values
        pert_reference = average_of_perturbation_centroids(
            X=train_adata.X,
            conditions=train_cond_arr,
            ctrl_label=data.ctrl_label,
        )
        generic_axis = np.asarray(pert_reference - ctrl_mean_train, dtype=np.float64).reshape(-1)

        payload_for_metadata_model = None
        payload_for_metadata = None
        active_test_conds = list(split_test_conds)

        for model_name in model_requests:
            spec = resolve_model_spec(model_name)
            if spec.kind != "payload":
                continue
            try:
                _, payload_for_metadata = load_payload_item(dataset=dataset_key, model_name=model_name, split_id=int(split_id), condition=None)
                payload_for_metadata_model = model_name
                active_test_conds = sorted({normalize_condition(str(c)) for c in payload_for_metadata.keys()})
                break
            except Exception:
                continue

        if not active_test_conds:
            continue

        truth_by_cond = _truth_centroids(data, active_test_conds)
        pred_by_model: dict[str, dict[str, np.ndarray]] = {}
        systema_pred_map = _systema_predictions_for_split(data=data, train_adata=train_adata, test_conds=active_test_conds)

        for model_name in model_requests:
            spec = resolve_model_spec(model_name)
            try:
                if spec.kind == "payload":
                    pred_map = _payload_predictions_for_split(dataset_key, model_name, int(split_id))
                elif spec.request_id in {"systema_nonctl_mean", "systema_matching_mean"}:
                    pred_map = systema_pred_map[spec.request_id]
                else:
                    skipped_models.append({"model_name": model_name, "split_id": int(split_id), "reason": "unsupported_kind"})
                    continue
            except Exception as exc:
                skipped_models.append({"model_name": model_name, "split_id": int(split_id), "reason": str(exc)})
                continue

            pred_by_model[model_name] = pred_map
            if model_name not in used_models:
                used_models.append(model_name)

            for cond in active_test_conds:
                if cond not in pred_map or cond not in truth_by_cond:
                    continue
                deg_idx = _get_deg_idx(data, cond)
                if deg_idx.size == 0:
                    continue
                truth_mean = truth_by_cond[cond]
                pred_mean = pred_map[cond]
                per_condition_rows.append(
                    {
                        "dataset": dataset_key,
                        "split_id": int(split_id),
                        "condition": cond,
                        "model_name": model_name,
                        "systema_corr_20de_allpert": _systema_corr(
                            truth_mean=truth_mean,
                            pred_mean=pred_mean,
                            reference=pert_reference,
                            deg_idx=deg_idx,
                        ),
                        "residualized_systema_corr_20de_allpert": _residualized_systema_corr(
                            truth_mean=truth_mean,
                            pred_mean=pred_mean,
                            reference=pert_reference,
                            generic_axis=generic_axis,
                            deg_idx=deg_idx,
                        ),
                        "generic_projection_ratio": _projection_ratio(pred_mean - ctrl_mean_train, generic_axis),
                    }
                )

        if payload_for_metadata is not None:
            metadata_frames.append(
                _build_metadata_from_payload(
                    dataset_key,
                    payload_for_metadata,
                    int(split_id),
                    [normalize_condition(c) for c in split_dict.get("train_conds", [])],
                    paths_path,
                )
            )

        centroid_rows.extend(
            _centroid_accuracy_rows(
                dataset=dataset_key,
                split_id=int(split_id),
                pred_by_model=pred_by_model,
                truth_by_cond=truth_by_cond,
            )
        )

    per_condition_df = pd.DataFrame(per_condition_rows)
    metadata_df = pd.concat(metadata_frames, ignore_index=True).drop_duplicates(["split_id", "condition"]) if metadata_frames else pd.DataFrame()
    if not metadata_df.empty and not per_condition_df.empty:
        metadata_df["train_distance_bin"], _ = _qcut_labels(metadata_df["train_test_distance"], ["near", "medium", "far"])
        metadata_df["effect_strength_bin"], _ = _qcut_labels(metadata_df["truth_ctrl_shift_norm"], ["weak", "medium", "strong"])
        per_condition_df = per_condition_df.merge(
            metadata_df[["split_id", "condition", "subgroup", "train_test_distance", "train_distance_bin", "truth_ctrl_shift_norm", "effect_strength_bin"]],
            on=["split_id", "condition"],
            how="left",
        )

    residual_summary_df = (
        per_condition_df.groupby(["dataset", "model_name"], as_index=False)[
            ["systema_corr_20de_allpert", "residualized_systema_corr_20de_allpert", "generic_projection_ratio"]
        ].mean(numeric_only=True)
        if not per_condition_df.empty
        else pd.DataFrame(columns=["dataset", "model_name", "systema_corr_20de_allpert", "residualized_systema_corr_20de_allpert", "generic_projection_ratio"])
    )
    centroid_df = pd.DataFrame(centroid_rows)
    centroid_summary_df = (
        centroid_df.groupby(["dataset", "model_name"], as_index=False)[
            ["centroid_accuracy", "centroid_top1", "centroid_top3", "centroid_mean_rank", "n_conditions"]
        ].mean(numeric_only=True)
        if not centroid_df.empty
        else pd.DataFrame(columns=["dataset", "model_name", "centroid_accuracy", "centroid_top1", "centroid_top3", "centroid_mean_rank", "n_conditions"])
    )
    difficulty_bin_df = (
        per_condition_df.groupby(["dataset", "model_name", "train_distance_bin"], as_index=False)[
            ["systema_corr_20de_allpert", "residualized_systema_corr_20de_allpert", "generic_projection_ratio"]
        ].mean(numeric_only=True)
        if not per_condition_df.empty and "train_distance_bin" in per_condition_df.columns
        else pd.DataFrame(columns=["dataset", "model_name", "train_distance_bin", "systema_corr_20de_allpert", "residualized_systema_corr_20de_allpert", "generic_projection_ratio"])
    )

    per_condition_df.to_csv(out_dir / "systema_mechanism_per_condition.csv", index=False, encoding="utf-8-sig")
    residual_summary_df.to_csv(out_dir / "residualized_systema_summary.csv", index=False, encoding="utf-8-sig")
    centroid_df.to_csv(out_dir / "centroid_accuracy_per_split.csv", index=False, encoding="utf-8-sig")
    centroid_summary_df.to_csv(out_dir / "centroid_accuracy_summary.csv", index=False, encoding="utf-8-sig")
    difficulty_bin_df.to_csv(out_dir / "difficulty_bin_generic_shift_summary.csv", index=False, encoding="utf-8-sig")
    metadata_df.to_csv(out_dir / "condition_metadata.csv", index=False, encoding="utf-8-sig")

    write_run_meta(
        out_dir / "run_meta.json",
        {
            "dataset": dataset_key,
            "models": model_requests,
            "used_models": used_models,
            "split_ids": split_list,
            "paths_path": str(_resolve_paths_yaml(paths_path)),
            "out_dir": str(out_dir),
            "skipped_models": skipped_models,
        },
    )
    return {
        "out_dir": out_dir,
        "per_condition_df": per_condition_df,
        "residual_summary_df": residual_summary_df,
        "centroid_df": centroid_df,
        "centroid_summary_df": centroid_summary_df,
        "difficulty_bin_df": difficulty_bin_df,
        "metadata_df": metadata_df,
        "skipped_models": skipped_models,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run Systema-inspired mechanism analyses on exported results.")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--models", default="trishift_nearest,trishift_random,systema_nonctl_mean,systema_matching_mean")
    ap.add_argument("--split_ids", default="1")
    ap.add_argument("--out_root", default="")
    ap.add_argument("--paths_path", default="configs/paths.yaml")
    args = ap.parse_args(argv)
    result = run_systema_mechanism_analysis(
        dataset=str(args.dataset).strip(),
        models=str(args.models).strip(),
        split_ids=str(args.split_ids).strip(),
        out_root=str(args.out_root).strip() or None,
        paths_path=str(args.paths_path).strip(),
    )
    print(f"out_dir: {result['out_dir']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
