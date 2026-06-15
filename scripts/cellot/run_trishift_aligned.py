from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "external" / "cellot"))

from scripts.cellot.prepare_trishift_heldout import (
    DATASET_CONFIG,
    DATASET_CONFIG_PATHS,
    _cellot_model_config,
    _load_profile_config,
    _resolve_repo_path,
    _split_ids,
    _trishift_unseen_ctrl_split,
    _write_yaml,
)


def _load_data_with_degs(dataset: str, defaults: dict, paths: dict):
    from scripts.trishift._core.run_dataset_core import (
        DEGS_CACHE_KEYS,
        _degs_cache_name,
        _degs_cache_meta_ok,
        _load_degs_cache,
        _save_degs_cache,
    )
    from trishift.TriShiftData import TriShiftData
    from trishift._utils import apply_alias_mapping, load_adata, load_embedding_df

    h5ad_path = _resolve_repo_path(paths["datasets"][dataset])
    emb_key = str(defaults.get("emb_key", DATASET_CONFIG[dataset]["emb_key"]))
    if emb_key not in paths.get("embeddings", {}):
        raise ValueError(f"Embedding key {emb_key!r} missing from paths config")
    emb_path = _resolve_repo_path(paths["embeddings"][emb_key])
    adata = load_adata(h5ad_path)
    adata.uns = {}
    cache_dir = REPO_ROOT / "artifacts" / "cache" / "degs"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / _degs_cache_name(dataset, h5ad_path, adata)
    degs_cache = _load_degs_cache(cache_path)
    if degs_cache:
        adata.uns.update(degs_cache)
    embd_df = apply_alias_mapping(load_embedding_df(emb_path), dataset)
    data = TriShiftData(adata, embd_df)
    data.setup_embedding_index()
    if degs_cache:
        meta_ok, _ = _degs_cache_meta_ok(adata.uns)
        if not meta_ok:
            for key in (*DEGS_CACHE_KEYS, "rank_genes_groups", "top20_degs_final"):
                adata.uns.pop(key, None)
            degs_cache = None
    data.build_or_load_degs()
    if not degs_cache:
        _save_degs_cache(cache_path, adata.uns)
    return data


def _dense(adata) -> np.ndarray:
    from trishift import _utils

    arr = np.asarray(_utils.densify_X(adata.X), dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def _gene_names(data) -> np.ndarray:
    if data.var_gene_key in data.adata_all.var.columns:
        return data.adata_all.var[data.var_gene_key].astype(str).values
    return data.adata_all.var_names.astype(str).values


def _cond_embeddings(data, adata, conds: list[str]) -> dict[str, np.ndarray]:
    import torch
    from trishift._model import aggregate_cond_embedding

    emb = torch.tensor(data.embd_df.values, dtype=torch.float32)
    lookup: dict[str, np.ndarray] = {}
    obs_cond = adata.obs[data.label_key].astype(str)
    for cond in conds:
        mask = obs_cond.eq(str(cond)).values
        if not np.any(mask):
            continue
        idx = obs_cond.index[mask][0]
        embd_idx = adata.obs["embd_index"].loc[idx]
        cond_vec = aggregate_cond_embedding(emb, embd_idx)
        lookup[str(cond)] = cond_vec.detach().cpu().numpy().astype(np.float32)
    return lookup


def _aligned_split(data, *, dataset: str, split_id: int, defaults: dict) -> dict:
    if dataset == "scgen_pbmc_celltype":
        from scripts.trishift._core.run_dataset_core import (
            DATASET_CONFIG as CORE_DATASET_CONFIG,
            _split_by_dataset_config_policy,
        )

        split_dict = _split_by_dataset_config_policy(
            data,
            dataset_cfg=CORE_DATASET_CONFIG[dataset],
            seed=int(split_id),
        )
        if split_dict is None:
            raise ValueError("PBMC CellOT alignment requires the celltype_seen_perturbation split policy")
        return split_dict
    return _trishift_unseen_ctrl_split(data, dataset=dataset, split_id=int(split_id), defaults=defaults)


def _nearest_train_condition(
    test_cond: str,
    train_map: dict[str, np.ndarray],
    test_map: dict[str, np.ndarray],
) -> str | None:
    if test_cond not in test_map or not train_map:
        return None
    test_vec = np.asarray(test_map[test_cond], dtype=np.float32).reshape(-1)
    t_norm = float(np.linalg.norm(test_vec))
    scored: list[tuple[str, float]] = []
    for cond, vec in train_map.items():
        train_vec = np.asarray(vec, dtype=np.float32).reshape(-1)
        denom = t_norm * float(np.linalg.norm(train_vec))
        score = -np.inf if denom <= 1e-12 else float(np.dot(test_vec, train_vec) / denom)
        scored.append((str(cond), score))
    if not scored:
        return None
    best = max(score for _, score in scored)
    cands = sorted(cond for cond, score in scored if abs(score - best) <= 1e-12)
    return cands[0] if cands else None


def _task_config(h5ad_path: Path, target: str, batch_size: int, random_state: int) -> dict[str, Any]:
    return {
        "data": {
            "type": "cell",
            "source": "ctrl",
            "target": str(target),
            "condition": "cellot_condition",
            "path": str(h5ad_path),
        },
        "dataloader": {
            "batch_size": int(batch_size),
            "shuffle": True,
        },
        "datasplit": {
            "name": "train_test",
            "groupby": "cellot_condition",
            "test_size": 0.2,
            "random_state": int(random_state),
        },
    }


def _patch_cellot_py310_compat() -> None:
    import collections
    import collections.abc

    for name in ("Iterable", "Mapping", "MutableMapping", "Sequence", "MutableSequence"):
        if not hasattr(collections, name) and hasattr(collections.abc, name):
            setattr(collections, name, getattr(collections.abc, name))


def _sanitize_cellot_obs(adata, label_key: str):
    keep = []
    for col in [label_key, "condition", "cell_type", "split", "transport", "cellot_condition"]:
        if col in adata.obs.columns and col not in keep:
            keep.append(col)
    adata.obs = adata.obs.loc[:, keep].copy()
    for col in adata.obs.columns:
        adata.obs[col] = adata.obs[col].astype(str).astype("category")
    adata.uns = {}
    return adata


def _materialize_train_condition(
    *,
    data,
    split_dict: dict,
    train_condition: str,
    h5ad_path: Path,
) -> dict[str, int]:
    import anndata as ad

    label_key = data.label_key
    ctrl_label = data.ctrl_label
    train = split_dict["train"]
    obs_cond = train.obs[label_key].astype(str)
    train_ctrl = train[obs_cond.eq(ctrl_label).values].copy()
    train_target = train[obs_cond.eq(str(train_condition)).values].copy()
    if train_ctrl.n_obs == 0 or train_target.n_obs == 0:
        return {"train_source_n": int(train_ctrl.n_obs), "train_target_n": int(train_target.n_obs)}
    parts = []
    for transport, adata_part in (
        ("source", train_ctrl),
        ("target", train_target),
    ):
        adata_part.obs["split"] = "input"
        adata_part.obs["transport"] = transport
        adata_part.obs["cellot_condition"] = ctrl_label if transport == "source" else str(train_condition)
        parts.append(_sanitize_cellot_obs(adata_part, label_key))
    combined = ad.concat(parts, axis=0, join="outer", merge="same", index_unique="-cellot")
    h5ad_path.parent.mkdir(parents=True, exist_ok=True)
    combined.write_h5ad(h5ad_path)
    return {
        "train_source_n": int(train_ctrl.n_obs),
        "train_target_n": int(train_target.n_obs),
        "h5ad_n_obs": int(combined.n_obs),
        "h5ad_n_vars": int(combined.n_vars),
    }


def _train_one_map(
    *,
    cond_dir: Path,
    condition: str,
    h5ad_path: Path,
    batch_size: int,
    n_iters: int,
    random_state: int,
    force: bool,
) -> Path:
    task_path = cond_dir / "task.yaml"
    model_path = cond_dir / "model.yaml"
    outdir = cond_dir / "model-cellot"
    model_cfg = _cellot_model_config()
    model_cfg["training"]["n_iters"] = int(n_iters)
    _write_yaml(task_path, _task_config(h5ad_path, condition, batch_size, random_state))
    _write_yaml(model_path, model_cfg)
    model_pt = outdir / "cache" / "model.pt"
    if model_pt.exists() and not force:
        return outdir
    train_py = REPO_ROOT / "external" / "cellot" / "scripts" / "train.py"
    bootstrap = (
        "import collections, collections.abc, runpy, sys; "
        "import pandas as pd; "
        "[(not hasattr(collections, n) and setattr(collections, n, getattr(collections.abc, n))) "
        "for n in ('Iterable','Mapping','MutableMapping','Sequence','MutableSequence')]; "
        "pd.DataFrame.to_hdf = lambda self, *args, **kwargs: None; "
        "import cellot.train.summary as cellot_summary; "
        "cellot_summary.Logger.flush = lambda self: None; "
        "sys.argv = sys.argv[1:]; "
        "runpy.run_path(sys.argv[0], run_name='__main__')"
    )
    cmd = [
        sys.executable,
        "-c",
        bootstrap,
        str(train_py),
        "--outdir",
        str(outdir),
        "--config",
        str(task_path),
        "--config",
        str(model_path),
    ]
    if force:
        cmd.append("--restart")
    env = os.environ.copy()
    cellot_root = str(REPO_ROOT / "external" / "cellot")
    env["PYTHONPATH"] = cellot_root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)
    return outdir


def _predict_with_cellot(outdir: Path, source: np.ndarray, batch_size: int) -> np.ndarray:
    _patch_cellot_py310_compat()
    import torch
    from cellot.utils import load_config
    from cellot.utils.loaders import load

    config = load_config(outdir / "config.yaml")
    (f_net, g_net), _, _ = load(config, restore=outdir / "cache" / "model.pt")
    f_net.eval()
    g_net.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, source.shape[0], max(1, int(batch_size))):
            chunk = torch.tensor(source[start : start + int(batch_size)], dtype=torch.float32)
            preds.append(g_net.transport(chunk).detach().cpu().numpy().astype(np.float32))
    return np.vstack(preds) if preds else np.zeros_like(source, dtype=np.float32)


def _stable_seed(base_seed: int, split_id: int, condition: str) -> int:
    raw = f"{int(base_seed)}::{int(split_id)}::{condition}"
    return int(np.frombuffer(raw.encode("utf-8"), dtype=np.uint8).sum()) % 1000003


def _metrics_payload_for_condition(
    *,
    data,
    split_dict: dict,
    split_id: int,
    base_seed: int,
    condition: str,
    nearest_condition: str,
    pred_expr: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from trishift._external_metrics import (
        average_of_perturbation_centroids,
        compute_distributional_systema_metrics_from_arrays,
        compute_mean_effect_metrics,
        compute_scpram_metrics_bundle_from_arrays,
        pearson_delta_reference_metrics,
    )
    label_key = data.label_key
    ctrl_label = data.ctrl_label
    test = split_dict["test"]
    train_val = __import__("anndata").concat([split_dict["train"], split_dict["val"]], join="outer", merge="same")
    test_cond = test.obs[label_key].astype(str)
    test_ctrl = test[test_cond.eq(ctrl_label).values].copy()
    test_target = test[test_cond.eq(str(condition)).values].copy()
    true_expr = _dense(test_target)
    ctrl_expr = _dense(test_ctrl)
    reference = average_of_perturbation_centroids(
        X=_dense(train_val),
        conditions=train_val.obs[label_key].astype(str).values,
        ctrl_label=ctrl_label,
    )
    degs = np.asarray(data.adata_all.uns.get("top20_degs_final", {}).get(str(condition), []), dtype=int)
    mean_metrics = compute_mean_effect_metrics(X_true=true_expr, X_pred=pred_expr, X_ctrl=ctrl_expr, deg_idx=degs)
    systema = pearson_delta_reference_metrics(
        X_true=true_expr.mean(axis=0),
        X_pred=pred_expr.mean(axis=0),
        reference=reference,
        top20_de_idxs=degs,
    )
    scpram = compute_scpram_metrics_bundle_from_arrays(
        X_true=true_expr,
        X_pred=pred_expr,
        deg_idx=degs,
        n_degs=100,
        sample_ratio=0.8,
        times=100,
        seed=_stable_seed(base_seed, split_id, condition),
    )
    dist = compute_distributional_systema_metrics_from_arrays(
        X_true=true_expr,
        X_pred=pred_expr,
        reference=reference,
        deg_idx=degs,
        sample_ratio=0.8,
        times=100,
        seed=_stable_seed(base_seed, split_id, condition),
    )
    metrics = {
        "condition": str(condition),
        **mean_metrics,
        "systema_corr_20de_allpert": float(systema["corr_20de_allpert"]),
        "systema_corr_deg_r2": float(systema["corr_deg_r2"]),
        **scpram["metrics"],
        **dist["metrics"],
        "split_id": int(split_id),
        "n_ensemble": int(pred_expr.shape[0]),
        "n_eval_ctrl": int(ctrl_expr.shape[0]),
        "eval_ctrl_source": "target_domain_test_ctrl",
        "prediction_ctrl_source": "cellot_nearest_train_condition_map",
        "nearest_train_condition": str(nearest_condition),
    }
    genes = _gene_names(data)
    payload = {
        "Pred": pred_expr[:, degs],
        "Ctrl": ctrl_expr[:, degs],
        "Truth": true_expr[:, degs],
        "Pred_full": pred_expr,
        "Ctrl_full": ctrl_expr,
        "Truth_full": true_expr,
        "DE_idx": degs,
        "DE_name": genes[degs] if degs.size > 0 else np.array([], dtype=genes.dtype),
        "gene_name_full": genes,
        "export_metadata": {
            "model": "cellot",
            "split_id": int(split_id),
            "split_policy": str(split_dict.get("split_policy", "unseen_ctrl_unseen_perturbation")),
            "eval_ctrl_source": "target_domain_test_ctrl",
            "prediction_ctrl_source": "cellot_nearest_train_condition_map",
            "nearest_train_condition": str(nearest_condition),
            "degs_source": "top20_degs_final",
        },
        "scpram_repeats": scpram["repeats"],
        "scpram_wasserstein_degs_by_gene": scpram["wasserstein_degs_by_gene"],
        "scpram_degs_used": scpram["degs_used"],
        "systema_distributional_repeats": dist["repeats"],
        "systema_distributional_degs_used": dist["degs_used"],
    }
    return metrics, payload


def run_aligned_cellot(
    *,
    dataset: str,
    config_path: Path,
    split_ids: list[int],
    out_root: Path,
    work_root: Path,
    batch_size: int,
    n_iters: int,
    max_train_conditions: int | None,
    force: bool,
    train: bool,
    evaluate: bool,
) -> dict[str, Any]:
    dataset_key, defaults, paths, _ = _load_profile_config(config_path)
    if dataset_key != dataset:
        raise ValueError(f"dataset mismatch: --dataset={dataset}, config={dataset_key}")
    data = _load_data_with_degs(dataset, defaults, paths)
    base_seed = int(defaults.get("seed", 24))
    out_dir = out_root / dataset
    work_dir = work_root / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    all_metrics: list[dict[str, Any]] = []
    train_rows: list[dict[str, Any]] = []
    split_policies: dict[int, str] = {}
    for split_id in split_ids:
        split_dict = _aligned_split(data, dataset=dataset, split_id=int(split_id), defaults=defaults)
        split_policies[int(split_id)] = str(split_dict.get("split_policy", "unknown"))
        train_conds = [str(c) for c in split_dict.get("train_conds", [])]
        test_conds = [str(c) for c in split_dict.get("test_conds", [])]
        if max_train_conditions is not None:
            train_conds = train_conds[: int(max_train_conditions)]
        train_map = _cond_embeddings(data, split_dict["train"], train_conds)
        test_map = _cond_embeddings(data, split_dict["test"], test_conds)
        split_work = work_dir / f"split{int(split_id)}"
        map_outdirs: dict[str, Path] = {}
        for train_cond in train_conds:
            safe_cond = str(train_cond).replace("/", "_")
            cond_dir = split_work / "train_maps" / safe_cond
            h5ad_path = cond_dir / "cellot_train_input.h5ad"
            counts = _materialize_train_condition(
                data=data,
                split_dict=split_dict,
                train_condition=train_cond,
                h5ad_path=h5ad_path,
            )
            status = "ready" if counts.get("train_source_n", 0) > 0 and counts.get("train_target_n", 0) > 0 else "empty_train_distribution"
            outdir = cond_dir / "model-cellot"
            if status == "ready" and train:
                outdir = _train_one_map(
                    cond_dir=cond_dir,
                    condition=train_cond,
                    h5ad_path=h5ad_path,
                    batch_size=batch_size,
                    n_iters=n_iters,
                    random_state=int(split_id),
                    force=force,
                )
            map_outdirs[train_cond] = outdir
            train_rows.append(
                {
                    "dataset": dataset,
                    "split_id": int(split_id),
                    "train_condition": train_cond,
                    "status": status,
                    "cellot_outdir": str(outdir),
                    **counts,
                }
            )
        split_payload: dict[str, Any] = {}
        if evaluate:
            obs_cond = split_dict["test"].obs[data.label_key].astype(str)
            test_ctrl = split_dict["test"][obs_cond.eq(data.ctrl_label).values].copy()
            ctrl_expr = _dense(test_ctrl)
            for test_cond in test_conds:
                nearest = _nearest_train_condition(test_cond, train_map, test_map)
                if nearest is None or nearest not in map_outdirs:
                    all_metrics.append(
                        {
                            "dataset": dataset,
                            "split_id": int(split_id),
                            "condition": test_cond,
                            "model": "CellOT",
                            "status": "missing_nearest_train_map",
                            "pearson": np.nan,
                            "nmse": np.nan,
                        }
                    )
                    continue
                model_pt = map_outdirs[nearest] / "cache" / "model.pt"
                if not model_pt.exists():
                    all_metrics.append(
                        {
                            "dataset": dataset,
                            "split_id": int(split_id),
                            "condition": test_cond,
                            "model": "CellOT",
                            "status": "missing_trained_model",
                            "nearest_train_condition": nearest,
                            "pearson": np.nan,
                            "nmse": np.nan,
                        }
                    )
                    continue
                pred_expr = _predict_with_cellot(map_outdirs[nearest], ctrl_expr, batch_size=batch_size)
                row, payload = _metrics_payload_for_condition(
                    data=data,
                    split_dict=split_dict,
                    split_id=int(split_id),
                    base_seed=base_seed,
                    condition=test_cond,
                    nearest_condition=nearest,
                    pred_expr=pred_expr,
                )
                row.update({"dataset": dataset, "model": "CellOT", "status": "ok"})
                all_metrics.append(row)
                split_payload[str(test_cond)] = payload
        if split_payload:
            with (out_dir / f"cellot_{dataset}_{int(split_id)}_unseen_ctrl.pkl").open("wb") as f:
                pickle.dump(split_payload, f)
    metrics_df = pd.DataFrame(all_metrics)
    metrics_path = out_dir / "metrics_unseen_ctrl.csv"
    metrics_df.to_csv(metrics_path, index=False)
    if not metrics_df.empty:
        numeric = metrics_df.mean(numeric_only=True)
        with (out_dir / "mean_pearson_unseen_ctrl.txt").open("w", encoding="utf-8") as f:
            f.write(f"{float(numeric.get('pearson', np.nan))}\n")
    pd.DataFrame(train_rows).to_csv(out_dir / "cellot_train_maps.csv", index=False)
    provenance = {
        "dataset": dataset,
        "split_ids": [int(x) for x in split_ids],
        "split_policy_by_split": split_policies,
        "prediction_strategy": "train CellOT maps only on training perturbation conditions; apply nearest training-condition map to unseen test perturbations",
        "deg_source": "TriShift top20_degs_final / shared DEG cache",
        "metrics_path": str(metrics_path),
        "work_root": str(work_dir),
        "batch_size": int(batch_size),
        "n_iters": int(n_iters),
    }
    (out_dir / "provenance_unseen_ctrl.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return provenance


def _parse_split_ids(value: str, defaults: dict, dataset: str, fast: bool) -> list[int]:
    raw = str(value or "").strip()
    if raw:
        return [int(x.strip()) for x in raw.split(",") if x.strip()]
    return _split_ids(defaults, dataset, fast=fast)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run CellOT under TriShift held-out-control/unseen-perturbation splits and metrics.")
    ap.add_argument("--dataset", choices=sorted(DATASET_CONFIG_PATHS), required=True)
    ap.add_argument("--config", default="")
    ap.add_argument("--split-ids", default="")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--out-root", default=str(REPO_ROOT / "artifacts" / "results" / "cellot"))
    ap.add_argument("--work-root", default=str(REPO_ROOT / "artifacts" / "results" / "cellot" / "trishift_aligned"))
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--n-iters", type=int, default=5000)
    ap.add_argument("--max-train-conditions", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-train", action="store_true")
    ap.add_argument("--no-evaluate", action="store_true")
    args = ap.parse_args(argv)
    config_path = Path(args.config).resolve() if str(args.config).strip() else DATASET_CONFIG_PATHS[str(args.dataset)]
    dataset_key, defaults, _, _ = _load_profile_config(config_path)
    split_ids = _parse_split_ids(str(args.split_ids), defaults, dataset_key, fast=bool(args.fast))
    prov = run_aligned_cellot(
        dataset=dataset_key,
        config_path=config_path,
        split_ids=split_ids,
        out_root=Path(args.out_root).resolve(),
        work_root=Path(args.work_root).resolve(),
        batch_size=int(args.batch_size),
        n_iters=int(args.n_iters),
        max_train_conditions=(int(args.max_train_conditions) if int(args.max_train_conditions) > 0 else None),
        force=bool(args.force),
        train=not bool(args.no_train),
        evaluate=not bool(args.no_evaluate),
    )
    print(json.dumps(prov, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
