from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from datetime import datetime
from typing import Any

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

from trishift._external_metrics import compute_scpram_metrics_from_arrays


DEFAULT_RESULT_ROOTS = {
    "trishift": REPO_ROOT / "artifacts" / "results",
    "gears": REPO_ROOT / "artifacts" / "results" / "gears",
    "genepert": REPO_ROOT / "artifacts" / "results" / "genepert",
    "scouter": REPO_ROOT / "artifacts" / "results" / "scouter",
    "scgpt": REPO_ROOT / "artifacts" / "results" / "scgpt",
}
DEFAULT_DEG_CACHE_ROOT = REPO_ROOT / "artifacts" / "cache" / "degs"


@dataclass
class DEG20ExperimentResult:
    out_dir: Path
    per_condition_df: pd.DataFrame
    split_summary_df: pd.DataFrame
    dataset_summary_df: pd.DataFrame
    gene_lists_df: pd.DataFrame
    enrichment_df: pd.DataFrame
    representative_df: pd.DataFrame
    figure_paths: dict[str, str]


def _ts_local() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _normalize_model_name(model_name: str) -> str:
    key = str(model_name).strip().lower()
    if key not in DEFAULT_RESULT_ROOTS:
        raise ValueError(f"Unsupported model_name={model_name}")
    return key


def _split_ids_from_value(split_ids: int | str | list[int] | tuple[int, ...]) -> list[int]:
    if isinstance(split_ids, int):
        return [int(split_ids)]
    if isinstance(split_ids, (list, tuple)):
        out = [int(x) for x in split_ids]
        if not out:
            raise ValueError("split_ids cannot be empty")
        return out
    text = str(split_ids).strip()
    if not text:
        raise ValueError("split_ids cannot be empty")
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _result_root(model_name: str, dataset: str, result_dir: str | Path | None) -> Path:
    if result_dir is not None and str(result_dir).strip():
        return Path(result_dir).resolve()
    return (DEFAULT_RESULT_ROOTS[_normalize_model_name(model_name)] / dataset).resolve()


def _pkl_path(
    *,
    model_name: str,
    dataset: str,
    split_id: int,
    result_root: Path,
    variant_tag: str | None = None,
) -> Path:
    model_key = _normalize_model_name(model_name)
    if model_key == "trishift":
        suffix = f"_{variant_tag}" if variant_tag else ""
        return result_root / f"trishift_{dataset}_{split_id}{suffix}.pkl"
    return result_root / f"{model_key}_{dataset}_{split_id}.pkl"


def _parse_condition_tokens(condition: str) -> list[str]:
    tokens = []
    for token in str(condition).split("+"):
        tok = str(token).strip()
        if not tok or tok.lower() == "ctrl":
            continue
        tokens.append(tok)
    return tokens


def _remove_perturbed_genes(genes: list[str], condition: str) -> list[str]:
    perturbed = set(_parse_condition_tokens(condition))
    return [g for g in genes if str(g) not in perturbed]


def _safe_token(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(text))


def _normalize_space(space: str | None) -> str:
    key = str(space or "auto").strip().lower()
    if key not in {"auto", "full_gene", "deg"}:
        raise ValueError("space must be one of: auto, full_gene, deg")
    return key


def _payload_item_arrays(
    obj: dict[str, Any],
    *,
    space: str = "auto",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    space_key = _normalize_space(space)
    has_full = all(k in obj for k in ["Pred_full", "Ctrl_full", "Truth_full", "gene_name_full"])
    use_full = has_full and space_key in {"auto", "full_gene"}
    if use_full:
        pred = np.asarray(obj["Pred_full"], dtype=np.float32)
        ctrl = np.asarray(obj["Ctrl_full"], dtype=np.float32)
        truth = np.asarray(obj["Truth_full"], dtype=np.float32)
        gene_names = np.asarray(obj["gene_name_full"]).astype(str)
        deg_idx = np.asarray(obj["DE_idx"], dtype=int).reshape(-1)
        deg_name = np.asarray(obj["DE_name"]).astype(str)
    else:
        if space_key == "full_gene" and not has_full:
            raise ValueError("full_gene space requested but payload does not contain full-gene arrays")
        pred = np.asarray(obj["Pred"], dtype=np.float32)
        ctrl = np.asarray(obj["Ctrl"], dtype=np.float32)
        truth = np.asarray(obj["Truth"], dtype=np.float32)
        deg_name = np.asarray(obj["DE_name"]).astype(str)
        if deg_name.size == 0:
            gene_names = np.asarray([f"gene_{i}" for i in range(pred.shape[1])], dtype=str)
        else:
            gene_names = deg_name.astype(str)
        # Legacy payloads only contain the DEG subspace, so local gene indices are 0..n_deg-1.
        deg_idx = np.arange(gene_names.shape[0], dtype=int)
    if pred.ndim != 2 or ctrl.ndim != 2 or truth.ndim != 2:
        raise ValueError("Prediction/control/truth arrays must be 2D")
    if pred.shape[1] != ctrl.shape[1] or pred.shape[1] != truth.shape[1]:
        raise ValueError("Prediction/control/truth arrays must share the same gene dimension")
    if gene_names.shape[0] != pred.shape[1]:
        raise ValueError("gene name array length must equal the gene dimension")
    return pred, ctrl, truth, gene_names, deg_idx, deg_name


def _truth_deg20(
    *,
    condition: str,
    truth: np.ndarray,
    ctrl: np.ndarray,
    gene_names: np.ndarray,
    deg_idx: np.ndarray,
    deg_name: np.ndarray,
    truth_deg_mode: str,
    remove_perturbed_genes: bool,
    n_degs: int = 20,
) -> tuple[list[str], np.ndarray]:
    mode_key = str(truth_deg_mode).strip().lower()
    if mode_key not in {"payload", "scanpy", "effect_size", "effect_size_non_dropout", "ttest_non_dropout"}:
        raise ValueError(
            "truth_deg_mode must be one of: payload, scanpy, effect_size, effect_size_non_dropout, ttest_non_dropout"
        )
    if mode_key == "scanpy":
        names = _pred_deg20_scanpy_rank(
            pred=truth,
            ctrl=ctrl,
            gene_names=gene_names,
            condition=condition,
            remove_perturbed_genes=remove_perturbed_genes,
            n_degs=n_degs,
        )
        name_to_idx = {str(g): i for i, g in enumerate(gene_names.astype(str).tolist())}
        idx = [int(name_to_idx[g]) for g in names if g in name_to_idx]
        return names[:n_degs], np.asarray(idx[:n_degs], dtype=int)
    if mode_key == "effect_size":
        names = _pred_deg20_effect_size(
            pred=truth,
            ctrl=ctrl,
            gene_names=gene_names,
            condition=condition,
            remove_perturbed_genes=remove_perturbed_genes,
            n_degs=n_degs,
        )
        name_to_idx = {str(g): i for i, g in enumerate(gene_names.astype(str).tolist())}
        idx = [int(name_to_idx[g]) for g in names if g in name_to_idx]
        return names[:n_degs], np.asarray(idx[:n_degs], dtype=int)
    if mode_key == "effect_size_non_dropout":
        names = _pred_deg20_effect_size_non_dropout(
            pred=truth,
            ctrl=ctrl,
            gene_names=gene_names,
            condition=condition,
            remove_perturbed_genes=remove_perturbed_genes,
            n_degs=n_degs,
        )
        name_to_idx = {str(g): i for i, g in enumerate(gene_names.astype(str).tolist())}
        idx = [int(name_to_idx[g]) for g in names if g in name_to_idx]
        return names[:n_degs], np.asarray(idx[:n_degs], dtype=int)
    if mode_key == "ttest_non_dropout":
        names = _pred_deg20_ttest_non_dropout(
            pred=truth,
            ctrl=ctrl,
            gene_names=gene_names,
            condition=condition,
            remove_perturbed_genes=remove_perturbed_genes,
            n_degs=n_degs,
        )
        name_to_idx = {str(g): i for i, g in enumerate(gene_names.astype(str).tolist())}
        idx = [int(name_to_idx[g]) for g in names if g in name_to_idx]
        return names[:n_degs], np.asarray(idx[:n_degs], dtype=int)

    names = [str(g) for g in deg_name.tolist()] if deg_name.size > 0 else [str(gene_names[i]) for i in deg_idx.tolist()]
    idx = [int(i) for i in deg_idx.tolist() if 0 <= int(i) < gene_names.shape[0]]
    if remove_perturbed_genes:
        filtered_names = _remove_perturbed_genes(names, condition)
        keep = set(filtered_names)
        idx = [i for i in idx if str(gene_names[i]) in keep]
        names = filtered_names
    return names[:n_degs], np.asarray(idx[:n_degs], dtype=int)


def _truth_deg_cache_path(
    *,
    cache_root: Path,
    pkl_path: Path,
    truth_deg_mode: str,
    remove_perturbed_genes: bool,
    n_degs: int,
    space: str = "auto",
) -> Path:
    mode_key = _safe_token(str(truth_deg_mode).strip().lower())
    keep_key = "drop_pert" if remove_perturbed_genes else "keep_pert"
    space_key = _safe_token(_normalize_space(space))
    return cache_root / f"{pkl_path.stem}__truth_{mode_key}__top{int(n_degs)}__{keep_key}__space_{space_key}.pkl"


def _build_truth_deg_cache_for_payload(
    *,
    payload: dict[str, Any],
    truth_deg_mode: str,
    remove_perturbed_genes: bool,
    n_degs: int,
    space: str = "auto",
) -> dict[str, Any]:
    conditions: dict[str, dict[str, Any]] = {}
    for condition, obj in payload.items():
        pred, ctrl, truth, gene_names, deg_idx, deg_name = _payload_item_arrays(obj, space=space)
        truth_genes, truth_idx = _truth_deg20(
            condition=str(condition),
            truth=truth,
            ctrl=ctrl,
            gene_names=gene_names,
            deg_idx=deg_idx,
            deg_name=deg_name,
            truth_deg_mode=truth_deg_mode,
            remove_perturbed_genes=remove_perturbed_genes,
            n_degs=n_degs,
        )
        conditions[str(condition)] = {
            "truth_deg20": list(truth_genes),
            "truth_deg_idx": np.asarray(truth_idx, dtype=int),
        }
    return {
        "truth_deg_mode": str(truth_deg_mode).strip().lower(),
        "remove_perturbed_genes": bool(remove_perturbed_genes),
        "n_degs": int(n_degs),
        "space": _normalize_space(space),
        "conditions": conditions,
    }


def _load_or_build_truth_deg_cache(
    *,
    payload: dict[str, Any],
    pkl_path: Path,
    truth_deg_mode: str,
    remove_perturbed_genes: bool,
    cache_root: Path,
    n_degs: int,
    space: str = "auto",
) -> dict[str, Any]:
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_path = _truth_deg_cache_path(
        cache_root=cache_root,
        pkl_path=pkl_path,
        truth_deg_mode=truth_deg_mode,
        remove_perturbed_genes=remove_perturbed_genes,
        n_degs=n_degs,
        space=space,
    )
    if cache_path.exists():
        with cache_path.open("rb") as fh:
            cached = pickle.load(fh)
        if (
            isinstance(cached, dict)
            and str(cached.get("truth_deg_mode", "")).strip().lower() == str(truth_deg_mode).strip().lower()
            and bool(cached.get("remove_perturbed_genes", True)) == bool(remove_perturbed_genes)
            and int(cached.get("n_degs", n_degs)) == int(n_degs)
            and str(cached.get("space", "auto")).strip().lower() == _normalize_space(space)
            and isinstance(cached.get("conditions"), dict)
        ):
            return cached
    cached = _build_truth_deg_cache_for_payload(
        payload=payload,
        truth_deg_mode=truth_deg_mode,
        remove_perturbed_genes=remove_perturbed_genes,
        n_degs=n_degs,
        space=space,
    )
    with cache_path.open("wb") as fh:
        pickle.dump(cached, fh)
    return cached


def _pred_deg20_effect_size(
    *,
    pred: np.ndarray,
    ctrl: np.ndarray,
    gene_names: np.ndarray,
    condition: str,
    remove_perturbed_genes: bool,
    n_degs: int,
) -> list[str]:
    scores = np.abs(pred.mean(axis=0) - ctrl.mean(axis=0))
    order = np.argsort(-scores, kind="stable")
    ranked = [str(gene_names[i]) for i in order.tolist()]
    if remove_perturbed_genes:
        ranked = _remove_perturbed_genes(ranked, condition)
    return ranked[:n_degs]


def _compute_nonzero_non_dropout(
    pert_mean: np.ndarray,
    ctrl_mean: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    non_zero = np.where(pert_mean != 0)[0]
    zero = np.where(pert_mean == 0)[0]
    true_zeros = np.intersect1d(zero, np.where(ctrl_mean == 0)[0])
    non_dropouts = np.concatenate((non_zero, true_zeros))
    return non_zero, non_dropouts


def _pred_deg20_effect_size_non_dropout(
    *,
    pred: np.ndarray,
    ctrl: np.ndarray,
    gene_names: np.ndarray,
    condition: str,
    remove_perturbed_genes: bool,
    n_degs: int,
) -> list[str]:
    pred_mean = np.asarray(pred.mean(axis=0)).ravel()
    ctrl_mean = np.asarray(ctrl.mean(axis=0)).ravel()
    scores = np.abs(pred_mean - ctrl_mean)
    _, non_dropouts = _compute_nonzero_non_dropout(pred_mean, ctrl_mean)
    keep = set(gene_names[np.asarray(non_dropouts, dtype=int)].astype(str).tolist())
    order = np.argsort(-scores, kind="stable")
    ranked = [str(gene_names[i]) for i in order.tolist() if str(gene_names[i]) in keep]
    if remove_perturbed_genes:
        ranked = _remove_perturbed_genes(ranked, condition)
    return ranked[:n_degs]


def _prepare_scanpy_rank_matrix(
    *,
    pred: np.ndarray,
    ctrl: np.ndarray,
) -> np.ndarray:
    x = np.vstack([ctrl, pred]).astype(np.float32, copy=False)
    if not np.isfinite(x).all():
        raise ValueError("pred/ctrl matrix contains non-finite values")
    min_val = float(np.min(x))
    if min_val < 0.0:
        x = x - min_val
    return np.log1p(x).astype(np.float32, copy=False)


def _pred_deg20_ttest_non_dropout(
    *,
    pred: np.ndarray,
    ctrl: np.ndarray,
    gene_names: np.ndarray,
    condition: str,
    remove_perturbed_genes: bool,
    n_degs: int,
) -> list[str]:
    if pred.ndim != 2 or ctrl.ndim != 2:
        raise ValueError("pred/ctrl must be 2D for t-test ranking")
    if pred.shape[1] != ctrl.shape[1]:
        raise ValueError("pred/ctrl must share the same gene dimension")
    if pred.shape[0] < 2 or ctrl.shape[0] < 2:
        return _pred_deg20_effect_size(
            pred=pred,
            ctrl=ctrl,
            gene_names=gene_names,
            condition=condition,
            remove_perturbed_genes=remove_perturbed_genes,
            n_degs=n_degs,
        )
    x = np.vstack([ctrl, pred]).astype(np.float32, copy=False)
    if not np.isfinite(x).all():
        return _pred_deg20_effect_size(
            pred=pred,
            ctrl=ctrl,
            gene_names=gene_names,
            condition=condition,
            remove_perturbed_genes=remove_perturbed_genes,
            n_degs=n_degs,
        )
    obs = pd.DataFrame(
        {
            "group": ["control"] * int(ctrl.shape[0]) + ["predicted"] * int(pred.shape[0])
        }
    )
    var = pd.DataFrame(index=pd.Index(gene_names.astype(str), name="gene_name"))
    adata = ad.AnnData(X=x, obs=obs, var=var)
    sc.tl.rank_genes_groups(
        adata,
        groupby="group",
        reference="control",
        rankby_abs=True,
        n_genes=adata.n_vars,
        method="t-test",
    )
    names = adata.uns["rank_genes_groups"]["names"]["predicted"]
    ranked = [str(g) for g in list(names)]
    pred_mean = np.asarray(pred.mean(axis=0)).ravel()
    ctrl_mean = np.asarray(ctrl.mean(axis=0)).ravel()
    _, non_dropouts = _compute_nonzero_non_dropout(pred_mean, ctrl_mean)
    keep = set(gene_names[np.asarray(non_dropouts, dtype=int)].astype(str).tolist())
    ranked = [g for g in ranked if g in keep]
    if remove_perturbed_genes:
        ranked = _remove_perturbed_genes(ranked, condition)
    return ranked[:n_degs]


def _pred_deg20_scanpy_rank(
    *,
    pred: np.ndarray,
    ctrl: np.ndarray,
    gene_names: np.ndarray,
    condition: str,
    remove_perturbed_genes: bool,
    n_degs: int,
) -> list[str]:
    x = _prepare_scanpy_rank_matrix(pred=pred, ctrl=ctrl)
    obs = pd.DataFrame(
        {
            "group": ["control"] * int(ctrl.shape[0]) + ["predicted"] * int(pred.shape[0])
        }
    )
    var = pd.DataFrame(index=pd.Index(gene_names.astype(str), name="gene_name"))
    adata = ad.AnnData(X=x, obs=obs, var=var)
    sc.tl.rank_genes_groups(
        adata,
        groupby="group",
        reference="control",
        rankby_abs=True,
        n_genes=adata.n_vars,
        method="wilcoxon",
    )
    names = adata.uns["rank_genes_groups"]["names"]["predicted"]
    ranked = [str(g) for g in list(names)]
    if remove_perturbed_genes:
        ranked = _remove_perturbed_genes(ranked, condition)
    return ranked[:n_degs]


def _scanpy_rank_is_safe(
    *,
    pred: np.ndarray,
    ctrl: np.ndarray,
) -> bool:
    # `rank_genes_groups` is unreliable on our reconstructed matrices once predictions
    # contain substantial negative values; in that case adaptive mode should fall back
    # to effect-size ranking instead of emitting deg lists with zero overlap.
    if pred.ndim != 2 or ctrl.ndim != 2:
        return False
    if pred.shape[0] < 2:
        return False
    if not np.isfinite(pred).all() or not np.isfinite(ctrl).all():
        return False
    return True


def _pred_deg20(
    *,
    pred: np.ndarray,
    ctrl: np.ndarray,
    gene_names: np.ndarray,
    condition: str,
    pred_deg_mode: str,
    remove_perturbed_genes: bool,
    n_degs: int = 20,
) -> tuple[list[str], str]:
    mode_key = str(pred_deg_mode).strip().lower()
    if mode_key not in {
        "adaptive",
        "scanpy",
        "effect_size",
        "effect_size_non_dropout",
        "ttest_non_dropout",
    }:
        raise ValueError(
            "pred_deg_mode must be one of: adaptive, scanpy, effect_size, effect_size_non_dropout, ttest_non_dropout"
        )
    if mode_key == "effect_size":
        return _pred_deg20_effect_size(
            pred=pred,
            ctrl=ctrl,
            gene_names=gene_names,
            condition=condition,
            remove_perturbed_genes=remove_perturbed_genes,
            n_degs=n_degs,
        ), "effect_size_fallback"
    if mode_key == "effect_size_non_dropout":
        return _pred_deg20_effect_size_non_dropout(
            pred=pred,
            ctrl=ctrl,
            gene_names=gene_names,
            condition=condition,
            remove_perturbed_genes=remove_perturbed_genes,
            n_degs=n_degs,
        ), "effect_size_non_dropout"
    if mode_key == "ttest_non_dropout":
        return _pred_deg20_ttest_non_dropout(
            pred=pred,
            ctrl=ctrl,
            gene_names=gene_names,
            condition=condition,
            remove_perturbed_genes=remove_perturbed_genes,
            n_degs=n_degs,
        ), "ttest_non_dropout"
    if mode_key == "scanpy":
        return _pred_deg20_scanpy_rank(
            pred=pred,
            ctrl=ctrl,
            gene_names=gene_names,
            condition=condition,
            remove_perturbed_genes=remove_perturbed_genes,
            n_degs=n_degs,
        ), "scanpy_rank"
    if _scanpy_rank_is_safe(pred=pred, ctrl=ctrl):
        return _pred_deg20_scanpy_rank(
            pred=pred,
            ctrl=ctrl,
            gene_names=gene_names,
            condition=condition,
            remove_perturbed_genes=remove_perturbed_genes,
            n_degs=n_degs,
        ), "scanpy_rank"
    return _pred_deg20_effect_size(
        pred=pred,
        ctrl=ctrl,
        gene_names=gene_names,
        condition=condition,
        remove_perturbed_genes=remove_perturbed_genes,
        n_degs=n_degs,
    ), "effect_size_fallback"


def _overlap_metrics(truth_genes: list[str], pred_genes: list[str]) -> dict[str, float | int]:
    truth = list(dict.fromkeys([str(g) for g in truth_genes]))
    pred = list(dict.fromkeys([str(g) for g in pred_genes]))
    truth_set = set(truth)
    pred_set = set(pred)
    common = truth_set & pred_set
    union = truth_set | pred_set
    return {
        "common_degs_at_20": int(len(common)),
        "jaccard_at_20": float(len(common) / len(union)) if union else float("nan"),
        "precision_at_20": float(len(common) / len(pred_set)) if pred_set else float("nan"),
        "recall_at_20": float(len(common) / len(truth_set)) if truth_set else float("nan"),
        "truth_deg_count": int(len(truth)),
        "pred_deg_count": int(len(pred)),
    }


def _condition_rows_from_payload(
    *,
    model_name: str,
    dataset: str,
    split_id: int,
    pkl_path: Path,
    payload: dict[str, Any],
    truth_deg_mode: str,
    pred_deg_mode: str,
    remove_perturbed_genes: bool,
    truth_deg_cache: dict[str, Any] | None = None,
    n_degs: int = 20,
    space: str = "auto",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    gene_rows: list[dict[str, Any]] = []
    for condition, obj in payload.items():
        pred, ctrl, truth, gene_names, deg_idx, deg_name = _payload_item_arrays(obj, space=space)
        cached_cond = None
        if truth_deg_cache is not None:
            cached_cond = (truth_deg_cache.get("conditions", {}) or {}).get(str(condition))
        if cached_cond is not None:
            truth_deg20 = [str(g) for g in cached_cond.get("truth_deg20", [])]
            truth_deg_idx = np.asarray(cached_cond.get("truth_deg_idx", []), dtype=int)
        else:
            truth_deg20, truth_deg_idx = _truth_deg20(
                condition=str(condition),
                truth=truth,
                ctrl=ctrl,
                gene_names=gene_names,
                deg_idx=deg_idx,
                deg_name=deg_name,
                truth_deg_mode=truth_deg_mode,
                remove_perturbed_genes=remove_perturbed_genes,
                n_degs=n_degs,
            )
        pred_deg20, pred_mode_used = _pred_deg20(
            pred=pred,
            ctrl=ctrl,
            gene_names=gene_names,
            condition=str(condition),
            pred_deg_mode=pred_deg_mode,
            remove_perturbed_genes=remove_perturbed_genes,
            n_degs=n_degs,
        )
        overlap = _overlap_metrics(truth_deg20, pred_deg20)
        common_deg20 = [g for g in pred_deg20 if g in set(truth_deg20)]
        scpram = compute_scpram_metrics_from_arrays(
            X_true=truth,
            X_pred=pred,
            deg_idx=truth_deg_idx,
            n_degs=int(n_degs),
            sample_ratio=0.8,
            times=100,
        )
        row = {
            "model_name": str(model_name),
            "dataset": str(dataset),
            "split_id": int(split_id),
            "condition": str(condition),
            "focus_key": f"{int(split_id)}:{str(condition)}",
            "pkl_path": str(pkl_path),
            "deg_top_k": int(n_degs),
            "space_used": _normalize_space(space) if _normalize_space(space) != "auto" else ("full_gene" if all(k in obj for k in ["Pred_full", "Ctrl_full", "Truth_full", "gene_name_full"]) else "deg"),
            "truth_deg_mode_used": str(truth_deg_mode).strip().lower(),
            "pred_deg_mode_used": str(pred_mode_used),
            **overlap,
            "scpram_r2_degs_mean_mean": float(scpram["scpram_r2_degs_mean_mean"]),
            "scpram_r2_degs_var_mean": float(scpram["scpram_r2_degs_var_mean"]),
            "scpram_wasserstein_degs_sum": float(scpram["scpram_wasserstein_degs_sum"]),
        }
        rows.append(row)
        for list_type, genes in (
            ("truth_deg20", truth_deg20),
            ("pred_deg20", pred_deg20),
            ("common_deg20", common_deg20),
        ):
            for rank, gene in enumerate(genes, start=1):
                gene_rows.append(
                    {
                        "model_name": str(model_name),
                        "dataset": str(dataset),
                        "split_id": int(split_id),
                        "condition": str(condition),
                        "focus_key": f"{int(split_id)}:{str(condition)}",
                        "list_type": str(list_type),
                        "rank": int(rank),
                        "gene": str(gene),
                        "deg_top_k": int(n_degs),
                        "space_used": row["space_used"],
                        "truth_deg_mode_used": str(truth_deg_mode).strip().lower(),
                        "pred_deg_mode_used": str(pred_mode_used),
                    }
                )
    return rows, gene_rows


def _summarize_by_split(per_condition_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "common_degs_at_20",
        "jaccard_at_20",
        "precision_at_20",
        "recall_at_20",
        "truth_deg_count",
        "pred_deg_count",
        "scpram_r2_degs_mean_mean",
        "scpram_r2_degs_var_mean",
        "scpram_wasserstein_degs_sum",
    ]
    rows = []
    for split_id, split_df in per_condition_df.groupby("split_id", sort=True):
        row: dict[str, Any] = {
            "model_name": str(split_df["model_name"].iloc[0]),
            "dataset": str(split_df["dataset"].iloc[0]),
            "split_id": int(split_id),
            "deg_top_k": int(pd.to_numeric(split_df["deg_top_k"], errors="coerce").iloc[0]),
            "n_conditions": int(len(split_df)),
            "n_scanpy_rank": int(split_df["pred_deg_mode_used"].eq("scanpy_rank").sum()),
            "n_effect_size_fallback": int(split_df["pred_deg_mode_used"].eq("effect_size_fallback").sum()),
        }
        for col in metric_cols:
            vals = pd.to_numeric(split_df[col], errors="coerce")
            row[f"{col}_mean"] = float(vals.mean())
            row[f"{col}_median"] = float(vals.median())
        rows.append(row)
    return pd.DataFrame(rows)


def _summarize_dataset(split_summary_df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [c for c in split_summary_df.columns if c.endswith("_mean") or c.endswith("_median")]
    row: dict[str, Any] = {
        "model_name": str(split_summary_df["model_name"].iloc[0]),
        "dataset": str(split_summary_df["dataset"].iloc[0]),
        "deg_top_k": int(pd.to_numeric(split_summary_df["deg_top_k"], errors="coerce").iloc[0]),
        "n_splits": int(split_summary_df["split_id"].nunique()),
    }
    for col in numeric_cols:
        vals = pd.to_numeric(split_summary_df[col], errors="coerce")
        row[col] = float(vals.mean())
    return pd.DataFrame([row])


def select_representative_conditions(
    per_condition_df: pd.DataFrame,
    *,
    focus_conditions: list[str] | None = None,
) -> pd.DataFrame:
    if focus_conditions:
        focus_set = {str(x) for x in focus_conditions}
        picked = per_condition_df[per_condition_df["condition"].astype(str).isin(focus_set)].copy()
        sort_cols = [c for c in ["split_id", "condition"] if c in picked.columns]
        if sort_cols:
            return picked.sort_values(by=sort_cols).reset_index(drop=True)
        return picked.reset_index(drop=True)
    ranked = per_condition_df.sort_values(
        by=["common_degs_at_20", "jaccard_at_20", "scpram_r2_degs_mean_mean"],
        ascending=[False, False, False],
        na_position="last",
    ).reset_index(drop=True)
    if ranked.empty:
        return ranked
    top_row = ranked.iloc[[0]]
    mid_row = ranked.iloc[[len(ranked) // 2]]
    worst_row = ranked.iloc[[-1]]
    return pd.concat([top_row, mid_row, worst_row], ignore_index=True).drop_duplicates(
        subset=["focus_key"]
    )


def _try_run_enrichment(
    *,
    gene_lists_df: pd.DataFrame,
    representative_df: pd.DataFrame,
    enrichment_mode: str,
    enrichment_library: str,
) -> pd.DataFrame:
    mode_key = str(enrichment_mode).strip().lower()
    if mode_key in {"disabled", "off", "none", "export_only"}:
        return pd.DataFrame()
    try:
        import gseapy as gp  # type: ignore
    except Exception:
        return pd.DataFrame()

    rep_keys = set(representative_df["focus_key"].astype(str).tolist())
    work_df = gene_lists_df[
        gene_lists_df["focus_key"].astype(str).isin(rep_keys)
        & gene_lists_df["list_type"].isin(["pred_deg20", "common_deg20"])
    ]
    rows: list[dict[str, Any]] = []
    for (focus_key, list_type), sub_df in work_df.groupby(["focus_key", "list_type"], sort=False):
        gene_list = sub_df.sort_values("rank")["gene"].astype(str).tolist()
        if not gene_list:
            continue
        try:
            enr = gp.enrichr(
                gene_list=gene_list,
                gene_sets=[str(enrichment_library)],
                organism="Human",
                outdir=None,
                no_plot=True,
            )
        except Exception:
            continue
        res2d = getattr(enr, "results", None)
        if res2d is None or len(res2d) == 0:
            continue
        res2d = pd.DataFrame(res2d).head(20)
        cond_info = sub_df.iloc[0]
        for _, rr in res2d.iterrows():
            rows.append(
                {
                    "model_name": str(cond_info["model_name"]),
                    "dataset": str(cond_info["dataset"]),
                    "split_id": int(cond_info["split_id"]),
                    "condition": str(cond_info["condition"]),
                    "focus_key": str(focus_key),
                    "list_type": str(list_type),
                    "library": str(enrichment_library),
                    "term": str(rr.get("Term", "")),
                    "adjusted_p_value": float(rr.get("Adjusted P-value", np.nan)),
                    "combined_score": float(rr.get("Combined Score", np.nan)),
                    "overlap": str(rr.get("Overlap", "")),
                    "genes": str(rr.get("Genes", "")),
                }
            )
    return pd.DataFrame(rows)


def load_condition_payload(
    *,
    model_name: str,
    dataset: str,
    split_id: int,
    condition: str,
    result_dir: str | Path | None = None,
    variant_tag: str | None = None,
) -> dict[str, Any]:
    result_root = _result_root(model_name, dataset, result_dir)
    pkl_path = _pkl_path(
        model_name=model_name,
        dataset=dataset,
        split_id=int(split_id),
        result_root=result_root,
        variant_tag=variant_tag,
    )
    with pkl_path.open("rb") as f:
        payload = pickle.load(f)
    if str(condition) not in payload:
        raise KeyError(f"Condition not found in payload: {condition}")
    return payload[str(condition)]


def summarize_condition_payload(
    *,
    payload_item: dict[str, Any],
    condition: str,
    truth_deg_mode: str = "effect_size_non_dropout",
    pred_deg_mode: str = "effect_size_non_dropout",
    n_degs: int = 100,
    remove_perturbed_genes: bool = True,
    space: str = "auto",
) -> dict[str, Any]:
    pred, ctrl, truth, gene_names, deg_idx, deg_name = _payload_item_arrays(payload_item, space=space)
    truth_deg20, truth_deg_idx = _truth_deg20(
        condition=str(condition),
        truth=truth,
        ctrl=ctrl,
        gene_names=gene_names,
        deg_idx=deg_idx,
        deg_name=deg_name,
        truth_deg_mode=truth_deg_mode,
        remove_perturbed_genes=remove_perturbed_genes,
        n_degs=n_degs,
    )
    pred_deg20, pred_mode_used = _pred_deg20(
        pred=pred,
        ctrl=ctrl,
        gene_names=gene_names,
        condition=str(condition),
        pred_deg_mode=pred_deg_mode,
        remove_perturbed_genes=remove_perturbed_genes,
        n_degs=n_degs,
    )
    common_deg20 = [g for g in pred_deg20 if g in set(truth_deg20)]
    return {
        "condition": str(condition),
        "pred": pred,
        "ctrl": ctrl,
        "truth": truth,
        "gene_name_full": gene_names,
        "truth_deg20": truth_deg20,
        "truth_deg_idx": truth_deg_idx,
        "deg_top_k": int(n_degs),
        "space_used": _normalize_space(space) if _normalize_space(space) != "auto" else ("full_gene" if all(k in payload_item for k in ["Pred_full", "Ctrl_full", "Truth_full", "gene_name_full"]) else "deg"),
        "truth_deg_mode_used": str(truth_deg_mode).strip().lower(),
        "pred_deg20": pred_deg20,
        "common_deg20": common_deg20,
        "pred_deg_mode_used": str(pred_mode_used),
    }


def build_mean_var_scatter(
    *,
    payload_item: dict[str, Any],
    truth_deg_idx: np.ndarray,
    title_prefix: str,
    n_degs: int = 20,
    space: str = "auto",
) -> tuple[plt.Figure, plt.Figure]:
    pred, _, truth, gene_names, _, _ = _payload_item_arrays(payload_item, space=space)
    degs = np.asarray(truth_deg_idx, dtype=int).reshape(-1)
    degs = degs[(degs >= 0) & (degs < pred.shape[1])]
    pred_mean = pred.mean(axis=0)[degs]
    truth_mean = truth.mean(axis=0)[degs]
    pred_var = pred.var(axis=0)[degs]
    truth_var = truth.var(axis=0)[degs]
    labels = gene_names[degs] if degs.size > 0 else np.array([], dtype=str)

    fig_mean, ax_mean = plt.subplots(figsize=(5, 4))
    ax_mean.scatter(truth_mean, pred_mean, s=18, alpha=0.8)
    ax_mean.set_xlabel("Truth mean")
    ax_mean.set_ylabel("Pred mean")
    ax_mean.set_title(f"{title_prefix}: DEG{int(n_degs)} mean")
    for x, y, label in zip(truth_mean[:10], pred_mean[:10], labels[:10]):
        ax_mean.text(float(x), float(y), str(label), fontsize=8)

    fig_var, ax_var = plt.subplots(figsize=(5, 4))
    ax_var.scatter(truth_var, pred_var, s=18, alpha=0.8)
    ax_var.set_xlabel("Truth var")
    ax_var.set_ylabel("Pred var")
    ax_var.set_title(f"{title_prefix}: DEG{int(n_degs)} variance")
    for x, y, label in zip(truth_var[:10], pred_var[:10], labels[:10]):
        ax_var.text(float(x), float(y), str(label), fontsize=8)
    return fig_mean, fig_var


def run_deg20_experiment(
    *,
    dataset: str,
    model_name: str,
    split_ids: int | str | list[int] | tuple[int, ...],
    result_dir: str | Path | None = None,
    out_root: str | Path | None = None,
    variant_tag: str | None = None,
    focus_conditions: list[str] | None = None,
    truth_deg_mode: str = "effect_size_non_dropout",
    pred_deg_mode: str = "effect_size_non_dropout",
    n_degs: int = 100,
    enrichment_mode: str = "export_only",
    enrichment_library: str = "Reactome_2022",
    remove_perturbed_genes: bool = True,
    truth_deg_cache_root: str | Path | None = None,
    space: str = "auto",
) -> DEG20ExperimentResult:
    dataset_key = str(dataset).strip()
    model_key = _normalize_model_name(model_name)
    split_list = _split_ids_from_value(split_ids)
    result_root = _result_root(model_key, dataset_key, result_dir)
    out_base = Path(out_root).resolve() if out_root else result_root / f"deg20_downstream_{_ts_local()}"
    out_base.mkdir(parents=True, exist_ok=True)
    truth_cache_root_path = (
        Path(truth_deg_cache_root).resolve()
        if truth_deg_cache_root
        else DEFAULT_DEG_CACHE_ROOT.resolve()
    )

    rows: list[dict[str, Any]] = []
    gene_rows: list[dict[str, Any]] = []
    for split_id in split_list:
        pkl_path = _pkl_path(
            model_name=model_key,
            dataset=dataset_key,
            split_id=int(split_id),
            result_root=result_root,
            variant_tag=variant_tag,
        )
        if not pkl_path.exists():
            raise FileNotFoundError(f"Missing pkl: {pkl_path}")
        with pkl_path.open("rb") as f:
            payload = pickle.load(f)
        if not isinstance(payload, dict):
            raise TypeError(f"Unexpected payload type at {pkl_path}: {type(payload)}")
        truth_deg_cache = None
        if str(truth_deg_mode).strip().lower() != "payload":
            truth_deg_cache = _load_or_build_truth_deg_cache(
                payload=payload,
                pkl_path=pkl_path,
                truth_deg_mode=truth_deg_mode,
                remove_perturbed_genes=remove_perturbed_genes,
                cache_root=truth_cache_root_path,
                n_degs=n_degs,
                space=space,
            )
        split_rows, split_gene_rows = _condition_rows_from_payload(
            model_name=model_key,
            dataset=dataset_key,
            split_id=int(split_id),
            pkl_path=pkl_path,
            payload=payload,
            truth_deg_mode=truth_deg_mode,
            pred_deg_mode=pred_deg_mode,
            remove_perturbed_genes=remove_perturbed_genes,
            truth_deg_cache=truth_deg_cache,
            n_degs=n_degs,
            space=space,
        )
        rows.extend(split_rows)
        gene_rows.extend(split_gene_rows)

    per_condition_df = pd.DataFrame(rows).sort_values(by=["split_id", "condition"]).reset_index(drop=True)
    gene_lists_df = pd.DataFrame(gene_rows).sort_values(by=["split_id", "condition", "list_type", "rank"]).reset_index(drop=True)
    split_summary_df = _summarize_by_split(per_condition_df)
    dataset_summary_df = _summarize_dataset(split_summary_df)
    representative_df = select_representative_conditions(per_condition_df, focus_conditions=focus_conditions)
    enrichment_df = _try_run_enrichment(
        gene_lists_df=gene_lists_df,
        representative_df=representative_df,
        enrichment_mode=enrichment_mode,
        enrichment_library=enrichment_library,
    )

    per_condition_path = out_base / "per_condition_metrics.csv"
    split_summary_path = out_base / "split_summary.csv"
    dataset_summary_path = out_base / "dataset_summary.csv"
    gene_lists_path = out_base / "deg_gene_lists_long.csv"
    representative_path = out_base / "representative_conditions.csv"
    enrichment_path = out_base / "enrichment_results.csv"
    meta_path = out_base / "run_meta.json"

    per_condition_df.to_csv(per_condition_path, index=False)
    split_summary_df.to_csv(split_summary_path, index=False)
    dataset_summary_df.to_csv(dataset_summary_path, index=False)
    gene_lists_df.to_csv(gene_lists_path, index=False)
    representative_df.to_csv(representative_path, index=False)
    enrichment_df.to_csv(enrichment_path, index=False)
    meta = {
        "dataset": dataset_key,
        "model_name": model_key,
        "split_ids": split_list,
        "result_root": str(result_root),
        "variant_tag": str(variant_tag or ""),
        "n_degs": int(n_degs),
        "truth_deg_mode": str(truth_deg_mode),
        "pred_deg_mode": str(pred_deg_mode),
        "enrichment_mode": str(enrichment_mode),
        "enrichment_library": str(enrichment_library),
        "remove_perturbed_genes": bool(remove_perturbed_genes),
        "truth_deg_cache_root": str(truth_cache_root_path),
        "space": _normalize_space(space),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    figure_paths: dict[str, str] = {}
    return DEG20ExperimentResult(
        out_dir=out_base,
        per_condition_df=per_condition_df,
        split_summary_df=split_summary_df,
        dataset_summary_df=dataset_summary_df,
        gene_lists_df=gene_lists_df,
        enrichment_df=enrichment_df,
        representative_df=representative_df,
        figure_paths=figure_paths,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Run multi-condition DEG20 downstream experiment from exported pkl files.")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model_name", required=True, choices=sorted(DEFAULT_RESULT_ROOTS.keys()))
    ap.add_argument("--split_ids", default="1")
    ap.add_argument("--result_dir", default="")
    ap.add_argument("--out_root", default="")
    ap.add_argument("--variant_tag", default="")
    ap.add_argument("--focus_conditions", default="")
    ap.add_argument("--n_degs", type=int, default=100)
    ap.add_argument(
        "--truth_deg_mode",
        default="effect_size_non_dropout",
        choices=[
            "payload",
            "scanpy",
            "effect_size",
            "effect_size_non_dropout",
            "ttest_non_dropout",
        ],
    )
    ap.add_argument(
        "--pred_deg_mode",
        default="effect_size_non_dropout",
        choices=[
            "adaptive",
            "scanpy",
            "effect_size",
            "effect_size_non_dropout",
            "ttest_non_dropout",
        ],
    )
    ap.add_argument("--enrichment_mode", default="export_only", choices=["export_only", "run_if_available", "disabled"])
    ap.add_argument("--enrichment_library", default="Reactome_2022")
    ap.add_argument("--keep_perturbed_genes", action="store_true")
    ap.add_argument("--space", default="auto", choices=["auto", "full_gene", "deg"])
    args = ap.parse_args()

    focus_conditions = [x.strip() for x in str(args.focus_conditions).split(",") if x.strip()]
    result = run_deg20_experiment(
        dataset=str(args.dataset).strip(),
        model_name=str(args.model_name).strip(),
        split_ids=str(args.split_ids).strip(),
        result_dir=str(args.result_dir).strip() or None,
        out_root=str(args.out_root).strip() or None,
        variant_tag=str(args.variant_tag).strip() or None,
        focus_conditions=focus_conditions or None,
        n_degs=int(args.n_degs),
        truth_deg_mode=str(args.truth_deg_mode).strip(),
        pred_deg_mode=str(args.pred_deg_mode).strip(),
        enrichment_mode=str(args.enrichment_mode).strip(),
        enrichment_library=str(args.enrichment_library).strip(),
        remove_perturbed_genes=not bool(args.keep_perturbed_genes),
        space=str(args.space).strip(),
    )
    print(f"[deg20] out_dir={result.out_dir}")
    print(result.dataset_summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
