from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import Any

import numpy as np


def stable_subset_seed(*parts: object) -> int:
    text = "::".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**32 - 1)


def _subset_rows(
    arr: Any,
    *,
    sample_size: int,
    seed: int,
) -> tuple[Any, np.ndarray | None, int]:
    sample_size = max(1, int(sample_size))
    if arr is None:
        return arr, None, 0
    x = np.asarray(arr)
    if x.ndim < 1:
        return arr, None, 0
    n_rows = int(x.shape[0])
    if n_rows <= 0:
        return x, np.asarray([], dtype=np.int64), n_rows
    if n_rows <= int(sample_size):
        idx = np.arange(n_rows, dtype=np.int64)
        return x, idx, n_rows
    rng = np.random.default_rng(int(seed))
    idx = np.sort(rng.choice(n_rows, size=int(sample_size), replace=False)).astype(np.int64)
    return x[idx], idx, n_rows


def _safe_de_subset(full: Any, deg_idx: Any, fallback: Any = None) -> Any:
    if full is None:
        return fallback
    try:
        arr = np.asarray(full)
        deg = np.asarray(deg_idx, dtype=int)
        if arr.ndim != 2 or deg.size == 0:
            return arr[:, :0] if arr.ndim == 2 else fallback
        if int(deg.max()) >= int(arr.shape[1]) or int(deg.min()) < 0:
            return fallback
        return arr[:, deg]
    except Exception:
        return fallback


def subset_payload_item(
    item: dict[str, Any],
    *,
    model_name: str,
    dataset: str,
    split_id: int | str,
    condition: str,
    sample_size: int = 300,
) -> dict[str, Any]:
    sample_size = max(1, int(sample_size))
    out = dict(item)
    pred = out.get("Pred_full")
    ctrl = out.get("Ctrl_full")
    truth = out.get("Truth_full")

    pred_arr = None if pred is None else np.asarray(pred)
    ctrl_arr = None if ctrl is None else np.asarray(ctrl)
    use_shared_pred_ctrl_idx = (
        pred_arr is not None
        and ctrl_arr is not None
        and pred_arr.ndim >= 1
        and ctrl_arr.ndim >= 1
        and int(pred_arr.shape[0]) == int(ctrl_arr.shape[0])
    )

    pred_seed = stable_subset_seed(model_name, dataset, split_id, condition, "pred_ctrl")
    truth_seed = stable_subset_seed(model_name, dataset, split_id, condition, "truth")
    if use_shared_pred_ctrl_idx:
        pred_sub, pred_idx, n_pred_full = _subset_rows(pred_arr, sample_size=sample_size, seed=pred_seed)
        if pred_idx is None:
            ctrl_sub, ctrl_idx, n_ctrl_full = _subset_rows(ctrl_arr, sample_size=sample_size, seed=pred_seed)
        else:
            n_ctrl_full = int(ctrl_arr.shape[0])
            ctrl_idx = pred_idx
            ctrl_sub = ctrl_arr[pred_idx] if n_ctrl_full > int(sample_size) else ctrl_arr
    else:
        pred_sub, pred_idx, n_pred_full = _subset_rows(pred, sample_size=sample_size, seed=pred_seed)
        ctrl_sub, ctrl_idx, n_ctrl_full = _subset_rows(
            ctrl,
            sample_size=sample_size,
            seed=stable_subset_seed(model_name, dataset, split_id, condition, "ctrl"),
        )
    truth_sub, truth_idx, n_truth_full = _subset_rows(truth, sample_size=sample_size, seed=truth_seed)

    out["Pred_full"] = np.asarray(pred_sub, dtype=np.float32) if pred_sub is not None else pred_sub
    out["Ctrl_full"] = np.asarray(ctrl_sub, dtype=np.float32) if ctrl_sub is not None else ctrl_sub
    out["Truth_full"] = np.asarray(truth_sub, dtype=np.float32) if truth_sub is not None else truth_sub

    deg_idx = out.get("DE_idx", np.asarray([], dtype=int))
    out["Pred"] = _safe_de_subset(out.get("Pred_full"), deg_idx, out.get("Pred"))
    out["Ctrl"] = _safe_de_subset(out.get("Ctrl_full"), deg_idx, out.get("Ctrl"))
    out["Truth"] = _safe_de_subset(out.get("Truth_full"), deg_idx, out.get("Truth"))

    meta = dict(out.get("export_metadata") or {})
    meta.update(
        {
            "export_is_subset": True,
            "export_sample_size": int(sample_size),
            "metrics_computed_on_full": bool(meta.get("metrics_computed_on_full", True)),
            "n_pred_full": int(n_pred_full),
            "n_ctrl_full": int(n_ctrl_full),
            "n_truth_full": int(n_truth_full),
        }
    )
    if pred_idx is not None:
        meta["pred_sample_idx"] = np.asarray(pred_idx, dtype=np.int64)
    if ctrl_idx is not None:
        meta["ctrl_sample_idx"] = np.asarray(ctrl_idx, dtype=np.int64)
    if truth_idx is not None:
        meta["truth_sample_idx"] = np.asarray(truth_idx, dtype=np.int64)
    out["export_metadata"] = meta
    return out


def subset_payload(
    payload: dict[str, Any],
    *,
    model_name: str,
    dataset: str,
    split_id: int | str,
    sample_size: int = 300,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for condition, item in payload.items():
        if isinstance(item, dict):
            out[condition] = subset_payload_item(
                item,
                model_name=model_name,
                dataset=dataset,
                split_id=split_id,
                condition=str(condition),
                sample_size=sample_size,
            )
        else:
            out[condition] = item
    return out


def shrink_payload_file(
    src: Path,
    dst: Path,
    *,
    model_name: str,
    dataset: str,
    split_id: int | str,
    sample_size: int = 300,
) -> dict[str, Any]:
    with Path(src).open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected dict payload at {src}, found {type(payload)}")
    shrunk = subset_payload(
        payload,
        model_name=model_name,
        dataset=dataset,
        split_id=split_id,
        sample_size=sample_size,
    )
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    with Path(dst).open("wb") as handle:
        pickle.dump(shrunk, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return {"n_conditions": len(shrunk)}
