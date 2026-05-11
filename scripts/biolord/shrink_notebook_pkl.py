from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np


def _stable_seed(file_path: Path, condition: str) -> int:
    return int(sum(ord(ch) for ch in f"{file_path.name}::{condition}") % (2**32 - 1))


def _shrink_condition_payload(payload: dict, sample_size: int, seed: int) -> tuple[dict, bool]:
    pred_full = payload.get("Pred_full")
    ctrl_full = payload.get("Ctrl_full")
    truth_full = payload.get("Truth_full")
    if pred_full is None or ctrl_full is None or truth_full is None:
        return payload, False

    pred_full = np.asarray(pred_full)
    ctrl_full = np.asarray(ctrl_full)
    truth_full = np.asarray(truth_full)
    if pred_full.ndim != 2 or ctrl_full.ndim != 2 or truth_full.ndim != 2:
        return payload, False

    pred_rows = int(pred_full.shape[0])
    ctrl_rows = int(ctrl_full.shape[0])
    truth_rows = int(truth_full.shape[0])
    if min(pred_rows, ctrl_rows, truth_rows) <= 0:
        return payload, False
    needs_pred = pred_rows > int(sample_size)
    needs_ctrl = ctrl_rows > int(sample_size)
    needs_truth = truth_rows > int(sample_size)
    if not needs_pred and not needs_ctrl and not needs_truth:
        return payload, False

    def pick_rows(n_rows: int, salt: int) -> np.ndarray:
        if n_rows > int(sample_size):
            local_rng = np.random.default_rng(int(seed) + int(salt))
            return np.sort(local_rng.choice(n_rows, size=int(sample_size), replace=False))
        return np.arange(n_rows, dtype=int)

    if pred_rows == ctrl_rows:
        pred_idx = pick_rows(pred_rows, 11)
        ctrl_idx = pred_idx
    else:
        pred_idx = pick_rows(pred_rows, 11)
        ctrl_idx = pick_rows(ctrl_rows, 17)
    truth_idx = pick_rows(truth_rows, 23)

    pred_small = np.asarray(pred_full[pred_idx], dtype=np.float32)
    ctrl_small = np.asarray(ctrl_full[ctrl_idx], dtype=np.float32)
    truth_small = np.asarray(truth_full[truth_idx], dtype=np.float32)
    deg_idx = np.asarray(payload.get("DE_idx", []), dtype=int).reshape(-1)

    updated = dict(payload)
    updated["Pred_full"] = pred_small
    updated["Ctrl_full"] = ctrl_small
    updated["Truth_full"] = truth_small
    updated["Pred"] = pred_small[:, deg_idx] if deg_idx.size > 0 else pred_small[:, :0]
    updated["Ctrl"] = ctrl_small[:, deg_idx] if deg_idx.size > 0 else ctrl_small[:, :0]
    updated["Truth"] = truth_small[:, deg_idx] if deg_idx.size > 0 else truth_small[:, :0]
    meta = dict(updated.get("export_metadata", {}) or {})
    meta.update(
        {
            "export_is_subset": True,
            "export_sample_size": int(sample_size),
            "pred_sample_idx": np.asarray(pred_idx, dtype=int),
            "ctrl_sample_idx": np.asarray(ctrl_idx, dtype=int),
            "truth_sample_idx": np.asarray(truth_idx, dtype=int),
        }
    )
    updated["export_metadata"] = meta
    return updated, True


def _rewrite_file(pkl_path: Path, sample_size: int) -> tuple[bool, str]:
    with pkl_path.open("rb") as handle:
        obj = pickle.load(handle)
    if not isinstance(obj, dict):
        return False, "payload_not_dict"

    changed = False
    rewritten = {}
    for condition, payload in obj.items():
        if not isinstance(payload, dict):
            rewritten[condition] = payload
            continue
        new_payload, did_change = _shrink_condition_payload(
            payload=payload,
            sample_size=int(sample_size),
            seed=_stable_seed(pkl_path, str(condition)),
        )
        rewritten[condition] = new_payload
        changed = changed or did_change

    if not changed:
        return False, "already_small"

    pkl_path.unlink()
    with pkl_path.open("wb") as handle:
        pickle.dump(rewritten, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return True, "rewritten"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Shrink existing biolord notebook pkl payloads to a fixed ctrl/pred sample size.")
    ap.add_argument("--root", default="artifacts/results/biolord", help="directory to scan for biolord_*.pkl")
    ap.add_argument("--sample_size", type=int, default=300)
    args = ap.parse_args(argv)

    root = Path(args.root).resolve()
    files = sorted(root.rglob("biolord_*.pkl"))
    print(f"[shrink_biolord_pkl] root={root} files={len(files)} sample_size={int(args.sample_size)}")
    for pkl_path in files:
        before_mb = round(pkl_path.stat().st_size / 1024**2, 1)
        changed, status = _rewrite_file(pkl_path, sample_size=int(args.sample_size))
        after_mb = round(pkl_path.stat().st_size / 1024**2, 1)
        print(f"[shrink_biolord_pkl] {status}: {pkl_path} {before_mb}MB -> {after_mb}MB")


if __name__ == "__main__":
    main()
