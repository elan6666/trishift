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
    if pred_full is None or ctrl_full is None:
        return payload, False

    pred_full = np.asarray(pred_full)
    ctrl_full = np.asarray(ctrl_full)
    if pred_full.ndim != 2 or ctrl_full.ndim != 2:
        return payload, False

    n_rows = min(int(pred_full.shape[0]), int(ctrl_full.shape[0]))
    if n_rows <= 0 or n_rows <= int(sample_size):
        return payload, False

    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(n_rows, size=int(sample_size), replace=False))

    pred_small = np.asarray(pred_full[idx], dtype=np.float32)
    ctrl_small = np.asarray(ctrl_full[idx], dtype=np.float32)
    deg_idx = np.asarray(payload.get("DE_idx", []), dtype=int).reshape(-1)

    updated = dict(payload)
    updated["Pred_full"] = pred_small
    updated["Ctrl_full"] = ctrl_small
    updated["Pred"] = pred_small[:, deg_idx] if deg_idx.size > 0 else pred_small[:, :0]
    updated["Ctrl"] = ctrl_small[:, deg_idx] if deg_idx.size > 0 else ctrl_small[:, :0]
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
