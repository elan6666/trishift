from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_TRISHIFT_DIR = Path("artifacts/results/norman")
DEFAULT_SCOUTER_DIR = Path("artifacts/results/scouter/norman")


@dataclass
class CondCompare:
    split_id: int
    condition: str
    tri_len: int
    sco_len: int
    exact_equal: bool
    set_equal: bool
    order_equal: bool
    de_name_equal: bool
    tri_only_count: int
    sco_only_count: int
    first_tri_only: str
    first_sco_only: str


def _load_pkl(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict):
        raise TypeError(f"{path} is not a dict pkl")
    return obj


def _to_int_list(v: Any) -> list[int]:
    if v is None:
        return []
    arr = np.asarray(v).reshape(-1)
    return [int(x) for x in arr.tolist()]


def _to_str_list(v: Any) -> list[str]:
    if v is None:
        return []
    arr = np.asarray(v).reshape(-1)
    return [str(x) for x in arr.tolist()]


def _detect_splits(tri_dir: Path, sco_dir: Path) -> list[int]:
    tri = {int(p.stem.rsplit("_", 1)[1]) for p in tri_dir.glob("trishift_norman_*.pkl")}
    sco = {int(p.stem.rsplit("_", 1)[1]) for p in sco_dir.glob("scouter_norman_*.pkl")}
    splits = sorted(tri & sco)
    if not splits:
        raise FileNotFoundError(
            f"No overlapping splits found between {tri_dir} (trishift_norman_*.pkl) "
            f"and {sco_dir} (scouter_norman_*.pkl)"
        )
    return splits


def _compare_split(split_id: int, tri_pkl: Path, sco_pkl: Path) -> tuple[list[CondCompare], dict[str, int]]:
    tri = _load_pkl(tri_pkl)
    sco = _load_pkl(sco_pkl)

    tri_conds = set(tri.keys())
    sco_conds = set(sco.keys())
    common = sorted(tri_conds & sco_conds)
    tri_only_conds = sorted(tri_conds - sco_conds)
    sco_only_conds = sorted(sco_conds - tri_conds)

    rows: list[CondCompare] = []
    for cond in common:
        tri_item = tri[cond]
        sco_item = sco[cond]
        tri_idx = _to_int_list(tri_item.get("DE_idx"))
        sco_idx = _to_int_list(sco_item.get("DE_idx"))
        tri_names = _to_str_list(tri_item.get("DE_name"))
        sco_names = _to_str_list(sco_item.get("DE_name"))

        tri_set = set(tri_idx)
        sco_set = set(sco_idx)
        exact_equal = tri_idx == sco_idx
        set_equal = tri_set == sco_set
        order_equal = exact_equal
        de_name_equal = tri_names == sco_names
        tri_only = sorted(tri_set - sco_set)
        sco_only = sorted(sco_set - tri_set)
        rows.append(
            CondCompare(
                split_id=split_id,
                condition=cond,
                tri_len=len(tri_idx),
                sco_len=len(sco_idx),
                exact_equal=exact_equal,
                set_equal=set_equal,
                order_equal=order_equal,
                de_name_equal=de_name_equal,
                tri_only_count=len(tri_only),
                sco_only_count=len(sco_only),
                first_tri_only=str(tri_only[0]) if tri_only else "",
                first_sco_only=str(sco_only[0]) if sco_only else "",
            )
        )

    summary = {
        "split_id": split_id,
        "n_tri_conditions": len(tri_conds),
        "n_sco_conditions": len(sco_conds),
        "n_common_conditions": len(common),
        "n_tri_only_conditions": len(tri_only_conds),
        "n_sco_only_conditions": len(sco_only_conds),
        "n_exact_equal": sum(r.exact_equal for r in rows),
        "n_set_equal": sum(r.set_equal for r in rows),
        "n_de_name_equal": sum(r.de_name_equal for r in rows),
    }
    return rows, summary


def _print_summary(summary_df: pd.DataFrame) -> None:
    print("\n=== Split Summary ===")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(summary_df.to_string(index=False))

    total_common = int(summary_df["n_common_conditions"].sum())
    total_exact = int(summary_df["n_exact_equal"].sum())
    total_set = int(summary_df["n_set_equal"].sum())
    print("\n=== Overall ===")
    print(f"total_common_conditions={total_common}")
    print(f"total_exact_equal={total_exact}")
    print(f"total_set_equal={total_set}")
    print(f"exact_match_rate={total_exact / total_common:.4f}" if total_common else "exact_match_rate=nan")
    print(f"set_match_rate={total_set / total_common:.4f}" if total_common else "set_match_rate=nan")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compare TriShift vs Scouter Norman pkl DE_idx/DE_name consistency by split"
    )
    ap.add_argument("--trishift-dir", type=Path, default=DEFAULT_TRISHIFT_DIR)
    ap.add_argument("--scouter-dir", type=Path, default=DEFAULT_SCOUTER_DIR)
    ap.add_argument("--splits", type=str, default="", help="Comma-separated split ids. Default: auto-detect overlap.")
    ap.add_argument("--out-csv", type=Path, default=None, help="Optional detailed mismatch/full comparison CSV")
    ap.add_argument("--only-mismatch", action="store_true", help="If set, save/print only mismatch rows")
    args = ap.parse_args()

    tri_dir = args.trishift_dir
    sco_dir = args.scouter_dir
    if not tri_dir.exists():
        raise FileNotFoundError(tri_dir)
    if not sco_dir.exists():
        raise FileNotFoundError(sco_dir)

    if args.splits.strip():
        splits = [int(x.strip()) for x in args.splits.split(",") if x.strip()]
    else:
        splits = _detect_splits(tri_dir, sco_dir)

    print(f"trishift_dir={tri_dir}")
    print(f"scouter_dir={sco_dir}")
    print(f"splits={splits}")

    all_rows: list[CondCompare] = []
    split_summaries: list[dict[str, int]] = []
    for split_id in splits:
        tri_pkl = tri_dir / f"trishift_norman_{split_id}.pkl"
        sco_pkl = sco_dir / f"scouter_norman_{split_id}.pkl"
        if not tri_pkl.exists():
            raise FileNotFoundError(tri_pkl)
        if not sco_pkl.exists():
            raise FileNotFoundError(sco_pkl)
        rows, summary = _compare_split(split_id, tri_pkl, sco_pkl)
        all_rows.extend(rows)
        split_summaries.append(summary)

    detail_df = pd.DataFrame([r.__dict__ for r in all_rows])
    summary_df = pd.DataFrame(split_summaries).sort_values("split_id").reset_index(drop=True)
    _print_summary(summary_df)

    mismatch_mask = ~(
        detail_df["exact_equal"] & detail_df["de_name_equal"] & (detail_df["tri_len"] == detail_df["sco_len"])
    )
    mismatch_df = detail_df.loc[mismatch_mask].copy().sort_values(["split_id", "condition"]).reset_index(drop=True)

    print("\n=== Mismatch Rows ===")
    print(f"n_mismatch_rows={len(mismatch_df)}")
    if len(mismatch_df) > 0:
        with pd.option_context("display.max_rows", 50, "display.max_columns", None, "display.width", 200):
            print(mismatch_df.head(20).to_string(index=False))

    if args.out_csv is not None:
        out_df = mismatch_df if args.only_mismatch else detail_df.sort_values(["split_id", "condition"]).reset_index(drop=True)
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(args.out_csv, index=False)
        print(f"\nSaved CSV: {args.out_csv}")


if __name__ == "__main__":
    main()
