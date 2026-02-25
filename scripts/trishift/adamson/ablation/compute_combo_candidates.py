from __future__ import annotations

import argparse
import csv
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


def _as_code(x: str) -> str:
    s = str(x).strip()
    if len(s) == 2 and s.isdigit():
        return s
    if s.isdigit():
        return s.zfill(2)
    # allow "03_A2_xxx" style
    if len(s) >= 2 and s[0:2].isdigit():
        return s[0:2]
    raise ValueError(f"Cannot parse module code from '{x}'")


def _discover_mean_metrics(source_root: Path) -> pd.DataFrame:
    mean_path = source_root / "all_runs_mean_metrics.csv"
    if mean_path.exists():
        return pd.read_csv(mean_path)

    # Fallback: aggregate from per-run metrics.csv
    run_dirs = sorted([p for p in source_root.iterdir() if p.is_dir() and p.name[:2].isdigit()])
    if not run_dirs:
        raise FileNotFoundError(f"No run dirs found under: {source_root}")

    rows: list[dict] = []
    for rd in run_dirs:
        m = rd / "metrics.csv"
        if not m.exists():
            continue
        df = pd.read_csv(m)
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        drop_cols = {"split_id", "n_ensemble"}
        metric_cols = [c for c in num_cols if c not in drop_cols]
        mean_vals = df[metric_cols].mean(numeric_only=True)

        idx = int(rd.name.split("_", 1)[0])
        label = rd.name.split("_", 1)[1] if "_" in rd.name else rd.name
        row = {"idx": idx, "run": rd.name, "label": label}
        row.update({k: float(v) for k, v in mean_vals.to_dict().items()})
        rows.append(row)

    out = pd.DataFrame(rows).sort_values("idx").reset_index(drop=True)
    out.to_csv(mean_path, index=False, encoding="utf-8-sig")
    return out


def _lower_better(metric: str) -> bool:
    m = metric.lower()
    return ("nmse" in m) or ("mse" in m) or ("rmse" in m) or ("wasserstein" in m)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute top-K two-way ablation combo candidates from single-factor sweep results."
        )
    )
    parser.add_argument(
        "--source_root",
        required=True,
        help="single-factor sweep root (contains all_runs_mean_metrics.csv or per-run metrics.csv dirs)",
    )
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument(
        "--exclude_codes",
        default="03",
        help="comma-separated 2-digit module codes to exclude from pairing (default: 03)",
    )
    parser.add_argument(
        "--out_csv",
        default="",
        help="output CSV path (default: <source_root>/top20_combo_candidates_excl03.csv)",
    )
    args = parser.parse_args()

    source_root = Path(args.source_root)
    if not source_root.exists():
        raise FileNotFoundError(f"source_root not found: {source_root}")
    if int(args.top_k) <= 0:
        raise ValueError("--top_k must be positive")

    exclude = [c.strip() for c in str(args.exclude_codes).split(",") if c.strip()]
    exclude_codes = {c.zfill(2) if c.isdigit() else c for c in exclude}
    # Map code -> idx (int)
    exclude_idxs = {int(c) for c in exclude_codes if c.isdigit()}

    mean_df = _discover_mean_metrics(source_root)
    need_cols = {"idx", "run"}
    if not need_cols.issubset(mean_df.columns):
        raise ValueError(f"mean metrics missing required columns: {need_cols}")

    metric_cols = [c for c in mean_df.columns if c not in ("idx", "run", "label")]
    if len(metric_cols) < 2:
        raise ValueError("Not enough metric columns to compute combos.")

    base_df = mean_df[mean_df["idx"] == 1]
    if base_df.empty:
        raise ValueError("Baseline (idx=1) not found in mean metrics.")
    base = base_df.iloc[0]

    mods = mean_df[(mean_df["idx"] != 1) & (~mean_df["idx"].isin(sorted(exclude_idxs)))].copy()
    mods = mods.sort_values("idx")
    if len(mods) < 2:
        raise ValueError("Not enough modules (after excluding) to form pairs.")

    # Δ metric (unified: positive is better)
    delta = pd.DataFrame(index=mods["idx"].astype(int), columns=metric_cols, dtype=float)
    for c in metric_cols:
        d = mods[c].astype(float).values - float(base[c])
        if _lower_better(c):
            d = -d
        delta[c] = d

    # z-score per metric dimension across modules
    z = delta.copy()
    for c in metric_cols:
        mu = float(z[c].mean())
        sd = float(z[c].std(ddof=0))
        if sd == 0.0 or np.isnan(sd):
            z[c] = 0.0
        else:
            z[c] = (z[c] - mu) / sd

    idxs = list(z.index.astype(int))
    rows: list[dict] = []
    for i, j in combinations(idxs, 2):
        vi = z.loc[i].values.astype(float)
        vj = z.loc[j].values.astype(float)

        gain_strength = float(np.linalg.norm(vi) + np.linalg.norm(vj))

        if np.std(vi) == 0.0 or np.std(vj) == 0.0:
            corr = 0.0
        else:
            corr = float(np.corrcoef(vi, vj)[0, 1])
            if np.isnan(corr):
                corr = 0.0
        overlap_risk = float(abs(corr))

        di = delta.loc[i].values.astype(float)
        dj = delta.loc[j].values.astype(float)
        coverage = float(np.mean((di > 0) | (dj > 0)))

        synergy = gain_strength * (1.0 - overlap_risk)
        score = float(synergy * coverage)

        rows.append(
            {
                "i": int(i),
                "j": int(j),
                "score": score,
                "gain_strength": float(gain_strength),
                "corr": float(corr),
                "overlap_risk": float(overlap_risk),
                "coverage": float(coverage),
            }
        )

    res = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    top = res.head(int(args.top_k))

    if str(args.out_csv).strip():
        out_csv = Path(str(args.out_csv))
    else:
        suffix = "_excl" + "_".join(sorted(exclude_codes)) if exclude_codes else ""
        out_csv = source_root / f"top{int(args.top_k)}_combo_candidates{suffix}.csv"
        # keep legacy-friendly name when excluding only 03 and top_k==20
        if exclude_codes == {"03"} and int(args.top_k) == 20:
            out_csv = source_root / "top20_combo_candidates_excl03.csv"

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    top.to_csv(out_csv, index=False, encoding="utf-8-sig")

    # Print in strict format (numbers only in reason).
    print("Top Recommended Ablation Combinations:\n")
    for rank, r in enumerate(top.itertuples(index=False), start=1):
        ci = f"{int(r.i):02d}"
        cj = f"{int(r.j):02d}"
        print(f"Rank {rank}: {ci} + {cj}")
        print(
            f"Reason: gain_strength={r.gain_strength:.3f}, coverage={r.coverage:.3f}, overlap_risk={r.overlap_risk:.3f}"
        )
        print(f"Complementarity score: {r.score:.3f}\n")

    print(str(out_csv))


if __name__ == "__main__":
    main()

