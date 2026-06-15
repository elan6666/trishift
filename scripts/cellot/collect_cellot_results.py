from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


METRIC_MAP = {
    "r2-means": "mean_scpram_r2_all_mean_mean",
    "r2-stds": "mean_scpram_r2_all_var_mean",
    "mmd": "mean_mmd",
    "l2-means": "mean_l2_means",
    "l2-stds": "mean_l2_stds",
    "enrichment-k50": "mean_enrichment_k50",
    "enrichment-k100": "mean_enrichment_k100",
}


def _read_plan(root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(root.glob("*/cellot_trishift_heldout_plan.csv")):
        df = pd.read_csv(path, keep_default_na=False)
        if "dataset" not in df.columns:
            df["dataset"] = path.parent.name
        frames.append(df)
    if not frames:
        return pd.DataFrame(
            columns=["dataset", "split_id", "condition", "status", "cellot_outdir"]
        )
    return pd.concat(frames, ignore_index=True)


def collect(root: Path, out_dir: Path) -> dict:
    plan_df = _read_plan(root)
    rows: list[dict] = []
    long_rows: list[dict] = []

    for rec in plan_df.to_dict("records"):
        dataset = str(rec.get("dataset", "")).strip()
        split_id = int(rec.get("split_id", 0) or 0)
        raw_condition = rec.get("condition", "")
        condition = "" if pd.isna(raw_condition) else str(raw_condition).strip()
        status = str(rec.get("status", "")).strip() or "unknown"
        outdir = Path(str(rec.get("cellot_outdir", "")).strip())
        eval_path = outdir / "evals_iid_data_space" / "evals.csv"
        if status != "ready":
            rows.append(
                {
                    "dataset": dataset,
                    "split_id": split_id,
                    "condition": condition,
                    "model": "CellOT",
                    "status": status,
                    "pearson": np.nan,
                    "nmse": np.nan,
                }
            )
            continue
        if not eval_path.exists():
            rows.append(
                {
                    "dataset": dataset,
                    "split_id": split_id,
                    "condition": condition,
                    "model": "CellOT",
                    "status": "missing_evals",
                    "pearson": np.nan,
                    "nmse": np.nan,
                    "evals_path": str(eval_path),
                }
            )
            continue
        eval_df = pd.read_csv(eval_path)
        metric_means = eval_df.groupby("metric", dropna=False)["value"].mean(numeric_only=True)
        row = {
            "dataset": dataset,
            "split_id": split_id,
            "condition": condition,
            "model": "CellOT",
            "status": "ok",
            "pearson": float(metric_means.get("r2-means", np.nan)),
            "nmse": np.nan,
            "evals_path": str(eval_path),
        }
        for source_metric, target_col in METRIC_MAP.items():
            row[target_col] = float(metric_means.get(source_metric, np.nan))
            long_rows.append(
                {
                    "dataset": dataset,
                    "split_id": split_id,
                    "condition": condition,
                    "model": "CellOT",
                    "source_metric": source_metric,
                    "metric": target_col,
                    "value": float(metric_means.get(source_metric, np.nan)),
                    "source_path": str(eval_path),
                }
            )
        rows.append(row)

    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_df = pd.DataFrame(rows)
    long_df = pd.DataFrame(long_rows)
    metrics_path = out_dir / "metrics_unseen_ctrl.csv"
    long_path = out_dir / "cellot_metrics_long.csv"
    summary_path = out_dir / "cellot_summary.csv"
    metrics_df.to_csv(metrics_path, index=False)
    long_df.to_csv(long_path, index=False)
    if not metrics_df.empty:
        summary_cols = [
            c for c in metrics_df.columns
            if c.startswith("mean_") or c in {"pearson", "nmse"}
        ]
        summary_df = (
            metrics_df.groupby(["dataset", "model", "status"], dropna=False)[summary_cols]
            .mean(numeric_only=True)
            .reset_index()
        )
    else:
        summary_df = pd.DataFrame()
    summary_df.to_csv(summary_path, index=False)
    provenance = {
        "input_root": str(root),
        "metrics_path": str(metrics_path),
        "long_path": str(long_path),
        "summary_path": str(summary_path),
        "n_plan_rows": int(len(plan_df)),
        "n_metric_rows": int(len(metrics_df)),
        "note": "CellOT native evals do not directly provide TriShift nMSE; prediction-level metrics require imputed h5ad post-processing.",
    }
    prov_path = out_dir / "provenance.json"
    prov_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return provenance


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Collect CellOT native evals into TriShift paper metric tables.")
    ap.add_argument("--root", default="artifacts/results/cellot/trishift_heldout")
    ap.add_argument("--out-dir", default="artifacts/results/cellot")
    args = ap.parse_args(argv)
    print(json.dumps(collect(Path(args.root), Path(args.out_dir)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
