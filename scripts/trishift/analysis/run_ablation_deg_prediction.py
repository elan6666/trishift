from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from scripts.trishift.analysis.deg_prediction import run_deg_prediction


DEFAULT_DATASET_SPLITS = {
    "adamson": [1, 2, 3, 4, 5],
    "dixit": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "norman": [1, 2, 3, 4, 5],
    "scgen_pbmc_celltype": [1, 2, 3, 4, 5],
}

DEFAULT_PRESETS = [
    "ref_random",
    "ref_knn",
    "ref_ot",
    "cond_no_reference",
    "cond_no_prior",
    "cond_full",
]


def _parse_csv(value: str) -> list[str]:
    return [x.strip() for x in str(value).split(",") if x.strip()]


def _parse_split_map(value: str) -> dict[str, list[int]]:
    if not value.strip():
        return DEFAULT_DATASET_SPLITS
    out: dict[str, list[int]] = {}
    for chunk in value.split(";"):
        if not chunk.strip():
            continue
        dataset, splits = chunk.split(":", 1)
        out[dataset.strip()] = [int(x.strip()) for x in splits.split(",") if x.strip()]
    return out


def run_ablation_deg_prediction(
    *,
    datasets: list[str],
    presets: list[str],
    split_map: dict[str, list[int]],
    result_root: Path,
    out_root: Path,
    label_top_k: int,
    overlap_ks: list[int],
) -> Path:
    frames: list[pd.DataFrame] = []
    summary_frames: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, object]] = []
    out_root.mkdir(parents=True, exist_ok=True)
    for dataset in datasets:
        split_ids = split_map.get(dataset, DEFAULT_DATASET_SPLITS.get(dataset, [1]))
        result_mode = "default" if dataset == "scgen_pbmc_celltype" else "unseen_ctrl"
        for preset in presets:
            result_dir = result_root / dataset / preset
            if not result_dir.exists():
                manifest_rows.append(
                    {
                        "dataset": dataset,
                        "preset": preset,
                        "result_dir": str(result_dir),
                        "status": "missing_result_dir",
                        "n_long_rows": 0,
                        "n_summary_rows": 0,
                        "n_missing_payloads": len(split_ids),
                    }
                )
                continue
            result = run_deg_prediction(
                dataset=dataset,
                model_name="trishift",
                split_ids=split_ids,
                result_dir=result_dir,
                out_root=out_root / dataset / preset,
                result_mode=result_mode,
                variant_tag=None,
                payload_path=None,
                label_top_k=label_top_k,
                overlap_ks=overlap_ks,
                remove_perturbed_genes=True,
            )
            long_df = result["long"].copy()
            summary_df = result["summary"].copy()
            missing_payloads = result.get("missing_payloads", [])
            if not summary_df.empty:
                summary_df["preset"] = preset
                summary_df["model"] = preset
                summary_frames.append(summary_df)
            manifest_rows.append(
                {
                    "dataset": dataset,
                    "preset": preset,
                    "result_dir": str(result_dir),
                    "out_dir": str(result["out_dir"]),
                    "status": "ok" if not long_df.empty else "no_metric_rows",
                    "n_long_rows": int(len(long_df)),
                    "n_summary_rows": int(len(summary_df)),
                    "n_missing_payloads": int(len(missing_payloads)),
                }
            )
            if long_df.empty:
                continue
            long_df["preset"] = preset
            long_df["model"] = preset
            frames.append(long_df)
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["dataset", "setting", "split", "condition", "model", "metric", "value", "preset"]
    )
    summary = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame(
        columns=["dataset", "model_name", "metric", "value", "n_conditions", "preset", "model"]
    )
    manifest = pd.DataFrame(manifest_rows)
    out_path = out_root / "ablation_deg_prediction_all_long.csv"
    combined.to_csv(out_path, index=False)
    summary.to_csv(out_root / "ablation_deg_prediction_all_summary.csv", index=False)
    manifest.to_csv(out_root / "ablation_deg_prediction_manifest.csv", index=False)
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser(description="Compute DEG overlap metrics for completed TriShift ablation payloads.")
    ap.add_argument("--datasets", default="adamson,dixit,norman,scgen_pbmc_celltype")
    ap.add_argument("--presets", default=",".join(DEFAULT_PRESETS))
    ap.add_argument("--split-map", default="", help="Optional dataset:s1,s2;dataset:s1 format")
    ap.add_argument("--result-root", default=str(REPO_ROOT / "artifacts" / "results" / "ablation"))
    ap.add_argument("--out-root", default=str(REPO_ROOT / "artifacts" / "analysis" / "deg_prediction" / "ablation"))
    ap.add_argument("--label-top-k", type=int, default=100)
    ap.add_argument("--overlap-ks", default="20,50")
    args = ap.parse_args()
    out = run_ablation_deg_prediction(
        datasets=_parse_csv(args.datasets),
        presets=_parse_csv(args.presets),
        split_map=_parse_split_map(args.split_map),
        result_root=Path(args.result_root),
        out_root=Path(args.out_root),
        label_top_k=int(args.label_top_k),
        overlap_ks=[int(x) for x in _parse_csv(args.overlap_ks)],
    )
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
