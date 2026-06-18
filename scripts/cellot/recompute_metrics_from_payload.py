from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "external" / "cellot"))

from scripts.cellot.run_trishift_aligned import (  # noqa: E402
    _aligned_split,
    _load_data_with_degs,
    _load_profile_config,
    _metrics_payload_for_condition,
    _write_mean_metrics_txt,
)


def _parse_split_ids(raw: str) -> list[int]:
    return [int(x.strip()) for x in str(raw).split(",") if x.strip()]


def recompute_from_payloads(
    *,
    dataset: str,
    config_path: Path,
    split_ids: list[int],
    result_dir: Path,
    metric_repeats: int,
) -> dict[str, object]:
    dataset_key, defaults, paths, _ = _load_profile_config(config_path)
    if dataset_key != dataset:
        raise ValueError(f"dataset mismatch: requested {dataset!r}, config has {dataset_key!r}")
    data = _load_data_with_degs(dataset_key, defaults, paths)
    base_seed = int(defaults.get("seed", 24))
    rows: list[dict[str, object]] = []
    missing: list[dict[str, object]] = []
    for split_id in split_ids:
        payload_path = result_dir / f"cellot_{dataset_key}_{int(split_id)}_unseen_ctrl.pkl"
        if not payload_path.exists():
            missing.append({"split_id": int(split_id), "path": str(payload_path), "reason": "missing_payload"})
            continue
        with payload_path.open("rb") as f:
            payload = pickle.load(f)
        if not isinstance(payload, dict):
            missing.append({"split_id": int(split_id), "path": str(payload_path), "reason": "unexpected_payload_type"})
            continue
        split_dict = _aligned_split(data, dataset=dataset_key, split_id=int(split_id), defaults=defaults)
        for condition, obj in payload.items():
            if not isinstance(obj, dict):
                continue
            pred = obj.get("Pred_full", obj.get("Pred"))
            pred_expr = np.asarray(pred, dtype=np.float32)
            if pred_expr.ndim != 2 or pred_expr.shape[0] == 0:
                missing.append(
                    {
                        "split_id": int(split_id),
                        "condition": str(condition),
                        "path": str(payload_path),
                        "reason": "missing_prediction_array",
                    }
                )
                continue
            meta = obj.get("export_metadata", {}) or {}
            nearest = str(meta.get("nearest_train_condition", condition))
            metric_row, _ = _metrics_payload_for_condition(
                data=data,
                split_dict=split_dict,
                split_id=int(split_id),
                base_seed=base_seed,
                condition=str(condition),
                nearest_condition=nearest,
                pred_expr=pred_expr,
                metric_repeats=max(1, int(metric_repeats)),
            )
            metric_row.update(
                {
                    "dataset": dataset_key,
                    "model": "CellOT",
                    "status": "ok",
                    "payload_path": str(payload_path),
                }
            )
            for key in (
                "split_policy",
                "eval_ctrl_source",
                "prediction_ctrl_source",
                "cellot_eval_setting",
                "include_test_ctrl_in_train",
                "pbmc_protocol",
                "degs_source",
            ):
                if key in meta:
                    metric_row[key] = meta[key]
            rows.append(metric_row)
    metrics_df = pd.DataFrame(rows)
    result_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = result_dir / "metrics_unseen_ctrl.csv"
    mean_path = result_dir / "mean_pearson_unseen_ctrl.txt"
    metrics_df.to_csv(metrics_path, index=False)
    _write_mean_metrics_txt(mean_path, metrics_df)
    provenance = {
        "dataset": dataset_key,
        "split_ids": [int(x) for x in split_ids],
        "source": "existing CellOT prediction payloads",
        "metrics_path": str(metrics_path),
        "mean_path": str(mean_path),
        "metric_repeats": int(metric_repeats),
        "missing": missing,
        "required_pbmc_protocol": "true_unseen_target_domain_ctrl",
    }
    (result_dir / "payload_metric_recompute_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return provenance


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Recompute CellOT metrics from existing TriShift-style prediction payloads.")
    ap.add_argument("--dataset", default="scgen_pbmc_celltype")
    ap.add_argument("--config", default="scripts/cellot/scgen_pbmc_celltype/config.yaml")
    ap.add_argument("--split-ids", default="1,2,3,4,5")
    ap.add_argument("--result-dir", default="artifacts/results/cellot/scgen_pbmc_celltype")
    ap.add_argument("--metric-repeats", type=int, default=30)
    args = ap.parse_args(argv)
    prov = recompute_from_payloads(
        dataset=str(args.dataset),
        config_path=(REPO_ROOT / str(args.config)).resolve(),
        split_ids=_parse_split_ids(str(args.split_ids)),
        result_dir=(REPO_ROOT / str(args.result_dir)).resolve(),
        metric_repeats=int(args.metric_repeats),
    )
    print(json.dumps(prov, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
