from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "external" / "cellot"))

from scripts.cellot.run_trishift_aligned import (
    _load_profile_config,
    _parse_split_ids,
    run_aligned_cellot,
)
from scripts.common.yaml_utils import load_yaml_file


def _arg_or_config(value, default_value, cast):
    return cast(default_value if value is None else value)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="CellOT dataset entry: scgen_pbmc_celltype")
    ap.add_argument("--config", default="", help="CellOT PBMC config; defaults to this folder's config.yaml")
    ap.add_argument("--split-ids", default="", help="Comma-separated split ids; defaults to config/default splits")
    ap.add_argument("--fast", action="store_true", help="Run split1 with small CellOT settings")
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--work-root", default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--n-iters", type=int, default=None)
    ap.add_argument("--n-inner-iters", type=int, default=None)
    ap.add_argument("--parallel-maps", type=int, default=None)
    ap.add_argument("--map-threads", type=int, default=None)
    ap.add_argument("--metric-repeats", type=int, default=None)
    ap.add_argument("--max-eval-ctrl", type=int, default=None)
    ap.add_argument("--max-train-conditions", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-train", action="store_true")
    ap.add_argument("--no-evaluate", action="store_true")
    args = ap.parse_args(argv)

    config_path = Path(args.config).resolve() if str(args.config).strip() else Path(__file__).with_name("config.yaml")
    cfg = load_yaml_file(config_path)
    cellot_args = cfg.get("cellot_args") or {}
    dataset_key, defaults, _, _ = _load_profile_config(config_path)
    if dataset_key != "scgen_pbmc_celltype":
        raise ValueError(f"Expected scgen_pbmc_celltype config, got {dataset_key!r}")

    split_ids = _parse_split_ids(str(args.split_ids), defaults, dataset_key, fast=bool(args.fast))
    batch_size = _arg_or_config(args.batch_size, cellot_args.get("batch_size", 256), int)
    n_iters = 5 if bool(args.fast) else _arg_or_config(args.n_iters, cellot_args.get("n_iters", 100), int)
    n_inner_iters = 1 if bool(args.fast) else _arg_or_config(args.n_inner_iters, cellot_args.get("n_inner_iters", 1), int)
    metric_repeats = 2 if bool(args.fast) else _arg_or_config(args.metric_repeats, cellot_args.get("metric_repeats", 30), int)
    max_eval_ctrl_default = int(cellot_args.get("max_eval_ctrl", 0))
    max_eval_ctrl = _arg_or_config(args.max_eval_ctrl, max_eval_ctrl_default, int)

    prov = run_aligned_cellot(
        dataset=dataset_key,
        config_path=config_path,
        split_ids=split_ids,
        out_root=(REPO_ROOT / str(args.out_root or cellot_args.get("out_root", "artifacts/results/cellot"))).resolve(),
        work_root=(REPO_ROOT / str(args.work_root or cellot_args.get("work_root", "artifacts/results/cellot/trishift_aligned"))).resolve(),
        batch_size=batch_size,
        n_iters=n_iters,
        n_inner_iters=n_inner_iters,
        parallel_maps=_arg_or_config(args.parallel_maps, cellot_args.get("parallel_maps", 1), int),
        map_threads=_arg_or_config(args.map_threads, cellot_args.get("map_threads", 4), int),
        metric_repeats=max(1, metric_repeats),
        max_train_conditions=(int(args.max_train_conditions) if int(args.max_train_conditions) > 0 else None),
        max_eval_ctrl=(max_eval_ctrl if max_eval_ctrl > 0 else None),
        force=bool(args.force),
        train=not bool(args.no_train),
        evaluate=not bool(args.no_evaluate),
    )
    print(json.dumps(prov, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
