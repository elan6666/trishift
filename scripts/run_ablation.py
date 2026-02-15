from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import copy
import csv
import json
import os
import subprocess
import sys
import time

import yaml


def _ts_local() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def _deep_update(dst: dict, src: dict) -> dict:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


def _dump_yaml(path: Path, obj: dict) -> None:
    path.write_text(
        yaml.safe_dump(obj, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )


def _safe_git_commit(repo_root: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(repo_root), stderr=subprocess.DEVNULL
        )
    except Exception:
        return None
    return out.decode("utf-8", "replace").strip() or None


def _read_mean_pearson(run_dir: Path) -> float | None:
    p = run_dir / "mean_pearson.txt"
    if not p.exists():
        return None
    try:
        first = p.read_text(encoding="utf-8", errors="replace").splitlines()[0].strip()
        return float(first)
    except Exception:
        return None


@dataclass(frozen=True)
class RunSpec:
    idx: int
    label: str
    overrides: dict


def _build_runs() -> list[RunSpec]:
    runs: list[RunSpec] = []

    def add(label: str, overrides: dict):
        runs.append(RunSpec(idx=len(runs) + 1, label=label, overrides=overrides))

    # Batch 1
    add("B0_baseline", {})
    add("A1_train_mode_sequential", {"train_mode": "sequential"})
    add("A2_train_mode_stage3_only", {"train_mode": "stage3_only"})
    add("A3_train_mode_latent_decoder", {"train_mode": "latent_decoder"})
    add("T1_topk_weighted_sample", {"ablation": {"topk_strategy": "weighted_sample"}})

    # Batch 2
    add("L1_latent_loss_mse", {"ablation": {"latent_loss_type": "mse"}})
    add("L2_latent_loss_smooth_l1", {"ablation": {"latent_loss_type": "smooth_l1"}})
    add("D1_predict_delta_false", {"model": {"stage2": {"predict_delta": False}}})
    add("S1_shift_input_source_state", {"model": {"stage2": {"shift_input_source": "state"}}})
    add("C1_cond_l2_norm_true", {"model": {"stage2": {"cond_l2_norm": True}}})

    # Batch 3
    add("X1_cross_attention", {"model": {"stage2": {"use_cross_attention": True}}})
    add(
        "TR1_transformer_first",
        {"model": {"stage2": {"use_transformer_block": True, "transformer_readout": "first"}}},
    )
    add(
        "TR2_transformer_mean",
        {"model": {"stage2": {"use_transformer_block": True, "transformer_readout": "mean"}}},
    )
    add(
        "TR3_transformer_concat",
        {"model": {"stage2": {"use_transformer_block": True, "transformer_readout": "concat"}}},
    )
    add("R1_use_residual_head_true", {"model": {"stage3": {"use_residual_head": True}}})

    return runs


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(description="Run TriShift ablation sweep")
    parser.add_argument("--dataset", required=True, help="dataset name (only adamson supported)")
    parser.add_argument("--num_batches", type=int, default=3)
    parser.add_argument("--batch_idx", type=int, required=True, help="1-based batch index")
    parser.add_argument("--reuse_root", default="", help="reuse existing sweep root directory")
    parser.add_argument("--keep_going", action="store_true", help="continue after a failed run")
    parser.add_argument("--dry_run", action="store_true", help="print commands without running")
    args = parser.parse_args()

    dataset = str(args.dataset)
    if dataset != "adamson":
        raise ValueError("Only --dataset adamson is supported for now")

    if args.num_batches != 3:
        raise ValueError("This sweep is defined for --num_batches 3")
    if args.batch_idx not in {1, 2, 3}:
        raise ValueError("--batch_idx must be 1, 2, or 3")

    base_defaults_path = repo_root / "configs" / "defaults.yaml"
    base_paths_path = repo_root / "configs" / "paths.yaml"
    base_defaults = yaml.safe_load(base_defaults_path.read_text(encoding="utf-8")) or {}
    base_paths = yaml.safe_load(base_paths_path.read_text(encoding="utf-8")) or {}

    runs = _build_runs()
    if len(runs) != 15:
        raise RuntimeError("Expected 15 runs")

    start = (args.batch_idx - 1) * 5
    end = start + 5
    batch_runs = runs[start:end]

    if args.reuse_root.strip():
        sweep_root = Path(args.reuse_root)
    else:
        sweep_root = repo_root / "artifacts" / "ablation" / dataset / f"{_ts_local()}_sweep"
    sweep_root.mkdir(parents=True, exist_ok=True)

    sweep_meta = {
        "dataset": dataset,
        "git_commit": _safe_git_commit(repo_root),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "num_batches": 3,
        "runs_total": 15,
        "global_overrides": {
            "run.multi_split": True,
            "run.n_splits": 3,
            "train.*.epochs": 30,
            "n_eval_ensemble": 300,
            "performance.num_workers": 4,
            "ablation.reuse_ot_cache": True,
            "ablation.reuse_z_mu_cache": True,
        },
    }
    (sweep_root / "sweep_meta.json").write_text(
        json.dumps(sweep_meta, indent=2, sort_keys=False), encoding="utf-8"
    )

    summary_rows: list[dict] = []
    summary_path = sweep_root / "summary.csv"
    if summary_path.exists():
        try:
            with summary_path.open("r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if not row:
                        continue
                    if "idx" in row and row["idx"]:
                        summary_rows.append(row)
        except Exception:
            summary_rows = []

    def upsert_row(new_row: dict) -> None:
        # Replace row with same idx if it exists, else append.
        idx = str(new_row.get("idx"))
        for i, row in enumerate(summary_rows):
            if str(row.get("idx")) == idx:
                summary_rows[i] = new_row
                return
        summary_rows.append(new_row)

    for spec in batch_runs:
        run_dir = sweep_root / f"{spec.idx:02d}_{spec.label}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # Skip if already finished.
        if (run_dir / "metrics.csv").exists():
            upsert_row(
                {
                    "idx": spec.idx,
                    "label": spec.label,
                    "status": "skipped_exists",
                    "run_dir": str(run_dir),
                    "mean_pearson": _read_mean_pearson(run_dir),
                }
            )
            continue

        # Build per-run defaults starting from base + global overrides + run overrides.
        defaults_run = copy.deepcopy(base_defaults)
        _deep_update(
            defaults_run,
            {
                "train_mode": "joint",
                "run": {"multi_split": True, "n_splits": 3},
                "train": {
                    "stage1": {"epochs": 30},
                    "stage23": {"epochs": 30},
                    "stage2": {"epochs": 30},
                    "stage3": {"epochs": 30},
                },
                "n_eval_ensemble": 300,
                "performance": {"num_workers": 4},
                "ablation": {"reuse_ot_cache": True, "reuse_z_mu_cache": True},
            },
        )
        _deep_update(defaults_run, spec.overrides)

        defaults_path = run_dir / "defaults.yaml"
        paths_path = run_dir / "paths.yaml"
        _dump_yaml(defaults_path, defaults_run)
        _dump_yaml(paths_path, base_paths)

        run_meta = {
            "idx": spec.idx,
            "label": spec.label,
            "batch_idx": args.batch_idx,
            "dataset": dataset,
            "git_commit": _safe_git_commit(repo_root),
            "defaults_path": str(defaults_path),
            "paths_path": str(paths_path),
            "run_dir": str(run_dir),
            "overrides": spec.overrides,
        }
        (run_dir / "run_meta.json").write_text(
            json.dumps(run_meta, indent=2, sort_keys=False), encoding="utf-8"
        )

        cmd = [
            sys.executable,
            str(repo_root / "scripts" / "run_dataset.py"),
            "--name",
            dataset,
            "--defaults",
            str(defaults_path),
            "--paths",
            str(paths_path),
            "--out_dir",
            str(run_dir),
        ]

        (run_dir / "command.txt").write_text(" ".join(cmd) + "\n", encoding="utf-8")

        if args.dry_run:
            print("[dry_run]", " ".join(cmd))
            upsert_row(
                {
                    "idx": spec.idx,
                    "label": spec.label,
                    "status": "dry_run",
                    "run_dir": str(run_dir),
                    "mean_pearson": None,
                }
            )
            continue

        stdout_path = run_dir / "stdout.log"
        stderr_path = run_dir / "stderr.log"
        print(f"[sweep] run {spec.idx:02d} {spec.label}")
        try:
            with stdout_path.open("w", encoding="utf-8") as out_f, stderr_path.open(
                "w", encoding="utf-8"
            ) as err_f:
                subprocess.run(
                    cmd,
                    cwd=str(repo_root),
                    stdout=out_f,
                    stderr=err_f,
                    check=True,
                    env=dict(os.environ),
                )
            status = "ok"
        except subprocess.CalledProcessError as exc:
            status = f"failed_rc{exc.returncode}"
            if not args.keep_going:
                raise

        upsert_row(
            {
                "idx": spec.idx,
                "label": spec.label,
                "status": status,
                "run_dir": str(run_dir),
                "mean_pearson": _read_mean_pearson(run_dir),
            }
        )

        # Write/update summary incrementally.
        with summary_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["idx", "label", "status", "run_dir", "mean_pearson"]
            )
            writer.writeheader()
            for row in sorted(summary_rows, key=lambda r: int(r["idx"])):
                writer.writerow(row)

    print(str(sweep_root))


if __name__ == "__main__":
    main()
