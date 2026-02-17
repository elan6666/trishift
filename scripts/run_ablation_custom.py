from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import copy
import csv
import json
import math
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
    # Per-run snapshots should be stable and ASCII-only (server-safe).
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


def _read_summary_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    try:
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not row:
                    continue
                if "idx" in row and row["idx"]:
                    rows.append(row)
    except Exception:
        return []
    return rows


def _acquire_lock(lock_path: Path, timeout_sec: float = 24 * 3600, poll_sec: float = 0.2) -> int:
    start = time.time()
    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
            os.write(fd, str(os.getpid()).encode("ascii", "ignore"))
            return fd
        except FileExistsError:
            if (time.time() - start) >= timeout_sec:
                raise TimeoutError(f"Timed out waiting for lock: {lock_path}")
            time.sleep(poll_sec)


def _release_lock(fd: int, lock_path: Path) -> None:
    try:
        os.close(fd)
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def _parse_run_ids(run_ids_raw: str, max_idx: int) -> list[int]:
    if not run_ids_raw.strip():
        return []
    out: list[int] = []
    for tok in run_ids_raw.split(","):
        s = tok.strip()
        if not s:
            continue
        try:
            i = int(s)
        except Exception as exc:
            raise ValueError(f"Invalid run id '{s}' in --run_ids") from exc
        if i < 1 or i > max_idx:
            raise ValueError(f"--run_ids contains out-of-range id {i}; expected 1..{max_idx}")
        out.append(i)
    return sorted(set(out))


@dataclass(frozen=True)
class RunSpec:
    idx: int
    label: str
    overrides: dict


def _build_runs() -> list[RunSpec]:
    runs: list[RunSpec] = []

    def add(label: str, overrides: dict):
        runs.append(RunSpec(idx=len(runs) + 1, label=label, overrides=overrides))

    # 01 baseline
    add("B0_baseline", {})

    # 02-20 custom single-factor overrides (based on current configs/defaults.yaml baseline).
    add("M1_lambda_expr_mse_0p2", {"loss": {"lambda_expr_mse": 0.2}})

    add(
        "CA1_cross_attn_latent_mu",
        {
            "model": {
                "stage2": {
                    "use_cross_attention": True,
                    "use_transformer_block": False,
                    "shift_input_source": "latent_mu",
                }
            }
        },
    )
    add(
        "CA2_cross_attn_state",
        {
            "model": {
                "stage2": {
                    "use_cross_attention": True,
                    "use_transformer_block": False,
                    "shift_input_source": "state",
                }
            }
        },
    )

    add(
        "TR1_transformer_latent_mu",
        {
            "model": {
                "stage2": {
                    "use_transformer_block": True,
                    "use_cross_attention": False,
                    "shift_input_source": "latent_mu",
                }
            }
        },
    )
    add(
        "TR2_transformer_state",
        {
            "model": {
                "stage2": {
                    "use_transformer_block": True,
                    "use_cross_attention": False,
                    "shift_input_source": "state",
                }
            }
        },
    )

    add("L1_deg_weight_2", {"loss": {"deg_weight": 2}})
    add("G1_use_residual_head_true", {"model": {"stage3": {"use_residual_head": True}}})
    add("S1_predict_delta_true", {"model": {"stage2": {"predict_delta": True}}})
    add("G2_input_mode_fusion_only", {"model": {"stage3": {"input_mode": "fusion_only"}}})
    add("G3_input_mode_state_fusion", {"model": {"stage3": {"input_mode": "state_fusion"}}})
    add(
        "TM1_stage3_only_plus_residual",
        {"train_mode": "stage3_only", "model": {"stage3": {"use_residual_head": True}}},
    )
    add("TM2_train_mode_latent_decoder", {"train_mode": "latent_decoder"})
    add("MM1_matching_mode_soft_ot", {"matching_mode": "soft_ot"})
    add("K1_k_topk_200", {"k_topk": 200})
    add("T1_topk_strategy_weighted_sample", {"ablation": {"topk_strategy": "weighted_sample"}})
    add(
        "S2_predict_delta_true_lambda_z_0p5",
        {"model": {"stage2": {"predict_delta": True}}, "loss": {"lambda_z": 0.5}},
    )
    add(
        "S3_predict_delta_true_latent_smooth_l1",
        {
            "model": {"stage2": {"predict_delta": True}},
            "ablation": {"latent_loss_type": "smooth_l1"},
        },
    )
    add("Z1_z_dim_64", {"model": {"stage1": {"z_dim": 64}}})
    add("S4_shift_input_source_state", {"model": {"stage2": {"shift_input_source": "state"}}})

    if len(runs) != 20:
        raise RuntimeError(f"Expected 20 runs, got {len(runs)}")
    return runs


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(
        description="Run TriShift custom single-factor sweep (adamson only)"
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--num_batches", type=int, default=2)
    parser.add_argument("--batch_idx", type=int, default=1, help="1-based batch index")
    parser.add_argument(
        "--run_ids",
        default="",
        help="comma-separated run ids to execute (e.g. 1,3,5); overrides batch slicing",
    )
    parser.add_argument("--reuse_root", default="", help="reuse existing sweep root directory")
    parser.add_argument("--keep_going", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--gpu_lock_path",
        default="",
        help="optional GPU mutex lock path; default uses artifacts/ablation/_locks/gpu0.lock",
    )
    parser.add_argument("--gpu_lock_timeout_sec", type=float, default=24 * 3600)
    args = parser.parse_args()

    dataset = str(args.dataset)
    if dataset != "adamson":
        raise ValueError("Only --dataset adamson is supported for now")
    if int(args.num_batches) <= 0:
        raise ValueError("--num_batches must be positive")
    if int(args.batch_idx) <= 0 or int(args.batch_idx) > int(args.num_batches):
        raise ValueError("--batch_idx must be within [1, --num_batches]")

    base_defaults_path = repo_root / "configs" / "defaults.yaml"
    base_paths_path = repo_root / "configs" / "paths.yaml"
    base_defaults = yaml.safe_load(base_defaults_path.read_text(encoding="utf-8")) or {}
    base_paths = yaml.safe_load(base_paths_path.read_text(encoding="utf-8")) or {}

    runs = _build_runs()
    requested_ids = _parse_run_ids(str(args.run_ids), max_idx=len(runs))
    if requested_ids:
        selected_runs = [runs[i - 1] for i in requested_ids]
        selection_mode = "run_ids"
    else:
        batch_size = math.ceil(len(runs) / int(args.num_batches))
        start = (int(args.batch_idx) - 1) * batch_size
        end = min(start + batch_size, len(runs))
        selected_runs = runs[start:end]
        selection_mode = "batch_slice"
    if not selected_runs:
        raise ValueError("No runs selected for execution.")

    if args.reuse_root.strip():
        sweep_root = Path(args.reuse_root)
    else:
        sweep_root = repo_root / "artifacts" / "ablation" / dataset / f"{_ts_local()}_custom_sweep"
    sweep_root.mkdir(parents=True, exist_ok=True)

    if str(args.gpu_lock_path).strip():
        gpu_lock_path = Path(str(args.gpu_lock_path))
    else:
        gpu_lock_path = repo_root / "artifacts" / "ablation" / "_locks" / "gpu0.lock"
    gpu_lock_path.parent.mkdir(parents=True, exist_ok=True)

    sweep_meta = {
        "dataset": dataset,
        "git_commit": _safe_git_commit(repo_root),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "num_batches": int(args.num_batches),
        "batch_idx": int(args.batch_idx),
        "run_ids": requested_ids,
        "selection_mode": selection_mode,
        "runs_total": len(runs),
        "runs_selected": [int(s.idx) for s in selected_runs],
        "gpu_lock_path": str(gpu_lock_path),
        "gpu_lock_timeout_sec": float(args.gpu_lock_timeout_sec),
        "baseline_defaults": str(base_defaults_path),
        "baseline_paths": str(base_paths_path),
    }
    (sweep_root / "sweep_meta.json").write_text(
        json.dumps(sweep_meta, indent=2, sort_keys=False), encoding="utf-8"
    )

    summary_path = sweep_root / "summary.csv"
    summary_fields = ["idx", "label", "status", "run_dir", "mean_pearson"]
    summary_rows: list[dict] = _read_summary_rows(summary_path)

    def upsert_row(new_row: dict) -> None:
        idx = str(new_row.get("idx"))
        for i, row in enumerate(summary_rows):
            if str(row.get("idx")) == idx:
                summary_rows[i] = new_row
                return
        summary_rows.append(new_row)

    def flush_summary() -> None:
        lock_path = Path(str(summary_path) + ".lock")
        fd = _acquire_lock(lock_path)
        try:
            disk_rows = _read_summary_rows(summary_path)
            merged_by_idx: dict[str, dict] = {}
            for row in disk_rows:
                merged_by_idx[str(row.get("idx"))] = row
            for row in summary_rows:
                merged_by_idx[str(row.get("idx"))] = row
            merged_rows = sorted(
                merged_by_idx.values(), key=lambda r: int(str(r.get("idx", "0")))
            )
            with summary_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=summary_fields)
                writer.writeheader()
                for row in merged_rows:
                    writer.writerow(row)
            summary_rows.clear()
            summary_rows.extend(merged_rows)
        finally:
            _release_lock(fd, lock_path)

    for spec in selected_runs:
        run_dir = sweep_root / f"{spec.idx:02d}_{spec.label}"
        run_dir.mkdir(parents=True, exist_ok=True)

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
            flush_summary()
            continue

        defaults_run = copy.deepcopy(base_defaults)
        _deep_update(defaults_run, spec.overrides)

        defaults_path = run_dir / "defaults.yaml"
        paths_path = run_dir / "paths.yaml"
        _dump_yaml(defaults_path, defaults_run)
        _dump_yaml(paths_path, base_paths)

        run_meta = {
            "idx": spec.idx,
            "label": spec.label,
            "batch_idx": int(args.batch_idx),
            "run_ids": requested_ids,
            "selection_mode": selection_mode,
            "dataset": dataset,
            "git_commit": _safe_git_commit(repo_root),
            "defaults_path": str(defaults_path),
            "paths_path": str(paths_path),
            "run_dir": str(run_dir),
            "overrides": spec.overrides,
            "gpu_lock_path": str(gpu_lock_path),
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
            flush_summary()
            continue

        stdout_path = run_dir / "stdout.log"
        stderr_path = run_dir / "stderr.log"
        print(f"[sweep] run {spec.idx:02d} {spec.label}")

        gpu_fd = None
        try:
            gpu_fd = _acquire_lock(
                gpu_lock_path, timeout_sec=float(args.gpu_lock_timeout_sec), poll_sec=0.5
            )
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
        finally:
            if gpu_fd is not None:
                _release_lock(gpu_fd, gpu_lock_path)

        upsert_row(
            {
                "idx": spec.idx,
                "label": spec.label,
                "status": status,
                "run_dir": str(run_dir),
                "mean_pearson": _read_mean_pearson(run_dir),
            }
        )
        flush_summary()

    if summary_rows:
        flush_summary()

    print(str(sweep_root))


if __name__ == "__main__":
    main()
