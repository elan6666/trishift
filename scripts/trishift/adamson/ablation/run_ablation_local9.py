from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
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
        yaml.safe_dump(obj, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def _safe_git_commit(repo_root: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(repo_root), stderr=subprocess.DEVNULL
        )
    except Exception:
        return None
    s = out.decode("utf-8", "replace").strip()
    return s or None


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


@dataclass(frozen=True)
class RunSpec:
    idx: int
    label: str
    overrides: dict


def _build_runs() -> list[RunSpec]:
    runs: list[RunSpec] = []

    def add(label: str, overrides: dict) -> None:
        runs.append(RunSpec(idx=len(runs) + 1, label=label, overrides=overrides))

    # 01 PD_lambda_z_0p3
    add(
        "PD_lambda_z_0p3",
        {"model": {"stage2": {"predict_delta": True}}, "loss": {"lambda_z": 0.3}},
    )
    # 02 PD_lambda_z_0p1
    add(
        "PD_lambda_z_0p1",
        {"model": {"stage2": {"predict_delta": True}}, "loss": {"lambda_z": 0.1}},
    )
    # 03 PD_lambda_z_0p5_plus_residual
    add(
        "PD_lambda_z_0p5_plus_residual",
        {
            "model": {"stage2": {"predict_delta": True}, "stage3": {"use_residual_head": True}},
            "loss": {"lambda_z": 0.5},
        },
    )
    # 04 TR_latent_mu_plus_residual
    add(
        "TR_latent_mu_plus_residual",
        {
            "model": {
                "stage2": {
                    "use_transformer_block": True,
                    "use_cross_attention": False,
                    "shift_input_source": "latent_mu",
                },
                "stage3": {"use_residual_head": True},
            }
        },
    )
    # 05 lambda_expr_mse_0p5
    add("lambda_expr_mse_0p5", {"loss": {"lambda_expr_mse": 0.5}})
    # 06 PD_lambda_z_0p7
    add(
        "PD_lambda_z_0p7",
        {"model": {"stage2": {"predict_delta": True}}, "loss": {"lambda_z": 0.7}},
    )
    # 07 TR_latent_mu_plus_PD_lambda_z_0p5
    add(
        "TR_latent_mu_plus_PD_lambda_z_0p5",
        {
            "model": {
                "stage2": {
                    "use_transformer_block": True,
                    "use_cross_attention": False,
                    "shift_input_source": "latent_mu",
                    "predict_delta": True,
                }
            },
            "loss": {"lambda_z": 0.5},
        },
    )
    # 08 PD_lambda_z_0p5_plus_expr_mse_0p2
    add(
        "PD_lambda_z_0p5_plus_expr_mse_0p2",
        {
            "model": {"stage2": {"predict_delta": True}},
            "loss": {"lambda_z": 0.5, "lambda_expr_mse": 0.2},
        },
    )
    # 09 TR_latent_mu_plus_expr_mse_0p2
    add(
        "TR_latent_mu_plus_expr_mse_0p2",
        {
            "model": {
                "stage2": {
                    "use_transformer_block": True,
                    "use_cross_attention": False,
                    "shift_input_source": "latent_mu",
                }
            },
            "loss": {"lambda_expr_mse": 0.2},
        },
    )

    return runs


def main() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    runner = repo_root / "scripts" / "run_dataset.py"
    if not runner.exists():
        raise SystemExit(f"missing runner: {runner}")

    parser = argparse.ArgumentParser(
        description="Adamson local 9-run combo ablation (serial) based on configs/defaults.yaml"
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--reuse_root", default="", help="reuse existing sweep root directory")
    parser.add_argument("--keep_going", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    dataset = str(args.dataset)
    if dataset != "adamson":
        raise SystemExit("Only --dataset adamson is supported for now")

    base_defaults_path = repo_root / "configs" / "defaults.yaml"
    base_paths_path = repo_root / "configs" / "paths.yaml"
    base_defaults = yaml.safe_load(base_defaults_path.read_text(encoding="utf-8")) or {}
    base_paths = yaml.safe_load(base_paths_path.read_text(encoding="utf-8")) or {}

    runs = _build_runs()
    if args.reuse_root.strip():
        sweep_root = Path(args.reuse_root)
    else:
        sweep_root = repo_root / "artifacts" / "ablation" / dataset / f"{_ts_local()}_local9_combo"
    sweep_root.mkdir(parents=True, exist_ok=True)

    sweep_meta = {
        "dataset": dataset,
        "git_commit": _safe_git_commit(repo_root),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "runs_total": len(runs),
        "baseline_source": "configs/defaults.yaml (no baseline run in this sweep; idx=1 used as delta reference)",
        "reuse_z_mu_cache_expected": False,
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

    for spec in runs:
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

        defaults = copy.deepcopy(base_defaults)
        _deep_update(defaults, copy.deepcopy(spec.overrides))
        # Ensure the decision is explicit per-run.
        defaults.setdefault("ablation", {})
        defaults["ablation"]["reuse_z_mu_cache"] = False

        defaults_path = run_dir / "defaults.yaml"
        paths_path = run_dir / "paths.yaml"
        _dump_yaml(defaults_path, defaults)
        _dump_yaml(paths_path, base_paths)

        run_meta = {
            "idx": int(spec.idx),
            "label": spec.label,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "git_commit": _safe_git_commit(repo_root),
            "dataset": dataset,
            "overrides": spec.overrides,
            "defaults_source": str(base_defaults_path),
            "paths_source": str(base_paths_path),
        }
        (run_dir / "run_meta.json").write_text(
            json.dumps(run_meta, indent=2, sort_keys=False), encoding="utf-8"
        )

        cmd = [
            sys.executable,
            str(runner),
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

        status = "unknown"
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
        flush_summary()

    print(str(sweep_root))


if __name__ == "__main__":
    main()
