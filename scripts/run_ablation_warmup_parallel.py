from __future__ import annotations

from pathlib import Path
import argparse
import os
import re
import subprocess
import sys
import time


def _ts_local() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def _run_cmd(cmd: list[str], repo_root: Path, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        f.write(" ".join(cmd) + "\n")
        f.flush()
        proc = subprocess.run(
            cmd,
            cwd=str(repo_root),
            stdout=f,
            stderr=f,
            env=dict(os.environ),
            check=False,
        )
    return int(proc.returncode)


def _spawn_cmd(cmd: list[str], repo_root: Path, log_path: Path) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    f = log_path.open("w", encoding="utf-8")
    f.write(" ".join(cmd) + "\n")
    f.flush()
    return subprocess.Popen(
        cmd,
        cwd=str(repo_root),
        stdout=f,
        stderr=f,
        env=dict(os.environ),
    )


def _extract_cache_splits_from_stdout(stdout_log: Path) -> set[int]:
    if not stdout_log.exists():
        return set()
    txt = stdout_log.read_text(encoding="utf-8", errors="replace")
    pat = re.compile(r"adamson_split(\d+)_s1[0-9a-fA-F]{12}\.npz")
    out = set()
    for m in pat.finditer(txt):
        try:
            out.add(int(m.group(1)))
        except Exception:
            continue
    return out


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Run baseline warmup first (to ensure z_mu cache), "
            "then launch two parallel single-factor batches."
        )
    )
    parser.add_argument("--dataset", default="adamson")
    parser.add_argument("--reuse_root", default="")
    parser.add_argument("--keep_going", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--stage1_epochs", type=int, default=100)
    parser.add_argument("--stage23_epochs", type=int, default=40)
    parser.add_argument("--stage2_epochs", type=int, default=40)
    parser.add_argument("--stage3_epochs", type=int, default=40)
    args = parser.parse_args()

    if str(args.dataset) != "adamson":
        raise ValueError("Only --dataset adamson is supported")

    if args.reuse_root.strip():
        sweep_root = Path(args.reuse_root)
    else:
        sweep_root = (
            repo_root
            / "artifacts"
            / "ablation"
            / str(args.dataset)
            / f"{_ts_local()}_warmup_parallel_e{int(args.stage1_epochs)}_{int(args.stage23_epochs)}"
        )
    sweep_root.mkdir(parents=True, exist_ok=True)

    py_exec = sys.executable
    run_ablation_py = repo_root / "scripts" / "run_ablation.py"

    common = [
        py_exec,
        str(run_ablation_py),
        "--dataset",
        str(args.dataset),
        "--reuse_root",
        str(sweep_root),
        "--stage1_epochs",
        str(int(args.stage1_epochs)),
        "--stage23_epochs",
        str(int(args.stage23_epochs)),
        "--stage2_epochs",
        str(int(args.stage2_epochs)),
        "--stage3_epochs",
        str(int(args.stage3_epochs)),
    ]
    if bool(args.keep_going):
        common.append("--keep_going")
    if bool(args.dry_run):
        common.append("--dry_run")

    # 1) Warmup baseline only.
    warmup_cmd = common + ["--run_ids", "1"]
    warmup_driver_log = sweep_root / "warmup_baseline.driver.log"
    print("[warmup] run baseline")
    rc = _run_cmd(warmup_cmd, repo_root=repo_root, log_path=warmup_driver_log)
    if rc != 0:
        raise SystemExit(rc)

    # 2) Validate z_mu cache from warmup stdout.
    if not bool(args.dry_run):
        baseline_stdout = sweep_root / "01_B0_baseline" / "stdout.log"
        seen_splits = _extract_cache_splits_from_stdout(baseline_stdout)
        missing = [sid for sid in (1, 2, 3) if sid not in seen_splits]
        if missing:
            raise RuntimeError(
                "Warmup baseline did not show z_mu cache usage for all splits. "
                f"Missing splits: {missing}. "
                f"Inspect: {baseline_stdout}"
            )

    # 3) Parallel phase: odd(except 1) vs even.
    run_ids_a = "3,5,7,9,11,13,15"
    run_ids_b = "2,4,6,8,10,12,14"
    cmd_a = common + ["--run_ids", run_ids_a]
    cmd_b = common + ["--run_ids", run_ids_b]

    log_a = sweep_root / "parallel_batchA.driver.log"
    log_b = sweep_root / "parallel_batchB.driver.log"
    print("[parallel] start batchA and batchB")
    proc_a = _spawn_cmd(cmd_a, repo_root=repo_root, log_path=log_a)
    proc_b = _spawn_cmd(cmd_b, repo_root=repo_root, log_path=log_b)

    rc_a = int(proc_a.wait())
    rc_b = int(proc_b.wait())
    print(f"[parallel] done: batchA={rc_a}, batchB={rc_b}")
    print(str(sweep_root))

    if rc_a != 0 or rc_b != 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
