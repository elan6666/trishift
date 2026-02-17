from __future__ import annotations

from pathlib import Path
import argparse
import os
import subprocess
import sys
import time


def _ts_local() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    runner = repo_root / "scripts" / "run_ablation_custom.py"
    analyzer = repo_root / "scripts" / "analyze_sweep_cn.py"

    ap = argparse.ArgumentParser(
        description="Warmup baseline then run two batches in parallel (single GPU safe via lock)."
    )
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--reuse_root", default="", help="reuse existing sweep root")
    ap.add_argument("--keep_going", action="store_true")
    ap.add_argument(
        "--batchA",
        default="2,4,6,8,10,12,14,16,18,20",
        help="comma-separated run ids for batch A",
    )
    ap.add_argument(
        "--batchB",
        default="3,5,7,9,11,13,15,17,19",
        help="comma-separated run ids for batch B",
    )
    args = ap.parse_args()

    if str(args.dataset) != "adamson":
        raise SystemExit("Only --dataset adamson is supported for now")

    if args.reuse_root.strip():
        root = Path(args.reuse_root)
    else:
        root = repo_root / "artifacts" / "ablation" / "adamson" / f"{_ts_local()}_custom_sweep_pdFalse_base"
    root.mkdir(parents=True, exist_ok=True)

    base_cmd = [sys.executable, str(runner), "--dataset", "adamson", "--reuse_root", str(root)]
    if args.keep_going:
        base_cmd.append("--keep_going")

    def run_blocking(run_ids: str, tag: str) -> int:
        cmd = base_cmd + ["--run_ids", run_ids]
        log = root / f"{tag}.driver.log"
        with log.open("a", encoding="utf-8") as f:
            f.write("[cmd] " + " ".join(cmd) + "\n")
            f.flush()
            p = subprocess.run(cmd, cwd=str(repo_root), stdout=f, stderr=subprocess.STDOUT)
            return int(p.returncode)

    # Warmup baseline.
    rc = run_blocking("1", "warmup")
    if rc != 0 and not args.keep_going:
        raise SystemExit(rc)

    # Parallel batches (run_ablation_custom itself uses GPU lock, so safe on single GPU).
    cmd_a = base_cmd + ["--run_ids", str(args.batchA)]
    cmd_b = base_cmd + ["--run_ids", str(args.batchB)]
    log_a = root / "batchA.driver.log"
    log_b = root / "batchB.driver.log"

    env = dict(os.environ)
    with log_a.open("a", encoding="utf-8") as fa, log_b.open("a", encoding="utf-8") as fb:
        fa.write("[cmd] " + " ".join(cmd_a) + "\n")
        fb.write("[cmd] " + " ".join(cmd_b) + "\n")
        fa.flush()
        fb.flush()
        pa = subprocess.Popen(cmd_a, cwd=str(repo_root), stdout=fa, stderr=subprocess.STDOUT, env=env)
        pb = subprocess.Popen(cmd_b, cwd=str(repo_root), stdout=fb, stderr=subprocess.STDOUT, env=env)
        ra = pa.wait()
        rb = pb.wait()

    # Analyze on server/local filesystem.
    try:
        subprocess.run(
            [sys.executable, str(analyzer), "--sweep_root", str(root), "--baseline_idx", "1"],
            cwd=str(repo_root),
            check=False,
        )
    except Exception:
        pass

    # Return combined status (non-zero if any batch failed).
    if ra != 0 or rb != 0:
        raise SystemExit(1)
    print(str(root))


if __name__ == "__main__":
    main()

