from __future__ import annotations

from pathlib import Path
import argparse
import subprocess
import sys


def _parse_root(stdout_text: str) -> str:
    lines = [ln.strip() for ln in stdout_text.splitlines() if ln.strip()]
    for line in reversed(lines):
        p = Path(line)
        if p.exists() and p.is_dir():
            return str(p)
    if lines:
        return lines[-1]
    raise RuntimeError("Failed to parse sweep root from run_combo_ablation output.")


def _run_batch(
    *,
    repo_root: Path,
    dataset: str,
    source_root: str,
    top_k: int,
    num_batches: int,
    batch_idx: int,
    reuse_root: str,
    keep_going: bool,
    dry_run: bool,
    stage1_epochs: int,
    stage23_epochs: int,
    stage2_epochs: int,
    stage3_epochs: int,
) -> tuple[int, str]:
    script = repo_root / "scripts" / "run_combo_ablation.py"
    cmd = [
        sys.executable,
        str(script),
        "--dataset",
        dataset,
        "--source_root",
        source_root,
        "--top_k",
        str(top_k),
        "--num_batches",
        str(num_batches),
        "--batch_idx",
        str(batch_idx),
        "--stage1_epochs",
        str(stage1_epochs),
        "--stage23_epochs",
        str(stage23_epochs),
        "--stage2_epochs",
        str(stage2_epochs),
        "--stage3_epochs",
        str(stage3_epochs),
    ]
    if reuse_root:
        cmd.extend(["--reuse_root", reuse_root])
    if keep_going:
        cmd.append("--keep_going")
    if dry_run:
        cmd.append("--dry_run")

    print(f"[combo_batches] batch {batch_idx}/{num_batches}")
    print("[combo_batches] cmd:", " ".join(cmd))
    proc = subprocess.run(
        cmd,
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.stdout:
        print(proc.stdout, end="" if proc.stdout.endswith("\n") else "\n")
    if proc.stderr:
        print(proc.stderr, end="" if proc.stderr.endswith("\n") else "\n", file=sys.stderr)

    root = ""
    if proc.stdout:
        try:
            root = _parse_root(proc.stdout)
        except Exception:
            root = ""
    return proc.returncode, root


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Run TopK combo ablation in sequential batches."
    )
    parser.add_argument("--dataset", default="adamson")
    parser.add_argument("--source_root", required=True)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--num_batches", type=int, default=4)
    parser.add_argument("--keep_going", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--stage1_epochs", type=int, default=100)
    parser.add_argument("--stage23_epochs", type=int, default=40)
    parser.add_argument("--stage2_epochs", type=int, default=40)
    parser.add_argument("--stage3_epochs", type=int, default=40)
    args = parser.parse_args()

    reuse_root = ""
    for batch_idx in range(1, int(args.num_batches) + 1):
        rc, root = _run_batch(
            repo_root=repo_root,
            dataset=str(args.dataset),
            source_root=str(args.source_root),
            top_k=int(args.top_k),
            num_batches=int(args.num_batches),
            batch_idx=batch_idx,
            reuse_root=reuse_root,
            keep_going=bool(args.keep_going),
            dry_run=bool(args.dry_run),
            stage1_epochs=int(args.stage1_epochs),
            stage23_epochs=int(args.stage23_epochs),
            stage2_epochs=int(args.stage2_epochs),
            stage3_epochs=int(args.stage3_epochs),
        )
        if not reuse_root and root:
            reuse_root = root
        if rc != 0:
            raise SystemExit(rc)

    if reuse_root:
        print(f"[combo_batches] done. root={reuse_root}")


if __name__ == "__main__":
    main()
