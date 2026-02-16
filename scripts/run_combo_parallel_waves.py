from __future__ import annotations

from pathlib import Path
import argparse
import subprocess
import sys
import time


def _ts_local() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def _build_cmd(
    *,
    py_exec: str,
    repo_root: Path,
    dataset: str,
    source_root: str,
    top_k: int,
    num_batches: int,
    batch_idx: int,
    reuse_root: str,
    keep_going: bool,
    stage1_epochs: int,
    stage23_epochs: int,
    stage2_epochs: int,
    stage3_epochs: int,
) -> list[str]:
    cmd = [
        py_exec,
        str(repo_root / "scripts" / "run_combo_ablation.py"),
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
        "--reuse_root",
        reuse_root,
        "--stage1_epochs",
        str(stage1_epochs),
        "--stage23_epochs",
        str(stage23_epochs),
        "--stage2_epochs",
        str(stage2_epochs),
        "--stage3_epochs",
        str(stage3_epochs),
    ]
    if keep_going:
        cmd.append("--keep_going")
    return cmd


def _run_wave(
    *,
    wave_name: str,
    batch_ids: list[int],
    repo_root: Path,
    py_exec: str,
    dataset: str,
    source_root: str,
    top_k: int,
    num_batches: int,
    reuse_root: str,
    keep_going: bool,
    stage1_epochs: int,
    stage23_epochs: int,
    stage2_epochs: int,
    stage3_epochs: int,
) -> dict[int, int]:
    procs: dict[int, subprocess.Popen] = {}
    log_handles: dict[int, object] = {}
    for batch_idx in batch_ids:
        cmd = _build_cmd(
            py_exec=py_exec,
            repo_root=repo_root,
            dataset=dataset,
            source_root=source_root,
            top_k=top_k,
            num_batches=num_batches,
            batch_idx=batch_idx,
            reuse_root=reuse_root,
            keep_going=keep_going,
            stage1_epochs=stage1_epochs,
            stage23_epochs=stage23_epochs,
            stage2_epochs=stage2_epochs,
            stage3_epochs=stage3_epochs,
        )
        log_path = Path(reuse_root) / f"{wave_name}_batch{batch_idx}.driver.log"
        with log_path.open("w", encoding="utf-8") as f:
            f.write(" ".join(cmd) + "\n")
        log_f = log_path.open("a", encoding="utf-8")
        log_handles[batch_idx] = log_f
        print(f"[parallel_waves] start {wave_name} batch{batch_idx}")
        print("[parallel_waves] cmd:", " ".join(cmd))
        procs[batch_idx] = subprocess.Popen(
            cmd,
            cwd=str(repo_root),
            stdout=log_f,
            stderr=log_f,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

    rc_map: dict[int, int] = {}
    for batch_idx, proc in procs.items():
        rc = proc.wait()
        rc_map[batch_idx] = int(rc)
        log_handles[batch_idx].close()
        print(f"[parallel_waves] done {wave_name} batch{batch_idx} rc={rc}")
    return rc_map


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Run combo ablation in two parallel waves: "
            "wave1(batch1+2) then wave2(batch3+4)."
        )
    )
    parser.add_argument("--dataset", default="adamson")
    parser.add_argument("--source_root", required=True)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--num_batches", type=int, default=4)
    parser.add_argument("--reuse_root", default="")
    parser.add_argument("--keep_going", action="store_true")
    parser.add_argument("--stage1_epochs", type=int, default=100)
    parser.add_argument("--stage23_epochs", type=int, default=40)
    parser.add_argument("--stage2_epochs", type=int, default=40)
    parser.add_argument("--stage3_epochs", type=int, default=40)
    parser.add_argument(
        "--python_exec",
        default=sys.executable,
        help="Python executable used for child batch runners.",
    )
    args = parser.parse_args()

    if int(args.num_batches) != 4:
        raise ValueError("--num_batches must be 4 for parallel waves mode")

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
            / f"{_ts_local()}_combo_sweep_parallel_e{args.stage1_epochs}_{args.stage23_epochs}"
        )
    sweep_root.mkdir(parents=True, exist_ok=True)

    wave1_rc = _run_wave(
        wave_name="wave1",
        batch_ids=[1, 2],
        repo_root=repo_root,
        py_exec=str(args.python_exec),
        dataset=str(args.dataset),
        source_root=str(args.source_root),
        top_k=int(args.top_k),
        num_batches=int(args.num_batches),
        reuse_root=str(sweep_root),
        keep_going=bool(args.keep_going),
        stage1_epochs=int(args.stage1_epochs),
        stage23_epochs=int(args.stage23_epochs),
        stage2_epochs=int(args.stage2_epochs),
        stage3_epochs=int(args.stage3_epochs),
    )

    wave2_rc = _run_wave(
        wave_name="wave2",
        batch_ids=[3, 4],
        repo_root=repo_root,
        py_exec=str(args.python_exec),
        dataset=str(args.dataset),
        source_root=str(args.source_root),
        top_k=int(args.top_k),
        num_batches=int(args.num_batches),
        reuse_root=str(sweep_root),
        keep_going=bool(args.keep_going),
        stage1_epochs=int(args.stage1_epochs),
        stage23_epochs=int(args.stage23_epochs),
        stage2_epochs=int(args.stage2_epochs),
        stage3_epochs=int(args.stage3_epochs),
    )

    print(str(sweep_root))
    all_rc = {**wave1_rc, **wave2_rc}
    bad = [b for b, rc in all_rc.items() if int(rc) != 0]
    if bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
