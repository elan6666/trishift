from __future__ import annotations

import argparse
import csv
import os
import queue
import subprocess
import threading
from datetime import datetime
from pathlib import Path


DEFAULT_DATASETS = ["adamson", "dixit", "norman", "scgen_pbmc_celltype"]
DEFAULT_PRESETS = [
    "ref_random",
    "ref_knn",
    "ref_ot",
    "cond_no_reference",
    "cond_no_prior",
    "cond_full",
]


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _parse_slots(raw: str) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        if ":" in token:
            slot_s, gpu_s = token.split(":", 1)
            out.append((int(slot_s), int(gpu_s)))
        else:
            idx = len(out)
            out.append((idx, int(token)))
    if not out:
        raise ValueError("at least one worker slot is required")
    return out


def _load_rows(plan_root: Path, datasets: list[str], presets: set[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for dataset in datasets:
        path = plan_root / dataset / "ablation_commands.csv"
        if not path.exists():
            raise FileNotFoundError(f"missing ablation command file: {path}")
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("preset", "") in presets:
                    rows.append(row)
    return rows


def _write_commands(log_dir: Path, rows: list[dict[str, str]]) -> None:
    path = log_dir / "commands_resolved.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["idx", "dataset", "preset", "group", "out_dir", "command"],
        )
        writer.writeheader()
        for idx, row in enumerate(rows):
            writer.writerow(
                {
                    "idx": idx,
                    "dataset": row["dataset"],
                    "preset": row["preset"],
                    "group": row["group"],
                    "out_dir": row["out_dir"],
                    "command": row["command"],
                }
            )


def run_plan(
    *,
    repo_root: Path,
    plan_root: Path,
    log_dir: Path,
    datasets: list[str],
    presets: list[str],
    slots: list[tuple[int, int]],
    stop_on_failure: bool,
    dry_run: bool,
) -> int:
    rows = _load_rows(plan_root, datasets, set(presets))
    log_dir.mkdir(parents=True, exist_ok=True)
    _write_commands(log_dir, rows)
    if dry_run:
        print(f"[ablation-plan] dry-run rows={len(rows)} log_dir={log_dir}")
        return 0

    work_queue: queue.Queue[tuple[int, dict[str, str]]] = queue.Queue()
    for idx, row in enumerate(rows):
        work_queue.put((idx, row))

    status_path = log_dir / "status.tsv"
    status_lock = threading.Lock()
    status_path.write_text(
        "time\tidx\tgpu\tdataset\tpreset\tstate\tlog\n",
        encoding="utf-8",
    )
    failures: list[tuple[int, str, str, int]] = []
    failure_lock = threading.Lock()
    stop_event = threading.Event()

    def write_status(idx: int, gpu: int, dataset: str, preset: str, state: str, log: Path) -> None:
        with status_lock:
            with status_path.open("a", encoding="utf-8") as handle:
                handle.write(f"{_now()}\t{idx}\t{gpu}\t{dataset}\t{preset}\t{state}\t{log}\n")

    def worker(slot: int, gpu: int) -> None:
        while not stop_event.is_set():
            try:
                idx, row = work_queue.get_nowait()
            except queue.Empty:
                return
            dataset = row["dataset"]
            preset = row["preset"]
            command = row["command"]
            log = log_dir / f"{idx:02d}_{dataset}_{preset}.log"
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            write_status(idx, gpu, dataset, preset, "start", log)
            with log.open("w", encoding="utf-8") as handle:
                handle.write(
                    f"# started {_now()} slot={slot} gpu={gpu} dataset={dataset} preset={preset}\n"
                )
                handle.write(f"# command: {command}\n")
                handle.flush()
                proc = subprocess.run(
                    command,
                    cwd=repo_root,
                    shell=True,
                    env=env,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                )
                handle.write(f"# finished {_now()} returncode={proc.returncode}\n")
            write_status(idx, gpu, dataset, preset, f"end:{proc.returncode}", log)
            if proc.returncode != 0:
                with failure_lock:
                    failures.append((idx, dataset, preset, proc.returncode))
                if stop_on_failure:
                    stop_event.set()
            work_queue.task_done()

    threads = []
    for slot, gpu in slots:
        thread = threading.Thread(target=worker, args=(slot, gpu), daemon=False)
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()

    summary_path = log_dir / "summary.txt"
    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write(f"finished_at={_now()}\n")
        handle.write(f"total={len(rows)}\n")
        handle.write(f"failures={len(failures)}\n")
        for idx, dataset, preset, code in failures:
            handle.write(f"failure\t{idx}\t{dataset}\t{preset}\t{code}\n")
    return 1 if failures else 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Run generated TriShift ablation commands with GPU slots.")
    ap.add_argument("--repo-root", default=".", help="repository root used as subprocess cwd")
    ap.add_argument("--plan-root", required=True, help="directory containing <dataset>/ablation_commands.csv")
    ap.add_argument("--log-dir", required=True, help="directory for runner logs and status files")
    ap.add_argument("--dataset", action="append", default=[], help="dataset key to include; may repeat")
    ap.add_argument("--preset", action="append", default=[], help="preset key to include; may repeat")
    ap.add_argument(
        "--slots",
        default="0:0,1:0,2:0,3:1,4:1,5:1",
        help="comma-separated worker slots as slot:gpu or gpu entries",
    )
    ap.add_argument("--stop-on-failure", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    datasets = args.dataset or DEFAULT_DATASETS
    presets = args.preset or DEFAULT_PRESETS
    return run_plan(
        repo_root=repo_root,
        plan_root=Path(args.plan_root),
        log_dir=Path(args.log_dir),
        datasets=datasets,
        presets=presets,
        slots=_parse_slots(args.slots),
        stop_on_failure=bool(args.stop_on_failure),
        dry_run=bool(args.dry_run),
    )


if __name__ == "__main__":
    raise SystemExit(main())
