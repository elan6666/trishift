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

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from scripts.common.yaml_utils import dump_yaml, load_yaml_file


def _ts_local() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def _deep_update(dst: dict, src: dict) -> dict:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


def _safe_git_commit(repo_root: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(repo_root), stderr=subprocess.DEVNULL
        )
    except Exception:
        return None
    return out.decode("utf-8", "replace").strip() or None


def _parse_mean_metrics(run_dir: Path) -> dict[str, float | None]:
    metrics = {
        "mean_pearson_nearest": None,
        "mean_nmse_nearest": None,
        "mean_deg_mean_r2_nearest": None,
    }
    candidates = [
        run_dir / "mean_pearson_nearest.txt",
        run_dir / "mean_pearson.txt",
    ]
    target = next((p for p in candidates if p.exists()), None)
    if target is None:
        return metrics
    try:
        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return metrics
    for i, raw in enumerate(lines):
        line = raw.strip()
        if not line:
            continue
        if i == 0:
            try:
                metrics["mean_pearson_nearest"] = float(line)
            except Exception:
                pass
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        try:
            val = float(str(value).strip())
        except Exception:
            continue
        key = str(key).strip()
        if key == "mean_pearson":
            metrics["mean_pearson_nearest"] = val
        elif key == "mean_nmse":
            metrics["mean_nmse_nearest"] = val
        elif key == "mean_deg_mean_r2":
            metrics["mean_deg_mean_r2_nearest"] = val
    return metrics


def _read_summary_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    try:
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row and row.get("idx"):
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


def _normalize_run_ids_default(raw: object) -> str:
    if raw is None:
        return ""
    if isinstance(raw, list):
        return ",".join(str(int(x)) for x in raw)
    return str(raw).strip()


@dataclass(frozen=True)
class RunSpec:
    idx: int
    label: str
    defaults_overrides: dict


def _build_runs() -> list[RunSpec]:
    runs: list[RunSpec] = []

    def add(label: str, defaults_overrides: dict) -> None:
        runs.append(
            RunSpec(
                idx=len(runs) + 1,
                label=label,
                defaults_overrides=defaults_overrides,
            )
        )

    add(
        "ablation_shift_input_latent_mu",
        {"model": {"stage2": {"shift_input_source": "latent_mu"}}},
    )
    add(
        "ablation_cross_attn_heads_4",
        {"model": {"stage2": {"cross_attn_heads": 4}}},
    )
    add(
        "ablation_shift_repr_dim_16",
        {"model": {"stage2": {"shift_repr_dim": 16}}},
    )
    add(
        "ablation_ecs_weight5_epochs20",
        {"train": {"stage1": {"ecs": {"enable": False, "weight": 5, "epochs": 20}}}},
    )
    return runs


def _load_local_config(path: Path) -> dict:
    obj = load_yaml_file(path)
    dataset = str(obj.get("dataset", "")).strip()
    if dataset != "adamson":
        raise ValueError(f"Expected dataset: adamson in {path}, got {dataset!r}")
    sweep = obj.get("sweep") or {}
    baseline_defaults = str(obj.get("baseline_defaults", "")).strip()
    baseline_paths = str(obj.get("baseline_paths", "")).strip()
    refs = obj.get("reference_results") or {}
    if not isinstance(sweep, dict) or not baseline_defaults or not baseline_paths:
        raise TypeError(f"Invalid targeted ablation config: {path}")
    if not isinstance(refs, dict):
        raise TypeError(f"reference_results must be a mapping in {path}")
    return {
        "dataset": dataset,
        "sweep": sweep,
        "baseline_defaults": baseline_defaults,
        "baseline_paths": baseline_paths,
        "reference_results": refs,
    }


def _write_pathway_manifest(
    *,
    sweep_root: Path,
    reference_results: dict,
    selected_runs: list[RunSpec],
) -> Path:
    runs = [
        {
            "label": "state",
            "result_dir": str((REPO_ROOT / str(reference_results["state"])).resolve()),
            "variant_tag": "nearest",
        },
        {
            "label": "best",
            "result_dir": str((REPO_ROOT / str(reference_results["best"])).resolve()),
            "variant_tag": "nearest",
        },
    ]
    for spec in selected_runs:
        runs.append(
            {
                "label": spec.label,
                "result_dir": str((sweep_root / f"{spec.idx:02d}_{spec.label}").resolve()),
                "variant_tag": "nearest",
            }
        )
    manifest_path = sweep_root / "pathway_trishift_runs.json"
    manifest_path.write_text(
        json.dumps({"runs": runs}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return manifest_path


def main() -> None:
    local_cfg_path = Path(__file__).with_name("config_state_targeted_ablation.yaml")
    local_cfg = _load_local_config(local_cfg_path)
    sweep_cfg = local_cfg["sweep"]
    dataset = "adamson"

    parser = argparse.ArgumentParser(description="Run Adamson state-based targeted TriShift ablations")
    parser.add_argument(
        "--run_ids",
        default=_normalize_run_ids_default(sweep_cfg.get("run_ids", "")),
        help="comma-separated run ids to execute (e.g. 1,3)",
    )
    parser.add_argument("--reuse_root", default="", help="reuse existing sweep root directory")
    parser.add_argument("--keep_going", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--gpu_lock",
        default=str(sweep_cfg.get("gpu_lock", "gpu0")),
        help="GPU mutex lock name. Use 'none' to disable locking.",
    )
    parser.add_argument("--gpu_lock_timeout_sec", type=float, default=24 * 3600)
    parser.add_argument(
        "--suffix",
        default=str(sweep_cfg.get("output_suffix", "state_targeted_ablation")),
        help="output directory suffix (timestamp prefix added)",
    )
    args = parser.parse_args()

    baseline_defaults_path = (REPO_ROOT / str(local_cfg["baseline_defaults"])).resolve()
    baseline_paths_path = (REPO_ROOT / str(local_cfg["baseline_paths"])).resolve()
    baseline_defaults = load_yaml_file(baseline_defaults_path)
    baseline_paths = load_yaml_file(baseline_paths_path)

    split_ids_raw = sweep_cfg.get("split_ids", [1, 2, 3, 4, 5])
    if not isinstance(split_ids_raw, list) or not split_ids_raw:
        raise ValueError("sweep.split_ids must be a non-empty list")
    split_ids = [int(x) for x in split_ids_raw]
    if any(x <= 0 for x in split_ids):
        raise ValueError("sweep.split_ids must contain positive integers")

    runs = _build_runs()
    requested_ids = _parse_run_ids(str(args.run_ids), max_idx=len(runs))
    selected_runs = [runs[i - 1] for i in requested_ids] if requested_ids else runs
    if not selected_runs:
        raise ValueError("No runs selected for execution.")

    if args.reuse_root.strip():
        sweep_root = Path(args.reuse_root)
    else:
        output_root = REPO_ROOT / str(sweep_cfg.get("output_root", f"artifacts/ablation/{dataset}"))
        sweep_root = output_root / f"{_ts_local()}_{str(args.suffix).strip()}"
    sweep_root.mkdir(parents=True, exist_ok=True)

    gpu_lock_path: Path | None
    if str(args.gpu_lock).strip().lower() in {"none", "off", "false", "0"}:
        gpu_lock_path = None
    else:
        gpu_lock_path = REPO_ROOT / "artifacts" / "ablation" / "_locks" / f"{args.gpu_lock}.lock"
        gpu_lock_path.parent.mkdir(parents=True, exist_ok=True)

    global_defaults_overrides = {
        "run": {
            "multi_split": True,
            "n_splits": len(split_ids),
            "split_ids": split_ids,
        }
    }

    manifest_path = _write_pathway_manifest(
        sweep_root=sweep_root,
        reference_results=local_cfg["reference_results"],
        selected_runs=selected_runs,
    )

    sweep_meta = {
        "dataset": dataset,
        "git_commit": _safe_git_commit(REPO_ROOT),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "run_ids": requested_ids,
        "runs_total": len(runs),
        "runs_selected": [int(s.idx) for s in selected_runs],
        "split_ids": split_ids,
        "gpu_lock": str(args.gpu_lock),
        "gpu_lock_path": (str(gpu_lock_path) if gpu_lock_path is not None else None),
        "gpu_lock_timeout_sec": float(args.gpu_lock_timeout_sec),
        "baseline_defaults": str(baseline_defaults_path),
        "baseline_paths": str(baseline_paths_path),
        "global_defaults_overrides": global_defaults_overrides,
        "config_source": str(local_cfg_path),
        "pathway_manifest": str(manifest_path),
    }
    (sweep_root / "sweep_meta.json").write_text(
        json.dumps(sweep_meta, indent=2, sort_keys=False), encoding="utf-8"
    )

    summary_path = sweep_root / "summary.csv"
    summary_fields = [
        "idx",
        "label",
        "status",
        "run_dir",
        "mean_pearson_nearest",
        "mean_nmse_nearest",
        "mean_deg_mean_r2_nearest",
    ]
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

        if (run_dir / "metrics_nearest.csv").exists() or (run_dir / "mean_pearson_nearest.txt").exists():
            metrics = _parse_mean_metrics(run_dir)
            upsert_row(
                {
                    "idx": spec.idx,
                    "label": spec.label,
                    "status": "skipped_exists",
                    "run_dir": str(run_dir),
                    **metrics,
                }
            )
            flush_summary()
            continue

        defaults_run = copy.deepcopy(baseline_defaults)
        _deep_update(defaults_run, copy.deepcopy(global_defaults_overrides))
        _deep_update(defaults_run, copy.deepcopy(spec.defaults_overrides))

        defaults_path = run_dir / "defaults.yaml"
        paths_path = run_dir / "paths.yaml"
        dump_yaml(defaults_path, defaults_run, allow_unicode=False)
        dump_yaml(paths_path, baseline_paths, allow_unicode=False)

        run_meta = {
            "idx": spec.idx,
            "label": spec.label,
            "dataset": dataset,
            "git_commit": _safe_git_commit(REPO_ROOT),
            "defaults_path": str(defaults_path),
            "paths_path": str(paths_path),
            "run_dir": str(run_dir),
            "global_defaults_overrides": global_defaults_overrides,
            "defaults_overrides": spec.defaults_overrides,
            "gpu_lock": str(args.gpu_lock),
            "gpu_lock_path": (str(gpu_lock_path) if gpu_lock_path is not None else None),
            "split_ids": split_ids,
        }
        (run_dir / "run_meta.json").write_text(
            json.dumps(run_meta, indent=2, sort_keys=False), encoding="utf-8"
        )

        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "trishift" / "train" / "main.py"),
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
                    "mean_pearson_nearest": None,
                    "mean_nmse_nearest": None,
                    "mean_deg_mean_r2_nearest": None,
                }
            )
            flush_summary()
            continue

        stdout_path = run_dir / "stdout.log"
        stderr_path = run_dir / "stderr.log"
        print(f"[sweep] run {spec.idx:02d} {spec.label}")

        gpu_fd = None
        try:
            if gpu_lock_path is not None:
                gpu_fd = _acquire_lock(
                    gpu_lock_path, timeout_sec=float(args.gpu_lock_timeout_sec), poll_sec=0.5
                )
            try:
                with stdout_path.open("w", encoding="utf-8") as out_f, stderr_path.open(
                    "w", encoding="utf-8"
                ) as err_f:
                    subprocess.run(
                        cmd,
                        cwd=str(REPO_ROOT),
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
            if gpu_fd is not None and gpu_lock_path is not None:
                _release_lock(gpu_fd, gpu_lock_path)

        upsert_row(
            {
                "idx": spec.idx,
                "label": spec.label,
                "status": status,
                "run_dir": str(run_dir),
                **_parse_mean_metrics(run_dir),
            }
        )
        flush_summary()

    print(str(sweep_root))


if __name__ == "__main__":
    main()
