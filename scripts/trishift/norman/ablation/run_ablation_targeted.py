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

from scripts.common.yaml_utils import dump_yaml, load_yaml_file, merged_dict


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
    paths_overrides: dict


def _build_runs() -> list[RunSpec]:
    runs: list[RunSpec] = []

    def add(label: str, defaults_overrides: dict | None = None, paths_overrides: dict | None = None) -> None:
        runs.append(
            RunSpec(
                idx=len(runs) + 1,
                label=label,
                defaults_overrides=defaults_overrides or {},
                paths_overrides=paths_overrides or {},
            )
        )

    add("B0_baseline_current")
    add("A1_cross_attn_heads_1", {"model": {"stage2": {"cross_attn_heads": 1}}})
    add("A2_predict_shift_false", {"model": {"stage2": {"predict_shift": False}}})
    add("A3_lambda_neg_expr_1", {"loss": {"lambda_neg_expr": 1}})
    add("A4_stage1_ecs_enable_true", {"train": {"stage1": {"ecs": {"enable": True}}}})
    add("A5_shift_repr_dim_null", {"model": {"stage2": {"shift_repr_dim": None}}})
    add(
        "A6_genept_model3_text",
        paths_overrides={
            "embeddings": {
                "emb_a": r"E:\CODE\trishift\src\data\Data_GeneEmbd\GenePT_gene_protein_embedding_model_3_text.pickle"
            }
        },
    )
    add("A7_shift_input_source_state", {"model": {"stage2": {"shift_input_source": "state"}}})
    return runs


def _load_local_config(path: Path) -> dict:
    obj = load_yaml_file(path)
    dataset = str(obj.get("dataset", "")).strip()
    if dataset != "norman":
        raise ValueError(f"Expected dataset: norman in {path}, got {dataset!r}")
    sweep = obj.get("sweep") or {}
    baseline_config = str(obj.get("baseline_config", "")).strip()
    if not isinstance(sweep, dict) or not baseline_config:
        raise TypeError(f"sweep must be a mapping and baseline_config is required in {path}")
    return {"dataset": dataset, "sweep": sweep, "baseline_config": baseline_config}


def main() -> None:
    local_cfg_path = Path(__file__).with_name("config_targeted_ablation.yaml")
    local_cfg = _load_local_config(local_cfg_path)
    sweep_cfg = local_cfg["sweep"]
    dataset = "norman"

    parser = argparse.ArgumentParser(
        description="Run targeted Norman TriShift ablations on selected splits"
    )
    parser.add_argument(
        "--run_ids",
        default=_normalize_run_ids_default(sweep_cfg.get("run_ids", "")),
        help="comma-separated run ids to execute (e.g. 1,3,5)",
    )
    parser.add_argument("--reuse_root", default="", help="reuse existing sweep root directory")
    parser.add_argument("--keep_going", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--gpu_lock",
        default=str(sweep_cfg.get("gpu_lock", "none")),
        help="GPU mutex lock name. Use 'none' to disable locking.",
    )
    parser.add_argument("--gpu_lock_timeout_sec", type=float, default=24 * 3600)
    parser.add_argument(
        "--suffix",
        default=str(sweep_cfg.get("output_suffix", "targeted_ablation_split25")),
        help="output directory suffix (timestamp prefix added)",
    )
    args = parser.parse_args()

    baseline_cfg_path = (REPO_ROOT / str(local_cfg["baseline_config"])).resolve()
    baseline_cfg = load_yaml_file(baseline_cfg_path)
    base_cfg = baseline_cfg.get("base") or {}
    baseline_defaults_overrides = baseline_cfg.get("defaults_overrides") or {}
    baseline_paths_overrides = baseline_cfg.get("paths_overrides") or {}
    if not isinstance(base_cfg, dict):
        raise TypeError(f"baseline base must be a mapping: {baseline_cfg_path}")
    if not isinstance(baseline_defaults_overrides, dict) or not isinstance(baseline_paths_overrides, dict):
        raise TypeError(f"baseline overrides must be mappings: {baseline_cfg_path}")

    base_defaults_path = (REPO_ROOT / str(base_cfg.get("defaults", "configs/defaults.yaml"))).resolve()
    base_paths_path = (REPO_ROOT / str(base_cfg.get("paths", "configs/paths.yaml"))).resolve()
    base_defaults = load_yaml_file(base_defaults_path)
    base_paths = load_yaml_file(base_paths_path)

    baseline_defaults = merged_dict(base_defaults, baseline_defaults_overrides)
    baseline_paths = merged_dict(base_paths, baseline_paths_overrides)

    split_ids_raw = sweep_cfg.get("split_ids", [2, 5])
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

    sweep_config_out = {
        "dataset": dataset,
        "config_source": str(local_cfg_path),
        "baseline_config": str(baseline_cfg_path),
        "baseline_defaults": str(base_defaults_path),
        "baseline_paths": str(base_paths_path),
        "sweep": {
            "split_ids": split_ids,
            "gpu_lock": str(args.gpu_lock),
            "gpu_lock_timeout_sec": float(args.gpu_lock_timeout_sec),
            "suffix": str(args.suffix),
            "dry_run": bool(args.dry_run),
            "keep_going": bool(args.keep_going),
            "run_ids": requested_ids,
        },
        "global_defaults_overrides": global_defaults_overrides,
        "runs": [
            {
                "idx": int(spec.idx),
                "label": spec.label,
                "defaults_overrides": spec.defaults_overrides,
                "paths_overrides": spec.paths_overrides,
            }
            for spec in runs
        ],
    }
    dump_yaml(sweep_root / "sweep_config.yaml", sweep_config_out, allow_unicode=False)

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
        "baseline_config": str(baseline_cfg_path),
        "baseline_defaults": str(base_defaults_path),
        "baseline_paths": str(base_paths_path),
        "global_defaults_overrides": global_defaults_overrides,
        "config_source": str(local_cfg_path),
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

        defaults_run = copy.deepcopy(baseline_defaults)
        paths_run = copy.deepcopy(baseline_paths)
        _deep_update(defaults_run, copy.deepcopy(global_defaults_overrides))
        _deep_update(defaults_run, copy.deepcopy(spec.defaults_overrides))
        _deep_update(paths_run, copy.deepcopy(spec.paths_overrides))

        defaults_path = run_dir / "defaults.yaml"
        paths_path = run_dir / "paths.yaml"
        dump_yaml(defaults_path, defaults_run, allow_unicode=False)
        dump_yaml(paths_path, paths_run, allow_unicode=False)

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
            "paths_overrides": spec.paths_overrides,
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
                "mean_pearson": _read_mean_pearson(run_dir),
            }
        )
        flush_summary()

    print(str(sweep_root))


if __name__ == "__main__":
    main()
