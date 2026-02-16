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


def _flatten_leaf(d: dict, prefix: str = "") -> dict[str, object]:
    out: dict[str, object] = {}
    for k, v in d.items():
        path = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(_flatten_leaf(v, prefix=path))
        else:
            out[path] = v
    return out


def _merge_overrides(left: dict, right: dict) -> tuple[dict, list[str]]:
    lf = _flatten_leaf(left)
    rf = _flatten_leaf(right)
    conflicts = sorted([k for k in lf.keys() & rf.keys() if lf[k] != rf[k]])
    merged = copy.deepcopy(left)
    _deep_update(merged, right)
    return merged, conflicts


def _ensure_constraints(defaults_run: dict) -> tuple[dict, list[str]]:
    notes: list[str] = []
    model = defaults_run.setdefault("model", {})
    stage2 = model.setdefault("stage2", {})
    stage3 = model.setdefault("stage3", {})

    use_transformer = bool(stage2.get("use_transformer_block", False))
    use_cross = bool(stage2.get("use_cross_attention", False))
    if use_transformer and use_cross:
        raise ValueError(
            "Invalid config: model.stage2.use_transformer_block and "
            "model.stage2.use_cross_attention cannot both be true."
        )

    train_mode = str(defaults_run.get("train_mode", "joint"))
    if train_mode == "latent_decoder":
        cur_src = str(stage2.get("shift_input_source", "latent_mu"))
        if cur_src != "latent_mu":
            stage2["shift_input_source"] = "latent_mu"
            notes.append(
                "train_mode=latent_decoder forces model.stage2.shift_input_source=latent_mu"
            )
    if train_mode == "stage3_only":
        input_mode = str(stage3.get("input_mode", "full"))
        if input_mode != "full":
            raise ValueError(
                "Invalid config: train_mode=stage3_only requires model.stage3.input_mode=full."
            )

    return defaults_run, notes


@dataclass(frozen=True)
class ModuleSpec:
    code: str
    label: str
    overrides: dict


@dataclass(frozen=True)
class ComboSpec:
    idx: int
    code_i: str
    code_j: str
    label_i: str
    label_j: str
    merged_overrides: dict
    merge_conflicts: list[str]
    score: float | None = None
    potential: float | None = None
    coverage: float | None = None
    corr: float | None = None

    @property
    def combo_label(self) -> str:
        return f"{self.code_i}_plus_{self.code_j}"


MODULES: dict[str, ModuleSpec] = {
    "01": ModuleSpec("01", "B0_baseline", {}),
    "02": ModuleSpec("02", "A1_train_mode_sequential", {"train_mode": "sequential"}),
    "03": ModuleSpec("03", "A2_train_mode_stage3_only", {"train_mode": "stage3_only"}),
    "04": ModuleSpec("04", "A3_train_mode_latent_decoder", {"train_mode": "latent_decoder"}),
    "05": ModuleSpec("05", "T1_topk_weighted_sample", {"ablation": {"topk_strategy": "weighted_sample"}}),
    "06": ModuleSpec("06", "L1_latent_loss_mse", {"ablation": {"latent_loss_type": "mse"}}),
    "07": ModuleSpec(
        "07", "L2_latent_loss_smooth_l1", {"ablation": {"latent_loss_type": "smooth_l1"}}
    ),
    "08": ModuleSpec("08", "D1_predict_delta_false", {"model": {"stage2": {"predict_delta": False}}}),
    "09": ModuleSpec(
        "09", "S1_shift_input_source_state", {"model": {"stage2": {"shift_input_source": "state"}}}
    ),
    "10": ModuleSpec("10", "C1_cond_l2_norm_true", {"model": {"stage2": {"cond_l2_norm": True}}}),
    "11": ModuleSpec(
        "11", "X1_cross_attention", {"model": {"stage2": {"use_cross_attention": True}}}
    ),
    "12": ModuleSpec(
        "12",
        "TR1_transformer_first",
        {"model": {"stage2": {"use_transformer_block": True, "transformer_readout": "first"}}},
    ),
    "13": ModuleSpec(
        "13",
        "TR2_transformer_mean",
        {"model": {"stage2": {"use_transformer_block": True, "transformer_readout": "mean"}}},
    ),
    "14": ModuleSpec(
        "14",
        "TR3_transformer_concat",
        {"model": {"stage2": {"use_transformer_block": True, "transformer_readout": "concat"}}},
    ),
    "15": ModuleSpec("15", "R1_use_residual_head_true", {"model": {"stage3": {"use_residual_head": True}}}),
}


# Fixed Top20 combos (locked by plan; order matters).
COMBO_CODE_PAIRS: list[tuple[str, str]] = [
    ("03", "13"),
    ("03", "14"),
    ("03", "09"),
    ("03", "07"),
    ("03", "10"),
    ("09", "12"),
    ("03", "11"),
    ("03", "06"),
    ("03", "05"),
    ("06", "09"),
    ("02", "03"),
    ("09", "11"),
    ("09", "10"),
    ("03", "12"),
    ("07", "09"),
    ("03", "08"),
    ("09", "14"),
    ("12", "13"),
    ("09", "13"),
    ("02", "09"),
]


def _load_score_table(source_root: Path) -> dict[tuple[str, str], dict[str, float]]:
    score_path = source_root / "top_combos_computed.csv"
    if not score_path.exists():
        return {}
    label_to_code = {m.label: m.code for m in MODULES.values()}
    out: dict[tuple[str, str], dict[str, float]] = {}
    with score_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            li = str(row.get("run_i", "")).strip()
            lj = str(row.get("run_j", "")).strip()
            ci = ""
            cj = ""
            if len(li) >= 2 and li[0:2].isdigit():
                ci = li[0:2]
            if len(lj) >= 2 and lj[0:2].isdigit():
                cj = lj[0:2]
            if not ci:
                ci = label_to_code.get(li, "")
            if not cj:
                cj = label_to_code.get(lj, "")
            if not ci or not cj:
                continue
            key = (ci, cj)
            parsed = {}
            for k in ("score", "potential", "coverage", "corr"):
                v = row.get(k)
                if v is None or str(v).strip() == "":
                    continue
                try:
                    parsed[k] = float(v)
                except Exception:
                    continue
            out[key] = parsed
    return out


def _build_combo_specs(source_root: Path, top_k: int) -> list[ComboSpec]:
    score_table = _load_score_table(source_root)
    specs: list[ComboSpec] = []
    for idx, (code_i, code_j) in enumerate(COMBO_CODE_PAIRS[:top_k], start=1):
        mod_i = MODULES[code_i]
        mod_j = MODULES[code_j]
        merged, conflicts = _merge_overrides(mod_i.overrides, mod_j.overrides)
        score_info = score_table.get((code_i, code_j), {})
        specs.append(
            ComboSpec(
                idx=idx,
                code_i=code_i,
                code_j=code_j,
                label_i=mod_i.label,
                label_j=mod_j.label,
                merged_overrides=merged,
                merge_conflicts=conflicts,
                score=score_info.get("score"),
                potential=score_info.get("potential"),
                coverage=score_info.get("coverage"),
                corr=score_info.get("corr"),
            )
        )
    return specs


def _write_combo_candidates(root: Path, combo_specs: list[ComboSpec]) -> None:
    path = root / "combo_candidates.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "idx",
                "combo_label",
                "code_i",
                "code_j",
                "label_i",
                "label_j",
                "score",
                "potential",
                "coverage",
                "corr",
                "merge_conflicts",
                "merged_overrides_json",
            ],
        )
        writer.writeheader()
        for s in combo_specs:
            writer.writerow(
                {
                    "idx": s.idx,
                    "combo_label": s.combo_label,
                    "code_i": s.code_i,
                    "code_j": s.code_j,
                    "label_i": s.label_i,
                    "label_j": s.label_j,
                    "score": s.score,
                    "potential": s.potential,
                    "coverage": s.coverage,
                    "corr": s.corr,
                    "merge_conflicts": json.dumps(s.merge_conflicts, ensure_ascii=True),
                    "merged_overrides_json": json.dumps(s.merged_overrides, ensure_ascii=True),
                }
            )


def _acquire_lock(lock_path: Path, timeout_sec: float = 600.0, poll_sec: float = 0.2) -> int:
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


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(description="Run TriShift combo ablation sweep")
    parser.add_argument("--dataset", required=True, help="dataset name (only adamson supported)")
    parser.add_argument("--source_root", required=True, help="source single-ablation root path")
    parser.add_argument("--top_k", type=int, default=20, help="number of fixed combos to run")
    parser.add_argument("--num_batches", type=int, default=4)
    parser.add_argument("--batch_idx", type=int, required=True, help="1-based batch index")
    parser.add_argument("--reuse_root", default="", help="reuse existing sweep root directory")
    parser.add_argument("--keep_going", action="store_true", help="continue after a failed run")
    parser.add_argument("--dry_run", action="store_true", help="print commands without running")
    parser.add_argument("--stage1_epochs", type=int, default=100)
    parser.add_argument("--stage23_epochs", type=int, default=40)
    parser.add_argument("--stage2_epochs", type=int, default=40)
    parser.add_argument("--stage3_epochs", type=int, default=40)
    args = parser.parse_args()

    dataset = str(args.dataset)
    if dataset != "adamson":
        raise ValueError("Only --dataset adamson is supported for now")

    source_root = Path(args.source_root)
    if not source_root.exists():
        raise FileNotFoundError(f"source_root not found: {source_root}")

    max_k = len(COMBO_CODE_PAIRS)
    if args.top_k <= 0 or args.top_k > max_k:
        raise ValueError(f"--top_k must be in [1, {max_k}]")
    if args.num_batches <= 0:
        raise ValueError("--num_batches must be positive")
    if args.batch_idx <= 0 or args.batch_idx > args.num_batches:
        raise ValueError("--batch_idx must be within [1, --num_batches]")

    base_defaults_path = repo_root / "configs" / "defaults.yaml"
    base_paths_path = repo_root / "configs" / "paths.yaml"
    base_defaults = yaml.safe_load(base_defaults_path.read_text(encoding="utf-8")) or {}
    base_paths = yaml.safe_load(base_paths_path.read_text(encoding="utf-8")) or {}

    combo_specs = _build_combo_specs(source_root=source_root, top_k=args.top_k)
    batch_size = math.ceil(len(combo_specs) / args.num_batches)
    start = (args.batch_idx - 1) * batch_size
    end = min(start + batch_size, len(combo_specs))
    batch_specs = combo_specs[start:end]
    if not batch_specs:
        raise ValueError("No combos in selected batch range.")

    if args.reuse_root.strip():
        sweep_root = Path(args.reuse_root)
    else:
        sweep_root = (
            repo_root
            / "artifacts"
            / "ablation"
            / dataset
            / f"{_ts_local()}_combo_sweep_e{args.stage1_epochs}_{args.stage23_epochs}"
        )
    sweep_root.mkdir(parents=True, exist_ok=True)

    sweep_meta = {
        "dataset": dataset,
        "git_commit": _safe_git_commit(repo_root),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "source_root": str(source_root),
        "num_batches": int(args.num_batches),
        "runs_total": int(len(combo_specs)),
        "fixed_pairs": [f"{c.code_i}+{c.code_j}" for c in combo_specs],
        "global_overrides": {
            "run.multi_split": True,
            "run.n_splits": 3,
            "train.stage1.epochs": int(args.stage1_epochs),
            "train.stage23.epochs": int(args.stage23_epochs),
            "train.stage2.epochs": int(args.stage2_epochs),
            "train.stage3.epochs": int(args.stage3_epochs),
            "n_eval_ensemble": 300,
            "performance.num_workers": 4,
            "ablation.reuse_ot_cache": True,
            "ablation.reuse_z_mu_cache": True,
        },
    }
    (sweep_root / "sweep_meta.json").write_text(
        json.dumps(sweep_meta, indent=2, sort_keys=False), encoding="utf-8"
    )
    _write_combo_candidates(sweep_root, combo_specs)

    summary_path = sweep_root / "summary.csv"
    summary_rows: list[dict] = _read_summary_rows(summary_path)

    def upsert_row(new_row: dict) -> None:
        idx = str(new_row.get("idx"))
        for i, row in enumerate(summary_rows):
            if str(row.get("idx")) == idx:
                summary_rows[i] = new_row
                return
        summary_rows.append(new_row)

    summary_fields = [
        "idx",
        "combo_label",
        "status",
        "run_dir",
        "mean_pearson",
        "code_i",
        "code_j",
        "label_i",
        "label_j",
        "merge_conflicts",
        "constraint_notes",
    ]

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

    for spec in batch_specs:
        run_dir = sweep_root / f"{spec.idx:02d}_{spec.combo_label}"
        run_dir.mkdir(parents=True, exist_ok=True)

        if (run_dir / "metrics.csv").exists():
            upsert_row(
                {
                    "idx": spec.idx,
                    "combo_label": spec.combo_label,
                    "status": "skipped_exists",
                    "run_dir": str(run_dir),
                    "mean_pearson": _read_mean_pearson(run_dir),
                    "code_i": spec.code_i,
                    "code_j": spec.code_j,
                    "label_i": spec.label_i,
                    "label_j": spec.label_j,
                    "merge_conflicts": json.dumps(spec.merge_conflicts, ensure_ascii=True),
                }
            )
            flush_summary()
            continue

        defaults_run = copy.deepcopy(base_defaults)
        _deep_update(
            defaults_run,
            {
                "train_mode": "joint",
                "run": {"multi_split": True, "n_splits": 3},
                "train": {
                    "stage1": {"epochs": int(args.stage1_epochs)},
                    "stage23": {"epochs": int(args.stage23_epochs)},
                    "stage2": {"epochs": int(args.stage2_epochs)},
                    "stage3": {"epochs": int(args.stage3_epochs)},
                },
                "n_eval_ensemble": 300,
                "performance": {"num_workers": 4},
                "ablation": {"reuse_ot_cache": True, "reuse_z_mu_cache": True},
            },
        )
        _deep_update(defaults_run, spec.merged_overrides)
        try:
            defaults_run, notes = _ensure_constraints(defaults_run)
            status_constraint = "ok"
        except Exception as exc:
            notes = [str(exc)]
            status_constraint = "invalid_constraints"
            upsert_row(
                {
                    "idx": spec.idx,
                    "combo_label": spec.combo_label,
                    "status": status_constraint,
                    "run_dir": str(run_dir),
                    "mean_pearson": None,
                    "code_i": spec.code_i,
                    "code_j": spec.code_j,
                    "label_i": spec.label_i,
                    "label_j": spec.label_j,
                    "merge_conflicts": json.dumps(spec.merge_conflicts, ensure_ascii=True),
                    "constraint_notes": json.dumps(notes, ensure_ascii=True),
                }
            )
            flush_summary()
            if not args.keep_going:
                raise
            continue

        defaults_path = run_dir / "defaults.yaml"
        paths_path = run_dir / "paths.yaml"
        _dump_yaml(defaults_path, defaults_run)
        _dump_yaml(paths_path, base_paths)

        run_meta = {
            "idx": spec.idx,
            "combo_label": spec.combo_label,
            "batch_idx": args.batch_idx,
            "dataset": dataset,
            "git_commit": _safe_git_commit(repo_root),
            "defaults_path": str(defaults_path),
            "paths_path": str(paths_path),
            "run_dir": str(run_dir),
            "code_i": spec.code_i,
            "code_j": spec.code_j,
            "label_i": spec.label_i,
            "label_j": spec.label_j,
            "merged_overrides": spec.merged_overrides,
            "merge_conflicts": spec.merge_conflicts,
            "constraint_notes": notes,
            "status_constraint": status_constraint,
            "score": spec.score,
            "potential": spec.potential,
            "coverage": spec.coverage,
            "corr": spec.corr,
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
                    "combo_label": spec.combo_label,
                    "status": "dry_run",
                    "run_dir": str(run_dir),
                    "mean_pearson": None,
                    "code_i": spec.code_i,
                    "code_j": spec.code_j,
                    "label_i": spec.label_i,
                    "label_j": spec.label_j,
                    "merge_conflicts": json.dumps(spec.merge_conflicts, ensure_ascii=True),
                    "constraint_notes": json.dumps(notes, ensure_ascii=True),
                }
            )
            flush_summary()
            continue

        stdout_path = run_dir / "stdout.log"
        stderr_path = run_dir / "stderr.log"
        print(f"[combo_sweep] run {spec.idx:02d} {spec.combo_label}")
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
                "combo_label": spec.combo_label,
                "status": status,
                "run_dir": str(run_dir),
                "mean_pearson": _read_mean_pearson(run_dir),
                "code_i": spec.code_i,
                "code_j": spec.code_j,
                "label_i": spec.label_i,
                "label_j": spec.label_j,
                "merge_conflicts": json.dumps(spec.merge_conflicts, ensure_ascii=True),
                "constraint_notes": json.dumps(notes, ensure_ascii=True),
            }
        )
        flush_summary()

    if summary_rows:
        flush_summary()

    print(str(sweep_root))


if __name__ == "__main__":
    main()
