from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts.common.yaml_utils import load_yaml_file, merged_dict


DATASET_CONFIG = {
    "adamson": {"emb_key": "emb_b", "multi_split_default": 5, "test_ratio": 0.2},
    "dixit": {"emb_key": "emb_c", "multi_split_default": 10, "test_ratio": 0.2},
    "norman": {"emb_key": "emb_a", "multi_split_default": 5, "test_ratio": 0.2},
    "scgen_pbmc_celltype": {
        "emb_key": "emb_scgen_ifnb1_zenodo_prott5",
        "multi_split_default": 5,
        "test_ratio": 0.2,
    },
}

DATASET_CONFIG_PATHS = {
    "adamson": REPO_ROOT / "scripts" / "trishift" / "adamson" / "config.yaml",
    "dixit": REPO_ROOT / "scripts" / "trishift" / "dixit" / "config.yaml",
    "norman": REPO_ROOT / "scripts" / "trishift" / "norman" / "config.yaml",
    "scgen_pbmc_celltype": REPO_ROOT / "scripts" / "trishift" / "scgen_pbmc_celltype" / "config.yaml",
}


def _resolve_repo_path(path_value: str | Path) -> str:
    path = Path(str(path_value))
    if path.is_absolute():
        return str(path)
    return str((REPO_ROOT / path).resolve())


def _trishift_unseen_ctrl_split(data: Any, *, dataset: str, split_id: int, defaults: dict) -> dict:
    # Import the private splitter lazily so dry-run planning does not require a
    # full torch-enabled TriShift training environment.
    from scripts.trishift._core.run_dataset_core import _split_unseen_ctrl_unseen_perturbation

    subgroup_df = None
    test_conds = None
    val_conds = None
    if dataset == "norman":
        from scripts.common.split_utils import norman_subgroup

        subgroup_df = norman_subgroup(
            list(data.adata_all.obs[data.label_key].astype(str).unique()),
            seed=int(split_id),
        )
        test_conds = list(subgroup_df[subgroup_df.group == "test"].index)
        val_conds = list(subgroup_df[subgroup_df.group == "val"].index)
    return _split_unseen_ctrl_unseen_perturbation(
        data,
        seed=int(split_id),
        test_ratio=float(DATASET_CONFIG[dataset].get("test_ratio", 0.2)),
        val_ratio=float(defaults.get("unseen_ctrl_pert_val_ratio", 0.1)),
        ctrl_test_ratio=float(defaults.get("unseen_ctrl_test_ratio", 0.2)),
        ctrl_val_ratio=float(defaults.get("unseen_ctrl_val_ratio", 0.1)),
        test_conds=test_conds,
        val_conds=val_conds,
    )


def _sha1_list(values: list[str]) -> str:
    raw = "\n".join(str(x) for x in values)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _obs_names(adata) -> list[str]:
    return [str(x) for x in adata.obs_names.tolist()]


def _load_profile_config(config_path: Path) -> tuple[str, dict, dict, dict]:
    profile = load_yaml_file(config_path)
    dataset = str(profile.get("dataset", "")).strip().lower()
    if dataset not in DATASET_CONFIG:
        raise ValueError(f"Unsupported dataset for CellOT held-out setup: {dataset}")
    base = profile.get("base") or {}
    defaults_base = load_yaml_file(REPO_ROOT / str(base.get("defaults", "configs/defaults.yaml")))
    paths_base = load_yaml_file(REPO_ROOT / str(base.get("paths", "configs/paths.yaml")))
    defaults = merged_dict(defaults_base, profile.get("defaults_overrides") or {})
    paths = merged_dict(paths_base, profile.get("paths_overrides") or {})
    return dataset, defaults, paths, profile


def _split_ids(defaults: dict, dataset: str, fast: bool) -> list[int]:
    run_cfg = defaults.get("run", {}) or {}
    raw = run_cfg.get("split_ids")
    if raw:
        out = []
        for item in raw:
            val = int(item)
            if val > 0 and val not in out:
                out.append(val)
        return out[:1] if fast else out
    n_splits = (
        int(run_cfg.get("n_splits", DATASET_CONFIG[dataset]["multi_split_default"]))
        if bool(run_cfg.get("multi_split", False))
        else 1
    )
    out = list(range(1, n_splits + 1))
    return out[:1] if fast else out


def _load_trishift_data(dataset: str, defaults: dict, paths: dict) -> Any:
    from trishift.TriShiftData import TriShiftData
    from trishift._utils import apply_alias_mapping, load_adata, load_embedding_df

    h5ad_path = _resolve_repo_path(paths["datasets"][dataset])
    emb_key = str(defaults.get("emb_key", DATASET_CONFIG[dataset]["emb_key"]))
    if emb_key not in paths.get("embeddings", {}):
        raise ValueError(f"Embedding key {emb_key!r} missing from paths config")
    emb_path = _resolve_repo_path(paths["embeddings"][emb_key])
    adata = load_adata(h5ad_path)
    adata.uns = {}
    embd_df = apply_alias_mapping(load_embedding_df(emb_path), dataset)
    data = TriShiftData(adata, embd_df)
    data.setup_embedding_index()
    return data


def _cellot_task_config(*, h5ad_path: Path, target: str, batch_size: int, random_state: int) -> dict[str, Any]:
    return {
        "data": {
            "type": "cell",
            "source": "ctrl",
            "target": str(target),
            "condition": "cellot_condition",
            "path": str(h5ad_path),
        },
        "dataloader": {
            "batch_size": int(batch_size),
            "shuffle": True,
        },
        "datasplit": {
            "name": "trishift_fixed",
            "groupby": "cellot_condition",
            "random_state": int(random_state),
        },
    }


def _cellot_model_config() -> dict[str, Any]:
    return {
        "model": {
            "name": "cellot",
            "hidden_units": [64, 64, 64, 64],
            "latent_dim": 50,
            "softplus_W_kernels": False,
            "g": {"fnorm_penalty": 1},
            "kernel_init_fxn": {"b": 0.1, "name": "uniform"},
        },
        "optim": {
            "optimizer": "Adam",
            "lr": 0.0001,
            "beta1": 0.5,
            "beta2": 0.9,
            "weight_decay": 0,
        },
        "training": {
            "n_iters": 100000,
            "n_inner_iters": 10,
            "cache_freq": 1000,
            "eval_freq": 250,
            "logs_freq": 50,
        },
    }


def _write_yaml(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(obj, sort_keys=False, allow_unicode=False), encoding="utf-8")


def _sanitize_cellot_obs(adata: ad.AnnData, label_key: str) -> ad.AnnData:
    keep = []
    for col in [label_key, "condition", "cell_type", "split", "transport", "cellot_condition"]:
        if col in adata.obs.columns and col not in keep:
            keep.append(col)
    adata.obs = adata.obs.loc[:, keep].copy()
    for col in adata.obs.columns:
        adata.obs[col] = adata.obs[col].astype(str).astype("category")
    adata.uns = {}
    return adata


def _materialize_split_h5ad(
    *,
    data: TriShiftData,
    split_dict: dict,
    target_condition: str,
    out_path: Path,
) -> dict[str, Any]:
    label_key = data.label_key
    ctrl_label = data.ctrl_label
    train = split_dict["train"]
    test = split_dict["test"]

    train_obs_cond = train.obs[label_key].astype(str)
    test_obs_cond = test.obs[label_key].astype(str)
    train_ctrl = train[train_obs_cond == ctrl_label].copy()
    train_target = train[train_obs_cond == str(target_condition)].copy()
    test_ctrl = test[test_obs_cond == ctrl_label].copy()
    test_target = test[test_obs_cond == str(target_condition)].copy()

    parts = []
    for split_name, transport, adata_part in (
        ("train", "source", train_ctrl),
        ("train", "target", train_target),
        ("test", "source", test_ctrl),
        ("test", "target", test_target),
    ):
        if adata_part.n_obs == 0:
            continue
        adata_part.obs["split"] = split_name
        adata_part.obs["transport"] = transport
        adata_part.obs["cellot_condition"] = (
            ctrl_label if transport == "source" else str(target_condition)
        )
        parts.append(_sanitize_cellot_obs(adata_part, label_key))
    if not parts:
        raise ValueError(f"No cells available for CellOT target {target_condition}")
    import anndata as ad

    combined = ad.concat(parts, axis=0, join="outer", merge="same", index_unique=None)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.write_h5ad(out_path)
    return {
        "n_obs": int(combined.n_obs),
        "n_vars": int(combined.n_vars),
        "train_source_n": int(train_ctrl.n_obs),
        "train_target_n": int(train_target.n_obs),
        "test_source_n": int(test_ctrl.n_obs),
        "test_target_n": int(test_target.n_obs),
    }


def build_plan(
    *,
    dataset: str,
    config_path: Path,
    out_root: Path,
    split_ids: list[int],
    materialize: bool,
    batch_size: int,
) -> dict[str, Any]:
    dataset_key, defaults, paths, profile = _load_profile_config(config_path)
    if dataset_key != dataset:
        raise ValueError(f"dataset mismatch: --dataset={dataset}, config={dataset_key}")
    out_root.mkdir(parents=True, exist_ok=True)

    data = _load_trishift_data(dataset, defaults, paths) if materialize else None
    rows: list[dict[str, Any]] = []
    commands: list[dict[str, str]] = []

    for split_id in split_ids:
        split_dict = None
        if data is not None:
            split_dict = _trishift_unseen_ctrl_split(
                data,
                dataset=dataset,
                split_id=int(split_id),
                defaults=defaults,
            )
            target_conditions = [str(x) for x in split_dict.get("test_conds", [])]
            split_meta = {
                "train_conds": [str(x) for x in split_dict.get("train_conds", [])],
                "val_conds": [str(x) for x in split_dict.get("val_conds", [])],
                "test_conds": target_conditions,
                "train_ctrl_n": int(split_dict.get("train_ctrl_n", 0)),
                "val_ctrl_n": int(split_dict.get("val_ctrl_n", 0)),
                "test_ctrl_n": int(split_dict.get("test_ctrl_n", 0)),
                "train_obs_sha1": _sha1_list(_obs_names(split_dict["train"])),
                "val_obs_sha1": _sha1_list(_obs_names(split_dict["val"])),
                "test_obs_sha1": _sha1_list(_obs_names(split_dict["test"])),
            }
        else:
            target_conditions = []
            split_meta = {
                "train_conds": [],
                "val_conds": [],
                "test_conds": [],
                "note": "dry plan without --materialize; run on a machine with prepared data to enumerate target conditions",
            }

        split_dir = out_root / dataset / f"split{split_id}"
        split_dir.mkdir(parents=True, exist_ok=True)
        (split_dir / "split_manifest.json").write_text(
            json.dumps(
                {
                    "dataset": dataset,
                    "split_id": int(split_id),
                    "eval_mode": "unseen_ctrl_eval",
                    "split_alignment_required": True,
                    "trishift_config_path": str(config_path),
                    **split_meta,
                },
                indent=2,
                sort_keys=False,
            ),
            encoding="utf-8",
        )

        if not target_conditions:
            rows.append(
                {
                    "dataset": dataset,
                    "split_id": int(split_id),
                    "condition": "",
                    "status": "needs_materialize",
                    "reason": "target conditions unavailable without prepared data",
                    "cellot_task_config": "",
                    "cellot_model_config": "",
                    "cellot_outdir": "",
                }
            )
            continue

        train_conds = set(split_meta["train_conds"])
        for condition in target_conditions:
            cond_dir = split_dir / condition.replace("/", "_")
            h5ad_path = cond_dir / "cellot_input.h5ad"
            task_path = cond_dir / "task.yaml"
            model_path = cond_dir / "model.yaml"
            outdir = cond_dir / "model-cellot"
            status = "ready"
            reason = ""
            cell_counts: dict[str, Any] = {}
            if condition not in train_conds:
                status = "requires_aligned_runner"
                reason = "use scripts/cellot/run_trishift_aligned.py to evaluate held-out targets with nearest training-condition maps"
            else:
                cell_counts = _materialize_split_h5ad(
                    data=data,
                    split_dict=split_dict,
                    target_condition=condition,
                    out_path=h5ad_path,
                )
                _write_yaml(task_path, _cellot_task_config(h5ad_path=h5ad_path, target=condition, batch_size=batch_size, random_state=split_id))
                _write_yaml(model_path, _cellot_model_config())
                commands.append(
                    {
                        "train": f"python external/cellot/scripts/train.py --outdir {outdir} --config {task_path} --config {model_path}",
                        "evaluate": f"python external/cellot/scripts/evaluate.py --outdir {outdir} --setting iid --where data_space --n_cells 100,250,500 --n_reps 10",
                    }
                )
            rows.append(
                {
                    "dataset": dataset,
                    "split_id": int(split_id),
                    "condition": condition,
                    "status": status,
                    "reason": reason,
                    "cellot_task_config": str(task_path if status == "ready" else ""),
                    "cellot_model_config": str(model_path if status == "ready" else ""),
                    "cellot_outdir": str(outdir if status == "ready" else ""),
                    **cell_counts,
                }
            )

    summary_path = out_root / dataset / "cellot_trishift_heldout_plan.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    commands_path = out_root / dataset / "cellot_commands.json"
    commands_path.write_text(json.dumps(commands, indent=2), encoding="utf-8")
    return {
        "dataset": dataset,
        "config_path": str(config_path),
        "out_root": str(out_root / dataset),
        "materialize": bool(materialize),
        "split_ids": [int(x) for x in split_ids],
        "summary_csv": str(summary_path),
        "commands_json": str(commands_path),
        "n_rows": int(len(rows)),
        "n_commands": int(len(commands)),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Prepare CellOT configs from TriShift held-out-control split manifests."
    )
    ap.add_argument("--dataset", choices=sorted(DATASET_CONFIG_PATHS), required=True)
    ap.add_argument("--config", default="", help="TriShift dataset config path; defaults to scripts/trishift/<dataset>/config.yaml")
    ap.add_argument("--out-root", default="artifacts/results/cellot/trishift_heldout")
    ap.add_argument("--split-id", action="append", type=int, default=[], help="Split id to prepare; may be repeated")
    ap.add_argument("--fast", action="store_true", help="Only prepare the first split id")
    ap.add_argument("--materialize", action="store_true", help="Load prepared data and write split h5ad/config files")
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args(argv)

    dataset = str(args.dataset).strip().lower()
    config_path = Path(args.config).resolve() if args.config else DATASET_CONFIG_PATHS[dataset]
    _, defaults, _, _ = _load_profile_config(config_path)
    split_ids = args.split_id or _split_ids(defaults, dataset, fast=bool(args.fast))
    if bool(args.fast):
        split_ids = split_ids[:1]
    result = build_plan(
        dataset=dataset,
        config_path=config_path,
        out_root=Path(args.out_root),
        split_ids=split_ids,
        materialize=bool(args.materialize),
        batch_size=int(args.batch_size),
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
