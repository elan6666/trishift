from __future__ import annotations

import argparse
import glob
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml
from gears import GEARS, PertData

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
SRC_DATA_ROOT = REPO_ROOT / "src" / "data"
GEARS_DATA_ROOT = SRC_DATA_ROOT / "Data_GEARS"
DEFAULT_PATHS_YAML = REPO_ROOT / "configs" / "paths.yaml"


@dataclass(frozen=True)
class DatasetPlan:
    dataset_name: str
    split_seeds: tuple[int, ...]
    prepare_kwargs: dict
    initialize_gears_model: bool = False


DEFAULT_PLANS: dict[str, DatasetPlan] = {
    "adamson": DatasetPlan(
        dataset_name="adamson",
        split_seeds=(1, 2, 3, 4, 5),
        prepare_kwargs={},
    ),
    "dixit": DatasetPlan(
        dataset_name="dixit",
        split_seeds=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
        prepare_kwargs={"train_gene_set_size": 0.9},
        initialize_gears_model=True,
    ),
    "norman": DatasetPlan(
        dataset_name="norman",
        split_seeds=(1, 2, 3, 4, 5),
        prepare_kwargs={},
    ),
    "replogle_k562_essential": DatasetPlan(
        dataset_name="replogle_k562_essential",
        split_seeds=(1, 2, 3, 4, 5),
        prepare_kwargs={},
    ),
    "replogle_rpe1_essential": DatasetPlan(
        dataset_name="replogle_rpe1_essential",
        split_seeds=(1, 2, 3, 4, 5),
        prepare_kwargs={},
    ),
}


def _load_paths_yaml(paths_yaml: Path) -> dict:
    with paths_yaml.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    return cfg


def _resolve_repo_path(path_value: str | Path) -> Path:
    path = Path(str(path_value))
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def _expected_outer_h5ad(dataset_name: str, paths_yaml: Path) -> Path:
    cfg = _load_paths_yaml(paths_yaml)
    datasets_cfg = cfg.get("datasets", {})
    configured = datasets_cfg.get(dataset_name)
    if configured:
        return _resolve_repo_path(configured)
    return (SRC_DATA_ROOT / dataset_name / "perturb_processed.h5ad").resolve()


def _sync_processed_h5ad(dataset_name: str, paths_yaml: Path) -> Path:
    src = (GEARS_DATA_ROOT / dataset_name / "perturb_processed.h5ad").resolve()
    if not src.exists():
        raise FileNotFoundError(f"Missing generated GEARS h5ad: {src}")
    dst = _expected_outer_h5ad(dataset_name, paths_yaml)
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"[trishift-data] sync {src} -> {dst}")
    shutil.copy2(src, dst)
    return dst


def _cleanup_download_archives(root: Path) -> None:
    for pattern in ("*.zip", "*.tar.gz"):
        for file in glob.glob(pattern, root_dir=str(root)):
            path = root / file
            path.unlink(missing_ok=True)
            print(f"[trishift-data] deleted cached archive: {path}")


def _prepare_dataset(plan: DatasetPlan) -> None:
    print(f"[trishift-data] download/load dataset={plan.dataset_name}")
    pert_data = PertData(str(GEARS_DATA_ROOT))
    pert_data.load(data_name=plan.dataset_name)
    for split_seed in plan.split_seeds:
        kwargs = {"split": "simulation", "seed": int(split_seed), **plan.prepare_kwargs}
        print(
            f"[trishift-data] prepare_split dataset={plan.dataset_name} "
            f"seed={split_seed} kwargs={kwargs}"
        )
        pert_data.prepare_split(**kwargs)
    if plan.initialize_gears_model:
        print(f"[trishift-data] initialize GEARS backbone for dataset={plan.dataset_name}")
        pert_data.get_dataloader(batch_size=32, test_batch_size=32)
        gears_model = GEARS(
            pert_data,
            device="cpu",
            weight_bias_track=False,
            proj_name=plan.dataset_name,
        )
        gears_model.model_initialize(hidden_size=64)


def run(
    datasets: list[str],
    *,
    paths_yaml: Path = DEFAULT_PATHS_YAML,
    sync_outer_h5ad: bool = True,
    cleanup_archives: bool = True,
) -> None:
    unknown = [name for name in datasets if name not in DEFAULT_PLANS]
    if unknown:
        raise ValueError(
            "Unknown dataset(s): "
            + ", ".join(unknown)
            + f". Available: {', '.join(DEFAULT_PLANS)}"
        )

    GEARS_DATA_ROOT.mkdir(parents=True, exist_ok=True)
    cwd = Path.cwd()
    try:
        os.chdir(GEARS_DATA_ROOT)
        for dataset_name in datasets:
            _prepare_dataset(DEFAULT_PLANS[dataset_name])
            if sync_outer_h5ad:
                synced = _sync_processed_h5ad(dataset_name, paths_yaml)
                print(f"[trishift-data] ready dataset={dataset_name} h5ad={synced}")
        if cleanup_archives:
            _cleanup_download_archives(GEARS_DATA_ROOT)
    finally:
        os.chdir(cwd)


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Download and prepare public single-cell perturbation benchmark datasets "
            "for TriShift. This script delegates raw dataset download to GEARS/PertData, "
            "then synchronizes the processed perturb_processed.h5ad files to the "
            "paths expected by TriShift and baseline runners."
        )
    )
    ap.add_argument(
        "--datasets",
        nargs="+",
        default=["adamson", "dixit", "norman"],
        help=(
            "Datasets to prepare. Default: adamson dixit norman. "
            "Also supports replogle_k562_essential and replogle_rpe1_essential."
        ),
    )
    ap.add_argument(
        "--paths-yaml",
        default=str(DEFAULT_PATHS_YAML),
        help="paths.yaml used to resolve final synced h5ad targets",
    )
    ap.add_argument(
        "--no-sync",
        action="store_true",
        help="Only prepare GEARS-native folders under src/data/Data_GEARS and skip outer h5ad sync",
    )
    ap.add_argument(
        "--keep-archives",
        action="store_true",
        help="Keep downloaded zip/tar.gz archives under src/data/Data_GEARS",
    )
    return ap


def main() -> None:
    ap = build_argparser()
    args = ap.parse_args()
    run(
        datasets=[str(name) for name in args.datasets],
        paths_yaml=Path(args.paths_yaml).resolve(),
        sync_outer_h5ad=not bool(args.no_sync),
        cleanup_archives=not bool(args.keep_archives),
    )


if __name__ == "__main__":
    main()
