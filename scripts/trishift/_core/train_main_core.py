from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from scripts.common.time_utils import ts_local
from scripts.common.yaml_utils import dump_yaml, load_yaml_file, merged_dict
from scripts.trishift._core.run_dataset_core import run_dataset_with_paths


PROFILE_DIR = Path(__file__).resolve().parents[1] / "train" / "configs"
PROFILE_ALLOWED_KEYS = {"dataset", "base", "defaults_overrides", "paths_overrides"}


def _load_profile(profile: str) -> dict:
    path = PROFILE_DIR / f"{profile}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"TriShift profile not found: {path}")
    obj = load_yaml_file(path)
    unknown = sorted(set(obj.keys()) - PROFILE_ALLOWED_KEYS)
    if unknown:
        raise ValueError(f"Unknown keys in TriShift profile {path.name}: {unknown}")
    dataset = str(obj.get("dataset", "")).strip()
    if dataset != profile:
        raise ValueError(f"Profile dataset mismatch: file={path.name} dataset={dataset!r}")
    base = obj.get("base") or {}
    if not isinstance(base, dict):
        raise TypeError(f"profile.base must be a mapping: {path}")
    base_defaults = str(base.get("defaults", "configs/defaults.yaml"))
    base_paths = str(base.get("paths", "configs/paths.yaml"))
    defaults_overrides = obj.get("defaults_overrides") or {}
    paths_overrides = obj.get("paths_overrides") or {}
    if not isinstance(defaults_overrides, dict) or not isinstance(paths_overrides, dict):
        raise TypeError(f"defaults_overrides/paths_overrides must be mappings: {path}")
    return {
        "dataset": dataset,
        "base_defaults": base_defaults,
        "base_paths": base_paths,
        "defaults_overrides": defaults_overrides,
        "paths_overrides": paths_overrides,
    }


def _materialize_profile_yaml(profile_name: str, profile_obj: dict) -> tuple[Path, Path]:
    defaults_base = load_yaml_file(REPO_ROOT / profile_obj["base_defaults"])
    paths_base = load_yaml_file(REPO_ROOT / profile_obj["base_paths"])
    defaults_merged = merged_dict(defaults_base, profile_obj["defaults_overrides"])
    paths_merged = merged_dict(paths_base, profile_obj["paths_overrides"])

    tmp_dir = REPO_ROOT / "artifacts" / "cache" / "profile_overlays" / "trishift" / f"{profile_name}_{ts_local()}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    defaults_path = tmp_dir / "defaults.yaml"
    paths_path = tmp_dir / "paths.yaml"
    dump_yaml(defaults_path, defaults_merged, allow_unicode=False)
    dump_yaml(paths_path, paths_merged, allow_unicode=False)
    return defaults_path, paths_path


def run_profile(profile: str, *, fast: bool = False, out_dir: str | None = None) -> None:
    prof = _load_profile(profile)
    defaults_path, paths_path = _materialize_profile_yaml(profile, prof)
    run_dataset_with_paths(
        name=prof["dataset"],
        fast=bool(fast),
        defaults_path=str(defaults_path),
        paths_path=str(paths_path),
        out_dir=out_dir,
    )


def run_from_dataset_config(
    config_path: str | Path,
    *,
    fast: bool = False,
    out_dir: str | None = None,
) -> None:
    path = Path(config_path).resolve()
    obj = load_yaml_file(path)
    unknown = sorted(set(obj.keys()) - PROFILE_ALLOWED_KEYS)
    if unknown:
        raise ValueError(f"Unknown keys in dataset config {path.name}: {unknown}")
    dataset = str(obj.get("dataset", "")).strip()
    if not dataset:
        raise ValueError(f"dataset is required in {path}")
    base = obj.get("base") or {}
    if not isinstance(base, dict):
        raise TypeError(f"base must be a mapping: {path}")
    prof = {
        "dataset": dataset,
        "base_defaults": str(base.get("defaults", "configs/defaults.yaml")),
        "base_paths": str(base.get("paths", "configs/paths.yaml")),
        "defaults_overrides": obj.get("defaults_overrides") or {},
        "paths_overrides": obj.get("paths_overrides") or {},
    }
    defaults_path, paths_path = _materialize_profile_yaml(dataset, prof)
    run_dataset_with_paths(
        name=dataset,
        fast=bool(fast),
        defaults_path=str(defaults_path),
        paths_path=str(paths_path),
        out_dir=out_dir,
    )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="TriShift official training entry (profile or legacy mode)")
    ap.add_argument("--profile", default="", help="dataset profile name under scripts/trishift/train/configs")
    ap.add_argument("--name", default="", help="legacy dataset name (compat mode)")
    ap.add_argument("--fast", action="store_true", help="use minimal epochs/splits")
    ap.add_argument("--defaults", default="configs/defaults.yaml", help="legacy defaults yaml path")
    ap.add_argument("--paths", default="configs/paths.yaml", help="legacy paths yaml path")
    ap.add_argument("--out_dir", default="", help="output directory override")
    args = ap.parse_args(argv)

    out_dir = args.out_dir.strip() or None
    profile = str(args.profile).strip()
    if profile:
        run_profile(profile, fast=bool(args.fast), out_dir=out_dir)
        return

    name = str(args.name).strip()
    if not name:
        raise ValueError("Either --profile or --name must be provided")

    run_dataset_with_paths(
        name=name,
        fast=bool(args.fast),
        defaults_path=str(args.defaults),
        paths_path=str(args.paths),
        out_dir=out_dir,
    )


if __name__ == "__main__":
    main()
