from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_ROOT = ROOT / "external"
OVERLAY_ROOT = ROOT / "patches" / "external_overlays"


@dataclass(frozen=True)
class BaselineSpec:
    key: str
    target_rel: str
    source_candidates: tuple[str, ...]
    repo_url: str | None = None

    @property
    def target_name(self) -> str:
        return Path(self.target_rel).name


BASELINES: dict[str, BaselineSpec] = {
    "scgpt": BaselineSpec(
        key="scgpt",
        target_rel="scGPT-main",
        source_candidates=("scGPT-main", "scGPT"),
        repo_url="https://github.com/bowang-lab/scGPT.git",
    ),
    "gears": BaselineSpec(
        key="gears",
        target_rel="gears",
        source_candidates=("gears", "GEARS"),
        repo_url="https://github.com/snap-stanford/GEARS.git",
    ),
    "biolord": BaselineSpec(
        key="biolord",
        target_rel=str(Path("biolord") / "biolord-main"),
        source_candidates=(
            str(Path("biolord") / "biolord-main"),
            "biolord-main",
            "biolord",
        ),
        repo_url="https://github.com/nitzanlab/biolord.git",
    ),
    "genepert": BaselineSpec(
        key="genepert",
        target_rel="GenePert-main",
        source_candidates=("GenePert-main", "GenePert"),
        repo_url="https://github.com/zou-group/GenePert.git",
    ),
}


def _split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [x.strip().lower() for x in value.split(",") if x.strip()]


def _parse_repo_overrides(values: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in values:
        if "=" not in raw:
            raise ValueError(f"--repo-url must use NAME=URL form, got: {raw}")
        name, url = raw.split("=", 1)
        name = name.strip().lower()
        url = url.strip()
        if name not in BASELINES:
            raise ValueError(f"Unknown baseline in --repo-url: {name}")
        if not url:
            raise ValueError(f"Empty URL for --repo-url {name}")
        out[name] = url
    return out


def _select_baselines(only: str | None) -> list[BaselineSpec]:
    names = _split_csv(only) or list(BASELINES)
    unknown = sorted(set(names) - set(BASELINES))
    if unknown:
        raise ValueError(f"Unknown baseline(s): {unknown}; choices={sorted(BASELINES)}")
    return [BASELINES[name] for name in names]


def _remove_target(target: Path, dry_run: bool) -> None:
    if not target.exists():
        return
    print(f"[remove] {target}")
    if not dry_run:
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()


def _copy_tree(src: Path, dst: Path, dry_run: bool) -> None:
    print(f"[copy] {src} -> {dst}")
    if dry_run:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst, dirs_exist_ok=True)


def _clone_repo(url: str, dst: Path, dry_run: bool) -> None:
    print(f"[clone] {url} -> {dst}")
    if dry_run:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "1", url, str(dst)], check=True)


def _find_source(source_root: Path, spec: BaselineSpec) -> Path | None:
    for candidate in spec.source_candidates:
        p = source_root / candidate
        if p.exists():
            return p
    return None


def _ensure_external_source(
    spec: BaselineSpec,
    *,
    source_root: Path | None,
    repo_overrides: dict[str, str],
    force: bool,
    apply_overlays_only: bool,
    dry_run: bool,
) -> Path:
    target = EXTERNAL_ROOT / spec.target_rel

    if apply_overlays_only:
        if not target.exists():
            print(f"[skip] {spec.key}: target does not exist for overlay-only mode: {target}")
        return target

    if target.exists():
        if not force:
            print(f"[exists] {spec.key}: {target}")
            return target
        _remove_target(target, dry_run)

    if source_root is not None:
        src = _find_source(source_root, spec)
        if src is not None:
            _copy_tree(src, target, dry_run)
            return target
        print(
            f"[warn] {spec.key}: no local source found under {source_root}; "
            f"tried {list(spec.source_candidates)}"
        )

    repo_url = repo_overrides.get(spec.key, spec.repo_url)
    if repo_url:
        _clone_repo(repo_url, target, dry_run)
        return target

    print(
        f"[skip] {spec.key}: no source was copied and no repository URL is configured. "
        "Pass --source-root containing a local checkout, or --repo-url NAME=https://..."
    )
    return target


def _apply_overlay(overlay: Path, target: Path, dry_run: bool) -> None:
    if not overlay.exists():
        return
    if not target.exists():
        print(f"[skip-overlay] target missing: {target}")
        return
    for src_file in sorted(p for p in overlay.rglob("*") if p.is_file()):
        rel = src_file.relative_to(overlay)
        dst_file = target / rel
        print(f"[overlay] {src_file.relative_to(ROOT)} -> {dst_file.relative_to(ROOT)}")
        if dry_run:
            continue
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_file, dst_file)


def _overlay_for_spec(spec: BaselineSpec) -> Path:
    return OVERLAY_ROOT / spec.target_name


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Populate ignored external baseline repositories and apply the tracked "
            "TriShift compatibility overlays."
        )
    )
    parser.add_argument(
        "--only",
        help="Comma-separated subset: scgpt,gears,biolord,genepert. Defaults to all.",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        help="Optional directory containing already downloaded baseline repositories.",
    )
    parser.add_argument(
        "--repo-url",
        action="append",
        default=[],
        help="Override clone URL using NAME=URL, for example genepert=https://...",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing external target before copying or cloning.",
    )
    parser.add_argument(
        "--apply-overlays-only",
        action="store_true",
        help="Do not copy or clone sources; only apply tracked overlays to existing external targets.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned actions without changing files.",
    )
    args = parser.parse_args(argv)

    try:
        specs = _select_baselines(args.only)
        repo_overrides = _parse_repo_overrides(args.repo_url)
    except ValueError as exc:
        parser.error(str(exc))

    print(f"[root] {ROOT}")
    print(f"[external] {EXTERNAL_ROOT}")
    print(f"[overlays] {OVERLAY_ROOT}")

    if not args.dry_run:
        EXTERNAL_ROOT.mkdir(parents=True, exist_ok=True)

    for spec in specs:
        target = _ensure_external_source(
            spec,
            source_root=args.source_root.resolve() if args.source_root else None,
            repo_overrides=repo_overrides,
            force=args.force,
            apply_overlays_only=args.apply_overlays_only,
            dry_run=args.dry_run,
        )
        _apply_overlay(_overlay_for_spec(spec), target, args.dry_run)

    print("[done] external baseline bootstrap complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
