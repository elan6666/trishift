from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts.common.payload_subset import shrink_payload_file


PKL_RE = re.compile(
    r"^(?P<model>trishift|scgpt|gears|genepert|biolord)_(?P<dataset>.+)_(?P<split>\d+)_unseen_ctrl\.pkl$"
)


def _iter_unseen_pkls(root: Path, models: set[str]) -> list[Path]:
    out: list[Path] = []
    for path in root.rglob("*_unseen_ctrl.pkl"):
        match = PKL_RE.match(path.name)
        if not match:
            continue
        model = match.group("model")
        if models and model not in models:
            continue
        out.append(path)
    return sorted(out, key=lambda p: str(p))


def _parse_models(raw: str) -> set[str]:
    text = str(raw).strip()
    if not text or text.lower() == "all":
        return set()
    return {x.strip().lower() for x in text.split(",") if x.strip()}


def _already_subset(path: Path, sample_size: int) -> bool:
    # Cheap size-based skip is intentionally not used. Some BioLORD files have
    # pred/ctrl subsetted but truth full-sized, so they still need rewriting.
    return False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Shrink unseen-control notebook PKLs by capping Pred/Pred_full only; "
            "Ctrl/Ctrl_full and Truth/Truth_full are preserved when present."
        )
    )
    ap.add_argument("--root", default="artifacts/results", help="result root to scan")
    ap.add_argument("--models", default="scgpt,gears,genepert,biolord", help="comma-separated models or all")
    ap.add_argument("--sample_size", type=int, default=300)
    ap.add_argument("--in-place", action="store_true", help="replace each input PKL atomically")
    ap.add_argument("--suffix", default="_subset300", help="output suffix when not using --in-place")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    root = Path(args.root).resolve()
    models = _parse_models(args.models)
    paths = _iter_unseen_pkls(root, models)
    print(f"[shrink] root={root}")
    print(f"[shrink] models={sorted(models) if models else 'all'} sample_size={int(args.sample_size)}")
    print(f"[shrink] files={len(paths)}")
    if args.dry_run:
        for path in paths:
            print(f"[dry-run] {path} size_mb={path.stat().st_size / 1024**2:.1f}")
        return 0

    if not args.in_place and not str(args.suffix):
        raise ValueError("--suffix must be non-empty unless --in-place is used")

    for idx, src in enumerate(paths, start=1):
        match = PKL_RE.match(src.name)
        if match is None:
            continue
        model = match.group("model")
        dataset = match.group("dataset")
        split_id = int(match.group("split"))
        before = src.stat().st_size
        if _already_subset(src, int(args.sample_size)):
            print(f"[shrink] skip already subset: {src}")
            continue
        if args.in_place:
            fd, tmp_name = tempfile.mkstemp(prefix=src.stem + ".", suffix=".tmp", dir=str(src.parent))
            os.close(fd)
            dst = Path(tmp_name)
        else:
            dst = src.with_name(f"{src.stem}{args.suffix}{src.suffix}")
        print(f"[shrink] {idx}/{len(paths)} {src} before_mb={before / 1024**2:.1f}", flush=True)
        try:
            info = shrink_payload_file(
                src,
                dst,
                model_name=model,
                dataset=dataset,
                split_id=split_id,
                sample_size=int(args.sample_size),
            )
            after = dst.stat().st_size
            if args.in_place:
                os.replace(dst, src)
                after = src.stat().st_size
            print(
                f"[shrink] done {src.name} conditions={info['n_conditions']} "
                f"after_mb={after / 1024**2:.1f} ratio={after / before:.3f}",
                flush=True,
            )
        except Exception:
            if args.in_place and dst.exists():
                dst.unlink()
            raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
