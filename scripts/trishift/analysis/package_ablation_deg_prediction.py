from __future__ import annotations

import argparse
import json
import sys
import tarfile
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from scripts.common.time_utils import ts_local
from scripts.trishift.analysis.run_ablation_deg_prediction import (
    DEFAULT_PRESETS,
    _parse_csv,
    _parse_split_map,
    run_ablation_deg_prediction,
)


def _arcname(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return path.name


def _iter_package_files(out_root: Path, include_gene_scores: bool) -> list[Path]:
    keep_names = {
        "ablation_deg_prediction_all_long.csv",
        "ablation_deg_prediction_all_summary.csv",
        "ablation_deg_prediction_manifest.csv",
        "deg_prediction_long.csv",
        "deg_prediction_per_condition.csv",
        "deg_prediction_summary.csv",
        "provenance.json",
        "package_manifest.json",
    }
    files: list[Path] = []
    for path in sorted(out_root.rglob("*")):
        if not path.is_file():
            continue
        if path.name == "deg_prediction_gene_scores.csv" and not include_gene_scores:
            continue
        if include_gene_scores or path.name in keep_names:
            files.append(path)
    return files


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Compute completed ablation DEG metrics and package lightweight recovery artifacts."
    )
    ap.add_argument("--datasets", default="adamson,dixit,norman,scgen_pbmc_celltype")
    ap.add_argument("--presets", default=",".join(DEFAULT_PRESETS))
    ap.add_argument("--split-map", default="")
    ap.add_argument("--result-root", default=str(REPO_ROOT / "artifacts" / "results" / "ablation"))
    ap.add_argument("--out-root", default=str(REPO_ROOT / "artifacts" / "analysis" / "deg_prediction" / "ablation"))
    ap.add_argument("--package", default=str(REPO_ROOT / "artifacts" / "analysis" / "deg_prediction" / "ablation_deg_prediction_recovery.tgz"))
    ap.add_argument("--label-top-k", type=int, default=100)
    ap.add_argument("--overlap-ks", default="20,50")
    ap.add_argument("--include-gene-scores", action="store_true")
    ap.add_argument("--allow-empty", action="store_true")
    args = ap.parse_args(argv)

    datasets = _parse_csv(args.datasets)
    presets = _parse_csv(args.presets)
    split_map = _parse_split_map(str(args.split_map))
    out_root = Path(args.out_root).resolve()
    out_path = run_ablation_deg_prediction(
        datasets=datasets,
        presets=presets,
        split_map=split_map,
        result_root=Path(args.result_root).resolve(),
        out_root=out_root,
        label_top_k=int(args.label_top_k),
        overlap_ks=[int(x) for x in _parse_csv(args.overlap_ks)],
    )
    long_df = pd.read_csv(out_path) if out_path.exists() and out_path.stat().st_size else pd.DataFrame()
    package_path = Path(args.package).resolve()
    package_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "generated_at": ts_local(),
        "repo_root": str(REPO_ROOT),
        "result_root": str(Path(args.result_root).resolve()),
        "out_root": str(out_root),
        "datasets": datasets,
        "presets": presets,
        "label_top_k": int(args.label_top_k),
        "overlap_ks": [int(x) for x in _parse_csv(args.overlap_ks)],
        "row_count": int(len(long_df)),
        "package": str(package_path),
        "include_gene_scores": bool(args.include_gene_scores),
    }
    manifest_path = out_root / "package_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    if long_df.empty and not bool(args.allow_empty):
        print(f"error: no ablation DEG rows were produced; see {out_root}", file=sys.stderr)
        return 2

    with tarfile.open(package_path, "w:gz") as tar:
        for path in _iter_package_files(out_root, include_gene_scores=bool(args.include_gene_scores)):
            tar.add(path, arcname=_arcname(path))
    print(f"long: {out_path}")
    print(f"package: {package_path}")
    print(f"rows: {len(long_df)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
