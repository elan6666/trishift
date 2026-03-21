from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from scripts.trishift.analysis.pathway_recovery import run_pathway_recovery


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pathway comparison across state/best/Adamson ablations")
    parser.add_argument("--ablation_root", required=True)
    parser.add_argument("--manifest", default="", help="optional explicit manifest path")
    parser.add_argument("--split_ids", default="1,2,3,4,5")
    parser.add_argument("--out_root", default="", help="default: <ablation_root>/pathway_compare")
    parser.add_argument("--truth_deg_mode", default="effect_size_non_dropout")
    parser.add_argument("--pred_deg_mode", default="effect_size_non_dropout")
    parser.add_argument("--enrichment_mode", default="run_if_available")
    parser.add_argument("--enrichment_library", default="Reactome_2022")
    parser.add_argument("--n_degs", type=int, default=100)
    parser.add_argument("--space", default="full_gene", choices=["auto", "full_gene", "deg"])
    args = parser.parse_args()

    ablation_root = Path(args.ablation_root).resolve()
    manifest_path = (
        Path(args.manifest).resolve()
        if str(args.manifest).strip()
        else ablation_root / "pathway_trishift_runs.json"
    )
    runs = json.loads(manifest_path.read_text(encoding="utf-8"))
    if isinstance(runs, dict):
        runs = runs.get("runs", [])
    out_root = (
        Path(args.out_root).resolve()
        if str(args.out_root).strip()
        else ablation_root / "pathway_compare"
    )
    result = run_pathway_recovery(
        dataset="adamson",
        split_ids=str(args.split_ids).strip(),
        out_root=out_root,
        truth_deg_mode=str(args.truth_deg_mode).strip(),
        pred_deg_mode=str(args.pred_deg_mode).strip(),
        enrichment_mode=str(args.enrichment_mode).strip(),
        enrichment_library=str(args.enrichment_library).strip(),
        n_degs=int(args.n_degs),
        space=str(args.space).strip(),
        trishift_runs=runs,
    )
    print(f"out_dir: {result['out_dir']}")


if __name__ == "__main__":
    main()
