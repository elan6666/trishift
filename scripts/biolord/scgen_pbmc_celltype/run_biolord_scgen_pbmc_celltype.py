from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.biolord._core.biolord_eval_core import run_biolord_scgen_pbmc_celltype_eval
from scripts.common.yaml_utils import load_yaml_file


def main() -> None:
    ap = argparse.ArgumentParser(description="BioLORD dataset entry: scgen_pbmc_celltype")
    ap.add_argument("--seed", type=int, default=24)
    ap.add_argument("--splits", default="", help="comma-separated split ids; default uses dataset config")
    ap.add_argument("--prior_key", default="", help="override BioLORD ordered attribute prior key")
    ap.add_argument("--max_epochs", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=0)
    ap.add_argument("--n_latent", type=int, default=0)
    ap.add_argument("--attribute_width", type=int, default=0)
    ap.add_argument("--attribute_depth", type=int, default=0)
    ap.add_argument("--fast", action="store_true", help="smoke test: split1, 1 epoch, no pkl export")
    ap.add_argument("--no_export_notebook_pkl", action="store_true")
    args = ap.parse_args()

    cfg = load_yaml_file(Path(__file__).with_name("config.yaml"))
    task = cfg.get("task_args") or {}
    split_ids = [int(x.strip()) for x in str(args.splits).split(",") if x.strip()] if str(args.splits).strip() else None
    run_biolord_scgen_pbmc_celltype_eval(
        prior_key=str(args.prior_key).strip() or str(task.get("prior_key", "emb_scgen_ifnb1_zenodo_prott5")),
        base_seed=int(task.get("seed", 24)) if int(args.seed) == 24 else int(args.seed),
        split_ids=[1] if bool(args.fast) else split_ids,
        max_epochs=1 if bool(args.fast) else (int(task.get("max_epochs", 100)) if int(args.max_epochs) == 0 else int(args.max_epochs)),
        batch_size=int(task.get("batch_size", 64)) if int(args.batch_size) == 0 else int(args.batch_size),
        n_latent=int(task.get("n_latent", 32)) if int(args.n_latent) == 0 else int(args.n_latent),
        attribute_width=int(task.get("attribute_width", 64)) if int(args.attribute_width) == 0 else int(args.attribute_width),
        attribute_depth=int(task.get("attribute_depth", 6)) if int(args.attribute_depth) == 0 else int(args.attribute_depth),
        export_notebook_pkl=False if bool(args.fast) or bool(args.no_export_notebook_pkl) else bool(task.get("export_notebook_pkl", True)),
    )


if __name__ == "__main__":
    main()
