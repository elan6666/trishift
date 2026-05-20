from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.common.yaml_utils import load_yaml_file
from scripts.scpram._core.scpram_eval_core import run_scpram_eval


def main() -> None:
    ap = argparse.ArgumentParser(description="scPRAM dataset entry: scgen_pbmc_celltype")
    ap.add_argument("--seed", type=int, default=24)
    ap.add_argument("--splits", default="", help="comma-separated split ids; default uses dataset config")
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=0)
    ap.add_argument("--latent_dim", type=int, default=0)
    ap.add_argument("--hidden_dim", type=int, default=0)
    ap.add_argument("--lr", type=float, default=0.0)
    ap.add_argument("--fast", action="store_true", help="smoke test: split1, 1 epoch, no pkl export")
    ap.add_argument("--no_export_notebook_pkl", action="store_true")
    args = ap.parse_args()

    cfg = load_yaml_file(Path(__file__).with_name("config.yaml"))
    task = cfg.get("task_args") or {}
    split_ids = [int(x.strip()) for x in str(args.splits).split(",") if x.strip()] if str(args.splits).strip() else None
    run_scpram_eval(
        "scgen_pbmc_celltype",
        base_seed=int(task.get("seed", 24)) if int(args.seed) == 24 else int(args.seed),
        split_ids=[1] if bool(args.fast) else split_ids,
        epochs=1 if bool(args.fast) else (int(task.get("epochs", 100)) if int(args.epochs) == 0 else int(args.epochs)),
        batch_size=int(task.get("batch_size", 128)) if int(args.batch_size) == 0 else int(args.batch_size),
        latent_dim=int(task.get("latent_dim", 100)) if int(args.latent_dim) == 0 else int(args.latent_dim),
        hidden_dim=int(task.get("hidden_dim", 1000)) if int(args.hidden_dim) == 0 else int(args.hidden_dim),
        lr=float(task.get("lr", 5e-4)) if float(args.lr) == 0.0 else float(args.lr),
        export_notebook_pkl=False if bool(args.fast) or bool(args.no_export_notebook_pkl) else bool(task.get("export_notebook_pkl", True)),
    )


if __name__ == "__main__":
    main()
