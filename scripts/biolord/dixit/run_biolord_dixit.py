from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.biolord._core.biolord_eval_core import run_biolord_eval, run_biolord_unseen_ctrl_eval
from scripts.common.yaml_utils import load_yaml_file


def main() -> None:
    ap = argparse.ArgumentParser(description="BioLORD dataset entry: dixit")
    ap.add_argument("--seed", type=int, default=24)
    ap.add_argument("--splits", default="", help="comma-separated split ids; default uses dataset config")
    ap.add_argument("--max_epochs", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=0)
    ap.add_argument("--n_latent", type=int, default=0)
    ap.add_argument("--attribute_width", type=int, default=0)
    ap.add_argument("--attribute_depth", type=int, default=0)
    ap.add_argument("--export_control_pool_size", type=int, default=0)
    ap.add_argument("--no_export_notebook_pkl", action="store_true")
    ap.add_argument(
        "--unseen_ctrl_eval",
        action="store_true",
        help="run held-out ctrl/unseen perturbation evaluation without overwriting default metrics",
    )
    args = ap.parse_args()
    cfg = load_yaml_file(Path(__file__).with_name("config.yaml")) if Path(__file__).with_name("config.yaml").exists() else {}
    task = cfg.get("task_args") or {}
    split_ids = [int(x.strip()) for x in str(args.splits).split(",") if x.strip()] if str(args.splits).strip() else None
    kwargs = dict(
        base_seed=int(task.get("seed", 24)) if int(args.seed) == 24 else int(args.seed),
        export_notebook_pkl=(
            False if bool(args.no_export_notebook_pkl) else bool(task.get("export_notebook_pkl", True))
        ),
        split_ids=split_ids,
        max_epochs=(int(task.get("max_epochs", 0)) if int(args.max_epochs) == 0 else int(args.max_epochs)) or None,
        batch_size=(int(task.get("batch_size", 0)) if int(args.batch_size) == 0 else int(args.batch_size)) or None,
        n_latent=(int(task.get("n_latent", 0)) if int(args.n_latent) == 0 else int(args.n_latent)) or None,
        attribute_width=(
            int(task.get("attribute_width", 0)) if int(args.attribute_width) == 0 else int(args.attribute_width)
        )
        or None,
        attribute_depth=(
            int(task.get("attribute_depth", 0)) if int(args.attribute_depth) == 0 else int(args.attribute_depth)
        )
        or None,
        export_control_pool_size=(
            int(task.get("export_control_pool_size", 0))
            if int(args.export_control_pool_size) == 0
            else int(args.export_control_pool_size)
        )
        or None,
    )
    runner = run_biolord_unseen_ctrl_eval if bool(args.unseen_ctrl_eval) else run_biolord_eval
    runner(
        "dixit",
        **kwargs,
    )


if __name__ == "__main__":
    main()
