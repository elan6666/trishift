from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.common.yaml_utils import load_yaml_file
from scripts.scgpt._core.scgpt_eval_core import run_scgpt_eval


def main() -> None:
    ap = argparse.ArgumentParser(description="scGPT dataset entry: norman")
    ap.add_argument("--seed", type=int, default=24)
    ap.add_argument("--control_pool_size", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--eval_batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--early_stop", type=int, default=10)
    ap.add_argument("--no_export_notebook_pkl", action="store_true")
    args = ap.parse_args()

    cfg = load_yaml_file(Path(__file__).with_name("config.yaml"))
    task = cfg.get("task_args") or {}
    run_scgpt_eval(
        "norman",
        base_seed=int(task.get("seed", 24)) if int(args.seed) == 24 else int(args.seed),
        export_notebook_pkl=(
            False if bool(args.no_export_notebook_pkl) else bool(task.get("export_notebook_pkl", True))
        ),
        control_pool_size=(
            int(task.get("control_pool_size", 300))
            if int(args.control_pool_size) == 300
            else int(args.control_pool_size)
        ),
        epochs=int(task.get("epochs", 15)) if int(args.epochs) == 15 else int(args.epochs),
        batch_size=int(task.get("batch_size", 64)) if int(args.batch_size) == 64 else int(args.batch_size),
        eval_batch_size=(
            int(task.get("eval_batch_size", 64))
            if int(args.eval_batch_size) == 64
            else int(args.eval_batch_size)
        ),
        lr=float(task.get("lr", 1e-4)) if float(args.lr) == 1e-4 else float(args.lr),
        early_stop=int(task.get("early_stop", 10)) if int(args.early_stop) == 10 else int(args.early_stop),
    )


if __name__ == "__main__":
    main()
