from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.biolord._core.biolord_eval_core import run_biolord_eval
from scripts.common.yaml_utils import load_yaml_file


def main() -> None:
    ap = argparse.ArgumentParser(description="biolord dataset entry: norman")
    ap.add_argument("--seed", type=int, default=24)
    ap.add_argument("--raw_result_root", default="")
    ap.add_argument("--no_export_notebook_pkl", action="store_true")
    args = ap.parse_args()

    cfg = load_yaml_file(Path(__file__).with_name("config.yaml"))
    task = cfg.get("task_args") or {}
    seed = int(task.get("seed", 24)) if int(args.seed) == 24 else int(args.seed)
    export_notebook_pkl = bool(task.get("export_notebook_pkl", True))
    if bool(args.no_export_notebook_pkl):
        export_notebook_pkl = False
    raw_result_root = str(args.raw_result_root).strip() or str(task.get("raw_result_root", "")).strip() or None

    run_biolord_eval("norman", base_seed=seed, export_notebook_pkl=export_notebook_pkl, raw_result_root=raw_result_root)


if __name__ == "__main__":
    main()

