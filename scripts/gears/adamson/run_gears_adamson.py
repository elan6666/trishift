from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.common.yaml_utils import load_yaml_file
from scripts.gears._core import gears_eval_core as gears_core
from scripts.gears._core.gears_eval_core import main as gears_core_main


def main() -> None:
    ap = argparse.ArgumentParser(description="GEARS dataset entry: adamson")
    ap.add_argument("--profile", default="", help="optional GEARS profile override")
    ap.add_argument("--seed", type=int, default=24)
    ap.add_argument("--no_export_notebook_pkl", action="store_true")
    args = ap.parse_args()

    profile = str(args.profile).strip()
    if profile:
        argv = ["--profile", profile, "--seed", str(int(args.seed))]
        if bool(args.no_export_notebook_pkl):
            argv.append("--no_export_notebook_pkl")
        gears_core_main(argv)
        return

    cfg = load_yaml_file(Path(__file__).with_name("config.yaml"))
    task = cfg.get("task_args") or {}
    seed = int(task.get("seed", 24)) if int(args.seed) == 24 else int(args.seed)
    export_notebook_pkl = bool(task.get("export_notebook_pkl", False))
    if bool(args.no_export_notebook_pkl):
        export_notebook_pkl = False

    gears_core.run_gears_eval(
        "adamson",
        base_seed=seed,
        export_notebook_pkl=export_notebook_pkl,
    )


if __name__ == "__main__":
    main()
