from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.common.yaml_utils import load_yaml_file
from scripts.scouter._core.scouter_eval_core import run_scouter_eval


def main() -> None:
    ap = argparse.ArgumentParser(description="Scouter dataset entry: replogle_rpe1_essential")
    ap.add_argument('--n_ensemble', type=int, default=300)
    ap.add_argument('--seed', type=int, default=24)
    ap.add_argument('--no_export_notebook_pkl', action='store_true')
    args = ap.parse_args()

    cfg = load_yaml_file(Path(__file__).with_name('config.yaml'))
    task = cfg.get('task_args') or {}
    n_ensemble = int(task.get('n_ensemble', 300)) if int(args.n_ensemble) == 300 else int(args.n_ensemble)
    seed = int(task.get('seed', 24)) if int(args.seed) == 24 else int(args.seed)
    export_notebook_pkl = bool(task.get('export_notebook_pkl', True))
    if bool(args.no_export_notebook_pkl):
        export_notebook_pkl = False

    run_scouter_eval('replogle_rpe1_essential', n_ensemble=n_ensemble, base_seed=seed, export_notebook_pkl=export_notebook_pkl)


if __name__ == "__main__":
    main()

