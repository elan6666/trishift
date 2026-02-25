from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.common.yaml_utils import load_yaml_file
from scripts.systema._core import baselines_core


def main() -> None:
    ap = argparse.ArgumentParser(description="Systema baselines dataset entry: norman")
    ap.add_argument('--splits', default='')
    ap.add_argument('--out_root', default='')
    ap.add_argument('--align_metrics_csv', default='')
    ap.add_argument('--align_to_run', default='')
    ap.add_argument('--sweep_root', default='')
    ap.add_argument('--n_ensemble', type=int, default=300)
    ap.add_argument('--defaults', default='configs/defaults.yaml')
    ap.add_argument('--paths', default='configs/paths.yaml')
    args = ap.parse_args()

    cfg = load_yaml_file(Path(__file__).with_name('systema_baselines.yaml'))
    task = cfg.get('task_args') or {}
    splits_val = str(args.splits).strip() or ','.join([str(x) for x in list(task.get('splits', [1,2,3]))])
    out_root_val = str(args.out_root).strip() or str(task.get('out_root', 'artifacts/results'))
    align_metrics_csv_val = str(args.align_metrics_csv).strip() or ('' if task.get('align_metrics_csv') in (None, '') else str(task.get('align_metrics_csv')))
    align_to_run_val = str(args.align_to_run).strip() or ('' if task.get('align_to_run') in (None, '') else str(task.get('align_to_run')))
    sweep_root_val = str(args.sweep_root).strip() or ''

    argv = [
        '--dataset', 'norman',
        '--splits', splits_val,
        '--n_ensemble', str(int(args.n_ensemble)),
        '--defaults', str(args.defaults),
        '--paths', str(args.paths),
        '--out_root', out_root_val,
    ]
    if align_metrics_csv_val:
        argv += ['--align_metrics_csv', align_metrics_csv_val]
    if align_to_run_val:
        argv += ['--align_to_run', align_to_run_val]
    if sweep_root_val:
        argv += ['--sweep_root', sweep_root_val]

    baselines_core.main(argv)


if __name__ == "__main__":
    main()

