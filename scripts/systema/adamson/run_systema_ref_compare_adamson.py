from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.common.yaml_utils import load_yaml_file
from scripts.systema._core import ref_compare_core


def main() -> None:
    ap = argparse.ArgumentParser(description="Systema ref_compare dataset entry: adamson")
    ap.add_argument('--sweep-dir', required=False, default='')
    ap.add_argument('--out-root', default='')
    ap.add_argument('--splits', default='')
    ap.add_argument('--timestamp', default='')
    ap.add_argument('--write-doc', action='store_true')
    ap.add_argument('--no-umap', action='store_true')
    args = ap.parse_args()

    cfg = load_yaml_file(Path(__file__).with_name('systema_ref_compare.yaml'))
    task = cfg.get('task_args') or {}
    sweep_dir = str(args.sweep_dir).strip() or ('' if task.get('sweep_dir') in (None, '') else str(task.get('sweep_dir')))
    if not sweep_dir:
        raise ValueError('Provide --sweep-dir or set task_args.sweep_dir in systema_ref_compare.yaml')
    argv = ['--dataset', 'adamson', '--sweep-dir', sweep_dir]
    out_root = str(args.out_root).strip() or ('' if task.get('out_root') in (None, '') else str(task.get('out_root')))
    if out_root:
        argv += ['--out-root', out_root]
    splits = str(args.splits).strip() or (','.join([str(x) for x in list(task.get('splits', []))]) if task.get('splits') else '')
    if splits:
        argv += ['--splits', splits]
    if str(args.timestamp).strip():
        argv += ['--timestamp', str(args.timestamp).strip()]
    if bool(args.write_doc):
        argv += ['--write-doc']
    if bool(args.no_umap) or bool(task.get('umap', True)) is False:
        argv += ['--no-umap']
    ref_compare_core.main(argv)


if __name__ == "__main__":
    main()

