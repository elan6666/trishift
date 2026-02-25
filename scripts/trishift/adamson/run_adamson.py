from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.trishift._core.train_main_core import run_from_dataset_config


def main() -> None:
    ap = argparse.ArgumentParser(description="TriShift dataset entry: adamson")
    ap.add_argument('--fast', action='store_true', help='use minimal epochs/splits')
    ap.add_argument('--out_dir', default='', help='output directory override')
    args = ap.parse_args()
    run_from_dataset_config(
        Path(__file__).with_name('config.yaml'),
        fast=bool(args.fast),
        out_dir=(args.out_dir.strip() or None),
    )


if __name__ == "__main__":
    main()

