from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_dataset import run_dataset


def main():
    parser = argparse.ArgumentParser(description="Run TriShift on adamson")
    parser.add_argument("--fast", action="store_true", help="use minimal epochs/splits")
    args = parser.parse_args()
    run_dataset("adamson", fast=args.fast)


if __name__ == "__main__":
    main()
