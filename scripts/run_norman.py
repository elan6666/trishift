from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_dataset import run_dataset


def main():
    run_dataset("norman")


if __name__ == "__main__":
    main()
