from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts.trishift._core.train_main_core import run_from_dataset_config


def main() -> None:
    config_path = Path(__file__).with_name("config.yaml")
    out_dir = REPO_ROOT / "artifacts" / "demo" / "adamson_mini"
    run_from_dataset_config(config_path, fast=True, out_dir=str(out_dir))
    print(f"[adamson-mini] finished. Outputs: {out_dir}")


if __name__ == "__main__":
    main()
