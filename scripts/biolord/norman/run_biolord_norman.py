from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.biolord._core.biolord_eval_core import run_biolord_unseen_ctrl_eval


def main() -> None:
    ap = argparse.ArgumentParser(description="BioLORD unseen-control eval entry: norman")
    ap.add_argument("--unseen_ctrl_eval", action="store_true", help="accepted for parity with other baselines")
    ap.add_argument("--source_dir", default="", help="directory containing BioLORD pkl payloads")
    ap.add_argument("--splits", default="", help="comma-separated split ids; default uses dataset config")
    args = ap.parse_args()
    split_ids = [int(x.strip()) for x in str(args.splits).split(",") if x.strip()] if str(args.splits).strip() else None
    run_biolord_unseen_ctrl_eval(
        "norman",
        source_dir=(args.source_dir.strip() or None),
        split_ids=split_ids,
    )


if __name__ == "__main__":
    main()
