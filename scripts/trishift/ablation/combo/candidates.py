from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from scripts.trishift.adamson.ablation.compute_combo_candidates import *  # noqa: F401,F403


if __name__ == "__main__":
    main()

