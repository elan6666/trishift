from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.genepert._core.genepert_eval_core import *  # noqa: F401,F403


if __name__ == "__main__":
    main()
