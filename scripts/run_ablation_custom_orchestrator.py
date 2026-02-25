from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.trishift.ablation.single_factor.custom_orchestrator import *  # noqa: F401,F403


if __name__ == "__main__":
    main()

