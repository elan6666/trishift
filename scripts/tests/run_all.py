from __future__ import annotations

from pathlib import Path
import subprocess
import sys


def main() -> int:
    root = Path(__file__).resolve().parent
    tests = [
        "test_utils.py",
        "test_data.py",
        "test_model.py",
        "test_train.py",
        "test_eval.py",
    ]
    for name in tests:
        path = root / name
        print(f"[run] {path}")
        result = subprocess.run([sys.executable, str(path)], check=False)
        if result.returncode != 0:
            return result.returncode
    print("ALL TESTS PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
