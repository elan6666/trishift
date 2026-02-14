from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from trishift import _utils


def test_normalize_condition():
    assert _utils.normalize_condition("B+A") == "A+B"
    assert _utils.normalize_condition("A") == "A"


def test_load_embedding_df(tmp_path: Path):
    obj = {"A": np.ones(3, dtype=np.float32), "B": np.zeros(3, dtype=np.float32)}
    path = tmp_path / "emb.pkl"
    with open(path, "wb") as handle:
        pickle.dump(obj, handle)
    df = _utils.load_embedding_df(str(path))
    assert "ctrl" in df.index
    assert df.shape[1] == 3


def main():
    tmp = Path(".tmp_test_utils")
    tmp.mkdir(exist_ok=True)
    test_normalize_condition()
    test_load_embedding_df(tmp)
    print("test_utils: PASS")


if __name__ == "__main__":
    main()
