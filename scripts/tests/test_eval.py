from __future__ import annotations

from pathlib import Path

import torch

import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.tests._helpers import make_data_and_model, train_stage1_and_cache, temp_dir


def test_evaluate_and_export():
    data, model = make_data_and_model()
    train_stage1_and_cache(model, data)
    emb_table = torch.tensor(data.embd_df.values, dtype=torch.float32)
    split = data.split_by_condition(seed=4, test_ratio=0.2, val_ratio=0.2)

    df = model.evaluate(
        split_dict=split,
        emb_table=emb_table,
        split_id=1,
        n_ensemble=10,
        base_seed=24,
    )
    assert df.shape[0] > 0
    assert "nmse" in df.columns
    assert "pearson" in df.columns

    with temp_dir() as td:
        out_path = Path(td) / "preds.pkl"
        preds = model.export_predictions(
            split_dict=split,
            emb_table=emb_table,
            split_id=1,
            n_ensemble=10,
            base_seed=24,
            out_path=str(out_path),
        )
        assert out_path.exists()
        assert isinstance(preds, dict)


def main():
    test_evaluate_and_export()
    print("test_eval: PASS")


if __name__ == "__main__":
    main()
