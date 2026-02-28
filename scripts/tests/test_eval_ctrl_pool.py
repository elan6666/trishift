from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.tests._helpers import make_data_and_model, train_stage1_and_cache


def test_eval_ctrl_pool_flatten_without_dedup():
    data, model = make_data_and_model()
    topk_map = np.asarray(
        [
            [1, 2, 1],
            [3, 3, 4],
            [5, 6, 5],
        ],
        dtype=int,
    )
    rows_by_cond = {"A": np.asarray([0, 2], dtype=int)}
    pool = model._build_eval_ctrl_pool_from_topk("A", topk_map, rows_by_cond)
    assert pool.tolist() == [1, 2, 1, 5, 6, 5]


def test_nearest_genept_ot_pool_eval_and_export():
    data, model = make_data_and_model(seed=1)
    train_stage1_and_cache(model, data)
    emb_table = torch.tensor(data.embd_df.values, dtype=torch.float32)
    split = data.split_by_condition(seed=4, test_ratio=0.2, val_ratio=0.2)

    train_split = split["train"]
    train_cond = train_split.obs[data.label_key].astype(str).values
    train_pert = train_split[train_cond != data.ctrl_label]
    _, ctrl_global_idx = model._get_ctrl_pool_from_split(train_split)
    topk_map = data.build_or_load_topk_map(
        split_adata=train_pert,
        mode="knn",
        k=3,
        seed=1,
        candidates=5,
        cache_path=None,
        ctrl_global_indices=ctrl_global_idx,
    )

    strategy = model.build_eval_ctrl_strategy(
        split_dict=split,
        emb_table=emb_table,
        topk_map=topk_map,
        distance_metric="cosine",
        sample_size=3,
    )
    assert strategy["mode"] == "nearest_genept_ot_pool"
    assert strategy["distance_metric"] == "cosine"
    assert strategy["sample_size"] == 3

    df = model.evaluate(
        split_dict=split,
        emb_table=emb_table,
        split_id=1,
        n_ensemble=10,
        base_seed=24,
        eval_ctrl_strategy=strategy,
    )
    assert df.shape[0] > 0
    assert set(df["n_ensemble"].astype(int).unique().tolist()) == {3}

    preds = model.export_predictions(
        split_dict=split,
        emb_table=emb_table,
        split_id=1,
        n_ensemble=10,
        base_seed=24,
        eval_ctrl_strategy=strategy,
        out_path=None,
    )
    assert isinstance(preds, dict) and preds
    for payload in preds.values():
        assert payload["Pred"].shape[0] == 3
        assert payload["Ctrl"].shape[0] == 3


def main():
    test_eval_ctrl_pool_flatten_without_dedup()
    test_nearest_genept_ot_pool_eval_and_export()
    print("test_eval_ctrl_pool: PASS")


if __name__ == "__main__":
    main()

