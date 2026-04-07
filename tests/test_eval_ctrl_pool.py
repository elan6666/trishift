from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from tests._helpers import make_data_and_model, train_stage1_and_cache
from trishift.TriShift import TriShift
from trishift.TriShiftData import TriShiftData


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
        sample_size=10,
    )
    assert strategy["mode"] == "nearest_genept_ot_pool"
    assert strategy["distance_metric"] == "cosine"
    assert strategy["sample_size"] == 10
    assert strategy["compare_mode"] == "aggregate_cond"

    df = model.evaluate(
        split_dict=split,
        emb_table=emb_table,
        split_id=1,
        n_ensemble=10,
        base_seed=24,
        eval_ctrl_strategy=strategy,
    )
    assert df.shape[0] > 0
    assert set(df["n_ensemble"].astype(int).unique().tolist()) == {10}

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
        assert payload["Pred"].shape[0] == 10
        assert payload["Ctrl"].shape[0] == 10
        assert payload["Pred_full"].shape[0] == 10
        assert payload["Ctrl_full"].shape[0] == 10
        assert payload["Truth_full"].shape[1] == len(payload["gene_name_full"])
        np.testing.assert_allclose(
            payload["Pred_full"][:, payload["DE_idx"]],
            payload["Pred"],
        )

    # Fallback branch: missing pool entry should still use n_ensemble-sized random sampling.
    strategy_fallback = dict(strategy)
    strategy_fallback["pool_idx_by_test_cond"] = {}
    df_fb = model.evaluate(
        split_dict=split,
        emb_table=emb_table,
        split_id=1,
        n_ensemble=10,
        base_seed=24,
        eval_ctrl_strategy=strategy_fallback,
    )
    assert df_fb.shape[0] > 0
    assert set(df_fb["n_ensemble"].astype(int).unique().tolist()) == {10}


def test_condition_tokens_no_ctrl():
    data, model = make_data_and_model(seed=2)
    assert model._condition_tokens_no_ctrl("A+ctrl") == ["A"]
    assert model._condition_tokens_no_ctrl("ctrl+A") == ["A"]
    assert model._condition_tokens_no_ctrl("A+B") == ["A", "B"]


def test_per_gene_nearest_strategy_concat_without_dedup():
    X = np.random.RandomState(0).rand(12, 6).astype(np.float32)
    obs = pd.DataFrame(
        {
            "condition": (
                ["ctrl"] * 4
                + ["A+ctrl"] * 2
                + ["B+ctrl"] * 2
                + ["A+B"] * 4
            )
        }
    )
    var = pd.DataFrame({"gene_name": ["A", "B", "G1", "G2", "G3", "G4"]})
    adata = ad.AnnData(X=X, obs=obs, var=var)

    embd_df = pd.DataFrame(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        index=["ctrl", "A", "B"],
    )
    data = TriShiftData(adata, embd_df)
    data.setup_embedding_index()

    model = TriShift(data, device="cpu")
    model.model_init(
        x_dim=adata.n_vars,
        z_dim=4,
        cond_dim=embd_df.shape[1],
        vae_enc_hidden=[8],
        vae_dec_hidden=[8],
        shift_hidden=[8],
        gen_hidden=[8],
        dropout=0.0,
    )

    emb_table = torch.tensor(embd_df.values, dtype=torch.float32)
    train_mask = adata.obs["condition"].isin(["ctrl", "A+ctrl", "B+ctrl"]).values
    test_mask = adata.obs["condition"] == "A+B"
    split_dict = {
        "train": adata[train_mask].copy(),
        "test": adata[test_mask].copy(),
        "test_conds": ["A+B"],
    }
    # train pert rows order: A+ctrl (2 rows), B+ctrl (2 rows)
    topk_map = np.asarray(
        [
            [1, 1],
            [1, 2],
            [7, 7],
            [7, 8],
        ],
        dtype=int,
    )
    strategy = model.build_eval_ctrl_strategy(
        split_dict=split_dict,
        emb_table=emb_table,
        topk_map=topk_map,
        distance_metric="cosine",
        sample_size=5,
        compare_mode="per_gene_nearest_cond",
    )
    assert strategy["compare_mode"] == "per_gene_nearest_cond"
    assert strategy["nearest_train_conds_by_test_cond"]["A+B"] == ["A+ctrl", "B+ctrl"]
    pool = strategy["pool_idx_by_test_cond"]["A+B"]
    assert pool.tolist() == [1, 1, 1, 2, 7, 7, 7, 8]


def test_aggregate_cond_train_candidate_filter():
    X = np.random.RandomState(7).rand(14, 6).astype(np.float32)
    obs = pd.DataFrame(
        {
            "condition": (
                ["ctrl"] * 4
                + ["A+ctrl"] * 4
                + ["B+ctrl"] * 4
                + ["A+ctrl"] * 2
            )
        }
    )
    var = pd.DataFrame({"gene_name": ["A", "B", "G1", "G2", "G3", "G4"]})
    adata = ad.AnnData(X=X, obs=obs, var=var)
    embd_df = pd.DataFrame(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        index=["ctrl", "A", "B"],
    )
    data = TriShiftData(adata, embd_df)
    data.setup_embedding_index()

    model = TriShift(data, device="cpu")
    model.model_init(
        x_dim=adata.n_vars,
        z_dim=4,
        cond_dim=embd_df.shape[1],
        vae_enc_hidden=[8],
        vae_dec_hidden=[8],
        shift_hidden=[8],
        gen_hidden=[8],
        dropout=0.0,
    )
    emb_table = torch.tensor(embd_df.values, dtype=torch.float32)
    train_mask = adata.obs["condition"].isin(["ctrl", "A+ctrl", "B+ctrl"]).values
    test_mask = adata.obs["condition"] == "A+ctrl"
    split_dict = {
        "train": adata[train_mask].copy(),
        "test": adata[test_mask].copy(),
        "test_conds": ["A+ctrl"],
    }
    # train pert rows order: A+ctrl (4 rows), B+ctrl (4 rows)
    topk_map = np.asarray(
        [
            [1, 1], [1, 2], [2, 2], [2, 3],
            [7, 7], [7, 8], [8, 8], [8, 9],
        ],
        dtype=int,
    )
    strategy = model.build_eval_ctrl_strategy(
        split_dict=split_dict,
        emb_table=emb_table,
        topk_map=topk_map,
        distance_metric="cosine",
        sample_size=6,
        compare_mode="aggregate_cond",
        train_candidate_conds=["B+ctrl"],
    )
    assert strategy["train_candidate_conds_source"] == "filtered"
    assert strategy["train_candidate_conds_count"] == 1
    assert strategy["nearest_train_cond_by_test_cond"]["A+ctrl"] == "B+ctrl"
    pool = strategy["pool_idx_by_test_cond"]["A+ctrl"]
    assert pool.tolist() == [7, 7, 7, 8, 8, 8, 8, 9]


def test_per_gene_nearest_train_candidate_filter():
    X = np.random.RandomState(11).rand(12, 6).astype(np.float32)
    obs = pd.DataFrame(
        {
            "condition": (
                ["ctrl"] * 4
                + ["A+ctrl"] * 2
                + ["B+ctrl"] * 2
                + ["A+B"] * 4
            )
        }
    )
    var = pd.DataFrame({"gene_name": ["A", "B", "G1", "G2", "G3", "G4"]})
    adata = ad.AnnData(X=X, obs=obs, var=var)
    embd_df = pd.DataFrame(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        index=["ctrl", "A", "B"],
    )
    data = TriShiftData(adata, embd_df)
    data.setup_embedding_index()

    model = TriShift(data, device="cpu")
    model.model_init(
        x_dim=adata.n_vars,
        z_dim=4,
        cond_dim=embd_df.shape[1],
        vae_enc_hidden=[8],
        vae_dec_hidden=[8],
        shift_hidden=[8],
        gen_hidden=[8],
        dropout=0.0,
    )
    emb_table = torch.tensor(embd_df.values, dtype=torch.float32)
    train_mask = adata.obs["condition"].isin(["ctrl", "A+ctrl", "B+ctrl"]).values
    test_mask = adata.obs["condition"] == "A+B"
    split_dict = {
        "train": adata[train_mask].copy(),
        "test": adata[test_mask].copy(),
        "test_conds": ["A+B"],
    }
    topk_map = np.asarray(
        [
            [1, 1],
            [1, 2],
            [7, 7],
            [7, 8],
        ],
        dtype=int,
    )
    strategy = model.build_eval_ctrl_strategy(
        split_dict=split_dict,
        emb_table=emb_table,
        topk_map=topk_map,
        distance_metric="cosine",
        sample_size=5,
        compare_mode="per_gene_nearest_cond",
        train_candidate_conds=["A+ctrl"],
    )
    assert strategy["train_candidate_conds_source"] == "filtered"
    assert strategy["train_candidate_conds_count"] == 1
    assert strategy["nearest_train_conds_by_test_cond"]["A+B"] == ["A+ctrl", "A+ctrl"]
    pool = strategy["pool_idx_by_test_cond"]["A+B"]
    assert pool.tolist() == [1, 1, 1, 2, 1, 1, 1, 2]


def test_empty_train_candidate_filter_fallback_sampling():
    data, model = make_data_and_model(seed=5)
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
        sample_size=10,
        compare_mode="aggregate_cond",
        train_candidate_conds=[],
    )
    assert strategy["train_candidate_conds_source"] == "filtered"
    assert strategy["train_candidate_conds_count"] == 0
    assert len(strategy["pool_idx_by_test_cond"]) == 0
    df = model.evaluate(
        split_dict=split,
        emb_table=emb_table,
        split_id=1,
        n_ensemble=10,
        base_seed=24,
        eval_ctrl_strategy=strategy,
    )
    assert df.shape[0] > 0
    assert set(df["n_ensemble"].astype(int).unique().tolist()) == {10}


def test_target_test_conds_filter_only_maps_subset():
    X = np.random.RandomState(21).rand(16, 6).astype(np.float32)
    obs = pd.DataFrame(
        {
            "condition": (
                ["ctrl"] * 4
                + ["A+ctrl"] * 4
                + ["B+ctrl"] * 4
                + ["A+B"] * 2
                + ["A+ctrl"] * 2
            )
        }
    )
    var = pd.DataFrame({"gene_name": ["A", "B", "G1", "G2", "G3", "G4"]})
    adata = ad.AnnData(X=X, obs=obs, var=var)
    embd_df = pd.DataFrame(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        index=["ctrl", "A", "B"],
    )
    data = TriShiftData(adata, embd_df)
    data.setup_embedding_index()

    model = TriShift(data, device="cpu")
    model.model_init(
        x_dim=adata.n_vars,
        z_dim=4,
        cond_dim=embd_df.shape[1],
        vae_enc_hidden=[8],
        vae_dec_hidden=[8],
        shift_hidden=[8],
        gen_hidden=[8],
        dropout=0.0,
    )
    emb_table = torch.tensor(embd_df.values, dtype=torch.float32)
    train_mask = adata.obs["condition"].isin(["ctrl", "A+ctrl", "B+ctrl"]).values
    test_mask = adata.obs["condition"].isin(["A+ctrl", "A+B"]).values
    split_dict = {
        "train": adata[train_mask].copy(),
        "test": adata[test_mask].copy(),
        "test_conds": ["A+ctrl", "A+B"],
    }
    topk_map = np.asarray(
        [
            [1, 1], [1, 2], [2, 2], [2, 3],
            [7, 7], [7, 8], [8, 8], [8, 9],
        ],
        dtype=int,
    )
    strategy = model.build_eval_ctrl_strategy(
        split_dict=split_dict,
        emb_table=emb_table,
        topk_map=topk_map,
        distance_metric="cosine",
        sample_size=6,
        compare_mode="aggregate_cond",
        target_test_conds=["A+B"],
    )
    assert strategy["target_test_conds_count"] == 1
    assert strategy["active_test_conds_count"] == 1
    assert strategy["skipped_test_conds_count"] == 1
    assert sorted(strategy["pool_idx_by_test_cond"].keys()) == ["A+B"]


def test_target_test_conds_empty_map_fallback_sampling():
    data, model = make_data_and_model(seed=9)
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
        sample_size=10,
        compare_mode="per_gene_nearest_cond",
        target_test_conds=["__not_in_test__"],
    )
    assert strategy["active_test_conds_count"] == 0
    assert strategy["skipped_test_conds_count"] == len(split["test_conds"])
    assert len(strategy["pool_idx_by_test_cond"]) == 0

    df = model.evaluate(
        split_dict=split,
        emb_table=emb_table,
        split_id=1,
        n_ensemble=10,
        base_seed=24,
        eval_ctrl_strategy=strategy,
    )
    assert df.shape[0] > 0
    assert set(df["n_ensemble"].astype(int).unique().tolist()) == {10}


def main():
    test_eval_ctrl_pool_flatten_without_dedup()
    test_nearest_genept_ot_pool_eval_and_export()
    test_condition_tokens_no_ctrl()
    test_per_gene_nearest_strategy_concat_without_dedup()
    test_aggregate_cond_train_candidate_filter()
    test_per_gene_nearest_train_candidate_filter()
    test_empty_train_candidate_filter_fallback_sampling()
    test_target_test_conds_filter_only_maps_subset()
    test_target_test_conds_empty_map_fallback_sampling()
    print("test_eval_ctrl_pool: PASS")


if __name__ == "__main__":
    main()
