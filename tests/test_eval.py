from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from tests._helpers import make_data_and_model, train_stage1_and_cache, temp_dir


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
    assert "deg_mean_r2" in df.columns
    assert "systema_corr_20de_allpert" in df.columns
    assert "systema_corr_deg_r2" in df.columns
    assert "scpram_r2_degs_mean_mean" in df.columns
    assert "scpram_r2_degs_var_mean" in df.columns
    assert "systema_corr_20de_allpert_dist" not in df.columns
    assert "systema_corr_deg_r2_dist" not in df.columns

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
        first_payload = next(iter(preds.values()))
        assert "Pred_full" in first_payload
        assert "Ctrl_full" in first_payload
        assert "Truth_full" in first_payload
        assert "gene_name_full" in first_payload
        assert first_payload["Pred_full"].shape[1] == len(first_payload["gene_name_full"])
        np.testing.assert_allclose(
            first_payload["Pred_full"][:, first_payload["DE_idx"]],
            first_payload["Pred"],
        )


def test_target_domain_ctrl_export_contract():
    data, model = make_data_and_model()
    train_stage1_and_cache(model, data)
    adata = data.adata_all
    donor_id = np.asarray(["donor_0"] * adata.n_obs, dtype=object)
    cond = adata.obs["condition"].astype(str).values
    ctrl_rows = np.where(cond == "ctrl")[0]
    a_rows = np.where(cond == "A")[0]
    b_rows = np.where(cond == "B")[0]
    donor_id[ctrl_rows[:7]] = "donor_2"
    donor_id[a_rows[:5]] = "donor_2"
    adata.obs["donor_id"] = donor_id

    train_mask = (adata.obs["donor_id"].astype(str).values == "donor_0") & np.isin(cond, ["ctrl", "B"])
    test_mask = (adata.obs["donor_id"].astype(str).values == "donor_2") & np.isin(cond, ["ctrl", "A"])
    split = {
        "train": adata[train_mask],
        "val": adata[train_mask],
        "test": adata[test_mask],
        "train_conds": ["B"],
        "val_conds": ["B"],
        "test_conds": ["A"],
        "split_policy": "domain_unseen_drug",
        "split_domain_key": "donor_id",
        "train_domain_values": ["donor_0"],
        "test_domain_values": ["donor_2"],
    }
    emb_table = torch.tensor(data.embd_df.values, dtype=torch.float32)

    df = model.evaluate(
        split_dict=split,
        emb_table=emb_table,
        split_id=1,
        n_ensemble=3,
        base_seed=24,
        eval_ctrl_source="target_domain_test_ctrl",
        eval_batch_size=4,
    )
    assert df.shape[0] == 1
    assert int(df.loc[0, "n_ensemble"]) == 7
    assert int(df.loc[0, "n_eval_ctrl"]) == 7
    assert str(df.loc[0, "eval_ctrl_source"]) == "target_domain_test_ctrl"
    assert "systema_corr_20de_allpert_dist" in df.columns
    assert "systema_corr_deg_r2_dist" in df.columns

    with temp_dir() as td:
        out_path = Path(td) / "target_domain.pkl"
        preds = model.export_predictions(
            split_dict=split,
            emb_table=emb_table,
            split_id=1,
            n_ensemble=3,
            base_seed=24,
            out_path=str(out_path),
            eval_ctrl_source="target_domain_test_ctrl",
            eval_batch_size=4,
        )
    payload = preds["A"]
    assert payload["Pred_full"].shape[0] <= 3
    assert payload["Ctrl_full"].shape[0] <= 3
    assert payload["Truth_full"].shape[0] <= 3
    assert bool(payload["export_metadata"]["metrics_computed_on_full"])
    assert payload["export_metadata"]["n_pred_full"] == 7
    assert payload["export_metadata"]["n_ctrl_full"] == 7
    assert "full_summary" in payload
    assert "systema_corr_20de_allpert_dist" in payload["full_summary"]["metrics"]
    np.testing.assert_allclose(payload["Pred_full"][:, payload["DE_idx"]], payload["Pred"])


def main():
    test_evaluate_and_export()
    test_target_domain_ctrl_export_contract()
    print("test_eval: PASS")


if __name__ == "__main__":
    main()
