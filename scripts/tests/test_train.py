from __future__ import annotations

import io
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from trishift import _utils
from trishift.TriShift import _weighted_stage1_recon_loss

from scripts.tests._helpers import make_data_and_model, make_tiny_pbmc_adata, train_stage1_and_cache


def test_stage1_and_cache():
    data, model = make_data_and_model()
    train_stage1_and_cache(model, data)
    assert "z_mu" in data.adata_all.obsm


def test_stage1_weighted_recon_loss_upweights_deg_columns():
    x_true = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    x_recon = torch.tensor([[0.0, 1.0, 1.0]], dtype=torch.float32)
    plain = _weighted_stage1_recon_loss(
        x_true=x_true,
        x_recon=x_recon,
        cond_batch=["A"],
        deg_idx_dict=None,
        deg_weight=1.0,
    )
    weighted = _weighted_stage1_recon_loss(
        x_true=x_true,
        x_recon=x_recon,
        cond_batch=["A"],
        deg_idx_dict={"A": np.array([0, 2], dtype=int)},
        deg_weight=3.0,
    )
    assert float(weighted.item()) > float(plain.item())


def test_stage1_weighted_recon_loss_leaves_ctrl_rows_unweighted():
    x_true = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], dtype=torch.float32)
    x_recon = torch.tensor([[0.0, 1.0, 1.0], [0.0, 1.0, 1.0]], dtype=torch.float32)
    weighted = _weighted_stage1_recon_loss(
        x_true=x_true,
        x_recon=x_recon,
        cond_batch=["ctrl", "stimulated::T"],
        deg_idx_dict={"stimulated::T": np.array([0, 2], dtype=int)},
        deg_weight=3.0,
    )
    plain = ((x_true - x_recon) ** 2).sum(dim=1)
    assert float(weighted[0].item()) == float(plain[0].item())
    assert float(weighted[1].item()) > float(plain[1].item())


def test_stage1_deg_weight_on_pert_pool_runs():
    data, model = make_data_and_model()
    logs = model.train_stage1_vae(
        data.adata_all,
        epochs=1,
        batch_size=8,
        lr=1e-3,
        deg_weight=2.0,
        amp=False,
        num_workers=0,
        pin_memory=False,
        grad_accum_steps=1,
    )
    assert "epochs" in logs and len(logs["epochs"]) == 1


def test_stage1_ctrl_only_deg_weight_falls_back():
    data, model = make_data_and_model()
    buf = io.StringIO()
    with redirect_stdout(buf):
        logs = model.train_stage1_vae(
            data.adata_ctrl,
            epochs=1,
            batch_size=8,
            lr=1e-3,
            deg_weight=2.0,
            amp=False,
            num_workers=0,
            pin_memory=False,
            grad_accum_steps=1,
        )
    out = buf.getvalue()
    assert "deg_weight_active=false" in out
    assert "ctrl_only_pool" in out
    assert "epochs" in logs


def test_stage1_missing_degs_deg_weight_falls_back():
    data, model = make_data_and_model(include_top20=False, build_degs=False)
    buf = io.StringIO()
    with redirect_stdout(buf):
        logs = model.train_stage1_vae(
            data.adata_all,
            epochs=1,
            batch_size=8,
            lr=1e-3,
            deg_weight=2.0,
            amp=False,
            num_workers=0,
            pin_memory=False,
            grad_accum_steps=1,
        )
    out = buf.getvalue()
    assert "deg_weight_active=false" in out
    assert "missing_degs" in out
    assert "epochs" in logs


def test_stage1_deg_weight_pbmc_stimulated_only_runs():
    from trishift.TriShiftData import TriShiftData
    from trishift.TriShift import TriShift

    adata = make_tiny_pbmc_adata()
    adata.obs["condition_internal"] = np.where(
        adata.obs["condition"].astype(str).eq("control"),
        "ctrl",
        "stimulated",
    )
    adata.obs["stage1_deg_key"] = np.where(
        adata.obs["condition"].astype(str).eq("stimulated"),
        "stimulated::" + adata.obs["cell_type"].astype(str),
        "ctrl",
    )
    adata.var["gene_name"] = adata.var["gene_symbol"].astype(str)
    adata.uns["top20_degs_non_dropout"] = {
        "stimulated::T": np.array([0, 1], dtype=int),
        "stimulated::B": np.array([2, 3], dtype=int),
    }
    embd_df = np.array([[0.0], [1.0]], dtype=np.float32)
    data = TriShiftData(
        adata,
        pd.DataFrame(embd_df, index=["ctrl", "stimulated"], columns=["dummy_cond"]),
        label_key="condition_internal",
    )
    data.setup_embedding_index()
    model = TriShift(data, device="cpu")
    model.model_init(
        x_dim=adata.n_vars,
        z_dim=8,
        cond_dim=1,
        vae_enc_hidden=[16],
        vae_dec_hidden=[16],
        shift_hidden=[16],
        gen_hidden=[16],
        dropout=0.0,
    )
    buf = io.StringIO()
    with redirect_stdout(buf):
        logs = model.train_stage1_vae(
            data.adata_all,
            epochs=1,
            batch_size=8,
            lr=1e-3,
            deg_weight=2.0,
            deg_key_obs_key="stage1_deg_key",
            amp=False,
            num_workers=0,
            pin_memory=False,
            grad_accum_steps=1,
        )
    out = buf.getvalue()
    assert "deg_weight_active=true" in out
    assert "epochs" in logs and len(logs["epochs"]) == 1


def test_stage23_joint():
    data, model = make_data_and_model()
    train_stage1_and_cache(model, data)
    emb_table = torch.tensor(data.embd_df.values, dtype=torch.float32)
    split = data.split_by_condition(seed=1, test_ratio=0.2, val_ratio=0.2)
    logs = model.train_stage23_joint(
        split_dict=split,
        emb_table=emb_table,
        mode="knn",
        k=3,
        split_id=1,
        epochs=1,
        batch_size=4,
        lr=1e-3,
        amp=False,
        num_workers=0,
        pin_memory=False,
        grad_accum_steps=1,
    )
    assert "epochs" in logs


def test_stage23_sequential():
    data, model = make_data_and_model()
    train_stage1_and_cache(model, data)
    emb_table = torch.tensor(data.embd_df.values, dtype=torch.float32)
    split = data.split_by_condition(seed=2, test_ratio=0.2, val_ratio=0.2)
    logs = model.train_stage23_sequential(
        split_dict=split,
        emb_table=emb_table,
        mode="knn",
        k=3,
        split_id=1,
        epochs_stage2=1,
        epochs_stage3=1,
        batch_size=4,
        lr_stage2=1e-3,
        lr_stage3=1e-3,
        amp=False,
        num_workers=0,
        pin_memory=False,
        grad_accum_steps=1,
    )
    assert "stage2" in logs and "stage3" in logs


def test_stage23_joint_predict_shift_false_skips_shift_head():
    data, model = make_data_and_model(
        model_init_overrides={"predict_shift": False},
    )
    train_stage1_and_cache(model, data)
    emb_table = torch.tensor(data.embd_df.values, dtype=torch.float32)
    split = data.split_by_condition(seed=5, test_ratio=0.2, val_ratio=0.2)

    def _raise_if_called(*_args, **_kwargs):
        raise AssertionError("predict_shift_repr should not be called when predict_shift=false")

    model.net.predict_shift_repr = _raise_if_called
    logs = model.train_stage23_joint(
        split_dict=split,
        emb_table=emb_table,
        mode="knn",
        k=3,
        split_id=1,
        epochs=1,
        batch_size=4,
        lr=1e-3,
        amp=False,
        num_workers=0,
        pin_memory=False,
        grad_accum_steps=1,
    )
    assert "epochs" in logs


def test_stage23_sequential_predict_shift_false_skips_stage2():
    data, model = make_data_and_model(
        model_init_overrides={"predict_shift": False},
    )
    train_stage1_and_cache(model, data)
    emb_table = torch.tensor(data.embd_df.values, dtype=torch.float32)
    split = data.split_by_condition(seed=6, test_ratio=0.2, val_ratio=0.2)

    def _raise_if_called(*_args, **_kwargs):
        raise AssertionError("predict_shift_repr should not be called when predict_shift=false")

    model.net.predict_shift_repr = _raise_if_called
    logs = model.train_stage23_sequential(
        split_dict=split,
        emb_table=emb_table,
        mode="knn",
        k=3,
        split_id=1,
        epochs_stage2=1,
        epochs_stage3=1,
        batch_size=4,
        lr_stage2=1e-3,
        lr_stage3=1e-3,
        amp=False,
        num_workers=0,
        pin_memory=False,
        grad_accum_steps=1,
    )
    assert logs["stage2"][0].get("stage2_skipped_predict_shift_false", False) is True


def test_stage3_only():
    data, model = make_data_and_model()
    train_stage1_and_cache(model, data)
    emb_table = torch.tensor(data.embd_df.values, dtype=torch.float32)
    split = data.split_by_condition(seed=3, test_ratio=0.2, val_ratio=0.2)
    logs = model.train_stage3_only(
        split_dict=split,
        emb_table=emb_table,
        split_id=1,
        epochs=1,
        batch_size=4,
        lr=1e-3,
        amp=False,
        num_workers=0,
        pin_memory=False,
        grad_accum_steps=1,
    )
    assert "epochs" in logs


def test_stage23_joint_requires_cached_z_mu():
    data, model = make_data_and_model()
    emb_table = torch.tensor(data.embd_df.values, dtype=torch.float32)
    split = data.split_by_condition(seed=4, test_ratio=0.2, val_ratio=0.2)
    try:
        model.train_stage23_joint(
            split_dict=split,
            emb_table=emb_table,
            mode="knn",
            k=3,
            split_id=1,
            epochs=1,
            batch_size=4,
            lr=1e-3,
            amp=False,
            num_workers=0,
            pin_memory=False,
            grad_accum_steps=1,
        )
        raise AssertionError("expected ValueError when z_mu cache is missing")
    except ValueError as exc:
        assert "z_mu cache missing" in str(exc)


def main():
    _utils.set_seeds(0)
    test_stage1_and_cache()
    test_stage1_weighted_recon_loss_upweights_deg_columns()
    test_stage1_weighted_recon_loss_leaves_ctrl_rows_unweighted()
    test_stage1_deg_weight_on_pert_pool_runs()
    test_stage1_ctrl_only_deg_weight_falls_back()
    test_stage1_missing_degs_deg_weight_falls_back()
    test_stage1_deg_weight_pbmc_stimulated_only_runs()
    test_stage23_joint()
    test_stage23_sequential()
    test_stage23_joint_predict_shift_false_skips_shift_head()
    test_stage23_sequential_predict_shift_false_skips_stage2()
    test_stage3_only()
    test_stage23_joint_requires_cached_z_mu()
    print("test_train: PASS")


if __name__ == "__main__":
    main()
