from __future__ import annotations

from pathlib import Path

import torch

import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from trishift import _utils

from scripts.tests._helpers import make_data_and_model, train_stage1_and_cache


def test_stage1_and_cache():
    data, model = make_data_and_model()
    train_stage1_and_cache(model, data)
    assert "z_mu" in data.adata_all.obsm


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


def test_latent_decoder_joint():
    data, model = make_data_and_model(
        model_init_overrides={
            "gen_input_mode": "full",
            "gen_state_source": "latent_mu",
            "shift_input_source": "latent_mu",
        }
    )
    train_stage1_and_cache(model, data)
    emb_table = torch.tensor(data.embd_df.values, dtype=torch.float32)
    split = data.split_by_condition(seed=5, test_ratio=0.2, val_ratio=0.2)
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


def main():
    _utils.set_seeds(0)
    test_stage1_and_cache()
    test_stage23_joint()
    test_stage23_sequential()
    test_stage3_only()
    test_stage23_joint_requires_cached_z_mu()
    test_latent_decoder_joint()
    print("test_train: PASS")


if __name__ == "__main__":
    main()
