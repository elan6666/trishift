from __future__ import annotations

from pathlib import Path

import torch

import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from trishift._model import ShiftNet, TriShiftNet, aggregate_cond_embedding


def test_aggregate_cond_embedding_empty_and_mean() -> None:
    emb_table = torch.tensor(
        [
            [1.0, 3.0, 5.0],
            [2.0, 4.0, 6.0],
        ],
        dtype=torch.float32,
    )
    out_empty = aggregate_cond_embedding(emb_table, [], mode="sum")
    assert torch.equal(out_empty, torch.zeros(3, dtype=torch.float32))

    out_mean = aggregate_cond_embedding(emb_table, [0, 1], mode="mean")
    expected = torch.tensor([1.5, 3.5, 5.5], dtype=torch.float32)
    assert torch.allclose(out_mean, expected)


def test_shift_net_output_dims() -> None:
    batch = 4
    model_dim = 8
    cond_dim = 6
    z_ctrl = torch.randn(batch, model_dim)
    cond = torch.randn(batch, cond_dim)

    net_delta = ShiftNet(
        z_dim=model_dim,
        cond_dim=cond_dim,
        hidden=[16],
        predict_delta=True,
        use_cross_attention=False,
        use_transformer_block=False,
        delta_out_dim=5,
    )
    out_delta = net_delta(z_ctrl, cond)
    assert out_delta.shape == (batch, 5)

    net_repr = ShiftNet(
        z_dim=model_dim,
        cond_dim=cond_dim,
        hidden=[16],
        predict_delta=False,
        use_cross_attention=False,
        use_transformer_block=False,
    )
    out_repr = net_repr(z_ctrl, cond)
    assert out_repr.shape == (batch, model_dim)


def test_shift_net_transformer_concat_repr_dim() -> None:
    batch = 3
    model_dim = 8
    cond_dim = 8
    z_ctrl = torch.randn(batch, model_dim)
    cond = torch.randn(batch, cond_dim)

    net = ShiftNet(
        z_dim=model_dim,
        cond_dim=cond_dim,
        hidden=[16],
        predict_delta=False,
        use_transformer_block=True,
        transformer_readout="concat",
        transformer_layers=1,
        cross_attn_heads=4,
    )
    out = net(z_ctrl, cond)
    assert out.shape == (batch, model_dim * 2)


def test_trishiftnet_state_source_resolution_and_forward_keys() -> None:
    model = TriShiftNet(
        x_dim=10,
        z_dim=6,
        cond_dim=8,
        vae_enc_hidden=[16],
        vae_dec_hidden=[16],
        shift_hidden=[16],
        gen_hidden=[16],
        gen_state_dim="cond_dim",
        shift_predict_delta=True,
        shift_use_cross_attention=True,
        shift_cross_attn_heads=4,
        shift_input_source="state",
    )
    assert model.gen.state_dim == 8
    assert model.shift.model_dim == 8

    x_ctrl = torch.randn(5, 10)
    cond_vec = torch.randn(5, 8)
    out = model.forward_joint(x_ctrl=x_ctrl, cond_vec=cond_vec, z_ctrl_mu=None)
    assert "delta_z" in out
    assert out["shift_repr"].shape == (5, 6)
    assert out["x_pred"].shape == (5, 10)

    model_no_delta = TriShiftNet(
        x_dim=10,
        z_dim=6,
        cond_dim=8,
        vae_enc_hidden=[16],
        vae_dec_hidden=[16],
        shift_hidden=[16],
        gen_hidden=[16],
        shift_predict_delta=False,
        shift_input_source="latent_mu",
    )
    out_no_delta = model_no_delta.forward_joint(
        x_ctrl=torch.randn(5, 10),
        cond_vec=torch.randn(5, 8),
        z_ctrl_mu=torch.randn(5, 6),
    )
    assert "delta_z" not in out_no_delta


def test_trishiftnet_state_requires_gen_state_dim() -> None:
    try:
        TriShiftNet(
            x_dim=10,
            z_dim=6,
            cond_dim=8,
            vae_enc_hidden=[16],
            vae_dec_hidden=[16],
            shift_hidden=[16],
            gen_hidden=[16],
            gen_encoder_hidden=[12],
            gen_decoder_hidden=[12],
            gen_state_dim=None,
            shift_input_source="state",
        )
        raise AssertionError("expected ValueError for missing gen_state_dim")
    except ValueError as exc:
        assert "shift_input_source=state requires gen_state_dim" in str(exc)


def main() -> None:
    test_aggregate_cond_embedding_empty_and_mean()
    test_shift_net_output_dims()
    test_shift_net_transformer_concat_repr_dim()
    test_trishiftnet_state_source_resolution_and_forward_keys()
    test_trishiftnet_state_requires_gen_state_dim()
    print("test_model: PASS")


if __name__ == "__main__":
    main()
