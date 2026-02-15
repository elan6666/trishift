from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import anndata as ad
import torch

import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from trishift import _utils
from trishift.TriShiftData import TriShiftData
from trishift.TriShift import TriShift


def make_tiny_adata(
    n_ctrl: int = 20,
    n_per: int = 12,
    n_genes: int = 40,
    seed: int = 0,
    include_top20: bool = True,
    include_gene_idx: bool = False,
) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    X_ctrl = rng.random((n_ctrl, n_genes), dtype=np.float32)
    X_a = rng.random((n_per, n_genes), dtype=np.float32)
    X_b = rng.random((n_per, n_genes), dtype=np.float32)
    X_ab = rng.random((n_per, n_genes), dtype=np.float32)
    X = np.vstack([X_ctrl, X_a, X_b, X_ab])
    obs = pd.DataFrame(
        {
            "condition": ["ctrl"] * n_ctrl
            + ["A"] * n_per
            + ["B"] * n_per
            + ["A+B"] * n_per
        }
    )
    gene_names = ["A", "B"] + [f"G{i}" for i in range(n_genes - 2)]
    var = pd.DataFrame({"gene_name": gene_names})
    adata = ad.AnnData(X=X, obs=obs, var=var)
    if include_top20:
        top20 = np.arange(min(20, n_genes), dtype=int)
        adata.uns["top20_degs_non_dropout"] = {
            "A": top20.copy(),
            "B": top20.copy(),
            "A+B": top20.copy(),
        }
    if include_gene_idx:
        non_zero = np.arange(min(20, n_genes), dtype=int)
        adata.uns["gene_idx_non_zeros"] = {"A": non_zero, "B": non_zero, "A+B": non_zero}
        adata.uns["gene_idx_non_dropout"] = {"A": non_zero, "B": non_zero, "A+B": non_zero}
    return adata


def make_embeddings_df(dim: int = 8) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    data = rng.random((3, dim), dtype=np.float32)
    return pd.DataFrame(data, index=["ctrl", "A", "B"])


def make_data_and_model(
    seed: int = 0,
    model_init_overrides: dict | None = None,
) -> tuple[TriShiftData, TriShift]:
    _utils.set_seeds(seed)
    adata = make_tiny_adata()
    embd_df = make_embeddings_df()
    data = TriShiftData(adata, embd_df)
    data.setup_embedding_index()
    data.build_or_load_degs()
    model = TriShift(data, device="cpu")
    model_init_kwargs = {
        "x_dim": adata.n_vars,
        "z_dim": 8,
        "cond_dim": embd_df.shape[1],
        "vae_enc_hidden": [16],
        "vae_dec_hidden": [16],
        "shift_hidden": [16],
        "gen_hidden": [16],
        "dropout": 0.0,
    }
    if model_init_overrides:
        model_init_kwargs.update(model_init_overrides)
    model.model_init(
        **model_init_kwargs,
    )
    return data, model


def train_stage1_and_cache(model: TriShift, data: TriShiftData) -> None:
    model.train_stage1_vae(
        data.adata_ctrl,
        epochs=1,
        batch_size=8,
        lr=1e-3,
        amp=False,
        num_workers=0,
        pin_memory=False,
        grad_accum_steps=1,
    )
    model.encode_and_cache_mu(
        data.adata_all,
        batch_size=16,
        amp=False,
        num_workers=0,
        pin_memory=False,
    )


def temp_dir() -> tempfile.TemporaryDirectory:
    return tempfile.TemporaryDirectory()
