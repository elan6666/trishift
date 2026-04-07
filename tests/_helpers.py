from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import anndata as ad
import torch

import sys

ROOT = Path(__file__).resolve().parents[1]
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


def make_tiny_pbmc_adata(
    n_per_group: int = 8,
    n_genes: int = 12,
    seed: int = 0,
) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    gene_names = [f"G{i}" for i in range(n_genes)]
    cell_types = []
    conditions = []
    xs = []
    for cell_type, up_idx in (("T", [0, 1]), ("B", [2, 3])):
        for condition in ("control", "stimulated"):
            base = rng.normal(loc=1.0, scale=0.05, size=(n_per_group, n_genes)).astype(np.float32)
            base = np.clip(base, 0.0, None)
            if condition == "stimulated":
                base[:, up_idx] += 2.0
            xs.append(base)
            cell_types.extend([cell_type] * n_per_group)
            conditions.extend([condition] * n_per_group)
    X = np.vstack(xs).astype(np.float32, copy=False)
    obs = pd.DataFrame(
        {
            "condition": conditions,
            "cell_type": cell_types,
        }
    )
    var = pd.DataFrame({"gene_symbol": gene_names})
    return ad.AnnData(X=X, obs=obs, var=var)


def make_tiny_scvi_pbmc_adata(
    n_per_group: int = 8,
    n_genes: int = 30,
    seed: int = 0,
) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    gene_symbols = [f"G{i}" for i in range(n_genes)]
    xs = []
    labels = []
    batches = []
    for batch_id in (0, 1):
        for cell_type, up_idx in (("B cells", [0, 1, 2]), ("CD4 T cells", [3, 4, 5])):
            base = rng.poisson(lam=3 + batch_id, size=(n_per_group, n_genes)).astype(np.float32)
            base[:, up_idx] += 2 + batch_id
            xs.append(base)
            labels.extend([cell_type] * n_per_group)
            batches.extend([batch_id] * n_per_group)
    X = np.vstack(xs).astype(np.float32, copy=False)
    obs = pd.DataFrame(
        {
            "batch": batches,
            "str_labels": labels,
        }
    )
    var = pd.DataFrame({"gene_symbols": gene_symbols})
    return ad.AnnData(X=X, obs=obs, var=var)


def make_data_and_model(
    seed: int = 0,
    include_top20: bool = True,
    build_degs: bool = True,
    model_init_overrides: dict | None = None,
) -> tuple[TriShiftData, TriShift]:
    _utils.set_seeds(seed)
    adata = make_tiny_adata(include_top20=include_top20)
    embd_df = make_embeddings_df()
    data = TriShiftData(adata, embd_df)
    data.setup_embedding_index()
    if build_degs:
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
