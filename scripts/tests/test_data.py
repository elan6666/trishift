from __future__ import annotations

from pathlib import Path

import numpy as np

import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from trishift.TriShiftData import TriShiftData
from trishift import _utils

from scripts.tests._helpers import make_tiny_adata, make_embeddings_df


def test_setup_embedding_index_filters_missing():
    adata = make_tiny_adata()
    embd_df = make_embeddings_df()
    embd_df = embd_df.drop(index=["B"])
    data = TriShiftData(adata, embd_df)
    data.setup_embedding_index()
    conds = set(data.adata_all.obs["condition"].astype(str).unique())
    assert "B" not in conds


def test_build_or_load_degs_precomputed():
    adata = make_tiny_adata(include_top20=True, include_gene_idx=False)
    embd_df = make_embeddings_df()
    data = TriShiftData(adata, embd_df)
    data.setup_embedding_index()
    data.build_or_load_degs()
    uns = data.adata_all.uns
    assert "top20_degs_final" in uns
    assert "gene_idx_non_dropout" in uns
    assert "gene_idx_non_zeros" in uns


def test_build_or_load_degs_compute():
    try:
        import scanpy as _  # noqa: F401
    except Exception:
        print("test_build_or_load_degs_compute: SKIP (scanpy missing)")
        return
    adata = make_tiny_adata(include_top20=False)
    embd_df = make_embeddings_df()
    data = TriShiftData(adata, embd_df)
    data.setup_embedding_index()
    data.build_or_load_degs(prefer_key="missing_key")
    uns = data.adata_all.uns
    assert "top20_degs_final" in uns
    assert "gene_idx_non_dropout" in uns
    assert "gene_idx_non_zeros" in uns


def test_topk_map_modes():
    adata = make_tiny_adata(include_gene_idx=True)
    embd_df = make_embeddings_df()
    data = TriShiftData(adata, embd_df)
    data.setup_embedding_index()
    data.build_or_load_degs()
    z_mu = np.random.default_rng(0).random((data.adata_all.n_obs, 8), dtype=np.float32)
    data.set_latent_mu(z_mu)

    for mode in ("knn", "ot", "knn_ot"):
        topk = data.build_or_load_topk_map(
            split_adata=data.adata_all,
            mode=mode,
            k=3,
            seed=1,
            candidates=5,
            cache_path=None,
        )
        assert topk.shape[1] == 3
        assert topk.min() >= 0
        assert topk.max() < len(data.ctrl_indices)


def main():
    test_setup_embedding_index_filters_missing()
    test_build_or_load_degs_precomputed()
    test_build_or_load_degs_compute()
    test_topk_map_modes()
    print("test_data: PASS")


if __name__ == "__main__":
    main()
