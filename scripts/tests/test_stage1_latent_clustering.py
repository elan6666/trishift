from pathlib import Path
import json

import numpy as np

import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.tests._helpers import make_tiny_pbmc_adata, temp_dir
from scripts.trishift.analysis.stage1_latent_clustering import (
    _can_reuse_stage1_cache,
    _load_or_build_pbmc_deg_cache,
    _prepare_pbmc_data,
    _stage1_cache_identity,
)


def test_prepare_pbmc_data_keeps_original_condition_and_internal_alias():
    with temp_dir() as tmp:
        tmp_path = Path(tmp)
        pbmc_path = tmp_path / "train_pbmc.h5ad"
        make_tiny_pbmc_adata().write_h5ad(pbmc_path)
        adata, data, embd_df, deg_cache_path = _prepare_pbmc_data(
            pbmc_path,
            stage1_deg_weight=2.0,
            pbmc_deg_mode="by_cell_type",
        )
    conds = set(adata.obs["condition"].astype(str).unique().tolist())
    cond_internal = set(adata.obs["condition_internal"].astype(str).unique().tolist())
    deg_keys = set(adata.obs["stage1_deg_key"].astype(str).unique().tolist())
    assert conds == {"control", "stimulated"}
    assert cond_internal == {"ctrl", "stimulated"}
    assert "stimulated" in set(data.adata_all.obs["condition_internal"].astype(str).unique().tolist())
    assert set(embd_df.index.astype(str).tolist()) == {"ctrl", "stimulated"}
    assert "stimulated::T" in deg_keys
    assert "stimulated::B" in deg_keys
    assert deg_cache_path is not None and Path(deg_cache_path).exists()


def test_load_or_build_pbmc_deg_cache_returns_by_cell_type_keys():
    adata = make_tiny_pbmc_adata()
    with temp_dir() as tmp:
        tmp_path = Path(tmp)
        pbmc_path = tmp_path / "train_pbmc.h5ad"
        adata.write_h5ad(pbmc_path)
        payload, cache_path = _load_or_build_pbmc_deg_cache(
            adata,
            pbmc_path=pbmc_path,
            pbmc_deg_mode="by_cell_type",
        )
    degs = payload["top20_degs_non_dropout_by_key"]
    assert cache_path.exists()
    assert "stimulated::T" in degs
    assert "stimulated::B" in degs
    assert np.asarray(degs["stimulated::T"], dtype=int).size > 0
    assert np.asarray(degs["stimulated::B"], dtype=int).size > 0


def test_stage1_cache_identity_detects_stage1_deg_changes():
    with temp_dir() as tmp:
        tmp_path = Path(tmp)
        meta_path = tmp_path / "run_meta.json"
        base = _stage1_cache_identity(
            mode="pbmc_celltype",
            dataset_name="pbmc",
            split_id=1,
            stage1_pool_mode="train_all_cells",
            random_seed=24,
            stage1_epochs=10,
            stage1_z_dim=32,
            stage1_batch_size=256,
            stage1_lr=5e-4,
            stage1_beta=1.0,
            stage1_deg_weight=1.0,
            stage1_sched_gamma=1.0,
            stage1_patience=5,
            stage1_min_delta=1e-3,
            pbmc_path=tmp_path / "train_pbmc.h5ad",
            pbmc_train_frac=0.9,
            pbmc_deg_mode="by_cell_type",
        )
        meta_path.write_text(json.dumps(base), encoding="utf-8")
        assert _can_reuse_stage1_cache(meta_path, expected_identity=base)
        changed = dict(base)
        changed["stage1_deg_weight"] = 3.0
        assert not _can_reuse_stage1_cache(meta_path, expected_identity=changed)


def main():
    test_prepare_pbmc_data_keeps_original_condition_and_internal_alias()
    test_load_or_build_pbmc_deg_cache_returns_by_cell_type_keys()
    test_stage1_cache_identity_detects_stage1_deg_changes()
    print("test_stage1_latent_clustering: PASS")


if __name__ == "__main__":
    main()
