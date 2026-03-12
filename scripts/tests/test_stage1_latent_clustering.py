from pathlib import Path
import json
import types

import numpy as np
import pandas as pd
import scanpy as sc

import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.tests._helpers import make_tiny_pbmc_adata, make_tiny_scvi_pbmc_adata, temp_dir
from scripts.trishift.analysis.stage1_latent_clustering import (
    _can_reuse_stage1_cache,
    _label_metrics,
    _load_or_build_pbmc_deg_cache,
    _prepare_pbmc_data,
    _stage1_cache_identity,
)
from trishift._scgpt_preprocess import Preprocessor


class _MiniMonkeyPatch:
    def setattr(self, obj, name, value):
        setattr(obj, name, value)

    def setitem(self, mapping, key, value):
        mapping[key] = value


def _install_fake_scvi(monkeypatch):
    calls = {}

    def _pbmc_dataset(**kwargs):
        calls["kwargs"] = dict(kwargs)
        return make_tiny_scvi_pbmc_adata()

    fake_scvi = types.SimpleNamespace(
        data=types.SimpleNamespace(
            pbmc_dataset=_pbmc_dataset,
        )
    )
    monkeypatch.setitem(sys.modules, "scvi", fake_scvi)
    return calls


def test_scgpt_preprocessor_emits_expected_layers(monkeypatch):
    adata = make_tiny_scvi_pbmc_adata()

    def _fake_hvg(
        adata_in,
        layer=None,
        n_top_genes=None,
        batch_key=None,
        flavor=None,
        subset=False,
    ):
        n_keep = min(int(n_top_genes or adata_in.n_vars), adata_in.n_vars)
        keep_mask = np.zeros(adata_in.n_vars, dtype=bool)
        keep_mask[:n_keep] = True
        adata_in.var["highly_variable"] = keep_mask
        if subset:
            adata_in._inplace_subset_var(keep_mask)

    monkeypatch.setattr(sc.pp, "highly_variable_genes", _fake_hvg)
    preprocessor = Preprocessor(
        use_key="X",
        filter_gene_by_counts=3,
        filter_cell_by_counts=False,
        normalize_total=1e4,
        result_normed_key="X_normed",
        log1p=True,
        result_log1p_key="X_log1p",
        subset_hvg=12,
        hvg_flavor="seurat_v3",
        binning=11,
        result_binned_key="X_binned",
    )
    preprocessor(adata, batch_key="batch")
    assert "X_normed" in adata.layers
    assert "X_log1p" in adata.layers
    assert "X_binned" in adata.layers
    assert "bin_edges" in adata.obsm
    assert adata.layers["X_binned"].shape == adata.layers["X_log1p"].shape


def test_scgpt_preprocessor_falls_back_when_skmisc_missing(monkeypatch):
    adata = make_tiny_scvi_pbmc_adata()
    calls = []

    def _fake_hvg(
        adata_in,
        layer=None,
        n_top_genes=None,
        batch_key=None,
        flavor=None,
        subset=False,
    ):
        calls.append(str(flavor))
        if str(flavor) == "seurat_v3":
            raise ImportError("No module named 'skmisc'")
        n_keep = min(int(n_top_genes or adata_in.n_vars), adata_in.n_vars)
        keep_mask = np.zeros(adata_in.n_vars, dtype=bool)
        keep_mask[:n_keep] = True
        adata_in.var["highly_variable"] = keep_mask
        if subset:
            adata_in._inplace_subset_var(keep_mask)

    monkeypatch.setattr(sc.pp, "highly_variable_genes", _fake_hvg)
    preprocessor = Preprocessor(
        use_key="X",
        filter_gene_by_counts=3,
        filter_cell_by_counts=False,
        normalize_total=1e4,
        result_normed_key="X_normed",
        log1p=True,
        result_log1p_key="X_log1p",
        subset_hvg=12,
        hvg_flavor="seurat_v3",
        binning=11,
        result_binned_key="X_binned",
    )
    preprocessor(adata, batch_key="batch")
    assert calls[:2] == ["seurat_v3", "cell_ranger"]
    assert "X_log1p" in adata.layers


def test_prepare_pbmc_data_uses_scvi_source_and_scgpt_preprocess(monkeypatch):
    calls = _install_fake_scvi(monkeypatch)

    def _fake_hvg(
        adata_in,
        layer=None,
        n_top_genes=None,
        batch_key=None,
        flavor=None,
        subset=False,
    ):
        n_keep = min(int(n_top_genes or adata_in.n_vars), adata_in.n_vars)
        keep_mask = np.zeros(adata_in.n_vars, dtype=bool)
        keep_mask[:n_keep] = True
        adata_in.var["highly_variable"] = keep_mask
        if subset:
            adata_in._inplace_subset_var(keep_mask)

    monkeypatch.setattr(sc.pp, "highly_variable_genes", _fake_hvg)
    adata, data, embd_df, deg_cache_path, deg_weight_status, pbmc_input_layer = _prepare_pbmc_data(
        pbmc_source="scvi",
        stage1_deg_weight=2.0,
        pbmc_deg_mode="by_cell_type",
        pbmc_filter_gene_by_counts=3,
        pbmc_normalize_total=1e4,
        pbmc_log1p=True,
        pbmc_n_hvg=12,
    )
    conds = set(adata.obs["condition"].astype(str).unique().tolist())
    cond_internal = set(adata.obs["condition_internal"].astype(str).unique().tolist())
    assert conds == {"0", "1"}
    assert cond_internal == {"ctrl"}
    assert set(embd_df.index.astype(str).tolist()) == {"ctrl"}
    assert deg_cache_path is None
    assert deg_weight_status["active"] is False
    assert deg_weight_status["reason"] == "missing_pbmc_perturbation_condition"
    assert "save_path" in calls["kwargs"]
    assert Path(calls["kwargs"]["save_path"]).name == "scvi_pbmc"
    assert calls["kwargs"]["remove_extracted_data"] is False
    assert "X_normed" in adata.layers
    assert "X_log1p" in adata.layers
    assert "X_binned" not in adata.layers
    assert "bin_edges" not in adata.obsm
    assert pbmc_input_layer == "X_log1p"
    np.testing.assert_allclose(np.asarray(adata.X), np.asarray(adata.layers["X_log1p"]))
    assert set(data.adata_all.obs["label_cell_type"].astype(str).unique().tolist()) == {
        "B cells",
        "CD4 T cells",
    }


def test_prepare_pbmc_data_can_disable_log1p_and_use_normed_input(monkeypatch):
    _install_fake_scvi(monkeypatch)

    def _fake_hvg(
        adata_in,
        layer=None,
        n_top_genes=None,
        batch_key=None,
        flavor=None,
        subset=False,
    ):
        n_keep = min(int(n_top_genes or adata_in.n_vars), adata_in.n_vars)
        keep_mask = np.zeros(adata_in.n_vars, dtype=bool)
        keep_mask[:n_keep] = True
        adata_in.var["highly_variable"] = keep_mask
        if subset:
            adata_in._inplace_subset_var(keep_mask)

    monkeypatch.setattr(sc.pp, "highly_variable_genes", _fake_hvg)
    adata, _, _, _, _, pbmc_input_layer = _prepare_pbmc_data(
        pbmc_source="scvi",
        stage1_deg_weight=1.0,
        pbmc_deg_mode="by_cell_type",
        pbmc_filter_gene_by_counts=3,
        pbmc_normalize_total=1e4,
        pbmc_log1p=False,
        pbmc_n_hvg=12,
    )
    assert pbmc_input_layer == "X_normed"
    assert "X_normed" in adata.layers
    assert "X_log1p" not in adata.layers
    np.testing.assert_allclose(np.asarray(adata.X), np.asarray(adata.layers["X_normed"]))


def test_load_or_build_pbmc_deg_cache_returns_by_cell_type_keys():
    adata = make_tiny_pbmc_adata()
    with temp_dir() as tmp:
        tmp_path = Path(tmp)
        pbmc_path = tmp_path / "train_pbmc.h5ad"
        adata.write_h5ad(pbmc_path)
        payload, cache_path = _load_or_build_pbmc_deg_cache(
            adata,
            pbmc_source="scvi",
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
            stage1_ecs_enable=False,
            stage1_ecs_epochs=10,
            stage1_ecs_lr=1e-4,
            stage1_ecs_sched_gamma=0.9,
            stage1_ecs_weight=10.0,
            stage1_ecs_threshold=0.8,
            stage1_ecs_patience=5,
            stage1_ecs_min_delta=1e-3,
            stage1_sched_gamma=1.0,
            stage1_patience=5,
            stage1_min_delta=1e-3,
            pbmc_source="scvi",
            pbmc_train_frac=0.9,
            pbmc_deg_mode="by_cell_type",
            pbmc_preprocess_mode="scgpt",
            pbmc_input_layer="X_log1p",
            pbmc_filter_gene_by_counts=3,
            pbmc_normalize_total=1e4,
            pbmc_log1p=True,
            pbmc_n_hvg=1200,
        )
        meta_path.write_text(json.dumps(base), encoding="utf-8")
        assert _can_reuse_stage1_cache(meta_path, expected_identity=base)
        changed = dict(base)
        changed["stage1_deg_weight"] = 3.0
        assert not _can_reuse_stage1_cache(meta_path, expected_identity=changed)
        changed_ecs = dict(base)
        changed_ecs["stage1_ecs_enable"] = True
        assert not _can_reuse_stage1_cache(meta_path, expected_identity=changed_ecs)
        changed_ecs_sched = dict(base)
        changed_ecs_sched["stage1_ecs_sched_gamma"] = 0.8
        assert not _can_reuse_stage1_cache(meta_path, expected_identity=changed_ecs_sched)


def test_label_metrics_prefers_scib_alias_values_when_available():
    x = np.random.RandomState(0).randn(8, 3).astype(np.float32)
    clusters = np.array(["0", "0", "1", "1", "0", "0", "1", "1"])
    labels = np.array(["A", "A", "B", "B", "A", "A", "B", "B"])
    metrics = _label_metrics(
        x=x,
        clusters=clusters,
        labels=pd.Series(labels),
        label_key="label_cell_type",
        scib_metrics={
            "ARI_cluster/label": 0.91,
            "NMI_cluster/label": 0.92,
            "ASW_label": 0.93,
            "ASW_label/batch": 0.81,
            "PCR_batch": 0.12,
            "graph_conn": 0.99,
            "avg_bio": 0.92,
        },
    )
    assert metrics["ari_leiden_vs_label"] == 1.0
    assert metrics["ARI_cluster/label"] == 0.91
    assert metrics["NMI_cluster/label"] == 0.92
    assert metrics["ASW_label"] == 0.93
    assert metrics["avg_bio"] == 0.92
    assert metrics["avg_bio_source"] == "scib"


def main():
    monkeypatch = _MiniMonkeyPatch()
    test_scgpt_preprocessor_emits_expected_layers(monkeypatch)
    test_scgpt_preprocessor_falls_back_when_skmisc_missing(monkeypatch)
    test_prepare_pbmc_data_uses_scvi_source_and_scgpt_preprocess(monkeypatch)
    test_prepare_pbmc_data_can_disable_log1p_and_use_normed_input(monkeypatch)
    test_load_or_build_pbmc_deg_cache_returns_by_cell_type_keys()
    test_stage1_cache_identity_detects_stage1_deg_changes()
    test_label_metrics_prefers_scib_alias_values_when_available()
    print("test_stage1_latent_clustering: PASS")


if __name__ == "__main__":
    main()
