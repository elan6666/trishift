from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import pickle
import sys

import anndata as ad
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.gears._core import gears_eval_core as core


def _make_eval_adata(conditions: list[str]) -> ad.AnnData:
    n_genes = 6
    rows = []
    obs_cond = []
    rng = np.random.default_rng(0)
    for cond in conditions:
        n = 5 if cond == "ctrl" else 3
        rows.append(rng.random((n, n_genes), dtype=np.float32))
        obs_cond.extend([cond] * n)
    X = np.vstack(rows)
    obs = pd.DataFrame({"condition": obs_cond})
    var = pd.DataFrame({"gene_name": ["A", "B", "G1", "G2", "G3", "G4"]})
    adata = ad.AnnData(X=X, obs=obs, var=var)
    top20 = np.asarray([0, 1, 2, 3, 4], dtype=int)
    adata.uns["top20_degs_non_dropout"] = {
        cond: top20.copy() for cond in conditions if cond != "ctrl"
    }
    return adata


def test_build_test_queries_norman_and_adamson():
    class FakeModel:
        subgroup = {"test_subgroup": {"unseen_single": ["A+ctrl"], "seen0": ["A+B"]}}
        set2conditions = {"test": ["A", "B"]}

    norman_queries = core._build_test_queries(FakeModel(), "norman")
    assert norman_queries == [("A+ctrl", ["A"]), ("A+B", ["A", "B"])]

    adamson_queries = core._build_test_queries(FakeModel(), "adamson")
    assert adamson_queries == [("A+ctrl", ["A"]), ("B+ctrl", ["B"])]


def test_run_gears_eval_norman_and_adamson_with_fake_backend(tmp_path, monkeypatch):
    class FakePertData:
        def __init__(self, root: str):
            self.root = root
            self.loaded = None
            self.prepare_kwargs = None
            self.loader_kwargs = None

        def load(self, data_name: str):
            self.loaded = data_name

        def prepare_split(self, **kwargs):
            self.prepare_kwargs = kwargs

        def get_dataloader(self, **kwargs):
            self.loader_kwargs = kwargs

    norman_adata = _make_eval_adata(["ctrl", "A+ctrl", "A+B"])
    adamson_adata = _make_eval_adata(["ctrl", "A+ctrl", "B+ctrl"])
    norman_path = tmp_path / "norman.h5ad"
    adamson_path = tmp_path / "adamson.h5ad"
    norman_adata.write_h5ad(norman_path)
    adamson_adata.write_h5ad(adamson_path)

    gears_root = tmp_path / "Data_GEARS"
    (gears_root / "norman").mkdir(parents=True)
    (gears_root / "adamson").mkdir(parents=True)

    class FakeGEARS:
        def __init__(self, pert_data, **kwargs):
            self.pert_data = pert_data
            self.kwargs = kwargs
            self.node_map = {"A": 0, "B": 1, "G1": 2, "G2": 3, "G3": 4, "G4": 5}
            if pert_data.loaded == "norman":
                self.adata = norman_adata.copy()
                self.subgroup = {"test_subgroup": {"unseen_single": ["A+ctrl"], "seen0": ["A+B"]}}
                self.set2conditions = {"train": ["A+ctrl"]}
            else:
                self.adata = adamson_adata.copy()
                self.subgroup = {"test_subgroup": {}}
                self.set2conditions = {"test": ["A", "B"], "train": ["A"]}

        def model_initialize(self, hidden_size: int):
            self.hidden_size = hidden_size

        def train(self, epochs: int):
            self.epochs = epochs

        def predict(self, genes_batch):
            genes = genes_batch[0]
            key = "+".join(genes) if len(genes) > 1 else genes[0]
            vec = np.linspace(0.1, 0.6, 6, dtype=np.float32)
            return {key: vec}

    monkeypatch.setattr(core, "_require_gears_classes", lambda: (FakePertData, FakeGEARS))
    monkeypatch.setattr(core, "_resolve_gears_data_root", lambda name: gears_root)
    monkeypatch.setattr(
        core,
        "_resolve_eval_data_path",
        lambda name, cfg: norman_path if name == "norman" else adamson_path,
    )
    monkeypatch.setattr(
        core,
        "subgroup",
        lambda pert_list, seed: pd.DataFrame(
            {"group": ["train", "test"], "subgroup": ["single", "seen1"]},
            index=["A+ctrl", "A+B"],
        ),
    )
    monkeypatch.setattr(core, "ROOT", tmp_path)
    monkeypatch.setitem(
        core.DATASET_CONFIG,
        "norman",
        replace(core.DATASET_CONFIG["norman"], splits=[1]),
    )
    monkeypatch.setitem(
        core.DATASET_CONFIG,
        "adamson",
        replace(core.DATASET_CONFIG["adamson"], splits=[1]),
    )

    core.run_gears_eval("norman", export_notebook_pkl=True)
    norman_out = tmp_path / "artifacts" / "results" / "gears" / "norman"
    norman_metrics = pd.read_csv(norman_out / "metrics.csv")
    assert "subgroup" in norman_metrics.columns
    assert set(norman_metrics["condition"].astype(str)) == {"A+ctrl", "A+B"}
    assert set(norman_metrics["n_ensemble"].astype(int)) == {1}
    with (norman_out / "gears_norman_1.pkl").open("rb") as f:
        norman_payload = pickle.load(f)
    assert set(norman_payload.keys()) == {"A+ctrl", "A+B"}
    assert norman_payload["A+ctrl"]["Pred"].shape == (1, 4)

    core.run_gears_eval("adamson", export_notebook_pkl=True)
    adamson_out = tmp_path / "artifacts" / "results" / "gears" / "adamson"
    adamson_metrics = pd.read_csv(adamson_out / "metrics.csv")
    assert set(adamson_metrics["condition"].astype(str)) == {"A+ctrl", "B+ctrl"}
    assert set(adamson_metrics["n_ensemble"].astype(int)) == {1}
    with (adamson_out / "gears_adamson_1.pkl").open("rb") as f:
        adamson_payload = pickle.load(f)
    assert set(adamson_payload.keys()) == {"A+ctrl", "B+ctrl"}
    assert adamson_payload["A+ctrl"]["Pred"].shape[0] == 1
