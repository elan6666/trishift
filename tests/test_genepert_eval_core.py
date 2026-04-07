from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import pickle
import sys

import anndata as ad
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.genepert._core import genepert_eval_core as core


def _make_eval_adata(conditions: list[str]) -> ad.AnnData:
    n_genes = 6
    rows = []
    obs_cond = []
    rng = np.random.default_rng(2)
    for cond in conditions:
        n = 5 if cond == "ctrl" else 3
        rows.append(rng.random((n, n_genes), dtype=np.float32))
        obs_cond.extend([cond] * n)
    X = np.vstack(rows)
    obs = pd.DataFrame({"condition": obs_cond})
    var = pd.DataFrame({"gene_name": ["KIAA1804", "A", "B", "G1", "G2", "G3"]})
    var.index = np.array(
        ["raw_KIAA1804", "raw_A", "raw_B", "raw_G1", "raw_G2", "raw_G3"],
        dtype=object,
    )
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.uns["top_non_dropout_de_20"] = {
        "A549_KIAA1804+ctrl_1+1": np.array(
            ["raw_KIAA1804", "raw_G1", "raw_G2"],
            dtype=object,
        ),
        "A549_A+B_1+1": np.array(["raw_A", "raw_B", "raw_G1", "raw_G2"], dtype=object),
        "A549_A+ctrl_1+1": np.array(["raw_A", "raw_G1", "raw_G2"], dtype=object),
        "A549_B+ctrl_1+1": np.array(["raw_B", "raw_G1", "raw_G2"], dtype=object),
    }
    return adata


def test_select_alpha_prefers_corr_then_mse_then_alpha():
    result = {
        "aggregate": {
            "ridge": {
                (("alpha", 10.0),): {"corr": 0.9, "mse": 2.0},
                (("alpha", 1.0),): {"corr": 0.9, "mse": 1.0},
                (("alpha", 0.5),): {"corr": 0.9, "mse": 1.0},
                (("alpha", 0.1),): {"corr": 0.8, "mse": 0.1},
            }
        }
    }
    assert core._select_alpha_from_result(result) == 0.5


def test_run_genepert_eval_norman_alias_and_export(tmp_path, monkeypatch):
    adata = _make_eval_adata(
        ["ctrl", "A+ctrl", "B+ctrl", "KIAA1804+ctrl", "A+B", "MISSING+ctrl"]
    )
    data_path = tmp_path / "norman.h5ad"
    adata.write_h5ad(data_path)

    calls: list[tuple[list[str], list[str]]] = []

    class FakeGenePertExperiment:
        def __init__(self, embeddings):
            self.embeddings = embeddings
            self.adata = None
            self.mean_expression = None

        def run_experiment_with_adata(
            self,
            adata_train,
            adata_test,
            ridge_params=None,
            knn_params=None,
        ):
            train_conds = sorted(set(adata_train.obs["condition"].astype(str).tolist()))
            test_conds = sorted(set(adata_test.obs["condition"].astype(str).tolist()))
            calls.append((train_conds, test_conds))
            aggregate = {"ridge": {}}
            per_gene = {}
            for ridge_param in ridge_params or []:
                alpha = float(ridge_param["alpha"])
                corr = 0.90 if alpha == 1.0 else 0.70
                mse_val = 0.10 if alpha == 1.0 else 0.30
                key = (("alpha", alpha),)
                aggregate["ridge"][key] = {"corr": corr, "mse": mse_val, "mae": 0.0}
                for cond in test_conds:
                    clean = core._clean_condition(cond)
                    y_test = np.asarray(
                        adata_test[adata_test.obs["condition"] == cond].X.mean(axis=0)
                    ).reshape(-1)
                    y_pred = y_test + alpha * 0.01
                    entry = per_gene.setdefault(clean, {"ridge": {}})
                    entry["ridge"][key] = (corr, mse_val, y_pred, y_test, 0.0, 0.0)
            return {"aggregate": aggregate, "per_gene": per_gene}

    monkeypatch.setattr(core, "_require_genepert_class", lambda: FakeGenePertExperiment)
    monkeypatch.setattr(core, "_resolve_eval_data_path", lambda name, cfg: data_path)
    monkeypatch.setattr(
        core,
        "_load_embedding_dict",
        lambda emb_path: {
            "A": np.array([1.0, 0.0], dtype=np.float32),
            "B": np.array([0.0, 1.0], dtype=np.float32),
            "MAP3K21": np.array([0.5, 0.5], dtype=np.float32),
        },
    )
    monkeypatch.setattr(core, "ROOT", tmp_path)
    monkeypatch.setattr(
        core,
        "subgroup",
        lambda pert_list, seed: pd.DataFrame(
            {
                "group": ["train", "val", "test", "test"],
                "subgroup": ["single", "single", "single", "seen1"],
            },
            index=["A+ctrl", "B+ctrl", "KIAA1804+ctrl", "A+B"],
        ),
    )
    monkeypatch.setitem(
        core.DATASET_CONFIG,
        "norman",
        replace(core.DATASET_CONFIG["norman"], splits=[1]),
    )

    core.run_genepert_eval("norman", export_notebook_pkl=True, alpha_grid=[0.1, 1.0])

    out_dir = tmp_path / "artifacts" / "results" / "genepert" / "norman"
    metrics = pd.read_csv(out_dir / "metrics.csv")
    assert "subgroup" in metrics.columns
    assert set(metrics["condition"].astype(str)) == {"KIAA1804+ctrl", "A+B"}
    assert "MISSING+ctrl" not in set(metrics["condition"].astype(str))
    assert set(metrics["n_ensemble"].astype(int)) == {3}

    with (out_dir / "genepert_norman_1.pkl").open("rb") as handle:
        payload = pickle.load(handle)
    assert set(payload.keys()) == {"KIAA1804+ctrl", "A+B"}
    assert payload["KIAA1804+ctrl"]["Pred"].shape == (3, 2)
    assert payload["KIAA1804+ctrl"]["Ctrl"].shape == (3, 2)
    assert payload["KIAA1804+ctrl"]["Truth"].shape == (3, 2)
    assert payload["KIAA1804+ctrl"]["Pred_full"].shape == (3, 6)
    assert payload["KIAA1804+ctrl"]["Ctrl_full"].shape == (3, 6)
    assert payload["KIAA1804+ctrl"]["Truth_full"].shape == (3, 6)
    assert len(payload["KIAA1804+ctrl"]["gene_name_full"]) == 6
    np.testing.assert_allclose(
        payload["KIAA1804+ctrl"]["Pred_full"][:, payload["KIAA1804+ctrl"]["DE_idx"]],
        payload["KIAA1804+ctrl"]["Pred"],
    )
    assert all("MISSING+ctrl" not in test for _, test in calls)
    assert calls[0][1] == ["B+ctrl"]
    assert set(calls[1][1]) == {"A+B", "KIAA1804+ctrl"}


def test_main_profile_passes_alpha_grid(monkeypatch, tmp_path):
    profile_dir = tmp_path / "profiles"
    profile_dir.mkdir(parents=True)
    profile_path = profile_dir / "adamson.yaml"
    profile_path.write_text(
        "\n".join(
            [
                "dataset: adamson",
                "task_args:",
                "  seed: 9",
                "  export_notebook_pkl: false",
                "  alpha_grid: [0.5, 2.0]",
            ]
        ),
        encoding="utf-8",
    )
    seen = {}

    def fake_run(name, base_seed=24, export_notebook_pkl=True, alpha_grid=None):
        seen["name"] = name
        seen["base_seed"] = base_seed
        seen["export_notebook_pkl"] = export_notebook_pkl
        seen["alpha_grid"] = alpha_grid

    monkeypatch.setattr(core, "PROFILE_DIR", profile_dir)
    monkeypatch.setattr(core, "run_genepert_eval", fake_run)

    core.main(["--profile", "adamson"])
    assert seen == {
        "name": "adamson",
        "base_seed": 9,
        "export_notebook_pkl": False,
        "alpha_grid": [0.5, 2.0],
    }


def main() -> int:
    import pytest

    return int(pytest.main([__file__, "-q"]))


if __name__ == "__main__":
    raise SystemExit(main())
