from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import pickle
import sys

import anndata as ad
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.scgpt._core import scgpt_eval_core as core
from scripts.trishift.analysis import _result_adapter as adapter
from trishift.TriShiftData import TriShiftData


def _make_eval_adata(conditions: list[str]) -> ad.AnnData:
    pert_genes = sorted({tok for cond in conditions for tok in str(cond).split("+") if tok and tok != "ctrl"})
    extra_genes = ["G1", "G2", "G3"]
    gene_names = pert_genes + extra_genes
    n_genes = len(gene_names)
    rows = []
    obs_cond = []
    rng = np.random.default_rng(7)
    for cond in conditions:
        n = 5 if cond == "ctrl" else 3
        rows.append(rng.random((n, n_genes), dtype=np.float32))
        obs_cond.extend([cond] * n)
    X = np.vstack(rows)
    obs = pd.DataFrame({"condition": obs_cond})
    var = pd.DataFrame({"gene_name": gene_names})
    adata = ad.AnnData(X=X, obs=obs, var=var)
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    deg_map = {}
    for cond in conditions:
        if cond == "ctrl":
            continue
        cond_tokens = [tok for tok in str(cond).split("+") if tok and tok != "ctrl"]
        degs = [gene_to_idx[tok] for tok in cond_tokens if tok in gene_to_idx]
        degs.extend([gene_to_idx["G1"], gene_to_idx["G2"]])
        deg_map[str(cond)] = np.asarray(degs, dtype=int)
    adata.uns["top20_degs_non_dropout"] = deg_map
    return adata


def test_build_split_dict_matches_shared_logic():
    genes = list("ABCDEFGHIJKLMNO")
    singles = [f"{g}+ctrl" for g in genes]
    combos = []
    for i in range(len(genes)):
        for j in range(i + 1, len(genes)):
            combos.append(f"{genes[i]}+{genes[j]}")
    norman_adata = _make_eval_adata(["ctrl", *singles, *combos])
    data = TriShiftData(norman_adata, core._dummy_embedding_df())
    split_seed = None
    expected = None
    for seed in range(1, 21):
        try:
            expected = core.subgroup(list(norman_adata.obs["condition"].astype(str).unique()), seed=seed)
        except ValueError:
            continue
        split_seed = seed
        break
    assert split_seed is not None
    split_dict, subgroup_df = core._build_split_dict("norman", data, split_id=int(split_seed), test_ratio=0.2)
    assert subgroup_df is not None
    assert set(split_dict["test_conds"]) == set(expected[expected.group == "test"].index)
    assert set(split_dict["val_conds"]) == set(expected[expected.group == "val"].index)

    adamson_adata = _make_eval_adata(["ctrl", "A+ctrl", "B+ctrl", "C+ctrl"])
    adamson_data = TriShiftData(adamson_adata, core._dummy_embedding_df())
    split_dict_adamson, subgroup_df_adamson = core._build_split_dict("adamson", adamson_data, split_id=2, test_ratio=0.2)
    expected_adamson = adamson_data.split_by_condition(seed=2, test_ratio=0.2)
    assert subgroup_df_adamson is None
    assert set(split_dict_adamson["test_conds"]) == set(expected_adamson["test_conds"])
    assert set(split_dict_adamson["val_conds"]) == set(expected_adamson["val_conds"])


def test_resolve_pretrained_root_requires_configured_entry(tmp_path, monkeypatch):
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "paths.yaml").write_text("datasets: {}\nembeddings: {}\n", encoding="utf-8")
    monkeypatch.setattr(core, "ROOT", tmp_path)
    try:
        core._resolve_pretrained_root()
    except FileNotFoundError as exc:
        assert "scgpt_pretrained_root" in str(exc)
    else:
        raise AssertionError("expected FileNotFoundError for missing scgpt_pretrained_root")


def test_run_scgpt_eval_writes_metrics_and_payload(tmp_path, monkeypatch):
    adata = _make_eval_adata(["ctrl", "A+ctrl", "B+ctrl", "C+ctrl", "A+B"])
    h5ad_path = tmp_path / "norman.h5ad"
    adata.write_h5ad(h5ad_path)

    class FakeLoader(list):
        def __init__(self, items, *args, **kwargs):
            super().__init__(items)

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))

    fake_stack = {
        "Data": object,
        "DataLoader": FakeLoader,
        "map_raw_id_to_vocab_id": lambda raw_ids, gene_ids: raw_ids,
        "masked_mse_loss": lambda output, target, mask: torch.tensor(0.0),
        "GeneVocab": object,
        "TransformerGenerator": object,
    }
    seen: dict[str, object] = {}

    def fake_graphs(*, split_adata, ctrl_adata, gene_names, data_cls, samples_per_cell=1):
        seen.setdefault("graph_conditions", []).append(sorted(set(split_adata.obs["condition"].astype(str).tolist())))
        assert ctrl_adata.n_obs > 0
        return [object()]

    def fake_train_model(**kwargs):
        seen["train_gene_ids"] = kwargs["gene_ids"].tolist()
        return kwargs["model"]

    def fake_compute(**kwargs):
        split_dict = kwargs["split_dict"]
        conds = list(split_dict["test_conds"])
        payload = {}
        rows = []
        for cond in conds:
            rows.append(
                {
                    "condition": cond,
                    "pearson": 0.8,
                    "nmse": 0.2,
                    "mse_pred": 0.1,
                    "mse_ctrl": 0.3,
                    "deg_mean_r2": 0.6,
                    "systema_corr_20de_allpert": 0.5,
                    "systema_corr_deg_r2": 0.4,
                    "scpram_r2_degs_mean_mean": 0.7,
                    "scpram_r2_degs_var_mean": 0.2,
                    "scpram_wasserstein_degs_sum": 1.0,
                    "split_id": int(kwargs["split_id"]),
                    "n_ensemble": 3,
                }
            )
            payload[cond] = {
                "Pred": np.ones((3, 2), dtype=np.float32),
                "Ctrl": np.ones((3, 2), dtype=np.float32),
                "Truth": np.ones((3, 2), dtype=np.float32),
                "Pred_full": np.ones((3, 6), dtype=np.float32),
                "Ctrl_full": np.ones((3, 6), dtype=np.float32),
                "Truth_full": np.ones((3, 6), dtype=np.float32),
                "DE_idx": np.array([3, 4], dtype=int),
                "DE_name": np.array(["G1", "G2"], dtype=object),
                "gene_name_full": np.array(["A", "B", "C", "G1", "G2", "G3"], dtype=object),
            }
        return pd.DataFrame(rows), payload

    monkeypatch.setattr(core, "ROOT", tmp_path)
    monkeypatch.setattr(core, "_require_scgpt_stack", lambda: fake_stack)
    monkeypatch.setattr(core, "_resolve_eval_data_path", lambda name, cfg: h5ad_path)
    monkeypatch.setattr(core, "_resolve_pretrained_root", lambda: tmp_path / "pretrained")
    monkeypatch.setattr(core, "_prepare_vocab_and_gene_ids", lambda adata, pretrained_root, stack: np.arange(adata.n_vars, dtype=int))
    monkeypatch.setattr(core, "_build_model", lambda pretrained_root, stack, cfg: FakeModel())
    monkeypatch.setattr(core, "_build_train_eval_graphs", fake_graphs)
    monkeypatch.setattr(core, "_train_model", fake_train_model)
    monkeypatch.setattr(core, "_compute_metrics_and_export_payload", fake_compute)
    monkeypatch.setattr(
        core,
        "subgroup",
        lambda pert_list, seed: pd.DataFrame(
            {
                "group": ["train", "val", "test", "test"],
                "subgroup": ["single", "single", "single", "seen1"],
            },
            index=["A+ctrl", "B+ctrl", "C+ctrl", "A+B"],
        ),
    )
    monkeypatch.setitem(
        core.DATASET_CONFIG,
        "norman",
        replace(core.DATASET_CONFIG["norman"], splits=[1]),
    )

    core.run_scgpt_eval("norman", export_notebook_pkl=True)

    out_dir = tmp_path / "artifacts" / "results" / "scgpt" / "norman"
    metrics = pd.read_csv(out_dir / "metrics.csv")
    assert "subgroup" in metrics.columns
    with (out_dir / "scgpt_norman_1.pkl").open("rb") as f:
        payload = pickle.load(f)
    assert payload
    assert set(metrics["condition"].astype(str)) == set(payload.keys())
    first = next(iter(payload.values()))
    assert first["Pred_full"].shape == (3, 6)
    assert len(first["gene_name_full"]) == 6


def test_result_adapter_supports_scgpt(monkeypatch, tmp_path):
    result_root = tmp_path / "scgpt"
    out_dir = result_root / "toy"
    out_dir.mkdir(parents=True)
    pd.DataFrame([{"condition": "A+ctrl", "split_id": 1, "pearson": 0.9}]).to_csv(out_dir / "metrics.csv", index=False)
    (out_dir / "mean_pearson.txt").write_text("0.9\nmean_pearson=0.9\n", encoding="utf-8")
    with (out_dir / "scgpt_toy_1.pkl").open("wb") as f:
        pickle.dump({"A+ctrl": {"Pred": np.ones((1, 1), dtype=np.float32)}}, f)

    monkeypatch.setitem(adapter.DEFAULT_PAYLOAD_ROOTS, "scgpt", result_root)
    resolved = adapter.resolve_result(dataset="toy", model_name="scgpt")
    assert resolved.metrics_path.name == "metrics.csv"
    assert resolved.available_split_ids == [1]


def test_main_profile_passes_task_args(monkeypatch, tmp_path):
    profile_dir = tmp_path / "profiles"
    profile_dir.mkdir(parents=True)
    (profile_dir / "adamson.yaml").write_text(
        "\n".join(
            [
                "dataset: adamson",
                "task_args:",
                "  seed: 9",
                "  export_notebook_pkl: false",
                "  control_pool_size: 11",
                "  epochs: 3",
                "  batch_size: 5",
                "  eval_batch_size: 7",
                "  lr: 0.002",
                "  early_stop: 4",
            ]
        ),
        encoding="utf-8",
    )
    seen = {}

    def fake_run(name, **kwargs):
        seen["name"] = name
        seen.update(kwargs)

    monkeypatch.setattr(core, "PROFILE_DIR", profile_dir)
    monkeypatch.setattr(core, "run_scgpt_eval", fake_run)
    core.main(["--profile", "adamson"])
    assert seen == {
        "name": "adamson",
        "base_seed": 9,
        "export_notebook_pkl": False,
        "control_pool_size": 11,
        "epochs": 3,
        "batch_size": 5,
        "eval_batch_size": 7,
        "lr": 0.002,
        "early_stop": 4,
    }


def main() -> int:
    import pytest

    return int(pytest.main([__file__, "-q"]))


if __name__ == "__main__":
    raise SystemExit(main())
