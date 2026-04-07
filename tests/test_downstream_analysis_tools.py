from __future__ import annotations

import pickle
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from tests._helpers import temp_dir
from scripts.trishift.analysis import _result_adapter as adapter
from scripts.trishift.analysis import baseline_panel as baseline_panel_mod
from scripts.trishift.analysis import pathway_recovery as pathway_recovery_mod
from scripts.trishift.analysis import split_structure_recovery as structure_mod
from scripts.trishift.analysis import stratified_benchmark as stratified_mod
from scripts.trishift.analysis.baseline_panel import run_baseline_panel
from scripts.trishift.analysis.pathway_recovery import run_pathway_recovery
from scripts.trishift.analysis.split_structure_recovery import run_split_structure_recovery
from scripts.trishift.analysis.stratified_benchmark import run_stratified_benchmark
from scripts.trishift.analysis import deg20_experiment as deg20_mod


def _write_pickle(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f)


def _write_metrics(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _full_payload(condition: str = "A+ctrl") -> dict:
    genes = np.array(["A", "B", "G0", "G1"], dtype=object)
    ctrl = np.ones((4, 4), dtype=np.float32)
    truth = ctrl.copy()
    pred = ctrl.copy()
    truth[:, 2] += 3.0
    truth[:, 3] += 2.0
    pred[:, 2] += 2.8
    pred[:, 3] += 2.2
    return {
        condition: {
            "Pred": pred[:, :2],
            "Truth": truth[:, :2],
            "Ctrl": ctrl[:, :2],
            "Pred_full": pred,
            "Truth_full": truth,
            "Ctrl_full": ctrl,
            "gene_name_full": genes,
            "DE_idx": np.array([2, 3], dtype=int),
            "DE_name": np.array(["G0", "G1"], dtype=object),
        }
    }


def _norman_combo_payload() -> dict:
    genes = np.array(["A", "B", "G0", "G1"], dtype=object)
    ctrl = np.ones((4, 4), dtype=np.float32)

    def item(delta_a: float, delta_b: float) -> dict:
        truth = ctrl.copy()
        pred = ctrl.copy()
        truth[:, 2] += delta_a
        truth[:, 3] += delta_b
        pred[:, 2] += delta_a + 0.1
        pred[:, 3] += delta_b - 0.1
        return {
            "Pred_full": pred,
            "Truth_full": truth,
            "Ctrl_full": ctrl,
            "gene_name_full": genes,
            "DE_idx": np.array([2, 3], dtype=int),
            "DE_name": np.array(["G0", "G1"], dtype=object),
        }

    payload = {
        "A+ctrl": item(1.0, 0.5),
        "B+ctrl": item(0.4, 1.2),
        "A+B": item(2.0, 2.3),
    }
    return payload


def _norman_combo_payload_deg_only() -> dict:
    genes = np.array(["G0", "G1", "G2"], dtype=object)
    ctrl = np.ones((4, 3), dtype=np.float32)

    def item(delta: tuple[float, float, float]) -> dict:
        truth = ctrl.copy()
        pred = ctrl.copy()
        truth += np.asarray(delta, dtype=np.float32)
        pred += np.asarray([delta[0] + 0.1, delta[1] - 0.1, delta[2] + 0.05], dtype=np.float32)
        return {
            "Pred": pred,
            "Truth": truth,
            "Ctrl": ctrl,
            "DE_idx": np.array([0, 1, 2], dtype=int),
            "DE_name": genes,
        }

    return {
        "A+ctrl": item((1.0, 0.5, 0.0)),
        "B+ctrl": item((0.4, 1.2, 0.2)),
        "A+B": item((2.0, 2.3, 0.5)),
    }


def _norman_combo_payload_missing_single() -> dict:
    payload = _norman_combo_payload()
    payload.pop("B+ctrl", None)
    return payload


def _norman_combo_payload_multi() -> dict:
    genes = np.array(["A", "B", "C", "G0", "G1"], dtype=object)
    ctrl = np.ones((4, 5), dtype=np.float32)

    def item(delta: tuple[float, float]) -> dict:
        truth = ctrl.copy()
        pred = ctrl.copy()
        truth[:, 3] += float(delta[0])
        truth[:, 4] += float(delta[1])
        pred[:, 3] += float(delta[0]) + 0.05
        pred[:, 4] += float(delta[1]) - 0.05
        return {
            "Pred_full": pred,
            "Truth_full": truth,
            "Ctrl_full": ctrl,
            "gene_name_full": genes,
            "DE_idx": np.array([3, 4], dtype=int),
            "DE_name": np.array(["G0", "G1"], dtype=object),
        }

    return {
        "A+ctrl": item((1.0, 0.5)),
        "B+ctrl": item((0.4, 1.2)),
        "C+ctrl": item((0.8, 0.3)),
        "A+B": item((2.0, 2.3)),
        "A+C": item((1.8, 1.7)),
    }


def _write_embedding_pickle(path: Path) -> None:
    obj = {
        "A": np.array([1.0, 0.0], dtype=np.float32),
        "B": np.array([0.0, 1.0], dtype=np.float32),
        "C": np.array([1.0, 1.0], dtype=np.float32),
    }
    _write_pickle(path, obj)


def _write_dense_embedding_pickle(path: Path) -> None:
    obj = {
        "A": np.array([1.0, 0.0], dtype=np.float32),
        "B": np.array([0.0, 1.0], dtype=np.float32),
        "C": np.array([1.0, 1.0], dtype=np.float32),
        "D": np.array([0.5, 1.5], dtype=np.float32),
        "E": np.array([1.5, 0.5], dtype=np.float32),
    }
    _write_pickle(path, obj)


def _write_h5ad(path: Path) -> None:
    X = np.ones((20, 4), dtype=np.float32)
    obs = pd.DataFrame(
        {
            "condition": ["ctrl"] * 8 + ["A"] * 4 + ["B"] * 4 + ["C"] * 4,
        }
    )
    var = pd.DataFrame({"gene_name": ["A", "B", "G0", "G1"]})
    ad.AnnData(X=X, obs=obs, var=var).write_h5ad(path)


def _write_norman_truth_h5ad(path: Path) -> None:
    genes = ["G0", "G1", "G2", "G3"]
    rows = []
    conditions = [
        ("ctrl", (0.0, 0.0, 0.0, 0.0)),
        ("A+ctrl", (1.2, 0.3, 0.1, 0.0)),
        ("B+ctrl", (0.4, 1.1, 0.2, 0.0)),
        ("C+ctrl", (0.8, 0.2, 0.9, 0.1)),
        ("D+ctrl", (0.1, 0.6, 0.2, 1.0)),
        ("A+B", (1.9, 1.8, 0.5, 0.0)),
        ("A+C", (2.0, 0.6, 1.4, 0.1)),
        ("B+C", (0.9, 1.3, 1.1, 0.0)),
        ("C+D", (0.8, 0.9, 1.2, 1.4)),
    ]
    X = []
    obs_rows = []
    for condition, delta in conditions:
        for _ in range(3):
            X.append(np.asarray(delta, dtype=np.float32))
            obs_rows.append(
                {
                    "condition": condition,
                    "cell_type": "A549",
                    "dose_val": "1+1",
                    "control": 1 if condition == "ctrl" else 0,
                    "condition_name": f"A549_{condition}_1+1",
                }
            )
    ad.AnnData(X=np.asarray(X, dtype=np.float32), obs=pd.DataFrame(obs_rows), var=pd.DataFrame({"gene_name": genes})).write_h5ad(path)


def _dense_norman_condition_deltas() -> dict[str, np.ndarray]:
    singles = {
        "A+ctrl": np.array([1.0, 0.1, 0.2, 0.0], dtype=np.float32),
        "B+ctrl": np.array([0.1, 1.1, 0.3, 0.2], dtype=np.float32),
        "C+ctrl": np.array([0.3, 0.2, 1.0, 0.4], dtype=np.float32),
        "D+ctrl": np.array([0.2, 0.4, 0.1, 1.0], dtype=np.float32),
        "E+ctrl": np.array([0.5, 0.3, 0.6, 0.2], dtype=np.float32),
    }
    combos: dict[str, np.ndarray] = {}
    genes = ["A", "B", "C", "D", "E"]
    for i, gene_a in enumerate(genes):
        for j, gene_b in enumerate(genes):
            if i >= j:
                continue
            cond = adapter.normalize_condition(f"{gene_a}+{gene_b}")
            a = singles[f"{gene_a}+ctrl"]
            b = singles[f"{gene_b}+ctrl"]
            interaction = np.array(
                [
                    0.15 * ((i + j) % 3 + 1),
                    0.03 * (i + 1),
                    -0.02 * (j + 1),
                    0.04 * ((i + 2 * j) % 2),
                ],
                dtype=np.float32,
            )
            combos[cond] = a + b + interaction
    return {**singles, **combos}


def _write_dense_norman_truth_h5ad(path: Path) -> None:
    feature_names = ["A", "B", "C", "D", "E", "G0", "G1", "G2", "G3"]
    ctrl = np.ones((3, len(feature_names)), dtype=np.float32)
    deltas = _dense_norman_condition_deltas()
    X = []
    obs_rows = []
    for _ in range(5):
        X.append(ctrl[0].copy())
        obs_rows.append({"condition": "ctrl"})
    for condition, delta in deltas.items():
        row = ctrl[0].copy()
        row[5:] += np.asarray(delta, dtype=np.float32)
        for _ in range(3):
            X.append(row.copy())
            obs_rows.append({"condition": condition})
    ad.AnnData(
        X=np.asarray(X, dtype=np.float32),
        obs=pd.DataFrame(obs_rows),
        var=pd.DataFrame({"gene_name": feature_names}),
    ).write_h5ad(path)


def _dense_norman_payload() -> dict:
    feature_names = np.array(["A", "B", "C", "D", "E", "G0", "G1", "G2", "G3"], dtype=object)
    ctrl = np.ones((4, len(feature_names)), dtype=np.float32)
    deltas = _dense_norman_condition_deltas()
    payload: dict[str, dict] = {}
    for condition, delta in deltas.items():
        truth = ctrl.copy()
        pred = ctrl.copy()
        truth[:, 5:] += np.asarray(delta, dtype=np.float32)
        pred[:, 5:] += (np.asarray(delta, dtype=np.float32) * 0.97) + 0.02
        payload[condition] = {
            "Pred_full": pred,
            "Truth_full": truth,
            "Ctrl_full": ctrl,
            "gene_name_full": feature_names,
            "DE_idx": np.array([5, 6, 7, 8], dtype=int),
            "DE_name": np.array(["G0", "G1", "G2", "G3"], dtype=object),
        }
    return payload


def test_run_baseline_panel_handles_payload_and_systema(monkeypatch):
    with temp_dir() as tmp:
        root = Path(tmp)
        tri_root = root / "tri"
        scouter_root = root / "scouter"
        systema_root = root / "systema_baselines_20260101_000000"

        _write_metrics(
            tri_root / "toy" / "metrics_nearest.csv",
            [
                {"condition": "A+ctrl", "split_id": 1, "pearson": 0.9, "nmse": 0.2, "deg_mean_r2": 0.7, "systema_corr_20de_allpert": 0.6, "systema_corr_deg_r2": 0.5, "scpram_r2_degs_mean_mean": 0.9, "scpram_r2_degs_var_mean": 0.4, "scpram_wasserstein_degs_sum": 3.0, "subgroup": "single"},
            ],
        )
        _write_metrics(
            scouter_root / "toy" / "metrics.csv",
            [
                {"condition": "A+ctrl", "split_id": 1, "pearson": 0.8, "nmse": 0.3, "deg_mean_r2": 0.6, "systema_corr_20de_allpert": 0.5, "systema_corr_deg_r2": 0.4, "scpram_r2_degs_mean_mean": 0.8, "scpram_r2_degs_var_mean": 0.3, "scpram_wasserstein_degs_sum": 4.0, "subgroup": "single"},
            ],
        )
        _write_metrics(
            systema_root / "systema_nonctl_mean" / "metrics.csv",
            [
                {"condition": "A+ctrl", "split_id": 1, "pearson": 0.7, "nmse": 0.4, "deg_mean_r2": 0.5, "systema_corr_20de_allpert": 0.3, "systema_corr_deg_r2": 0.2, "scpram_r2_degs_mean_mean": np.nan, "scpram_r2_degs_var_mean": np.nan, "scpram_wasserstein_degs_sum": np.nan, "subgroup": "single"},
            ],
        )

        monkeypatch.setitem(adapter.DEFAULT_PAYLOAD_ROOTS, "trishift", tri_root)
        monkeypatch.setitem(adapter.DEFAULT_PAYLOAD_ROOTS, "scouter", scouter_root)
        result = run_baseline_panel(
            dataset="toy",
            models=["trishift_nearest", "scouter", "systema_nonctl_mean"],
            split_ids="1",
            out_root=root / "out",
            systema_root=systema_root,
        )
        assert len(result["summary_df"]) == 3
        assert (result["out_dir"] / "baseline_panel_summary.csv").exists()
        assert (result["out_dir"] / "baseline_panel_heatmap.png").exists()


def test_run_pathway_recovery_export_only_builds_gene_lists(monkeypatch):
    with temp_dir() as tmp:
        root = Path(tmp)
        tri_root = root / "tri"
        _write_pickle(tri_root / "toy" / "trishift_toy_1_nearest.pkl", _full_payload())

        monkeypatch.setitem(deg20_mod.DEFAULT_RESULT_ROOTS, "trishift", tri_root)
        result = run_pathway_recovery(
            dataset="toy",
            models=["trishift_nearest"],
            split_ids="1",
            out_root=root / "out",
            enrichment_mode="export_only",
        )
        assert not result["gene_lists_df"].empty
        assert result["enrichment_df"].empty
        assert (result["out_dir"] / "pathway_enrichment_all.csv").exists()
        assert (result["out_dir"] / "pathway_nes_scatter.png").exists()


def test_run_stratified_benchmark_builds_metadata_and_summary(monkeypatch):
    with temp_dir() as tmp:
        root = Path(tmp)
        tri_root = root / "tri"
        notebook_cwd = root / "notebooks"
        paths_yaml = root / "paths.yaml"
        h5ad_path = root / "toy.h5ad"
        emb_path = root / "emb.pkl"
        notebook_cwd.mkdir()

        _write_h5ad(h5ad_path)
        _write_embedding_pickle(emb_path)
        paths_yaml.write_text(
            (
                "datasets:\n"
                f"  adamson: \"{str(h5ad_path).replace('\\', '\\\\')}\"\n"
                "embeddings:\n"
                f"  emb_b: \"{str(emb_path).replace('\\', '\\\\')}\"\n"
            ),
            encoding="utf-8",
        )

        payload = {
            "A+ctrl": _full_payload("A+ctrl")["A+ctrl"],
            "B+ctrl": _full_payload("B+ctrl")["B+ctrl"],
        }
        _write_pickle(tri_root / "adamson" / "trishift_adamson_1_nearest.pkl", payload)
        _write_metrics(
            tri_root / "adamson" / "metrics_nearest.csv",
            [
                {"condition": "A+ctrl", "split_id": 1, "pearson": 0.9, "nmse": 0.2, "deg_mean_r2": 0.7, "systema_corr_20de_allpert": 0.6, "systema_corr_deg_r2": 0.5, "scpram_r2_degs_var_mean": 0.4},
                {"condition": "B+ctrl", "split_id": 1, "pearson": 0.8, "nmse": 0.3, "deg_mean_r2": 0.6, "systema_corr_20de_allpert": 0.5, "systema_corr_deg_r2": 0.4, "scpram_r2_degs_var_mean": 0.3},
            ],
        )

        monkeypatch.setitem(adapter.DEFAULT_PAYLOAD_ROOTS, "trishift", tri_root)
        old_cwd = Path.cwd()
        monkeypatch.chdir(notebook_cwd)
        result = run_stratified_benchmark(
            dataset="adamson",
            models=["trishift_nearest"],
            split_ids="1",
            out_root=root / "out",
            paths_path=paths_yaml,
        )
        monkeypatch.chdir(old_cwd)
        assert not result["metadata_df"].empty
        assert not result["stratified_df"].empty
        assert (result["out_dir"] / "stratified_summary.csv").exists()
        assert (result["out_dir"] / "difficulty_scatter.png").exists()


def test_run_split_structure_recovery_builds_panel(monkeypatch):
    with temp_dir() as tmp:
        root = Path(tmp)
        tri_root = root / "tri"
        gears_root = root / "gears"
        notebook_cwd = root / "notebooks"
        paths_yaml = root / "paths.yaml"
        h5ad_path = root / "norman.h5ad"
        emb_path = root / "emb.pkl"
        notebook_cwd.mkdir()

        _write_dense_norman_truth_h5ad(h5ad_path)
        _write_dense_embedding_pickle(emb_path)
        paths_yaml.write_text(
            (
                "datasets:\n"
                f"  norman: \"{str(h5ad_path).replace('\\', '\\\\')}\"\n"
                "embeddings:\n"
                f"  emb_a: \"{str(emb_path).replace('\\', '\\\\')}\"\n"
            ),
            encoding="utf-8",
        )

        payload_tri = _dense_norman_payload()
        payload_gears = _dense_norman_payload()
        for item in payload_gears.values():
            item["Pred_full"] = np.asarray(item["Pred_full"], dtype=np.float32) * 0.985

        _write_pickle(tri_root / "norman" / "trishift_norman_1_nearest.pkl", payload_tri)
        _write_pickle(gears_root / "norman" / "gears_norman_1.pkl", payload_gears)

        monkeypatch.setitem(adapter.DEFAULT_PAYLOAD_ROOTS, "trishift", tri_root)
        monkeypatch.setitem(adapter.DEFAULT_PAYLOAD_ROOTS, "gears", gears_root)
        old_cwd = Path.cwd()
        monkeypatch.chdir(notebook_cwd)
        result = run_split_structure_recovery(
            dataset="norman",
            models=["trishift_nearest", "gears"],
            split_ids="1",
            out_root=root / "out",
            paths_path=paths_yaml,
            max_cells_per_condition=2,
            cluster_k=3,
            seed=7,
        )
        monkeypatch.chdir(old_cwd)
        assert not result.summary_df.empty
        assert {"Experimental data", "TriShift nearest", "GEARS", "Mean perturbation"} <= set(
            result.summary_df["row_name"].astype(str).tolist()
        )
        assert result.figure_path.exists()
        assert (result.out_dir / "figs4e_structure_recovery_norman_metrics.csv").exists()
