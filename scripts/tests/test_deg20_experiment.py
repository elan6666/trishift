from pathlib import Path
import pickle

import numpy as np
import pandas as pd

import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.tests._helpers import temp_dir
from scripts.trishift.analysis.deg20_experiment import (
    _pred_deg20,
    _truth_deg20,
    load_condition_payload,
    run_deg20_experiment,
    select_representative_conditions,
)


def _write_payload(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(payload, f)


def _multirow_payload() -> dict:
    gene_names = np.array([f"G{i}" for i in range(6)], dtype=object)
    ctrl = np.full((6, 6), 1.0, dtype=np.float32)
    truth = np.full((6, 6), 1.0, dtype=np.float32)
    truth[:, 0] += 4.0
    truth[:, 1] += 3.0
    pred = np.full((6, 6), 1.0, dtype=np.float32)
    pred[:, 0] += 4.5
    pred[:, 1] += 2.5
    return {
        "X+ctrl": {
            "Pred_full": pred,
            "Ctrl_full": ctrl,
            "Truth_full": truth,
            "gene_name_full": gene_names,
            "DE_idx": np.array([0, 1], dtype=int),
            "DE_name": np.array(["G0", "G1"], dtype=object),
        }
    }


def _single_row_payload() -> dict:
    gene_names = np.array([f"G{i}" for i in range(6)], dtype=object)
    ctrl = np.full((6, 6), 1.0, dtype=np.float32)
    truth = np.full((6, 6), 1.0, dtype=np.float32)
    truth[:, 2] += 5.0
    truth[:, 3] += 2.0
    pred = np.full((1, 6), 1.0, dtype=np.float32)
    pred[:, 2] += 5.5
    pred[:, 3] += 2.5
    return {
        "Y+ctrl": {
            "Pred_full": pred,
            "Ctrl_full": ctrl,
            "Truth_full": truth,
            "gene_name_full": gene_names,
            "DE_idx": np.array([2, 3], dtype=int),
            "DE_name": np.array(["G2", "G3"], dtype=object),
        }
    }


def _multirow_negative_pred_payload() -> dict:
    gene_names = np.array([f"G{i}" for i in range(6)], dtype=object)
    ctrl = np.full((6, 6), 1.0, dtype=np.float32)
    truth = np.full((6, 6), 1.0, dtype=np.float32)
    truth[:, 0] += 4.0
    truth[:, 1] += 3.0
    pred = np.full((6, 6), 1.0, dtype=np.float32)
    pred[:, 0] += 4.5
    pred[:, 1] += 2.5
    pred[:, 4] = -0.5
    return {
        "Z+ctrl": {
            "Pred_full": pred,
            "Ctrl_full": ctrl,
            "Truth_full": truth,
            "gene_name_full": gene_names,
            "DE_idx": np.array([0, 1], dtype=int),
            "DE_name": np.array(["G0", "G1"], dtype=object),
        }
    }


def test_run_deg20_experiment_adaptive_uses_scanpy_and_effect_size():
    with temp_dir() as tmp:
        root = Path(tmp)
        tri_dir = root / "tri"
        gears_dir = root / "gears"
        _write_payload(tri_dir / "trishift_toy_1.pkl", _multirow_payload())
        _write_payload(gears_dir / "gears_toy_1.pkl", _single_row_payload())

        tri_result = run_deg20_experiment(
            dataset="toy",
            model_name="trishift",
            split_ids=[1],
            result_dir=tri_dir,
            pred_deg_mode="adaptive",
            enrichment_mode="export_only",
        )
        gears_result = run_deg20_experiment(
            dataset="toy",
            model_name="gears",
            split_ids=[1],
            result_dir=gears_dir,
            pred_deg_mode="adaptive",
            enrichment_mode="export_only",
        )

        tri_row = tri_result.per_condition_df.iloc[0]
        gears_row = gears_result.per_condition_df.iloc[0]
        assert tri_row["pred_deg_mode_used"] == "scanpy_rank"
        assert gears_row["pred_deg_mode_used"] == "effect_size_fallback"
        assert int(tri_row["common_degs_at_20"]) >= 2
        assert int(gears_row["common_degs_at_20"]) >= 2
        assert (tri_result.out_dir / "per_condition_metrics.csv").exists()
        assert (tri_result.out_dir / "deg_gene_lists_long.csv").exists()


def test_pred_deg20_adaptive_falls_back_to_effect_size_for_negative_predictions():
    payload = _multirow_negative_pred_payload()["Z+ctrl"]
    pred_genes, mode_used = _pred_deg20(
        pred=np.asarray(payload["Pred_full"], dtype=np.float32),
        ctrl=np.asarray(payload["Ctrl_full"], dtype=np.float32),
        gene_names=np.asarray(payload["gene_name_full"]).astype(str),
        condition="Z+ctrl",
        pred_deg_mode="adaptive",
        remove_perturbed_genes=True,
    )
    assert mode_used == "scanpy_rank"
    assert len(set(pred_genes[:5]) & {"G0", "G1"}) >= 1


def test_pred_deg20_ttest_non_dropout_matches_truth_style_filtering():
    payload = _multirow_payload()["X+ctrl"]
    pred_genes, mode_used = _pred_deg20(
        pred=np.asarray(payload["Pred_full"], dtype=np.float32),
        ctrl=np.asarray(payload["Ctrl_full"], dtype=np.float32),
        gene_names=np.asarray(payload["gene_name_full"]).astype(str),
        condition="X+ctrl",
        pred_deg_mode="ttest_non_dropout",
        remove_perturbed_genes=True,
    )
    assert mode_used == "ttest_non_dropout"
    assert "X" not in pred_genes
    assert len(set(pred_genes[:5]) & {"G0", "G1"}) >= 1


def test_pred_deg20_effect_size_non_dropout_filters_dropout_genes():
    gene_names = np.array(["G0", "G1", "G2", "G3"], dtype=object)
    ctrl = np.array(
        [
            [0.0, 1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    pred = np.array(
        [
            [0.0, 4.0, 2.5, 1.0],
            [0.0, 4.0, 2.5, 1.0],
            [0.0, 4.0, 2.5, 1.0],
        ],
        dtype=np.float32,
    )
    pred_genes, mode_used = _pred_deg20(
        pred=pred,
        ctrl=ctrl,
        gene_names=gene_names.astype(str),
        condition="X+ctrl",
        pred_deg_mode="effect_size_non_dropout",
        remove_perturbed_genes=False,
    )
    assert mode_used == "effect_size_non_dropout"
    assert pred_genes[0] == "G1"
    assert pred_genes[1] == "G2"


def test_truth_deg20_effect_size_non_dropout_uses_truth_vs_ctrl_ranking():
    payload = _single_row_payload()["Y+ctrl"]
    truth_genes, truth_idx = _truth_deg20(
        condition="Y+ctrl",
        truth=np.asarray(payload["Truth_full"], dtype=np.float32),
        ctrl=np.asarray(payload["Ctrl_full"], dtype=np.float32),
        gene_names=np.asarray(payload["gene_name_full"]).astype(str),
        deg_idx=np.asarray(payload["DE_idx"], dtype=int),
        deg_name=np.asarray(payload["DE_name"]).astype(str),
        truth_deg_mode="effect_size_non_dropout",
        remove_perturbed_genes=True,
    )
    assert "Y" not in truth_genes
    assert truth_genes[:2] == ["G2", "G3"]
    assert truth_idx.tolist()[:2] == [2, 3]


def test_truth_deg20_scanpy_uses_truth_vs_ctrl_ranking():
    payload = _multirow_payload()["X+ctrl"]
    truth_genes, truth_idx = _truth_deg20(
        condition="X+ctrl",
        truth=np.asarray(payload["Truth_full"], dtype=np.float32),
        ctrl=np.asarray(payload["Ctrl_full"], dtype=np.float32),
        gene_names=np.asarray(payload["gene_name_full"]).astype(str),
        deg_idx=np.asarray(payload["DE_idx"], dtype=int),
        deg_name=np.asarray(payload["DE_name"]).astype(str),
        truth_deg_mode="scanpy",
        remove_perturbed_genes=True,
    )
    assert "X" not in truth_genes
    assert len(set(truth_genes[:5]) & {"G0", "G1"}) >= 1
    assert len(truth_idx) >= 1


def test_truth_deg20_ttest_non_dropout_uses_truth_vs_ctrl_ranking():
    payload = _multirow_payload()["X+ctrl"]
    truth_genes, truth_idx = _truth_deg20(
        condition="X+ctrl",
        truth=np.asarray(payload["Truth_full"], dtype=np.float32),
        ctrl=np.asarray(payload["Ctrl_full"], dtype=np.float32),
        gene_names=np.asarray(payload["gene_name_full"]).astype(str),
        deg_idx=np.asarray(payload["DE_idx"], dtype=int),
        deg_name=np.asarray(payload["DE_name"]).astype(str),
        truth_deg_mode="ttest_non_dropout",
        remove_perturbed_genes=True,
    )
    assert "X" not in truth_genes
    assert len(set(truth_genes[:5]) & {"G0", "G1"}) >= 1
    assert len(truth_idx) >= 1


def test_run_deg20_experiment_summaries_are_macro_over_conditions():
    with temp_dir() as tmp:
        root = Path(tmp)
        tri_dir = root / "tri"
        payload = _multirow_payload()
        payload.update(_single_row_payload())
        _write_payload(tri_dir / "trishift_toy_1.pkl", payload)

        result = run_deg20_experiment(
            dataset="toy",
            model_name="trishift",
            split_ids="1",
            result_dir=tri_dir,
            truth_deg_mode="effect_size_non_dropout",
            pred_deg_mode="adaptive",
            enrichment_mode="export_only",
        )
        assert len(result.per_condition_df) == 2
        assert len(result.split_summary_df) == 1
        split_row = result.split_summary_df.iloc[0]
        expected_mean = float(result.per_condition_df["common_degs_at_20"].mean())
        assert np.isclose(float(split_row["common_degs_at_20_mean"]), expected_mean)
        assert int(split_row["n_conditions"]) == 2
        assert set(result.per_condition_df["truth_deg_mode_used"]) == {"effect_size_non_dropout"}
        assert len(result.dataset_summary_df) == 1


def test_non_payload_truth_modes_write_cache_pkls():
    with temp_dir() as tmp:
        root = Path(tmp)
        tri_dir = root / "tri"
        cache_dir = root / "cache"
        _write_payload(tri_dir / "trishift_toy_1_nearest.pkl", _multirow_payload())

        run_deg20_experiment(
            dataset="toy",
            model_name="trishift",
            split_ids="1",
            result_dir=tri_dir,
            variant_tag="nearest",
            truth_deg_mode="scanpy",
            pred_deg_mode="effect_size",
            enrichment_mode="export_only",
            truth_deg_cache_root=cache_dir,
            n_degs=20,
        )
        run_deg20_experiment(
            dataset="toy",
            model_name="trishift",
            split_ids="1",
            result_dir=tri_dir,
            variant_tag="nearest",
            truth_deg_mode="effect_size_non_dropout",
            pred_deg_mode="effect_size",
            enrichment_mode="export_only",
            truth_deg_cache_root=cache_dir,
            n_degs=20,
        )

        scanpy_cache = cache_dir / "trishift_toy_1_nearest__truth_scanpy__top20__drop_pert.pkl"
        effect_nd_cache = cache_dir / "trishift_toy_1_nearest__truth_effect_size_non_dropout__top20__drop_pert.pkl"
        assert scanpy_cache.exists()
        assert effect_nd_cache.exists()


def test_run_deg20_experiment_respects_n_degs():
    with temp_dir() as tmp:
        root = Path(tmp)
        tri_dir = root / "tri"
        cache_dir = root / "cache"
        _write_payload(tri_dir / "trishift_toy_1_nearest.pkl", _multirow_payload())

        result = run_deg20_experiment(
            dataset="toy",
            model_name="trishift",
            split_ids="1",
            result_dir=tri_dir,
            variant_tag="nearest",
            truth_deg_mode="effect_size_non_dropout",
            pred_deg_mode="effect_size_non_dropout",
            enrichment_mode="export_only",
            truth_deg_cache_root=cache_dir,
            n_degs=3,
        )

        row = result.per_condition_df.iloc[0]
        assert int(row["deg_top_k"]) == 3
        assert int(row["truth_deg_count"]) == 3
        assert int(row["pred_deg_count"]) == 3
        cache_path = cache_dir / "trishift_toy_1_nearest__truth_effect_size_non_dropout__top3__drop_pert.pkl"
        assert cache_path.exists()


def test_load_condition_payload_reads_model_specific_file():
    with temp_dir() as tmp:
        root = Path(tmp)
        tri_dir = root / "tri"
        payload = _multirow_payload()
        _write_payload(tri_dir / "trishift_toy_2_nearest.pkl", payload)
        obj = load_condition_payload(
            model_name="trishift",
            dataset="toy",
            split_id=2,
            condition="X+ctrl",
            result_dir=tri_dir,
            variant_tag="nearest",
        )
        assert "Pred_full" in obj
        assert np.asarray(obj["Pred_full"]).shape == (6, 6)


def test_select_representative_conditions_uses_focus_conditions():
    df = pd.DataFrame(
        {
            "focus_key": ["1:A", "1:B", "1:C"],
            "condition": ["A", "B", "C"],
            "common_degs_at_20": [5, 10, 1],
            "jaccard_at_20": [0.2, 0.5, 0.1],
            "scpram_r2_degs_mean_mean": [0.1, 0.2, 0.3],
        }
    )
    picked = select_representative_conditions(df, focus_conditions=["C", "A"])
    assert picked["condition"].tolist() == ["A", "C"]
