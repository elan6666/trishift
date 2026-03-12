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
            pred_deg_mode="adaptive",
            enrichment_mode="export_only",
        )
        assert len(result.per_condition_df) == 2
        assert len(result.split_summary_df) == 1
        split_row = result.split_summary_df.iloc[0]
        expected_mean = float(result.per_condition_df["common_degs_at_20"].mean())
        assert np.isclose(float(split_row["common_degs_at_20_mean"]), expected_mean)
        assert int(split_row["n_conditions"]) == 2
        assert len(result.dataset_summary_df) == 1


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
