from pathlib import Path
import pickle

import numpy as np

import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.tests._helpers import temp_dir
from scripts.trishift.analysis.scpram_deg_experiment import (
    _full_payload_item_arrays_strict,
    run_scpram_deg_experiment,
    summarize_condition_payload,
)


def _write_payload(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(payload, fh)


def _toy_full_payload() -> dict:
    gene_names = np.array(["G0", "G1", "G2", "G3"], dtype=object)
    ctrl = np.array(
        [
            [0.05, 0.10, 0.05, 0.08],
            [0.07, 0.09, 0.06, 0.07],
            [0.06, 0.08, 0.07, 0.09],
            [0.05, 0.11, 0.04, 0.08],
            [0.08, 0.10, 0.05, 0.07],
            [0.07, 0.09, 0.06, 0.08],
        ],
        dtype=np.float32,
    )
    truth = np.array(
        [
            [5.0, 4.5, 0.05, 0.08],
            [5.2, 4.6, 0.06, 0.07],
            [4.9, 4.7, 0.04, 0.09],
            [5.1, 4.4, 0.05, 0.08],
            [5.3, 4.5, 0.05, 0.07],
            [5.0, 4.6, 0.06, 0.09],
        ],
        dtype=np.float32,
    )
    pred = np.array(
        [
            [5.1, 0.10, 4.8, 0.08],
            [5.2, 0.09, 4.7, 0.07],
            [5.0, 0.11, 4.9, 0.09],
            [5.3, 0.10, 4.6, 0.08],
            [5.1, 0.12, 4.8, 0.07],
            [5.2, 0.09, 4.7, 0.09],
        ],
        dtype=np.float32,
    )
    return {
        "A+ctrl": {
            "Pred_full": pred,
            "Ctrl_full": ctrl,
            "Truth_full": truth,
            "gene_name_full": gene_names,
        }
    }


def _legacy_payload() -> dict:
    return {
        "A+ctrl": {
            "Pred": np.ones((3, 2), dtype=np.float32),
            "Ctrl": np.ones((3, 2), dtype=np.float32),
            "Truth": np.ones((3, 2), dtype=np.float32),
            "DE_idx": np.array([0, 1], dtype=int),
            "DE_name": np.array(["G0", "G1"], dtype=object),
        }
    }


def test_strict_full_gene_payload_rejects_legacy_payload():
    with temp_dir() as tmp:
        pkl_path = Path(tmp) / "legacy.pkl"
        payload = _legacy_payload()
        _write_payload(pkl_path, payload)
        try:
            _full_payload_item_arrays_strict(
                payload["A+ctrl"],
                pkl_path=pkl_path,
                condition="A+ctrl",
            )
        except ValueError as exc:
            text = str(exc)
            assert "Pred_full" in text
            assert str(pkl_path) in text
            assert "full-gene pkl" in text
        else:
            raise AssertionError("Expected strict full-gene validation to fail")


def test_summarize_condition_payload_uses_scpram_style_wilcoxon_topk():
    payload_item = _toy_full_payload()["A+ctrl"]
    summary = summarize_condition_payload(
        payload_item=payload_item,
        condition="A+ctrl",
        pkl_path="toy.pkl",
        top_k=2,
        remove_perturbed_genes=False,
    )
    assert summary["truth_deg_topk"][:2] == ["G0", "G1"]
    assert summary["pred_deg_topk"][:2] == ["G0", "G2"]
    assert summary["common_deg_topk"] == ["G0"]
    assert summary["truth_deg_idx"].tolist() == [0, 1]


def test_run_scpram_deg_experiment_writes_outputs_and_respects_top_k():
    with temp_dir() as tmp:
        root = Path(tmp)
        result_dir = root / "tri"
        out_root = root / "out"
        _write_payload(result_dir / "trishift_toy_1_nearest.pkl", _toy_full_payload())

        result = run_scpram_deg_experiment(
            dataset="toy",
            model_name="trishift",
            split_ids=[1],
            result_dir=result_dir,
            out_root=out_root,
            variant_tag="nearest",
            top_k=2,
            remove_perturbed_genes=False,
        )

        row = result.per_condition_df.iloc[0]
        assert int(row["deg_top_k"]) == 2
        assert int(row["common_degs_at_k"]) == 1
        assert np.isclose(float(row["jaccard_at_k"]), 1.0 / 3.0)
        assert (out_root / "per_condition_metrics.csv").exists()
        assert (out_root / "split_summary.csv").exists()
        assert (out_root / "dataset_summary.csv").exists()
        assert (out_root / "deg_gene_lists_long.csv").exists()
        assert (out_root / "representative_conditions.csv").exists()
        assert (out_root / "run_meta.json").exists()
