from pathlib import Path
import pickle

import numpy as np

import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.tests._helpers import temp_dir
from scripts.trishift.analysis.condition_centroid_vis import (
    _feature_arrays,
    run_condition_centroid_visualization,
)


def _write_payload(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(payload, f)


def _legacy_payload(pred_rows: int = 4, truth_rows: int = 5, ctrl_rows: int = 6) -> dict:
    pred = np.tile(np.array([[3.0, 2.0, 1.0]], dtype=np.float32), (pred_rows, 1))
    truth = np.tile(np.array([[2.0, 4.0, 1.0]], dtype=np.float32), (truth_rows, 1))
    ctrl = np.tile(np.array([[1.0, 1.0, 1.0]], dtype=np.float32), (ctrl_rows, 1))
    return {
        "A+ctrl": {
            "Pred": pred,
            "Truth": truth,
            "Ctrl": ctrl,
            "DE_idx": np.array([0, 1, 2], dtype=int),
            "DE_name": np.array(["G0", "G1", "G2"], dtype=object),
        },
        "B+ctrl": {
            "Pred": pred + 1.0,
            "Truth": truth + 0.5,
            "Ctrl": ctrl,
            "DE_idx": np.array([0, 1, 2], dtype=int),
            "DE_name": np.array(["G0", "G1", "G2"], dtype=object),
        },
    }


def _full_payload() -> dict:
    gene_names = np.array(["G0", "G1", "G2", "G3"], dtype=object)
    pred = np.array([[3.0, 2.0, 1.0, 0.0], [3.0, 2.0, 1.0, 0.0]], dtype=np.float32)
    truth = np.array([[2.0, 4.0, 1.0, 0.0], [2.0, 4.0, 1.0, 0.0]], dtype=np.float32)
    ctrl = np.array([[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 0.0]], dtype=np.float32)
    return {
        "A+ctrl": {
            "Pred": pred[:, :3],
            "Truth": truth[:, :3],
            "Ctrl": ctrl[:, :3],
            "DE_idx": np.array([0, 1, 2], dtype=int),
            "DE_name": np.array(["G0", "G1", "G2"], dtype=object),
            "Pred_full": pred,
            "Truth_full": truth,
            "Ctrl_full": ctrl,
            "gene_name_full": gene_names,
        },
        "B+ctrl": {
            "Pred": pred[:, :3] + 1.0,
            "Truth": truth[:, :3] + 0.5,
            "Ctrl": ctrl[:, :3],
            "DE_idx": np.array([0, 1, 2], dtype=int),
            "DE_name": np.array(["G0", "G1", "G2"], dtype=object),
            "Pred_full": pred + 1.0,
            "Truth_full": truth + 0.5,
            "Ctrl_full": ctrl,
            "gene_name_full": gene_names,
        },
    }


def test_feature_arrays_full_requires_full_fields():
    payload = _legacy_payload()["A+ctrl"]
    try:
        _feature_arrays(payload=payload, feature_mode="full")
    except ValueError as exc:
        assert "feature_mode='full'" in str(exc)
    else:
        raise AssertionError("Expected ValueError when full payload fields are missing")


def test_run_condition_centroid_visualization_supports_all_models_with_deg_payload():
    with temp_dir() as tmp:
        root = Path(tmp)
        payload = _legacy_payload()
        paths = {
            "trishift": root / "tri" / "trishift_toy_1_nearest.pkl",
            "scouter": root / "scouter" / "scouter_toy_1_nearest.pkl",
            "gears": root / "gears" / "gears_toy_1.pkl",
            "genepert": root / "genepert" / "genepert_toy_1.pkl",
        }
        for p in paths.values():
            _write_payload(p, payload)

        for model_name, result_dir in {
            "trishift": root / "tri",
            "scouter": root / "scouter",
            "gears": root / "gears",
            "genepert": root / "genepert",
        }.items():
            result = run_condition_centroid_visualization(
                model_name=model_name,
                dataset="toy",
                split_id=1,
                result_dir=result_dir,
                variant_tag="nearest",
                feature_mode="deg",
                include_ctrl=True,
                plot_delta=True,
                save_dpi=120,
            )
            assert len(result.metrics_df) == 2
            assert (result.out_dir / "condition_centroid_points.csv").exists()
            assert (result.out_dir / "condition_centroid_metrics.csv").exists()
            assert (result.out_dir / "condition_centroid_summary.csv").exists()
            assert "truth_vs_pred_centroid_umap" in result.figure_paths


def test_run_condition_centroid_visualization_full_mode_uses_full_payload():
    with temp_dir() as tmp:
        root = Path(tmp)
        pkl_path = root / "tri" / "trishift_toy_1_nearest.pkl"
        _write_payload(pkl_path, _full_payload())
        result = run_condition_centroid_visualization(
            model_name="trishift",
            dataset="toy",
            split_id=1,
            result_dir=root / "tri",
            variant_tag="nearest",
            feature_mode="full",
            include_ctrl=True,
            plot_delta=True,
            save_dpi=120,
        )
        assert set(result.points_df["feature_mode_used"].tolist()) == {"full"}
        assert set(result.points_df["feature_dim"].tolist()) == {4}


def test_delta_metrics_nan_when_ctrl_missing():
    with temp_dir() as tmp:
        root = Path(tmp)
        payload = _legacy_payload()
        for item in payload.values():
            item.pop("Ctrl")
        _write_payload(root / "tri" / "trishift_toy_1_nearest.pkl", payload)
        result = run_condition_centroid_visualization(
            model_name="trishift",
            dataset="toy",
            split_id=1,
            result_dir=root / "tri",
            variant_tag="nearest",
            feature_mode="deg",
            include_ctrl=True,
            plot_delta=True,
            save_dpi=120,
        )
        assert result.metrics_df["delta_centroid_dist"].isna().all()
        assert result.metrics_df["delta_cosine"].isna().all()
