from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pickle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.biolord import shrink_notebook_pkl  # noqa: E402
from scripts.trishift.analysis import recompute_metrics_from_pkl  # noqa: E402


def test_biolord_shrinker_limits_pred_ctrl_truth_independently():
    payload = {
        "Pred_full": np.ones((1000, 4), dtype=np.float32),
        "Ctrl_full": np.ones((100, 4), dtype=np.float32),
        "Truth_full": np.ones((800, 4), dtype=np.float32),
        "DE_idx": np.array([1, 3], dtype=int),
    }
    out, changed = shrink_notebook_pkl._shrink_condition_payload(payload, sample_size=300, seed=7)
    assert changed
    assert out["Pred_full"].shape[0] == 300
    assert out["Ctrl_full"].shape[0] == 100
    assert out["Truth_full"].shape[0] == 300
    assert out["Pred"].shape == (300, 2)
    assert out["Ctrl"].shape == (100, 2)
    assert out["Truth"].shape == (300, 2)
    assert bool(out["export_metadata"]["export_is_subset"])


def test_recompute_metrics_from_pkl_uses_full_summary(tmp_path):
    payload = {
        "A": {
            "Pred": np.zeros((2, 2), dtype=np.float32),
            "Ctrl": np.zeros((2, 2), dtype=np.float32),
            "Truth": np.zeros((2, 2), dtype=np.float32),
            "export_metadata": {
                "metrics_computed_on_full": True,
                "n_pred_full": 17,
                "n_ctrl_full": 17,
                "eval_ctrl_source": "target_domain_test_ctrl",
            },
            "full_summary": {
                "metrics": {
                    "pearson": 0.91,
                    "nmse": 0.12,
                    "systema_corr_20de_allpert_dist": 0.77,
                    "systema_corr_deg_r2_dist": 0.66,
                }
            },
        }
    }
    pkl_path = tmp_path / "toy.pkl"
    with pkl_path.open("wb") as handle:
        pickle.dump(payload, handle)
    rows = recompute_metrics_from_pkl._compute_rows_from_pkl(
        pkl_path,
        split_id=1,
        subgroup_map=None,
    )
    assert rows[0]["n_ensemble"] == 17
    assert rows[0]["n_eval_ctrl"] == 17
    assert rows[0]["eval_ctrl_source"] == "target_domain_test_ctrl"
    assert abs(rows[0]["systema_corr_20de_allpert_dist"] - 0.77) < 1e-12
