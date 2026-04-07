from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.metrics import r2_score

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from trishift._external_metrics import (  # noqa: E402
    compute_scpram_metrics_from_arrays,
    pearson_delta_reference_metrics,
    regression_r2_safe,
)


def test_regression_r2_safe_matches_sklearn():
    y_true = np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    y_pred = np.asarray([0.1, 0.9, 2.2, 2.8], dtype=np.float64)
    expected = float(r2_score(y_true, y_pred))
    got = regression_r2_safe(y_true, y_pred)
    assert abs(got - expected) < 1e-12


def test_regression_r2_safe_nan_on_invalid_inputs():
    y_true_const = np.asarray([1.0, 1.0, 1.0], dtype=np.float64)
    y_pred_const = np.asarray([1.0, 1.2, 0.8], dtype=np.float64)
    assert np.isnan(regression_r2_safe(y_true_const, y_pred_const))

    y_true_short = np.asarray([1.0], dtype=np.float64)
    y_pred_short = np.asarray([1.0], dtype=np.float64)
    assert np.isnan(regression_r2_safe(y_true_short, y_pred_short))


def test_pearson_delta_reference_metrics_adds_regression_r2():
    x_true = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    x_pred = np.asarray([1.1, 2.2, 2.9, 3.8], dtype=np.float64)
    ref = np.asarray([0.5, 0.5, 0.5, 0.5], dtype=np.float64)
    deg_idx = np.asarray([1, 3], dtype=int)
    out = pearson_delta_reference_metrics(
        X_true=x_true,
        X_pred=x_pred,
        reference=ref,
        top20_de_idxs=deg_idx,
    )
    assert "corr_all_r2" in out
    assert "corr_deg_r2" in out
    assert np.isfinite(float(out["corr_all_r2"]))
    assert np.isfinite(float(out["corr_deg_r2"]))


def test_compute_scpram_metrics_contains_regression_var_r2():
    rng = np.random.RandomState(0)
    x_true = rng.normal(size=(40, 12)).astype(np.float32)
    x_pred = (x_true * 0.85 + 0.15 * rng.normal(size=(40, 12))).astype(np.float32)
    deg_idx = np.asarray([0, 1, 2, 3, 4], dtype=int)
    out = compute_scpram_metrics_from_arrays(
        X_true=x_true,
        X_pred=x_pred,
        deg_idx=deg_idx,
        n_degs=5,
        sample_ratio=0.8,
        times=20,
    )
    assert "r2_all_var_mean" in out
    assert "r2_degs_var_mean" in out
    assert "scpram_r2_all_var_mean" in out
    assert "scpram_r2_degs_var_mean" in out
    assert np.isfinite(float(out["r2_all_var_mean"])) or np.isnan(float(out["r2_all_var_mean"]))
    assert np.isfinite(float(out["r2_degs_var_mean"])) or np.isnan(float(out["r2_degs_var_mean"]))


def main():
    test_regression_r2_safe_matches_sklearn()
    test_regression_r2_safe_nan_on_invalid_inputs()
    test_pearson_delta_reference_metrics_adds_regression_r2()
    test_compute_scpram_metrics_contains_regression_var_r2()
    print("test_regression_r2_metrics: PASS")


if __name__ == "__main__":
    main()

