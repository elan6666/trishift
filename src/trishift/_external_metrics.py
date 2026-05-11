from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from scipy.stats import wasserstein_distance
from sklearn.metrics import r2_score


def _safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64).reshape(-1)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    if aa.size != bb.size or aa.size < 2:
        return np.nan
    if (not np.isfinite(aa).all()) or (not np.isfinite(bb).all()):
        return np.nan
    if float(np.std(aa)) <= 0.0 or float(np.std(bb)) <= 0.0:
        return np.nan
    return float(pearsonr(aa, bb)[0])


def _safe_corrcoef_squared(a: np.ndarray, b: np.ndarray) -> float:
    corr = _safe_pearson(a, b)
    if not np.isfinite(corr):
        return np.nan
    return float(corr * corr)


def _nanmean_or_nan(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    return float(arr.mean())


def _regression_r2_safe(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """Compute regression R^2 safely; return NaN when undefined."""
    yt = np.asarray(y_true, dtype=np.float64).reshape(-1)
    yp = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if yt.size != yp.size or yt.size < 2:
        return np.nan
    if (not np.isfinite(yt).all()) or (not np.isfinite(yp).all()):
        return np.nan
    if float(np.var(yt)) <= float(eps):
        return np.nan
    return float(r2_score(yt, yp))


def regression_r2_safe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Public wrapper for safe regression R^2."""
    return _regression_r2_safe(y_true=y_true, y_pred=y_pred)


def _validate_gene_idx(idx: np.ndarray, n_genes: int, n_degs: int | None = None) -> np.ndarray:
    out = np.asarray(idx, dtype=int).reshape(-1)
    out = out[(out >= 0) & (out < int(n_genes))]
    if n_degs is not None:
        out = out[: int(n_degs)]
    return out


def _mean_vector(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 1:
        return arr.reshape(-1)
    return arr.mean(axis=0).reshape(-1)


def _var_vector(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 1:
        return np.zeros_like(arr.reshape(-1), dtype=np.float32)
    if arr.shape[0] <= 1:
        return np.zeros(arr.shape[1], dtype=np.float32)
    return arr.var(axis=0, ddof=1).reshape(-1)


def compute_mean_effect_metrics(
    X_true: np.ndarray,
    X_pred: np.ndarray,
    X_ctrl: np.ndarray,
    deg_idx: np.ndarray,
) -> dict[str, float]:
    """Compute DE-gene metrics on mean perturbation effects."""
    X_true = np.asarray(X_true, dtype=np.float32)
    X_pred = np.asarray(X_pred, dtype=np.float32)
    X_ctrl = np.asarray(X_ctrl, dtype=np.float32)
    if X_true.ndim != 2 or X_pred.ndim != 2 or X_ctrl.ndim != 2:
        raise ValueError("X_true, X_pred, and X_ctrl must be 2D arrays")
    if X_true.shape[1] != X_pred.shape[1] or X_true.shape[1] != X_ctrl.shape[1]:
        raise ValueError("X_true, X_pred, and X_ctrl must share the same number of genes")

    degs = _validate_gene_idx(deg_idx, X_true.shape[1], None)
    if degs.size == 0:
        return {
            "mse_pred": np.nan,
            "mse_ctrl": np.nan,
            "nmse": np.nan,
            "pearson": np.nan,
            "deg_mean_r2": np.nan,
        }

    true_mean = _mean_vector(X_true)
    pred_mean = _mean_vector(X_pred)
    ctrl_mean = _mean_vector(X_ctrl)
    true_delta = true_mean[degs] - ctrl_mean[degs]
    pred_delta = pred_mean[degs] - ctrl_mean[degs]
    zero_delta = np.zeros_like(true_delta)
    mse_ctrl_val = float(np.mean((true_delta - zero_delta) ** 2))
    mse_pred_val = float(np.mean((true_delta - pred_delta) ** 2))
    return {
        "mse_pred": mse_pred_val,
        "mse_ctrl": mse_ctrl_val,
        "nmse": float(mse_pred_val / mse_ctrl_val) if mse_ctrl_val > 0 else np.nan,
        "pearson": _safe_pearson(true_delta, pred_delta),
        "deg_mean_r2": regression_r2_safe(true_delta, pred_delta),
    }


def pearson_delta_reference_metrics(
    X_true: np.ndarray,
    X_pred: np.ndarray,
    reference: np.ndarray,
    top20_de_idxs: np.ndarray,
) -> dict:
    """Systema evaluation/pearson_delta_reference_metrics.py compatible metric."""
    delta_true_allpert = np.asarray(X_true - reference, dtype=np.float64).reshape(-1)
    delta_pred_allpert = np.asarray(X_pred - reference, dtype=np.float64).reshape(-1)

    top20_de_idxs = np.asarray(top20_de_idxs, dtype=int).reshape(-1)
    top20_de_idxs = top20_de_idxs[
        (top20_de_idxs >= 0) & (top20_de_idxs < int(delta_true_allpert.size))
    ]
    out = {
        "corr_all_allpert": _safe_pearson(delta_true_allpert, delta_pred_allpert),
        "corr_all_r2": _regression_r2_safe(delta_true_allpert, delta_pred_allpert),
        "corr_20de_allpert": _safe_pearson(
            delta_true_allpert[top20_de_idxs], delta_pred_allpert[top20_de_idxs]
        )
        if top20_de_idxs.size > 0
        else np.nan,
        "corr_deg_r2": _regression_r2_safe(
            delta_true_allpert[top20_de_idxs], delta_pred_allpert[top20_de_idxs]
        )
        if top20_de_idxs.size > 0
        else np.nan,
    }
    return out


def average_of_perturbation_centroids(
    X: np.ndarray,
    conditions: np.ndarray,
    ctrl_label: str = "ctrl",
) -> np.ndarray:
    """Systema evaluation/eval_utils.py compatible centroid reference."""
    pert_means = []
    cond_arr = np.asarray(conditions).astype(str)
    for cond in np.unique(cond_arr):
        if cond == ctrl_label:
            continue
        adata_cond = X[cond_arr == cond]
        if adata_cond.shape[0] == 0:
            continue
        pert_mean = np.asarray(adata_cond.mean(axis=0), dtype=np.float32).reshape(-1)
        pert_means.append(pert_mean)
    if len(pert_means) == 0:
        return np.asarray(X.mean(axis=0), dtype=np.float32).reshape(-1)
    pert_means = np.asarray(pert_means, dtype=np.float32)
    return np.asarray(pert_means.mean(axis=0), dtype=np.float32).reshape(-1)


def get_pearson2_from_df(
    df_case: pd.DataFrame,
    df_pred: pd.DataFrame,
    degs: list[int],
    sample_ratio: float = 0.8,
    times: int = 100,
) -> tuple[pd.Series, pd.Series]:
    """scPRAM evaluate.py:get_pearson2 compatible core computation."""
    data = np.zeros((times, 4), dtype=np.float64)
    for i in range(times):
        stim = df_case.sample(frac=sample_ratio, random_state=i)
        pred = df_pred.sample(frac=sample_ratio, random_state=i)

        stim_mean = stim.mean().values.reshape(1, -1)
        pred_mean = pred.mean().values.reshape(1, -1)
        stim_var = stim.var().values.reshape(1, -1)
        pred_var = pred.var().values.reshape(1, -1)

        r2_mean = (np.corrcoef(stim_mean, pred_mean)[0, 1]) ** 2
        r2_var = (np.corrcoef(stim_var, pred_var)[0, 1]) ** 2

        if len(degs) > 0:
            stim_degs_mean = stim.loc[:, degs].mean().values.reshape(1, -1)
            pred_degs_mean = pred.loc[:, degs].mean().values.reshape(1, -1)
            stim_degs_var = stim.loc[:, degs].var().values.reshape(1, -1)
            pred_degs_var = pred.loc[:, degs].var().values.reshape(1, -1)
            r2_degs_mean = (np.corrcoef(stim_degs_mean, pred_degs_mean)[0, 1]) ** 2
            r2_degs_var = (np.corrcoef(stim_degs_var, pred_degs_var)[0, 1]) ** 2
        else:
            r2_degs_mean = np.nan
            r2_degs_var = np.nan
        data[i, :] = [r2_mean, r2_var, r2_degs_mean, r2_degs_var]

    df = pd.DataFrame(
        data, columns=["r2_all_mean", "r2_all_var", "r2_degs_mean", "r2_degs_var"]
    )
    r2_mean = df.mean(axis=0)
    r2_std = df.std(axis=0)
    return r2_mean, r2_std


def get_regression_r2_from_df(
    df_case: pd.DataFrame,
    df_pred: pd.DataFrame,
    degs: list[int],
    sample_ratio: float = 0.8,
    times: int = 100,
) -> tuple[pd.Series, pd.Series]:
    """Compute scPRAM-style repeated-sampling regression R^2 statistics."""
    data = np.zeros((times, 4), dtype=np.float64)
    for i in range(times):
        stim = df_case.sample(frac=sample_ratio, random_state=i)
        pred = df_pred.sample(frac=sample_ratio, random_state=i)

        stim_mean = stim.mean().values.reshape(-1)
        pred_mean = pred.mean().values.reshape(-1)
        stim_var = stim.var().values.reshape(-1)
        pred_var = pred.var().values.reshape(-1)

        r2_mean = _regression_r2_safe(stim_mean, pred_mean)
        r2_var = _regression_r2_safe(stim_var, pred_var)

        if len(degs) > 0:
            stim_degs_mean = stim.loc[:, degs].mean().values.reshape(-1)
            pred_degs_mean = pred.loc[:, degs].mean().values.reshape(-1)
            stim_degs_var = stim.loc[:, degs].var().values.reshape(-1)
            pred_degs_var = pred.loc[:, degs].var().values.reshape(-1)
            r2_degs_mean = _regression_r2_safe(stim_degs_mean, pred_degs_mean)
            r2_degs_var = _regression_r2_safe(stim_degs_var, pred_degs_var)
        else:
            r2_degs_mean = np.nan
            r2_degs_var = np.nan

        data[i, :] = [r2_mean, r2_var, r2_degs_mean, r2_degs_var]

    df = pd.DataFrame(
        data,
        columns=[
            "r2_all_mean_reg",
            "r2_all_var_reg",
            "r2_degs_mean_reg",
            "r2_degs_var_reg",
        ],
    )
    r2_mean = df.mean(axis=0)
    r2_std = df.std(axis=0)
    return r2_mean, r2_std


def get_wasserstein_distance_from_df(
    df_case: pd.DataFrame,
    df_pred: pd.DataFrame,
    top_genes: list[int] | None = None,
    cal_type: str = "sum",
):
    """scPRAM evaluate.py:get_wasserstein_distance compatible core computation."""
    dist_list = []
    dist_list_top = []

    for i in range(df_pred.shape[1]):
        gene_pred = df_pred.iloc[:, i].values
        gene_case = df_case.iloc[:, i].values
        dist = wasserstein_distance(gene_pred, gene_case)
        dist_list.append(dist)

    if top_genes is None:
        if cal_type == "mean":
            return np.mean(dist_list)
        return np.sum(dist_list)

    for gene in top_genes:
        gene_pred = df_pred.loc[:, gene].values
        gene_case = df_case.loc[:, gene].values
        dist = wasserstein_distance(gene_pred, gene_case)
        dist_list_top.append(dist)
    if cal_type == "mean":
        return [np.mean(dist_list), np.mean(dist_list_top)]
    return [np.sum(dist_list), np.sum(dist_list_top)]


def _subsample_rows(
    n_rows: int,
    sample_ratio: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if n_rows <= 0:
        return np.asarray([], dtype=int)
    sample_n = int(np.floor(float(n_rows) * float(sample_ratio)))
    sample_n = max(1, min(int(n_rows), sample_n))
    return np.sort(rng.choice(int(n_rows), size=sample_n, replace=False))


def compute_scpram_metrics_bundle_from_arrays(
    X_true: np.ndarray,
    X_pred: np.ndarray,
    deg_idx: np.ndarray,
    n_degs: int = 100,
    sample_ratio: float = 0.8,
    times: int = 100,
    seed: int = 0,
) -> dict:
    """Compute scPRAM-style metrics and retain repeat-level details."""
    X_true = np.asarray(X_true, dtype=np.float32)
    X_pred = np.asarray(X_pred, dtype=np.float32)
    if X_true.ndim != 2 or X_pred.ndim != 2:
        raise ValueError("X_true and X_pred must be 2D arrays")
    if X_true.shape[1] != X_pred.shape[1]:
        raise ValueError("X_true and X_pred must share the same number of genes")

    degs = _validate_gene_idx(deg_idx, X_true.shape[1], int(n_degs))
    df_case = pd.DataFrame(X_true)
    df_pred = pd.DataFrame(X_pred)
    data = np.full((int(times), 4), np.nan, dtype=np.float64)
    for i in range(int(times)):
        random_state = int(seed) + i
        stim = df_case.sample(frac=sample_ratio, random_state=random_state)
        pred = df_pred.sample(frac=sample_ratio, random_state=random_state)
        if stim.shape[0] == 0 or pred.shape[0] == 0:
            continue

        stim_mean = stim.mean().values.reshape(-1)
        pred_mean = pred.mean().values.reshape(-1)
        stim_var = stim.var().values.reshape(-1)
        pred_var = pred.var().values.reshape(-1)

        data[i, 0] = _safe_corrcoef_squared(stim_mean, pred_mean)
        data[i, 1] = _safe_corrcoef_squared(stim_var, pred_var)
        if degs.size > 0:
            data[i, 2] = _safe_corrcoef_squared(stim_mean[degs], pred_mean[degs])
            data[i, 3] = _safe_corrcoef_squared(stim_var[degs], pred_var[degs])

    wasserstein_degs = np.asarray(
        [
            wasserstein_distance(X_pred[:, gene], X_true[:, gene])
            for gene in degs
        ],
        dtype=np.float64,
    )
    wasserstein_degs_sum = float(np.sum(wasserstein_degs)) if wasserstein_degs.size else np.nan
    metrics = {
        "r2_all_mean_mean": _nanmean_or_nan(data[:, 0]),
        "r2_all_var_mean": _nanmean_or_nan(data[:, 1]),
        "r2_degs_mean_mean": _nanmean_or_nan(data[:, 2]),
        "r2_degs_var_mean": _nanmean_or_nan(data[:, 3]),
        "scpram_r2_all_mean_mean": _nanmean_or_nan(data[:, 0]),
        "scpram_r2_all_var_mean": _nanmean_or_nan(data[:, 1]),
        "scpram_r2_degs_mean_mean": _nanmean_or_nan(data[:, 2]),
        "scpram_r2_degs_var_mean": _nanmean_or_nan(data[:, 3]),
        "scpram_wasserstein_degs_sum": wasserstein_degs_sum,
    }
    return {
        "metrics": metrics,
        "repeats": {
            "r2_all_mean": data[:, 0].astype(np.float32),
            "r2_all_var": data[:, 1].astype(np.float32),
            "r2_degs_mean": data[:, 2].astype(np.float32),
            "r2_degs_var": data[:, 3].astype(np.float32),
        },
        "wasserstein_degs_by_gene": wasserstein_degs.astype(np.float32),
        "degs_used": degs.astype(int),
        "sample_ratio": float(sample_ratio),
        "times": int(times),
        "seed": int(seed),
    }


def compute_scpram_metrics_from_arrays(
    X_true: np.ndarray,
    X_pred: np.ndarray,
    deg_idx: np.ndarray,
    n_degs: int = 100,
    sample_ratio: float = 0.8,
    times: int = 100,
    seed: int = 0,
) -> dict:
    """Compute scPRAM-style R2/Wasserstein metrics on one condition."""
    return compute_scpram_metrics_bundle_from_arrays(
        X_true=X_true,
        X_pred=X_pred,
        deg_idx=deg_idx,
        n_degs=n_degs,
        sample_ratio=sample_ratio,
        times=times,
        seed=seed,
    )["metrics"]


def compute_distributional_systema_metrics_from_arrays(
    X_true: np.ndarray,
    X_pred: np.ndarray,
    reference: np.ndarray,
    deg_idx: np.ndarray,
    sample_ratio: float = 0.8,
    times: int = 100,
    seed: int = 0,
) -> dict:
    """Compute Systema-style delta metrics over repeated cell subsamples."""
    X_true = np.asarray(X_true, dtype=np.float32)
    X_pred = np.asarray(X_pred, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32).reshape(-1)
    if X_true.ndim != 2 or X_pred.ndim != 2:
        raise ValueError("X_true and X_pred must be 2D arrays")
    if X_true.shape[1] != X_pred.shape[1] or X_true.shape[1] != reference.size:
        raise ValueError("X_true, X_pred, and reference must share the same number of genes")

    degs = _validate_gene_idx(deg_idx, X_true.shape[1], None)
    corr_repeats = np.full(int(times), np.nan, dtype=np.float64)
    r2_repeats = np.full(int(times), np.nan, dtype=np.float64)
    for i in range(int(times)):
        true_rng = np.random.default_rng(int(seed) + i)
        pred_rng = np.random.default_rng(int(seed) + 1000003 + i)
        true_idx = _subsample_rows(X_true.shape[0], sample_ratio, true_rng)
        pred_idx = _subsample_rows(X_pred.shape[0], sample_ratio, pred_rng)
        if true_idx.size == 0 or pred_idx.size == 0 or degs.size == 0:
            continue
        delta_true = _mean_vector(X_true[true_idx]) - reference
        delta_pred = _mean_vector(X_pred[pred_idx]) - reference
        corr_repeats[i] = _safe_pearson(delta_true[degs], delta_pred[degs])
        r2_repeats[i] = regression_r2_safe(delta_true[degs], delta_pred[degs])

    return {
        "metrics": {
            "systema_corr_20de_allpert_dist": _nanmean_or_nan(corr_repeats),
            "systema_corr_deg_r2_dist": _nanmean_or_nan(r2_repeats),
        },
        "repeats": {
            "systema_corr_20de_allpert_dist": corr_repeats.astype(np.float32),
            "systema_corr_deg_r2_dist": r2_repeats.astype(np.float32),
        },
        "degs_used": degs.astype(int),
        "sample_ratio": float(sample_ratio),
        "times": int(times),
        "seed": int(seed),
    }
