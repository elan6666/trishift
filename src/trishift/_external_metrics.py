import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from scipy.stats import wasserstein_distance


def pearson_delta_reference_metrics(
    X_true: np.ndarray,
    X_pred: np.ndarray,
    reference: np.ndarray,
    top20_de_idxs: np.ndarray,
) -> dict:
    """Systema evaluation/pearson_delta_reference_metrics.py compatible metric."""
    delta_true_allpert = X_true - reference
    delta_pred_allpert = X_pred - reference

    top20_de_idxs = np.asarray(top20_de_idxs, dtype=int).reshape(-1)
    out = {
        "corr_all_allpert": pearsonr(delta_true_allpert, delta_pred_allpert)[0],
        "corr_20de_allpert": pearsonr(
            delta_true_allpert[top20_de_idxs], delta_pred_allpert[top20_de_idxs]
        )[0]
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


def compute_scpram_metrics_from_arrays(
    X_true: np.ndarray,
    X_pred: np.ndarray,
    deg_idx: np.ndarray,
    n_degs: int = 100,
    sample_ratio: float = 0.8,
    times: int = 100,
) -> dict:
    """Compute scPRAM-style R2/Wasserstein metrics on one condition."""
    X_true = np.asarray(X_true, dtype=np.float32)
    X_pred = np.asarray(X_pred, dtype=np.float32)
    if X_true.ndim != 2 or X_pred.ndim != 2:
        raise ValueError("X_true and X_pred must be 2D arrays")
    if X_true.shape[1] != X_pred.shape[1]:
        raise ValueError("X_true and X_pred must share the same number of genes")

    df_case = pd.DataFrame(X_true)
    df_pred = pd.DataFrame(X_pred)

    deg_idx = np.asarray(deg_idx, dtype=int).reshape(-1)
    degs = [int(i) for i in deg_idx.tolist() if 0 <= int(i) < X_true.shape[1]]
    degs = degs[: int(n_degs)]

    r2_mean, r2_std = get_pearson2_from_df(
        df_case=df_case,
        df_pred=df_pred,
        degs=degs,
        sample_ratio=sample_ratio,
        times=times,
    )
    wd_all, wd_top = get_wasserstein_distance_from_df(
        df_case=df_case,
        df_pred=df_pred,
        top_genes=degs,
        cal_type="sum",
    )
    return {
        "scpram_r2_all_mean_mean": float(r2_mean["r2_all_mean"]),
        "scpram_r2_all_var_mean": float(r2_mean["r2_all_var"]),
        "scpram_r2_degs_mean_mean": float(r2_mean["r2_degs_mean"]),
        "scpram_r2_degs_var_mean": float(r2_mean["r2_degs_var"]),
        "scpram_r2_all_mean_std": float(r2_std["r2_all_mean"]),
        "scpram_r2_all_var_std": float(r2_std["r2_all_var"]),
        "scpram_r2_degs_mean_std": float(r2_std["r2_degs_mean"]),
        "scpram_r2_degs_var_std": float(r2_std["r2_degs_var"]),
        "scpram_wasserstein_all_sum": float(wd_all),
        "scpram_wasserstein_degs_sum": float(wd_top),
    }
