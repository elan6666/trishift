import random
import pickle

import numpy as np
import pandas as pd
import torch
import yaml
import anndata as ad
import scipy.sparse as sp


def load_yaml(path: str) -> dict:
    """Load YAML content from a file path and return a dict."""
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return {} if data is None else data


def set_seeds(seed: int):
    """Set seeds for random, numpy, and torch (including CUDA if available)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_condition(cond: str) -> str:
    """Normalize a condition string by sorting '+'-separated tokens."""
    return "+".join(sorted(cond.split("+")))


def load_adata(h5ad_path: str) -> ad.AnnData:
    """Load h5ad and apply condition normalization and cleanup."""
    adata = ad.read_h5ad(h5ad_path)
    adata.obs["condition"] = (
        adata.obs["condition"].astype(str).apply(normalize_condition).astype("category")
    )
    if "condition_name" in adata.obs:
        adata.obs.drop("condition_name", axis=1, inplace=True)
    return adata


def load_embedding_df(pickle_path: str) -> pd.DataFrame:
    """Load a pickle object into a DataFrame, transpose, and add a ctrl row."""
    with open(pickle_path, "rb") as handle:
        embd_df = pd.DataFrame(pickle.load(handle)).T
    ctrl_row = pd.DataFrame(
        [np.zeros(embd_df.shape[1])], columns=embd_df.columns, index=["ctrl"]
    )
    embd_df = pd.concat([ctrl_row, embd_df])
    return embd_df


def apply_alias_mapping(embd_df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """Apply dataset-specific alias mapping using hardcoded rules."""
    if dataset_name == "adamson":
        mapping = {
            "SARS1": "SARS",
            "DARS1": "DARS",
            "QARS1": "QARS",
            "TARS1": "TARS",
            "HARS1": "HARS",
            "CARS1": "CARS",
            "SRPRA": "SRPR",
            "MARS1": "MARS",
            "AARS1": "AARS",
            "PRELID3B": "SLMO2",
        }
        return embd_df.rename(index=mapping)
    if dataset_name == "replogle_k562_essential":
        mapping = {
            "AARS1": "AARS",
            "CENATAC": "CCDC84",
            "POLR1G": "CD3EAP",
            "DARS1": "DARS",
            "EPRS1": "EPRS",
            "HARS1": "HARS",
            "IARS1": "IARS",
            "KARS1": "KARS",
            "LARS1": "LARS",
            "MARS1": "MARS",
            "QARS1": "QARS",
            "RARS1": "RARS",
            "SARS1": "SARS",
            "TARS1": "TARS",
            "POLR1F": "TWISTNB",
            "VARS1": "VARS",
            "POLR1H": "ZNRD1",
        }
        return embd_df.rename(index=mapping)
    if dataset_name == "replogle_rpe1_essential":
        return embd_df.rename(index={"ZZZ3": "AC118549.1"})
    return embd_df


def densify_X(X) -> np.ndarray:
    """Convert a matrix to a float32 dense numpy array."""
    if sp.issparse(X):
        return X.toarray().astype(np.float32)
    return np.asarray(X, dtype=np.float32)


def nmse(pred: np.ndarray, true: np.ndarray, ctrl: np.ndarray) -> float:
    """Compute mean((pred-true)^2) / mean((ctrl-true)^2)."""
    pred = np.asarray(pred)
    true = np.asarray(true)
    ctrl = np.asarray(ctrl)
    numerator = np.mean((pred - true) ** 2)
    denominator = np.mean((ctrl - true) ** 2)
    return float(numerator / denominator)


def pearson_on_degs(
    pred: np.ndarray, true: np.ndarray, ctrl: np.ndarray, deg_idx: np.ndarray
) -> float:
    """Compute Pearson on deg_idx using (pred-ctrl) vs (true-ctrl)."""
    pred = np.asarray(pred)
    true = np.asarray(true)
    ctrl = np.asarray(ctrl)
    idx = np.asarray(deg_idx, dtype=int)
    delta_pred = pred[:, idx] - ctrl[:, idx]
    delta_true = true[:, idx] - ctrl[:, idx]
    a = delta_pred.reshape(-1)
    b = delta_true.reshape(-1)
    if a.size == 0 or np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def sample_control_indices(n_ctrl: int, size: int, seed: int) -> np.ndarray:
    """Sample control indices with replacement using a fixed seed."""
    np.random.seed(seed)
    return np.random.choice(n_ctrl, size=size, replace=True)
