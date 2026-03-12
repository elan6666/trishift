from __future__ import annotations

import logging
from typing import Dict, Optional, Union

import numpy as np
import scanpy as sc
import torch
from anndata import AnnData
from scanpy.get import _get_obs_rep, _set_obs_rep
from scipy.sparse import issparse

logger = logging.getLogger(__name__)


class Preprocessor:
    """
    Local copy of the scGPT preprocessing pipeline used in the integration example.

    The implementation intentionally mirrors external/scGPT-main/scgpt/preprocess.py
    with only minimal logger adaptation so this repo can preprocess PBMC data
    without importing the external package at runtime.
    """

    def __init__(
        self,
        use_key: Optional[str] = None,
        filter_gene_by_counts: Union[int, bool] = False,
        filter_cell_by_counts: Union[int, bool] = False,
        normalize_total: Union[float, bool] = 1e4,
        result_normed_key: Optional[str] = "X_normed",
        log1p: bool = False,
        result_log1p_key: str = "X_log1p",
        subset_hvg: Union[int, bool] = False,
        hvg_use_key: Optional[str] = None,
        hvg_flavor: str = "seurat_v3",
        binning: Optional[int] = None,
        result_binned_key: str = "X_binned",
    ):
        self.use_key = use_key
        self.filter_gene_by_counts = filter_gene_by_counts
        self.filter_cell_by_counts = filter_cell_by_counts
        self.normalize_total = normalize_total
        self.result_normed_key = result_normed_key
        self.log1p = log1p
        self.result_log1p_key = result_log1p_key
        self.subset_hvg = subset_hvg
        self.hvg_use_key = hvg_use_key
        self.hvg_flavor = hvg_flavor
        self.binning = binning
        self.result_binned_key = result_binned_key

    def __call__(self, adata: AnnData, batch_key: Optional[str] = None) -> Dict:
        key_to_process = self.use_key
        if key_to_process == "X":
            key_to_process = None
        is_logged = self.check_logged(adata, obs_key=key_to_process)

        if self.filter_gene_by_counts:
            logger.info("Filtering genes by counts ...")
            sc.pp.filter_genes(
                adata,
                min_counts=self.filter_gene_by_counts
                if isinstance(self.filter_gene_by_counts, int)
                else None,
            )

        if (
            isinstance(self.filter_cell_by_counts, int)
            and self.filter_cell_by_counts > 0
        ):
            logger.info("Filtering cells by counts ...")
            sc.pp.filter_cells(
                adata,
                min_counts=self.filter_cell_by_counts
                if isinstance(self.filter_cell_by_counts, int)
                else None,
            )

        if self.normalize_total:
            logger.info("Normalizing total counts ...")
            normed_ = sc.pp.normalize_total(
                adata,
                target_sum=self.normalize_total
                if isinstance(self.normalize_total, float)
                else None,
                layer=key_to_process,
                inplace=False,
            )["X"]
            key_to_process = self.result_normed_key or key_to_process
            _set_obs_rep(adata, normed_, layer=key_to_process)

        if self.log1p:
            logger.info("Log1p transforming ...")
            if is_logged:
                logger.warning(
                    "The input data seems to be already log1p transformed. "
                    "Set `log1p=False` to avoid double log1p transform."
                )
            if self.result_log1p_key:
                _set_obs_rep(
                    adata,
                    _get_obs_rep(adata, layer=key_to_process),
                    layer=self.result_log1p_key,
                )
                key_to_process = self.result_log1p_key
            sc.pp.log1p(adata, layer=key_to_process)

        if self.subset_hvg:
            logger.info("Subsetting highly variable genes ...")
            if batch_key is None:
                logger.warning(
                    "No batch_key is provided, will use all cells for HVG selection."
                )
            try:
                sc.pp.highly_variable_genes(
                    adata,
                    layer=self.hvg_use_key,
                    n_top_genes=self.subset_hvg
                    if isinstance(self.subset_hvg, int)
                    else None,
                    batch_key=batch_key,
                    flavor=self.hvg_flavor,
                    subset=True,
                )
            except ImportError as exc:
                # scanpy's seurat_v3 flavor requires scikit-misc. Fall back so the
                # local notebook can still run in lighter environments.
                if str(self.hvg_flavor) != "seurat_v3" or "skmisc" not in str(exc).lower():
                    raise
                logger.warning(
                    "scikit-misc is missing; falling back from HVG flavor "
                    "`seurat_v3` to `cell_ranger`."
                )
                sc.pp.highly_variable_genes(
                    adata,
                    layer=self.hvg_use_key,
                    n_top_genes=self.subset_hvg
                    if isinstance(self.subset_hvg, int)
                    else None,
                    batch_key=batch_key,
                    flavor="cell_ranger",
                    subset=True,
                )

        if self.binning:
            logger.info("Binning data ...")
            if not isinstance(self.binning, int):
                raise ValueError(f"Binning arg must be an integer, but got {self.binning}.")
            n_bins = self.binning
            binned_rows = []
            bin_edges = []
            layer_data = _get_obs_rep(adata, layer=key_to_process)
            layer_data = layer_data.toarray() if issparse(layer_data) else layer_data
            if layer_data.min() < 0:
                raise ValueError(
                    f"Assuming non-negative data, but got min value {layer_data.min()}."
                )
            for row in layer_data:
                if row.max() == 0:
                    logger.warning(
                        "The input data contains all zero rows. Please make sure "
                        "this is expected. You can use `filter_cell_by_counts` "
                        "to filter out all zero rows."
                    )
                    binned_rows.append(np.zeros_like(row, dtype=np.int64))
                    bin_edges.append(np.array([0] * n_bins))
                    continue
                non_zero_ids = row.nonzero()
                non_zero_row = row[non_zero_ids]
                bins = np.quantile(non_zero_row, np.linspace(0, 1, n_bins - 1))
                non_zero_digits = _digitize(non_zero_row, bins)
                assert non_zero_digits.min() >= 1
                assert non_zero_digits.max() <= n_bins - 1
                binned_row = np.zeros_like(row, dtype=np.int64)
                binned_row[non_zero_ids] = non_zero_digits
                binned_rows.append(binned_row)
                bin_edges.append(np.concatenate([[0], bins]))
            adata.layers[self.result_binned_key] = np.stack(binned_rows)
            adata.obsm["bin_edges"] = np.stack(bin_edges)
        return {}

    def check_logged(self, adata: AnnData, obs_key: Optional[str] = None) -> bool:
        data = _get_obs_rep(adata, layer=obs_key)
        max_, min_ = data.max(), data.min()
        if max_ > 30:
            return False
        if min_ < 0:
            return False

        non_zero = data[data > 0]
        if non_zero.size == 0:
            return False
        non_zero_min = non_zero.min()
        if non_zero_min >= 1:
            return False
        return True


def _digitize(x: np.ndarray, bins: np.ndarray, side: str = "both") -> np.ndarray:
    assert x.ndim == 1 and bins.ndim == 1
    if side == "one":
        return np.digitize(x, bins)
    if side == "both":
        left_digits = np.digitize(x, bins)
        right_digits = np.digitize(x, bins, right=True)

        rands = torch.rand(len(x))
        digits = rands * (right_digits - left_digits) + left_digits
        digits = np.ceil(digits.detach().cpu().numpy()).astype(np.int64)
        return digits
    raise ValueError(f"side must be 'one' or 'both', got {side}.")
