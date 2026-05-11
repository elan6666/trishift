from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq
import scanpy as sc
from scipy import sparse


META_COLUMNS = ["cell_type", "sm_name", "sm_lincs_id", "SMILES", "control"]


def _read_gene_names(raw_dir: Path) -> list[str]:
    train_path = raw_dir / "adata_train.parquet"
    if train_path.exists():
        pf = pq.ParquetFile(train_path)
        if "gene" in pf.schema.names:
            genes: set[str] = set()
            for rg in range(pf.metadata.num_row_groups):
                arr = pf.read_row_group(rg, columns=["gene"]).column("gene")
                genes.update(str(x.as_py()) for x in pc.unique(arr))
                if (rg + 1) % 100 == 0 or rg + 1 == pf.metadata.num_row_groups:
                    print(
                        f"[prepare] scanned genes row_group={rg + 1}/{pf.metadata.num_row_groups} "
                        f"unique={len(genes):,}",
                        flush=True,
                    )
            return sorted(genes)

    de_path = raw_dir / "de_train.parquet"
    if de_path.exists():
        names = pq.ParquetFile(de_path).schema.names
        return [name for name in names if name not in META_COLUMNS]

    sample_path = raw_dir / "sample_submission.csv"
    if sample_path.exists():
        return [name for name in pd.read_csv(sample_path, nrows=0).columns if name != "id"]

    raise FileNotFoundError(
        "Could not infer gene names; expected de_train.parquet or sample_submission.csv"
    )


def _condition_from_obs(obs: pd.DataFrame, condition_source: str) -> pd.Series:
    control = obs["control"].astype(bool)
    values = obs[condition_source].astype(str)
    condition = values.where(~control, "ctrl")
    return condition.astype(str)


def _build_obs(raw_dir: Path, condition_source: str) -> pd.DataFrame:
    obs_path = raw_dir / "adata_obs_meta.csv"
    if not obs_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {obs_path}")

    obs = pd.read_csv(obs_path)
    required = {"obs_id", "control", condition_source}
    missing = sorted(required - set(obs.columns))
    if missing:
        raise ValueError(f"adata_obs_meta.csv is missing required columns: {missing}")

    obs = obs.sort_values("obs_id", kind="mergesort").set_index("obs_id", drop=False)
    obs.index = obs.index.astype(str)
    obs.index.name = None
    obs["condition"] = _condition_from_obs(obs, condition_source)
    obs["condition_name"] = obs["sm_name"].astype(str).where(
        ~obs["control"].astype(bool),
        "ctrl",
    )
    obs["drug_name"] = obs["sm_name"].astype(str)
    obs["is_control"] = obs["control"].astype(bool)
    return obs


def _create_template_h5ad(
    output_path: Path,
    obs: pd.DataFrame,
    genes: list[str],
    *,
    raw_dir: Path,
    matrix_value: str,
    condition_source: str,
) -> None:
    var = pd.DataFrame(index=pd.Index(genes, name=None))
    var["gene_name"] = genes

    empty = sparse.csr_matrix((obs.shape[0], len(genes)), dtype=np.float32)
    adata = ad.AnnData(X=empty, obs=obs, var=var)
    adata.uns["openproblems_source"] = {
        "raw_dir": str(raw_dir.resolve()),
        "matrix_value": matrix_value,
        "condition_source": condition_source,
        "condition_encoding": "ctrl for control rows; otherwise condition_source",
        "source_files": [
            "adata_train.parquet",
            "adata_obs_meta.csv",
            "de_train.parquet",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(output_path)


def _replace_x_with_streamed_csr(
    output_path: Path,
    parquet_path: Path,
    obs: pd.DataFrame,
    gene_to_idx: dict[str, int],
    *,
    matrix_value: str,
    compression: str | None,
    max_row_groups: int | None,
) -> tuple[int, int]:
    pf = pq.ParquetFile(parquet_path)
    if matrix_value not in pf.schema.names:
        raise ValueError(
            f"{parquet_path.name} has no column {matrix_value!r}; "
            f"available columns: {pf.schema.names}"
        )

    row_groups = pf.metadata.num_row_groups
    if max_row_groups is not None:
        row_groups = min(row_groups, int(max_row_groups))

    nnz_expected = (
        int(pf.metadata.num_rows)
        if max_row_groups is None
        else sum(int(pf.metadata.row_group(i).num_rows) for i in range(row_groups))
    )
    n_obs = int(obs.shape[0])
    n_vars = int(len(gene_to_idx))
    obs_to_idx = {obs_id: i for i, obs_id in enumerate(obs.index.astype(str))}

    with h5py.File(output_path, "a") as handle:
        x_group = handle["X"]
        for key in ("data", "indices", "indptr"):
            if key in x_group:
                del x_group[key]

        chunks = min(max(nnz_expected, 1), 1_000_000)
        data_ds = x_group.create_dataset(
            "data",
            shape=(nnz_expected,),
            dtype="float32",
            chunks=(chunks,),
            compression=compression,
        )
        indices_ds = x_group.create_dataset(
            "indices",
            shape=(nnz_expected,),
            dtype="int32",
            chunks=(chunks,),
            compression=compression,
        )
        indptr_ds = x_group.create_dataset(
            "indptr",
            shape=(n_obs + 1,),
            dtype="int64",
            chunks=(min(n_obs + 1, 1_000_000),),
            compression=compression,
        )
        x_group.attrs["encoding-type"] = "csr_matrix"
        x_group.attrs["encoding-version"] = "0.1.0"
        x_group.attrs["shape"] = np.array([n_obs, n_vars], dtype=np.int64)

        counts = np.zeros(n_obs, dtype=np.int64)
        write_pos = 0
        last_obs_id = ""

        for rg in range(row_groups):
            table = pf.read_row_group(rg, columns=["obs_id", "gene", matrix_value])
            df = table.to_pandas()
            if df.empty:
                continue

            first_obs_id = str(df["obs_id"].iloc[0])
            if last_obs_id and first_obs_id < last_obs_id:
                raise ValueError(
                    "adata_train.parquet is not sorted by obs_id; "
                    "streamed CSR construction would produce invalid row pointers"
                )
            last_obs_id = str(df["obs_id"].iloc[-1])

            gene_idx = df["gene"].map(gene_to_idx)
            if gene_idx.isna().any():
                missing = sorted(df.loc[gene_idx.isna(), "gene"].astype(str).unique())[:10]
                raise ValueError(f"adata_train.parquet contains unknown genes: {missing}")

            sizes = df.groupby("obs_id", sort=False).size()
            row_idx = sizes.index.astype(str).map(obs_to_idx)
            if pd.isna(row_idx).any():
                missing = sizes.index[pd.isna(row_idx)].astype(str).tolist()[:10]
                raise ValueError(f"adata_train.parquet contains unknown obs_id values: {missing}")

            counts[np.asarray(row_idx, dtype=np.int64)] += sizes.to_numpy(dtype=np.int64)

            n_rows = int(df.shape[0])
            next_pos = write_pos + n_rows
            data_ds[write_pos:next_pos] = df[matrix_value].to_numpy(dtype=np.float32, copy=False)
            indices_ds[write_pos:next_pos] = gene_idx.to_numpy(dtype=np.int32, copy=False)
            write_pos = next_pos

            if (rg + 1) % 50 == 0 or rg + 1 == row_groups:
                print(
                    f"[stream] row_group={rg + 1}/{row_groups} "
                    f"nnz={write_pos:,}/{nnz_expected:,}",
                    flush=True,
                )

        if write_pos != nnz_expected:
            raise RuntimeError(f"Expected {nnz_expected} entries, wrote {write_pos}")

        indptr = np.empty(n_obs + 1, dtype=np.int64)
        indptr[0] = 0
        np.cumsum(counts, out=indptr[1:])
        if int(indptr[-1]) != write_pos:
            raise RuntimeError(
                f"CSR pointer mismatch: indptr[-1]={int(indptr[-1])}, wrote={write_pos}"
            )
        indptr_ds[:] = indptr

    return write_pos, row_groups


def _compute_hvg_indices(
    input_path: Path,
    *,
    n_top_genes: int,
    n_bins: int = 20,
    data_chunk_size: int = 5_000_000,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Compute Scanpy/Seurat-style HVGs from log-normalized sparse X.

    GEARS preprocessing applies normalize_total + log1p and then calls
    scanpy.pp.highly_variable_genes with the default Seurat flavor. The Seurat
    flavor estimates dispersion on expm1(log-normalized) values and normalizes
    dispersion within mean-expression bins. This implementation mirrors that
    computation from the on-disk CSR arrays so large Kaggle matrices do not need
    to be fully loaded into memory.
    """
    if n_top_genes <= 0:
        raise ValueError("n_top_genes must be positive")

    with h5py.File(input_path, "r") as handle:
        shape = tuple(int(x) for x in handle["X"].attrs["shape"])
        n_obs, n_vars = shape
        data_ds = handle["X/data"]
        indices_ds = handle["X/indices"]
        nnz = int(data_ds.shape[0])

        sums = np.zeros(n_vars, dtype=np.float64)
        sums_sq = np.zeros(n_vars, dtype=np.float64)
        for start in range(0, nnz, data_chunk_size):
            end = min(start + data_chunk_size, nnz)
            values = np.expm1(data_ds[start:end].astype(np.float64))
            gene_idx = indices_ds[start:end].astype(np.int64)
            sums += np.bincount(gene_idx, weights=values, minlength=n_vars)
            sums_sq += np.bincount(gene_idx, weights=values * values, minlength=n_vars)
            if end == nnz or end % (data_chunk_size * 10) == 0:
                print(f"[hvg] stats nnz={end:,}/{nnz:,}", flush=True)

    means_raw = sums / float(n_obs)
    variances = (sums_sq / float(n_obs)) - (means_raw * means_raw)
    variances = np.maximum(variances, 0.0)
    if n_obs > 1:
        variances *= float(n_obs) / float(n_obs - 1)

    dispersions = np.full_like(means_raw, np.nan, dtype=np.float64)
    positive = means_raw > 0
    dispersions[positive] = variances[positive] / means_raw[positive]
    dispersions[positive] = np.log(np.maximum(dispersions[positive], 1e-12))
    means = np.log1p(means_raw)

    metrics = pd.DataFrame({"means": means, "dispersions": dispersions})
    metrics["mean_bin"] = pd.cut(metrics["means"], bins=int(n_bins))
    grouped = metrics.groupby("mean_bin", observed=False)["dispersions"]
    bin_mean = grouped.transform("mean").to_numpy(dtype=np.float64)
    bin_std = grouped.transform("std").to_numpy(dtype=np.float64)
    bin_std = np.where(np.isfinite(bin_std) & (bin_std > 0), bin_std, 1.0)

    dispersions_norm = (dispersions - bin_mean) / bin_std
    dispersions_norm[~np.isfinite(dispersions_norm)] = -np.inf
    metrics["dispersions_norm"] = dispersions_norm

    n_select = min(int(n_top_genes), int(n_vars))
    selected_unsorted = np.argpartition(-dispersions_norm, n_select - 1)[:n_select]
    selected_unsorted = selected_unsorted[
        np.argsort(-dispersions_norm[selected_unsorted], kind="mergesort")
    ]
    selected = np.sort(selected_unsorted.astype(np.int64))
    metrics["highly_variable"] = False
    metrics.iloc[selected, metrics.columns.get_loc("highly_variable")] = True
    return selected, metrics


def _write_hvg_subset_h5ad(
    input_path: Path,
    output_path: Path,
    selected_idx: np.ndarray,
    hvg_metrics: pd.DataFrame,
    *,
    compression: str | None,
    data_chunk_size_rows: int = 2048,
) -> None:
    selected_idx = np.asarray(selected_idx, dtype=np.int64)
    input_adata = ad.read_h5ad(input_path, backed="r")
    try:
        obs = input_adata.obs.copy()
        var = input_adata.var.iloc[selected_idx].copy()
        metrics_sel = hvg_metrics.iloc[selected_idx]
        for col in ("highly_variable", "means", "dispersions", "dispersions_norm"):
            var[col] = metrics_sel[col].to_numpy()
        var["highly_variable"] = var["highly_variable"].astype(bool)
        uns = dict(input_adata.uns)
        shape = input_adata.shape
    finally:
        input_adata.file.close()

    n_obs = int(shape[0])
    n_vars_old = int(shape[1])
    n_vars_new = int(selected_idx.size)
    selected_map = np.full(n_vars_old, -1, dtype=np.int32)
    selected_map[selected_idx] = np.arange(n_vars_new, dtype=np.int32)

    empty = sparse.csr_matrix((n_obs, n_vars_new), dtype=np.float32)
    out = ad.AnnData(X=empty, obs=obs, var=var)
    uns["hvg_filter"] = {
        "method": "scanpy.pp.highly_variable_genes compatible",
        "flavor": "seurat",
        "n_top_genes": int(n_vars_new),
        "input_h5ad": str(input_path.resolve()),
    }
    out.uns = uns
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.write_h5ad(output_path)

    with h5py.File(input_path, "r") as src, h5py.File(output_path, "a") as dst:
        src_indptr = src["X/indptr"][:]
        src_indices = src["X/indices"]
        src_data = src["X/data"]

        row_counts = np.zeros(n_obs, dtype=np.int64)
        for row_start in range(0, n_obs, data_chunk_size_rows):
            row_end = min(row_start + data_chunk_size_rows, n_obs)
            start = int(src_indptr[row_start])
            end = int(src_indptr[row_end])
            idx = src_indices[start:end].astype(np.int64)
            mapped = selected_map[idx]
            keep = mapped >= 0
            if np.any(keep):
                local_rows = np.repeat(
                    np.arange(row_end - row_start, dtype=np.int64),
                    np.diff(src_indptr[row_start : row_end + 1]),
                )
                row_counts[row_start:row_end] = np.bincount(
                    local_rows[keep],
                    minlength=row_end - row_start,
                )

        out_indptr = np.empty(n_obs + 1, dtype=np.int64)
        out_indptr[0] = 0
        np.cumsum(row_counts, out=out_indptr[1:])
        out_nnz = int(out_indptr[-1])

        x_group = dst["X"]
        for key in ("data", "indices", "indptr"):
            if key in x_group:
                del x_group[key]
        chunks = min(max(out_nnz, 1), 1_000_000)
        data_ds = x_group.create_dataset(
            "data",
            shape=(out_nnz,),
            dtype="float32",
            chunks=(chunks,),
            compression=compression,
        )
        indices_ds = x_group.create_dataset(
            "indices",
            shape=(out_nnz,),
            dtype="int32",
            chunks=(chunks,),
            compression=compression,
        )
        indptr_ds = x_group.create_dataset(
            "indptr",
            data=out_indptr,
            dtype="int64",
            chunks=(min(n_obs + 1, 1_000_000),),
            compression=compression,
        )
        x_group.attrs["encoding-type"] = "csr_matrix"
        x_group.attrs["encoding-version"] = "0.1.0"
        x_group.attrs["shape"] = np.array([n_obs, n_vars_new], dtype=np.int64)

        write_pos = 0
        for row_start in range(0, n_obs, data_chunk_size_rows):
            row_end = min(row_start + data_chunk_size_rows, n_obs)
            start = int(src_indptr[row_start])
            end = int(src_indptr[row_end])
            idx = src_indices[start:end].astype(np.int64)
            mapped = selected_map[idx]
            keep = mapped >= 0
            n_keep = int(np.sum(keep))
            if n_keep:
                next_pos = write_pos + n_keep
                data_ds[write_pos:next_pos] = src_data[start:end][keep].astype(np.float32)
                indices_ds[write_pos:next_pos] = mapped[keep].astype(np.int32)
                write_pos = next_pos
            if row_end == n_obs or row_end % (data_chunk_size_rows * 25) == 0:
                print(
                    f"[hvg] subset rows={row_end:,}/{n_obs:,} nnz={write_pos:,}/{out_nnz:,}",
                    flush=True,
                )

        if write_pos != out_nnz:
            raise RuntimeError(f"HVG subset write mismatch: wrote={write_pos}, expected={out_nnz}")
        _ = indptr_ds


def filter_h5ad_to_hvg(
    input_path: Path,
    output_path: Path,
    *,
    n_top_genes: int,
    compression: str | None = "lzf",
    force: bool = False,
) -> None:
    input_path = input_path.resolve()
    output_path = output_path.resolve()
    if output_path.exists() and not force:
        raise FileExistsError(f"Output already exists: {output_path}; pass --force to overwrite")
    if output_path.exists():
        output_path.unlink()

    print(f"[hvg] compute top {int(n_top_genes):,} genes from {input_path}", flush=True)
    selected, metrics = _compute_hvg_indices(input_path, n_top_genes=int(n_top_genes))
    print(f"[hvg] write filtered h5ad genes={selected.size:,}", flush=True)
    _write_hvg_subset_h5ad(
        input_path,
        output_path,
        selected,
        metrics,
        compression=compression,
    )
    print(f"[done] wrote HVG h5ad {output_path}", flush=True)


def _dense_mean_vector(X) -> np.ndarray:
    mean = X.mean(axis=0)
    return np.asarray(mean).reshape(-1).astype(np.float64, copy=False)


def _prepare_gears_obs_columns(adata: ad.AnnData) -> None:
    if "condition" not in adata.obs:
        raise ValueError("adata.obs['condition'] is required for GEARS-style DEG")
    if "cell_type" not in adata.obs:
        raise ValueError("adata.obs['cell_type'] is required for GEARS-style DEG")
    if "gene_name" not in adata.var:
        adata.var["gene_name"] = adata.var_names.astype(str)

    condition = adata.obs["condition"].astype(str)
    cell_type = adata.obs["cell_type"].astype(str)
    adata.obs["dose_val"] = "1"
    adata.obs["control"] = (condition == "ctrl").astype(int)
    adata.obs["condition_name"] = cell_type + "_" + condition + "_1"
    adata.obs["cov_drug_dose_name"] = adata.obs["condition_name"].astype(str)
    for col in ("condition", "cell_type", "condition_name", "cov_drug_dose_name"):
        adata.obs[col] = adata.obs[col].astype("category")


def add_gears_style_deg(
    input_path: Path,
    output_path: Path,
    *,
    force: bool = False,
    n_genes: int | None = None,
) -> None:
    """Add GEARS-compatible cell-type-aware DEG caches to an h5ad file."""
    input_path = input_path.resolve()
    output_path = output_path.resolve()
    same_output = input_path == output_path
    if output_path.exists() and not force and not same_output:
        raise FileExistsError(f"Output already exists: {output_path}; pass --force to overwrite")

    print(f"[deg] read {input_path}", flush=True)
    adata = ad.read_h5ad(input_path)
    _prepare_gears_obs_columns(adata)
    n_rank_genes = int(n_genes) if n_genes is not None and int(n_genes) > 0 else int(adata.n_vars)

    gene_dict: dict[str, list[str]] = {}
    cov_categories = adata.obs["cell_type"].astype(str).unique().tolist()
    for cov_cat in cov_categories:
        control_group_cov = f"{cov_cat}_ctrl_1"
        cov_mask = adata.obs["cell_type"].astype(str).values == str(cov_cat)
        adata_cov = adata[cov_mask].copy()
        groups = set(adata_cov.obs["condition_name"].astype(str).unique().tolist())
        if control_group_cov not in groups:
            print(f"[deg] skip cell_type={cov_cat!r}: missing {control_group_cov}", flush=True)
            continue
        group_counts = adata_cov.obs["condition_name"].astype(str).value_counts()
        valid_groups = [
            str(group)
            for group, count in group_counts.items()
            if str(group) != control_group_cov and int(count) >= 2
        ]
        skipped_small = int(sum(1 for group, count in group_counts.items() if str(group) != control_group_cov and int(count) < 2))
        if not valid_groups:
            print(f"[deg] skip cell_type={cov_cat!r}: no non-control group has >=2 cells", flush=True)
            continue

        print(
            f"[deg] rank_genes_groups cell_type={cov_cat!r} "
            f"groups={len(valid_groups):,} cells={adata_cov.n_obs:,} "
            f"skipped_small_groups={skipped_small:,}",
            flush=True,
        )
        sc.tl.rank_genes_groups(
            adata_cov,
            groupby="condition_name",
            groups=valid_groups,
            reference=control_group_cov,
            rankby_abs=True,
            n_genes=n_rank_genes,
            use_raw=False,
            method="t-test",
        )
        names_df = pd.DataFrame(adata_cov.uns["rank_genes_groups"]["names"])
        for group in names_df.columns:
            group_name = str(group)
            if group_name == control_group_cov:
                continue
            gene_dict[group_name] = names_df[group].astype(str).tolist()

    print(f"[deg] groups with DEG rankings={len(gene_dict):,}", flush=True)
    gene_id2idx = {gene: idx for idx, gene in enumerate(adata.var_names.astype(str))}
    gene_idx2id = np.asarray(adata.var_names.astype(str))
    condition_names = adata.obs["condition_name"].astype(str)
    cell_types = adata.obs["cell_type"].astype(str)

    non_zeros_gene_idx: dict[str, np.ndarray] = {}
    non_dropout_gene_idx: dict[str, np.ndarray] = {}
    top_non_dropout_de_20: dict[str, np.ndarray] = {}
    top_non_zero_de_20: dict[str, np.ndarray] = {}

    ctrl_mean_by_cell_type: dict[str, np.ndarray] = {}
    for cov_cat in cov_categories:
        ctrl_key = f"{cov_cat}_ctrl_1"
        mask = condition_names.values == ctrl_key
        if np.any(mask):
            ctrl_mean_by_cell_type[str(cov_cat)] = _dense_mean_vector(adata[mask].X)

    key_meta = (
        adata.obs[["condition_name", "cell_type", "condition"]]
        .astype(str)
        .drop_duplicates("condition_name")
        .set_index("condition_name")
    )
    for group_name, ranked_genes in gene_dict.items():
        if group_name not in key_meta.index:
            continue
        cov_cat = str(key_meta.loc[group_name, "cell_type"])
        ctrl_mean = ctrl_mean_by_cell_type.get(cov_cat)
        if ctrl_mean is None:
            continue
        pert_mask = condition_names.values == group_name
        if not np.any(pert_mask):
            continue
        pert_mean = _dense_mean_vector(adata[pert_mask].X)

        non_zero = np.where(pert_mean != 0)[0]
        zero = np.where(pert_mean == 0)[0]
        true_zeros = np.intersect1d(zero, np.where(ctrl_mean == 0)[0])
        non_dropouts = np.concatenate((non_zero, true_zeros))

        gene_idx_top = [gene_id2idx[g] for g in ranked_genes if g in gene_id2idx]
        non_dropout_20 = [i for i in gene_idx_top if i in non_dropouts][:20]
        non_zero_20 = [i for i in gene_idx_top if i in non_zero][:20]

        non_zeros_gene_idx[group_name] = np.sort(non_zero).astype(np.int64)
        non_dropout_gene_idx[group_name] = np.sort(non_dropouts).astype(np.int64)
        top_non_dropout_de_20[group_name] = gene_idx2id[non_dropout_20].astype(object)
        top_non_zero_de_20[group_name] = gene_idx2id[non_zero_20].astype(object)

    adata.uns["rank_genes_groups_cov_all"] = gene_dict
    adata.uns["top_non_dropout_de_20"] = top_non_dropout_de_20
    adata.uns["top_non_zero_de_20"] = top_non_zero_de_20
    adata.uns["non_dropout_gene_idx"] = non_dropout_gene_idx
    adata.uns["non_zeros_gene_idx"] = non_zeros_gene_idx
    adata.uns["rank_genes_groups_cov_all_meta"] = {
        "method": "scanpy.tl.rank_genes_groups",
        "scanpy_method": "t-test",
        "groupby": "condition_name",
        "covariate": "cell_type",
        "reference_pattern": "{cell_type}_ctrl_1",
        "rankby_abs": True,
        "n_genes": int(n_rank_genes),
        "top_n": 20,
        "condition_name_encoding": "cell_type + '_' + condition + '_1'",
    }

    target = output_path
    temp_path: Path | None = None
    if same_output:
        fd, temp_name = tempfile.mkstemp(
            prefix=output_path.stem + "_deg_",
            suffix=".h5ad",
            dir=output_path.parent,
        )
        os.close(fd)
        temp_path = Path(temp_name)
        target = temp_path
    elif target.exists():
        target.unlink()
    print(f"[deg] write {target}", flush=True)
    adata.write_h5ad(target)
    if temp_path is not None:
        os.replace(temp_path, output_path)
        print(f"[deg] replaced {output_path}", flush=True)
    print(
        f"[done] DEG caches groups={len(gene_dict):,} "
        f"top_non_dropout={len(top_non_dropout_de_20):,}",
        flush=True,
    )


def convert_openproblems(
    raw_dir: Path,
    output_path: Path,
    *,
    condition_source: str = "sm_lincs_id",
    matrix_value: str = "normalized_count",
    compression: str | None = "lzf",
    hvg_n_top_genes: int = 0,
    keep_full: bool = False,
    max_row_groups: int | None = None,
    force: bool = False,
) -> None:
    raw_dir = raw_dir.resolve()
    output_path = output_path.resolve()
    parquet_path = raw_dir / "adata_train.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"Missing expression file: {parquet_path}")
    if output_path.exists() and not force:
        raise FileExistsError(f"Output already exists: {output_path}; pass --force to overwrite")
    if output_path.exists():
        output_path.unlink()

    final_output_path = output_path
    temp_full_path: Path | None = None
    if int(hvg_n_top_genes) > 0:
        if keep_full:
            temp_full_path = output_path.with_name(output_path.stem + "_full" + output_path.suffix)
        else:
            fd, temp_name = tempfile.mkstemp(
                prefix=output_path.stem + "_full_",
                suffix=".h5ad",
                dir=output_path.parent,
            )
            os.close(fd)
            temp_full_path = Path(temp_name)
        output_path = temp_full_path
        if output_path.exists():
            output_path.unlink()

    print("[prepare] read metadata", flush=True)
    obs = _build_obs(raw_dir, condition_source=condition_source)
    genes = _read_gene_names(raw_dir)
    gene_to_idx = {gene: i for i, gene in enumerate(genes)}

    print(
        f"[prepare] obs={obs.shape[0]:,} genes={len(genes):,} "
        f"conditions={obs['condition'].nunique():,}",
        flush=True,
    )
    _create_template_h5ad(
        output_path,
        obs,
        genes,
        raw_dir=raw_dir,
        matrix_value=matrix_value,
        condition_source=condition_source,
    )

    nnz, row_groups = _replace_x_with_streamed_csr(
        output_path,
        parquet_path,
        obs,
        gene_to_idx,
        matrix_value=matrix_value,
        compression=compression,
        max_row_groups=max_row_groups,
    )
    print(
        f"[done] wrote {output_path} row_groups={row_groups:,} nnz={nnz:,}",
        flush=True,
    )
    if int(hvg_n_top_genes) > 0:
        filter_h5ad_to_hvg(
            output_path,
            final_output_path,
            n_top_genes=int(hvg_n_top_genes),
            compression=compression,
            force=True,
        )
        if temp_full_path is not None and not keep_full:
            temp_full_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Kaggle Open Problems single-cell perturbation RNA data to TriShift-style h5ad."
    )
    parser.add_argument("--raw-dir", default="src/data/openproblems_raw")
    parser.add_argument("--output", default="src/data/openproblems/perturb_processed.h5ad")
    parser.add_argument(
        "--hvg-input",
        default="",
        help="Existing h5ad to filter to HVGs; skips parquet-to-h5ad conversion.",
    )
    parser.add_argument(
        "--deg-input",
        default="",
        help="Existing h5ad to annotate with GEARS-style DEG caches; skips parquet-to-h5ad conversion.",
    )
    parser.add_argument(
        "--hvg-n-top-genes",
        type=int,
        default=0,
        help="If >0, filter output to this many Seurat-style HVGs.",
    )
    parser.add_argument(
        "--keep-full",
        action="store_true",
        help="When using --hvg-n-top-genes during conversion, keep the intermediate full-gene h5ad.",
    )
    parser.add_argument(
        "--add-gears-deg",
        action="store_true",
        help="After conversion/HVG filtering, add GEARS-style cell-type-aware DEG caches.",
    )
    parser.add_argument(
        "--deg-n-genes",
        type=int,
        default=0,
        help="Number of ranked genes per group for DEG; default ranks all genes.",
    )
    parser.add_argument(
        "--condition-source",
        default="sm_lincs_id",
        choices=["sm_lincs_id", "sm_name"],
        help="Column used for non-control condition labels.",
    )
    parser.add_argument(
        "--matrix-value",
        default="normalized_count",
        choices=["normalized_count", "count"],
        help="Expression value written to adata.X.",
    )
    parser.add_argument(
        "--compression",
        default="lzf",
        choices=["lzf", "gzip", "none"],
        help="HDF5 compression for X CSR arrays.",
    )
    parser.add_argument("--max-row-groups", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    compression = None if args.compression == "none" else args.compression
    if args.deg_input.strip():
        add_gears_style_deg(
            Path(args.deg_input),
            Path(args.output),
            force=bool(args.force),
            n_genes=int(args.deg_n_genes) if int(args.deg_n_genes) > 0 else None,
        )
        return

    if args.hvg_input.strip():
        if int(args.hvg_n_top_genes) <= 0:
            raise ValueError("--hvg-input requires --hvg-n-top-genes > 0")
        filter_h5ad_to_hvg(
            Path(args.hvg_input),
            Path(args.output),
            n_top_genes=int(args.hvg_n_top_genes),
            compression=compression,
            force=bool(args.force),
        )
        if bool(args.add_gears_deg):
            add_gears_style_deg(
                Path(args.output),
                Path(args.output),
                force=True,
                n_genes=int(args.deg_n_genes) if int(args.deg_n_genes) > 0 else None,
            )
        return

    convert_openproblems(
        Path(args.raw_dir),
        Path(args.output),
        condition_source=args.condition_source,
        matrix_value=args.matrix_value,
        compression=compression,
        hvg_n_top_genes=int(args.hvg_n_top_genes),
        keep_full=bool(args.keep_full),
        max_row_groups=args.max_row_groups,
        force=bool(args.force),
    )
    if bool(args.add_gears_deg):
        add_gears_style_deg(
            Path(args.output),
            Path(args.output),
            force=True,
            n_genes=int(args.deg_n_genes) if int(args.deg_n_genes) > 0 else None,
        )


if __name__ == "__main__":
    main()
