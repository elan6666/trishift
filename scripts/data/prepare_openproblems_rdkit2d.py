from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


def _load_chemprop_rdkit2d():
    try:
        from chemprop.features.features_generators import (
            rdkit_2d_normalized_features_generator,
        )
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise ImportError(
            "BioLORD-style RDKit2D requires chemprop 1.x and descriptastorus. "
            "Install with: python -m pip install 'chemprop==1.6.1' descriptastorus rdkit"
        ) from exc
    return rdkit_2d_normalized_features_generator


def _load_rdkit_descriptors():
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise ImportError(
            "RDKit is required for descriptor fallback. Install with: python -m pip install rdkit"
        ) from exc
    return Chem, Descriptors


def _condition_smiles_table(adata: ad.AnnData) -> pd.DataFrame:
    required = {"condition", "SMILES"}
    missing = sorted(required - set(adata.obs.columns))
    if missing:
        raise ValueError(f"h5ad obs is missing required columns: {missing}")

    obs = adata.obs.copy()
    obs["condition"] = obs["condition"].astype(str)
    obs = obs[obs["condition"] != "ctrl"]
    cols = ["condition", "SMILES"]
    if "dose_uM" in obs.columns:
        cols.append("dose_uM")
    optional = [c for c in ["sm_lincs_id", "sm_name", "drug_name"] if c in obs.columns]
    table = obs[cols + optional].drop_duplicates("condition").set_index("condition")
    table.index = table.index.astype(str)
    table["SMILES"] = table["SMILES"].astype(str)
    if "dose_uM" in table.columns:
        table["dose_uM"] = pd.to_numeric(table["dose_uM"], errors="coerce").fillna(0.0)
    return table.sort_index()


def _calc_chemprop_rdkit2d_matrix(smiles_by_condition: pd.Series) -> tuple[pd.DataFrame, list[str]]:
    generator = _load_chemprop_rdkit2d()
    rows: list[list[float]] = []
    invalid: list[str] = []
    width: int | None = None

    for condition, smiles in smiles_by_condition.items():
        try:
            vals = list(generator(str(smiles)))
        except Exception:
            invalid.append(str(condition))
            vals = [] if width is None else [0.0] * width
        if width is None:
            width = len(vals)
        if len(vals) != width:
            invalid.append(str(condition))
            vals = [0.0] * int(width)
        rows.append([float(v) if np.isfinite(float(v)) else 0.0 for v in vals])

    if width is None:
        raise ValueError("No SMILES values were provided")
    columns = [f"rdkit2d_{i:03d}" for i in range(width)]
    desc = pd.DataFrame(rows, index=smiles_by_condition.index.astype(str), columns=columns)
    return desc.astype(np.float32), invalid


def _calc_rdkit_descriptor_matrix(smiles_by_condition: pd.Series) -> tuple[pd.DataFrame, list[str]]:
    Chem, Descriptors = _load_rdkit_descriptors()
    desc_items = list(Descriptors._descList)
    desc_names = [name for name, _ in desc_items]
    rows: list[list[float]] = []
    invalid: list[str] = []

    for condition, smiles in smiles_by_condition.items():
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            invalid.append(str(condition))
            rows.append([np.nan] * len(desc_items))
            continue
        vals = []
        for _name, fn in desc_items:
            try:
                val = float(fn(mol))
            except Exception:
                val = np.nan
            vals.append(val if np.isfinite(val) else np.nan)
        rows.append(vals)

    desc = pd.DataFrame(rows, index=smiles_by_condition.index.astype(str), columns=desc_names)
    return desc, invalid


def _zscore_descriptors(desc: pd.DataFrame) -> pd.DataFrame:
    values = desc.astype(float)
    med = values.median(axis=0, skipna=True)
    values = values.fillna(med).fillna(0.0)
    std = values.std(axis=0, ddof=0)
    keep = std.replace(0.0, np.nan).notna()
    values = values.loc[:, keep]
    mean = values.mean(axis=0)
    std = values.std(axis=0, ddof=0).replace(0.0, 1.0)
    z = (values - mean) / std
    return z.astype(np.float32)


def _write_pickle_embedding(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {str(idx): row.astype(float).tolist() for idx, row in df.iterrows()}
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def _attach_obsm(
    adata_path: Path,
    rdkit_df: pd.DataFrame,
    *,
    normalization: str,
    backend: str,
) -> None:
    adata = ad.read_h5ad(adata_path)
    cond = adata.obs["condition"].astype(str)
    rdkit_lookup = {idx: i for i, idx in enumerate(rdkit_df.index.astype(str))}
    rdkit_arr = np.zeros((adata.n_obs, rdkit_df.shape[1]), dtype=np.float32)
    rdkit_values = rdkit_df.to_numpy(dtype=np.float32)

    for condition, idx in rdkit_lookup.items():
        mask = cond.values == condition
        if np.any(mask):
            rdkit_arr[mask] = rdkit_values[idx]

    adata.obsm["rdkit2d"] = rdkit_arr
    if "rdkit2d_dose" in adata.obsm:
        del adata.obsm["rdkit2d_dose"]
    adata.uns["rdkit2d_meta"] = {
        "generator": "scripts/data/prepare_openproblems_rdkit2d.py",
        "backend": str(backend),
        "descriptor_source": str(normalization),
        "normalization": str(normalization),
        "condition_key": "condition",
        "smiles_key": "SMILES",
        "n_descriptors": int(rdkit_df.shape[1]),
    }
    adata.write_h5ad(adata_path)


def build_rdkit2d(
    h5ad_path: Path,
    output_pickle: Path,
    *,
    output_csv: Path | None = None,
    attach_obsm: bool = False,
    backend: str = "chemprop",
) -> None:
    h5ad_path = h5ad_path.resolve()
    print(f"[rdkit2d] read {h5ad_path}", flush=True)
    adata = ad.read_h5ad(h5ad_path, backed="r")
    try:
        table = _condition_smiles_table(adata)
    finally:
        adata.file.close()

    print(f"[rdkit2d] conditions={len(table):,}", flush=True)
    backend = str(backend).strip().lower()
    if backend == "chemprop":
        rdkit_df, invalid = _calc_chemprop_rdkit2d_matrix(table["SMILES"])
        normalization = "chemprop.features.features_generators.rdkit_2d_normalized_features_generator"
    elif backend == "rdkit-descriptors":
        desc_raw, invalid = _calc_rdkit_descriptor_matrix(table["SMILES"])
        rdkit_df = _zscore_descriptors(desc_raw)
        normalization = "median impute, drop zero-variance descriptors, z-score across drug conditions"
    else:
        raise ValueError("backend must be one of: chemprop, rdkit-descriptors")
    if invalid:
        print(f"[rdkit2d] invalid SMILES skipped/zero-imputed: {invalid[:10]}", flush=True)

    _write_pickle_embedding(output_pickle.resolve(), rdkit_df)
    print(f"[rdkit2d] wrote embedding pickle {output_pickle.resolve()}", flush=True)
    if output_csv is not None:
        output_csv = output_csv.resolve()
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        meta = table[[c for c in ["sm_lincs_id", "sm_name", "drug_name", "SMILES", "dose_uM"] if c in table.columns]]
        pd.concat([meta, rdkit_df], axis=1).to_csv(output_csv, encoding="utf-8-sig")
        print(f"[rdkit2d] wrote csv {output_csv}", flush=True)
    if attach_obsm:
        print("[rdkit2d] attach obsm rdkit2d", flush=True)
        _attach_obsm(h5ad_path, rdkit_df, normalization=normalization, backend=backend)

    print(f"[done] rdkit2d dims={rdkit_df.shape[1]:,}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build RDKit2D condition embeddings for OpenProblems drug perturbations."
    )
    parser.add_argument("--h5ad", default="src/data/openproblems/perturb_processed_hvg5000.h5ad")
    parser.add_argument("--output", default="src/data/openproblems/openproblems_rdkit2d.pickle")
    parser.add_argument("--output-csv", default="src/data/openproblems/openproblems_rdkit2d.csv")
    parser.add_argument("--attach-obsm", action="store_true")
    parser.add_argument(
        "--backend",
        default="chemprop",
        choices=["chemprop", "rdkit-descriptors"],
        help="RDKit2D backend. chemprop matches BioLORD's preprocessing notebook.",
    )
    args = parser.parse_args()

    build_rdkit2d(
        Path(args.h5ad),
        Path(args.output),
        output_csv=Path(args.output_csv) if args.output_csv else None,
        attach_obsm=bool(args.attach_obsm),
        backend=str(args.backend),
    )


if __name__ == "__main__":
    main()
