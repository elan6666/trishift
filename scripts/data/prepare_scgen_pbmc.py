from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scanpy as sc


REPO_ROOT = Path(__file__).resolve().parents[2]
IFNB1_GENE = "IFNB1"
IFNB1_UNIPROT = "P01574"
IFNB1_ENTRY = "IFNB_HUMAN"
SCGEN_CONDITION = "stimulated"


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def _as_float32_vector(vec: np.ndarray | list[float], *, name: str) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} produced an empty embedding vector")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} embedding contains non-finite values")
    return arr


def _write_processed_h5ad(input_path: Path, output_path: Path, *, compression: str | None) -> None:
    adata = sc.read_h5ad(input_path)
    if "condition" not in adata.obs.columns:
        raise ValueError(f"{input_path} is missing obs['condition']")
    if "cell_type" not in adata.obs.columns:
        raise ValueError(f"{input_path} is missing obs['cell_type']")

    obs = adata.obs.copy()
    obs["condition_raw"] = obs["condition"].astype(str)
    obs["condition"] = obs["condition_raw"].replace({"control": "ctrl"}).astype("category")
    obs["condition_name"] = obs["condition"].astype(str)
    adata.obs = obs

    adata.var = adata.var.copy()
    if "gene_name" not in adata.var.columns:
        if "gene_symbol" in adata.var.columns:
            adata.var["gene_name"] = adata.var["gene_symbol"].astype(str).values
        else:
            adata.var["gene_name"] = adata.var_names.astype(str).values

    condition_values = set(adata.obs["condition"].astype(str).unique())
    required = {"ctrl", SCGEN_CONDITION}
    missing = sorted(required - condition_values)
    if missing:
        raise ValueError(f"Processed scGen data is missing required conditions: {missing}")

    adata.uns["trishift_scgen_source"] = {
        "input_path": str(input_path),
        "condition_mapping": {"control": "ctrl", "stimulated": SCGEN_CONDITION},
        "perturbation": "IFN-beta stimulation",
        "condition_label": SCGEN_CONDITION,
        "prior_symbol": IFNB1_GENE,
        "prior_uniprot": IFNB1_UNIPROT,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(output_path, compression=compression)
    print(f"[scgen] wrote processed h5ad: {output_path}")
    print(f"[scgen] shape={adata.shape}")
    print(f"[scgen] conditions={adata.obs['condition'].astype(str).value_counts().to_dict()}")
    print(f"[scgen] cell_types={adata.obs['cell_type'].astype(str).value_counts().to_dict()}")


def _load_uniprot_prott5(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as h5:
        if IFNB1_UNIPROT not in h5:
            raise KeyError(f"{path} does not contain UniProt accession {IFNB1_UNIPROT}")
        return _as_float32_vector(h5[IFNB1_UNIPROT][()], name="UniProt ProtT5")


def _load_zenodo_prott5(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as h5:
        for key in h5.keys():
            original_id = h5[key].attrs.get("original_id", "")
            if isinstance(original_id, bytes):
                original_id = original_id.decode("utf-8")
            parts = str(original_id).split("|")
            if len(parts) >= 3 and parts[1] == IFNB1_UNIPROT:
                return _as_float32_vector(h5[key][()], name="Zenodo ProtT5")
    raise KeyError(f"{path} does not contain {IFNB1_UNIPROT}/{IFNB1_ENTRY}")


def _load_hf_esm2(embedding_path: Path, metadata_path: Path) -> np.ndarray:
    meta = pd.read_csv(metadata_path)
    accession = meta["Accession"].astype(str) if "Accession" in meta.columns else pd.Series([], dtype=str)
    organism = meta["Organism"].astype(str) if "Organism" in meta.columns else pd.Series("", index=meta.index)
    mask = (accession == IFNB1_UNIPROT) & organism.str.contains("Human", case=False, na=False)
    rows = meta.index[mask].tolist()
    if not rows:
        raise KeyError(f"{metadata_path} does not contain human accession {IFNB1_UNIPROT}")
    arr = np.load(embedding_path, mmap_mode="r")
    row = int(rows[0])
    if row >= arr.shape[0]:
        raise IndexError(f"Metadata row {row} is outside embedding matrix with shape {arr.shape}")
    return _as_float32_vector(arr[row], name="HuggingFace ESM2")


def _load_genept(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        obj = pickle.load(handle)
    if IFNB1_GENE not in obj:
        raise KeyError(f"{path} does not contain {IFNB1_GENE}")
    return _as_float32_vector(obj[IFNB1_GENE], name="GenePT")


def _write_prior(path: Path, vector: np.ndarray) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump({SCGEN_CONDITION: vector.astype(np.float32, copy=False)}, handle)
    return {
        "path": str(path),
        "condition_key": SCGEN_CONDITION,
        "shape": [int(vector.shape[0])],
        "dtype": "float32",
    }


def _write_priors(args: argparse.Namespace) -> None:
    out_dir = _resolve(args.prior_dir)
    protein_dir = _resolve(args.protein_embedding_dir)
    genept_path = _resolve(args.genept_path)
    sources = {
        "emb_scgen_ifnb1_uniprot_prott5": {
            "file": out_dir / "scgen_ifnb1_uniprot_prott5.pickle",
            "loader": lambda: _load_uniprot_prott5(protein_dir / "uniprot_prott5_human_per_protein.h5"),
            "source": "UniProt official human per-protein ProtT5 embedding",
        },
        "emb_scgen_ifnb1_zenodo_prott5": {
            "file": out_dir / "scgen_ifnb1_zenodo_prott5.pickle",
            "loader": lambda: _load_zenodo_prott5(protein_dir / "zenodo_prott5_human_reduced_embeddings_file.h5"),
            "source": "Zenodo human SwissProt reduced ProtT5 embedding",
        },
        "emb_scgen_ifnb1_esm2_15b": {
            "file": out_dir / "scgen_ifnb1_esm2_15b.pickle",
            "loader": lambda: _load_hf_esm2(
                protein_dir / "hf_esm2_15b_human_mouse_embeddings.npy",
                protein_dir / "hf_esm2_15b_human_mouse_metadata.csv.gz",
            ),
            "source": "HuggingFace ESM2-15B human/mouse precomputed embedding",
        },
        "emb_scgen_ifnb1_genept": {
            "file": out_dir / "scgen_ifnb1_genept.pickle",
            "loader": lambda: _load_genept(genept_path),
            "source": "GenePT IFNB1 gene text embedding",
        },
    }

    metadata = {
        "condition_key": SCGEN_CONDITION,
        "biological_target": "IFN-beta stimulation",
        "gene_symbol": IFNB1_GENE,
        "uniprot_accession": IFNB1_UNIPROT,
        "uniprot_entry": IFNB1_ENTRY,
        "priors": {},
    }
    for key, spec in sources.items():
        vec = spec["loader"]()
        rec = _write_prior(spec["file"], vec)
        rec["source"] = spec["source"]
        metadata["priors"][key] = rec
        print(f"[scgen] wrote {key}: {spec['file']} dim={vec.shape[0]}")

    meta_path = out_dir / "scgen_ifnb1_priors_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=False), encoding="utf-8")
    print(f"[scgen] wrote prior metadata: {meta_path}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Prepare scGen PBMC IFN-beta data and TriShift priors")
    ap.add_argument("--input", default="src/data/scgen/train_kang_scgen.h5ad")
    ap.add_argument("--output", default="src/data/scgen/perturb_processed.h5ad")
    ap.add_argument("--prior_dir", default="src/data/scgen/priors")
    ap.add_argument("--protein_embedding_dir", default="src/data/protein_embeddings")
    ap.add_argument("--genept_path", default="src/data/Data_GeneEmbd/GenePT_gene_embedding_ada_text.pickle")
    ap.add_argument("--compression", default="gzip", help="h5ad compression, or empty for none")
    ap.add_argument("--skip_h5ad", action="store_true")
    ap.add_argument("--skip_priors", action="store_true")
    args = ap.parse_args(argv)

    if not args.skip_h5ad:
        _write_processed_h5ad(
            _resolve(args.input),
            _resolve(args.output),
            compression=(args.compression.strip() or None),
        )
    if not args.skip_priors:
        _write_priors(args)


if __name__ == "__main__":
    main()
