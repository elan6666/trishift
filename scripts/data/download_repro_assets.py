from __future__ import annotations

import argparse
import importlib.util
import urllib.request
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

PROTEIN_ASSETS = {
    "uniprot_prott5": {
        "url": "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/embeddings/UP000005640_9606/per-protein.h5",
        "path": "src/data/protein_embeddings/uniprot_prott5_human_per_protein.h5",
    },
    "zenodo_prott5": {
        "url": "https://zenodo.org/records/5047020/files/reduced_embeddings_file.h5?download=1",
        "path": "src/data/protein_embeddings/zenodo_prott5_human_reduced_embeddings_file.h5",
    },
    "hf_esm2_embeddings": {
        "url": "https://huggingface.co/datasets/Darkadin/ESM2_embeddings_Human_Mouse/resolve/main/ESM2_15B_Human_Mouse_Embeddings.npy",
        "path": "src/data/protein_embeddings/hf_esm2_15b_human_mouse_embeddings.npy",
    },
    "hf_esm2_metadata": {
        "url": "https://huggingface.co/datasets/Darkadin/ESM2_embeddings_Human_Mouse/resolve/main/ESM2_15B_Human_Mouse_Metadata.csv.gz",
        "path": "src/data/protein_embeddings/hf_esm2_15b_human_mouse_metadata.csv.gz",
    },
}

SCGPT_DRIVE_FILES = {
    "args.json": "1hh2zGKyWAx3DyovD30GStZ3QlzmSqdk1",
    "best_model.pt": "14AebJfGOUF047Eg40hk57HCtrb0fyDTm",
    "vocab.json": "1H3E_MJ-Dl36AQV6jLbna2EdvgPaqvqcC",
}


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else (REPO_ROOT / path)


def _download_url(url: str, dst: Path, *, overwrite: bool) -> None:
    if dst.exists() and not overwrite:
        print(f"[exists] {dst}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download] {url}")
    print(f"[to] {dst}")
    urllib.request.urlretrieve(url, dst)


def _download_protein_assets(*, overwrite: bool) -> None:
    for name, spec in PROTEIN_ASSETS.items():
        print(f"[asset] {name}")
        _download_url(str(spec["url"]), _resolve(spec["path"]), overwrite=overwrite)


def _require_gdown():
    if importlib.util.find_spec("gdown") is None:
        raise RuntimeError(
            "Downloading the official scGPT Google Drive checkpoint requires gdown. "
            "Install it with `pip install gdown`, or manually place args.json, "
            "best_model.pt, and vocab.json under artifacts/models/scGPT_human."
        )
    import gdown

    return gdown


def _download_scgpt(*, output_dir: Path, overwrite: bool) -> None:
    gdown = _require_gdown()
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, file_id in SCGPT_DRIVE_FILES.items():
        dst = output_dir / filename
        if dst.exists() and not overwrite:
            print(f"[exists] {dst}")
            continue
        print(f"[download] scGPT {filename} -> {dst}")
        gdown.download(id=file_id, output=str(dst), quiet=False, fuzzy=True)
    print(f"[ready] scGPT pretrained root: {output_dir}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download optional large reproduction assets that are not tracked in git: "
            "protein embeddings for the scGen IFN-beta prior and the scGPT human pretrained checkpoint."
        )
    )
    parser.add_argument(
        "--assets",
        nargs="+",
        choices=["protein", "scgpt", "all"],
        default=["all"],
        help="Asset groups to download.",
    )
    parser.add_argument("--scgpt-output", type=Path, default=REPO_ROOT / "artifacts" / "models" / "scGPT_human")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    assets = set(args.assets)
    if "all" in assets:
        assets = {"protein", "scgpt"}
    if "protein" in assets:
        _download_protein_assets(overwrite=bool(args.overwrite))
    if "scgpt" in assets:
        _download_scgpt(output_dir=_resolve(args.scgpt_output), overwrite=bool(args.overwrite))


if __name__ == "__main__":
    main()
