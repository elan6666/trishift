from __future__ import annotations

import argparse
import gzip
import hashlib
import shutil
import urllib.request
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCGEN_ZENODO_URL = "https://zenodo.org/records/14607156/files/kangCrossCell.h5ad.gz?download=1"
SCGEN_MD5 = "f787133ae1fefa3ee05d002bc7be26da"
DEFAULT_GZ_PATH = REPO_ROOT / "src" / "data" / "scgen" / "kangCrossCell.h5ad.gz"
DEFAULT_H5AD_PATH = REPO_ROOT / "src" / "data" / "scgen" / "train_kang_scgen.h5ad"


def _download(url: str, dst: Path, *, overwrite: bool) -> None:
    if dst.exists() and not overwrite:
        print(f"[exists] {dst}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download] {url}")
    print(f"[to] {dst}")
    urllib.request.urlretrieve(url, dst)


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _gunzip(src: Path, dst: Path, *, overwrite: bool) -> None:
    if dst.exists() and not overwrite:
        print(f"[exists] {dst}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"[decompress] {src} -> {dst}")
    with gzip.open(src, "rb") as inp, dst.open("wb") as out:
        shutil.copyfileobj(inp, out)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download the Kang IFN-beta PBMC cross-cell dataset used by the "
            "scGen/TriShift cell-type transfer case. The file is sourced from "
            "Zenodo record 14607156 as kangCrossCell.h5ad.gz and decompressed "
            "to the path expected by scripts/data/prepare_scgen_pbmc.py."
        )
    )
    parser.add_argument("--url", default=SCGEN_ZENODO_URL)
    parser.add_argument("--gz-path", type=Path, default=DEFAULT_GZ_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_H5AD_PATH)
    parser.add_argument("--skip-md5", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    gz_path = args.gz_path if args.gz_path.is_absolute() else (REPO_ROOT / args.gz_path)
    out_path = args.output if args.output.is_absolute() else (REPO_ROOT / args.output)
    _download(str(args.url), gz_path, overwrite=bool(args.overwrite))
    if not bool(args.skip_md5):
        observed = _md5(gz_path)
        if observed != SCGEN_MD5:
            raise RuntimeError(f"MD5 mismatch for {gz_path}: expected {SCGEN_MD5}, observed {observed}")
        print(f"[md5-ok] {gz_path}")
    _gunzip(gz_path, out_path, overwrite=bool(args.overwrite))
    print(f"[ready] {out_path}")


if __name__ == "__main__":
    main()
