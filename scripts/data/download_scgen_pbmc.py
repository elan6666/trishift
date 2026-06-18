from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCGEN_SOURCE_URL = "https://drive.google.com/uc?id=1r87vhoLLq6PXAYdmyyd89zG90eJOFYLk"
SCGEN_MD5 = "9af3ddd4697fc3f8009da237f7cd5d4e"
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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download the Kang IFN-beta PBMC cross-cell dataset used by the "
            "scGen/TriShift cell-type transfer case. The file is sourced from "
            "the scGen perturbation-prediction tutorial backup file and written "
            "to the path expected by scripts/data/prepare_scgen_pbmc.py."
        )
    )
    parser.add_argument("--url", default=SCGEN_SOURCE_URL)
    parser.add_argument("--output", type=Path, default=DEFAULT_H5AD_PATH)
    parser.add_argument("--skip-md5", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    out_path = args.output if args.output.is_absolute() else (REPO_ROOT / args.output)
    _download(str(args.url), out_path, overwrite=bool(args.overwrite))
    if not bool(args.skip_md5):
        observed = _md5(out_path)
        if observed != SCGEN_MD5:
            raise RuntimeError(f"MD5 mismatch for {out_path}: expected {SCGEN_MD5}, observed {observed}")
        print(f"[md5-ok] {out_path}")
    print(f"[ready] {out_path}")


if __name__ == "__main__":
    main()
