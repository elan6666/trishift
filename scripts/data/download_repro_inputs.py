from __future__ import annotations

import argparse
import hashlib
import importlib
import shutil
import sys
import zipfile
from pathlib import Path

import urllib.request


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DOWNLOAD_CACHE = REPO_ROOT / "src" / "data" / "_downloads"
GENE_EMBEDDING_DIR = REPO_ROOT / "src" / "data" / "Data_GeneEmbd"

GENEPT_V2_ZIP_URL = "https://zenodo.org/records/10833191/files/GenePT_emebdding_v2.zip?download=1"
GENEPT_V2_ZIP_MD5 = "3f6ce4317e3a0091978ae5cb8fbf05a3"
GENEPT_V2_FILES = {
    "GenePT_gene_embedding_ada_text.pickle": GENE_EMBEDDING_DIR / "GenePT_gene_embedding_ada_text.pickle",
    "GenePT_gene_protein_embedding_model_3_text.pickle": GENE_EMBEDDING_DIR
    / "GenePT_gene_protein_embedding_model_3_text.pickle",
}

GENEPT_LEGACY_ZIP_URL = "https://zenodo.org/records/10030426/files/data_embedding.zip?download=1"
GENEPT_LEGACY_ZIP_MD5 = "eabb1f71dfb3580f5855ff0d3217c715"
GENEPT_LEGACY_FILES = {
    "GPT_3_5_gene_embeddings.pickle": GENE_EMBEDDING_DIR / "GPT_3_5_gene_embeddings.pickle",
}


def _download_url(url: str, dst: Path, *, overwrite: bool) -> None:
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


def _verify_md5(path: Path, expected: str | None, *, skip_md5: bool) -> None:
    if skip_md5 or not expected:
        return
    observed = _md5(path)
    if observed != expected:
        raise RuntimeError(f"MD5 mismatch for {path}: expected {expected}, observed {observed}")
    print(f"[md5-ok] {path}")


def _extract_selected(zip_path: Path, members: dict[str, Path], *, overwrite: bool) -> None:
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        basename_to_member = {Path(name).name: name for name in names if not name.endswith("/")}
        missing = sorted(set(members) - set(basename_to_member))
        if missing:
            preview = "\n".join(names[:50])
            raise FileNotFoundError(
                f"Missing expected file(s) in {zip_path}: {missing}\n"
                f"First archive members:\n{preview}"
            )
        for basename, dst in members.items():
            if dst.exists() and not overwrite:
                print(f"[exists] {dst}")
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            member = basename_to_member[basename]
            print(f"[extract] {member} -> {dst}")
            with archive.open(member) as src, dst.open("wb") as out:
                shutil.copyfileobj(src, out)


def _run_module(module_name: str, argv: list[str] | None = None) -> None:
    module = importlib.import_module(module_name)
    main = getattr(module, "main", None)
    if main is None:
        raise AttributeError(f"{module_name} has no main()")
    if argv is None:
        main()
    else:
        main(argv)


def _download_benchmark(args: argparse.Namespace) -> None:
    module = importlib.import_module("scripts.data.download_and_prepare_benchmark_data")
    module.run(
        datasets=list(args.benchmark_datasets),
        paths_yaml=Path(args.paths_yaml).resolve(),
        sync_outer_h5ad=not bool(args.no_sync),
        cleanup_archives=not bool(args.keep_archives),
    )


def _download_scgen(args: argparse.Namespace) -> None:
    argv: list[str] = []
    if args.overwrite:
        argv.append("--overwrite")
    if args.skip_md5:
        argv.append("--skip-md5")
    _run_module("scripts.data.download_scgen_pbmc", argv)
    if args.prepare_scgen:
        prep_argv = []
        if args.skip_scgen_priors:
            prep_argv.append("--skip_priors")
        _run_module("scripts.data.prepare_scgen_pbmc", prep_argv)


def _download_genept(args: argparse.Namespace) -> None:
    v2_zip = DOWNLOAD_CACHE / "GenePT_emebdding_v2.zip"
    _download_url(GENEPT_V2_ZIP_URL, v2_zip, overwrite=bool(args.overwrite))
    _verify_md5(v2_zip, GENEPT_V2_ZIP_MD5, skip_md5=bool(args.skip_md5))
    _extract_selected(v2_zip, GENEPT_V2_FILES, overwrite=bool(args.overwrite))

    if not args.skip_legacy_genept:
        legacy_zip = DOWNLOAD_CACHE / "data_embedding.zip"
        _download_url(GENEPT_LEGACY_ZIP_URL, legacy_zip, overwrite=bool(args.overwrite))
        _verify_md5(legacy_zip, GENEPT_LEGACY_ZIP_MD5, skip_md5=bool(args.skip_md5))
        _extract_selected(legacy_zip, GENEPT_LEGACY_FILES, overwrite=bool(args.overwrite))


def _download_assets(args: argparse.Namespace, assets: list[str]) -> None:
    argv = ["--assets", *assets]
    if args.overwrite:
        argv.append("--overwrite")
    _run_module("scripts.data.download_repro_assets", argv)


def _prepare_biolord(args: argparse.Namespace) -> None:
    argv = ["--datasets", *args.benchmark_datasets]
    if args.overwrite:
        argv.append("--overwrite")
    _run_module("scripts.data.prepare_biolord_perturbation_data", argv)


def _check_inputs(args: argparse.Namespace) -> None:
    argv = ["--scope", args.check_scope]
    if args.strict_check:
        argv.append("--strict")
    _run_module("scripts.setup.check_repro_inputs", argv)


def _print_manual_notes(selected: set[str]) -> None:
    if "genept" in selected:
        print(
            "[note] emb_a / scELMo (`src/data/Data_GeneEmbd/ensem_emb_gpt3.5all_new.pickle`) "
            "is not downloaded by this script because the repository does not have a stable tracked "
            "direct-download source for that exact legacy file. The default TriShift configs use GenePT "
            "`emb_b`, which is downloaded here."
        )


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Unified downloader/preparer for TriShift reproducibility inputs: benchmark datasets, "
            "scGen PBMC, GenePT embeddings, protein embeddings, scGPT checkpoint, and BioLORD inputs."
        )
    )
    parser.add_argument(
        "--items",
        nargs="+",
        choices=["benchmark", "scgen", "genept", "protein", "scgpt", "biolord", "all"],
        default=["all"],
        help=(
            "What to fetch or prepare. `all` means benchmark, scgen, genept, protein, scgpt, and biolord. "
            "Use smaller groups for partial setup."
        ),
    )
    parser.add_argument(
        "--benchmark-datasets",
        nargs="+",
        default=["adamson", "dixit", "norman"],
        help="Benchmark datasets for GEARS/PertData download and BioLORD preprocessing.",
    )
    parser.add_argument("--paths-yaml", default=str(REPO_ROOT / "configs" / "paths.yaml"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-md5", action="store_true")
    parser.add_argument("--keep-archives", action="store_true")
    parser.add_argument("--no-sync", action="store_true", help="Do not sync GEARS h5ad files to src/data/<dataset>.")
    parser.add_argument(
        "--no-prepare-scgen",
        dest="prepare_scgen",
        action="store_false",
        help="Only download scGen raw h5ad.gz and decompressed h5ad; do not run prepare_scgen_pbmc.py.",
    )
    parser.add_argument(
        "--skip-scgen-priors",
        action="store_true",
        help="Pass --skip_priors to prepare_scgen_pbmc.py.",
    )
    parser.add_argument(
        "--skip-legacy-genept",
        action="store_true",
        help="Skip Zenodo record 10030426 and do not extract GPT_3_5_gene_embeddings.pickle for emb_c.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Run scripts/setup/check_repro_inputs.py after downloads/preparation.",
    )
    parser.add_argument(
        "--check-scope",
        choices=["all", "benchmark", "baselines", "scgen", "scgpt"],
        default="all",
    )
    parser.add_argument("--strict-check", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_argparser()
    args = parser.parse_args(argv)
    selected = set(args.items)
    if "all" in selected:
        selected = {"benchmark", "scgen", "genept", "protein", "scgpt", "biolord"}

    if "benchmark" in selected:
        _download_benchmark(args)
    if "genept" in selected:
        _download_genept(args)
    if "protein" in selected:
        _download_assets(args, ["protein"])
    if "scgen" in selected:
        _download_scgen(args)
    if "scgpt" in selected:
        _download_assets(args, ["scgpt"])
    if "biolord" in selected:
        _prepare_biolord(args)

    _print_manual_notes(selected)
    if args.check:
        _check_inputs(args)


if __name__ == "__main__":
    main()
