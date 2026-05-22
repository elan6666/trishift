from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
DATASETS = ("adamson", "dixit", "norman")


@dataclass(frozen=True)
class CheckItem:
    scope: str
    path: Path
    required: bool
    note: str


def _repo_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def _load_paths_yaml() -> dict:
    path = ROOT / "configs" / "paths.yaml"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _dataset_items(paths_cfg: dict) -> list[CheckItem]:
    datasets_cfg = paths_cfg.get("datasets", {})
    items: list[CheckItem] = []
    for dataset in DATASETS:
        h5ad = datasets_cfg.get(dataset, f"src/data/{dataset}/perturb_processed.h5ad")
        items.append(
            CheckItem(
                "benchmark-data",
                _repo_path(h5ad),
                True,
                f"{dataset} processed h5ad; create with scripts/data/download_and_prepare_benchmark_data.py",
            )
        )
        items.append(
            CheckItem(
                "gears-data",
                ROOT / "src" / "data" / "Data_GEARS" / dataset / "go.csv",
                True,
                f"{dataset} GEARS GO graph/cache; created by the benchmark data script",
            )
        )
    return items


def _embedding_items(paths_cfg: dict) -> list[CheckItem]:
    emb_cfg = paths_cfg.get("embeddings", {})
    keys = ("emb_a", "emb_b", "emb_c", "emb_d")
    defaults = {
        "emb_a": "src/data/Data_GeneEmbd/ensem_emb_gpt3.5all_new.pickle",
        "emb_b": "src/data/Data_GeneEmbd/GenePT_gene_embedding_ada_text.pickle",
        "emb_c": "src/data/Data_GeneEmbd/GPT_3_5_gene_embeddings.pickle",
        "emb_d": "src/data/Data_GeneEmbd/GenePT_gene_protein_embedding_model_3_text.pickle",
    }
    return [
        CheckItem(
            "gene-embeddings",
            _repo_path(emb_cfg.get(key, defaults[key])),
            key in {"emb_b"},
            f"{key}; required if selected by a config",
        )
        for key in keys
    ]


def _biolord_items() -> list[CheckItem]:
    items: list[CheckItem] = []
    for dataset in DATASETS:
        items.extend(
            [
                CheckItem(
                    "biolord-data",
                    ROOT / "src" / "data" / dataset / f"{dataset}_biolord.h5ad",
                    True,
                    f"{dataset} full BioLORD input; create with scripts/data/prepare_biolord_perturbation_data.py",
                ),
                CheckItem(
                    "biolord-data",
                    ROOT / "src" / "data" / dataset / f"{dataset}_single_biolord.h5ad",
                    True,
                    f"{dataset} condition-mean BioLORD input",
                ),
            ]
        )
    return items


def _external_items() -> list[CheckItem]:
    return [
        CheckItem("external", ROOT / "external" / "scGPT-main", False, "scGPT source tree"),
        CheckItem("external", ROOT / "external" / "gears", False, "GEARS source tree"),
        CheckItem("external", ROOT / "external" / "GenePert-main", False, "GenePert source tree"),
        CheckItem("external", ROOT / "external" / "scPRAM-main", False, "scPRAM source tree"),
        CheckItem("external", ROOT / "external" / "biolord" / "biolord-main", False, "BioLORD source tree"),
    ]


def _scgen_items() -> list[CheckItem]:
    return [
        CheckItem(
            "scgen",
            ROOT / "src" / "data" / "scgen" / "train_kang_scgen.h5ad",
            True,
            "download with scripts/data/download_scgen_pbmc.py",
        ),
        CheckItem(
            "scgen",
            ROOT / "src" / "data" / "scgen" / "perturb_processed.h5ad",
            True,
            "create with scripts/data/prepare_scgen_pbmc.py",
        ),
        CheckItem(
            "scgen",
            ROOT / "src" / "data" / "scgen" / "priors" / "scgen_ifnb1_genept.pickle",
            False,
            "IFNB1 GenePT prior; create with scripts/data/prepare_scgen_pbmc.py",
        ),
    ]


def _protein_items() -> list[CheckItem]:
    base = ROOT / "src" / "data" / "protein_embeddings"
    return [
        CheckItem("protein", base / "uniprot_prott5_human_per_protein.h5", False, "UniProt ProtT5 human proteome"),
        CheckItem("protein", base / "zenodo_prott5_human_reduced_embeddings_file.h5", False, "Zenodo ProtT5 human proteome"),
        CheckItem("protein", base / "hf_esm2_15b_human_mouse_embeddings.npy", False, "HF ESM2-15B embeddings"),
        CheckItem("protein", base / "hf_esm2_15b_human_mouse_metadata.csv.gz", False, "HF ESM2-15B metadata"),
    ]


def _scgpt_items(paths_cfg: dict) -> list[CheckItem]:
    root = _repo_path(paths_cfg.get("scgpt_pretrained_root", "artifacts/models/scGPT_human"))
    return [
        CheckItem("scgpt", root / "args.json", True, "scGPT pretrained args"),
        CheckItem("scgpt", root / "best_model.pt", True, "scGPT pretrained weights"),
        CheckItem("scgpt", root / "vocab.json", True, "scGPT gene vocabulary"),
    ]


def _collect_items() -> list[CheckItem]:
    paths_cfg = _load_paths_yaml()
    return (
        _dataset_items(paths_cfg)
        + _embedding_items(paths_cfg)
        + _biolord_items()
        + _external_items()
        + _scgen_items()
        + _protein_items()
        + _scgpt_items(paths_cfg)
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check whether local reproduction inputs are present.")
    parser.add_argument(
        "--scope",
        choices=["all", "benchmark", "baselines", "scgen", "scgpt"],
        default="all",
        help="Limit checks to a reproduction scope.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero if any selected missing item is marked required.",
    )
    args = parser.parse_args(argv)

    scope_map = {
        "benchmark": {"benchmark-data", "gears-data", "gene-embeddings"},
        "baselines": {"benchmark-data", "gears-data", "gene-embeddings", "biolord-data", "external", "scgpt"},
        "scgen": {"scgen", "protein", "gene-embeddings"},
        "scgpt": {"scgpt", "external"},
    }
    allowed = None if args.scope == "all" else scope_map[args.scope]
    selected = [item for item in _collect_items() if allowed is None or item.scope in allowed]

    missing_required = 0
    print(f"{'status':<8} {'required':<8} {'scope':<16} path")
    print("-" * 100)
    for item in selected:
        exists = item.path.exists()
        status = "OK" if exists else "MISSING"
        required = "yes" if item.required else "no"
        if item.required and not exists:
            missing_required += 1
        print(f"{status:<8} {required:<8} {item.scope:<16} {item.path.relative_to(ROOT) if item.path.is_relative_to(ROOT) else item.path}")
        if not exists:
            print(f"{'':<34} note: {item.note}")

    if missing_required:
        print(f"\n[summary] missing required inputs: {missing_required}")
    else:
        print("\n[summary] required inputs present for selected scope")
    return 1 if bool(args.strict) and missing_required else 0


if __name__ == "__main__":
    raise SystemExit(main())
