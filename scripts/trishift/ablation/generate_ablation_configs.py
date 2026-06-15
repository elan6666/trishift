from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from scripts.common.yaml_utils import deep_update, dump_yaml, load_yaml_file, merged_dict


DATASET_CONFIGS = {
    "adamson": REPO_ROOT / "scripts" / "trishift" / "adamson" / "config.yaml",
    "dixit": REPO_ROOT / "scripts" / "trishift" / "dixit" / "config.yaml",
    "norman": REPO_ROOT / "scripts" / "trishift" / "norman" / "config.yaml",
}


ABLATION_PRESETS: dict[str, dict[str, Any]] = {
    "ref_random": {
        "group": "reference_construction",
        "label": "Random top-k reference",
        "overrides": {
            "matching_mode": "random",
            "ablation": {"topk_strategy": "random", "sample_soft_ctrl": False},
        },
    },
    "ref_knn": {
        "group": "reference_construction",
        "label": "kNN reference",
        "overrides": {
            "matching_mode": "knn",
            "ablation": {"topk_strategy": "random", "sample_soft_ctrl": False},
        },
    },
    "ref_ot": {
        "group": "reference_construction",
        "label": "OT reference",
        "overrides": {
            "matching_mode": "ot",
            "ablation": {"topk_strategy": "random", "sample_soft_ctrl": False},
        },
    },
    "cond_no_reference": {
        "group": "conditioning_input",
        "label": "No reference state",
        "overrides": {
            "model": {
                "stage2": {"zero_reference_state": True},
                "stage3": {"input_mode": "fusion_only"},
            },
        },
    },
    "cond_no_prior": {
        "group": "conditioning_input",
        "label": "No perturbation prior",
        "overrides": {
            "model": {"stage2": {"cond_l2_norm": False, "zero_condition_prior": True}},
        },
    },
    "cond_full": {
        "group": "conditioning_input",
        "label": "Full TriShift",
        "overrides": {},
    },
    "fusion_mlp": {
        "group": "shift_fusion",
        "label": "MLP fusion",
        "overrides": {
            "model": {
                "stage2": {
                    "use_cross_attention": False,
                    "use_transformer_block": False,
                    "transformer_layers": 0,
                }
            }
        },
    },
    "fusion_transformer": {
        "group": "shift_fusion",
        "label": "Transformer fusion",
        "overrides": {
            "model": {
                "stage2": {
                    "use_cross_attention": False,
                    "use_transformer_block": True,
                    "transformer_readout": "concat",
                    "transformer_layers": 1,
                }
            }
        },
    },
    "fusion_full": {
        "group": "shift_fusion",
        "label": "Full TriShift",
        "overrides": {},
    },
    "loss_mse": {
        "group": "expression_objective",
        "label": "L_mse",
        "overrides": {
            "loss": {
                "gamma": 0,
                "lambda_dir_expr": 0,
                "lambda_expr_mse": 1,
                "lambda_neg_expr": 0,
            },
            "ablation": {"latent_loss_type": "mse"},
        },
    },
    "loss_gears": {
        "group": "expression_objective",
        "label": "L_gears",
        "overrides": {
            "loss": {
                "gamma": 0,
                "lambda_dir_expr": 0.05,
                "lambda_expr_mse": 0,
                "lambda_neg_expr": 0,
            },
            "ablation": {"latent_loss_type": "gears"},
        },
    },
    "loss_gears_neg": {
        "group": "expression_objective",
        "label": "L_gears + L_neg",
        "overrides": {
            "loss": {
                "gamma": 0,
                "lambda_dir_expr": 0.05,
                "lambda_expr_mse": 0,
                "lambda_neg_expr": 0.5,
                "neg_expr_penalty": "mae",
            },
            "ablation": {"latent_loss_type": "gears"},
        },
    },
}


def _load_dataset_profile(dataset: str) -> dict:
    profile = load_yaml_file(DATASET_CONFIGS[dataset])
    if str(profile.get("dataset", "")).strip().lower() != dataset:
        raise ValueError(f"Dataset profile mismatch for {dataset}")
    return profile


def _preset_config(dataset: str, preset_name: str, split_ids: list[int] | None) -> dict:
    profile = _load_dataset_profile(dataset)
    preset = ABLATION_PRESETS[preset_name]
    out = {
        "dataset": profile["dataset"],
        "base": profile.get("base", {}),
        "defaults_overrides": merged_dict(profile.get("defaults_overrides") or {}, preset["overrides"]),
        "paths_overrides": profile.get("paths_overrides") or {},
    }
    if split_ids:
        deep_update(out["defaults_overrides"], {"run": {"split_ids": [int(x) for x in split_ids]}})
    return out


def generate(dataset: str, out_root: Path, split_ids: list[int] | None) -> dict:
    config_dir = out_root / dataset / "configs"
    command_rows: list[dict[str, Any]] = []
    config_dir.mkdir(parents=True, exist_ok=True)
    for preset_name, preset in ABLATION_PRESETS.items():
        cfg = _preset_config(dataset, preset_name, split_ids)
        cfg_path = config_dir / f"{dataset}_{preset_name}.yaml"
        dump_yaml(cfg_path, cfg, allow_unicode=False)
        meta_path = config_dir / f"{dataset}_{preset_name}.meta.json"
        meta_path.write_text(
            json.dumps(
                {
                    "dataset": dataset,
                    "preset": preset_name,
                    "group": preset["group"],
                    "label": preset["label"],
                    "overrides": preset["overrides"],
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        out_dir = Path("artifacts") / "results" / "ablation" / dataset / preset_name
        command_rows.append(
            {
                "dataset": dataset,
                "preset": preset_name,
                "group": preset["group"],
                "label": preset["label"],
                "requires_code_flag": bool(preset.get("requires_code_flag", False)),
                "config_path": str(cfg_path),
                "meta_path": str(meta_path),
                "out_dir": str(out_dir),
                "command": (
                    "python scripts/trishift/train/main.py "
                    f"--config {cfg_path} --unseen_ctrl_eval --out_dir {out_dir}"
                ),
            }
        )
    command_path = out_root / dataset / "ablation_commands.json"
    csv_path = out_root / dataset / "ablation_commands.csv"
    command_path.write_text(json.dumps(command_rows, indent=2), encoding="utf-8")
    import pandas as pd

    pd.DataFrame(command_rows).to_csv(csv_path, index=False)
    return {
        "dataset": dataset,
        "config_dir": str(config_dir),
        "commands_json": str(command_path),
        "commands_csv": str(csv_path),
        "n_presets": len(command_rows),
        "requires_code_flag": [
            row["preset"] for row in command_rows if row["requires_code_flag"]
        ],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Generate TriShift ablation preset configs and server commands.")
    ap.add_argument("--dataset", choices=sorted(DATASET_CONFIGS), action="append", required=True)
    ap.add_argument("--out-root", default="artifacts/results/ablation/plan")
    ap.add_argument("--split-id", action="append", type=int, default=[])
    args = ap.parse_args(argv)
    out = []
    for dataset in args.dataset:
        out.append(generate(dataset, Path(args.out_root), args.split_id or None))
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
