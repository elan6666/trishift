from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import pickle
import sys

import anndata as ad
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = ROOT / "src"
LOCAL_DATA_ROOT = ROOT / "src" / "data"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SRC_ROOT))

from trishift import _utils
from trishift._external_metrics import (
    compute_mean_effect_metrics,
    pearson_delta_reference_metrics,
)
from trishift.TriShiftData import TriShiftData
from scripts.common.split_utils import (
    condition_sort,
    split_unseen_ctrl_unseen_perturbation,
)


@dataclass(frozen=True)
class BioLordDatasetConfig:
    data_rel: str
    splits: list[int]
    test_ratio: float = 0.2


DATASET_CONFIG = {
    "adamson": BioLordDatasetConfig(
        data_rel="data/adamson/perturb_processed.h5ad",
        splits=[1, 2, 3, 4, 5],
    ),
    "dixit": BioLordDatasetConfig(
        data_rel="data/dixit/perturb_processed.h5ad",
        splits=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    ),
    "norman": BioLordDatasetConfig(
        data_rel="data/norman/perturb_processed.h5ad",
        splits=[1, 2, 3, 4, 5],
    ),
}


def _resolve_eval_data_path(name: str, cfg: BioLordDatasetConfig) -> Path:
    primary = (LOCAL_DATA_ROOT / Path(cfg.data_rel).relative_to("data")).resolve()
    if primary.exists():
        return primary
    paths_cfg_path = ROOT / "configs" / "paths.yaml"
    if paths_cfg_path.exists():
        paths_cfg = _utils.load_yaml(str(paths_cfg_path))
        datasets_map = paths_cfg.get("datasets", {})
        alt = Path(datasets_map[name]) if name in datasets_map else primary
        if alt.exists():
            return alt.resolve()
    raise FileNotFoundError(f"Missing BioLORD evaluation h5ad for dataset={name}: {primary}")


def _prepare_data(data_path: Path) -> TriShiftData:
    adata = _utils.load_adata(str(data_path))
    data = TriShiftData(adata, pd.DataFrame(index=["ctrl"]))
    data.build_or_load_degs()
    return data


def _write_mean_metrics(path: Path, metrics_df: pd.DataFrame) -> None:
    numeric_means = metrics_df.mean(numeric_only=True)
    mean_pearson = float(numeric_means.get("pearson", float("nan")))
    keys = [
        "pearson",
        "nmse",
        "mse_pred",
        "mse_ctrl",
        "deg_mean_r2",
        "systema_corr_20de_allpert",
        "systema_corr_deg_r2",
    ]
    lines = [f"{mean_pearson}\n"]
    for key in keys:
        if key in numeric_means.index:
            lines.append(f"mean_{key}={float(numeric_means[key])}\n")
    if "subgroup" in metrics_df.columns and not metrics_df["subgroup"].isna().all():
        ordered = ["single", "seen0", "seen1", "seen2"]
        present = set(metrics_df["subgroup"].dropna().astype(str).tolist())
        lines.append("# subgroup_means_row_weighted\n")
        lines.append("subgroup_order=single,seen0,seen1,seen2\n\n")
        for subgroup in [x for x in ordered if x in present]:
            sub_df = metrics_df[metrics_df["subgroup"].astype(str) == subgroup]
            lines.append(f"subgroup_{subgroup}_n_rows={int(len(sub_df))}\n")
            sub_means = sub_df.mean(numeric_only=True)
            for key in keys:
                if key in sub_means.index:
                    lines.append(f"subgroup_{subgroup}_mean_{key}={float(sub_means[key])}\n")
            lines.append("\n")
    path.write_text("".join(lines), encoding="utf-8")


def _subgroup_map_from_df(subgroup_df: pd.DataFrame | None) -> dict[str, str]:
    if subgroup_df is None or "subgroup" not in subgroup_df.columns:
        return {}
    return {
        condition_sort(str(idx)): str(row["subgroup"])
        for idx, row in subgroup_df.iterrows()
        if condition_sort(str(idx)) != "ctrl"
    }


def _payload_by_condition(payload: dict) -> dict[str, dict]:
    return {condition_sort(str(k)): v for k, v in payload.items() if isinstance(v, dict)}


def _top20_degs_by_condition(adata: ad.AnnData) -> dict[str, np.ndarray]:
    top20 = adata.uns.get("top20_degs_final", {})
    if not isinstance(top20, dict):
        top20 = adata.uns.get("top20_degs_non_dropout", {})
    if not isinstance(top20, dict):
        raise TypeError("expected top20 DEG mapping in adata.uns")
    return {
        condition_sort(str(cond)): np.asarray(degs, dtype=int).reshape(-1)
        for cond, degs in top20.items()
    }


def _gene_names(adata: ad.AnnData) -> np.ndarray:
    if "gene_name" in adata.var.columns:
        return adata.var["gene_name"].astype(str).values
    return adata.var_names.astype(str).values


def _nan_scpram_metrics() -> dict[str, float]:
    return {
        "scpram_r2_degs_mean_mean": np.nan,
        "scpram_r2_degs_var_mean": np.nan,
        "scpram_wasserstein_degs_sum": np.nan,
    }


def _evaluate_payload_split(
    *,
    dataset_name: str,
    split_id: int,
    payload: dict,
    split_dict: dict,
    subgroup_map: dict[str, str],
) -> tuple[pd.DataFrame, dict]:
    test_adata = split_dict["test"]
    cond_arr = test_adata.obs["condition"].astype(str).map(condition_sort).values
    ctrl = _utils.densify_X(test_adata[test_adata.obs["condition"].astype(str) == "ctrl"].X)
    if ctrl.shape[0] == 0:
        raise ValueError("unseen BioLORD evaluation found no test ctrl cells")
    reference = ctrl.mean(axis=0).astype(np.float32)
    payload_map = _payload_by_condition(payload)
    top20 = _top20_degs_by_condition(test_adata)
    gene_names = _gene_names(test_adata)
    results: list[dict] = []
    export_payload: dict[str, dict] = {}

    for cond in sorted({condition_sort(str(c)) for c in split_dict.get("test_conds", [])}):
        if cond not in payload_map:
            print(f"[biolord] skip condition missing from source payload: {cond}")
            continue
        cond_mask = cond_arr == cond
        if not np.any(cond_mask):
            print(f"[biolord] skip condition missing from test split: {cond}")
            continue
        true = _utils.densify_X(test_adata[cond_mask].X)
        pred = np.asarray(payload_map[cond].get("Pred_full"), dtype=np.float32)
        if pred.ndim == 1:
            pred = pred.reshape(1, -1)
        if pred.ndim != 2 or pred.shape[1] != true.shape[1]:
            raise ValueError(
                f"BioLORD payload shape mismatch for {cond}: pred={pred.shape}, true={true.shape}"
            )
        degs = np.asarray(top20.get(cond, []), dtype=int).reshape(-1)
        if degs.size == 0:
            print(f"[biolord] skip condition without DEGs: {cond}")
            continue
        mean_metrics = compute_mean_effect_metrics(
            X_true=true,
            X_pred=pred,
            X_ctrl=ctrl,
            deg_idx=degs,
        )
        systema_metrics = pearson_delta_reference_metrics(
            X_true=true.mean(axis=0),
            X_pred=pred.mean(axis=0),
            reference=reference,
            top20_de_idxs=degs,
        )
        row = {
            "condition": cond,
            **mean_metrics,
            "systema_corr_20de_allpert": float(systema_metrics["corr_20de_allpert"]),
            "systema_corr_deg_r2": float(systema_metrics["corr_deg_r2"]),
            **_nan_scpram_metrics(),
            "split_id": int(split_id),
            "n_ensemble": int(pred.shape[0]),
            "n_eval_ctrl": int(ctrl.shape[0]),
            "eval_ctrl_source": "target_domain_test_ctrl",
        }
        if subgroup_map:
            row["subgroup"] = subgroup_map.get(cond, "unknown")
        results.append(row)
        export_payload[cond] = {
            "Pred": pred[:, degs] if degs.size > 0 else pred[:, :0],
            "Ctrl": ctrl[:, degs] if degs.size > 0 else ctrl[:, :0],
            "Truth": true[:, degs] if degs.size > 0 else true[:, :0],
            "Pred_full": pred,
            "Ctrl_full": ctrl,
            "Truth_full": true,
            "DE_idx": degs,
            "DE_name": gene_names[degs] if degs.size > 0 else np.array([], dtype=gene_names.dtype),
            "gene_name_full": gene_names,
            "export_metadata": {
                "model": "biolord",
                "dataset": str(dataset_name),
                "split_id": int(split_id),
                "split_policy": split_dict.get("split_policy"),
                "eval_ctrl_source": "target_domain_test_ctrl",
                "source_payload_reused": True,
            },
        }
    return pd.DataFrame(results), export_payload


def run_biolord_unseen_ctrl_eval(
    name: str,
    *,
    source_dir: str | Path | None = None,
    split_ids: list[int] | tuple[int, ...] | None = None,
) -> None:
    if name not in DATASET_CONFIG:
        raise ValueError(f"Unknown BioLORD dataset: {name}")
    cfg = DATASET_CONFIG[name]
    splits_eff = [int(x) for x in (split_ids if split_ids is not None else cfg.splits)]
    out_dir = ROOT / "artifacts" / "results" / "biolord" / name
    source_path = Path(source_dir).resolve() if source_dir else out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    data = _prepare_data(_resolve_eval_data_path(name, cfg))

    metrics_all: list[pd.DataFrame] = []
    for split_id in splits_eff:
        source_pkl = source_path / f"biolord_{name}_{split_id}.pkl"
        if not source_pkl.exists():
            raise FileNotFoundError(
                f"BioLORD source payload not found: {source_pkl}. "
                "Generate the BioLORD notebook payload before recomputing unseen-control metrics."
            )
        with source_pkl.open("rb") as handle:
            source_payload = pickle.load(handle)
        split_dict, subgroup_df = split_unseen_ctrl_unseen_perturbation(
            data,
            name,
            seed=int(split_id),
            test_ratio=float(cfg.test_ratio),
        )
        metrics_df, export_payload = _evaluate_payload_split(
            dataset_name=name,
            split_id=int(split_id),
            payload=source_payload,
            split_dict=split_dict,
            subgroup_map=_subgroup_map_from_df(subgroup_df),
        )
        metrics_all.append(metrics_df)
        out_pkl = out_dir / f"biolord_{name}_{split_id}_unseen_ctrl.pkl"
        with out_pkl.open("wb") as handle:
            pickle.dump(export_payload, handle)
        print(f"[biolord] saved notebook payload: {out_pkl}")

    metrics_df_all = pd.concat(metrics_all, ignore_index=True)
    metrics_path = out_dir / "metrics_unseen_ctrl.csv"
    mean_path = out_dir / "mean_pearson_unseen_ctrl.txt"
    metrics_df_all.to_csv(metrics_path, index=False)
    _write_mean_metrics(mean_path, metrics_df_all)
    print(f"[biolord] saved metrics: {metrics_path}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Recompute BioLORD held-out ctrl metrics from saved notebook payloads"
    )
    parser.add_argument("--name", required=True, choices=sorted(DATASET_CONFIG))
    parser.add_argument("--source_dir", default="", help="directory containing biolord_<dataset>_<split>.pkl")
    parser.add_argument("--splits", default="", help="comma-separated split ids; default uses dataset config")
    args = parser.parse_args(argv)

    split_ids = None
    if str(args.splits).strip():
        split_ids = [int(x.strip()) for x in str(args.splits).split(",") if x.strip()]
    run_biolord_unseen_ctrl_eval(
        str(args.name),
        source_dir=str(args.source_dir).strip() or None,
        split_ids=split_ids,
    )


if __name__ == "__main__":
    main()
