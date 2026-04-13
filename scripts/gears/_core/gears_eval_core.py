from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import importlib
import os
import pickle
import random
import sys

import anndata as ad
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error as mse

ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = ROOT / "src"
LOCAL_DATA_ROOT = ROOT / "src" / "data"
LOCAL_GEARS_DATA_ROOT = LOCAL_DATA_ROOT / "Data_GEARS"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SRC_ROOT))

from trishift import _utils
from trishift._external_metrics import (
    average_of_perturbation_centroids,
    compute_scpram_metrics_from_arrays,
    pearson_delta_reference_metrics,
    regression_r2_safe,
)
from trishift.TriShiftData import TriShiftData
from scripts.common.split_utils import (
    condition_sort as _shared_condition_sort,
    norman_subgroup as _shared_norman_subgroup,
)
from scripts.common.yaml_utils import load_yaml_file


@dataclass(frozen=True)
class GearsDatasetConfig:
    gears_data_name: str
    eval_data_rel: str
    splits: list[int]
    project_name: str
    hidden_size: int = 64
    epochs: int = 15
    batch_size: int = 32
    test_batch_size: int = 32
    train_gene_set_size: float | None = None
    norman_split: bool = False
    test_ratio: float = 0.2


@dataclass(frozen=True)
class GearsSplitMeta:
    train_conds: list[str]
    val_conds: list[str]
    test_conds: list[str]
    train_conds_raw: list[str]
    val_conds_raw: list[str]
    test_conds_raw: list[str]
    subgroup_map: dict[str, str]


PROFILE_DIR = Path(__file__).resolve().parents[1] / "eval" / "configs"
PROFILE_ALLOWED_KEYS = {"dataset", "task_args"}


DATASET_CONFIG = {
    "adamson": GearsDatasetConfig(
        gears_data_name="adamson",
        eval_data_rel="data/Data_GEARS/adamson/perturb_processed.h5ad",
        splits=[1, 2, 3, 4, 5],
        project_name="adamson",
    ),
    "dixit": GearsDatasetConfig(
        gears_data_name="dixit",
        eval_data_rel="data/Data_GEARS/dixit/perturb_processed.h5ad",
        splits=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        project_name="dixit",
        train_gene_set_size=0.9,
        test_ratio=0.1,
    ),
    "norman": GearsDatasetConfig(
        gears_data_name="norman",
        eval_data_rel="data/Data_GEARS/norman/perturb_processed.h5ad",
        splits=[1, 2, 3, 4, 5],
        project_name="norman",
        norman_split=True,
        test_ratio=0.2,
    ),
    "replogle_k562_essential": GearsDatasetConfig(
        gears_data_name="replogle_k562_essential",
        eval_data_rel="data/Data_GEARS/replogle_k562_essential/perturb_processed.h5ad",
        splits=[1, 2, 3, 4, 5],
        project_name="rep_k562",
        test_ratio=0.2,
    ),
    "replogle_rpe1_essential": GearsDatasetConfig(
        gears_data_name="replogle_rpe1_essential",
        eval_data_rel="data/Data_GEARS/replogle_rpe1_essential/perturb_processed.h5ad",
        splits=[1, 2, 3, 4, 5],
        project_name="rpe1",
        test_ratio=0.2,
    ),
}


def _load_profile(profile: str) -> dict:
    p = PROFILE_DIR / f"{profile}.yaml"
    if not p.exists():
        raise FileNotFoundError(f"GEARS profile not found: {p}")
    obj = load_yaml_file(p)
    unknown = sorted(set(obj.keys()) - PROFILE_ALLOWED_KEYS)
    if unknown:
        raise ValueError(f"Unknown keys in GEARS profile {p.name}: {unknown}")
    dataset = str(obj.get("dataset", "")).strip()
    if dataset != profile:
        raise ValueError(f"Profile dataset mismatch: file={p.name} dataset={dataset!r}")
    task_args = obj.get("task_args") or {}
    if not isinstance(task_args, dict):
        raise TypeError(f"profile.task_args must be a mapping: {p}")
    return {"dataset": dataset, "task_args": task_args}


def run_profile(profile: str) -> None:
    prof = _load_profile(profile)
    task_args = prof["task_args"]
    run_gears_eval(
        prof["dataset"],
        base_seed=int(task_args.get("seed", 24)),
        export_notebook_pkl=bool(task_args.get("export_notebook_pkl", True)),
    )


def _resolve_mean_metric_keys(numeric_means: pd.Series) -> list[str]:
    preferred_order = [
        "pearson",
        "nmse",
        "mse_pred",
        "mse_ctrl",
        "deg_mean_r2",
        "systema_corr_20de_allpert",
        "systema_corr_deg_r2",
        "scpram_r2_degs_mean_mean",
        "scpram_r2_degs_var_mean",
        "scpram_wasserstein_degs_sum",
    ]
    exclude_keys = {"split_id", "n_ensemble"}
    keys = [k for k in preferred_order if k in numeric_means.index and k not in exclude_keys]
    keys.extend([k for k in numeric_means.index if k not in exclude_keys and k not in keys])
    return keys


def _append_subgroup_mean_lines(lines: list[str], metrics_df: pd.DataFrame, keys: list[str]) -> None:
    if "subgroup" not in metrics_df.columns:
        return
    subgroup_series = metrics_df["subgroup"]
    if subgroup_series.isna().all():
        return

    ordered = ["single", "seen0", "seen1", "seen2"]
    present = set(subgroup_series.dropna().astype(str).tolist())
    subgroup_names = [g for g in ordered if g in present]
    if "unknown" in present:
        subgroup_names.append("unknown")
    if not subgroup_names:
        return

    lines.append("# subgroup_means_row_weighted\n")
    lines.append("subgroup_order=single,seen0,seen1,seen2\n")
    lines.append("\n")

    for g in subgroup_names:
        sub_df = metrics_df[metrics_df["subgroup"].astype(str) == g]
        lines.append(f"subgroup_{g}_n_rows={int(len(sub_df))}\n")
        sub_numeric_means = sub_df.mean(numeric_only=True)
        for key in keys:
            if key not in sub_numeric_means.index:
                continue
            lines.append(f"subgroup_{g}_mean_{key}={float(sub_numeric_means[key])}\n")
        lines.append("\n")


def _write_mean_metrics(path: Path, metrics_df: pd.DataFrame) -> None:
    numeric_means = metrics_df.mean(numeric_only=True)
    mean_pearson = float(numeric_means.get("pearson", float("nan")))
    keys = _resolve_mean_metric_keys(numeric_means)

    lines = [f"{mean_pearson}\n"]
    for key in keys:
        lines.append(f"mean_{key}={float(numeric_means[key])}\n")
    _append_subgroup_mean_lines(lines, metrics_df, keys)
    path.write_text("".join(lines), encoding="utf-8")


def _attach_subgroup_column(metrics_df: pd.DataFrame, subgroup_df: pd.DataFrame | None) -> pd.DataFrame:
    if subgroup_df is None or "condition" not in metrics_df.columns:
        return metrics_df
    if "subgroup" not in subgroup_df.columns:
        return metrics_df

    out = metrics_df.copy()
    if "subgroup" in out.columns:
        out = out.drop(columns=["subgroup"])
    subgroup_map = subgroup_df["subgroup"].astype(str).to_dict()
    subgroup_vals = out["condition"].astype(str).map(subgroup_map).fillna("unknown")
    insert_at = list(out.columns).index("condition") + 1
    out.insert(insert_at, "subgroup", subgroup_vals)
    return out


def set_seeds(seed: int = 24) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def condition_sort(x: str) -> str:
    return _shared_condition_sort(x)


def subgroup(pert_list: list[str], seed: int) -> pd.DataFrame:
    return _shared_norman_subgroup(pert_list=pert_list, seed=seed)


def _split_conditions_random(conditions: list[str], *, ratio: float, seed: int) -> tuple[list[str], list[str]]:
    work = np.asarray(list(conditions), dtype=object)
    if work.size == 0:
        return [], []
    rng = np.random.RandomState(int(seed))
    rng.shuffle(work)
    n_pick = int(round(float(ratio) * int(work.size)))
    if float(ratio) > 0.0 and int(work.size) > 0 and n_pick == 0:
        n_pick = 1
    if n_pick >= int(work.size):
        n_pick = max(1, int(work.size) - 1)
    picked_arr, remain_arr = np.split(work, [n_pick])
    return [str(x) for x in picked_arr.tolist()], [str(x) for x in remain_arr.tolist()]


def _build_shared_split_meta(
    eval_adata: ad.AnnData,
    *,
    dataset_name: str,
    cfg: GearsDatasetConfig,
    split_id: int,
    available_raw_conditions: list[str] | None = None,
) -> GearsSplitMeta:
    eval_raw_conditions = [str(x) for x in eval_adata.obs["condition"].astype(str).unique().tolist()]
    raw_condition_pool = [str(x) for x in (available_raw_conditions or eval_raw_conditions)]
    raw_by_canonical: dict[str, list[str]] = {}
    for cond in raw_condition_pool:
        raw_by_canonical.setdefault(condition_sort(cond), []).append(str(cond))
    canonical_conditions = sorted({condition_sort(cond) for cond in eval_raw_conditions})
    non_ctrl = [cond for cond in canonical_conditions if cond != "ctrl"]

    if cfg.norman_split:
        split_df = subgroup(canonical_conditions, seed=int(split_id))
        split_df.index = [condition_sort(str(x)) for x in split_df.index.tolist()]
        train_conds = [str(x) for x in split_df[split_df["group"].astype(str) == "train"].index.astype(str).tolist() if str(x) != "ctrl"]
        val_conds = [str(x) for x in split_df[split_df["group"].astype(str) == "val"].index.astype(str).tolist() if str(x) != "ctrl"]
        test_conds = [str(x) for x in split_df[split_df["group"].astype(str) == "test"].index.astype(str).tolist() if str(x) != "ctrl"]
        subgroup_map = {
            condition_sort(str(idx)): str(row["subgroup"])
            for idx, row in split_df.iterrows()
            if condition_sort(str(idx)) != "ctrl"
        }
    else:
        test_conds, remain_conds = _split_conditions_random(non_ctrl, ratio=float(cfg.test_ratio), seed=int(split_id))
        val_conds, train_conds = _split_conditions_random(remain_conds, ratio=0.1, seed=int(split_id))
        subgroup_map = {}

    available_canonical = set(raw_by_canonical)
    train_conds = [cond for cond in train_conds if cond in available_canonical]
    val_conds = [cond for cond in val_conds if cond in available_canonical]
    test_conds = [cond for cond in test_conds if cond in available_canonical]

    def _flatten_raw(split_conds: list[str]) -> list[str]:
        out: list[str] = []
        for cond in split_conds:
            out.extend(raw_by_canonical.get(cond, []))
        return list(dict.fromkeys(out))

    return GearsSplitMeta(
        train_conds=train_conds,
        val_conds=val_conds,
        test_conds=test_conds,
        train_conds_raw=_flatten_raw(train_conds),
        val_conds_raw=_flatten_raw(val_conds),
        test_conds_raw=_flatten_raw(test_conds),
        subgroup_map={k: v for k, v in subgroup_map.items() if k in available_canonical},
    )


def _apply_shared_split_to_pert_data(
    pert_data,
    split_meta: GearsSplitMeta,
    *,
    split_id: int,
    train_gene_set_size: float | None,
) -> None:
    pert_data.split = "custom"
    pert_data.seed = int(split_id)
    pert_data.train_gene_set_size = float(train_gene_set_size) if train_gene_set_size is not None else 0.75
    pert_data.subgroup = None
    pert_data.set2conditions = {
        "train": list(split_meta.train_conds_raw) + ["ctrl"],
        "val": list(split_meta.val_conds_raw),
        "test": list(split_meta.test_conds_raw),
    }


def _upgrade_legacy_dataset_processed(pert_data, split_meta: GearsSplitMeta) -> None:
    dataset_processed = getattr(pert_data, "dataset_processed", None)
    if not isinstance(dataset_processed, dict):
        return
    gene_names_src = getattr(pert_data, "gene_names", None)
    if gene_names_src is None or len(gene_names_src) == 0:
        adata = getattr(pert_data, "adata", None)
        if adata is not None:
            if "gene_name" in adata.var.columns:
                gene_names_src = adata.var["gene_name"].astype(str).tolist()
            else:
                gene_names_src = adata.var_names.astype(str).tolist()
    gene_names = np.asarray(gene_names_src or [], dtype=str)
    if gene_names.size == 0:
        return
    gene_to_idx = {str(g): int(i) for i, g in enumerate(gene_names.tolist())}
    needed_raw = set(split_meta.train_conds_raw) | set(split_meta.val_conds_raw) | set(split_meta.test_conds_raw) | {"ctrl"}
    for cond in needed_raw:
        graphs = dataset_processed.get(cond)
        if not graphs:
            continue
        sample_x = getattr(graphs[0], "x", None)
        if sample_x is None or int(sample_x.ndim) != 2 or int(sample_x.shape[1]) != 1:
            continue
        pert_vec = torch.zeros((int(sample_x.shape[0]), 1), dtype=sample_x.dtype)
        for gene in str(cond).split("+"):
            gene = str(gene)
            if gene == "ctrl":
                continue
            idx = gene_to_idx.get(gene)
            if idx is not None and 0 <= idx < int(sample_x.shape[0]):
                pert_vec[idx, 0] = 1
        for graph in graphs:
            graph.x = torch.cat([graph.x, pert_vec.to(graph.x.device)], dim=1)


def _require_gears_classes():
    try:
        old_sys_path = list(sys.path)
        try:
            sys.path = [
                p
                for p in sys.path
                if Path(p).resolve()
                not in {
                    ROOT.resolve(),
                    (ROOT / "scripts").resolve(),
                }
            ]
            gears_mod = importlib.import_module("gears")
            gears_utils_mod = importlib.import_module("gears.utils")
            gears_gears_mod = importlib.import_module("gears.gears")
        finally:
            sys.path = old_sys_path
        def _safe_uncertainty_loss_fct(pred, logvar, y, perts, reg=0.1, ctrl=None, direction_lambda=1e-3, dict_filter=None):
            gamma = 2
            perts_arr = np.asarray(perts)
            unique_perts = list(dict.fromkeys(perts_arr.tolist()))
            losses = pred.new_tensor(0.0)
            for pert in unique_perts:
                idx = np.where(perts_arr == pert)[0]
                if pert != "ctrl":
                    retain_idx = dict_filter[pert]
                    pred_p = pred[idx][:, retain_idx]
                    y_p = y[idx][:, retain_idx]
                    logvar_p = logvar[idx][:, retain_idx]
                else:
                    pred_p = pred[idx]
                    y_p = y[idx]
                    logvar_p = logvar[idx]
                losses = losses + torch.sum(
                    (pred_p - y_p) ** (2 + gamma)
                    + reg * torch.exp(-logvar_p) * (pred_p - y_p) ** (2 + gamma)
                ) / pred_p.shape[0] / pred_p.shape[1]
                if pert != "ctrl":
                    losses = losses + torch.sum(
                        direction_lambda
                        * (torch.sign(y_p - ctrl[retain_idx]) - torch.sign(pred_p - ctrl[retain_idx])) ** 2
                    ) / pred_p.shape[0] / pred_p.shape[1]
                else:
                    losses = losses + torch.sum(
                        direction_lambda * (torch.sign(y_p - ctrl) - torch.sign(pred_p - ctrl)) ** 2
                    ) / pred_p.shape[0] / pred_p.shape[1]
            return losses / max(1, len(unique_perts))

        def _safe_loss_fct(pred, y, perts, ctrl=None, direction_lambda=1e-3, dict_filter=None):
            gamma = 2
            perts_arr = np.asarray(perts)
            unique_perts = list(dict.fromkeys(perts_arr.tolist()))
            losses = pred.new_tensor(0.0)
            for pert in unique_perts:
                idx = np.where(perts_arr == pert)[0]
                if pert != "ctrl":
                    retain_idx = dict_filter[pert]
                    pred_p = pred[idx][:, retain_idx]
                    y_p = y[idx][:, retain_idx]
                else:
                    pred_p = pred[idx]
                    y_p = y[idx]
                losses = losses + torch.sum((pred_p - y_p) ** (2 + gamma)) / pred_p.shape[0] / pred_p.shape[1]
                if pert != "ctrl":
                    losses = losses + torch.sum(
                        direction_lambda
                        * (torch.sign(y_p - ctrl[retain_idx]) - torch.sign(pred_p - ctrl[retain_idx])) ** 2
                    ) / pred_p.shape[0] / pred_p.shape[1]
                else:
                    losses = losses + torch.sum(
                        direction_lambda * (torch.sign(y_p - ctrl) - torch.sign(pred_p - ctrl)) ** 2
                    ) / pred_p.shape[0] / pred_p.shape[1]
            return losses / max(1, len(unique_perts))

        def _safe_gears_init(self, pert_data, device="cuda", weight_bias_track=False, proj_name="GEARS", exp_name="GEARS"):
            self.weight_bias_track = weight_bias_track
            if self.weight_bias_track:
                import wandb

                wandb.init(project=proj_name, name=exp_name)
                self.wandb = wandb
            else:
                self.wandb = None

            self.device = device
            self.config = None

            self.dataloader = pert_data.dataloader
            self.adata = pert_data.adata
            self.node_map = pert_data.node_map
            self.data_path = pert_data.data_path
            self.dataset_name = pert_data.dataset_name
            self.split = pert_data.split
            self.seed = pert_data.seed
            self.train_gene_set_size = pert_data.train_gene_set_size
            self.set2conditions = pert_data.set2conditions
            self.subgroup = pert_data.subgroup
            self.gene_list = pert_data.gene_names.values.tolist()
            self.num_genes = len(self.gene_list)

            cond_series = self.adata.obs["condition"].astype(str)
            ctrl_mask = (cond_series.to_numpy() == "ctrl")
            self.ctrl_expression = torch.tensor(np.mean(self.adata.X[ctrl_mask], axis=0)).reshape(-1).to(self.device)
            pert_full_id2pert = dict(self.adata.obs[["condition_name", "condition"]].values)
            self.dict_filter = {pert_full_id2pert[i]: j for i, j in self.adata.uns["non_zeros_gene_idx"].items()}
            self.ctrl_adata = self.adata[ctrl_mask]

        gears_utils_mod.loss_fct = _safe_loss_fct
        gears_utils_mod.uncertainty_loss_fct = _safe_uncertainty_loss_fct
        gears_gears_mod.loss_fct = _safe_loss_fct
        gears_gears_mod.uncertainty_loss_fct = _safe_uncertainty_loss_fct
        GEARS = getattr(gears_mod, "GEARS")
        GEARS.__init__ = _safe_gears_init
        gears_gears_mod.GEARS.__init__ = _safe_gears_init
        PertData = getattr(gears_mod, "PertData")
    except ImportError as exc:
        raise ImportError(
            "GEARS is not installed. Install the GEARS dependency stack first "
            "(for example `pip install cell-gears torch-geometric`) and ensure "
            "the `gears` package is importable."
        ) from exc
    return PertData, GEARS


def _resolve_eval_data_path(name: str, cfg: GearsDatasetConfig) -> Path:
    data_path_primary = LOCAL_GEARS_DATA_ROOT / name / "perturb_processed.h5ad"
    if data_path_primary.exists():
        return data_path_primary

    paths_cfg_path = ROOT / "configs" / "paths.yaml"
    if paths_cfg_path.exists():
        paths_cfg = _utils.load_yaml(str(paths_cfg_path))
        datasets_map = paths_cfg.get("datasets", {})
        data_path_alt = Path(datasets_map[name]) if name in datasets_map else data_path_primary
        if data_path_alt.exists():
            return data_path_alt

    raise FileNotFoundError(
        "Missing evaluation h5ad for GEARS. "
        f"Tried primary data={data_path_primary}; fallback via {paths_cfg_path}. "
        "Provide files under src/data/Data_GEARS or configs/paths.yaml."
    )


def _resolve_gears_data_root(name: str) -> Path:
    candidates: list[Path] = []
    env_root = str(os.environ.get("TRISHIFT_GEARS_DATA_ROOT", "")).strip()
    if env_root:
        candidates.append(Path(env_root))
    candidates.append(LOCAL_GEARS_DATA_ROOT)

    for root in candidates:
        if (root / name).exists():
            return root

    tried = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        "Missing GEARS dataset directory. "
        f"Expected `<root>/{name}` under one of: {tried}. "
        "Generate the GEARS data first, for example via "
        "`src/data/Data_GEARS/generating_data.py`, "
        "or set TRISHIFT_GEARS_DATA_ROOT."
    )


def _prepare_eval_adata(data_path: Path) -> ad.AnnData:
    adata = _utils.load_adata(str(data_path))
    data = TriShiftData(adata, pd.DataFrame(index=["ctrl"]))
    data.build_or_load_degs()
    return data.adata_all


def _coerce_prediction_matrix(pred) -> np.ndarray:
    arr = np.asarray(pred, dtype=np.float32)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    if arr.ndim == 2:
        return arr
    raise ValueError(f"Unexpected GEARS prediction shape: {arr.shape}")


def _predict_single_condition(gears_model, genes: list[str]) -> np.ndarray:
    pred_dict = gears_model.predict([genes])
    if not isinstance(pred_dict, dict) or len(pred_dict) == 0:
        raise ValueError(f"Unexpected GEARS predict output for genes={genes!r}: {type(pred_dict)}")
    pred = list(pred_dict.values())[0]
    return _coerce_prediction_matrix(pred)


def _dataset_topgene_prefix(dataset_name: str) -> str:
    if dataset_name == "adamson":
        return "K562(?)_"
    if dataset_name == "norman":
        return "A549_"
    if dataset_name in {"dixit", "replogle_k562_essential"}:
        return "K562_"
    if dataset_name == "replogle_rpe1_essential":
        return "rpe1_"
    raise ValueError(f"Unsupported GEARS dataset for topgene prefix: {dataset_name}")


def _build_prediction_bundle(gears_model, dataset_name: str, split_meta: GearsSplitMeta) -> list[dict]:
    bundle = []
    seen: set[str] = set()
    prefix = _dataset_topgene_prefix(dataset_name)
    for condition in split_meta.test_conds:
        cond_key = condition_sort(str(condition))
        if cond_key in seen:
            continue
        seen.add(cond_key)
        genes = [token for token in cond_key.split("+") if token != "ctrl"]
        if not genes:
            continue
        pred = _predict_single_condition(gears_model, genes)
        bundle.append(
            {
                "condition": cond_key,
                "genes": genes,
                "pred": pred,
                "source_group": str(split_meta.subgroup_map.get(cond_key, "test")),
                "topgene_key": f"{prefix}{cond_key}_1+1",
            }
        )
    return bundle


def _infer_reference_adata(eval_adata: ad.AnnData, split_meta: GearsSplitMeta) -> ad.AnnData:
    train_conds = set(condition_sort(x) for x in split_meta.train_conds)
    if not train_conds:
        return eval_adata
    eval_conds = eval_adata.obs["condition"].astype(str).map(condition_sort)
    mask = eval_conds.isin(train_conds | {"ctrl"}).values
    if not bool(mask.any()):
        return eval_adata
    return eval_adata[mask].copy()


def _compute_metrics_and_export_payload(
    gears_model,
    eval_adata: ad.AnnData,
    dataset_name: str,
    split_id: int,
    split_meta: GearsSplitMeta,
) -> tuple[pd.DataFrame, dict]:
    results = []
    export_payload = {}
    eval_conds = eval_adata.obs["condition"].astype(str).map(condition_sort).values
    ref_adata = _infer_reference_adata(eval_adata, split_meta)
    pert_reference = average_of_perturbation_centroids(
        X=_utils.densify_X(ref_adata.X),
        conditions=ref_adata.obs["condition"].astype(str).map(condition_sort).values,
        ctrl_label="ctrl",
    )

    if "gene_name" in eval_adata.var.columns:
        gene_names = eval_adata.var["gene_name"].astype(str).values
    else:
        gene_names = eval_adata.var_names.astype(str).values

    ctrl = _utils.densify_X(eval_adata[eval_adata.obs["condition"] == "ctrl"].X)
    prediction_bundle = _build_prediction_bundle(gears_model, dataset_name, split_meta)
    node_map = getattr(gears_model, "node_map", {})
    if not isinstance(node_map, dict):
        raise TypeError("Expected gears_model.node_map to be a mapping")
    top20_degs_final = eval_adata.uns.get("top20_degs_final", {})
    if not isinstance(top20_degs_final, dict):
        raise TypeError("Expected eval_adata.uns['top20_degs_final'] to be a mapping")
    top20_degs_by_canonical = {
        condition_sort(str(cond)): np.asarray(degs, dtype=int).reshape(-1)
        for cond, degs in top20_degs_final.items()
    }

    for item in prediction_bundle:
        cond = str(item["condition"])
        genes = [str(g) for g in item["genes"]]
        if cond == "ctrl":
            continue
        cond_mask = eval_conds == cond
        if not bool(cond_mask.any()):
            print(f"[gears] skip condition missing in eval adata: {cond}")
            continue
        true = _utils.densify_X(eval_adata[cond_mask].X)
        degs = np.asarray(top20_degs_by_canonical.get(cond, []), dtype=int).reshape(-1)
        if degs.size == 0:
            print(f"[gears] skip condition without top20_degs_final entry: {cond}")
            continue

        pred = _coerce_prediction_matrix(item["pred"])
        pred_vec = pred[:, degs].mean(axis=0)
        ctrl_vec = ctrl[:, degs].mean(axis=0)
        true_vec = true[:, degs].mean(axis=0)

        mse_ctrl_val = float(mse(true_vec, ctrl_vec))
        mse_pred_val = float(mse(true_vec, pred_vec))
        nmse_val = float(mse_pred_val / mse_ctrl_val) if mse_ctrl_val > 0 else np.nan
        pearson_val = float(pearsonr(true_vec - ctrl_vec, pred_vec - ctrl_vec)[0])
        deg_mean_r2_val = regression_r2_safe(true_vec - ctrl_vec, pred_vec - ctrl_vec)
        systema_metrics = pearson_delta_reference_metrics(
            X_true=true.mean(axis=0),
            X_pred=pred.mean(axis=0),
            reference=pert_reference,
            top20_de_idxs=degs,
        )
        scpram_metrics = compute_scpram_metrics_from_arrays(
            X_true=true,
            X_pred=pred,
            deg_idx=degs,
            n_degs=100,
            sample_ratio=0.8,
            times=100,
        )
        results.append(
            {
                "condition": cond,
                "mse_pred": mse_pred_val,
                "mse_ctrl": mse_ctrl_val,
                "nmse": nmse_val,
                "pearson": pearson_val,
                "deg_mean_r2": float(deg_mean_r2_val),
                "systema_corr_20de_allpert": float(systema_metrics["corr_20de_allpert"]),
                "systema_corr_deg_r2": float(systema_metrics["corr_deg_r2"]),
                **scpram_metrics,
                "split_id": int(split_id),
                "n_ensemble": 1,
            }
        )

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
        }

    return pd.DataFrame(results), export_payload


def run_gears_eval(
    name: str,
    base_seed: int = 24,
    export_notebook_pkl: bool = True,
    split_ids: list[int] | tuple[int, ...] | None = None,
) -> None:
    if name not in DATASET_CONFIG:
        raise ValueError(f"Unknown dataset: {name}")
    cfg = DATASET_CONFIG[name]
    splits_eff = [int(x) for x in (split_ids if split_ids is not None else cfg.splits)]
    if not splits_eff:
        raise ValueError("split_ids must not be empty")
    PertData, GEARS = _require_gears_classes()

    gears_data_root = _resolve_gears_data_root(cfg.gears_data_name)
    eval_data_path = _resolve_eval_data_path(name, cfg)

    out_dir = ROOT / "artifacts" / "results" / "gears" / name
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_all = []
    for split in splits_eff:
        print(f"[gears] dataset={name} split={split}")
        set_seeds(base_seed + int(split))
        eval_adata = _prepare_eval_adata(eval_data_path)
        pert_data = PertData(str(gears_data_root))
        pert_data.load(data_name=cfg.gears_data_name)
        split_meta = _build_shared_split_meta(
            eval_adata,
            dataset_name=name,
            cfg=cfg,
            split_id=int(split),
            available_raw_conditions=list(getattr(pert_data, "dataset_processed", {}).keys()),
        )
        subgroup_df = None
        if split_meta.subgroup_map:
            subgroup_df = pd.DataFrame({"subgroup": split_meta.subgroup_map})

        _apply_shared_split_to_pert_data(
            pert_data,
            split_meta,
            split_id=int(split),
            train_gene_set_size=cfg.train_gene_set_size,
        )
        _upgrade_legacy_dataset_processed(pert_data, split_meta)
        pert_data.get_dataloader(batch_size=cfg.batch_size, test_batch_size=cfg.test_batch_size)

        gears_model = GEARS(
            pert_data,
            device="cuda" if torch.cuda.is_available() else "cpu",
            weight_bias_track=False,
            proj_name=cfg.project_name,
            exp_name=f"gears_seed{split}",
        )
        gears_model.model_initialize(hidden_size=cfg.hidden_size)
        gears_model.train(epochs=cfg.epochs)

        metrics_df, export_payload = _compute_metrics_and_export_payload(
            gears_model=gears_model,
            eval_adata=eval_adata,
            dataset_name=name,
            split_id=int(split),
            split_meta=split_meta,
        )
        metrics_df = _attach_subgroup_column(metrics_df, subgroup_df)
        metrics_all.append(metrics_df)

        if export_notebook_pkl:
            out_pkl = out_dir / f"gears_{name}_{split}.pkl"
            with out_pkl.open("wb") as f:
                pickle.dump(export_payload, f)
            print(f"[gears] saved notebook payload: {out_pkl}")

    metrics_df_all = pd.concat(metrics_all, ignore_index=True)
    metrics_df_all.to_csv(out_dir / "metrics.csv", index=False)
    _write_mean_metrics(out_dir / "mean_pearson.txt", metrics_df_all)
    print(f"[gears] saved metrics: {out_dir / 'metrics.csv'}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run GEARS eval with TriShift metrics")
    parser.add_argument("--profile", default="", help="dataset profile under scripts/gears/eval/configs")
    parser.add_argument("--name", default="", help="dataset name (legacy mode)")
    parser.add_argument("--seed", type=int, default=24, help="base seed")
    parser.add_argument(
        "--no_export_notebook_pkl",
        action="store_true",
        help="disable notebook-compatible pickle export",
    )
    args = parser.parse_args(argv)
    profile = str(args.profile).strip()
    if profile:
        prof = _load_profile(profile)
        task_args = prof["task_args"]
        seed = int(task_args.get("seed", 24)) if int(args.seed) == 24 else int(args.seed)
        export_notebook_pkl = bool(task_args.get("export_notebook_pkl", True))
        if bool(args.no_export_notebook_pkl):
            export_notebook_pkl = False
        run_gears_eval(
            prof["dataset"],
            base_seed=seed,
            export_notebook_pkl=export_notebook_pkl,
        )
        return
    if not str(args.name).strip():
        raise ValueError("Either --profile or --name must be provided")
    run_gears_eval(
        args.name,
        base_seed=args.seed,
        export_notebook_pkl=not args.no_export_notebook_pkl,
    )


if __name__ == "__main__":
    main()
