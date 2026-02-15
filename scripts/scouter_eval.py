from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import pickle
import random
import sys

import numpy as np
import pandas as pd
import torch
import anndata as ad
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error as mse

# Ensure local src and scouter package are importable
ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
SCOUTER_ROOT = ROOT / "external" / "scouter" / "scouter-master" / "scouter-master"
SCOUTER_MISC = ROOT / "external" / "scouter" / "scouter_misc-main" / "scouter_misc-main"

sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(0, str(SCOUTER_ROOT))

from trishift import _utils
from trishift._external_metrics import (
    average_of_perturbation_centroids,
    compute_scpram_metrics_from_arrays,
    pearson_delta_reference_metrics,
)
from scouter import Scouter, ScouterData


@dataclass(frozen=True)
class ScouterDatasetConfig:
    data_rel: str
    emb_rel: str
    emb_key: str
    emb_rename: dict
    splits: list[int]
    loss_lambda: float
    lr: float
    test_ratio: float | None = None
    norman_split: bool = False


def _write_mean_metrics(path: Path, metrics_df: pd.DataFrame) -> None:
    """Write mean pearson and other averaged numeric metrics to text file.

    Backward compatibility:
    - First line remains plain mean pearson float for old readers.
    """
    numeric_means = metrics_df.mean(numeric_only=True)
    mean_pearson = float(numeric_means.get("pearson", float("nan")))

    preferred_order = [
        "pearson",
        "nmse",
        "systema_corr_all_allpert",
        "systema_corr_20de_allpert",
        "scpram_r2_all_mean_mean",
        "scpram_r2_all_var_mean",
        "scpram_r2_degs_mean_mean",
        "scpram_r2_degs_var_mean",
        "scpram_r2_all_mean_std",
        "scpram_r2_all_var_std",
        "scpram_r2_degs_mean_std",
        "scpram_r2_degs_var_std",
        "scpram_wasserstein_all_sum",
        "scpram_wasserstein_degs_sum",
    ]
    exclude_keys = {"split_id", "n_ensemble"}
    keys = [k for k in preferred_order if k in numeric_means.index and k not in exclude_keys]
    keys.extend(
        [
            k
            for k in numeric_means.index
            if k not in exclude_keys and k not in keys
        ]
    )

    lines = [f"{mean_pearson}\n"]
    for key in keys:
        val = float(numeric_means[key])
        lines.append(f"mean_{key}={val}\n")

    path.write_text("".join(lines), encoding="utf-8")


def set_seeds(seed: int = 24) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def condition_sort(x: str) -> str:
    return "+".join(sorted(x.split("+")))


def split_list_by_ratio(lst: list[str], ratios: list[float], seed: int) -> list[list[str]]:
    random.seed(seed)
    random.shuffle(lst)

    total_ratio = sum(ratios)
    total_length = len(lst)
    num_splits = len(ratios)

    if total_length < num_splits:
        raise ValueError("The length of the list must be >= number of ratios")

    sizes = [max(1, int(total_length * ratio / total_ratio)) for ratio in ratios]

    while sum(sizes) > total_length:
        sizes[sizes.index(max(sizes))] -= 1

    sizes[-1] += total_length - sum(sizes)

    splits: list[list[str]] = []
    current_index = 0
    for size in sizes:
        splits.append(lst[current_index : current_index + size])
        current_index += size

    return splits


def subgroup(pert_list: list[str], seed: int) -> pd.DataFrame:
    uniq_perts = pert_list.copy()
    uniq_perts.remove("ctrl")
    uniq_combos = [p for p in uniq_perts if "ctrl" not in p.split("+")]
    uniq_singles = [p for p in uniq_perts if "ctrl" in p.split("+")]

    test_single, val_single, train_single = split_list_by_ratio(
        uniq_singles, [0.2, 0.05, 0.75], seed=seed
    )

    combo_seen0 = [
        p
        for p in uniq_combos
        if sum([i + "+ctrl" in train_single for i in p.split("+")]) == 0
    ]
    combo_seen1 = [
        p
        for p in uniq_combos
        if sum([i + "+ctrl" in train_single for i in p.split("+")]) == 1
    ]
    combo_seen2 = [
        p
        for p in uniq_combos
        if sum([i + "+ctrl" in train_single for i in p.split("+")]) == 2
    ]

    test_seen0, val_seen0, train_seen0 = split_list_by_ratio(
        combo_seen0, [0.2, 0.1, 0.7], seed=seed
    )
    test_seen1, val_seen1, train_seen1 = split_list_by_ratio(
        combo_seen1, [0.2, 0.1, 0.7], seed=seed
    )
    test_seen2, val_seen2, train_seen2 = split_list_by_ratio(
        combo_seen2, [0.2, 0.1, 0.7], seed=seed
    )

    test_all = test_single + test_seen0 + test_seen1 + test_seen2
    val_all = val_single + val_seen0 + val_seen1 + val_seen2
    train_all = train_single + train_seen0 + train_seen1 + train_seen2

    group = [
        "train" if p in train_all else "val" if p in val_all else "test"
        for p in uniq_perts
    ]
    subgroup_list = []
    for p in uniq_perts:
        if p in uniq_singles:
            subgroup_list.append("single")
        elif p in combo_seen0:
            subgroup_list.append("seen0")
        elif p in combo_seen1:
            subgroup_list.append("seen1")
        elif p in combo_seen2:
            subgroup_list.append("seen2")

    df = pd.DataFrame({"group": group, "subgroup": subgroup_list}, index=uniq_perts)
    return df


DATASET_CONFIG = {
    "adamson": ScouterDatasetConfig(
        data_rel="data/Data_GEARS/adamson/perturb_processed.h5ad",
        emb_rel="data/Data_GeneEmbd/GenePT_V2.pickle",
        emb_key="emb_b",
        emb_rename={
            "SARS1": "SARS",
            "DARS1": "DARS",
            "QARS1": "QARS",
            "TARS1": "TARS",
            "HARS1": "HARS",
            "CARS1": "CARS",
            "SRPRA": "SRPR",
            "MARS1": "MARS",
            "AARS1": "AARS",
            "PRELID3B": "SLMO2",
        },
        splits=[1, 2, 3, 4, 5],
        loss_lambda=0.01,
        lr=0.001,
    ),
    "dixit": ScouterDatasetConfig(
        data_rel="data/Data_GEARS/dixit/perturb_processed.h5ad",
        emb_rel="data/Data_GeneEmbd/GenePT_V1.pickle",
        emb_key="emb_c",
        emb_rename={},
        splits=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        loss_lambda=0.05,
        lr=0.01,
        test_ratio=0.1,
    ),
    "norman": ScouterDatasetConfig(
        data_rel="data/Data_GEARS/norman/perturb_processed.h5ad",
        emb_rel="data/Data_GeneEmbd/scELMo.pickle",
        emb_key="emb_a",
        emb_rename={
            "MAP3K21": "KIAA1804",
            "FOXL2NB": "C3orf72",
            "RHOXF2B": "RHOXF2BB",
            "MIDEAS": "ELMSAN1",
            "CBARP": "C19orf26",
        },
        splits=[1, 2, 3, 4, 5],
        loss_lambda=0.05,
        lr=0.001,
        norman_split=True,
    ),
    "replogle_k562_essential": ScouterDatasetConfig(
        data_rel="data/Data_GEARS/replogle_k562_essential/perturb_processed.h5ad",
        emb_rel="data/Data_GeneEmbd/GenePT_V1.pickle",
        emb_key="emb_c",
        emb_rename={
            "AARS1": "AARS",
            "CENATAC": "CCDC84",
            "POLR1G": "CD3EAP",
            "DARS1": "DARS",
            "EPRS1": "EPRS",
            "HARS1": "HARS",
            "IARS1": "IARS",
            "KARS1": "KARS",
            "LARS1": "LARS",
            "MARS1": "MARS",
            "QARS1": "QARS",
            "RARS1": "RARS",
            "SARS1": "SARS",
            "TARS1": "TARS",
            "POLR1F": "TWISTNB",
            "VARS1": "VARS",
            "POLR1H": "ZNRD1",
        },
        splits=[1, 2, 3, 4, 5],
        loss_lambda=0.5,
        lr=0.001,
    ),
    "replogle_rpe1_essential": ScouterDatasetConfig(
        data_rel="data/Data_GEARS/replogle_rpe1_essential/perturb_processed.h5ad",
        emb_rel="data/Data_GeneEmbd/GenePT_V2.pickle",
        emb_key="emb_b",
        emb_rename={"ZZZ3": "AC118549.1"},
        splits=[1, 2, 3, 4, 5],
        loss_lambda=0.5,
        lr=0.001,
    ),
}


def _resolve_data_and_emb_paths(name: str, cfg: ScouterDatasetConfig) -> tuple[Path, Path]:
    data_path_primary = SCOUTER_MISC / cfg.data_rel
    emb_path_primary = SCOUTER_MISC / cfg.emb_rel
    if data_path_primary.exists() and emb_path_primary.exists():
        return data_path_primary, emb_path_primary

    paths_cfg_path = ROOT / "configs" / "paths.yaml"
    if paths_cfg_path.exists():
        paths_cfg = _utils.load_yaml(str(paths_cfg_path))
        datasets_map = paths_cfg.get("datasets", {})
        embeddings_map = paths_cfg.get("embeddings", {})
        data_path_alt = Path(datasets_map[name]) if name in datasets_map else data_path_primary
        emb_path_alt = (
            Path(embeddings_map[cfg.emb_key])
            if cfg.emb_key in embeddings_map
            else emb_path_primary
        )
        if data_path_alt.exists() and emb_path_alt.exists():
            return data_path_alt, emb_path_alt

    raise FileNotFoundError(
        "Missing dataset/embedding files. "
        f"Tried primary data={data_path_primary} emb={emb_path_primary}; "
        f"fallback via {paths_cfg_path} with emb_key={cfg.emb_key}."
    )


def _load_embedding_df(emb_path: Path, emb_rename: dict) -> pd.DataFrame:
    with emb_path.open("rb") as handle:
        embd_df = pd.DataFrame(pickle.load(handle)).T
    ctrl_row = pd.DataFrame(
        [np.zeros(embd_df.shape[1])], columns=embd_df.columns, index=["ctrl"]
    )
    embd_df = pd.concat([ctrl_row, embd_df])
    if emb_rename:
        embd_df = embd_df.rename(index=emb_rename)
    return embd_df


def _prepare_adata(data_path: Path) -> ad.AnnData:
    adata = ad.read_h5ad(str(data_path))
    adata.obs["condition"] = (
        adata.obs["condition"].astype(str).apply(condition_sort).astype("category")
    )
    adata.uns = {}
    if "condition_name" in adata.obs:
        adata.obs.drop("condition_name", axis=1, inplace=True)
    return adata


def _compute_metrics_and_export_payload(
    scouter_model: Scouter, split_id: int, n_ensemble: int, base_seed: int
) -> tuple[pd.DataFrame, dict]:
    conds = list(scouter_model.test_adata.obs[scouter_model.key_label].unique())
    if "ctrl" in conds:
        conds.remove("ctrl")
    results = []
    export_payload = {}
    train_cond_arr = scouter_model.train_adata.obs[scouter_model.key_label].astype(str).values
    pert_reference = average_of_perturbation_centroids(
        X=_utils.densify_X(scouter_model.train_adata.X),
        conditions=train_cond_arr,
        ctrl_label="ctrl",
    )

    if scouter_model.key_var_genename in scouter_model.all_adata.var.columns:
        gene_names = (
            scouter_model.all_adata.var[scouter_model.key_var_genename]
            .astype(str)
            .values
        )
    else:
        gene_names = scouter_model.all_adata.var_names.astype(str).values

    for cond in conds:
        degs = scouter_model.all_adata.uns["top20_degs_non_dropout"][cond]
        pert_genes = np.where(
            np.isin(
                scouter_model.all_adata.var[scouter_model.key_var_genename].values,
                cond.split("+"),
            )
        )
        degs = np.setdiff1d(degs, pert_genes)

        pred = scouter_model.pred([cond], n_pred=n_ensemble, seed=base_seed + split_id)[cond]
        ctrl = _utils.densify_X(scouter_model.ctrl_adata.X)
        true = _utils.densify_X(
            scouter_model.all_adata[
                scouter_model.all_adata.obs[scouter_model.key_label] == cond
            ].X
        )
        degs = np.asarray(degs, dtype=int)
        if degs.size == 0:
            print(f"[scouter] skip condition without DEGs: {cond}")
            continue
        pred_vec = pred[:, degs].mean(axis=0)
        ctrl_vec = ctrl[:, degs].mean(axis=0)
        true_vec = true[:, degs].mean(axis=0)
        denom = mse(true_vec, ctrl_vec)
        nmse_val = float(mse(true_vec, pred_vec) / denom) if denom > 0 else np.nan
        pearson_val = float(pearsonr(true_vec - ctrl_vec, pred_vec - ctrl_vec)[0])
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
                "nmse": nmse_val,
                "pearson": pearson_val,
                "systema_corr_all_allpert": float(systema_metrics["corr_all_allpert"]),
                "systema_corr_20de_allpert": float(systema_metrics["corr_20de_allpert"]),
                **scpram_metrics,
                "split_id": split_id,
                "n_ensemble": n_ensemble,
            }
        )

        # Notebook-compatible payload format (same schema used by TriShift export).
        # Keep matrices on DE genes to control file size.
        deg_names = gene_names[degs] if degs.size > 0 else np.array([], dtype=gene_names.dtype)
        ctrl_idx = _utils.sample_control_indices(
            n_ctrl=ctrl.shape[0], size=n_ensemble, seed=base_seed + split_id
        )
        ctrl_sampled = ctrl[ctrl_idx]
        export_payload[cond] = {
            "Pred": pred[:, degs] if degs.size > 0 else pred[:, :0],
            "Ctrl": ctrl_sampled[:, degs] if degs.size > 0 else ctrl_sampled[:, :0],
            "Truth": true[:, degs] if degs.size > 0 else true[:, :0],
            "DE_idx": degs,
            "DE_name": deg_names,
        }

    return pd.DataFrame(results), export_payload


def run_scouter_eval(
    name: str,
    n_ensemble: int = 300,
    base_seed: int = 24,
    export_notebook_pkl: bool = True,
) -> None:
    if name not in DATASET_CONFIG:
        raise ValueError(f"Unknown dataset: {name}")
    cfg = DATASET_CONFIG[name]

    data_path, emb_path = _resolve_data_and_emb_paths(name, cfg)

    out_dir = ROOT / "artifacts" / "results" / "scouter" / name
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_all = []
    for split in cfg.splits:
        print(f"[scouter] dataset={name} split={split}")
        set_seeds(base_seed)
        adata = _prepare_adata(data_path)
        embd_df = _load_embedding_df(emb_path, cfg.emb_rename)

        pertdata = ScouterData(adata, embd_df, "condition", "gene_name")
        pertdata.setup_ad("embd_index")
        pertdata.gene_ranks()
        pertdata.get_dropout_non_zero_genes()

        if cfg.norman_split:
            splt_df = subgroup(list(adata.obs.condition.unique()), seed=split)
            test_conds = list(splt_df[splt_df.group == "test"].index)
            val_conds = list(splt_df[splt_df.group == "val"].index)
            pertdata.split_Train_Val_Test(val_conds=val_conds, test_conds=test_conds, seed=split)
        elif cfg.test_ratio is not None:
            pertdata.split_Train_Val_Test(test_ratio=cfg.test_ratio, seed=split)
        else:
            pertdata.split_Train_Val_Test(seed=split)

        scouter_model = Scouter(pertdata)
        scouter_model.model_init()
        scouter_model.train(loss_lambda=cfg.loss_lambda, lr=cfg.lr)

        metrics_df, export_payload = _compute_metrics_and_export_payload(
            scouter_model, split, n_ensemble, base_seed
        )
        metrics_all.append(metrics_df)
        if export_notebook_pkl:
            out_pkl = out_dir / f"scouter_{name}_{split}.pkl"
            with out_pkl.open("wb") as f:
                pickle.dump(export_payload, f)
            # Alias name for notebooks already using TriShift naming convention.
            out_pkl_alias = out_dir / f"trishift_{name}_{split}.pkl"
            with out_pkl_alias.open("wb") as f:
                pickle.dump(export_payload, f)
            print(f"[scouter] saved notebook payload: {out_pkl}")

    metrics_df_all = pd.concat(metrics_all, ignore_index=True)
    metrics_df_all.to_csv(out_dir / "metrics.csv", index=False)
    _write_mean_metrics(out_dir / "mean_pearson.txt", metrics_df_all)
    print(f"[scouter] saved metrics: {out_dir / 'metrics.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Scouter eval with TriShift metrics")
    parser.add_argument("--name", required=True, help="dataset name")
    parser.add_argument("--n_ensemble", type=int, default=300, help="ensemble size")
    parser.add_argument("--seed", type=int, default=24, help="base seed")
    parser.add_argument(
        "--no_export_notebook_pkl",
        action="store_true",
        help="disable notebook-compatible pickle export",
    )
    args = parser.parse_args()
    run_scouter_eval(
        args.name,
        n_ensemble=args.n_ensemble,
        base_seed=args.seed,
        export_notebook_pkl=not args.no_export_notebook_pkl,
    )


if __name__ == "__main__":
    main()
