from __future__ import annotations

import argparse
from pathlib import Path
import sys
import warnings

import anndata as ad
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error as mse


ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SRC_ROOT))

from trishift import _utils
from trishift._external_metrics import (
    average_of_perturbation_centroids,
    pearson_delta_reference_metrics,
)


# Defaults transcribed from the screenshot in the working thread.
# Edit these lists directly or pass --train-perts / --test-perts on the CLI.
DEFAULT_TRAIN_PERTURBATIONS = [
    "ARHGAP22",
    "ASCC3",
    "ATP5B",
    "BHLHE40",
    "CAD",
    "CARS",
    "CCND3",
    "CHERP",
    "COPB1",
    "COPZ1",
    "CREB1",
    "DAD1",
    "DARS",
    "DDIT3",
    "DDOST",
]
DEFAULT_TEST_PERTURBATIONS = [
    "AARS",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train/evaluate BioLORD on Adamson with a manual perturbation split."
    )
    parser.add_argument(
        "--data",
        default=str(ROOT / "src" / "data" / "adamson" / "perturb_processed.h5ad"),
        help="Path to Adamson perturb_processed.h5ad",
    )
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "external" / "biolord" / "results" / "adamson_manual_split"),
        help="Directory to save metrics and payloads",
    )
    parser.add_argument(
        "--train-perts",
        nargs="*",
        default=DEFAULT_TRAIN_PERTURBATIONS,
        help="Train perturbation gene symbols, e.g. ARHGAP22 ASCC3 ...",
    )
    parser.add_argument(
        "--test-perts",
        nargs="*",
        default=DEFAULT_TEST_PERTURBATIONS,
        help="Test perturbation gene symbols, e.g. AARS",
    )
    parser.add_argument("--seed", type=int, default=24)
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-latent", type=int, default=32)
    parser.add_argument("--attribute-width", type=int, default=64)
    parser.add_argument("--attribute-depth", type=int, default=4)
    return parser.parse_args()


def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def bool2idx(x: np.ndarray) -> np.ndarray:
    return np.where(np.asarray(x).reshape(-1))[0]


def repeat_n(x: torch.Tensor, n: int) -> torch.Tensor:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return x.to(device).view(1, -1).repeat(n, 1)


def normalize_single_condition(pert: str) -> str:
    pert = str(pert).strip()
    if pert == "ctrl":
        return "ctrl"
    if pert.endswith("+ctrl") or pert.endswith("ctrl+"):
        return _utils.normalize_condition(pert)
    return _utils.normalize_condition(f"{pert}+ctrl")


def build_manual_split_adata(
    adata: ad.AnnData,
    train_perts: list[str],
    test_perts: list[str],
) -> tuple[ad.AnnData, list[str], list[str]]:
    train_conditions = [normalize_single_condition(p) for p in train_perts]
    test_conditions = [normalize_single_condition(p) for p in test_perts]
    selected_conditions = ["ctrl"] + train_conditions + test_conditions

    cond_series = adata.obs["condition"].astype(str).map(_utils.normalize_condition)
    keep_mask = cond_series.isin(selected_conditions)
    adata_single = adata[keep_mask].copy()
    cond_series_single = adata_single.obs["condition"].astype(str).map(_utils.normalize_condition)

    split_vals = np.where(cond_series_single.eq("ctrl"), "train", "ignore")
    split_vals = np.where(cond_series_single.isin(train_conditions), "train", split_vals)
    split_vals = np.where(cond_series_single.isin(test_conditions), "test", split_vals)
    adata_single.obs["manual_split"] = pd.Categorical(split_vals, categories=["train", "test", "ignore"])

    # BioLORD requires an ordered attribute. The original reproducibility data ships a
    # precomputed perturbation-neighbor embedding; here we build a simple one-hot code so the
    # script is self-contained on top of perturb_processed.h5ad.
    ordered_conditions = selected_conditions
    emb = np.zeros((adata_single.n_obs, len(ordered_conditions)), dtype=np.float32)
    cond_to_idx = {c: i for i, c in enumerate(ordered_conditions)}
    for row_idx, cond in enumerate(cond_series_single.tolist()):
        emb[row_idx, cond_to_idx[cond]] = 1.0
    adata_single.obsm["perturbation_neighbors"] = emb

    return adata_single, train_conditions, test_conditions


def gene_names_from_adata(adata: ad.AnnData) -> np.ndarray:
    if "gene_name" in adata.var.columns:
        return adata.var["gene_name"].astype(str).values
    return adata.var_names.astype(str).values


def build_name_map(adata: ad.AnnData) -> dict[str, str]:
    out: dict[str, str] = {}
    if "condition_name" in adata.obs.columns:
        pairs = adata.obs[["condition", "condition_name"]].drop_duplicates().values
        for cond, cname in pairs:
            out[_utils.normalize_condition(str(cond))] = str(cname)
    for cond in adata.obs["condition"].astype(str).unique():
        out.setdefault(_utils.normalize_condition(cond), cond)
    return out


def main() -> None:
    args = parse_args()
    set_seeds(int(args.seed))

    try:
        import biolord  # type: ignore
    except Exception as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError(
            "biolord is not installed in the current environment. "
            "Run this script in the BioLORD/Scouter environment."
        ) from exc

    data_path = Path(args.data).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    adata = ad.read_h5ad(str(data_path))
    adata.obs["condition"] = adata.obs["condition"].astype(str).map(_utils.normalize_condition).astype("category")
    adata_single, train_conditions, test_conditions = build_manual_split_adata(
        adata=adata,
        train_perts=list(args.train_perts),
        test_perts=list(args.test_perts),
    )

    adata_single = adata_single[adata_single.obs["manual_split"].astype(str).isin(["train", "test"])].copy()
    gene_names = gene_names_from_adata(adata)
    name_map = build_name_map(adata)

    module_params = {
        "attribute_nn_width": int(args.attribute_width),
        "attribute_nn_depth": int(args.attribute_depth),
        "use_batch_norm": False,
        "use_layer_norm": False,
        "attribute_dropout_rate": 0.1,
        "seed": int(args.seed),
        "n_latent_attribute_ordered": 256,
        "n_latent_attribute_categorical": 16,
        "reconstruction_penalty": 1000.0,
        "unknown_attribute_penalty": 10000.0,
        "decoder_width": 64,
        "decoder_depth": 1,
        "decoder_activation": False,
        "attribute_nn_activation": False,
        "unknown_attributes": False,
    }
    trainer_params = {
        "n_epochs_warmup": 0,
        "attribute_nn_lr": 0.001,
        "attribute_nn_wd": 4e-8,
        "step_size_lr": 45,
        "cosine_scheduler": True,
        "scheduler_final_lr": 1e-5,
        "decoder_lr": 0.001,
        "decoder_wd": 0.01,
    }

    biolord.Biolord.setup_anndata(
        adata_single,
        ordered_attributes_keys=["perturbation_neighbors"],
        categorical_attributes_keys=None,
        retrieval_attribute_key=None,
    )

    model = biolord.Biolord(
        adata=adata_single,
        n_latent=int(args.n_latent),
        model_name="adamson_manual_split",
        module_params=module_params,
        train_classifiers=False,
        split_key="manual_split",
        train_split="train",
        valid_split="test",
        test_split="test",
    )
    model.train(
        max_epochs=int(args.max_epochs),
        batch_size=int(args.batch_size),
        plan_kwargs=trainer_params,
        early_stopping=True,
        early_stopping_patience=10,
        check_val_every_n_epoch=5,
        num_workers=0,
        enable_checkpointing=False,
    )

    cond_all = adata.obs["condition"].astype(str).map(_utils.normalize_condition)
    ctrl_mask_all = cond_all.eq("ctrl").values
    ctrl = np.asarray(_utils.densify_X(adata[ctrl_mask_all].X).mean(axis=0), dtype=np.float32).reshape(-1)

    ref_train = adata_single[adata_single.obs["manual_split"].astype(str).eq("train").values].copy()
    pert_reference = average_of_perturbation_centroids(
        X=_utils.densify_X(ref_train.X),
        conditions=ref_train.obs["condition"].astype(str).values,
        ctrl_label="ctrl",
    )

    adata_control = adata_single[adata_single.obs["condition"].astype(str) == "ctrl"].copy()
    dataset_control = model.get_dataset(adata_control)
    dataset_reference = model.get_dataset(adata_single)

    cond_lookup = adata_single.obs["condition"].astype(str).tolist()
    n_obs = adata_control.n_obs

    rows: list[dict] = []
    payload: dict[str, dict] = {}
    for cond in test_conditions:
        cond_mask_all = cond_all.eq(cond).values
        if not np.any(cond_mask_all):
            warnings.warn(f"test condition not found in adata: {cond}", RuntimeWarning, stacklevel=2)
            continue

        idx_ref = cond_lookup.index(cond)
        true_expr = _utils.densify_X(adata[cond_mask_all].X)
        if true_expr.ndim == 1:
            true_expr = true_expr.reshape(1, -1)
        dataset_pred = dataset_control.copy()
        dataset_pred["perturbation_neighbors"] = repeat_n(dataset_reference["perturbation_neighbors"][idx_ref, :], n_obs)
        pred_expr, _ = model.module.get_expression(dataset_pred)
        pred_expr = pred_expr.detach().cpu().numpy()

        condition_name = name_map.get(cond, cond)
        de_genes = np.array(adata.uns["top_non_zero_de_20"][condition_name])
        deg_mask = adata.var_names.isin(de_genes)
        deg_idx = bool2idx(deg_mask)

        true_vec = true_expr[:, deg_idx].mean(axis=0)
        pred_vec = pred_expr[:, deg_idx].mean(axis=0)
        mse_val = float(mse(true_vec, pred_vec))
        pearson_delta = float(
            pearson_delta_reference_metrics(
                X_true=true_expr.mean(axis=0),
                X_pred=pred_expr.mean(axis=0),
                reference=pert_reference,
                top20_de_idxs=deg_idx,
            )["corr_20de_allpert"]
        )

        rows.append(
            {
                "condition": cond,
                "condition_name": condition_name,
                "split": "manual",
                "mse": mse_val,
                "pearson_delta": pearson_delta,
                "n_cells_true": int(true_expr.shape[0]),
                "n_cells_pred": int(pred_expr.shape[0]),
            }
        )
        payload[cond] = {
            "Pred": pred_expr[:, deg_idx],
            "Ctrl": np.repeat(ctrl.reshape(1, -1)[:, deg_idx], pred_expr.shape[0], axis=0),
            "Truth": true_expr[:, deg_idx],
            "Pred_full": pred_expr,
            "Ctrl_full": np.repeat(ctrl.reshape(1, -1), pred_expr.shape[0], axis=0),
            "Truth_full": true_expr,
            "DE_idx": deg_idx,
            "DE_name": gene_names[deg_idx],
            "gene_name_full": gene_names,
        }

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(out_dir / "metrics_per_condition.csv", index=False)
    summary_df = pd.DataFrame(
        [
            {
                "n_train_perts": len(train_conditions),
                "n_test_perts": len(test_conditions),
                "mean_mse": float(metrics_df["mse"].mean()) if not metrics_df.empty else np.nan,
                "mean_pearson_delta": float(metrics_df["pearson_delta"].mean()) if not metrics_df.empty else np.nan,
            }
        ]
    )
    summary_df.to_csv(out_dir / "summary.csv", index=False)

    meta = pd.DataFrame(
        {
            "role": ["train"] * len(train_conditions) + ["test"] * len(test_conditions),
            "condition": train_conditions + test_conditions,
        }
    )
    meta.to_csv(out_dir / "manual_split_conditions.csv", index=False)

    import pickle

    with (out_dir / "biolord_adamson_manual_split.pkl").open("wb") as handle:
        pickle.dump(payload, handle)

    print(f"saved metrics: {out_dir / 'metrics_per_condition.csv'}")
    print(f"saved summary: {out_dir / 'summary.csv'}")
    print(f"saved split spec: {out_dir / 'manual_split_conditions.csv'}")
    print(f"saved payload: {out_dir / 'biolord_adamson_manual_split.pkl'}")


if __name__ == "__main__":
    main()
