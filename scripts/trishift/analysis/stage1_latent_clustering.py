from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
import json
import pickle
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

from scripts.common.split_utils import norman_subgroup as _shared_norman_subgroup

if TYPE_CHECKING:
    from trishift.TriShiftData import TriShiftData


def _get_dataset_config() -> dict:
    from scripts.trishift._core.run_dataset_core import DATASET_CONFIG

    return DATASET_CONFIG


def _get_init_model():
    from scripts.trishift._core.run_dataset_core import _init_model

    return _init_model


def _get_trishift_data_cls():
    from trishift.TriShiftData import TriShiftData

    return TriShiftData


def _get_utils():
    from trishift._utils import (
        apply_alias_mapping,
        load_adata,
        load_embedding_df,
        load_yaml,
        set_seeds,
    )

    return {
        "apply_alias_mapping": apply_alias_mapping,
        "load_adata": load_adata,
        "load_embedding_df": load_embedding_df,
        "load_yaml": load_yaml,
        "set_seeds": set_seeds,
    }


@dataclass(frozen=True)
class Stage1LatentRunResult:
    out_dir: Path
    latent_h5ad_path: Path
    z_mu_path: Path
    metrics_csv_path: Path
    train_logs_path: Path
    figure_paths: dict[str, str]
    metrics_df: pd.DataFrame
    latent_adata: ad.AnnData
    train_logs: dict
    metadata: dict


def _repo_root() -> Path:
    root = Path.cwd().resolve()
    if (root / "scripts").exists():
        return root
    if (root.parent / "scripts").exists():
        return root.parent
    return REPO_ROOT


def _set_repo_root() -> Path:
    root = _repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    src_root = root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    return root


def _make_out_dir(
    *,
    mode: str,
    dataset_name: str,
    split_id: int,
    stage1_pool_mode: str,
    random_seed: int,
) -> Path:
    root = _repo_root()
    safe_mode = str(mode).strip().lower()
    safe_dataset = str(dataset_name).strip().lower()
    safe_pool = str(stage1_pool_mode).strip().lower()
    out_dir = (
        root
        / "artifacts"
        / "stage1_latent_clustering"
        / safe_mode
        / safe_dataset
        / f"split{int(split_id)}"
        / f"{safe_pool}_seed{int(random_seed)}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _load_defaults(defaults_path: str | Path = "configs/defaults.yaml") -> dict:
    return _get_utils()["load_yaml"](str((_repo_root() / defaults_path).resolve()))


def _load_paths(paths_path: str | Path = "configs/paths.yaml") -> dict:
    return _get_utils()["load_yaml"](str((_repo_root() / paths_path).resolve()))


def _l2_normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, float(eps))
    return x / norms


def _dump_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _dump_train_logs_json(path: Path, train_logs: dict) -> None:
    path.write_text(json.dumps(train_logs, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _compute_nonzero_non_dropout(
    pert_mean: np.ndarray,
    ctrl_mean: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    non_zero = np.where(pert_mean != 0)[0]
    zero = np.where(pert_mean == 0)[0]
    true_zeros = np.intersect1d(zero, np.where(ctrl_mean == 0)[0])
    non_dropouts = np.concatenate((non_zero, true_zeros))
    return non_zero, non_dropouts


def _pbmc_deg_cache_path(
    pbmc_path: str | Path,
    pbmc_deg_mode: str,
) -> Path:
    pbmc_file = Path(pbmc_path).resolve()
    safe_mode = str(pbmc_deg_mode).strip().lower()
    return (
        _repo_root()
        / "artifacts"
        / "cache"
        / "pbmc_degs"
        / f"{pbmc_file.stem}__{safe_mode}__top20_non_dropout.pkl"
    )


def _build_pbmc_deg_cache_payload(
    adata: ad.AnnData,
    *,
    pbmc_path: str | Path,
    pbmc_deg_mode: str,
) -> dict:
    mode_key = str(pbmc_deg_mode).strip().lower()
    if mode_key != "by_cell_type":
        raise ValueError("pbmc_deg_mode must be 'by_cell_type'")

    if "cell_type" not in adata.obs.columns:
        raise ValueError("PBMC adata must contain obs['cell_type']")
    if "condition" not in adata.obs.columns:
        raise ValueError("PBMC adata must contain obs['condition']")

    cond_series = adata.obs["condition"].astype(str)
    if "gene_name" in adata.var.columns:
        gene_names = adata.var["gene_name"].astype(str).values
    elif "gene_symbol" in adata.var.columns:
        gene_names = adata.var["gene_symbol"].astype(str).values
    else:
        gene_names = adata.var_names.astype(str).values
    gene_id2idx = {g: i for i, g in enumerate(gene_names)}

    top_non_dropout_de_20: dict[str, np.ndarray] = {}
    cell_type_stats: dict[str, dict] = {}
    for cell_type in sorted(adata.obs["cell_type"].astype(str).unique().tolist()):
        subset = adata[adata.obs["cell_type"].astype(str).eq(cell_type).values].copy()
        subset_cond = subset.obs["condition"].astype(str)
        ctrl_mask = subset_cond.eq("control").values
        stim_mask = subset_cond.eq("stimulated").values
        if not bool(np.any(ctrl_mask)) or not bool(np.any(stim_mask)):
            cell_type_stats[cell_type] = {
                "status": "skipped_missing_group",
                "n_ctrl": int(np.sum(ctrl_mask)),
                "n_stim": int(np.sum(stim_mask)),
            }
            continue

        var_names_backup = subset.var_names.copy()
        if "gene_name" in subset.var.columns:
            subset.var_names = subset.var["gene_name"].astype(str).values
        elif "gene_symbol" in subset.var.columns:
            subset.var_names = subset.var["gene_symbol"].astype(str).values
        sc.tl.rank_genes_groups(
            subset,
            groupby="condition",
            reference="control",
            groups=["stimulated"],
            rankby_abs=True,
            n_genes=subset.n_vars,
            method="t-test",
        )
        names_df = pd.DataFrame(subset.uns["rank_genes_groups"]["names"])
        rank_genes = names_df["stimulated"].astype(str).tolist()
        subset.var_names = var_names_backup

        ctrl_mean = np.asarray(subset[ctrl_mask].X.mean(axis=0)).ravel()
        pert_mean = np.asarray(subset[stim_mask].X.mean(axis=0)).ravel()
        _, non_dropouts = _compute_nonzero_non_dropout(pert_mean, ctrl_mean)
        gene_idx_top = [gene_id2idx[g] for g in rank_genes if g in gene_id2idx]
        non_dropout_20 = np.array(
            [i for i in gene_idx_top if i in non_dropouts][:20], dtype=int
        )
        key = f"stimulated::{cell_type}"
        top_non_dropout_de_20[key] = non_dropout_20
        cell_type_stats[cell_type] = {
            "status": "ok",
            "n_ctrl": int(np.sum(ctrl_mask)),
            "n_stim": int(np.sum(stim_mask)),
            "n_top20_non_dropout": int(non_dropout_20.size),
        }

    return {
        "top20_degs_non_dropout_by_key": top_non_dropout_de_20,
        "degs_meta": {
            "rank_genes_groups_method": "t-test",
            "rank_genes_groups_reference": "control",
            "rankby_abs": True,
            "generator": "stage1_latent_clustering.pbmc_by_cell_type",
            "version": 1,
            "pbmc_deg_mode": mode_key,
            "source_path": str(Path(pbmc_path).resolve()),
        },
        "cell_type_stats": cell_type_stats,
    }


def _load_or_build_pbmc_deg_cache(
    adata: ad.AnnData,
    *,
    pbmc_path: str | Path,
    pbmc_deg_mode: str,
) -> tuple[dict, Path]:
    cache_path = _pbmc_deg_cache_path(pbmc_path, pbmc_deg_mode)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            payload = pickle.load(f)
        meta = payload.get("degs_meta", {})
        if (
            isinstance(meta, dict)
            and str(meta.get("pbmc_deg_mode", "")).strip().lower() == str(pbmc_deg_mode).strip().lower()
            and str(meta.get("source_path", "")) == str(Path(pbmc_path).resolve())
        ):
            return payload, cache_path

    payload = _build_pbmc_deg_cache_payload(
        adata,
        pbmc_path=pbmc_path,
        pbmc_deg_mode=pbmc_deg_mode,
    )
    with open(cache_path, "wb") as f:
        pickle.dump(payload, f)
    return payload, cache_path


def _stage1_cache_identity(
    *,
    mode: str,
    dataset_name: str,
    split_id: int,
    stage1_pool_mode: str,
    random_seed: int,
    stage1_epochs: int,
    stage1_z_dim: int,
    stage1_batch_size: int,
    stage1_lr: float,
    stage1_beta: float,
    stage1_deg_weight: float,
    stage1_sched_gamma: float,
    stage1_patience: int,
    stage1_min_delta: float,
    pbmc_path: str | Path,
    pbmc_train_frac: float,
    pbmc_deg_mode: str,
) -> dict:
    return {
        "mode": str(mode),
        "dataset_name": str(dataset_name),
        "split_id": int(split_id),
        "stage1_pool_mode": str(stage1_pool_mode),
        "random_seed": int(random_seed),
        "stage1_epochs": int(stage1_epochs),
        "stage1_z_dim": int(stage1_z_dim),
        "stage1_batch_size": int(stage1_batch_size),
        "stage1_lr": float(stage1_lr),
        "stage1_beta": float(stage1_beta),
        "stage1_deg_weight": float(stage1_deg_weight),
        "stage1_sched_gamma": float(stage1_sched_gamma),
        "stage1_patience": int(stage1_patience),
        "stage1_min_delta": float(stage1_min_delta),
        "pbmc_path": str(Path(pbmc_path).resolve()) if str(mode) == "pbmc_celltype" else "",
        "pbmc_train_frac": float(pbmc_train_frac) if str(mode) == "pbmc_celltype" else 0.0,
        "pbmc_deg_mode": str(pbmc_deg_mode).strip().lower() if str(mode) == "pbmc_celltype" else "",
    }


def _can_reuse_stage1_cache(
    metadata_path: Path,
    *,
    expected_identity: dict,
) -> bool:
    if not metadata_path.exists():
        return False
    try:
        meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    for key, value in expected_identity.items():
        if meta.get(key) != value:
            return False
    return True


def _norman_split(data: "TriShiftData", split_id: int) -> tuple[dict, pd.DataFrame]:
    subgroup_df = _shared_norman_subgroup(
        list(data.adata_all.obs["condition"].astype(str).unique()),
        seed=int(split_id),
    )
    test_conds = list(subgroup_df[subgroup_df.group == "test"].index)
    val_conds = list(subgroup_df[subgroup_df.group == "val"].index)
    split_dict = data.split_by_condition(
        seed=int(split_id),
        test_conds=test_conds,
        val_conds=val_conds,
    )
    return split_dict, subgroup_df


def _random_cell_split(
    adata: ad.AnnData,
    *,
    train_frac: float,
    seed: int,
) -> tuple[ad.AnnData, ad.AnnData]:
    if not 0.0 < float(train_frac) < 1.0:
        raise ValueError("train_frac must be in (0, 1)")
    rng = np.random.default_rng(int(seed))
    idx = np.arange(adata.n_obs, dtype=int)
    rng.shuffle(idx)
    n_train = max(1, min(int(round(float(train_frac) * adata.n_obs)), adata.n_obs - 1))
    train_idx = idx[:n_train]
    val_idx = idx[n_train:]
    return adata[train_idx].copy(), adata[val_idx].copy()


def _prepare_trishift_data(
    dataset_name: str,
    *,
    stage1_deg_weight: float = 1.0,
    defaults_path: str | Path = "configs/defaults.yaml",
    paths_path: str | Path = "configs/paths.yaml",
) -> tuple[ad.AnnData, "TriShiftData", pd.DataFrame, dict, dict]:
    dataset_config = _get_dataset_config()
    if dataset_name not in dataset_config:
        raise ValueError(f"Unsupported dataset_name={dataset_name}")
    defaults = _load_defaults(defaults_path)
    paths_cfg = _load_paths(paths_path)
    h5ad_path = paths_cfg["datasets"][dataset_name]
    emb_key = dataset_config[dataset_name]["emb_key"]
    emb_path = paths_cfg["embeddings"][emb_key]
    utils = _get_utils()

    adata = utils["load_adata"](h5ad_path)
    adata.uns = {}
    if "condition_name" in adata.obs.columns:
        adata.obs["condition_name_raw"] = adata.obs["condition_name"].astype(str)
    adata.obs["label_condition"] = adata.obs["condition"].astype(str)
    adata.obs["label_ctrl_pert"] = np.where(
        adata.obs["condition"].astype(str).eq("ctrl"),
        "ctrl",
        "pert",
    )
    if "cell_type" in adata.obs.columns:
        adata.obs["label_cell_type"] = adata.obs["cell_type"].astype(str)

    embd_df = utils["load_embedding_df"](emb_path)
    embd_df = utils["apply_alias_mapping"](embd_df, dataset_name)
    data = _get_trishift_data_cls()(adata, embd_df)
    data.setup_embedding_index()
    if float(stage1_deg_weight) != 1.0:
        data.build_or_load_degs()
    return adata, data, embd_df, defaults, paths_cfg


def _prepare_pbmc_data(
    pbmc_path: str | Path,
    *,
    stage1_deg_weight: float = 1.0,
    pbmc_deg_mode: str = "by_cell_type",
) -> tuple[ad.AnnData, "TriShiftData", pd.DataFrame, Path | None]:
    adata = ad.read_h5ad(str(Path(pbmc_path).resolve()))
    adata = sc.AnnData(adata.X, obs=adata.obs.copy(), var=adata.var.copy())
    adata.obs_names_make_unique()
    adata.uns = {}
    adata.obs["label_cell_type"] = adata.obs["cell_type"].astype(str)
    adata.obs["label_condition"] = adata.obs["condition"].astype(str)
    adata.obs["label_ctrl_pert"] = np.where(
        adata.obs["condition"].astype(str).str.lower().eq("control"),
        "ctrl",
        "pert",
    )
    adata.obs["condition_raw"] = adata.obs["condition"].astype(str)
    adata.obs["condition_internal"] = np.where(
        adata.obs["condition"].astype(str).str.lower().eq("control"),
        "ctrl",
        "stimulated",
    )
    adata.obs["stage1_deg_key"] = np.where(
        adata.obs["condition"].astype(str).str.lower().eq("stimulated"),
        "stimulated::" + adata.obs["cell_type"].astype(str),
        "ctrl",
    )
    if "gene_name" not in adata.var.columns:
        if "gene_symbol" in adata.var.columns:
            adata.var["gene_name"] = adata.var["gene_symbol"].astype(str)
        else:
            adata.var["gene_name"] = adata.var_names.astype(str)
    embd_df = pd.DataFrame(
        [[0.0], [1.0]],
        index=["ctrl", "stimulated"],
        columns=["dummy_cond"],
    )
    deg_cache_path = None
    if float(stage1_deg_weight) != 1.0:
        payload, deg_cache_path = _load_or_build_pbmc_deg_cache(
            adata,
            pbmc_path=pbmc_path,
            pbmc_deg_mode=pbmc_deg_mode,
        )
        adata.uns["top20_degs_non_dropout"] = payload["top20_degs_non_dropout_by_key"]
        adata.uns["degs_meta"] = payload["degs_meta"]
    data = _get_trishift_data_cls()(adata, embd_df, label_key="condition_internal")
    data.setup_embedding_index()
    return adata, data, embd_df, deg_cache_path


def _build_model(
    *,
    data: "TriShiftData",
    adata: ad.AnnData,
    embd_df: pd.DataFrame,
    defaults: dict,
    stage1_z_dim: int | None = None,
) -> tuple[Any, dict, dict]:
    stage1_model_cfg = dict(defaults.get("model", {}).get("stage1", {}))
    if stage1_z_dim is not None:
        stage1_model_cfg["z_dim"] = int(stage1_z_dim)
    stage2_model_cfg = dict(defaults.get("model", {}).get("stage2", {}))
    model_cfg = dict(defaults.get("model", {}).get("stage3", {}))
    model = _get_init_model()(
        data,
        adata,
        embd_df,
        stage1_model_cfg,
        stage2_model_cfg,
        model_cfg,
        predict_shift=True,
        shift_input_source_override=None,
        gen_state_source="compressor",
    )
    return model, stage1_model_cfg, defaults.get("train", {}).get("stage1", {})


def _resolve_stage1_train_pool(
    *,
    mode: str,
    data: "TriShiftData",
    split_dict: dict | None,
    stage1_pool_mode: str,
    adata_all: ad.AnnData,
    pbmc_train_frac: float,
    seed: int,
) -> tuple[ad.AnnData, ad.AnnData | None, dict]:
    pool_mode = str(stage1_pool_mode)
    if mode == "pbmc_celltype":
        train_split, val_split = _random_cell_split(adata_all, train_frac=pbmc_train_frac, seed=seed)
        if pool_mode == "train_all_cells":
            return train_split, val_split, {"train_n": int(train_split.n_obs), "val_n": int(val_split.n_obs)}
        if pool_mode != "ctrl_only":
            raise ValueError("stage1_pool_mode must be train_all_cells or ctrl_only")
        train_mask = train_split.obs["label_ctrl_pert"].astype(str).eq("ctrl").values
        val_mask = val_split.obs["label_ctrl_pert"].astype(str).eq("ctrl").values
        train_pool = train_split[train_mask].copy()
        val_pool = val_split[val_mask].copy()
        if train_pool.n_obs == 0:
            raise ValueError("PBMC ctrl_only stage1 pool is empty")
        return train_pool, val_pool if val_pool.n_obs > 0 else None, {
            "train_n": int(train_pool.n_obs),
            "val_n": int(val_pool.n_obs),
        }

    if split_dict is None:
        raise ValueError("split_dict is required for trishift_condition mode")
    if pool_mode == "train_all_cells":
        return split_dict["train"], split_dict.get("val"), {
            "train_n": int(split_dict["train"].n_obs),
            "val_n": int(split_dict.get("val").n_obs if split_dict.get("val") is not None else 0),
        }
    if pool_mode != "ctrl_only":
        raise ValueError("stage1_pool_mode must be train_all_cells or ctrl_only")

    stage1_train = data.adata_ctrl
    stage1_val = None
    val_split = split_dict.get("val")
    if val_split is not None:
        val_mask = val_split.obs[data.label_key].astype(str).eq(data.ctrl_label).values
        if bool(np.any(val_mask)):
            stage1_val = val_split[val_mask].copy()
    return stage1_train, stage1_val, {
        "train_n": int(stage1_train.n_obs),
        "val_n": int(stage1_val.n_obs if stage1_val is not None else 0),
    }


def _fit_stage1_and_get_latent(
    *,
    model,
    data: "TriShiftData",
    adata_all: ad.AnnData,
    train_pool: ad.AnnData,
    val_pool: ad.AnnData | None,
    stage1_epochs: int,
    stage1_batch_size: int,
    stage1_lr: float,
    stage1_beta: float,
    stage1_deg_weight: float,
    stage1_deg_key_obs_key: str | None,
    stage1_sched_gamma: float,
    stage1_patience: int,
    stage1_min_delta: float,
    amp: bool,
    num_workers: int,
    pin_memory: bool,
    grad_accum_steps: int,
) -> tuple[dict, np.ndarray]:
    train_logs = model.train_stage1_vae(
        adata_ctrl_pool=train_pool,
        epochs=int(stage1_epochs),
        batch_size=int(stage1_batch_size),
        lr=float(stage1_lr),
        beta=float(stage1_beta),
        deg_weight=float(stage1_deg_weight),
        deg_key_obs_key=stage1_deg_key_obs_key,
        sched_gamma=float(stage1_sched_gamma),
        patience=int(stage1_patience),
        min_delta=float(stage1_min_delta),
        amp=bool(amp),
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        grad_accum_steps=int(grad_accum_steps),
        adata_val=val_pool,
    )
    model.encode_and_cache_mu(
        adata_all,
        batch_size=int(stage1_batch_size),
        amp=bool(amp),
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
    )
    z_mu = np.asarray(data.adata_all.obsm["z_mu"], dtype=np.float32)
    return train_logs, z_mu


def _compute_silhouette(x: np.ndarray, labels: np.ndarray) -> float:
    uniq, counts = np.unique(labels, return_counts=True)
    if uniq.size < 2:
        return float("nan")
    if np.any(counts < 2):
        return float("nan")
    return float(silhouette_score(x, labels))


def _label_metrics(
    *,
    x: np.ndarray,
    clusters: np.ndarray,
    labels: pd.Series,
    label_key: str,
) -> dict:
    label_arr = labels.astype(str).to_numpy()
    uniq_labels = np.unique(label_arr)
    uniq_clusters = np.unique(clusters.astype(str))
    if uniq_labels.size < 2 or uniq_clusters.size < 2:
        ari = float("nan")
        nmi = float("nan")
    else:
        ari = float(adjusted_rand_score(label_arr, clusters))
        nmi = float(normalized_mutual_info_score(label_arr, clusters))
    silhouette_label = _compute_silhouette(x, label_arr)
    avg_bio = float(np.mean([nmi, ari, silhouette_label]))
    return {
        "label_key": str(label_key),
        "n_obs": int(x.shape[0]),
        "n_unique_labels": int(uniq_labels.size),
        "n_unique_clusters": int(uniq_clusters.size),
        # Existing notebook-local names.
        "ari_leiden_vs_label": ari,
        "nmi_leiden_vs_label": nmi,
        "silhouette_label": silhouette_label,
        # scGPT/scIB-style aliases.
        "ARI_cluster/label": ari,
        "NMI_cluster/label": nmi,
        "ASW_label": silhouette_label,
        "avg_bio": avg_bio,
    }


def _fit_cluster_graph(
    latent_adata: ad.AnnData,
    *,
    neighbors_k: int,
    leiden_resolution: float,
    umap_min_dist: float,
    random_seed: int,
    fallback_clusters: int,
) -> None:
    sc.pp.neighbors(
        latent_adata,
        use_rep="X_stage1",
        n_neighbors=max(2, int(neighbors_k)),
        random_state=int(random_seed),
    )
    cluster_method = "leiden"
    try:
        sc.tl.leiden(latent_adata, resolution=float(leiden_resolution), key_added="leiden")
    except ImportError:
        n_clusters = max(2, int(fallback_clusters))
        labels = KMeans(n_clusters=n_clusters, random_state=int(random_seed), n_init=10).fit_predict(
            np.asarray(latent_adata.obsm["X_stage1"], dtype=np.float32)
        )
        latent_adata.obs["leiden"] = pd.Categorical(labels.astype(str))
        cluster_method = "kmeans_fallback"
    sc.tl.umap(latent_adata, min_dist=float(umap_min_dist), random_state=int(random_seed))
    latent_adata.uns["cluster_method"] = cluster_method


def _save_umap(
    latent_adata: ad.AnnData,
    *,
    color: str,
    out_path: Path,
    title: str,
    figsize: tuple[float, float],
    save_dpi: int,
    point_size: float | None,
) -> None:
    plot_kwargs = dict(
        color=[color],
        title=[title],
        frameon=False,
        return_fig=True,
        show=False,
    )
    if point_size is not None:
        plot_kwargs["size"] = float(point_size)
    fig = sc.pl.umap(latent_adata, **plot_kwargs)
    fig.set_size_inches(float(figsize[0]), float(figsize[1]))
    fig.savefig(out_path, dpi=int(save_dpi), bbox_inches="tight")
    plt.close(fig)


def _save_confusion_heatmap(
    *,
    labels: pd.Series,
    clusters: pd.Series,
    out_path: Path,
    title: str,
) -> None:
    table = pd.crosstab(labels.astype(str), clusters.astype(str))
    fig_w = max(6.0, 0.5 * table.shape[1] + 2.0)
    fig_h = max(4.0, 0.4 * table.shape[0] + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(table.to_numpy(dtype=float), aspect="auto", cmap="Blues")
    ax.set_xticks(np.arange(table.shape[1]))
    ax.set_yticks(np.arange(table.shape[0]))
    ax.set_xticklabels(table.columns.astype(str).tolist(), rotation=45, ha="right")
    ax.set_yticklabels(table.index.astype(str).tolist())
    ax.set_xlabel("Leiden cluster")
    ax.set_ylabel("Label")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            val = int(table.iloc[i, j])
            ax.text(j, i, str(val), ha="center", va="center", color="black", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _sanitize_for_h5ad(adata: ad.AnnData) -> ad.AnnData:
    """Return a copy with obs columns coerced to h5ad-safe scalar types."""
    clean = adata.copy()
    drop_cols: list[str] = []
    for col in clean.obs.columns:
        series = clean.obs[col]
        if pd.api.types.is_categorical_dtype(series) or pd.api.types.is_bool_dtype(series):
            continue
        if pd.api.types.is_numeric_dtype(series):
            continue
        if series.dtype == object:
            sample = next((v for v in series.tolist() if v is not None), None)
            if isinstance(sample, (list, tuple, dict, set, np.ndarray)):
                if col == "embd_index":
                    drop_cols.append(col)
                else:
                    clean.obs[col] = series.map(lambda v: "" if v is None else str(v)).astype("string")
            else:
                clean.obs[col] = series.astype("string")
    if drop_cols:
        clean.obs = clean.obs.drop(columns=drop_cols)
    return clean


def _collect_label_keys(
    *,
    mode: str,
    latent_adata: ad.AnnData,
    subgroup_df: pd.DataFrame | None,
) -> list[str]:
    keys: list[str] = []
    if mode == "pbmc_celltype":
        for key in ("label_cell_type", "label_condition", "label_ctrl_pert"):
            if key in latent_adata.obs.columns and latent_adata.obs[key].nunique() > 1:
                keys.append(key)
        return keys

    if "label_condition" in latent_adata.obs.columns and latent_adata.obs["label_condition"].nunique() > 1:
        keys.append("label_condition")
    if "label_ctrl_pert" in latent_adata.obs.columns and latent_adata.obs["label_ctrl_pert"].nunique() > 1:
        keys.append("label_ctrl_pert")
    if subgroup_df is not None:
        keys.append("subgroup")
    return keys


def _infer_fallback_cluster_count(
    *,
    mode: str,
    latent_adata: ad.AnnData,
    subgroup_df: pd.DataFrame | None,
) -> int:
    label_keys = _collect_label_keys(mode=mode, latent_adata=latent_adata, subgroup_df=subgroup_df)
    counts = []
    for key in label_keys:
        if key in latent_adata.obs.columns:
            counts.append(int(latent_adata.obs[key].astype(str).nunique()))
    if counts:
        return max(2, max(counts))
    return 8


def run_stage1_latent_clustering(
    *,
    mode: str,
    dataset_name: str,
    split_id: int = 1,
    stage1_pool_mode: str = "train_all_cells",
    random_seed: int = 24,
    stage1_epochs: int | None = None,
    stage1_z_dim: int | None = None,
    stage1_batch_size: int | None = None,
    stage1_lr: float | None = None,
    stage1_beta: float | None = None,
    stage1_deg_weight: float | None = None,
    pbmc_path: str | Path = "src/data/pbmc/train_pbmc.h5ad",
    pbmc_train_frac: float = 0.9,
    pbmc_deg_mode: str = "by_cell_type",
    l2_normalize_latent: bool = True,
    neighbors_k: int | None = None,
    leiden_resolution: float = 1.0,
    umap_min_dist: float = 0.3,
    reuse_z_mu_cache: bool = True,
    save_dpi: int = 300,
    cluster_figsize: tuple[float, float] = (4.0, 4.0),
    label_figsize: tuple[float, float] = (4.0, 4.0),
    point_size_cluster: float | None = None,
    point_size_label: float | None = None,
    defaults_path: str | Path = "configs/defaults.yaml",
    paths_path: str | Path = "configs/paths.yaml",
) -> Stage1LatentRunResult:
    _set_repo_root()
    mode_key = str(mode).strip().lower()
    if mode_key not in {"trishift_condition", "pbmc_celltype"}:
        raise ValueError("mode must be one of: trishift_condition, pbmc_celltype")
    _get_utils()["set_seeds"](int(random_seed))
    defaults = _load_defaults(defaults_path)
    perf = defaults.get("performance", {})
    train_stage1_cfg = defaults.get("train", {}).get("stage1", {})
    dataset_config = _get_dataset_config() if mode_key == "trishift_condition" else {}

    stage1_epochs_eff = int(
        stage1_epochs
        if stage1_epochs is not None
        else (10 if mode_key == "pbmc_celltype" else dataset_config[dataset_name]["defaults"]["stage1"])
    )
    stage1_z_dim_eff = int(
        stage1_z_dim
        if stage1_z_dim is not None
        else defaults.get("model", {}).get("stage1", {}).get("z_dim", 100)
    )
    stage1_batch_size_eff = int(stage1_batch_size if stage1_batch_size is not None else train_stage1_cfg.get("batch_size", 128))
    stage1_lr_eff = float(stage1_lr if stage1_lr is not None else train_stage1_cfg.get("lr", 1e-3))
    stage1_beta_eff = float(stage1_beta if stage1_beta is not None else train_stage1_cfg.get("beta", 1.0))
    stage1_deg_weight_eff = float(
        stage1_deg_weight
        if stage1_deg_weight is not None
        else train_stage1_cfg.get("deg_weight", 1.0)
    )
    stage1_sched_gamma_eff = float(train_stage1_cfg.get("sched_gamma", 0.9))
    stage1_patience_eff = int(train_stage1_cfg.get("patience", 5))
    stage1_min_delta_eff = float(train_stage1_cfg.get("min_delta", 1e-3))
    pbmc_deg_mode_key = str(pbmc_deg_mode).strip().lower()
    if mode_key == "pbmc_celltype" and pbmc_deg_mode_key != "by_cell_type":
        raise ValueError("pbmc_deg_mode must be 'by_cell_type'")
    stage1_deg_key_obs_key = None
    pbmc_deg_cache_path = None

    if mode_key == "trishift_condition":
        adata, data, embd_df, defaults, _ = _prepare_trishift_data(
            dataset_name,
            stage1_deg_weight=stage1_deg_weight_eff,
            defaults_path=defaults_path,
            paths_path=paths_path,
        )
        adata = data.adata_all
        split_dict, subgroup_df = (
            _norman_split(data, split_id)
            if dataset_name == "norman"
            else (data.split_by_condition(seed=int(split_id), test_ratio=float(dataset_config[dataset_name]["test_ratio"])), None)
        )
        if subgroup_df is not None:
            subgroup_map = subgroup_df["subgroup"].astype(str).to_dict()
            adata.obs["subgroup"] = adata.obs["label_condition"].map(subgroup_map).fillna("unknown")
        train_pool, val_pool, split_meta = _resolve_stage1_train_pool(
            mode=mode_key,
            data=data,
            split_dict=split_dict,
            stage1_pool_mode=stage1_pool_mode,
            adata_all=adata,
            pbmc_train_frac=pbmc_train_frac,
            seed=random_seed + int(split_id),
        )
    else:
        adata, data, embd_df, pbmc_deg_cache_path = _prepare_pbmc_data(
            pbmc_path,
            stage1_deg_weight=stage1_deg_weight_eff,
            pbmc_deg_mode=pbmc_deg_mode_key,
        )
        adata = data.adata_all
        subgroup_df = None
        split_dict = None
        stage1_deg_key_obs_key = "stage1_deg_key"
        train_pool, val_pool, split_meta = _resolve_stage1_train_pool(
            mode=mode_key,
            data=data,
            split_dict=None,
            stage1_pool_mode=stage1_pool_mode,
            adata_all=adata,
            pbmc_train_frac=pbmc_train_frac,
            seed=random_seed + int(split_id),
        )

    out_dir = _make_out_dir(
        mode=mode_key,
        dataset_name=dataset_name,
        split_id=split_id,
        stage1_pool_mode=stage1_pool_mode,
        random_seed=random_seed,
    )
    z_mu_path = out_dir / "z_mu.npz"
    train_logs_path = out_dir / "stage1_train_logs.json"
    latent_h5ad_path = out_dir / "latent_with_clusters.h5ad"
    metrics_csv_path = out_dir / "cluster_metrics.csv"
    metadata_path = out_dir / "run_meta.json"
    cache_identity = _stage1_cache_identity(
        mode=mode_key,
        dataset_name=dataset_name,
        split_id=split_id,
        stage1_pool_mode=stage1_pool_mode,
        random_seed=random_seed,
        stage1_epochs=stage1_epochs_eff,
        stage1_z_dim=stage1_z_dim_eff,
        stage1_batch_size=stage1_batch_size_eff,
        stage1_lr=stage1_lr_eff,
        stage1_beta=stage1_beta_eff,
        stage1_deg_weight=stage1_deg_weight_eff,
        stage1_sched_gamma=stage1_sched_gamma_eff,
        stage1_patience=stage1_patience_eff,
        stage1_min_delta=stage1_min_delta_eff,
        pbmc_path=pbmc_path,
        pbmc_train_frac=pbmc_train_frac,
        pbmc_deg_mode=pbmc_deg_mode_key,
    )

    if (
        reuse_z_mu_cache
        and z_mu_path.exists()
        and _can_reuse_stage1_cache(metadata_path, expected_identity=cache_identity)
    ):
        cached = np.load(z_mu_path, allow_pickle=False)
        z_mu = np.asarray(cached["z_mu"], dtype=np.float32)
        if z_mu.ndim != 2 or z_mu.shape[0] != adata.n_obs:
            raise ValueError(
                f"Cached z_mu shape mismatch at {z_mu_path}: {z_mu.shape} expected ({adata.n_obs}, z_dim)"
            )
        train_logs = {"loaded_from_cache": True, "epochs": []}
        data.set_latent_mu(z_mu, key="z_mu")
    else:
        model, _, _ = _build_model(
            data=data,
            adata=adata,
            embd_df=embd_df,
            defaults=defaults,
            stage1_z_dim=stage1_z_dim_eff,
        )
        train_logs, z_mu = _fit_stage1_and_get_latent(
            model=model,
            data=data,
            adata_all=adata,
            train_pool=train_pool,
            val_pool=val_pool,
            stage1_epochs=stage1_epochs_eff,
            stage1_batch_size=stage1_batch_size_eff,
            stage1_lr=stage1_lr_eff,
            stage1_beta=stage1_beta_eff,
            stage1_deg_weight=stage1_deg_weight_eff,
            stage1_deg_key_obs_key=stage1_deg_key_obs_key,
            stage1_sched_gamma=stage1_sched_gamma_eff,
            stage1_patience=stage1_patience_eff,
            stage1_min_delta=stage1_min_delta_eff,
            amp=bool(perf.get("amp", True)),
            num_workers=int(perf.get("num_workers", 0)),
            pin_memory=bool(perf.get("pin_memory", True)),
            grad_accum_steps=int(perf.get("grad_accum_steps", 1)),
        )
        np.savez_compressed(z_mu_path, z_mu=z_mu)
        _dump_train_logs_json(train_logs_path, train_logs)

    z_mu_used = np.asarray(data.adata_all.obsm["z_mu"], dtype=np.float32)
    if bool(l2_normalize_latent):
        z_mu_used = _l2_normalize_rows(z_mu_used)

    latent_adata = sc.AnnData(X=z_mu_used.copy(), obs=adata.obs.copy())
    latent_adata.obsm["X_stage1"] = z_mu_used.copy()
    latent_adata.uns["stage1_latent_meta"] = {
        "mode": mode_key,
        "dataset_name": dataset_name,
        "split_id": int(split_id),
        "stage1_pool_mode": str(stage1_pool_mode),
        "l2_normalize_latent": bool(l2_normalize_latent),
    }
    fallback_clusters = _infer_fallback_cluster_count(
        mode=mode_key,
        latent_adata=latent_adata,
        subgroup_df=subgroup_df,
    )
    neighbors_k_eff = 15 if neighbors_k is None else int(neighbors_k)
    _fit_cluster_graph(
        latent_adata,
        neighbors_k=neighbors_k_eff,
        leiden_resolution=leiden_resolution,
        umap_min_dist=umap_min_dist,
        random_seed=random_seed,
        fallback_clusters=fallback_clusters,
    )

    label_keys = _collect_label_keys(mode=mode_key, latent_adata=latent_adata, subgroup_df=subgroup_df)
    clusters = latent_adata.obs["leiden"].astype(str).to_numpy()
    metrics_rows = []
    for key in label_keys:
        metrics_rows.append(
            _label_metrics(
                x=latent_adata.obsm["X_stage1"],
                clusters=clusters,
                labels=latent_adata.obs[key],
                label_key=key,
            )
        )
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(metrics_csv_path, index=False)

    figure_paths: dict[str, str] = {}
    cluster_path = out_dir / "umap_by_cluster.png"
    _save_umap(
        latent_adata,
        color="leiden",
        out_path=cluster_path,
        title="Stage1 latent UMAP by Leiden cluster",
        figsize=cluster_figsize,
        save_dpi=save_dpi,
        point_size=point_size_cluster,
    )
    figure_paths["umap_by_cluster"] = str(cluster_path)
    for key in label_keys:
        fig_path = out_dir / f"umap_by_{key}.png"
        _save_umap(
            latent_adata,
            color=key,
            out_path=fig_path,
            title=f"Stage1 latent UMAP by {key}",
            figsize=label_figsize,
            save_dpi=save_dpi,
            point_size=point_size_label,
        )
        figure_paths[f"umap_by_{key}"] = str(fig_path)
        heatmap_path = out_dir / f"cluster_vs_{key}.png"
        _save_confusion_heatmap(
            labels=latent_adata.obs[key],
            clusters=latent_adata.obs["leiden"],
            out_path=heatmap_path,
            title=f"Leiden cluster vs {key}",
        )
        figure_paths[f"cluster_vs_{key}"] = str(heatmap_path)

    latent_adata_to_save = _sanitize_for_h5ad(latent_adata)
    latent_adata_to_save.write_h5ad(latent_h5ad_path)
    metadata = {
        "mode": mode_key,
        "dataset_name": dataset_name,
        "split_id": int(split_id),
        "stage1_pool_mode": str(stage1_pool_mode),
        "random_seed": int(random_seed),
        "stage1_epochs": int(stage1_epochs_eff),
        "stage1_z_dim": int(stage1_z_dim_eff),
        "stage1_batch_size": int(stage1_batch_size_eff),
        "stage1_lr": float(stage1_lr_eff),
        "stage1_beta": float(stage1_beta_eff),
        "stage1_deg_weight": float(stage1_deg_weight_eff),
        "pbmc_path": str(Path(pbmc_path).resolve()) if mode_key == "pbmc_celltype" else "",
        "pbmc_train_frac": float(pbmc_train_frac),
        "pbmc_deg_mode": pbmc_deg_mode_key if mode_key == "pbmc_celltype" else "",
        "pbmc_deg_cache_path": str(pbmc_deg_cache_path) if pbmc_deg_cache_path is not None else "",
        "stage1_sched_gamma": float(stage1_sched_gamma_eff),
        "stage1_patience": int(stage1_patience_eff),
        "stage1_min_delta": float(stage1_min_delta_eff),
        "l2_normalize_latent": bool(l2_normalize_latent),
        "neighbors_k": None if neighbors_k is None else int(neighbors_k),
        "neighbors_k_effective": int(neighbors_k_eff),
        "leiden_resolution": float(leiden_resolution),
        "umap_min_dist": float(umap_min_dist),
        "reuse_z_mu_cache": bool(reuse_z_mu_cache),
        "save_dpi": int(save_dpi),
        "cluster_figsize": [float(cluster_figsize[0]), float(cluster_figsize[1])],
        "label_figsize": [float(label_figsize[0]), float(label_figsize[1])],
        "point_size_cluster": None if point_size_cluster is None else float(point_size_cluster),
        "point_size_label": None if point_size_label is None else float(point_size_label),
        "cluster_method": str(latent_adata.uns.get("cluster_method", "leiden")),
        "fallback_clusters": int(fallback_clusters),
        "label_keys": label_keys,
        "split_meta": split_meta,
    }
    _dump_json(metadata_path, metadata)

    if not train_logs_path.exists():
        _dump_train_logs_json(train_logs_path, train_logs)

    return Stage1LatentRunResult(
        out_dir=out_dir,
        latent_h5ad_path=latent_h5ad_path,
        z_mu_path=z_mu_path,
        metrics_csv_path=metrics_csv_path,
        train_logs_path=train_logs_path,
        figure_paths=figure_paths,
        metrics_df=metrics_df,
        latent_adata=latent_adata,
        train_logs=train_logs,
        metadata=metadata,
    )
