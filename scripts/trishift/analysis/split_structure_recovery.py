from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler

import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts.common.paper_plot_style import CLUSTER_COLORS, apply_gears_paper_style, style_axis
from scripts.trishift.analysis._result_adapter import (
    load_payload_item,
    parse_models,
    parse_split_ids,
    resolve_model_spec,
    write_run_meta,
)
from trishift.TriShiftData import TriShiftData
from trishift._external_metrics import average_of_perturbation_centroids
from trishift._utils import apply_alias_mapping, load_adata, load_embedding_df, load_yaml, normalize_condition


@dataclass
class SplitStructureRecoveryResult:
    out_dir: Path
    summary_df: pd.DataFrame
    figure_path: Path


def _load_split_dataset(dataset: str, split_id: int, paths_path: str | Path) -> tuple[TriShiftData, dict[str, Any]]:
    cfg = load_yaml(str(Path(paths_path).resolve()))
    emb_key_map = {
        "adamson": "emb_b",
        "dixit": "emb_c",
        "norman": "emb_a",
        "replogle_k562_essential": "emb_c",
        "replogle_rpe1_essential": "emb_b",
    }
    adata = load_adata(cfg["datasets"][dataset])
    embd_df = load_embedding_df(cfg["embeddings"][emb_key_map[dataset]])
    embd_df = apply_alias_mapping(embd_df, dataset)
    data = TriShiftData(adata, embd_df)
    split_dict = data.split_by_condition(seed=int(split_id))
    return data, split_dict


def _payload_full_arrays(item: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    needed = ["Pred_full", "Truth_full", "Ctrl_full"]
    missing = [k for k in needed if k not in item]
    if missing:
        raise ValueError(f"structure recovery requires full-gene payload arrays: missing={missing}")
    pred = np.asarray(item["Pred_full"], dtype=np.float32)
    truth = np.asarray(item["Truth_full"], dtype=np.float32)
    ctrl = np.asarray(item["Ctrl_full"], dtype=np.float32)
    if pred.ndim != 2 or truth.ndim != 2 or ctrl.ndim != 2:
        raise ValueError("Pred_full/Truth_full/Ctrl_full must all be 2D")
    return pred, truth, ctrl


def _sample_rows(arr: np.ndarray, max_rows: int, rng: np.random.RandomState) -> np.ndarray:
    if arr.shape[0] <= int(max_rows):
        return np.asarray(arr, dtype=np.float32)
    idx = rng.choice(arr.shape[0], size=int(max_rows), replace=False)
    return np.asarray(arr[idx], dtype=np.float32)


def _truth_condition_centroids(payload: dict[str, Any], conditions: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cond in conditions:
        item = payload.get(cond)
        if not isinstance(item, dict):
            continue
        _pred, truth, _ctrl = _payload_full_arrays(item)
        rows.append({"condition": cond, "centroid": truth.mean(axis=0).astype(np.float32)})
    if not rows:
        raise ValueError("No truth centroids could be constructed")
    return pd.DataFrame(rows)


def _cluster_conditions(centroid_df: pd.DataFrame, cluster_k: int, seed: int) -> dict[str, int]:
    vecs = np.vstack(centroid_df["centroid"].to_list()).astype(np.float32, copy=False)
    n_clusters = max(1, min(int(cluster_k), int(vecs.shape[0])))
    if n_clusters == 1:
        return {str(cond): 0 for cond in centroid_df["condition"].astype(str).tolist()}
    kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=int(seed))
    labels = kmeans.fit_predict(vecs)
    counts = pd.Series(labels).value_counts().sort_values(ascending=False)
    ordered = counts.index.tolist()
    relabel = {int(old): int(new) for new, old in enumerate(ordered)}
    return {
        str(cond): int(relabel[int(label)])
        for cond, label in zip(centroid_df["condition"].astype(str).tolist(), labels.tolist())
    }


def _mean_perturb_delta(train_adata, ctrl_label: str = "ctrl") -> np.ndarray:
    x_train = train_adata.X
    if sp.issparse(x_train):
        x_train = x_train.toarray()
    x_train = np.asarray(x_train, dtype=np.float32)
    cond_train = train_adata.obs["condition"].astype(str).map(normalize_condition).to_numpy()
    ref = average_of_perturbation_centroids(X=x_train, conditions=cond_train, ctrl_label=str(ctrl_label))
    ctrl_mask = cond_train == str(ctrl_label)
    if not bool(ctrl_mask.any()):
        ctrl_mean = x_train.mean(axis=0).astype(np.float32)
    else:
        ctrl_mean = x_train[ctrl_mask].mean(axis=0).astype(np.float32)
    return np.asarray(ref - ctrl_mean, dtype=np.float32).reshape(-1)


def _fit_embedding(x_truth: np.ndarray, seed: int) -> tuple[Any, Any, np.ndarray, dict[str, Any]]:
    scaler = StandardScaler(with_mean=True, with_std=True)
    x_scaled = scaler.fit_transform(x_truth)
    n_components = max(2, min(20, x_scaled.shape[0] - 1, x_scaled.shape[1]))
    pca = PCA(n_components=n_components, random_state=int(seed))
    x_pca = pca.fit_transform(x_scaled)
    meta = {"embed_mode": "pca2d"}
    try:
        import umap  # type: ignore

        n_neighbors = max(5, min(15, x_truth.shape[0] - 1))
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=int(n_neighbors),
            min_dist=0.25,
            random_state=int(seed),
        )
        emb_truth = reducer.fit_transform(x_pca)
        meta = {"embed_mode": "umap", "umap_n_neighbors": int(n_neighbors), "umap_min_dist": 0.25}
        return scaler, pca, reducer, emb_truth, meta
    except Exception:
        return scaler, pca, None, x_pca[:, :2], meta


def _transform_embedding(
    x: np.ndarray,
    scaler: Any,
    pca: Any,
    reducer: Any,
) -> np.ndarray:
    x_scaled = scaler.transform(x)
    x_pca = pca.transform(x_scaled)
    if reducer is None:
        return x_pca[:, :2]
    return np.asarray(reducer.transform(x_pca), dtype=np.float32)


def _row_cluster_metrics(emb: np.ndarray, labels_true: np.ndarray, cluster_k: int, seed: int) -> tuple[float, float]:
    n_clusters = max(1, min(int(cluster_k), int(np.unique(labels_true).size), int(emb.shape[0])))
    if n_clusters <= 1:
        return float("nan"), float("nan")
    pred_labels = KMeans(n_clusters=n_clusters, n_init=20, random_state=int(seed)).fit_predict(emb)
    return (
        float(adjusted_rand_score(labels_true, pred_labels)),
        float(normalized_mutual_info_score(labels_true, pred_labels)),
    )


def _build_split_rows(
    *,
    dataset: str,
    split_id: int,
    conditions: list[str],
    model_payloads: dict[str, dict[str, Any]],
    train_adata,
    row_models: list[str],
    max_cells_per_condition: int,
    cluster_k: int,
    seed: int,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    ref_model = row_models[0]
    centroid_df = _truth_condition_centroids(model_payloads[ref_model], conditions)
    cluster_map = _cluster_conditions(centroid_df, cluster_k=cluster_k, seed=seed + int(split_id))
    mean_delta = _mean_perturb_delta(train_adata)
    rng = np.random.RandomState(int(seed) + int(split_id))

    row_frames: dict[str, pd.DataFrame] = {}
    truth_rows: list[dict[str, Any]] = []
    model_rows: dict[str, list[dict[str, Any]]] = {model_name: [] for model_name in row_models}
    baseline_rows: list[dict[str, Any]] = []

    for cond in conditions:
        cluster_id = int(cluster_map[str(cond)])
        ref_item = model_payloads[ref_model][str(cond)]
        _ref_pred, ref_truth, ref_ctrl = _payload_full_arrays(ref_item)
        truth_sample = _sample_rows(ref_truth, max_cells_per_condition, rng)
        ctrl_sample = _sample_rows(ref_ctrl, max_cells_per_condition, rng)

        for row in truth_sample:
            truth_rows.append({"condition": cond, "cluster_id": cluster_id, "vector": row})
        for row in ctrl_sample + mean_delta.reshape(1, -1):
            baseline_rows.append({"condition": cond, "cluster_id": cluster_id, "vector": row})

        for model_name in row_models:
            pred, _truth, _ctrl = _payload_full_arrays(model_payloads[model_name][str(cond)])
            pred_sample = _sample_rows(pred, max_cells_per_condition, rng)
            for row in pred_sample:
                model_rows[model_name].append({"condition": cond, "cluster_id": cluster_id, "vector": row})

    truth_df = pd.DataFrame(truth_rows)
    if truth_df.empty:
        raise ValueError(f"No truth rows available for dataset={dataset} split={split_id}")
    x_truth = np.vstack(truth_df["vector"].to_list()).astype(np.float32, copy=False)
    scaler, pca, reducer, truth_emb, embed_meta = _fit_embedding(x_truth, seed=seed + int(split_id))
    truth_plot = truth_df.drop(columns=["vector"]).copy()
    truth_plot["x"] = truth_emb[:, 0]
    truth_plot["y"] = truth_emb[:, 1]
    row_frames["Experimental data"] = truth_plot

    summary_rows = [
        {
            "dataset": dataset,
            "split_id": int(split_id),
            "row_name": "Experimental data",
            "ari": np.nan,
            "nmi": np.nan,
            **embed_meta,
        }
    ]

    for model_name in row_models:
        display_name = resolve_model_spec(model_name).label
        df = pd.DataFrame(model_rows[model_name])
        x = np.vstack(df["vector"].to_list()).astype(np.float32, copy=False)
        emb = _transform_embedding(x, scaler=scaler, pca=pca, reducer=reducer)
        plot_df = df.drop(columns=["vector"]).copy()
        plot_df["x"] = emb[:, 0]
        plot_df["y"] = emb[:, 1]
        row_frames[display_name] = plot_df
        labels_true = plot_df["cluster_id"].to_numpy(dtype=int, copy=False)
        ari, nmi = _row_cluster_metrics(emb, labels_true, cluster_k=cluster_k, seed=seed + int(split_id))
        summary_rows.append(
            {
                "dataset": dataset,
                "split_id": int(split_id),
                "row_name": display_name,
                "ari": ari,
                "nmi": nmi,
                **embed_meta,
            }
        )

    baseline_df = pd.DataFrame(baseline_rows)
    x_base = np.vstack(baseline_df["vector"].to_list()).astype(np.float32, copy=False)
    emb_base = _transform_embedding(x_base, scaler=scaler, pca=pca, reducer=reducer)
    baseline_plot = baseline_df.drop(columns=["vector"]).copy()
    baseline_plot["x"] = emb_base[:, 0]
    baseline_plot["y"] = emb_base[:, 1]
    row_frames["Mean perturbation"] = baseline_plot
    labels_true = baseline_plot["cluster_id"].to_numpy(dtype=int, copy=False)
    ari, nmi = _row_cluster_metrics(emb_base, labels_true, cluster_k=cluster_k, seed=seed + 100 + int(split_id))
    summary_rows.append(
        {
            "dataset": dataset,
            "split_id": int(split_id),
            "row_name": "Mean perturbation",
            "ari": ari,
            "nmi": nmi,
            **embed_meta,
        }
    )
    return row_frames, pd.DataFrame(summary_rows)


def _plot_panel(
    *,
    dataset_label: str,
    split_ids: list[int],
    row_order: list[str],
    split_row_frames: dict[int, dict[str, pd.DataFrame]],
    summary_df: pd.DataFrame,
    out_path: Path,
) -> None:
    apply_gears_paper_style(font_scale=1.05)
    fig, axes = plt.subplots(len(row_order), len(split_ids), figsize=(2.55 * len(split_ids), 2.15 * len(row_order)), dpi=220)
    axes_arr = np.asarray(axes, dtype=object).reshape(len(row_order), len(split_ids))
    for col_idx, split_id in enumerate(split_ids):
        split_rows = split_row_frames.get(int(split_id), {})
        for row_idx, row_name in enumerate(row_order):
            ax = axes_arr[row_idx, col_idx]
            plot_df = split_rows.get(row_name, pd.DataFrame())
            if plot_df.empty:
                ax.text(0.5, 0.5, "No rows", ha="center", va="center")
                ax.axis("off")
                continue
            for cluster_id in sorted(plot_df["cluster_id"].astype(int).unique().tolist()):
                sub = plot_df[plot_df["cluster_id"].astype(int) == int(cluster_id)]
                ax.scatter(
                    sub["x"].to_numpy(dtype=float),
                    sub["y"].to_numpy(dtype=float),
                    s=8,
                    alpha=0.9,
                    color=CLUSTER_COLORS[int(cluster_id) % len(CLUSTER_COLORS)],
                    edgecolors="none",
                )
            style_axis(ax)
            ax.set_xticks([])
            ax.set_yticks([])
            if row_idx == 0:
                ax.set_title(f"Split {split_id}")
            if col_idx == 0:
                ax.set_ylabel(row_name)
            else:
                ax.set_ylabel("")
            if row_idx == len(row_order) - 1:
                ax.set_xlabel("UMAP 1")
            else:
                ax.set_xlabel("")
            if row_name != "Experimental data":
                metric_row = summary_df[
                    (summary_df["split_id"].astype(int) == int(split_id))
                    & (summary_df["row_name"].astype(str) == str(row_name))
                ]
                if not metric_row.empty:
                    ari = float(metric_row["ari"].iloc[0])
                    nmi = float(metric_row["nmi"].iloc[0])
                    if np.isfinite(ari) and np.isfinite(nmi):
                        ax.text(
                            0.03,
                            0.97,
                            f"ARI: {ari:.2f}\nNMI: {nmi:.2f}",
                            ha="left",
                            va="top",
                            transform=ax.transAxes,
                            fontsize=7.5,
                            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.25},
                        )
    fig.suptitle(f"{dataset_label}: split-wise global structure recovery", y=1.01, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def run_split_structure_recovery(
    *,
    dataset: str,
    models: str | list[str] | tuple[str, ...] = ("trishift_nearest", "gears"),
    split_ids: int | str | list[int] | tuple[int, ...] = (1, 2, 3, 4, 5),
    dataset_label: str | None = None,
    out_root: str | Path | None = None,
    paths_path: str | Path = "configs/paths.yaml",
    max_cells_per_condition: int = 28,
    cluster_k: int = 4,
    seed: int = 24,
) -> SplitStructureRecoveryResult:
    dataset_key = str(dataset).strip()
    model_names = parse_models(models)
    if not model_names:
        raise ValueError("models cannot be empty")
    split_list = parse_split_ids(split_ids)
    out_dir = Path(out_root).resolve() if out_root else (REPO_ROOT / "artifacts" / "analysis" / f"structure_recovery_{dataset_key}").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    split_row_frames: dict[int, dict[str, pd.DataFrame]] = {}
    summary_frames: list[pd.DataFrame] = []
    row_order = ["Experimental data"] + [resolve_model_spec(m).label for m in model_names] + ["Mean perturbation"]

    for split_id in split_list:
        _data, split_dict = _load_split_dataset(dataset_key, int(split_id), paths_path=paths_path)
        test_conds = [normalize_condition(str(c)) for c in split_dict.get("test_conds", [])]
        train_adata = split_dict.get("train")
        if train_adata is None:
            continue

        model_payloads: dict[str, dict[str, Any]] = {}
        available_cond_sets: list[set[str]] = []
        for model_name in model_names:
            try:
                _pkl_path, payload = load_payload_item(
                    dataset=dataset_key,
                    model_name=model_name,
                    split_id=int(split_id),
                    condition=None,
                )
            except Exception:
                payload = {}
            normalized = {
                normalize_condition(str(cond)): item
                for cond, item in payload.items()
                if isinstance(item, dict) and all(k in item for k in ["Pred_full", "Truth_full", "Ctrl_full"])
            }
            if not normalized:
                break
            model_payloads[model_name] = normalized
            available_cond_sets.append(set(normalized))
        if len(model_payloads) != len(model_names):
            continue

        conditions = [cond for cond in test_conds if all(cond in cond_set for cond_set in available_cond_sets)]
        if not conditions:
            continue
        row_frames, split_summary = _build_split_rows(
            dataset=dataset_key,
            split_id=int(split_id),
            conditions=conditions,
            model_payloads=model_payloads,
            train_adata=train_adata,
            row_models=model_names,
            max_cells_per_condition=max_cells_per_condition,
            cluster_k=cluster_k,
            seed=seed,
        )
        split_row_frames[int(split_id)] = row_frames
        summary_frames.append(split_summary)

    summary_df = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
    summary_csv = out_dir / f"figs4e_structure_recovery_{dataset_key}_metrics.csv"
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    figure_path = out_dir / f"figs4e_structure_recovery_{dataset_key}.png"
    if split_row_frames:
        _plot_panel(
            dataset_label=str(dataset_label or dataset_key.title()),
            split_ids=[int(x) for x in split_list if int(x) in split_row_frames],
            row_order=row_order,
            split_row_frames=split_row_frames,
            summary_df=summary_df,
            out_path=figure_path,
        )
    meta = {
        "dataset": dataset_key,
        "dataset_label": str(dataset_label or dataset_key.title()),
        "models": model_names,
        "split_ids": split_list,
        "max_cells_per_condition": int(max_cells_per_condition),
        "cluster_k": int(cluster_k),
        "seed": int(seed),
        "summary_csv": str(summary_csv),
        "figure_path": str(figure_path),
    }
    write_run_meta(out_dir / "figs4e_structure_recovery_run_meta.json", meta)
    return SplitStructureRecoveryResult(out_dir=out_dir, summary_df=summary_df, figure_path=figure_path)
