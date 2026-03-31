from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scripts.common.paper_plot_style import apply_gears_paper_style, style_axis

try:
    import umap
except ImportError as exc:  # pragma: no cover - dependency check
    raise SystemExit("Missing dependency: umap-learn. Install with `pip install umap-learn`.") from exc


DEFAULT_RESULT_ROOTS = {
    "trishift": REPO_ROOT / "artifacts" / "results",
    "gears": REPO_ROOT / "artifacts" / "results" / "gears",
    "genepert": REPO_ROOT / "artifacts" / "results" / "genepert",
    "scouter": REPO_ROOT / "artifacts" / "results" / "scouter",
    "scgpt": REPO_ROOT / "artifacts" / "results" / "scgpt",
}


@dataclass
class ConditionCentroidVisResult:
    out_dir: Path
    points_raw_df: pd.DataFrame
    points_df: pd.DataFrame
    metrics_df: pd.DataFrame
    summary_df: pd.DataFrame
    figure_paths: dict[str, str]


def _ts_local() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _normalize_model_name(model_name: str) -> str:
    key = str(model_name).strip().lower()
    if key not in DEFAULT_RESULT_ROOTS:
        raise ValueError(f"Unsupported model_name={model_name}")
    return key


def _result_root(model_name: str, dataset: str, result_dir: str | Path | None) -> Path:
    if result_dir is not None and str(result_dir).strip():
        return Path(result_dir).resolve()
    return (DEFAULT_RESULT_ROOTS[_normalize_model_name(model_name)] / dataset).resolve()


def _pkl_path(
    *,
    model_name: str,
    dataset: str,
    split_id: int,
    result_root: Path,
    variant_tag: str | None = None,
) -> Path:
    model_key = _normalize_model_name(model_name)
    if model_key in {"trishift", "scouter"}:
        suffix = f"_{variant_tag}" if variant_tag else ""
        return result_root / f"{model_key}_{dataset}_{split_id}{suffix}.pkl"
    return result_root / f"{model_key}_{dataset}_{split_id}.pkl"


def _load_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected dict payload in {path}")
    return obj


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 0.0 or nb <= 0.0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def _feature_arrays(
    *,
    payload: dict[str, Any],
    feature_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray, str]:
    mode_key = str(feature_mode).strip().lower()
    if mode_key not in {"deg", "full"}:
        raise ValueError("feature_mode must be one of: deg, full")

    if mode_key == "full":
        missing = [k for k in ("Pred_full", "Truth_full", "Ctrl_full", "gene_name_full") if k not in payload]
        if missing:
            raise ValueError(
                f"feature_mode='full' requires {missing}, but the payload only contains keys: {sorted(payload.keys())}"
            )
        pred = np.asarray(payload["Pred_full"], dtype=np.float32)
        truth = np.asarray(payload["Truth_full"], dtype=np.float32)
        ctrl = np.asarray(payload["Ctrl_full"], dtype=np.float32) if "Ctrl_full" in payload else None
        genes = np.asarray(payload["gene_name_full"]).astype(str)
        used = "full"
    else:
        pred = np.asarray(payload["Pred"], dtype=np.float32)
        truth = np.asarray(payload["Truth"], dtype=np.float32)
        ctrl = np.asarray(payload["Ctrl"], dtype=np.float32) if "Ctrl" in payload else None
        de_name = payload.get("DE_name")
        if de_name is None:
            genes = np.asarray([f"deg_{i}" for i in range(pred.shape[1])], dtype=object)
        else:
            genes = np.asarray(de_name).astype(str)
        used = "deg"

    if pred.ndim != 2 or truth.ndim != 2:
        raise ValueError("Pred and Truth must be 2D arrays")
    if pred.shape[1] != truth.shape[1]:
        raise ValueError("Pred and Truth feature dimensions must match")
    if ctrl is not None:
        if ctrl.ndim != 2:
            raise ValueError("Ctrl must be a 2D array")
        if ctrl.shape[1] != pred.shape[1]:
            raise ValueError("Ctrl feature dimension must match Pred/Truth")
    if genes.shape[0] != pred.shape[1]:
        raise ValueError("Gene name length must match the feature dimension")
    return pred, truth, ctrl, genes, used


def _global_gene_space(payload: dict[str, Any], feature_mode: str) -> np.ndarray | None:
    mode_key = str(feature_mode).strip().lower()
    if mode_key != "deg":
        return None
    names: list[str] = []
    for condition in sorted(payload.keys()):
        item = payload[condition]
        if not isinstance(item, dict):
            continue
        _pred, _truth, _ctrl, genes, _mode_used = _feature_arrays(payload=item, feature_mode=feature_mode)
        names.extend([str(g) for g in genes.tolist()])
    if not names:
        return None
    return np.asarray(sorted(set(names)), dtype=object)


def _align_vector_to_gene_space(
    vector: np.ndarray,
    genes: np.ndarray,
    global_genes: np.ndarray | None,
) -> np.ndarray:
    vec = np.asarray(vector, dtype=np.float32).reshape(-1)
    local_genes = np.asarray(genes).astype(str).reshape(-1)
    if global_genes is None:
        return vec
    global_names = np.asarray(global_genes).astype(str).reshape(-1)
    if local_genes.shape[0] != vec.shape[0]:
        raise ValueError("Local gene names must match the vector dimension")
    if global_names.shape[0] == local_genes.shape[0] and np.array_equal(global_names, local_genes):
        return vec
    aligned = np.zeros(global_names.shape[0], dtype=np.float32)
    index_map = {str(g): i for i, g in enumerate(global_names.tolist())}
    for i, gene in enumerate(local_genes.tolist()):
        j = index_map.get(str(gene))
        if j is not None:
            aligned[j] = float(vec[i])
    return aligned


def _centroid_rows(
    *,
    condition: str,
    split_id: int,
    pred: np.ndarray,
    truth: np.ndarray,
    ctrl: np.ndarray | None,
    genes: np.ndarray,
    global_genes: np.ndarray | None,
    feature_mode_used: str,
    feature_dim: int,
    include_ctrl: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    truth_centroid = truth.mean(axis=0).astype(np.float32, copy=False)
    pred_centroid = pred.mean(axis=0).astype(np.float32, copy=False)
    ctrl_centroid = None if ctrl is None or ctrl.shape[0] == 0 else ctrl.mean(axis=0).astype(np.float32, copy=False)
    truth_plot = _align_vector_to_gene_space(truth_centroid, genes, global_genes)
    pred_plot = _align_vector_to_gene_space(pred_centroid, genes, global_genes)
    ctrl_plot = None if ctrl_centroid is None else _align_vector_to_gene_space(ctrl_centroid, genes, global_genes)
    feature_names = np.asarray(global_genes if global_genes is not None else genes).astype(str)

    point_rows = [
        {
            "split_id": int(split_id),
            "condition": str(condition),
            "kind": "Truth",
            "feature_mode_used": str(feature_mode_used),
            "feature_dim": int(feature_dim),
            "n_truth": int(truth.shape[0]),
            "n_pred": int(pred.shape[0]),
            "n_ctrl": (0 if ctrl is None else int(ctrl.shape[0])),
            "vector": truth_plot,
            "feature_names": feature_names,
        },
        {
            "split_id": int(split_id),
            "condition": str(condition),
            "kind": "Pred",
            "feature_mode_used": str(feature_mode_used),
            "feature_dim": int(feature_dim),
            "n_truth": int(truth.shape[0]),
            "n_pred": int(pred.shape[0]),
            "n_ctrl": (0 if ctrl is None else int(ctrl.shape[0])),
            "vector": pred_plot,
            "feature_names": feature_names,
        },
    ]
    if include_ctrl and ctrl_plot is not None:
        point_rows.append(
            {
                "split_id": int(split_id),
                "condition": str(condition),
                "kind": "Ctrl",
                "feature_mode_used": str(feature_mode_used),
                "feature_dim": int(feature_dim),
                "n_truth": int(truth.shape[0]),
                "n_pred": int(pred.shape[0]),
                "n_ctrl": int(ctrl.shape[0]),
                "vector": ctrl_plot,
                "feature_names": feature_names,
            }
        )

    metric_row: dict[str, Any] = {
        "split_id": int(split_id),
        "condition": str(condition),
        "feature_mode_used": str(feature_mode_used),
        "feature_dim": int(feature_dim),
        "n_truth": int(truth.shape[0]),
        "n_pred": int(pred.shape[0]),
        "n_ctrl": (0 if ctrl is None else int(ctrl.shape[0])),
        "centroid_dist": float(np.linalg.norm(truth_centroid - pred_centroid)),
        "centroid_cosine": _cosine(truth_centroid, pred_centroid),
    }
    if ctrl_centroid is None:
        metric_row["truth_ctrl_shift_norm"] = float("nan")
        metric_row["pred_ctrl_shift_norm"] = float("nan")
        metric_row["delta_centroid_dist"] = float("nan")
        metric_row["delta_cosine"] = float("nan")
    else:
        truth_delta = truth_centroid - ctrl_centroid
        pred_delta = pred_centroid - ctrl_centroid
        metric_row["truth_ctrl_shift_norm"] = float(np.linalg.norm(truth_delta))
        metric_row["pred_ctrl_shift_norm"] = float(np.linalg.norm(pred_delta))
        metric_row["delta_centroid_dist"] = float(np.linalg.norm(truth_delta - pred_delta))
        metric_row["delta_cosine"] = _cosine(truth_delta, pred_delta)
    return point_rows, metric_row


def _build_points_and_metrics(
    *,
    payload: dict[str, Any],
    split_id: int,
    feature_mode: str,
    include_ctrl: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    point_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    global_genes = _global_gene_space(payload, feature_mode)
    for condition in sorted(payload.keys()):
        item = payload[condition]
        if not isinstance(item, dict):
            continue
        pred, truth, ctrl, genes, mode_used = _feature_arrays(payload=item, feature_mode=feature_mode)
        rows, metrics = _centroid_rows(
            condition=condition,
            split_id=split_id,
            pred=pred,
            truth=truth,
            ctrl=ctrl,
            genes=genes,
            global_genes=global_genes,
            feature_mode_used=mode_used,
            feature_dim=int(pred.shape[1]),
            include_ctrl=include_ctrl,
        )
        point_rows.extend(rows)
        metric_rows.append(metrics)
    if not point_rows or not metric_rows:
        raise ValueError("No valid condition centroid rows were constructed")
    points_df = pd.DataFrame(point_rows)
    metrics_df = pd.DataFrame(metric_rows).sort_values(["split_id", "condition"]).reset_index(drop=True)
    return points_df, metrics_df


def _fit_umap(points_df: pd.DataFrame, seed: int, umap_n_neighbors: int | None, umap_min_dist: float) -> tuple[pd.DataFrame, dict[str, Any]]:
    vecs = np.vstack(points_df["vector"].to_list()).astype(np.float32, copy=False)
    if vecs.ndim != 2 or vecs.shape[0] < 2:
        raise ValueError("Need at least two centroid points to fit an embedding")
    if vecs.shape[0] < 3 or vecs.shape[1] < 2:
        emb = np.zeros((vecs.shape[0], 2), dtype=np.float32)
        emb[:, 0] = vecs[:, 0]
        if vecs.shape[1] > 1:
            emb[:, 1] = vecs[:, 1]
        meta = {"n_neighbors": None, "min_dist": float(umap_min_dist), "embed_mode": "identity_2d"}
    else:
        if umap_n_neighbors is None:
            n_neighbors = max(2, min(15, vecs.shape[0] - 1))
        else:
            n_neighbors = max(2, min(int(umap_n_neighbors), vecs.shape[0] - 1))
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=int(n_neighbors),
            min_dist=float(umap_min_dist),
            random_state=int(seed),
        )
        emb = reducer.fit_transform(vecs)
        meta = {"n_neighbors": int(n_neighbors), "min_dist": float(umap_min_dist), "embed_mode": "umap"}
    out = points_df.drop(columns=["vector", "feature_names"], errors="ignore").copy()
    out["umap1"] = emb[:, 0]
    out["umap2"] = emb[:, 1]
    return out, meta


def _build_delta_points(metrics_df: pd.DataFrame, points_df: pd.DataFrame) -> pd.DataFrame:
    vec_map = {
        (int(row.split_id), str(row.condition), str(row.kind)): row.vector
        for row in points_df.itertuples(index=False)
    }
    name_map = {
        (int(row.split_id), str(row.condition), str(row.kind)): np.asarray(row.feature_names).astype(str)
        for row in points_df.itertuples(index=False)
        if hasattr(row, "feature_names")
    }
    rows: list[dict[str, Any]] = []
    for row in metrics_df.itertuples(index=False):
        key_truth = (int(row.split_id), str(row.condition), "Truth")
        key_pred = (int(row.split_id), str(row.condition), "Pred")
        key_ctrl = (int(row.split_id), str(row.condition), "Ctrl")
        if key_ctrl not in vec_map:
            continue
        feature_names = name_map.get(key_truth, name_map.get(key_ctrl))
        truth_delta = np.asarray(vec_map[key_truth], dtype=np.float32) - np.asarray(vec_map[key_ctrl], dtype=np.float32)
        pred_delta = np.asarray(vec_map[key_pred], dtype=np.float32) - np.asarray(vec_map[key_ctrl], dtype=np.float32)
        rows.append(
            {
                "split_id": int(row.split_id),
                "condition": str(row.condition),
                "kind": "TruthDelta",
                "vector": truth_delta,
                "feature_names": feature_names,
            }
        )
        rows.append(
            {
                "split_id": int(row.split_id),
                "condition": str(row.condition),
                "kind": "PredDelta",
                "vector": pred_delta,
                "feature_names": feature_names,
            }
        )
    return pd.DataFrame(rows)


def _summary_df(metrics_df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        "centroid_dist",
        "delta_centroid_dist",
        "centroid_cosine",
        "delta_cosine",
        "truth_ctrl_shift_norm",
        "pred_ctrl_shift_norm",
    ]
    row: dict[str, Any] = {
        "n_conditions": int(len(metrics_df)),
        "n_splits": int(metrics_df["split_id"].nunique()),
    }
    for col in numeric_cols:
        vals = pd.to_numeric(metrics_df[col], errors="coerce")
        row[f"{col}_mean"] = float(vals.mean()) if vals.notna().any() else float("nan")
        row[f"{col}_median"] = float(vals.median()) if vals.notna().any() else float("nan")
        row[f"{col}_std"] = float(vals.std()) if vals.notna().sum() > 1 else float("nan")
    return pd.DataFrame([row])


def _condition_colors(conditions: list[str]) -> dict[str, Any]:
    cmap = plt.colormaps.get_cmap("tab20").resampled(max(len(conditions), 1))
    return {cond: cmap(i) for i, cond in enumerate(conditions)}


def _plot_truth_pred(
    *,
    points_df: pd.DataFrame,
    out_path: Path,
    title: str,
    save_dpi: int,
) -> None:
    apply_gears_paper_style(font_scale=1.0)
    conds = sorted(points_df["condition"].astype(str).unique().tolist())
    colors = _condition_colors(conds)
    fig, ax = plt.subplots(figsize=(12, 9), dpi=220, facecolor="white")
    kind_style = {
        "Truth": dict(marker="o", size=70, alpha=0.92),
        "Pred": dict(marker="x", size=95, alpha=0.95),
    }
    for cond in conds:
        sub = points_df[points_df["condition"].astype(str) == cond]
        t = sub[sub["kind"].astype(str) == "Truth"]
        p = sub[sub["kind"].astype(str) == "Pred"]
        if len(t) == 1 and len(p) == 1:
            ax.plot(
                [float(t["umap1"].iloc[0]), float(p["umap1"].iloc[0])],
                [float(t["umap2"].iloc[0]), float(p["umap2"].iloc[0])],
                color=colors[cond],
                alpha=0.45,
                linewidth=1.1,
            )
        for kind, style in kind_style.items():
            cur = sub[sub["kind"].astype(str) == kind]
            if cur.empty:
                continue
            ax.scatter(
                cur["umap1"],
                cur["umap2"],
                s=style["size"],
                alpha=style["alpha"],
                marker=style["marker"],
                c=[colors[cond]],
                linewidths=1.0 if kind == "Pred" else 0.0,
                edgecolors="none" if kind == "Truth" else None,
            )
    ax.set_title(title)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    style_axis(ax, grid_axis="both")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=int(save_dpi), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_truth_pred_ctrl(
    *,
    points_df: pd.DataFrame,
    out_path: Path,
    title: str,
    save_dpi: int,
) -> None:
    apply_gears_paper_style(font_scale=1.0)
    conds = sorted(points_df["condition"].astype(str).unique().tolist())
    colors = _condition_colors(conds)
    fig, ax = plt.subplots(figsize=(12, 9), dpi=220, facecolor="white")
    kind_style = {
        "Truth": dict(marker="o", size=70, alpha=0.88),
        "Pred": dict(marker="x", size=95, alpha=0.95),
        "Ctrl": dict(marker="^", size=78, alpha=0.8),
    }
    for kind, style in kind_style.items():
        sub = points_df[points_df["kind"].astype(str) == kind]
        if sub.empty:
            continue
        ax.scatter(
            sub["umap1"],
            sub["umap2"],
            s=style["size"],
            alpha=style["alpha"],
            marker=style["marker"],
            c=[colors[c] for c in sub["condition"].astype(str).tolist()],
            linewidths=1.0 if kind == "Pred" else 0.0,
            edgecolors="none" if kind in {"Truth", "Ctrl"} else None,
        )
    ax.set_title(title)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    style_axis(ax, grid_axis="both")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=int(save_dpi), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_delta(
    *,
    delta_plot_df: pd.DataFrame,
    out_path: Path,
    title: str,
    save_dpi: int,
) -> None:
    apply_gears_paper_style(font_scale=1.0)
    conds = sorted(delta_plot_df["condition"].astype(str).unique().tolist())
    colors = _condition_colors(conds)
    fig, ax = plt.subplots(figsize=(12, 9), dpi=220, facecolor="white")
    for cond in conds:
        sub = delta_plot_df[delta_plot_df["condition"].astype(str) == cond]
        t = sub[sub["kind"].astype(str) == "TruthDelta"]
        p = sub[sub["kind"].astype(str) == "PredDelta"]
        if len(t) == 1 and len(p) == 1:
            ax.plot(
                [float(t["umap1"].iloc[0]), float(p["umap1"].iloc[0])],
                [float(t["umap2"].iloc[0]), float(p["umap2"].iloc[0])],
                color=colors[cond],
                alpha=0.45,
                linewidth=1.1,
            )
        if not t.empty:
            ax.scatter(t["umap1"], t["umap2"], s=70, alpha=0.9, marker="o", c=[colors[cond]], edgecolors="none")
        if not p.empty:
            ax.scatter(p["umap1"], p["umap2"], s=95, alpha=0.95, marker="x", c=[colors[cond]], linewidths=1.0)
    ax.set_title(title)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    style_axis(ax, grid_axis="both")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=int(save_dpi), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def run_condition_centroid_visualization(
    *,
    model_name: str,
    dataset: str,
    split_id: int = 1,
    result_dir: str | Path | None = None,
    out_root: str | Path | None = None,
    variant_tag: str | None = None,
    feature_mode: str = "deg",
    include_ctrl: bool = True,
    plot_delta: bool = True,
    umap_n_neighbors: int | None = None,
    umap_min_dist: float = 0.15,
    save_dpi: int = 420,
    seed: int = 24,
) -> ConditionCentroidVisResult:
    model_key = _normalize_model_name(model_name)
    root = _result_root(model_key, dataset, result_dir)
    variant_clean = str(variant_tag).strip() or None
    if model_key not in {"trishift", "scouter"}:
        variant_clean = None
    pkl_path = _pkl_path(
        model_name=model_key,
        dataset=str(dataset),
        split_id=int(split_id),
        result_root=root,
        variant_tag=variant_clean,
    )
    payload = _load_payload(pkl_path)

    points_df_raw, metrics_df = _build_points_and_metrics(
        payload=payload,
        split_id=int(split_id),
        feature_mode=feature_mode,
        include_ctrl=bool(include_ctrl),
    )
    points_plot_df, umap_meta = _fit_umap(
        points_df_raw,
        seed=int(seed),
        umap_n_neighbors=umap_n_neighbors,
        umap_min_dist=float(umap_min_dist),
    )
    summary_df = _summary_df(metrics_df)

    if out_root is None or not str(out_root).strip():
        out_dir = root / "condition_centroid_plots"
    else:
        out_dir = Path(out_root).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    figure_paths: dict[str, str] = {}
    p1 = out_dir / "truth_vs_pred_centroid_umap.png"
    _plot_truth_pred(
        points_df=points_plot_df,
        out_path=p1,
        title=f"{model_key}:{dataset} split{split_id} | Truth vs Pred centroid UMAP",
        save_dpi=int(save_dpi),
    )
    figure_paths["truth_vs_pred_centroid_umap"] = str(p1)

    if include_ctrl and (points_plot_df["kind"].astype(str) == "Ctrl").any():
        p2 = out_dir / "truth_pred_ctrl_centroid_umap.png"
        _plot_truth_pred_ctrl(
            points_df=points_plot_df,
            out_path=p2,
            title=f"{model_key}:{dataset} split{split_id} | Truth / Pred / Ctrl centroid UMAP",
            save_dpi=int(save_dpi),
        )
        figure_paths["truth_pred_ctrl_centroid_umap"] = str(p2)

    if plot_delta:
        delta_points = _build_delta_points(metrics_df=metrics_df, points_df=points_df_raw)
        if not delta_points.empty:
            delta_plot_df, _ = _fit_umap(
                delta_points,
                seed=int(seed) + 17,
                umap_n_neighbors=umap_n_neighbors,
                umap_min_dist=float(umap_min_dist),
            )
            p3 = out_dir / "delta_centroid_umap.png"
            _plot_delta(
                delta_plot_df=delta_plot_df,
                out_path=p3,
                title=f"{model_key}:{dataset} split{split_id} | Delta centroid UMAP",
                save_dpi=int(save_dpi),
            )
            figure_paths["delta_centroid_umap"] = str(p3)

    points_export = points_plot_df.copy()
    points_export.to_csv(out_dir / "condition_centroid_points.csv", index=False)
    metrics_df.to_csv(out_dir / "condition_centroid_metrics.csv", index=False)
    summary_export = summary_df.copy()
    summary_export["model_name"] = model_key
    summary_export["dataset"] = str(dataset)
    summary_export["split_id"] = int(split_id)
    summary_export["feature_mode"] = str(feature_mode).strip().lower()
    summary_export["variant_tag"] = str(variant_clean or "")
    summary_export["include_ctrl"] = bool(include_ctrl)
    summary_export["plot_delta"] = bool(plot_delta)
    summary_export["umap_n_neighbors"] = ("" if umap_meta.get("n_neighbors") is None else umap_meta["n_neighbors"])
    summary_export["umap_min_dist"] = float(umap_meta["min_dist"])
    summary_export["embed_mode"] = str(umap_meta.get("embed_mode", "umap"))
    summary_export.to_csv(out_dir / "condition_centroid_summary.csv", index=False)

    run_meta = {
        "model_name": model_key,
        "dataset": str(dataset),
        "split_id": int(split_id),
        "variant_tag": str(variant_clean or ""),
        "feature_mode": str(feature_mode).strip().lower(),
        "include_ctrl": bool(include_ctrl),
        "plot_delta": bool(plot_delta),
        "umap_n_neighbors": umap_meta.get("n_neighbors"),
        "umap_min_dist": float(umap_meta["min_dist"]),
        "embed_mode": str(umap_meta.get("embed_mode", "umap")),
        "pkl_path": str(pkl_path),
        "out_dir": str(out_dir),
    }
    (out_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")
    return ConditionCentroidVisResult(
        out_dir=out_dir,
        points_raw_df=points_df_raw,
        points_df=points_plot_df,
        metrics_df=metrics_df,
        summary_df=summary_export,
        figure_paths=figure_paths,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Condition centroid visualization for Trishift-family result PKLs")
    ap.add_argument("--model_name", required=True, choices=sorted(DEFAULT_RESULT_ROOTS))
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split_id", type=int, default=1)
    ap.add_argument("--result_dir", default="")
    ap.add_argument("--out_root", default="")
    ap.add_argument("--variant_tag", default="nearest")
    ap.add_argument("--feature_mode", choices=["deg", "full"], default="deg")
    ap.add_argument("--no_include_ctrl", action="store_true")
    ap.add_argument("--no_plot_delta", action="store_true")
    ap.add_argument("--umap_n_neighbors", type=int, default=None)
    ap.add_argument("--umap_min_dist", type=float, default=0.15)
    ap.add_argument("--save_dpi", type=int, default=420)
    ap.add_argument("--seed", type=int, default=24)
    args = ap.parse_args(argv)

    result = run_condition_centroid_visualization(
        model_name=args.model_name,
        dataset=args.dataset,
        split_id=int(args.split_id),
        result_dir=(str(args.result_dir).strip() or None),
        out_root=(str(args.out_root).strip() or None),
        variant_tag=(str(args.variant_tag).strip() or None),
        feature_mode=str(args.feature_mode),
        include_ctrl=not bool(args.no_include_ctrl),
        plot_delta=not bool(args.no_plot_delta),
        umap_n_neighbors=args.umap_n_neighbors,
        umap_min_dist=float(args.umap_min_dist),
        save_dpi=int(args.save_dpi),
        seed=int(args.seed),
    )
    print(f"out_dir: {result.out_dir}")
    print(result.summary_df.to_string(index=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
