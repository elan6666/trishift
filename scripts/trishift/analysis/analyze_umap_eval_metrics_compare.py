from __future__ import annotations

import argparse
import pickle
import zlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.manifold import trustworthiness as sklearn_trustworthiness
from sklearn.metrics import silhouette_score

try:
    import umap
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing dependency: umap-learn. Install with `pip install umap-learn`.") from exc


REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = REPO_ROOT / "output" / "spreadsheet" / "umap_compare_scouter_trishift"

GLOBAL_METRICS = [
    ("silhouette_kind_umap2d_abs", "lower"),
    ("trustworthiness_umap2d", "higher"),
    ("centroid_dist_umap2d_mean", "lower"),
    ("wasserstein_umap2d_mean_mean", "lower"),
    ("mmd_rbf_umap2d_mean", "lower"),
]

PER_CONDITION_METRICS = [
    ("centroid_dist_umap2d", "lower"),
    ("wasserstein_umap2d_mean", "lower"),
    ("mmd_rbf_umap2d", "lower"),
]


def _equal_sample_pair(pred: np.ndarray, truth: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if pred.ndim != 2 or truth.ndim != 2:
        raise ValueError("pred/truth must be 2D arrays")
    if pred.shape[1] != truth.shape[1]:
        raise ValueError("pred/truth feature dims mismatch")
    n = min(pred.shape[0], truth.shape[0])
    if n <= 0:
        return pred[:0], truth[:0]
    if pred.shape[0] == n and truth.shape[0] == n:
        return pred, truth
    rng = np.random.default_rng(seed)
    pred_idx = rng.choice(pred.shape[0], size=n, replace=False) if pred.shape[0] > n else np.arange(n)
    truth_idx = rng.choice(truth.shape[0], size=n, replace=False) if truth.shape[0] > n else np.arange(n)
    return pred[pred_idx], truth[truth_idx]


def _collect_gene_names(cond: str, item: dict, width: int) -> np.ndarray:
    genes = item.get("DE_name", None)
    if genes is None:
        return np.array([f"{cond}__g{i}" for i in range(width)], dtype=object)
    genes_arr = np.asarray(genes).astype(str, copy=False)
    if genes_arr.size != width:
        return np.array([f"{cond}__g{i}" for i in range(width)], dtype=object)
    return genes_arr


def _build_aggregate_frame(obj: dict, equal_sample: bool = True, seed: int = 24) -> tuple[pd.DataFrame, str]:
    rows: list[pd.DataFrame] = []
    cond_keys = sorted(obj.keys())
    gene_sets: list[set[str]] = []
    cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    for cond in cond_keys:
        item = obj[cond]
        if not isinstance(item, dict):
            continue
        pred = np.asarray(item.get("Pred", np.empty((0, 0), dtype=np.float32)))
        truth = np.asarray(item.get("Truth", np.empty((0, 0), dtype=np.float32)))
        if pred.ndim != 2 or truth.ndim != 2 or pred.shape[1] != truth.shape[1] or pred.shape[1] == 0:
            continue

        if equal_sample:
            cond_seed = seed + (zlib.crc32(cond.encode("utf-8")) % (2**31 - 1))
            pred, truth = _equal_sample_pair(pred, truth, cond_seed)
            if pred.shape[0] == 0:
                continue

        genes = _collect_gene_names(cond, item, pred.shape[1])
        gene_set = set(genes.tolist())
        gene_sets.append(gene_set)
        cache[cond] = (pred, truth, genes)

    if not cache:
        raise ValueError("no valid condition matrices for aggregate UMAP")

    gene_intersection = set.intersection(*gene_sets) if gene_sets else set()
    if len(gene_intersection) >= 2:
        feature_names = sorted(gene_intersection)
        mode = "intersection"
    else:
        feature_names = sorted(set.union(*gene_sets))
        mode = "union_impute"

    feat_idx = {g: i for i, g in enumerate(feature_names)}
    for cond in sorted(cache.keys()):
        pred, truth, genes = cache[cond]
        gene_pos = {g: i for i, g in enumerate(genes.tolist())}
        keep = [g for g in feature_names if g in gene_pos]
        if not keep:
            continue
        src_idx = np.array([gene_pos[g] for g in keep], dtype=int)
        dst_idx = np.array([feat_idx[g] for g in keep], dtype=int)
        truth_mat = np.full((truth.shape[0], len(feature_names)), np.nan, dtype=np.float32)
        pred_mat = np.full((pred.shape[0], len(feature_names)), np.nan, dtype=np.float32)
        truth_mat[:, dst_idx] = truth[:, src_idx]
        pred_mat[:, dst_idx] = pred[:, src_idx]
        rows.append(pd.DataFrame(truth_mat, columns=feature_names).assign(condition=cond, kind="Truth"))
        rows.append(pd.DataFrame(pred_mat, columns=feature_names).assign(condition=cond, kind="Pred"))

    df = pd.concat(rows, ignore_index=True)
    feat_cols = [c for c in df.columns if c not in {"condition", "kind"}]
    if mode == "union_impute":
        feat = df[feat_cols].to_numpy(dtype=np.float32, copy=True)
        col_mean = np.nanmean(feat, axis=0)
        inds = np.where(np.isnan(feat))
        feat[inds] = col_mean[inds[1]]
        df[feat_cols] = feat
    return df, mode


def _fit_umap_embedding(
    df: pd.DataFrame,
    seed: int,
    umap_n_neighbors: int | None = None,
    umap_min_dist: float = 0.15,
) -> tuple[pd.DataFrame, list[str], np.ndarray, dict]:
    feat_cols = [c for c in df.columns if c not in {"condition", "kind"}]
    x = df[feat_cols].to_numpy(dtype=np.float32, copy=False)
    if x.shape[0] < 3 or x.shape[1] < 2:
        raise ValueError("aggregate matrix too small for UMAP")
    if umap_n_neighbors is None:
        n_neighbors = max(2, min(25, x.shape[0] - 1))
    else:
        n_neighbors = max(2, min(int(umap_n_neighbors), x.shape[0] - 1))
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=float(umap_min_dist),
        random_state=seed,
    )
    emb = reducer.fit_transform(x)
    df_plot = df.copy()
    df_plot["umap1"] = emb[:, 0]
    df_plot["umap2"] = emb[:, 1]
    return df_plot, feat_cols, x, {"n_neighbors": int(n_neighbors), "min_dist": float(umap_min_dist)}


def _subsample_rows(
    x: np.ndarray,
    labels_dict: dict[str, np.ndarray],
    max_rows: int | None,
    seed: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    n = int(x.shape[0])
    if max_rows is None or max_rows <= 0 or n <= int(max_rows):
        return x, labels_dict
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(n, size=int(max_rows), replace=False))
    x_sub = x[idx]
    labels_sub = {k: np.asarray(v)[idx] for k, v in labels_dict.items()}
    return x_sub, labels_sub


def _safe_silhouette(x: np.ndarray, labels: np.ndarray, max_rows: int | None, seed: int) -> float:
    labels = np.asarray(labels).astype(str)
    x_use, labs = _subsample_rows(x, {"labels": labels}, max_rows=max_rows, seed=seed)
    labels_use = labs["labels"]
    uniq, counts = np.unique(labels_use, return_counts=True)
    if uniq.size < 2 or np.any(counts < 2):
        return float("nan")
    try:
        return float(silhouette_score(x_use, labels_use))
    except Exception:
        return float("nan")


def _safe_trustworthiness(
    x_high: np.ndarray,
    x_low: np.ndarray,
    max_rows: int | None,
    seed: int,
    n_neighbors: int,
) -> float:
    xh_use, labs = _subsample_rows(x_high, {"idx": np.arange(x_high.shape[0])}, max_rows=max_rows, seed=seed)
    xl_use = x_low[labs["idx"]]
    if xh_use.shape[0] < 3:
        return float("nan")
    n_neighbors = max(1, min(int(n_neighbors), xh_use.shape[0] - 1))
    try:
        return float(sklearn_trustworthiness(xh_use, xl_use, n_neighbors=n_neighbors))
    except Exception:
        return float("nan")


def _sample_rows_fixed(x: np.ndarray, max_rows: int | None, seed: int) -> np.ndarray:
    if max_rows is None or max_rows <= 0 or x.shape[0] <= int(max_rows):
        return x
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(x.shape[0], size=int(max_rows), replace=False))
    return x[idx]


def _median_bandwidth_rbf(x: np.ndarray) -> float:
    if x.shape[0] < 2:
        return 1.0
    diffs = x[:, None, :] - x[None, :, :]
    d2 = np.sum(diffs * diffs, axis=2)
    vals = d2[np.triu_indices_from(d2, k=1)]
    vals = vals[vals > 0]
    if vals.size == 0:
        return 1.0
    med = float(np.median(vals))
    return float(np.sqrt(med)) if med > 0 else 1.0


def _rbf_mmd2_biased(x: np.ndarray, y: np.ndarray, bandwidth: float | None = None) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1] or x.shape[0] == 0 or y.shape[0] == 0:
        return float("nan")
    if bandwidth is None:
        bandwidth = _median_bandwidth_rbf(np.vstack([x, y]))
    bandwidth = max(float(bandwidth), 1e-8)
    gamma = 1.0 / (2.0 * bandwidth * bandwidth)

    def _k(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        d = a[:, None, :] - b[None, :, :]
        d2 = np.sum(d * d, axis=2)
        return np.exp(-gamma * d2)

    val = float(_k(x, x).mean() + _k(y, y).mean() - 2.0 * _k(x, y).mean())
    return max(val, 0.0)


def _compute_per_condition_alignment_metrics(
    df: pd.DataFrame,
    df_plot: pd.DataFrame,
    feat_cols: list[str],
    seed: int,
    per_condition_max_points: int | None = 800,
    mmd_bandwidth: float | None = None,
) -> pd.DataFrame:
    rows: list[dict] = []
    conds = sorted(df_plot["condition"].astype(str).unique().tolist())
    for cond in conds:
        sub_low = df_plot[df_plot["condition"].astype(str) == cond]
        sub_high = df[df["condition"].astype(str) == cond]
        t_low = sub_low[sub_low["kind"].astype(str) == "Truth"][["umap1", "umap2"]].to_numpy(dtype=np.float32, copy=False)
        p_low = sub_low[sub_low["kind"].astype(str) == "Pred"][["umap1", "umap2"]].to_numpy(dtype=np.float32, copy=False)
        t_high = sub_high[sub_high["kind"].astype(str) == "Truth"][feat_cols].to_numpy(dtype=np.float32, copy=False)
        p_high = sub_high[sub_high["kind"].astype(str) == "Pred"][feat_cols].to_numpy(dtype=np.float32, copy=False)
        if t_low.size == 0 or p_low.size == 0 or t_high.size == 0 or p_high.size == 0:
            continue
        cond_seed = int(seed + (zlib.crc32(cond.encode("utf-8")) % (2**31 - 1)))
        t_low_mmd = _sample_rows_fixed(t_low, per_condition_max_points, cond_seed)
        p_low_mmd = _sample_rows_fixed(p_low, per_condition_max_points, cond_seed + 1)
        rows.append(
            {
                "row_type": "per_condition",
                "condition": cond,
                "n_truth": int(t_low.shape[0]),
                "n_pred": int(p_low.shape[0]),
                "centroid_dist_highdim": float(np.linalg.norm(t_high.mean(axis=0) - p_high.mean(axis=0))),
                "centroid_dist_umap2d": float(np.linalg.norm(t_low.mean(axis=0) - p_low.mean(axis=0))),
                "wasserstein_umap2d_dim1": float(wasserstein_distance(t_low[:, 0], p_low[:, 0])),
                "wasserstein_umap2d_dim2": float(wasserstein_distance(t_low[:, 1], p_low[:, 1])),
                "mmd_rbf_umap2d": float(_rbf_mmd2_biased(t_low_mmd, p_low_mmd, bandwidth=mmd_bandwidth)),
            }
        )
    per_cond = pd.DataFrame(rows)
    if len(per_cond) == 0:
        return per_cond
    per_cond["wasserstein_umap2d_mean"] = (
        per_cond["wasserstein_umap2d_dim1"] + per_cond["wasserstein_umap2d_dim2"]
    ) / 2.0
    return per_cond.sort_values("condition").reset_index(drop=True)


def _compute_umap_eval_metrics(
    df: pd.DataFrame,
    df_plot: pd.DataFrame,
    feat_cols: list[str],
    x_high: np.ndarray,
    seed: int,
    umap_meta: dict,
    silhouette_max_rows: int | None = 3000,
    trustworthiness_max_rows: int | None = 3000,
    trustworthiness_n_neighbors: int = 15,
    per_condition_max_points: int | None = 800,
    mmd_bandwidth: float | None = None,
) -> pd.DataFrame:
    labels_condition = df_plot["condition"].astype(str).to_numpy()
    labels_kind = df_plot["kind"].astype(str).to_numpy()
    x_low = df_plot[["umap1", "umap2"]].to_numpy(dtype=np.float32, copy=False)
    per_cond_df = _compute_per_condition_alignment_metrics(
        df, df_plot, feat_cols, seed, per_condition_max_points=per_condition_max_points, mmd_bandwidth=mmd_bandwidth
    )
    summary = {
        "row_type": "global",
        "condition": "",
        "n_rows_total": int(len(df_plot)),
        "n_conditions": int(df_plot["condition"].nunique()),
        "n_truth_total": int((df_plot["kind"].astype(str) == "Truth").sum()),
        "n_pred_total": int((df_plot["kind"].astype(str) == "Pred").sum()),
        "umap_n_neighbors": int(umap_meta["n_neighbors"]),
        "umap_min_dist": float(umap_meta["min_dist"]),
        "silhouette_condition_highdim": _safe_silhouette(x_high, labels_condition, silhouette_max_rows, seed + 11),
        "silhouette_condition_umap2d": _safe_silhouette(x_low, labels_condition, silhouette_max_rows, seed + 12),
        "silhouette_kind_umap2d": _safe_silhouette(x_low, labels_kind, silhouette_max_rows, seed + 13),
        "trustworthiness_umap2d": _safe_trustworthiness(
            x_high, x_low, trustworthiness_max_rows, seed + 21, int(trustworthiness_n_neighbors)
        ),
    }
    for col in [
        "centroid_dist_highdim",
        "centroid_dist_umap2d",
        "wasserstein_umap2d_mean",
        "mmd_rbf_umap2d",
    ]:
        vals = pd.to_numeric(per_cond_df[col], errors="coerce")
        summary[f"{col}_mean"] = float(vals.mean())
        summary[f"{col}_median"] = float(vals.median())
        summary[f"{col}_std"] = float(vals.std()) if len(vals) > 1 else float("nan")
    summary_df = pd.DataFrame([summary])
    common_cols = list(summary_df.columns)
    for col in per_cond_df.columns:
        if col not in common_cols:
            common_cols.append(col)
    return pd.concat(
        [summary_df.reindex(columns=common_cols), per_cond_df.reindex(columns=common_cols)],
        ignore_index=True,
    )


def _load_pickle(path: Path) -> dict:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _resolve_paths(model: str, dataset: str, split_id: int = 1, variant_tag: str = "nearest") -> tuple[Path, Path]:
    if model == "trishift":
        pkl = REPO_ROOT / "artifacts" / "results" / dataset / f"trishift_{dataset}_{split_id}_{variant_tag}.pkl"
        csv = REPO_ROOT / "artifacts" / "results" / dataset / "umap_plots" / f"trishift_{dataset}_{split_id}_{variant_tag}_umap_eval_metrics.csv"
    elif model == "scouter":
        pkl = REPO_ROOT / "artifacts" / "results" / "scouter" / dataset / f"scouter_{dataset}_{split_id}.pkl"
        csv = REPO_ROOT / "artifacts" / "results" / "scouter" / dataset / "umap_plots" / f"scouter_{dataset}_{split_id}_umap_eval_metrics.csv"
    else:
        raise ValueError(model)
    return pkl, csv


def ensure_metrics_csv(model: str, dataset: str, split_id: int = 1, variant_tag: str = "nearest") -> Path:
    pkl_path, csv_path = _resolve_paths(model, dataset, split_id=split_id, variant_tag=variant_tag)
    if csv_path.exists():
        return csv_path
    if not pkl_path.exists():
        raise FileNotFoundError(pkl_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    obj = _load_pickle(pkl_path)
    agg_df, _mode = _build_aggregate_frame(obj, equal_sample=True, seed=24)
    df_plot, feat_cols, x_high, umap_meta = _fit_umap_embedding(agg_df, seed=24, umap_n_neighbors=None, umap_min_dist=0.15)
    export_df = _compute_umap_eval_metrics(
        agg_df,
        df_plot,
        feat_cols,
        x_high,
        seed=24,
        umap_meta=umap_meta,
        silhouette_max_rows=3000,
        trustworthiness_max_rows=3000,
        trustworthiness_n_neighbors=15,
        per_condition_max_points=800,
        mmd_bandwidth=None,
    )
    export_df.to_csv(csv_path, index=False)
    return csv_path


def _load_metrics(csv_path: Path, model: str, dataset: str) -> tuple[pd.Series, pd.DataFrame]:
    df = pd.read_csv(csv_path)
    global_row = df[df["row_type"] == "global"].iloc[0].copy()
    global_row["model"] = model
    global_row["dataset"] = dataset
    per_cond = df[df["row_type"] == "per_condition"].copy()
    per_cond["model"] = model
    per_cond["dataset"] = dataset
    return global_row, per_cond


def _build_outputs(
    global_df: pd.DataFrame,
    per_cond_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    global_df = global_df.copy()
    global_df["silhouette_kind_umap2d_abs"] = global_df["silhouette_kind_umap2d"].abs()
    keep_global = ["dataset", "model"] + [m for m, _ in GLOBAL_METRICS]
    global_keep = global_df[keep_global].copy()

    summary_rows: list[dict] = []
    for dataset in sorted(global_keep["dataset"].unique()):
        sub = global_keep[global_keep["dataset"] == dataset].set_index("model")
        for metric, direction in GLOBAL_METRICS:
            t_val = float(sub.loc["trishift", metric])
            s_val = float(sub.loc["scouter", metric])
            if direction == "higher":
                winner = "trishift" if t_val > s_val else "scouter"
            else:
                winner = "trishift" if t_val < s_val else "scouter"
            summary_rows.append(
                {
                    "dataset": dataset,
                    "metric": metric,
                    "direction": direction,
                    "trishift": t_val,
                    "scouter": s_val,
                    "winner": winner,
                }
            )
    global_compare = pd.DataFrame(summary_rows)

    per_rows: list[dict] = []
    for dataset in sorted(per_cond_df["dataset"].unique()):
        t_sub = per_cond_df[(per_cond_df["dataset"] == dataset) & (per_cond_df["model"] == "trishift")]
        s_sub = per_cond_df[(per_cond_df["dataset"] == dataset) & (per_cond_df["model"] == "scouter")]
        merged = t_sub.merge(s_sub, on="condition", suffixes=("_trishift", "_scouter"))
        for metric, direction in PER_CONDITION_METRICS:
            t_col = f"{metric}_trishift"
            s_col = f"{metric}_scouter"
            if merged.empty:
                continue
            if direction == "lower":
                wins = (merged[t_col] < merged[s_col]).mean()
                mean_delta = (merged[s_col] - merged[t_col]).mean()
            else:
                wins = (merged[t_col] > merged[s_col]).mean()
                mean_delta = (merged[t_col] - merged[s_col]).mean()
            per_rows.append(
                {
                    "dataset": dataset,
                    "metric": metric,
                    "n_shared_conditions": int(len(merged)),
                    "trishift_mean": float(merged[t_col].mean()),
                    "scouter_mean": float(merged[s_col].mean()),
                    "trishift_median": float(merged[t_col].median()),
                    "scouter_median": float(merged[s_col].median()),
                    "trishift_win_rate": float(wins),
                    "mean_margin_in_trishift_direction": float(mean_delta),
                }
            )
    per_compare = pd.DataFrame(per_rows)
    return global_keep, global_compare, per_compare


def _plot_global(global_compare: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), dpi=180, facecolor="white")
    axes = axes.ravel()
    datasets = ["adamson", "norman"]
    colors = {"trishift": "#1f77b4", "scouter": "#ff7f0e"}
    for ax, (metric, direction) in zip(axes, GLOBAL_METRICS):
        sub = global_compare[global_compare["metric"] == metric].set_index("dataset")
        x = np.arange(len(datasets))
        width = 0.35
        t_vals = [sub.loc[d, "trishift"] for d in datasets]
        s_vals = [sub.loc[d, "scouter"] for d in datasets]
        ax.bar(x - width / 2, t_vals, width=width, color=colors["trishift"], label="TriShift")
        ax.bar(x + width / 2, s_vals, width=width, color=colors["scouter"], label="Scouter")
        ax.set_xticks(x)
        ax.set_xticklabels([d.title() for d in datasets])
        arrow = "higher better" if direction == "higher" else "lower better"
        ax.set_title(f"{metric}\n({arrow})", fontsize=10)
        ax.grid(axis="y", alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("UMAP Global Metrics | TriShift vs Scouter", y=0.98, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, bbox_inches="tight", facecolor="white", dpi=300)
    plt.close(fig)


def _plot_per_condition(per_cond_df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), dpi=180, facecolor="white")
    axes = axes.ravel()
    datasets = ["adamson", "norman"]
    colors = {"trishift": "#1f77b4", "scouter": "#ff7f0e"}
    for idx, dataset in enumerate(datasets):
        for jdx, (metric, _direction) in enumerate(PER_CONDITION_METRICS):
            ax = axes[idx * 3 + jdx]
            sub = per_cond_df[per_cond_df["dataset"] == dataset]
            data = [
                pd.to_numeric(sub[sub["model"] == "trishift"][metric], errors="coerce").dropna().to_numpy(),
                pd.to_numeric(sub[sub["model"] == "scouter"][metric], errors="coerce").dropna().to_numpy(),
            ]
            bp = ax.boxplot(data, patch_artist=True, tick_labels=["TriShift", "Scouter"], showfliers=False)
            for patch, color in zip(bp["boxes"], [colors["trishift"], colors["scouter"]]):
                patch.set_facecolor(color)
                patch.set_alpha(0.5)
            ax.set_title(f"{dataset.title()} | {metric}", fontsize=10)
            ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Per-condition UMAP Metrics Distribution", y=0.98, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, bbox_inches="tight", facecolor="white", dpi=300)
    plt.close(fig)


def _to_markdown_table(df: pd.DataFrame, decimals: int = 4) -> str:
    show = df.copy()
    for col in show.columns:
        if pd.api.types.is_float_dtype(show[col]):
            show[col] = show[col].map(lambda x: f"{x:.{decimals}f}")
    return show.to_markdown(index=False)


def _write_report(
    out_path: Path,
    global_keep: pd.DataFrame,
    global_compare: pd.DataFrame,
    per_compare: pd.DataFrame,
    csv_paths: dict[tuple[str, str], Path],
) -> None:
    wins = global_compare.groupby(["dataset", "winner"]).size().unstack(fill_value=0)
    lines = [
        "# TriShift vs Scouter UMAP Comparison",
        "",
        "## Scope",
        "",
        "- Datasets: `adamson`, `norman`",
        "- Models: `TriShift`, `Scouter`",
        "- TriShift inputs are fixed to `nearest` pkl.",
        "- This report excludes every `highdim` metric.",
        "- Metrics are based on split `1` UMAP evaluation CSVs.",
        "",
        "## Source Files",
        "",
    ]
    for (dataset, model), path in sorted(csv_paths.items()):
        lines.append(f"- `{dataset} / {model}`: `{path.as_posix()}`")
    lines.extend(
        [
            "",
            "## Global Metrics",
            "",
            _to_markdown_table(global_keep.sort_values(["dataset", "model"])),
            "",
            "## Global Winner Table",
            "",
            _to_markdown_table(global_compare),
            "",
            "## Per-condition Comparison Summary",
            "",
            _to_markdown_table(per_compare),
            "",
            "## Interpretation",
            "",
        ]
    )
    for dataset in ["adamson", "norman"]:
        ds_global = global_compare[global_compare["dataset"] == dataset]
        ds_per = per_compare[per_compare["dataset"] == dataset]
        trishift_global_wins = int((ds_global["winner"] == "trishift").sum())
        scouter_global_wins = int((ds_global["winner"] == "scouter").sum())
        lines.append(f"### {dataset.title()}")
        lines.append("")
        lines.append(
            f"- Global UMAP metrics: TriShift wins `{trishift_global_wins}` / `{len(ds_global)}` tracked metrics; "
            f"Scouter wins `{scouter_global_wins}`."
        )
        if not ds_per.empty:
            for metric in PER_CONDITION_METRICS:
                metric_name = metric[0]
                row = ds_per[ds_per["metric"] == metric_name].iloc[0]
                lines.append(
                    f"- `{metric_name}`: TriShift mean `{row['trishift_mean']:.4f}` vs Scouter `{row['scouter_mean']:.4f}`, "
                    f"TriShift win-rate over shared conditions `{row['trishift_win_rate']:.2%}`."
                )
        lines.append("")
    lines.extend(
        [
            "## Reading Guide",
            "",
            "- `silhouette_kind_umap2d_abs`: smaller is better; Truth and Pred are less separable in 2D UMAP.",
            "- `trustworthiness_umap2d`: larger is better; the 2D embedding preserves local neighborhoods better.",
            "- `centroid_dist_umap2d_mean`: smaller is better; predicted condition centers are closer to truth.",
            "- `wasserstein_umap2d_mean_mean`: smaller is better; Truth/Pred distributions are closer in 2D UMAP.",
            "- `mmd_rbf_umap2d_mean`: smaller is better; the two 2D distributions are more similar.",
            "",
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def run() -> dict[str, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_paths: dict[tuple[str, str], Path] = {}
    global_rows: list[pd.Series] = []
    per_rows: list[pd.DataFrame] = []

    for dataset in ["adamson", "norman"]:
        tri_csv = ensure_metrics_csv("trishift", dataset, split_id=1, variant_tag="nearest")
        sco_csv = ensure_metrics_csv("scouter", dataset, split_id=1, variant_tag="")
        csv_paths[(dataset, "trishift")] = tri_csv
        csv_paths[(dataset, "scouter")] = sco_csv
        for model, csv_path in [("trishift", tri_csv), ("scouter", sco_csv)]:
            global_row, per_cond = _load_metrics(csv_path, model, dataset)
            global_rows.append(global_row)
            per_rows.append(per_cond)

    global_df = pd.DataFrame(global_rows)
    per_cond_df = pd.concat(per_rows, ignore_index=True)
    global_keep, global_compare, per_compare = _build_outputs(global_df, per_cond_df)

    global_csv = OUTPUT_DIR / "global_metrics_comparison.csv"
    per_csv = OUTPUT_DIR / "per_condition_comparison.csv"
    raw_csv = OUTPUT_DIR / "per_condition_raw_metrics.csv"
    xlsx_path = OUTPUT_DIR / "umap_metrics_comparison.xlsx"
    report_md = OUTPUT_DIR / "umap_compare_scouter_trishift.md"
    global_plot = OUTPUT_DIR / "global_metric_bars.png"
    per_plot = OUTPUT_DIR / "per_condition_boxplots.png"

    global_compare.to_csv(global_csv, index=False)
    per_compare.to_csv(per_csv, index=False)
    per_cond_df.to_csv(raw_csv, index=False)

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        global_keep.sort_values(["dataset", "model"]).to_excel(writer, sheet_name="global_raw", index=False)
        global_compare.to_excel(writer, sheet_name="global_compare", index=False)
        per_compare.to_excel(writer, sheet_name="per_condition_compare", index=False)
        per_cond_df.to_excel(writer, sheet_name="per_condition_raw", index=False)

    _plot_global(global_compare, global_plot)
    _plot_per_condition(per_cond_df, per_plot)
    _write_report(report_md, global_keep, global_compare, per_compare, csv_paths)

    return {
        "global_csv": global_csv,
        "per_csv": per_csv,
        "raw_csv": raw_csv,
        "xlsx": xlsx_path,
        "report_md": report_md,
        "global_plot": global_plot,
        "per_plot": per_plot,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare TriShift vs Scouter UMAP metrics on Adamson and Norman.")
    _ = parser.parse_args()
    outputs = run()
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
