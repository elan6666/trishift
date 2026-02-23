from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import pearsonr
try:
    from zoneinfo import ZoneInfo  # py3.9+
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore


# Make `import trishift.*` work when running as a script.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from trishift._utils import (  # noqa: E402
    apply_alias_mapping,
    load_adata,
    load_embedding_df,
    load_yaml,
)
from trishift.TriShiftData import TriShiftData  # noqa: E402


DATASET_CONFIG = {
    # Keep in sync with scripts/run_dataset.py
    "adamson": {"emb_key": "emb_b"},
    "dixit": {"emb_key": "emb_c"},
    "norman": {"emb_key": "emb_a"},
    "replogle_k562_essential": {"emb_key": "emb_c"},
    "replogle_rpe1_essential": {"emb_key": "emb_b"},
}


@dataclass(frozen=True)
class RunInfo:
    run_dir: Path
    run_name: str
    label: str
    dataset: str


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

def _cn_timestamp() -> str:
    """Timestamp in China local time (Asia/Shanghai, UTC+8)."""
    if ZoneInfo is not None:
        try:
            return datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
        except Exception:
            pass
    # Fallback: fixed offset (no DST needed for China).
    return datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S")


def _is_run_dir(p: Path) -> bool:
    return p.is_dir() and p.name[:2].isdigit() and "_" in p.name


def _discover_runs(sweep_dir: Path) -> list[RunInfo]:
    runs: list[RunInfo] = []
    for child in sorted(sweep_dir.iterdir(), key=lambda x: x.name):
        if not _is_run_dir(child):
            continue
        meta_path = child / "run_meta.json"
        label = child.name
        dataset = ""
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                label = str(meta.get("label") or label)
                dataset = str(meta.get("dataset") or dataset)
            except Exception:
                pass
        runs.append(RunInfo(run_dir=child, run_name=child.name, label=label, dataset=dataset))
    return runs


def _discover_splits(run_dir: Path, dataset: str) -> list[int]:
    splits: list[int] = []
    for pkl in sorted(run_dir.glob(f"trishift_{dataset}_*.pkl")):
        # expected: trishift_<dataset>_<split_id>.pkl
        stem = pkl.stem
        parts = stem.split("_")
        if not parts:
            continue
        try:
            split_id = int(parts[-1])
        except ValueError:
            continue
        splits.append(split_id)
    return sorted(set(splits))


def _load_predictions_pkl(path: Path) -> dict:
    with path.open("rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict):
        raise TypeError(f"unexpected pkl payload: {path} type={type(obj)}")
    return obj


def _mean_dense(X) -> np.ndarray:
    if sp.issparse(X):
        out = np.asarray(X.mean(axis=0), dtype=np.float32).reshape(-1)
        return out
    return np.asarray(X, dtype=np.float32).mean(axis=0).reshape(-1)


def _cond_centroid_full(adata, label_key: str, cond: str) -> np.ndarray:
    mask = adata.obs[label_key].astype(str).values == str(cond)
    if not np.any(mask):
        raise KeyError(f"condition not found in adata: {cond}")
    X = adata[mask].X
    return _mean_dense(X)


def _build_o_pert_vectors(
    adata,
    label_key: str,
    train_conds: list[str],
    test_conds: list[str],
    *,
    ctrl_label: str = "ctrl",
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Return (o_pert_train, o_pert_test, centroid_cache) on full gene space."""
    train_pert = [c for c in train_conds if c != ctrl_label]
    test_pert = [c for c in test_conds if c != ctrl_label]
    needed = sorted(set(train_pert + test_pert))
    centroids: dict[str, np.ndarray] = {}
    for cond in needed:
        centroids[cond] = _cond_centroid_full(adata, label_key, cond)

    if len(train_pert) == 0:
        raise ValueError("train perturbation set is empty; cannot build o_pert_train")
    if len(test_pert) == 0:
        raise ValueError("test perturbation set is empty; cannot build o_pert_test")

    o_train = np.mean([centroids[c] for c in train_pert], axis=0).astype(np.float32, copy=False)
    o_test = np.mean([centroids[c] for c in test_pert], axis=0).astype(np.float32, copy=False)
    return o_train, o_test, centroids


def _pearson_safe(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.size == 0 or b.size == 0:
        return float("nan")
    # pearsonr returns nan if std==0; keep that behavior explicit.
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(pearsonr(a, b)[0])


def _compute_systema_pearson_de20(
    pred_obj: dict,
    o_pert_train: np.ndarray,
    o_pert_test: np.ndarray,
) -> list[dict]:
    rows: list[dict] = []
    for cond, payload in pred_obj.items():
        if not isinstance(payload, dict):
            continue
        if "Pred" not in payload or "Truth" not in payload or "DE_idx" not in payload:
            continue
        pred = np.asarray(payload["Pred"], dtype=np.float32)
        truth = np.asarray(payload["Truth"], dtype=np.float32)
        de_idx = np.asarray(payload["DE_idx"], dtype=int).reshape(-1)
        if pred.ndim != 2 or truth.ndim != 2:
            raise ValueError(
                f"unexpected Pred/Truth dims for {cond}: pred={pred.shape}, truth={truth.shape}"
            )
        if pred.shape[1] != truth.shape[1]:
            raise ValueError(
                f"mismatched Pred/Truth feature dims for {cond}: pred={pred.shape}, truth={truth.shape}"
            )
        if de_idx.shape[0] != pred.shape[1]:
            raise ValueError(
                f"DE_idx length mismatch for {cond}: de_idx={de_idx.shape}, pred={pred.shape}"
            )
        if pred.shape[1] == 0:
            continue

        true_mean = truth.mean(axis=0)
        pred_mean = pred.mean(axis=0)
        ref_train = o_pert_train[de_idx]
        ref_test = o_pert_test[de_idx]
        p_train = _pearson_safe(true_mean - ref_train, pred_mean - ref_train)
        p_test = _pearson_safe(true_mean - ref_test, pred_mean - ref_test)
        rows.append(
            {
                "condition": str(cond),
                "pearson_ref_train": p_train,
                "pearson_ref_test": p_test,
                "delta_test_minus_train": (p_test - p_train) if np.isfinite(p_test) and np.isfinite(p_train) else np.nan,
            }
        )
    return rows


def _plot_umap(
    points: np.ndarray,
    labels: list[str],
    *,
    out_path: Path,
    pca_dim: int = 50,
    n_neighbors: int = 15,
    min_dist: float = 0.3,
    random_state: int = 0,
) -> None:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    import umap
    import matplotlib.pyplot as plt

    X = np.asarray(points, dtype=np.float32)
    n_obs, n_feat = X.shape
    if n_obs < 3:
        raise ValueError("UMAP requires at least 3 points (test perts + 2 refs)")

    scaler = StandardScaler(with_mean=True, with_std=True)
    Xs = scaler.fit_transform(X)

    n_comp = int(min(pca_dim, max(n_obs - 1, 1), n_feat))
    Xp = PCA(n_components=n_comp, random_state=random_state).fit_transform(Xs)

    reducer = umap.UMAP(
        n_neighbors=int(min(n_neighbors, max(n_obs - 1, 2))),
        min_dist=float(min_dist),
        random_state=int(random_state),
        metric="euclidean",
    )
    emb = reducer.fit_transform(Xp)

    ref_train_idx = labels.index("O_pert_train")
    ref_test_idx = labels.index("O_pert_test")
    test_idx = [i for i, l in enumerate(labels) if l == "test_true"]

    fig = plt.figure(figsize=(7.5, 6.5), dpi=160)
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(
        emb[test_idx, 0],
        emb[test_idx, 1],
        s=18,
        c="#9aa0a6",
        alpha=0.75,
        linewidths=0,
        label="test true centroids",
    )
    ax.scatter(
        emb[ref_train_idx, 0],
        emb[ref_train_idx, 1],
        s=220,
        c="#d93025",
        marker="*",
        edgecolors="k",
        linewidths=0.5,
        label="O_pert(train)",
    )
    ax.scatter(
        emb[ref_test_idx, 0],
        emb[ref_test_idx, 1],
        s=220,
        c="#1a73e8",
        marker="*",
        edgecolors="k",
        linewidths=0.5,
        label="O_pert(test)",
    )
    ax.set_title("UMAP: test true centroids + perturbed centroid references")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(frameon=True, fontsize=9, loc="best")
    ax.grid(False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _write_design_doc(
    *,
    out_dir: Path,
    sweep_dir: Path,
    dataset: str,
    splits: list[int],
    timestamp: str,
) -> Path:
    # Keep the implementation available, but do not call it by default.
    # The user prefers a one-time, manually maintained design doc in docs/.
    doc_path = REPO_ROOT / "docs" / f"systema_ref_compare_{dataset}_{timestamp}.md"
    rel_out = (
        out_dir.relative_to(REPO_ROOT)
        if out_dir.is_absolute()
        and out_dir.parts[: len(REPO_ROOT.parts)] == REPO_ROOT.parts
        else out_dir
    )
    content = f"""# Systema Reference 对比（train vs test）：{dataset}

生成时间（北京时间/Asia-Shanghai）：`{timestamp}`

本文件由脚本可选生成（`--write-doc`），默认不生成。推荐使用手工维护的中文说明文档。

输出目录：`{rel_out}`
"""
    doc_path.write_text(content, encoding="utf-8")
    return doc_path


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare Systema Pearson under train vs test perturbed centroid references.")
    ap.add_argument("--sweep-dir", required=True, help="Sweep root directory containing run subfolders.")
    ap.add_argument("--dataset", default=None, help="Dataset name (default: infer from run_meta.json).")
    ap.add_argument("--splits", default=None, help="Comma-separated split ids (default: infer from pkl files).")
    ap.add_argument("--out-root", default=None, help="Output root directory (default: <sweep_dir>/_systema_ref_compare).")
    ap.add_argument("--timestamp", default=None, help="Timestamp in China local time (default: now).")
    ap.add_argument("--write-doc", action="store_true", help="Write a docs/ markdown file for this run (off by default).")
    ap.add_argument("--no-umap", action="store_true", help="Disable UMAP generation.")
    ap.add_argument("--umap-pca-dim", type=int, default=50)
    ap.add_argument("--umap-n-neighbors", type=int, default=15)
    ap.add_argument("--umap-min-dist", type=float, default=0.3)
    ap.add_argument("--umap-random-state", type=int, default=0)
    args = ap.parse_args()

    sweep_dir = Path(args.sweep_dir).resolve()
    if not sweep_dir.exists():
        raise FileNotFoundError(f"sweep_dir not found: {sweep_dir}")

    runs = _discover_runs(sweep_dir)
    if not runs:
        raise ValueError(f"no run directories found under: {sweep_dir}")

    dataset = args.dataset
    if dataset is None:
        # Infer from first run with non-empty dataset.
        for r in runs:
            if r.dataset:
                dataset = r.dataset
                break
    if not dataset:
        raise ValueError("dataset cannot be inferred; pass --dataset")
    dataset = str(dataset)

    if dataset not in DATASET_CONFIG:
        raise ValueError(f"unsupported dataset: {dataset} (known: {sorted(DATASET_CONFIG)})")

    # Determine splits by scanning first run.
    if args.splits:
        splits = [int(s.strip()) for s in str(args.splits).split(",") if s.strip()]
    else:
        splits = _discover_splits(runs[0].run_dir, dataset)
    if not splits:
        raise ValueError("no splits discovered; pass --splits explicitly")

    timestamp = str(args.timestamp) if args.timestamp else _cn_timestamp()
    out_root = Path(args.out_root).resolve() if args.out_root else (sweep_dir / "_systema_ref_compare").resolve()
    out_dir = out_root / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[ref-compare] sweep_dir={sweep_dir}")
    print(f"[ref-compare] dataset={dataset}, splits={splits}")
    print(f"[ref-compare] out_dir={out_dir}")

    # Load dataset + embedding, then build TriShiftData for split reproduction.
    paths = load_yaml(str(REPO_ROOT / "configs" / "paths.yaml"))
    h5ad_path = Path(paths["datasets"][dataset])
    emb_key = DATASET_CONFIG[dataset]["emb_key"]
    emb_path = Path(paths["embeddings"][emb_key])
    print(f"[ref-compare] load adata: {h5ad_path}")
    adata = load_adata(str(h5ad_path))
    print(f"[ref-compare] load embedding: key={emb_key} path={emb_path}")
    embd_df = load_embedding_df(str(emb_path))
    embd_df = apply_alias_mapping(embd_df, dataset_name=dataset)

    data = TriShiftData(adata=adata, embd_df=embd_df, label_key="condition", var_gene_key="gene_name")
    data.setup_embedding_index(key_name="embd_index")

    gene_names = (
        data.adata_all.var[data.var_gene_key].astype(str).values
        if data.var_gene_key in data.adata_all.var.columns
        else data.adata_all.var_names.astype(str).values
    )
    label_key = data.label_key

    # Build references per split + optional UMAP (once per split).
    split_refs: dict[int, tuple[np.ndarray, np.ndarray, list[str]]] = {}
    for split_id in splits:
        split_dict = data.split_by_condition(seed=int(split_id))
        train_conds = [str(c) for c in split_dict["train_conds"]]
        test_conds = [str(c) for c in split_dict["test_conds"]]

        o_train, o_test, centroid_cache = _build_o_pert_vectors(
            data.adata_all,
            label_key=label_key,
            train_conds=train_conds,
            test_conds=test_conds,
            ctrl_label=data.ctrl_label,
        )
        split_refs[int(split_id)] = (o_train, o_test, test_conds)

        np.savez_compressed(
            out_dir / f"o_pert_vectors_split{split_id}.npz",
            o_pert_train=o_train.astype(np.float32, copy=False),
            o_pert_test=o_test.astype(np.float32, copy=False),
            gene_names=np.asarray(gene_names, dtype=object),
            train_conds=np.asarray(train_conds, dtype=object),
            test_conds=np.asarray(test_conds, dtype=object),
        )

        if not args.no_umap:
            test_centroids = []
            for cond in test_conds:
                if cond in centroid_cache:
                    test_centroids.append(centroid_cache[cond])
                else:
                    test_centroids.append(_cond_centroid_full(data.adata_all, label_key, cond))
            pts = np.vstack([np.stack(test_centroids, axis=0), o_train.reshape(1, -1), o_test.reshape(1, -1)])
            lbls = ["test_true"] * len(test_centroids) + ["O_pert_train", "O_pert_test"]
            _plot_umap(
                pts,
                lbls,
                out_path=out_dir / f"umap_split{split_id}_test_true_plus_refs.png",
                pca_dim=int(args.umap_pca_dim),
                n_neighbors=int(args.umap_n_neighbors),
                min_dist=float(args.umap_min_dist),
                random_state=int(args.umap_random_state),
            )

    # Compute per-run per-split Pearson on DE20 using the split references.
    long_rows: list[dict] = []
    for r in runs:
        for split_id in splits:
            pkl_path = r.run_dir / f"trishift_{dataset}_{split_id}.pkl"
            if not pkl_path.exists():
                print(f"[warn] missing pkl: {pkl_path}")
                continue
            pred_obj = _load_predictions_pkl(pkl_path)

            # Consistency check: pkl conditions vs reproduced test_conds.
            _, _, test_conds = split_refs[int(split_id)]
            pkl_conds = set(map(str, pred_obj.keys()))
            test_set = set(map(str, test_conds))
            overlap = len(pkl_conds & test_set)
            if overlap < max(1, int(0.8 * len(pkl_conds))):
                print(
                    f"[warn] low overlap between pkl conditions and split test_conds: "
                    f"run={r.run_name} split={split_id} overlap={overlap}/{len(pkl_conds)}"
                )

            o_train, o_test, _ = split_refs[int(split_id)]
            rows = _compute_systema_pearson_de20(pred_obj, o_train, o_test)
            for row in rows:
                long_rows.append(
                    {
                        "run": r.run_name,
                        "label": r.label,
                        "split_id": int(split_id),
                        **row,
                    }
                )

    df_long = pd.DataFrame(long_rows)
    long_path = out_dir / "systema_ref_compare_de20_long.csv"
    df_long.to_csv(long_path, index=False)
    print(f"[ref-compare] wrote: {long_path}")

    if df_long.empty:
        raise ValueError("no rows computed; check pkl paths and dataset/splits")

    def _nanmean(x):
        x = pd.to_numeric(x, errors="coerce")
        return float(np.nanmean(x.values)) if len(x) else float("nan")

    def _nanstd(x):
        x = pd.to_numeric(x, errors="coerce")
        return float(np.nanstd(x.values)) if len(x) else float("nan")

    df_summary = (
        df_long.groupby(["run", "label", "split_id"], dropna=False)
        .agg(
            n_conditions=("condition", "count"),
            n_nan_train=("pearson_ref_train", lambda s: int(pd.isna(s).sum())),
            n_nan_test=("pearson_ref_test", lambda s: int(pd.isna(s).sum())),
            mean_ref_train=("pearson_ref_train", _nanmean),
            std_ref_train=("pearson_ref_train", _nanstd),
            mean_ref_test=("pearson_ref_test", _nanmean),
            std_ref_test=("pearson_ref_test", _nanstd),
            mean_delta=("delta_test_minus_train", _nanmean),
            std_delta=("delta_test_minus_train", _nanstd),
        )
        .reset_index()
    )
    summary_path = out_dir / "systema_ref_compare_de20_summary.csv"
    df_summary.to_csv(summary_path, index=False)
    print(f"[ref-compare] wrote: {summary_path}")

    # Over-splits aggregation (equal-weight over splits) + a row-weighted reference table.
    def _nanmean_series(x: pd.Series) -> float:
        x = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
        return float(np.nanmean(x)) if x.size else float("nan")

    def _nanstd_series(x: pd.Series) -> float:
        x = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
        return float(np.nanstd(x)) if x.size else float("nan")

    agg_rows = []
    for (run, label), g in df_summary.groupby(["run", "label"], dropna=False):
        g = g.sort_values("split_id")
        agg_rows.append(
            {
                "run": run,
                "label": label,
                "splits_present": int(g["split_id"].nunique()),
                "split_ids": ",".join(map(str, g["split_id"].tolist())),
                "total_conditions": int(g["n_conditions"].sum()),
                "total_nan_train": int(g["n_nan_train"].sum()),
                "total_nan_test": int(g["n_nan_test"].sum()),
                "mean_ref_train_over_splits": _nanmean_series(g["mean_ref_train"]),
                "mean_ref_test_over_splits": _nanmean_series(g["mean_ref_test"]),
                "mean_delta_over_splits": _nanmean_series(g["mean_delta"]),
                "std_split_mean_ref_train": _nanstd_series(g["mean_ref_train"]),
                "std_split_mean_ref_test": _nanstd_series(g["mean_ref_test"]),
                "std_split_mean_delta": _nanstd_series(g["mean_delta"]),
                "avg_within_split_std_ref_train": _nanmean_series(g["std_ref_train"]),
                "avg_within_split_std_ref_test": _nanmean_series(g["std_ref_test"]),
                "avg_within_split_std_delta": _nanmean_series(g["std_delta"]),
            }
        )
    df_over_splits = pd.DataFrame(agg_rows).sort_values(["run"])
    over_splits_path = out_dir / "systema_ref_compare_de20_over_splits.csv"
    df_over_splits.to_csv(over_splits_path, index=False)
    print(f"[ref-compare] wrote: {over_splits_path}")

    rows2 = []
    for (run, label), g in df_long.groupby(["run", "label"], dropna=False):
        rows2.append(
            {
                "run": run,
                "label": label,
                "n_rows": int(len(g)),
                "mean_ref_train_all_rows": float(
                    np.nanmean(g["pearson_ref_train"].to_numpy(dtype=float))
                ),
                "mean_ref_test_all_rows": float(
                    np.nanmean(g["pearson_ref_test"].to_numpy(dtype=float))
                ),
                "mean_delta_all_rows": float(
                    np.nanmean(g["delta_test_minus_train"].to_numpy(dtype=float))
                ),
            }
        )
    df_weighted = pd.DataFrame(rows2).sort_values(["run"])
    weighted_path = out_dir / "systema_ref_compare_de20_over_splits_weighted_by_rows.csv"
    df_weighted.to_csv(weighted_path, index=False)
    print(f"[ref-compare] wrote: {weighted_path}")

    if bool(args.write_doc):
        doc_path = _write_design_doc(
            out_dir=out_dir,
            sweep_dir=sweep_dir,
            dataset=dataset,
            splits=splits,
            timestamp=timestamp,
        )
        print(f"[ref-compare] wrote: {doc_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
