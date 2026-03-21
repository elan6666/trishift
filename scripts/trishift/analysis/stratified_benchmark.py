from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts.common.split_utils import norman_subgroup
from scripts.trishift.analysis._result_adapter import (
    DATASET_EMBEDDING_KEYS,
    condition_embedding,
    load_metrics_df,
    load_payload_item,
    parse_models,
    parse_split_ids,
    resolve_model_spec,
    resolve_result,
    ts_local,
    warn_skip,
    write_run_meta,
)
from trishift.TriShiftData import TriShiftData
from trishift._utils import apply_alias_mapping, load_adata, load_embedding_df, load_yaml, normalize_condition


SUMMARY_METRICS = [
    "pearson",
    "nmse",
    "deg_mean_r2",
    "systema_corr_20de_allpert",
    "systema_corr_deg_r2",
    "scpram_r2_degs_var_mean",
]

LOWER_IS_BETTER = {"nmse"}


def _qcut_labels(series: pd.Series, labels: list[str]) -> tuple[pd.Series, list[float]]:
    s = pd.to_numeric(series, errors="coerce")
    valid = s.dropna()
    if valid.empty:
        return pd.Series([np.nan] * len(series), index=series.index, dtype=object), []
    try:
        bins = pd.qcut(valid.rank(method="first"), q=len(labels), labels=labels, duplicates="drop")
        q = list(np.quantile(valid.to_numpy(), np.linspace(0.0, 1.0, len(labels) + 1)))
    except Exception:
        bins = pd.Series([labels[len(labels) // 2]] * len(valid), index=valid.index, dtype=object)
        q = [float(valid.min()), float(valid.max())]
    out = pd.Series([np.nan] * len(series), index=series.index, dtype=object)
    out.loc[valid.index] = bins.astype(str)
    return out, q


def _load_dataset_split(data_name: str, split_id: int, paths_path: str | Path) -> dict[str, Any]:
    cfg = load_yaml(str(Path(paths_path).resolve()))
    adata = load_adata(cfg["datasets"][data_name])
    emb_key = DATASET_EMBEDDING_KEYS[data_name]
    embd_df = load_embedding_df(cfg["embeddings"][emb_key])
    embd_df = apply_alias_mapping(embd_df, data_name)
    data = TriShiftData(adata, embd_df)
    return data.split_by_condition(seed=int(split_id))


def _build_metadata_from_payload(dataset: str, payload: dict[str, Any], split_id: int, train_conds: list[str]) -> pd.DataFrame:
    subgroup_lookup: dict[str, str] = {}
    conds = [normalize_condition(str(k)) for k in payload.keys()]
    if dataset == "norman":
        plist = ["ctrl"] + conds if "ctrl" not in conds else conds
        subgroup_df = norman_subgroup(plist, seed=int(split_id))
        subgroup_lookup = {normalize_condition(str(idx)): str(row["subgroup"]) for idx, row in subgroup_df.iterrows()}

    train_embs = {
        normalize_condition(c): condition_embedding(dataset, c)
        for c in train_conds
    }
    rows: list[dict[str, Any]] = []
    for condition, item in payload.items():
        cond_key = normalize_condition(str(condition))
        truth_full = np.asarray(item["Truth_full"] if "Truth_full" in item else item["Truth"], dtype=np.float32)
        ctrl_full = np.asarray(item["Ctrl_full"] if "Ctrl_full" in item else item["Ctrl"], dtype=np.float32)
        truth_delta = truth_full.mean(axis=0) - ctrl_full.mean(axis=0)
        cond_emb = condition_embedding(dataset, cond_key)
        dists = [float(np.linalg.norm(cond_emb - emb)) for emb in train_embs.values()] if train_embs else [float("nan")]
        rows.append(
            {
                "dataset": str(dataset),
                "split_id": int(split_id),
                "condition": cond_key,
                "subgroup": subgroup_lookup.get(cond_key, ""),
                "truth_ctrl_shift_norm": float(np.linalg.norm(truth_delta)),
                "deg_count": int(len(np.asarray(item.get("DE_name", [])))),
                "train_test_distance": float(np.nanmin(np.asarray(dists, dtype=float))),
            }
        )
    return pd.DataFrame(rows)


def _choose_reference_payload_model(dataset: str, models: list[str], split_id: int) -> str | None:
    for model_name in models:
        spec = resolve_model_spec(model_name)
        if spec.kind != "payload":
            continue
        try:
            load_payload_item(dataset=dataset, model_name=model_name, split_id=int(split_id), condition=None)
            return model_name
        except Exception:
            continue
    return None


def _render_boxplot(df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(8, 5), dpi=220)
    if df.empty:
        plt.text(0.5, 0.5, "No stratified rows", ha="center", va="center")
        plt.axis("off")
    else:
        work = df[df["stratum_name"] == "effect_strength_bin"].copy()
        if work.empty:
            work = df.copy()
        labels = sorted(work["stratum_value"].dropna().astype(str).unique().tolist())
        vals = [pd.to_numeric(work[work["stratum_value"].astype(str) == lab]["pearson"], errors="coerce").dropna().to_numpy() for lab in labels]
        vals = [v for v in vals if v.size > 0]
        labels = [lab for lab, v in zip(labels, [pd.to_numeric(work[work["stratum_value"].astype(str) == lab]["pearson"], errors="coerce").dropna().to_numpy() for lab in labels]) if v.size > 0]
        if vals:
            plt.boxplot(vals, tick_labels=labels)
            plt.ylabel("pearson")
            plt.title("Stratified performance")
        else:
            plt.text(0.5, 0.5, "No valid boxplot rows", ha="center", va="center")
            plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _render_scatter(df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(6, 5), dpi=220)
    if df.empty:
        plt.text(0.5, 0.5, "No difficulty rows", ha="center", va="center")
        plt.axis("off")
    else:
        x = pd.to_numeric(df["train_test_distance"], errors="coerce")
        y = pd.to_numeric(df["pearson"], errors="coerce")
        plt.scatter(x, y, alpha=0.7, s=20)
        plt.xlabel("train_test_distance")
        plt.ylabel("pearson")
        plt.title("Difficulty vs performance")
        plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def run_stratified_benchmark(
    *,
    dataset: str,
    models: str | list[str] | tuple[str, ...] | None = None,
    split_ids: int | str | list[int] | tuple[int, ...] = "1",
    out_root: str | Path | None = None,
    paths_path: str | Path = "configs/paths.yaml",
    systema_root: str | Path | None = None,
) -> dict[str, Any]:
    dataset_key = str(dataset).strip()
    model_requests = parse_models(models)
    split_list = parse_split_ids(split_ids)
    out_dir = Path(out_root).resolve() if out_root else (Path("artifacts/analysis") / f"{ts_local()}_stratified_{dataset_key}").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata_frames: list[pd.DataFrame] = []
    thresholds: dict[str, list[float]] = {}
    skipped_models: list[dict[str, Any]] = []

    for split_id in split_list:
        ref_model = _choose_reference_payload_model(dataset_key, model_requests, int(split_id))
        if ref_model is None:
            warn_skip(f"[stratified] no payload-backed reference model for dataset={dataset_key} split={split_id}")
            continue
        try:
            split_dict = _load_dataset_split(dataset_key, int(split_id), paths_path)
            _, payload = load_payload_item(dataset=dataset_key, model_name=ref_model, split_id=int(split_id), condition=None)
            metadata_frames.append(
                _build_metadata_from_payload(
                    dataset_key,
                    payload,
                    int(split_id),
                    [normalize_condition(c) for c in split_dict.get("train_conds", [])],
                )
            )
        except Exception as exc:
            warn_skip(f"[stratified] skip split={split_id}: {exc}")

    metadata_df = pd.concat(metadata_frames, ignore_index=True).drop_duplicates(["split_id", "condition"]) if metadata_frames else pd.DataFrame()
    if not metadata_df.empty:
        metadata_df["effect_strength_bin"], thresholds["effect_strength_bin"] = _qcut_labels(
            metadata_df["truth_ctrl_shift_norm"], ["weak", "medium", "strong"]
        )
        metadata_df["train_distance_bin"], thresholds["train_distance_bin"] = _qcut_labels(
            metadata_df["train_test_distance"], ["near", "medium", "far"]
        )
        metadata_df["deg_difficulty_bin"], thresholds["deg_difficulty_bin"] = _qcut_labels(
            metadata_df["deg_count"], ["easy", "medium", "hard"]
        )

    merged_frames: list[pd.DataFrame] = []
    for model_name in model_requests:
        try:
            resolved = resolve_result(dataset=dataset_key, model_name=model_name, systema_root=systema_root)
            metrics_df = load_metrics_df(resolved)
        except Exception as exc:
            warn_skip(f"[stratified] skip model={model_name}: {exc}")
            skipped_models.append({"model_name": str(model_name), "reason": str(exc)})
            continue
        if "split_id" in metrics_df.columns:
            metrics_df = metrics_df[metrics_df["split_id"].isin(split_list)].copy()
        if metrics_df.empty:
            skipped_models.append({"model_name": str(model_name), "reason": "no rows after split filter"})
            continue
        metrics_df["condition"] = metrics_df["condition"].astype(str).map(normalize_condition)
        metrics_df["model_name"] = str(model_name)
        merged = metrics_df.merge(metadata_df, on=["split_id", "condition"], how="left")
        merged_frames.append(merged)

    merged_df = pd.concat(merged_frames, ignore_index=True) if merged_frames else pd.DataFrame()
    strata_rows: list[dict[str, Any]] = []
    if not merged_df.empty:
        for stratum_name in ["subgroup", "effect_strength_bin", "train_distance_bin", "deg_difficulty_bin"]:
            if stratum_name not in merged_df.columns:
                continue
            work = merged_df[merged_df[stratum_name].astype(str).ne("") & merged_df[stratum_name].notna()].copy()
            for keys, g in work.groupby(["model_name", stratum_name], sort=True):
                model_name, stratum_value = keys
                row = {
                    "dataset": dataset_key,
                    "model_name": str(model_name),
                    "stratum_name": str(stratum_name),
                    "stratum_value": str(stratum_value),
                    "n_rows": int(len(g)),
                }
                for metric in SUMMARY_METRICS:
                    if metric in g.columns:
                        row[metric] = float(pd.to_numeric(g[metric], errors="coerce").mean())
                strata_rows.append(row)
    stratified_df = pd.DataFrame(strata_rows)

    summary_rows: list[dict[str, Any]] = []
    win_rows: list[dict[str, Any]] = []
    if not stratified_df.empty:
        for (stratum_name, stratum_value), g in stratified_df.groupby(["stratum_name", "stratum_value"], sort=True):
            best_metric_rows = {}
            for metric in SUMMARY_METRICS:
                if metric not in g.columns:
                    continue
                series = pd.to_numeric(g[metric], errors="coerce")
                if series.notna().sum() == 0:
                    continue
                idx = series.idxmin() if metric in LOWER_IS_BETTER else series.idxmax()
                best_metric_rows[metric] = str(g.loc[idx, "model_name"])
            win_row = {"dataset": dataset_key, "stratum_name": str(stratum_name), "stratum_value": str(stratum_value)}
            for metric, model_name in best_metric_rows.items():
                win_row[f"best_{metric}"] = model_name
            win_rows.append(win_row)
        summary_rows = (
            stratified_df.groupby(["model_name", "stratum_name", "stratum_value"], as_index=False)[SUMMARY_METRICS]
            .mean()
            .to_dict(orient="records")
        )
    summary_df = pd.DataFrame(summary_rows)
    win_df = pd.DataFrame(win_rows)

    merged_df.to_csv(out_dir / "difficulty_scatter.csv", index=False, encoding="utf-8-sig")
    stratified_df.to_csv(out_dir / "stratified_metrics.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(out_dir / "stratified_summary.csv", index=False, encoding="utf-8-sig")
    win_df.to_csv(out_dir / "model_winrate_by_strata.csv", index=False, encoding="utf-8-sig")
    metadata_df.to_csv(out_dir / "condition_difficulty_metadata.csv", index=False, encoding="utf-8-sig")
    _render_boxplot(stratified_df, out_dir / "stratified_boxplot.png")
    _render_scatter(merged_df, out_dir / "difficulty_scatter.png")

    write_run_meta(
        out_dir / "run_meta.json",
        {
            "dataset": dataset_key,
            "models": model_requests,
            "split_ids": split_list,
            "paths_path": str(Path(paths_path).resolve()),
            "thresholds": thresholds,
            "skipped_models": skipped_models,
            "out_dir": str(out_dir),
        },
    )
    return {
        "out_dir": out_dir,
        "metadata_df": metadata_df,
        "merged_df": merged_df,
        "stratified_df": stratified_df,
        "summary_df": summary_df,
        "win_df": win_df,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run stratified robustness benchmark on exported metrics.")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--models", default="trishift_nearest,scouter,gears,genepert,systema_nonctl_mean,systema_matching_mean")
    ap.add_argument("--split_ids", default="1")
    ap.add_argument("--out_root", default="")
    ap.add_argument("--paths_path", default="configs/paths.yaml")
    ap.add_argument("--systema_root", default="")
    args = ap.parse_args(argv)

    result = run_stratified_benchmark(
        dataset=str(args.dataset).strip(),
        models=str(args.models).strip(),
        split_ids=str(args.split_ids).strip(),
        out_root=str(args.out_root).strip() or None,
        paths_path=str(args.paths_path).strip(),
        systema_root=str(args.systema_root).strip() or None,
    )
    print(f"out_dir: {result['out_dir']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
