from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts.trishift.analysis._result_adapter import (
    parse_models,
    parse_split_ids,
    resolve_model_spec,
    ts_local,
    warn_skip,
    write_run_meta,
)
from scripts.trishift.analysis.deg20_experiment import run_deg20_experiment


def _try_import_gseapy():
    try:
        import gseapy as gp  # type: ignore
    except Exception:
        return None
    return gp


def _run_enrichment_for_gene_lists(
    *,
    gene_lists_df: pd.DataFrame,
    enrichment_mode: str,
    enrichment_library: str,
) -> pd.DataFrame:
    mode_key = str(enrichment_mode).strip().lower()
    if mode_key in {"disabled", "off", "none", "export_only"}:
        return pd.DataFrame()
    gp = _try_import_gseapy()
    if gp is None:
        return pd.DataFrame()

    work_df = gene_lists_df[gene_lists_df["list_type"].isin(["truth_deg20", "pred_deg20"])].copy()
    rows: list[dict[str, Any]] = []
    for (model_name, dataset, split_id, condition, list_type), sub_df in work_df.groupby(
        ["model_name", "dataset", "split_id", "condition", "list_type"],
        sort=False,
    ):
        genes = sub_df.sort_values("rank")["gene"].astype(str).tolist()
        if not genes:
            continue
        try:
            enr = gp.enrichr(
                gene_list=genes,
                gene_sets=[str(enrichment_library)],
                organism="human",
                outdir=None,
                no_plot=True,
            )
        except Exception:
            continue
        res = getattr(enr, "results", None)
        if res is None or len(res) == 0:
            continue
        res_df = pd.DataFrame(res).head(20)
        for _, rr in res_df.iterrows():
            rows.append(
                {
                    "model_name": str(model_name),
                    "dataset": str(dataset),
                    "split_id": int(split_id),
                    "condition": str(condition),
                    "list_type": str(list_type),
                    "library": str(enrichment_library),
                    "term": str(rr.get("Term", "")),
                    "adjusted_p_value": float(rr.get("Adjusted P-value", np.nan)),
                    "combined_score": float(rr.get("Combined Score", np.nan)),
                    "overlap": str(rr.get("Overlap", "")),
                    "genes": str(rr.get("Genes", "")),
                }
            )
    return pd.DataFrame(rows)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=float).reshape(-1)
    y = np.asarray(b, dtype=float).reshape(-1)
    if x.size == 0 or y.size == 0 or np.allclose(np.std(x), 0.0) or np.allclose(np.std(y), 0.0):
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    x = pd.Series(np.asarray(a, dtype=float).reshape(-1))
    y = pd.Series(np.asarray(b, dtype=float).reshape(-1))
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    return float(x.rank().corr(y.rank(), method="pearson"))


def _build_overlap_tables(enrichment_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if enrichment_df.empty:
        empty = pd.DataFrame()
        return empty, empty, empty

    truth = enrichment_df[enrichment_df["list_type"] == "truth_deg20"].copy()
    pred = enrichment_df[enrichment_df["list_type"] == "pred_deg20"].copy()

    overlap_rows: list[dict[str, Any]] = []
    corr_rows: list[dict[str, Any]] = []
    hitk_rows: list[dict[str, Any]] = []
    for keys, truth_df in truth.groupby(["model_name", "dataset", "split_id", "condition"], sort=False):
        model_name, dataset, split_id, condition = keys
        pred_df = pred[
            (pred["model_name"].astype(str) == str(model_name))
            & (pred["dataset"].astype(str) == str(dataset))
            & (pred["split_id"] == int(split_id))
            & (pred["condition"].astype(str) == str(condition))
        ].copy()
        if pred_df.empty:
            continue
        truth_top = truth_df.sort_values(["adjusted_p_value", "combined_score"], ascending=[True, False]).head(10)
        pred_top = pred_df.sort_values(["adjusted_p_value", "combined_score"], ascending=[True, False]).head(10)
        truth_terms = truth_top["term"].astype(str).tolist()
        pred_terms = pred_top["term"].astype(str).tolist()
        truth_set = set(truth_terms)
        pred_set = set(pred_terms)
        inter = truth_set & pred_set
        union = truth_set | pred_set
        overlap_rows.append(
            {
                "model_name": str(model_name),
                "dataset": str(dataset),
                "split_id": int(split_id),
                "condition": str(condition),
                "top10_truth_terms": int(len(truth_set)),
                "top10_pred_terms": int(len(pred_set)),
                "top10_overlap": int(len(inter)),
                "top10_jaccard": (float(len(inter)) / float(len(union))) if union else float("nan"),
            }
        )
        hitk_rows.append(
            {
                "model_name": str(model_name),
                "dataset": str(dataset),
                "split_id": int(split_id),
                "condition": str(condition),
                "hit_at_1": float(len(set(truth_terms[:1]) & set(pred_terms[:1])) > 0),
                "hit_at_3": float(len(set(truth_terms[:3]) & set(pred_terms[:3])) > 0),
                "hit_at_5": float(len(set(truth_terms[:5]) & set(pred_terms[:5])) > 0),
                "hit_at_10": float(len(inter) > 0),
            }
        )
        merged = truth_df.merge(
            pred_df,
            on=["model_name", "dataset", "split_id", "condition", "term"],
            how="inner",
            suffixes=("_truth", "_pred"),
        )
        corr_rows.append(
            {
                "model_name": str(model_name),
                "dataset": str(dataset),
                "split_id": int(split_id),
                "condition": str(condition),
                "n_matched_terms": int(len(merged)),
                "combined_score_pearson": _pearson(
                    merged["combined_score_truth"].to_numpy(),
                    merged["combined_score_pred"].to_numpy(),
                ),
                "combined_score_spearman": _spearman(
                    merged["combined_score_truth"].to_numpy(),
                    merged["combined_score_pred"].to_numpy(),
                ),
            }
        )
    return pd.DataFrame(overlap_rows), pd.DataFrame(corr_rows), pd.DataFrame(hitk_rows)


def _render_scatter(corr_df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(6, 5), dpi=220)
    if corr_df.empty:
        plt.text(0.5, 0.5, "No enrichment results available", ha="center", va="center")
        plt.axis("off")
    else:
        x = pd.to_numeric(corr_df["combined_score_pearson"], errors="coerce")
        y = pd.to_numeric(corr_df["combined_score_spearman"], errors="coerce")
        plt.scatter(x, y, alpha=0.8, s=22)
        plt.xlabel("Combined score Pearson")
        plt.ylabel("Combined score Spearman")
        plt.title("Pathway score correlation by condition")
        plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _metric_summary(
    *,
    overlap_df: pd.DataFrame,
    corr_df: pd.DataFrame,
    hitk_df: pd.DataFrame,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for df in [overlap_df, corr_df, hitk_df]:
        if df.empty:
            continue
        value_cols = [
            c
            for c in df.columns
            if c not in {"model_name", "dataset", "split_id", "condition", "top10_truth_terms", "top10_pred_terms", "n_matched_terms"}
        ]
        if not value_cols:
            continue
        grouped = (
            df.groupby("model_name", sort=False)[value_cols]
            .mean(numeric_only=True)
            .reset_index()
        )
        frames.append(grouped)
    if not frames:
        return pd.DataFrame()
    out = frames[0]
    for df in frames[1:]:
        out = out.merge(df, on="model_name", how="outer")
    preferred_order = [
        "model_name",
        "top10_overlap",
        "top10_jaccard",
        "combined_score_pearson",
        "combined_score_spearman",
        "hit_at_1",
        "hit_at_3",
        "hit_at_5",
        "hit_at_10",
    ]
    cols = [c for c in preferred_order if c in out.columns] + [c for c in out.columns if c not in preferred_order]
    return out[cols].sort_values(by="model_name").reset_index(drop=True)


def _render_metric_heatmap(summary_df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(8, max(3.5, 0.75 * max(1, len(summary_df)))), dpi=220)
    if summary_df.empty:
        plt.text(0.5, 0.5, "No pathway summary available", ha="center", va="center")
        plt.axis("off")
    else:
        metric_cols = [c for c in summary_df.columns if c != "model_name"]
        values = summary_df[metric_cols].to_numpy(dtype=float)
        im = plt.imshow(values, aspect="auto", cmap="viridis")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.xticks(range(len(metric_cols)), metric_cols, rotation=35, ha="right")
        plt.yticks(range(len(summary_df)), summary_df["model_name"].astype(str).tolist())
        plt.title("Pathway Recovery Metric Means")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _load_trishift_runs_manifest(path: str | Path) -> list[dict[str, Any]]:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(obj, dict):
        runs = obj.get("runs", [])
    else:
        runs = obj
    if not isinstance(runs, list):
        raise TypeError("trishift_runs_manifest must contain a list or {\"runs\": [...]}")
    out: list[dict[str, Any]] = []
    for item in runs:
        if not isinstance(item, dict):
            raise TypeError("each trishift run spec must be a mapping")
        label = str(item.get("label", "")).strip()
        result_dir = str(item.get("result_dir", "")).strip()
        if not label or not result_dir:
            raise ValueError("each trishift run spec requires non-empty label and result_dir")
        out.append(
            {
                "label": label,
                "result_dir": result_dir,
                "variant_tag": str(item.get("variant_tag", "nearest")).strip() or "nearest",
            }
        )
    if not out:
        raise ValueError("trishift run spec list must not be empty")
    return out


def run_pathway_recovery(
    *,
    dataset: str,
    models: str | list[str] | tuple[str, ...] | None = None,
    split_ids: int | str | list[int] | tuple[int, ...] = "1",
    out_root: str | Path | None = None,
    truth_deg_mode: str = "effect_size_non_dropout",
    pred_deg_mode: str = "effect_size_non_dropout",
    enrichment_mode: str = "run_if_available",
    enrichment_library: str = "Reactome_2022",
    n_degs: int = 100,
    space: str = "auto",
    trishift_runs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    dataset_key = str(dataset).strip()
    model_requests = parse_models(models) if trishift_runs is None else []
    split_list = parse_split_ids(split_ids)
    out_dir = Path(out_root).resolve() if out_root else (Path("artifacts/analysis") / f"{ts_local()}_pathway_recovery_{dataset_key}").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    per_model_dirs: dict[str, str] = {}
    per_condition_frames: list[pd.DataFrame] = []
    gene_list_frames: list[pd.DataFrame] = []
    skipped_models: list[dict[str, Any]] = []
    used_models: list[str] = []

    if trishift_runs is not None:
        for run_spec in trishift_runs:
            label = str(run_spec["label"]).strip()
            result_dir = Path(str(run_spec["result_dir"]).strip()).resolve()
            variant_tag = str(run_spec.get("variant_tag", "nearest")).strip() or "nearest"
            model_out = out_dir / label
            try:
                result = run_deg20_experiment(
                    dataset=dataset_key,
                    model_name="trishift",
                    split_ids=split_list,
                    result_dir=result_dir,
                    out_root=model_out,
                    variant_tag=variant_tag,
                    truth_deg_mode=truth_deg_mode,
                    pred_deg_mode=pred_deg_mode,
                    enrichment_mode="export_only",
                    enrichment_library=enrichment_library,
                    n_degs=n_degs,
                    remove_perturbed_genes=True,
                    space=space,
                )
            except Exception as exc:
                warn_skip(f"[pathway_recovery] skip trishift label={label}: {exc}")
                skipped_models.append({"model_name": label, "reason": str(exc), "result_dir": str(result_dir)})
                continue
            used_models.append(label)
            per_model_dirs[label] = str(result.out_dir)
            df_metrics = result.per_condition_df.copy()
            df_metrics["model_request"] = label
            df_metrics["model_name"] = label
            per_condition_frames.append(df_metrics)
            df_genes = result.gene_lists_df.copy()
            df_genes["model_request"] = label
            df_genes["model_name"] = label
            gene_list_frames.append(df_genes)
    else:
        for model_name in model_requests:
            spec = resolve_model_spec(model_name)
            if spec.kind != "payload" or spec.base_model is None:
                warn_skip(f"[pathway_recovery] skip model={model_name}: metrics-only baseline")
                skipped_models.append({"model_name": str(model_name), "reason": "metrics-only baseline"})
                continue
            model_out = out_dir / model_name
            try:
                result = run_deg20_experiment(
                    dataset=dataset_key,
                    model_name=spec.base_model,
                    split_ids=split_list,
                    out_root=model_out,
                    variant_tag=spec.variant_tag,
                    truth_deg_mode=truth_deg_mode,
                    pred_deg_mode=pred_deg_mode,
                    enrichment_mode="export_only",
                    enrichment_library=enrichment_library,
                    n_degs=n_degs,
                    remove_perturbed_genes=True,
                    space=space,
                )
            except Exception as exc:
                warn_skip(f"[pathway_recovery] skip model={model_name}: {exc}")
                skipped_models.append({"model_name": str(model_name), "reason": str(exc)})
                continue
            used_models.append(str(model_name))
            per_model_dirs[str(model_name)] = str(result.out_dir)
            df_metrics = result.per_condition_df.copy()
            df_metrics["model_request"] = str(model_name)
            per_condition_frames.append(df_metrics)
            df_genes = result.gene_lists_df.copy()
            df_genes["model_request"] = str(model_name)
            gene_list_frames.append(df_genes)

    per_condition_df = pd.concat(per_condition_frames, ignore_index=True) if per_condition_frames else pd.DataFrame()
    gene_lists_df = pd.concat(gene_list_frames, ignore_index=True) if gene_list_frames else pd.DataFrame()
    enrichment_df = _run_enrichment_for_gene_lists(
        gene_lists_df=gene_lists_df,
        enrichment_mode=enrichment_mode,
        enrichment_library=enrichment_library,
    )
    overlap_df, corr_df, hitk_df = _build_overlap_tables(enrichment_df)
    metric_summary_df = _metric_summary(
        overlap_df=overlap_df,
        corr_df=corr_df,
        hitk_df=hitk_df,
    )

    per_condition_df.to_csv(out_dir / "deg20_per_condition_all.csv", index=False, encoding="utf-8-sig")
    gene_lists_df.to_csv(out_dir / "deg_gene_lists_all.csv", index=False, encoding="utf-8-sig")
    enrichment_df.to_csv(out_dir / "pathway_enrichment_all.csv", index=False, encoding="utf-8-sig")
    overlap_df.to_csv(out_dir / "pathway_overlap_summary.csv", index=False, encoding="utf-8-sig")
    corr_df.to_csv(out_dir / "pathway_nes_correlation.csv", index=False, encoding="utf-8-sig")
    hitk_df.to_csv(out_dir / "pathway_hitk_summary.csv", index=False, encoding="utf-8-sig")
    metric_summary_df.to_csv(out_dir / "pathway_metric_means.csv", index=False, encoding="utf-8-sig")
    _render_scatter(corr_df, out_dir / "pathway_nes_scatter.png")
    _render_metric_heatmap(metric_summary_df, out_dir / "pathway_metric_heatmap.png")

    write_run_meta(
        out_dir / "run_meta.json",
        {
            "dataset": dataset_key,
            "models_requested": model_requests,
            "models_used": used_models,
            "split_ids": split_list,
            "truth_deg_mode": str(truth_deg_mode),
            "pred_deg_mode": str(pred_deg_mode),
            "enrichment_mode": str(enrichment_mode),
            "enrichment_library": str(enrichment_library),
            "n_degs": int(n_degs),
            "space": str(space),
            "mode": "trishift_runs" if trishift_runs is not None else "models",
            "trishift_runs": trishift_runs or [],
            "per_model_dirs": per_model_dirs,
            "skipped_models": skipped_models,
            "has_enrichment_results": bool(not enrichment_df.empty),
            "score_column": "combined_score",
            "out_dir": str(out_dir),
        },
    )
    return {
        "out_dir": out_dir,
        "per_condition_df": per_condition_df,
        "gene_lists_df": gene_lists_df,
        "enrichment_df": enrichment_df,
        "overlap_df": overlap_df,
        "corr_df": corr_df,
        "hitk_df": hitk_df,
        "metric_summary_df": metric_summary_df,
        "skipped_models": skipped_models,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run pathway/mechanism recovery on top of DEG20 outputs.")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--models", default="trishift_nearest,scouter,gears,genepert")
    ap.add_argument("--split_ids", default="1")
    ap.add_argument("--out_root", default="")
    ap.add_argument("--truth_deg_mode", default="effect_size_non_dropout")
    ap.add_argument("--pred_deg_mode", default="effect_size_non_dropout")
    ap.add_argument("--enrichment_mode", default="run_if_available", choices=["export_only", "run_if_available", "disabled"])
    ap.add_argument("--enrichment_library", default="Reactome_2022")
    ap.add_argument("--n_degs", type=int, default=100)
    ap.add_argument("--space", default="auto", choices=["auto", "full_gene", "deg"])
    ap.add_argument("--trishift_runs_manifest", default="")
    args = ap.parse_args(argv)

    trishift_runs = None
    if str(args.trishift_runs_manifest).strip():
        trishift_runs = _load_trishift_runs_manifest(str(args.trishift_runs_manifest).strip())

    result = run_pathway_recovery(
        dataset=str(args.dataset).strip(),
        models=str(args.models).strip(),
        split_ids=str(args.split_ids).strip(),
        out_root=str(args.out_root).strip() or None,
        truth_deg_mode=str(args.truth_deg_mode).strip(),
        pred_deg_mode=str(args.pred_deg_mode).strip(),
        enrichment_mode=str(args.enrichment_mode).strip(),
        enrichment_library=str(args.enrichment_library).strip(),
        n_degs=int(args.n_degs),
        space=str(args.space).strip(),
        trishift_runs=trishift_runs,
    )
    print(f"out_dir: {result['out_dir']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
