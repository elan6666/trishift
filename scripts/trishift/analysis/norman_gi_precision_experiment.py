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

from scripts.trishift.analysis._result_adapter import (
    load_payload_item,
    normalize_condition,
    parse_models,
    parse_split_ids,
    ts_local,
    warn_skip,
    write_run_meta,
)
from scripts.trishift.analysis.norman_nonadd_experiment import (
    DEFAULT_MODELS,
    _combo_single_keys,
    _compute_condition_gi,
)

GI_TYPE_ORDER = [
    "synergy",
    "suppressor",
    "neomorphic",
    "redundancy",
    "epistasis",
    "strong_interaction",
    "balanced_synergy",
    "dominant_epistasis",
]
GI_TYPE_LABELS = {
    "synergy": "Synergy",
    "suppressor": "Suppression",
    "neomorphic": "Neomorphism",
    "redundancy": "Redundancy",
    "epistasis": "Epistasis",
    "strong_interaction": "Strong Interaction",
    "balanced_synergy": "Balanced Synergy",
    "dominant_epistasis": "Dominant Epistasis",
}
GI_TYPE_SPECS = {
    "synergy": {"metric_name": "mag", "rank_direction": "max", "truth_label_col": "is_synergy"},
    "suppressor": {"metric_name": "mag", "rank_direction": "min", "truth_label_col": "is_suppressor"},
    "neomorphic": {"metric_name": "corr_fit", "rank_direction": "min", "truth_label_col": "is_neomorphic"},
    "redundancy": {"metric_name": "dcor", "rank_direction": "max", "truth_label_col": "is_redundancy"},
    "epistasis": {"metric_name": "eq_contr", "rank_direction": "min", "truth_label_col": "is_epistasis"},
    "strong_interaction": {"metric_name": "mag", "rank_direction": "max", "truth_label_col": "is_strong_interaction"},
    "balanced_synergy": {"metric_name": "mag", "rank_direction": "max", "truth_label_col": "is_balanced_synergy"},
    "dominant_epistasis": {"metric_name": "dominance", "rank_direction": "max", "truth_label_col": "is_dominant_epistasis"},
}
DEFAULT_TRUTH_LABELS_PATH = (REPO_ROOT / "artifacts" / "analysis" / "norman_gi_truth" / "norman_gi_truth_labels.csv").resolve()


def _load_truth_labels(path: str | Path, *, gi_types: list[str]) -> pd.DataFrame:
    csv_path = Path(path).resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"Truth labels file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if "condition" not in df.columns:
        raise ValueError(f"Truth labels file is missing condition column: {csv_path}")
    work = df.copy()
    work["condition"] = work["condition"].astype(str).map(normalize_condition)
    required_cols = {"condition"}
    for gi_type in gi_types:
        spec = GI_TYPE_SPECS[gi_type]
        required_cols.add(str(spec["truth_label_col"]))
        required_cols.add(f"truth_{spec['metric_name']}")
    missing = sorted(required_cols - set(work.columns))
    if missing:
        raise ValueError(f"Truth labels file is missing required columns: {missing}")
    for gi_type in gi_types:
        label_col = str(GI_TYPE_SPECS[gi_type]["truth_label_col"])
        work[label_col] = work[label_col].fillna(False).astype(bool)
    return work


def _collect_condition_rows(
    *,
    model_name: str,
    split_id: int,
    payload: dict[str, Any],
    known_combos: set[str],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    normalized_payload = {normalize_condition(str(k)): v for k, v in payload.items() if isinstance(v, dict)}
    rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []
    for condition, item_ab in sorted(normalized_payload.items()):
        if condition not in known_combos:
            continue
        tokens = [x for x in condition.split("+") if x and x != "ctrl"]
        if len(tokens) != 2:
            continue
        try:
            single_a, single_b = _combo_single_keys(condition)
        except Exception as exc:
            skipped_rows.append({"model_name": str(model_name), "split_id": int(split_id), "condition": str(condition), "skip_reason": str(exc)})
            continue
        if single_a not in normalized_payload or single_b not in normalized_payload:
            missing = single_a if single_a not in normalized_payload else single_b
            skipped_rows.append(
                {
                    "model_name": str(model_name),
                    "split_id": int(split_id),
                    "condition": str(condition),
                    "skip_reason": f"missing single perturbation: {missing}",
                }
            )
            continue
        try:
            gi_row = _compute_condition_gi(item_ab=item_ab, item_a=normalized_payload[single_a], item_b=normalized_payload[single_b])
        except Exception as exc:
            skipped_rows.append({"model_name": str(model_name), "split_id": int(split_id), "condition": str(condition), "skip_reason": str(exc)})
            continue
        rows.append({"model_name": str(model_name), "split_id": int(split_id), "condition": str(condition), **gi_row})
    return (pd.DataFrame(rows), skipped_rows)


def _compute_precision_for_type(
    df: pd.DataFrame,
    *,
    gi_type: str,
    top_k: int,
    truth_labels_df: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    spec = GI_TYPE_SPECS[gi_type]
    metric_name = str(spec["metric_name"])
    rank_direction = str(spec["rank_direction"])
    truth_label_col = str(spec["truth_label_col"])
    truth_col = f"truth_{metric_name}"
    pred_col = f"pred_{metric_name}"
    merge_cols = ["condition", truth_label_col, truth_col]
    if df.empty or "condition" not in df.columns or truth_labels_df.empty:
        work = pd.DataFrame(columns=["condition", truth_label_col, truth_col, pred_col])
    else:
        work = df.merge(truth_labels_df[merge_cols], on="condition", how="inner", suffixes=("", "_label"))
        if f"{truth_col}_label" in work.columns:
            work[truth_col] = pd.to_numeric(work[f"{truth_col}_label"], errors="coerce")
    coverage_n = int(len(work))
    effective_k = int(min(int(top_k), coverage_n))
    positive_truth_n = int(work[truth_label_col].fillna(False).astype(bool).sum()) if not work.empty else 0
    row = {
        "gi_type": str(gi_type),
        "gi_label": GI_TYPE_LABELS[gi_type],
        "metric_name": metric_name,
        "rank_direction": rank_direction,
        "coverage_n": coverage_n,
        "positive_truth_n": positive_truth_n,
        "effective_k": effective_k,
        "precision_at_10": float("nan"),
    }
    if coverage_n == 0:
        return row, {"gi_type": gi_type, "skip_reason": "no overlapping known GI combos"}
    if positive_truth_n == 0:
        return row, {"gi_type": gi_type, "skip_reason": "no truth positives in local truth labels"}
    if effective_k <= 0:
        return row, {"gi_type": gi_type, "skip_reason": "effective_k <= 0"}
    ascending = rank_direction == "min"
    top = work.sort_values(pred_col, ascending=ascending).head(effective_k).copy()
    hits = int(top[truth_label_col].fillna(False).astype(bool).sum())
    row["precision_at_10"] = float(hits / effective_k)
    return row, None


def _summarize_precision(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    numeric_cols = ["precision_at_10", "coverage_n", "positive_truth_n", "effective_k"]
    grouped = df.groupby(["model_name", "gi_type", "gi_label", "metric_name", "rank_direction"], as_index=False)
    summary = grouped[numeric_cols].agg(["mean", "std", "min", "max"]).reset_index()
    summary.columns = [
        "_".join([str(x) for x in col if str(x)])
        .rstrip("_")
        .replace("precision_at_10_mean", "precision_at_10_mean")
        for col in summary.columns.to_flat_index()
    ]
    if "precision_at_10_std" not in summary.columns:
        summary["precision_at_10_std"] = np.nan
    valid_counts = (
        df.assign(_valid=pd.to_numeric(df["precision_at_10"], errors="coerce").notna().astype(int))
        .groupby(["model_name", "gi_type"], as_index=False)["_valid"]
        .sum()
        .rename(columns={"_valid": "valid_split_n"})
    )
    attempted_counts = (
        df.groupby(["model_name", "gi_type"], as_index=False)
        .size()
        .rename(columns={"size": "attempted_split_n"})
    )
    summary = summary.merge(valid_counts, on=["model_name", "gi_type"], how="left")
    summary = summary.merge(attempted_counts, on=["model_name", "gi_type"], how="left")
    return summary


def _build_coverage_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    grouped = df.groupby(["model_name", "gi_type", "gi_label"], as_index=False)[["coverage_n", "positive_truth_n", "effective_k"]].agg(["mean", "min", "max"]).reset_index()
    grouped.columns = ["_".join([str(x) for x in col if str(x)]).rstrip("_") for col in grouped.columns.to_flat_index()]
    return grouped


def _render_precision_barplot(df: pd.DataFrame, out_path: Path, *, model_order: list[str], top_k: int) -> None:
    fig, ax = plt.subplots(figsize=(15.5, 6.8), dpi=220)
    if df.empty:
        ax.text(0.5, 0.5, "No Precision@10 summary rows", ha="center", va="center")
        ax.axis("off")
    else:
        x = np.arange(len(GI_TYPE_ORDER))
        width = 0.8 / max(1, len(model_order))
        cmap = plt.get_cmap("tab10")
        ymax = 1.22
        for idx, model_name in enumerate(model_order):
            sub = df[df["model_name"].astype(str) == model_name].copy()
            sub["gi_type"] = pd.Categorical(sub["gi_type"], categories=GI_TYPE_ORDER, ordered=True)
            sub = sub.sort_values("gi_type").set_index("gi_type").reindex(GI_TYPE_ORDER)
            y = pd.to_numeric(sub.get("precision_at_10_mean"), errors="coerce")
            yerr = pd.to_numeric(sub.get("precision_at_10_std"), errors="coerce")
            xpos = x + (idx - (len(model_order) - 1) / 2.0) * width
            ax.bar(
                xpos,
                y,
                width=width,
                color=cmap(idx % 10),
                label=model_name,
                alpha=0.9,
                edgecolor="white",
                linewidth=0.6,
            )
            valid_mask = y.notna() & yerr.notna()
            if valid_mask.any():
                ax.errorbar(xpos[valid_mask.to_numpy()], y[valid_mask].to_numpy(), yerr=yerr[valid_mask].to_numpy(), fmt="none", ecolor="black", elinewidth=1.4, capsize=3)
            cov = pd.to_numeric(sub.get("coverage_n_mean"), errors="coerce")
            for xi, yi, cov_i in zip(xpos, y.fillna(0.0), cov.fillna(np.nan)):
                if np.isnan(cov_i):
                    ax.text(
                        float(xi),
                        0.04,
                        "NA",
                        ha="center",
                        va="bottom",
                        rotation=0,
                        fontsize=7,
                        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 0.4},
                    )
                elif float(cov_i) < float(top_k):
                    label_y = min(float(yi) + 0.04, ymax - 0.08)
                    ax.text(
                        float(xi),
                        label_y,
                        f"n={int(round(float(cov_i)))}",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        rotation=0,
                        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 0.4},
                    )
        ax.set_xticks(x)
        ax.set_xticklabels([GI_TYPE_LABELS[k] for k in GI_TYPE_ORDER], rotation=20, ha="right")
        ax.set_ylabel("Precision@10")
        ax.set_title("Model precision in identifying genetic interactions", pad=16)
        ax.set_ylim(0, ymax)
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.22, linewidth=0.8)
        ax.axhline(1.0, color="#666666", linewidth=0.9, linestyle="--", alpha=0.45)
        ax.legend(
            ncol=min(3, len(model_order)),
            loc="upper right",
            bbox_to_anchor=(0.995, 0.995),
            frameon=True,
            framealpha=0.92,
            borderpad=0.35,
            handlelength=1.4,
            columnspacing=0.9,
            fontsize=8,
        )
        ax.margins(x=0.02)
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.22, top=0.88)
    plt.savefig(out_path)
    plt.close(fig)


def _render_coverage_barplot(df: pd.DataFrame, out_path: Path, *, model_order: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(15.5, 6.8), dpi=220)
    if df.empty:
        ax.text(0.5, 0.5, "No coverage rows", ha="center", va="center")
        ax.axis("off")
    else:
        x = np.arange(len(GI_TYPE_ORDER))
        width = 0.8 / max(1, len(model_order))
        cmap = plt.get_cmap("tab10")
        ymax = 0.0
        for idx, model_name in enumerate(model_order):
            sub = df[df["model_name"].astype(str) == model_name].copy()
            sub["gi_type"] = pd.Categorical(sub["gi_type"], categories=GI_TYPE_ORDER, ordered=True)
            sub = sub.sort_values("gi_type").set_index("gi_type").reindex(GI_TYPE_ORDER)
            y = pd.to_numeric(sub.get("coverage_n_mean"), errors="coerce")
            xpos = x + (idx - (len(model_order) - 1) / 2.0) * width
            ymax = max(ymax, float(np.nanmax(y.to_numpy(dtype=float))) if y.notna().any() else ymax)
            ax.bar(
                xpos,
                y,
                width=width,
                color=cmap(idx % 10),
                label=model_name,
                alpha=0.9,
                edgecolor="white",
                linewidth=0.6,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([GI_TYPE_LABELS[k] for k in GI_TYPE_ORDER], rotation=20, ha="right")
        ax.set_ylabel("Known-GI coverage")
        ax.set_title("Known GI combo coverage by model", pad=16)
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.22, linewidth=0.8)
        ax.set_ylim(0, max(1.0, ymax + 0.8))
        ax.legend(
            ncol=min(3, len(model_order)),
            loc="upper right",
            bbox_to_anchor=(0.995, 0.995),
            frameon=True,
            framealpha=0.92,
            borderpad=0.35,
            handlelength=1.4,
            columnspacing=0.9,
            fontsize=8,
        )
        ax.margins(x=0.02)
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.22, top=0.88)
    plt.savefig(out_path)
    plt.close(fig)


def run_norman_gi_precision_experiment(
    *,
    dataset: str,
    models: str | list[str] | tuple[str, ...] | None = None,
    split_ids: int | str | list[int] | tuple[int, ...] = "1",
    out_root: str | Path | None = None,
    top_k: int = 10,
    truth_labels_path: str | Path | None = None,
) -> dict[str, Any]:
    dataset_key = normalize_condition(str(dataset))
    if dataset_key != "norman":
        raise ValueError("norman_gi_precision_experiment only supports dataset=norman")
    model_requests = parse_models(models or DEFAULT_MODELS)
    split_list = parse_split_ids(split_ids)
    out_dir = Path(out_root).resolve() if out_root else (Path("artifacts/analysis") / "norman_gi_precision_experiment").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    truth_labels_resolved = Path(truth_labels_path).resolve() if truth_labels_path else DEFAULT_TRUTH_LABELS_PATH
    truth_labels_df = _load_truth_labels(truth_labels_resolved, gi_types=GI_TYPE_ORDER)
    known_combo_union = set(truth_labels_df["condition"].astype(str).tolist())

    per_split_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []
    used_models: list[str] = []
    for model_name in model_requests:
        try:
            _, payload = load_payload_item(dataset="norman", model_name=model_name, split_id=int(split_list[0]), condition=None)
        except Exception:
            payload = None
        if payload is None:
            warn_skip(f"[norman_gi_precision] skip model={model_name}: payload unavailable")
            skipped_rows.append({"model_name": str(model_name), "split_id": "", "gi_type": "", "condition": "", "skip_reason": "payload unavailable"})
            continue
        used_models.append(str(model_name))

        for split_id in split_list:
            try:
                _, payload = load_payload_item(dataset="norman", model_name=model_name, split_id=int(split_id), condition=None)
            except Exception as exc:
                warn_skip(f"[norman_gi_precision] skip model={model_name} split={split_id}: {exc}")
                skipped_rows.append({"model_name": str(model_name), "split_id": int(split_id), "gi_type": "", "condition": "", "skip_reason": str(exc)})
                continue
            condition_df, condition_skips = _collect_condition_rows(
                model_name=str(model_name),
                split_id=int(split_id),
                payload=payload,
                known_combos=known_combo_union,
            )
            for row in condition_skips:
                skipped_rows.append({"gi_type": "", **row})
            for gi_type in GI_TYPE_ORDER:
                row, skip = _compute_precision_for_type(
                    condition_df,
                    gi_type=gi_type,
                    top_k=int(top_k),
                    truth_labels_df=truth_labels_df,
                )
                per_split_rows.append(
                    {
                        "dataset": "norman",
                        "model_name": str(model_name),
                        "split_id": int(split_id),
                        **row,
                    }
                )
                if skip is not None:
                    skipped_rows.append(
                        {
                            "model_name": str(model_name),
                            "split_id": int(split_id),
                            "condition": "",
                            **skip,
                        }
                    )

    per_split_df = pd.DataFrame(per_split_rows)
    summary_df = _summarize_precision(per_split_df)
    coverage_df = _build_coverage_df(per_split_df)
    skipped_df = pd.DataFrame(skipped_rows)

    per_split_df.to_csv(out_dir / "norman_gi_precision_per_split.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(out_dir / "norman_gi_precision_summary.csv", index=False, encoding="utf-8-sig")
    coverage_df.to_csv(out_dir / "norman_gi_precision_coverage.csv", index=False, encoding="utf-8-sig")
    skipped_df.to_csv(out_dir / "norman_gi_precision_skipped.csv", index=False, encoding="utf-8-sig")
    _render_precision_barplot(summary_df, out_dir / "norman_gi_precision_barplot.png", model_order=used_models, top_k=int(top_k))
    _render_coverage_barplot(coverage_df, out_dir / "norman_gi_precision_coverage_barplot.png", model_order=used_models)

    coverage_summary = coverage_df.to_dict(orient="records") if not coverage_df.empty else []
    write_run_meta(
        out_dir / "run_meta.json",
        {
            "dataset": "norman",
            "models_requested": model_requests,
            "models_used": used_models,
            "split_ids": split_list,
            "top_k": int(top_k),
            "gi_types": GI_TYPE_ORDER,
            "truth_labels_path": str(truth_labels_resolved),
            "metric_map": {
                k: {
                    "metric_name": v["metric_name"],
                    "rank_direction": v["rank_direction"],
                    "truth_label_col": v["truth_label_col"],
                }
                for k, v in GI_TYPE_SPECS.items()
            },
            "coverage_summary": coverage_summary,
            "skipped_rows": skipped_rows,
            "out_dir": str(out_dir),
        },
    )
    return {
        "out_dir": out_dir,
        "per_split_df": per_split_df,
        "summary_df": summary_df,
        "coverage_df": coverage_df,
        "skipped_df": skipped_df,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run paper-style Norman GI Precision@10 analysis from exported PKLs.")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--models", default="trishift_nearest,scouter,gears,genepert,scgpt")
    ap.add_argument("--split_ids", default="1")
    ap.add_argument("--out_root", default="")
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--truth_labels_path", default="")
    args = ap.parse_args(argv)

    result = run_norman_gi_precision_experiment(
        dataset=str(args.dataset).strip(),
        models=str(args.models).strip(),
        split_ids=str(args.split_ids).strip(),
        out_root=str(args.out_root).strip() or None,
        top_k=int(args.top_k),
        truth_labels_path=str(args.truth_labels_path).strip() or None,
    )
    print(f"out_dir: {result['out_dir']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
