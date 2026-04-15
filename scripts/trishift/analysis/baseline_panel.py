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
    DEFAULT_MODEL_REQUESTS,
    load_payload_item,
    load_metrics_df,
    parse_models,
    parse_split_ids,
    resolve_model_spec,
    resolve_result,
    ts_local,
    warn_skip,
    write_run_meta,
)
from scripts.trishift.analysis.stratified_benchmark import (
    _build_metadata_from_payload,
    _load_dataset_split,
)


PANEL_METRICS = [
    "pearson",
    "nmse",
    "deg_mean_r2",
    "systema_corr_20de_allpert",
    "systema_corr_deg_r2",
    "scpram_r2_degs_mean_mean",
    "scpram_r2_degs_var_mean",
    "scpram_wasserstein_degs_sum",
]

LOWER_IS_BETTER = {"nmse", "scpram_wasserstein_degs_sum"}


def _mean_or_nan(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns:
        return float("nan")
    s = pd.to_numeric(df[col], errors="coerce")
    if s.notna().sum() == 0:
        return float("nan")
    return float(s.mean())


def _normalize_for_heatmap(table: pd.DataFrame) -> pd.DataFrame:
    out = table.copy()
    for col in out.columns:
        s = pd.to_numeric(out[col], errors="coerce")
        if s.notna().sum() == 0:
            out[col] = np.nan
            continue
        lo = float(s.min())
        hi = float(s.max())
        if np.isclose(lo, hi):
            out[col] = 0.5
            continue
        scaled = (s - lo) / (hi - lo)
        out[col] = 1.0 - scaled if col in LOWER_IS_BETTER else scaled
    return out


def _render_heatmap(table: pd.DataFrame, out_path: Path) -> None:
    if table.empty:
        return
    score = _normalize_for_heatmap(table)
    plt.figure(figsize=(max(8, table.shape[1] * 1.2), max(5, table.shape[0] * 0.55)), dpi=220)
    im = plt.imshow(score.values, aspect="auto", cmap="RdYlGn", vmin=0.0, vmax=1.0)
    plt.colorbar(im, label="normalized score")
    plt.xticks(np.arange(table.shape[1]), table.columns, rotation=40, ha="right")
    plt.yticks(np.arange(table.shape[0]), table.index)
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            val = table.iat[i, j]
            txt = "NA" if pd.isna(val) else f"{float(val):.4f}"
            bg = score.iat[i, j]
            color = "black" if pd.isna(bg) or bg > 0.45 else "white"
            plt.text(j, i, txt, ha="center", va="center", fontsize=8, color=color)
    plt.title("Baseline Panel")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _infer_subgroup_metadata(
    *,
    dataset: str,
    split_ids: list[int],
    model_requests: list[str],
) -> pd.DataFrame:
    ref_models = [
        str(model_name)
        for model_name in model_requests
        if resolve_model_spec(model_name).kind == "payload"
    ]
    if not ref_models:
        return pd.DataFrame(columns=["split_id", "condition", "subgroup"])

    rows: list[pd.DataFrame] = []
    for split_id in split_ids:
        split_dict = None
        for model_name in ref_models:
            try:
                if split_dict is None:
                    split_dict = _load_dataset_split(dataset, int(split_id), "configs/paths.yaml")
                _, payload = load_payload_item(
                    dataset=dataset,
                    model_name=model_name,
                    split_id=int(split_id),
                    condition=None,
                )
                metadata_df = _build_metadata_from_payload(
                    dataset,
                    payload,
                    int(split_id),
                    [str(c) for c in split_dict.get("train_conds", [])],
                    "configs/paths.yaml",
                )
                if metadata_df.empty or "subgroup" not in metadata_df.columns:
                    continue
                rows.append(metadata_df[["split_id", "condition", "subgroup"]].copy())
                break
            except Exception:
                continue
    if not rows:
        return pd.DataFrame(columns=["split_id", "condition", "subgroup"])
    out = pd.concat(rows, ignore_index=True).drop_duplicates(["split_id", "condition"])
    out["condition"] = out["condition"].astype(str)
    return out


def run_baseline_panel(
    *,
    dataset: str,
    models: str | list[str] | tuple[str, ...] | None = None,
    split_ids: int | str | list[int] | tuple[int, ...] | None = None,
    subgroup_filter: str | list[str] | tuple[str, ...] | None = None,
    out_root: str | Path | None = None,
    systema_root: str | Path | None = None,
) -> dict[str, Any]:
    dataset_key = str(dataset).strip()
    model_requests = parse_models(models)
    split_list = parse_split_ids(split_ids or "1")
    subgroup_values = (
        []
        if subgroup_filter is None
        else [str(x).strip() for x in ([subgroup_filter] if isinstance(subgroup_filter, str) else subgroup_filter) if str(x).strip()]
    )
    subgroup_metadata = _infer_subgroup_metadata(
        dataset=dataset_key,
        split_ids=split_list,
        model_requests=model_requests,
    )
    out_dir = Path(out_root).resolve() if out_root else (Path("artifacts/analysis") / f"{ts_local()}_baseline_panel_{dataset_key}").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    subgroup_rows: list[dict[str, Any]] = []
    skipped_models: list[dict[str, Any]] = []

    for model_name in model_requests:
        try:
            resolved = resolve_result(dataset=dataset_key, model_name=model_name, systema_root=systema_root)
            metrics_df = load_metrics_df(resolved)
        except Exception as exc:
            warn_skip(f"[baseline_panel] skip model={model_name}: {exc}")
            skipped_models.append({"model_name": str(model_name), "reason": str(exc)})
            continue

        if "split_id" in metrics_df.columns:
            split_filtered = metrics_df[metrics_df["split_id"].isin(split_list)].copy()
        else:
            split_filtered = metrics_df.copy()
        if (
            "subgroup" not in split_filtered.columns
            and not subgroup_metadata.empty
            and {"split_id", "condition"}.issubset(split_filtered.columns)
        ):
            split_filtered["condition"] = split_filtered["condition"].astype(str)
            split_filtered = split_filtered.merge(
                subgroup_metadata,
                on=["split_id", "condition"],
                how="left",
            )
        if subgroup_values:
            if "subgroup" not in split_filtered.columns:
                if "subgroup" not in split_filtered.columns:
                    warn_skip(f"[baseline_panel] skip model={model_name}: subgroup filter requested but subgroup column missing")
                    skipped_models.append({"model_name": str(model_name), "reason": "subgroup column missing"})
                    continue
            split_filtered = split_filtered[split_filtered["subgroup"].astype(str).isin(subgroup_values)].copy()
        if split_filtered.empty:
            warn_skip(f"[baseline_panel] skip model={model_name}: no rows after split filter {split_list}")
            skipped_models.append({"model_name": str(model_name), "reason": "no rows after split filter"})
            continue

        split_filtered["model_name"] = str(model_name)
        split_filtered["label"] = str(resolved.spec.label)
        raw_rows.extend(split_filtered.to_dict(orient="records"))

        summary = {
            "dataset": dataset_key,
            "model_name": str(model_name),
            "label": str(resolved.spec.label),
            "n_rows": int(len(split_filtered)),
            "split_ids_used": ",".join(map(str, sorted(set(pd.to_numeric(split_filtered.get("split_id"), errors="coerce").dropna().astype(int).tolist()))))
            if "split_id" in split_filtered.columns
            else "",
            "metrics_path": str(resolved.metrics_path),
        }
        for metric in PANEL_METRICS:
            summary[f"mean_{metric}"] = _mean_or_nan(split_filtered, metric)
        summary_rows.append(summary)

        if "subgroup" in split_filtered.columns:
            for subgroup, g in split_filtered.groupby(split_filtered["subgroup"].astype(str), sort=True):
                row = {
                    "dataset": dataset_key,
                    "model_name": str(model_name),
                    "label": str(resolved.spec.label),
                    "subgroup": str(subgroup),
                    "n_rows": int(len(g)),
                }
                for metric in PANEL_METRICS:
                    row[f"mean_{metric}"] = _mean_or_nan(g, metric)
                subgroup_rows.append(row)

    raw_df = pd.DataFrame(raw_rows)
    summary_df = pd.DataFrame(summary_rows)
    subgroup_df = pd.DataFrame(subgroup_rows)

    ranking_df = summary_df.sort_values("mean_pearson", ascending=False).reset_index(drop=True) if not summary_df.empty else summary_df

    raw_df.to_csv(out_dir / "baseline_panel_raw.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(out_dir / "baseline_panel_summary.csv", index=False, encoding="utf-8-sig")
    ranking_df.to_csv(out_dir / "baseline_panel_ranking.csv", index=False, encoding="utf-8-sig")
    subgroup_df.to_csv(out_dir / "norman_subgroup_panel.csv", index=False, encoding="utf-8-sig")

    if not summary_df.empty:
        heatmap_table = summary_df.set_index("label")[[f"mean_{m}" for m in PANEL_METRICS if f"mean_{m}" in summary_df.columns]].copy()
        heatmap_table.columns = [c.replace("mean_", "") for c in heatmap_table.columns]
        _render_heatmap(heatmap_table, out_dir / "baseline_panel_heatmap.png")

    write_run_meta(
        out_dir / "run_meta.json",
        {
            "dataset": dataset_key,
            "models": model_requests,
            "split_ids": split_list,
            "subgroup_filter": subgroup_values,
            "systema_root": ("" if systema_root is None else str(systema_root)),
            "default_models": DEFAULT_MODEL_REQUESTS,
            "skipped_models": skipped_models,
            "out_dir": str(out_dir),
        },
    )
    return {
        "out_dir": out_dir,
        "raw_df": raw_df,
        "summary_df": summary_df,
        "ranking_df": ranking_df,
        "subgroup_df": subgroup_df,
        "skipped_models": skipped_models,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build a strong-baseline panel across result directories.")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--models", default=",".join(DEFAULT_MODEL_REQUESTS))
    ap.add_argument("--split_ids", default="1")
    ap.add_argument("--subgroup_filter", default="")
    ap.add_argument("--out_root", default="")
    ap.add_argument("--systema_root", default="")
    args = ap.parse_args(argv)

    result = run_baseline_panel(
        dataset=str(args.dataset).strip(),
        models=str(args.models).strip(),
        split_ids=str(args.split_ids).strip(),
        subgroup_filter=str(args.subgroup_filter).strip() or None,
        out_root=str(args.out_root).strip() or None,
        systema_root=str(args.systema_root).strip() or None,
    )
    print(f"out_dir: {result['out_dir']}")
    if not result["ranking_df"].empty:
        print(result["ranking_df"].to_string(index=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
