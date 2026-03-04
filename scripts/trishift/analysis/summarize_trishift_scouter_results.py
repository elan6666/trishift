from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FOCUS_METRICS = [
    "mean_pearson",
    "mean_nmse",
    "mean_scpram_r2_degs_var_mean",
    "mean_systema_corr_20de_allpert",
    "mean_scpram_wasserstein_degs_sum",
]

NORMAN_SUBGROUPS = ["single", "seen0", "seen1", "seen2"]


@dataclass
class ExperimentEntry:
    dataset: str
    model: str
    source_group: str
    label: str
    mean_file: Path


def _is_float(text: str) -> bool:
    try:
        float(text)
        return True
    except Exception:
        return False


def _parse_mean_file(path: Path) -> dict[str, float]:
    metrics: dict[str, float] = {}
    lines = path.read_text(encoding="utf-8").splitlines()
    for i, raw in enumerate(lines):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            if i == 0 and _is_float(line):
                metrics["mean_pearson_line"] = float(line)
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if _is_float(value):
            metrics[key] = float(value)
    return metrics


def _collect_ablation_entries(dataset: str, source_group: str, root: Path) -> list[ExperimentEntry]:
    items: list[ExperimentEntry] = []
    for mean_file in sorted(root.glob("*/mean_pearson.txt")):
        label = mean_file.parent.name
        items.append(
            ExperimentEntry(
                dataset=dataset,
                model="trishift",
                source_group=source_group,
                label=label,
                mean_file=mean_file,
            )
        )
    return items


def _collect_single_entry(
    dataset: str, model: str, source_group: str, label: str, result_dir: Path
) -> ExperimentEntry:
    mean_file = result_dir / "mean_pearson.txt"
    return ExperimentEntry(
        dataset=dataset,
        model=model,
        source_group=source_group,
        label=label,
        mean_file=mean_file,
    )


def _to_dataframe(entries: Iterable[ExperimentEntry]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for entry in entries:
        if not entry.mean_file.exists():
            continue
        metrics = _parse_mean_file(entry.mean_file)
        row: dict[str, object] = {
            "dataset": entry.dataset,
            "model": entry.model,
            "source_group": entry.source_group,
            "label": entry.label,
            "mean_file": str(entry.mean_file),
        }
        row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows)


def _metric_columns(df: pd.DataFrame, exclude_std_suffix: str = "_std") -> list[str]:
    keep: list[str] = []
    for col in df.columns:
        if col == "mean_file":
            continue
        if not col.startswith("mean_"):
            continue
        if col.startswith("mean_pearson_line"):
            continue
        if col.startswith("subgroup_"):
            continue
        if col.endswith(exclude_std_suffix):
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.notna().sum() == 0:
            continue
        keep.append(col)
    preferred = [
        "mean_pearson",
        "mean_nmse",
        "mean_mse_pred",
        "mean_mse_ctrl",
        "mean_systema_corr_all_allpert",
        "mean_systema_corr_20de_allpert",
        "mean_scpram_r2_all_mean_mean",
        "mean_scpram_r2_all_var_mean",
        "mean_scpram_r2_degs_mean_mean",
        "mean_scpram_r2_degs_var_mean",
        "mean_scpram_wasserstein_all_sum",
        "mean_scpram_wasserstein_degs_sum",
    ]
    head = [k for k in preferred if k in keep]
    tail = [k for k in keep if k not in head]
    return head + tail


def _render_bar(df: pd.DataFrame, out_path: Path) -> None:
    if "mean_pearson" not in df.columns or df.empty:
        return
    data = df.sort_values("mean_pearson", ascending=False).reset_index(drop=True)
    colors = data["model"].map({"trishift": "#2E8B57", "scouter": "#1F77B4"}).fillna("#7F7F7F")
    plt.figure(figsize=(max(10, len(data) * 0.6), 6), dpi=220)
    plt.bar(data["label"], data["mean_pearson"], color=colors)
    plt.xticks(rotation=70, ha="right")
    plt.ylabel("mean_pearson")
    plt.title(f"{data['dataset'].iloc[0]} mean_pearson comparison")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _normalize_for_heatmap(values: pd.DataFrame) -> pd.DataFrame:
    out = values.copy()
    for col in out.columns:
        s = out[col].astype(float)
        if s.isna().all():
            out[col] = np.nan
            continue
        vmin = s.min(skipna=True)
        vmax = s.max(skipna=True)
        if pd.isna(vmin) or pd.isna(vmax) or np.isclose(vmax, vmin):
            out[col] = 0.5
            continue
        scaled = (s - vmin) / (vmax - vmin)
        lower_is_better = any(k in col for k in ("nmse", "wasserstein", "mse_pred", "mse_ctrl"))
        out[col] = 1.0 - scaled if lower_is_better else scaled
    return out


def _render_heatmap(df: pd.DataFrame, metrics: list[str], out_path: Path) -> None:
    if not metrics or df.empty:
        return
    table = df.set_index("label")[metrics].copy()
    table = table.dropna(axis=1, how="all")
    if table.empty:
        return
    score = _normalize_for_heatmap(table)
    fig_w = max(8, score.shape[1] * 0.8)
    fig_h = max(6, score.shape[0] * 0.45)
    plt.figure(figsize=(fig_w, fig_h), dpi=220)
    im = plt.imshow(score.values, aspect="auto", cmap="RdYlGn", vmin=0.0, vmax=1.0)
    plt.colorbar(im, label="normalized score (higher is better)")
    plt.xticks(ticks=np.arange(score.shape[1]), labels=score.columns, rotation=70, ha="right")
    plt.yticks(ticks=np.arange(score.shape[0]), labels=score.index)
    plt.title(f"{df['dataset'].iloc[0]} key metrics heatmap (normalized)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _render_heatmap_with_values(
    table: pd.DataFrame, out_path: Path, title: str, lower_is_better_cols: set[str] | None = None
) -> None:
    if table.empty:
        return
    lower_is_better_cols = lower_is_better_cols or set()
    score = _normalize_for_heatmap(table)
    fig_w = max(8, score.shape[1] * 1.1)
    fig_h = max(6, score.shape[0] * 0.5)
    plt.figure(figsize=(fig_w, fig_h), dpi=240)
    im = plt.imshow(score.values, aspect="auto", cmap="RdYlGn", vmin=0.0, vmax=1.0)
    plt.colorbar(im, label="normalized score (higher is better)")
    plt.xticks(ticks=np.arange(score.shape[1]), labels=score.columns, rotation=40, ha="right")
    plt.yticks(ticks=np.arange(score.shape[0]), labels=score.index)
    plt.title(title)
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            v = table.iat[i, j]
            if pd.isna(v):
                txt = "NA"
            else:
                txt = f"{float(v):.4f}"
            bg = score.iat[i, j]
            color = "black" if pd.isna(bg) or bg > 0.45 else "white"
            plt.text(j, i, txt, ha="center", va="center", fontsize=8, color=color)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _write_focus_outputs(df_all: pd.DataFrame, dataset: str, out_dir: Path) -> None:
    ds = df_all[df_all["dataset"] == dataset].copy()
    if ds.empty:
        return
    present_metrics = [m for m in FOCUS_METRICS if m in ds.columns]
    core_cols = ["dataset", "model", "source_group", "label", "mean_file"]
    focus_df = ds[core_cols + present_metrics].copy()
    focus_df = focus_df.sort_values(["model", "label"]).reset_index(drop=True)
    focus_df.to_csv(out_dir / f"{dataset}_focus_metrics.csv", index=False, encoding="utf-8-sig")
    table = focus_df.set_index("label")[present_metrics]
    _render_heatmap_with_values(
        table=table,
        out_path=out_dir / f"{dataset}_focus_metrics_heatmap_values.png",
        title=f"{dataset} focus metrics (with values, no *_std)",
    )


def _build_norman_subgroup_table(df_all: pd.DataFrame) -> pd.DataFrame:
    ds = df_all[df_all["dataset"] == "norman"].copy()
    if ds.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for _, r in ds.iterrows():
        for subgroup in NORMAN_SUBGROUPS:
            one: dict[str, object] = {
                "dataset": "norman",
                "model": r.get("model"),
                "source_group": r.get("source_group"),
                "label": r.get("label"),
                "subgroup": subgroup,
            }
            any_metric = False
            for m in FOCUS_METRICS:
                key = f"subgroup_{subgroup}_{m}"
                value = r.get(key, np.nan)
                one[m] = value
                if pd.notna(value):
                    any_metric = True
            if any_metric:
                rows.append(one)
    return pd.DataFrame(rows)


def _write_norman_subgroup_outputs(df_all: pd.DataFrame, out_dir: Path) -> None:
    subgroup_df = _build_norman_subgroup_table(df_all)
    if subgroup_df.empty:
        return
    subgroup_df = subgroup_df.sort_values(["subgroup", "model", "label"]).reset_index(drop=True)
    subgroup_df.to_csv(out_dir / "norman_subgroup_focus_metrics.csv", index=False, encoding="utf-8-sig")

    for subgroup in NORMAN_SUBGROUPS:
        one = subgroup_df[subgroup_df["subgroup"] == subgroup].copy()
        if one.empty:
            continue
        metrics = [m for m in FOCUS_METRICS if m in one.columns]
        table = one.set_index("label")[metrics]
        _render_heatmap_with_values(
            table=table,
            out_path=out_dir / f"norman_subgroup_{subgroup}_focus_heatmap_values.png",
            title=f"norman subgroup={subgroup} focus metrics (with values)",
        )


def _write_dataset_outputs(df_all: pd.DataFrame, dataset: str, out_dir: Path) -> None:
    ds = df_all[df_all["dataset"] == dataset].copy()
    if ds.empty:
        return
    metrics = _metric_columns(ds, exclude_std_suffix="_std")
    core_cols = ["dataset", "model", "source_group", "label", "mean_file"]
    table = ds[core_cols + [m for m in metrics if m in ds.columns]].copy()
    table = table.sort_values(["model", "label"]).reset_index(drop=True)
    table.to_csv(out_dir / f"{dataset}_summary_no_std.csv", index=False, encoding="utf-8-sig")

    if "mean_pearson" in table.columns:
        ranking = table.sort_values("mean_pearson", ascending=False).reset_index(drop=True)
        ranking.to_csv(out_dir / f"{dataset}_ranking_by_mean_pearson.csv", index=False, encoding="utf-8-sig")

    _render_bar(table, out_dir / f"{dataset}_mean_pearson_bar.png")
    _render_heatmap(table, metrics, out_dir / f"{dataset}_metrics_heatmap_no_std.png")
    _write_focus_outputs(df_all, dataset, out_dir)


def _write_overall_note(df: pd.DataFrame, out_path: Path) -> None:
    lines: list[str] = []
    lines.append("# TriShift vs Scouter Summary (excluding *_std metrics)")
    lines.append("")
    for dataset in sorted(df["dataset"].dropna().unique()):
        ds = df[df["dataset"] == dataset]
        lines.append(f"## {dataset}")
        if "mean_pearson" in ds.columns:
            best_idx = ds["mean_pearson"].idxmax()
            best = ds.loc[best_idx]
            lines.append(
                f"- Best mean_pearson: `{best['label']}` ({best['model']}) = {best['mean_pearson']:.6f}"
            )
            sc = ds[ds["model"] == "scouter"]
            if not sc.empty:
                sc_row = sc.sort_values("mean_pearson", ascending=False).iloc[0]
                lines.append(
                    f"- Scouter mean_pearson: `{sc_row['label']}` = {sc_row['mean_pearson']:.6f}"
                )
                diff = best["mean_pearson"] - sc_row["mean_pearson"]
                lines.append(f"- Best minus Scouter: {diff:+.6f}")
        lines.append(f"- Focus metrics: {', '.join(FOCUS_METRICS)}")
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize TriShift/Scouter mean metrics and visualize.")
    parser.add_argument(
        "--norman-ablation-root",
        default="artifacts/ablation/norman/20260227_144917_custom_sweep_scpram_ot_pd_ktopk_n5",
    )
    parser.add_argument("--norman-scouter-root", default="artifacts/results/scouter/norman")
    parser.add_argument(
        "--adamson-ablation-root", default="artifacts/ablation/adamson/20260218_231523_local9_combo"
    )
    parser.add_argument("--adamson-trishift-root", default="artifacts/results/adamson/best2")
    parser.add_argument("--adamson-scouter-root", default="artifacts/results/scouter/adamson")
    parser.add_argument("--out-dir", default="")
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else Path("artifacts/analysis") / f"{ts}_trishift_scouter_summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    entries: list[ExperimentEntry] = []
    entries.extend(
        _collect_ablation_entries("norman", "norman_ablation_scpram_ot", Path(args.norman_ablation_root))
    )
    entries.append(
        _collect_single_entry(
            dataset="norman",
            model="scouter",
            source_group="norman_scouter_result",
            label="scouter_norman",
            result_dir=Path(args.norman_scouter_root),
        )
    )

    entries.extend(
        _collect_ablation_entries("adamson", "adamson_ablation_local9_combo", Path(args.adamson_ablation_root))
    )
    entries.append(
        _collect_single_entry(
            dataset="adamson",
            model="trishift",
            source_group="adamson_trishift_best2",
            label="trishift_adamson_best2",
            result_dir=Path(args.adamson_trishift_root),
        )
    )
    entries.append(
        _collect_single_entry(
            dataset="adamson",
            model="scouter",
            source_group="adamson_scouter_result",
            label="scouter_adamson",
            result_dir=Path(args.adamson_scouter_root),
        )
    )

    df = _to_dataframe(entries)
    if df.empty:
        raise RuntimeError("No mean_pearson.txt files were parsed.")
    df = df.sort_values(["dataset", "model", "label"]).reset_index(drop=True)
    df.to_csv(out_dir / "all_experiments_raw.csv", index=False, encoding="utf-8-sig")

    for dataset in ("norman", "adamson"):
        _write_dataset_outputs(df, dataset, out_dir)
    _write_norman_subgroup_outputs(df, out_dir)

    _write_overall_note(df, out_dir / "README_summary.md")
    print(f"[done] outputs: {out_dir}")


if __name__ == "__main__":
    main()
