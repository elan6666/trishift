from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import math
import re
import time

import pandas as pd


LOWER_BETTER_SUBSTR = ("nmse", "mse", "rmse", "wasserstein")
EXCLUDE_COLS = {"split_id", "n_ensemble"}


@dataclass(frozen=True)
class RunRow:
    idx: int
    run: str
    run_dir: Path


def _is_lower_better(metric: str) -> bool:
    m = metric.lower()
    return any(s in m for s in LOWER_BETTER_SUBSTR)


def _fmt(x: float | int | None, nd: int = 6) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return ""
    if isinstance(x, int):
        return str(x)
    return f"{float(x):.{nd}f}"


def _discover_runs(sweep_root: Path) -> list[RunRow]:
    runs: list[RunRow] = []
    for p in sorted(sweep_root.iterdir()):
        if not p.is_dir():
            continue
        m = re.match(r"^(\d{2})_", p.name)
        if not m:
            continue
        runs.append(RunRow(idx=int(m.group(1)), run=p.name, run_dir=p))
    return sorted(runs, key=lambda r: r.idx)


def _read_metrics_mean(run_dir: Path) -> dict[str, float] | None:
    p = run_dir / "metrics.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    means = df.mean(numeric_only=True).to_dict()
    # Drop noisy/informational fields.
    for k in list(means.keys()):
        if k in EXCLUDE_COLS:
            means.pop(k, None)
    return {str(k): float(v) for k, v in means.items() if v is not None}


def _write_md_full(
    sweep_root: Path,
    mean_df: pd.DataFrame,
    delta_df: pd.DataFrame,
    baseline_run: str,
    out_name: str,
) -> Path:
    out_path = sweep_root / out_name

    metrics = [c for c in mean_df.columns if c not in {"idx", "run"}]
    now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    lines: list[str] = []
    lines.append(f"# 全指标对比报告（Adamson 自定义单因子消融）")
    lines.append("")
    lines.append("## 1. 数据说明")
    lines.append("")
    lines.append(f"- 目录：`{sweep_root.as_posix()}`")
    lines.append(f"- 生成时间：{now}")
    lines.append(f"- 运行数：{len(mean_df)}")
    lines.append(f"- Baseline：`{baseline_run}`")
    lines.append(f"- 指标数：{len(metrics)}")
    lines.append(
        "- 说明：`nmse` 和 `*wasserstein*` 按“越小越好”，其他指标按“越大越好”。"
    )
    lines.append("")
    lines.append("## 2. 所有指标绝对值对比（全表）")
    lines.append("")
    lines.append(mean_df.to_markdown(index=False))
    lines.append("")
    lines.append("## 3. 相对 Baseline 增益/退化对比（全表）")
    lines.append("")
    lines.append(delta_df.to_markdown(index=False))
    lines.append("")

    # Best run per metric (direction-aware).
    lines.append("## 4. 各指标最优 Run（全指标）")
    lines.append("")
    best_rows = []
    for m in metrics:
        if _is_lower_better(m):
            best = mean_df.sort_values(m, ascending=True).iloc[0]
            direction = "lower_better"
        else:
            best = mean_df.sort_values(m, ascending=False).iloc[0]
            direction = "higher_better"
        best_rows.append(
            {
                "metric": m,
                "direction": direction,
                "best_run": f"`{best['run']}`",
                "best_value": float(best[m]),
            }
        )
    best_df = pd.DataFrame(best_rows)
    if not best_df.empty:
        best_df["best_value"] = best_df["best_value"].map(lambda x: _fmt(x))
    lines.append(best_df.to_markdown(index=False))
    lines.append("")

    # Top3 per metric (direction-aware).
    lines.append("## 5. 各指标前三名（全指标）")
    lines.append("")
    top3_rows = []
    for m in metrics:
        if _is_lower_better(m):
            top = mean_df.sort_values(m, ascending=True).head(3)
            direction = "lower_better"
        else:
            top = mean_df.sort_values(m, ascending=False).head(3)
            direction = "higher_better"
        vals = [f"`{r['run']}` ({_fmt(r[m])})" for _, r in top.iterrows()]
        while len(vals) < 3:
            vals.append("")
        top3_rows.append(
            {"metric": m, "direction": direction, "top1": vals[0], "top2": vals[1], "top3": vals[2]}
        )
    top3_df = pd.DataFrame(top3_rows)
    lines.append(top3_df.to_markdown(index=False))
    lines.append("")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def _write_md_brief(
    sweep_root: Path,
    mean_df: pd.DataFrame,
    baseline_run: str,
    out_name: str,
) -> Path:
    out_path = sweep_root / out_name
    now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    def rank_table(metric: str, lower_better: bool, topn: int = 10) -> str:
        df = mean_df.sort_values(metric, ascending=lower_better).head(topn)[["run", metric]].copy()
        df.insert(0, "rank", range(1, len(df) + 1))
        df[metric] = df[metric].map(lambda x: _fmt(x))
        return df.to_markdown(index=False)

    lines: list[str] = []
    lines.append("# Adamson 单因子消融结果整理与分析（中文）")
    lines.append("")
    lines.append(f"- 结果目录：`{sweep_root.as_posix()}`")
    lines.append(f"- 生成时间：{now}")
    lines.append(f"- 完成 run 数：{len(mean_df)} / {len(mean_df)}")
    lines.append(f"- baseline：`{baseline_run}`")
    lines.append("")

    main_metrics = [
        ("pearson", False),
        ("nmse", True),
        ("systema_corr_all_allpert", False),
        ("systema_corr_20de_allpert", False),
    ]
    for metric, lower_better in main_metrics:
        if metric not in mean_df.columns:
            continue
        lines.append(f"## {metric} 排名（Top10）")
        lines.append("")
        lines.append(rank_table(metric, lower_better, topn=10))
        lines.append("")

    # Full Top3 summary.
    metrics = [c for c in mean_df.columns if c not in {"idx", "run"}]
    lines.append("## 全指标 Top3（按指标方向）")
    lines.append("")
    lines.append("说明：`nmse` 与 `wasserstein` 按“越小越好”，其余指标按“越大越好”。")
    lines.append("")
    rows = []
    for m in metrics:
        lower_better = _is_lower_better(m)
        top = mean_df.sort_values(m, ascending=lower_better).head(3)
        vals = [f"`{r['run']}` ({_fmt(r[m])})" for _, r in top.iterrows()]
        while len(vals) < 3:
            vals.append("")
        rows.append(
            {
                "指标": m,
                "方向": ("越小越好" if lower_better else "越大越好"),
                "第1名": vals[0],
                "第2名": vals[1],
                "第3名": vals[2],
            }
        )
    pd.DataFrame(rows).to_markdown(index=False)
    lines.append(pd.DataFrame(rows).to_markdown(index=False))
    lines.append("")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze a sweep root and write Chinese markdown reports.")
    ap.add_argument("--sweep_root", required=True, help="sweep root containing per-run dirs with metrics.csv")
    ap.add_argument("--baseline_idx", type=int, default=1, help="baseline idx (default 1)")
    args = ap.parse_args()

    sweep_root = Path(args.sweep_root)
    runs = _discover_runs(sweep_root)
    if not runs:
        raise SystemExit(f"No run dirs found under {sweep_root}")

    rows = []
    for r in runs:
        means = _read_metrics_mean(r.run_dir)
        if means is None:
            continue
        row = {"idx": r.idx, "run": r.run}
        row.update(means)
        rows.append(row)

    if not rows:
        raise SystemExit("No metrics.csv found; cannot analyze.")

    mean_df = pd.DataFrame(rows).sort_values("idx").reset_index(drop=True)
    # Ensure deterministic column order: idx/run then metrics.
    metric_cols = [c for c in mean_df.columns if c not in {"idx", "run"}]
    mean_df = mean_df[["idx", "run"] + metric_cols]

    baseline_idx = int(args.baseline_idx)
    base = mean_df[mean_df["idx"] == baseline_idx]
    if base.empty:
        raise SystemExit(f"baseline_idx={baseline_idx} not found in sweep.")
    baseline_run = str(base.iloc[0]["run"])

    # Delta vs baseline (raw delta; direction handling is documented in markdown).
    base_vals = base.iloc[0].to_dict()
    delta_df = mean_df.copy()
    for c in metric_cols:
        delta_df[c] = delta_df[c] - float(base_vals.get(c, 0.0))

    # Persist CSVs (common artifacts used by downstream scripts).
    mean_df.to_csv(sweep_root / "all_runs_mean_metrics.csv", index=False)
    delta_df.to_csv(sweep_root / "all_runs_delta_vs_baseline.csv", index=False)

    _write_md_brief(sweep_root, mean_df, baseline_run, out_name="analysis_results_cn.md")
    _write_md_full(
        sweep_root,
        mean_df,
        delta_df,
        baseline_run,
        out_name="analysis_full_metrics_comparison_cn.md",
    )

    print(str(sweep_root))


if __name__ == "__main__":
    main()
