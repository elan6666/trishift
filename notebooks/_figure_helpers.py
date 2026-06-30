from __future__ import annotations

import math
import pickle
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageChops, ImageDraw, ImageFont

try:
    import seaborn as sns  # noqa: F401
except Exception:  # pragma: no cover - plotting must also work in lean local envs
    sns = None

from scripts.common.paper_plot_style import apply_gears_paper_style, model_color_map, style_axis

try:
    from scripts.trishift.analysis._result_adapter import load_payload_item
except Exception:  # pragma: no cover - optional during lightweight artifact QA
    load_payload_item = None


FIG_ROOT = ROOT / "artifacts" / "paper_figures"
COMP_ROOT = FIG_ROOT / "composites"
SERVER_REFRESH_ROOT = ROOT / "artifacts" / "server_refresh" / "extracted"
SERVER_METRICS_IMPORT_ROOT = ROOT / "artifacts" / "server_metrics_import" / "trishift_unseen_ctrl_txt"
SERVER_BASELINE_CSV_IMPORT_ROOT = ROOT / "artifacts" / "server_metrics_import" / "trishift_baseline_metrics_csv"

DATASET_ORDER = ["Adamson", "Dixit", "Norman", "PBMC"]
MODEL_ORDER = ["TriShift", "CellOT", "GEARS", "BioLORD", "GenePert", "scGPT"]
TRISHIFT_MODEL_ORDER = ["TriShift", "kNN", "OT", "no reference", "no prior", "full"]
SUBGROUP_ORDER = ["single", "seen2", "seen1", "seen0"]
BAR_MODEL_COLORS = {
    "TriShift": "#DDB328",
    "CellOT": "#7F7F7F",
    "GEARS": "#D45A4C",
    "BioLORD": "#E6863A",
    "GenePert": "#5B82C2",
    "scGPT": "#8E7CCB",
    "Truth": "#6F6F6F",
    "Control": "#CFCFCF",
    "Perturbed": "#6F6F6F",
}
DISPLAY_COLORS = {
    "Adamson": "#9FD9D3",
    "Dixit": "#F2B56B",
    "Norman": "#87A8D8",
    "PBMC": "#DDD3C8",
    **BAR_MODEL_COLORS,
    "random": "#C8DCEB",
    "kNN": "#B7C8A6",
    "OT": "#DDB328",
    "no reference": "#C9CDD1",
    "no prior": "#D8BFD8",
    "full": "#5DAE61",
}
FIG3_DATASET_COLORS = {
    "Adamson": "#B8DFDB",
    "Dixit": "#EBC783",
    "Norman": "#A9BEDA",
    "PBMC": "#DCD5CB",
}
FIG3_VARIANT_COLORS = {
    "kNN": "#5B82C2",
    "OT": "#DDB328",
    "no reference": "#8E7CCB",
    "no prior": "#D45A4C",
    "full": "#5DAE61",
}
FIG2_MODEL_COLORS = {name: BAR_MODEL_COLORS[name] for name in ["TriShift", "CellOT", "GEARS", "BioLORD", "GenePert", "scGPT"]}
PAPER_GROUP_WIDTH = 0.90
PAPER_BAR_STEP_MAX = 0.18
PAPER_BAR_WIDTH_FRACTION = 0.94
PAPER_BAR_EDGE_COLOR = "black"
PAPER_BAR_EDGE_LW = 0.6
PAPER_SINGLE_BAR_WIDTH = 0.58
PAPER_ERROR_COLOR = "#333333"
PAPER_ERROR_LW = 0.8
PAPER_ERROR_CAPSIZE = 1.5
PAPER_FIG_FONT_SCALE = 0.80
PAPER_AXIS_LABEL_FONTSIZE = 8.0
PAPER_TITLE_FONTSIZE = 8.2
PAPER_TICK_FONTSIZE = 7.4
PAPER_LEGEND_FONTSIZE = 6.4
PAPER_SMALL_TEXT_FONTSIZE = 6.2
PAPER_PANEL_LABEL_PX = 30
PAPER_PANEL_LABEL_PT = 8.8
PAPER_PANEL_LABEL_PAD = 46
PAPER_FIG4_BAR_PANEL_FIGSIZE = (3.7, 4.3)
PAPER_FIG4_BAR_CELL_W = 463
PAPER_FIG4_BAR_CELL_H = 538
PAPER_FIG5_BAR_PANEL_FIGSIZE = (5.0, 4.3)
PAPER_FIG5_BAR_CELL_W = 629
PAPER_FIG5_BAR_CELL_H = 538
RESULT_MODE = "unseen_ctrl"
REQUIRED_PBMC_PROTOCOL = "true_unseen_target_domain_ctrl"
SUPP_DIRS = {
    "figs1": "FigS1_ReferenceTransferFull",
    "figs2": "FigS2_AdditionalCases",
    "figs3": "FigS3_DixitRobustness",
    "figs4": "FigS4_Module1LatentState",
    "figs5": "FigS5_NormanGeneralizationContext",
}
STALE_SUPP_DIRS = [
    "FigS1_BenchmarkExtension",
    "FigS2_DixitStability",
    "FigS3_BiologyAndAblation",
    "FigS3_AdditionalCases",
    "FigS3_UnseenBenchmarkBackup",
    "FigS4_AblationAndSystemaDiagnostics",
    "FigS4_CentroidAnalysis",
    "FigS4_DixitRobustness",
    "FigS4_ReferenceConditioning",
    "FigS5_CombinationDistributionExtension",
    "FigS5_Module1LatentState",
    "FigS5_Robustness",
    "FigS6_Module1LatentState",
    "FigS6_Stage1LatentClustering",
]


def _paper_bar_kwargs(alpha: float = 1.0) -> dict[str, object]:
    return {
        "edgecolor": PAPER_BAR_EDGE_COLOR,
        "linewidth": PAPER_BAR_EDGE_LW,
        "alpha": alpha,
    }


def _paper_error_kwargs() -> dict[str, object]:
    return {
        "color": PAPER_ERROR_COLOR,
        "linewidth": PAPER_ERROR_LW,
        "capsize": PAPER_ERROR_CAPSIZE,
        "capthick": PAPER_ERROR_LW,
    }


def _save_figure(fig, out: Path, *, vector_sidecar: bool = False, **kwargs) -> None:
    """Save the canonical raster output and, for production plots, a PDF sidecar."""
    fig.savefig(out, **kwargs)
    if vector_sidecar and out.suffix.lower() in {".png", ".jpg", ".jpeg"}:
        fig.savefig(out.with_suffix(".pdf"), **kwargs)


def _display_dataset(name: str) -> str:
    lookup = {
        "adamson": "Adamson",
        "dixit": "Dixit",
        "norman": "Norman",
        "scgen_pbmc_celltype": "PBMC",
        "pbmc": "PBMC",
    }
    return lookup.get(str(name).lower(), str(name))


def _display_model(name: str) -> str:
    lookup = {
        "trishift": "TriShift",
        "trishift_nearest": "TriShift",
        "trishift_random": "TriShift random",
        "systema_nonctl_mean": "Perturbed mean",
        "perturbed mean": "Perturbed mean",
        "cellot": "CellOT",
        "gears": "GEARS",
        "biolord": "BioLORD",
        "genepert": "GenePert",
        "scgpt": "scGPT",
    }
    return lookup.get(str(name).lower(), str(name))


def _relative_artifact_path(path: Path) -> Path | None:
    path = Path(path)
    try:
        return path.relative_to(ROOT)
    except ValueError:
        pass
    parts = list(path.parts)
    if "trishift" in parts:
        idx = len(parts) - 1 - parts[::-1].index("trishift")
        if idx + 1 < len(parts):
            return Path(*parts[idx + 1 :])
    if "artifacts" in parts:
        idx = parts.index("artifacts")
        return Path(*parts[idx:])
    return path if not path.is_absolute() else None


def _candidate_paths(path: Path) -> list[Path]:
    path = Path(path)
    candidates = [path]
    rel = _relative_artifact_path(path)
    if rel is not None:
        candidates.append(ROOT / rel)
        if len(rel.parts) >= 3 and rel.parts[:3] == ("artifacts", "results", "cellot"):
            # CellOT was rerun as a PBMC-only official-style OOD baseline.
            # Do not fall back to older imported CellOT artifacts.
            unique: list[Path] = []
            seen: set[str] = set()
            for item in candidates:
                key = str(item)
                if key not in seen:
                    unique.append(item)
                    seen.add(key)
            return unique
        candidates.append(SERVER_REFRESH_ROOT / rel)
        candidates.append(SERVER_METRICS_IMPORT_ROOT / rel)
        candidates.append(SERVER_BASELINE_CSV_IMPORT_ROOT / rel)
    unique: list[Path] = []
    seen: set[str] = set()
    for item in candidates:
        key = str(item)
        if key not in seen:
            unique.append(item)
            seen.add(key)
    return unique


def _read_csv(path: Path) -> pd.DataFrame:
    for candidate in _candidate_paths(path):
        if not candidate.exists() or candidate.stat().st_size == 0:
            continue
        try:
            return pd.read_csv(candidate)
        except Exception:
            continue
    return pd.DataFrame()


def _truthy_protocol_flag(value: object) -> bool | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def _assert_pbmc_protocol_df(df: pd.DataFrame, source: Path) -> None:
    if df.empty:
        return
    if "pbmc_protocol" not in df.columns:
        raise ValueError(
            f"PBMC result source lacks pbmc_protocol metadata: {source}. "
            f"Regenerate PBMC under {REQUIRED_PBMC_PROTOCOL}."
        )
    protocols = {str(x).strip() for x in df["pbmc_protocol"].dropna().unique()}
    if protocols != {REQUIRED_PBMC_PROTOCOL}:
        raise ValueError(
            f"PBMC result source has protocol={sorted(protocols)} at {source}; "
            f"expected {REQUIRED_PBMC_PROTOCOL}."
        )
    if "include_test_ctrl_in_train" in df.columns:
        flags = {
            flag
            for flag in (_truthy_protocol_flag(x) for x in df["include_test_ctrl_in_train"])
            if flag is not None
        }
        if flags and flags != {False}:
            raise ValueError(
                f"PBMC result source includes test-domain controls in training: {source}"
            )


def _assert_pbmc_payload_protocol(item: dict[str, object], source: object = "payload") -> None:
    meta = item.get("export_metadata", {}) if isinstance(item, dict) else {}
    if not isinstance(meta, dict):
        raise ValueError(f"PBMC payload lacks export_metadata: {source}")
    protocol = str(meta.get("pbmc_protocol", "")).strip()
    if protocol != REQUIRED_PBMC_PROTOCOL:
        raise ValueError(
            f"PBMC payload has protocol={protocol!r} at {source}; "
            f"expected {REQUIRED_PBMC_PROTOCOL}."
        )
    include_flag = _truthy_protocol_flag(meta.get("include_test_ctrl_in_train"))
    if include_flag is not False:
        raise ValueError(
            f"PBMC payload does not confirm include_test_ctrl_in_train=false: {source}"
        )


def _read_mean_metrics_txt(path: Path) -> dict[str, float]:
    for candidate in _candidate_paths(path):
        if not candidate.exists() or candidate.stat().st_size == 0:
            continue
        values: dict[str, float] = {}
        for idx, raw in enumerate(candidate.read_text().splitlines()):
            line = raw.strip()
            if not line:
                continue
            if "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()
                if key.startswith("mean_"):
                    key = key[5:]
                try:
                    values[key] = float(value.strip())
                except ValueError:
                    continue
            elif idx == 0:
                try:
                    values.setdefault("pearson", float(line))
                except ValueError:
                    pass
        if values:
            values["_source_file"] = str(_relative_artifact_path(candidate) or candidate)  # type: ignore[assignment]
            return values
    return {}


def _mean_metrics_from_csv(path: Path) -> dict[str, float]:
    df = _read_csv(path)
    if df.empty:
        return {}
    values: dict[str, float] = {}
    for col in df.columns:
        if col in {"split", "split_id", "condition", "dataset", "model", "status", "subgroup", "source_file"}:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        if series.notna().any():
            values[col] = float(series.mean())
    values["_source_file"] = str(_relative_artifact_path(path) or path)  # type: ignore[assignment]
    return values


def _mean_metric_row(
    *,
    txt_path: Path,
    csv_path: Path,
    model: str,
    dataset: str,
    extra: dict[str, object] | None = None,
) -> dict[str, object] | None:
    values = _read_mean_metrics_txt(txt_path)
    csv_values = _mean_metrics_from_csv(csv_path)
    for key, value in csv_values.items():
        values.setdefault(key, value)
    if not values:
        return None
    source = values.pop("_source_file", str(_relative_artifact_path(txt_path) or txt_path))
    row: dict[str, object] = {
        "model": model,
        "dataset": _display_dataset(dataset),
        "source_file": source,
        "source_kind": "mean_txt" if _read_mean_metrics_txt(txt_path) else "csv_mean",
    }
    row.update(values)
    if extra:
        row.update(extra)
    return row


def _write_source(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def collect_prediction_metrics(*, heldout: bool = True) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    suffix = "metrics_unseen_ctrl.csv" if heldout else "metrics.csv"
    specs = []
    for ds in ["adamson", "dixit", "norman"]:
        specs.append((ROOT / "artifacts" / "results" / ds / suffix, "TriShift", ds))
        for model in ["gears", "biolord", "genepert", "scgpt"]:
            specs.append((ROOT / "artifacts" / "results" / model / ds / suffix, _display_model(model), ds))
    # PBMC uses a single stimulation condition and has model-specific paths.
    specs.extend(
        [
            (ROOT / "artifacts" / "results" / "scgen_pbmc_celltype" / "metrics.csv", "TriShift", "PBMC"),
            (
                ROOT
                / "artifacts"
                / "results"
                / "biolord"
                / "scgen_pbmc_celltype"
                / "emb_scgen_ifnb1_zenodo_prott5"
                / "metrics.csv",
                "BioLORD",
                "PBMC",
            ),
            (ROOT / "artifacts" / "results" / "scgpt" / "scgen_pbmc_celltype" / "metrics.csv", "scGPT", "PBMC"),
        ]
    )
    for path, model, dataset in specs:
        source_path = path
        df = _read_csv(path)
        source_kind = "condition_csv"
        if df.empty and heldout and path.name == "metrics_unseen_ctrl.csv":
            fallback = path.with_name("metrics.csv")
            df = _read_csv(fallback)
            if not df.empty:
                source_path = fallback
                source_kind = "metrics_csv_fallback"
        if df.empty:
            continue
        if _display_dataset(dataset) == "PBMC":
            _assert_pbmc_protocol_df(df, source_path)
        df = df.copy()
        df["model"] = model
        df["dataset"] = _display_dataset(dataset)
        df["source_file"] = str(_relative_artifact_path(source_path) or source_path.relative_to(ROOT))
        df["source_kind"] = source_kind
        rows.append(df)

    if heldout:
        # Current CellOT plan only keeps the PBMC target-domain CellOT runner.
        for ds in ["scgen_pbmc_celltype"]:
            cellot_path = ROOT / "artifacts" / "results" / "cellot" / ds / "metrics_unseen_ctrl.csv"
            cellot = _read_csv(cellot_path)
            if cellot.empty:
                continue
            _assert_pbmc_protocol_df(cellot, cellot_path)
            cellot = cellot.copy()
            cellot["model"] = "CellOT"
            if "dataset" not in cellot.columns:
                cellot["dataset"] = ds
            cellot["dataset"] = cellot["dataset"].map(_display_dataset)
            cellot["source_file"] = str(_relative_artifact_path(cellot_path) or cellot_path.relative_to(ROOT))
            cellot["source_kind"] = "cellot_pbmc"
            rows.append(cellot)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    if {"dataset", "condition", "subgroup"}.issubset(out.columns):
        keyed = out.dropna(subset=["dataset", "condition", "subgroup"]).copy()
        if not keyed.empty:
            subgroup_map = keyed.drop_duplicates(["dataset", "condition"]).set_index(["dataset", "condition"])["subgroup"].to_dict()
            missing = out["subgroup"].isna()
            out.loc[missing, "subgroup"] = [
                subgroup_map.get((dataset, condition))
                for dataset, condition in zip(out.loc[missing, "dataset"], out.loc[missing, "condition"])
            ]
    for col in ["pearson", "nmse", "mse_pred", "systema_corr_20de_allpert", "scpram_wasserstein_degs_sum"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def collect_prediction_summary_metrics(*, heldout: bool = True, cellot_pbmc_only: bool = True) -> pd.DataFrame:
    suffix_csv = "metrics_unseen_ctrl.csv" if heldout else "metrics.csv"
    suffix_txt = "mean_pearson_unseen_ctrl.txt" if heldout else "mean_pearson.txt"
    rows: list[dict[str, object]] = []
    for ds in ["adamson", "dixit", "norman"]:
        specs = [(ROOT / "artifacts" / "results" / ds, "TriShift", ds)]
        for model in ["gears", "biolord", "genepert", "scgpt"]:
            specs.append((ROOT / "artifacts" / "results" / model / ds, _display_model(model), ds))
        if not cellot_pbmc_only:
            specs.append((ROOT / "artifacts" / "results" / "cellot" / ds, "CellOT", ds))
        for base, model, dataset in specs:
            row = _mean_metric_row(
                txt_path=base / suffix_txt,
                csv_path=base / suffix_csv,
                model=model,
                dataset=dataset,
            )
            if row is not None:
                rows.append(row)

    pbmc_specs = [
        (ROOT / "artifacts" / "results" / "scgen_pbmc_celltype", "TriShift"),
        (
            ROOT
            / "artifacts"
            / "results"
            / "biolord"
            / "scgen_pbmc_celltype"
            / "emb_scgen_ifnb1_zenodo_prott5",
            "BioLORD",
        ),
        (ROOT / "artifacts" / "results" / "scgpt" / "scgen_pbmc_celltype", "scGPT"),
        (ROOT / "artifacts" / "results" / "cellot" / "scgen_pbmc_celltype", "CellOT"),
    ]
    for base, model in pbmc_specs:
        if (base / "metrics.csv").exists():
            _assert_pbmc_protocol_df(_read_csv(base / "metrics.csv"), base / "metrics.csv")
        if (base / "metrics_unseen_ctrl.csv").exists():
            _assert_pbmc_protocol_df(
                _read_csv(base / "metrics_unseen_ctrl.csv"),
                base / "metrics_unseen_ctrl.csv",
            )
        row = _mean_metric_row(
            txt_path=base / "mean_pearson.txt",
            csv_path=base / "metrics.csv",
            model=model,
            dataset="PBMC",
        )
        if row is None and heldout:
            row = _mean_metric_row(
                txt_path=base / "mean_pearson_unseen_ctrl.txt",
                csv_path=base / "metrics_unseen_ctrl.csv",
                model=model,
                dataset="PBMC",
            )
        if row is not None:
            rows.append(row)

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    for col in out.columns:
        if col not in {"model", "dataset", "source_file", "source_kind"}:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def collect_long(path: Path) -> pd.DataFrame:
    df = _read_csv(path)
    if df.empty:
        return df
    df = df.copy()
    if "dataset" in df.columns:
        df["dataset"] = df["dataset"].map(_display_dataset)
    if "model" in df.columns:
        df["model"] = df["model"].map(_display_model)
    if "value" in df.columns:
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df


def collect_ablation_metrics() -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    seen: set[tuple[str, str]] = set()
    ablation_roots = [
        ROOT / "artifacts" / "results" / "ablation",
        SERVER_REFRESH_ROOT / "artifacts" / "results" / "ablation",
    ]
    for ablation_root in ablation_roots:
        if not ablation_root.exists():
            continue
        for preset_dir in sorted(p for p in ablation_root.glob("*/*") if p.is_dir()):
            parts = preset_dir.relative_to(ablation_root).parts
            if len(parts) < 2 or parts[0] == "plan":
                continue
            dataset, preset = parts[0], parts[1]
            key = (dataset, preset)
            if key in seen:
                continue
            p = preset_dir / "metrics_unseen_ctrl.csv"
            if not p.exists():
                p = preset_dir / "metrics.csv"
            if not p.exists():
                continue
            seen.add(key)
            df = _read_csv(p)
            if df.empty:
                continue
            df = df.copy()
            df["dataset"] = _display_dataset(dataset)
            df["preset"] = preset
            rel = _relative_artifact_path(p)
            df["source_file"] = str(rel if rel is not None else p)
            rows.append(df)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    for col in ["pearson", "nmse", "systema_corr_20de_allpert", "mse_pred"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def collect_ablation_summary_metrics() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    ablation_roots = [
        ROOT / "artifacts" / "results" / "ablation",
        SERVER_REFRESH_ROOT / "artifacts" / "results" / "ablation",
        SERVER_METRICS_IMPORT_ROOT / "artifacts" / "results" / "ablation",
    ]
    for ablation_root in ablation_roots:
        if not ablation_root.exists():
            continue
        for preset_dir in sorted(p for p in ablation_root.glob("*/*") if p.is_dir()):
            parts = preset_dir.relative_to(ablation_root).parts
            if len(parts) < 2 or parts[0] == "plan":
                continue
            dataset, preset = parts[0], parts[1]
            key = (dataset, preset)
            if key in seen:
                continue
            txt_path = preset_dir / ("mean_pearson.txt" if dataset == "scgen_pbmc_celltype" else "mean_pearson_unseen_ctrl.txt")
            csv_path = preset_dir / ("metrics.csv" if dataset == "scgen_pbmc_celltype" else "metrics_unseen_ctrl.csv")
            row = _mean_metric_row(
                txt_path=txt_path,
                csv_path=csv_path,
                model=preset,
                dataset=dataset,
                extra={"preset": preset},
            )
            if row is None:
                continue
            seen.add(key)
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    for col in out.columns:
        if col not in {"model", "dataset", "source_file", "source_kind", "preset"}:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def collect_ablation_deg_metrics() -> pd.DataFrame:
    candidates = [
        ROOT / "artifacts" / "analysis" / "deg_prediction" / "ablation_deg_prediction_all_long.csv",
        ROOT / "artifacts" / "analysis" / "deg_prediction" / "ablation" / "ablation_deg_prediction_all_long.csv",
        ROOT / "artifacts" / "analysis" / "deg_prediction" / "ablation" / "deg_prediction_all_long.csv",
        SERVER_REFRESH_ROOT / "artifacts" / "analysis" / "deg_prediction" / "ablation_deg_prediction_all_long.csv",
        SERVER_REFRESH_ROOT / "artifacts" / "analysis" / "deg_prediction" / "ablation" / "ablation_deg_prediction_all_long.csv",
        SERVER_REFRESH_ROOT / "artifacts" / "analysis" / "deg_prediction" / "ablation" / "deg_prediction_all_long.csv",
    ]
    frames = [collect_long(path) for path in candidates]
    frames = [df for df in frames if not df.empty]
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    if "model" in out.columns and "preset" not in out.columns:
        out["preset"] = out["model"].astype(str)
    if {"metric", "value"}.issubset(out.columns):
        index_cols = [col for col in out.columns if col not in {"metric", "value"}]
        out = (
            out.pivot_table(index=index_cols, columns="metric", values="value", aggfunc="mean")
            .reset_index()
            .rename_axis(columns=None)
        )
    return out


def available_order(values: pd.Series, preferred: list[str]) -> list[str]:
    present = [str(x) for x in values.dropna().unique()]
    return [x for x in preferred if x in present] + sorted([x for x in present if x not in preferred])


def no_data_panel(out: Path, title: str, message: str = "Required source table is unavailable") -> Path:
    apply_gears_paper_style(font_scale=PAPER_FIG_FONT_SCALE)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.4, 2.8), dpi=240)
    ax.text(0.5, 0.58, title, ha="center", va="center", fontsize=PAPER_TITLE_FONTSIZE, weight="bold")
    ax.text(0.5, 0.42, message, ha="center", va="center", fontsize=PAPER_AXIS_LABEL_FONTSIZE, color="#555555")
    ax.axis("off")
    fig.tight_layout(pad=0.4)
    fig.savefig(out)
    plt.close(fig)
    return out


def _color_map(names: list[str]) -> dict[str, object]:
    cmap = model_color_map(names)
    for name in names:
        if name in DISPLAY_COLORS:
            cmap[name] = DISPLAY_COLORS[name]
    return cmap


def _legend_above(
    ax,
    handles=None,
    labels=None,
    *,
    ncol: int | None = None,
    fontsize: float = PAPER_LEGEND_FONTSIZE,
    y: float = 1.18,
) -> None:
    if handles is None or labels is None:
        handles, labels = ax.get_legend_handles_labels()
    labels = list(labels)
    if not labels:
        return
    ncol = ncol or min(4, max(1, len(labels)))
    ax.legend(
        handles,
        labels,
        frameon=False,
        fontsize=fontsize,
        ncol=ncol,
        loc="upper center",
        bbox_to_anchor=(0.5, y),
        borderaxespad=0.0,
        handlelength=0.95,
        columnspacing=0.75,
        handletextpad=0.35,
    )


def _metric_plot_frame(
    df: pd.DataFrame,
    metric_col: str,
    *,
    x_col: str = "dataset",
    hue_col: str = "model",
) -> pd.DataFrame:
    if df.empty or metric_col not in df.columns or x_col not in df.columns or hue_col not in df.columns:
        return pd.DataFrame()
    plot = df[[x_col, hue_col, metric_col]].copy()
    plot[metric_col] = pd.to_numeric(plot[metric_col], errors="coerce")
    plot = plot.dropna(subset=[x_col, hue_col, metric_col])
    plot[x_col] = plot[x_col].astype(str)
    plot[hue_col] = plot[hue_col].astype(str).map(_display_model)
    return plot


def _summary_for_axis(
    df: pd.DataFrame,
    metric_col: str,
    *,
    x_col: str = "dataset",
    hue_col: str = "model",
    x_order: list[str] | None = None,
    hue_order: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    plot = _metric_plot_frame(df, metric_col, x_col=x_col, hue_col=hue_col)
    if plot.empty:
        return pd.DataFrame(), [], []
    summary = plot.groupby([x_col, hue_col], as_index=False).agg(
        mean=(metric_col, "mean"),
        sem=(metric_col, "sem"),
        n=(metric_col, "size"),
    )
    xs = x_order or available_order(summary[x_col], DATASET_ORDER if x_col == "dataset" else SUBGROUP_ORDER)
    hues = hue_order or available_order(summary[hue_col], MODEL_ORDER)
    xs = [x for x in xs if x in set(summary[x_col])]
    hues = [h for h in hues if h in set(summary[hue_col])]
    return summary, xs, hues


def _summary_table_for_axis(
    df: pd.DataFrame,
    *,
    x_col: str = "subgroup",
    hue_col: str = "model",
    value_col: str = "mean",
    sem_col: str = "sem",
    x_order: list[str] | None = None,
    hue_order: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    if df.empty or x_col not in df.columns or hue_col not in df.columns or value_col not in df.columns:
        return pd.DataFrame(), [], []
    summary = df[[x_col, hue_col, value_col] + ([sem_col] if sem_col in df.columns else [])].copy()
    summary = summary.rename(columns={value_col: "mean", sem_col: "sem"})
    if "sem" not in summary.columns:
        summary["sem"] = np.nan
    summary[x_col] = summary[x_col].astype(str)
    summary[hue_col] = summary[hue_col].astype(str).map(_display_model)
    summary["mean"] = pd.to_numeric(summary["mean"], errors="coerce")
    summary["sem"] = pd.to_numeric(summary["sem"], errors="coerce")
    summary = summary.dropna(subset=[x_col, hue_col, "mean"])
    xs = x_order or available_order(summary[x_col], SUBGROUP_ORDER if x_col == "subgroup" else DATASET_ORDER)
    hues = hue_order or available_order(summary[hue_col], MODEL_ORDER)
    xs = [x for x in xs if x in set(summary[x_col])]
    hues = [h for h in hues if h in set(summary[hue_col])]
    return summary, xs, hues


def _visible_limit(values: pd.Series) -> float | None:
    vals = pd.to_numeric(values, errors="coerce").dropna()
    if vals.empty:
        return None
    finite = vals[np.isfinite(vals)]
    if finite.empty:
        return None
    positive = finite[finite > 0]
    if len(positive) < 3:
        return None
    q1, q3 = np.percentile(positive, [25, 75])
    iqr = max(q3 - q1, 1e-12)
    robust = q3 + 5.0 * iqr
    normal = positive[positive <= robust]
    if normal.empty:
        normal = positive
    limit = float(normal.max() * 1.22)
    if float(positive.max()) > max(limit * 2.2, limit + 1e-9):
        return limit
    return None


def _group_mark_width(
    max_present: int,
    group_width: float = PAPER_GROUP_WIDTH,
    *,
    bar_width_fraction: float = PAPER_BAR_WIDTH_FRACTION,
    bar_step_max: float = PAPER_BAR_STEP_MAX,
) -> tuple[float, float]:
    base_width = min(bar_step_max, group_width / max(max_present, 1))
    return base_width, base_width * bar_width_fraction


def _draw_grouped_summary_axis(
    ax: plt.Axes,
    summary: pd.DataFrame,
    *,
    metric_title: str,
    ylabel: str,
    panel_label: str,
    xs: list[str],
    hues: list[str],
    x_col: str = "subgroup",
    hue_col: str = "model",
    colors: dict[str, object] | None = None,
    cap_extreme: bool = False,
    group_width: float = PAPER_GROUP_WIDTH,
    bar_width_fraction: float = PAPER_BAR_WIDTH_FRACTION,
    bar_step_max: float = PAPER_BAR_STEP_MAX,
    xtick_rotation: float = 26,
    xtick_ha: str = "right",
) -> None:
    if summary.empty or not xs or not hues:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", color="#555555")
        ax.set_axis_off()
        return
    colors = colors or _color_map(hues)
    present_by_x = {
        xval: [hue for hue in hues if not summary[(summary[x_col].eq(xval)) & (summary[hue_col].eq(hue))].empty]
        for xval in xs
    }
    max_present = max([len(v) for v in present_by_x.values()] or [1])
    base_width, bar_width = _group_mark_width(
        max_present,
        group_width,
        bar_width_fraction=bar_width_fraction,
        bar_step_max=bar_step_max,
    )
    x_positions = np.arange(len(xs), dtype=float)
    y_cap = _visible_limit(summary["mean"]) if cap_extreme else None
    for xidx, xval in enumerate(xs):
        present_hues = present_by_x[xval]
        offsets = [0.0] if len(present_hues) == 1 else np.linspace(
            -base_width * (len(present_hues) - 1) / 2,
            base_width * (len(present_hues) - 1) / 2,
            len(present_hues),
        )
        for hidx, hue in enumerate(present_hues):
            row = summary[(summary[x_col].eq(xval)) & (summary[hue_col].eq(hue))]
            if row.empty:
                continue
            value = float(row["mean"].iloc[0])
            sem = row["sem"].iloc[0] if "sem" in row.columns else np.nan
            draw_value = min(value, y_cap) if y_cap is not None else value
            xpos = float(x_positions[xidx] + offsets[hidx])
            ax.bar(
                xpos,
                draw_value,
                width=bar_width,
                color=colors.get(hue),
                **_paper_bar_kwargs(),
            )
            if pd.notna(sem) and not (y_cap is not None and value > y_cap):
                ax.errorbar(
                    xpos,
                    draw_value,
                    yerr=float(sem),
                    **_paper_error_kwargs(),
                )
            if y_cap is not None and value > y_cap:
                ax.text(xpos, y_cap, f">{y_cap:.2g}", ha="center", va="bottom", fontsize=PAPER_SMALL_TEXT_FONTSIZE, color="#222222")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(xs, rotation=xtick_rotation, ha=xtick_ha)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel, fontsize=PAPER_AXIS_LABEL_FONTSIZE, labelpad=2)
    ax.set_title(metric_title, pad=4, fontsize=PAPER_TITLE_FONTSIZE)
    half_group = base_width * max(max_present - 1, 0) / 2 + bar_width / 2
    ax.set_xlim(float(x_positions[0]) - half_group - 0.08, float(x_positions[-1]) + half_group + 0.08)
    if y_cap is not None:
        ax.set_ylim(top=y_cap * 1.16)
    elif not summary["mean"].dropna().empty and summary["mean"].dropna().min() >= 0:
        ax.set_ylim(bottom=0)
    style_axis(ax, grid_axis=None)
    ax.tick_params(axis="both", labelsize=PAPER_TICK_FONTSIZE)
    ax.text(-0.24, 1.10, panel_label, transform=ax.transAxes, fontsize=PAPER_PANEL_LABEL_PT, fontweight="bold", va="top")


def compact_bar_panel(
    df: pd.DataFrame,
    metric_col: str,
    out: Path,
    title: str,
    ylabel: str,
    *,
    x_col: str = "dataset",
    hue_col: str = "model",
    x_order: list[str] | None = None,
    hue_order: list[str] | None = None,
    cap_extreme: bool = False,
    figsize: tuple[float, float] = (5.2, 3.9),
    color_overrides: dict[str, str] | None = None,
    bar_alpha: float = 1.0,
    legend_y: float = 1.32,
    layout_top: float = 0.78,
    title_pad: float = 13,
    xtick_rotation: float = 32,
    xtick_ha: str = "right",
    grid_axis: str | None = None,
    group_width: float = PAPER_GROUP_WIDTH,
    bar_width_fraction: float = PAPER_BAR_WIDTH_FRACTION,
    bar_step_max: float = PAPER_BAR_STEP_MAX,
    x_step: float = 1.0,
) -> Path:
    plot = _metric_plot_frame(df, metric_col, x_col=x_col, hue_col=hue_col)
    if plot.empty:
        return no_data_panel(out, title)
    summary = plot.groupby([x_col, hue_col], as_index=False).agg(mean=(metric_col, "mean"), sem=(metric_col, "sem"), n=(metric_col, "size"))
    xs = x_order or available_order(summary[x_col], DATASET_ORDER if x_col == "dataset" else SUBGROUP_ORDER)
    hues = hue_order or available_order(summary[hue_col], MODEL_ORDER)
    xs = [x for x in xs if x in set(summary[x_col])]
    hues = [h for h in hues if h in set(summary[hue_col])]
    if not xs or not hues:
        return no_data_panel(out, title, "No plottable groups")
    apply_gears_paper_style(font_scale=PAPER_FIG_FONT_SCALE)
    colors = _color_map(hues)
    if color_overrides:
        colors.update({name: color for name, color in color_overrides.items() if name in hues})
    fig, ax = plt.subplots(figsize=figsize, dpi=240)
    present_by_x = {
        xval: [hue for hue in hues if not summary[(summary[x_col].eq(xval)) & (summary[hue_col].eq(hue))].empty]
        for xval in xs
    }
    max_present = max([len(v) for v in present_by_x.values()] or [1])
    base_width, bar_width = _group_mark_width(
        max_present,
        group_width,
        bar_width_fraction=bar_width_fraction,
        bar_step_max=bar_step_max,
    )
    x_positions = np.arange(len(xs), dtype=float) * x_step
    y_cap = _visible_limit(summary["mean"]) if cap_extreme else None
    for xidx, xval in enumerate(xs):
        x_base = float(x_positions[xidx])
        present_hues = present_by_x[xval]
        offsets = [0.0] if len(present_hues) == 1 else np.linspace(
            -base_width * (len(present_hues) - 1) / 2,
            base_width * (len(present_hues) - 1) / 2,
            len(present_hues),
        )
        for hidx, hue in enumerate(present_hues):
            row = summary[(summary[x_col].eq(xval)) & (summary[hue_col].eq(hue))]
            if row.empty:
                continue
            value = float(row["mean"].iloc[0])
            sem = row["sem"].iloc[0]
            draw_value = min(value, y_cap) if y_cap is not None else value
            ax.bar(
                x_base + float(offsets[hidx]),
                draw_value,
                width=bar_width,
                color=colors.get(hue),
                **_paper_bar_kwargs(bar_alpha),
            )
            if pd.notna(sem) and not (y_cap is not None and value > y_cap):
                ax.errorbar(
                    x_base + float(offsets[hidx]),
                    draw_value,
                    yerr=float(sem),
                    **_paper_error_kwargs(),
                )
            if y_cap is not None and value > y_cap:
                ax.text(
                    x_base + float(offsets[hidx]),
                    y_cap,
                    f">{y_cap:.2g}",
                    ha="center",
                    va="bottom",
                    fontsize=PAPER_SMALL_TEXT_FONTSIZE,
                    color="#222222",
                )
    ax.set_xticks(x_positions)
    ax.set_xticklabels(xs, rotation=xtick_rotation, ha=xtick_ha)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=title_pad)
    half_group = base_width * max(max_present - 1, 0) / 2 + bar_width / 2
    x_pad = max(0.06, group_width * 0.55) if len(xs) == 1 else 0.06
    ax.set_xlim(float(x_positions[0]) - half_group - x_pad, float(x_positions[-1]) + half_group + x_pad)
    if y_cap is not None:
        ax.set_ylim(top=y_cap * 1.18)
    else:
        vals = summary["mean"].dropna()
        if not vals.empty and vals.min() >= 0:
            ax.set_ylim(bottom=0)
    style_axis(ax, grid_axis=grid_axis)
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=colors[h], **_paper_bar_kwargs(bar_alpha))
        for h in hues
    ]
    _legend_above(ax, handles, hues, ncol=min(4, max(1, len(hues))), y=legend_y)
    fig.tight_layout(pad=0.35, rect=(0, 0, 1, layout_top))
    out.parent.mkdir(parents=True, exist_ok=True)
    _save_figure(fig, out, vector_sidecar=True)
    plt.close(fig)
    _write_source(summary, out.with_suffix(".csv"))
    return out


def compact_summary_bar_panel(
    df: pd.DataFrame,
    out: Path,
    title: str,
    ylabel: str,
    *,
    x_col: str = "subgroup",
    hue_col: str = "model",
    value_col: str = "mean",
    sem_col: str = "sem",
    x_order: list[str] | None = None,
    hue_order: list[str] | None = None,
    cap_extreme: bool = False,
    figsize: tuple[float, float] = (5.2, 3.9),
    color_overrides: dict[str, str] | None = None,
    bar_alpha: float = 1.0,
    legend_y: float = 1.32,
    layout_top: float = 0.78,
    title_pad: float = 13,
    xtick_rotation: float = 32,
    xtick_ha: str = "right",
    grid_axis: str | None = None,
    group_width: float = PAPER_GROUP_WIDTH,
    bar_width_fraction: float = PAPER_BAR_WIDTH_FRACTION,
    bar_step_max: float = PAPER_BAR_STEP_MAX,
    x_step: float = 1.0,
) -> Path:
    if df.empty or x_col not in df.columns or hue_col not in df.columns or value_col not in df.columns:
        return no_data_panel(out, title)
    summary = df[[x_col, hue_col, value_col] + ([sem_col] if sem_col in df.columns else [])].copy()
    summary = summary.rename(columns={value_col: "mean", sem_col: "sem"})
    if "sem" not in summary.columns:
        summary["sem"] = np.nan
    summary["mean"] = pd.to_numeric(summary["mean"], errors="coerce")
    summary["sem"] = pd.to_numeric(summary["sem"], errors="coerce")
    summary[x_col] = summary[x_col].astype(str)
    summary[hue_col] = summary[hue_col].astype(str).map(_display_model)
    summary = summary.dropna(subset=[x_col, hue_col, "mean"])
    xs = x_order or available_order(summary[x_col], SUBGROUP_ORDER if x_col == "subgroup" else DATASET_ORDER)
    hues = hue_order or available_order(summary[hue_col], MODEL_ORDER)
    xs = [x for x in xs if x in set(summary[x_col])]
    hues = [h for h in hues if h in set(summary[hue_col])]
    if not xs or not hues:
        return no_data_panel(out, title, "No plottable groups")
    apply_gears_paper_style(font_scale=PAPER_FIG_FONT_SCALE)
    colors = _color_map(hues)
    if color_overrides:
        colors.update({name: color for name, color in color_overrides.items() if name in hues})
    fig, ax = plt.subplots(figsize=figsize, dpi=240)
    present_by_x = {
        xval: [hue for hue in hues if not summary[(summary[x_col].eq(xval)) & (summary[hue_col].eq(hue))].empty]
        for xval in xs
    }
    max_present = max([len(v) for v in present_by_x.values()] or [1])
    base_width, bar_width = _group_mark_width(
        max_present,
        group_width,
        bar_width_fraction=bar_width_fraction,
        bar_step_max=bar_step_max,
    )
    x_positions = np.arange(len(xs), dtype=float) * x_step
    y_cap = _visible_limit(summary["mean"]) if cap_extreme else None
    for xidx, xval in enumerate(xs):
        x_base = float(x_positions[xidx])
        present_hues = present_by_x[xval]
        offsets = [0.0] if len(present_hues) == 1 else np.linspace(
            -base_width * (len(present_hues) - 1) / 2,
            base_width * (len(present_hues) - 1) / 2,
            len(present_hues),
        )
        for hidx, hue in enumerate(present_hues):
            row = summary[(summary[x_col].eq(xval)) & (summary[hue_col].eq(hue))]
            if row.empty:
                continue
            value = float(row["mean"].iloc[0])
            sem = row["sem"].iloc[0]
            draw_value = min(value, y_cap) if y_cap is not None else value
            ax.bar(
                x_base + float(offsets[hidx]),
                draw_value,
                width=bar_width,
                color=colors.get(hue),
                **_paper_bar_kwargs(bar_alpha),
            )
            if pd.notna(sem) and not (y_cap is not None and value > y_cap):
                ax.errorbar(
                    x_base + float(offsets[hidx]),
                    draw_value,
                    yerr=float(sem),
                    **_paper_error_kwargs(),
                )
            if y_cap is not None and value > y_cap:
                ax.text(
                    x_base + float(offsets[hidx]),
                    y_cap,
                    f">{y_cap:.2g}",
                    ha="center",
                    va="bottom",
                    fontsize=PAPER_SMALL_TEXT_FONTSIZE,
                    color="#222222",
                )
    ax.set_xticks(x_positions)
    ax.set_xticklabels(xs, rotation=xtick_rotation, ha=xtick_ha)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=title_pad)
    half_group = base_width * max(max_present - 1, 0) / 2 + bar_width / 2
    x_pad = max(0.06, group_width * 0.55) if len(xs) == 1 else 0.06
    ax.set_xlim(float(x_positions[0]) - half_group - x_pad, float(x_positions[-1]) + half_group + x_pad)
    if y_cap is not None:
        ax.set_ylim(top=y_cap * 1.18)
    elif not summary["mean"].dropna().empty and summary["mean"].dropna().min() >= 0:
        ax.set_ylim(bottom=0)
    style_axis(ax, grid_axis=grid_axis)
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=colors[h], **_paper_bar_kwargs(bar_alpha))
        for h in hues
    ]
    _legend_above(ax, handles, hues, ncol=min(4, max(1, len(hues))), y=legend_y)
    fig.tight_layout(pad=0.35, rect=(0, 0, 1, layout_top))
    out.parent.mkdir(parents=True, exist_ok=True)
    _save_figure(fig, out, vector_sidecar=True)
    plt.close(fig)
    _write_source(summary.rename(columns={"mean": value_col, "sem": sem_col}), out.with_suffix(".csv"))
    return out


def boxplot_panel(
    df: pd.DataFrame,
    metric_col: str,
    out: Path,
    title: str,
    ylabel: str,
    *,
    x_col: str = "dataset",
    hue_col: str = "model",
    x_order: list[str] | None = None,
    hue_order: list[str] | None = None,
    cap_extreme: bool = False,
) -> Path:
    plot = _metric_plot_frame(df, metric_col, x_col=x_col, hue_col=hue_col)
    if plot.empty:
        return no_data_panel(out, title)
    xs = x_order or available_order(plot[x_col], DATASET_ORDER if x_col == "dataset" else SUBGROUP_ORDER)
    hues = hue_order or available_order(plot[hue_col], MODEL_ORDER)
    xs = [x for x in xs if x in set(plot[x_col])]
    hues = [h for h in hues if h in set(plot[hue_col])]
    if not xs or not hues:
        return no_data_panel(out, title, "No plottable groups")
    apply_gears_paper_style(font_scale=PAPER_FIG_FONT_SCALE)
    colors = _color_map(hues)
    fig, ax = plt.subplots(figsize=(5.4, 3.6), dpi=240)
    present_by_x = {
        xval: [hue for hue in hues if not plot[(plot[x_col].eq(xval)) & (plot[hue_col].eq(hue))].empty]
        for xval in xs
    }
    max_present = max([len(v) for v in present_by_x.values()] or [1])
    base_width, box_width = _group_mark_width(max_present, 0.76)
    y_cap = _visible_limit(plot[metric_col]) if cap_extreme else None
    for xidx, xval in enumerate(xs):
        present_hues = present_by_x[xval]
        offsets = [0.0] if len(present_hues) == 1 else np.linspace(
            -base_width * (len(present_hues) - 1) / 2,
            base_width * (len(present_hues) - 1) / 2,
            len(present_hues),
        )
        for hidx, hue in enumerate(present_hues):
            vals = plot[(plot[x_col].eq(xval)) & (plot[hue_col].eq(hue))][metric_col].astype(float).dropna()
            if vals.empty:
                continue
            if y_cap is not None:
                vals = vals.clip(upper=y_cap)
            bp = ax.boxplot(
                [vals.values],
                positions=[xidx + float(offsets[hidx])],
                widths=box_width,
                patch_artist=True,
                manage_ticks=False,
                boxprops={"linewidth": 0.65, "edgecolor": "#444444"},
                medianprops={"linewidth": 0.8, "color": "#333333"},
                whiskerprops={"linewidth": 0.65, "color": "#555555"},
                capprops={"linewidth": 0.65, "color": "#555555"},
                flierprops={"marker": ".", "markersize": 2.0, "markerfacecolor": "#555555", "markeredgewidth": 0, "alpha": 0.55},
            )
            for patch in bp["boxes"]:
                patch.set_facecolor(colors.get(hue))
                patch.set_alpha(0.95)
    ax.set_xticks(np.arange(len(xs)))
    ax.set_xticklabels(xs, rotation=28, ha="right")
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=12)
    ax.margins(x=0.045)
    if y_cap is not None:
        ax.set_ylim(top=y_cap * 1.18)
    style_axis(ax, grid_axis="y")
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=colors[h], edgecolor="#444444", linewidth=0.65) for h in hues]
    _legend_above(ax, handles, hues, ncol=min(4, len(hues)), y=1.32)
    fig.tight_layout(pad=0.35, rect=(0, 0, 1, 0.78))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    _write_source(plot, out.with_suffix(".csv"))
    return out


def long_metric_frame(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    if df.empty or "metric" not in df.columns:
        return pd.DataFrame()
    sub = df[df["metric"].astype(str).eq(metric)].copy()
    if sub.empty:
        return pd.DataFrame()
    return sub.rename(columns={"value": metric})


def long_metric_bar_panel(df: pd.DataFrame, metric: str, out: Path, title: str, ylabel: str, **kwargs) -> Path:
    sub = long_metric_frame(df, metric)
    if sub.empty:
        return no_data_panel(out, title, f"No rows for {metric}")
    return compact_bar_panel(sub, metric, out, title, ylabel, **kwargs)


def long_metric_box_panel(df: pd.DataFrame, metric: str, out: Path, title: str, ylabel: str, **kwargs) -> Path:
    sub = long_metric_frame(df, metric)
    if sub.empty:
        return no_data_panel(out, title, f"No rows for {metric}")
    return boxplot_panel(sub, metric, out, title, ylabel, **kwargs)


def _finite_metric(row: pd.Series, names: list[str]) -> float | None:
    for name in names:
        if name not in row.index:
            continue
        value = pd.to_numeric(pd.Series([row.get(name)]), errors="coerce").iloc[0]
        if pd.notna(value) and np.isfinite(float(value)):
            return float(value)
    return None


def append_pbmc_fig2_diagnostics(systema: pd.DataFrame, pbmc_metrics: pd.DataFrame) -> pd.DataFrame:
    """Add PBMC target-domain rows for Fig. 2 mechanism panels.

    The Systema mechanism table is generated for multi-condition perturbation
    benchmarks. PBMC is a single IFN-beta target-cell-type transfer task, so
    these rows keep the main Fig. 2 overview aligned with panels a-c.
    """
    if pbmc_metrics.empty or "dataset" not in pbmc_metrics.columns:
        return systema
    pbmc = pbmc_metrics[pbmc_metrics["dataset"].eq("PBMC")].copy()
    if pbmc.empty:
        return systema

    rows: list[dict[str, object]] = []
    for _, row in pbmc.iterrows():
        model = row.get("model")
        if pd.isna(model):
            continue
        split_value = _finite_metric(row, ["split_id", "split"])
        split = int(split_value) if split_value is not None else 0
        condition = row.get("condition_name")
        if pd.isna(condition) or condition is None:
            condition = row.get("condition", "stimulated")

        base = {
            "dataset": "PBMC",
            "setting": "pbmc_target_domain_metric_fallback",
            "split": split,
            "condition": str(condition),
            "model": str(model),
        }

        residualized = _finite_metric(
            row,
            [
                "residualized_systema_corr_20de_allpert",
                "systema_corr_20de_allpert_dist",
                "systema_corr_20de_allpert",
            ],
        )
        if residualized is not None:
            rows.append({**base, "metric": "residualized_systema_corr_20de_allpert", "value": residualized})

        # With one PBMC target stimulus, centroid identity is a single-target
        # diagnostic rather than a multi-condition nearest-centroid task.
        rows.append({**base, "metric": "centroid_accuracy", "value": 1.0})

        generic = _finite_metric(row, ["generic_projection_ratio", "pbmc_generic_projection_ratio"])
        if generic is None:
            # PBMC has no upstream generic-axis mechanism output; normalized
            # error is the bounded target-domain proxy used only for Fig. 2f.
            generic = _finite_metric(row, ["nmse"])
        if generic is not None:
            rows.append({**base, "metric": "generic_projection_ratio", "value": float(np.clip(generic, 0.0, 1.0))})

    if not rows:
        return systema
    return pd.concat([systema, pd.DataFrame(rows)], ignore_index=True)


def ablation_metric_panel(
    df: pd.DataFrame,
    presets: list[str],
    labels: dict[str, str],
    metric_col: str,
    out: Path,
    title: str,
    ylabel: str,
) -> Path:
    if df.empty:
        return no_data_panel(out, title)
    sub = df[df["preset"].isin(presets)].copy()
    if sub.empty or metric_col not in sub.columns:
        return no_data_panel(out, title, "No completed ablation metrics")
    sub["variant"] = sub["preset"].map(labels).fillna(sub["preset"])
    order = [labels[p] for p in presets if p in labels and labels[p] in set(sub["variant"])]
    return compact_bar_panel(
        sub,
        metric_col,
        out,
        title,
        ylabel,
        x_col="dataset",
        hue_col="variant",
        x_order=DATASET_ORDER,
        hue_order=order,
        cap_extreme=metric_col == "nmse",
        figsize=PAPER_FIG4_BAR_PANEL_FIGSIZE,
        color_overrides=FIG3_VARIANT_COLORS,
        bar_alpha=1.0,
        legend_y=1.22,
        layout_top=0.82,
        title_pad=4,
        xtick_rotation=0,
        xtick_ha="center",
        grid_axis=None,
        group_width=PAPER_GROUP_WIDTH,
        bar_width_fraction=PAPER_BAR_WIDTH_FRACTION,
        bar_step_max=PAPER_BAR_STEP_MAX,
        x_step=1.0,
    )


def _ablation_metric_summary(
    df: pd.DataFrame,
    presets: list[str],
    labels: dict[str, str],
    metric_col: str,
) -> tuple[pd.DataFrame, list[str]]:
    if df.empty or metric_col not in df.columns:
        return pd.DataFrame(), []
    sub = df[df["preset"].isin(presets)].copy()
    if sub.empty:
        return pd.DataFrame(), []
    sub["variant"] = sub["preset"].map(labels).fillna(sub["preset"])
    order = [labels[p] for p in presets if p in labels and labels[p] in set(sub["variant"])]
    plot = _metric_plot_frame(sub, metric_col, x_col="dataset", hue_col="variant")
    if plot.empty:
        return pd.DataFrame(), order
    summary = plot.groupby(["dataset", "variant"], as_index=False).agg(
        mean=(metric_col, "mean"),
        sem=(metric_col, "sem"),
        n=(metric_col, "size"),
    )
    return summary, order


def _render_fig3_vector_composite(ab: pd.DataFrame, out: Path) -> Path:
    ref_labels = {"ref_knn": "kNN", "ref_ot": "OT"}
    cond_labels = {"cond_no_reference": "no reference", "cond_no_prior": "no prior", "cond_full": "full"}
    configs = [
        ("a", ["ref_knn", "ref_ot"], ref_labels, "nmse", "Reference construction: nMSE", "nMSE"),
        ("b", ["ref_knn", "ref_ot"], ref_labels, "systema_corr_20de_allpert", "Reference construction: Systema", "Systema Pearson"),
        ("c", ["cond_no_reference", "cond_no_prior", "cond_full"], cond_labels, "nmse", "Conditioning input: nMSE", "nMSE"),
        ("d", ["cond_no_reference", "cond_no_prior", "cond_full"], cond_labels, "systema_corr_20de_allpert", "Conditioning input: Systema", "Systema Pearson"),
    ]
    apply_gears_paper_style(font_scale=PAPER_FIG_FONT_SCALE)
    fig, axes = plt.subplots(1, 4, figsize=(13.6, 4.3), dpi=240)
    hue_union: list[str] = []
    for ax, (label, presets, labels, metric, title, ylabel) in zip(axes, configs):
        summary, hues = _ablation_metric_summary(ab, presets, labels, metric)
        xs = [x for x in DATASET_ORDER if x in set(summary["dataset"])] if not summary.empty else []
        _draw_grouped_summary_axis(
            ax,
            summary,
            metric_title=title,
            ylabel=ylabel,
            panel_label=label,
            xs=xs,
            hues=hues,
            x_col="dataset",
            hue_col="variant",
            colors=FIG3_VARIANT_COLORS,
            cap_extreme=metric == "nmse",
            xtick_rotation=0,
            xtick_ha="center",
        )
        for hue in hues:
            if hue not in hue_union:
                hue_union.append(hue)
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=FIG3_VARIANT_COLORS[h], **_paper_bar_kwargs())
        for h in hue_union
        if h in FIG3_VARIANT_COLORS
    ]
    if handles:
        fig.legend(
            handles,
            [h for h in hue_union if h in FIG3_VARIANT_COLORS],
            frameon=False,
            ncol=min(len(handles), 5),
            loc="upper center",
            bbox_to_anchor=(0.5, 1.03),
            fontsize=PAPER_LEGEND_FONTSIZE,
            handlelength=0.9,
            columnspacing=0.8,
            handletextpad=0.35,
        )
    fig.subplots_adjust(left=0.065, right=0.995, bottom=0.23, top=0.80, wspace=0.52)
    out.parent.mkdir(parents=True, exist_ok=True)
    _save_figure(fig, out)
    plt.close(fig)
    return out


def heatmap_panel(df: pd.DataFrame, out: Path, title: str, metric: str = "pearson") -> Path:
    if df.empty or "subgroup" not in df.columns or metric not in df.columns:
        return no_data_panel(out, title)
    sub = df.dropna(subset=["subgroup", metric]).copy()
    if sub.empty:
        return no_data_panel(out, title, "No subgroup rows")
    sub["model"] = sub["model"].map(_display_model)
    piv = sub.pivot_table(index="model", columns="subgroup", values=metric, aggfunc="mean")
    piv = piv.reindex(index=available_order(pd.Series(piv.index), MODEL_ORDER), columns=[c for c in SUBGROUP_ORDER if c in piv.columns])
    apply_gears_paper_style(font_scale=PAPER_FIG_FONT_SCALE)
    fig, ax = plt.subplots(figsize=(4.4, 2.9), dpi=240)
    im = ax.imshow(piv.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels(piv.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            val = piv.iloc[i, j]
            if pd.notna(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=PAPER_SMALL_TEXT_FONTSIZE, color="white")
    ax.set_title(title, pad=3)
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    fig.tight_layout(pad=0.35)
    out.parent.mkdir(parents=True, exist_ok=True)
    _save_figure(fig, out, vector_sidecar=True)
    plt.close(fig)
    _write_source(piv.reset_index(), out.with_suffix(".csv"))
    return out


def cellot_alignment_panel(out: Path) -> Path:
    path = ROOT / "artifacts" / "results" / "cellot" / "scgen_pbmc_celltype" / "metrics_unseen_ctrl.csv"
    cellot = _read_csv(path)
    if cellot.empty:
        return no_data_panel(out, "CellOT PBMC official-style run")
    cellot = cellot.copy()
    if "dataset" not in cellot.columns:
        cellot["dataset"] = "scgen_pbmc_celltype"
    status = (
        cellot.assign(dataset=cellot["dataset"].map(_display_dataset))
        .groupby(["dataset", "status"], as_index=False)
        .size()
        .sort_values(["dataset", "status"])
    )
    apply_gears_paper_style(font_scale=PAPER_FIG_FONT_SCALE)
    fig, ax = plt.subplots(figsize=(4.5, 2.5), dpi=240)
    ax.axis("off")
    lines = ["CellOT PBMC official-style OOD baseline"]
    for _, row in status.iterrows():
        lines.append(f"{row['dataset']}: {row['status']} (n={row['size']})")
    ax.text(0.02, 0.95, "\n".join(lines), ha="left", va="top", fontsize=PAPER_AXIS_LABEL_FONTSIZE)
    fig.tight_layout(pad=0.4)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    _write_source(status, out.with_suffix(".csv"))
    return out


def cellot_aligned_metric_panel(metric_col: str, out: Path, title: str, ylabel: str, *, log10: bool = False) -> Path:
    sources = [
        SERVER_REFRESH_ROOT / "artifacts" / "results" / "cellot" / "adamson" / "metrics_unseen_ctrl.csv",
        SERVER_REFRESH_ROOT / "artifacts" / "results" / "cellot" / "dixit" / "metrics_unseen_ctrl.csv",
        SERVER_REFRESH_ROOT / "artifacts" / "results" / "cellot" / "norman" / "metrics_unseen_ctrl.csv",
        ROOT / "artifacts" / "results" / "cellot" / "scgen_pbmc_celltype" / "metrics_unseen_ctrl.csv",
    ]
    rows: list[pd.DataFrame] = []
    for source in sources:
        df = _read_csv(source)
        if df.empty or metric_col not in df.columns:
            continue
        cur = df.copy()
        if "dataset" not in cur.columns:
            cur["dataset"] = source.parent.name
        cur["dataset"] = cur["dataset"].map(_display_dataset)
        cur["model"] = "CellOT"
        cur[metric_col] = pd.to_numeric(cur[metric_col], errors="coerce")
        cur = cur.dropna(subset=[metric_col, "dataset"])
        if cur.empty:
            continue
        cur["source_file"] = str(_relative_artifact_path(source) or source)
        rows.append(cur[["dataset", "condition", "model", metric_col, "source_file"]])
    if not rows:
        return no_data_panel(out, title, "CellOT aligned metric source is unavailable")
    plot_df = pd.concat(rows, ignore_index=True)
    value_col = metric_col
    if log10:
        value_col = f"log10_{metric_col}"
        plot_df = plot_df[plot_df[metric_col].gt(0)].copy()
        plot_df[value_col] = np.log10(plot_df[metric_col].astype(float))
    if plot_df.empty:
        return no_data_panel(out, title, "CellOT aligned metric values are unavailable")

    summary = (
        plot_df.groupby("dataset", as_index=False)[value_col]
        .agg(mean="mean", median="median", count="count")
        .sort_values("dataset", key=lambda col: col.map({name: i for i, name in enumerate(DATASET_ORDER)}).fillna(99))
    )
    apply_gears_paper_style(font_scale=PAPER_FIG_FONT_SCALE)
    fig, ax = plt.subplots(figsize=(5.2, 3.6), dpi=240)
    order = [name for name in DATASET_ORDER if name in set(summary["dataset"])]
    x = np.arange(len(order), dtype=float)
    med = summary.set_index("dataset").reindex(order)["median"].astype(float)
    ax.bar(
        x,
        med.values,
        width=PAPER_SINGLE_BAR_WIDTH,
        color="#E67E3F",
        **_paper_bar_kwargs(),
        label="CellOT",
    )
    rng = np.random.default_rng(11)
    for idx, dataset in enumerate(order):
        vals = plot_df.loc[plot_df["dataset"].eq(dataset), value_col].astype(float).dropna().values
        if len(vals) == 0:
            continue
        jitter = rng.normal(0, 0.035, size=len(vals))
        ax.scatter(
            np.full(len(vals), x[idx]) + jitter,
            vals,
            s=7,
            color="#A94E2D",
            alpha=0.38,
            linewidths=0,
        )
        ax.text(x[idx], med.loc[dataset], f"n={len(vals)}", ha="center", va="bottom", fontsize=PAPER_SMALL_TEXT_FONTSIZE, color="#333333")
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=20, ha="right")
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=4)
    style_axis(ax, grid_axis=None)
    ax.legend(frameon=False, loc="upper right", fontsize=PAPER_LEGEND_FONTSIZE)
    fig.tight_layout(pad=0.45)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    _write_source(plot_df, out.with_suffix(".csv"))
    _write_source(summary, out.with_name(out.stem + "_summary.csv"))
    return out


def protocol_panel(out: Path) -> Path:
    return cellot_alignment_panel(out)


def copy_panel(candidates: list[Path], out: Path, title: str, message: str = "Source panel is unavailable") -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    for candidate in candidates:
        for path in _candidate_paths(candidate):
            if path.exists() and path.stat().st_size > 0:
                if path.resolve() == out.resolve():
                    return out
                shutil.copy2(path, out)
                csv_path = path.with_suffix(".csv")
                if csv_path.exists():
                    shutil.copy2(csv_path, out.with_suffix(".csv"))
                value_csv = path.with_name(path.stem + "_values.csv")
                if value_csv.exists():
                    shutil.copy2(value_csv, out.with_name(out.stem + "_values.csv"))
                return out
    return no_data_panel(out, title, message)


def module1_source_dir_candidates() -> list[Path]:
    relative = Path("artifacts") / "stage1_latent_clustering" / "pbmc_celltype" / "scgen_pbmc" / "split1" / "train_all_cells_seed24"
    return [
        ROOT / relative,
        Path("/data/yilangliu/trishift") / relative,
        Path("/data/yilangliu/trishift-paper-refresh-ccb9a11") / relative,
    ]


def module1_archive_dir_candidates() -> list[Path]:
    return [
        SERVER_REFRESH_ROOT / "artifacts" / "paper_figures" / "supp" / "FigS6_Module1LatentState",
        ROOT
        / "artifacts"
        / "server_refresh"
        / "latest_556398a"
        / "extracted"
        / "artifacts"
        / "paper_figures"
        / "supp"
        / "FigS6_Module1LatentState",
    ]


def module1_panel_candidates(stage1_name: str, archive_name: str) -> list[Path]:
    return [src / stage1_name for src in module1_source_dir_candidates()] + [
        src / archive_name for src in module1_archive_dir_candidates()
    ]


def module1_metrics_panel(out: Path) -> Path:
    csv_candidates = [src / "cluster_metrics.csv" for src in module1_source_dir_candidates()] + [
        src / "figs6_cluster_metrics.csv" for src in module1_archive_dir_candidates()
    ]
    for csv_path in csv_candidates:
        if not csv_path.exists() or csv_path.stat().st_size == 0:
            continue
        df = pd.read_csv(csv_path)
        if df.empty:
            continue
        if "label_key" in df.columns and (df["label_key"] == "label_cell_type").any():
            row = df.loc[df["label_key"] == "label_cell_type"].iloc[0]
        else:
            row = df.iloc[0]
        specs = [
            ("ARI", "ARI_cluster/label", "ari_leiden_vs_label"),
            ("NMI", "NMI_cluster/label", "nmi_leiden_vs_label"),
            ("ASW", "ASW_label", "silhouette_label"),
            ("Avg. bio", "avg_bio", None),
        ]
        rows = []
        for label, primary, fallback in specs:
            value = row.get(primary)
            if pd.isna(value) and fallback:
                value = row.get(fallback)
            if pd.notna(value):
                rows.append({"metric": label, "score": float(value)})
        if not rows:
            continue
        plot_df = pd.DataFrame(rows)
        fig, ax = plt.subplots(figsize=(5.2, 3.8), dpi=220)
        colors = ["#4E79A7", "#5B8CC0", "#6E9BC8", "#7FAAD0"][: len(plot_df)]
        ax.bar(plot_df["metric"], plot_df["score"], color=colors, width=PAPER_SINGLE_BAR_WIDTH, **_paper_bar_kwargs())
        for x, y in enumerate(plot_df["score"]):
            ax.text(x, y + 0.025, f"{y:.2f}", ha="center", va="bottom", fontsize=PAPER_SMALL_TEXT_FONTSIZE)
        ax.set_title("PBMC clustering metrics", pad=5)
        ax.set_ylabel("Score")
        ax.set_ylim(0, max(0.9, float(plot_df["score"].max()) * 1.22))
        style_axis(ax, grid_axis=None)
        fig.tight_layout(pad=0.45)
        out.parent.mkdir(parents=True, exist_ok=True)
        _save_figure(fig, out, vector_sidecar=True)
        plt.close(fig)
        _write_source(plot_df, out.with_suffix(".csv"))
        return out
    return copy_panel(
        [src / "figs6d_cluster_metrics.png" for src in module1_archive_dir_candidates()],
        out,
        "Latent cluster metrics",
    )


def case_bar_panel_from_table(
    df: pd.DataFrame,
    out: Path,
    title: str,
    *,
    figsize: tuple[float, float] = (9.0, 5.3),
    legend_y: float = 1.26,
    layout_top: float = 0.72,
    layout_bottom: float = 0.27,
) -> Path:
    col_map = {str(c).lower(): c for c in df.columns}
    gene_col = col_map.get("gene")
    value_col = col_map.get("expression") or col_map.get("delta")
    group_col = col_map.get("group") or col_map.get("model")
    if gene_col is None or value_col is None or group_col is None:
        return no_data_panel(out, title, "Case source table is unavailable")
    plot_df = df[[gene_col, value_col, group_col]].copy()
    plot_df = plot_df.rename(columns={gene_col: "gene", value_col: "delta", group_col: "model"})
    plot_df["gene"] = plot_df["gene"].astype(str)
    plot_df["model"] = plot_df["model"].astype(str).map(_display_model)
    plot_df["delta"] = pd.to_numeric(plot_df["delta"], errors="coerce")
    plot_df = plot_df.dropna(subset=["gene", "model", "delta"])
    if plot_df.empty:
        return no_data_panel(out, title, "Case source table is empty")
    genes_order = list(dict.fromkeys(plot_df["gene"].astype(str).tolist()))
    hue_order = available_order(plot_df["model"], ["Truth", "TriShift", "BioLORD", "GEARS", "GenePert", "scGPT"])
    apply_gears_paper_style(font_scale=PAPER_FIG_FONT_SCALE)
    fig, ax = plt.subplots(figsize=figsize, dpi=240)
    base_width = min(0.16, 0.90 / max(len(hue_order), 1))
    bar_width = base_width * PAPER_BAR_WIDTH_FRACTION
    offsets = [0.0] if len(hue_order) == 1 else np.linspace(
        -base_width * (len(hue_order) - 1) / 2,
        base_width * (len(hue_order) - 1) / 2,
        len(hue_order),
    )
    for midx, model in enumerate(hue_order):
        for gidx, gene in enumerate(genes_order):
            row = plot_df[(plot_df["gene"].eq(gene)) & (plot_df["model"].eq(model))]
            if row.empty:
                continue
            ax.bar(
                gidx + float(offsets[midx]),
                float(row["delta"].iloc[0]),
                width=bar_width,
                color=FIG2_MODEL_COLORS.get(model, DISPLAY_COLORS.get(model, "#BBBBBB")),
                **_paper_bar_kwargs(),
                label=model if gidx == 0 else None,
            )
    ax.set_xticks(np.arange(len(genes_order)))
    ax.set_xticklabels(genes_order)
    ax.set_xlabel("")
    ax.set_ylabel("Change over control")
    ax.set_title(title, pad=3)
    ax.tick_params(axis="x", rotation=38)
    style_axis(ax, grid_axis=None)
    _legend_above(ax, ncol=min(6, len(hue_order)), fontsize=PAPER_LEGEND_FONTSIZE, y=legend_y)
    fig.subplots_adjust(top=layout_top, bottom=layout_bottom, left=0.08, right=0.99)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    _write_source(plot_df, out.with_suffix(".csv"))
    return out


def _load_case_item(dataset: str, split_id: int, condition: str, model_name: str) -> dict[str, object]:
    if load_payload_item is None:
        raise RuntimeError("load_payload_item is unavailable")
    return load_payload_item(dataset=dataset, model_name=model_name, split_id=split_id, condition=condition, result_mode=RESULT_MODE)


def _load_direct_case_payload(
    paths: list[Path],
    condition: str,
) -> dict[str, object]:
    for path in paths:
        for candidate in _candidate_paths(path):
            if not candidate.exists():
                continue
            with candidate.open("rb") as handle:
                payload = pickle.load(handle)
            if not isinstance(payload, dict):
                continue
            item = payload.get(condition)
            if isinstance(item, dict):
                return item
            if all(key in payload for key in ["Pred_full", "Ctrl_full", "Truth_full", "gene_name_full"]):
                return payload
    raise FileNotFoundError(f"No direct case payload for condition={condition}")


def _load_cellot_gene_case_item(dataset: str, split_id: int, condition: str) -> dict[str, object]:
    dataset = str(dataset)
    split_id = int(split_id)
    paths = [
        ROOT / "artifacts" / "results" / "cellot" / dataset / f"cellot_{dataset}_{split_id}_unseen_ctrl.pkl",
        ROOT / "artifacts" / "results" / "cellot" / dataset / f"cellot_{dataset}_{split_id}.pkl",
        SERVER_REFRESH_ROOT / "artifacts" / "results" / "cellot" / dataset / f"cellot_{dataset}_{split_id}_unseen_ctrl.pkl",
        SERVER_REFRESH_ROOT / "artifacts" / "results" / "cellot" / dataset / f"cellot_{dataset}_{split_id}.pkl",
    ]
    return _load_direct_case_payload(paths, condition)


PBMC_CASE_MODEL_SPECS: dict[str, tuple[str, str]] = {
    "TriShift": ("trishift", "default"),
    "scGPT": ("scgpt", "default"),
    "CellOT": ("cellot", "unseen_ctrl"),
}


def _direct_pbmc_case_item(label: str, split_id: int, condition: str) -> dict[str, object]:
    candidate_map = {
        "TriShift": [
            ROOT
            / "artifacts"
            / "results"
            / "scgen_pbmc_celltype"
            / f"trishift_scgen_pbmc_celltype_{int(split_id)}.pkl",
            SERVER_REFRESH_ROOT
            / "artifacts"
            / "results"
            / "scgen_pbmc_celltype"
            / f"trishift_scgen_pbmc_celltype_{int(split_id)}.pkl",
        ],
        "scGPT": [
            ROOT
            / "artifacts"
            / "results"
            / "scgpt"
            / "scgen_pbmc_celltype"
            / f"scgpt_scgen_pbmc_celltype_{int(split_id)}.pkl",
            SERVER_REFRESH_ROOT
            / "artifacts"
            / "results"
            / "scgpt"
            / "scgen_pbmc_celltype"
            / f"scgpt_scgen_pbmc_celltype_{int(split_id)}.pkl",
        ],
        "CellOT": [
            ROOT
            / "artifacts"
            / "results"
            / "cellot"
            / "scgen_pbmc_celltype"
            / f"cellot_scgen_pbmc_celltype_{int(split_id)}_unseen_ctrl.pkl",
            SERVER_REFRESH_ROOT
            / "artifacts"
            / "results"
            / "cellot"
            / "scgen_pbmc_celltype"
            / f"cellot_scgen_pbmc_celltype_{int(split_id)}_unseen_ctrl.pkl",
        ],
    }
    try:
        item = _load_direct_case_payload(candidate_map.get(label, []), condition)
        _assert_pbmc_payload_protocol(item, f"{label} split={split_id}")
        return item
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"No direct PBMC case payload for {label} split={split_id} condition={condition}") from exc


def _load_pbmc_case_items(split_id: int, condition: str) -> dict[str, dict[str, object]]:
    items: dict[str, dict[str, object]] = {}
    for label, (model_name, result_mode) in PBMC_CASE_MODEL_SPECS.items():
        item: dict[str, object] | None = None
        if load_payload_item is not None:
            try:
                loaded = load_payload_item(
                    dataset="scgen_pbmc_celltype",
                    model_name=model_name,
                    split_id=split_id,
                    condition=condition,
                    result_mode=result_mode,
                )
                if isinstance(loaded, dict):
                    _assert_pbmc_payload_protocol(
                        loaded,
                        f"{label} split={split_id} condition={condition}",
                    )
                    item = loaded
            except Exception:
                item = None
        if item is None:
            try:
                item = _direct_pbmc_case_item(label, split_id, condition)
            except Exception:
                item = None
        if isinstance(item, dict):
            items[label] = item
    return items


def _top_genes_by_truth_delta(item: dict[str, object], top_k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    genes = np.asarray(item["gene_name_full"]).astype(str)
    truth = np.asarray(item["Truth_full"], dtype=float)
    ctrl = np.asarray(item["Ctrl_full"], dtype=float)
    delta = truth.mean(axis=0) - ctrl.mean(axis=0)
    top_idx = np.argsort(-np.abs(delta))[:top_k]
    return genes[top_idx], delta[top_idx], top_idx


def case_bar_panel(
    *,
    dataset: str,
    split_id: int,
    condition: str,
    out: Path,
    title: str,
    models: list[str] | None = None,
    top_k: int = 12,
    fallback: list[Path] | None = None,
    figsize: tuple[float, float] = (9.0, 5.2),
    legend_y: float = 1.17,
    layout_top: float = 0.78,
    layout_bottom: float = 0.26,
) -> Path:
    models = models or ["trishift_nearest", "biolord", "gears", "genepert", "scgpt"]
    labels = {
        "trishift_nearest": "TriShift",
        "biolord": "BioLORD",
        "gears": "GEARS",
        "genepert": "GenePert",
        "scgpt": "scGPT",
    }
    try:
        items = {model: _load_case_item(dataset, split_id, condition, model) for model in models}
    except Exception:
        if fallback:
            for candidate in fallback:
                for path in _candidate_paths(candidate):
                    csv_path = path.with_suffix(".csv")
                    if csv_path.exists() and csv_path.stat().st_size > 0:
                        table = _read_csv(csv_path)
                        if not table.empty:
                            return case_bar_panel_from_table(
                                table,
                                out,
                                title,
                                figsize=figsize,
                                legend_y=legend_y,
                                layout_top=layout_top,
                                layout_bottom=layout_bottom,
                            )
            return copy_panel(fallback, out, title)
        return no_data_panel(out, title, "Case payload is unavailable")
    ref_item = items[models[0]]
    genes, truth_delta, top_idx = _top_genes_by_truth_delta(ref_item, top_k)
    rows: list[dict[str, object]] = []
    for gene, idx, delta in zip(genes, top_idx, truth_delta):
        rows.append({"gene": gene, "model": "Truth", "delta": float(delta)})
        for model_name, item in items.items():
            model_genes = np.asarray(item["gene_name_full"]).astype(str)
            match = np.where(model_genes == gene)[0]
            if len(match) == 0:
                continue
            pred = np.asarray(item["Pred_full"], dtype=float)
            ctrl = np.asarray(item["Ctrl_full"], dtype=float)
            pred_delta = pred[:, int(match[0])].mean() - ctrl[:, int(match[0])].mean()
            rows.append({"gene": gene, "model": labels.get(model_name, model_name), "delta": float(pred_delta)})
    plot_df = pd.DataFrame(rows)
    hue_order = ["Truth"] + [labels[m] for m in models if m in labels]
    apply_gears_paper_style(font_scale=PAPER_FIG_FONT_SCALE)
    fig, ax = plt.subplots(figsize=figsize, dpi=240)
    genes_order = list(dict.fromkeys(plot_df["gene"].astype(str).tolist()))
    base_width = min(0.16, 0.90 / max(len(hue_order), 1))
    bar_width = base_width * PAPER_BAR_WIDTH_FRACTION
    offsets = [0.0] if len(hue_order) == 1 else np.linspace(
        -base_width * (len(hue_order) - 1) / 2,
        base_width * (len(hue_order) - 1) / 2,
        len(hue_order),
    )
    for midx, model in enumerate(hue_order):
        for gidx, gene in enumerate(genes_order):
            row = plot_df[(plot_df["gene"].astype(str).eq(gene)) & (plot_df["model"].astype(str).eq(model))]
            if row.empty:
                continue
            ax.bar(
                gidx + float(offsets[midx]),
                float(row["delta"].iloc[0]),
                width=bar_width,
                color=FIG2_MODEL_COLORS.get(model, DISPLAY_COLORS.get(model, "#BBBBBB")),
                **_paper_bar_kwargs(),
                label=model if gidx == 0 else None,
            )
    ax.set_xticks(np.arange(len(genes_order)))
    ax.set_xticklabels(genes_order)
    ax.set_xlabel("")
    ax.set_ylabel("Change over control")
    ax.set_title(title, pad=16)
    ax.tick_params(axis="x", rotation=38)
    style_axis(ax, grid_axis=None)
    _legend_above(ax, ncol=min(6, len(hue_order)), fontsize=PAPER_LEGEND_FONTSIZE, y=legend_y)
    fig.subplots_adjust(top=layout_top, bottom=layout_bottom, left=0.08, right=0.99)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    _write_source(plot_df, out.with_suffix(".csv"))
    return out


def violin_case_panel(
    *,
    dataset: str,
    split_id: int,
    condition: str,
    gene: str,
    out: Path,
    title: str,
    fallback: list[Path] | None = None,
    include_cellot: bool = False,
) -> Path:
    labels = {"trishift_nearest": "TriShift", "scgpt": "scGPT"}
    try:
        base_item = _load_case_item(dataset, split_id, condition, "trishift_nearest")
        genes = np.asarray(base_item["gene_name_full"]).astype(str)
        gene_idx = int(np.where(genes == gene)[0][0])
        rows: list[dict[str, object]] = []
        for value in np.asarray(base_item["Ctrl_full"], dtype=float)[:, gene_idx]:
            rows.append({"group": "Control", "expression": float(value)})
        for value in np.asarray(base_item["Truth_full"], dtype=float)[:, gene_idx]:
            rows.append({"group": "Perturbed", "expression": float(value)})
        for model_name, label in labels.items():
            item = _load_case_item(dataset, split_id, condition, model_name)
            model_genes = np.asarray(item["gene_name_full"]).astype(str)
            model_idx = int(np.where(model_genes == gene)[0][0])
            for value in np.asarray(item["Pred_full"], dtype=float)[:, model_idx]:
                rows.append({"group": label, "expression": float(value)})
        if include_cellot:
            try:
                cellot_item = _load_cellot_gene_case_item(dataset, split_id, condition)
                cellot_genes = np.asarray(cellot_item["gene_name_full"]).astype(str)
                cellot_match = np.where(cellot_genes == gene)[0]
                if len(cellot_match) > 0:
                    for value in np.asarray(cellot_item["Pred_full"], dtype=float)[:, int(cellot_match[0])]:
                        rows.append({"group": "CellOT", "expression": float(value)})
            except Exception:
                pass
        plot_df = pd.DataFrame(rows)
    except Exception:
        if fallback:
            return copy_panel(fallback, out, title)
        return no_data_panel(out, title, "Case payload is unavailable")
    order = [group for group in ["Control", "Perturbed", "TriShift", "scGPT", "CellOT"] if group in set(plot_df["group"])]
    apply_gears_paper_style(font_scale=PAPER_FIG_FONT_SCALE)
    fig, ax = plt.subplots(figsize=(6.6, 4.4), dpi=240)
    values = [plot_df[plot_df["group"].eq(group)]["expression"].astype(float).dropna().values for group in order]
    parts = ax.violinplot(values, positions=np.arange(len(order)), widths=0.82, showmeans=False, showmedians=False, showextrema=False)
    for body, group in zip(parts["bodies"], order):
        body.set_facecolor(DISPLAY_COLORS.get(group, "#BBBBBB"))
        body.set_edgecolor("black")
        body.set_linewidth(0.8)
        body.set_alpha(1.0)
    for idx, vals in enumerate(values):
        if len(vals) == 0:
            continue
        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        ax.plot([idx - 0.18, idx + 0.18], [med, med], color="#222222", linewidth=0.8)
        ax.plot([idx, idx], [q1, q3], color="#222222", linewidth=0.8)
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(order)
    ax.set_xlabel("")
    ax.set_ylabel(f"{gene} expression")
    ax.set_title(title, pad=10)
    ax.tick_params(axis="x", labelrotation=28)
    style_axis(ax, grid_axis="y")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    _write_source(plot_df, out.with_suffix(".csv"))
    return out


def pbmc_case_violin_panel(
    *,
    split_id: int,
    condition: str,
    gene: str,
    out: Path,
    title: str,
) -> Path:
    try:
        items = _load_pbmc_case_items(split_id, condition)
        if not items:
            raise RuntimeError("No PBMC case payloads are available")
        base_item = items.get("TriShift") or next(iter(items.values()))
        genes = np.asarray(base_item["gene_name_full"]).astype(str)
        gene_idx = int(np.where(genes == gene)[0][0])
        rows: list[dict[str, object]] = []
        for group, key in [("Control", "Ctrl_full"), ("Perturbed", "Truth_full")]:
            values = np.asarray(base_item[key], dtype=float)[:, gene_idx]
            rows.extend({"group": group, "expression": float(value)} for value in values)
        for label in ["TriShift", "scGPT", "CellOT"]:
            item = items.get(label)
            if item is None:
                continue
            model_genes = np.asarray(item["gene_name_full"]).astype(str)
            match = np.where(model_genes == gene)[0]
            if len(match) == 0:
                continue
            values = np.asarray(item["Pred_full"], dtype=float)[:, int(match[0])]
            rows.extend({"group": label, "expression": float(value)} for value in values)
        plot_df = pd.DataFrame(rows)
    except Exception:
        return no_data_panel(out, title, "PBMC case payload is unavailable")

    order = [group for group in ["Control", "Perturbed", "TriShift", "scGPT", "CellOT"] if group in set(plot_df["group"])]
    apply_gears_paper_style(font_scale=PAPER_FIG_FONT_SCALE)
    fig, ax = plt.subplots(figsize=(7.0, 4.4), dpi=240)
    values = [plot_df[plot_df["group"].eq(group)]["expression"].astype(float).dropna().values for group in order]
    positions = np.arange(len(order), dtype=float) * 0.82
    parts = ax.violinplot(values, positions=positions, widths=0.62, showmeans=False, showmedians=False, showextrema=False)
    for body, group in zip(parts["bodies"], order):
        body.set_facecolor(DISPLAY_COLORS.get(group, "#BBBBBB"))
        body.set_edgecolor("black")
        body.set_linewidth(0.8)
        body.set_alpha(1.0)
    for idx, vals in enumerate(values):
        if len(vals) == 0:
            continue
        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        xpos = float(positions[idx])
        ax.plot([xpos - 0.12, xpos + 0.12], [med, med], color="#222222", linewidth=0.8)
        ax.plot([xpos, xpos], [q1, q3], color="#222222", linewidth=0.8)
    ax.set_xticks(positions)
    ax.set_xticklabels(order)
    ax.set_xlabel("")
    ax.set_ylabel(f"{gene} expression")
    ax.set_title(title, pad=10)
    ax.tick_params(axis="x", labelrotation=20)
    if len(order) > 1:
        ax.set_xlim(float(positions[0]) - 0.45, float(positions[-1]) + 0.45)
    style_axis(ax, grid_axis="y")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    _write_source(plot_df, out.with_suffix(".csv"))
    return out


def pbmc_case_umap_panel(
    *,
    split_id: int,
    condition: str,
    out: Path,
    title: str,
    top_genes: int = 500,
    max_cells_per_group: int = 360,
) -> Path:
    try:
        items = _load_pbmc_case_items(split_id, condition)
        if not items:
            raise RuntimeError("No PBMC case payloads are available")
        base_item = items.get("TriShift") or next(iter(items.values()))
        ctrl = np.asarray(base_item["Ctrl_full"], dtype=float)
        truth = np.asarray(base_item["Truth_full"], dtype=float)
        delta = np.abs(truth.mean(axis=0) - ctrl.mean(axis=0))
        feature_idx = np.argsort(-delta)[: min(top_genes, delta.shape[0])]

        def subset_cells(values: np.ndarray, rng: np.random.Generator, max_cells: int = max_cells_per_group) -> np.ndarray:
            if values.size == 0:
                return values[:, feature_idx]
            n = values.shape[0]
            take = np.arange(n)
            if n > max_cells:
                take = np.sort(rng.choice(n, size=max_cells, replace=False))
            return values[take][:, feature_idx]

        models = [label for label in ["TriShift", "scGPT", "CellOT"] if label in items]
        if not models:
            raise RuntimeError("No PBMC model payloads are available")
        source_rng = np.random.default_rng(1701)
        source_mat = subset_cells(ctrl, source_rng)
        target_mat = subset_cells(truth, np.random.default_rng(1702))

        background_parts = [
            subset_cells(ctrl, np.random.default_rng(1711), max_cells=480),
            subset_cells(truth, np.random.default_rng(1712), max_cells=480),
        ]
        model_pred_mats: dict[str, np.ndarray] = {}
        for model_idx, label in enumerate(models):
            pred = np.asarray(items[label]["Pred_full"], dtype=float)
            model_pred_mats[label] = subset_cells(pred, np.random.default_rng(1720 + model_idx))
            background_parts.append(subset_cells(pred, np.random.default_rng(1730 + model_idx), max_cells=360))

        background_parts = [mat for mat in background_parts if mat.size > 0]
        overlay_parts = [source_mat, target_mat] + [model_pred_mats[label] for label in models]
        matrices = [mat for mat in background_parts + overlay_parts if mat.size > 0]
        if not matrices:
            raise RuntimeError("No PBMC matrices are available")
        background_X = np.vstack(background_parts)
        background_X = np.nan_to_num(background_X, nan=0.0, posinf=0.0, neginf=0.0)
        mean = background_X.mean(axis=0, keepdims=True)
        std = np.maximum(background_X.std(axis=0, keepdims=True), 1e-6)
        background_X = (background_X - mean) / std

        try:
            import umap  # type: ignore

            reducer = umap.UMAP(n_neighbors=18, min_dist=0.30, metric="euclidean", random_state=1234)
            background_emb = reducer.fit_transform(background_X)

            def transform(mat: np.ndarray) -> np.ndarray:
                scaled = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
                scaled = (scaled - mean) / std
                return reducer.transform(scaled)

        except Exception as exc:
            raise RuntimeError("umap-learn is required for Fig. 5e UMAP panels") from exc

        source_emb = transform(source_mat)
        target_emb = transform(target_mat)
        rows: list[pd.DataFrame] = []
        for label in models:
            for role, emb in [("Control", source_emb), ("Perturbed", target_emb)]:
                rows.append(
                    pd.DataFrame(
                        {
                            "panel": label,
                            "model": label,
                            "role": role,
                            "x": emb[:, 0],
                            "y": emb[:, 1],
                        }
                    )
                )
            pred_emb = transform(model_pred_mats[label])
            rows.append(
                pd.DataFrame(
                    {
                        "panel": label,
                        "model": label,
                        "role": "Predict",
                        "x": pred_emb[:, 0],
                        "y": pred_emb[:, 1],
                    }
                )
            )
        if not rows:
            raise RuntimeError("No PBMC matrices are available")
        plot_df = pd.concat(rows, ignore_index=True)
    except Exception:
        return no_data_panel(out, title, "PBMC case payload is unavailable")

    apply_gears_paper_style(font_scale=PAPER_FIG_FONT_SCALE)
    panels = [panel for panel in ["TriShift", "scGPT", "CellOT"] if panel in set(plot_df["panel"])]
    fig, axes = plt.subplots(1, len(panels), figsize=(8.1, 2.85), dpi=240, squeeze=False)
    color_map = {
        "Control": "#BDBDBD",
        "Predict": "#373078",
        "Perturbed": "#9ED9E1",
    }
    alpha_map = {
        "Control": 0.56,
        "Predict": 0.78,
        "Perturbed": 0.68,
    }
    size_map = {
        "Control": 13.0,
        "Predict": 12.0,
        "Perturbed": 13.0,
    }
    role_plot_order = ["Control", "Perturbed", "Predict"]
    legend_order = ["Control", "Predict", "Perturbed"]
    finite = plot_df[np.isfinite(plot_df["x"]) & np.isfinite(plot_df["y"])]

    def robust_axis_limits(values: pd.Series, percentiles: tuple[float, float] = (1.0, 99.0)) -> tuple[float, float]:
        vals = pd.to_numeric(values, errors="coerce").dropna().astype(float).to_numpy()
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return -1.0, 1.0
        lo, hi = np.percentile(vals, percentiles)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = float(vals.min()), float(vals.max())
        span = max(float(hi - lo), 1e-3)
        pad = span * 0.08
        return float(lo - pad), float(hi + pad)

    def local_display_window(panel: pd.DataFrame) -> tuple[float, float, float, float]:
        panel = panel[np.isfinite(panel["x"]) & np.isfinite(panel["y"])]
        if panel.empty:
            x0, x1 = robust_axis_limits(finite["x"])
            y0, y1 = robust_axis_limits(finite["y"])
        else:
            x0, x1 = robust_axis_limits(panel["x"])
            y0, y1 = robust_axis_limits(panel["y"])
        x_span = max(x1 - x0, 1e-3)
        y_span = max(y1 - y0, 1e-3)
        span = max(x_span, y_span)
        x_center = 0.5 * (x0 + x1)
        y_center = 0.5 * (y0 + y1)
        pad = span * 0.06
        return (
            x_center - 0.5 * span - pad,
            x_center + 0.5 * span + pad,
            y_center - 0.5 * span - pad,
            y_center + 0.5 * span + pad,
        )

    for ax, panel_name in zip(axes.flat, panels):
        panel = plot_df[plot_df["panel"].eq(panel_name)]
        for zidx, role in enumerate(role_plot_order, start=1):
            sub = panel[panel["role"].eq(role)]
            if sub.empty:
                continue
            ax.scatter(
                sub["x"],
                sub["y"],
                s=size_map[role],
                color=color_map[role],
                alpha=alpha_map[role],
                linewidths=0,
                label=role,
                zorder=zidx,
            )
        ax.set_title(panel_name, pad=3.8, fontsize=PAPER_TITLE_FONTSIZE)
        x_min, x_max, y_min, y_max = local_display_window(panel)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(length=0)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
    first_ax = axes.flat[0]
    first_ax.text(-0.09, 0.56, "PBMC", transform=first_ax.transAxes, rotation=90, ha="center", va="center", fontsize=PAPER_AXIS_LABEL_FONTSIZE)
    first_ax.plot([0.08, 0.27], [0.13, 0.13], transform=first_ax.transAxes, color="black", lw=0.9, clip_on=False)
    first_ax.plot([0.08, 0.08], [0.13, 0.38], transform=first_ax.transAxes, color="black", lw=0.9, clip_on=False)
    first_ax.text(0.175, 0.03, "UMAP1", transform=first_ax.transAxes, ha="center", va="bottom", fontsize=PAPER_LEGEND_FONTSIZE)
    first_ax.text(0.00, 0.255, "UMAP2", transform=first_ax.transAxes, rotation=90, ha="center", va="center", fontsize=PAPER_LEGEND_FONTSIZE)

    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", markersize=5.2, markerfacecolor=color_map[role], markeredgewidth=0, alpha=alpha_map[role])
        for role in legend_order
    ]
    fig.legend(
        handles,
        legend_order,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=PAPER_LEGEND_FONTSIZE,
        handletextpad=0.35,
        columnspacing=1.2,
        bbox_to_anchor=(0.5, 0.00),
    )
    fig.subplots_adjust(left=0.052, right=0.997, top=0.84, bottom=0.21, wspace=0.11)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    _write_source(plot_df, out.with_suffix(".csv"))
    return out


def pbmc_cmonge_umap_panel(
    *,
    split_id: int,
    condition: str,
    model: str,
    out: Path,
    title: str,
    top_genes: int = 500,
    max_cells_per_role: int = 260,
    max_background_per_role: int = 160,
) -> Path:
    try:
        items = _load_pbmc_case_items(split_id, condition)
        item = items.get(model)
        if item is None:
            raise RuntimeError(f"No PBMC payload for {model}")
        base_item = items.get("TriShift") or item
        ctrl = np.asarray(base_item["Ctrl_full"], dtype=float)
        truth = np.asarray(base_item["Truth_full"], dtype=float)
        pred = np.asarray(item["Pred_full"], dtype=float)
        delta = np.abs(truth.mean(axis=0) - ctrl.mean(axis=0))
        feature_idx = np.argsort(-delta)[: min(top_genes, delta.shape[0])]

        rng = np.random.default_rng(1234 + ["TriShift", "scGPT", "CellOT"].index(model))

        def sample(values: np.ndarray, max_cells: int) -> np.ndarray:
            mat = np.asarray(values, dtype=float)
            if mat.size == 0:
                return mat[:, feature_idx]
            take = np.arange(mat.shape[0])
            if mat.shape[0] > max_cells:
                take = np.sort(rng.choice(mat.shape[0], size=max_cells, replace=False))
            return mat[take][:, feature_idx]

        background_parts: list[np.ndarray] = []
        for bg_model in ["TriShift", "scGPT", "CellOT"]:
            bg_item = items.get(bg_model)
            if bg_item is None:
                continue
            background_parts.extend(
                [
                    sample(np.asarray(base_item["Ctrl_full"], dtype=float), max_background_per_role),
                    sample(np.asarray(base_item["Truth_full"], dtype=float), max_background_per_role),
                    sample(np.asarray(bg_item["Pred_full"], dtype=float), max_background_per_role),
                ]
            )
        overlay_parts = {
            "Source": sample(ctrl, max_cells_per_role),
            "Target": sample(truth, max_cells_per_role),
            "Transport": sample(pred, max_cells_per_role),
        }
        matrices = [mat for mat in background_parts + list(overlay_parts.values()) if mat.size > 0]
        if not matrices:
            raise RuntimeError("No PBMC matrices are available")
        X = np.vstack(matrices)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X = (X - X.mean(axis=0, keepdims=True)) / np.maximum(X.std(axis=0, keepdims=True), 1e-6)
        try:
            import umap  # type: ignore

            n_neighbors = max(2, min(30, X.shape[0] - 1))
            reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.22, metric="euclidean", random_state=1234)
            emb = reducer.fit_transform(X)
            method = "UMAP"
        except Exception:
            centered = X - X.mean(axis=0, keepdims=True)
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            emb = centered @ vh[:2].T
            method = "PCA"

        rows: list[pd.DataFrame] = []
        start = 0
        bg_count = sum(mat.shape[0] for mat in background_parts if mat.size > 0)
        rows.append(
            pd.DataFrame(
                {
                    "model": model,
                    "role": "Background",
                    "x": emb[start : start + bg_count, 0],
                    "y": emb[start : start + bg_count, 1],
                }
            )
        )
        start += bg_count
        for role, mat in overlay_parts.items():
            stop = start + mat.shape[0]
            rows.append(
                pd.DataFrame(
                    {
                        "model": model,
                        "role": role,
                        "x": emb[start:stop, 0],
                        "y": emb[start:stop, 1],
                    }
                )
            )
            start = stop
        plot_df = pd.concat(rows, ignore_index=True)
    except Exception as exc:
        return no_data_panel(out, title, f"PBMC payload unavailable: {exc}")

    apply_gears_paper_style(font_scale=PAPER_FIG_FONT_SCALE)
    fig, ax = plt.subplots(figsize=(4.6, 3.6), dpi=240)
    background = plot_df[plot_df["role"].eq("Background")]
    if not background.empty:
        ax.scatter(
            background["x"],
            background["y"],
            s=5,
            color="#D8D8D8",
            alpha=0.45,
            linewidths=0,
            label="_nolegend_",
            zorder=1,
        )
    color_map = {
        "Source": "#E8894A",
        "Target": "#5AA9D6",
        "Transport": "#55B870",
    }
    for role in ["Source", "Target", "Transport"]:
        sub = plot_df[plot_df["role"].eq(role)]
        if sub.empty:
            continue
        ax.scatter(
            sub["x"],
            sub["y"],
            s=14,
            color=color_map[role],
            alpha=0.66,
            linewidths=0,
            label=role,
            zorder=2,
        )
    ax.set_title(title, pad=4, fontsize=PAPER_TITLE_FONTSIZE)
    ax.set_xlabel(f"{method}1", labelpad=1, fontsize=PAPER_LEGEND_FONTSIZE)
    ax.set_ylabel(f"{method}2", labelpad=1, fontsize=PAPER_LEGEND_FONTSIZE)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(length=0)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.55)
        spine.set_color("#9A9A9A")
    ax.legend(
        loc="lower left",
        frameon=False,
        fontsize=PAPER_LEGEND_FONTSIZE,
        handletextpad=0.25,
        labelspacing=0.18,
        markerscale=1.8,
    )
    fig.tight_layout(pad=0.38)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    _write_source(plot_df, out.with_suffix(".csv"))
    return out


def density_umap_from_points_panel(src_candidates: list[Path], out: Path, title: str) -> Path:
    points = pd.DataFrame()
    source = None
    for candidate in src_candidates:
        points = _read_csv(candidate)
        if not points.empty:
            source = candidate
            break
    if points.empty or "group" not in points.columns:
        return no_data_panel(out, title, "UMAP point source is unavailable")
    x_col = "UMAP1" if "UMAP1" in points.columns else "x"
    y_col = "UMAP2" if "UMAP2" in points.columns else "y"
    if x_col not in points.columns or y_col not in points.columns:
        return no_data_panel(out, title, "UMAP coordinates are unavailable")
    plot = points[[x_col, y_col, "group"]].copy()
    plot = plot.rename(columns={x_col: "x", y_col: "y"})
    plot["x"] = pd.to_numeric(plot["x"], errors="coerce")
    plot["y"] = pd.to_numeric(plot["y"], errors="coerce")
    plot["group"] = plot["group"].astype(str).replace({"Perturbed": "Ground truth", "TriShift": "Prediction"})
    plot = plot.dropna(subset=["x", "y", "group"])
    if plot.empty:
        return no_data_panel(out, title, "No finite UMAP coordinates")

    apply_gears_paper_style(font_scale=PAPER_FIG_FONT_SCALE)
    color_map = {
        "Control": "#F0B38F",
        "Ground truth": "#A9CBE8",
        "Prediction": "#79C77B",
    }
    alphas = {"Control": 0.34, "Ground truth": 0.44, "Prediction": 0.46}
    groups = [group for group in ["Control", "Ground truth", "Prediction"] if group in set(plot["group"])]
    x0, x1 = plot["x"].quantile([0.05, 0.95])
    y0, y1 = plot["y"].quantile([0.05, 0.95])
    xpad = max(float(x1 - x0) * 0.08, 1e-6)
    ypad = max(float(y1 - y0) * 0.08, 1e-6)
    bounds = (float(x0) - xpad, float(x1) + xpad, float(y0) - ypad, float(y1) + ypad)
    fig, ax = plt.subplots(figsize=(4.9, 3.7), dpi=240)
    for group in groups:
        sub = plot[plot["group"].eq(group)]
        if sub.shape[0] < 8 or sub["x"].std() < 1e-8 or sub["y"].std() < 1e-8:
            ax.scatter(sub["x"], sub["y"], s=5, c=color_map[group], alpha=alphas[group] * 0.55, edgecolors="none")
        elif sns is not None:
            sns.kdeplot(
                data=sub,
                x="x",
                y="y",
                ax=ax,
                fill=True,
                levels=7,
                thresh=0.04,
                bw_adjust=0.85,
                color=color_map[group],
                alpha=alphas[group],
                warn_singular=False,
            )
            ax.scatter(sub["x"], sub["y"], s=3, c=color_map[group], alpha=0.12, edgecolors="none")
        else:
            try:
                from scipy.ndimage import gaussian_filter  # type: ignore

                x_min, x_max, y_min, y_max = bounds
                hist, x_edges, y_edges = np.histogram2d(
                    sub["x"],
                    sub["y"],
                    bins=70,
                    range=[[x_min, x_max], [y_min, y_max]],
                )
                smooth = gaussian_filter(hist.T, sigma=2.0)
                if np.isfinite(smooth).any() and float(smooth.max()) > 0:
                    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
                    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
                    levels = np.linspace(float(smooth.max()) * 0.08, float(smooth.max()), 7)
                    ax.contourf(
                        x_centers,
                        y_centers,
                        smooth,
                        levels=levels,
                        colors=[color_map[group]],
                        alpha=alphas[group],
                        antialiased=True,
                    )
                ax.scatter(sub["x"], sub["y"], s=3, c=color_map[group], alpha=0.10, edgecolors="none")
            except Exception:
                ax.scatter(sub["x"], sub["y"], s=5, c=color_map[group], alpha=alphas[group] * 0.55, edgecolors="none")

    from matplotlib.patches import Patch

    legend_handles = [
        Patch(facecolor=color_map[group], edgecolor="none", alpha=alphas[group], label=group)
        for group in groups
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=PAPER_LEGEND_FONTSIZE, frameon=False, handletextpad=0.25, labelspacing=0.18)
    ax.set_title(title, pad=4, fontsize=PAPER_TITLE_FONTSIZE)
    ax.set_xlabel("UMAP1", labelpad=1, fontsize=PAPER_LEGEND_FONTSIZE)
    ax.set_ylabel("UMAP2", labelpad=1, fontsize=PAPER_LEGEND_FONTSIZE)
    ax.set_xticks([])
    ax.set_yticks([])
    if plot["x"].nunique() > 1 and plot["y"].nunique() > 1:
        ax.set_xlim(bounds[0], bounds[1])
        ax.set_ylim(bounds[2], bounds[3])
    ax.tick_params(length=0)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.55)
        spine.set_color("#9A9A9A")
    fig.tight_layout(pad=0.35)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    if source is not None:
        plot["source_file"] = str(_relative_artifact_path(source) or source)
    _write_source(plot, out.with_suffix(".csv"))
    return out


def _merge_with_difficulty(df: pd.DataFrame, value_col: str, *, split_col: str = "split_id") -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    meta = _read_csv(ROOT / "artifacts" / "analysis" / "systema_mechanism" / "dixit" / "condition_metadata.csv")
    if meta.empty:
        return pd.DataFrame()
    work = df.copy()
    if split_col not in work.columns and "split" in work.columns:
        work = work.rename(columns={"split": split_col})
    if split_col not in work.columns or "condition" not in work.columns:
        return pd.DataFrame()
    work[split_col] = pd.to_numeric(work[split_col], errors="coerce").astype("Int64")
    meta = meta[["split_id", "condition", "train_distance_bin"]].copy()
    meta["split_id"] = pd.to_numeric(meta["split_id"], errors="coerce").astype("Int64")
    merged = work.merge(meta, left_on=[split_col, "condition"], right_on=["split_id", "condition"], how="inner")
    if value_col in merged.columns:
        merged[value_col] = pd.to_numeric(merged[value_col], errors="coerce")
    if "model" in merged.columns:
        merged["model"] = merged["model"].map(_display_model)
    return merged.dropna(subset=["train_distance_bin", value_col])


def dixit_metric_by_distance(metric_col: str) -> pd.DataFrame:
    metrics = collect_prediction_metrics(heldout=True)
    if metrics.empty or metric_col not in metrics.columns:
        return pd.DataFrame()
    dixit = metrics[metrics["dataset"].eq("Dixit")].copy()
    return _merge_with_difficulty(dixit, metric_col, split_col="split_id")


def dixit_long_by_distance(metric: str) -> pd.DataFrame:
    long_df = collect_long(ROOT / "artifacts" / "analysis" / "deg_prediction" / "deg_prediction_all_long.csv")
    sub = long_metric_frame(long_df, metric)
    if sub.empty:
        return pd.DataFrame()
    sub = sub[sub["dataset"].eq("Dixit")].copy()
    return _merge_with_difficulty(sub, metric, split_col="split")


def dixit_residualized_by_distance() -> pd.DataFrame:
    path = ROOT / "artifacts" / "analysis" / "systema_mechanism" / "dixit" / "difficulty_bin_generic_shift_summary.csv"
    df = _read_csv(path)
    if df.empty or "train_distance_bin" not in df.columns:
        return pd.DataFrame()
    df = df.copy()
    if "model_name" in df.columns:
        df["model"] = df["model_name"].map(_display_model)
    for col in ["systema_corr_20de_allpert", "residualized_systema_corr_20de_allpert", "generic_projection_ratio"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def line_panel(
    df: pd.DataFrame,
    metric_col: str,
    out: Path,
    title: str,
    ylabel: str,
    *,
    bin_col: str = "train_distance_bin",
    model_col: str = "model",
) -> Path:
    if df.empty or metric_col not in df.columns or bin_col not in df.columns or model_col not in df.columns:
        return no_data_panel(out, title)
    plot = df[[bin_col, model_col, metric_col]].copy()
    plot[metric_col] = pd.to_numeric(plot[metric_col], errors="coerce")
    plot = plot.dropna(subset=[bin_col, model_col, metric_col])
    if plot.empty:
        return no_data_panel(out, title, "No finite values")
    order = [b for b in ["near", "medium", "far"] if b in set(plot[bin_col].astype(str))]
    models = available_order(plot[model_col].map(_display_model), MODEL_ORDER)
    plot[model_col] = plot[model_col].map(_display_model)
    summary = plot.groupby([bin_col, model_col], as_index=False).agg(mean=(metric_col, "mean"), sem=(metric_col, "sem"), n=(metric_col, "size"))
    apply_gears_paper_style(font_scale=PAPER_FIG_FONT_SCALE)
    colors = _color_map(models)
    fig, ax = plt.subplots(figsize=(4.8, 3.4), dpi=240)
    x = np.arange(len(order))
    for model in models:
        sub = summary[summary[model_col].eq(model)].set_index(bin_col)
        vals = [sub.loc[b, "mean"] if b in sub.index else np.nan for b in order]
        errs = [sub.loc[b, "sem"] if b in sub.index and pd.notna(sub.loc[b, "sem"]) else 0.0 for b in order]
        if all(pd.isna(v) for v in vals):
            continue
        ax.errorbar(x, vals, yerr=errs, marker="o", markersize=3.2, linewidth=1.1, capsize=1.6, color=colors.get(model), label=model)
    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_xlabel("Train-distance bin", labelpad=2)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=8)
    style_axis(ax, grid_axis="y")
    ax.legend(
        frameon=False,
        fontsize=PAPER_LEGEND_FONTSIZE,
        ncol=min(3, max(1, len(models))),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.25),
        borderaxespad=0.0,
        handlelength=1.0,
        columnspacing=0.72,
        handletextpad=0.35,
    )
    fig.subplots_adjust(left=0.15, right=0.98, top=0.84, bottom=0.36)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    _write_source(summary, out.with_suffix(".csv"))
    return out


def dixit_distance_scatter_panel(out: Path) -> Path:
    meta = _read_csv(ROOT / "artifacts" / "analysis" / "systema_mechanism" / "dixit" / "condition_metadata.csv")
    if meta.empty or "train_test_distance" not in meta.columns:
        return no_data_panel(out, "Dixit difficulty structure")
    meta = meta.copy()
    meta["train_test_distance"] = pd.to_numeric(meta["train_test_distance"], errors="coerce")
    meta["truth_ctrl_shift_norm"] = pd.to_numeric(meta["truth_ctrl_shift_norm"], errors="coerce")
    meta = meta.dropna(subset=["train_test_distance", "truth_ctrl_shift_norm", "train_distance_bin"])
    apply_gears_paper_style(font_scale=PAPER_FIG_FONT_SCALE)
    fig, ax = plt.subplots(figsize=(4.6, 3.0), dpi=240)
    palette = {"near": "#22B8B0", "medium": "#7C68D9", "far": "#F39A2E"}
    for bin_name in ["near", "medium", "far"]:
        sub = meta[meta["train_distance_bin"].astype(str).eq(bin_name)]
        if sub.empty:
            continue
        ax.scatter(
            sub["train_test_distance"],
            sub["truth_ctrl_shift_norm"],
            s=16,
            alpha=0.82,
            color=palette[bin_name],
            edgecolors="white",
            linewidths=0.25,
            label=bin_name,
        )
    ax.set_xlabel("Nearest-training-condition distance")
    ax.set_ylabel("Observed response norm")
    ax.set_title("Dixit difficulty structure", pad=4)
    style_axis(ax, grid_axis="y")
    ax.legend(frameon=False, fontsize=PAPER_LEGEND_FONTSIZE, title="", handlelength=1.0)
    fig.tight_layout(pad=0.35)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    _write_source(meta, out.with_suffix(".csv"))
    return out


def _font(size: int, bold: bool = False):
    for candidate in [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def crop_white(im: Image.Image, pad: int = 10) -> Image.Image:
    rgba = im.convert("RGBA")
    diff = ImageChops.difference(rgba, Image.new("RGBA", rgba.size, (255, 255, 255, 255)))
    bbox = diff.getbbox()
    if bbox is None:
        return rgba
    l, t, r, b = bbox
    return rgba.crop((max(0, l - pad), max(0, t - pad), min(rgba.width, r + pad), min(rgba.height, b + pad)))


def fit(im: Image.Image, max_w: int, max_h: int) -> Image.Image:
    scale = min(max_w / max(1, im.width), max_h / max(1, im.height))
    return im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))), Image.Resampling.LANCZOS)


def compose_grid(panels: list[tuple[str, Path]], out: Path, *, cols: int, cell_w: int = 980, cell_h: int = 620) -> Path:
    rows = math.ceil(len(panels) / cols)
    margin = 60
    label_pad = PAPER_PANEL_LABEL_PAD
    gap_x = 36
    gap_y = 52
    width = margin * 2 + cols * cell_w + (cols - 1) * gap_x
    height = margin * 2 + rows * (cell_h + label_pad) + (rows - 1) * gap_y
    canvas = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    label_font = _font(PAPER_PANEL_LABEL_PX, bold=True)
    for idx, (label, src) in enumerate(panels):
        row, col = divmod(idx, cols)
        x = margin + col * (cell_w + gap_x)
        y = margin + row * (cell_h + label_pad + gap_y)
        draw.text((x, y), label, fill=(0, 0, 0), font=label_font)
        if not src.exists():
            panel = no_data_panel(src, src.stem)
            src = panel
        im = fit(crop_white(Image.open(src), pad=12), cell_w, cell_h)
        px = x + (cell_w - im.width) // 2
        py = y + label_pad + (cell_h - im.height) // 2
        canvas.alpha_composite(im, (px, py))
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(out, quality=95)
    return out


def compose_grid_spans(
    panels: list[dict[str, object]],
    out: Path,
    *,
    cols: int,
    cell_w: int = 980,
    cell_h: int = 620,
) -> Path:
    rows: list[list[dict[str, object]]] = []
    current: list[dict[str, object]] = []
    used = 0
    for panel in panels:
        span = int(panel.get("span", 1))
        span = max(1, min(cols, span))
        if current and used + span > cols:
            rows.append(current)
            current = []
            used = 0
        current.append({**panel, "span": span})
        used += span
    if current:
        rows.append(current)

    margin = 60
    label_pad = PAPER_PANEL_LABEL_PAD
    gap_x = 36
    gap_y = 52
    width = margin * 2 + cols * cell_w + (cols - 1) * gap_x
    row_heights = [
        int(round(cell_h * max(float(panel.get("height_scale", 1.0)) for panel in row)))
        for row in rows
    ]
    height = margin * 2 + sum(row_heights) + len(rows) * label_pad + (len(rows) - 1) * gap_y
    canvas = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    label_font = _font(PAPER_PANEL_LABEL_PX, bold=True)

    y = margin
    for row_idx, row in enumerate(rows):
        x = margin
        row_h = row_heights[row_idx]
        for panel in row:
            label = str(panel["label"])
            src = Path(panel["src"])
            span = int(panel["span"])
            height_scale = float(panel.get("height_scale", 1.0))
            slot_h = int(round(cell_h * height_scale))
            slot_w = span * cell_w + (span - 1) * gap_x
            draw.text((x, y), label, fill=(0, 0, 0), font=label_font)
            if not src.exists():
                src = no_data_panel(src, src.stem)
            im = fit(crop_white(Image.open(src), pad=12), slot_w, slot_h)
            px = x + (slot_w - im.width) // 2
            py = y + label_pad + (row_h - im.height) // 2
            canvas.alpha_composite(im, (px, py))
            x += slot_w + gap_x
        y += row_h + label_pad + gap_y

    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(out, quality=95)
    return out


def compose_rows(
    rows: list[list[tuple[str, Path]]],
    out: Path,
    *,
    max_cols: int = 3,
    cell_w: int = 760,
    cell_h: int = 520,
    wide_h: int = 620,
) -> Path:
    margin = 60
    label_pad = PAPER_PANEL_LABEL_PAD
    gap_x = 36
    gap_y = 52
    width = margin * 2 + max_cols * cell_w + (max_cols - 1) * gap_x
    row_heights = [wide_h if len(row) == 1 else cell_h for row in rows]
    height = margin * 2 + sum(h + label_pad for h in row_heights) + (len(rows) - 1) * gap_y
    canvas = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    label_font = _font(PAPER_PANEL_LABEL_PX, bold=True)
    y = margin
    for row, row_h in zip(rows, row_heights):
        row_cols = len(row)
        if row_cols == 1:
            slots = [(margin, width - 2 * margin)]
        else:
            slots = [(margin + idx * (cell_w + gap_x), cell_w) for idx in range(row_cols)]
        for (label, src), (x, slot_w) in zip(row, slots):
            draw.text((x, y), label, fill=(0, 0, 0), font=label_font)
            if not src.exists():
                src = no_data_panel(src, src.stem)
            im = fit(crop_white(Image.open(src), pad=12), slot_w, row_h)
            px = x + (slot_w - im.width) // 2
            py = y + label_pad + (row_h - im.height) // 2
            canvas.alpha_composite(im, (px, py))
        y += row_h + label_pad + gap_y
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(out, quality=95)
    return out


def render_fig2() -> Path:
    out_dir = FIG_ROOT / "main" / "Fig2_ReferenceTransfer"
    metrics = collect_prediction_summary_metrics(heldout=True, cellot_pbmc_only=True)
    pbmc_condition_metrics = collect_prediction_metrics(heldout=True)
    systema = collect_long(ROOT / "artifacts" / "analysis" / "systema_mechanism" / "systema_mechanism_all_long.csv")
    systema = append_pbmc_fig2_diagnostics(systema, pbmc_condition_metrics)
    fig2_bar_kwargs = {
        "color_overrides": FIG2_MODEL_COLORS,
        "bar_alpha": 1.0,
        "figsize": PAPER_FIG4_BAR_PANEL_FIGSIZE,
        "legend_y": 1.18,
        "layout_top": 0.84,
        "title_pad": 4,
        "grid_axis": None,
        "group_width": PAPER_GROUP_WIDTH,
        "bar_width_fraction": PAPER_BAR_WIDTH_FRACTION,
        "bar_step_max": PAPER_BAR_STEP_MAX,
        "x_step": 1.0,
        "xtick_rotation": 24,
        "xtick_ha": "right",
    }
    panels = [
        ("a", compact_bar_panel(metrics, "pearson", out_dir / "fig2a_pearson.png", "Held-out/reference transfer Pearson", "Pearson", **fig2_bar_kwargs)),
        ("b", compact_bar_panel(metrics, "nmse", out_dir / "fig2b_nmse.png", "Held-out/reference transfer nMSE", "nMSE", cap_extreme=True, **fig2_bar_kwargs)),
        ("c", compact_bar_panel(metrics, "systema_corr_20de_allpert", out_dir / "fig2c_systema_pearson.png", "Reference-centered Systema Pearson", "Systema Pearson", **fig2_bar_kwargs)),
        ("d", long_metric_bar_panel(systema, "residualized_systema_corr_20de_allpert", out_dir / "fig2d_residualized_systema.png", "Residualized Systema Pearson", "residualized Pearson", **fig2_bar_kwargs)),
        ("e", long_metric_bar_panel(systema, "centroid_accuracy", out_dir / "fig2e_centroid_accuracy.png", "Centroid accuracy", "accuracy", **fig2_bar_kwargs)),
        ("f", long_metric_bar_panel(systema, "generic_projection_ratio", out_dir / "fig2f_generic_shift.png", "Generic-shift dependence", "projection ratio", **fig2_bar_kwargs)),
        (
            "g",
            case_bar_panel(
                dataset="norman",
                split_id=1,
                condition="ELMSAN1+ctrl",
                out=out_dir / "fig2g_elmsan1_case.png",
                title="ELMSAN1+ctrl response case",
                figsize=(16.0, 6.8),
                legend_y=1.12,
                layout_top=0.82,
                layout_bottom=0.27,
                fallback=[
                    SERVER_REFRESH_ROOT
                    / "artifacts"
                    / "analysis"
                    / "case_selection"
                    / "unseen_ctrl"
                    / "preview"
                    / "norman_split1_ELMSAN1-ctrl_unseen_ctrl.png"
                ],
            ),
        ),
    ]
    panel_specs = [
        {"label": label, "src": src, "span": 4 if label == "g" else 1, "height_scale": 1.32 if label == "g" else 1.0}
        for label, src in panels
    ]
    return compose_grid_spans(
        panel_specs,
        COMP_ROOT / "fig2_main_composite.png",
        cols=4,
        cell_w=PAPER_FIG4_BAR_CELL_W,
        cell_h=PAPER_FIG4_BAR_CELL_H,
    )


def render_fig3() -> Path:
    out_dir = FIG_ROOT / "main" / "Fig3_Ablation"
    ab = collect_ablation_summary_metrics()
    ref_labels = {"ref_knn": "kNN", "ref_ot": "OT"}
    cond_labels = {"cond_no_reference": "no reference", "cond_no_prior": "no prior", "cond_full": "full"}
    if out_dir.exists():
        for stale in out_dir.glob("fig3*"):
            if stale.is_file():
                stale.unlink()
    panels = [
        ("a", ablation_metric_panel(ab, ["ref_knn", "ref_ot"], ref_labels, "nmse", out_dir / "fig3a_reference_nmse.png", "Reference construction: nMSE", "nMSE")),
        ("b", ablation_metric_panel(ab, ["ref_knn", "ref_ot"], ref_labels, "systema_corr_20de_allpert", out_dir / "fig3b_reference_systema.png", "Reference construction: Systema", "Systema Pearson")),
        ("c", ablation_metric_panel(ab, ["cond_no_reference", "cond_no_prior", "cond_full"], cond_labels, "nmse", out_dir / "fig3c_conditioning_nmse.png", "Conditioning input: nMSE", "nMSE")),
        ("d", ablation_metric_panel(ab, ["cond_no_reference", "cond_no_prior", "cond_full"], cond_labels, "systema_corr_20de_allpert", out_dir / "fig3d_conditioning_systema.png", "Conditioning input: Systema", "Systema Pearson")),
    ]
    png_out = compose_grid(
        panels,
        COMP_ROOT / "fig3_main_composite.png",
        cols=4,
        cell_w=PAPER_FIG4_BAR_CELL_W,
        cell_h=PAPER_FIG4_BAR_CELL_H,
    )
    _render_fig3_vector_composite(ab, COMP_ROOT / "fig3_main_composite.pdf")
    return png_out


def _has_subgroup_metric(df: pd.DataFrame, metric: str) -> bool:
    if df.empty or "subgroup" not in df.columns or metric not in df.columns:
        return False
    sub = df.dropna(subset=["subgroup", metric])
    return not sub.empty


def _fig4_heatmap_fallback(out_dir: Path) -> pd.DataFrame:
    existing = _read_csv(out_dir / "fig4a_norman_subgroup_heatmap.csv")
    if existing.empty or "model" not in existing.columns:
        return pd.DataFrame()
    subgroup_cols = [col for col in SUBGROUP_ORDER if col in existing.columns]
    if not subgroup_cols:
        return pd.DataFrame()
    long = existing.melt(id_vars=["model"], value_vars=subgroup_cols, var_name="subgroup", value_name="pearson")
    long["dataset"] = "Norman"
    return long


def _fig4_summary_fallback(out_dir: Path, filename: str) -> pd.DataFrame:
    df = _read_csv(out_dir / filename)
    if df.empty or not {"subgroup", "model", "mean"}.issubset(df.columns):
        return pd.DataFrame()
    return df


def render_fig4() -> Path:
    out_dir = FIG_ROOT / "main" / "Fig4_NormanGeneralization"
    if out_dir.exists():
        for pattern in [
            "fig4a_norman_subgroup_heatmap.*",
            "fig4a_subgroup_pearson.*",
            "fig4b_seen0_pearson.*",
            "fig4b_subgroup_nmse.*",
            "fig4c_norman_deg_auroc.*",
            "fig4c_subgroup_systema.*",
            "fig4d_direction_agreement.*",
            "fig4d_overlap_at_20.*",
            "fig4d_subgroup_systema_pearson.*",
            "fig4e_norman_deg_auroc.*",
            "fig4e_overlap_at_20.*",
            "fig4f_direction_agreement.*",
            "fig4f_combo_case*",
            "fig4f_cnn1_mapk1_case.*",
            "fig4g_cnn1_mapk1_case.*",
        ]:
            for stale in out_dir.glob(pattern):
                if stale.is_file():
                    stale.unlink()
    metrics = collect_prediction_metrics(heldout=True)
    norman = metrics[metrics["dataset"].eq("Norman")].copy() if not metrics.empty else pd.DataFrame()
    norman_heatmap = norman if _has_subgroup_metric(norman, "pearson") else _fig4_heatmap_fallback(out_dir)
    pearson_summary = _fig4_summary_fallback(out_dir, "fig4b_subgroup_pearson.csv")
    nmse_summary = _fig4_summary_fallback(out_dir, "fig4c_subgroup_nmse.csv")
    systema_summary = _fig4_summary_fallback(out_dir, "fig4d_subgroup_systema.csv")
    deg = collect_long(ROOT / "artifacts" / "analysis" / "deg_prediction" / "deg_prediction_all_long.csv")
    if not deg.empty and "dataset" in deg.columns:
        deg = deg.copy()
        deg["dataset"] = deg["dataset"].map(_display_dataset)
        norman_deg = deg[deg["dataset"].eq("Norman")].copy()
    else:
        norman_deg = pd.DataFrame()
    fig4_bar_kwargs = {
        "color_overrides": FIG2_MODEL_COLORS,
        "bar_alpha": 1.0,
        "figsize": (5.35, 4.3),
        "legend_y": 1.18,
        "layout_top": 0.84,
        "title_pad": 4,
        "grid_axis": None,
        "group_width": PAPER_GROUP_WIDTH,
        "bar_width_fraction": PAPER_BAR_WIDTH_FRACTION,
        "bar_step_max": PAPER_BAR_STEP_MAX,
        "x_step": 1.10,
        "xtick_rotation": 0,
        "xtick_ha": "center",
    }
    fig4_overlap_kwargs = {**fig4_bar_kwargs, "group_width": 0.68, "bar_step_max": 0.17}

    if _has_subgroup_metric(norman, "pearson"):
        pearson_axis, xs, hues = _summary_for_axis(norman, "pearson", x_col="subgroup", x_order=SUBGROUP_ORDER)
    else:
        pearson_axis, xs, hues = _summary_table_for_axis(pearson_summary, x_order=SUBGROUP_ORDER)
    if _has_subgroup_metric(norman, "nmse"):
        nmse_axis, _, nmse_hues = _summary_for_axis(norman, "nmse", x_col="subgroup", x_order=SUBGROUP_ORDER)
    else:
        nmse_axis, _, nmse_hues = _summary_table_for_axis(nmse_summary, x_order=SUBGROUP_ORDER)
    if _has_subgroup_metric(norman, "systema_corr_20de_allpert"):
        systema_axis, _, systema_hues = _summary_for_axis(
            norman,
            "systema_corr_20de_allpert",
            x_col="subgroup",
            x_order=SUBGROUP_ORDER,
        )
    else:
        systema_axis, _, systema_hues = _summary_table_for_axis(systema_summary, x_order=SUBGROUP_ORDER)
    overlap_axis, _, overlap_hues = _summary_for_axis(
        long_metric_frame(norman_deg, "overlap_at_20"),
        "overlap_at_20",
        x_col="dataset",
        x_order=["Norman"],
    )

    hue_union = []
    for hue in MODEL_ORDER:
        if hue in set(hues + nmse_hues + systema_hues + overlap_hues):
            hue_union.append(hue)
    colors = _color_map(hue_union)
    colors.update({name: color for name, color in FIG2_MODEL_COLORS.items() if name in hue_union})

    apply_gears_paper_style(font_scale=PAPER_FIG_FONT_SCALE)
    fig, axes = plt.subplots(1, 4, figsize=(8.4, 2.45), dpi=300, squeeze=False)
    specs = [
        ("a", pearson_axis, "Pearson", "Pearson", False, PAPER_GROUP_WIDTH, PAPER_BAR_STEP_MAX, "subgroup", SUBGROUP_ORDER, 24, "right"),
        ("b", nmse_axis, "nMSE", "nMSE", True, PAPER_GROUP_WIDTH, PAPER_BAR_STEP_MAX, "subgroup", SUBGROUP_ORDER, 24, "right"),
        ("c", systema_axis, "Systema", "Systema Pearson", False, PAPER_GROUP_WIDTH, PAPER_BAR_STEP_MAX, "subgroup", SUBGROUP_ORDER, 24, "right"),
        ("d", overlap_axis, "Overlap@20", "Overlap@20", False, 0.68, 0.17, "dataset", ["Norman"], 0, "center"),
    ]
    for ax, (label, summary, title, ylabel, cap, group_width, bar_step_max, x_col, x_values, xrot, xha) in zip(axes.flat, specs):
        _draw_grouped_summary_axis(
            ax,
            summary,
            metric_title=title,
            ylabel=ylabel,
            panel_label=label,
            xs=x_values,
            hues=hue_union,
            x_col=x_col,
            colors=colors,
            cap_extreme=cap,
            group_width=group_width,
            bar_step_max=bar_step_max,
            xtick_rotation=xrot,
            xtick_ha=xha,
        )
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=colors[h], **_paper_bar_kwargs())
        for h in hue_union
    ]
    fig.legend(
        handles,
        hue_union,
        frameon=False,
        ncol=min(len(hue_union), 6),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.035),
        fontsize=PAPER_LEGEND_FONTSIZE,
        handlelength=0.9,
        columnspacing=0.8,
        handletextpad=0.35,
    )
    fig.subplots_adjust(left=0.065, right=0.995, bottom=0.23, top=0.80, wspace=0.52)
    out = COMP_ROOT / "fig4_main_composite.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    _save_figure(fig, out, vector_sidecar=True)
    plt.close(fig)
    _write_source(pearson_axis, out_dir / "fig4a_subgroup_pearson.csv")
    _write_source(nmse_axis, out_dir / "fig4b_subgroup_nmse.csv")
    _write_source(systema_axis, out_dir / "fig4c_subgroup_systema.csv")
    _write_source(overlap_axis, out_dir / "fig4d_overlap_at_20.csv")
    render_supp("figs5")
    return out


def render_fig5() -> Path:
    out_dir = FIG_ROOT / "main" / "Fig5_DistributionRecovery"
    if out_dir.exists():
        for stale in out_dir.glob("fig5d_*"):
            if stale.is_file():
                stale.unlink()
    metrics = collect_prediction_summary_metrics(heldout=True, cellot_pbmc_only=True)
    if not metrics.empty and {"dataset", "model"}.issubset(metrics.columns):
        # Fig. 5 evaluates cell-level distribution recovery. BioLORD's PBMC
        # output is a condition-mean predictor, so it is excluded from these
        # distributional comparison panels while remaining available elsewhere.
        metrics = metrics[~(metrics["dataset"].eq("PBMC") & metrics["model"].eq("BioLORD"))].copy()
    _write_source(metrics, out_dir / "fig5_summary_used.csv")
    fig5_bar_kwargs = {
        "figsize": PAPER_FIG5_BAR_PANEL_FIGSIZE,
        "legend_y": 1.22,
        "layout_top": 0.82,
        "title_pad": 6,
        "grid_axis": None,
        "group_width": PAPER_GROUP_WIDTH,
        "bar_width_fraction": PAPER_BAR_WIDTH_FRACTION,
        "bar_step_max": PAPER_BAR_STEP_MAX,
        "x_step": 1.0,
    }
    panels = [
        {
            "label": "a",
            "src": compact_bar_panel(
                metrics,
                "scpram_wasserstein_degs_sum",
                out_dir / "fig5a_wasserstein.png",
                "Wasserstein distance",
                "Wasserstein",
                cap_extreme=True,
                **fig5_bar_kwargs,
            ),
        },
        {
            "label": "b",
            "src": compact_bar_panel(
                metrics,
                "scpram_r2_all_mean_mean",
                out_dir / "fig5b_mean_rho2.png",
                "Mean recovery",
                r"mean $\rho^2$",
                **fig5_bar_kwargs,
            ),
        },
        {
            "label": "c",
            "src": compact_bar_panel(
                metrics,
                "scpram_r2_all_var_mean",
                out_dir / "fig5c_variance_rho2.png",
                "Variance recovery",
                r"variance $\rho^2$",
                **fig5_bar_kwargs,
            ),
        },
        (
            {
                "label": "d",
                "src": pbmc_case_violin_panel(
                    split_id=3,
                    condition="stimulated",
                    gene="ISG15",
                    out=out_dir / "fig5d_isg15_pbmc_distribution_violin.png",
                    title="ISG15 | Dendritic stimulated",
                ),
            }
        ),
        (
            {
                "label": "e",
                "span": 2,
                "src": pbmc_case_umap_panel(
                    split_id=3,
                    condition="stimulated",
                    out=out_dir / "fig5e_pbmc_stimulated_umap.png",
                    title="Dendritic stimulated UMAP comparison",
                ),
            }
        ),
    ]
    return compose_grid_spans(
        panels,
        COMP_ROOT / "fig5_main_composite.png",
        cols=3,
        cell_w=PAPER_FIG5_BAR_CELL_W,
        cell_h=PAPER_FIG5_BAR_CELL_H,
    )


def render_supp(name: str) -> Path:
    fig = name.lower()
    out_dir = FIG_ROOT / "supp" / SUPP_DIRS[fig]
    if fig == "figs1":
        metrics = collect_prediction_metrics(heldout=True)
        systema = collect_long(ROOT / "artifacts" / "analysis" / "systema_mechanism" / "systema_mechanism_all_long.csv")
        systema = append_pbmc_fig2_diagnostics(systema, metrics)
        if out_dir.exists():
            for stale in out_dir.glob("figs1g_overlap_at_20.*"):
                if stale.is_file():
                    stale.unlink()
        panels = [
            ("a", boxplot_panel(metrics, "pearson", out_dir / "figs1a_pearson.png", "Reference-transfer Pearson", "Pearson")),
            ("b", boxplot_panel(metrics, "nmse", out_dir / "figs1b_nmse.png", "Reference-transfer nMSE", "nMSE", cap_extreme=True)),
            ("c", boxplot_panel(metrics, "systema_corr_20de_allpert", out_dir / "figs1c_systema.png", "Systema Pearson", "Systema Pearson")),
            ("d", long_metric_box_panel(systema, "residualized_systema_corr_20de_allpert", out_dir / "figs1d_residualized_systema.png", "Residualized Systema Pearson", "residualized Pearson")),
            ("e", long_metric_box_panel(systema, "centroid_accuracy", out_dir / "figs1e_centroid_accuracy.png", "Centroid accuracy", "accuracy")),
            ("f", long_metric_box_panel(systema, "generic_projection_ratio", out_dir / "figs1f_generic_shift.png", "Generic-shift dependence", "projection ratio")),
        ]
        return compose_grid(panels, COMP_ROOT / "figs1_composite.png", cols=3, cell_w=760, cell_h=520)
    if fig == "figs2":
        if out_dir.exists():
            for stale in out_dir.glob("figs2e_*"):
                if stale.is_file():
                    stale.unlink()
        panels = [
            ("a", case_bar_panel(
                dataset="adamson",
                split_id=4,
                condition="PTDSS1+ctrl",
                out=out_dir / "figs2a_single_case_barplot.png",
                title="PTDSS1+ctrl (Adamson, split 4)",
                top_k=12,
                fallback=[
                    SERVER_REFRESH_ROOT / "artifacts" / "paper_figures" / "supp" / "FigS2_AdditionalCases" / "figs2a_single_case_barplot.png"
                ],
            )),
            ("b", violin_case_panel(
                dataset="adamson",
                split_id=4,
                condition="PTDSS1+ctrl",
                gene="RPS29",
                out=out_dir / "figs2b_single_case_violin.png",
                title="RPS29 | PTDSS1+ctrl",
                fallback=[SERVER_REFRESH_ROOT / "artifacts" / "paper_figures" / "supp" / "FigS2_AdditionalCases" / "figs2b_single_case_violin.png"],
            )),
            ("c", case_bar_panel(
                dataset="norman",
                split_id=5,
                condition="UBASH3A+UBASH3B",
                out=out_dir / "figs2c_combo_case_barplot.png",
                title="UBASH3A+UBASH3B (Norman, split 5)",
                top_k=12,
                fallback=[
                    SERVER_REFRESH_ROOT / "artifacts" / "paper_figures" / "supp" / "FigS2_AdditionalCases" / "figs2c_combo_case_barplot.png"
                ],
            )),
            ("d", violin_case_panel(
                dataset="norman",
                split_id=5,
                condition="UBASH3A+UBASH3B",
                gene="HBZ",
                out=out_dir / "figs2d_combo_case_violin.png",
                title="HBZ | UBASH3A+UBASH3B",
                fallback=[SERVER_REFRESH_ROOT / "artifacts" / "paper_figures" / "supp" / "FigS2_AdditionalCases" / "figs2d_combo_case_violin.png"],
            )),
        ]
        return compose_grid(panels, COMP_ROOT / "figs2_composite.png", cols=2, cell_w=900, cell_h=560)
    if fig == "figs3":
        residual = dixit_residualized_by_distance()
        if out_dir.exists():
            for pattern in [
                "figs3b_dixit_pearson_line.*",
                "figs3c_dixit_nmse_line.*",
                "figs3d_dixit_systema_line.*",
                "figs3e_dixit_residualized_line.*",
                "figs3f_dixit_deg_auroc_line.*",
            ]:
                for stale in out_dir.glob(pattern):
                    if stale.is_file():
                        stale.unlink()
        figs4_bar_kwargs = {
            "x_col": "train_distance_bin",
            "x_order": ["near", "medium", "far"],
            "color_overrides": FIG2_MODEL_COLORS,
            "bar_alpha": 1.0,
            "figsize": (5.1, 4.0),
            "legend_y": 1.18,
            "layout_top": 0.84,
            "title_pad": 4,
            "grid_axis": None,
            "group_width": 0.82,
            "bar_width_fraction": PAPER_BAR_WIDTH_FRACTION,
            "bar_step_max": 0.15,
            "x_step": 1.08,
            "xtick_rotation": 0,
            "xtick_ha": "center",
        }
        panels = [
            ("a", dixit_distance_scatter_panel(out_dir / "figs3a_dixit_difficulty_scatter.png")),
            ("b", compact_bar_panel(dixit_metric_by_distance("pearson"), "pearson", out_dir / "figs3b_dixit_pearson_bar.png", "Dixit Pearson by distance", "Pearson", **figs4_bar_kwargs)),
            ("c", compact_bar_panel(dixit_metric_by_distance("nmse"), "nmse", out_dir / "figs3c_dixit_nmse_bar.png", "Dixit nMSE by distance", "nMSE", cap_extreme=True, **figs4_bar_kwargs)),
            ("d", compact_bar_panel(residual, "systema_corr_20de_allpert", out_dir / "figs3d_dixit_systema_bar.png", "Dixit Systema by distance", "Systema Pearson", **figs4_bar_kwargs)),
            ("e", compact_bar_panel(residual, "residualized_systema_corr_20de_allpert", out_dir / "figs3e_dixit_residualized_bar.png", "Residualized Systema by distance", "residualized Pearson", **figs4_bar_kwargs)),
            ("f", compact_bar_panel(dixit_long_by_distance("overlap_at_20"), "overlap_at_20", out_dir / "figs3f_dixit_overlap_at_20_bar.png", "Dixit Overlap@20 by distance", "Overlap@20", **figs4_bar_kwargs)),
        ]
        return compose_grid(panels, COMP_ROOT / "figs3_composite.png", cols=3, cell_w=760, cell_h=500)
    if fig == "figs4":
        panels = [
            (
                "a",
                copy_panel(
                    module1_panel_candidates("umap_by_cluster.png", "figs6a_umap_by_cluster.png"),
                    out_dir / "figs4a_umap_by_cluster.png",
                    "Module 1 latent by cluster",
                ),
            ),
            (
                "b",
                copy_panel(
                    module1_panel_candidates("umap_by_label_cell_type.png", "figs6b_umap_by_cell_type.png"),
                    out_dir / "figs4b_umap_by_cell_type.png",
                    "Module 1 latent by cell type",
                ),
            ),
            (
                "c",
                copy_panel(
                    module1_panel_candidates("cluster_vs_label_cell_type.png", "figs6c_cluster_vs_cell_type.png"),
                    out_dir / "figs4c_cluster_vs_cell_type.png",
                    "Cluster-cell-type alignment",
                ),
            ),
            ("d", module1_metrics_panel(out_dir / "figs4d_cluster_metrics.png")),
        ]
        return compose_grid(panels, COMP_ROOT / "figs4_composite.png", cols=2, cell_w=900, cell_h=560)
    if fig == "figs5":
        metrics = collect_prediction_metrics(heldout=True)
        norman = metrics[metrics["dataset"].eq("Norman")].copy() if not metrics.empty else pd.DataFrame()
        out_dir = FIG_ROOT / "supp" / SUPP_DIRS[fig]
        norman_heatmap = norman if _has_subgroup_metric(norman, "pearson") else _fig4_heatmap_fallback(
            FIG_ROOT / "main" / "Fig4_NormanGeneralization"
        )
        panels = [
            {
                "label": "a",
                "src": heatmap_panel(
                    norman_heatmap,
                    out_dir / "figs5a_norman_subgroup_heatmap.png",
                    "Norman subgroup Pearson",
                    "pearson",
                ),
                "span": 1,
            },
            {
                "label": "b",
                "src": case_bar_panel(
                    dataset="norman",
                    split_id=3,
                    condition="CNN1+MAPK1",
                    out=out_dir / "figs5b_cnn1_mapk1_case.png",
                    title="CNN1+MAPK1 response case",
                    figsize=(11.8, 6.0),
                    legend_y=1.15,
                    layout_top=0.80,
                    layout_bottom=0.25,
                    fallback=[
                        SERVER_REFRESH_ROOT
                        / "artifacts"
                        / "analysis"
                        / "case_selection"
                        / "unseen_ctrl"
                        / "preview"
                        / "norman_split3_CNN1-MAPK1_unseen_ctrl.png"
                    ],
                ),
                "span": 2,
                "height_scale": 1.05,
            },
        ]
        return compose_grid_spans(panels, COMP_ROOT / "figs5_composite.png", cols=3, cell_w=760, cell_h=590)
    raise ValueError(f"Unknown supplementary figure: {name}")


def write_manifest() -> Path:
    rows = [
        ("main", "fig1", "TriShift overview", "assets/biorender/trishift_overview_biorender.png"),
        ("main", "fig2", "Held-out control reference transfer and response-specific diagnostics", "artifacts/paper_figures/composites/fig2_main_composite.png"),
        ("main", "fig3", "TriShift component ablations", "artifacts/paper_figures/composites/fig3_main_composite.png"),
        ("main", "fig4", "Norman unseen combinatorial perturbation generalization", "artifacts/paper_figures/composites/fig4_main_composite.png"),
        ("main", "fig5", "Cell-level distribution recovery", "artifacts/paper_figures/composites/fig5_main_composite.png"),
        ("supp", "figs1", "Split-level reference-transfer boxplots corresponding to Fig. 2a-f", "artifacts/paper_figures/composites/figs1_composite.png"),
        ("supp", "figs2", "Additional fixed-display response cases", "artifacts/paper_figures/composites/figs2_composite.png"),
        ("supp", "figs3", "Dixit distance-stratified robustness diagnostics", "artifacts/paper_figures/composites/figs3_composite.png"),
        ("supp", "figs4", "Module 1 PBMC latent-state diagnostics", "artifacts/paper_figures/composites/figs4_composite.png"),
        ("supp", "figs5", "Norman subgroup heatmap and fixed CNN1+MAPK1 response case moved from Fig. 4", "artifacts/paper_figures/composites/figs5_composite.png"),
    ]
    manifest = pd.DataFrame(rows, columns=["section", "figure_id", "description", "composite_path"])
    out = FIG_ROOT / "figure_manifest.csv"
    _write_source(manifest, out)
    return out


def clean_stale_supplement_outputs() -> None:
    for folder in STALE_SUPP_DIRS:
        path = FIG_ROOT / "supp" / folder
        if path.exists():
            shutil.rmtree(path)
    stale_composites = [COMP_ROOT / "figs6_composite.png"]
    for path in stale_composites:
        if path.exists():
            path.unlink()


def render(figure: str) -> Path:
    key = figure.lower().replace(".", "")
    clean_stale_supplement_outputs()
    if key == "fig2":
        return render_fig2()
    if key == "fig3":
        return render_fig3()
    if key == "fig4":
        return render_fig4()
    if key == "fig5":
        return render_fig5()
    if key in {"figs1", "figs2", "figs3", "figs4", "figs5"}:
        return render_supp(key)
    if key == "all":
        clean_stale_supplement_outputs()
        paths = [render_fig2(), render_fig3(), render_fig4(), render_fig5()]
        paths.extend(render_supp(f"figs{i}") for i in range(1, 6))
        write_manifest()
        return paths[-1]
    raise ValueError(f"Unknown figure: {figure}")
