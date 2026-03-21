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
    parse_models,
    parse_split_ids,
    load_payload_item,
    normalize_condition,
    ts_local,
    warn_skip,
    write_run_meta,
)


def _vector_mean(item: dict[str, Any], key: str) -> np.ndarray:
    arr = np.asarray(item[key], dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"{key} must be a 2D array")
    return arr.mean(axis=0).astype(np.float32, copy=False)


def _normalize_space(space: str) -> str:
    key = normalize_condition(str(space))
    aliases = {
        "full": "full_gene",
        "fullgene": "full_gene",
        "full_gene": "full_gene",
        "deg": "deg",
        "degs": "deg",
        "auto": "auto",
    }
    if key not in aliases:
        raise ValueError("space must be one of: full_gene, deg, auto")
    return aliases[key]


def _gene_name_array(item: dict[str, Any], key: str) -> np.ndarray:
    if key not in item:
        raise KeyError(f"Missing {key}")
    arr = np.asarray(item[key], dtype=object).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{key} is empty")
    return arr


def _delta_on_gene_names(item: dict[str, Any], genes: list[str]) -> tuple[np.ndarray, np.ndarray]:
    gene_names = _gene_name_array(item, "DE_name")
    index = {str(g): i for i, g in enumerate(gene_names.tolist())}
    missing = [g for g in genes if g not in index]
    if missing:
        raise KeyError(f"Missing genes in DE_name: {', '.join(missing[:5])}")
    idx = np.asarray([index[g] for g in genes], dtype=int)
    truth = np.asarray(item["Truth"], dtype=np.float32)
    pred = np.asarray(item["Pred"], dtype=np.float32)
    ctrl = np.asarray(item["Ctrl"], dtype=np.float32)
    if truth.ndim != 2 or pred.ndim != 2 or ctrl.ndim != 2:
        raise ValueError("Pred/Truth/Ctrl must be 2D arrays for DEG space")
    truth_delta = truth[:, idx].mean(axis=0).astype(np.float32, copy=False) - ctrl[:, idx].mean(axis=0).astype(np.float32, copy=False)
    pred_delta = pred[:, idx].mean(axis=0).astype(np.float32, copy=False) - ctrl[:, idx].mean(axis=0).astype(np.float32, copy=False)
    return truth_delta, pred_delta


def _extract_nonadd_vectors(
    *,
    item_ab: dict[str, Any],
    item_a: dict[str, Any],
    item_b: dict[str, Any],
    space: str,
) -> tuple[np.ndarray, np.ndarray, str]:
    requested = _normalize_space(space)
    if requested in {"full_gene", "auto"}:
        needed = ["Pred_full", "Truth_full", "Ctrl_full"]
        if all(k in item_ab and k in item_a and k in item_b for k in needed):
            truth_a = _vector_mean(item_a, "Truth_full") - _vector_mean(item_a, "Ctrl_full")
            truth_b = _vector_mean(item_b, "Truth_full") - _vector_mean(item_b, "Ctrl_full")
            truth_ab = _vector_mean(item_ab, "Truth_full") - _vector_mean(item_ab, "Ctrl_full")
            pred_a = _vector_mean(item_a, "Pred_full") - _vector_mean(item_a, "Ctrl_full")
            pred_b = _vector_mean(item_b, "Pred_full") - _vector_mean(item_b, "Ctrl_full")
            pred_ab = _vector_mean(item_ab, "Pred_full") - _vector_mean(item_ab, "Ctrl_full")
            return truth_ab - (truth_a + truth_b), pred_ab - (pred_a + pred_b), "full_gene"
        if requested == "full_gene":
            raise KeyError("Missing Pred_full/Truth_full/Ctrl_full for full_gene space")

    genes_ab = set(map(str, _gene_name_array(item_ab, "DE_name").tolist()))
    genes_a = set(map(str, _gene_name_array(item_a, "DE_name").tolist()))
    genes_b = set(map(str, _gene_name_array(item_b, "DE_name").tolist()))
    common_genes = sorted(genes_ab & genes_a & genes_b)
    if len(common_genes) < 2:
        raise ValueError("Need at least 2 shared DE genes across combo and both singles for DEG space")

    truth_a, pred_a = _delta_on_gene_names(item_a, common_genes)
    truth_b, pred_b = _delta_on_gene_names(item_b, common_genes)
    truth_ab, pred_ab = _delta_on_gene_names(item_ab, common_genes)
    return truth_ab - (truth_a + truth_b), pred_ab - (pred_a + pred_b), "deg"


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=float).reshape(-1)
    y = np.asarray(b, dtype=float).reshape(-1)
    if x.size == 0 or y.size == 0 or np.allclose(np.std(x), 0.0) or np.allclose(np.std(y), 0.0):
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _r2(a: np.ndarray, b: np.ndarray) -> float:
    truth = np.asarray(a, dtype=float).reshape(-1)
    pred = np.asarray(b, dtype=float).reshape(-1)
    if truth.size == 0:
        return float("nan")
    denom = float(np.sum((truth - truth.mean()) ** 2))
    if np.isclose(denom, 0.0):
        return float("nan")
    num = float(np.sum((truth - pred) ** 2))
    return float(1.0 - num / denom)


def _synergy_sign_accuracy(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    truth = np.asarray(a, dtype=float).reshape(-1)
    pred = np.asarray(b, dtype=float).reshape(-1)
    mask = np.abs(truth) > float(eps)
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.sign(truth[mask]) == np.sign(pred[mask])))


def _interaction_strength_mae(a: np.ndarray, b: np.ndarray) -> float:
    truth = np.asarray(a, dtype=float).reshape(-1)
    pred = np.asarray(b, dtype=float).reshape(-1)
    if truth.size == 0:
        return float("nan")
    return float(np.mean(np.abs(truth - pred)))


def _combo_single_keys(condition: str) -> tuple[str, str]:
    tokens = [x for x in normalize_condition(str(condition)).split("+") if x and x != "ctrl"]
    if len(tokens) != 2:
        raise ValueError("Only 2-gene combo conditions are supported")
    a, b = sorted(tokens)
    return normalize_condition(f"{a}+ctrl"), normalize_condition(f"{b}+ctrl")


def _build_subgroup_lookup(conditions: list[str], split_id: int) -> dict[str, str]:
    pert_list = [normalize_condition(x) for x in conditions]
    if "ctrl" not in pert_list:
        pert_list = ["ctrl"] + pert_list
    try:
        df = norman_subgroup(pert_list, seed=int(split_id))
        return {normalize_condition(str(idx)): str(row["subgroup"]) for idx, row in df.iterrows()}
    except Exception:
        out: dict[str, str] = {}
        for cond in pert_list:
            key = normalize_condition(cond)
            tokens = [x for x in key.split("+") if x and x != "ctrl"]
            if key == "ctrl":
                out[key] = "unknown"
            elif len(tokens) == 1:
                out[key] = "single"
            else:
                out[key] = "seen0"
        return out


def _render_scatter(df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(6, 5), dpi=220)
    if df.empty:
        plt.text(0.5, 0.5, "No valid combo conditions", ha="center", va="center")
        plt.axis("off")
    else:
        x = pd.to_numeric(df["nonadd_pearson"], errors="coerce")
        y = pd.to_numeric(df["nonadd_r2"], errors="coerce")
        plt.scatter(x, y, alpha=0.8, s=28)
        plt.xlabel("nonadd_pearson")
        plt.ylabel("nonadd_r2")
        plt.title("Norman non-additive metrics")
        plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _render_subgroup_barplot(df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(7, 4.5), dpi=220)
    if df.empty:
        plt.text(0.5, 0.5, "No subgroup results", ha="center", va="center")
        plt.axis("off")
    else:
        order = ["seen0", "seen1", "seen2"]
        work = df.copy()
        work["subgroup"] = pd.Categorical(work["subgroup"], categories=order, ordered=True)
        work = work.sort_values("subgroup")
        plt.bar(work["subgroup"].astype(str), work["nonadd_pearson_mean"], color="#2E8B57")
        plt.ylabel("mean nonadd_pearson")
        plt.title("Norman non-additive by subgroup")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def run_norman_nonadd_experiment(
    *,
    dataset: str,
    models: str | list[str] | tuple[str, ...] | None = None,
    split_ids: int | str | list[int] | tuple[int, ...] = "1",
    out_root: str | Path | None = None,
    space: str = "full_gene",
) -> dict[str, Any]:
    dataset_key = normalize_condition(str(dataset))
    if dataset_key != "norman":
        raise ValueError("norman_nonadd_experiment only supports dataset=norman")
    space_key = _normalize_space(space)
    model_requests = parse_models(models or ["trishift_nearest", "scouter", "gears", "genepert"])
    split_list = parse_split_ids(split_ids)
    out_dir = Path(out_root).resolve() if out_root else (Path("artifacts/analysis") / f"{ts_local()}_norman_nonadd").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []
    used_models: list[str] = []
    for model_name in model_requests:
        try:
            _, payload = load_payload_item(dataset="norman", model_name=model_name, split_id=int(split_list[0]), condition=None)
        except Exception:
            payload = None
        if payload is None:
            warn_skip(f"[norman_nonadd] skip model={model_name}: payload unavailable")
            skipped_rows.append({"model_name": str(model_name), "split_id": "", "condition": "", "skip_reason": "payload unavailable"})
            continue
        used_models.append(str(model_name))

        for split_id in split_list:
            try:
                _, payload = load_payload_item(dataset="norman", model_name=model_name, split_id=int(split_id), condition=None)
            except Exception as exc:
                warn_skip(f"[norman_nonadd] skip model={model_name} split={split_id}: {exc}")
                skipped_rows.append({"model_name": str(model_name), "split_id": int(split_id), "condition": "", "skip_reason": str(exc)})
                continue
            normalized_payload = {normalize_condition(str(k)): v for k, v in payload.items() if isinstance(v, dict)}
            subgroup_lookup = _build_subgroup_lookup(list(normalized_payload.keys()), int(split_id))
            for condition, item_ab in sorted(normalized_payload.items()):
                tokens = [x for x in condition.split("+") if x and x != "ctrl"]
                if len(tokens) != 2:
                    continue
                try:
                    single_a, single_b = _combo_single_keys(condition)
                except Exception as exc:
                    skipped_rows.append({"model_name": str(model_name), "split_id": int(split_id), "condition": str(condition), "skip_reason": str(exc)})
                    continue
                if single_a not in normalized_payload or single_b not in normalized_payload:
                    skipped_rows.append(
                        {
                            "model_name": str(model_name),
                            "split_id": int(split_id),
                            "condition": str(condition),
                            "skip_reason": f"missing single perturbation: {single_a if single_a not in normalized_payload else single_b}",
                        }
                    )
                    continue
                try:
                    truth_nonadd, pred_nonadd, space_used = _extract_nonadd_vectors(
                        item_ab=item_ab,
                        item_a=normalized_payload[single_a],
                        item_b=normalized_payload[single_b],
                        space=space_key,
                    )
                except Exception as exc:
                    skipped_rows.append({"model_name": str(model_name), "split_id": int(split_id), "condition": str(condition), "skip_reason": str(exc)})
                    continue
                rows.append(
                    {
                        "dataset": "norman",
                        "model_name": str(model_name),
                        "split_id": int(split_id),
                        "condition": str(condition),
                        "subgroup": subgroup_lookup.get(condition, "unknown"),
                        "space_used": str(space_used),
                        "nonadd_pearson": _pearson(truth_nonadd, pred_nonadd),
                        "nonadd_r2": _r2(truth_nonadd, pred_nonadd),
                        "synergy_sign_accuracy": _synergy_sign_accuracy(truth_nonadd, pred_nonadd),
                        "interaction_strength_mae": _interaction_strength_mae(truth_nonadd, pred_nonadd),
                    }
                )

    per_condition_df = pd.DataFrame(rows).sort_values(["model_name", "split_id", "condition"]).reset_index(drop=True) if rows else pd.DataFrame()
    skipped_df = pd.DataFrame(skipped_rows)
    summary_df = (
        per_condition_df.groupby(["model_name"], as_index=False)[["nonadd_pearson", "nonadd_r2", "synergy_sign_accuracy", "interaction_strength_mae"]]
        .mean()
        .rename(columns={c: f"{c}_mean" for c in ["nonadd_pearson", "nonadd_r2", "synergy_sign_accuracy", "interaction_strength_mae"]})
        if not per_condition_df.empty
        else pd.DataFrame()
    )
    subgroup_df = (
        per_condition_df.groupby(["model_name", "subgroup"], as_index=False)[["nonadd_pearson", "nonadd_r2", "synergy_sign_accuracy", "interaction_strength_mae"]]
        .mean()
        .rename(columns={c: f"{c}_mean" for c in ["nonadd_pearson", "nonadd_r2", "synergy_sign_accuracy", "interaction_strength_mae"]})
        if not per_condition_df.empty
        else pd.DataFrame()
    )

    per_condition_df.to_csv(out_dir / "norman_nonadd_per_condition.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(out_dir / "norman_nonadd_summary.csv", index=False, encoding="utf-8-sig")
    subgroup_df.to_csv(out_dir / "norman_nonadd_by_subgroup.csv", index=False, encoding="utf-8-sig")
    skipped_df.to_csv(out_dir / "norman_nonadd_skipped.csv", index=False, encoding="utf-8-sig")
    _render_scatter(per_condition_df, out_dir / "norman_nonadd_scatter.png")
    subgroup_plot_df = subgroup_df[subgroup_df["model_name"] == subgroup_df["model_name"].iloc[0]].copy() if not subgroup_df.empty else subgroup_df
    _render_subgroup_barplot(subgroup_plot_df, out_dir / "norman_nonadd_by_subgroup.png")

    write_run_meta(
        out_dir / "run_meta.json",
        {
            "dataset": "norman",
            "models_requested": model_requests,
            "models_used": used_models,
            "split_ids": split_list,
            "space": str(space_key),
            "skipped_rows": skipped_rows,
            "out_dir": str(out_dir),
        },
    )
    return {
        "out_dir": out_dir,
        "per_condition_df": per_condition_df,
        "summary_df": summary_df,
        "subgroup_df": subgroup_df,
        "skipped_df": skipped_df,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run Norman non-additive interaction analysis from exported PKLs.")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--models", default="trishift_nearest,scouter,gears,genepert")
    ap.add_argument("--split_ids", default="1")
    ap.add_argument("--out_root", default="")
    ap.add_argument("--space", default="full_gene", choices=["full_gene", "deg", "auto"])
    args = ap.parse_args(argv)

    result = run_norman_nonadd_experiment(
        dataset=str(args.dataset).strip(),
        models=str(args.models).strip(),
        split_ids=str(args.split_ids).strip(),
        out_root=str(args.out_root).strip() or None,
        space=str(args.space).strip(),
    )
    print(f"out_dir: {result['out_dir']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
