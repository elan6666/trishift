from __future__ import annotations

import json
import pickle
import warnings
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from trishift._utils import apply_alias_mapping, load_embedding_df, load_yaml, normalize_condition


DEFAULT_PAYLOAD_ROOTS = {
    "trishift": REPO_ROOT / "artifacts" / "results",
    "scouter": REPO_ROOT / "artifacts" / "results" / "scouter",
    "biolord": REPO_ROOT / "artifacts" / "results" / "biolord",
    "gears": REPO_ROOT / "artifacts" / "results" / "gears",
    "genepert": REPO_ROOT / "artifacts" / "results" / "genepert",
    "scgpt": REPO_ROOT / "artifacts" / "results" / "scgpt",
}

DATASET_EMBEDDING_KEYS = {
    "adamson": "emb_b",
    "dixit": "emb_c",
    "norman": "emb_a",
    "replogle_k562_essential": "emb_c",
    "replogle_rpe1_essential": "emb_b",
}

DEFAULT_MODEL_REQUESTS = [
    "trishift_nearest",
    "trishift_random",
    "biolord",
    "gears",
    "genepert",
    "scgpt",
    "systema_nonctl_mean",
    "systema_matching_mean",
]


@dataclass(frozen=True)
class ModelSpec:
    request_id: str
    kind: str
    base_model: str | None
    variant_tag: str | None
    label: str


@dataclass
class ResolvedResult:
    spec: ModelSpec
    dataset: str
    result_root: Path
    metrics_path: Path
    mean_path: Path
    payload_dir: Path | None
    available_split_ids: list[int]


def ts_local() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def parse_split_ids(split_ids: int | str | list[int] | tuple[int, ...]) -> list[int]:
    if isinstance(split_ids, int):
        return [int(split_ids)]
    if isinstance(split_ids, (list, tuple)):
        out = [int(x) for x in split_ids]
        if not out:
            raise ValueError("split_ids cannot be empty")
        return out
    text = str(split_ids).strip()
    if not text:
        raise ValueError("split_ids cannot be empty")
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_models(models: str | list[str] | tuple[str, ...] | None) -> list[str]:
    if models is None:
        return list(DEFAULT_MODEL_REQUESTS)
    if isinstance(models, (list, tuple)):
        out = [str(x).strip() for x in models if str(x).strip()]
        if not out:
            raise ValueError("models cannot be empty")
        return out
    out = [x.strip() for x in str(models).split(",") if x.strip()]
    if not out:
        raise ValueError("models cannot be empty")
    return out


def resolve_model_spec(model_name: str) -> ModelSpec:
    key = str(model_name).strip().lower()
    if key == "trishift":
        return ModelSpec(key, "payload", "trishift", None, "TriShift")
    if key == "trishift_nearest":
        return ModelSpec(key, "payload", "trishift", "nearest", "TriShift nearest")
    if key == "trishift_random":
        return ModelSpec(key, "payload", "trishift", "random", "TriShift random")
    if key in {"scouter", "biolord", "gears", "genepert", "scgpt"}:
        label_map = {
            "scouter": "Scouter",
            "biolord": "biolord",
            "gears": "GEARS",
            "genepert": "GenePert",
            "scgpt": "scGPT",
        }
        return ModelSpec(key, "payload", key, None, label_map[key])
    if key == "systema_nonctl_mean":
        return ModelSpec(key, "systema", None, None, "Systema nonctl-mean")
    if key == "systema_matching_mean":
        return ModelSpec(key, "systema", None, None, "Systema matching-mean")
    raise ValueError(f"Unsupported model request: {model_name}")


def default_payload_root(base_model: str, dataset: str) -> Path:
    root = DEFAULT_PAYLOAD_ROOTS[base_model]
    return (root / dataset).resolve()


def default_systema_root(dataset: str) -> Path:
    base = (REPO_ROOT / "artifacts" / "results" / dataset).resolve()
    candidates = sorted([p for p in base.glob("systema_baselines_*") if p.is_dir()])
    if not candidates:
        raise FileNotFoundError(f"No systema_baselines_* directory found under {base}")
    return candidates[-1]


def metrics_filename_for_spec(spec: ModelSpec) -> str:
    if spec.kind == "systema":
        return "metrics.csv"
    if spec.base_model == "trishift" and spec.variant_tag:
        return f"metrics_{spec.variant_tag}.csv"
    return "metrics.csv"


def mean_filename_for_spec(spec: ModelSpec) -> str:
    if spec.kind == "systema":
        return "mean_pearson.txt"
    if spec.base_model == "trishift" and spec.variant_tag:
        return f"mean_pearson_{spec.variant_tag}.txt"
    return "mean_pearson.txt"


def payload_filename(base_model: str, dataset: str, split_id: int, variant_tag: str | None) -> str:
    suffix = f"_{variant_tag}" if variant_tag and base_model in {"trishift", "scouter"} else ""
    return f"{base_model}_{dataset}_{int(split_id)}{suffix}.pkl"


def _discover_available_payload_splits(root: Path, base_model: str, dataset: str, variant_tag: str | None) -> list[int]:
    prefix = f"{base_model}_{dataset}_"
    suffix = f"_{variant_tag}.pkl" if variant_tag and base_model in {"trishift", "scouter"} else ".pkl"
    out: list[int] = []
    for p in root.glob(f"{prefix}*{suffix}"):
        name = p.name
        if not name.startswith(prefix) or not name.endswith(suffix):
            continue
        mid = name[len(prefix) : -len(suffix)]
        if mid.isdigit():
            out.append(int(mid))
    return sorted(set(out))


def _discover_metric_splits(path: Path) -> list[int]:
    if not path.exists():
        return []
    try:
        df = pd.read_csv(path, usecols=["split_id"])
    except Exception:
        return []
    split_series = pd.to_numeric(df.get("split_id"), errors="coerce").dropna().astype(int)
    return sorted(set(split_series.tolist()))


def _select_metrics_root(
    root: Path,
    metrics_name: str,
    payload_split_ids: list[int],
) -> Path:
    expected = set(int(x) for x in payload_split_ids)
    candidates = [root]
    candidates.extend(sorted([p for p in root.iterdir() if p.is_dir()]))

    best_root = root
    best_score = (-1, -1, -1.0)
    for candidate in candidates:
        metrics_path = candidate / metrics_name
        metric_splits = _discover_metric_splits(metrics_path)
        metric_set = set(metric_splits)
        overlap = len(metric_set & expected) if expected else len(metric_set)
        complete = int(bool(expected) and expected.issubset(metric_set))
        mtime = metrics_path.stat().st_mtime if metrics_path.exists() else -1.0
        score = (complete, overlap, mtime)
        if score > best_score:
            best_score = score
            best_root = candidate
    return best_root


def resolve_result(
    *,
    dataset: str,
    model_name: str,
    result_dir: str | Path | None = None,
    systema_root: str | Path | None = None,
) -> ResolvedResult:
    spec = resolve_model_spec(model_name)
    dataset_key = str(dataset).strip()
    if spec.kind == "payload":
        payload_root = Path(result_dir).resolve() if result_dir else default_payload_root(spec.base_model or "", dataset_key)
        split_ids = _discover_available_payload_splits(payload_root, spec.base_model or "", dataset_key, spec.variant_tag)
        metrics_root = _select_metrics_root(
            payload_root,
            metrics_filename_for_spec(spec),
            split_ids,
        )
        metrics_path = metrics_root / metrics_filename_for_spec(spec)
        mean_path = metrics_root / mean_filename_for_spec(spec)
        return ResolvedResult(
            spec=spec,
            dataset=dataset_key,
            result_root=metrics_root,
            metrics_path=metrics_path,
            mean_path=mean_path,
            payload_dir=payload_root,
            available_split_ids=split_ids,
        )

    systema_base = Path(systema_root).resolve() if systema_root else default_systema_root(dataset_key)
    subdir = "systema_nonctl_mean" if spec.request_id.endswith("nonctl_mean") else "systema_matching_mean"
    root = systema_base / subdir
    return ResolvedResult(
        spec=spec,
        dataset=dataset_key,
        result_root=root,
        metrics_path=root / "metrics.csv",
        mean_path=root / "mean_pearson.txt",
        payload_dir=None,
        available_split_ids=[],
    )


def parse_mean_file(path: Path) -> dict[str, float]:
    metrics: dict[str, float] = {}
    if not path.exists():
        return metrics
    for i, raw in enumerate(path.read_text(encoding="utf-8").splitlines()):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            if i == 0:
                try:
                    metrics["mean_pearson_line"] = float(line)
                except Exception:
                    pass
            continue
        key, value = line.split("=", 1)
        try:
            metrics[str(key).strip()] = float(str(value).strip())
        except Exception:
            continue
    return metrics


def load_metrics_df(resolved: ResolvedResult) -> pd.DataFrame:
    if not resolved.metrics_path.exists():
        raise FileNotFoundError(resolved.metrics_path)
    return pd.read_csv(resolved.metrics_path)


def load_payload_item(
    *,
    dataset: str,
    model_name: str,
    split_id: int,
    condition: str | None = None,
    result_dir: str | Path | None = None,
) -> dict[str, Any] | tuple[Path, dict[str, Any]]:
    resolved = resolve_result(dataset=dataset, model_name=model_name, result_dir=result_dir)
    if resolved.payload_dir is None or resolved.spec.base_model is None:
        raise ValueError(f"{model_name} does not provide payload PKLs")
    pkl_path = resolved.payload_dir / payload_filename(
        resolved.spec.base_model,
        dataset,
        split_id,
        resolved.spec.variant_tag,
    )
    if not pkl_path.exists():
        raise FileNotFoundError(pkl_path)
    with pkl_path.open("rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict):
        raise TypeError(f"Unexpected payload type at {pkl_path}: {type(payload)}")
    if condition is None:
        return pkl_path, payload
    cond_key = normalize_condition(str(condition))
    normalized = {normalize_condition(str(k)): v for k, v in payload.items()}
    if cond_key not in normalized:
        raise KeyError(f"Condition not found in payload: {condition}")
    return normalized[cond_key]


def condition_tokens(condition: str) -> list[str]:
    tokens = []
    for token in normalize_condition(str(condition)).split("+"):
        tok = str(token).strip()
        if not tok or tok.lower() == "ctrl":
            continue
        tokens.append(tok)
    return tokens


@lru_cache(maxsize=16)
def _cached_embedding_table(dataset: str, paths_path: str) -> pd.DataFrame:
    dataset_key = str(dataset).strip()
    cfg = load_yaml(str(Path(paths_path).resolve()))
    emb_key = DATASET_EMBEDDING_KEYS[dataset_key]
    emb_path = Path(cfg["embeddings"][emb_key])
    embd_df = load_embedding_df(str(emb_path))
    embd_df = apply_alias_mapping(embd_df, dataset_key)
    return embd_df


def condition_embedding(dataset: str, condition: str, paths_path: str | Path = "configs/paths.yaml") -> np.ndarray:
    dataset_key = str(dataset).strip()
    embd_df = _cached_embedding_table(dataset_key, str(Path(paths_path).resolve()))
    tokens = condition_tokens(condition)
    if not tokens:
        return np.zeros(int(embd_df.shape[1]), dtype=np.float32)
    rows: list[np.ndarray] = []
    for token in tokens:
        if token in embd_df.index:
            rows.append(np.asarray(embd_df.loc[token].values, dtype=np.float32))
    if not rows:
        return np.zeros(int(embd_df.shape[1]), dtype=np.float32)
    return np.mean(np.vstack(rows).astype(np.float32, copy=False), axis=0)


def available_conditions_from_payload(payload: dict[str, Any]) -> list[str]:
    return [normalize_condition(str(k)) for k, v in payload.items() if isinstance(v, dict)]


def write_run_meta(path: Path, meta: dict[str, Any]) -> None:
    path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def warn_skip(message: str) -> None:
    warnings.warn(message, RuntimeWarning, stacklevel=2)
