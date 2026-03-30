from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import copy
import importlib
import json
import pickle
import random
import sys

import anndata as ad
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error as mse
try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = ROOT / "src"
LOCAL_DATA_ROOT = ROOT / "src" / "data"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SRC_ROOT))

from trishift import _utils
from trishift._external_metrics import (
    average_of_perturbation_centroids,
    compute_scpram_metrics_from_arrays,
    pearson_delta_reference_metrics,
    regression_r2_safe,
)
from trishift.TriShiftData import TriShiftData
from scripts.common.split_utils import (
    condition_sort as _shared_condition_sort,
    norman_subgroup as _shared_norman_subgroup,
)
from scripts.common.yaml_utils import load_yaml_file


@dataclass(frozen=True)
class ScgptDatasetConfig:
    data_rel: str
    splits: list[int]
    test_ratio: float
    norman_split: bool = False
    lr: float = 1e-4
    batch_size: int = 64
    eval_batch_size: int = 64
    epochs: int = 15
    early_stop: int = 10
    control_pool_size: int = 300
    include_zero_gene: str = "all"
    max_seq_len: int = 1536
    dropout: float = 0.0
    use_fast_transformer: bool = True


PROFILE_DIR = Path(__file__).resolve().parents[1] / "eval" / "configs"
PROFILE_ALLOWED_KEYS = {"dataset", "task_args"}
SPECIAL_TOKENS = ["<pad>", "<cls>", "<eoc>"]
PAD_TOKEN = "<pad>"
PAD_VALUE = 0
PERT_PAD_ID = 0
LOAD_PARAM_PREFIXES = ["encoder", "value_encoder", "transformer_encoder"]


DATASET_CONFIG = {
    "adamson": ScgptDatasetConfig(
        data_rel="data/adamson/perturb_processed.h5ad",
        splits=[1, 2, 3, 4, 5],
        test_ratio=0.2,
    ),
    "dixit": ScgptDatasetConfig(
        data_rel="data/dixit/perturb_processed.h5ad",
        splits=[1, 2, 3, 4, 5],
        test_ratio=0.2,
    ),
    "norman": ScgptDatasetConfig(
        data_rel="data/norman/perturb_processed.h5ad",
        splits=[1, 2, 3, 4, 5],
        test_ratio=0.2,
        norman_split=True,
    ),
    "replogle_k562_essential": ScgptDatasetConfig(
        data_rel="data/replogle_k562_essential/perturb_processed.h5ad",
        splits=[1, 2, 3, 4, 5],
        test_ratio=0.2,
    ),
    "replogle_rpe1_essential": ScgptDatasetConfig(
        data_rel="data/replogle_rpe1_essential/perturb_processed.h5ad",
        splits=[1, 2, 3, 4, 5],
        test_ratio=0.2,
    ),
}


def _progress(iterable, **kwargs):
    if tqdm is None:
        return iterable
    return tqdm(iterable, **kwargs)


def _load_profile(profile: str) -> dict:
    p = PROFILE_DIR / f"{profile}.yaml"
    if not p.exists():
        raise FileNotFoundError(f"scGPT profile not found: {p}")
    obj = load_yaml_file(p)
    unknown = sorted(set(obj.keys()) - PROFILE_ALLOWED_KEYS)
    if unknown:
        raise ValueError(f"Unknown keys in scGPT profile {p.name}: {unknown}")
    dataset = str(obj.get("dataset", "")).strip()
    if dataset != profile:
        raise ValueError(f"Profile dataset mismatch: file={p.name} dataset={dataset!r}")
    task_args = obj.get("task_args") or {}
    if not isinstance(task_args, dict):
        raise TypeError(f"profile.task_args must be a mapping: {p}")
    return {"dataset": dataset, "task_args": task_args}


def run_profile(profile: str) -> None:
    prof = _load_profile(profile)
    task_args = prof["task_args"]
    run_scgpt_eval(
        prof["dataset"],
        base_seed=int(task_args.get("seed", 24)),
        export_notebook_pkl=bool(task_args.get("export_notebook_pkl", True)),
        control_pool_size=int(task_args.get("control_pool_size", 300)),
        epochs=int(task_args.get("epochs", 15)),
        batch_size=int(task_args.get("batch_size", 64)),
        eval_batch_size=int(task_args.get("eval_batch_size", 64)),
        lr=float(task_args.get("lr", 1e-4)),
        early_stop=int(task_args.get("early_stop", 10)),
    )


def _resolve_mean_metric_keys(numeric_means: pd.Series) -> list[str]:
    preferred_order = [
        "pearson",
        "nmse",
        "mse_pred",
        "mse_ctrl",
        "deg_mean_r2",
        "systema_corr_20de_allpert",
        "systema_corr_deg_r2",
        "scpram_r2_degs_mean_mean",
        "scpram_r2_degs_var_mean",
        "scpram_wasserstein_degs_sum",
    ]
    exclude_keys = {"split_id", "n_ensemble"}
    keys = [k for k in preferred_order if k in numeric_means.index and k not in exclude_keys]
    keys.extend([k for k in numeric_means.index if k not in exclude_keys and k not in keys])
    return keys


def _append_subgroup_mean_lines(lines: list[str], metrics_df: pd.DataFrame, keys: list[str]) -> None:
    if "subgroup" not in metrics_df.columns:
        return
    subgroup_series = metrics_df["subgroup"]
    if subgroup_series.isna().all():
        return

    ordered = ["single", "seen0", "seen1", "seen2"]
    present = set(subgroup_series.dropna().astype(str).tolist())
    subgroup_names = [g for g in ordered if g in present]
    if "unknown" in present:
        subgroup_names.append("unknown")
    if not subgroup_names:
        return

    lines.append("# subgroup_means_row_weighted\n")
    lines.append("subgroup_order=single,seen0,seen1,seen2\n")
    lines.append("\n")

    for g in subgroup_names:
        sub_df = metrics_df[metrics_df["subgroup"].astype(str) == g]
        lines.append(f"subgroup_{g}_n_rows={int(len(sub_df))}\n")
        sub_numeric_means = sub_df.mean(numeric_only=True)
        for key in keys:
            if key not in sub_numeric_means.index:
                continue
            lines.append(f"subgroup_{g}_mean_{key}={float(sub_numeric_means[key])}\n")
        lines.append("\n")


def _attach_subgroup_column(metrics_df: pd.DataFrame, subgroup_df: pd.DataFrame | None) -> pd.DataFrame:
    if subgroup_df is None or "condition" not in metrics_df.columns:
        return metrics_df
    if "subgroup" not in subgroup_df.columns:
        return metrics_df

    out = metrics_df.copy()
    if "subgroup" in out.columns:
        out = out.drop(columns=["subgroup"])
    subgroup_map = subgroup_df["subgroup"].astype(str).to_dict()
    subgroup_vals = out["condition"].astype(str).map(subgroup_map).fillna("unknown")
    insert_at = list(out.columns).index("condition") + 1
    out.insert(insert_at, "subgroup", subgroup_vals)
    return out


def _write_mean_metrics(path: Path, metrics_df: pd.DataFrame) -> None:
    numeric_means = metrics_df.mean(numeric_only=True)
    mean_pearson = float(numeric_means.get("pearson", float("nan")))
    keys = _resolve_mean_metric_keys(numeric_means)

    lines = [f"{mean_pearson}\n"]
    for key in keys:
        lines.append(f"mean_{key}={float(numeric_means[key])}\n")
    _append_subgroup_mean_lines(lines, metrics_df, keys)
    path.write_text("".join(lines), encoding="utf-8")


def set_seeds(seed: int = 24) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def condition_sort(x: str) -> str:
    return _shared_condition_sort(x)


def subgroup(pert_list: list[str], seed: int) -> pd.DataFrame:
    return _shared_norman_subgroup(pert_list=pert_list, seed=seed)


def _dummy_embedding_df() -> pd.DataFrame:
    return pd.DataFrame([[0.0]], index=["ctrl"], columns=["dummy"])


def _resolve_eval_data_path(name: str, cfg: ScgptDatasetConfig) -> Path:
    primary = (LOCAL_DATA_ROOT / Path(cfg.data_rel).relative_to("data")).resolve()
    if primary.exists():
        return primary

    paths_cfg_path = ROOT / "configs" / "paths.yaml"
    if paths_cfg_path.exists():
        paths_cfg = _utils.load_yaml(str(paths_cfg_path))
        datasets_map = paths_cfg.get("datasets", {})
        alt = Path(datasets_map[name]) if name in datasets_map else primary
        if alt.exists():
            return alt.resolve()

    raise FileNotFoundError(
        f"Missing dataset file for scGPT dataset={name}. "
        f"Tried {primary} and fallback via {paths_cfg_path}."
    )


def _resolve_pretrained_root() -> Path:
    required = ["args.json", "vocab.json", "best_model.pt"]

    def _is_complete(root: Path) -> bool:
        return all((root / name).exists() for name in required)

    # Prefer the repo-local checkpoint directory so Linux servers do not inherit
    # a Windows absolute path from configs/paths.yaml.
    local_root = (ROOT / "artifacts" / "models" / "scGPT_human").resolve()
    if _is_complete(local_root):
        return local_root

    paths_cfg_path = ROOT / "configs" / "paths.yaml"
    if not paths_cfg_path.exists():
        raise FileNotFoundError(
            f"Missing paths.yaml required for scGPT pretrained weights: {paths_cfg_path}"
        )
    paths_cfg = _utils.load_yaml(str(paths_cfg_path))
    raw = str(paths_cfg.get("scgpt_pretrained_root", "")).strip()
    if not raw:
        raise FileNotFoundError(
            "configs/paths.yaml is missing `scgpt_pretrained_root`. "
            "Set it to a directory containing args.json, vocab.json, and best_model.pt."
        )
    root = Path(raw).resolve()

    # If the configured path is a machine-specific absolute path that does not
    # exist here, fall back to the repo-local checkpoint directory.
    if not root.exists() and _is_complete(local_root):
        return local_root

    missing = [name for name in required if not (root / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"scGPT pretrained directory is incomplete: {root}. Missing files: {missing}"
        )
    return root


def _require_scgpt_stack() -> dict:
    try:
        model_mod = importlib.import_module("scgpt.model")
        loss_mod = importlib.import_module("scgpt.loss")
        tok_mod = importlib.import_module("scgpt.tokenizer.gene_tokenizer")
        util_mod = importlib.import_module("scgpt.utils")
        tg_data_mod = importlib.import_module("torch_geometric.data")
        tg_loader_mod = importlib.import_module("torch_geometric.loader")
    except ImportError as exc:
        raise ImportError(
            "scGPT integration requires `import scgpt` and `torch_geometric` in the current environment."
        ) from exc

    return {
        "TransformerGenerator": getattr(model_mod, "TransformerGenerator"),
        "masked_mse_loss": getattr(loss_mod, "masked_mse_loss"),
        "GeneVocab": getattr(tok_mod, "GeneVocab"),
        "map_raw_id_to_vocab_id": getattr(util_mod, "map_raw_id_to_vocab_id"),
        "Data": getattr(tg_data_mod, "Data"),
        "DataLoader": getattr(tg_loader_mod, "DataLoader"),
    }


def _prepare_eval_adata(data_path: Path) -> tuple[ad.AnnData, TriShiftData]:
    adata = _utils.load_adata(str(data_path))
    data = TriShiftData(adata, _dummy_embedding_df())
    data.build_or_load_degs()
    return data.adata_all, data


def _subset_by_conditions(adata: ad.AnnData, conditions: list[str]) -> ad.AnnData:
    cond_set = set(map(str, conditions)) | {"ctrl"}
    mask = adata.obs["condition"].astype(str).isin(cond_set).values
    return adata[mask].copy()


def _build_split_dict(
    name: str,
    data: TriShiftData,
    split_id: int,
    test_ratio: float,
) -> tuple[dict, pd.DataFrame | None]:
    subgroup_df = None
    if name == "norman":
        subgroup_df = subgroup(
            list(data.adata_all.obs["condition"].astype(str).unique()),
            seed=int(split_id),
        )
        test_conds = list(subgroup_df[subgroup_df.group == "test"].index)
        val_conds = list(subgroup_df[subgroup_df.group == "val"].index)
        split_dict = data.split_by_condition(
            seed=int(split_id),
            test_conds=test_conds,
            val_conds=val_conds,
        )
        return split_dict, subgroup_df
    return data.split_by_condition(seed=int(split_id), test_ratio=float(test_ratio)), None


def _condition_tokens_no_ctrl(condition: str) -> list[str]:
    return [tok for tok in str(condition).split("+") if tok and tok != "ctrl"]


def _make_pert_flags(condition: str, gene_names: np.ndarray) -> np.ndarray:
    perturbed = set(_condition_tokens_no_ctrl(condition))
    return np.asarray([1 if str(g) in perturbed else 0 for g in gene_names], dtype=np.int64)


def _dense_rows(adata: ad.AnnData) -> np.ndarray:
    return _utils.densify_X(adata.X)


def _make_graph(
    *,
    data_cls,
    ctrl_expr: np.ndarray,
    pert_flags: np.ndarray,
    condition: str,
    target_expr: np.ndarray | None = None,
):
    ctrl_vec = np.asarray(ctrl_expr, dtype=np.float32).reshape(1, -1)
    flags_vec = np.asarray(pert_flags, dtype=np.float32).reshape(1, -1)
    x = np.concatenate([ctrl_vec.T, flags_vec.T], axis=1)
    payload = {"x": torch.tensor(x, dtype=torch.float32), "pert": str(condition)}
    if target_expr is not None:
        payload["y"] = torch.tensor(
            np.asarray(target_expr, dtype=np.float32).reshape(1, -1),
            dtype=torch.float32,
        )
    return data_cls(**payload)


def _build_train_eval_graphs(
    *,
    split_adata: ad.AnnData,
    ctrl_adata: ad.AnnData,
    gene_names: np.ndarray,
    data_cls,
    samples_per_cell: int = 1,
) -> list:
    if ctrl_adata.n_obs == 0:
        raise ValueError("Control pool is empty for scGPT graph construction")
    ctrl_rows = _dense_rows(ctrl_adata)
    graphs = []
    for condition in sorted(set(split_adata.obs["condition"].astype(str).tolist())):
        if condition == "ctrl":
            continue
        pert_flags = _make_pert_flags(condition, gene_names)
        target_rows = _dense_rows(split_adata[split_adata.obs["condition"] == condition])
        sample_size = int(max(samples_per_cell, 1))
        for target_row in target_rows:
            ctrl_idx = np.random.randint(0, ctrl_rows.shape[0], size=sample_size)
            for idx in ctrl_idx:
                graphs.append(
                    _make_graph(
                        data_cls=data_cls,
                        ctrl_expr=ctrl_rows[int(idx)],
                        pert_flags=pert_flags,
                        condition=condition,
                        target_expr=target_row,
                    )
                )
    return graphs


def _build_prediction_graphs(
    *,
    ctrl_adata: ad.AnnData,
    gene_names: np.ndarray,
    condition: str,
    data_cls,
    num_samples: int,
) -> tuple[list, np.ndarray]:
    if ctrl_adata.n_obs == 0:
        raise ValueError("Control pool is empty for scGPT prediction")
    ctrl_rows = _dense_rows(ctrl_adata)
    sample_idx = np.random.randint(0, ctrl_rows.shape[0], size=int(max(num_samples, 1)))
    sampled_ctrl = ctrl_rows[sample_idx]
    pert_flags = _make_pert_flags(condition, gene_names)
    graphs = [
        _make_graph(
            data_cls=data_cls,
            ctrl_expr=row,
            pert_flags=pert_flags,
            condition=condition,
        )
        for row in sampled_ctrl
    ]
    return graphs, sampled_ctrl


def _prepare_vocab_and_gene_ids(adata: ad.AnnData, pretrained_root: Path, stack: dict) -> np.ndarray:
    vocab = stack["GeneVocab"].from_file(pretrained_root / "vocab.json")
    for token in SPECIAL_TOKENS:
        if token not in vocab:
            vocab.append_token(token)
    genes = adata.var["gene_name"].astype(str).tolist()
    adata.var["id_in_vocab"] = [1 if gene in vocab else -1 for gene in genes]
    vocab.set_default_index(vocab[PAD_TOKEN])
    return np.asarray([vocab[gene] if gene in vocab else vocab[PAD_TOKEN] for gene in genes], dtype=int)


def _build_model(
    *,
    pretrained_root: Path,
    stack: dict,
    cfg: ScgptDatasetConfig,
):
    with (pretrained_root / "args.json").open("r", encoding="utf-8") as f:
        model_cfg = json.load(f)

    vocab = stack["GeneVocab"].from_file(pretrained_root / "vocab.json")
    for token in SPECIAL_TOKENS:
        if token not in vocab:
            vocab.append_token(token)
    vocab.set_default_index(vocab[PAD_TOKEN])

    model = stack["TransformerGenerator"](
        len(vocab),
        int(model_cfg["embsize"]),
        int(model_cfg["nheads"]),
        int(model_cfg["d_hid"]),
        int(model_cfg["nlayers"]),
        nlayers_cls=int(model_cfg["n_layers_cls"]),
        n_cls=1,
        vocab=vocab,
        dropout=float(cfg.dropout),
        pad_token=PAD_TOKEN,
        pad_value=PAD_VALUE,
        pert_pad_id=PERT_PAD_ID,
        use_fast_transformer=bool(cfg.use_fast_transformer),
    )
    pretrained_dict = torch.load(pretrained_root / "best_model.pt", map_location="cpu")
    model_dict = model.state_dict()
    pretrained_dict = {
        k: v
        for k, v in pretrained_dict.items()
        if any(k.startswith(prefix) for prefix in LOAD_PARAM_PREFIXES)
        and k in model_dict
        and tuple(v.shape) == tuple(model_dict[k].shape)
    }
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model


def _forward_batch(
    *,
    batch_data,
    model,
    gene_ids: np.ndarray,
    map_raw_id_to_vocab_id,
    masked_mse_loss,
    include_zero_gene: str,
    max_seq_len: int,
    amp: bool,
    device: torch.device,
):
    batch_data.to(device)
    batch_size = int(batch_data.y.shape[0])
    x = batch_data.x
    n_genes = int(x.shape[0] / max(batch_size, 1))
    ori_gene_values = x[:, 0].view(batch_size, n_genes)
    pert_flags = x[:, 1].long().view(batch_size, n_genes)

    if include_zero_gene == "all":
        input_gene_ids = torch.arange(n_genes, device=device, dtype=torch.long)
    elif include_zero_gene == "batch-wise":
        input_gene_ids = ori_gene_values.nonzero()[:, 1].flatten().unique().sort()[0]
    else:
        raise ValueError("include_zero_gene must be one of: all, batch-wise")

    if len(input_gene_ids) > int(max_seq_len):
        input_gene_ids = torch.randperm(len(input_gene_ids), device=device)[: int(max_seq_len)]

    input_values = ori_gene_values[:, input_gene_ids]
    input_pert_flags = pert_flags[:, input_gene_ids]
    mapped_input_gene_ids = map_raw_id_to_vocab_id(input_gene_ids, gene_ids)
    mapped_input_gene_ids = mapped_input_gene_ids.repeat(batch_size, 1)
    src_key_padding_mask = torch.zeros_like(input_values, dtype=torch.bool, device=device)

    with torch.cuda.amp.autocast(enabled=bool(amp and device.type == "cuda")):
        output_dict = model(
            mapped_input_gene_ids,
            input_values,
            input_pert_flags,
            src_key_padding_mask=src_key_padding_mask,
            CLS=False,
            CCE=False,
            MVC=False,
            ECS=False,
        )
        output_values = output_dict["mlm_output"]
        target_values = batch_data.y[:, input_gene_ids]
        masked_positions = torch.ones_like(input_values, dtype=torch.bool, device=device)
        return masked_mse_loss(output_values, target_values, masked_positions)


def _collect_loader_predictions(
    *,
    loader,
    model,
    gene_ids: np.ndarray,
    cfg: ScgptDatasetConfig,
    map_raw_id_to_vocab_id,
    device: torch.device,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    conds: list[str] = []
    pred_rows: list[np.ndarray] = []
    truth_rows: list[np.ndarray] = []
    for batch in loader:
        batch.to(device)
        with torch.no_grad():
            pred = model.pred_perturb(
                batch,
                include_zero_gene=cfg.include_zero_gene,
                gene_ids=gene_ids,
                amp=bool(device.type == "cuda"),
            )
        conds.extend([str(x) for x in list(batch.pert)])
        pred_rows.append(np.asarray(pred.detach().cpu().numpy(), dtype=np.float32))
        truth_rows.append(np.asarray(batch.y.detach().cpu().numpy(), dtype=np.float32))
    if not pred_rows:
        return [], np.empty((0, 0), dtype=np.float32), np.empty((0, 0), dtype=np.float32)
    return conds, np.vstack(pred_rows), np.vstack(truth_rows)


def _mean_pearson_by_condition(
    *,
    conds: list[str],
    pred_rows: np.ndarray,
    truth_rows: np.ndarray,
    ctrl_mean: np.ndarray,
    top20_degs_final: dict,
) -> float:
    if not conds:
        return float("-inf")
    rows = []
    cond_arr = np.asarray(conds, dtype=object)
    for condition in sorted(set(cond_arr.tolist())):
        idx = np.where(cond_arr == condition)[0]
        degs = np.asarray(top20_degs_final.get(condition, []), dtype=int)
        if idx.size == 0 or degs.size == 0:
            continue
        pred_vec = pred_rows[idx][:, degs].mean(axis=0)
        truth_vec = truth_rows[idx][:, degs].mean(axis=0)
        ctrl_vec = np.asarray(ctrl_mean, dtype=np.float32)[degs]
        if np.std(pred_vec - ctrl_vec) <= 0 or np.std(truth_vec - ctrl_vec) <= 0:
            continue
        rows.append(float(pearsonr(truth_vec - ctrl_vec, pred_vec - ctrl_vec)[0]))
    if not rows:
        return float("-inf")
    return float(np.nanmean(rows))


def _train_model(
    *,
    model,
    train_loader,
    val_loader,
    gene_ids: np.ndarray,
    cfg: ScgptDatasetConfig,
    device: torch.device,
    top20_degs_final: dict,
    ctrl_mean: np.ndarray,
    stack: dict,
):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.lr))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(device.type == "cuda"))
    best_score = float("-inf")
    best_state = copy.deepcopy(model.state_dict())
    patience = 0
    epoch_iter = _progress(
        range(1, int(cfg.epochs) + 1),
        desc="scGPT epochs",
        total=int(cfg.epochs),
        leave=True,
        dynamic_ncols=True,
    )

    for _epoch in epoch_iter:
        model.train()
        batch_iter = _progress(
            train_loader,
            desc=f"epoch {_epoch}/{int(cfg.epochs)}",
            total=len(train_loader) if hasattr(train_loader, "__len__") else None,
            leave=False,
            dynamic_ncols=True,
        )
        train_losses: list[float] = []
        for batch in batch_iter:
            loss = _forward_batch(
                batch_data=batch,
                model=model,
                gene_ids=gene_ids,
                map_raw_id_to_vocab_id=stack["map_raw_id_to_vocab_id"],
                masked_mse_loss=stack["masked_mse_loss"],
                include_zero_gene=cfg.include_zero_gene,
                max_seq_len=int(cfg.max_seq_len),
                amp=bool(device.type == "cuda"),
                device=device,
            )
            loss_value = float(loss.detach().item())
            train_losses.append(loss_value)
            if tqdm is not None and hasattr(batch_iter, "set_postfix"):
                batch_iter.set_postfix(loss=f"{loss_value:.4f}")
            model.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=False)
            scaler.step(optimizer)
            scaler.update()

        model.eval()
        conds, pred_rows, truth_rows = _collect_loader_predictions(
            loader=val_loader,
            model=model,
            gene_ids=gene_ids,
            cfg=cfg,
            map_raw_id_to_vocab_id=stack["map_raw_id_to_vocab_id"],
            device=device,
        )
        score = _mean_pearson_by_condition(
            conds=conds,
            pred_rows=pred_rows,
            truth_rows=truth_rows,
            ctrl_mean=ctrl_mean,
            top20_degs_final=top20_degs_final,
        )
        if score > best_score:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
            improved = True
        else:
            patience += 1
            improved = False
        mean_train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        if tqdm is not None and hasattr(epoch_iter, "set_postfix"):
            epoch_iter.set_postfix(
                train_loss=f"{mean_train_loss:.4f}",
                val=f"{score:.4f}",
                best=f"{best_score:.4f}",
                patience=f"{patience}/{int(cfg.early_stop)}",
            )
        print(
            "[scgpt] "
            f"epoch={_epoch}/{int(cfg.epochs)} "
            f"train_loss={mean_train_loss:.4f} "
            f"val_score={score:.4f} "
            f"best_score={best_score:.4f} "
            f"patience={patience}/{int(cfg.early_stop)} "
            f"improved={str(improved).lower()}",
            flush=True,
        )
        if tqdm is not None and hasattr(batch_iter, "close"):
            batch_iter.close()
        if patience >= int(cfg.early_stop):
            print(
                f"[scgpt] early stopping at epoch={_epoch} with best_score={best_score:.4f}",
                flush=True,
            )
            break
        scheduler.step()

    if tqdm is not None and hasattr(epoch_iter, "close"):
        epoch_iter.close()
    model.load_state_dict(best_state)
    model.eval()
    return model


def _predict_condition(
    *,
    model,
    ctrl_adata: ad.AnnData,
    gene_names: np.ndarray,
    condition: str,
    data_cls,
    dataloader_cls,
    gene_ids: np.ndarray,
    cfg: ScgptDatasetConfig,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    graphs, sampled_ctrl = _build_prediction_graphs(
        ctrl_adata=ctrl_adata,
        gene_names=gene_names,
        condition=condition,
        data_cls=data_cls,
        num_samples=int(cfg.control_pool_size),
    )
    loader = dataloader_cls(graphs, batch_size=int(cfg.eval_batch_size), shuffle=False)
    pred_rows: list[np.ndarray] = []
    for batch in loader:
        batch.to(device)
        with torch.no_grad():
            pred = model.pred_perturb(
                batch,
                include_zero_gene=cfg.include_zero_gene,
                gene_ids=gene_ids,
                amp=bool(device.type == "cuda"),
            )
        pred_rows.append(np.asarray(pred.detach().cpu().numpy(), dtype=np.float32))
    if not pred_rows:
        return np.empty((0, gene_names.shape[0]), dtype=np.float32), sampled_ctrl
    return np.vstack(pred_rows), sampled_ctrl


def _compute_metrics_and_export_payload(
    *,
    model,
    eval_adata: ad.AnnData,
    reference_adata: ad.AnnData,
    split_dict: dict,
    split_id: int,
    gene_ids: np.ndarray,
    cfg: ScgptDatasetConfig,
    stack: dict,
) -> tuple[pd.DataFrame, dict]:
    results = []
    export_payload = {}
    gene_names = (
        eval_adata.var["gene_name"].astype(str).values
        if "gene_name" in eval_adata.var.columns
        else eval_adata.var_names.astype(str).values
    )
    top20_degs_final = eval_adata.uns.get("top20_degs_final", {})
    if not isinstance(top20_degs_final, dict):
        raise TypeError("Expected eval_adata.uns['top20_degs_final'] to be a mapping")

    pert_reference = average_of_perturbation_centroids(
        X=_utils.densify_X(reference_adata.X),
        conditions=reference_adata.obs["condition"].astype(str).values,
        ctrl_label="ctrl",
    )
    ctrl_adata = eval_adata[eval_adata.obs["condition"].astype(str) == "ctrl"]
    device = next(model.parameters()).device

    for condition in sorted(set(map(str, split_dict.get("test_conds", [])))):
        cond_mask = eval_adata.obs["condition"].astype(str) == condition
        if not bool(cond_mask.any()):
            print(f"[scgpt] skip condition missing in eval adata: {condition}")
            continue
        true = _utils.densify_X(eval_adata[cond_mask].X)
        degs = np.asarray(top20_degs_final.get(condition, []), dtype=int).reshape(-1)
        if degs.size == 0:
            print(f"[scgpt] skip condition without DEGs: {condition}")
            continue
        pred, ctrl_sampled = _predict_condition(
            model=model,
            ctrl_adata=ctrl_adata,
            gene_names=gene_names,
            condition=condition,
            data_cls=stack["Data"],
            dataloader_cls=stack["DataLoader"],
            gene_ids=gene_ids,
            cfg=cfg,
            device=device,
        )
        if pred.size == 0:
            print(f"[scgpt] skip condition without predictions: {condition}")
            continue
        pred_vec = pred[:, degs].mean(axis=0)
        ctrl_vec = ctrl_sampled[:, degs].mean(axis=0)
        true_vec = true[:, degs].mean(axis=0)

        mse_ctrl_val = float(mse(true_vec, ctrl_vec))
        mse_pred_val = float(mse(true_vec, pred_vec))
        nmse_val = float(mse_pred_val / mse_ctrl_val) if mse_ctrl_val > 0 else np.nan
        pearson_val = float(pearsonr(true_vec - ctrl_vec, pred_vec - ctrl_vec)[0])
        deg_mean_r2_val = regression_r2_safe(true_vec - ctrl_vec, pred_vec - ctrl_vec)
        systema_metrics = pearson_delta_reference_metrics(
            X_true=true.mean(axis=0),
            X_pred=pred.mean(axis=0),
            reference=pert_reference,
            top20_de_idxs=degs,
        )
        scpram_metrics = compute_scpram_metrics_from_arrays(
            X_true=true,
            X_pred=pred,
            deg_idx=degs,
            n_degs=100,
            sample_ratio=0.8,
            times=100,
        )
        results.append(
            {
                "condition": condition,
                "mse_pred": mse_pred_val,
                "mse_ctrl": mse_ctrl_val,
                "nmse": nmse_val,
                "pearson": pearson_val,
                "deg_mean_r2": float(deg_mean_r2_val),
                "systema_corr_20de_allpert": float(systema_metrics["corr_20de_allpert"]),
                "systema_corr_deg_r2": float(systema_metrics["corr_deg_r2"]),
                **scpram_metrics,
                "split_id": int(split_id),
                "n_ensemble": int(pred.shape[0]),
            }
        )
        export_payload[condition] = {
            "Pred": pred[:, degs] if degs.size > 0 else pred[:, :0],
            "Ctrl": ctrl_sampled[:, degs] if degs.size > 0 else ctrl_sampled[:, :0],
            "Truth": true[:, degs] if degs.size > 0 else true[:, :0],
            "Pred_full": pred,
            "Ctrl_full": ctrl_sampled,
            "Truth_full": true,
            "DE_idx": degs,
            "DE_name": gene_names[degs] if degs.size > 0 else np.array([], dtype=gene_names.dtype),
            "gene_name_full": gene_names,
        }

    return pd.DataFrame(results), export_payload


def run_scgpt_eval(
    name: str,
    base_seed: int = 24,
    export_notebook_pkl: bool = True,
    control_pool_size: int | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    eval_batch_size: int | None = None,
    lr: float | None = None,
    early_stop: int | None = None,
) -> None:
    if name not in DATASET_CONFIG:
        raise ValueError(f"Unknown dataset: {name}")
    stack = _require_scgpt_stack()
    base_cfg = DATASET_CONFIG[name]
    cfg = ScgptDatasetConfig(
        data_rel=base_cfg.data_rel,
        splits=list(base_cfg.splits),
        test_ratio=float(base_cfg.test_ratio),
        norman_split=bool(base_cfg.norman_split),
        lr=float(base_cfg.lr if lr is None else lr),
        batch_size=int(base_cfg.batch_size if batch_size is None else batch_size),
        eval_batch_size=int(base_cfg.eval_batch_size if eval_batch_size is None else eval_batch_size),
        epochs=int(base_cfg.epochs if epochs is None else epochs),
        early_stop=int(base_cfg.early_stop if early_stop is None else early_stop),
        control_pool_size=int(base_cfg.control_pool_size if control_pool_size is None else control_pool_size),
        include_zero_gene=str(base_cfg.include_zero_gene),
        max_seq_len=int(base_cfg.max_seq_len),
        dropout=float(base_cfg.dropout),
        use_fast_transformer=bool(base_cfg.use_fast_transformer),
    )

    data_path = _resolve_eval_data_path(name, cfg)
    pretrained_root = _resolve_pretrained_root()
    eval_adata, data = _prepare_eval_adata(data_path)
    gene_ids = _prepare_vocab_and_gene_ids(eval_adata, pretrained_root, stack)
    gene_names = (
        eval_adata.var["gene_name"].astype(str).values
        if "gene_name" in eval_adata.var.columns
        else eval_adata.var_names.astype(str).values
    )

    out_dir = ROOT / "artifacts" / "results" / "scgpt" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    metrics_all = []
    for split in cfg.splits:
        print(f"[scgpt] dataset={name} split={split}")
        set_seeds(base_seed + int(split))
        split_dict, subgroup_df = _build_split_dict(name, data, int(split), float(cfg.test_ratio))
        train_graphs = _build_train_eval_graphs(
            split_adata=split_dict["train"],
            ctrl_adata=split_dict["train"][split_dict["train"].obs["condition"].astype(str) == "ctrl"],
            gene_names=gene_names,
            data_cls=stack["Data"],
        )
        val_graphs = _build_train_eval_graphs(
            split_adata=split_dict["val"],
            ctrl_adata=split_dict["val"][split_dict["val"].obs["condition"].astype(str) == "ctrl"],
            gene_names=gene_names,
            data_cls=stack["Data"],
        )
        if not train_graphs:
            raise ValueError(f"scGPT train split is empty for dataset={name} split={split}")
        if not val_graphs:
            raise ValueError(f"scGPT val split is empty for dataset={name} split={split}")

        train_loader = stack["DataLoader"](train_graphs, batch_size=int(cfg.batch_size), shuffle=True, drop_last=True)
        val_loader = stack["DataLoader"](val_graphs, batch_size=int(cfg.eval_batch_size), shuffle=False)

        model = _build_model(
            pretrained_root=pretrained_root,
            stack=stack,
            cfg=cfg,
        )
        ctrl_mean = _utils.densify_X(
            split_dict["val"][split_dict["val"].obs["condition"].astype(str) == "ctrl"].X
        ).mean(axis=0)
        model = _train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            gene_ids=gene_ids,
            cfg=cfg,
            device=device,
            top20_degs_final=eval_adata.uns.get("top20_degs_final", {}),
            ctrl_mean=np.asarray(ctrl_mean, dtype=np.float32),
            stack=stack,
        )

        train_val_conds = list(split_dict["train_conds"]) + list(split_dict["val_conds"])
        reference_adata = _subset_by_conditions(eval_adata, train_val_conds)
        metrics_df, export_payload = _compute_metrics_and_export_payload(
            model=model,
            eval_adata=eval_adata,
            reference_adata=reference_adata,
            split_dict=split_dict,
            split_id=int(split),
            gene_ids=gene_ids,
            cfg=cfg,
            stack=stack,
        )
        metrics_df = _attach_subgroup_column(metrics_df, subgroup_df)
        metrics_all.append(metrics_df)

        if export_notebook_pkl:
            out_pkl = out_dir / f"scgpt_{name}_{split}.pkl"
            with out_pkl.open("wb") as f:
                pickle.dump(export_payload, f)
            print(f"[scgpt] saved notebook payload: {out_pkl}")

    metrics_df_all = pd.concat(metrics_all, ignore_index=True)
    metrics_df_all.to_csv(out_dir / "metrics.csv", index=False)
    _write_mean_metrics(out_dir / "mean_pearson.txt", metrics_df_all)
    print(f"[scgpt] saved metrics: {out_dir / 'metrics.csv'}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run scGPT eval with TriShift metrics")
    parser.add_argument("--profile", default="", help="dataset profile under scripts/scgpt/eval/configs")
    parser.add_argument("--name", default="", help="dataset name (legacy mode)")
    parser.add_argument("--seed", type=int, default=24, help="base seed")
    parser.add_argument("--control_pool_size", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--early_stop", type=int, default=10)
    parser.add_argument(
        "--no_export_notebook_pkl",
        action="store_true",
        help="disable notebook-compatible pickle export",
    )
    args = parser.parse_args(argv)

    profile = str(args.profile).strip()
    if profile:
        prof = _load_profile(profile)
        task_args = prof["task_args"]
        run_scgpt_eval(
            prof["dataset"],
            base_seed=int(task_args.get("seed", 24)) if int(args.seed) == 24 else int(args.seed),
            export_notebook_pkl=(
                False
                if bool(args.no_export_notebook_pkl)
                else bool(task_args.get("export_notebook_pkl", True))
            ),
            control_pool_size=(
                int(task_args.get("control_pool_size", 300))
                if int(args.control_pool_size) == 300
                else int(args.control_pool_size)
            ),
            epochs=int(task_args.get("epochs", 15)) if int(args.epochs) == 15 else int(args.epochs),
            batch_size=(
                int(task_args.get("batch_size", 64))
                if int(args.batch_size) == 64
                else int(args.batch_size)
            ),
            eval_batch_size=(
                int(task_args.get("eval_batch_size", 64))
                if int(args.eval_batch_size) == 64
                else int(args.eval_batch_size)
            ),
            lr=float(task_args.get("lr", 1e-4)) if float(args.lr) == 1e-4 else float(args.lr),
            early_stop=(
                int(task_args.get("early_stop", 10))
                if int(args.early_stop) == 10
                else int(args.early_stop)
            ),
        )
        return

    if not str(args.name).strip():
        raise ValueError("Either --profile or --name must be provided")
    run_scgpt_eval(
        args.name,
        base_seed=int(args.seed),
        export_notebook_pkl=not bool(args.no_export_notebook_pkl),
        control_pool_size=int(args.control_pool_size),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        eval_batch_size=int(args.eval_batch_size),
        lr=float(args.lr),
        early_stop=int(args.early_stop),
    )


if __name__ == "__main__":
    main()
