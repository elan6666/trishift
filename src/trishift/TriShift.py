import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn.functional as F
import scipy.sparse as sp
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error as mse
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm

from trishift._model import TriShiftNet, aggregate_cond_embedding
from trishift import _utils
from trishift._external_metrics import (
    average_of_perturbation_centroids,
    compute_scpram_metrics_from_arrays,
    pearson_delta_reference_metrics,
)


def gears_loss(
    pred_expr: torch.Tensor,
    true_expr: torch.Tensor,
    ctrl_expr: torch.Tensor,
    group: list[str],
    nonzero_idx_dict: dict,
    gamma: float = 1.0,
    lambda_dir: float = 1.0,
    deg_idx_dict: dict | None = None,
    deg_weight: float = 1.0,
) -> torch.Tensor:
    """Compute autofocus + direction-aware loss with per-condition gene selection."""
    if not nonzero_idx_dict:
        raise ValueError("nonzero_idx_dict is empty")
    if deg_weight <= 0:
        raise ValueError("deg_weight must be positive")
    autofocus = (true_expr - pred_expr).abs() ** (2.0 + gamma)
    direction = (torch.sign(true_expr - ctrl_expr) - torch.sign(pred_expr - ctrl_expr)) ** 2
    total = autofocus + lambda_dir * direction

    group_arr = np.asarray(group)
    unique_conds, inv = np.unique(group_arr, return_inverse=True)
    loss_scalar = torch.tensor(0.0, device=pred_expr.device)
    for i, cond in enumerate(unique_conds):
        cell_idx = np.where(inv == i)[0]
        if cond not in nonzero_idx_dict:
            raise KeyError(f"missing nonzero indices for condition: {cond}")
        retain = nonzero_idx_dict[cond]
        if len(retain) == 0:
            raise ValueError(f"nonzero indices empty for condition: {cond}")
        retain_arr = np.asarray(retain, dtype=int)
        retain_idx = torch.as_tensor(retain_arr, device=pred_expr.device, dtype=torch.long)
        total_slice = total[cell_idx][:, retain_idx]
        if deg_idx_dict and deg_weight != 1.0 and cond in deg_idx_dict:
            deg_idx = np.asarray(deg_idx_dict[cond], dtype=int)
            if deg_idx.size > 0:
                mask = np.isin(retain_arr, deg_idx)
                if mask.any():
                    weights = torch.ones(
                        len(retain_arr), device=pred_expr.device, dtype=total_slice.dtype
                    )
                    weights[torch.as_tensor(mask, device=pred_expr.device)] = deg_weight
                    total_slice = total_slice * weights
        loss_scalar += total_slice.mean()
    return loss_scalar / max(len(unique_conds), 1)


class _EarlyStopper:
    def __init__(self, patience: int, min_delta: float):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best = np.inf
        self.epochs_no_improve = 0

    def update(self, value: float) -> tuple[bool, bool]:
        if self.patience <= 0:
            return False, False
        improvement = self.best - value
        if improvement > self.min_delta:
            self.best = value
            self.epochs_no_improve = 0
            return True, False
        self.epochs_no_improve += 1
        return False, self.epochs_no_improve >= self.patience


class _TopKTrainDataset(Dataset):
    def __init__(
        self,
        pert_adata,
        ctrl_adata,
        topk,
        topk_weights,
        key_name,
        label_key,
        z_ctrl_mu,
        z_pert_mu,
        soft_ot_weighted_delta: bool = False,
        sample_soft_ctrl: bool = True,
        topk_strategy: str = "random",
    ):
        """Dataset that pairs perturbed cells with top-k matched controls and latent mu."""
        self.pert_adata = pert_adata
        self.ctrl_adata = ctrl_adata
        self.topk = topk
        self.topk_weights = topk_weights
        self.key_name = key_name
        self.label_key = label_key
        self.ctrl_X = _utils.densify_X(self.ctrl_adata.X).astype(np.float32, copy=False)
        self.pert_X = _utils.densify_X(self.pert_adata.X).astype(np.float32, copy=False)
        self.z_ctrl_mu = z_ctrl_mu
        self.z_pert_mu = z_pert_mu
        self.soft_ot_weighted_delta = bool(soft_ot_weighted_delta)
        self.sample_soft_ctrl = bool(sample_soft_ctrl)
        self.topk_strategy = str(topk_strategy)
        if self.topk_strategy not in {"random", "weighted", "expand", "weighted_sample"}:
            raise ValueError(
                "topk_strategy must be one of: random, weighted, expand, weighted_sample"
            )
        self.k = int(self.topk.shape[1])

    def __len__(self) -> int:
        if self.topk_strategy == "expand":
            return int(self.pert_adata.n_obs * self.k)
        return self.pert_adata.n_obs

    def __getitem__(self, i: int) -> tuple[np.ndarray, np.ndarray, list, str, np.ndarray, np.ndarray, np.ndarray]:
        if self.topk_strategy == "expand":
            pert_i = i // self.k
            neigh_j = i % self.k
        else:
            pert_i = i
            neigh_j = None

        true_expr = self.pert_X[pert_i]
        z_pert_mu = self.z_pert_mu[pert_i]
        ctrl_rows = self.topk[pert_i]

        if self.topk_weights is None:
            weights = np.full(self.k, 1.0 / max(self.k, 1), dtype=np.float32)
        else:
            weights = self.topk_weights[pert_i].astype(np.float32, copy=False)
            weight_sum = weights.sum()
            if weight_sum <= 0:
                weights = np.full_like(weights, 1.0 / max(len(weights), 1), dtype=np.float32)
            else:
                weights = weights / weight_sum

        if self.topk_strategy == "expand":
            ctrl_row = ctrl_rows[neigh_j]
            ctrl_expr = self.ctrl_X[ctrl_row]
            z_ctrl_mu = self.z_ctrl_mu[ctrl_row]
            delta_target = z_pert_mu - z_ctrl_mu
        elif self.topk_strategy == "weighted":
            ctrl_expr = (self.ctrl_X[ctrl_rows] * weights[:, None]).sum(axis=0)
            z_ctrl_mu = (self.z_ctrl_mu[ctrl_rows] * weights[:, None]).sum(axis=0)
            delta_target = z_pert_mu - z_ctrl_mu
        elif self.topk_strategy == "weighted_sample":
            if self.sample_soft_ctrl:
                pick = np.random.choice(len(ctrl_rows), p=weights)
            else:
                pick = int(np.argmax(weights))
            ctrl_row = ctrl_rows[pick]
            ctrl_expr = self.ctrl_X[ctrl_row]
            z_ctrl_mu = self.z_ctrl_mu[ctrl_row]
            if self.soft_ot_weighted_delta:
                delta_target = ((z_pert_mu - self.z_ctrl_mu[ctrl_rows]) * weights[:, None]).sum(
                    axis=0
                )
            else:
                delta_target = z_pert_mu - z_ctrl_mu
        elif self.soft_ot_weighted_delta:
            if self.sample_soft_ctrl:
                pick = np.random.choice(len(ctrl_rows), p=weights)
            else:
                pick = int(np.argmax(weights))
            ctrl_row = ctrl_rows[pick]
            ctrl_expr = self.ctrl_X[ctrl_row]
            z_ctrl_mu = self.z_ctrl_mu[ctrl_row]
            delta_target = ((z_pert_mu - self.z_ctrl_mu[ctrl_rows]) * weights[:, None]).sum(axis=0)
        else:
            pick = np.random.randint(0, self.k)
            ctrl_row = ctrl_rows[pick]
            ctrl_expr = self.ctrl_X[ctrl_row]
            z_ctrl_mu = self.z_ctrl_mu[ctrl_row]
            delta_target = z_pert_mu - z_ctrl_mu

        idx_list = self.pert_adata.obs[self.key_name].iloc[pert_i]
        cond_str = self.pert_adata.obs[self.label_key].iloc[pert_i]
        return ctrl_expr, true_expr, idx_list, cond_str, z_ctrl_mu, z_pert_mu, delta_target


class _Stage3PairDataset(Dataset):
    def __init__(self, ctrl_adata, pert_adata, key_name, label_key, shuffle_pairs=True):
        """Dataset that pairs perturbed cells with random control cells (stage3-only)."""
        self.ctrl_X = _utils.densify_X(ctrl_adata.X).astype(np.float32, copy=False)
        self.pert_X = _utils.densify_X(pert_adata.X).astype(np.float32, copy=False)
        self.idx_list = pert_adata.obs[key_name].values
        self.cond_str = pert_adata.obs[label_key].astype(str).values
        self.n_pert = self.pert_X.shape[0]
        self.n_ctrl = self.ctrl_X.shape[0]
        self.pert_idx = np.arange(self.n_pert)
        self.ctrl_idx = np.resize(np.arange(self.n_ctrl), self.n_pert)
        if shuffle_pairs:
            np.random.shuffle(self.pert_idx)
            np.random.shuffle(self.ctrl_idx)

    def __len__(self) -> int:
        return self.n_pert

    def __getitem__(self, i: int) -> tuple[np.ndarray, np.ndarray, list, str]:
        pi = self.pert_idx[i]
        ci = self.ctrl_idx[i]
        ctrl_expr = self.ctrl_X[ci]
        true_expr = self.pert_X[pi]
        idx_list = self.idx_list[pi]
        cond_str = self.cond_str[pi]
        return ctrl_expr, true_expr, idx_list, cond_str


def _collate_with_latent(
    batch,
) -> tuple[torch.Tensor, torch.Tensor, list, list, torch.Tensor, torch.Tensor, torch.Tensor]:
    ctrl_expr, true_expr, idx_list, cond_str, z_ctrl_mu, z_pert_mu, delta_target = zip(*batch)
    ctrl_t = torch.tensor(np.stack(ctrl_expr), dtype=torch.float32)
    true_t = torch.tensor(np.stack(true_expr), dtype=torch.float32)
    z_ctrl_mu_t = torch.tensor(np.stack(z_ctrl_mu), dtype=torch.float32)
    z_pert_mu_t = torch.tensor(np.stack(z_pert_mu), dtype=torch.float32)
    delta_target_t = torch.tensor(np.stack(delta_target), dtype=torch.float32)
    return ctrl_t, true_t, list(idx_list), list(cond_str), z_ctrl_mu_t, z_pert_mu_t, delta_target_t


def _collate_no_latent(batch) -> tuple[torch.Tensor, torch.Tensor, list, list]:
    ctrl_expr, true_expr, idx_list, cond_str = zip(*batch)
    ctrl_t = torch.tensor(np.stack(ctrl_expr), dtype=torch.float32)
    true_t = torch.tensor(np.stack(true_expr), dtype=torch.float32)
    return ctrl_t, true_t, list(idx_list), list(cond_str)


class TriShift:
    def __init__(self, data, device: str = "cuda"):
        """Main API class for training and evaluation."""
        self.data = data
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)
        self.net = None
        self.hparams = {}

    def set_base_seed(self, base_seed: int):
        """Set base seed used by training/evaluation."""
        self.hparams["base_seed"] = int(base_seed)

    def _get_nonzero_idx_dict(self) -> dict:
        """Return non-zero/non-dropout gene indices per condition from adata.uns."""
        uns = self.data.adata_all.uns
        for key in (
            "gene_idx_non_zeros",
            "non_zeros_gene_idx",
            "gene_idx_non_dropout",
            "non_dropout_gene_idx",
        ):
            if key in uns:
                return uns[key]
        raise ValueError("nonzero index dict not found in adata.uns")

    def _get_degs_idx_dict(self) -> dict:
        """Return DE gene indices per condition from adata.uns if available."""
        return self.data.adata_all.uns.get("top20_degs_final", {})

    @staticmethod
    def _select_dense(X, rows) -> np.ndarray:
        """Return a dense float32 slice of X for the requested rows."""
        if sp.issparse(X):
            return X[rows].toarray().astype(np.float32)
        return np.asarray(X[rows], dtype=np.float32)

    def _get_cond_embd_idx(self, cond_mask) -> list[int] | np.ndarray:
        """Get embedding index list for the first cell of a condition."""
        idx_list = self.data.adata_all.obs[self.data.label_key].loc[cond_mask].index
        idx_first = idx_list[0]
        return self.data.adata_all.obs["embd_index"].loc[idx_first]

    def _sample_ctrl_expr(
        self,
        X_ctrl,
        n_ensemble: int,
        rng: np.random.RandomState,
    ) -> np.ndarray:
        """Sample control expressions from one global RNG stream (Scouter-compatible)."""
        n_ctrl = X_ctrl.shape[0]
        ctrl_idx = rng.choice(n_ctrl, size=n_ensemble, replace=True)
        return self._select_dense(X_ctrl, ctrl_idx)

    def _get_cond_pool_cfg(self) -> tuple[str, bool]:
        if self.hparams.get("stage3_only", False):
            # Strict Scouter-style condition construction for stage3_only.
            return "sum", False
        return (
            str(self.hparams.get("cond_pool_mode", "sum")),
            bool(self.hparams.get("cond_l2_norm", False)),
        )

    def _get_gene_names(self) -> np.ndarray:
        if self.data.var_gene_key in self.data.adata_all.var.columns:
            return self.data.adata_all.var[self.data.var_gene_key].astype(str).values
        return self.data.adata_all.var_names.astype(str).values

    def _build_cond_vec(
        self,
        emb_table: torch.Tensor,
        embd_idx_list,
        n_ensemble: int,
    ) -> torch.Tensor:
        """Build a condition embedding matrix repeated for an ensemble."""
        cond_mode, cond_norm = self._get_cond_pool_cfg()
        cond_vec = aggregate_cond_embedding(
            emb_table,
            embd_idx_list,
            mode=cond_mode,
            normalize=cond_norm,
        )
        return cond_vec.unsqueeze(0).repeat(n_ensemble, 1)

    def _build_cond_vec_batch(self, emb_table: torch.Tensor, idx_list) -> torch.Tensor:
        """Build a batch of condition embeddings from index lists."""
        cond_mode, cond_norm = self._get_cond_pool_cfg()
        # Scouter-style vectorized path: gather all pert indices in one tensor and reduce once.
        # Falls back to per-sample aggregation if unexpected index format is encountered.
        try:
            seqs: list[torch.Tensor] = []
            max_len = 0
            for idxs in idx_list:
                if torch.is_tensor(idxs):
                    t = idxs.to(device=emb_table.device, dtype=torch.long).view(-1)
                elif isinstance(idxs, np.ndarray):
                    t = torch.as_tensor(idxs, device=emb_table.device, dtype=torch.long).view(-1)
                elif isinstance(idxs, (list, tuple, pd.Series)):
                    t = torch.tensor(list(idxs), device=emb_table.device, dtype=torch.long).view(-1)
                elif idxs is None:
                    t = torch.empty(0, device=emb_table.device, dtype=torch.long)
                else:
                    t = torch.tensor([int(idxs)], device=emb_table.device, dtype=torch.long)
                seqs.append(t)
                if t.numel() > max_len:
                    max_len = int(t.numel())

            batch_size = len(seqs)
            if batch_size == 0:
                return emb_table.new_zeros((0, emb_table.shape[1]))
            if max_len == 0:
                out = emb_table.new_zeros((batch_size, emb_table.shape[1]))
                if cond_norm:
                    out = F.normalize(out, p=2, dim=1, eps=1e-12)
                return out

            padded = torch.zeros((batch_size, max_len), device=emb_table.device, dtype=torch.long)
            mask = torch.zeros(
                (batch_size, max_len), device=emb_table.device, dtype=emb_table.dtype
            )
            for i, t in enumerate(seqs):
                n = int(t.numel())
                if n == 0:
                    continue
                padded[i, :n] = t
                mask[i, :n] = 1.0

            selected = emb_table.index_select(0, padded.reshape(-1)).view(batch_size, max_len, -1)
            out = (selected * mask.unsqueeze(-1)).sum(dim=1)
            if cond_mode == "mean":
                denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
                out = out / denom
            elif cond_mode != "sum":
                raise ValueError("mode must be one of: sum, mean")

            if cond_norm:
                out = F.normalize(out, p=2, dim=1, eps=1e-12)
            return out
        except Exception:
            return torch.stack(
                [
                    aggregate_cond_embedding(
                        emb_table,
                        idxs,
                        mode=cond_mode,
                        normalize=cond_norm,
                    )
                    for idxs in idx_list
                ],
                dim=0,
            )

    def _predict_expr_from_ctrl(
        self,
        ctrl_expr: np.ndarray,
        cond_vec: torch.Tensor,
    ) -> np.ndarray:
        """Predict expression given control expression and condition vectors."""
        x_ctrl_t = torch.tensor(ctrl_expr, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            if self.hparams.get("stage3_only", False):
                x_pred = self.net.gen.forward_no_delta(x_ctrl_t, cond_vec).cpu().numpy()
            else:
                out = self.net.forward_joint(x_ctrl_t, cond_vec)
                x_pred = out["x_pred"].cpu().numpy()
        return x_pred

    @staticmethod
    def _need_topk_weights(mode: str, topk_strategy: str) -> bool:
        return mode == "soft_ot" or (
            mode == "ot" and topk_strategy in {"weighted", "weighted_sample"}
        )

    def _derive_disable_loss_z_supervision(
        self,
        requested_disable: bool,
        when_disabled_message: str,
    ) -> bool:
        use_shift_delta = bool(getattr(self.net.shift, "predict_delta", True))
        derived_disable = not use_shift_delta
        if requested_disable != derived_disable:
            print(
                "[stage23] disable_loss_z_supervision is now derived from "
                f"shift_predict_delta={use_shift_delta}; overriding to {derived_disable}"
            )
        if derived_disable:
            print(when_disabled_message)
        return derived_disable

    def _build_topk_map_for_split(
        self,
        split_adata,
        ctrl_global_idx: np.ndarray,
        mode: str,
        k: int,
        seed: int,
        candidates: int,
        cache_path: str | None,
        per_condition_ot: bool,
        need_topk_weights: bool,
        reuse_ot_cache: bool = False,
        cache_key: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        if need_topk_weights:
            topk_map, topk_weights = self.data.build_or_load_topk_map(
                split_adata=split_adata,
                mode=mode,
                k=k,
                seed=seed,
                candidates=candidates,
                cache_path=cache_path,
                return_weights=True,
                per_condition_ot=per_condition_ot,
                reuse_ot_cache=reuse_ot_cache,
                cache_key=cache_key,
                ctrl_global_indices=ctrl_global_idx,
            )
            return topk_map, topk_weights
        topk_map = self.data.build_or_load_topk_map(
            split_adata=split_adata,
            mode=mode,
            k=k,
            seed=seed,
            candidates=candidates,
            cache_path=cache_path,
            per_condition_ot=per_condition_ot,
            reuse_ot_cache=reuse_ot_cache,
            cache_key=cache_key,
            ctrl_global_indices=ctrl_global_idx,
        )
        return topk_map, None

    def _get_val_pert_adata(self, split_dict: dict):
        val_split = split_dict.get("val")
        if val_split is None:
            return None
        val_mask = val_split.obs[self.data.label_key].astype(str) != self.data.ctrl_label
        if np.any(val_mask):
            return val_split[val_mask]
        return None

    def _get_ctrl_pool_from_split(self, split_adata):
        """Return ctrl AnnData and global row indices from the requested split."""
        if split_adata is None:
            return self.data.adata_ctrl, np.asarray(self.data.ctrl_indices, dtype=int)
        cond = split_adata.obs[self.data.label_key].astype(str).values
        ctrl_mask = cond == self.data.ctrl_label
        if not np.any(ctrl_mask):
            print("[ctrl] warning: split has no ctrl; falling back to global ctrl pool")
            return self.data.adata_ctrl, np.asarray(self.data.ctrl_indices, dtype=int)
        ctrl_adata = split_adata[ctrl_mask]
        all_idx_map = {n: i for i, n in enumerate(self.data.adata_all.obs_names)}
        ctrl_global_idx = np.array([all_idx_map[n] for n in ctrl_adata.obs_names], dtype=int)
        return ctrl_adata, ctrl_global_idx

    def _require_cached_z_mu(self) -> np.ndarray:
        """Return cached z_mu from adata_all.obsm or raise a clear error."""
        z_mu = self.data.adata_all.obsm.get("z_mu")
        if z_mu is None:
            raise ValueError(
                "z_mu cache missing for stage23 training. "
                "Run stage1 training and encode_and_cache_mu() before stage23."
            )
        return np.asarray(z_mu)

    @staticmethod
    def _build_topk_loader(
        dataset,
        batch_size: int,
        shuffle: bool,
        num_workers: int,
        pin_memory: bool,
    ) -> DataLoader | None:
        if dataset is None:
            return None
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=_collate_with_latent,
        )

    @staticmethod
    def _compute_expression_loss(
        x_pred: torch.Tensor,
        x_true: torch.Tensor,
        x_ctrl: torch.Tensor,
        cond_str: list[str],
        nonzero_idx_dict: dict,
        gamma: float,
        lambda_dir_expr: float,
        deg_idx_dict: dict | None,
        deg_weight: float,
        lambda_expr_mse: float,
    ) -> torch.Tensor:
        loss_expr = gears_loss(
            x_pred,
            x_true,
            x_ctrl,
            cond_str,
            nonzero_idx_dict,
            gamma=gamma,
            lambda_dir=lambda_dir_expr,
            deg_idx_dict=deg_idx_dict,
            deg_weight=deg_weight,
        )
        if lambda_expr_mse > 0:
            loss_expr = loss_expr + lambda_expr_mse * F.mse_loss(x_pred, x_true)
        return loss_expr

    @staticmethod
    def _compute_latent_supervision_loss(
        shift_pred: torch.Tensor,
        delta_target: torch.Tensor,
        cond_str: list[str],
        latent_loss_type: str,
        latent_idx_dict: dict | None,
        gamma: float,
        lambda_dir_z: float,
    ) -> torch.Tensor:
        if latent_loss_type == "mse":
            return F.mse_loss(shift_pred, delta_target)
        if latent_loss_type == "smooth_l1":
            return F.smooth_l1_loss(shift_pred, delta_target)
        if latent_loss_type == "gears":
            if latent_idx_dict is None:
                raise RuntimeError("latent_idx_dict is required for latent_loss_type=gears")
            zero_ctrl = torch.zeros_like(shift_pred)
            return gears_loss(
                shift_pred,
                delta_target,
                zero_ctrl,
                cond_str,
                latent_idx_dict,
                gamma=gamma,
                lambda_dir=lambda_dir_z,
            )
        raise ValueError("latent_loss_type must be one of: gears, mse, smooth_l1")

    def model_init(
        self,
        x_dim: int,
        z_dim: int,
        cond_dim: int,
        vae_enc_hidden: list[int],
        vae_dec_hidden: list[int],
        shift_hidden: list[int],
        gen_hidden: list[int],
        vae_hidden_dim: int = 1000,
        vae_noise_rate: float = 0.1,
        vae_kl_weight: float = 5e-4,
        dropout: float = 0.0,
        gen_encoder_hidden: list[int] | None = None,
        gen_state_dim: int | str | None = None,
        gen_decoder_hidden: list[int] | None = None,
        gen_input_mode: str = "full",
        gen_use_batchnorm: bool = True,
        gen_use_layernorm: bool = False,
        shift_predict_delta: bool = True,
        shift_use_cross_attention: bool = False,
        shift_cross_attn_heads: int = 4,
        shift_cross_attn_dropout: float = 0.0,
        shift_use_transformer_block: bool = False,
        shift_transformer_layers: int = 1,
        shift_transformer_ff_mult: int = 4,
        shift_transformer_dropout: float = 0.0,
        shift_transformer_readout: str = "first",
        cond_pool_mode: str = "sum",
        cond_l2_norm: bool = False,
        gen_use_residual_head: bool = False,
        gen_state_source: str = "compressor",
        shift_input_source: str = "latent_mu",
    ):
        """Initialize the network modules and move them to the configured device."""
        self.net = TriShiftNet(
            x_dim=x_dim,
            z_dim=z_dim,
            cond_dim=cond_dim,
            vae_enc_hidden=vae_enc_hidden,
            vae_dec_hidden=vae_dec_hidden,
            shift_hidden=shift_hidden,
            gen_hidden=gen_hidden,
            vae_hidden_dim=vae_hidden_dim,
            vae_noise_rate=vae_noise_rate,
            vae_kl_weight=vae_kl_weight,
            dropout=dropout,
            gen_encoder_hidden=gen_encoder_hidden,
            gen_state_dim=gen_state_dim,
            gen_decoder_hidden=gen_decoder_hidden,
            gen_input_mode=gen_input_mode,
            gen_use_batchnorm=gen_use_batchnorm,
            gen_use_layernorm=gen_use_layernorm,
            shift_predict_delta=shift_predict_delta,
            shift_use_cross_attention=shift_use_cross_attention,
            shift_cross_attn_heads=shift_cross_attn_heads,
            shift_cross_attn_dropout=shift_cross_attn_dropout,
            shift_use_transformer_block=shift_use_transformer_block,
            shift_transformer_layers=shift_transformer_layers,
            shift_transformer_ff_mult=shift_transformer_ff_mult,
            shift_transformer_dropout=shift_transformer_dropout,
            shift_transformer_readout=shift_transformer_readout,
            gen_state_source=gen_state_source,
            gen_use_residual_head=gen_use_residual_head,
            shift_input_source=shift_input_source,
        ).to(self.device)
        self.hparams["cond_pool_mode"] = str(cond_pool_mode)
        self.hparams["cond_l2_norm"] = bool(cond_l2_norm)
        self.hparams["shift_predict_delta"] = bool(shift_predict_delta)
        self.hparams["shift_transformer_readout"] = str(shift_transformer_readout)
        self.hparams["shift_input_source"] = str(shift_input_source)
        self.hparams["gen_input_mode"] = str(gen_input_mode)
        self.hparams["gen_state_source"] = str(gen_state_source)
        return self

    def train_stage1_vae(
        self,
        adata_ctrl_pool,
        epochs: int,
        batch_size: int,
        lr: float,
        beta: float = 1.0,
        sched_gamma: float = 0.9,
        patience: int = 5,
        min_delta: float = 1e-3,
        amp: bool = True,
        num_workers: int = 0,
        pin_memory: bool = True,
        grad_accum_steps: int = 1,
        adata_val=None,
    ) -> dict:
        """Train the VAE on the provided stage1 pool and freeze it afterward.

        Args:
            adata_ctrl_pool: AnnData used for stage1 training (ctrl-only or train split pool).
            adata_val: Optional AnnData used for validation loss/early stopping.
            epochs: Number of epochs to train.
            batch_size: Batch size for training.
            lr: Learning rate for VAE optimizer.
            beta: KL weight for ELBO.
            sched_gamma: ExponentialLR decay rate.
            patience: Early stopping patience (0 to disable).
            min_delta: Minimum improvement for early stopping.
            amp: Enable mixed precision if CUDA is available.
            num_workers: DataLoader workers.
            pin_memory: DataLoader pin_memory.
            grad_accum_steps: Gradient accumulation steps.

        Returns:
            dict with per-epoch loss logs.
        """
        if self.net is None:
            raise ValueError("model_init must be called before training")
        print(f"[stage1] start vae training: epochs={epochs}, batch_size={batch_size}")
        beta = float(beta)

        x_np = _utils.densify_X(adata_ctrl_pool.X).astype(np.float32, copy=False)
        x_tensor = torch.from_numpy(x_np)
        dataset = TensorDataset(x_tensor)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        val_loader = None
        if adata_val is not None and getattr(adata_val, "n_obs", 0) > 0:
            x_val = _utils.densify_X(adata_val.X).astype(np.float32, copy=False)
            val_dataset = TensorDataset(torch.from_numpy(x_val))
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )

        self.net.vae.train()
        optimizer = torch.optim.Adam(self.net.vae.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=sched_gamma)
        use_amp = amp and self.device.type == "cuda"
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        best_state = None
        stopper = _EarlyStopper(patience, min_delta)
        logs = []
        for epoch in range(epochs):
            scpram_total = 0.0
            recon_loss = 0.0
            kl_loss = 0.0
            optimizer.zero_grad(set_to_none=True)
            for step, (x_batch,) in enumerate(
                tqdm(loader, desc=f"stage1 {epoch+1}/{epochs}", leave=False)
            ):
                x_batch = x_batch.to(self.device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    loss_rec, loss_kl = self.net.vae.get_loss(x_batch)
                    scpram_loss = (
                        0.5 * loss_rec + 0.5 * beta * (loss_kl * self.net.vae.kl_weight)
                    ).mean()
                    loss = scpram_loss / grad_accum_steps
                scaler.scale(loss).backward()
                if (step + 1) % grad_accum_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.net.vae.parameters(), 10.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                scpram_total += scpram_loss.item()
                recon_loss += loss_rec.mean().item()
                kl_loss += loss_kl.mean().item()

            avg_total = scpram_total / max(len(loader), 1)
            logs.append(
                {
                    "scpram_loss": avg_total,
                    "recon": recon_loss / max(len(loader), 1),
                    "kl": kl_loss / max(len(loader), 1),
                    "val_loss": None,
                }
            )
            monitor = avg_total
            if val_loader is not None:
                self.net.vae.eval()
                val_total = 0.0
                with torch.no_grad():
                    for (x_batch,) in val_loader:
                        x_batch = x_batch.to(self.device, non_blocking=True)
                        with torch.cuda.amp.autocast(enabled=use_amp):
                            loss_rec, loss_kl = self.net.vae.get_loss(x_batch)
                            scpram_loss = (
                                0.5 * loss_rec
                                + 0.5 * beta * (loss_kl * self.net.vae.kl_weight)
                            ).mean()
                        val_total += scpram_loss.item()
                val_loss = val_total / max(len(val_loader), 1)
                logs[-1]["val_loss"] = val_loss
                monitor = val_loss
                self.net.vae.train()
            scheduler.step()
            improved, should_stop = stopper.update(monitor)
            if improved:
                best_state = {k: v.detach().cpu() for k, v in self.net.vae.state_dict().items()}
            if should_stop:
                print(f"[stage1] early stop at epoch {epoch+1}")
                break

        if best_state is not None:
            self.net.vae.load_state_dict(best_state)

        for p in self.net.vae.parameters():
            p.requires_grad = False
        self.net.vae.eval()
        print("[stage1] done vae training; vae frozen")
        return {"epochs": logs}

    def encode_and_cache_mu(
        self,
        adata,
        batch_size: int,
        amp: bool = True,
        num_workers: int = 0,
        pin_memory: bool = True,
        key: str = "z_mu",
    ) -> None:
        """Encode latent mu for all cells and cache into adata.obsm.

        Args:
            adata: AnnData to encode.
            batch_size: Batch size for encoding.
            amp: Enable mixed precision if CUDA is available.
            num_workers: DataLoader workers.
            pin_memory: DataLoader pin_memory.
            key: Key to store in adata.obsm.
        """
        if self.net is None:
            raise ValueError("model_init must be called before encoding")
        print(f"[stage1] start encoding z_mu: batch_size={batch_size}, key={key}")

        x_np = _utils.densify_X(adata.X).astype(np.float32, copy=False)
        x_tensor = torch.from_numpy(x_np)
        dataset = TensorDataset(x_tensor)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        use_amp = amp and self.device.type == "cuda"
        mu_list = []
        self.net.vae.eval()
        with torch.no_grad():
            for (x_batch,) in loader:
                x_batch = x_batch.to(self.device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    mu = self.net.vae.encode_mu(x_batch)
                mu_list.append(mu.detach().cpu())
        z_mu_all = torch.cat(mu_list, dim=0).numpy()
        self.data.set_latent_mu(z_mu_all, key=key)
        print(f"[stage1] cached z_mu: shape={z_mu_all.shape}")

    def train_stage23_joint(
        self,
        split_dict: dict,
        emb_table: torch.Tensor,
        mode: str,
        k: int,
        split_id: int,
        epochs: int,
        batch_size: int,
        lr: float,
        sched_gamma: float = 0.9,
        patience: int = 5,
        min_delta: float = 1e-3,
        amp: bool = True,
        num_workers: int = 0,
        pin_memory: bool = True,
        grad_accum_steps: int = 1,
        cache_topk_path: str | None = None,
        gamma: float = 1.0,
        lambda_dir: float = 1.0,
        lambda_dir_expr: float | None = None,
        lambda_dir_z: float | None = None,
        lambda_z: float = 1.0,
        deg_weight: float = 1.0,
        topk_strategy: str = "random",
        sample_soft_ctrl: bool = True,
        latent_loss_type: str = "gears",
        lambda_expr_mse: float = 0.0,
        per_condition_ot: bool = False,
        disable_loss_z_supervision: bool = False,
        reuse_ot_cache: bool = False,
        topk_cache_key: str | None = None,
    ) -> dict:
        """Jointly train shift predictor and generator with OT-matched pairs.

        Args:
            split_dict: Dict containing train/val/test splits.
            emb_table: Condition embedding table tensor.
            mode: Matching mode (knn/ot/knn_ot/soft_ot).
            k: Top-k controls per perturbed cell.
            split_id: Split id for seeding.
            epochs: Training epochs.
            batch_size: Batch size.
            lr: Learning rate.
            sched_gamma: ExponentialLR decay rate.
            patience: Early stopping patience (0 to disable).
            min_delta: Minimum improvement for early stopping.
            amp: Mixed precision toggle.
            num_workers: DataLoader workers.
            pin_memory: DataLoader pin_memory.
            grad_accum_steps: Gradient accumulation steps.
            cache_topk_path: Optional cache for top-k map.
            gamma: Autofocus exponent.
            lambda_dir: Direction loss weight (fallback).
            lambda_dir_expr: Direction loss weight for expression.
            lambda_dir_z: Direction loss weight for latent shift.
            lambda_z: Latent loss weight.
            deg_weight: Upweight factor for DE genes in expression loss.
            disable_loss_z_supervision: If true, do not supervise delta_z (no loss_z).

        Returns:
            dict with per-epoch loss logs.
        """
        if self.net is None:
            raise ValueError("model_init must be called before training")
        self.hparams["stage3_only"] = False
        if lambda_dir_expr is None:
            lambda_dir_expr = lambda_dir
        if lambda_dir_z is None:
            lambda_dir_z = lambda_dir
        print(
            f"[stage23] start joint training: mode={mode}, k={k}, split_id={split_id}, epochs={epochs}"
        )
        disable_loss_z_supervision = self._derive_disable_loss_z_supervision(
            disable_loss_z_supervision,
            "[stage23] disable_loss_z_supervision=True -> optimize expression loss only",
        )

        base_seed = int(self.hparams.get("base_seed", 24))
        z_mu_all = self._require_cached_z_mu()
        train_ctrl_adata, train_ctrl_global_idx = self._get_ctrl_pool_from_split(
            split_dict.get("train")
        )
        need_topk_weights = self._need_topk_weights(mode, topk_strategy)
        topk_map, topk_weights = self._build_topk_map_for_split(
            split_adata=split_dict["train"],
            ctrl_global_idx=train_ctrl_global_idx,
            mode=mode,
            k=k,
            seed=base_seed + split_id,
            candidates=100,
            cache_path=cache_topk_path,
            per_condition_ot=per_condition_ot,
            need_topk_weights=need_topk_weights,
            reuse_ot_cache=reuse_ot_cache,
            cache_key=topk_cache_key,
        )
        print(f"[stage23] topk_map built: shape={topk_map.shape}")

        pert_mask = split_dict["train"].obs[self.data.label_key].astype(str) != self.data.ctrl_label
        pert_adata = split_dict["train"][pert_mask]
        val_adata = self._get_val_pert_adata(split_dict)

        val_topk = None
        val_topk_weights = None
        val_ctrl_adata = None
        val_ctrl_global_idx = None
        if val_adata is not None:
            val_ctrl_adata, val_ctrl_global_idx = self._get_ctrl_pool_from_split(
                split_dict.get("val")
            )
            val_topk, val_topk_weights = self._build_topk_map_for_split(
                split_adata=split_dict["val"],
                ctrl_global_idx=val_ctrl_global_idx,
                mode=mode,
                k=k,
                seed=base_seed + split_id,
                candidates=100,
                cache_path=None,
                per_condition_ot=per_condition_ot,
                need_topk_weights=need_topk_weights,
                reuse_ot_cache=reuse_ot_cache,
                cache_key=topk_cache_key,
            )
        z_ctrl_mu_all = z_mu_all[train_ctrl_global_idx]
        all_idx_map = {n: i for i, n in enumerate(self.data.adata_all.obs_names)}
        pert_global_idx = np.array([all_idx_map[n] for n in pert_adata.obs_names])
        z_pert_mu_all = z_mu_all[pert_global_idx]
        dataset = _TopKTrainDataset(
            pert_adata,
            train_ctrl_adata,
            topk_map,
            topk_weights,
            "embd_index",
            self.data.label_key,
            z_ctrl_mu_all,
            z_pert_mu_all,
            soft_ot_weighted_delta=(mode == "soft_ot"),
            sample_soft_ctrl=sample_soft_ctrl,
            topk_strategy=topk_strategy,
        )
        val_dataset = None
        if (
            val_adata is not None
            and val_topk is not None
            and val_ctrl_adata is not None
            and val_ctrl_global_idx is not None
        ):
            val_global_idx = np.array([all_idx_map[n] for n in val_adata.obs_names])
            z_pert_val = z_mu_all[val_global_idx]
            z_ctrl_mu_val = z_mu_all[val_ctrl_global_idx]
            val_dataset = _TopKTrainDataset(
                val_adata,
                val_ctrl_adata,
                val_topk,
                val_topk_weights,
                "embd_index",
                self.data.label_key,
                z_ctrl_mu_val,
                z_pert_val,
                soft_ot_weighted_delta=(mode == "soft_ot"),
                sample_soft_ctrl=False,
                topk_strategy=topk_strategy,
            )

        loader = self._build_topk_loader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        val_loader = self._build_topk_loader(
            dataset=val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        for p in self.net.vae.parameters():
            p.requires_grad = False
        self.net.vae.eval()
        self.net.shift.train()
        self.net.gen.train()

        params = list(self.net.shift.parameters()) + list(self.net.gen.parameters())
        optimizer = torch.optim.Adam(params, lr=lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=sched_gamma)
        use_amp = amp and self.device.type == "cuda"
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        emb_table = emb_table.to(self.device)
        nonzero_idx_dict = self._get_nonzero_idx_dict()
        deg_idx_dict = None
        if deg_weight != 1.0:
            deg_idx_dict = self._get_degs_idx_dict()
            if not deg_idx_dict:
                print("[degs] warning: deg_weight set but no degs found; using unweighted loss")
        latent_idx_dict = None
        if not disable_loss_z_supervision and latent_loss_type == "gears":
            latent_dims = int(getattr(self.net.gen, "z_dim", z_ctrl_mu_all.shape[1]))
            latent_idx = list(range(latent_dims))
            cond_vals = pert_adata.obs[self.data.label_key].astype(str).values
            if val_adata is not None:
                cond_vals = np.concatenate(
                    [cond_vals, val_adata.obs[self.data.label_key].astype(str).values]
                )
            latent_idx_dict = {c: latent_idx for c in np.unique(cond_vals)}

        best_state = None
        stopper = _EarlyStopper(patience, min_delta)
        logs = []
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_loss_expr = 0.0
            epoch_loss_z = 0.0
            optimizer.zero_grad(set_to_none=True)
            for step, (x_ctrl, x_true, idx_list, cond_str, z_ctrl_mu, _, delta_target) in enumerate(
                tqdm(loader, desc=f"stage23 {epoch+1}/{epochs}", leave=False)
            ):
                x_ctrl = x_ctrl.to(self.device, non_blocking=True)
                x_true = x_true.to(self.device, non_blocking=True)
                z_ctrl_mu = z_ctrl_mu.to(self.device, non_blocking=True)
                delta_target = delta_target.to(self.device, non_blocking=True)
                cond_vec = self._build_cond_vec_batch(emb_table, idx_list)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    out = self.net.forward_joint(x_ctrl, cond_vec, z_ctrl_mu=z_ctrl_mu)
                    x_pred = out["x_pred"]
                    shift_pred = out["shift_repr"]
                    loss_expr = self._compute_expression_loss(
                        x_pred=x_pred,
                        x_true=x_true,
                        x_ctrl=x_ctrl,
                        cond_str=cond_str,
                        nonzero_idx_dict=nonzero_idx_dict,
                        gamma=gamma,
                        lambda_dir_expr=lambda_dir_expr,
                        deg_idx_dict=deg_idx_dict,
                        deg_weight=deg_weight,
                        lambda_expr_mse=lambda_expr_mse,
                    )
                    if disable_loss_z_supervision:
                        loss_z = torch.zeros((), device=x_pred.device, dtype=x_pred.dtype)
                        loss = loss_expr / grad_accum_steps
                    else:
                        loss_z = self._compute_latent_supervision_loss(
                            shift_pred=shift_pred,
                            delta_target=delta_target,
                            cond_str=cond_str,
                            latent_loss_type=latent_loss_type,
                            latent_idx_dict=latent_idx_dict,
                            gamma=gamma,
                            lambda_dir_z=lambda_dir_z,
                        )
                        loss = (loss_expr + lambda_z * loss_z) / grad_accum_steps
                scaler.scale(loss).backward()
                if (step + 1) % grad_accum_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                epoch_loss += loss.item() * grad_accum_steps
                epoch_loss_expr += loss_expr.item()
                epoch_loss_z += loss_z.item()

            denom = max(len(loader), 1)
            avg_loss = epoch_loss / denom
            val_loss = None
            if val_loader is not None:
                self.net.shift.eval()
                self.net.gen.eval()
                val_total = 0.0
                with torch.no_grad():
                    for x_ctrl, x_true, idx_list, cond_str, z_ctrl_mu, _, delta_target in val_loader:
                        x_ctrl = x_ctrl.to(self.device, non_blocking=True)
                        x_true = x_true.to(self.device, non_blocking=True)
                        z_ctrl_mu = z_ctrl_mu.to(self.device, non_blocking=True)
                        delta_target = delta_target.to(self.device, non_blocking=True)
                        cond_vec = self._build_cond_vec_batch(emb_table, idx_list)
                        out = self.net.forward_joint(x_ctrl, cond_vec, z_ctrl_mu=z_ctrl_mu)
                        x_pred = out["x_pred"]
                        shift_pred = out["shift_repr"]
                        loss_expr = self._compute_expression_loss(
                            x_pred=x_pred,
                            x_true=x_true,
                            x_ctrl=x_ctrl,
                            cond_str=cond_str,
                            nonzero_idx_dict=nonzero_idx_dict,
                            gamma=gamma,
                            lambda_dir_expr=lambda_dir_expr,
                            deg_idx_dict=deg_idx_dict,
                            deg_weight=deg_weight,
                            lambda_expr_mse=lambda_expr_mse,
                        )

                        if disable_loss_z_supervision:
                            val_total += loss_expr.item()
                        else:
                            loss_z = self._compute_latent_supervision_loss(
                                shift_pred=shift_pred,
                                delta_target=delta_target,
                                cond_str=cond_str,
                                latent_loss_type=latent_loss_type,
                                latent_idx_dict=latent_idx_dict,
                                gamma=gamma,
                                lambda_dir_z=lambda_dir_z,
                            )
                            val_total += (loss_expr + lambda_z * loss_z).item()
                val_loss = val_total / max(len(val_loader), 1)
                self.net.shift.train()
                self.net.gen.train()
            logs.append(
                {
                    "loss": avg_loss,
                    "loss_expr": epoch_loss_expr / denom,
                    "loss_z": epoch_loss_z / denom,
                    "val_loss": val_loss,
                }
            )
            scheduler.step()
            monitor = val_loss if val_loss is not None else avg_loss
            improved, should_stop = stopper.update(monitor)
            if improved:
                best_state = {
                    "shift": {k: v.detach().cpu() for k, v in self.net.shift.state_dict().items()},
                    "gen": {k: v.detach().cpu() for k, v in self.net.gen.state_dict().items()},
                }
            if should_stop:
                print(f"[stage23] early stop at epoch {epoch+1}")
                break

        if best_state is not None:
            self.net.shift.load_state_dict(best_state["shift"])
            self.net.gen.load_state_dict(best_state["gen"])
        print("[stage23] done joint training")
        return {"epochs": logs}

    def train_stage23_sequential(
        self,
        split_dict: dict,
        emb_table: torch.Tensor,
        mode: str,
        k: int,
        split_id: int,
        epochs_stage2: int,
        epochs_stage3: int,
        batch_size: int,
        lr_stage2: float,
        lr_stage3: float,
        sched_gamma_stage2: float = 0.9,
        sched_gamma_stage3: float = 0.9,
        patience_stage2: int = 5,
        patience_stage3: int = 5,
        min_delta_stage2: float = 1e-3,
        min_delta_stage3: float = 1e-3,
        amp: bool = True,
        num_workers: int = 0,
        pin_memory: bool = True,
        grad_accum_steps: int = 1,
        cache_topk_path: str | None = None,
        gamma: float = 1.0,
        lambda_dir: float = 1.0,
        lambda_dir_expr: float | None = None,
        deg_weight: float = 1.0,
        topk_strategy: str = "random",
        sample_soft_ctrl: bool = True,
        lambda_expr_mse: float = 0.0,
        per_condition_ot: bool = False,
        disable_loss_z_supervision: bool = False,
        reuse_ot_cache: bool = False,
        topk_cache_key: str | None = None,
    ) -> dict:
        """Sequential training: first shift predictor, then generator.

        Args:
            split_dict: Dict containing train/val/test splits.
            emb_table: Condition embedding table tensor.
            mode: Matching mode (knn/ot/knn_ot/soft_ot).
            k: Top-k controls per perturbed cell.
            split_id: Split id for seeding.
            epochs_stage2: Shift predictor epochs.
            epochs_stage3: Generator epochs.
            batch_size: Batch size.
            lr_stage2: Shift predictor learning rate.
            lr_stage3: Generator learning rate.
            sched_gamma_stage2: ExponentialLR decay for stage2.
            sched_gamma_stage3: ExponentialLR decay for stage3.
            patience_stage2: Early stopping patience for stage2.
            patience_stage3: Early stopping patience for stage3.
            min_delta_stage2: Minimum improvement for stage2.
            min_delta_stage3: Minimum improvement for stage3.
            amp: Mixed precision toggle.
            num_workers: DataLoader workers.
            pin_memory: DataLoader pin_memory.
            grad_accum_steps: Gradient accumulation steps.
            cache_topk_path: Optional cache for top-k map.
            gamma: Autofocus exponent.
            lambda_dir: Direction loss weight (fallback).
            lambda_dir_expr: Direction loss weight for expression.
            deg_weight: Upweight factor for DE genes in expression loss.
            disable_loss_z_supervision: If true, skip stage2 delta supervision and
                learn shift implicitly from stage3 expression loss.

        Returns:
            dict with stage2/stage3 loss logs.
        """
        if self.net is None:
            raise ValueError("model_init must be called before training")
        self.hparams["stage3_only"] = False
        if lambda_dir_expr is None:
            lambda_dir_expr = lambda_dir
        print(
            f"[stage23] start sequential training: mode={mode}, k={k}, split_id={split_id}"
        )
        disable_loss_z_supervision = self._derive_disable_loss_z_supervision(
            disable_loss_z_supervision,
            "[stage23] disable_loss_z_supervision=True -> skip supervised stage2; "
            "stage3 trains shift+generator with expression loss only",
        )

        base_seed = int(self.hparams.get("base_seed", 24))
        z_mu_all = self._require_cached_z_mu()
        train_ctrl_adata, train_ctrl_global_idx = self._get_ctrl_pool_from_split(
            split_dict.get("train")
        )
        need_topk_weights = self._need_topk_weights(mode, topk_strategy)
        topk_map, topk_weights = self._build_topk_map_for_split(
            split_adata=split_dict["train"],
            ctrl_global_idx=train_ctrl_global_idx,
            mode=mode,
            k=k,
            seed=base_seed + split_id,
            candidates=100,
            cache_path=cache_topk_path,
            per_condition_ot=per_condition_ot,
            need_topk_weights=need_topk_weights,
            reuse_ot_cache=reuse_ot_cache,
            cache_key=topk_cache_key,
        )
        print(f"[stage23] topk_map built: shape={topk_map.shape}")

        pert_mask = split_dict["train"].obs[self.data.label_key].astype(str) != self.data.ctrl_label
        pert_adata = split_dict["train"][pert_mask]

        z_ctrl_mu_all = z_mu_all[train_ctrl_global_idx]
        all_idx_map = {n: i for i, n in enumerate(self.data.adata_all.obs_names)}
        pert_global_idx = np.array([all_idx_map[n] for n in pert_adata.obs_names])
        z_pert_mu_all = z_mu_all[pert_global_idx]

        dataset = _TopKTrainDataset(
            pert_adata,
            train_ctrl_adata,
            topk_map,
            topk_weights,
            "embd_index",
            self.data.label_key,
            z_ctrl_mu_all,
            z_pert_mu_all,
            soft_ot_weighted_delta=(mode == "soft_ot"),
            sample_soft_ctrl=sample_soft_ctrl,
            topk_strategy=topk_strategy,
        )

        loader = self._build_topk_loader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        val_adata = self._get_val_pert_adata(split_dict)

        val_topk = None
        val_topk_weights = None
        val_ctrl_adata = None
        val_ctrl_global_idx = None
        if val_adata is not None:
            val_ctrl_adata, val_ctrl_global_idx = self._get_ctrl_pool_from_split(
                split_dict.get("val")
            )
            val_topk, val_topk_weights = self._build_topk_map_for_split(
                split_adata=split_dict["val"],
                ctrl_global_idx=val_ctrl_global_idx,
                mode=mode,
                k=k,
                seed=base_seed + split_id,
                candidates=100,
                cache_path=None,
                per_condition_ot=per_condition_ot,
                need_topk_weights=need_topk_weights,
                reuse_ot_cache=reuse_ot_cache,
                cache_key=topk_cache_key,
            )

        val_dataset = None
        if (
            val_adata is not None
            and val_topk is not None
            and val_ctrl_adata is not None
            and val_ctrl_global_idx is not None
        ):
            val_global_idx = np.array([all_idx_map[n] for n in val_adata.obs_names])
            z_pert_val = z_mu_all[val_global_idx]
            z_ctrl_mu_val = z_mu_all[val_ctrl_global_idx]
            val_dataset = _TopKTrainDataset(
                val_adata,
                val_ctrl_adata,
                val_topk,
                val_topk_weights,
                "embd_index",
                self.data.label_key,
                z_ctrl_mu_val,
                z_pert_val,
                soft_ot_weighted_delta=(mode == "soft_ot"),
                sample_soft_ctrl=False,
                topk_strategy=topk_strategy,
            )

        val_loader = self._build_topk_loader(
            dataset=val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        for p in self.net.vae.parameters():
            p.requires_grad = False
        self.net.vae.eval()
        emb_table = emb_table.to(self.device)
        use_amp = amp and self.device.type == "cuda"
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        nonzero_idx_dict = self._get_nonzero_idx_dict()
        deg_idx_dict = None
        if deg_weight != 1.0:
            deg_idx_dict = self._get_degs_idx_dict()
            if not deg_idx_dict:
                print("[degs] warning: deg_weight set but no degs found; using unweighted loss")

        logs_stage2 = []
        if disable_loss_z_supervision:
            logs_stage2.append({"loss": np.nan, "val_loss": np.nan, "skipped": True})
        else:
            # Phase A: shift only
            print(f"[stage2] start shift training: epochs={epochs_stage2}")
            for p in self.net.shift.parameters():
                p.requires_grad = True
            for p in self.net.gen.parameters():
                p.requires_grad = False
            if getattr(self.net, "shift_input_source", "latent_mu") == "state":
                for p in self.net.gen.compressor.parameters():
                    p.requires_grad = True
            self.net.shift.train()
            self.net.gen.eval()
            params_stage2 = list(self.net.shift.parameters())
            if getattr(self.net, "shift_input_source", "latent_mu") == "state":
                params_stage2.extend(list(self.net.gen.compressor.parameters()))
            opt_shift = torch.optim.Adam(params_stage2, lr=lr_stage2)
            sched_shift = torch.optim.lr_scheduler.ExponentialLR(opt_shift, gamma=sched_gamma_stage2)
            best_state = None
            stopper = _EarlyStopper(patience_stage2, min_delta_stage2)
            for epoch in range(epochs_stage2):
                epoch_loss = 0.0
                opt_shift.zero_grad(set_to_none=True)
                for step, (x_ctrl, _, idx_list, _, z_ctrl_mu, _, delta_target) in enumerate(
                    tqdm(loader, desc=f"stage2 {epoch+1}/{epochs_stage2}", leave=False)
                ):
                    x_ctrl = x_ctrl.to(self.device, non_blocking=True)
                    z_ctrl_mu = z_ctrl_mu.to(self.device, non_blocking=True)
                    delta_target = delta_target.to(self.device, non_blocking=True)
                    cond_vec = self._build_cond_vec_batch(emb_table, idx_list)
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        delta_pred = self.net.predict_shift_repr(
                            z_ctrl_mu,
                            cond_vec,
                            x_ctrl=x_ctrl,
                        )
                        loss = F.mse_loss(delta_pred, delta_target) / grad_accum_steps
                    scaler.scale(loss).backward()
                    if (step + 1) % grad_accum_steps == 0:
                        scaler.step(opt_shift)
                        scaler.update()
                        opt_shift.zero_grad(set_to_none=True)
                    epoch_loss += loss.item() * grad_accum_steps
                avg_loss = epoch_loss / max(len(loader), 1)
                val_loss = None
                if val_loader is not None:
                    self.net.shift.eval()
                    val_total = 0.0
                    with torch.no_grad():
                        for x_ctrl, _, idx_list, _, z_ctrl_mu, _, delta_target in val_loader:
                            x_ctrl = x_ctrl.to(self.device, non_blocking=True)
                            z_ctrl_mu = z_ctrl_mu.to(self.device, non_blocking=True)
                            delta_target = delta_target.to(self.device, non_blocking=True)
                            cond_vec = self._build_cond_vec_batch(emb_table, idx_list)
                            delta_pred = self.net.predict_shift_repr(
                                z_ctrl_mu,
                                cond_vec,
                                x_ctrl=x_ctrl,
                            )
                            val_total += F.mse_loss(delta_pred, delta_target).item()
                    val_loss = val_total / max(len(val_loader), 1)
                    self.net.shift.train()
                logs_stage2.append({"loss": avg_loss, "val_loss": val_loss})
                sched_shift.step()
                monitor = val_loss if val_loss is not None else avg_loss
                improved, should_stop = stopper.update(monitor)
                if improved:
                    best_state = {
                        "shift": {k: v.detach().cpu() for k, v in self.net.shift.state_dict().items()}
                    }
                    if getattr(self.net, "shift_input_source", "latent_mu") == "state":
                        best_state["compressor"] = {
                            k: v.detach().cpu() for k, v in self.net.gen.compressor.state_dict().items()
                        }
                if should_stop:
                    print(f"[stage2] early stop at epoch {epoch+1}")
                    break

            if best_state is not None:
                self.net.shift.load_state_dict(best_state["shift"])
                if "compressor" in best_state:
                    self.net.gen.compressor.load_state_dict(best_state["compressor"])

        # Phase B: generator only
        print(f"[stage3] start generator training: epochs={epochs_stage3}")
        for p in self.net.shift.parameters():
            p.requires_grad = bool(disable_loss_z_supervision)
        for p in self.net.gen.parameters():
            p.requires_grad = True
        if disable_loss_z_supervision:
            self.net.shift.train()
        else:
            self.net.shift.eval()
        self.net.gen.train()
        if disable_loss_z_supervision:
            params_stage3 = list(self.net.shift.parameters()) + list(self.net.gen.parameters())
        else:
            params_stage3 = list(self.net.gen.parameters())
        opt_gen = torch.optim.Adam(params_stage3, lr=lr_stage3)
        sched_gen = torch.optim.lr_scheduler.ExponentialLR(opt_gen, gamma=sched_gamma_stage3)
        logs_stage3 = []
        best_state = None
        stopper = _EarlyStopper(patience_stage3, min_delta_stage3)
        for epoch in range(epochs_stage3):
            epoch_loss = 0.0
            opt_gen.zero_grad(set_to_none=True)
            for step, (x_ctrl, x_true, idx_list, cond_str, z_ctrl_mu, _, _) in enumerate(
                tqdm(loader, desc=f"stage3 {epoch+1}/{epochs_stage3}", leave=False)
            ):
                x_ctrl = x_ctrl.to(self.device, non_blocking=True)
                x_true = x_true.to(self.device, non_blocking=True)
                z_ctrl_mu = z_ctrl_mu.to(self.device, non_blocking=True)
                cond_vec = self._build_cond_vec_batch(emb_table, idx_list)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    shift_repr = self.net.predict_shift_repr(
                        z_ctrl_mu,
                        cond_vec,
                        x_ctrl=x_ctrl,
                    )
                    x_pred = self.net.gen(x_ctrl, cond_vec, shift_repr)
                    loss_expr = self._compute_expression_loss(
                        x_pred=x_pred,
                        x_true=x_true,
                        x_ctrl=x_ctrl,
                        cond_str=cond_str,
                        nonzero_idx_dict=nonzero_idx_dict,
                        gamma=gamma,
                        lambda_dir_expr=lambda_dir_expr,
                        deg_idx_dict=deg_idx_dict,
                        deg_weight=deg_weight,
                        lambda_expr_mse=lambda_expr_mse,
                    )
                    loss = loss_expr / grad_accum_steps
                scaler.scale(loss).backward()
                if (step + 1) % grad_accum_steps == 0:
                    scaler.step(opt_gen)
                    scaler.update()
                    opt_gen.zero_grad(set_to_none=True)
                epoch_loss += loss.item() * grad_accum_steps
            avg_loss = epoch_loss / max(len(loader), 1)
            val_loss = None
            if val_loader is not None:
                self.net.gen.eval()
                val_total = 0.0
                with torch.no_grad():
                    for x_ctrl, x_true, idx_list, cond_str, z_ctrl_mu, _, _ in val_loader:
                        x_ctrl = x_ctrl.to(self.device, non_blocking=True)
                        x_true = x_true.to(self.device, non_blocking=True)
                        z_ctrl_mu = z_ctrl_mu.to(self.device, non_blocking=True)
                        cond_vec = self._build_cond_vec_batch(emb_table, idx_list)
                        shift_repr = self.net.predict_shift_repr(
                            z_ctrl_mu,
                            cond_vec,
                            x_ctrl=x_ctrl,
                        )
                        x_pred = self.net.gen(x_ctrl, cond_vec, shift_repr)
                        loss_expr = self._compute_expression_loss(
                            x_pred=x_pred,
                            x_true=x_true,
                            x_ctrl=x_ctrl,
                            cond_str=cond_str,
                            nonzero_idx_dict=nonzero_idx_dict,
                            gamma=gamma,
                            lambda_dir_expr=lambda_dir_expr,
                            deg_idx_dict=deg_idx_dict,
                            deg_weight=deg_weight,
                            lambda_expr_mse=lambda_expr_mse,
                        )
                        val_total += loss_expr.item()
                val_loss = val_total / max(len(val_loader), 1)
                self.net.gen.train()
                if disable_loss_z_supervision:
                    self.net.shift.train()
            logs_stage3.append({"loss": avg_loss, "val_loss": val_loss})
            sched_gen.step()
            monitor = val_loss if val_loss is not None else avg_loss
            improved, should_stop = stopper.update(monitor)
            if improved:
                best_state = {"gen": {k: v.detach().cpu() for k, v in self.net.gen.state_dict().items()}}
                if disable_loss_z_supervision:
                    best_state["shift"] = {
                        k: v.detach().cpu() for k, v in self.net.shift.state_dict().items()
                    }
            if should_stop:
                print(f"[stage3] early stop at epoch {epoch+1}")
                break

        if best_state is not None:
            self.net.gen.load_state_dict(best_state["gen"])
            if disable_loss_z_supervision and "shift" in best_state:
                self.net.shift.load_state_dict(best_state["shift"])

        print("[stage23] done sequential training")
        return {"stage2": logs_stage2, "stage3": logs_stage3}

    def train_stage3_only(
        self,
        split_dict: dict,
        emb_table: torch.Tensor,
        split_id: int,
        epochs: int,
        batch_size: int,
        lr: float,
        sched_gamma: float = 0.9,
        patience: int = 5,
        min_delta: float = 1e-3,
        amp: bool = True,
        num_workers: int = 0,
        pin_memory: bool = True,
        grad_accum_steps: int = 1,
        gamma: float = 1.0,
        lambda_dir: float = 1.0,
        lambda_dir_expr: float | None = None,
        deg_weight: float = 1.0,
        lambda_expr_mse: float = 0.0,
        shuffle: bool = True,
    ) -> dict:
        """Train generator only with random control pairing (stage3-only mode).

        Args:
            split_dict: Dict containing train/val/test splits.
            emb_table: Condition embedding table tensor.
            split_id: Split id for seeding.
            epochs: Training epochs.
            batch_size: Batch size.
            lr: Learning rate.
            sched_gamma: ExponentialLR decay rate.
            patience: Early stopping patience (0 to disable).
            min_delta: Minimum improvement for early stopping.
            amp: Mixed precision toggle.
            num_workers: DataLoader workers.
            pin_memory: DataLoader pin_memory.
            grad_accum_steps: Gradient accumulation steps.
            gamma: Autofocus exponent.
            lambda_dir: Direction loss weight (fallback).
            lambda_dir_expr: Direction loss weight for expression.
            deg_weight: Upweight factor for DE genes in expression loss.
            shuffle: Shuffle control/pert pairing.

        Returns:
            dict with per-epoch loss logs.
        """
        if self.net is None:
            raise ValueError("model_init must be called before training")
        if lambda_dir_expr is None:
            lambda_dir_expr = lambda_dir
        self.hparams["stage3_only"] = True
        print(f"[stage3_only] start: split_id={split_id}, epochs={epochs}")

        train_adata = split_dict["train"]
        train_ctrl = train_adata[train_adata.obs[self.data.label_key].astype(str) == self.data.ctrl_label]
        train_pert = train_adata[train_adata.obs[self.data.label_key].astype(str) != self.data.ctrl_label]

        val_adata = split_dict.get("val", None)
        if val_adata is not None:
            val_ctrl = val_adata[val_adata.obs[self.data.label_key].astype(str) == self.data.ctrl_label]
            val_pert = val_adata[val_adata.obs[self.data.label_key].astype(str) != self.data.ctrl_label]
        else:
            val_ctrl = None
            val_pert = None

        if train_pert.n_obs == 0 or train_ctrl.n_obs == 0:
            raise ValueError("train split lacks ctrl or pert cells for stage3_only")

        dataset = _Stage3PairDataset(
            train_ctrl, train_pert, "embd_index", self.data.label_key, shuffle_pairs=shuffle
        )

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=_collate_no_latent,
        )

        val_loader = None
        if val_ctrl is not None and val_pert is not None and val_pert.n_obs > 0 and val_ctrl.n_obs > 0:
            val_dataset = _Stage3PairDataset(
                val_ctrl, val_pert, "embd_index", self.data.label_key, shuffle_pairs=False
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                collate_fn=_collate_no_latent,
            )

        for p in self.net.shift.parameters():
            p.requires_grad = False
        for p in self.net.gen.parameters():
            p.requires_grad = True
        self.net.shift.eval()
        self.net.gen.train()

        opt_gen = torch.optim.Adam(self.net.gen.parameters(), lr=lr)
        sched_gen = torch.optim.lr_scheduler.ExponentialLR(opt_gen, gamma=sched_gamma)
        use_amp = amp and self.device.type == "cuda"
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        emb_table = emb_table.to(self.device)
        nonzero_idx_dict = self._get_nonzero_idx_dict()
        deg_idx_dict = None
        if deg_weight != 1.0:
            deg_idx_dict = self._get_degs_idx_dict()
            if not deg_idx_dict:
                print("[degs] warning: deg_weight set but no degs found; using unweighted loss")

        best_state = None
        stopper = _EarlyStopper(patience, min_delta)
        logs = []
        for epoch in range(epochs):
            epoch_loss = 0.0
            opt_gen.zero_grad(set_to_none=True)
            for step, (x_ctrl, x_true, idx_list, cond_str) in enumerate(
                tqdm(loader, desc=f"stage3_only {epoch+1}/{epochs}", leave=False)
            ):
                x_ctrl = x_ctrl.to(self.device, non_blocking=True)
                x_true = x_true.to(self.device, non_blocking=True)
                cond_vec = self._build_cond_vec_batch(emb_table, idx_list)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    x_pred = self.net.gen.forward_no_delta(x_ctrl, cond_vec)
                    loss_expr = self._compute_expression_loss(
                        x_pred=x_pred,
                        x_true=x_true,
                        x_ctrl=x_ctrl,
                        cond_str=cond_str,
                        nonzero_idx_dict=nonzero_idx_dict,
                        gamma=gamma,
                        lambda_dir_expr=lambda_dir_expr,
                        deg_idx_dict=deg_idx_dict,
                        deg_weight=deg_weight,
                        lambda_expr_mse=lambda_expr_mse,
                    )
                    loss = loss_expr / grad_accum_steps
                scaler.scale(loss).backward()
                if (step + 1) % grad_accum_steps == 0:
                    scaler.step(opt_gen)
                    scaler.update()
                    opt_gen.zero_grad(set_to_none=True)
                epoch_loss += loss_expr.item()

            avg_loss = epoch_loss / max(len(loader), 1)
            val_loss = None
            if val_loader is not None:
                self.net.gen.eval()
                val_total = 0.0
                with torch.no_grad():
                    for x_ctrl, x_true, idx_list, cond_str in val_loader:
                        x_ctrl = x_ctrl.to(self.device, non_blocking=True)
                        x_true = x_true.to(self.device, non_blocking=True)
                        cond_vec = self._build_cond_vec_batch(emb_table, idx_list)
                        x_pred = self.net.gen.forward_no_delta(x_ctrl, cond_vec)
                        loss_expr = self._compute_expression_loss(
                            x_pred=x_pred,
                            x_true=x_true,
                            x_ctrl=x_ctrl,
                            cond_str=cond_str,
                            nonzero_idx_dict=nonzero_idx_dict,
                            gamma=gamma,
                            lambda_dir_expr=lambda_dir_expr,
                            deg_idx_dict=deg_idx_dict,
                            deg_weight=deg_weight,
                            lambda_expr_mse=lambda_expr_mse,
                        )
                        val_total += loss_expr.item()
                val_loss = val_total / max(len(val_loader), 1)
                self.net.gen.train()

            logs.append({"loss": avg_loss, "val_loss": val_loss})
            sched_gen.step()
            monitor = val_loss if val_loss is not None else avg_loss
            improved, should_stop = stopper.update(monitor)
            if improved:
                best_state = {k: v.detach().cpu() for k, v in self.net.gen.state_dict().items()}
            if should_stop:
                print(f"[stage3_only] early stop at epoch {epoch+1}")
                break

        if best_state is not None:
            self.net.gen.load_state_dict(best_state)
        print("[stage3_only] done")
        return {"epochs": logs}

    def evaluate(
        self,
        split_dict: dict,
        emb_table: torch.Tensor,
        split_id: int,
        n_ensemble: int = 300,
        base_seed: int = 24,
    ) -> pd.DataFrame:
        """Evaluate per condition with ensemble control sampling.

        Returns:
            DataFrame with per-condition metrics.
        """
        if self.net is None:
            raise ValueError("model_init must be called before evaluation")
        print(
            f"[eval] start: split_id={split_id}, n_ensemble={n_ensemble}, n_conds={len(split_dict['test_conds'])}"
        )

        emb_table = emb_table.to(self.device)
        self.net.eval()
        eval_seed = base_seed + split_id
        rng = np.random.RandomState(eval_seed)
        conds = split_dict["test_conds"]
        results = []

        train_ctrl_adata, _ = self._get_ctrl_pool_from_split(split_dict.get("train"))
        X_ctrl = train_ctrl_adata.X
        X_all = self.data.adata_all.X
        cond_series = self.data.adata_all.obs[self.data.label_key].astype(str).values
        if sp.issparse(X_ctrl):
            ctrl_mean_all = np.asarray(X_ctrl.mean(axis=0), dtype=np.float32).reshape(1, -1)
        else:
            ctrl_mean_all = np.asarray(X_ctrl, dtype=np.float32).mean(axis=0, keepdims=True)

        degs_non_dropout = self.data.adata_all.uns.get("top20_degs_non_dropout", {})
        degs_final = self.data.adata_all.uns.get("top20_degs_final", {})
        train_adata = split_dict.get("train", None)
        if train_adata is not None:
            train_cond_arr = train_adata.obs[self.data.label_key].astype(str).values
            pert_reference = average_of_perturbation_centroids(
                X=train_adata.X,
                conditions=train_cond_arr,
                ctrl_label=self.data.ctrl_label,
            )
        else:
            pert_reference = np.asarray(ctrl_mean_all, dtype=np.float32).reshape(-1)

        for cond in conds:
            cond_mask = cond_series == cond
            if not np.any(cond_mask):
                print(f"[eval] skip missing condition: {cond}")
                continue
            embd_idx_list = self._get_cond_embd_idx(cond_mask)
            ctrl_expr = self._sample_ctrl_expr(X_ctrl, n_ensemble, rng)

            true_expr = self._select_dense(X_all, cond_mask)
            true_mean = true_expr.mean(axis=0, keepdims=True)
            if true_expr.size == 0 or np.isnan(true_mean).any():
                print(f"[eval] empty/invalid true_mean: {cond}")

            cond_vec = self._build_cond_vec(emb_table, embd_idx_list, n_ensemble)
            x_pred = self._predict_expr_from_ctrl(ctrl_expr, cond_vec)
            pred_mean = x_pred.mean(axis=0, keepdims=True)
            if np.isnan(pred_mean).any():
                print(f"[eval] invalid pred_mean: {cond}")

            if cond in degs_non_dropout:
                deg_idx = np.asarray(degs_non_dropout.get(cond, []), dtype=int)
            else:
                deg_idx = np.asarray(degs_final.get(cond, []), dtype=int)
            remove_idx = np.asarray(self.data.cond_to_gene_idx.get(cond, []), dtype=int)
            if remove_idx.size > 0:
                deg_idx = np.setdiff1d(deg_idx, remove_idx)
            if deg_idx.size == 0:
                print(f"[eval] skip condition without DEGs: {cond}")
                continue

            pred_vec = pred_mean[:, deg_idx].reshape(-1)
            true_vec = true_mean[:, deg_idx].reshape(-1)
            ctrl_vec = ctrl_mean_all[:, deg_idx].reshape(-1)
            denom = mse(true_vec, ctrl_vec)
            nmse_val = float(mse(true_vec, pred_vec) / denom) if denom > 0 else np.nan
            pearson_val = float(pearsonr(true_vec - ctrl_vec, pred_vec - ctrl_vec)[0])
            if not np.isfinite(nmse_val) or not np.isfinite(pearson_val):
                print(f"[eval] non-finite metrics: {cond} nmse={nmse_val} pearson={pearson_val}")

            systema_metrics = pearson_delta_reference_metrics(
                X_true=true_mean.reshape(-1),
                X_pred=pred_mean.reshape(-1),
                reference=pert_reference,
                top20_de_idxs=deg_idx,
            )
            scpram_metrics = compute_scpram_metrics_from_arrays(
                X_true=true_expr,
                X_pred=x_pred,
                deg_idx=deg_idx,
                n_degs=100,
                sample_ratio=0.8,
                times=100,
            )

            results.append(
                {
                    "condition": cond,
                    "nmse": nmse_val,
                    "pearson": pearson_val,
                    "systema_corr_all_allpert": float(systema_metrics["corr_all_allpert"]),
                    "systema_corr_20de_allpert": float(systema_metrics["corr_20de_allpert"]),
                    **scpram_metrics,
                    "split_id": split_id,
                    "n_ensemble": n_ensemble,
                }
            )

        print(f"[eval] done: n_results={len(results)}")
        return pd.DataFrame(results)

    def export_predictions(
        self,
        split_dict: dict,
        emb_table: torch.Tensor,
        split_id: int,
        n_ensemble: int = 300,
        base_seed: int = 24,
        out_path: str | None = None,
    ) -> dict:
        """Export per-condition predictions for downstream plotting.

        Returns:
            Dict mapping condition -> prediction arrays and DE gene info.
        """
        if self.net is None:
            raise ValueError("model_init must be called before export")
        print(
            f"[export] start: split_id={split_id}, n_ensemble={n_ensemble}, n_conds={len(split_dict['test_conds'])}"
        )

        emb_table = emb_table.to(self.device)
        self.net.eval()
        eval_seed = base_seed + split_id
        rng = np.random.RandomState(eval_seed)
        conds = split_dict["test_conds"]
        results = {}

        train_ctrl_adata, _ = self._get_ctrl_pool_from_split(split_dict.get("train"))
        X_ctrl = train_ctrl_adata.X
        X_all = self.data.adata_all.X
        cond_series = self.data.adata_all.obs[self.data.label_key].astype(str).values

        gene_names = self._get_gene_names()

        for cond in conds:
            cond_mask = cond_series == cond
            if not np.any(cond_mask):
                continue
            embd_idx_list = self._get_cond_embd_idx(cond_mask)
            ctrl_expr = self._sample_ctrl_expr(X_ctrl, n_ensemble, rng)
            true_expr = self._select_dense(X_all, cond_mask)

            cond_vec = self._build_cond_vec(emb_table, embd_idx_list, n_ensemble)
            x_pred = self._predict_expr_from_ctrl(ctrl_expr, cond_vec)

            deg_idx = self.data.adata_all.uns["top20_degs_final"].get(
                cond, np.array([], dtype=int)
            )
            if deg_idx is None:
                deg_idx = np.array([], dtype=int)
            deg_idx = np.asarray(deg_idx, dtype=int)
            deg_names = gene_names[deg_idx] if deg_idx.size > 0 else np.array([], dtype=gene_names.dtype)

            results[cond] = {
                "Pred": x_pred[:, deg_idx] if deg_idx.size > 0 else x_pred[:, :0],
                "Ctrl": ctrl_expr[:, deg_idx] if deg_idx.size > 0 else ctrl_expr[:, :0],
                "Truth": true_expr[:, deg_idx] if deg_idx.size > 0 else true_expr[:, :0],
                "DE_idx": deg_idx,
                "DE_name": deg_names,
            }

        if out_path:
            with open(out_path, "wb") as f:
                pickle.dump(results, f)
            print(f"[export] saved: {out_path}")
        print(f"[export] done: n_results={len(results)}")
        return results
