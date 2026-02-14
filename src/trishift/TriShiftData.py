import hashlib
import numpy as np
import pandas as pd
import torch
import anndata as ad
import scanpy as sc

from trishift import _utils


class TriShiftData:
    def __init__(
        self,
        adata: ad.AnnData,
        embd_df: pd.DataFrame,
        label_key: str = "condition",
        var_gene_key: str = "gene_name",
    ):
        """Initialize data containers and condition metadata."""
        self.adata_all = adata
        self.embd_df = embd_df
        self.label_key = label_key
        self.var_gene_key = var_gene_key
        self.ctrl_label = "ctrl"
        self.gene_to_var_index = {
            g: i for i, g in enumerate(self._get_gene_names(self.adata_all))
        }

        self._refresh_condition_cache()

    def _get_gene_names(self, adata: ad.AnnData) -> np.ndarray:
        if self.var_gene_key in adata.var.columns:
            return adata.var[self.var_gene_key].astype(str).values
        return adata.var_names.astype(str).values

    @staticmethod
    def _compute_nonzero_non_dropout(
        pert_mean: np.ndarray,
        ctrl_mean: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        non_zero = np.where(pert_mean != 0)[0]
        zero = np.where(pert_mean == 0)[0]
        true_zeros = np.intersect1d(zero, np.where(ctrl_mean == 0)[0])
        non_dropouts = np.concatenate((non_zero, true_zeros))
        return non_zero, non_dropouts

    def _refresh_condition_cache(self) -> None:
        """Refresh condition-derived caches after adata changes."""
        cond_series = self.adata_all.obs[self.label_key].astype(str)
        self.ctrl_mask = cond_series == self.ctrl_label
        self.pert_mask = ~self.ctrl_mask
        self.ctrl_indices = np.where(self.ctrl_mask.values)[0]

        self.adata_ctrl = self.adata_all[self.ctrl_mask]
        self.adata_pert = self.adata_all[self.pert_mask]

        self.conditions_all = list(cond_series.unique())
        self.conditions_pert = [c for c in self.conditions_all if c != self.ctrl_label]

        self.cond_to_genes = {}
        self.cond_to_gene_idx = {}
        for cond in self.conditions_all:
            if cond == self.ctrl_label:
                genes = []
            else:
                genes = _utils.normalize_condition(cond).split("+")
            self.cond_to_genes[cond] = genes
            idx_list = [self.gene_to_var_index[g] for g in genes if g in self.gene_to_var_index]
            self.cond_to_gene_idx[cond] = idx_list

    def setup_embedding_index(self, key_name: str = "embd_index") -> None:
        """Map condition genes to embedding indices and filter missing entries.

        This also drops conditions whose genes are missing from the embedding table
        and refreshes cached condition metadata.
        """
        embd_index = {g: i for i, g in enumerate(self.embd_df.index.astype(str))}
        if self.ctrl_label not in embd_index:
            raise ValueError("ctrl index is missing in embedding data")

        missing_conditions = set()
        cond_to_embd_idx = {}
        for cond in self.conditions_all:
            if cond == self.ctrl_label:
                cond_to_embd_idx[cond] = [embd_index[self.ctrl_label]]
                continue
            genes = self.cond_to_genes.get(cond, [])
            idx_list = []
            has_missing = False
            for g in genes:
                if g not in embd_index:
                    has_missing = True
                    break
                idx_list.append(embd_index[g])
            if has_missing:
                missing_conditions.add(cond)
            else:
                cond_to_embd_idx[cond] = idx_list

        if missing_conditions:
            print(
                f"[embd] filtering {len(missing_conditions)} conditions with missing embeddings"
            )
            keep_mask = ~self.adata_all.obs[self.label_key].isin(missing_conditions)
            self.adata_all = self.adata_all[keep_mask].copy()
            self._refresh_condition_cache()

        cond_series = self.adata_all.obs[self.label_key].astype(str)
        self.adata_all.obs[key_name] = [
            cond_to_embd_idx[c] for c in cond_series.values
        ]
        print(f"[embd] embedding index ready: key={key_name}")

    def build_or_load_degs(self, prefer_key: str = "top20_degs_non_dropout") -> None:
        """Populate top20 DE genes per condition and store in uns.

        If precomputed DE genes are available, they are reused and filtered.
        Otherwise, Scanpy is used to compute DE genes and non-dropout indices.
        """
        uns = self.adata_all.uns
        if prefer_key not in uns and "top20_degs" not in uns:
            print("[degs] computing with scanpy")
            var_names_backup = None
            if self.var_gene_key in self.adata_all.var.columns:
                var_names_backup = self.adata_all.var_names.copy()
                self.adata_all.var_names = (
                    self.adata_all.var[self.var_gene_key].astype(str).values
                )
            sc.tl.rank_genes_groups(
                self.adata_all,
                groupby=self.label_key,
                reference=self.ctrl_label,
                rankby_abs=True,
                n_genes=self.adata_all.n_vars,
                method="wilcoxon",
            )
            names_df = pd.DataFrame(self.adata_all.uns["rank_genes_groups"]["names"])
            gene_dict = {g: names_df[g].tolist() for g in names_df.columns}
            self.adata_all.uns["rank_genes_groups"] = gene_dict
            if var_names_backup is not None:
                self.adata_all.var_names = var_names_backup

            cond_series = self.adata_all.obs[self.label_key].astype(str).values
            ctrl_mask = cond_series == self.ctrl_label
            ctrl_mean = np.asarray(self.adata_all[ctrl_mask].X.mean(axis=0)).ravel()
            gene_names = self._get_gene_names(self.adata_all)
            gene_id2idx = {g: i for i, g in enumerate(gene_names)}

            non_zeros_gene_idx = {}
            non_dropout_gene_idx = {}
            top_non_dropout_de_20 = {}
            top_non_zero_de_20 = {}
            top_de_20 = {}

            for cond in self.conditions_pert:
                pert_mask = cond_series == cond
                if not np.any(pert_mask):
                    continue
                pert_mean = np.asarray(self.adata_all[pert_mask].X.mean(axis=0)).ravel()
                non_zero, non_dropouts = self._compute_nonzero_non_dropout(
                    pert_mean,
                    ctrl_mean,
                )

                rank_genes = self.adata_all.uns["rank_genes_groups"].get(cond, [])
                gene_idx_top = [gene_id2idx[g] for g in rank_genes if g in gene_id2idx]

                de_20 = np.array(gene_idx_top[:20], dtype=int)
                non_dropout_20 = np.array(
                    [i for i in gene_idx_top if i in non_dropouts][:20], dtype=int
                )
                non_zero_20 = np.array(
                    [i for i in gene_idx_top if i in non_zero][:20], dtype=int
                )

                non_zeros_gene_idx[cond] = non_zero
                non_dropout_gene_idx[cond] = non_dropouts
                top_non_dropout_de_20[cond] = non_dropout_20
                top_non_zero_de_20[cond] = non_zero_20
                top_de_20[cond] = de_20

            self.adata_all.uns["top20_degs"] = top_de_20
            self.adata_all.uns["top20_degs_non_zero"] = top_non_zero_de_20
            self.adata_all.uns["top20_degs_non_dropout"] = top_non_dropout_de_20
            self.adata_all.uns["gene_idx_non_dropout"] = non_dropout_gene_idx
            self.adata_all.uns["gene_idx_non_zeros"] = non_zeros_gene_idx
            print("[degs] scanpy degs ready")

        if prefer_key in uns:
            degs_src = uns[prefer_key]
        elif "top20_degs" in uns:
            degs_src = uns["top20_degs"]
        else:
            degs_src = {}

        if degs_src:
            print(f"[degs] using precomputed degs: key={prefer_key}")
            degs_final = {}
            for cond in self.conditions_pert:
                if cond not in degs_src:
                    continue
                degs = np.array(degs_src[cond], dtype=int)
                remove_idx = np.array(self.cond_to_gene_idx.get(cond, []), dtype=int)
                if remove_idx.size > 0:
                    degs = np.setdiff1d(degs, remove_idx)
                degs_final[cond] = degs
            self.adata_all.uns["top20_degs_final"] = degs_final
            if "gene_idx_non_dropout" not in uns or "gene_idx_non_zeros" not in uns:
                cond_series = self.adata_all.obs[self.label_key].astype(str).values
                ctrl_mask = cond_series == self.ctrl_label
                ctrl_mean = np.asarray(self.adata_all[ctrl_mask].X.mean(axis=0)).ravel()
                non_zeros_gene_idx = {} if "gene_idx_non_zeros" not in uns else uns["gene_idx_non_zeros"]
                non_dropout_gene_idx = (
                    {} if "gene_idx_non_dropout" not in uns else uns["gene_idx_non_dropout"]
                )
                for cond in self.conditions_pert:
                    if cond in non_zeros_gene_idx and cond in non_dropout_gene_idx:
                        continue
                    pert_mask = cond_series == cond
                    if not np.any(pert_mask):
                        continue
                    pert_mean = np.asarray(self.adata_all[pert_mask].X.mean(axis=0)).ravel()
                    non_zero, non_dropouts = self._compute_nonzero_non_dropout(
                        pert_mean,
                        ctrl_mean,
                    )
                    if cond not in non_zeros_gene_idx:
                        non_zeros_gene_idx[cond] = non_zero
                    if cond not in non_dropout_gene_idx:
                        non_dropout_gene_idx[cond] = non_dropouts
                if "gene_idx_non_zeros" not in uns:
                    self.adata_all.uns["gene_idx_non_zeros"] = non_zeros_gene_idx
                if "gene_idx_non_dropout" not in uns:
                    self.adata_all.uns["gene_idx_non_dropout"] = non_dropout_gene_idx
            return

        raise ValueError("DE genes missing after scanpy computation")

    def _split_train_val(
        self,
        adata: ad.AnnData,
        val_conds_include: list[str] | None,
        val_ratio: float,
        seed: int,
        disjoint_ctrl: bool = False,
    ) -> tuple[list[str], ad.AnnData, list[str], ad.AnnData]:
        """Train/val split by condition.

        When disjoint_ctrl=True, ctrl cells are partitioned between train and val
        (no ctrl overlap). Non-ctrl cells are still split by condition.
        """
        all_conds = adata.obs[self.label_key].astype(str).unique().tolist()
        if self.ctrl_label in all_conds:
            all_conds.remove(self.ctrl_label)

        if val_conds_include is None:
            conds = np.array(all_conds, dtype=object)
            # Match Scouter split behavior (MT19937 + shuffle + round).
            rng = np.random.RandomState(seed)
            rng.shuffle(conds)
            n_val = round(val_ratio * len(conds))
            val_conds_arr, train_conds_arr = np.split(conds, [n_val])
            train_conds = list(train_conds_arr) + [self.ctrl_label]
            val_conds = list(val_conds_arr) + [self.ctrl_label]
        else:
            val_conds = list(val_conds_include) + [self.ctrl_label]
            train_conds = list(np.setdiff1d(all_conds, val_conds)) + [self.ctrl_label]

        cond_series = adata.obs[self.label_key].astype(str)
        train_mask = cond_series.isin(train_conds)
        val_mask = cond_series.isin(val_conds)
        if disjoint_ctrl:
            ctrl_mask = cond_series == self.ctrl_label
            ctrl_idx = np.where(ctrl_mask.values)[0]
            train_non_ctrl = cond_series.isin([c for c in train_conds if c != self.ctrl_label]).values
            val_non_ctrl = cond_series.isin([c for c in val_conds if c != self.ctrl_label]).values

            train_ctrl_mask = np.zeros(adata.n_obs, dtype=bool)
            val_ctrl_mask = np.zeros(adata.n_obs, dtype=bool)
            if ctrl_idx.size > 1:
                rng = np.random.RandomState(seed + 10007)
                perm = rng.permutation(ctrl_idx)
                n_val_ctrl = int(round(val_ratio * ctrl_idx.size))
                n_val_ctrl = min(max(n_val_ctrl, 1), ctrl_idx.size - 1)
                val_ctrl_idx = perm[:n_val_ctrl]
                train_ctrl_idx = perm[n_val_ctrl:]
                val_ctrl_mask[val_ctrl_idx] = True
                train_ctrl_mask[train_ctrl_idx] = True
            elif ctrl_idx.size == 1:
                train_ctrl_mask[ctrl_idx[0]] = True

            train_mask = train_non_ctrl | train_ctrl_mask
            val_mask = val_non_ctrl | val_ctrl_mask

        train_adata = adata[train_mask]
        val_adata = adata[val_mask]
        return train_conds, train_adata, val_conds, val_adata

    def split_by_condition(
        self,
        seed: int,
        test_ratio: float = 0.2,
        val_ratio: float = 0.1,
        test_conds: list[str] | None = None,
        val_conds: list[str] | None = None,
        if_test: bool = True,
    ) -> dict:
        """Split train/val/test by condition.

        Follows Scouter split behavior where ctrl can overlap across splits.
        """
        if if_test:
            _, train_val_adata, test_conds_out, test_adata = self._split_train_val(
                adata=self.adata_all,
                val_conds_include=test_conds,
                val_ratio=test_ratio,
                seed=seed,
            )
            train_conds_out, train_adata, val_conds_out, val_adata = self._split_train_val(
                adata=train_val_adata,
                val_conds_include=val_conds,
                val_ratio=val_ratio,
                seed=seed,
                disjoint_ctrl=False,
            )
        else:
            train_conds_out, train_adata, val_conds_out, val_adata = self._split_train_val(
                adata=self.adata_all,
                val_conds_include=val_conds,
                val_ratio=val_ratio,
                seed=seed,
                disjoint_ctrl=False,
            )
            test_conds_out = [self.ctrl_label]
            test_adata = self.adata_all[
                self.adata_all.obs[self.label_key].astype(str).isin(test_conds_out)
            ]

        return {
            "train": train_adata,
            "val": val_adata,
            "test": test_adata,
            "train_conds": [c for c in train_conds_out if c != self.ctrl_label],
            "val_conds": [c for c in val_conds_out if c != self.ctrl_label],
            "test_conds": [c for c in test_conds_out if c != self.ctrl_label],
        }

    def set_latent_mu(self, z_mu_all: np.ndarray, key: str = "z_mu") -> None:
        """Attach latent mean vectors to adata_all.obsm under key."""
        if z_mu_all.shape[0] != self.adata_all.n_obs:
            raise ValueError("z_mu_all row count must match number of cells")
        self.adata_all.obsm[key] = z_mu_all

    def build_or_load_topk_map(
        self,
        split_adata: ad.AnnData,
        mode: str,
        k: int,
        seed: int,
        candidates: int = 100,
        cache_path: str | None = None,
        return_weights: bool = False,
        per_condition_ot: bool = False,
        reuse_ot_cache: bool = False,
        cache_key: str | None = None,
        ctrl_global_indices: np.ndarray | None = None,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Build or load top-k control indices for pert cells in split_adata.

        Supports knn, ot (Sinkhorn), knn_ot, and soft_ot modes.
        """
        if mode not in {"knn", "ot", "knn_ot", "soft_ot"}:
            raise ValueError("mode must be one of: knn, ot, knn_ot, soft_ot")
        # Reuse KNN cache by default; OT-family cache reuse is opt-in.
        reuse_topk_cache = mode == "knn" or (
            bool(reuse_ot_cache) and mode in {"ot", "soft_ot", "knn_ot"}
        )
        cache_path_eff = cache_path if reuse_topk_cache else None
        if cache_path is not None and cache_path_eff is None:
            print(
                f"[topk] cache disabled for mode={mode} "
                f"(reuse_ot_cache={bool(reuse_ot_cache)}); recomputing"
            )

        print(
            f"[topk] start: mode={mode}, k={k}, seed={seed}, "
            f"candidates={candidates}, per_condition_ot={per_condition_ot}"
        )
        split_cond = split_adata.obs[self.label_key].astype(str).values
        expected_n_pert = int(np.sum(split_cond != self.ctrl_label))
        split_pert_names = split_adata.obs_names[split_cond != self.ctrl_label].astype(str).tolist()
        split_sig = hashlib.sha1("||".join(split_pert_names).encode("utf-8")).hexdigest()
        if ctrl_global_indices is None:
            ctrl_global_idx = np.asarray(self.ctrl_indices, dtype=int)
        else:
            ctrl_global_idx = np.asarray(ctrl_global_indices, dtype=int).reshape(-1)
        if ctrl_global_idx.size == 0:
            raise ValueError("ctrl_global_indices is empty; cannot build top-k map")
        ctrl_sig = hashlib.sha1(
            "||".join(map(str, ctrl_global_idx.tolist())).encode("utf-8")
        ).hexdigest()
        if cache_path_eff is not None:
            try:
                cached = np.load(cache_path_eff, allow_pickle=True)
                if "topk_map" in cached:
                    cached_topk = cached["topk_map"]
                    cache_ok = (
                        cached_topk.ndim == 2
                        and cached_topk.shape[0] == expected_n_pert
                        and cached_topk.shape[1] == k
                    )
                    if not cache_ok:
                        print(
                            "[topk] cache shape mismatch; rebuilding: "
                            f"path={cache_path_eff}, cached={cached_topk.shape}, expected=({expected_n_pert}, {k})"
                        )
                    elif "mode" in cached and str(cached["mode"]) != str(mode):
                        print(
                            "[topk] cache matching mode mismatch; rebuilding: "
                            f"path={cache_path_eff}, cached_mode={str(cached['mode'])}, expected={mode}"
                        )
                    elif "split_sig" in cached and str(cached["split_sig"]) != split_sig:
                        print(
                            "[topk] cache split signature mismatch; rebuilding: "
                            f"path={cache_path_eff}"
                        )
                    elif (
                        ("ctrl_sig" in cached and str(cached["ctrl_sig"]) != ctrl_sig)
                        or ("ctrl_sig" not in cached)
                    ):
                        print(
                            "[topk] cache control-pool signature mismatch; rebuilding: "
                            f"path={cache_path_eff}"
                        )
                    elif (
                        ("per_condition_ot" in cached and bool(cached["per_condition_ot"]) != bool(per_condition_ot))
                        or ("per_condition_ot" not in cached and bool(per_condition_ot))
                    ):
                        print(
                            "[topk] cache OT mode mismatch; rebuilding: "
                            f"path={cache_path_eff}, cached_per_condition_ot="
                            f"{bool(cached['per_condition_ot']) if 'per_condition_ot' in cached else False}, "
                            f"expected={bool(per_condition_ot)}"
                        )
                    elif cache_key is not None and (
                        "cache_key" not in cached or str(cached["cache_key"]) != str(cache_key)
                    ):
                        print(
                            "[topk] cache key mismatch; rebuilding: "
                            f"path={cache_path_eff}"
                        )
                    elif not return_weights:
                        print(f"[topk] loaded cache: {cache_path_eff}")
                        return cached_topk
                    elif "topk_weights" in cached:
                        cached_w = cached["topk_weights"]
                        if cached_w.ndim == 2 and cached_w.shape == cached_topk.shape:
                            print(f"[topk] loaded cache: {cache_path_eff}")
                            return cached_topk, cached_w
                        print(
                            "[topk] cache weight shape mismatch; rebuilding: "
                            f"path={cache_path_eff}, weights={cached_w.shape}, topk={cached_topk.shape}"
                        )
            except FileNotFoundError:
                pass

        if "z_mu" not in self.adata_all.obsm:
            raise ValueError("z_mu is missing; call set_latent_mu first")

        z_mu = np.asarray(self.adata_all.obsm["z_mu"])

        all_idx_map = {n: i for i, n in enumerate(self.adata_all.obs_names)}
        split_global_idx = np.array([all_idx_map[n] for n in split_adata.obs_names])
        split_pert_mask = split_cond != self.ctrl_label
        pert_global_idx = split_global_idx[split_pert_mask]
        if pert_global_idx.size == 0:
            return np.empty((0, k), dtype=int)

        torch.manual_seed(seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        z_mu_t = torch.as_tensor(z_mu, dtype=torch.float32, device=device)
        ctrl_idx_t = torch.as_tensor(ctrl_global_idx, dtype=torch.long, device=device)
        pert_idx_t = torch.as_tensor(pert_global_idx, dtype=torch.long, device=device)
        z_ctrl = torch.index_select(z_mu_t, 0, ctrl_idx_t)
        z_pert = torch.index_select(z_mu_t, 0, pert_idx_t)
        n_ctrl = z_ctrl.shape[0]
        n_pert = z_pert.shape[0]
        topk_weights = None

        if mode == "knn":
            print("[topk] computing knn")
            dist = torch.cdist(z_pert, z_ctrl, p=2)
            k_eff = min(k, dist.shape[1])
            _, idx = torch.topk(dist, k_eff, dim=1, largest=False)
            if k_eff < k:
                pad = idx[:, -1:].repeat(1, k - k_eff)
                idx = torch.cat([idx, pad], dim=1)
            topk_map = idx.cpu().numpy().astype(int)

        elif mode in {"ot", "soft_ot"}:
            print("[topk] computing ot")
            reg = 0.05
            need_weights = bool(return_weights and mode in {"ot", "soft_ot"})

            def _compute_ot_topk(
                z_ctrl_local: torch.Tensor,
                z_pert_local: torch.Tensor,
                reg_local: float,
                k_local: int,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                n_ctrl_local = z_ctrl_local.shape[0]
                n_pert_local = z_pert_local.shape[0]
                a_local = torch.full(
                    (n_ctrl_local,),
                    1.0 / max(n_ctrl_local, 1),
                    device=z_ctrl_local.device,
                )
                b_local = torch.full(
                    (n_pert_local,),
                    1.0 / max(n_pert_local, 1),
                    device=z_pert_local.device,
                )
                u_local = torch.ones_like(a_local)
                v_local = torch.ones_like(b_local)
                cost_local = torch.cdist(z_ctrl_local, z_pert_local, p=2)
                K_local = torch.exp(-cost_local / reg_local)
                for _ in range(200):
                    Kv_local = K_local @ v_local
                    Kv_local = torch.clamp(Kv_local, min=1e-12)
                    u_local = a_local / Kv_local
                    KTu_local = K_local.T @ u_local
                    KTu_local = torch.clamp(KTu_local, min=1e-12)
                    v_local = b_local / KTu_local
                coupling_local = (u_local[:, None] * K_local) * v_local[None, :]
                k_eff_local = min(k_local, coupling_local.shape[0])
                vals_local, idx_local = torch.topk(
                    coupling_local.T, k_eff_local, dim=1, largest=True
                )
                return vals_local, idx_local

            def _compute_with_fallback(
                z_ctrl_local: torch.Tensor,
                z_pert_local: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                try:
                    return _compute_ot_topk(z_ctrl_local, z_pert_local, reg, k)
                except RuntimeError as exc:
                    oom = "out of memory" in str(exc).lower()
                    if z_ctrl_local.device.type != "cuda" or not oom:
                        raise
                    print("[topk] cuda oom in ot; falling back to cpu")
                    torch.cuda.empty_cache()
                    return _compute_ot_topk(
                        z_ctrl_local.detach().cpu(),
                        z_pert_local.detach().cpu(),
                        reg,
                        k,
                    )

            if per_condition_ot:
                split_pert_cond = split_cond[split_pert_mask]
                unique_conds = pd.unique(split_pert_cond)
                topk_map = np.zeros((n_pert, k), dtype=int)
                if need_weights:
                    topk_weights = np.zeros((n_pert, k), dtype=np.float32)

                for cond in unique_conds:
                    cond_local_idx = np.where(split_pert_cond == cond)[0]
                    if cond_local_idx.size == 0:
                        continue
                    cond_t = torch.as_tensor(cond_local_idx, dtype=torch.long, device=z_pert.device)
                    z_pert_cond = torch.index_select(z_pert, 0, cond_t)
                    vals_cond, idx_cond = _compute_with_fallback(z_ctrl, z_pert_cond)

                    k_eff = idx_cond.shape[1]
                    if k_eff < k:
                        pad_idx = idx_cond[:, -1:].repeat(1, k - k_eff)
                        idx_cond = torch.cat([idx_cond, pad_idx], dim=1)
                        vals_cond = torch.cat([vals_cond, vals_cond[:, -1:].repeat(1, k - k_eff)], dim=1)

                    idx_np = idx_cond.cpu().numpy().astype(int)
                    topk_map[cond_local_idx] = idx_np
                    if need_weights:
                        weight_sum = vals_cond.sum(dim=1, keepdim=True).clamp_min(1e-12)
                        w_cond = (vals_cond / weight_sum).cpu().numpy().astype(np.float32)
                        topk_weights[cond_local_idx] = w_cond
            else:
                vals, idx = _compute_with_fallback(z_ctrl, z_pert)
                k_eff = idx.shape[1]
                if k_eff < k:
                    pad = idx[:, -1:].repeat(1, k - k_eff)
                    idx = torch.cat([idx, pad], dim=1)
                    vals = torch.cat([vals, vals[:, -1:].repeat(1, k - k_eff)], dim=1)
                topk_map = idx.cpu().numpy().astype(int)
                if need_weights:
                    weight_sum = vals.sum(dim=1, keepdim=True).clamp_min(1e-12)
                    weights = vals / weight_sum
                    topk_weights = weights.cpu().numpy().astype(np.float32)

        else:
            print("[topk] computing knn_ot")
            reg = 0.05
            dist = torch.cdist(z_pert, z_ctrl, p=2)
            k_prime = min(candidates, dist.shape[1])
            cand_vals, cand_idx = torch.topk(dist, k_prime, dim=1, largest=False)
            weights = torch.softmax(-cand_vals / reg, dim=1)
            k_eff = min(k, k_prime)
            _, idx_local = torch.topk(weights, k_eff, dim=1, largest=True)
            if k_eff < k:
                pad = idx_local[:, -1:].repeat(1, k - k_eff)
                idx_local = torch.cat([idx_local, pad], dim=1)
            topk_idx = torch.gather(cand_idx, 1, idx_local)
            topk_map = topk_idx.cpu().numpy().astype(int)

        if cache_path_eff is not None:
            save_kwargs = dict(
                topk_map=topk_map,
                mode=mode,
                k=k,
                seed=seed,
                candidates=candidates,
                split_sig=split_sig,
                ctrl_sig=ctrl_sig,
                per_condition_ot=bool(per_condition_ot),
            )
            if cache_key is not None:
                save_kwargs["cache_key"] = str(cache_key)
            if topk_weights is not None:
                save_kwargs["topk_weights"] = topk_weights
            np.savez(cache_path_eff, **save_kwargs)
            print(f"[topk] saved cache: {cache_path_eff}")
        if mode in {"ot", "soft_ot"} and return_weights:
            return topk_map, topk_weights
        return topk_map
