import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions import kl_divergence as kl


def _resolve_activation_layer(activation: str) -> type[nn.Module]:
    if activation == "gelu":
        return nn.GELU
    if activation == "relu":
        return nn.ReLU
    if activation == "silu":
        return nn.SiLU
    if activation == "selu":
        return nn.SELU
    raise ValueError("activation must be one of: gelu, relu, silu, selu")


def _zero_cond_vec(emb_table: torch.Tensor) -> torch.Tensor:
    return torch.zeros(emb_table.size(1), device=emb_table.device, dtype=emb_table.dtype)


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dims: list[int],
        out_dim: int,
        dropout: float = 0.0,
        activation: str = "gelu",
        use_layernorm: bool = False,
        use_batchnorm: bool = False,
        use_alpha_dropout: bool = False,
    ):
        super().__init__()
        act_layer = _resolve_activation_layer(activation)

        layers: list[nn.Module] = []
        prev_dim = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h))
            if use_layernorm:
                layers.append(nn.LayerNorm(h))
            layers.append(act_layer())
            if dropout > 0:
                if use_alpha_dropout:
                    layers.append(nn.AlphaDropout(dropout))
                else:
                    layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VAE(nn.Module):
    def __init__(
        self,
        x_dim: int,
        z_dim: int,
        hidden_dim: int = 1000,
        noise_rate: float = 0.1,
        kl_weight: float = 5e-4,
    ):
        """scPRAM-style denoising VAE.

        input_dim is dynamic via x_dim from current dataset.
        """
        super().__init__()
        self.input_dim = int(x_dim)
        self.hidden_dim = int(hidden_dim)
        self.latent_dim = int(z_dim)
        self.noise_rate = float(noise_rate)
        self.kl_weight = float(kl_weight)

        # Match scPRAM architecture as closely as possible.
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.latent_dim * 2),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.input_dim),
            nn.ReLU(),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        mu, logvar = torch.chunk(h, 2, dim=1)
        z = self.reparam(mu, logvar)
        return z, mu, logvar

    def reparam(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        noise = torch.randn_like(x)
        x_noisy = x + noise * self.noise_rate
        z, mu, logvar = self.encode(x_noisy)
        x_recon = self.decode(z)

        std = torch.exp(logvar / 2)
        loss_kl = kl(Normal(mu, std), Normal(0, 1)).sum(dim=1)
        loss_rec = ((x - x_recon) ** 2).sum(dim=1)
        return x_recon, loss_rec, loss_kl

    def encode_mu(self, x: torch.Tensor) -> torch.Tensor:
        # Keep deterministic latent for downstream matching/eval.
        _, mu, _ = self.encode(x)
        return mu

    def get_loss(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, loss_rec, loss_kl = self.forward(x)
        return loss_rec, loss_kl


def aggregate_cond_embedding(
    emb_table: torch.Tensor,
    idx_list: list[int] | torch.Tensor,
    mode: str = "sum",
    normalize: bool = False,
    eps: float = 1e-12,
) -> torch.Tensor:
    if isinstance(idx_list, list):
        if len(idx_list) == 0:
            return _zero_cond_vec(emb_table)
        idx = torch.tensor(idx_list, device=emb_table.device, dtype=torch.long)
    else:
        idx = idx_list.to(device=emb_table.device, dtype=torch.long).view(-1)
        if idx.numel() == 0:
            return _zero_cond_vec(emb_table)

    selected = emb_table.index_select(0, idx)
    if mode == "sum":
        out = selected.sum(dim=0)
    elif mode == "mean":
        out = selected.mean(dim=0)
    else:
        raise ValueError("mode must be one of: sum, mean")
    if normalize:
        out = F.normalize(out, p=2, dim=0, eps=eps)
    return out


class ShiftNet(nn.Module):
    def __init__(
        self,
        z_dim: int,
        cond_dim: int,
        hidden: list[int],
        dropout: float = 0.0,
        predict_delta: bool = True,
        use_cross_attention: bool = False,
        cross_attn_heads: int = 4,
        cross_attn_dropout: float = 0.0,
        use_transformer_block: bool = False,
        transformer_layers: int = 1,
        transformer_ff_mult: int = 4,
        transformer_dropout: float = 0.0,
        transformer_readout: str = "first",
        delta_out_dim: int | None = None,
    ):
        super().__init__()
        self.predict_delta = bool(predict_delta)
        self.model_dim = int(z_dim)
        self.cond_dim = int(cond_dim)
        self.delta_out_dim = (
            int(delta_out_dim) if delta_out_dim is not None else int(self.model_dim)
        )
        self.use_transformer_block = bool(use_transformer_block)
        self.use_cross_attention = bool(use_cross_attention)
        if self.use_transformer_block and self.use_cross_attention:
            raise ValueError("use_transformer_block and use_cross_attention cannot both be true")
        self.transformer_readout = str(transformer_readout)
        if self.transformer_readout not in {"first", "mean", "concat"}:
            raise ValueError("transformer_readout must be one of: first, mean, concat")
        transformer_fused_dim = (
            self.model_dim * 2 if self.transformer_readout == "concat" else self.model_dim
        )
        in_dim = self.model_dim + self.cond_dim
        self.cond_proj: nn.Module = nn.Identity()
        if self.use_transformer_block:
            if cross_attn_heads <= 0 or self.model_dim % cross_attn_heads != 0:
                raise ValueError("transformer heads must be >0 and divide model_dim")
            if transformer_layers <= 0:
                raise ValueError("transformer_layers must be > 0")
            if transformer_ff_mult <= 0:
                raise ValueError("transformer_ff_mult must be > 0")
            self.cond_proj = self._build_cond_proj()
            enc_layer = nn.TransformerEncoderLayer(
                d_model=self.model_dim,
                nhead=cross_attn_heads,
                dim_feedforward=int(self.model_dim * transformer_ff_mult),
                dropout=transformer_dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(enc_layer, num_layers=transformer_layers)
            in_dim += transformer_fused_dim
        elif self.use_cross_attention:
            if cross_attn_heads <= 0 or self.model_dim % cross_attn_heads != 0:
                raise ValueError("cross_attn_heads must be >0 and divide model_dim")
            self.cond_proj = self._build_cond_proj()
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=self.model_dim,
                num_heads=cross_attn_heads,
                dropout=cross_attn_dropout,
                batch_first=True,
            )
            in_dim += self.model_dim
        self.context_fallback = None
        if self.predict_delta:
            self.output_dim = self.delta_out_dim
        elif self.use_transformer_block:
            self.output_dim = transformer_fused_dim
        else:
            self.output_dim = self.model_dim
        if self.predict_delta:
            self.net = MLP(
                in_dim=in_dim,
                hidden_dims=hidden,
                out_dim=self.delta_out_dim,
                dropout=dropout,
                activation="selu",
                use_batchnorm=True,
                use_alpha_dropout=True,
            )
        else:
            self.net = None
            if not self.use_transformer_block and not self.use_cross_attention:
                self.context_fallback = MLP(
                    in_dim=self.model_dim + self.cond_dim,
                    hidden_dims=hidden,
                    out_dim=self.model_dim,
                    dropout=dropout,
                    activation="selu",
                    use_batchnorm=True,
                    use_alpha_dropout=True,
                )

    def _build_cond_proj(self) -> nn.Module:
        if self.cond_dim == self.model_dim:
            return nn.Identity()
        return nn.Linear(self.cond_dim, self.model_dim)

    def _transformer_tokens_to_repr(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.transformer_readout == "first":
            return tokens[:, 0, :]
        if self.transformer_readout == "mean":
            return tokens.mean(dim=1)
        if self.transformer_readout == "concat":
            return torch.cat([tokens[:, 0, :], tokens[:, 1, :]], dim=1)
        raise RuntimeError(f"unsupported transformer_readout: {self.transformer_readout}")

    def _forward_with_transformer(self, z_ctrl_mu: torch.Tensor, cond_vec: torch.Tensor) -> torch.Tensor:
        cond_token = self.cond_proj(cond_vec)
        tokens = torch.stack([z_ctrl_mu, cond_token], dim=1)
        tokens = self.transformer(tokens)
        return self._transformer_tokens_to_repr(tokens)

    def _forward_with_cross_attention(
        self,
        z_ctrl_mu: torch.Tensor,
        cond_vec: torch.Tensor,
    ) -> torch.Tensor:
        cond_token = self.cond_proj(cond_vec)
        query = cond_token.unsqueeze(1)
        key_value = torch.stack([z_ctrl_mu, cond_token], dim=1)
        fused, _ = self.cross_attn(query, key_value, key_value, need_weights=False)
        return fused.squeeze(1)

    def _forward_mlp_only(self, z_ctrl_mu: torch.Tensor, cond_vec: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z_ctrl_mu, cond_vec], dim=1)
        if self.predict_delta:
            return self.net(x)
        if self.context_fallback is None:
            raise RuntimeError("context_fallback is not initialized")
        return self.context_fallback(x)

    def _forward_with_fused(
        self,
        z_ctrl_mu: torch.Tensor,
        cond_vec: torch.Tensor,
        z_fused: torch.Tensor,
    ) -> torch.Tensor:
        if self.predict_delta:
            x = torch.cat([z_ctrl_mu, cond_vec, z_fused], dim=1)
            return self.net(x)
        return z_fused

    def forward(self, z_ctrl_mu: torch.Tensor, cond_vec: torch.Tensor) -> torch.Tensor:
        if self.use_transformer_block:
            return self._forward_with_fused(
                z_ctrl_mu,
                cond_vec,
                self._forward_with_transformer(z_ctrl_mu, cond_vec),
            )
        if self.use_cross_attention:
            return self._forward_with_fused(
                z_ctrl_mu,
                cond_vec,
                self._forward_with_cross_attention(z_ctrl_mu, cond_vec),
            )
        return self._forward_mlp_only(z_ctrl_mu, cond_vec)


class GeneratorNet(nn.Module):
    def __init__(
        self,
        x_dim: int,
        cond_dim: int,
        z_dim: int,
        hidden: list[int] | None,
        shift_dim: int | None = None,
        input_mode: str = "full",
        dropout: float = 0.0,
        encoder_hidden: list[int] | None = None,
        state_dim: int | None = None,
        decoder_hidden: list[int] | None = None,
        use_batchnorm: bool = True,
        use_layernorm: bool = False,
        use_residual_head: bool = False,
    ):
        super().__init__()
        self.z_dim = int(shift_dim) if shift_dim is not None else int(z_dim)
        self.input_mode = str(input_mode)
        if self.input_mode not in {"full", "fusion_only", "state_fusion"}:
            raise ValueError("input_mode must be one of: full, fusion_only, state_fusion")
        self.use_residual_head = bool(use_residual_head)

        if encoder_hidden is None and state_dim is None and decoder_hidden is None:
            hidden = hidden or []
            if hidden:
                state_dim = hidden[0]
                comp_hidden = hidden[1:]
                gen_hidden = hidden[1:]
            else:
                state_dim = x_dim
                comp_hidden = []
                gen_hidden = []
        else:
            if state_dim is None:
                raise ValueError("state_dim must be provided when using encoder_hidden/decoder_hidden")
            comp_hidden = encoder_hidden or []
            gen_hidden = decoder_hidden or []
        self.state_dim = int(state_dim)

        self.compressor = MLP(
            in_dim=x_dim,
            hidden_dims=comp_hidden,
            out_dim=self.state_dim,
            dropout=dropout,
            activation="selu",
            use_batchnorm=use_batchnorm,
            use_layernorm=use_layernorm,
            use_alpha_dropout=True,
        )
        if self.input_mode == "fusion_only":
            gen_in_dim = self.z_dim
        elif self.input_mode == "state_fusion":
            gen_in_dim = self.state_dim + self.z_dim
        else:
            gen_in_dim = self.state_dim + cond_dim + self.z_dim
        self.generator = MLP(
            in_dim=gen_in_dim,
            hidden_dims=gen_hidden,
            out_dim=x_dim,
            dropout=dropout,
            activation="selu",
            use_batchnorm=use_batchnorm,
            use_layernorm=use_layernorm,
            use_alpha_dropout=True,
        )
        # Scouter-style path used by stage3_only:
        # encoder(ctrl) + sum(pert_emb) -> generator (no latent delta input).
        self.generator_no_delta = MLP(
            in_dim=self.state_dim + cond_dim,
            hidden_dims=gen_hidden,
            out_dim=x_dim,
            dropout=dropout,
            activation="selu",
            use_batchnorm=use_batchnorm,
            use_layernorm=use_layernorm,
            use_alpha_dropout=True,
        )

    def _build_generator_input(
        self,
        x_ctrl: torch.Tensor,
        cond_vec: torch.Tensor,
        delta_z: torch.Tensor,
    ) -> torch.Tensor:
        if self.input_mode == "fusion_only":
            return delta_z
        state = self.compressor(x_ctrl)
        if self.input_mode == "state_fusion":
            return torch.cat([state, delta_z], dim=1)
        return torch.cat([state, cond_vec, delta_z], dim=1)

    def forward(
        self, x_ctrl: torch.Tensor, cond_vec: torch.Tensor, delta_z: torch.Tensor
    ) -> torch.Tensor:
        x = self._build_generator_input(x_ctrl, cond_vec, delta_z)
        out = self.generator(x)
        if self.use_residual_head:
            return x_ctrl + out
        return out

    def forward_no_delta(self, x_ctrl: torch.Tensor, cond_vec: torch.Tensor) -> torch.Tensor:
        if self.input_mode != "full":
            raise RuntimeError("forward_no_delta is available only when gen input_mode=full")
        state = self.compressor(x_ctrl)
        x = torch.cat([state, cond_vec], dim=1)
        out = self.generator_no_delta(x)
        if self.use_residual_head:
            return x_ctrl + out
        return out


class TriShiftNet(nn.Module):
    @staticmethod
    def _resolve_gen_state_dim(gen_state_dim: int | str | None, cond_dim: int) -> int | None:
        if isinstance(gen_state_dim, str):
            key = gen_state_dim.strip().lower()
            if key in {"cond_dim", "cond", "cond_vec"}:
                return int(cond_dim)
            try:
                return int(gen_state_dim)
            except ValueError as exc:
                raise ValueError(
                    "gen_state_dim must be int-like or one of: cond_dim, cond, cond_vec"
                ) from exc
        if gen_state_dim is None:
            return None
        return int(gen_state_dim)

    @staticmethod
    def _resolve_shift_model_dim(
        shift_input_source: str,
        resolved_gen_state_dim: int | None,
        gen_encoder_hidden: list[int] | None,
        gen_decoder_hidden: list[int] | None,
        gen_hidden: list[int],
        x_dim: int,
        z_dim: int,
    ) -> int:
        if shift_input_source == "state":
            if resolved_gen_state_dim is not None:
                return resolved_gen_state_dim
            if gen_encoder_hidden is None and gen_decoder_hidden is None:
                return int(gen_hidden[0]) if len(gen_hidden) > 0 else int(x_dim)
            raise ValueError("shift_input_source=state requires gen_state_dim to be set")
        return int(z_dim)

    def __init__(
        self,
        x_dim: int,
        z_dim: int,
        cond_dim: int,
        vae_enc_hidden: list[int] | None,
        vae_dec_hidden: list[int] | None,
        shift_hidden: list[int],
        gen_hidden: list[int],
        vae_hidden_dim: int = 1000,
        vae_noise_rate: float = 0.1,
        vae_kl_weight: float = 5e-4,
        dropout: float = 0.0,
        gen_encoder_hidden: list[int] | None = None,
        gen_state_dim: int | str | None = None,
        gen_decoder_hidden: list[int] | None = None,
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
        gen_input_mode: str = "full",
        gen_use_residual_head: bool = False,
        shift_input_source: str = "latent_mu",
    ):
        super().__init__()
        resolved_gen_state_dim = self._resolve_gen_state_dim(gen_state_dim, cond_dim)

        self.vae = VAE(
            x_dim=x_dim,
            z_dim=z_dim,
            hidden_dim=vae_hidden_dim,
            noise_rate=vae_noise_rate,
            kl_weight=vae_kl_weight,
        )
        self.shift_input_source = str(shift_input_source)
        if self.shift_input_source not in {"latent_mu", "state"}:
            raise ValueError("shift_input_source must be one of: latent_mu, state")
        shift_model_dim = self._resolve_shift_model_dim(
            shift_input_source=self.shift_input_source,
            resolved_gen_state_dim=resolved_gen_state_dim,
            gen_encoder_hidden=gen_encoder_hidden,
            gen_decoder_hidden=gen_decoder_hidden,
            gen_hidden=gen_hidden,
            x_dim=x_dim,
            z_dim=z_dim,
        )
        self.shift = ShiftNet(
            shift_model_dim,
            cond_dim,
            shift_hidden,
            dropout=dropout,
            predict_delta=shift_predict_delta,
            use_cross_attention=shift_use_cross_attention,
            cross_attn_heads=shift_cross_attn_heads,
            cross_attn_dropout=shift_cross_attn_dropout,
            use_transformer_block=shift_use_transformer_block,
            transformer_layers=shift_transformer_layers,
            transformer_ff_mult=shift_transformer_ff_mult,
            transformer_dropout=shift_transformer_dropout,
            transformer_readout=shift_transformer_readout,
            delta_out_dim=int(z_dim) if shift_predict_delta else None,
        )
        self.gen = GeneratorNet(
            x_dim=x_dim,
            cond_dim=cond_dim,
            z_dim=z_dim,
            shift_dim=self.shift.output_dim,
            hidden=gen_hidden,
            input_mode=gen_input_mode,
            dropout=dropout,
            encoder_hidden=gen_encoder_hidden,
            state_dim=resolved_gen_state_dim,
            decoder_hidden=gen_decoder_hidden,
            use_batchnorm=gen_use_batchnorm,
            use_layernorm=gen_use_layernorm,
            use_residual_head=gen_use_residual_head,
        )

    def encode_mu(self, x: torch.Tensor) -> torch.Tensor:
        return self.vae.encode_mu(x)

    def _build_shift_input(
        self,
        z_ctrl_mu: torch.Tensor | None,
        x_ctrl: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.shift_input_source == "latent_mu":
            if z_ctrl_mu is None:
                if x_ctrl is None:
                    raise RuntimeError("x_ctrl is required when z_ctrl_mu is None")
                z_ctrl_mu = self.encode_mu(x_ctrl)
            return z_ctrl_mu
        if x_ctrl is None:
            raise RuntimeError("x_ctrl is required when shift_input_source=state")
        return self.gen.compressor(x_ctrl)

    def predict_shift_repr(
        self,
        z_ctrl_mu: torch.Tensor | None,
        cond_vec: torch.Tensor,
        x_ctrl: torch.Tensor | None = None,
    ) -> torch.Tensor:
        shift_in = self._build_shift_input(z_ctrl_mu, x_ctrl)
        return self.shift(shift_in, cond_vec)

    def predict_delta(
        self,
        z_ctrl_mu: torch.Tensor | None,
        cond_vec: torch.Tensor,
        x_ctrl: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self.shift.predict_delta:
            raise RuntimeError("predict_delta is disabled; use predict_shift_repr instead")
        return self.predict_shift_repr(z_ctrl_mu, cond_vec, x_ctrl=x_ctrl)

    def predict_expr(
        self, x_ctrl: torch.Tensor, cond_vec: torch.Tensor, delta_z: torch.Tensor
    ) -> torch.Tensor:
        return self.gen(x_ctrl, cond_vec, delta_z)

    def forward_joint(
        self,
        x_ctrl: torch.Tensor,
        cond_vec: torch.Tensor,
        x_true: torch.Tensor | None = None,
        z_ctrl_mu: torch.Tensor | None = None,
    ) -> dict:
        if z_ctrl_mu is None and self.shift_input_source == "latent_mu":
            z_ctrl_mu = self.encode_mu(x_ctrl)
        shift_repr = self.predict_shift_repr(z_ctrl_mu, cond_vec, x_ctrl=x_ctrl)
        x_pred = self.predict_expr(x_ctrl, cond_vec, shift_repr)
        out = {"z_ctrl_mu": z_ctrl_mu, "shift_repr": shift_repr, "x_pred": x_pred}
        if self.shift.predict_delta:
            out["delta_z"] = shift_repr
        return out
