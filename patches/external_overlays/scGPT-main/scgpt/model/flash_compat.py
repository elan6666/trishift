from __future__ import annotations

import warnings

from torch import nn


flash_attn_available = False

try:
    from flash_attn.flash_attention import FlashMHA  # type: ignore

    flash_attn_available = True
except ImportError:
    try:
        from flash_attn.modules.mha import MHA as _FlashAttn2MHA  # type: ignore

        class FlashMHA(nn.Module):
            """Compatibility wrapper for scGPT's flash-attn v1-style API."""

            def __init__(
                self,
                embed_dim,
                num_heads,
                batch_first=True,
                attention_dropout=0.0,
                **factory_kwargs,
            ):
                super().__init__()
                self.batch_first = batch_first
                self.mha = _FlashAttn2MHA(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    dropout=attention_dropout,
                    use_flash_attn=True,
                    device=factory_kwargs.get("device"),
                    dtype=factory_kwargs.get("dtype"),
                )

            def forward(self, src, key_padding_mask=None, **kwargs):
                if key_padding_mask is not None and not key_padding_mask.all().item():
                    raise ValueError(
                        "flash-attn 2.x compatibility wrapper only supports unpadded "
                        "inputs. Set use_fast_transformer=False for padded batches."
                    )
                if not self.batch_first:
                    src = src.transpose(0, 1)
                out = self.mha(src, **kwargs)
                if not self.batch_first:
                    out = out.transpose(0, 1)
                return out, None

        flash_attn_available = True
    except ImportError:
        FlashMHA = None  # type: ignore
        warnings.warn("flash_attn is not installed")
