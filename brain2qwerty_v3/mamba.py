# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Nemotron-H-style hybrid Mamba-2 / attention sequence stack (Brain2Qwerty V3).

This module provides ``MambaHybrid``, a drop-in replacement for the
``TransformerEncoder`` / ``Conformer`` sequence core used by
``neuraltrain.models.conv_transformer.ConvTransformerModel``: it builds an
``nn.Module`` with a ``forward`` of signature ``(B, T, D) -> (B, T, D)`` from a
pydantic config dict, exactly like the reference configs.

Design (following NVIDIA's Nemotron-H hybrid pattern):
  * the stack is mostly Mamba-2 blocks (selective state-space mixers, SSD
    parametrisation) with one global self-attention block every
    ``attention_every`` blocks (default pattern: M, M, M, A, M, M, M, A);
  * blocks are pre-norm (RMSNorm) residual blocks;
  * a final RMSNorm is applied to the stack output.

The Mamba-2 mixer is implemented in pure PyTorch using the quadratic "dual"
form of the SSD kernel (Dao & Gu, 2024, sec. 4-6), computed in float32 and
chunked over heads to bound memory. It has no dependency on the ``mamba-ssm``
CUDA kernels, so it runs on any GPU/CPU; for long contexts the mixer can later
be swapped for the reference kernels without changing the surrounding code.
"""

import math
import typing as tp

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuraltrain.models.base import BaseModelConfig


# --------------------------------------------------------------------------- #
# Primitives
# --------------------------------------------------------------------------- #
class RMSNorm(nn.Module):
    """Root-mean-square layer norm (no bias, learned gain).

    ``group_size`` splits the feature dim into groups normalised separately
    (e.g. ``headdim`` reproduces the gated per-head norm of the official
    Mamba-2 / HuggingFace implementations; ``None`` normalises over the full
    feature dim).
    """

    def __init__(self, dim: int, eps: float = 1e-6, group_size: int | None = None):
        super().__init__()
        if group_size is not None and dim % group_size != 0:
            raise ValueError(f"{dim=} not divisible by {group_size=}")
        self.eps = eps
        self.group_size = group_size or dim
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        shape = x.shape
        x = x.float().view(*shape[:-1], -1, self.group_size)
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x.view(shape) * self.weight.float()).to(dtype)


def _segsum(x: torch.Tensor) -> torch.Tensor:
    """Stable segment sum: (..., T) -> (..., T, T), lower-triangular.

    ``out[..., t, s] = sum_{u=s+1}^{t} x[..., u]`` for ``s <= t`` and ``-inf``
    elsewhere, so that ``exp(segsum(a))`` is the SSD decay mask.
    """
    T = x.size(-1)
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device), -1)
    x = x.unsqueeze(-1).expand(*x.shape, T)
    x = x.masked_fill(~mask, 0)
    x_segsum = torch.cumsum(x, dim=-2)
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device), 0)
    return x_segsum.masked_fill(~mask, float("-inf"))


# --------------------------------------------------------------------------- #
# Mamba-2 mixer (pure PyTorch SSD, quadratic dual form)
# --------------------------------------------------------------------------- #
class Mamba2Mixer(nn.Module):
    """Single Mamba-2 (SSD) mixer layer.

    Input ``x``: ``(B, T, d_model)``. Internally:
      * ``in_proj`` produces the gate ``z``, the conv branch ``[x, B, C]`` and
        the step-size logits ``dt``;
      * a causal depthwise conv (kernel ``d_conv``) + SiLU smooths ``[x, B, C]``;
      * the SSD recurrence ``h_t = exp(dt_t A) h_{t-1} + dt_t B_t x_t``,
        ``y_t = C_t h_t + D x_t`` is evaluated in its quadratic dual form;
      * the output is gated by ``silu(z)`` and projected back to ``d_model``.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 128,
        headdim: int = 64,
        expand: int = 2,
        d_conv: int = 4,
        ngroups: int = 1,
        dropout: float = 0.0,
        head_chunk: int = 8,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        norm_group_size: int | None = None,
    ):
        super().__init__()
        self.d_inner = expand * d_model
        if self.d_inner % headdim != 0:
            raise ValueError(f"d_inner={self.d_inner} not divisible by {headdim=}")
        self.nheads = self.d_inner // headdim
        self.headdim = headdim
        self.d_state = d_state
        self.ngroups = ngroups
        if self.nheads % ngroups != 0:
            raise ValueError(f"nheads={self.nheads} not divisible by {ngroups=}")
        self.head_chunk = head_chunk

        conv_dim = self.d_inner + 2 * ngroups * d_state  # x, B, C
        self.in_proj = nn.Linear(
            d_model, 2 * self.d_inner + 2 * ngroups * d_state + self.nheads, bias=False
        )
        self.conv1d = nn.Conv1d(
            conv_dim, conv_dim, kernel_size=d_conv, padding=d_conv - 1,
            groups=conv_dim, bias=True,
        )

        # Step-size bias: dt = softplus(raw + bias), initialised in [dt_min, dt_max]
        dt = torch.exp(
            torch.rand(self.nheads) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        # Per-head negative decay and skip connection
        self.A_log = nn.Parameter(torch.log(torch.empty(self.nheads).uniform_(1, 16)))
        self.D = nn.Parameter(torch.ones(self.nheads))

        self.norm = RMSNorm(self.d_inner, group_size=norm_group_size)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def _ssd(
        self,
        x: torch.Tensor,   # (B, T, H, P)
        Bmat: torch.Tensor,  # (B, T, G, N)
        Cmat: torch.Tensor,  # (B, T, G, N)
        dt: torch.Tensor,  # (B, T, H)
    ) -> torch.Tensor:
        """Quadratic dual form of the SSD kernel, chunked over heads.

        Memory per chunk is ``B * head_chunk * T^2`` floats; at the sentence
        lengths produced by the temporal downsampling (T ~ a few hundred
        frames) this is modest. Computed in float32 for stability.
        """
        b, t, h, p = x.shape
        g = Bmat.shape[2]
        A = -torch.exp(self.A_log.float())  # (H,), negative

        # Decay mask L[b, h, t, s] = exp(sum_{u=s+1..t} dt_u * A_h) for s <= t
        # (segsum treats the last dim as time, hence the permute)
        dA = dt.float() * A  # (B, T, H)
        L = torch.exp(_segsum(dA.permute(0, 2, 1)))  # (B, H, T, S)

        Bh = Bmat.repeat_interleave(h // g, dim=2).float()  # (B, T, H, N)
        Ch = Cmat.repeat_interleave(h // g, dim=2).float()
        dt_s = dt.float().permute(0, 2, 1)  # (B, H, S)

        y = torch.empty(b, t, h, p, device=x.device, dtype=torch.float32)
        for h0 in range(0, h, self.head_chunk):
            h1 = min(h0 + self.head_chunk, h)
            # (B, hs, T, S): content similarity x decay x step size
            cb = torch.einsum("bthn,bshn->bhts", Ch[:, :, h0:h1], Bh[:, :, h0:h1])
            m = cb * L[:, h0:h1] * dt_s[:, h0:h1].unsqueeze(2)
            y[:, :, h0:h1] = torch.einsum("bhts,bshp->bthp", m, x[:, :, h0:h1].float())
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        z, xbc, dt_raw = self.in_proj(x).split(
            [self.d_inner, self.d_inner + 2 * self.ngroups * self.d_state, self.nheads],
            dim=-1,
        )
        # Causal depthwise conv + SiLU on [x, B, C]
        xbc = self.conv1d(xbc.transpose(1, 2))[:, :, :t].transpose(1, 2)
        xbc = F.silu(xbc)
        xs, Bmat, Cmat = xbc.split(
            [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state],
            dim=-1,
        )
        xs = xs.view(b, t, self.nheads, self.headdim)
        Bmat = Bmat.view(b, t, self.ngroups, self.d_state)
        Cmat = Cmat.view(b, t, self.ngroups, self.d_state)
        dt = F.softplus(dt_raw + self.dt_bias)  # (B, T, H)

        y = self._ssd(xs, Bmat, Cmat, dt)
        y = y + self.D.float() * xs.float()  # per-head skip
        y = y.reshape(b, t, self.d_inner).to(x.dtype)

        y = self.norm(y)
        y = y * F.silu(z)
        return self.dropout(self.out_proj(y))


class MambaBlock(nn.Module):
    """Pre-norm residual Mamba-2 block."""

    def __init__(self, dim: int, dropout: float = 0.0, **mamba_kwargs):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.mixer = Mamba2Mixer(dim, dropout=dropout, **mamba_kwargs)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.drop(self.mixer(self.norm(x)))


class AttentionBlock(nn.Module):
    """Pre-norm residual global self-attention block (x_transformers encoder).

    A single x-transformers ``Encoder`` layer with rotary positional embeddings
    and RMSNorm; includes the small feed-forward sublayer (``ff_mult``), as in
    the Nemotron-H hybrid pattern.
    """

    def __init__(
        self,
        dim: int,
        heads: int = 4,
        ff_mult: int = 1,
        attn_dropout: float = 0.1,
        ff_dropout: float = 0.0,
        rotary_pos_emb: bool = True,
    ):
        super().__init__()
        from x_transformers import Encoder

        self.enc = Encoder(
            dim=dim,
            depth=1,
            heads=heads,
            attn_dim_head=dim // heads,
            attn_dropout=attn_dropout,
            ff_mult=ff_mult,
            ff_dropout=ff_dropout,
            rotary_pos_emb=rotary_pos_emb,
            use_scalenorm=False,
            use_rmsnorm=True,
            scale_residual=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.enc(x)


class HybridMambaEncoder(nn.Module):
    """Nemotron-H-style stack: Mamba-2 blocks with periodic attention blocks.

    Block ``i`` (0-based) is an attention block iff ``(i + 1) % attention_every
    == 0``; all other blocks are Mamba-2 blocks. With ``n_layer=8`` and
    ``attention_every=4`` the pattern is ``M M M A M M M A`` (25% attention).
    Signature: ``(B, T, D) -> (B, T, D)``, matching the neuraltrain sequence
    cores (``TransformerEncoder`` / ``Conformer``) it replaces.
    """

    def __init__(
        self,
        dim: int,
        n_layer: int = 8,
        attention_every: int = 4,
        heads: int = 4,
        ff_mult: int = 1,
        attn_dropout: float = 0.1,
        ff_dropout: float = 0.0,
        dropout: float = 0.1,
        rotary_pos_emb: bool = True,
        **mamba_kwargs: tp.Any,
    ):
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by heads ({heads})")
        if rotary_pos_emb and dim // heads < 32:
            raise ValueError(
                f"dim_head ({dim // heads}) < 32: x-transformers clamps the rotary "
                f"embedding dimension to min 32. Increase dim or reduce heads, "
                f"or disable rotary_pos_emb."
            )
        blocks: list[nn.Module] = []
        for i in range(n_layer):
            if (i + 1) % attention_every == 0:
                blocks.append(
                    AttentionBlock(
                        dim,
                        heads=heads,
                        ff_mult=ff_mult,
                        attn_dropout=attn_dropout,
                        ff_dropout=ff_dropout,
                        rotary_pos_emb=rotary_pos_emb,
                    )
                )
            else:
                blocks.append(MambaBlock(dim, dropout=dropout, **mamba_kwargs))
        self.blocks = nn.ModuleList(blocks)
        self.final_norm = RMSNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return self.final_norm(x)


# --------------------------------------------------------------------------- #
# Config (neuraltrain BaseModelConfig; registered under name="MambaHybrid")
# --------------------------------------------------------------------------- #
class MambaHybrid(BaseModelConfig):
    """Config for :class:`HybridMambaEncoder` (Nemotron-H-style hybrid stack).

    Use as the ``transformer_config`` of ``ConvMambaHybrid``; ``build(dim)``
    returns a module with ``forward: (B, T, D) -> (B, T, D)``.

    Parameters
    ----------
    n_layer :
        Total number of blocks in the stack.
    attention_every :
        One attention block every ``attention_every`` blocks (last block of
        each group). ``attention_every=4`` gives the M-M-M-A pattern.
    d_state, headdim, expand, d_conv, ngroups :
        Mamba-2 (SSD) mixer hyperparameters.
    heads, ff_mult, attn_dropout, ff_dropout, rotary_pos_emb :
        Attention-block hyperparameters (x-transformers encoder).
    dropout :
        Dropout on the Mamba mixer output / block residual branch.
    head_chunk :
        Heads processed per SSD chunk (bounds the ``B * chunk * T^2`` memory).
    """

    n_layer: int = 8
    attention_every: int = 4

    # Mamba-2 mixer
    d_state: int = 128
    headdim: int = 64
    expand: int = 2
    d_conv: int = 4
    ngroups: int = 1
    head_chunk: int = 8
    dt_min: float = 0.001
    dt_max: float = 0.1
    # per-group gated RMSNorm (set to headdim to match the official/HF Mamba-2
    # gated norm exactly; None normalises over the full inner dim)
    norm_group_size: int | None = None

    # Attention blocks
    heads: int = 4
    ff_mult: int = 1
    attn_dropout: float = 0.1
    ff_dropout: float = 0.0
    rotary_pos_emb: bool = True

    dropout: float = 0.1

    def build(self, dim: int) -> nn.Module:
        kwargs = self.model_dump()
        del kwargs["name"]
        return HybridMambaEncoder(dim, **kwargs)
