# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Brain2Qwerty V3 Sequence Cores:
1. Conformer (V2 reference baseline on SpanishBCBL)
2. BiMambaGatedMLP (Round 3 Best Mamba Champion: BiMamba-2 + Gated Fusion + FFN MLP)
3. Mamba3StabilizedHybrid (Mamba-3 BCNorm + Data-Dependent RoPE + Adaptive Delta-t Clamping + Attention Hybrid)
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
    """Root-mean-square layer norm with optional gating."""

    def __init__(self, dim: int, eps: float = 1e-6, group_size: int | None = None):
        super().__init__()
        if group_size is not None and dim % group_size != 0:
            raise ValueError(f"{dim=} not divisible by {group_size=}")
        self.eps = eps
        self.group_size = group_size or dim
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor, gate: torch.Tensor | None = None) -> torch.Tensor:
        dtype = x.dtype
        shape = x.shape
        x = x.float()
        if gate is not None:
            x = x * F.silu(gate.float())
        x = x.view(*shape[:-1], -1, self.group_size)
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x.view(shape) * self.weight.float()).to(dtype)


def _segsum(x: torch.Tensor) -> torch.Tensor:
    """Stable segment sum: (..., T) -> (..., T, T), lower-triangular.
    
    Optimized via 1D prefix-sum broadcasting to avoid allocating O(T^3) intermediate tensors.
    """
    T = x.size(-1)
    c = torch.cumsum(x, dim=-1)
    # diff[..., t, s] = c[..., t] - c[..., s] = sum_{k=s+1}^t x[..., k]
    diff = c.unsqueeze(-1) - c.unsqueeze(-2)
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device), 0)
    return diff.masked_fill(~mask, float("-inf"))


def _apply_rope_2d(m: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Apply 2D rotary embedding along the last dimension of (B, T, ..., D)."""
    d = m.shape[-1]
    m_even = m[..., 0::2]
    m_odd = m[..., 1::2]
    cos = torch.cos(theta)
    sin = torch.sin(theta)
    out_even = m_even * cos - m_odd * sin
    out_odd = m_even * sin + m_odd * cos
    return torch.stack([out_even, out_odd], dim=-1).flatten(-2)


# --------------------------------------------------------------------------- #
# Mamba-2 Mixer (Base SSD)
# --------------------------------------------------------------------------- #
class Mamba2Mixer(nn.Module):
    """Single Mamba-2 (SSD) mixer layer."""

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

        dt = torch.exp(
            torch.rand(self.nheads) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.A_log = nn.Parameter(torch.log(torch.empty(self.nheads).uniform_(1, 16)))
        self.D = nn.Parameter(torch.ones(self.nheads))

        self.norm = RMSNorm(self.d_inner, group_size=norm_group_size)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def _ssd(
        self,
        x: torch.Tensor,
        Bmat: torch.Tensor,
        Cmat: torch.Tensor,
        dt: torch.Tensor,
    ) -> torch.Tensor:
        b, t, h, p = x.shape
        g = Bmat.shape[2]
        A = -torch.exp(self.A_log.float())
        dA = dt.float() * A
        L = torch.exp(_segsum(dA.permute(0, 2, 1)))

        Bh = Bmat.repeat_interleave(h // g, dim=2).float()
        Ch = Cmat.repeat_interleave(h // g, dim=2).float()
        dt_s = dt.float().permute(0, 2, 1)

        y = torch.empty(b, t, h, p, device=x.device, dtype=torch.float32)
        for h0 in range(0, h, self.head_chunk):
            h1 = min(h0 + self.head_chunk, h)
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
        xbc = self.conv1d(xbc.transpose(1, 2))[:, :, :t].transpose(1, 2)
        xbc = F.silu(xbc)
        xs, Bmat, Cmat = xbc.split(
            [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state],
            dim=-1,
        )
        xs = xs.view(b, t, self.nheads, self.headdim)
        Bmat = Bmat.view(b, t, self.ngroups, self.d_state)
        Cmat = Cmat.view(b, t, self.ngroups, self.d_state)
        dt = F.softplus(dt_raw + self.dt_bias)

        y = self._ssd(xs, Bmat, Cmat, dt)
        y = y + self.D.float()[:, None] * xs.float()
        y = y.reshape(b, t, self.d_inner).to(x.dtype)
        y = self.norm(y, gate=z)
        return self.dropout(self.out_proj(y))


# --------------------------------------------------------------------------- #
# Mamba-3 Stabilized Mixer (BCNorm + RoPE + Adaptive Delta-t Clamping)
# --------------------------------------------------------------------------- #
class Mamba3Mixer(Mamba2Mixer):
    """Mamba-3 Stabilized Mixer.

    Improves continuous long-sequence state tracking via:
    1. BCNorm on B and C matrices (RMSNorm over d_state).
    2. Learned B and C biases.
    3. Continuous Data-Dependent RoPE state rotations.
    4. Clamped step-size softplus.
    """

    def __init__(self, d_model: int, rope_base: float = 10000.0, **kwargs):
        super().__init__(d_model, **kwargs)
        self.rope_base = rope_base
        self.b_norm = RMSNorm(self.d_state)
        self.c_norm = RMSNorm(self.d_state)
        self.b_bias = nn.Parameter(torch.zeros(self.ngroups, self.d_state))
        self.c_bias = nn.Parameter(torch.zeros(self.ngroups, self.d_state))

        # RoPE frequency bands
        half_dim = self.d_state // 2
        freqs = torch.exp(
            -torch.arange(0, half_dim, dtype=torch.float32)
            * (math.log(rope_base) / max(half_dim, 1))
        )
        self.register_buffer("rope_freqs", freqs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        z, xbc, dt_raw = self.in_proj(x).split(
            [self.d_inner, self.d_inner + 2 * self.ngroups * self.d_state, self.nheads],
            dim=-1,
        )
        xbc = self.conv1d(xbc.transpose(1, 2))[:, :, :t].transpose(1, 2)
        xbc = F.silu(xbc)
        xs, Bmat, Cmat = xbc.split(
            [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state],
            dim=-1,
        )
        xs = xs.view(b, t, self.nheads, self.headdim)
        Bmat = Bmat.view(b, t, self.ngroups, self.d_state) + self.b_bias
        Cmat = Cmat.view(b, t, self.ngroups, self.d_state) + self.c_bias

        # BCNorm
        Bmat = self.b_norm(Bmat)
        Cmat = self.c_norm(Cmat)

        # Adaptive softplus step-size
        dt = F.softplus(dt_raw + self.dt_bias).clamp(min=1e-4, max=0.5)

        # Data-dependent RoPE state rotation
        dt_mean = dt.mean(dim=-1, keepdim=True)  # (B, T, 1)
        cum_t = torch.cumsum(dt_mean, dim=1)     # (B, T, 1)
        theta = cum_t * self.rope_freqs.view(1, 1, -1)  # (B, T, half_dim)
        theta = theta.unsqueeze(2).expand(b, t, self.ngroups, -1)

        Bmat = _apply_rope_2d(Bmat, theta)
        Cmat = _apply_rope_2d(Cmat, theta)

        y = self._ssd(xs, Bmat, Cmat, dt)
        y = y + self.D.float()[:, None] * xs.float()
        y = y.reshape(b, t, self.d_inner).to(x.dtype)
        y = self.norm(y, gate=z)
        return self.dropout(self.out_proj(y))


# --------------------------------------------------------------------------- #
# Attention Block (ALiBi / Rotary Self-Attention + MLP)
# --------------------------------------------------------------------------- #
class AttentionBlock(nn.Module):
    """Pre-norm residual global self-attention block."""

    def __init__(
        self,
        dim: int,
        heads: int = 4,
        ff_mult: int = 2,
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


# --------------------------------------------------------------------------- #
# BiMamba Block with Gated Fusion + FFN MLP Sublayer
# --------------------------------------------------------------------------- #
class BiMambaGatedMLPBlock(nn.Module):
    """Bidirectional Mamba-2 Block with Learned Non-Linear Gated Fusion and FFN."""

    def __init__(self, dim: int, dropout: float = 0.1, ff_mult: int = 4, **mamba_kwargs):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.fwd = Mamba2Mixer(dim, dropout=dropout, **mamba_kwargs)
        self.bwd = Mamba2Mixer(dim, dropout=dropout, **mamba_kwargs)
        self.fuse_proj = nn.Linear(2 * dim, dim, bias=False)
        self.fuse_gate = nn.Linear(2 * dim, dim, bias=True)
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = RMSNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, ff_mult * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_mult * dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        yf = self.fwd(h)
        yb = torch.flip(self.bwd(torch.flip(h, dims=[1])), dims=[1])
        cat = torch.cat([yf, yb], dim=-1)
        y = self.fuse_proj(cat) * F.silu(self.fuse_gate(cat))
        x = x + self.drop1(y)
        x = x + self.mlp(self.norm2(x))
        return x


# --------------------------------------------------------------------------- #
# Sequence Stack Implementations
# --------------------------------------------------------------------------- #
class BiMambaGatedMLPEncoder(nn.Module):
    """Stack of BiMambaGatedMLPBlocks (Round 3 Champion Architecture)."""

    def __init__(
        self,
        dim: int,
        n_layer: int = 8,
        dropout: float = 0.1,
        gradient_checkpointing: bool = True,
        **mamba_kwargs,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            BiMambaGatedMLPBlock(dim, dropout=dropout, **mamba_kwargs)
            for _ in range(n_layer)
        )
        self.final_norm = RMSNorm(dim)
        # Recompute block internals (the O(H*T^2) SSD maps) during backward
        # instead of storing them: ~35% slower step, ~4x lower peak memory.
        self.gradient_checkpointing = gradient_checkpointing

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ckpt = (
            self.gradient_checkpointing and self.training and torch.is_grad_enabled()
        )
        for blk in self.blocks:
            if ckpt:
                x = torch.utils.checkpoint.checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        return self.final_norm(x)


class HybridMambaEncoder(nn.Module):
    """Nemotron-H style Hybrid Stack (Mamba-2 / Mamba-3 + Attention)."""

    def __init__(
        self,
        dim: int,
        n_layer: int = 8,
        attention_every: int = 4,
        heads: int = 4,
        ff_mult: int = 2,
        attn_dropout: float = 0.1,
        ff_dropout: float = 0.0,
        dropout: float = 0.1,
        rotary_pos_emb: bool = True,
        gradient_checkpointing: bool = True,
        mixer_cls=Mamba2Mixer,
        **mamba_kwargs: tp.Any,
    ):
        super().__init__()
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
                norm = RMSNorm(dim)
                mixer = mixer_cls(dim, dropout=dropout, **mamba_kwargs)
                blocks.append(nn.ModuleDict({"norm": norm, "mixer": mixer, "drop": nn.Dropout(dropout)}))
        self.blocks = nn.ModuleList(blocks)
        self.final_norm = RMSNorm(dim)
        # Recompute block internals (the O(H*T^2) SSD maps) during backward
        # instead of storing them: ~35% slower step, ~4x lower peak memory.
        self.gradient_checkpointing = gradient_checkpointing

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ckpt = (
            self.gradient_checkpointing and self.training and torch.is_grad_enabled()
        )
        for blk in self.blocks:
            if isinstance(blk, AttentionBlock):
                if ckpt:
                    x = torch.utils.checkpoint.checkpoint(blk, x, use_reentrant=False)
                else:
                    x = blk(x)
            else:
                def _mamba_block(inp: torch.Tensor, blk=blk) -> torch.Tensor:
                    h = blk["norm"](inp)
                    y = blk["mixer"](h)
                    return inp + blk["drop"](y)

                if ckpt:
                    x = torch.utils.checkpoint.checkpoint(
                        _mamba_block, x, use_reentrant=False
                    )
                else:
                    x = _mamba_block(x)
        return self.final_norm(x)


# --------------------------------------------------------------------------- #
# Pydantic Configs for NeuralTrain Registry
# --------------------------------------------------------------------------- #
class BiMambaGatedMLP(BaseModelConfig):
    """Config for :class:`BiMambaGatedMLPEncoder`."""

    n_layer: int = 8
    dropout: float = 0.1
    d_state: int = 128
    headdim: int = 64
    expand: int = 2
    d_conv: int = 4
    ngroups: int = 1
    head_chunk: int = 8
    ff_mult: int = 4
    gradient_checkpointing: bool = True

    def build(self, dim: int) -> nn.Module:
        kwargs = self.model_dump()
        kwargs.pop("name", None)
        return BiMambaGatedMLPEncoder(dim, **kwargs)


class MambaHybrid(BaseModelConfig):
    """Config for :class:`HybridMambaEncoder` (Nemotron-H hybrid)."""

    n_layer: int = 8
    attention_every: int = 4
    d_state: int = 128
    headdim: int = 64
    expand: int = 2
    d_conv: int = 4
    ngroups: int = 1
    head_chunk: int = 8
    dt_min: float = 0.001
    dt_max: float = 0.1
    norm_group_size: int | None = None

    heads: int = 4
    ff_mult: int = 2
    attn_dropout: float = 0.1
    ff_dropout: float = 0.0
    rotary_pos_emb: bool = True
    dropout: float = 0.1
    gradient_checkpointing: bool = True

    def build(self, dim: int) -> nn.Module:
        kwargs = self.model_dump()
        kwargs.pop("name", None)
        return HybridMambaEncoder(dim, mixer_cls=Mamba2Mixer, **kwargs)


class Mamba3StabilizedHybrid(MambaHybrid):
    """Config for Stabilized Mamba-3 Hybrid (BCNorm + RoPE + Attention)."""

    rope_base: float = 10000.0

    def build(self, dim: int) -> nn.Module:
        kwargs = self.model_dump()
        kwargs.pop("name", None)
        return HybridMambaEncoder(dim, mixer_cls=Mamba3Mixer, **kwargs)
