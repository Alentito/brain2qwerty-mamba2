# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Bidirectional Mamba-2 sentence core for the Brain2Qwerty V1 ablation.

This is the ONLY component that differs from the V1 reference
(``brain2qwerty_v1``): V1 refines per-keystroke conv embeddings with a
bidirectional sentence-level ``TransformerEncoder`` (depth 4, ALiBi). Here that
module is replaced by a stack of bidirectional Mamba-2 (SSD) blocks.

Everything else — the 500 ms keystroke windows, the SimpleConvTimeAgg encoder,
the 29-class character head, the cross-entropy loss, the optimiser — is shared
verbatim with V1, so any test-CER delta is attributable to the sequence core.

Design notes
------------
* ``Mamba2Mixer`` is copied verbatim from ``brain2qwerty_colab/mamba.py``
  (pure-PyTorch quadratic dual form of the SSD kernel, float32 internally, no
  ``mamba-ssm`` CUDA dependency — it runs on Kaggle/Colab T4s and CPU).
* The V1 transformer is *bidirectional* (each keystroke embedding can attend
  to the whole sentence). A causal Mamba would not be a fair comparison, so
  each block runs two mixers — one on the sequence, one on the reversed
  sequence — and sums their outputs (``BiMambaBlock``).
* V1's pl_module calls the core as ``core(x, mask=mask)`` on zero-padded
  ``(B, T_max, D)`` tensors. Padding a bidirectional state-space model with
  zeros would contaminate the backward direction, so ``BiMambaSentenceCore``
  runs each sentence unpadded (sentences are tens of keystrokes long; the
  Python loop is negligible).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuraltrain.models.base import BaseModelConfig


# --------------------------------------------------------------------------- #
# Primitives (verbatim from brain2qwerty_colab/mamba.py)
# --------------------------------------------------------------------------- #
class RMSNorm(nn.Module):
    """Root-mean-square layer norm (no bias, learned gain), optional gating."""

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
    """Stable segment sum: (..., T) -> (..., T, T), lower-triangular."""
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
    """Single causal Mamba-2 (SSD) mixer layer. Input/output: (B, T, d_model)."""

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

    def _ssd(self, x, Bmat, Cmat, dt) -> torch.Tensor:
        """Quadratic dual form of the SSD kernel (float32)."""
        h = x.shape[2]
        g = Bmat.shape[2]
        Bh = Bmat.repeat_interleave(h // g, dim=2).float()
        Ch = Cmat.repeat_interleave(h // g, dim=2).float()
        return self._ssd_expanded(x, Bh, Ch, dt)

    def _ssd_expanded(self, x, Bh, Ch, dt) -> torch.Tensor:
        """SSD with per-head (float32) B/C, chunked over heads."""
        b, t, h, p = x.shape
        A = -torch.exp(self.A_log.float())  # (H,), negative

        dA = dt.float() * A  # (B, T, H)
        L = torch.exp(_segsum(dA.permute(0, 2, 1)))  # (B, H, T, S)

        dt_s = dt.float().permute(0, 2, 1)  # (B, H, S)

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
        dt = F.softplus(dt_raw + self.dt_bias)  # (B, T, H)

        y = self._ssd(xs, Bmat, Cmat, dt)
        y = y + self.D.float()[:, None] * xs.float()
        y = y.reshape(b, t, self.d_inner).to(x.dtype)

        y = self.norm(y, gate=z)
        return self.dropout(self.out_proj(y))


# --------------------------------------------------------------------------- #
# Mamba-3-style mixer (Lahoti et al. 2026, arXiv:2603.15569)
# --------------------------------------------------------------------------- #
class Mamba3Mixer(Mamba2Mixer):
    """Mamba-3-inspired upgrades on top of :class:`Mamba2Mixer`.

    Implements two of the three Mamba-3 core changes, the ones targeting
    training stability and state-tracking expressivity:

    * **BCNorm + B/C biases** — RMSNorm on the B and C projections (mirrors
      QKNorm in modern transformers) plus learnable channelwise biases. This
      directly targets the known Mamba instability path of unbounded B/C norm
      growth, and lets the model drop reliance on the output gate norm.
    * **Complex-valued state via data-dependent RoPE** — a per-head rotation
      rate ``theta_t`` is projected from the input; the cumulative angle
      ``phi_t = cumsum(theta)`` rotates B by ``-phi`` and C by ``+phi`` across
      state-dim pairs, so every pairwise interaction ``c_t . b_s`` carries the
      relative rotation ``R(phi_t - phi_s)`` — the paper's RoPE trick, exact
      in this quadratic (single-chunk) formulation.

    NOT included: the exponential-trapezoidal discretization (the paper shows
    it mainly makes the short conv redundant; we keep ``d_conv``) and the MIMO
    state update (inference-efficiency feature, irrelevant at our scale).
    """

    def __init__(self, d_model: int, *args, rope_base: float = 10000.0, **kwargs):
        super().__init__(d_model, *args, **kwargs)
        if self.d_state % 2:
            raise ValueError(f"Mamba3Mixer needs even d_state, got {self.d_state}")
        self.rope_base = rope_base
        # extra +nheads outputs: the per-head rotation rate theta
        self.in_proj = nn.Linear(
            d_model,
            2 * self.d_inner + 2 * self.ngroups * self.d_state + 2 * self.nheads,
            bias=False,
        )
        self.bc_norm = RMSNorm(self.ngroups * self.d_state, group_size=self.d_state)
        self.b_bias = nn.Parameter(torch.zeros(self.ngroups * self.d_state))
        self.c_bias = nn.Parameter(torch.zeros(self.ngroups * self.d_state))

    def _rope(self, x: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        """Rotate state-dim pairs of x (B, T, H, N) by angles phi (B, T, H)."""
        half = x.shape[-1] // 2
        omega = self.rope_base ** (
            -torch.arange(half, device=x.device, dtype=torch.float32) / half
        )
        ang = phi.unsqueeze(-1) * omega  # (B, T, H, half)
        cos, sin = ang.cos(), ang.sin()
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        z, xbc, dt_raw, theta = self.in_proj(x).split(
            [self.d_inner, self.d_inner + 2 * self.ngroups * self.d_state,
             self.nheads, self.nheads],
            dim=-1,
        )
        xbc = self.conv1d(xbc.transpose(1, 2))[:, :, :t].transpose(1, 2)
        xbc = F.silu(xbc)
        xs, Bmat, Cmat = xbc.split(
            [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state],
            dim=-1,
        )
        # Mamba-3: BCNorm + learnable biases (stability + approximation power)
        Bmat = self.bc_norm(Bmat) + self.b_bias
        Cmat = self.bc_norm(Cmat) + self.c_bias

        xs = xs.view(b, t, self.nheads, self.headdim)
        dt = F.softplus(dt_raw + self.dt_bias)  # (B, T, H)

        # Mamba-3: data-dependent rotation (complex state via RoPE trick)
        h, g = self.nheads, self.ngroups
        phi = torch.cumsum(theta.float(), dim=1)  # (B, T, H)
        Bh = Bmat.view(b, t, g, self.d_state).repeat_interleave(h // g, dim=2).float()
        Ch = Cmat.view(b, t, g, self.d_state).repeat_interleave(h // g, dim=2).float()
        Bh = self._rope(Bh, -phi)
        Ch = self._rope(Ch, phi)

        y = self._ssd_expanded(xs, Bh, Ch, dt)
        y = y + self.D.float()[:, None] * xs.float()
        y = y.reshape(b, t, self.d_inner).to(x.dtype)

        y = self.norm(y, gate=z)
        return self.dropout(self.out_proj(y))


# --------------------------------------------------------------------------- #
# Bidirectional block + sentence core
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Attention block (ALiBi relative position bias + MLP sublayer)
# --------------------------------------------------------------------------- #
class AttentionBlock(nn.Module):
    """Pre-norm Self-Attention Block with ALiBi relative position bias + MLP."""

    def __init__(self, dim: int, heads: int = 4, dropout: float = 0.1, ff_mult: int = 4):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.heads = heads
        if dim % heads != 0:
            raise ValueError(f"{dim=} not divisible by {heads=}")
        self.headdim = dim // heads
        self.scale = 1.0 / math.sqrt(self.headdim)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.drop1 = nn.Dropout(dropout)

        # Feed-forward network (MLP) sublayer
        self.norm2 = RMSNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, ff_mult * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_mult * dim, dim),
            nn.Dropout(dropout),
        )
        # ALiBi slope calculation
        slopes = [2 ** (-8 * (i + 1) / heads) for i in range(heads)]
        self.register_buffer("slopes", torch.tensor(slopes, dtype=torch.float32).view(1, heads, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).view(b, t, 3, self.heads, self.headdim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (b, heads, t, headdim)

        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # (b, heads, t, t)

        pos = torch.arange(t, device=x.device)
        dist = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs()  # (t, t)
        alibi = -self.slopes * dist.unsqueeze(0).unsqueeze(0)  # (1, heads, t, t)
        scores = scores + alibi

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).permute(0, 2, 1, 3).reshape(b, t, d)
        x = x + self.drop1(self.out_proj(out))
        x = x + self.mlp(self.norm2(x))
        return x


# --------------------------------------------------------------------------- #
# Bidirectional block + sentence core
# --------------------------------------------------------------------------- #
class BiMambaBlock(nn.Module):
    """Pre-norm residual block with a forward and a backward Mamba mixer.

    Supports optional FFN/MLP sublayer and learned gated non-linear fusion.
    """

    def __init__(
        self,
        dim: int,
        dropout: float = 0.0,
        mixer_cls=Mamba2Mixer,
        use_mlp: bool = False,
        gated_fusion: bool = False,
        ff_mult: int = 4,
        **mamba_kwargs,
    ):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.fwd = mixer_cls(dim, dropout=dropout, **mamba_kwargs)
        self.bwd = mixer_cls(dim, dropout=dropout, **mamba_kwargs)
        self.drop = nn.Dropout(dropout)
        self.gated_fusion = gated_fusion
        if gated_fusion:
            self.fuse_proj = nn.Linear(2 * dim, dim, bias=False)
            self.fuse_gate = nn.Linear(2 * dim, dim, bias=True)

        self.use_mlp = use_mlp
        if use_mlp:
            self.norm_mlp = RMSNorm(dim)
            self.mlp = nn.Sequential(
                nn.Linear(dim, ff_mult * dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(ff_mult * dim, dim),
                nn.Dropout(dropout),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        yf = self.fwd(h)
        yb = torch.flip(self.bwd(torch.flip(h, dims=[1])), dims=[1])
        if self.gated_fusion:
            cat = torch.cat([yf, yb], dim=-1)
            y = self.fuse_proj(cat) * F.silu(self.fuse_gate(cat))
        else:
            y = yf + yb
        x = x + self.drop(y)
        if self.use_mlp:
            x = x + self.mlp(self.norm_mlp(x))
        return x


class BiMambaSentenceCoreModule(nn.Module):
    """Stack of BiMambaBlocks with the V1 sentence-transformer interface."""

    def __init__(
        self,
        dim: int,
        n_layer: int = 4,
        dropout: float = 0.1,
        mixer_cls=Mamba2Mixer,
        use_mlp: bool = False,
        gated_fusion: bool = False,
        **mamba_kwargs,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            BiMambaBlock(
                dim,
                dropout=dropout,
                mixer_cls=mixer_cls,
                use_mlp=use_mlp,
                gated_fusion=gated_fusion,
                **mamba_kwargs,
            )
            for _ in range(n_layer)
        )
        self.final_norm = RMSNorm(dim)

    def _run(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return self.final_norm(x)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is None:
            return self._run(x)
        out = torch.zeros_like(x)
        for i in range(x.shape[0]):
            L = int(mask[i].sum().item())
            if L > 0:
                out[i, :L] = self._run(x[i : i + 1, :L])
        return out


class HybridSentenceCoreModule(nn.Module):
    """Nemotron-H style Hybrid Mamba-Attention Sentence Core.

    Interleaves BiMamba blocks with periodic Global ALiBi Attention blocks.
    Default: attention every 4 blocks -> [M, M, M, A] (or [M, M, M, A, M, M, M, A]).
    """

    def __init__(
        self,
        dim: int,
        n_layer: int = 4,
        attention_every: int = 4,
        heads: int = 4,
        dropout: float = 0.1,
        mixer_cls=Mamba2Mixer,
        use_mlp: bool = False,
        gated_fusion: bool = False,
        **mamba_kwargs,
    ):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(n_layer):
            if (i + 1) % attention_every == 0:
                self.blocks.append(AttentionBlock(dim, heads=heads, dropout=dropout))
            else:
                self.blocks.append(
                    BiMambaBlock(
                        dim,
                        dropout=dropout,
                        mixer_cls=mixer_cls,
                        use_mlp=use_mlp,
                        gated_fusion=gated_fusion,
                        **mamba_kwargs,
                    )
                )
        self.final_norm = RMSNorm(dim)

    def _run(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return self.final_norm(x)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is None:
            return self._run(x)
        out = torch.zeros_like(x)
        for i in range(x.shape[0]):
            L = int(mask[i].sum().item())
            if L > 0:
                out[i, :L] = self._run(x[i : i + 1, :L])
        return out


class BiMamba3SentenceCoreModule(BiMambaSentenceCoreModule):
    """Bidirectional stack of Mamba-3-style blocks (BCNorm + RoPE)."""

    def __init__(self, dim: int, **kwargs):
        super().__init__(dim, mixer_cls=Mamba3Mixer, **kwargs)


class HybridMamba3SentenceCoreModule(HybridSentenceCoreModule):
    """Nemotron-H style Hybrid Mamba-3 + Attention Sentence Core."""

    def __init__(self, dim: int, **kwargs):
        super().__init__(dim, mixer_cls=Mamba3Mixer, **kwargs)


# --------------------------------------------------------------------------- #
# Configs (neuraltrain BaseModelConfig)
# --------------------------------------------------------------------------- #
class BiMambaSentenceCore(BaseModelConfig):
    """Config for :class:`BiMambaSentenceCoreModule`."""

    n_layer: int = 4
    dropout: float = 0.1
    use_mlp: bool = False
    gated_fusion: bool = False

    # Mamba-2 mixer hyperparameters
    d_state: int = 128
    headdim: int = 64
    expand: int = 2
    d_conv: int = 4
    ngroups: int = 1
    head_chunk: int = 8
    dt_min: float = 0.001
    dt_max: float = 0.1
    norm_group_size: int | None = None

    def build(self, dim: int) -> BiMambaSentenceCoreModule:
        kwargs = self.model_dump()
        kwargs.pop("name", None)
        return BiMambaSentenceCoreModule(dim, **kwargs)


class BiMamba3SentenceCore(BiMambaSentenceCore):
    """Config for Mamba-3 sentence core (BCNorm + RoPE)."""

    rope_base: float = 10000.0

    def build(self, dim: int) -> BiMambaSentenceCoreModule:
        kwargs = self.model_dump()
        kwargs.pop("name", None)
        return BiMambaSentenceCoreModule(dim, mixer_cls=Mamba3Mixer, **kwargs)


class HybridSentenceCore(BaseModelConfig):
    """Config for Nemotron-H style Hybrid Mamba-Transformer Core."""

    n_layer: int = 4
    attention_every: int = 4
    heads: int = 4
    dropout: float = 0.1
    use_mlp: bool = False
    gated_fusion: bool = False

    d_state: int = 128
    headdim: int = 64
    expand: int = 2
    d_conv: int = 4
    ngroups: int = 1
    head_chunk: int = 8
    dt_min: float = 0.001
    dt_max: float = 0.1
    norm_group_size: int | None = None

    def build(self, dim: int) -> HybridSentenceCoreModule:
        kwargs = self.model_dump()
        kwargs.pop("name", None)
        return HybridSentenceCoreModule(dim, mixer_cls=Mamba2Mixer, **kwargs)


class HybridMamba3SentenceCore(HybridSentenceCore):
    """Config for Hybrid Mamba-3 + Attention Core."""

    rope_base: float = 10000.0

    def build(self, dim: int) -> HybridSentenceCoreModule:
        kwargs = self.model_dump()
        kwargs.pop("name", None)
        return HybridSentenceCoreModule(dim, mixer_cls=Mamba3Mixer, **kwargs)

