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
        """Quadratic dual form of the SSD kernel, chunked over heads (float32)."""
        b, t, h, p = x.shape
        g = Bmat.shape[2]
        A = -torch.exp(self.A_log.float())  # (H,), negative

        dA = dt.float() * A  # (B, T, H)
        L = torch.exp(_segsum(dA.permute(0, 2, 1)))  # (B, H, T, S)

        Bh = Bmat.repeat_interleave(h // g, dim=2).float()
        Ch = Cmat.repeat_interleave(h // g, dim=2).float()
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
# Bidirectional block + sentence core
# --------------------------------------------------------------------------- #
class BiMambaBlock(nn.Module):
    """Pre-norm residual block with a forward and a backward Mamba-2 mixer.

    The backward mixer sees the time-reversed sequence; the two outputs are
    summed. This restores the bidirectional context V1's transformer has,
    which is required for a fair core-only ablation.
    """

    def __init__(self, dim: int, dropout: float = 0.0, **mamba_kwargs):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.fwd = Mamba2Mixer(dim, dropout=dropout, **mamba_kwargs)
        self.bwd = Mamba2Mixer(dim, dropout=dropout, **mamba_kwargs)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        y = self.fwd(h) + torch.flip(self.bwd(torch.flip(h, dims=[1])), dims=[1])
        return x + self.drop(y)


class BiMambaSentenceCoreModule(nn.Module):
    """Stack of BiMambaBlocks with the V1 sentence-transformer interface.

    ``forward(x, mask)``: ``x`` is ``(B, T_max, D)`` zero-padded, ``mask`` is a
    ``(B, T_max)`` boolean (True = real keystroke). Each sentence is processed
    unpadded to avoid contaminating the backward direction; outputs are
    scattered back into a padded tensor, exactly like V1's transformer.
    """

    def __init__(self, dim: int, n_layer: int = 4, dropout: float = 0.1, **mamba_kwargs):
        super().__init__()
        self.blocks = nn.ModuleList(
            BiMambaBlock(dim, dropout=dropout, **mamba_kwargs) for _ in range(n_layer)
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


# --------------------------------------------------------------------------- #
# Config (neuraltrain BaseModelConfig; registered as "BiMambaSentenceCore")
# --------------------------------------------------------------------------- #
class BiMambaSentenceCore(BaseModelConfig):
    """Config for :class:`BiMambaSentenceCoreModule`.

    Use as the ``transformer_config`` of the V1 experiment in place of
    ``{"name": "TransformerEncoder", ...}``; ``build(dim)`` returns a module
    with ``forward(x, mask) -> (B, T, D)``, the exact V1 interface.

    ``n_layer=4`` matches the V1 transformer's depth. ``expand`` doubles the
    inner dimension of each mixer (and there are two mixers per block), so at
    the V1 width (dim 2048) the Mamba core is substantially larger than the
    transformer; for Kaggle-scale runs use the ``small`` preset (dim 512)
    where both cores are compared at the same width.
    """

    n_layer: int = 4
    dropout: float = 0.1

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
