# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""DeltaNet sentence core for the Brain2Qwerty V1 ablation (Stage 2).

DeltaNet (Yang et al. 2024, "Parallelizing Linear Transformers with the Delta
Rule over Sequence Length") is linear attention whose state update is the
associative *delta rule* — an error-correcting write that removes the old
value associated with a key before writing the new one::

    W_t = W_{t-1} (I - beta_t k_t k_t^T) + beta_t v_t k_t^T,   W_0 = 0
    o_t = W_t q_t

This module adds DeltaNet as a Stage-2 sentence-core ablation, replacing the
transformer self-attention of V1 (and sitting next to the Mamba-2/Mamba-3
cores in ``mamba_core.py``). Sentences here are tens of keystrokes long, so
the exact O(T^2) "WY" parallel form in pure PyTorch (float32 internally, no
CUDA-only kernels) is entirely adequate and runs on CPU and Apple MPS.

Bidirectionality (required for parity with V1's bidirectional transformer) is
obtained for free from the shared plumbing: ``BiMambaSentenceCoreModule``
runs each sentence unpadded and ``BiMambaBlock`` instantiates the mixer twice
(forward + backward on the flipped sequence).

Derivation of the parallel form: writing W_t = sum_{s<=t} u_s k_s^T and
substituting into the recurrence gives

    u_t = beta_t ( v_t - sum_{s<t} (k_t . k_s) u_s )
    o_t = sum_{s<=t} (q_t . k_s) u_s

i.e. with A[t, s] = beta_t (k_t . k_s) for s < t (strictly lower triangular),
(I + A) U = diag(beta) V is a unit lower-triangular system solved for U, and
O = tril(Q K^T) U. Keys are L2-normalized (DeltaNet uses unit-norm keys) and
queries scaled by P**-0.5; beta = sigmoid(beta_raw).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuraltrain.models.base import BaseModelConfig

from .mamba_core import BiMambaSentenceCoreModule, RMSNorm


# --------------------------------------------------------------------------- #
# Delta rule — exact quadratic "WY" parallel form (float32)
# --------------------------------------------------------------------------- #
def _solve_unit_lower(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Solve (I + A) U = B for U, with A strictly lower triangular.

    A: (B, H, T, T) strictly lower triangular; B: (B, H, T, P).
    Prefers ``torch.linalg.solve_triangular``; falls back to an explicit
    forward-substitution loop (T is small here) if the linalg op errors
    (e.g. an unimplemented backend).
    """
    T = A.size(-1)
    eye = torch.eye(T, dtype=A.dtype, device=A.device)
    try:
        return torch.linalg.solve_triangular(
            A + eye, B, upper=False, unitriangular=True
        )
    except RuntimeError:
        U = torch.empty_like(B)
        for t in range(T):
            rhs = B[..., t, :]
            if t > 0:
                # U_t = B_t - sum_{s<t} A[t, s] U_s
                rhs = rhs - (A[..., t, :t].unsqueeze(-1) * U[..., :t, :]).sum(dim=-2)
            U[..., t, :] = rhs
        return U


def _delta_rule_parallel(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta_raw: torch.Tensor,
) -> torch.Tensor:
    """Delta-rule attention output, exact O(T^2) parallel form (float32).

    Args:
        q, k, v: (B, T, H, P)
        beta_raw: (B, T, H) pre-sigmoid write strengths

    Returns:
        O: (B, T, H, P), float32 — o_t = W_t q_t with
        W_t = W_{t-1}(I - beta_t k_t k_t^T) + beta_t v_t k_t^T, W_0 = 0.
    """
    P = q.size(-1)
    q = q.float() * P**-0.5
    k = F.normalize(k.float(), p=2, dim=-1)  # DeltaNet: unit-norm keys
    v = v.float()
    beta = torch.sigmoid(beta_raw.float())  # (B, T, H)

    # -> (B, H, T, P) / (B, H, T)
    q, k, v = (t.transpose(1, 2) for t in (q, k, v))
    beta = beta.transpose(1, 2)

    # A[t, s] = beta_t (k_t . k_s) for s < t — strictly lower triangular
    A = torch.tril((k * beta.unsqueeze(-1)) @ k.transpose(-1, -2), diagonal=-1)
    # (I + A) U = diag(beta) V
    U = _solve_unit_lower(A, v * beta.unsqueeze(-1))  # (B, H, T, P)
    # O = tril(Q K^T, 0) U  (causal mask includes the diagonal)
    O = torch.tril(q @ k.transpose(-1, -2), diagonal=0) @ U  # (B, H, T, P)
    return O.transpose(1, 2)  # (B, T, H, P)


# --------------------------------------------------------------------------- #
# DeltaNet mixer (Mamba2Mixer-style interface: (B, T, D) -> (B, T, D))
# --------------------------------------------------------------------------- #
class DeltaNetMixer(nn.Module):
    """Single causal DeltaNet mixer layer. Input/output: (B, T, d_model).

    Mirrors :class:`~brain2qwerty_v1_mamba.mamba_core.Mamba2Mixer`'s
    in_proj / gated-RMSNorm / out_proj / dropout pattern, with the SSD kernel
    replaced by the delta rule (no conv, no dt/A/D parameters — DeltaNet's
    recurrence is fully data-dependent through q/k/v/beta).
    """

    def __init__(
        self,
        d_model: int,
        headdim: int = 64,
        expand: int = 1,
        dropout: float = 0.0,
        norm_group_size: int | None = None,
    ):
        super().__init__()
        self.d_inner = expand * d_model
        if self.d_inner % headdim != 0:
            raise ValueError(f"d_inner={self.d_inner} not divisible by {headdim=}")
        self.nheads = self.d_inner // headdim
        self.headdim = headdim

        # q, k, v (d_inner each) + beta (per head) + z output gate (d_inner)
        self.in_proj = nn.Linear(
            d_model, 3 * self.d_inner + self.nheads + self.d_inner, bias=False
        )
        self.norm = RMSNorm(self.d_inner, group_size=norm_group_size)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        q, k, v, beta_raw, z = self.in_proj(x).split(
            [self.d_inner, self.d_inner, self.d_inner, self.nheads, self.d_inner],
            dim=-1,
        )
        q = q.view(b, t, self.nheads, self.headdim)
        k = k.view(b, t, self.nheads, self.headdim)
        v = v.view(b, t, self.nheads, self.headdim)

        y = _delta_rule_parallel(q, k, v, beta_raw)
        y = y.reshape(b, t, self.d_inner).to(x.dtype)

        y = self.norm(y, gate=z)
        return self.dropout(self.out_proj(y))


# --------------------------------------------------------------------------- #
# Config (neuraltrain BaseModelConfig)
# --------------------------------------------------------------------------- #
class BiDeltaNetSentenceCore(BaseModelConfig):
    """Config for a bidirectional DeltaNet sentence core."""

    n_layer: int = 4
    dropout: float = 0.1
    use_mlp: bool = False
    gated_fusion: bool = False

    # DeltaNet mixer hyperparameters
    headdim: int = 64
    expand: int = 1
    norm_group_size: int | None = None

    def build(self, dim: int) -> BiMambaSentenceCoreModule:
        kwargs = self.model_dump()
        kwargs.pop("name", None)
        return BiMambaSentenceCoreModule(dim, mixer_cls=DeltaNetMixer, **kwargs)
