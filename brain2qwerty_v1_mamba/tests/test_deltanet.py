# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the DeltaNet sentence core (no data, no GPU needed).

Run from the repo root:

    pytest brain2qwerty_v1_mamba/tests/test_deltanet.py -v

These verify the properties the V1 ablation depends on:
  1. correctness of the O(T^2) "WY" parallel form against the literal
     delta-rule recurrence W_t = W_{t-1}(I - beta k k^T) + beta v k^T
  2. shape/interface parity with V1's transformer: forward(x, mask) -> (B, T, D),
     padded positions stay zero, valid positions match the unpadded run
  3. gradient flow through the whole stack
  4. the config registry path used by the experiment config (name resolution
     + build at both 512 and 2048 width)
"""

import torch
import torch.nn.functional as F

from neuraltrain.models.base import BaseModelConfig

from ..deltanet_core import (
    BiDeltaNetSentenceCore,
    DeltaNetMixer,
    _delta_rule_parallel,
    _solve_unit_lower,
)

DIM, N_LAYER = 64, 2  # tiny config for CPU-speed tests


def _core(**overrides) -> torch.nn.Module:
    torch.manual_seed(0)
    cfg = dict(n_layer=N_LAYER, dropout=0.0, headdim=16, expand=1)
    cfg.update(overrides)
    return BiDeltaNetSentenceCore(**cfg).build(DIM)


def _naive_delta_rule(q, k, v, beta_raw):
    """Literal recurrence: W_t = W_{t-1}(I - beta k k^T) + beta v k^T, o_t = W_t q_t."""
    B, T, H, P = q.shape
    q = q.float() * P**-0.5
    k = F.normalize(k.float(), p=2, dim=-1)
    v = v.float()
    beta = torch.sigmoid(beta_raw.float())
    W = torch.zeros(B, H, P, P)
    outs = []
    for t in range(T):
        kt = k[:, t].unsqueeze(-1)  # (B, H, P, 1)
        bt = beta[:, t].reshape(B, H, 1, 1)
        vt = v[:, t].unsqueeze(-1)  # (B, H, P, 1)
        W = W - bt * (W @ kt) @ kt.transpose(-1, -2) + bt * vt @ kt.transpose(-1, -2)
        outs.append((W @ q[:, t].unsqueeze(-1)).squeeze(-1))  # (B, H, P)
    return torch.stack(outs, dim=1)  # (B, T, H, P)


def test_parallel_form_matches_naive_recurrence():
    """The factored pure function must equal the literal delta-rule loop."""
    torch.manual_seed(3)
    B, T, H, P = 2, 9, 4, 16
    q = torch.randn(B, T, H, P)
    k = torch.randn(B, T, H, P)
    v = torch.randn(B, T, H, P)
    beta_raw = torch.randn(B, T, H)
    fast = _delta_rule_parallel(q, k, v, beta_raw)
    slow = _naive_delta_rule(q, k, v, beta_raw)
    assert fast.shape == slow.shape == (B, T, H, P)
    assert torch.allclose(fast, slow, atol=1e-4), (
        f"max abs diff {(fast - slow).abs().max().item():.2e}"
    )


def test_triangular_solve_fallback_matches():
    """The forward-substitution fallback must agree with solve_triangular."""
    torch.manual_seed(4)
    B, H, T, P = 2, 3, 7, 8
    A = torch.tril(torch.randn(B, H, T, T), -1) * 0.3
    Bm = torch.randn(B, H, T, P)
    eye = torch.eye(T)
    ref = torch.linalg.solve_triangular(A + eye, Bm, upper=False, unitriangular=True)

    real_solve = torch.linalg.solve_triangular
    def _boom(*args, **kwargs):
        raise RuntimeError("simulated unimplemented backend")
    torch.linalg.solve_triangular = _boom
    try:
        out = _solve_unit_lower(A, Bm)
    finally:
        torch.linalg.solve_triangular = real_solve
    assert torch.allclose(out, ref, atol=1e-5)


def test_mixer_shape_and_dtype():
    mixer = DeltaNetMixer(DIM, headdim=16, expand=1, dropout=0.0)
    x = torch.randn(3, 11, DIM)
    y = mixer(x)
    assert y.shape == x.shape
    assert y.dtype == x.dtype
    assert torch.isfinite(y).all()


def test_interface_shape_and_padding():
    """Padded (B, T_max, D) + mask -> (B, T_max, D); padding stays zero and
    valid positions match running the unpadded sentence (mask=None path)."""
    core = _core().eval()
    torch.manual_seed(1)
    x = torch.randn(2, 12, DIM)
    mask = torch.zeros(2, 12, dtype=torch.bool)
    mask[0, :12] = True
    mask[1, :7] = True
    with torch.no_grad():
        out = core(x, mask=mask)
        ref0 = core(x[0:1, :12])  # unpadded, mask=None
        ref1 = core(x[1:2, :7])
    assert out.shape == (2, 12, DIM)
    assert torch.isfinite(out).all()
    # padded positions stay zero
    assert (out[1, 7:] == 0).all()
    # valid positions match the unpadded runs
    assert torch.allclose(out[0:1, :12], ref0, atol=1e-5)
    assert torch.allclose(out[1:2, :7], ref1, atol=1e-5)


def test_bidirectionality():
    """A change at the LAST keystroke must affect the FIRST output."""
    core = _core().eval()
    torch.manual_seed(2)
    x = torch.randn(1, 8, DIM)
    with torch.no_grad():
        y1 = core(x)
        x2 = x.clone()
        x2[0, -1] += 1.0
        y2 = core(x2)
    assert not torch.allclose(y1[0, 0], y2[0, 0]), (
        "first output did not change when the last input changed — "
        "the core is not bidirectional"
    )


def test_gradient_flow():
    core = _core(use_mlp=True, gated_fusion=True)
    torch.manual_seed(5)
    x = torch.randn(2, 6, DIM, requires_grad=True)
    mask = torch.ones(2, 6, dtype=torch.bool)
    loss = core(x, mask=mask).sum()
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for name, p in core.named_parameters():
        assert p.grad is not None, f"no gradient for {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite gradient for {name}"


def test_config_wiring():
    """sentence_core() dicts must resolve in the BaseModelConfig registry and
    build at both the small (512) and full (2048) widths."""
    from ..config.model_config import sentence_core

    for core_name, expect_mlp in (("deltanet", False), ("deltanet_mlp", True)):
        cfg = sentence_core(core_name, small=True)
        assert cfg["name"] == "BiDeltaNetSentenceCore"
        obj = BaseModelConfig(**cfg)  # registry resolution by class name
        assert isinstance(obj, BiDeltaNetSentenceCore)
        assert obj.use_mlp is expect_mlp
        assert obj.gated_fusion is expect_mlp
        for dim in (512, 2048):  # full width catches headdim divisibility issues
            module = obj.build(dim)
            out = module(torch.randn(1, 5, dim))
            assert out.shape == (1, 5, dim)
            assert torch.isfinite(out).all()
