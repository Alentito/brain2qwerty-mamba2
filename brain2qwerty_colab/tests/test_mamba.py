# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Smoke, correctness, and parity tests for the V3 hybrid Mamba-2 stack.

Run on the cluster (or any machine with the b2q environment):

    pytest brain2qwerty_colab/tests/test_mamba.py -v

The parity test compares our pure-PyTorch ``Mamba2Mixer`` against the
HuggingFace reference implementation (``transformers`` >= 4.45, already in
requirements.lock) with identical weights. If it passes, the from-scratch
implementation is numerically validated, not just paper-derived.
"""

import inspect

import pytest
import torch

from brain2qwerty_colab.mamba import (
    HybridMambaEncoder,
    Mamba2Mixer,
    MambaHybridCore,
)


# --------------------------------------------------------------------------- #
# Small shared fixtures
# --------------------------------------------------------------------------- #
D_MODEL, T, B = 128, 37, 2
MIXER_KW = dict(d_state=32, headdim=16, expand=2, d_conv=4, ngroups=1, dropout=0.0)


def _make_mixer(**over):
    kw = {**MIXER_KW, **over}
    return Mamba2Mixer(D_MODEL, **kw).eval()


# --------------------------------------------------------------------------- #
# 1. Shapes and autograd
# --------------------------------------------------------------------------- #
def test_mixer_shapes_and_backward():
    mixer = _make_mixer()
    x = torch.randn(B, T, D_MODEL, requires_grad=True)
    y = mixer(x)
    assert y.shape == (B, T, D_MODEL)
    y.sum().backward()
    assert torch.isfinite(x.grad).all()
    for name, p in mixer.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), f"bad grad: {name}"


def test_stack_shapes_and_backward():
    # dim 256 / heads 4 -> dim_head 64 (>= 32, required by x-transformers rotary)
    # MIXER_KW carries its own 'dropout' (the mixer's internal dropout), which
    # is a different knob from the block-level `dropout` below -- splat only
    # the mixer-config keys so the two don't collide as duplicate kwargs.
    mixer_kw = {k: v for k, v in MIXER_KW.items() if k != "dropout"}
    stack = HybridMambaEncoder(dim=256, n_layer=8, attention_every=4, heads=4,
                               ff_mult=1, dropout=0.0, **mixer_kw).eval()
    x = torch.randn(B, T, 256, requires_grad=True)
    y = stack(x)
    assert y.shape == (B, T, 256)
    y.sum().backward()
    assert torch.isfinite(x.grad).all()
    # block pattern: M M M A M M M A
    types = [type(b).__name__ for b in stack.blocks]
    assert types == ["MambaBlock"] * 3 + ["AttentionBlock"] + \
                    ["MambaBlock"] * 3 + ["AttentionBlock"]


def test_config_build():
    """The pydantic config builds the stack the way neuraltrain cores do."""
    cfg = MambaHybridCore(n_layer=8, attention_every=4, heads=4)
    stack = cfg.build(dim=256)
    assert isinstance(stack, HybridMambaEncoder)
    y = stack(torch.randn(B, T, 256))
    assert y.shape == (B, T, 256)


# --------------------------------------------------------------------------- #
# 2. Causality: future frames must not influence earlier outputs
# --------------------------------------------------------------------------- #
def test_causality():
    torch.manual_seed(0)
    mixer = _make_mixer()
    x1 = torch.randn(B, T, D_MODEL)
    x2 = x1.clone()
    t0 = T // 2
    x2[:, t0:, :] = torch.randn(B, T - t0, D_MODEL)  # scramble the future
    with torch.no_grad():
        y1, y2 = mixer(x1), mixer(x2)
    assert torch.allclose(y1[:, :t0], y2[:, :t0], atol=1e-5), \
        "mixer is not causal: future frames leak into past outputs"


# --------------------------------------------------------------------------- #
# 3. Determinism
# --------------------------------------------------------------------------- #
def test_determinism():
    torch.manual_seed(1)
    mixer = _make_mixer()
    x = torch.randn(B, T, D_MODEL)
    with torch.no_grad():
        assert torch.equal(mixer(x), mixer(x))


# --------------------------------------------------------------------------- #
# 4. Parity with the HuggingFace reference Mamba-2
# --------------------------------------------------------------------------- #
def test_hf_parity():
    """Numerical parity vs transformers' Mamba2Mixer (identical weights).

    transformers==4.52.4 ships a pure-PyTorch Mamba-2 path, so no mamba-ssm
    install is needed. Our parameter names mirror the reference, so the state
    dict copies directly. Tolerance is loose enough for float32 reassociation
    differences between the two evaluation orders.
    """
    pytest.importorskip("transformers")
    from transformers.models.mamba2.configuration_mamba2 import Mamba2Config
    from transformers.models.mamba2.modeling_mamba2 import (
        Mamba2Mixer as HFMamba2Mixer,
    )

    d_inner = MIXER_KW["expand"] * D_MODEL
    nheads = d_inner // MIXER_KW["headdim"]

    cfg = Mamba2Config(
        hidden_size=D_MODEL,
        state_size=MIXER_KW["d_state"],
        head_dim=MIXER_KW["headdim"],
        num_heads=nheads,
        expand=MIXER_KW["expand"],
        conv_kernel=MIXER_KW["d_conv"],
        n_groups=MIXER_KW["ngroups"],
        chunk_size=64,
        hidden_act="silu",
        layer_norm_epsilon=1e-6,
        use_cache=False,
    )
    ref = HFMamba2Mixer(cfg, layer_idx=0).eval()
    # HF's MambaRMSNormGated normalises over the *full* d_inner dim (no
    # per-head/per-group split -- see transformers modeling_mamba2.py:
    # `variance = hidden_states.pow(2).mean(-1, keepdim=True)` over the whole
    # last axis). With ngroups=1 there's no grouping on the reference side,
    # so norm_group_size must be None to match, not headdim.
    mine = _make_mixer(norm_group_size=None)

    # Copy weights (parameter names match by design)
    with torch.no_grad():
        mine.in_proj.weight.copy_(ref.in_proj.weight)
        mine.conv1d.weight.copy_(ref.conv1d.weight)
        mine.conv1d.bias.copy_(ref.conv1d.bias)
        mine.dt_bias.copy_(ref.dt_bias)
        mine.A_log.copy_(ref.A_log)
        mine.D.copy_(ref.D)
        mine.norm.weight.copy_(ref.norm.weight)
        mine.out_proj.weight.copy_(ref.out_proj.weight)

    x = torch.randn(B, T, D_MODEL)
    with torch.no_grad():
        sig = inspect.signature(ref.forward)
        kwargs = {}
        if "attention_mask" in sig.parameters:
            kwargs["attention_mask"] = torch.ones(B, T, dtype=torch.long)
        ref_out = ref(x, **kwargs)
        my_out = mine(x)

    assert ref_out.shape == my_out.shape
    max_err = (ref_out - my_out).abs().max().item()
    assert torch.allclose(ref_out, my_out, atol=1e-3, rtol=1e-2), \
        f"parity failed: max abs error {max_err:.2e}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))