# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the BiMamba sentence core (no data, no GPU needed).

Run from the repo root:

    pytest brain2qwerty_v1_mamba/tests/test_bimamba.py -v

These verify the properties the V1 ablation depends on:
  1. shape/interface parity with V1's transformer: forward(x, mask) -> (B, T, D)
  2. padding invariance: padded zeros must not change real positions' outputs
     (critical: a contaminated backward direction would silently corrupt the
     sentence embeddings)
  3. bidirectionality: changing a LATER keystroke must change EARLIER outputs
     (V1's transformer has this; a causal-only Mamba would fail)
  4. gradient flow through the whole stack
"""

import torch

from ..mamba_core import BiMamba3SentenceCoreModule, BiMambaSentenceCoreModule

DIM, N_LAYER = 64, 2  # tiny config for CPU-speed tests


def _core() -> BiMambaSentenceCoreModule:
    torch.manual_seed(0)
    return BiMambaSentenceCoreModule(
        dim=DIM,
        n_layer=N_LAYER,
        dropout=0.0,
        d_state=16,
        headdim=16,
        expand=2,
        d_conv=4,
        ngroups=1,
        head_chunk=2,
    )


def _core3() -> BiMamba3SentenceCoreModule:
    torch.manual_seed(0)
    return BiMamba3SentenceCoreModule(
        dim=DIM,
        n_layer=N_LAYER,
        dropout=0.0,
        d_state=16,
        headdim=16,
        expand=2,
        d_conv=4,
        ngroups=1,
        head_chunk=2,
    )


def test_interface_and_shape():
    core = _core()
    x = torch.randn(3, 10, DIM)
    mask = torch.zeros(3, 10, dtype=torch.bool)
    mask[0, :7] = True  # sentence of 7 keystrokes
    mask[1, :10] = True  # sentence of 10
    mask[2, :4] = True  # sentence of 4
    out = core(x, mask=mask)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_padding_invariance():
    """Real positions must be unaffected by whatever sits in the padding."""
    core = _core().eval()
    torch.manual_seed(1)
    x = torch.randn(2, 12, DIM)
    mask = torch.zeros(2, 12, dtype=torch.bool)
    mask[0, :5] = True
    mask[1, :8] = True
    with torch.no_grad():
        out = core(x, mask=mask)
        # corrupt the padded region and re-run: real outputs must not move
        x2 = x.clone()
        x2[0, 5:] = 999.0
        x2[1, 8:] = -999.0
        out2 = core(x2, mask=mask)
    assert torch.allclose(out[0, :5], out2[0, :5], atol=1e-5)
    assert torch.allclose(out[1, :8], out2[1, :8], atol=1e-5)
    # padded positions stay zero (they are discarded by the caller anyway)
    assert (out[0, 5:] == 0).all()
    assert (out[1, 8:] == 0).all()


def test_bidirectionality():
    """A change at the LAST keystroke must affect the FIRST output — the
    property V1's bidirectional transformer has and a causal Mamba lacks."""
    core = _core().eval()
    torch.manual_seed(2)
    x = torch.randn(1, 8, DIM)
    with torch.no_grad():
        y1 = core(x)
        x2 = x.clone()
        x2[0, -1] += 1.0  # perturb the final keystroke only
        y2 = core(x2)
    assert not torch.allclose(y1[0, 0], y2[0, 0]), (
        "first output did not change when the last input changed — "
        "the core is not bidirectional"
    )


def test_gradient_flow():
    core = _core()
    x = torch.randn(2, 6, DIM, requires_grad=True)
    mask = torch.ones(2, 6, dtype=torch.bool)
    loss = core(x, mask=mask).pow(2).mean()
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    grads = [
        p.grad for p in core.parameters() if p.requires_grad and p.grad is not None
    ]
    assert grads, "no parameter received a gradient"
    assert all(torch.isfinite(g).all() for g in grads)


def test_unmasked_forward():
    """mask=None path (full sequences) must also work and be finite."""
    core = _core()
    out = core(torch.randn(4, 9, DIM))
    assert out.shape == (4, 9, DIM)
    assert torch.isfinite(out).all()


# --------------------------------------------------------------------------- #
# Mamba-3-style variant (BCNorm + biases + data-dependent RoPE)
# --------------------------------------------------------------------------- #
def test_v3_interface_and_shape():
    core = _core3()
    x = torch.randn(3, 10, DIM)
    mask = torch.zeros(3, 10, dtype=torch.bool)
    mask[0, :7] = True
    mask[1, :10] = True
    mask[2, :4] = True
    out = core(x, mask=mask)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_v3_padding_invariance():
    core = _core3().eval()
    torch.manual_seed(1)
    x = torch.randn(2, 12, DIM)
    mask = torch.zeros(2, 12, dtype=torch.bool)
    mask[0, :5] = True
    mask[1, :8] = True
    with torch.no_grad():
        out = core(x, mask=mask)
        x2 = x.clone()
        x2[0, 5:] = 999.0
        x2[1, 8:] = -999.0
        out2 = core(x2, mask=mask)
    assert torch.allclose(out[0, :5], out2[0, :5], atol=1e-5)
    assert torch.allclose(out[1, :8], out2[1, :8], atol=1e-5)


def test_v3_bidirectionality():
    core = _core3().eval()
    torch.manual_seed(2)
    x = torch.randn(1, 8, DIM)
    with torch.no_grad():
        y1 = core(x)
        x2 = x.clone()
        x2[0, -1] += 1.0
        y2 = core(x2)
    assert not torch.allclose(y1[0, 0], y2[0, 0])


def test_v3_gradient_flow():
    core = _core3()
    x = torch.randn(2, 6, DIM, requires_grad=True)
    mask = torch.ones(2, 6, dtype=torch.bool)
    loss = core(x, mask=mask).pow(2).mean()
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    grads = [
        p.grad for p in core.parameters() if p.requires_grad and p.grad is not None
    ]
    assert grads, "no parameter received a gradient"
    assert all(torch.isfinite(g).all() for g in grads)
    # the Mamba-3 extras (BC biases) must actually receive gradients
    named = dict(core.named_parameters())
    for frag in ("b_bias", "c_bias"):
        params = [p for n, p in named.items() if frag in n]
        assert params, f"no parameter matching {frag!r}"
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)


def test_v3_config_build():
    """The registry path used by the experiment config must work."""
    from ..mamba_core import BiMamba3SentenceCore

    cfg = BiMamba3SentenceCore(
        n_layer=2, d_state=16, headdim=16, expand=2, d_conv=4,
        ngroups=1, head_chunk=2, dropout=0.0,
    )
    module = cfg.build(dim=DIM)
    out = module(torch.randn(2, 7, DIM))
    assert out.shape == (2, 7, DIM)
    assert torch.isfinite(out).all()


def test_hybrid_core_and_alibi():
    """Test Nemotron-H style Hybrid Mamba-Attention Core."""
    from ..mamba_core import HybridSentenceCore, HybridMamba3SentenceCore

    for core_cls in (HybridSentenceCore, HybridMamba3SentenceCore):
        cfg = core_cls(
            n_layer=4, attention_every=4, heads=4, d_state=16, headdim=16,
            expand=2, d_conv=4, ngroups=1, head_chunk=2, dropout=0.0,
        )
        module = cfg.build(dim=DIM)
        x = torch.randn(2, 8, DIM)
        out = module(x)
        assert out.shape == (2, 8, DIM)
        assert torch.isfinite(out).all()

        # Gradient flow test
        loss = out.pow(2).mean()
        loss.backward()
        grads = [p.grad for p in module.parameters() if p.requires_grad and p.grad is not None]
        assert len(grads) > 0
        assert all(torch.isfinite(g).all() for g in grads)


def test_mamba_mlp_and_gated_fusion():
    """Test BiMambaBlock with FFN MLP sublayer and learned gated fusion."""
    from ..mamba_core import BiMambaSentenceCore

    cfg = BiMambaSentenceCore(
        n_layer=2, d_state=16, headdim=16, expand=2, d_conv=4,
        ngroups=1, head_chunk=2, dropout=0.0, use_mlp=True, gated_fusion=True,
    )
    module = cfg.build(dim=DIM)
    x = torch.randn(2, 6, DIM)
    out = module(x)
    assert out.shape == (2, 6, DIM)
    assert torch.isfinite(out).all()

