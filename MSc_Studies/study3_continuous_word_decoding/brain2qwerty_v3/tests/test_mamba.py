"""Correctness, parity, and shape tests for Brain2Qwerty V3 Sequence Cores."""

import inspect
import pytest
import torch

from brain2qwerty_v3.mamba import (
    BiMambaGatedMLP,
    BiMambaGatedMLPEncoder,
    HybridMambaEncoder,
    Mamba2Mixer,
    Mamba3Mixer,
    Mamba3StabilizedHybrid,
    MambaHybrid,
)

D_MODEL, T, B = 128, 37, 2
MIXER_KW = dict(d_state=32, headdim=16, expand=2, d_conv=4, ngroups=1, dropout=0.0)


def _make_mixer(**over):
    kw = {**MIXER_KW, **over}
    return Mamba2Mixer(D_MODEL, **kw).eval()


def test_mixer_shapes_and_backward():
    mixer = _make_mixer()
    x = torch.randn(B, T, D_MODEL, requires_grad=True)
    y = mixer(x)
    assert y.shape == (B, T, D_MODEL)
    y.sum().backward()
    assert torch.isfinite(x.grad).all()


def test_mamba3_stabilized_mixer():
    mixer = Mamba3Mixer(D_MODEL, d_state=32, headdim=16, expand=2, d_conv=4, ngroups=1).eval()
    x = torch.randn(B, T, D_MODEL, requires_grad=True)
    y = mixer(x)
    assert y.shape == (B, T, D_MODEL)
    assert torch.isfinite(y).all()
    y.sum().backward()
    assert torch.isfinite(x.grad).all()


def test_bimamba_gated_mlp_encoder():
    cfg = BiMambaGatedMLP(n_layer=4, d_state=32, headdim=16, expand=2, d_conv=4, ngroups=1, ff_mult=2)
    module = cfg.build(dim=D_MODEL)
    assert isinstance(module, BiMambaGatedMLPEncoder)
    x = torch.randn(B, T, D_MODEL, requires_grad=True)
    y = module(x)
    assert y.shape == (B, T, D_MODEL)
    assert torch.isfinite(y).all()
    y.sum().backward()
    assert torch.isfinite(x.grad).all()


def test_mamba3_stabilized_hybrid_encoder():
    cfg = Mamba3StabilizedHybrid(n_layer=4, attention_every=2, heads=4, d_state=32, headdim=16, expand=2)
    module = cfg.build(dim=256)
    assert isinstance(module, HybridMambaEncoder)
    x = torch.randn(B, T, 256, requires_grad=True)
    y = module(x)
    assert y.shape == (B, T, 256)
    assert torch.isfinite(y).all()
    y.sum().backward()
    assert torch.isfinite(x.grad).all()


def test_hf_parity():
    pytest.importorskip("transformers")
    from transformers.models.mamba2.configuration_mamba2 import Mamba2Config
    from transformers.models.mamba2.modeling_mamba2 import Mamba2Mixer as HFMamba2Mixer

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
    mine = _make_mixer(norm_group_size=None)

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
    assert torch.allclose(ref_out, my_out, atol=1e-3, rtol=1e-2)
