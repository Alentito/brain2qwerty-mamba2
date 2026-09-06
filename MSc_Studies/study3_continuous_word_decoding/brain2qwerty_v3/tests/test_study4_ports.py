"""Study-4 port tests: GNN continuous frontend + DeltaNet CTC core for V3.

No data, no GPU, no LLM downloads — encoder-level builds only.

Run from the repo root:

    pytest brain2qwerty_v3/tests/test_study4_ports.py -v

Coverage:
  1. GnnContinuousEncoder: shapes with T EXACTLY preserved (the
     compute_output_lens contract), (C,2)/(B,C,2)/None position handling,
     finite gradients, and chunk equivalence (t_chunk must not change the
     output — chunking is a memory device, not architecture).
  2. Whole-model builds: ConvMambaHybrid with frontend="gnn" (and
     core="deltanet") via the config dict; z_enc length must equal
     (T - 16) // 4 + 1 from the temporal downsampling conv alone.
  3. BiDeltaNetCTCCore: (B,T,D)->(B,T,D) no-mask contract, finite gradients,
     and strict weight-copy equivalence with v1_mamba's
     BiDeltaNetSentenceCore mask=None path.
  4. Registry: v1_mamba and v3 importable in the same process (no exca
     name collision); both new config names resolve through
     BaseModelConfig(**{"name": ...}).
  5. Config wiring: experiment_config(core="deltanet", frontend="gnn") is
     sane; conv defaults are byte-identical to the pre-Study-4 behaviour.
"""

import pytest
import torch

from neuraltrain.models.base import BaseModelConfig

from brain2qwerty_v3.config.model_config import build_encoder_config
from brain2qwerty_v3.deltanet import BiDeltaNetCTCCore, DeltaNetCTCCoreModule
from brain2qwerty_v3.gnn_frontend import (
    GnnContinuousEncoder,
    GnnContinuousEncoderModel,
    adjacency_to_neighbors,
)
from brain2qwerty_v3.models import ConvMambaHybrid

C = 306  # Elekta channel count


def _frontend(out: int = 512, **overrides) -> GnnContinuousEncoderModel:
    torch.manual_seed(0)
    cfg = dict(d_node=64, n_layers=2, heads=4, k_neighbors=8, conv_mult=4,
               dropout=0.0, t_chunk=128)
    cfg.update(overrides)
    return GnnContinuousEncoder(**cfg).build(n_in_channels=C, n_outputs=out)


def _positions(n: int = C, seed: int = 7) -> torch.Tensor:
    """Random toy sensor layout (no near-duplicates -> no top-k ties)."""
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n, 2, generator=g)


# --------------------------------------------------------------------------- #
# 1. GNN frontend: shapes, length preservation, positions, gradients
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("T", [96, 256])
@pytest.mark.parametrize("positions", ["shared", "batched", "none"])
def test_gnn_frontend_length_preserving(T, positions):
    """(B, C, T) -> (B, dim, T) with T preserved EXACTLY (compute_output_lens
    assumes only the kernel-16/stride-4 downsampling conv changes length)."""
    B, dim = 2, 512
    model = _frontend(out=dim).eval()
    x = torch.randn(B, C, T)
    if positions == "shared":
        pos = _positions()
    elif positions == "batched":
        pos = torch.stack([_positions(seed=50 + i) for i in range(B)])
    else:
        pos = None
    with torch.no_grad():
        out = model(x, None, pos)
    assert out.shape == (B, dim, T), f"T not preserved: {out.shape}"
    assert torch.isfinite(out).all()


def test_gnn_frontend_per_sample_graphs_differ():
    """Different layouts per sample must use per-sample graphs (not collapse)."""
    model = _frontend().eval()
    x = torch.randn(2, C, 96)
    pos_same = _positions().unsqueeze(0).repeat(2, 1, 1)
    pos_diff = torch.stack([_positions(seed=50), _positions(seed=51)])
    idx_same, _ = model.build_neighbors(pos_same, 2, x.device)
    idx_diff, _ = model.build_neighbors(pos_diff, 2, x.device)
    assert idx_same.dim() == 2  # identical layouts collapse to a shared graph
    assert idx_diff.dim() == 3  # differing layouts keep per-sample graphs


def test_gnn_frontend_gradient_flow():
    model = _frontend()
    model.train()
    x = torch.randn(2, C, 96, requires_grad=True)
    out = model(x, None, _positions())
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    n_graded = sum(
        1 for p in model.parameters() if p.grad is not None and torch.isfinite(p.grad).all()
    )
    n_total = sum(1 for p in model.parameters() if p.requires_grad)
    assert n_graded == n_total, f"only {n_graded}/{n_total} params have finite grads"


@pytest.mark.parametrize("positions", ["shared", "batched", "none"])
def test_gnn_frontend_chunk_equivalence(positions):
    """t_chunk is a memory device only: same weights, eval mode, chunking the
    frame axis must not change the output."""
    B, T = 2, 96
    model = _frontend().eval()
    x = torch.randn(B, C, T)
    if positions == "shared":
        pos = _positions()
    elif positions == "batched":  # differing layouts -> per-sample graphs
        pos = torch.stack([_positions(seed=50), _positions(seed=51)])
    else:
        pos = None
    with torch.no_grad():
        model.t_chunk = T  # single chunk
        full = model(x, None, pos)
        model.t_chunk = 32
        chunked = model(x, None, pos)
    assert torch.allclose(full, chunked, atol=1e-5), (
        f"max abs diff {(full - chunked).abs().max().item():.2e}"
    )


def test_adjacency_to_neighbors_pads_safely():
    """Rows with different degrees get padded; padded slots are masked out."""
    from brain2qwerty_v1_mamba.gnn_encoder import knn_adjacency

    pos = _positions()
    adj = knn_adjacency(pos, 8)
    idx, mask = adjacency_to_neighbors(adj)
    assert idx.shape == mask.shape == (C, 9)  # k+1 incl. self
    assert mask.all()  # no padding needed in the non-degenerate case
    # each row lists exactly its True neighbors
    for i in (0, 100, 305):
        assert set(idx[i].tolist()) == set(adj[i].nonzero().flatten().tolist())
        assert i in idx[i].tolist()  # self-loop


# --------------------------------------------------------------------------- #
# 2. Whole-model: compute_output_lens end-to-end
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("core", ["conformer", "deltanet"])
@pytest.mark.parametrize("with_positions", [True, False])
def test_whole_model_gnn_frontend_output_lens(core, with_positions):
    """ConvMambaHybrid(frontend='gnn', small): a synthetic (2, 300, 306)
    sentence batch must yield z_enc/z_final/c_out with exactly
    (300 - 16) // 4 + 1 frames — the number compute_output_lens produces
    from the temporal downsampling conv alone."""
    from brain2qwerty_v3.utils import compute_output_lens

    cfg_dict = build_encoder_config(core=core, small=True, frontend="gnn")
    cfg = ConvMambaHybrid(**cfg_dict)
    assert cfg.encoder_config.__class__.__name__ == "GnnContinuousEncoder"
    model = cfg.build(n_in_channels=C, n_outputs=29)
    model.eval()

    B, T = 2, 300
    x = torch.randn(B, T, C)
    days = torch.zeros(B, dtype=torch.long)
    chan_pos = torch.stack([_positions(seed=50), _positions(seed=51)]) if with_positions else None
    with torch.no_grad():
        out = model(x, days, chan_pos)
        out_lens = compute_output_lens(model, torch.tensor([T, T]))

    expect = (T - 16) // 4 + 1
    assert out_lens.tolist() == [expect, expect]
    assert out["z_enc"].shape[1] == expect, f"[{core}] z_enc len {out['z_enc'].shape[1]} != {expect}"
    assert out["z_final"].shape[1] == expect
    assert out["c_out"].shape[1] == expect
    assert out["c_out"].shape == (B, expect, 29)
    assert torch.isfinite(out["c_out"]).all()
    assert torch.isfinite(out["z_final"]).all()


def test_whole_model_conv_frontend_regression():
    """The conv frontend path is untouched: same dict, same shapes as before."""
    cfg_dict = build_encoder_config(core="conformer", small=True)
    assert cfg_dict["encoder_config"]["name"] == "SimpleConv"
    assert cfg_dict["encoder_config"]["hidden"] == 750
    assert cfg_dict["encoder_config"]["merger_config"]["fourier_emb_config"]["total_dim"] == 512
    cfg = ConvMambaHybrid(**cfg_dict)
    model = cfg.build(n_in_channels=C, n_outputs=29)
    model.eval()
    x = torch.randn(2, 300, C)
    days = torch.zeros(2, dtype=torch.long)
    chan_pos = torch.randn(2, C, 2)
    with torch.no_grad():
        out = model(x, days, chan_pos)
    assert out["z_enc"].shape[1] == (300 - 16) // 4 + 1


def test_full_width_gnn_build():
    """Full width (dim=1024, d_node=256) must build and divide heads evenly."""
    cfg_dict = build_encoder_config(core="deltanet", small=False, frontend="gnn")
    cfg = ConvMambaHybrid(**cfg_dict)
    assert cfg.dim == 1024
    assert cfg.encoder_config.d_node == 256
    model = cfg.build(n_in_channels=C, n_outputs=29)
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(1, 96, C), torch.zeros(1, dtype=torch.long), _positions())
    assert out["z_enc"].shape[0] == 1


# --------------------------------------------------------------------------- #
# 3. DeltaNet CTC core
# --------------------------------------------------------------------------- #
def test_deltanet_core_shapes_and_gradients():
    """(B, T, D) -> (B, T, D), no mask, finite gradients everywhere."""
    torch.manual_seed(0)
    core = BiDeltaNetCTCCore(n_layer=2, dropout=0.0, headdim=64, expand=1).build(512)
    core.train()
    x = torch.randn(2, 40, 512, requires_grad=True)
    y = core(x)
    assert y.shape == (2, 40, 512)
    assert torch.isfinite(y).all()
    y.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for name, p in core.named_parameters():
        assert p.grad is not None, f"no gradient for {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite gradient for {name}"


def test_deltanet_core_matches_v1mamba_mask_none_path():
    """With weights copied (strict load), DeltaNetCTCCoreModule must produce
    bit-identical output to v1_mamba's BiDeltaNetSentenceCore mask=None path —
    the V3 core is that path, minus the per-sentence unpadding loop."""
    from brain2qwerty_v1_mamba.deltanet_core import BiDeltaNetSentenceCore

    torch.manual_seed(0)
    ref = BiDeltaNetSentenceCore(n_layer=2, dropout=0.0, headdim=16, expand=1).build(64)
    torch.manual_seed(1)  # different seed: equivalence must come from the copy
    new = BiDeltaNetCTCCore(n_layer=2, dropout=0.0, headdim=16, expand=1).build(64)
    assert isinstance(new, DeltaNetCTCCoreModule)
    new.load_state_dict(ref.state_dict(), strict=True)  # same parameter layout

    ref.eval()
    new.eval()
    x = torch.randn(2, 17, 64)
    with torch.no_grad():
        y_ref = ref(x)          # mask=None path of the sentence core
        y_new = new(x)          # the V3 no-mask contract
    assert torch.allclose(y_ref, y_new, atol=1e-6), (
        f"max abs diff {(y_ref - y_new).abs().max().item():.2e}"
    )


def test_deltanet_core_bidirectionality():
    """A change at the LAST frame must affect the FIRST output."""
    torch.manual_seed(2)
    core = BiDeltaNetCTCCore(n_layer=2, dropout=0.0, headdim=16, expand=1).build(64)
    core.eval()
    x = torch.randn(1, 12, 64)
    with torch.no_grad():
        y1 = core(x)
        x2 = x.clone()
        x2[0, -1] += 1.0
        y2 = core(x2)
    assert not torch.allclose(y1[0, 0], y2[0, 0]), "core is not bidirectional"


# --------------------------------------------------------------------------- #
# 4. Registry: both packages in one process, no exca collision
# --------------------------------------------------------------------------- #
def test_registry_no_collision_and_name_resolution():
    """Importing v1_mamba.main AND v3.main in the same process must not raise
    exca's duplicate-class-name error, and the two new Study-4 config names
    must resolve through the BaseModelConfig registry."""
    import brain2qwerty_v1_mamba.main  # noqa: F401
    import brain2qwerty_v3.main  # noqa: F401

    gnn = BaseModelConfig(**{"name": "GnnContinuousEncoder"})
    assert isinstance(gnn, GnnContinuousEncoder)
    delta = BaseModelConfig(**{"name": "BiDeltaNetCTCCore", "n_layer": 8})
    assert isinstance(delta, BiDeltaNetCTCCore)

    # the v1_mamba names still resolve to the v1_mamba classes
    from brain2qwerty_v1_mamba.deltanet_core import BiDeltaNetSentenceCore
    from brain2qwerty_v1_mamba.gnn_encoder import GnnWindowEncoder

    assert isinstance(BaseModelConfig(**{"name": "GnnWindowEncoder"}), GnnWindowEncoder)
    assert isinstance(
        BaseModelConfig(**{"name": "BiDeltaNetSentenceCore"}), BiDeltaNetSentenceCore
    )


# --------------------------------------------------------------------------- #
# 5. Config wiring
# --------------------------------------------------------------------------- #
def test_experiment_config_study4_combo():
    from brain2qwerty_v3.config.xp_config import experiment_config

    cfg = experiment_config(core="deltanet", frontend="gnn", small=True, subjects=["S16"])
    bm = cfg["brain_model_config"]
    assert bm["encoder_config"]["name"] == "GnnContinuousEncoder"
    assert bm["encoder_config"]["d_node"] == 128  # small width
    assert bm["transformer_config"]["name"] == "BiDeltaNetCTCCore"
    assert bm["transformer_config"]["n_layer"] == 8  # matches the other v3 cores
    assert bm["dim"] == 512
    assert cfg["optimizer_config"]["lr"] == 3e-4  # deltanet uses the mamba LR
    assert "S16" in cfg["data"]["study"]["query"]


def test_experiment_config_conv_defaults_regression():
    """Conv naming/values must be exactly what they were before Study 4."""
    from brain2qwerty_v3.config.xp_config import experiment_config

    cfg = experiment_config(core="mamba3_hybrid_stabilized", small=True)
    bm = cfg["brain_model_config"]
    assert bm["name"] == "ConvMambaHybrid"
    assert bm["encoder_config"]["name"] == "SimpleConv"
    assert bm["encoder_config"]["hidden"] == 750
    assert bm["encoder_config"]["initial_linear"] == 256
    assert bm["encoder_config"]["merger_config"]["fourier_emb_config"]["total_dim"] == 512
    assert bm["transformer_config"]["name"] == "Mamba3StabilizedHybrid"
    assert bm["temporal_downsampling_config"] == {"kernel_size": 16, "stride": 4}
    assert cfg["optimizer_config"]["lr"] == 3e-4  # mamba heuristic unchanged

    cfg_conf = experiment_config(core="conformer")
    assert cfg_conf["optimizer_config"]["lr"] == 8e-4  # conformer LR unchanged
    assert cfg_conf["brain_model_config"]["encoder_config"]["hidden"] == 1500
    assert (
        cfg_conf["brain_model_config"]["encoder_config"]["merger_config"]
        ["fourier_emb_config"]["total_dim"] == 2048
    )


def test_output_dir_tag_rule():
    """conv naming is exactly ``v3-<core>-<tag>``; gnn arms get the infix."""
    from brain2qwerty_v3.main import tagged_output_dir

    assert tagged_output_dir("conformer", "conv", "study4").endswith("v3-conformer-study4")
    assert tagged_output_dir("deltanet", "conv", "study4").endswith("v3-deltanet-study4")
    assert tagged_output_dir("conformer", "gnn", "study4").endswith("v3-gnn-conformer-study4")
    assert tagged_output_dir("deltanet", "gnn", "study4").endswith("v3-gnn-deltanet-study4")


def test_unknown_frontend_and_core_raise():
    with pytest.raises(ValueError, match="Unknown frontend"):
        build_encoder_config(core="conformer", frontend="transformer")
    with pytest.raises(ValueError, match="Unknown core"):
        build_encoder_config(core="gnn")
