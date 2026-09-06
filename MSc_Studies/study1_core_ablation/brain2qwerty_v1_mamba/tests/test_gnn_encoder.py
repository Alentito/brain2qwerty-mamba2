# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the GNN window encoder (Stage 1) — no data, no GPU needed.

Run from the repo root:

    pytest brain2qwerty_v1_mamba/tests/test_gnn_encoder.py -v

These verify the properties the Stage-1 ablation depends on:
  1. forward shapes / interface parity with SimpleConvTimeAgg:
     (B, 306, 25) + (C,2) | (B,C,2) | None positions -> (B, hidden)
  2. k-NN graph correctness (neighbor sets = true nearest neighbors + self)
  3. gradient flow through the whole stack
  4. the config registry path used by experiment_config (name resolution +
     build at both 512 and 2048 width) and the output-dir tag rule
  5. position sensitivity: different layouts -> different adjacency
  6. Stage-1 -> Stage-2 compatibility: GNN embeddings feed the DeltaNet
     sentence core through the same padded/masked interface as V1
"""

import torch

from neuraltrain.models.base import BaseModelConfig

from ..deltanet_core import BiDeltaNetSentenceCore
from ..gnn_encoder import GnnWindowEncoder, GnnWindowEncoderModel, knn_adjacency

C, T = 306, 25  # Elekta channels, 500 ms @ 50 Hz


def _encoder(hidden: int = 512, **overrides) -> GnnWindowEncoderModel:
    torch.manual_seed(0)
    cfg = dict(hidden=hidden, d_node=64, n_layers=2, heads=4,
               k_neighbors=8, dropout=0.0)
    cfg.update(overrides)
    return GnnWindowEncoder(**cfg).build(n_in_channels=C, n_outputs=hidden)


def _positions(n: int = C, seed: int = 7) -> torch.Tensor:
    """Random toy sensor layout (no near-duplicates -> no top-k ties)."""
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n, 2, generator=g)


# --------------------------------------------------------------------------- #
# 1. Forward shapes / interface
# --------------------------------------------------------------------------- #
def test_forward_shapes():
    model = _encoder().eval()
    assert model.out_channels == 512
    x = torch.randn(4, C, T)
    pos_shared = _positions()
    pos_batched = torch.stack([_positions(seed=50 + i) for i in range(4)])
    with torch.no_grad():
        out_shared = model(x, None, pos_shared)
        out_batched = model(x, None, pos_batched)
        out_none = model(x, None, None)
        # subject_ids are accepted (and ignored) like SimpleConvTimeAgg
        out_subj = model(x, torch.zeros(4, dtype=torch.long), pos_shared)
    for out in (out_shared, out_batched, out_none, out_subj):
        assert out.shape == (4, 512)
        assert torch.isfinite(out).all()
    assert torch.equal(out_shared, out_subj)  # subject_ids must not matter


def test_batched_identical_positions_collapse_to_shared_graph():
    """(B, C, 2) with identical layouts must collapse to one shared graph."""
    model = _encoder()
    pos = _positions()
    batched_same = pos.unsqueeze(0).repeat(4, 1, 1)
    adj = model.build_adjacency(batched_same, batch_size=4)
    assert adj.shape == (1, C, C)
    assert torch.equal(adj[0], knn_adjacency(pos, model.k_neighbors))
    # genuinely different layouts -> per-sample graphs
    batched_diff = torch.stack([_positions(seed=100 + i) for i in range(4)])
    assert model.build_adjacency(batched_diff, batch_size=4).shape == (4, C, C)


# --------------------------------------------------------------------------- #
# 2. k-NN graph sanity
# --------------------------------------------------------------------------- #
def test_knn_graph_sanity():
    k = 8
    pos = _positions(60)
    adj = knn_adjacency(pos, k)  # (60, 60) bool
    dist = torch.cdist(pos, pos)
    expected = dist.argsort(dim=-1)[:, : k + 1]  # self is always closest
    for i in range(pos.size(0)):
        neigh = adj[i].nonzero().flatten()
        assert neigh.numel() == k + 1, f"node {i}: {neigh.numel()} neighbors"
        assert set(neigh.tolist()) == set(expected[i].tolist())
        assert adj[i, i], f"node {i} missing self-loop"


# --------------------------------------------------------------------------- #
# 3. Gradient flow
# --------------------------------------------------------------------------- #
def test_gradient_flow():
    model = _encoder()
    x = torch.randn(2, C, T, requires_grad=True)
    loss = model(x, None, _positions()).sum()
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"no gradient for {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite gradient for {name}"


# --------------------------------------------------------------------------- #
# 4. Config / registry / output-tag wiring
# --------------------------------------------------------------------------- #
def test_config_wiring_and_tags():
    from ..config.model_config import gnn_encoder
    from ..config.xp_config import experiment_config

    xp = experiment_config(subjects=["S16"], core="mamba", small=True,
                           encoder_kind="gnn")
    cfg = xp["brain_model_config"]
    assert cfg["name"] == "GnnWindowEncoder"
    obj = BaseModelConfig(**cfg)  # registry resolution by class name
    assert isinstance(obj, GnnWindowEncoder)
    assert obj.hidden == 512
    model = obj.build(n_in_channels=C, n_outputs=obj.hidden)
    assert model.out_channels == 512
    out = model(torch.randn(2, C, T), None, _positions())
    assert out.shape == (2, 512) and torch.isfinite(out).all()

    # full width (2048) catches d_node/heads divisibility issues
    obj_full = BaseModelConfig(**gnn_encoder(small=False))
    assert obj_full.hidden == 2048
    model_full = obj_full.build(n_in_channels=C, n_outputs=obj_full.hidden)
    out_full = model_full(torch.randn(1, C, T))
    assert out_full.shape == (1, 2048) and torch.isfinite(out_full).all()

    # tag rule: gnn inserts "gnn-" after "small-"; conv naming unchanged
    assert xp["output_dir"].endswith("small-gnn-mamba-S16")
    xp_conv = experiment_config(subjects=["S16"], core="mamba", small=True)
    assert xp_conv["output_dir"].endswith("small-mamba-S16")

    # unknown encoder kind is rejected
    try:
        experiment_config(subjects=["S16"], encoder_kind="lstm")
    except ValueError:
        pass
    else:
        raise AssertionError("encoder_kind='lstm' should raise ValueError")


# --------------------------------------------------------------------------- #
# 5. Position sensitivity
# --------------------------------------------------------------------------- #
def test_position_sensitivity():
    """Clearly different layouts must produce different graphs."""
    model = _encoder()
    pos = _positions()
    g = torch.Generator().manual_seed(11)
    shuffled = pos[torch.randperm(C, generator=g)]
    adj_a = model.build_adjacency(pos, batch_size=1)
    adj_b = model.build_adjacency(shuffled, batch_size=1)
    assert not torch.equal(adj_a, adj_b)


# --------------------------------------------------------------------------- #
# 6. Stage-1 -> Stage-2 compatibility
# --------------------------------------------------------------------------- #
def test_stage2_compatibility():
    """GNN keystroke embeddings feed the DeltaNet sentence core through the
    same padded/masked interface as V1's BrainModule._transformer_forward."""
    hidden = 512
    encoder = _encoder(hidden=hidden).eval()
    core = BiDeltaNetSentenceCore(
        n_layer=2, dropout=0.0, headdim=64, expand=1
    ).build(hidden).eval()

    # 6 keystrokes grouped into sentences of length 4 and 2 (as in pl_module)
    x = torch.randn(6, C, T)
    with torch.no_grad():
        emb = encoder(x, None, _positions())  # (6, 512)
    assert emb.shape == (6, hidden)

    grouped = [emb[:4], emb[4:]]
    max_len = max(len(g) for g in grouped)
    padded = torch.zeros(2, max_len, hidden)
    mask = torch.zeros(2, max_len, dtype=torch.bool)
    for i, g in enumerate(grouped):
        padded[i, : len(g)] = g
        mask[i, : len(g)] = 1

    with torch.no_grad():
        out = core(padded, mask=mask)
    assert out.shape == (2, max_len, hidden)
    assert torch.isfinite(out).all()
    assert (out[1, 2:] == 0).all()  # padding stays zero
