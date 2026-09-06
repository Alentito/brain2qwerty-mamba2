# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""GNN continuous frontend for the Brain2Qwerty V3 pipeline (Study 4, Stage 1).

Motivation
----------
V3's default frontend (``SimpleConv`` + per-subject 2D-Fourier channel merger)
must *learn* the 306-sensor Elekta topology from data and ties that topology
to a per-subject embedding table. This module ports the Stage-1 graph
inductive bias of ``brain2qwerty_v1_mamba.gnn_encoder`` to V3's *continuous*
uncut sentence windows: the sensors are nodes of a k-nearest-neighbor graph
built from their physical 2D layout coordinates, and message passing is
restricted to true spatial neighborhoods. Unlike V1's GnnWindowEncoder — which
pools a whole 500 ms keystroke window into one embedding — this frontend is
**per-frame**: it maps (B, C, T) -> (B, dim, T) so the sequence core still
sees one token per (downsampled) time step.

Interface contract (traced through ``ConvTransformerModel``)
------------------------------------------------------------
* Built via ``config.encoder_config.build(in_channels, dim)``.
* Called as ``forward(x, subject_ids=..., channel_positions=...)`` with
  ``x`` of shape ``(B, C, T)``; MUST return **channel-first (B, dim, T)**.
* **Length preservation is a hard contract**: ``brain2qwerty_v3.utils.
  compute_output_lens`` derives CTC output lengths from the temporal
  downsampling conv alone — ``(T - 16) // 4 + 1`` — so this frontend's output
  length must equal its input length exactly, or every CTC length is wrong.
  The depthwise temporal conv uses kernel 5 / padding 2 ("same") and all
  pooling is over the sensor axis only.
* ``subject_ids`` is accepted and intentionally ignored: the graph is purely
  spatial and the Elekta 306 layout is identical across subjects — unlike the
  conv frontend's per-subject Fourier merger, which is exactly what this
  ablation isolates.
* ``channel_positions`` handling is identical to V1's GNN encoder:
  ``(C, 2)`` -> one graph shared by the batch; ``(B, C, 2)`` -> per-sample
  graphs only when layouts actually differ (V3's dataloader always supplies
  ``(B, C, 2)`` via the ``chan_pos`` extractor); ``None`` -> fully-connected
  graph (graceful degradation).

Architecture (per frame)
------------------------
1. Depthwise temporal Conv1d (groups=C, kernel 5, padding 2, C -> C*conv_mult)
   + GELU: short per-sensor temporal features, length preserved.
2. Node embedding: Linear(conv_mult -> d_node) per frame/sensor + learned
   per-sensor embedding (C, d_node).
3. ``n_layers`` GAT-style blocks restricted to the k-NN neighborhood
   (+ self-loops). Same additive-attention math as V1's
   ``GraphAttentionBlock`` (reused graph construction via ``knn_adjacency``),
   but implemented with *neighbor gathering* — scores/values are
   (N, C, k+1, ·) tensors, never the dense (N, C, C, ·) attention matrix.
4. Per-frame readout: LayerNorm -> concat(mean-pool, max-pool over sensors)
   -> Linear(2*d_node -> dim); output (B, T, dim) -> transpose -> (B, dim, T).

Memory
------
Frames are processed in chunks of ``t_chunk`` along T: the node batch is
N = B * t_chunk, so the (N, C, k+1, d_node) gather intermediates are bounded
independently of the (up to 1280-frame) window length. Chunking is exact —
frames never interact inside this frontend — so changing ``t_chunk`` does not
change the output (covered by a chunk-equivalence test). The fully-connected
``None`` fallback gathers all C neighbors per node (k+1 = C); that is the
same O(N * C^2) cost as V1's dense block, only reached when positions are
unavailable.

Pure PyTorch, float32 where numerically sensitive (attention scores), runs on
CPU / MPS / CUDA.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuraltrain.models.base import BaseModelConfig

# Graph construction is reused verbatim from the V1 ablation; the attention
# block below mirrors GraphAttentionBlock's math in a neighbor-gathered
# (memory-bounded) form. No new registry names are taken from v1_mamba.
from brain2qwerty_v1_mamba.gnn_encoder import knn_adjacency


# --------------------------------------------------------------------------- #
# Adjacency -> padded neighbor-index lists
# --------------------------------------------------------------------------- #
def adjacency_to_neighbors(adj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a bool adjacency (..., C, C) to padded neighbor indices + mask.

    Returns:
        nbr_idx: (..., C, K) long, K = max row degree; padded slots point at
            the node itself (they are masked out, so the target is irrelevant
            as long as it is a valid index — self is always valid).
        nbr_mask: (..., C, K) bool, True for real neighbors.

    Rows of ``knn_adjacency`` have exactly k+1 True entries outside degenerate
    tie cases, so K = k+1 in practice; padding only absorbs near-duplicate
    position edge cases without changing the softmax (masked slots are
    excluded, never duplicated).
    """
    # stable sort: real neighbors (0 = ~adj is False) first, in column order
    order = torch.argsort((~adj).to(torch.uint8), dim=-1, stable=True)
    counts = adj.sum(dim=-1)  # (..., C)
    K = int(counts.max())
    nbr_idx = order[..., :K]
    nbr_mask = adj.gather(-1, nbr_idx)
    # padded slots must be gatherable: point them at the self index
    self_idx = torch.arange(adj.size(-1), device=adj.device)
    self_idx = self_idx.view(*([1] * (nbr_idx.dim() - 2)), -1, 1).expand_as(nbr_idx)
    nbr_idx = torch.where(nbr_mask, nbr_idx, self_idx)
    return nbr_idx, nbr_mask


# --------------------------------------------------------------------------- #
# Neighbor-gathered GAT block (memory-bounded GraphAttentionBlock)
# --------------------------------------------------------------------------- #
class NeighborGraphAttentionBlock(nn.Module):
    """Pre-norm GAT block over explicit neighbor lists.

    Same parameterization and math as
    ``brain2qwerty_v1_mamba.gnn_encoder.GraphAttentionBlock`` — additive
    GAT scores LeakyReLU(a_dst . Wh_i + a_src . Wh_j), softmax over neighbors
    j, pre-norm FFN sublayer, residual + dropout — but the neighborhood is
    given as padded index lists ``nbr_idx``/``nbr_mask`` of shape
    ``(C, K)`` (shared graph) or ``(N, C, K)`` (per-sample graphs), so no
    dense (N, C, C, ·) attention tensor is ever materialized.

    Input/output: node features ``x`` of shape (N, C, D).
    """

    def __init__(self, d_node: int, heads: int, dropout: float, ffn_mult: int = 2):
        super().__init__()
        if d_node % heads != 0:
            raise ValueError(f"d_node={d_node} not divisible by {heads=}")
        self.heads = heads
        self.head_dim = d_node // heads

        self.norm1 = nn.LayerNorm(d_node)
        self.w = nn.Linear(d_node, d_node, bias=False)
        self.a_src = nn.Parameter(torch.empty(heads, self.head_dim))
        self.a_dst = nn.Parameter(torch.empty(heads, self.head_dim))
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)
        self.out_proj = nn.Linear(d_node, d_node, bias=False)

        self.norm2 = nn.LayerNorm(d_node)
        self.ffn = nn.Sequential(
            nn.Linear(d_node, ffn_mult * d_node),
            nn.GELU(),
            nn.Linear(ffn_mult * d_node, d_node),
        )
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        nbr_idx: torch.Tensor,
        nbr_mask: torch.Tensor,
    ) -> torch.Tensor:
        N, C, D = x.shape
        H, P = self.heads, self.head_dim
        h = self.norm1(x)
        wh = self.w(h).view(N, C, H, P)  # (N, C, H, P)

        # additive GAT scores over the gathered neighborhood only
        s_dst = (wh * self.a_dst).sum(-1)  # (N, C, H)   — node i as destination
        s_src = (wh * self.a_src).sum(-1)  # (N, C, H)   — node j as source
        if nbr_idx.dim() == 2:  # (C, K) shared graph
            s_src_nbr = s_src[:, nbr_idx]  # (N, C, K, H)
            mask = nbr_mask  # (C, K)
        else:  # (N, C, K) per-sample graphs
            s_src_nbr = s_src[
                torch.arange(N, device=x.device)[:, None, None], nbr_idx
            ]  # (N, C, K, H)
            mask = nbr_mask  # (N, C, K)
        scores = s_dst.unsqueeze(2) + s_src_nbr  # (N, C, K, H)
        scores = F.leaky_relu(scores.float(), negative_slope=0.2)
        scores = scores.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        alpha = torch.softmax(scores, dim=2)  # over neighbors j
        alpha = self.attn_dropout(alpha).to(wh.dtype)

        # Aggregate slot-by-slot: peak extra memory is one (N, C, H, P)
        # gather per neighbor slot, never the full (N, C, K, H, P) tensor.
        K = nbr_idx.size(-1)
        agg = torch.zeros_like(wh)
        for j in range(K):
            if nbr_idx.dim() == 2:
                v_j = wh[:, nbr_idx[:, j]]  # (N, C, H, P)
            else:
                v_j = wh[torch.arange(N, device=x.device)[:, None], nbr_idx[:, :, j]]
            agg = agg + alpha[:, :, j].unsqueeze(-1) * v_j
        agg = agg.reshape(N, C, D)

        x = x + self.dropout(self.out_proj(agg))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


# --------------------------------------------------------------------------- #
# The frontend module
# --------------------------------------------------------------------------- #
class GnnContinuousEncoderModel(nn.Module):
    """Length-preserving per-frame GNN frontend for continuous MEG windows.

    Maps ``x`` of shape (B, C, T) -> (B, out_channels, T) with T preserved
    exactly (see module docstring for the compute_output_lens contract).

    Forward signature matches ``SimpleConv``:
    ``forward(x, subject_ids=None, channel_positions=None)``; ``subject_ids``
    is ignored on purpose (purely spatial graph, identical across subjects).
    ``t_chunk`` is read at forward time, so tests/checkpoints can retune the
    memory/speed trade-off without touching weights.
    """

    def __init__(self, config: "GnnContinuousEncoder", n_in_channels: int, out_channels: int):
        super().__init__()
        self.out_channels = out_channels
        self.n_in_channels = n_in_channels
        self.k_neighbors = config.k_neighbors
        self.conv_mult = config.conv_mult
        self.t_chunk = config.t_chunk

        C = n_in_channels
        # 1. depthwise per-sensor temporal features ("same" length)
        self.temporal = nn.Conv1d(
            C, C * config.conv_mult, kernel_size=5, padding=2, groups=C
        )

        # 2. per-frame node embedding
        d = config.d_node
        self.node_proj = nn.Linear(config.conv_mult, d)
        self.node_emb = nn.Parameter(torch.zeros(1, C, d))
        nn.init.normal_(self.node_emb, std=0.02)
        self.input_dropout = nn.Dropout(config.dropout)

        # 3. spatial message passing
        self.layers = nn.ModuleList(
            NeighborGraphAttentionBlock(d, config.heads, config.dropout)
            for _ in range(config.n_layers)
        )
        self.final_norm = nn.LayerNorm(d)

        # 4. per-frame readout -> out_channels
        self.readout = nn.Linear(2 * d, out_channels)

    # ------------------------------------------------------------------ #
    def build_neighbors(
        self, channel_positions: torch.Tensor | None, batch_size: int, device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Neighbor indices/mask of shape (C, K) or (B, C, K).

        * ``None`` -> fully-connected graph (uniform neighborhood).
        * ``(C, 2)`` -> single graph shared by the batch.
        * ``(B, C, 2)`` -> per-sample graphs, collapsed to a shared graph
          when every batch element carries the same layout (the usual case).
        """
        C = self.n_in_channels
        if channel_positions is None:
            adj = torch.ones(1, C, C, dtype=torch.bool, device=device)
            idx, mask = adjacency_to_neighbors(adj)
            return idx[0], mask[0]

        pos = channel_positions.detach()
        if pos.dim() == 2:
            idx, mask = adjacency_to_neighbors(knn_adjacency(pos, self.k_neighbors))
        elif pos.dim() == 3:
            if pos.size(0) == 1 or bool((pos == pos[0:1]).all()):
                idx, mask = adjacency_to_neighbors(
                    knn_adjacency(pos[0], self.k_neighbors)
                )
            else:
                idx, mask = adjacency_to_neighbors(
                    knn_adjacency(pos, self.k_neighbors)
                )
        else:
            raise ValueError(
                "channel_positions must be (C, 2), (B, C, 2) or None, "
                f"got {tuple(pos.shape)}"
            )
        return idx.to(device), mask.to(device)

    # ------------------------------------------------------------------ #
    def forward(
        self,
        x: torch.Tensor,
        subject_ids: torch.Tensor | None = None,
        channel_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del subject_ids  # graph is purely spatial; identical across subjects
        B, C, T = x.shape

        # depthwise temporal features, length preserved: (B, C*f, T)
        h = F.gelu(self.temporal(x))
        # conv output channels are group-major: [ch0_f0..ch0_f{f-1}, ch1_f0...]
        h = h.view(B, C, self.conv_mult, T).permute(0, 3, 1, 2)  # (B, T, C, f)

        nbr_idx, nbr_mask = self.build_neighbors(channel_positions, B, x.device)

        outs: list[torch.Tensor] = []
        for t0 in range(0, T, self.t_chunk):
            hc = h[:, t0 : t0 + self.t_chunk]  # (B, t, C, f)
            t_len = hc.shape[1]
            N = B * t_len
            hn = hc.reshape(N, C, self.conv_mult)
            hn = self.node_proj(hn) + self.node_emb  # (N, C, d)
            hn = self.input_dropout(hn)

            if nbr_idx.dim() == 3:  # per-sample graphs: one per batch element
                idx_c = nbr_idx.repeat_interleave(t_len, dim=0)  # (N, C, K)
                mask_c = nbr_mask.repeat_interleave(t_len, dim=0)
            else:
                idx_c, mask_c = nbr_idx, nbr_mask

            for layer in self.layers:
                hn = layer(hn, idx_c, mask_c)
            hn = self.final_norm(hn)

            pooled = torch.cat([hn.mean(dim=1), hn.max(dim=1).values], dim=-1)
            outs.append(self.readout(pooled).reshape(B, t_len, self.out_channels))

        y = torch.cat(outs, dim=1)  # (B, T, out_channels)
        return y.transpose(1, 2)  # (B, out_channels, T) — channel-first contract


# --------------------------------------------------------------------------- #
# Config (neuraltrain BaseModelConfig)
# --------------------------------------------------------------------------- #
class GnnContinuousEncoder(BaseModelConfig):
    """Config for the V3 continuous GNN frontend.

    Unique registry name (exca's discriminated-model registry is global by
    class name; ``GnnWindowEncoder`` is taken by brain2qwerty_v1_mamba).
    """

    d_node: int = 128
    n_layers: int = 3
    heads: int = 4
    k_neighbors: int = 8
    conv_mult: int = 8
    dropout: float = 0.1
    t_chunk: int = 128

    def build(self, n_in_channels: int, n_outputs: int) -> GnnContinuousEncoderModel:
        return GnnContinuousEncoderModel(self, n_in_channels, n_outputs)
