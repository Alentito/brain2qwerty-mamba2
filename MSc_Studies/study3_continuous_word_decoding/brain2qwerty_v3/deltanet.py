# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""DeltaNet sequence core for the Brain2Qwerty V3 pipeline (Study 4, Stage 2).

Ports the bidirectional DeltaNet stack of
``brain2qwerty_v1_mamba.deltanet_core`` to V3's sequence-core contract:
``transformer_config.build(dim)`` -> a module called as ``module(c_in)`` with
``c_in`` of shape (B, T', dim), **no mask**, returning (B, T', dim).

The mixer (``DeltaNetMixer``, exact O(T^2) "WY" parallel delta rule) and the
bidirectional wrapper (``BiMambaBlock``) are reused verbatim from v1_mamba.
What does NOT carry over is v1_mamba's ``BiMambaSentenceCoreModule.forward``
mask handling — a per-sentence Python loop that unpads zero-padded V1
sentences. V3's batches are padded continuous windows and every other V3 core
(Conformer, MambaHybrid, Mamba3StabilizedHybrid, BiMambaGatedMLP) processes
padding frames *unmasked* — the CTC loss ignores them via
``compute_output_lens`` — so this core does the same, preserving exact parity
with the Conformer baseline. With ``mask=None`` (the only path V3 exercises)
v1_mamba's loop module reduces to the same plain stack, so the two modules
are weight-compatible: a checkpoint/state_dict can be copied between them
(strict load), which the Study-4 port tests verify.

Registry: the config class is deliberately named ``BiDeltaNetCTCCore`` —
exca's discriminated-model registry is global by class name and
``BiDeltaNetSentenceCore`` is taken by brain2qwerty_v1_mamba.
"""

import torch
import torch.nn as nn

from neuraltrain.models.base import BaseModelConfig

from brain2qwerty_v1_mamba.deltanet_core import BiDeltaNetSentenceCore, DeltaNetMixer
from brain2qwerty_v1_mamba.mamba_core import BiMambaBlock, RMSNorm


class DeltaNetCTCCoreModule(nn.Module):
    """Plain bidirectional DeltaNet stack with the V3 (B, T, D) -> (B, T, D)
    no-mask interface: ``n_layer`` x ``BiMambaBlock(mixer_cls=DeltaNetMixer)``
    + final ``RMSNorm``.

    State-dict layout is identical to v1_mamba's
    ``BiMambaSentenceCoreModule(dim, mixer_cls=DeltaNetMixer, ...)``
    (``blocks.{i}.*`` + ``final_norm.weight``), so weights copy across with a
    strict ``load_state_dict``.
    """

    def __init__(
        self,
        dim: int,
        n_layer: int = 8,
        dropout: float = 0.1,
        use_mlp: bool = False,
        gated_fusion: bool = False,
        **mixer_kwargs,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            BiMambaBlock(
                dim,
                dropout=dropout,
                mixer_cls=DeltaNetMixer,
                use_mlp=use_mlp,
                gated_fusion=gated_fusion,
                **mixer_kwargs,
            )
            for _ in range(n_layer)
        )
        self.final_norm = RMSNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return self.final_norm(x)


class BiDeltaNetCTCCore(BiDeltaNetSentenceCore):
    """Config for :class:`DeltaNetCTCCoreModule` (V3 CTC-pipeline core).

    Inherits all fields from v1_mamba's ``BiDeltaNetSentenceCore``
    (``n_layer``, ``dropout``, ``use_mlp``, ``gated_fusion``, ``headdim``,
    ``expand``, ``norm_group_size``); only the build target and the registry
    name differ.
    """

    def build(self, dim: int) -> DeltaNetCTCCoreModule:
        kwargs = self.model_dump()
        kwargs.pop("name", None)
        return DeltaNetCTCCoreModule(dim, **kwargs)
