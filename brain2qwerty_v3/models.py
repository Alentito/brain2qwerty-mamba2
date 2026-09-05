# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

from neuraltrain.models.conformer import Conformer
from neuraltrain.models.simpleconv import SimpleConv
from neuraltrain.models.simplerconv import SimplerConv
from neuraltrain.models.transformer import TransformerEncoder

# Aliased from V2 rather than redefined: exca's discriminated-model registry is
# global by class name, so redefining ``ConvConformer`` here would collide with
# V2's whenever both packages are imported in one process.
from brain2qwerty_v2.models import ConvConformer, ConvConformerModel  # noqa: E402

from .deltanet import BiDeltaNetCTCCore
from .gnn_frontend import GnnContinuousEncoder
from .mamba import BiMambaGatedMLP, Mamba3StabilizedHybrid, MambaHybrid


class ConvMambaHybrid(ConvConformer):
    """V3 encoder config: conv/GNN front-end + hybrid Mamba / Conformer / Gated MLP / DeltaNet sequence core."""

    # Union is widened (not redefined) so Study-4 configs can swap the Stage-1
    # frontend (GnnContinuousEncoder) and the Stage-2 core (BiDeltaNetCTCCore)
    # purely at the config-dict level.
    encoder_config: SimplerConv | SimpleConv | GnnContinuousEncoder

    transformer_config: (
        TransformerEncoder
        | Conformer
        | MambaHybrid
        | Mamba3StabilizedHybrid
        | BiMambaGatedMLP
        | BiDeltaNetCTCCore
        | None
    ) = None

    def build(
        self, n_in_channels: int, n_outputs: int | None = None
    ) -> "ConvMambaHybridModel":
        return ConvMambaHybridModel(
            n_in_channels, n_outputs or self.output_layer_dim, config=self
        )


class ConvMambaHybridModel(ConvConformerModel):
    """Conv + hybrid-Mamba encoder with the auxiliary CTC head.

    The forward pass is inherited from :class:`ConvConformerModel` unchanged:
    the shared ``ConvTransformerModel.__init__`` builds the sequence core via
    ``config.transformer_config.build(dim)``, so swapping in the hybrid stack
    is purely a config-level change. ``self.transformer(c_in)`` receives and
    returns ``(B, T, D)`` tensors exactly as the Conformer did.
    """
