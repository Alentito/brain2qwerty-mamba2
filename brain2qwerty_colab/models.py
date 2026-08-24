# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""V3 architecture, self-contained: conv front-end + hybrid Mamba-2/attention core.

``ConvConformer`` (V2) and ``ConvMambaHybrid`` (V3) are merged here into a single
``ConvMambaEncoder`` config (registered under that collision-free name) so this
package does not import ``brain2qwerty_v2`` / ``brain2qwerty_v3``. The forward
pass and parameter layout are identical to V3's ``ConvMambaHybridModel``.
"""

import torch
import torch.nn as nn

from neuraltrain.models.conformer import Conformer
from neuraltrain.models.conv_transformer import ConvTransformer, ConvTransformerModel
from neuraltrain.models.transformer import TransformerEncoder

from .mamba import MambaHybridCore


class ConvMambaEncoder(ConvTransformer):
    """Encoder config: conv front-end + hybrid Mamba-2/attention sequence core.

    Identical to V2's ``ConvConformer`` except that ``transformer_config`` also
    accepts :class:`~brain2qwerty_colab.mamba.MambaHybridCore`, the
    Nemotron-H-style hybrid Mamba-2/attention stack that replaces the Conformer.
    ``aux_prediction`` enables the auxiliary CTC head (``z`` / ``z_aux``) blended
    back into the stack input.
    """

    transformer_config: TransformerEncoder | Conformer | MambaHybridCore | None = None
    aux_prediction: bool = False

    def build(
        self, n_in_channels: int, n_outputs: int | None = None
    ) -> "ConvMambaEncoderModel":
        return ConvMambaEncoderModel(
            n_in_channels, n_outputs or self.output_layer_dim, config=self
        )


class ConvMambaEncoderModel(ConvTransformerModel):
    """Conv + hybrid-Mamba encoder with the auxiliary CTC head and per-frame outputs.

    The sequence core is built via ``config.transformer_config.build(dim)`` in the
    shared ``ConvTransformerModel.__init__``, so swapping in the hybrid stack is
    purely a config-level change. Returns ``c_out`` (CTC logits), ``z_final``
    (per-frame features for the word segmenter) and, when ``aux_prediction`` is
    set, the auxiliary logits ``z`` / ``z_aux``.
    """

    def __init__(
        self, in_channels: int, out_channels: int | None, config: ConvMambaEncoder
    ):
        super().__init__(in_channels, out_channels, config)
        self.aux_prediction = config.aux_prediction
        if config.aux_prediction:
            self.intermediate_linear = nn.Linear(out_channels or self.dim, self.dim)
            self.shared_layer_norm = nn.LayerNorm(self.dim)

    def forward(  # type: ignore[override]
        self,
        x: torch.Tensor,
        day_idx: torch.Tensor | None = None,
        channel_positions: torch.Tensor | None = None,
        neuro_device_type: str | None = None,
    ) -> dict[str, torch.Tensor]:
        # Sentence batches arrive as (B, T, C); the conv encoder wants (B, C, T).
        x = x.transpose(1, 2)
        z = self._encoder_and_downsampling_forward(
            x, subject_ids=day_idx, channel_positions=channel_positions
        )
        z_enc = z

        z_aux = None
        if self.aux_prediction:
            z = self.shared_layer_norm(z)
            z_aux = self.output_layer(z)
            z = z + self.intermediate_linear(torch.softmax(z_aux, dim=-1))

        c_in = self._pre_transformer_forward(z, neuro_device_type=neuro_device_type)
        z_final = self.transformer(c_in)

        c_out = z_final
        if self.output_avg_pool:
            c_out = c_out.mean(dim=1)
        if self.aux_prediction:
            c_out = self.shared_layer_norm(c_out)
        c_out = self.output_layer(c_out)

        out = {
            "z": z_aux if self.aux_prediction else z_enc,
            "z_enc": z_enc,
            "z_final": z_final,
            "c_out": c_out,
        }
        if z_aux is not None:
            out["z_aux"] = z_aux
        return out
