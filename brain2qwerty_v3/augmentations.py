# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pydantic
import torch
from pydantic import BaseModel
from torch import nn
from torch.nn import functional as F
from torchaudio.transforms import FrequencyMasking, TimeMasking


class PreprocessConfig(BaseModel):
    """Hyperparameters for the on-device MEG augmentation applied during training."""

    whiteNoiseSD: float = 0.8
    constantOffsetSD: float = 0.2
    time_mask_param: int = 0
    freq_mask_param: int = 0
    iid_masks: bool = False
    p_time_mask: float = 0.2
    time_stretch: bool = False
    channel_dropout: float = 0.0  # fraction of MEG channels zeroed per sample
    model_config = pydantic.ConfigDict(extra="forbid")


class Preprocess(nn.Module):
    """Adds white noise, a constant per-channel offset, optional time-stretch,
    optional time/frequency masking (SpecAugment-style), and optional sensor
    (channel) dropout to a (B, T, C) MEG batch."""

    def __init__(
        self,
        whiteNoiseSD: float = 0.8,
        constantOffsetSD: float = 0.2,
        time_mask_param: int = 0,
        freq_mask_param: int = 0,
        iid_masks: bool = False,
        p_time_mask: float = 0.2,
        time_stretch: bool = False,
        channel_dropout: float = 0.0,
    ):
        super().__init__()
        self.whiteNoiseSD = whiteNoiseSD
        self.constantOffsetSD = constantOffsetSD
        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param
        self.iid_masks = iid_masks
        self.p_time_mask = p_time_mask
        self.time_stretch = time_stretch
        self.channel_dropout = channel_dropout

        if self.time_mask_param > 0:
            self.time_mask = TimeMasking(
                time_mask_param, iid_masks=self.iid_masks, p=self.p_time_mask
            )
        if self.freq_mask_param > 0:
            self.freq_mask = FrequencyMasking(freq_mask_param, iid_masks=self.iid_masks)

    @torch.no_grad()
    def forward(self, batch: dict) -> dict:
        neuro = batch["neuros"]  # (B, T, C)

        if self.whiteNoiseSD > 0:
            neuro += torch.randn(neuro.shape, device=neuro.device) * self.whiteNoiseSD

        if self.constantOffsetSD > 0:
            neuro += (
                torch.randn([neuro.shape[0], 1, neuro.shape[2]], device=neuro.device)
                * self.constantOffsetSD
            )

        if self.time_stretch:
            stretch_factor = torch.empty(1).uniform_(0.8, 1.2).item()
            T = neuro.shape[1]
            stretched_T = int(T * stretch_factor)
            neuro = F.interpolate(
                neuro.transpose(1, 2),
                size=stretched_T,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
            batch["neuro_sizes"] = (batch["neuro_sizes"] * stretch_factor).long()

        if self.channel_dropout > 0:
            # Zero whole sensor channels per sample (B, 1, C) keep-mask.
            keep = (
                torch.rand(neuro.shape[0], 1, neuro.shape[2], device=neuro.device)
                > self.channel_dropout
            )
            neuro = neuro * keep

        if self.time_mask_param > 0:
            neuro = self.time_mask(neuro)
        if self.freq_mask_param > 0:
            neuro = self.freq_mask(neuro)

        batch["neuros"] = neuro
        return batch
