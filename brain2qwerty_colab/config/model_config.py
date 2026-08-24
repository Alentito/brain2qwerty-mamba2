# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Convolutional encoder with a per-subject 2D-Fourier channel merger
# (identical to V2/V3).
_ENCODER = {
    "name": "SimpleConv",
    "dropout_input": 0.2,
    "conv_dropout": 0.5,
    "hidden": 1500,
    "batch_norm": True,
    "depth": 4,
    "dilation_period": 3,
    "kernel_size": 5,
    "relu_leakiness": 0.01,
    "initial_linear": 512,
    "gelu": True,
    "skip": True,
    "scale": 0.1,
    "subject_layers_config": {},
    "merger_config": {
        "n_virtual_channels": 270,
        "fourier_emb_config": {"n_freqs": None, "total_dim": 2048, "n_dims": 2},
        "dropout": 0.2,
        "usage_penalty": 1.0,
        "per_subject": True,
        "embed_ref": False,
    },
}

# Full encoder (V3 architecture): conv encoder -> temporal downsampling ->
# hybrid Mamba-2 / attention stack (Nemotron-H-style M-M-M-A pattern), with an
# auxiliary CTC head (z_aux) blended back into the stack input.
ENCODER = {
    "name": "ConvMambaEncoder",
    "dim": 1024,
    "encoder_config": {**_ENCODER},
    "transformer_config": {
        "name": "MambaHybridCore",
        # 8 blocks total, attention at positions 4 and 8 -> M M M A M M M A
        "n_layer": 8,
        "attention_every": 4,
        # Mamba-2 (SSD) mixer
        "d_state": 128,
        "headdim": 64,
        "expand": 2,
        "d_conv": 4,
        "ngroups": 1,
        "head_chunk": 8,
        # attention blocks (x-transformers encoder layer, rotary + RMSNorm)
        "heads": 4,
        "ff_mult": 1,
        "attn_dropout": 0.1,
        "ff_dropout": 0.0,
        "rotary_pos_emb": True,
        "dropout": 0.1,
    },
    "temporal_downsampling_config": {"kernel_size": 16, "stride": 4},
    "aux_prediction": True,
}


def small_encoder() -> dict:
    """Smaller V3 variant for quick Colab/Kaggle iterations (dim 512, 6 blocks).

    Same architecture family (conv + hybrid Mamba-2/attention + aux CTC head),
    ~4x fewer sequence-core FLOPs than the full 1024-dim encoder.
    """
    cfg = {
        "name": "ConvMambaEncoder",
        "dim": 512,
        "encoder_config": {
            **_ENCODER,
            "hidden": 768,
            "initial_linear": 512,
            "merger_config": {
                **_ENCODER["merger_config"],
                "fourier_emb_config": {"n_freqs": None, "total_dim": 1024, "n_dims": 2},
            },
        },
        "transformer_config": {
            "name": "MambaHybridCore",
            "n_layer": 6,
            "attention_every": 3,
            "d_state": 64,
            "headdim": 64,
            "expand": 2,
            "d_conv": 4,
            "ngroups": 1,
            "head_chunk": 8,
            "heads": 4,
            "ff_mult": 1,
            "attn_dropout": 0.1,
            "ff_dropout": 0.0,
            "rotary_pos_emb": True,
            "dropout": 0.1,
        },
        "temporal_downsampling_config": {"kernel_size": 16, "stride": 4},
        "aux_prediction": True,
    }
    return cfg
