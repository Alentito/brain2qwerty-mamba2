# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Brain2Qwerty V3 Model Configs for Word-Level Decoding on SpanishBCBL."""

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


def build_encoder_config(core: str = "mamba3_hybrid_stabilized", small: bool = False) -> dict:
    """Build the Conv + Sequence Core encoder config.

    Options:
    * ``"conformer"``: Original Brain2Qwerty V2 baseline (Conformer CTC core).
    * ``"mamba_mlp"``: Round 3 Best Mamba Champion (BiMamba-2 + Gated Fusion + FFN MLP).
    * ``"mamba3_hybrid_stabilized"``: Deep Research Upgrade (BCNorm + RoPE + Adaptive Delta-t Clamping + Attention Hybrid).
    * ``"hybrid"``: Standard Mamba-2 Nemotron-H hybrid stack.
    """
    dim = 512 if small else 1024
    d_state = 64 if small else 128

    if core == "conformer":
        transformer_cfg = {
            "name": "Conformer",
            "num_layers": 8,
            "num_heads": 4,
            "ffn_dim": dim * 4,
            "depthwise_conv_kernel_size": 31,
            "dropout": 0.1,
        }
    elif core == "mamba_mlp":
        transformer_cfg = {
            "name": "BiMambaGatedMLP",
            "n_layer": 8,
            "d_state": d_state,
            "headdim": 64,
            "expand": 2,
            "d_conv": 4,
            "ngroups": 1,
            "head_chunk": 8,
            "ff_mult": 4,
            "dropout": 0.1,
        }
    elif core in ("mamba3_hybrid_stabilized", "mamba3_hybrid"):
        transformer_cfg = {
            "name": "Mamba3StabilizedHybrid",
            "n_layer": 8,
            "attention_every": 4,
            "d_state": d_state,
            "headdim": 64,
            "expand": 2,
            "d_conv": 4,
            "ngroups": 1,
            "head_chunk": 8,
            "rope_base": 10000.0,
            "heads": 4,
            "ff_mult": 2,
            "attn_dropout": 0.1,
            "ff_dropout": 0.0,
            "rotary_pos_emb": True,
            "dropout": 0.1,
        }
    elif core in ("hybrid", "mamba_hybrid"):
        transformer_cfg = {
            "name": "MambaHybrid",
            "n_layer": 8,
            "attention_every": 4,
            "d_state": d_state,
            "headdim": 64,
            "expand": 2,
            "d_conv": 4,
            "ngroups": 1,
            "head_chunk": 8,
            "heads": 4,
            "ff_mult": 2,
            "attn_dropout": 0.1,
            "ff_dropout": 0.0,
            "rotary_pos_emb": True,
            "dropout": 0.1,
        }
    else:
        raise ValueError(f"Unknown core: {core!r}")

    enc_cfg = {**_ENCODER}
    if small:
        enc_cfg["hidden"] = 750
        enc_cfg["initial_linear"] = 256
        enc_cfg["merger_config"]["fourier_emb_config"]["total_dim"] = 512

    return {
        "name": "ConvMambaHybrid",
        "dim": dim,
        "encoder_config": enc_cfg,
        "transformer_config": transformer_cfg,
        "temporal_downsampling_config": {"kernel_size": 16, "stride": 4},
        "aux_prediction": True,
    }


ENCODER = build_encoder_config(core="mamba3_hybrid_stabilized", small=False)
