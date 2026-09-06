# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Brain2Qwerty V3 Model Configs for Word-Level Decoding on SpanishBCBL."""

import copy

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


def build_encoder_config(
    core: str = "mamba3_hybrid_stabilized",
    small: bool = False,
    frontend: str = "conv",
) -> dict:
    """Build the Frontend + Sequence Core encoder config.

    Cores:
    * ``"conformer"``: Original Brain2Qwerty V2 baseline (Conformer CTC core).
    * ``"mamba_mlp"``: Round 3 Best Mamba Champion (BiMamba-2 + Gated Fusion + FFN MLP).
    * ``"mamba3_hybrid_stabilized"``: Deep Research Upgrade (BCNorm + RoPE + Adaptive Delta-t Clamping + Attention Hybrid).
    * ``"hybrid"``: Standard Mamba-2 Nemotron-H hybrid stack.
    * ``"deltanet"``: Bidirectional DeltaNet (delta-rule linear attention), 8 layers
      like the other V3 cores (Study 4 port from brain2qwerty_v1_mamba).

    Frontends:
    * ``"conv"``: SimpleConv + per-subject 2D-Fourier merger (default; the
      ``small`` width adjustments — ``hidden`` 750, merger ``total_dim`` 512 —
      apply only here).
    * ``"gnn"``: GnnContinuousEncoder — length-preserving per-frame k-NN graph
      attention over the sensor layout (Study 4). Widths follow ``dim``:
      ``d_node`` 128 (small) / 256 (full).
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
            "head_chunk": 4,
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
    elif core == "deltanet":
        # Study 4 port: bidirectional DeltaNet (delta-rule linear attention).
        # 8 layers to match the depth of the other V3 cores; expand=1 /
        # headdim=64 as in the v1_mamba ablation.
        transformer_cfg = {
            "name": "BiDeltaNetCTCCore",
            "n_layer": 8,
            "headdim": 64,
            "expand": 1,
            "dropout": 0.1,
        }
    else:
        raise ValueError(f"Unknown core: {core!r}")

    if frontend == "conv":
        enc_cfg = copy.deepcopy(_ENCODER)
        if small:
            enc_cfg["hidden"] = 750
            enc_cfg["initial_linear"] = 256
            enc_cfg["merger_config"]["fourier_emb_config"]["total_dim"] = 512
    elif frontend == "gnn":
        enc_cfg = {
            "name": "GnnContinuousEncoder",
            "d_node": 128 if small else 256,
            "n_layers": 3,
            "heads": 4,
            "k_neighbors": 8,
            "conv_mult": 8,
            "dropout": 0.1,
            "t_chunk": 128,
        }
    else:
        raise ValueError(f"Unknown frontend: {frontend!r}")

    return {
        "name": "ConvMambaHybrid",
        "dim": dim,
        "encoder_config": enc_cfg,
        "transformer_config": transformer_cfg,
        "temporal_downsampling_config": {"kernel_size": 16, "stride": 4},
        "aux_prediction": True,
    }


ENCODER = build_encoder_config(core="mamba3_hybrid_stabilized", small=False)
