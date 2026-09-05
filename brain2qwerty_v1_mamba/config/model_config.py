# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Model configs for the V1-Mamba ablation.

``ENCODER`` is V1's conv encoder verbatim. The sentence core is switchable:
``sentence_core(core="mamba"|"transformer")`` — everything else in the
experiment is identical, so the comparison is a clean core-only ablation.

Widths: V1's paper config uses hidden=2048 for both the encoder output and
the sentence core. ``small=True`` drops this to 512 (encoder hidden 512) for
Kaggle/Colab free-tier GPUs; keep ``small`` the same across both cores.
"""

from brain2qwerty_v1.config.model_config import ENCODER as _V1_ENCODER
from brain2qwerty_v1.config.model_config import TRANSFORMER as _V1_TRANSFORMER


def encoder(small: bool = False) -> dict:
    """V1's SimpleConvTimeAgg encoder (per-subject 2D-Fourier merger)."""
    cfg = {**_V1_ENCODER}
    if small:
        cfg["hidden"] = 512
        cfg["initial_linear"] = 256
        cfg["merger_config"] = {
            **cfg["merger_config"],
            # 2D Fourier emb requires (total_dim / 2) ** (1 / n_dims) to be an
            # integer: 512 -> sqrt(256) = 16. (1024 would give sqrt(512) — invalid.)
            "fourier_emb_config": {"n_freqs": None, "total_dim": 512, "n_dims": 2},
        }
    return cfg


def sentence_core(core: str = "mamba", small: bool = False) -> dict:
    """Sentence-level sequence core: the ablation switch.

    Options:
    * ``"transformer"``: V1 reference (depth 4, heads 2, ALiBi)
    * ``"transformer_deep"``: 8-layer Transformer (depth 8, heads 4, ALiBi)
    * ``"mamba"``: BiMamba-2 stack (depth 4, forward + backward SSD)
    * ``"mamba3"``: BiMamba-3 stack (depth 4, BCNorm + complex-state RoPE)
    * ``"mamba_mlp"``: BiMamba-2 + Gated Non-linear Fusion + FFN MLP sublayer
    * ``"mamba3_mlp"``: BiMamba-3 + Gated Non-linear Fusion + FFN MLP sublayer
    * ``"deltanet"``: BiDeltaNet stack (depth 4, delta-rule linear attention)
    * ``"deltanet_mlp"``: BiDeltaNet + Gated Non-linear Fusion + FFN MLP sublayer
    * ``"hybrid"``: Nemotron-H style [M, M, M, A] (depth 4, attention every 4)
    * ``"hybrid3"``: Mamba-3 + Attention [M3, M3, M3, A]
    * ``"hybrid_8l"``: 8-layer Hybrid [M, M, M, A, M, M, M, A]
    * ``"hybrid3_8l"``: 8-layer Mamba-3 Hybrid
    """
    if core == "transformer":
        return {**_V1_TRANSFORMER}
    if core == "transformer_deep":
        return {"name": "TransformerEncoder", "alibi_pos_bias": True, "depth": 8, "heads": 4}

    d_state = 64 if small else 128
    base_mamba = {
        "dropout": 0.1,
        "d_state": d_state,
        "headdim": 64,
        "expand": 2,
        "d_conv": 4,
        "ngroups": 1,
        "head_chunk": 8,
    }

    if core in ("mamba", "mamba3"):
        return {
            "name": "BiMamba3SentenceCore" if core == "mamba3" else "BiMambaSentenceCore",
            "n_layer": 4,
            "use_mlp": False,
            "gated_fusion": False,
            **base_mamba,
        }

    if core in ("mamba_mlp", "mamba3_mlp"):
        return {
            "name": "BiMamba3SentenceCore" if core == "mamba3_mlp" else "BiMambaSentenceCore",
            "n_layer": 4,
            "use_mlp": True,
            "gated_fusion": True,
            **base_mamba,
        }

    if core in ("deltanet", "deltanet_mlp"):
        mlp = core == "deltanet_mlp"
        return {
            "name": "BiDeltaNetSentenceCore",
            "n_layer": 4,
            "use_mlp": mlp,
            "gated_fusion": mlp,
            "headdim": 64,
            "expand": 1,
            "dropout": 0.1,
        }

    if core in ("hybrid", "hybrid3"):
        return {
            "name": "HybridMamba3SentenceCore" if core == "hybrid3" else "HybridSentenceCore",
            "n_layer": 4,
            "attention_every": 4,
            "heads": 4,
            "use_mlp": False,
            "gated_fusion": False,
            **base_mamba,
        }

    if core in ("hybrid_8l", "hybrid3_8l"):
        return {
            "name": "HybridMamba3SentenceCore" if core == "hybrid3_8l" else "HybridSentenceCore",
            "n_layer": 8,
            "attention_every": 4,
            "heads": 4,
            "use_mlp": False,
            "gated_fusion": False,
            **base_mamba,
        }

    raise ValueError(f"unknown core {core!r}; valid: transformer, mamba, mamba3, mamba_mlp, mamba3_mlp, deltanet, deltanet_mlp, hybrid, hybrid3, hybrid_8l, hybrid3_8l, transformer_deep")
