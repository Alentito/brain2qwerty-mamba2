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

    ``core="transformer"`` is V1's reference (bidirectional, ALiBi, depth 4).
    ``core="mamba"`` is the bidirectional Mamba-2 stack (depth 4, one forward
    and one backward SSD mixer per block) — same interface ``(x, mask)``.
    ``core="mamba3"`` adds the Mamba-3-style stability/expressivity upgrades
    (BCNorm + B/C biases + data-dependent RoPE state rotation).
    """
    if core == "transformer":
        # V1's reference sentence transformer, unchanged (depth 4, heads 2,
        # ALiBi) — identical at both widths so the ablation stays clean.
        return {**_V1_TRANSFORMER}
    if core in ("mamba", "mamba3"):
        return {
            "name": "BiMamba3SentenceCore" if core == "mamba3" else "BiMambaSentenceCore",
            "n_layer": 4,            # match V1 transformer depth
            "dropout": 0.1,
            "d_state": 64 if small else 128,
            "headdim": 64,
            "expand": 2,
            "d_conv": 4,
            "ngroups": 1,
            "head_chunk": 8,
        }
    raise ValueError(f"unknown core {core!r}; expected 'mamba', 'mamba3' or 'transformer'")
