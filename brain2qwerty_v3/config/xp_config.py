# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from pathlib import Path
import typing as tp

from .model_config import build_encoder_config

STUDY_PATH = os.environ.get(
    "BRAIN2QWERTY_STUDIES", str(Path.home() / "brain2qwerty_data" / "studies")
)
CACHE = os.environ.get("BRAIN2QWERTY_CACHE", str(Path.home() / ".cache" / "brain2qwerty"))
RESULTS = os.environ.get("BRAIN2QWERTY_RESULTS", str(Path(CACHE) / "results"))

# Word-level contrastive target + LoRA decoder LLM (TinyLlama multilingual embeddings)
LLM = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
WORD_EXTRACTOR = {"model_name": LLM, "layers": 0, "contextualized": False}


def experiment_config(
    core: str = "mamba3_hybrid_stabilized",
    small: bool = False,
    subjects: list[str] | None = None,
    lr: float | None = None,
    wd: float | None = None,
    output_dir: str | None = None,
) -> dict:
    """Full Brain2Qwerty V3 Word-Level Configuration on SpanishBCBL (Pinet2024Meg).

    Options:
    * ``core="conformer"``: Version 1 (V2 Conformer baseline on SpanishBCBL)
    * ``core="mamba_mlp"``: Version 2 (Round 3 Champion BiMamba-2 + Gated MLP on SpanishBCBL)
    * ``core="mamba3_hybrid_stabilized"``: Version 3 (Deep Research Stabilized Mamba-3 Hybrid on SpanishBCBL)
    """
    study_query = None
    if subjects:
        formatted = [f"Pinet2024Meg-{s}" if not s.startswith("Pinet2024Meg-") else s for s in subjects]
        study_query = f"subject in {formatted!r}"

    base_lr = lr if lr is not None else (3e-4 if "mamba" in core else 8e-4)
    base_wd = wd if wd is not None else 1e-3
    out = output_dir or RESULTS

    return {
        "output_dir": out,
        "seed": 123,
        "max_epochs": 275,
        "data": {
            "study": {
                "name": "Pinet2024Meg",
                "path": STUDY_PATH,
                "query": study_query,
                "infra": {"folder": CACHE},
                "infra_timelines": {"folder": CACHE, "cluster": None},
            },
            "transforms": [
                {"name": "SpanishBCBLV2Preprocessing"},
                {"name": "SpanishBCBLV2Splitter", "seed": 1},
                {"name": "WordCreator"},
            ],
            "neuro": {
                "name": "MegExtractor",
                "frequency": 100,
                "filter": (0.5, 45.0),
                "scaler": "RobustScaler",
                "apply_proj": False,
                "clamp": 5,
                "picks": "meg",
                "notch_filter": 50,
                "allow_maxshield": True,
                "infra": {"folder": CACHE, "cluster": None},
            },
            "extractor": {
                "name": "SentenceKeySeq",
                "mode": "typed_label",
                "infra": {"folder": CACHE},
            },
            "batch_size": 32 if small else 64,
            "val_batch_size": 64 if small else 128,
            "test_batch_size": 8,
            "num_workers": 4,
            "pin_memory": True,
            "persistent_workers": True,
        },
        "preprocess_config": {
            "whiteNoiseSD": 0.0,
            "constantOffsetSD": 0.3,
            "time_mask_param": 50,
            "p_time_mask": 0.2,
            "freq_mask_param": 400,
            "time_stretch": True,
        },
        "brain_model_config": build_encoder_config(core=core, small=small),
        # Staged 3-loss schedule: CTC from 0, +contrastive at 150, +LLM at 225
        "alpha": 0.1,
        "beta": 0.01,
        "loss_alpha": 0.7,
        "ctc_start_epoch": 0,
        "contrastive_start_epoch": 150,
        "llm_start_epoch": 225,
        "llm_name": LLM,
        "lora_rank": 2,
        "word_extractor_config": WORD_EXTRACTOR,
        "num_beams": 16,
        "optimizer_config": {"lr": base_lr, "weight_decay": base_wd},
        "scheduler_config": {
            "name": "WarmupCosine",
            "warmup_steps": 500,
            "eta_min": 1e-6,
        },
        "accumulate_gradient_batches": 2,
        "precision": "bf16-mixed",
    }


def debug_config(core: str = "mamba3_hybrid_stabilized") -> dict:
    """Smoke-test config: single timeline, fast sanity check."""
    cfg = experiment_config(core=core, small=True)
    cfg["data"]["study"]["query"] = "timeline_index == 0"
    cfg["data"]["batch_size"] = 4
    cfg["data"]["val_batch_size"] = 4
    cfg["data"]["num_workers"] = 0
    cfg["data"]["persistent_workers"] = False
    cfg["max_epochs"] = 3
    cfg["ctc_start_epoch"] = 0
    cfg["contrastive_start_epoch"] = 0
    cfg["llm_start_epoch"] = 0
    cfg["accumulate_gradient_batches"] = 1
    return cfg
