# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from pathlib import Path
import typing as tp

from .model_config import build_encoder_config


def _normalise_subject(s: str) -> str:
    """Map S15 / Pinet2024Meg-S15 / Pinet2024Meg/S15 -> Pinet2024Meg/S15."""
    s = str(s).strip()
    if s.startswith("Pinet2024Meg/"):
        return s
    if s.startswith("Pinet2024Meg-"):
        return "Pinet2024Meg/" + s[len("Pinet2024Meg-"):]
    return f"Pinet2024Meg/{s}"


def _find_default_study_path() -> str:
    if "BRAIN2QWERTY_STUDIES" in os.environ:
        return os.environ["BRAIN2QWERTY_STUDIES"]
    cluster_path = Path.home() / "sharedscratch" / "B2Q" / "code" / "SpanishBCBL_3subj"
    if cluster_path.exists():
        return str(cluster_path)
    if Path("SpanishBCBL_3subj").exists():
        return str(Path("SpanishBCBL_3subj").resolve())
    return str(Path.home() / "brain2qwerty_data" / "studies")


def _find_default_cache() -> str:
    if "BRAIN2QWERTY_CACHE" in os.environ:
        return os.environ["BRAIN2QWERTY_CACHE"]
    cluster_cache = Path.home() / "sharedscratch" / "B2Q" / "cache_v1mamba"
    if cluster_cache.exists():
        return str(cluster_cache)
    return str(Path.home() / ".cache" / "b2q_v1mamba")


STUDY_PATH = _find_default_study_path()
CACHE = _find_default_cache()
RESULTS = os.environ.get("BRAIN2QWERTY_RESULTS", str(Path(CACHE) / "results"))

# Word-level contrastive target + LoRA decoder LLM.
# Qwen3.5-0.8B (Apache 2.0, 201 languages incl. Spanish) replaces the
# English-centric TinyLlama-1.1B at a similar parameter count.
LLM = "Qwen/Qwen3.5-0.8B"
WORD_EXTRACTOR = {"model_name": LLM, "layers": 0, "contextualized": False}


def experiment_config(
    core: str = "mamba3_hybrid_stabilized",
    small: bool = False,
    subjects: list[str] | None = None,
    lr: float | None = None,
    wd: float | None = None,
    output_dir: str | None = None,
    batch_size: int | None = None,
    val_batch_size: int | None = None,
    accumulate_grad_batches: int | None = None,
) -> dict:
    """Full Brain2Qwerty V3 Word-Level Configuration on SpanishBCBL (Pinet2024Meg).

    Options:
    * ``core="conformer"``: Version 1 (V2 Conformer baseline on SpanishBCBL)
    * ``core="mamba_mlp"``: Version 2 (Round 3 Champion BiMamba-2 + Gated MLP on SpanishBCBL)
    * ``core="mamba3_hybrid_stabilized"``: Version 3 (Deep Research Stabilized Mamba-3 Hybrid on SpanishBCBL)
    """
    study_query = None
    if subjects:
        formatted = [_normalise_subject(s) for s in subjects]
        study_query = f"subject in {formatted!r}"

    base_lr = lr if lr is not None else (3e-4 if "mamba" in core else 8e-4)
    base_wd = wd if wd is not None else 1e-3
    out = output_dir or RESULTS

    bs = batch_size if batch_size is not None else (16 if not small else 32)
    v_bs = val_batch_size if val_batch_size is not None else (16 if not small else 32)
    accum = accumulate_grad_batches if accumulate_grad_batches is not None else (4 if bs <= 16 else 2)

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
            "batch_size": bs,
            "val_batch_size": v_bs,
            "test_batch_size": min(bs, 8),
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
            # Sensor dropout: zero 10% of MEG channels per batch for spatial
            # robustness / cross-session invariance.
            "channel_dropout": 0.1,
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
        # Qwen3.5-0.8B has more capacity than TinyLlama-1.1B; raise the LoRA rank
        # from 2 -> 8 so the adapter can actually learn the neural->text mapping.
        "lora_rank": 8,
        "lora_alpha_value": 16,
        "word_extractor_config": WORD_EXTRACTOR,
        "num_beams": 16,
        "optimizer_config": {"lr": base_lr, "weight_decay": base_wd},
        "scheduler_config": {
            "name": "WarmupCosine",
            "warmup_steps": 500,
            "eta_min": 1e-6,
        },
        "accumulate_gradient_batches": accum,
        "precision": "bf16-mixed",
        # Benchmark runs must complete all 3 stages (LLM starts at 225); early
        # stopping on stage-1 CER amputates them. Rely on checkpoints + manual
        # log monitoring instead.
        "early_stop_patience": None,
    }


def debug_config(core: str = "mamba3_hybrid_stabilized") -> dict:
    """Smoke-test config: fast sanity check with 3 subjects."""
    cfg = experiment_config(core=core, small=True, subjects=["S15", "S16", "S6"])
    cfg["data"]["batch_size"] = 2
    cfg["data"]["val_batch_size"] = 2
    cfg["data"]["test_batch_size"] = 2
    cfg["data"]["num_workers"] = 0
    cfg["data"]["persistent_workers"] = False
    cfg["data"]["pin_memory"] = False
    cfg["max_epochs"] = 2
    cfg["ctc_start_epoch"] = 0
    cfg["contrastive_start_epoch"] = 0
    cfg["llm_start_epoch"] = 0
    cfg["accumulate_gradient_batches"] = 1
    cfg["save_checkpoints"] = False
    return cfg
