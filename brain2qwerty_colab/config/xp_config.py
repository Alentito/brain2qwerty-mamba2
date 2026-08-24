# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Experiment configurations for the combined pipeline.

Three presets, all sharing the V3 architecture and the combined V1+V3
preprocessing:

* ``experiment_config()`` — full run (275-epoch staged schedule, 8 GPUs with
  automatic fallback), matching the V3 paper configuration.
* ``colab_config()`` — single-GPU preset for Google Colab / Kaggle: fewer
  workers, smaller batches and beams, and a shorter staged schedule.
* ``debug_config()`` — 1-timeline smoke test, all losses from epoch 0.

All presets accept ``subjects=[...]`` to train on a subset of the 19
participants (e.g. ``subjects=["S1", "S3", "S7"]``) and an optional
``timeline_query`` (neuralset study query, e.g. ``"subject in ['S1', 'S3']"``)
to also restrict which raw recordings are loaded/processed — recommended on
Colab/Kaggle to cut download and feature-extraction time.
"""

import os
from pathlib import Path

from .model_config import ENCODER, small_encoder

STUDY_PATH = os.environ.get(
    "BRAIN2QWERTY_STUDIES", str(Path.home() / "brain2qwerty_data" / "studies")
)
CACHE = os.environ.get("BRAIN2QWERTY_CACHE", str(Path.home() / ".cache" / "brain2qwerty"))
RESULTS = os.environ.get("BRAIN2QWERTY_RESULTS", str(Path(CACHE) / "results"))

# Word-level contrastive target + LoRA decoder LLM. TinyLlama is the V2/V3
# default; swap both entries for a Spanish/multilingual model if desired.
LLM = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Frozen LLM token embeddings used as the word-level contrastive target.
WORD_EXTRACTOR = {"model_name": LLM, "layers": 0, "contextualized": False}


def _transforms(subjects: list | None) -> list[dict]:
    """Combined preprocessing chain: V1 cleaning + V3 CTC targets (+ subject
    subset), then the V1 TF-IDF cluster split and V3 word-event creation."""
    return [
        {"name": "CombinedBCBLPreprocessing", "subjects": subjects},
        {"name": "CombinedBCBLSplitter", "seed": 1},
        {"name": "CombinedWordCreator"},
    ]


def experiment_config(
    subjects: list | None = None,
    timeline_query: str | None = None,
) -> dict:
    """Full configuration (SpanishBCBL MEG; CTC + contrastive + LLM staged
    schedule; V3 hybrid Mamba-2/attention encoder; combined V1+V3 preprocessing).

    Parameters
    ----------
    subjects :
        Optional subset of participants, e.g. ``["S1", "S3"]`` or ``[1, 3]``.
        ``None`` trains on all 19 unique participants.
    timeline_query :
        Optional neuralset study query restricting which recordings are loaded
        at all (e.g. ``"subject in ['S1', 'S3']"``). Combine with ``subjects``
        on Colab/Kaggle to avoid processing recordings you will not use.
    """
    study = {
        "name": "Pinet2024Meg",
        "path": STUDY_PATH,
        "infra": {"folder": CACHE},
        "infra_timelines": {"folder": CACHE, "cluster": None},
    }
    if timeline_query:
        study["query"] = timeline_query
    return {
        "output_dir": RESULTS,
        "seed": 123,
        "max_epochs": 275,
        "data": {
            "study": study,
            "transforms": _transforms(subjects),
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
                "name": "CombinedSentenceKeySeq",
                "mode": "typed_label",
                "infra": {"folder": CACHE},
            },
            "batch_size": 64,
            "val_batch_size": 128,
            "test_batch_size": 8,
            "num_workers": 16,
            "pin_memory": True,
            "persistent_workers": True,
        },
        # MEG augmentation (on-device, train only): per-channel offset + SpecAugment
        # masking + time-stretch (no white noise), matching the paper.
        "preprocess_config": {
            "whiteNoiseSD": 0.0,
            "constantOffsetSD": 0.3,
            "time_mask_param": 50,
            "p_time_mask": 0.2,
            "freq_mask_param": 400,
            "time_stretch": True,
        },
        "brain_model_config": ENCODER,
        # staged 3-loss schedule: CTC from 0, +contrastive at 150, +LLM at 225
        "alpha": 0.1,
        "beta": 0.01,
        "loss_alpha": 0.7,
        "ctc_start_epoch": 0,
        "contrastive_start_epoch": 150,
        "llm_start_epoch": 225,
        # LLM + LoRA all-subjects rank=2
        "llm_name": LLM,
        "lora_rank": 2,
        "word_extractor_config": WORD_EXTRACTOR,
        "num_beams": 16,
        "optimizer_config": {"lr": 8e-4, "weight_decay": 1e-3},
        "scheduler_config": {
            "name": "WarmupCosine",
            "warmup_steps": 500,
            "eta_min": 1e-6,
        },
        "accumulate_gradient_batches": 2,
        "precision": "bf16-mixed",
    }


def colab_config(
    subjects: list | None = None,
    timeline_query: str | None = None,
    small: bool = False,
) -> dict:
    """Single-GPU preset for Google Colab / Kaggle (T4/A100 class).

    Same combined preprocessing and V3 architecture as ``experiment_config``,
    with notebook-friendly settings: 1 device, 2 dataloader workers, smaller
    validation/test batches, greedy validation decoding, fewer test beams, and
    a compressed staged schedule (CTC 0-99, +contrastive at 100, +LLM at 150,
    200 epochs total).

    ``small=True`` additionally swaps the 1024-dim encoder for the 512-dim
    ``small_encoder()`` variant for fast iteration on free-tier GPUs.
    """
    cfg = experiment_config(subjects=subjects, timeline_query=timeline_query)
    cfg["devices"] = 1
    cfg["max_epochs"] = 200
    cfg["contrastive_start_epoch"] = 100
    cfg["llm_start_epoch"] = 150
    cfg["data"]["batch_size"] = 32
    cfg["data"]["val_batch_size"] = 64
    cfg["data"]["num_workers"] = 2
    cfg["data"]["persistent_workers"] = False
    cfg["num_beams"] = 4
    cfg["val_num_beams"] = 1
    cfg["accumulate_gradient_batches"] = 4
    if small:
        cfg["brain_model_config"] = small_encoder()
    return cfg


def debug_config(subjects: list | None = None) -> dict:
    """Smoke-test config: one timeline, all losses from epoch 0, single GPU."""
    cfg = experiment_config(subjects=subjects, timeline_query="timeline_index == 0")
    cfg["data"]["batch_size"] = 4
    cfg["data"]["val_batch_size"] = 4
    cfg["data"]["test_batch_size"] = 4
    cfg["data"]["num_workers"] = 0
    cfg["data"]["pin_memory"] = False
    cfg["data"]["persistent_workers"] = False
    cfg["max_epochs"] = 2
    cfg["contrastive_start_epoch"] = 0
    cfg["llm_start_epoch"] = 0
    cfg["num_beams"] = 1
    cfg["devices"] = 1
    cfg["save_checkpoints"] = False
    return cfg
