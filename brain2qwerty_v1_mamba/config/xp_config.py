# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Experiment configs for the V1-Mamba ablation (subjects S15, S16, S6).

Identical to V1's ``experiment_config`` except:

* the event chain starts with ``V1MambaSubjectFilter`` (subject subset),
* the sentence core is switchable (``core="mamba"|"transformer"``),
* ``small=True`` shrinks the model width for Kaggle/Colab free-tier GPUs,
* ``output_dir`` is namespaced by core and size so the two runs never clash.
"""

import os
from pathlib import Path

from brain2qwerty_v1.utils import BUTTON_MAPPING, NUM_CLASSES

from .model_config import encoder, sentence_core

STUDY_PATH = os.environ.get(
    "BRAIN2QWERTY_STUDIES", str(Path.home() / "brain2qwerty_data" / "studies")
)
CACHE = os.environ.get("BRAIN2QWERTY_CACHE", str(Path.home() / ".cache" / "brain2qwerty"))
RESULTS = os.environ.get("BRAIN2QWERTY_RESULTS", str(Path(CACHE) / "results"))

DEFAULT_SUBJECTS = ["S15", "S16", "S6"]


def experiment_config(
    subjects: list | None = None,
    core: str = "mamba",
    small: bool = False,
    timeline_query: str | None = None,
) -> dict:
    """V1 keystroke-decoding config with a switchable sentence core.

    Parameters
    ----------
    subjects :
        Participants to train on (default: S15, S16, S6 — the three strongest
        MEG decoders; S15 is the paper's best subject).
    core :
        ``"mamba"`` (bidirectional Mamba-2 stack) or ``"transformer"`` (V1
        reference). Both share every other setting — the ablation is clean.
    small :
        Model width 512 instead of the paper's 2048 (Kaggle/Colab preset).
    timeline_query :
        Optional neuralset study query (e.g. ``"subject in ['S15','S16','S6']"``)
        to skip loading other recordings entirely — recommended when warming
        the cache, as it cuts feature extraction to the selected subjects.
    """
    subjects = list(subjects or DEFAULT_SUBJECTS)
    tag = f"{'small-' if small else ''}{core}-" + "-".join(str(s) for s in subjects)

    study = {
        "name": "Pinet2024Meg",
        "path": STUDY_PATH,
        # mode="retry": recompute steps whose previous run failed (cached
        # error) — e.g. study scan ran before the raw data was extracted.
        "infra": {"folder": CACHE, "mode": "retry"},
        "infra_timelines": {"folder": CACHE, "cluster": None, "mode": "retry"},
    }
    if timeline_query:
        study["query"] = timeline_query

    return {
        "output_dir": str(Path(RESULTS) / tag),
        "seed": 33,
        "n_epochs": 300,
        "patience": 30,
        "save_checkpoints": True,
        "data": {
            "study": study,
            "transforms": [
                {"name": "V1MambaSubjectFilter", "subjects": subjects},
                {"name": "SpanishBCBLPreprocessing"},
                {"name": "Brain2QwertyV1Splitter", "seed": 1},
            ],
            "neuro": {
                "name": "MegExtractor",
                "frequency": 50,
                "filter": (0.1, 20.0),
                "baseline": (0.0, 0.2),
                "apply_proj": False,
                "clamp": 5,
                "scaler": "RobustScaler",
                "allow_maxshield": True,
                "infra": {"folder": CACHE, "cluster": None},
            },
            "feature": {
                "name": "LabelEncoder",
                "aggregation": "trigger",
                "predefined_mapping": BUTTON_MAPPING,
                "event_types": "Keystroke",
                "event_field": "button",
                "return_one_hot": False,
            },
            "num_classes": NUM_CLASSES,
            "start": -0.2,
            "duration": 0.5,
            "batch_size": 64,
            "val_batch_size": 2048,
            "test_batch_size": 2048,
            "num_workers": 16,
            "pin_memory": True,
            "persistent_workers": True,
        },
        "brain_model_config": encoder(small=small),
        "transformer_config": sentence_core(core=core, small=small),
        "loss": {"name": "CrossEntropyLoss"},
        "optimizer": {
            "name": "LightningOptimizer",
            "optimizer": {"name": "AdamW", "lr": 5e-5, "kwargs": {"weight_decay": 1e-4}},
            "scheduler": {
                "name": "OneCycleLR",
                "kwargs": {"max_lr": 5e-5, "pct_start": 0.1},
            },
            "interval": "step",
        },
    }


def colab_config(
    subjects: list | None = None,
    core: str = "mamba",
    small: bool = True,
) -> dict:
    """Single-GPU Kaggle/Colab preset (T4-class): small model, 1 device,
    small batches, 2 workers, shorter schedule."""
    cfg = experiment_config(subjects=subjects, core=core, small=small)
    cfg["devices"] = 1
    cfg["n_epochs"] = 200
    cfg["patience"] = 25
    cfg["data"]["batch_size"] = 32
    cfg["data"]["val_batch_size"] = 256
    cfg["data"]["test_batch_size"] = 256
    cfg["data"]["num_workers"] = 2
    cfg["data"]["persistent_workers"] = False
    return cfg


def debug_config(subjects: list | None = None, core: str = "mamba") -> dict:
    """Smoke test: one recording of the first subject, 2 epochs, single GPU."""
    from ..transforms import _normalise_subject

    subjects = list(subjects or DEFAULT_SUBJECTS)
    cfg = experiment_config(
        subjects=subjects,
        core=core,
        small=True,
        # the timeline index stores subjects in long form ("Pinet2024Meg/S15")
        timeline_query=f"subject == '{_normalise_subject(subjects[0])}'",
    )
    cfg["n_epochs"] = 2
    cfg["patience"] = 2
    cfg["devices"] = 1
    cfg["data"]["batch_size"] = 16
    cfg["data"]["val_batch_size"] = 64
    cfg["data"]["test_batch_size"] = 64
    cfg["data"]["num_workers"] = 0
    cfg["data"]["pin_memory"] = False
    cfg["data"]["persistent_workers"] = False
    cfg["save_checkpoints"] = False
    return cfg
