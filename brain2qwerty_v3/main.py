# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import typing as tp
from pathlib import Path

import lightning.pytorch as pl
import pydantic
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.strategies import DDPStrategy
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
from torchmetrics.text import CharErrorRate, WordErrorRate
from transformers import AutoModelForCausalLM, AutoTokenizer

import neuralset as ns
from neuralset.events.study import EventsTransform
from neuraltrain.models.base import BaseModelConfig
from neuraltrain.utils import WandbLoggerConfig

import studies  # registers Pinet2024Meg
from . import models as _models  # registers ConvMambaHybrid
from . import transforms as _transforms  # registers EventsTransforms
from .callbacks import PredictionCSVCallback
from .config.xp_config import LLM, RESULTS, WORD_EXTRACTOR, debug_config, experiment_config
from .data import SentenceDataset
from .metrics import SemanticErrorRate
from .pl_module import NeuroLLMModule
from .utils import ChannelPositions2D, accelerator, build_events, prepare_word_embeddings

log = logging.getLogger(__name__)


class Data(pydantic.BaseModel):
    """Sentence-level dataloaders for Brain2Qwerty V3."""

    model_config = pydantic.ConfigDict(extra="ignore", arbitrary_types_allowed=True)

    study: ns.events.Study
    transforms: list[EventsTransform] = pydantic.Field(default_factory=list)
    neuro: ns.extractors.BaseExtractor
    extractor: ns.extractors.BaseExtractor

    start: float = -0.4
    duration: float | None = None
    jitter: bool = True
    num_classes: int = 29
    tail_min: float = 0.4
    tail_max: float = 0.5

    batch_size: int = 64
    val_batch_size: int = 128
    test_batch_size: int = 8
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True

    def build(self) -> dict[str, DataLoader]:
        events = build_events(self.study, self.transforms)
        splits = [s for s in events.split.unique() if pd_notna(s)]
        loaders = {}
        for split in splits:
            ds = SentenceDataset(
                events,
                self.neuro,
                self.extractor,
                split=split,
                start=self.start,
                duration=self.duration,
                jitter=self.jitter,
                num_classes=self.num_classes,
                tail_min=self.tail_min,
                tail_max=self.tail_max,
            )
            is_train = split == "train"
            bs = self.batch_size if is_train else (self.val_batch_size if split == "val" else self.test_batch_size)
            loaders[split] = DataLoader(
                ds,
                batch_size=bs,
                shuffle=is_train,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=self.persistent_workers and self.num_workers > 0,
                collate_fn=ds.collate_fn,
            )
        return loaders


def pd_notna(val) -> bool:
    import pandas as pd
    return pd.notna(val)


class Experiment(pydantic.BaseModel):
    """Brain2Qwerty V3 Word-Level End-to-End Experiment."""

    model_config = pydantic.ConfigDict(extra="ignore", arbitrary_types_allowed=True)

    output_dir: str
    seed: int = 123
    max_epochs: int = 275

    data: Data
    preprocess_config: dict = pydantic.Field(default_factory=dict)
    brain_model_config: BaseModelConfig

    alpha: float = 0.1
    beta: float = 0.01
    loss_alpha: float = 0.7
    ctc_start_epoch: int = 0
    contrastive_start_epoch: int = 150
    llm_start_epoch: int = 225

    llm_name: str = LLM
    lora_rank: int = 2
    word_extractor_config: dict = pydantic.Field(default_factory=lambda: WORD_EXTRACTOR)
    num_beams: int = 16

    optimizer_config: dict = pydantic.Field(default_factory=lambda: {"lr": 3e-4, "weight_decay": 1e-3})
    scheduler_config: dict = pydantic.Field(default_factory=lambda: {"name": "WarmupCosine", "warmup_steps": 500, "eta_min": 1e-6})

    devices: int | None = None
    gradient_clip_val: float | None = 1.0
    accumulate_gradient_batches: int = 2
    precision: str = "bf16-mixed"

    wandb_config: WandbLoggerConfig | None = None
    eval_only: bool = False
    ckpt_path: str | None = None
    resume_ckpt: str | None = None
    save_checkpoints: bool = True

    def _build_module(self, loaders: dict[str, DataLoader]) -> NeuroLLMModule:
        sample_batch = next(iter(loaders["train"]))
        data = sample_batch["data"]
        n_channels = data["neuro"].shape[1]

        encoder = self.brain_model_config.build(n_in_channels=n_channels, n_outputs=self.brain_model_config.dim)
        tokenizer = AutoTokenizer.from_pretrained(self.llm_name)
        llm = AutoModelForCausalLM.from_pretrained(self.llm_name, torch_dtype=torch.bfloat16)

        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.lora_rank,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj"],
        )
        llm = get_peft_model(llm, lora_cfg)

        metrics = {
            "val/CER": CharErrorRate(),
            "val/WER": WordErrorRate(),
            "val/SemER": SemanticErrorRate(),
            "test/CER": CharErrorRate(),
            "test/WER": WordErrorRate(),
            "test/SemER": SemanticErrorRate(),
        }

        module = NeuroLLMModule(
            encoder=encoder,
            llm=llm,
            tokenizer=tokenizer,
            word_extractor_config=self.word_extractor_config,
            metrics=metrics,
            preprocess_config=self.preprocess_config,
            optimizer_config=self.optimizer_config,
            scheduler_config=self.scheduler_config,
            alpha=self.alpha,
            beta=self.beta,
            loss_alpha=self.loss_alpha,
            ctc_start_epoch=self.ctc_start_epoch,
            contrastive_start_epoch=self.contrastive_start_epoch,
            llm_start_epoch=self.llm_start_epoch,
            num_beams=self.num_beams,
        )
        return module

    def _trainer_setup(self) -> pl.Trainer:
        accel, devices = accelerator(self.devices)
        if self.eval_only:
            devices = 1
        callbacks: list[pl.Callback] = [PredictionCSVCallback(save_dir=self.output_dir)]
        if self.save_checkpoints:
            callbacks += [
                ModelCheckpoint(
                    dirpath=self.output_dir,
                    filename="best_ctc",
                    save_last=True,
                    save_top_k=1,
                    monitor="val/cer_epo",
                    mode="min",
                ),
                ModelCheckpoint(
                    dirpath=self.output_dir,
                    filename="best_llm",
                    save_top_k=1,
                    monitor="val/WER",
                    mode="min",
                ),
            ]
        loggers: list = [CSVLogger(self.output_dir, name="logs")]
        return pl.Trainer(
            accelerator=accel,
            devices=devices,
            strategy=DDPStrategy(find_unused_parameters=True) if devices > 1 else "auto",
            max_epochs=self.max_epochs,
            gradient_clip_val=self.gradient_clip_val,
            accumulate_grad_batches=self.accumulate_gradient_batches,
            precision=self.precision,
            callbacks=callbacks,
            logger=loggers,
            log_every_n_steps=5,
        )

    def run(self) -> None:
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        loaders = self.data.build()
        self._module = self._build_module(loaders)
        self._trainer = self._trainer_setup()
        if not self.eval_only:
            self._trainer.fit(
                self._module,
                loaders["train"],
                loaders.get("val"),
                ckpt_path=self.resume_ckpt,
            )
        if "test" in loaders:
            self._trainer.test(
                self._module,
                dataloaders=loaders["test"],
                ckpt_path=self.ckpt_path if self.eval_only else None,
            )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="brain2qwerty_v3")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p):
        p.add_argument("--core", choices=["conformer", "mamba_mlp", "mamba3_hybrid_stabilized", "hybrid"],
                       default="mamba3_hybrid_stabilized", help="sequence core variant")
        p.add_argument("--small", action="store_true", help="smaller 512-dim preset")
        p.add_argument("--subjects", nargs="+", default=None, help="subjects list (e.g. S15 S16 S6)")
        p.add_argument("--lr", type=float, default=None, help="learning rate")
        p.add_argument("--wd", type=float, default=None, help="weight decay")
        p.add_argument("--devices", type=int, default=None, help="GPU count")
        p.add_argument("--tag", default=None, help="output directory suffix tag")
        p.add_argument("--resume", default=None, help="checkpoint to resume from")

    p_debug = sub.add_parser("debug", help="smoke test")
    add_common(p_debug)

    p_train = sub.add_parser("train", help="train model")
    add_common(p_train)

    p_eval = sub.add_parser("eval", help="eval model")
    add_common(p_eval)
    p_eval.add_argument("--ckpt", required=True)

    args = parser.parse_args(argv)

    if args.command == "debug":
        cfg = debug_config(core=args.core)
    else:
        out_dir = None
        if args.tag:
            out_dir = str(Path(RESULTS) / f"v3-{args.core}-{args.tag}")
        cfg = experiment_config(
            core=args.core,
            small=args.small,
            subjects=args.subjects,
            lr=args.lr,
            wd=args.wd,
            output_dir=out_dir,
        )

    if args.command == "eval":
        cfg["eval_only"] = True
        cfg["ckpt_path"] = args.ckpt
    if getattr(args, "resume", None):
        cfg["resume_ckpt"] = args.resume
    if getattr(args, "devices", None) is not None:
        cfg["devices"] = args.devices

    print(f"[brain2qwerty_v3] running in '{args.command}' mode (core={args.core})")
    Experiment(**cfg).run()


if __name__ == "__main__":
    main()
