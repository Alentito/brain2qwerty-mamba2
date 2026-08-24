# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""CLI for the V1-Mamba ablation.

Reuses V1's ``Experiment``/``Data``/``BrainModule`` unchanged — only the
config differs (subject filter + switchable sentence core).

    python -m brain2qwerty_v1_mamba.main cache [--debug] [--subjects ...]
    python -m brain2qwerty_v1_mamba.main debug [--core mamba|transformer]
    python -m brain2qwerty_v1_mamba.main train --core mamba        [--small]
    python -m brain2qwerty_v1_mamba.main train --core transformer  [--small]  # baseline
    python -m brain2qwerty_v1_mamba.main colab --core mamba                   # T4 preset
    python -m brain2qwerty_v1_mamba.main eval --ckpt <path> --core mamba [--small]

IMPORTANT: keep --small and --subjects identical between the mamba run and
the transformer baseline, and pass the same flags to eval as to train.
"""

import argparse
from pathlib import Path

import lightning.pytorch as pl
import torch

import studies  # noqa: F401  (registers Pinet2024Meg / Pinet2024Eeg)

import brain2qwerty_v1.transforms  # noqa: F401  (registers V1 transforms)
from brain2qwerty_v1.main import Experiment as _V1Experiment
from brain2qwerty_v1.metrics import CER
from brain2qwerty_v1.pl_module import BrainModule
from brain2qwerty_v1.utils import materialize_lazy_params

from . import mamba_core as _mamba_core  # noqa: F401  (registers BiMambaSentenceCore)
from . import transforms as _transforms  # noqa: F401  (registers V1MambaSubjectFilter)
from .config.xp_config import colab_config, debug_config, experiment_config


class Experiment(_V1Experiment):
    """V1 experiment with Apple Silicon (MPS) support and checkpoint resume.

    * ``_accelerator``: V1 only checks CUDA and otherwise falls back to CPU;
      this override uses the M1/M2/M3 GPU via MPS when available. If an op is
      not implemented for MPS, run with ``PYTORCH_ENABLE_MPS_FALLBACK=1``.
    * ``resume_from``: path to a ``last.ckpt`` to resume interrupted training
      (Kaggle sessions die at ~12 h); optimizer/scheduler/epoch state are
      restored by Lightning, so the run continues where it stopped.
    """

    resume_from: str | None = None

    def _accelerator(self) -> tuple[str, int]:
        if torch.cuda.is_available():
            return "gpu", max(1, min(self.devices, torch.cuda.device_count()))
        if torch.backends.mps.is_available():
            return "mps", 1
        return "cpu", 1

    def run(self) -> None:
        if not self.resume_from or self.eval_only:
            return super().run()
        pl.seed_everything(self.seed, workers=True)
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        loaders = self.data.build()
        brain, transformer = self._build_modules(loaders["train"])
        metrics = {"CER": CER()}
        self._module = BrainModule(
            model=brain,
            transformer=transformer,
            loss=self.loss.build(),
            metrics=metrics,
            optimizer=self.optimizer,
        )
        materialize_lazy_params(self._module, loaders["train"])
        self._trainer = self._trainer_setup()
        self._trainer.fit(
            self._module, loaders["train"], loaders["val"], ckpt_path=self.resume_from
        )
        if "test" in loaders:
            self._trainer.test(self._module, dataloaders=loaders["test"])


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--subjects", nargs="+", default=None,
                   help="participants, e.g. S15 S16 S6 (default: S15 S16 S6)")
    p.add_argument("--core", choices=["mamba", "transformer"], default="mamba",
                   help="sentence-level sequence core (transformer = V1 baseline)")
    p.add_argument("--small", action="store_true",
                   help="512-dim model instead of the paper's 2048 (Kaggle preset)")
    p.add_argument("--devices", type=int, default=None,
                   help="override GPU count (e.g. 2 for Kaggle T4 x2)")
    p.add_argument("--resume", default=None,
                   help="path to last.ckpt to resume interrupted training")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="brain2qwerty_v1_mamba")
    sub = parser.add_subparsers(dest="command", required=True)

    p_debug = sub.add_parser("debug", help="1-recording smoke test (small model)")
    _add_common(p_debug)

    p_train = sub.add_parser("train", help="full training (paper width unless --small)")
    _add_common(p_train)
    p_train.add_argument("--seed", type=int, default=None)

    p_colab = sub.add_parser("colab", help="Kaggle/Colab single-GPU preset (implies --small)")
    _add_common(p_colab)
    p_colab.add_argument("--seed", type=int, default=None)

    p_eval = sub.add_parser("eval", help="evaluate a checkpoint on the test split")
    _add_common(p_eval)
    p_eval.add_argument("--ckpt", required=True)

    p_cache = sub.add_parser("cache", help="pre-warm the feature cache")
    _add_common(p_cache)
    p_cache.add_argument("--debug", action="store_true", help="only the debug subset")
    p_cache.add_argument(
        "--timeline-query", default=None,
        help="neuralset study query on LONG-form ids, e.g. "
        "\"subject in ['Pinet2024Meg/S15','Pinet2024Meg/S16']\" "
        "(restricts which recordings are processed at all)",
    )

    args = parser.parse_args(argv)

    if args.command == "cache":
        if args.debug:
            cfg = debug_config(subjects=args.subjects, core=args.core)
        else:
            cfg = experiment_config(
                subjects=args.subjects, core=args.core, small=args.small,
                timeline_query=args.timeline_query,
            )
        print("[v1_mamba] pre-warming the feature cache...")
        Experiment(**cfg).data.build()
        print("[v1_mamba] cache warmed.")
        return

    if args.command == "debug":
        cfg = debug_config(subjects=args.subjects, core=args.core)
    elif args.command == "colab":
        cfg = colab_config(subjects=args.subjects, core=args.core, small=True)
    else:  # train / eval
        cfg = experiment_config(subjects=args.subjects, core=args.core, small=args.small)

    if args.command == "eval":
        cfg["eval_only"] = True
        cfg["ckpt_path"] = args.ckpt
    if getattr(args, "seed", None) is not None:
        cfg["seed"] = args.seed
    if getattr(args, "devices", None) is not None:
        cfg["devices"] = args.devices
    if getattr(args, "resume", None):
        cfg["resume_from"] = args.resume

    print(
        f"[v1_mamba] mode={args.command} core={args.core} "
        f"small={cfg['brain_model_config']['hidden'] == 512} "
        f"subjects={cfg['data']['transforms'][0]['subjects']} "
        f"out={cfg['output_dir']}"
    )
    Experiment(**cfg).run()


if __name__ == "__main__":
    main()
