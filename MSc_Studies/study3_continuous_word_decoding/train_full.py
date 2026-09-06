#!/usr/bin/env python
"""Full Brain2Qwerty V3 training run with explicit checkpoint saving.

Runs the complete staged schedule from `brain2qwerty_v3.config.xp_config
.experiment_config()` (SpanishBCBL MEG, hybrid Mamba-2 encoder):

    CTC from epoch 0  ->  +word-contrastive at 150  ->  +LoRA-LLM at 225
    (275 epochs total, AdamW + warmup/cosine, bf16-mixed, DDP if >1 GPU)

Checkpoints written to $BRAIN2QWERTY_RESULTS (or --output-dir):

    best_ctc.ckpt   best val/cer_epo  (best encoder)
    best_llm.ckpt   best val/WER      (best full pipeline)
    last.ckpt       last epoch        (auto-saved by ModelCheckpoint)
    final.ckpt      saved explicitly after fit() (weights + full trainer
                    state: optimizer, scheduler, epoch — use to resume)

Usage:
    python train_full.py                      # full run, 8 GPUs (auto-falls back)
    python train_full.py --devices 1          # single GPU
    python train_full.py --resume last.ckpt   # resume from a checkpoint
    python train_full.py --seed 7             # different split/init seed
"""

import argparse
import logging
import sys
from pathlib import Path

import studies  # noqa: F401  (registers the Pinet2024Meg / SpanishBCBL study)

from brain2qwerty_v3.config.xp_config import experiment_config
from brain2qwerty_v3.main import Experiment

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("train_full")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=None,
                        help="checkpoint/log dir (default: $BRAIN2QWERTY_RESULTS)")
    parser.add_argument("--devices", type=int, default=None,
                        help="GPUs per node (default: 8, auto-falls back to 1)")
    parser.add_argument("--seed", type=int, default=None, help="override the seed")
    parser.add_argument("--resume", default=None,
                        help="checkpoint (.ckpt) to resume full trainer state from")
    parser.add_argument("--skip-test", action="store_true",
                        help="do not evaluate the best checkpoint on the test split")
    args = parser.parse_args()

    cfg = experiment_config()
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    if args.devices:
        cfg["devices"] = args.devices
    if args.seed is not None:
        cfg["seed"] = args.seed

    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    log.info("checkpoints -> %s", out)

    xp = Experiment(**cfg)

    # --- build data, model, trainer -------------------------------------
    loaders = xp.data.build()
    xp._module = xp._build_module(loaders)
    xp._trainer = xp._trainer_setup()  # attaches best_ctc / best_llm / last callbacks

    # --- full training ---------------------------------------------------
    xp._trainer.fit(
        xp._module,
        loaders["train"],
        loaders.get("val"),
        ckpt_path=args.resume,
    )

    # --- explicit final checkpoint (weights + optimizer/scheduler/epoch) --
    final_path = out / "final.ckpt"
    xp._trainer.save_checkpoint(str(final_path))
    log.info("saved final checkpoint: %s", final_path)

    # --- evaluate the best pipeline checkpoint on the test split ---------
    if not args.skip_test and "test" in loaders:
        best = out / "best_llm.ckpt"
        ckpt = str(best) if best.exists() else None
        log.info("test evaluation with %s", ckpt or "final weights")
        xp._trainer.test(xp._module, dataloaders=loaders["test"], ckpt_path=ckpt)

    log.info("done. checkpoints in %s:", out)
    for name in ("best_ctc.ckpt", "best_llm.ckpt", "last.ckpt", "final.ckpt"):
        p = out / name
        log.info("  %s %s", name, "(saved)" if p.exists() else "(missing)")

    if not (out / "best_ctc.ckpt").exists():
        sys.exit("training finished but no checkpoint was saved — check the logs above")


if __name__ == "__main__":
    main()
