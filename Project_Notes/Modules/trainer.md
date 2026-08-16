# Trainer Module

## Overview
Brain2Qwerty entirely delegates the training loop execution to **PyTorch Lightning**.

## `Experiment` class (`main.py`)
- Serves as the high-level API for configuring the trainer.
- Instantiates `pl.Trainer`.
- Configures Distributed Data Parallel (DDP) for multi-GPU training.

## Callbacks & Utilities
- **`callbacks.py`**:
  - `LogSentencePredictions`: A custom callback that captures test/val predictions, pairs them with ground truths, and dumps them into JSON for later LM processing.
- **`metrics.py`**:
  - Computes standard Character Error Rate (`CER`).
- **WandB**: Integrated via CLI arguments in `cli.py` to seamlessly track loss curves, configurations, and system metrics.

## Artifacts Generated
When training concludes, the trainer generates:
1. `logs/` (CSV metrics)
2. Wandb run syncs
3. `best.ckpt` and `last.ckpt` (model weights)
4. JSON predictions containing raw logits for each sentence.
