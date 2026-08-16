# Training Loop

## Overview
The training loop is abstracted away by PyTorch Lightning. The orchestration happens inside `main.py` -> `Experiment._trainer_setup()`.

## Flow
1. **DDP Strategy**: If multiple GPUs are requested, PyTorch Lightning sets up `DDPStrategy(find_unused_parameters=True)`.
2. **Optimizer**: Configured in `config/xp_config.py` as an `AdamW` optimizer (learning rate `5e-5`, weight decay `1e-4`).
3. **Scheduler**: `OneCycleLR` (max_lr `5e-5`, pct_start `0.1`).
4. **Loss**: standard `CrossEntropyLoss` since this is a classification problem over 29 classes.
5. **Execution**:
   - `Trainer.fit()` handles iterating through epochs and batches.
   - For every batch, `BrainModule.training_step()` runs the forward pass and computes the loss.
   - Lightning automatically handles `.backward()` and `optimizer.step()`.
   - Gradients are clipped based on `grad_max_norm`.

## Checkpointing and Logging
- **EarlyStopping**: Halts training if `val_CER` (Validation Character Error Rate) doesn't improve for `patience` (30) epochs.
- **ModelCheckpoint**: Monitors `val_CER` and saves the best model to `$BRAIN2QWERTY_RESULTS/best.ckpt`.
- **Logging**:
   - Logs to CSV in `output_dir`.
   - Optionally logs to Weights & Biases (Wandb) if configured.
   - Uses a custom callback `LogSentencePredictions` to dump actual predictions to a JSON file during validation/testing.
