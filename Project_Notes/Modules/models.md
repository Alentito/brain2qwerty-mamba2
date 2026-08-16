# Models Module

## Overview
The architecture is divided into the PyTorch models themselves (imported dynamically from `neuraltrain`) and the PyTorch Lightning module that coordinates them.

## Configuration (`config/model_config.py`)
- `ENCODER`: A CNN configured as `SimpleConvTimeAgg` with 2D-Fourier embedding features to handle individual subjects. 
- `TRANSFORMER`: A `TransformerEncoder` designed to ingest sequences of keystrokes.

## `BrainModule` (`pl_module.py`)
- **Initialization**: Holds the encoder, transformer, final linear classifier, loss, optimizer, and metrics.
- **Forward Pass (`forward`)**: Feeds MEG data (`neuro`), subject identifiers (`subject_id`), and sensor layout (`channel_positions`) into the `model` (encoder).
- **Transformer Forward (`_transformer_forward`)**: Takes the resulting batch of isolated keystrokes and regroups them by their original sentence IDs. Creates variable-length padded sequences, feeds them to the transformer with an attention mask, and then unravels them back for the linear layer prediction.
- **Outputs**: Computes logits over characters which are fed to the loss function.
