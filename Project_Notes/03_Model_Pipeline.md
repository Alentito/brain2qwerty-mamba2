# Model Pipeline

## Overview
The architecture is defined by two main components instantiated in `Experiment._build_modules()` via the `neuraltrain` library configurations.

## 1. Convolutional Encoder
- **Defined by**: `ENCODER` dictionary in `config/model_config.py`.
- **Name**: `SimpleConvTimeAgg`
- **Function**: Takes raw MEG signal windows (e.g., 0.5 seconds, 50 Hz, 270 virtual channels after merging).
- **Features**: Uses a per-subject 2D-Fourier channel merger. It has 8 depth layers, residual skips, GeLU activations, and produces a single embedding (`total_dim: 2048`) per keystroke window.

## 2. Sentence-level Transformer
- **Defined by**: `TRANSFORMER` dictionary in `config/model_config.py`.
- **Name**: `TransformerEncoder`
- **Function**: Takes a sequence of encoder embeddings grouped by sentence.
- **Features**: 4 layers deep, 2 attention heads, uses ALiBi positional biases. Refines the local keystroke representation using contextual information from surrounding keystrokes.

## 3. The PyTorch Lightning Wrapper (`BrainModule`)
- Located in `pl_module.py`.
- `forward()`: Passes the batch through the Convolutional Encoder.
- `_transformer_forward()`: Restructures the flat batch of embeddings into sequences grouped by `sentence_UID`. Pads the sequences, creates a mask, passes them through the Transformer, and flattens them back.
- **Output**: A final Linear layer maps the `hidden` dimension (2048) to `NUM_CLASSES` (29).
