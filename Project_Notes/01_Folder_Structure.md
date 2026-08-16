# Folder Structure

## `brain2qwerty_v1/`

| Folder/File | Purpose | Used By |
| --- | --- | --- |
| `config/` | Stores Hydra/Pydantic experiment and model settings (`xp_config.py`, `model_config.py`). | `main.py` |
| `scripts/` | Post-training scripts for inference (`extract_predictions.py`, `ngram_decoding.py`). | User (CLI) |
| `resources/` | Static assets like architecture diagrams. | Documentation |
| `main.py` | The main entry point for the pipeline. Initializes data, models, and starts training. | User (CLI) |
| `cli.py` | Argument parsing extensions (e.g., WandB integration). | `main.py` |
| `pl_module.py` | PyTorch Lightning module that glues the encoder, transformer, and loss function together. | `main.py` |
| `callbacks.py` | Custom PyTorch Lightning callbacks (e.g., logging sentence predictions). | `main.py` |
| `metrics.py` | Evaluation metrics (Character Error Rate - CER). | `pl_module.py` |
| `transforms.py` | Data preprocessing transforms (e.g., `Brain2QwertyV1Splitter`, `SpanishBCBLPreprocessing`). | `main.py` -> `neuralset` |
| `utils.py` | Helper functions, constants, and distributed samplers. | Multiple |

## External Dependencies
The project heavily relies on:
- **`neuralset`**: Handles dataset downloading, event alignment, segment extraction, and dataloading.
- **`neuraltrain`**: Provides generic training utilities, model definitions, losses, and optimizers.
- **`lightning`**: PyTorch Lightning for the training loop and multi-GPU orchestration.
