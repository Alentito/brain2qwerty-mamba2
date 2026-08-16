# Project Overview: Brain2Qwerty V1

## Core Questions Answered

### 1. What happens first?
Execution starts at `brain2qwerty_v1/main.py` where the CLI parser determines the run mode (`train`, `debug`, `eval`, or `cache`). It selects the configuration from `config/xp_config.py` and initializes an `Experiment` Pydantic model. 

### 2. Where does the data come from?
The dataset (SpanishBCBL, containing MEG recordings and behavioral logs) is managed by the `neuralset` library. Raw data is downloaded from the Hugging Face Hub (`bcbl190626/SpanishBCBL`) into a local directory (`$BRAIN2QWERTY_STUDIES`). Features are cached locally in `$BRAIN2QWERTY_CACHE`.

### 3. How does it change?
Raw MEG `.fif` files go through several transformations:
1. **Event Extraction:** Keystroke timings are extracted.
2. **Windowing:** Sliced into 0.5s windows (from -0.2s to +0.3s around the keystroke).
3. **Preprocessing:** Extracted into `neuralset` segments, filtered (0.1-20 Hz), baselined, and scaled (RobustScaler).
4. **Batching:** The dataloader groups windows, ensuring that windows belonging to the same sentence stay on the same GPU rank.

### 4. When is the model created?
The model is instantiated inside `Experiment._build_modules()` when `Experiment.run()` is called. It creates:
- `brain`: A convolutional encoder (`SimpleConvTimeAgg`)
- `transformer`: A sentence-level transformer (`TransformerEncoder`)
These are then wrapped in a PyTorch Lightning `BrainModule` (`pl_module.py`).

### 5. How is training executed?
Training relies entirely on **PyTorch Lightning**. The `Experiment` class configures a `pl.Trainer` with a `DDPStrategy` for multi-GPU training. The trainer calls `.fit()` on the `BrainModule`, which iterates over the `DataLoader` batches.

### 6. How are checkpoints and metrics saved?
Checkpoints are saved by Lightning's `ModelCheckpoint` callback based on the validation Character Error Rate (`val_CER`). Metrics and losses are logged via `CSVLogger` and `WandbLogger`. Predictions are logged to JSON by a custom `LogSentencePredictions` callback.

### 7. How do all folders communicate?
- `main.py` is the conductor. It reads from `config/`, initializes models defined via `neuraltrain`, sets up data via `neuralset`, and wraps them in `pl_module.py`.
- PyTorch Lightning abstracts the training loop, calling `callbacks.py` for logging and `metrics.py` for CER computation.
- Output metrics and models are saved to the `output_dir` (default: `$BRAIN2QWERTY_RESULTS`).
