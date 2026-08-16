# Dataset Module

## Overview
The dataset layer acts as the bridge between raw MEG/EEG signals and PyTorch tensors, leveraging the `neuralset` library.

## Components

### `Data` Class (in `main.py`)
- A Pydantic config class that constructs the data pipeline.
- Downloads data using `ns.events.Study.run()`.
- Applies dataset-specific transforms (e.g., specific to SpanishBCBL).

### `EventsTransform` (in `transforms.py`)
- Contains classes like `SpanishBCBLPreprocessing` to fix metadata in the dataset and `Brain2QwertyV1Splitter` to allocate data splits based on timeline constraints.

### `Extractors`
- `MegExtractor`: Reads from the `.fif` files, processes the signal (filtering, scaling, baselining).
- `LabelEncoder`: Encodes the ground truth (`button` presses on the keyboard) to a categorical label.

### DataLoaders
- Uses `ns.SegmentDataset` to group the slices of time into PyTorch Datasets.
- Employs a custom `SentenceGroupedDistributedSampler` to guarantee that all characters of a sentence are processed on the exact same GPU node, which is essential for the Transformer's sentence-level attention.
