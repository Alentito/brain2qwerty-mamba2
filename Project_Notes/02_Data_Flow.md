# Data Flow

How a single MEG recording becomes a prediction:

1. **Raw `.fif` File**: Hosted on Hugging Face Hub (SpanishBCBL). Downloaded via `neuralset`.
2. **Event Parsing**: Keystrokes, Words, and Sentences timings are parsed into a standardized dataframe (`Data.build_events`).
3. **Transforms**: `SpanishBCBLPreprocessing` cleans up specific quirks of the dataset, and `Brain2QwertyV1Splitter` splits the data into train/val/test sets based on timeline indices.
4. **Windowing (Segments)**: Continuous MEG data is sliced into discrete 0.5-second windows (`start=-0.2`, `duration=0.5` around each keystroke) via `ns.SegmentDataset`.
5. **Preprocessing**: The `MegExtractor` applies bandpass filtering (0.1 - 20.0 Hz), baseline correction (0.0 to 0.2), and RobustScaling.
6. **DataLoader**: Windows are grouped into batches. Crucially, `SentenceGroupedDistributedSampler` ensures that all keystrokes belonging to a single sentence stay grouped together on the same GPU rank.
7. **GPU / PyTorch Lightning**: Batches move to the GPU inside `BrainModule.forward()`.
8. **Encoder**: The convolutional model extracts features from each individual MEG window (shape `[B, Channels, Time]`).
9. **Sentence Grouping**: Embeddings are grouped back into sentences based on their `sentence_UID`.
10. **Transformer**: Sequence modeling is applied over the sequence of keystroke embeddings to refine the predictions using surrounding context.
11. **Linear Projection**: Outputs are mapped to probabilities over `NUM_CLASSES` (29 characters).
12. **Loss**: CrossEntropyLoss is calculated against the ground truth keys.
