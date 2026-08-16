# Inference and Decoding

## Overview
Once a model is trained and `best.ckpt` is saved, the evaluation process happens in multiple stages.

## 1. Running the Checkpoint (`main.py eval`)
- The user runs `python -m brain2qwerty_v1.main eval --ckpt $BRAIN2QWERTY_RESULTS/best.ckpt`.
- The data pipeline is built for the test set.
- Lightning's `Trainer.test()` is called with `BrainModule.load_from_checkpoint()`.
- The model outputs log-probabilities per keystroke.
- The `LogSentencePredictions` callback writes these predictions to JSON files.

## 2. Prediction Extraction (`scripts/extract_predictions.py`)
- Takes the raw JSON callbacks output from testing.
- Formats them into an analysis-ready CSV file.
- Calculates initial Character Error Rates (CER) and Word Error Rates (WER) using naive argmax predictions.

## 3. N-gram Rescoring (`scripts/ngram_decoding.py`)
- This is a post-processing step entirely independent of PyTorch Lightning training.
- Takes the extracted CSV and applies a character-level N-gram language model (built with KenLM).
- Runs **beam search** over the network's keystroke predictions to find the most probable sentence that aligns with both the brain signal and the Spanish language rules.
- Outputs the final metrics (`predictions_with_lm.csv`).
