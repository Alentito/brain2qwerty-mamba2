# Brain2Qwerty V1 — Mamba Sentence-Core Ablation (`brain2qwerty_v1_mamba`)

Strict Option-A ablation: **everything is V1** (keystroke-aligned 500 ms MEG
windows, `SimpleConvTimeAgg` conv encoder, 29-class character head,
cross-entropy, AdamW + OneCycleLR) **except the sentence-level sequence core**:

| | V1 reference | This package |
|---|---|---|
| Sentence core | `TransformerEncoder` (depth 4, heads 2, ALiBi, bidirectional) | `BiMambaSentenceCore` — 4 bidirectional Mamba-2 (SSD) blocks: one forward + one backward mixer per block, summed |
| Default subjects | all 19 | **S15, S16, S6** (the strongest MEG subjects; S15 is the paper's best, CER 0.287) |

The Mamba-2 mixer is pure PyTorch (copied from `brain2qwerty_colab/mamba.py`,
no `mamba-ssm` CUDA dependency), so it runs on Kaggle/Colab T4s and CPU.
Bidirectionality restores the context the V1 transformer has — a causal Mamba
would not be a fair comparison.

## Commands

```bash
pip install -r requirements.lock && pip install -e . --no-deps   # once

# smoke test (one S15 recording, small model, 2 epochs)
python -m brain2qwerty_v1_mamba.main debug

# the two ablation arms — identical except the core:
python -m brain2qwerty_v1_mamba.main train --core mamba       --small   # treatment
python -m brain2qwerty_v1_mamba.main train --core transformer --small   # baseline

# Kaggle/Colab single-GPU preset (implies --small, batch 32, 200 epochs)
python -m brain2qwerty_v1_mamba.main colab --core mamba
python -m brain2qwerty_v1_mamba.main colab --core transformer

# evaluate (pass the same --core/--small/--subjects used at train time)
python -m brain2qwerty_v1_mamba.main eval --core mamba --small \
    --ckpt $BRAIN2QWERTY_RESULTS/small-mamba-S15-S16-S6/best.ckpt
```

Keep `--core`, `--small` and `--subjects` identical between train and eval.
Each (core, size, subjects) combo writes to its own output dir
(`$BRAIN2QWERTY_RESULTS/<small-><core>-S15-S16-S6/`), so runs never clash.

## Preprocess on the cluster, train on Kaggle

Warm the cache on k2-hipri (CPU-only), restricted to the three subjects:

```bash
export BRAIN2QWERTY_CACHE=/path/to/b2q_cache
python -m brain2qwerty_v1_mamba.main cache \
    --subjects S15 S16 S6 --timeline-query "subject in ['S15','S16','S6']"
tar -czf b2q_cache_v1mamba.tar.gz -C /path/to b2q_cache   # upload as Kaggle Dataset
```

On Kaggle: attach the dataset, extract, set `BRAIN2QWERTY_CACHE` to the
extracted folder **before** importing the package, run `debug` first to verify
the cache is hit (it must not start reading FIF files), then `colab`.

## Output

Same artifacts as V1: `best.ckpt` / `last.ckpt`, CSV logs, and the test-time
per-sentence prediction JSON under `callbacks/` — post-process with V1's
`scripts/extract_predictions.py` (per-sentence CER/WER per subject) and
optionally `scripts/ngram_decoding.py` for the +LM numbers. The comparison
metric is **test CER of mamba vs transformer** on the same subjects/split.
