# Brain2Qwerty — Combined Pipeline (`brain2qwerty_colab`)

A self-contained training package that combines:

- **Preprocessing: V1 + V3 combined** — V1's full SpanishBCBL cleaning and metadata
  (`SpanishBCBLPreprocessing`) merged with V3's CTC-target construction
  (`SpanishBCBLV2Preprocessing`) into a single `CombinedBCBLPreprocessing` transform,
  plus the V1 TF-IDF paraphrase-cluster split and V3's word-event creation.
- **Architecture: V3** — conv front-end + Nemotron-H-style **hybrid Mamba-2 / attention**
  sequence core (`MambaHybridCore`, pure PyTorch SSD, no `mamba-ssm` dependency),
  auxiliary CTC head, word-level SigLIP contrastive aligner, and a LoRA-adapted LLM
  decoder, trained with the staged 3-loss schedule.
- **Subject subset selection** — train on any subset of the 19 participants
  (`subjects=["S1", "S3"]`, `--subjects S1 S3`), resolved after V1's
  participant-merge rules.
- **Colab / Kaggle ready** — single-GPU preset (`colab_config`), ready-made
  notebooks, pinned `requirements.txt`, and a bundled copy of the `studies/`
  dataset definition so the folder works standalone.

## Folder layout

```
brain2qwerty_colab/
├── main.py                  # Data + Experiment; CLI: debug / train / colab / eval / cache
├── transforms.py            # CombinedBCBLPreprocessing (V1+V3), CombinedBCBLSplitter,
│                            # CombinedWordCreator, CombinedSentenceKeySeq
├── models.py                # ConvMambaEncoder (V3 encoder, self-contained)
├── mamba.py                 # HybridMambaEncoder + MambaHybridCore config (V3 core)
├── pl_module.py             # NeuroLLMModule: CTC + contrastive + LLM staged losses
├── data.py                  # SentenceDataset (onset jitter, padded collation)
├── augmentations.py         # on-device MEG augmentation (offset / SpecAugment / stretch)
├── ctc_segmenter.py         # CTC-space pseudo-word segmenter
├── losses.py, metrics.py, callbacks.py, utils.py, cli.py
├── config/
│   ├── model_config.py      # ENCODER (1024-dim) + small_encoder() (512-dim)
│   └── xp_config.py         # experiment_config / colab_config / debug_config
├── studies/                 # Pinet2024Meg (SpanishBCBL) study definition
├── scripts/extract_predictions.py
├── notebooks/
│   ├── Brain2Qwerty_Colab.ipynb
│   └── Brain2Qwerty_Kaggle.ipynb
├── tests/test_mamba.py      # shape/causality/HF-parity tests for the Mamba-2 stack
└── requirements.txt
```

## Preprocessing steps (what the combined transform does, in order)

`CombinedBCBLPreprocessing` runs both halves in one pass:

**Part 1 — V1 cleaning & metadata** (`brain2qwerty_v1/transforms.py`):

1. Normalise event types: `Button` / `DetectedButton` → `Keystroke`.
2. Drop the two practice trials (`trial_id` 0 and 1) of every block.
3. Participant bookkeeping: drop the 5 no-keyboard controls and the excluded
   subject (metallic implant), merge duplicate recordings into the same person
   (`S18→S1`, `S14→S4`, `S10→S5`, `S21→S5`) → the 19 unique participants.
4. **(new) Optional subject subset selection** — before integer factorisation.
5. Factorise `subject` to integer ids; build `sentence_UID` per sentence; drop the
   known corrupted S1 block-1 sentence.
6. V1 metadata: per-keystroke `button_unique_id`; rebuild one `Sentence` event per
   keystroke group spanning its keystrokes; propagate the ground-truth text;
   `sentence_typed` (full typed string with `<special>→@`, `<space>→" "`,
   `<number>→9`); drop unused study-level columns.

**Part 2 — V3/V2 CTC targets** (`brain2qwerty_v3/transforms.py`):

7. Normalise buttons: `<space>` → `&`, drop `<special>` / `<number>` tokens and any
   keystroke outside the 27-character CTC vocabulary (`a–z` + `&`, blank = 0).
8. Integer `typed_key_int` per keystroke; space-separated `typed_label` CTC target
   per sentence; drop sentences with too few keystrokes (< 50% of events).
9. Keep only `Sentence` / `Keystroke` / `Meg` rows (RSVP perception-phase `Word`
   rows are dropped — the contrastive loss uses its own Word events).

Then:

- `CombinedBCBLSplitter(seed=1)` — V1's TF-IDF paraphrase-cluster split (clusters of
  similar sentences are allocated greedily to hit 80/10/10 ratios in keystrokes, so
  paraphrases never leak across splits), propagated per `sentence_UID` so Sentence
  rows carry the split (V3 behaviour).
- `CombinedWordCreator` — one `Word` event per whitespace token of each Sentence
  (word order + left context) for the frozen-LLM contrastive target.
- `CombinedSentenceKeySeq` — the integer character sequence the CTC head predicts
  (`typed_label` of what was actually typed, or the reference text).
- Segment extraction: sentences windowed from −0.4 s with a random 0.4–0.5 s tail;
  MEG at 100 Hz, 0.5–45 Hz bandpass + 50 Hz notch, RobustScaler, clamp 5; train-time
  sentence-onset jitter; on-device augmentation (per-channel offset, SpecAugment
  time/freq masking, time-stretch).

## Architecture (V3)

`ConvMambaEncoder`: SimpleConv front-end (depth 4, hidden 1500, per-subject 2D-Fourier
channel merger → 270 virtual channels) → temporal downsampling (kernel 16, stride 4)
→ `MambaHybridCore`: 8 pre-norm RMSNorm blocks in the pattern **M M M A M M M A**
(Mamba-2 SSD mixers, d_state 128 / headdim 64 / expand 2, with one global
x-transformers attention block every 4 blocks), dim 1024 → auxiliary CTC head
(`z_aux` blended back) + final CTC logits (29 classes). The CTC head then drives the
word segmenter → SigLIP contrastive alignment to frozen TinyLlama word embeddings →
LoRA (rank 2) TinyLlama decoder generating the sentence from `[CTC text, MEG word
embeds]`. Losses are staged: CTC from epoch 0, +contrastive at 150, +LLM at 225
(full config) or 0/100/150 (Colab preset).

## Quickstart (local / cluster)

```bash
pip install -r brain2qwerty_colab/requirements.txt   # plus torch==2.6.0 on a bare env

# run from the repository root (the directory containing brain2qwerty_colab/)
python -m brain2qwerty_colab.main debug                      # 1-timeline smoke test
python -m brain2qwerty_colab.main train                      # full run (all 19 subjects)
python -m brain2qwerty_colab.main train --subjects S1 S3 S7  # subset of participants
python -m brain2qwerty_colab.main colab --small --subjects S1  # single-GPU preset
python -m brain2qwerty_colab.main eval --ckpt $BRAIN2QWERTY_RESULTS/best_llm.ckpt
python -m brain2qwerty_colab.scripts.extract_predictions \
    --input $BRAIN2QWERTY_RESULTS --split test
```

Environment variables: `BRAIN2QWERTY_STUDIES` (raw data), `BRAIN2QWERTY_CACHE`
(feature cache), `BRAIN2QWERTY_RESULTS` (checkpoints/logs). The SpanishBCBL dataset
is a gated Hugging Face repo (`bcbl190626/SpanishBCBL`) — authenticate once with
`hf auth login` or `HF_TOKEN`; it downloads automatically on first use.

## Subject selection

Three levels, combinable:

| Level | Where | Effect |
|---|---|---|
| `subjects=["S1", "S3"]` | `CombinedBCBLPreprocessing` | keeps only those participants' events (after V1 merge rules; accepts `"S1"`, `1`, `"Pinet2024Meg/S1"`) |
| `timeline_query="subject in ['S1','S3']"` | study config | skips loading/processing other recordings entirely (raw, pre-merge ids) — recommended on Colab/Kaggle |
| `--subjects S1 S3` | CLI | same as `subjects=`, for `debug`/`train`/`colab`/`cache` |

## Colab / Kaggle

Use the notebooks in `notebooks/` — they install the pinned dependencies, mount
Drive / attach the code, authenticate to HF, pre-warm the cache, train with
`colab_config()` (single GPU, batch 32, 2 workers, greedy val decoding, 200-epoch
compressed schedule) and evaluate. On free-tier GPUs, use `small=True`
(512-dim encoder) and 1–3 subjects.

## Preprocess on a cluster, train on Kaggle/Colab

Everything expensive (study event building, MEG feature extraction, CTC labels)
is cached on disk under `$BRAIN2QWERTY_CACHE`, so you can do the slow CPU work on
your cluster and ship only the warmed cache to Kaggle/Colab:

**1. On the cluster** — warm the cache with *exactly* the config you will train
with (same subjects, same transform code; cache keys are content/config-based):

```bash
export BRAIN2QWERTY_CACHE=/path/to/b2q_cache
# debug first (1 timeline) to validate the path end-to-end, then the full subset:
python -m brain2qwerty_colab.main cache --debug --subjects S1 S2 S3
python -m brain2qwerty_colab.main cache --subjects S1 S2 S3
```

**2. Package and upload** — archive the cache folder and upload it as a Kaggle
Dataset (or to Drive for Colab):

```bash
tar -czf b2q_cache_s1-s3.tar.gz -C /path/to b2q_cache
```

**3. On Kaggle/Colab** — attach the dataset, extract, and point the env var at it
*before* importing the package:

```python
import os
os.environ["BRAIN2QWERTY_CACHE"] = "/kaggle/working/b2q_cache"   # extracted cache
# raw study path can stay empty as long as every lookup hits the cache
```

Caveats:

- **Cache keys must match exactly**: install the pinned `requirements.txt` on
  Kaggle/Colab (same `neuralset`/`exca` versions) and use the same `subjects`
  list and the same code version of this folder. A mismatch silently recomputes
  (and would then need the raw data).
- **Verify before training**: run the debug cell first
  (`Experiment(**debug_config(subjects=...)).data.build()`). If it finishes
  quickly without downloading anything, the cache is being hit; if it starts
  reading FIF files, the raw study folder must be uploaded too.
- The LLM (TinyLlama) and RoBERTa weights are *not* in this cache — they
  download from HF at module-build/test time regardless (small, one-time).

## Tests

```bash
pytest brain2qwerty_colab/tests/test_mamba.py -v   # shapes, causality, HF parity
```

## Naming note

Registered config classes use collision-free names (`ConvMambaEncoder`,
`MambaHybridCore`, `CombinedBCBL*`, `MegChannelPositions2D`) so this package can be
imported alongside `brain2qwerty_v1/v2/v3` in one process without exca registry
conflicts. The architecture and numerics are identical to V3.
