# MSc Project — Brain2Qwerty-Mamba: Studies 1–3

**Owner:** Alen (atito) · **Repo of origin:** github.com/Alentito/brain2qwerty-mamba2
**Research question:** Can Structured State-Space Models (Mamba-2 / Mamba-3) replace
the Transformer sentence-level core in non-invasive brain-to-text (B2T) decoding
from MEG?

This folder contains **only our own study code** for the three dissertation
studies (Chapter 4 of the thesis), plus shared tooling. Meta's reference
pipelines (`brain2qwerty_v1`, `brain2qwerty_v2`) are **not** included — they are
external dependencies, listed below.

## Folder map

| Folder | Study | Paradigm | Own package |
|---|---|---|---|
| `study1_core_ablation/` | **Study 1** — core-only sentence-core ablation (Transformer vs BiMamba-2/3) | 500 ms keystroke windows, 29-class CE | `brain2qwerty_v1_mamba/` |
| `study2_architecture_benchmark/` | **Study 2** — 7-architecture char-level benchmark + n-gram LM rescoring | same windows, CTC-style decoding + LM | `brain2qwerty_v1_mamba/` (extended cores) |
| `study3_continuous_word_decoding/` | **Study 3** — continuous multi-second word-level decoding (V2-style pipeline on SpanishBCBL) | uncut 3.0–12.8 s sentences, CTC + contrastive + LoRA-LLM | `brain2qwerty_v3/` |
| `shared_tools/` | EDA, animation, cluster scoring, auxiliary experiments | — | — |

Each study folder has a **`SUMMARY.md`** with the design, dataset split, sizes,
hyperparameters, results, and how to run.

## Shared dataset

All three studies use the **SpanishBCBL** MEG typing corpus (internally
`Pinet2024Meg`; 306-channel Elekta Neuromag Vectorview @ 1 kHz), restricted to a
**3-subject pilot cohort: S15, S16, S6** (9 usable session timelines after
excluding S6's corrupted 230502 block2/block3). The `studies/` package inside
each study folder registers this dataset with the `neuralset` framework.

## External dependencies (NOT included here)

- `brain2qwerty_v1` (Meta reference V1 pipeline) — **required to run Studies 1
  and 2**: our `brain2qwerty_v1_mamba` imports its `Experiment`, `BrainModule`,
  CER metric, char vocabulary and event transforms unchanged, so the ablation
  stays strictly core-only. Install from the main repo.
- `brain2qwerty_v2` (Meta V2 pipeline) — **not needed at runtime**;
  `brain2qwerty_v3` is a self-contained re-implementation of that pipeline
  retargeted to SpanishBCBL.
- Third-party: `neuralset`, `neuraltrain`, PyTorch, Lightning, and for Study 3
  also `transformers`, `peft`, `torchmetrics`. KenLM is optional (LM rescoring
  in Study 2; does not build on Kaggle).

## Headline results (verified — see each SUMMARY.md for sources)

| Study | Best SSM result | Best attention-baseline result | Reading |
|---|---|---|---|
| 1 (windows, no LM) | BiMamba-2, CER **0.304** (lr 3e-4) | Transformer, CER **0.246** (lr 1e-4) | attention wins on short windows; ~80% of the initial gap was LR tuning, not architecture |
| 2 (windows, LM rescoring) | BiMamba-2 + Gated MLP, CER **29.4%** | deep 8-layer Transformer, CER **25.9%** | SSM within 3.5 pts at half the depth |
| 3 (continuous, LLM decode) | Mamba-3 Hybrid, WER **75.4%** (90.8M params); BiMamba-2+MLP, CER **57.7%** | Conformer, WER 92.0% (125.1M params) | SSMs beat Conformer by ~16% absolute WER on continuous decoding |

## Environments used

- **Kelvin-2 HPC (QUB)** — primary training (k2-gpu-a100/h100/v100, ~3 h
  practical job wall-time; k2-hipri CPU); conda env `~/sharedscratch/conda/envs/b2q`
- **Kaggle T4×2** — backup/visualisation track (30 h/week)
- **Mac M1 Pro** — development; MPS support added to the Experiment class
