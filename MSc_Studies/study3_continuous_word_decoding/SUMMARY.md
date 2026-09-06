# Study 3 — Continuous, Multi-Second, Word-Level Decoding

> **Question:** can SSM-based cores decode **whole, uncut, multi-second MEG
> sentence recordings** into free text — the condition closest to a deployable
> real-time brain-to-text system — and do they beat the Conformer there?
>
> Sources: thesis Chapter 4 §4.7, `Project_Notes/10_Complete_Project_Experiments_and_Results_Summary.md`,
> `brain2qwerty_v3/config/xp_config.py`, `brain2qwerty_v3/README.md`,
> `slurm/13_v3_word_decoding.sbatch` / `18_v3_eval.sbatch` headers.

---

## 1. Design

This study discards the 500 ms keystroke-aligned window entirely. Models
consume **uncut sentence recordings** and decode free-text Spanish via the
Brain2Qwerty-V2-style three-stage pipeline, re-implemented self-contained in
`brain2qwerty_v3/` and retargeted from EnglishBCBL to **SpanishBCBL**:

1. **Character level:** conv frontend + sequence core → per-frame CTC head.
2. **Word level:** segmented neural word embeddings aligned to frozen LLM word
   embeddings with a SigLIP-style contrastive loss using **Hard DTW** alignment.
3. **Sentence level:** a LoRA-adapted causal LLM autoregressively generates the
   sentence conditioned on the neural word embeddings.

Three cores compared under an identical curriculum:

| Arm | Core | Params |
|---|---|---:|
| V1 | `conformer` — the V2 reference Conformer, ported to SpanishBCBL | 125.1M |
| V2 | `mamba_mlp` — BiMamba-2 + learned gated fusion + FFN (the Study-2 champion) | 128.3M |
| V3 | `mamba3_hybrid_stabilized` — Mamba-3 (BCNorm + RoPE + adaptive Δt clamping) interleaved with attention | **90.8M** |

## 2. Dataset & split

- SpanishBCBL / `Pinet2024Meg`, subjects S15, S16, S6 (9 session blocks).
- **No windowing:** whole sentence recordings of **3.0–12.8 s** (mean 6.84 s),
  **300–1,280 frames @ 100 Hz**; target sentences 15–78 characters.
- MEG preprocessing (`MegExtractor`): resample 100 Hz, band-pass 0.5–45 Hz,
  50 Hz notch, RobustScaler, clamp at ±5, 306 MEG channels (maxshield allowed).
- On-device train-time augmentation: constant per-channel offset (SD 0.3),
  SpecAugment-style time mask (50 frames, p 0.2) and freq mask (400), time
  stretch, and **10% channel dropout** for spatial robustness (added on top of
  the V2 recipe).
- Split: **leakage-free 80/10/10 via TF–IDF paraphrase clustering** (cosine
  threshold 0.5, seed 1) → **62 held-out test sentences**. Stronger than a
  random split: typing corpora contain many near-paraphrase variants.
- Label extractor: `SentenceKeySeq` (`typed_label` mode); transforms:
  `SpanishBCBLV2Preprocessing` → `SpanishBCBLV2Splitter(seed=1)` → `WordCreator`.

## 3. Training configuration

Staged 3-loss curriculum over **275 epochs**, seed 123, bf16-mixed,
**no early stopping** (early stopping on stage-1 CER would amputate the later
stages; checkpoints + manual log monitoring instead):

| Stage | Epochs | Active losses | Weights |
|---|---|---|---|
| 1 — CTC warmup | 0–149 | CTC | w_ctc = 1.0 |
| 2 — word alignment | 150–224 | CTC + SigLIP word-contrastive (Hard DTW) | w_ctc = 0.90, w_con = 0.10 |
| 3 — LLM decoding | 225–274 | + LoRA-adapted LLM | w_ctc = 0.89, w_con = 0.10, w_llm = 0.01 |

- Optimizer: AdamW + WarmupCosine (500 warmup steps, η_min 1e-6); **lr 8e-4
  for the Conformer** (V2's value), **3e-4 for the Mamba/linear-recurrence
  cores** (per the Study-1 finding); weight decay 1e-3; batch 16 ×
  gradient-accumulation 4.
- Contrastive weighting: `loss_alpha = 0.7`, `alpha = 0.1`, `beta = 0.01`.
- Inference: **16-beam search, length penalty 0.2**.
- LLM decoder: the dissertation results (below) used **TinyLlama-1.1B + LoRA
  rank 2**. The current code default is **Qwen3.5-0.8B + LoRA rank 8
  (α = 16)** — the later decoder-swap campaign (`slurm/17_v3_decoder_swap_qwen.sbatch`);
  pass matching flags when reproducing the published numbers.

## 4. Results (62-sentence held-out test set)

| Architecture | Params | Test WER ↓ | Word acc ↑ | Test CER ↓ | CTC CER ↓ | SemER ↓ | LLM loss ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Conformer (baseline) | 125.1M | 92.0% | 8.0% | 68.6% | 48.1% | 0.0970 | 3.812 |
| BiMamba-2 + Gated MLP | 128.3M | 76.0% | 24.0% | **57.7%** | **45.0%** | **0.0940** | **3.348** |
| Mamba-3 Stabilised Hybrid | **90.8M** | **75.4%** | **24.6%** | 60.6% | 50.4% | 0.0967 | 3.654 |

### Statistical validation (10,000-resample bootstrap + sentence-matched paired tests, vs Conformer)

| Comparison | Metric | Δ (reduction) | Bootstrap 95% CI on Δ | Paired p | Wilcoxon p | Cohen's d |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| BiMamba-2+MLP vs Conformer | WER | +16.0% | [+14.1%, +17.9%] | < 10⁻⁴ | < 10⁻¹⁰ | 2.17 |
| Mamba-3 Hybrid vs Conformer | WER | +16.6% | [+14.9%, +18.4%] | < 10⁻⁴ | < 10⁻¹⁰ | 2.34 |
| BiMamba-2+MLP vs Conformer | CER | +10.9% | [+9.2%, +12.7%] | < 10⁻⁴ | < 10⁻¹⁰ | 1.54 |
| Mamba-3 Hybrid vs Conformer | CER | +8.0% | [+6.9%, +9.2%] | < 10⁻⁴ | < 10⁻¹⁰ | 1.87 |

### Per-subject (Mamba-3 Stabilised Hybrid)

| Subject | Test sentences | Word acc | WER | Char acc | CER | SemER |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| S15 | 22 | 23.4% | 76.6% | 38.1% | 61.9% | 0.0984 |
| S16 | 27 | **28.2%** | **71.8%** | **42.5%** | **57.5%** | **0.0882** |
| S6 | 13 | 20.1% | 79.9% | 35.8% | 64.2% | 0.1085 |
| **Overall** | **62** | **24.6%** | **75.4%** | **39.4%** | **60.6%** | **0.0967** |

### Qualitative decodes (LLM stage)

- True `la silla ocasiona las lesiones` → CTC `las ciollreindascna las veieasoio`
  → output `las coseñas ocasionan las divisiones` (CER 0.40, SemER 0.0682) —
  verb root and feminine-plural syntax recovered.
- True `el signo conserva los resultados` → CTC `el ro cea los reutaos s`
  → output `el hilo conecta los nucleos` (CER 0.41, SemER 0.0718) — perfect
  syntactic frame, lexical substitution ("semantic healing" failure mode).

### Findings

1. **SSMs invert the Study-1 ordering on continuous signals:** both Mamba
   cores beat the Conformer by ~16% absolute WER — a **3× word-accuracy gain**
   (8.0% → 24.6%).
2. **BiMamba-2+MLP leads on characters/semantics** (CER 57.7%, CTC CER 45.0%,
   SemER 0.0940); **Mamba-3 Hybrid leads on words and efficiency** (WER 75.4%
   with 27.4% fewer params than the Conformer).
3. External anchoring: 75.4% WER in a 3-subject pilot (~10% of full-cohort
   data) sits between Défossez et al. 2023 (~55–65%, 169 subjects, zero-shot
   speech) and the full-cohort Brain2Qwerty V2 (~45–55%) — competitive under
   severe data scarcity, though high in absolute clinical terms.

## 5. Explainability findings (supporting this study)

- Mamba's step-size Δ_t shows **3.8× spikes at space characters** — an
  emergent word-boundary reset that drives the state decay toward zero between
  words (consistent with the non-uniform, space-dominated character
  distribution of the corpus).
- DTW cost matrices are near-diagonal and monotonic — learned neural word
  embeddings preserve chronological order (no token hallucination at the
  alignment stage).

## 6. Code layout (this folder)

```
brain2qwerty_v3/           # self-contained V2-style pipeline on SpanishBCBL
  models.py / mamba.py     # ConvConformer + pure-PyTorch Mamba-2/3 hybrid cores
  deltanet.py              # DeltaNet core (Study-4 extension, kept for completeness)
  gnn_frontend.py          # k-NN graph-attention frontend (Study-4 ablation)
  ctc_segmenter.py         # Hard-DTW word segmentation
  pl_module.py, losses.py, metrics.py, data.py, transforms.py, augmentations.py
  config/                  # xp_config (the curriculum above) + model_config (core registry)
  tests/                   # smoke suite (forward/backward/joint-loss; 10 tests)
studies/                   # SpanishBCBL (Pinet2024Meg) study registration
evaluate_accuracy.py       # overall + per-subject WER/CER/SemER scoring
statistical_testing.py     # bootstrap CIs + paired tests (the §4 table)
aggregate_multi_seed_results.py  # multi-seed aggregation
benchmark_complexity.py    # O(N) vs O(N²) runtime/VRAM profiler (A100)
train_full.py              # standalone full-training entry with explicit checkpointing
slurm/
  13_v3_word_decoding.sbatch               # the 3-core Study-3 suite (Conformer / M2+MLP / M3-hybrid)
  14_v3_full_dataset_and_benchmark.sbatch  # 1024-dim full-dataset arms + complexity profiler
  15_v3_multi_seed_evaluation.sbatch       # multi-seed robustness campaign
  16_v3_19subjects_full_dataset.sbatch     # 19-subject full-dataset campaign
  17_v3_decoder_swap_qwen.sbatch           # TinyLlama → Qwen3.5-0.8B decoder swap
  18_v3_eval.sbatch                        # evaluation suite for the 3 cores
```

Scripts 14–17 are extension campaigns beyond the dissertation's core Study-3
comparison; the headline numbers above come from the `13`/`18` suite.

## 7. How to run (Kelvin-2)

```bash
cd ~/sharedscratch/B2Q/B2Q_Mamba/brain2qwerty-mamba2
sbatch slurm/13_v3_word_decoding.sbatch           # train all 3 cores (array 0-2)
sbatch slurm/18_v3_eval.sbatch                    # evaluate all 3 (array 0-2)

# single arm, manually:
python -m brain2qwerty_v3.main eval --core mamba3_hybrid_stabilized \
  --frontend conv --ckpt <path>/best.ckpt
```

Note: v3's config auto-detects the cluster study/cache paths
(`~/sharedscratch/B2Q/...`); the LLM must be in the cluster's HuggingFace
cache (compute nodes have no internet).

## 8. Limitations

3-subject pilot; single seed for the headline numbers (multi-seed campaign in
`15_...`); the LLM decoder can "heal" into fluent but lexically wrong Spanish
(SemER partly mitigates); TinyLlama→Qwen decoder swap means exact reproduction
of the published numbers requires the TinyLlama configuration; no real-time
latency evaluation yet.
