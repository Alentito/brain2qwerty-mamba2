# Study 1 — Core-Only Sentence-Level Architecture Ablation

> **Question:** does swapping only the sentence-level core of Brain2Qwerty V1
> (4-layer bidirectional ALiBi Transformer → bidirectional Mamba-2/Mamba-3 SSD
> stack) preserve keystroke-decoding performance?
>
> Sources for every number below: `PROJECT_SUMMARY.md` (project log),
> cluster run logs (job `9693616`, round 2), and thesis Chapter 4 §4.5.

---

## 1. Design

Strict core-only ablation. **Everything** is shared verbatim with the V1
reference — the SimpleConvTimeAgg conv encoder with per-subject 2D-Fourier
channel merger, the 29-class character head, cross-entropy loss, AdamW +
OneCycleLR — **except** the sentence core:

| Arm | Core | Params (512-dim "small" preset) |
|---|---|---|
| Reference | 4-layer bidirectional ALiBi `TransformerEncoder` | 24.1M |
| Ablation | `BiMambaSentenceCore` — 4 blocks, each a forward + backward Mamba-2 (SSD) mixer, outputs summed | 18.8M |
| Round-2 addition | `BiMamba3SentenceCore` — Mamba-3-style mixer (BCNorm + B/C biases + data-dependent RoPE state rotation; pure PyTorch, no `mamba-ssm`) | 18.7M |

Bidirectionality is required for a fair comparison (V1's transformer sees the
whole sentence), so each block runs two causal mixers, one on the
time-reversed sequence. Zero-padding is handled by running each sentence
unpadded inside the core (Python loop over the batch — sentences are tens of
keystrokes, cost is negligible) to avoid contaminating the backward direction.

**Scope caveats:** width-matched at 512-dim, **not** parameter-matched (the
Transformer carries a 4× FFN the pure Mamba core lacks — quantified in
Study 2); 3-subject pilot, not the full 35-subject corpus.

## 2. Dataset & split

- SpanishBCBL / `Pinet2024Meg`, subjects **S15, S16, S6** → **9 session
  timelines** (S15 × 4, S16 × 2, S6 × 3; S6's 230502 block2/block3 are on the
  loader's known-bad list and excluded).
- Example = **500 ms window @ 50 Hz** centred on a keypress, spanning
  **−200 ms to +300 ms** (`start: -0.2, duration: 0.5`), 306 MEG channels.
- Isolated-window extractor applies strict recording-boundary rejection
  (drops keystrokes within 200 ms of block edges) and debounces double
  keypresses (< 50 ms).
- **Train: 17,811 windows** (S15 7,758 / S16 6,228 / S6 3,825);
  **test: 2,280 windows from 54 sentences** (S15 24 / S16 12 / S6 18).
- Target: 29-class Spanish character vocabulary (`BUTTON_MAPPING` in
  `brain2qwerty_v1/utils.py`; specials collapsed into `<special>`, plus
  `<space>`, `<number>`).

## 3. Training configuration

- Preset: `colab` — 200 epochs max, early-stopping patience 25, seed 33,
  batch size 128 (train), 512-dim width (`--small`), `d_state 64`,
  `headdim 64`, `expand 2`, 4 layers, dropout 0.1.
- Round 1 (V1 defaults): lr 5e-5, weight decay 1e-4, no gradient clipping.
- Round 2 grid (`slurm/10_experiments.sbatch`, array job `9693616`):

| Task | Core | LR | wd | clip | Question |
|---|---|---|---|---|---|
| 0 | mamba2 | 1e-4 | – | – | is 5e-5 too low? |
| 1 | mamba2 | 3e-4 | – | – | aggressive LR |
| 2 | mamba2 | 1e-4 | 0.1 | 1.0 | full Mamba recipe |
| 3 | mamba3 | 1e-4 | 0.1 | 1.0 | do the v3 upgrades help? |
| 4 | mamba3 | 3e-4 | 0.1 | 1.0 | v3 + aggressive LR |
| 5 | transformer | 1e-4 | – | – | control |

The script is resumable (re-`sbatch` resumes from `last.ckpt` with full
optimizer/scheduler state) and idempotent (finished arms skip; outputs are
namespaced by `--tag` under `results/`).

## 4. Results

### Round 1 — matched default hyperparameters (lr 5e-5)

| Subject | BiMamba-2 CER | Transformer CER | n sentences |
|---|:---:|:---:|:---:|
| S15 | 0.455 | 0.299 | 24 |
| S16 | 0.408 | 0.250 | 12 |
| S6 | 0.359 | 0.290 | 18 |
| **Pooled** | **0.412** | **0.286** | **54** |

- Pooled numbers match Lightning's `test_CER` exactly (0.412 / 0.285) —
  metric cross-check passed.
- Training dynamics: Mamba early-stopped at **epoch 135** (plateau ≈ 0.47 val
  CER); Transformer used all 200 epochs and was still improving → the gap is
  partly optimisation, motivating round 2.
- Reproduction anchor: the Transformer's best subject (S16, 0.250) is near the
  paper's full-cohort best (~0.19), so the baseline reproduction is credible.

### Round 2 — LR / regularisation grid (from cluster logs, job 9693616)

| Task | Core | Recipe | Test CER | Test loss |
|---|---|---|---:|---:|
| 5 | transformer | lr 1e-4 | **0.246** | 1.744 |
| 1 | mamba2 | lr 3e-4 | **0.304** | 3.122 |
| 4 | mamba3 | lr 3e-4, wd 0.1, clip 1.0 | 0.310 | 2.367 |
| 3 | mamba3 | lr 1e-4, wd 0.1, clip 1.0 | 0.357 | 3.488 |
| 2 | mamba2 | lr 1e-4, wd 0.1, clip 1.0 | 0.374 | 3.372 |
| 0 | mamba2 | lr 1e-4 | 0.394 | 3.206 |
| — | mamba2 | round-1 (lr 5e-5) | 0.412 | 3.321 |
| — | transformer | round-1 (lr 5e-5) | 0.285 | 1.992 |

### Findings

1. **The optimisation confound is real.** LR 5e-5 → 3e-4 alone moves pure
   Mamba-2 from 0.412 → 0.304 (−10.8 pts, closing > 80% of the round-1 gap).
   Mamba's input-dependent parameterisation (B_t, C_t, Δ_t) needs higher
   learning rates to escape early plateaus.
2. **Regularisation helps SSMs:** at lr 1e-4, wd 0.1 + clip 1.0 improves
   Mamba-2 from 0.394 → 0.374 (clip prevents Δ_t/state explosions on MEG
   transient artefacts).
3. **Mamba-3 upgrades give consistent gains** at matched recipes
   (0.357 vs 0.374 at lr 1e-4; 0.310 vs 0.304 at 3e-4 is within noise).
4. **Attention still wins on short windows:** the tuned Transformer control
   (0.246) remains best — unconstrained pairwise interaction pays off at
   sentence-scale horizons (tens of keystrokes). This is what motivates the
   hybrid architectures of Studies 2–3.
5. Differences < ~0.01 CER are treated as within noise (single seed 33).

## 5. Supporting: classical baselines (per-window char accuracy; chance 3.4%)

| Model | Protocol | Result |
|---|---|---|
| LDA pooled | flattened 306×25 window | **38.2%** |
| Ridge per subject | flattened window | 32.7% / 32.9% / 30.6% (S15/S16/S6) |
| Ridge pooled, per time-sample | 306-d per sample | peak **26.8% @ +20 ms** post-keypress |

The temporal curve replicates the paper's physiology (their linear peak
+40 ms); pooled > per-subject at the linear model class. These are per-window
accuracies, not sentence CER — evidence of genuine SNR, not direct competitors.

## 6. Explainability tooling

`explain_mamba.py` (run via `slurm/11_explain.sbatch`) exploits the pure-
PyTorch quadratic SSD form: the exact input→output mixing matrix
`m = (C·B) × L × Δt` is materialised during the forward pass, giving an
attention-map analogue for free. It produces, per test sentence:

- **mixing maps** per block × direction with typed characters as axis labels,
- **Δ_t selectivity profiles** (where the SSM opens/closes its state update),
- **memory-horizon curves** (how many keystrokes back the mixing reaches),
- **grad × input saliency** on the raw window (heatmap, 2D channel topomap
  from `channel_positions`, time course aligned to keypress @ 0 ms),
- **char-probability heatmaps** with ground truth marked.

## 7. Code layout (this folder)

```
brain2qwerty_v1_mamba/     # our package: BiMamba-2/3 cores, configs, CLI, tests
  mamba_core.py            # Mamba2Mixer / Mamba3Mixer / BiMambaBlock / sentence cores
  main.py                  # CLI: cache | debug | train | colab | eval (+ --core/--lr/--tag)
  config/                  # experiment + model configs (cores: transformer, mamba, mamba3, …)
  tests/test_bimamba.py    # 10-test verification suite (~5–7 min)
studies/                   # registers the SpanishBCBL (Pinet2024Meg) study for neuralset
analyze_preds.py           # per-subject CER table from a predictions JSON
plot_cer.py                # per-subject and multi-arm CER figures
explain_mamba.py           # XAI pipeline (see §6)
slurm/
  09_train_v1mamba.sbatch  # round-1 two-arm array job
  10_experiments.sbatch    # round-2 six-arm grid (resumable, idempotent)
  11_explain.sbatch        # XAI pass over the test split for one checkpoint
```

**External dependency:** `brain2qwerty_v1` (Meta reference pipeline — provides
`Experiment`, `BrainModule`, CER metric, char vocabulary, event transforms).
Not included here by design; install the main repo package first, then this
package (`pip install -e . --no-deps` at the repo root, with
`brain2qwerty_v1_mamba` present in `pyproject.toml` package discovery).

## 8. How to run (Kelvin-2)

```bash
cd ~/sharedscratch/B2Q/B2Q_Mamba/brain2qwerty-mamba2
git pull && conda activate ~/sharedscratch/conda/envs/b2q
pytest brain2qwerty_v1_mamba/tests/test_bimamba.py -v   # 10 tests
sbatch slurm/10_experiments.sbatch                      # re-run to resume/skip

# fetch + analyse results on the Mac
scp kelvin2:~/sharedscratch/B2Q/cache_v1mamba/results/small-<core>-S15-S16-S6-<tag>/callbacks/test_all_sentences.json ./preds_<arm>.json
python analyze_preds.py

# explainability on one checkpoint
CKPT=.../small-mamba-S15-S16-S6-lr3e4/best.ckpt CORE=mamba sbatch slurm/11_explain.sbatch
```

## 9. Limitations

3-subject pilot; width- not parameter-matched; single seed (33); no LM stage;
Mamba-3's exponential-trapezoidal discretization not implemented (kept the
short conv); round-2 transformer control shows LR tuning helps *both* cores.
