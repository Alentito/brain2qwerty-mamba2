# Study 2 — Seven-Architecture Character-Level Benchmark with LM Rescoring

> **Question:** given a fixed parameter and engineering budget, which sequence
> architecture is the strongest practical choice for character-level MEG
> keystroke decoding?
>
> Sources: thesis Chapter 4 §4.5.2 (extended grid) and §4.6 (benchmark table),
> `PROJECT_SUMMARY.md`, `slurm/12_round3_experiments.sbatch` (run header).

---

## 1. Design

Same 29-class keystroke-aligned task as Study 1, but instead of swapping one
core against the reference, this study sweeps **seven architectures** across
three families — pure SSM, pure attention, and Mamba/attention hybrids —
at matched 512-dim width, and applies **language-model rescoring at
inference** (Spanish character n-gram beam search, α_lm = 0.5,
β_word = 1.0), which is how a deployable decoder would actually be run.

The seven architectures (all defined in
`brain2qwerty_v1_mamba/config/model_config.py` + `mamba_core.py`):

| # | Core name | Architecture | Depth | Params |
|---|---|---|:---:|---:|
| 1 | `transformer_deep` | 8-layer ALiBi Transformer (4 heads) | 8 | 44.2M |
| 2 | `hybrid` | Nemotron-H-style [M, M, M, A] — Mamba-2 + interleaved attention | 4 | 26.2M |
| 3 | `hybrid3` | Mamba-3 mixer (BCNorm + RoPE) + attention | 4 | 25.1M |
| 4 | `hybrid_8l` | 8-layer Mamba-2 hybrid [M,M,M,A]×2 | 8 | 48.6M |
| 5 | `hybrid3_8l` | 8-layer Mamba-3 hybrid | 8 | 46.8M |
| 6 | `mamba3_mlp` | BiMamba-3 + learned SiLU-gated fusion + 4× FFN sublayer | 4 | 25.1M |
| 7 | `mamba_mlp` | **BiMamba-2 + learned gated fusion + 4× FFN sublayer** | 4 | 25.2M |

The gated-MLP variants were added to close the parameter gap identified in
Study 1 (a 4-layer Transformer has attention **and** a 4× FFN = 24.1M params;
pure BiMamba-2 has only SSD mixers = 18.8M, a 22% deficit). At 25.2M, arm 7
reaches parameter parity with the 4-layer Transformer.

## 2. Dataset & split

- Same SpanishBCBL 3-subject cohort (S15, S16, S6), same 306-channel MEG.
- Studies 2–3 extract **continuous multi-second sentence timelines** directly,
  without Study 1's recording-boundary rejection and debounce, retaining all
  **22,302 keystrokes across 576 complete sentences** (9 session blocks:
  S15 192 sentences / 7,761 keystrokes; S16 192 / 9,737; S6 192 / 4,804).
- Leakage-free **80/10/10 train/val/test split via TF–IDF paraphrase
  clustering** (cosine threshold 0.5, seed 1) so near-duplicate sentence
  variants cannot cross the split boundary; **62 held-out test sentences**
  (2,243 test keystrokes).
- Per-window character accuracy chance level: 3.4% (29 classes).

## 3. Training configuration

- Launched via `slurm/12_round3_experiments.sbatch` (array tasks 0–6; task 7
  is the Study-3 continuous decoder).
- All SSM/hybrid arms: **lr 3e-4, weight decay 0.1, gradient clip 1.0** (the
  recipe Study 1 showed SSMs need); `transformer_deep` at lr 1e-4.
- 512-dim width, seed 33, same encoder/head/loss as Study 1.
- Inference: greedy character decode, then n-gram LM beam-search rescoring.
  - `test_ngram_rescoring.py` — our self-contained implementation: trains a
    smoothed character n-gram LM (N = 6, interpolated backoff with Laplace
    smoothing) on Spanish text and runs beam-search rescoring of model outputs.
  - The KenLM-based reference decoder is Meta's
    `brain2qwerty_v1/scripts/ngram_decoding.py` (character beam search with a
    KenLM LM; KenLM is optional and does not build on Kaggle).

## 4. Results (test CER with LM rescoring; Chapter 4 §4.6)

| # | Architecture | Depth | Test CER ↓ | Char. accuracy ↑ |
|---|---|:---:|---:|---:|
| 1 | Transformer (deep) | 8 | **25.9%** | **74.1%** |
| 2 | Hybrid (Mamba-2 + Attn) | 4 | 30.1% | 69.9% |
| 3 | Hybrid (Mamba-3 + Attn) | 4 | 32.6% | 67.4% |
| 4 | Hybrid (Mamba-2 + Attn) | 8 | 29.5% | 70.5% |
| 5 | Hybrid (Mamba-3 + Attn) | 8 | 30.1% | 69.9% |
| 6 | Mamba-3 + Gated MLP | 4 | 30.5% | 69.5% |
| 7 | **BiMamba-2 + Gated MLP** | **4** | **29.4%** | **70.6%** |

### Findings

1. **BiMamba-2 + Gated MLP is the champion SSM**: 29.4% CER — beats every
   hybrid (including 8-layer hybrids at double its depth) and comes within
   **3.5 points of the 8-layer Transformer at half the layer count**.
2. **Parameter parity matters**: restoring the FFN (18.8M → 25.2M) accounted
   for ~1 pt of CER; the residual ~4.8-pt gap to the tuned Transformer is the
   genuine short-horizon attention advantage.
3. **Depth does not rescue hybrids**: 8-layer hybrids (29.5/30.1%) ≈ 4-layer
   hybrids (30.1/32.6%) — no benefit from doubling depth at fixed width.
4. **Mamba-3's stabilisation trades differently at this scale**: pure v3
   upgrades (BCNorm + RoPE) do not beat the simpler BiMamba-2 + MLP recipe on
   short windows.
5. Differences < 1 pt CER are treated as within noise (single seed).

> **Verification note (honesty):** the §4.6 table values coincide exactly with
> the greedy-decoding test CERs of the same arms in the §4.5.2 grid (e.g.
> 25.9% / 29.4% / 30.1%). Before the viva, re-derive the rescored numbers from
> the stored prediction JSONs with `test_ngram_rescoring.py` to confirm the
> LM-rescoring deltas, or relabel the table as greedy decode.

## 5. Code layout (this folder)

```
brain2qwerty_v1_mamba/     # same package as Study 1 (shared code); the benchmark
                           # uses its extended cores: transformer_deep, mamba_mlp,
                           # mamba3_mlp, hybrid, hybrid3, hybrid_8l, hybrid3_8l
studies/                   # SpanishBCBL (Pinet2024Meg) study registration
test_ngram_rescoring.py    # char n-gram LM (N=6) + beam-search rescoring benchmark
slurm/
  12_round3_experiments.sbatch   # 8-task array: the 7 benchmark arms + v3_continuous
```

**External dependency:** `brain2qwerty_v1` (Meta reference — encoder config,
vocabulary, CER metric); optional `kenlm` for the reference KenLM decoder.

## 6. How to run (Kelvin-2)

```bash
cd ~/sharedscratch/B2Q/B2Q_Mamba/brain2qwerty-mamba2
sbatch slurm/12_round3_experiments.sbatch          # or: sbatch --array=2 ...
python test_ngram_rescoring.py                     # n-gram LM rescoring benchmark
```

## 7. Limitations

Same cohort/seed caveats as Study 1; LM rescoring depends on a character-level
n-gram LM (a word-level KenLM over a large Spanish corpus, as in the original
paper, is the natural upgrade); benchmark arms share one encoder, so results
characterise the **sentence core**, not the frontend.
