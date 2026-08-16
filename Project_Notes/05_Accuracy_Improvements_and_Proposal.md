# Brain2Qwerty V1 — Accuracy Improvement Levers & Proposal Directions

Baseline reproduced: test CER 0.389 (micro) / 0.409 (macro, per-sentence), WER 0.778.
Reference: paper reports ~0.38 for Conv+Transformer without LM rescoring.

---

## Part 1 — Improving accuracy (ranked by effort vs. expected gain)

### Tier 0 — Free wins (no retraining)

1. **N-gram LM rescoring** — `scripts/ngram_decoding.py` exists but was never run.
   The paper's optional KenLM + beam-search step typically recovers several CER points
   (character-level LM fixes exactly the "regicen/reducen"-type substitution errors seen
   in your best decodes). Run this first — it gives you a second, stronger headline number
   at zero training cost, and makes the comparison to the paper complete.
2. **Seed ensembling** — train 3–5 seeds (current: `seed: 33`), average log-probabilities
   before argmax/CER. Soft-voting usually buys 1–2 CER points on small neural datasets.
3. **Test-time augmentation** — average predictions over small temporal jitters of the
   window (±20–40 ms around `start=-0.2`). Motor signals are not perfectly time-locked.

### Tier 1 — Training/data tweaks (one retraining cycle each)

4. **Label smoothing** on CrossEntropyLoss (ε=0.05–0.1). With 29 classes and noisy motor
   labels this regularizes the softmax and typically helps CER slightly.
5. **EMA of weights** — evaluate an exponential moving average of parameters instead of the
   final weights; cheap, standard, usually +small consistent gain.
6. **Longer/wider windows** — current window is `start=-0.2, duration=0.5` (i.e., -0.2→0.3 s).
   Try -0.4→0.4 s or -0.3→0.5 s: keystroke-related motor preparation can precede the press
   by >200 ms. Watch the cache size (4.5 GB now; will scale up).
7. **Frequency content** — current pipeline: resample 50 Hz, bandpass 0.1–20 Hz. Motor-cortex
   typing signal likely extends into beta/low-gamma (20–40 Hz). Try filter (0.1, 40) +
   frequency 100–200. This is a hypothesis-driven ablation, good for the report either way.
8. **Augmentation** — Gaussian noise, channel dropout on the merged virtual channels,
   temporal jitter during training. Especially useful given only 81 recordings / 19 subjects.

### Tier 2 — Architectural changes (the "proposal-worthy" improvements)

9. **CTC loss instead of per-keystroke CE** — the current loss assumes the logged keystroke
   timestamp is the true neural-event time, which is only approximately true. CTC lets the
   model learn the alignment and is the standard fix in speech decoding; this alone is a
   publishable ablation vs. the V1 baseline.
10. **Bigger sentence transformer** — 4 layers / 2 heads is small. Try 6–8 layers, 4–8 heads,
    RoPE instead of ALiBi. The transformer is where sentence context enters; your qualitative
    errors (single-character substitutions in otherwise correct sentences) suggest under-used
    context.
11. **Target the subject variance directly** — S15 (0.287) to S20 (0.552) is nearly 2× spread,
    and all 5 worst sentences are S20. Options:
    - per-subject fine-tuning on top of the pooled model (few-shot calibration angle),
    - subject-adversarial invariance + subject-specific heads,
    - stronger per-subject normalization than the global RobustScaler.
    Framed as "inter-subject variability mitigation", this is a clean research contribution.
12. **Longer training / LR search** — OneCycleLR max_lr 5e-5 is conservative (repo comment
    says it's more stable than the paper's). With augmentation + label smoothing you can
    often push LR up and train longer before early stopping.

### Reporting hygiene
- Pick one CER convention (micro from eval vs. macro from predictions.csv) and state it.
- Report with/without LM rescoring as two columns, like the paper does.

---

## Part 2 — Proposal directions (another approach on SpanishBCBL)

### Recommended flagship: Cross-modal MEG→EEG distillation
SpanishBCBL is unusual in containing **both** MEG (~5.1K sentences) and EEG (~4K sentences)
on the same task, with **5 subjects recorded in both modalities**.

- **Research question:** Can a cheap, scalable modality (EEG) approach MEG-level typing-decoding
  accuracy by transferring knowledge from an MEG teacher?
- **Method:** train your validated V1 model on MEG (teacher) → train EEG student with a
  distillation loss (KL between keystroke logit distributions on shared/paired sentences +
  CE on labels), possibly with a shared latent space via contrastive alignment on the 5
  dual-modality subjects.
- **Why it's a strong proposal:** clinically motivated (EEG is deployable, MEG is not),
  uses a property of the dataset no published baseline exploits, and your reproduced V1
  pipeline is the exact teacher you need. Falsifiable, incremental, well-scoped for COM865.
- **Metrics:** EEG-only CER baseline vs. distilled EEG; ablation on paired vs. unpaired data.

### Alternative B: Self-supervised pretraining on continuous MEG
The current pipeline throws away everything outside 0.5 s keystroke windows. Masked-autoencoder
or contrastive (à la wav2vec) pretraining on the full continuous recordings (~262 GB raw),
then fine-tune for keystroke decoding. Question: does SSL beat supervised-only under limited
labeled data? More compute-hungry, less novel framing, but very "foundation model" friendly.

### Alternative C: CTC + LM end-to-end sentence decoder
Reframe keystroke decoding as speech recognition: CTC alignment-free training + character
LM beam search, benchmarked against the per-keystroke CE baseline you already have. Narrower
scope, cleanest single-variable comparison — good as a second experiment inside proposal A.

### Suggested proposal structure
1. Reproduced baseline (done — CER 0.389, this is your credibility anchor).
2. Improvement ablations (Tier 0–1 items, each one training run).
3. Main contribution: MEG→EEG distillation (or CTC reformulation if you prefer methods novelty).
4. Analysis chapter: per-subject variability, error taxonomy, what the LM fixes vs. what it can't.
