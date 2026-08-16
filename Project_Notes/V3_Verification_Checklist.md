# Brain2Qwerty V3 (Mamba Hybrid) — Run Order & Verification Checklist

Environment: Kelvin2 HPC, `b2q` conda env (torch 2.6.0, transformers 4.52.4 — already pinned).

## Stage 1 — Unit tests (CPU is fine, minutes)

```bash
cd /path/to/brain2qwerty
pytest brain2qwerty_v3/tests/test_mamba.py -v
```

| # | Test | What it proves | Pass criterion |
|---|------|----------------|----------------|
| 1.1 | `test_mixer_shapes_and_backward` | Mixer builds, output shape correct, grads finite everywhere | no NaN/Inf in any parameter grad |
| 1.2 | `test_stack_shapes_and_backward` | Hybrid stack builds; block pattern is exactly M M M A M M M A | pattern assertion + finite grads |
| 1.3 | `test_config_build` | `MambaHybrid` pydantic config builds via `build(dim)` like Conformer did | shape check |
| 1.4 | `test_causality` | Scrambling future frames does not change past outputs | exact match (atol 1e-5) |
| 1.5 | `test_determinism` | Same input → same output | bitwise equal |
| 1.6 | `test_hf_parity` | Our from-scratch Mamba-2 matches HuggingFace reference with identical weights | allclose atol=1e-3 |

If 1.6 fails: send me the max-abs-error value — small drift (~1e-4) is float reassociation, large error means a real bug to fix before any GPU run.

## Stage 2 — Config registration & model build (CPU, ~1 min)

```bash
python -m brain2qwerty_v3.main debug
```

Verify:
- [ ] No pydantic/discriminated-model error on `"name": "ConvMambaHybrid"` / `"MambaHybrid"` (proves config registration)
- [ ] Model builds: merger, conv encoder, downsampling, **HybridMambaEncoder**, aux CTC head
- [ ] Parameter count printed/logged — compare with V2's Conformer run (should be same order of magnitude, ~±20%)
- [ ] One training step completes, loss is finite
- [ ] Loss decreases over the 2 debug epochs (sanity, not convergence)

## Stage 3 — Feature cache (CPU node, high RAM)

```bash
python -m brain2qwerty_v3.main cache        # add --debug for the small subset first
```

Verify:
- [ ] Run with 64G RAM allocation (the 16G OOM/corruption from V1 applies here too)
- [ ] Cache entry count matches the recording count; re-run after interruption picks up cleanly

## Stage 4 — Short real training (GPU, ~30–60 min)

Debug config but on GPU with ~10–20 epochs on a few timelines:

- [ ] CTC loss starts high (~log 28 ≈ 3.3 for random 28-class output) and decreases steadily
- [ ] `val/cer_epo` below 1.0 and trending down
- [ ] bf16-mixed stability: no loss spikes/NaN in first 500 steps (the SSD core is float32 internally, so spikes here would indicate a pipeline issue, not the SSM)
- [ ] Throughput (it/s) vs V2 run — expect somewhat slower than Conformer; if >3× slower, consider head_chunk tuning or swapping to mamba-ssm kernels

## Stage 5 — Full training + evaluation (4× V100 DDP)

```bash
python -m brain2qwerty_v3.main train
python -m brain2qwerty_v3.main eval --ckpt <output_dir>/best_ctc*.ckpt
```

Verify:
- [ ] Staged losses activate at the right epochs (CTC @0, contrastive @150, LLM @225) — check `train/loss_*` logs
- [ ] Best val CTC-CER comparable to or better than the V2 Conformer reference run
- [ ] Test metrics: CTC-CER, LLM CER/WER, SemER, predictions CSV written
- [ ] Per-subject CER spread — compare against V1 reproduction's 0.287–0.552 range

## Cross-cutting checks

- [ ] Reproducibility: same seed twice → same val loss curve (up to DDP nondeterminism)
- [ ] Checkpoint reload: `eval --ckpt best_ctc` reproduces the logged test numbers
- [ ] Ablation knobs work from config only: `attention_every` (e.g. 2 vs 4 vs 999=pure Mamba), `n_layer`, `d_state` — each change should alter the block pattern/parameter count as expected without code edits

## Known open item (not blocking)

V3 inherits V2's data config, which targets the **English** study (`PinetAudio2025`) with `EnglishBCBLPreprocessing`. For SpanishBCBL, the study name becomes `Pinet2024Meg` and the preprocessing transform needs a Spanish variant (based on V1's `SpanishBCBLPreprocessing`). This is a data-layer change, independent of the architecture.
