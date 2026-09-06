# Shared / Cross-Study Tools

Auxiliary scripts that serve the whole experimental programme rather than a
single study.

| Script | Purpose |
|---|---|
| `dataset_analysis.py` | Exploratory data analysis on SpanishBCBL (S15/S16/S6): character distribution, per-subject split balance, sentence-length distribution, 306-channel evoked-response / GFP plots. Output: `dataset_eda_out/` |
| `animate_brain_typing.py` | Animated brain-activation video/GIF + interactive HTML viewer for MEG sensor activity during typing |
| `run_two_studies.py` | **Auxiliary experiments — naming collision warning.** This script's internal "STUDY 1 / STUDY 2" are *not* the dissertation's Study 1/2: they are (1) a single-subject (S15-only) benchmark and (2) a cross-subject transfer test (train S15+S16, zero-shot eval on S6), both run on the Study-1 windowed pipeline |
| `cluster_score.py` | Score all completed model runs directly on Kelvin-2 (Levenshtein-based CER from stored prediction files) |

## Dependencies

These scripts import from `brain2qwerty_v1` (Meta reference — char vocabulary,
extractors) and/or the study packages in the sibling folders; run them from the
main repo root, not from this folder.
