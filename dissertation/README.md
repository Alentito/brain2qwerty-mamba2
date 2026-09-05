# Dissertation Package

`main.tex` + `references.bib` + `figures/` = the full MSc Data Science
dissertation. Every figure and statistic is regenerable from the scripts in
`analysis/`.

## Compile (Overleaf or local LaTeX)

Upload `main.tex`, `references.bib`, and the whole `figures/` directory to
Overleaf, or locally:

```bash
cd dissertation
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

The document compiles **today** — the six cluster-pending figures render as
labelled placeholders and upgrade automatically once the PDFs exist in
`figures/` (see `\figorplaceholder` in `main.tex`).

## Figure inventory

| Figure | Source | Status |
|---|---|---|
| `fig_training_curves_cer.pdf` | Lightning `metrics.csv` logs | done |
| `fig_round2_grid.pdf`, `fig_leaderboard.pdf`, `fig_phase1_benchmark.pdf` | audited result tables | done |
| `fig_phase2_metrics.pdf`, `fig_bootstrap_delta.pdf` | `statistical_reports/*.csv` | done |
| `fig_per_subject.pdf` | per-subject table | done |
| `fig_latency_scaling.pdf` | `benchmark_out/*.json` | done |
| `01..04_*.png` (EDA) | `dataset_eda_out/` | done |
| `distribution_wer_cer_violin.png` | `statistical_reports/` | done |
| `fig_confusion_matrix / error_types / cer_vs_length / ctc_vs_final.pdf` | cluster predictions CSVs | **pending** |
| `fig_delta_t_selectivity.pdf`, `fig_dtw_alignment.pdf` | cluster checkpoint hooks | **pending (template)** |

## Regenerate local figures

```bash
python dissertation/analysis/make_figures.py
```

## Regenerate cluster figures + missing stats (kelvin2)

```bash
cd ~/sharedscratch/B2Q/B2Q_Mamba/brain2qwerty-mamba2
source $HOME/sharedscratch/conda/envs/b2q/bin/activate
bash dissertation/analysis/cluster/run_all.sh
```

This produces the four error-analysis figures and three new significance
tables in `dissertation/stats/` — including **Mamba-3 vs Mamba-2**, the
comparison the current draft is missing. After running, copy the repo back
(or commit/pull) and recompile.

`fig_delta_t_selectivity.py` is a documented template: it explains exactly
where to hook `dt = F.softplus(dt_raw + self.dt_bias)`
(`brain2qwerty_v3/mamba.py`, Mamba-2 ~line 166, Mamba-3 ~line 225). Complete
it only if you want the Δt figure; the text reads fine with the placeholder.

## Known to-dos before submission

- Verify the `levy2026brain2qwertyv2` bib entry (marked in `references.bib`).
- Replace placeholder affiliation/date on the title page.
- After `run_all.sh`, update Table `tab:bootstrap` with the Mamba-3 vs
  Mamba-2 row from `dissertation/stats/stats_mamba3_vs_mamba2.csv`.
