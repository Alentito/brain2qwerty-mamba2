#!/bin/bash
# Regenerate all cluster-side dissertation figures + stats.
# Run from the repo root on kelvin2 with the b2q env:
#   source $HOME/sharedscratch/conda/envs/b2q/bin/activate
#   bash dissertation/analysis/cluster/run_all.sh
set -euo pipefail

export BRAIN2QWERTY_STUDIES="${BRAIN2QWERTY_STUDIES:-/mnt/scratch2/users/atito/B2Q/dataset/SpanishBCBL}"
export BRAIN2QWERTY_CACHE="${BRAIN2QWERTY_CACHE:-$HOME/sharedscratch/B2Q/cache_v1mamba}"
export HF_HOME="${HF_HOME:-$HOME/sharedscratch/B2Q/hf_cache}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

R=$HOME/sharedscratch/B2Q/cache_v1mamba/results
CONF=$R/v3-conformer-v3-conformer-baseline/predictions_test.csv
M2=$R/v3-mamba_mlp-v3-mamba-gated-mlp/predictions_test.csv
M3=$R/v3-mamba3_hybrid_stabilized-v3-mamba3-stabilized-hybrid/predictions_test.csv

OUT=dissertation/figures
mkdir -p "$OUT" dissertation/stats

# 1. error-analysis figures (confusion, error types, CER-vs-length, decoder gain)
python dissertation/analysis/cluster/fig_error_analysis.py \
    --csvs Conformer="$CONF" BiMamba2="$M2" Mamba3Hybrid="$M3" \
    --out "$OUT"

# 2. significance tests missing from the current report:
#    Mamba-3 vs Mamba-2 (is the WER gap real?) and both vs Conformer
python dissertation/analysis/cluster/stats_paired_bootstrap.py \
    --a Conformer="$CONF" --b Mamba3Hybrid="$M3" \
    --metrics CER WER CTC_CER SemER \
    --out dissertation/stats/stats_mamba3_vs_conformer.csv

python dissertation/analysis/cluster/stats_paired_bootstrap.py \
    --a Conformer="$CONF" --b BiMamba2="$M2" \
    --metrics CER WER CTC_CER SemER \
    --out dissertation/stats/stats_mamba2_vs_conformer.csv

python dissertation/analysis/cluster/stats_paired_bootstrap.py \
    --a BiMamba2="$M2" --b Mamba3Hybrid="$M3" \
    --metrics CER WER CTC_CER SemER \
    --out dissertation/stats/stats_mamba3_vs_mamba2.csv

echo "=== cluster figure/stat bundle complete ==="
echo "If any predictions_test.csv path is missing, list available runs with:"
echo "  ls $R"
