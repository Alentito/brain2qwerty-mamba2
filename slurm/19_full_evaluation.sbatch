#!/bin/bash
#SBATCH --job-name=b2q_full_eval
#SBATCH --partition=k2-gpu-a100        # fallback: k2-gpu-h100 or k2-gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --array=0-4
#SBATCH --output=slurm/logs/fulleval_%A_%a.out

# ============================================================================ #
# FULL EVALUATION BATTERY — Studies 1-3 (SpanishBCBL 3-subject)
#
#   0: Study 1  -> eval every trained round-1/2 arm with a checkpoint
#                  (idempotent: arms with test_all_sentences.json are skipped)
#                  + XAI explainability pass on the best Mamba + Transformer
#   1: Study 2  -> character n-gram LM rescoring benchmark (greedy vs +LM CER)
#   2: Study 3  -> eval the 3 continuous-decoding cores (Conformer /
#                  BiMamba-2+MLP / Mamba-3 Hybrid) + per-subject metrics
#                  + bootstrap statistical testing
#   3: Scaling  -> O(N) vs O(N^2) latency / throughput / VRAM profiler
#                  vs sequence length (the "Mamba decode-time advantage" figure)
#   4: Encoder  -> linear-probe evaluation of the learned encoder
#                  (raw window vs 512-d embedding, char accuracy + PCA plots)
#
# Submit everything:   sbatch slurm/19_full_evaluation.sbatch
# Submit one section:  sbatch --array=2 slurm/19_full_evaluation.sbatch
# ============================================================================ #

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm/logs

if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
elif [ -f "$HOME/sharedscratch/conda/envs/b2q/bin/activate" ]; then
    source "$HOME/sharedscratch/conda/envs/b2q/bin/activate"
else
    source "$HOME/miniconda3/etc/profile.d/conda.sh" && conda activate "$HOME/sharedscratch/conda/envs/b2q"
fi

export BRAIN2QWERTY_STUDIES="${BRAIN2QWERTY_STUDIES:-$HOME/sharedscratch/B2Q/code/SpanishBCBL_3subj}"
[ -d "$BRAIN2QWERTY_STUDIES" ] || export BRAIN2QWERTY_STUDIES="$PWD/SpanishBCBL_3subj"
export BRAIN2QWERTY_CACHE="${BRAIN2QWERTY_CACHE:-$HOME/sharedscratch/B2Q/cache_v1mamba}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1

RESULTS="$BRAIN2QWERTY_CACHE/results"
TASK=${SLURM_ARRAY_TASK_ID:-0}
echo "=== full-eval task $TASK on $(hostname) at $(date) ==="

# --------------------------------------------------------------------------- #
run_study1_arm () {  # $1=core  $2=tag ("" for round-1)
    local core="$1" tag="$2"
    local dir="$RESULTS/small-$core-S15-S16-S6${tag:+-$tag}"
    local ckpt="$dir/best.ckpt"
    [ -f "$ckpt" ] || ckpt="$dir/last.ckpt"
    if [ ! -f "$ckpt" ]; then echo "[s1] SKIP $core/$tag (no checkpoint)"; return 0; fi
    if [ -f "$dir/callbacks/test_all_sentences.json" ]; then
        echo "[s1] SKIP $core/$tag (already evaluated)"; return 0
    fi
    echo "[s1] EVAL $core/$tag <- $ckpt"
    python -m brain2qwerty_v1_mamba.main eval --core "$core" --small \
        --subjects S15 S16 S6 ${tag:+--tag "$tag"} --ckpt "$ckpt"
}

case $TASK in

0)  # ---- Study 1: all windowed arms + XAI ------------------------------- #
    run_study1_arm mamba ""
    run_study1_arm transformer ""
    run_study1_arm mamba lr1e4
    run_study1_arm mamba lr3e4
    run_study1_arm mamba lr1e4-wd01-gc1
    run_study1_arm mamba3 v3lr1e4-wd01-gc1
    run_study1_arm mamba3 v3lr3e4-wd01-gc1
    run_study1_arm transformer lr1e4

    # XAI on the best Mamba-2 and the tuned Transformer control
    for pair in "mamba lr3e4" "transformer lr1e4"; do
        set -- $pair; core=$1; tag=$2
        ckpt="$RESULTS/small-$core-S15-S16-S6-$tag/best.ckpt"
        if [ -f "$ckpt" ]; then
            echo "[s1] XAI $core/$tag"
            python explain_mamba.py --ckpt "$ckpt" --core "$core" --small \
                --subjects S15 S16 S6 --n-sentences 6 \
                --out "explain_out/$core-$tag"
        else
            echo "[s1] XAI SKIP $core/$tag (no best.ckpt)"
        fi
    done
    ;;

1)  # ---- Study 2: n-gram LM rescoring benchmark -------------------------- #
    python test_ngram_rescoring.py
    ;;

2)  # ---- Study 3: continuous decoding eval + stats ----------------------- #
    for spec in "conformer v3-conformer-baseline" \
                "mamba_mlp v3-mamba-gated-mlp" \
                "mamba3_hybrid_stabilized v3-mamba3-stabilized-hybrid"; do
        set -- $spec; core=$1; tag=$2
        dir="$RESULTS/v3-$core-$tag"
        ckpt="$dir/best_llm.ckpt"
        [ -f "$ckpt" ] || ckpt="$dir/best_ctc.ckpt"
        if [ ! -f "$ckpt" ]; then
            ckpt=$(find "$dir" -name "*.ckpt" 2>/dev/null | head -n 1 || true)
        fi
        if [ -z "$ckpt" ] || [ ! -f "$ckpt" ]; then
            echo "[s3] SKIP $core (no checkpoint in $dir)"; continue
        fi
        echo "[s3] EVAL $core <- $ckpt"
        python -m brain2qwerty_v3.main eval --core "$core" --frontend conv \
            --tag "$tag" --subjects S15 S16 S6 --devices 1 --ckpt "$ckpt"
        python -m brain2qwerty_v3.scripts.extract_predictions --input "$dir" --split test
    done
    echo "[s3] per-subject metrics + bootstrap statistics"
    python evaluate_accuracy.py "$RESULTS" || python evaluate_accuracy.py
    python statistical_testing.py --dir "$RESULTS"
    ;;

3)  # ---- Sequence-length scaling / decode-time benchmark ----------------- #
    python benchmark_complexity.py --device cuda
    ;;

4)  # ---- Encoder-extraction probe ---------------------------------------- #
    for pair in "mamba lr3e4" "transformer lr1e4"; do
        set -- $pair; core=$1; tag=$2
        ckpt="$RESULTS/small-$core-S15-S16-S6-$tag/best.ckpt"
        if [ -f "$ckpt" ]; then
            echo "[enc] probe $core/$tag"
            python probe_encoder.py --ckpt "$ckpt" --core "$core" --small \
                --subjects S15 S16 S6 --n-train 10000 \
                --out "probe_out/$core-$tag"
        else
            echo "[enc] SKIP $core/$tag (no best.ckpt)"
        fi
    done
    ;;

*)  echo "unknown task $TASK"; exit 1 ;;
esac

echo "=== full-eval task $TASK done at $(date) ==="
