#!/usr/bin/env python
"""Word-segmentation figure for Study 3 (RUN ON CLUSTER, needs a checkpoint).

Visualises how CTCSpaceSegmenter turns the continuous stream into words:
  top:    CTC space-symbol probability per frame, with predicted boundaries
  bottom: predicted word spans vs the true word sequence of the sentence

Hook point (brain2qwerty_v3/ctc_segmenter.py, class CTCSpaceSegmenter):
    forward(z_final, ctc_logits)  ->  preds = ctc_logits.argmax(dim=-1)
    frames between predicted space symbols form a word.

So the figure needs, for ONE test sentence:
  1. ctc_logits (T, 29)  -> space index probability trace  p_space(t)
  2. preds (T,)          -> boundary frames where preds == space_idx
  3. the sentence's true word list (from the dataset / batch metadata)

Output:
  fig_word_segmentation.pdf
  word_segmentation_stats.csv  (n true words, n predicted segments, boundary
                                precision/recall at +/-3-frame tolerance)

USAGE:
  python fig_word_segmentation.py \
      --ckpt /path/to/v3-mamba3_hybrid_stabilized/best_ctc.ckpt \
      --sentence-uid 13.0_S16_1_block1 \
      --out dissertation/figures/

Must run inside the repo with the b2q env and the usual
BRAIN2QWERTY_STUDIES / BRAIN2QWERTY_CACHE env vars (see run_all.sh).
"""
import argparse
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--sentence-uid", default=None,
                    help="sentence_UID from predictions_test.csv; default: "
                         "first test sentence of S16")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)

    raise SystemExit(
        "TEMPLATE: build the NeuroLLMModule from ckpt['hyper_parameters'], "
        "load_state_dict(ckpt['state_dict'], strict=False), then pull ONE "
        "test batch through the repo's data pipeline. Run module.network + "
        "CTC head to get ctc_logits (T,29); space_idx is the index of the "
        "space token in the 29-class vocab (check model_config / the CTC "
        "head's class list). Plot: (top) softmax(ctc_logits)[..., space_idx] "
        "vs frame index, vertical lines at frames where argmax == space_idx; "
        "(bottom) predicted word spans as blocks, true words printed under "
        "their approximate span (uniform interpolation is fine, or use the "
        "hard-DTW path from module.word_contrastive_loss for exact spans). "
        "Write word_segmentation_stats.csv with boundary precision/recall "
        "at +/-3 frames. See ctc_segmenter.py CTCSpaceSegmenter.forward "
        "for the exact grouping logic to mirror."
    )


if __name__ == "__main__":
    main()
