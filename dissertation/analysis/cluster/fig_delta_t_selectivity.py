#!/usr/bin/env python
"""Delta-t selectivity figure for Mamba cores (RUN ON CLUSTER, needs a checkpoint).

Captures the input-dependent step size dt = softplus(dt_raw + dt_bias)
inside every Mamba mixer via forward hooks, aligned with the CTC frame
posteriors, and tests the claim from the dissertation:

    dt at SPACE characters is ~3.8x higher than at letter characters
    (the SSM learns to flush its state at word boundaries).

Output:
  fig_delta_t_selectivity.pdf   dt trace for one sentence + space/letter bar
  delta_t_stats.csv             per-layer mean dt at spaces vs letters, ratio

USAGE (adjust paths):
  python fig_delta_t_selectivity.py \
      --ckpt /path/to/v3-mamba3_hybrid_stabilized/best_ctc.ckpt \
      --out dissertation/figures/

NOTE: this script must run inside the repo (brain2qwerty_v3 importable) with
the b2q env. It reuses the repo's own data pipeline, so BRAIN2QWERTY_STUDIES
and BRAIN2QWERTY_CACHE must be set as in the training sbatch.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--subject", default="S16")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # -- load the trained module through the repo's own plumbing -------------- #
    from brain2qwerty_v3.pl_module import NeuroLLMModule  # noqa

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]

    # Build a minimal module skeleton matching the checkpoint.
    # (The exact constructor call depends on the repo version; the safest
    #  route is to instantiate via the training config saved in
    #  ckpt["hyper_parameters"] if present, else edit the two lines below.)
    raise SystemExit(
        "TEMPLATE: instantiate NeuroLLMModule from ckpt['hyper_parameters'] "
        "here, load_state_dict(sd, strict=False), then register forward "
        "hooks on every submodule whose forward computes "
        "'dt = F.softplus(dt_raw + self.dt_bias)' (see brain2qwerty_v3/"
        "mamba.py lines ~166 and ~225). Capture dt (B,T,H), average over "
        "heads, align with CTC argmax frames for one test sentence of "
        f"subject {args.subject}, and compare mean dt on frames decoded "
        "as space vs letters. Plot: (top) dt trace with space frames "
        "shaded; (bottom) bar of mean dt space vs letter per layer; "
        "write delta_t_stats.csv with the ratio (expected ~3.8x)."
    )


if __name__ == "__main__":
    main()
