#!/usr/bin/env python
"""Paired bootstrap + Wilcoxon comparison of two predictions_test.csv files.

Fully self-contained (pandas/numpy/scipy only). Run on cluster or locally.

Usage:
  python stats_paired_bootstrap.py \
      --a Conformer=/path/conformer/predictions_test.csv \
      --b Mamba3=/path/mamba3/predictions_test.csv \
      --metrics CER WER CTC_CER SemER --n-boot 10000 \
      --out stats_mamba3_vs_conformer.csv

Pairs are matched on sentence_UID (inner join). Reports, per metric:
  mean_A, mean_B, delta (A - B), bootstrap 95% CI on delta,
  paired-bootstrap p-value, Wilcoxon signed-rank p, Cohen's d_z, Cliff's delta.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def cohens_dz(d):
    return d.mean() / (d.std(ddof=1) + 1e-12)


def cliffs_delta(a, b):
    gt = sum(np.sum(a[:, None] > b[None, :]) for _ in [0])
    lt = sum(np.sum(a[:, None] < b[None, :]) for _ in [0])
    n = len(a) * len(b)
    return (gt - lt) / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="Name=path (baseline)")
    ap.add_argument("--b", required=True, help="Name=path (model)")
    ap.add_argument("--metrics", nargs="+", default=["CER", "WER"])
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    name_a, path_a = args.a.split("=", 1)
    name_b, path_b = args.b.split("=", 1)
    A = pd.read_csv(path_a).set_index("sentence_UID")
    B = pd.read_csv(path_b).set_index("sentence_UID")
    common = A.index.intersection(B.index)
    A, B = A.loc[common], B.loc[common]
    print(f"paired on {len(common)} sentences")

    rng = np.random.default_rng(0)
    rows = []
    for met in args.metrics:
        a, b = A[met].to_numpy(float), B[met].to_numpy(float)
        d = a - b  # positive => B (model) is better for error metrics
        idx = rng.integers(0, len(d), size=(args.n_boot, len(d)))
        boots = d[idx].mean(axis=1)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        p_boot = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
        w = stats.wilcoxon(a, b)
        rows.append({
            "comparison": f"{name_a} - {name_b}",
            "metric": met,
            f"mean_{name_a}": a.mean(),
            f"mean_{name_b}": b.mean(),
            "delta": d.mean(),
            "ci_lo": lo, "ci_hi": hi,
            "p_bootstrap": p_boot,
            "p_wilcoxon": w.pvalue,
            "cohens_dz": cohens_dz(d),
            "cliffs_delta": cliffs_delta(a, b),
        })
    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    print(out.to_string(index=False))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
