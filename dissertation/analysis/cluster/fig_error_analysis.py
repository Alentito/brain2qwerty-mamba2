#!/usr/bin/env python
"""Error-analysis figures from Phase-2 predictions_test.csv files (RUN ON CLUSTER).

Works on any predictions CSV with columns:
    sentence_UID, subject, true_text, ctc_text, CTC_CER, pred_text, CER, WER, SemER

Produces (into --out):
  fig_confusion_matrix.pdf   character substitution matrix (top confusions)
  fig_error_types.pdf        substitution / insertion / deletion profile per model
  fig_cer_vs_length.pdf      CER as a function of sentence length
  fig_ctc_vs_final.pdf       CTC CER vs final CER scatter (decoder gain per sentence)
  error_analysis_table.csv   numeric summary used in the dissertation tables

Usage:
  python fig_error_analysis.py \
      --csvs Conformer=/path/to/conformer/predictions_test.csv \
             Mamba2=/path/to/mamba2/predictions_test.csv \
             Mamba3=/path/to/mamba3/predictions_test.csv \
      --out dissertation/figures/
"""
import argparse
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def edit_ops(ref: str, hyp: str):
    """Levenshtein backtrace -> list of (op, ref_char, hyp_char)."""
    n, m = len(ref), len(hyp)
    dp = np.zeros((n + 1, m + 1), dtype=int)
    for i in range(n + 1):
        dp[i, 0] = i
    for j in range(m + 1):
        dp[0, j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i, j] = min(
                dp[i - 1, j] + 1,
                dp[i, j - 1] + 1,
                dp[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]),
            )
    ops = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i, j] == dp[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]):
            if ref[i - 1] != hyp[j - 1]:
                ops.append(("sub", ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and dp[i, j] == dp[i - 1, j] + 1:
            ops.append(("del", ref[i - 1], ""))
            i -= 1
        else:
            ops.append(("ins", "", hyp[j - 1]))
            j -= 1
    return ops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csvs", nargs="+", required=True,
                    help="Name=/path/predictions_test.csv pairs")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    dfs = {}
    for pair in args.csvs:
        name, path = pair.split("=", 1)
        dfs[name] = pd.read_csv(path)

    # ---------------- confusion matrix (first model) ---------------- #
    name0 = list(dfs)[0]
    df0 = dfs[name0]
    subs = Counter()
    for ref, hyp in zip(df0.true_text, df0.pred_text):
        for op, r, h in edit_ops(str(ref), str(hyp)):
            if op == "sub":
                subs[(r, h)] += 1
    top = subs.most_common(20)
    if top:
        fig, ax = plt.subplots(figsize=(6.4, 3.2))
        labels = [f"'{r}'→'{h}'" for (r, h), _ in top][::-1]
        vals = [c for _, c in top][::-1]
        ax.barh(range(len(vals)), vals, color="#1f77b4", alpha=0.85)
        ax.set_yticks(range(len(vals)))
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel("substitution count (test set)")
        ax.set_title(f"Top-20 character substitutions — {name0}", fontsize=9)
        fig.tight_layout()
        fig.savefig(out / "fig_confusion_matrix.pdf", bbox_inches="tight")
        plt.close(fig)

    # ---------------- error-type profile per model ------------------ #
    summary = {}
    for name, df in dfs.items():
        c = Counter()
        for ref, hyp in zip(df.true_text, df.pred_text):
            for op, _, _ in edit_ops(str(ref), str(hyp)):
                c[op] += 1
        summary[name] = c
    fig, ax = plt.subplots(figsize=(5.6, 2.8))
    x = np.arange(len(dfs))
    bottoms = np.zeros(len(dfs))
    for op, color in [("sub", "#1f77b4"), ("del", "#d62728"), ("ins", "#2ca02c")]:
        vals = np.array([summary[n][op] for n in dfs], dtype=float)
        ax.bar(x, vals, 0.5, bottom=bottoms, label=op, color=color, alpha=0.85)
        bottoms += vals
    ax.set_xticks(x)
    ax.set_xticklabels(list(dfs), fontsize=8)
    ax.set_ylabel("edit operations (test set)")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Error-type profile per model", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig_error_types.pdf", bbox_inches="tight")
    plt.close(fig)

    # ---------------- CER vs sentence length ------------------------ #
    fig, ax = plt.subplots(figsize=(5.6, 2.8))
    for name, df in dfs.items():
        L = df.true_text.astype(str).str.len()
        bins = pd.cut(L, bins=5)
        g = df.groupby(bins, observed=True)["CER"].agg(["mean", "sem"])
        centers = [iv.mid for iv in g.index]
        ax.errorbar(centers, g["mean"], yerr=g["sem"], marker="o", ms=4,
                    lw=1.4, capsize=3, label=name)
    ax.set_xlabel("sentence length (characters)")
    ax.set_ylabel("mean CER")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("CER vs sentence length", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig_cer_vs_length.pdf", bbox_inches="tight")
    plt.close(fig)

    # ---------------- CTC vs final CER scatter ---------------------- #
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    for name, df in dfs.items():
        ax.scatter(df.CTC_CER, df.CER, s=14, alpha=0.6, label=name)
    lim = [0, 1.05]
    ax.plot(lim, lim, "k--", lw=1, label="no decoder gain")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("CTC CER (pre-decoder)")
    ax.set_ylabel("final CER (post-decoder)")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Decoder gain per sentence", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig_ctc_vs_final.pdf", bbox_inches="tight")
    plt.close(fig)

    # ---------------- numeric summary ------------------------------- #
    rows = []
    for name, df in dfs.items():
        c = summary[name]
        rows.append({
            "model": name,
            "n_sentences": len(df),
            "mean_CER": df.CER.mean(),
            "mean_WER": df.WER.mean(),
            "mean_CTC_CER": df.CTC_CER.mean(),
            "mean_SemER": df.SemER.mean(),
            "n_sub": c["sub"], "n_del": c["del"], "n_ins": c["ins"],
            "sub_rate": c["sub"] / max(1, sum(c.values())),
        })
    pd.DataFrame(rows).to_csv(out / "error_analysis_table.csv", index=False)
    print("wrote figures +", out / "error_analysis_table.csv")


if __name__ == "__main__":
    main()
