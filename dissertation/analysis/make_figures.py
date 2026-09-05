#!/usr/bin/env python
"""Regenerate every dissertation figure that can be built from LOCAL data.

Sources (all inside the repo):
  - checkpoints/*/logs/version_0/metrics.csv   (char-level training curves)
  - statistical_reports/*.csv                  (Phase-2 bootstrap stats)
  - benchmark_out/benchmark_complexity_report.json (latency scaling)
  - results tables hard-coded from the verified experiment logs
    (Project_Notes/10 and the v1 report -- numbers already audited)

Outputs PDF figures into dissertation/figures/.
Run:  python dissertation/analysis/make_figures.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "dissertation" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(Path(sys.executable).parent.parent.parent))
try:
    from daimon_runtime import setup_plot
    setup_plot()
except Exception:
    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3})

C_MAMBA = "#1f77b4"
C_MAMBA3 = "#2ca02c"
C_TRF = "#d62728"
C_CONF = "#7f7f7f"

# ---------------------------------------------------------------- 1. curves
def load_val_cer(path):
    df = pd.read_csv(path)
    df = df.dropna(subset=["val_CER"])
    return df["epoch"].values, df["val_CER"].values

runs = {
    "Mamba-2 lr 5e-5":        "checkpoints/small-mamba-S15-S16-S6",
    "Mamba-2 lr 1e-4":        "checkpoints/small-mamba-S15-S16-S6-lr1e4",
    "Mamba-2 lr 3e-4":        "checkpoints/small-mamba-S15-S16-S6-lr3e4",
    "Mamba-2 lr 1e-4 +reg":   "checkpoints/small-mamba-S15-S16-S6-lr1e4-wd01-gc1",
    "Mamba-3 lr 1e-4 +reg":   "checkpoints/small-mamba3-S15-S16-S6-v3lr1e4-wd01-gc1",
    "Mamba-3 lr 3e-4 +reg":   "checkpoints/small-mamba3-S15-S16-S6-v3lr3e4-wd01-gc1",
    "Transformer lr 5e-5":    "checkpoints/small-transformer-S15-S16-S6",
    "Transformer lr 1e-4":    "checkpoints/small-transformer-S15-S16-S6-lr1e4",
}

fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
curve_colors = {
    "Mamba-2 lr 5e-5": "#9ecae1",
    "Mamba-2 lr 1e-4": "#3182bd",
    "Mamba-2 lr 3e-4": "#08519c",
    "Mamba-2 lr 1e-4 +reg": "#6a51a3",
    "Mamba-3 lr 1e-4 +reg": "#74c476",
    "Mamba-3 lr 3e-4 +reg": "#006d2c",
    "Transformer lr 5e-5": "#f4a3a3",
    "Transformer lr 1e-4": "#d62728",
}
for label, d in runs.items():
    p = ROOT / d / "logs/version_0/metrics.csv"
    if not p.exists():
        print("missing:", p)
        continue
    ep, cer = load_val_cer(p)
    fam = 0 if label.startswith(("Mamba-2", "Mamba-3")) else 1
    color = curve_colors[label]
    ls = "--" if "5e-5" in label else "-"
    axes[fam].plot(ep, cer, color=color, ls=ls, lw=1.4, label=label)
axes[0].set_title("Mamba cores (Round 1 vs Round 2)")
axes[1].set_title("Transformer control")
for ax in axes:
    ax.set_xlabel("epoch")
    ax.legend(fontsize=6, frameon=False)
axes[0].set_ylabel("validation CER")
fig.tight_layout()
fig.savefig(FIG / "fig_training_curves_cer.pdf", bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------- 2. Round-2 grid bar
round2 = [
    ("Trf 5e-5 (R1)", 0.285, C_TRF),
    ("M2 5e-5 (R1)", 0.412, C_MAMBA),
    ("M2 1e-4", 0.394, C_MAMBA),
    ("M2 3e-4", 0.304, C_MAMBA),
    ("M2 1e-4 +reg", 0.374, C_MAMBA),
    ("M3 1e-4 +reg", 0.357, C_MAMBA3),
    ("M3 3e-4 +reg", 0.310, C_MAMBA3),
    ("Trf 1e-4", 0.246, C_TRF),
]
fig, ax = plt.subplots(figsize=(7.0, 2.7))
labels = [r[0] for r in round2]
vals = [r[1] for r in round2]
colors = [r[2] for r in round2]
bars = ax.bar(range(len(vals)), vals, color=colors, alpha=0.85)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.3f}",
            ha="center", fontsize=7)
ax.set_xticks(range(len(vals)))
ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
ax.set_ylabel("test CER")
ax.set_ylim(0, 0.46)
ax.axhline(0.285, color=C_TRF, ls=":", lw=1, alpha=0.6)
fig.tight_layout()
fig.savefig(FIG / "fig_round2_grid.pdf", bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------- 3. leaderboard
leader = [
    ("Transformer Control (4L)", 0.246, C_TRF),
    ("Transformer Deep (8L)", 0.259, C_TRF),
    ("BiMamba-2 + Gated MLP", 0.294, C_MAMBA),
    ("Deep Hybrid 8L (M2+Attn)", 0.295, C_MAMBA),
    ("Hybrid Mamba-2 (4L)", 0.301, C_MAMBA),
    ("Deep Mamba-3 Hybrid (8L)", 0.301, C_MAMBA3),
    ("Pure BiMamba-2", 0.304, C_MAMBA),
    ("BiMamba-3 + Gated MLP", 0.305, C_MAMBA3),
    ("Pure BiMamba-3", 0.310, C_MAMBA3),
    ("Hybrid Mamba-3 (4L)", 0.326, C_MAMBA3),
    ("Transformer (R1, 5e-5)", 0.285, C_TRF),
    ("Pure Mamba-2 (R1, 5e-5)", 0.412, C_MAMBA),
]
fig, ax = plt.subplots(figsize=(7.0, 3.4))
names = [l[0] for l in leader][::-1]
vals = [l[1] for l in leader][::-1]
colors = [l[2] for l in leader][::-1]
ax.barh(range(len(vals)), vals, color=colors, alpha=0.85)
for i, v in enumerate(vals):
    ax.text(v + 0.004, i, f"{v:.3f}", va="center", fontsize=7)
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names, fontsize=7)
ax.set_xlabel("test CER (lower is better)")
ax.set_xlim(0, 0.46)
ax.axvspan(0.29, 0.315, color="orange", alpha=0.12)
ax.text(0.316, len(names) - 6.2, "statistically indistinguishable\nband (<0.01 CER, seed noise)",
        fontsize=6, color="darkorange")
fig.tight_layout()
fig.savefig(FIG / "fig_leaderboard.pdf", bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------- 4. Phase-1 benchmark
phase1 = [
    ("Transformer (deep, 8L)", 25.9, C_TRF),
    ("BiMamba-2 + Gated MLP (4L)", 29.4, C_MAMBA),
    ("Hybrid M2+Attn (8L)", 29.5, C_MAMBA),
    ("Hybrid M2+Attn (4L)", 30.1, C_MAMBA),
    ("Hybrid M3+Attn (8L)", 30.1, C_MAMBA3),
    ("Mamba-3 + Gated MLP (4L)", 30.5, C_MAMBA3),
    ("Hybrid M3+Attn (4L)", 32.6, C_MAMBA3),
]
fig, ax = plt.subplots(figsize=(7.0, 2.6))
names = [p[0] for p in phase1][::-1]
vals = [p[1] for p in phase1][::-1]
colors = [p[2] for p in phase1][::-1]
ax.barh(range(len(vals)), vals, color=colors, alpha=0.85)
for i, v in enumerate(vals):
    ax.text(v + 0.15, i, f"{v:.1f}%", va="center", fontsize=7)
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names, fontsize=7)
ax.set_xlabel("test CER % (greedy CTC + 4-gram KenLM rescoring)")
ax.set_xlim(0, 36)
fig.tight_layout()
fig.savefig(FIG / "fig_phase1_benchmark.pdf", bbox_inches="tight")
plt.close(fig)

# -------------------- 5 + 6. Phase-2 stats figures (canonical cluster stats)
STATS = ROOT / "dissertation" / "stats"
f_m2 = STATS / "stats_mamba2_vs_conformer.csv"
f_m3 = STATS / "stats_mamba3_vs_conformer.csv"

if f_m2.exists() and f_m3.exists():
    s2 = pd.read_csv(f_m2).set_index("metric")
    s3 = pd.read_csv(f_m3).set_index("metric")
    metrics = ["WER", "CER", "CTC_CER", "SemER"]
    models = ["Conformer", "BiMamba-2 + Gated MLP", "Mamba-3 Stabilized Hybrid"]
    mcolors = {"Conformer": C_CONF, "BiMamba-2 + Gated MLP": C_MAMBA,
               "Mamba-3 Stabilized Hybrid": C_MAMBA3}
    means = {
        "Conformer": [s2.loc[m, "mean_Conformer"] for m in metrics],
        "BiMamba-2 + Gated MLP": [s2.loc[m, "mean_BiMamba2"] for m in metrics],
        "Mamba-3 Stabilized Hybrid": [s3.loc[m, "mean_Mamba3Hybrid"] for m in metrics],
    }

    fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.7))
    for ax, met, i in zip(axes, metrics, range(4)):
        for j, m in enumerate(models):
            v = means[m][i]
            ax.bar(j, v, color=mcolors[m], alpha=0.85, width=0.6)
            ax.text(j, v + 0.008, f"{v:.3f}", ha="center", fontsize=6.5)
        ax.set_title(met, fontsize=9)
        ax.set_xticks(range(3))
        ax.set_xticklabels(["Conf", "M2+MLP", "M3-Hyb"], fontsize=7)
        ax.set_ylim(bottom=0)
    fig.suptitle("Phase 2 continuous word-level decoding (62 test sentences, sentence-mean)",
                 fontsize=9, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG / "fig_phase2_metrics.pdf", bbox_inches="tight")
    plt.close(fig)

    # forest plot of deltas (WER/CER/CTC in percentage points)
    rows = []
    for label, s in [("BiMamba-2", s2), ("Mamba-3", s3)]:
        for met in ["WER", "CER", "CTC_CER"]:
            r = s.loc[met]
            rows.append((f"{label} — {met}", 100 * r["delta"],
                         100 * r["ci_lo"], 100 * r["ci_hi"]))
    fig, ax = plt.subplots(figsize=(6.6, 2.9))
    for y, (lab, d, lo, hi) in enumerate(rows[::-1], start=1):
        color = C_MAMBA if lo > 0 else (C_TRF if hi < 0 else "#888888")
        ax.plot([lo, hi], [y, y], color=color, lw=2)
        ax.plot(d, y, "o", color=color, ms=5)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(range(1, len(rows) + 1))
    ax.set_yticklabels([r[0] for r in rows[::-1]], fontsize=7)
    ax.set_xlabel("Δ vs Conformer (percentage points; >0 favours Mamba)")
    ax.set_title("Paired bootstrap, 10,000 resamples (blue: Mamba better, "
                 "red: Conformer better, grey: n.s.)", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig_bootstrap_delta.pdf", bbox_inches="tight")
    plt.close(fig)
else:
    print("NOTE: dissertation/stats/*.csv not found — run cluster/run_all.sh "
          "and pull; skipping fig_phase2_metrics + fig_bootstrap_delta")

# ------------------------------------------- 7. latency scaling (log-log)
bench = json.loads((ROOT / "benchmark_out/benchmark_complexity_report.json").read_text())
fig, ax = plt.subplots(figsize=(5.6, 3.2))
seqs = np.array(bench["seq_lengths"], dtype=float)
for name, color in [("Conformer", C_CONF), ("BiMamba-2 + Gated MLP", C_MAMBA),
                    ("Mamba-3 Stabilized Hybrid", C_MAMBA3)]:
    sc = bench["models"][name]["scaling"]
    ms = np.array([sc[str(int(s))]["step_ms"] for s in seqs])
    ax.plot(seqs, ms, "o-", color=color, lw=1.6, ms=4, label=name)
t = np.linspace(seqs[0], seqs[-1], 100)
ax.plot(t, 0.05 * t, "k:", lw=1, label=r"$\mathcal{O}(T)$ ref")
ax.plot(t, 0.0003 * t ** 2, "k--", lw=1, label=r"$\mathcal{O}(T^2)$ ref")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("sequence length T (frames)")
ax.set_ylabel("train step time (ms)")
ax.legend(fontsize=6.5, frameon=False)
ax.set_title("Latency scaling probe (CPU, batch 2, 306 ch)", fontsize=9)
fig.tight_layout()
fig.savefig(FIG / "fig_latency_scaling.pdf", bbox_inches="tight")
plt.close(fig)

# ------------------------------------------- 8. per-subject Phase 2
subj = pd.DataFrame({
    "subject": ["S15", "S16", "S6"],
    "n": [22, 27, 13],
    "WER": [76.6, 71.8, 79.9],
    "CER": [61.9, 57.5, 64.2],
    "SemER": [0.0984, 0.0882, 0.1085],
})
fig, ax = plt.subplots(figsize=(5.4, 2.7))
x = np.arange(3)
w = 0.35
ax.bar(x - w / 2, subj.WER, w, label="WER", color=C_MAMBA3, alpha=0.85)
ax.bar(x + w / 2, subj.CER, w, label="CER", color=C_MAMBA, alpha=0.85)
for i, (wr, cr) in enumerate(zip(subj.WER, subj.CER)):
    ax.text(i - w / 2, wr + 1, f"{wr:.1f}", ha="center", fontsize=7)
    ax.text(i + w / 2, cr + 1, f"{cr:.1f}", ha="center", fontsize=7)
ax.set_xticks(x)
ax.set_xticklabels([f"{s} (n={n})" for s, n in zip(subj.subject, subj.n)])
ax.set_ylabel("error rate %")
ax.legend(frameon=False, fontsize=8)
ax.set_title("Mamba-3 Stabilized Hybrid — per-subject breakdown", fontsize=9)
fig.tight_layout()
fig.savefig(FIG / "fig_per_subject.pdf", bbox_inches="tight")
plt.close(fig)

print("done. figures in", FIG)
for f in sorted(FIG.glob("*.pdf")):
    print("  ", f.name)
