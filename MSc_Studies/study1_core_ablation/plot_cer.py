"""Per-subject and multi-arm CER figures for the V1-Mamba ablation report."""

import sys
from pathlib import Path

try:
    sys.path.insert(0, str(Path(sys.executable).parent.parent.parent))
    from daimon_runtime import setup_plot
    setup_plot()
except Exception:
    pass

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------- #
# 1. Round 1: Per-Subject CER Comparison (V1 Baseline)
# --------------------------------------------------------------------------- #
r1_rows = [
    # subject, arm, CER, n_sentences
    ("S15", "V1-Mamba (ours)", 0.455, 24),
    ("S15", "V1 Transformer (ref)", 0.299, 24),
    ("S16", "V1-Mamba (ours)", 0.408, 12),
    ("S16", "V1 Transformer (ref)", 0.250, 12),
    ("S6", "V1-Mamba (ours)", 0.359, 18),
    ("S6", "V1 Transformer (ref)", 0.290, 18),
    ("Pooled", "V1-Mamba (ours)", 0.412, 54),
    ("Pooled", "V1 Transformer (ref)", 0.286, 54),
]
df_r1 = pd.DataFrame(r1_rows, columns=["subject", "arm", "CER", "n_sent"])
df_r1["subject"] = pd.Categorical(df_r1["subject"], ["S15", "S16", "S6", "Pooled"])

fig, ax = plt.subplots(figsize=(8, 4.5))
sns.barplot(
    data=df_r1,
    x="subject",
    y="CER",
    hue="arm",
    palette={"V1-Mamba (ours)": "#d62728", "V1 Transformer (ref)": "#1f77b4"},
    ax=ax,
)
for c in ax.containers:
    ax.bar_label(c, fmt="%.3f", padding=2, fontsize=9)
ax.set_ylim(0, 0.55)
ax.set_ylabel("Test CER (sentence level, vs typed)")
ax.set_xlabel("Subject (n = test sentences)")
ax.set_title(
    "Mamba-2 vs Transformer sentence core — 3-subject ablation\n"
    "identical V1 encoder, 512-width, same splits/optimizer (lr=5e-5)"
)
ax.legend(title="")
fig.savefig("cer_by_subject.png", dpi=220, bbox_inches="tight")
plt.close(fig)
print("saved cer_by_subject.png")

# --------------------------------------------------------------------------- #
# 2. Round 2: 8-Arm Comprehensive Ablation Grid (Pooled Test CER)
# --------------------------------------------------------------------------- #
r2_arms = [
    {"arm": "Transformer\n(baseline)", "family": "Transformer", "lr": "5e-5", "other": "default", "cer": 0.285},
    {"arm": "Mamba2\n(R1 hybrid)", "family": "Mamba2", "lr": "5e-5", "other": "default", "cer": 0.412},
    {"arm": "Mamba2\n(Task 0)", "family": "Mamba2", "lr": "1e-4", "other": "—", "cer": 0.394},
    {"arm": "Mamba2\n(Task 1)", "family": "Mamba2", "lr": "3e-4", "other": "—", "cer": 0.304},
    {"arm": "Mamba2\n(Task 2)", "family": "Mamba2", "lr": "1e-4", "other": "wd 0.1 / gc 1.0", "cer": 0.374},
    {"arm": "Mamba3\n(Task 3)", "family": "Mamba3", "lr": "1e-4", "other": "wd 0.1 / gc 1.0", "cer": 0.357},
    {"arm": "Mamba3\n(Task 4)", "family": "Mamba3", "lr": "3e-4", "other": "wd 0.1 / gc 1.0", "cer": 0.310},
    {"arm": "Transformer\n(Task 5 control)", "family": "Transformer", "lr": "1e-4", "other": "—", "cer": 0.246},
]
df_r2 = pd.DataFrame(r2_arms)

fig2, ax2 = plt.subplots(figsize=(11, 5))
palette = {
    "Transformer": "#1f77b4",
    "Mamba2": "#d62728",
    "Mamba3": "#2ca02c",
}

bars = sns.barplot(
    data=df_r2,
    x="arm",
    y="cer",
    hue="family",
    palette=palette,
    dodge=False,
    ax=ax2,
)
for c in ax2.containers:
    ax2.bar_label(c, fmt="%.3f", padding=3, fontsize=9, fontweight="bold")

# Add horizontal reference lines
ax2.axhline(0.285, color="#1f77b4", linestyle="--", alpha=0.6, label="Transformer R1 Baseline (0.285)")
ax2.axhline(0.246, color="#0b407a", linestyle=":", alpha=0.8, label="Transformer Tuned Control (0.246)")

ax2.set_ylim(0, 0.48)
ax2.set_ylabel("Test CER (Character Error Rate, lower is better)", fontsize=10)
ax2.set_xlabel("Experimental Arm", fontsize=10)
ax2.set_title(
    "Brain2Qwerty V1 Core Ablation: Transformer vs Mamba-2 vs Mamba-3\n"
    "Hyperparameter Tuning & Architectural Comparison (Pooled 3-Subject Test Split)",
    fontsize=11,
    fontweight="bold",
)
ax2.legend(loc="upper right", framealpha=0.9)
fig2.savefig("cer_round2_grid.png", dpi=220, bbox_inches="tight")
plt.close(fig2)
print("saved cer_round2_grid.png")

