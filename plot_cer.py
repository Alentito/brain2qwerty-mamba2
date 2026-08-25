"""Per-subject CER figure for the V1-Mamba ablation report."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(sys.executable).parent.parent.parent))
from daimon_runtime import setup_plot

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

setup_plot()

rows = [
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
df = pd.DataFrame(rows, columns=["subject", "arm", "CER", "n_sent"])
df["subject"] = pd.Categorical(df["subject"], ["S15", "S16", "S6", "Pooled"])

fig, ax = plt.subplots(figsize=(8, 4.5))
sns.barplot(data=df, x="subject", y="CER", hue="arm",
            palette={"V1-Mamba (ours)": "#d62728", "V1 Transformer (ref)": "#1f77b4"},
            ax=ax)
for c in ax.containers:
    ax.bar_label(c, fmt="%.3f", padding=2, fontsize=9)
ax.set_ylim(0, 0.55)
ax.set_ylabel("test CER (sentence level, vs typed)")
ax.set_xlabel("subject (n = test sentences)")
ax.set_title("Mamba-2 vs Transformer sentence core — 3-subject ablation\n"
             "identical V1 encoder, 512-width, same splits/optimizer")
ax.legend(title="")
fig.savefig("cer_by_subject.png", dpi=220, bbox_inches="tight")
print("saved cer_by_subject.png")
