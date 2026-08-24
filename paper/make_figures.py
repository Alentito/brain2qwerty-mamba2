"""Figures for the IEEE paper: single-column-width versions."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(sys.executable).parent.parent.parent))
from daimon_runtime import setup_plot

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

setup_plot()

OUT = Path(__file__).resolve().parent
plt.rcParams.update({"font.family": "DejaVu Sans"})

CW = 3.45  # IEEE column width (inches)


def box(ax, x, y, w, h, text, fc="#dbeafe", ec="#1e3a5f", fs=6.8, bold=False):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.8,rounding_size=1.6",
                       facecolor=fc, edgecolor=ec, linewidth=1.0)
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color="#111827")


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=9, color="#1e3a5f", linewidth=1.0))


# ------------------------------------------------- Fig 1: vertical pipeline
fig, ax = plt.subplots(figsize=(CW, 6.6))
ax.set_xlim(0, 100)
ax.set_ylim(0, 150)
ax.axis("off")

W = 64
XC = 18  # main column x (centers at 50)

box(ax, XC, 138, W, 10, "M/EEG segments\n(500 ms windows, 0.5-45 Hz)", fc="#f3f4f6", fs=6.6)
box(ax, XC, 123, W, 10, "Convolutional module\nsubject layer, spatial/temporal convs", fc="#dbeafe", fs=6.4)
box(ax, XC, 106, W, 12, "Hybrid Mamba-2 / attention\n[M,M,M,A] x 2, Nemotron-H style\nRMSNorm; RoPE attention blocks", fc="#c7d2fe", fs=6.4, bold=True)
box(ax, XC, 91, W, 10, "CTC character head\n(auxiliary + final logits)", fc="#dbeafe", fs=6.6)
box(ax, XC, 76, W, 10, "Greedy CTC text\n'stamistosasigue la distribucion'", fc="#fef3c7", fs=6.6)

box(ax, 1, 56, 47, 12, "CTC space segmenter\nframe pooling between spaces\n-> pseudo-word embeddings", fc="#dbeafe", fs=6.2)
box(ax, 53, 56, 46, 12, "Word-contrastive loss\nencoder word embeds vs.\nLLM text embeddings", fc="#ede9fe", fs=6.2)
box(ax, 1, 38, 47, 10, "Projection adapter\n-> LLM embedding space", fc="#dbeafe", fs=6.2)
box(ax, 1, 16, 47, 16, "LoRA-adapted LLM\n(1.1 B params, 0.05% trainable)\nprompt: 'CTC: <greedy text>\nMEG: <word embeds> Out:'\nbeam search (16 beams)", fc="#d1fae5", fs=6.0)
box(ax, 1, 1, 47, 10, "Corrected sentence\n'la estadistica sigue\nla distribucion'", fc="#fef3c7", fs=6.2)

CXC = XC + W / 2  # 50
# main column arrows
arrow(ax, CXC, 138, CXC, 133)
arrow(ax, CXC, 123, CXC, 118)
arrow(ax, CXC, 106, CXC, 101)
arrow(ax, CXC, 91, CXC, 86)
# hybrid stack -> contrastive
arrow(ax, 82, 112, 76, 68)
# CTC head -> segmenter
arrow(ax, 40, 91, 24, 68)
# segmenter -> adapter
arrow(ax, 24, 56, 24, 48)
# adapter -> LLM
arrow(ax, 24, 38, 24, 32)
# greedy text -> LLM
arrow(ax, 82, 81, 48, 28)
# LLM -> corrected
arrow(ax, 24, 16, 24, 11)

fig.savefig(OUT / "fig_architecture.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------- Fig 2: stacked complexity + CER
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(CW, 4.9))

T = np.linspace(16, 4096, 200)
ax1.plot(T, (T ** 2) / (T[-1] ** 2) * 100, label="Self-attention  $\\mathcal{O}(T^2)$",
         color="#dc2626", lw=1.6)
ax1.plot(T, T / T[-1] * 100, label="Mamba-2 (SSD)  $\\mathcal{O}(T)$",
         color="#2563eb", lw=1.6)
ax1.set_xlabel("Sequence length $T$ (frames)", fontsize=8)
ax1.set_ylabel("Relative mixer cost (%)", fontsize=8)
ax1.set_title("(a) Sequence-mixer scaling (illustrative)", fontsize=8.5)
ax1.legend(fontsize=7, frameon=False, loc="upper left")
ax1.grid(alpha=0.25, lw=0.4)
ax1.tick_params(labelsize=7)

labels = ["Crell &\nM.-Putz\nEEG, 10 ch.", "EEGNet\nEEG*", "B2Q\nEEG",
          "B2Q MEG\nworst subj.", "B2Q MEG", "B2Q MEG\nbest subj."]
cer = [75.8, 76.4, 67.0, 45.0, 32.0, 19.0]
colors = ["#9ca3af", "#9ca3af", "#60a5fa", "#93c5fd", "#2563eb", "#1e40af"]
bars = ax2.bar(range(len(labels)), cer, color=colors, width=0.62)
for b, v in zip(bars, cer):
    ax2.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}", ha="center", fontsize=6.5)
ax2.set_xticks(range(len(labels)))
ax2.set_xticklabels(labels, fontsize=6)
ax2.set_ylabel("Character error rate (%)", fontsize=8)
ax2.set_ylim(0, 90)
ax2.set_title("(b) Reported brain-to-text CERs (literature)", fontsize=8.5)
ax2.grid(axis="y", alpha=0.25, lw=0.4)
ax2.tick_params(axis="y", labelsize=7)

fig.tight_layout()
fig.savefig(OUT / "fig_results.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print("figures written")
