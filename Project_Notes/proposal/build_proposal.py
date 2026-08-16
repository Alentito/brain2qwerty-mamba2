"""COM865 proposal PDF: Brain2Qwerty V1/V2 analysis + two proposed architectures.

Generates Project_Notes/COM865_Proposal_Mamba_Brain2Qwerty.pdf
"""

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, PageBreak, KeepTogether,
)

ROOT = Path(__file__).resolve().parent
OUT_PDF = ROOT / "COM865_Proposal_Mamba_Brain2Qwerty.pdf"
FIG_A = ROOT / "fig_arch_a.png"
FIG_B = ROOT / "fig_arch_b.png"

# ----------------------------------------------------------------------------
# Architecture diagrams
# ----------------------------------------------------------------------------

NAVY = "#1f3864"
BLUE = "#2e75b6"
TEAL = "#2a9d8f"
ORANGE = "#e76f51"
GRAY = "#666666"


def draw_flow(blocks, path, title, subtitle):
    """blocks: list of (label, color, width_scale)."""
    fig, ax = plt.subplots(figsize=(12.5, 2.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 26)
    ax.axis("off")

    total_w = sum(w for _, _, w in blocks)
    gap = 2.2
    scale = (100 - gap * (len(blocks) + 1)) / total_w
    x = gap
    centers = []
    for label, color, w in blocks:
        bw = w * scale
        box = FancyBboxPatch(
            (x, 7), bw, 12,
            boxstyle="round,pad=0.35,rounding_size=1.2",
            linewidth=1.2, edgecolor=color, facecolor=color + "22",
        )
        ax.add_patch(box)
        # auto-shrink font so the longest label line fits the box
        # (x-axis 0-100 spans ~12.5in; one char ~0.062 data units per font pt)
        longest = max(len(line) for line in label.split("\n"))
        fs = min(8.2, (bw - 1.6) / (0.072 * max(longest, 1)))
        ax.text(x + bw / 2, 13, label, ha="center", va="center",
                fontsize=fs, color="#222222")
        centers.append((x, x + bw))
        x += bw + gap

    for (l0, l1), (r0, r1) in zip(centers[:-1], centers[1:]):
        ax.add_patch(FancyArrowPatch((l1 + 0.4, 13), (r0 - 0.4, 13),
                                     arrowstyle="-|>", mutation_scale=13,
                                     color=GRAY, linewidth=1.4))

    ax.text(50, 23.2, title, ha="center", va="center",
            fontsize=11.5, fontweight="bold", color=NAVY)
    ax.text(50, 1.5, subtitle, ha="center", va="center",
            fontsize=8, color=GRAY, style="italic")
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


draw_flow(
    [
        ("Raw MEG\nsentence segment\n(0.5-45 Hz, 100 Hz)", BLUE, 14),
        ("Per-subject 2D-Fourier\nchannel merger\n(270 virtual ch, V1)", NAVY, 16),
        ("Deep dilated Conv encoder\n(depth 8, hidden 2048, V1)", NAVY, 16),
        ("Temporal downsampling\n(k16 / s4, V2)", BLUE, 13),
        ("Conformer stack\n(4L, local conv + attention, V2)", BLUE, 16),
        ("Dual heads:\nCTC (char) + keystroke-CE aux", ORANGE, 16),
        ("Word contrastive +\nLLM / KenLM decoding (V2)", TEAL, 14),
    ],
    FIG_A,
    "Architecture A — B2Q-Hybrid (best of V1 + V2)",
    "V1's stronger channel merger and deep conv encoder, V2's whole-sentence CTC framework, augmentation and staged losses",
)

draw_flow(
    [
        ("Raw MEG\nsentence segment", BLUE, 12),
        ("Per-subject 2D-Fourier\nchannel merger (V1)", NAVY, 14),
        ("Deep dilated Conv encoder\n(V1)", NAVY, 13),
        ("Temporal downsampling\n(k16 / s4)", BLUE, 12),
        ("Depthwise conv\nlocal mixer (Conformer frontend)", BLUE, 14),
        ("NeMo-style hybrid stack\nMamba-2 blocks (SSD) +\nperiodic attention (M-M-M-A)", TEAL, 18),
        ("CTC head (char)\n+ word contrastive", ORANGE, 14),
        ("LLM / KenLM\ndecoding", TEAL, 11),
    ],
    FIG_B,
    "Architecture B — B2Q-Mamba (NeMo-style Mamba/Attention hybrid)",
    "Attention blocks of the sequence stack replaced by interleaved Mamba-2 state-space blocks with sparse global attention layers",
)

# ----------------------------------------------------------------------------
# PDF document
# ----------------------------------------------------------------------------

styles = getSampleStyleSheet()

body = ParagraphStyle(
    "Body", parent=styles["Normal"], fontName="Times-Roman", fontSize=10.5,
    leading=16, alignment=TA_JUSTIFY, textColor=HexColor("#333333"),
    spaceAfter=8,
)
h1 = ParagraphStyle(
    "H1", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=15,
    leading=19, textColor=HexColor(NAVY), spaceBefore=16, spaceAfter=8,
)
h2 = ParagraphStyle(
    "H2", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=12,
    leading=15, textColor=HexColor(BLUE), spaceBefore=12, spaceAfter=6,
)
bullet = ParagraphStyle(
    "Bullet", parent=body, leftIndent=16, bulletIndent=6, spaceAfter=4,
)
caption = ParagraphStyle(
    "Caption", parent=styles["Normal"], fontName="Helvetica", fontSize=9,
    textColor=HexColor(GRAY), alignment=TA_CENTER, spaceBefore=4, spaceAfter=10,
)
cell = ParagraphStyle(
    "Cell", parent=styles["Normal"], fontName="Helvetica", fontSize=8.6,
    leading=11.5, textColor=HexColor("#333333"),
)
cellb = ParagraphStyle("CellB", parent=cell, fontName="Helvetica-Bold")
ref = ParagraphStyle(
    "Ref", parent=styles["Normal"], fontName="Helvetica", fontSize=9.5,
    leading=13.5, leftIndent=24, firstLineIndent=-24, spaceAfter=4,
    textColor=HexColor("#333333"),
)


def three_line_table(data, col_widths, header_bold=True):
    t = Table(data, colWidths=col_widths, repeatRows=1)
    cmds = [
        ("LINEABOVE", (0, 0), (-1, 0), 1.5, HexColor("#000000")),
        ("LINEBELOW", (0, 0), (-1, 0), 0.75, HexColor("#000000")),
        ("LINEBELOW", (0, -1), (-1, -1), 1.5, HexColor("#000000")),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]
    t.setStyle(TableStyle(cmds))
    return t


story = []

# --- Title -------------------------------------------------------------------
story.append(Spacer(1, 0.4 * cm))
story.append(Paragraph(
    "Decoding Typed Sentences from Non-Invasive Brain Recordings:<br/>"
    "A Hybrid Conv-Conformer and a NeMo-Style Mamba Architecture "
    "for the SpanishBCBL MEG Dataset",
    ParagraphStyle("Title", parent=styles["Title"], fontName="Helvetica-Bold",
                   fontSize=17, leading=22, textColor=HexColor(NAVY),
                   alignment=TA_CENTER, spaceAfter=14),
))
story.append(Paragraph(
    "COM865 Research Project Proposal — Brain2Qwerty Extension",
    ParagraphStyle("Sub", parent=body, alignment=TA_CENTER,
                   fontName="Helvetica", fontSize=12, textColor=HexColor(GRAY),
                   spaceAfter=4),
))
story.append(Paragraph(
    "August 2026 — baseline reproduced on Kelvin2 HPC (test CER 0.389, matching the published V1 baseline of ~0.38)",
    ParagraphStyle("Sub2", parent=body, alignment=TA_CENTER,
                   fontName="Helvetica-Oblique", fontSize=9.5,
                   textColor=HexColor(GRAY), spaceAfter=16),
))

# --- 1. Background -----------------------------------------------------------
story.append(Paragraph("1. Background and Motivation", h1))
story.append(Paragraph(
    "Brain2Qwerty (Lévy et al., 2025) demonstrated that sentences typed on a QWERTY "
    "keyboard can be decoded from non-invasive magnetoencephalography (MEG) with a "
    "character error rate (CER) of roughly 0.32–0.38, using a convolutional encoder "
    "followed by a sentence-level transformer trained with per-keystroke "
    "cross-entropy. The underlying SpanishBCBL dataset (BCBL, San Sebastián) contains "
    "MEG recordings of 35 healthy native Spanish speakers typing briefly memorised "
    "5–8 word sentences (~5.1K sentences, ~193K characters), together with a parallel "
    "EEG cohort (~4K sentences). Meta has since released a second-generation codebase "
    "(brain2qwerty_v2) targeting the companion English dataset, which replaces the "
    "per-keystroke classification framing with a whole-sentence, CTC-based, "
    "speech-recognition-style pipeline augmented by word-level contrastive alignment "
    "and a LoRA-adapted LLM decoder.",
    body))
story.append(Paragraph(
    "As the foundation for this project, the published V1 baseline has been "
    "successfully reproduced on the Kelvin2 HPC cluster (4× V100, DDP): test CER "
    "0.389 (micro-averaged) against the paper's ~0.38, best validation CER 0.381, "
    "on 81 usable MEG recordings across 19 held-out test sentences per subject. "
    "This proposal builds on that validated baseline in two directions: "
    "<b>Architecture A</b>, a principled combination of the strongest components of "
    "V1 and V2; and <b>Architecture B</b>, a novel sequence model in which the "
    "attention-based sequence stack is replaced by a NeMo-style hybrid of Mamba-2 "
    "state-space blocks with sparse global attention layers.",
    body))

# --- 2. V1 vs V2 -------------------------------------------------------------
story.append(Paragraph("2. Code-Level Comparison: V1 vs V2", h1))
story.append(Paragraph(
    "Table 1 summarises the differences extracted directly from the two released "
    "packages (<font face='Courier'>brain2qwerty_v1</font> and "
    "<font face='Courier'>brain2qwerty_v2</font>). The two versions are not "
    "incremental refinements of one model; they frame the decoding problem "
    "differently — per-keystroke classification over fixed 0.5 s windows (V1) versus "
    "alignment-free sequence-to-sequence transcription of the whole sentence "
    "segment (V2).",
    body))

cmp_rows = [
    [Paragraph("Component", cellb), Paragraph("V1 (SpanishBCBL)", cellb), Paragraph("V2 (EnglishBCBL)", cellb)],
    [Paragraph("Input unit", cell),
     Paragraph("0.5 s window per keystroke (start −0.2 s), 50 Hz, bandpass 0.1–20 Hz, baseline correction", cell),
     Paragraph("Whole-sentence segment, 100 Hz, bandpass 0.5–45 Hz + 50 Hz notch, no baseline", cell)],
    [Paragraph("Channel merger", cell),
     Paragraph("Per-subject 2D-Fourier merger → 270 virtual channels", cell),
     Paragraph("Same per-subject 2D-Fourier merger (unchanged)", cell)],
    [Paragraph("Encoder", cell),
     Paragraph("SimpleConvTimeAgg: depth 8, hidden 2048, kernel 3, attention time-pooling → one embedding per keystroke", cell),
     Paragraph("SimpleConv: depth 4, hidden 1500, kernel 5, per-frame output (no time pooling)", cell)],
    [Paragraph("Temporal compression", cell),
     Paragraph("None (one window = one token)", cell),
     Paragraph("Learned strided downsampling, kernel 16 / stride 4", cell)],
    [Paragraph("Sequence model", cell),
     Paragraph("TransformerEncoder over keystroke embeddings, 4 layers / 2 heads, ALiBi", cell),
     Paragraph("Conformer, 4 layers / 4 heads, FFN 1024, depthwise conv 17, group norm", cell)],
    [Paragraph("Auxiliary pathway", cell),
     Paragraph("None", cell),
     Paragraph("Auxiliary CTC head whose softmax is linearly blended back into the Conformer input", cell)],
    [Paragraph("Training objective", cell),
     Paragraph("Per-keystroke cross-entropy (29 classes), labels from logged key timestamps", cell),
     Paragraph("Staged 3-loss schedule: char-level CTC (aux+final blend, α=0.7) from epoch 0; word-level SigLIP contrastive (DTW-matched to frozen TinyLlama token embeddings) from epoch 150; LLM loss from epoch 225", cell)],
    [Paragraph("Augmentation", cell),
     Paragraph("None", cell),
     Paragraph("On-device: per-channel constant offset (SD 0.3), SpecAugment time/frequency masking, time-stretch", cell)],
    [Paragraph("Decoder", cell),
     Paragraph("Argmax per keystroke; optional offline KenLM n-gram beam rescoring", cell),
     Paragraph("TinyLlama-1.1B + LoRA (rank 2), conditioned on CTC greedy text and neural word embeddings; beam 16, length penalty 0.2", cell)],
    [Paragraph("Optimiser", cell),
     Paragraph("AdamW lr 5e-5, wd 1e-4, OneCycleLR", cell),
     Paragraph("AdamW lr 8e-4, wd 1e-3, warmup + cosine, bf16, grad-accum 2", cell)],
    [Paragraph("Metrics", cell),
     Paragraph("CER, WER (argmax keystrokes)", cell),
     Paragraph("CTC-CER (greedy, blank-collapsed), LLM CER/WER, SemER (RoBERTa-large semantic error rate, test only)", cell)],
]
story.append(three_line_table(cmp_rows, [3.1 * cm, 5.95 * cm, 6.45 * cm]))
story.append(Paragraph("Table 1 — V1 vs V2, extracted from the released configs and module code.", caption))

story.append(Paragraph(
    "Three observations motivate the proposed architectures. <b>(i)</b> V1's encoder "
    "is the stronger front-end — deeper (8 vs 4 layers), wider (2048 vs 1500), with "
    "attention-based temporal pooling — but its per-keystroke framing hard-codes the "
    "logged keypress timestamp as ground truth, discards all neural activity outside "
    "the 0.5 s windows, and cannot recover when the label timing is noisy. "
    "<b>(ii)</b> V2's CTC objective removes the timing assumption and its staged "
    "contrastive + LLM losses inject linguistic structure, but its encoder is "
    "shallower and V2 has only been applied to the English cohort. <b>(iii)</b> Both "
    "versions rely on attention as the sole long-range sequence mixer, whose "
    "quadratic cost and fixed positional biases are a poor match for the irregular "
    "inter-keystroke intervals of natural typing.",
    body))

# --- 3. Architecture A --------------------------------------------------------
story.append(Paragraph("3. Architecture A — B2Q-Hybrid (Best of Both Worlds)", h1))
story.append(Image(str(FIG_A), width=15.5 * cm, height=3.25 * cm))
story.append(Paragraph("Figure 1 — Architecture A data flow.", caption))
story.append(Paragraph(
    "Architecture A ports the V2 training framework to SpanishBCBL while restoring "
    "V1's stronger front-end. Concretely: whole-sentence MEG segments are preprocessed "
    "with V2's settings (100 Hz, 0.5–45 Hz, notch 50 Hz) and V2's on-device "
    "augmentations; the per-subject 2D-Fourier merger (retained by both versions) feeds "
    "V1's deep dilated convolutional encoder (depth 8, hidden 2048) with per-frame "
    "outputs instead of V1's attention pooling; V2's learned temporal downsampling "
    "and Conformer stack model the sentence sequence. Training uses a dual objective: "
    "the CTC loss (with V2's auxiliary head and blending) as the primary signal, plus "
    "V1's per-keystroke cross-entropy re-introduced as an auxiliary alignment loss at "
    "CTC-segmented keystroke positions, giving the encoder the explicit motor-event "
    "supervision that made V1 effective. The word-level SigLIP contrastive loss and "
    "LLM decoding stage follow V2's staged schedule; the cheaper KenLM n-gram rescoring "
    "from V1 is retained as a lightweight decoder baseline.",
    body))
story.append(Paragraph(
    "<b>Hypothesis A:</b> combining V1's encoder capacity and keystroke supervision "
    "with V2's alignment-free CTC framework and linguistic losses yields a lower CER "
    "on SpanishBCBL than either released model family, because the two training "
    "signals are complementary (explicit motor alignment vs. free alignment plus "
    "language structure).",
    body))

# --- 4. Architecture B --------------------------------------------------------
story.append(Paragraph("4. Architecture B — B2Q-Mamba (NeMo-Style Hybrid)", h1))
story.append(Image(str(FIG_B), width=15.5 * cm, height=3.25 * cm))
story.append(Paragraph("Figure 2 — Architecture B data flow.", caption))
story.append(Paragraph(
    "Architecture B keeps Architecture A's front-end and losses but replaces the "
    "attention-based sequence stack (V2's Conformer / V1's sentence transformer) with "
    "a NeMo-style hybrid Mamba–Transformer block pattern. Mamba-2 (structured "
    "state-space duality, SSD) blocks perform linear-time selective sequence "
    "modelling: their input-dependent state transitions are a natural fit for typing "
    "signals, where the relevant neural events occur at irregular, subject-specific "
    "intervals rather than on a fixed clock. Following the Nemotron-H / NeMo hybrid "
    "design, Mamba-2 blocks are interleaved with periodic global self-attention "
    "layers in an M-M-M-A pattern (three Mamba blocks per attention layer), so that "
    "cheap selective state-space mixing handles the bulk of the temporal structure "
    "while a small number of attention layers provide exact long-range retrieval "
    "across the sentence. The Conformer's depthwise convolution is kept as the local "
    "mixer in front of the hybrid stack, preserving the local feature smoothing that "
    "both released versions rely on.",
    body))
story.append(Paragraph(
    "Expected advantages over the pure-attention stacks: (i) linear scaling in "
    "sentence length, allowing longer context (multiple sentences, or cross-sentence "
    "language context) at the same compute budget; (ii) an inductive bias toward "
    "smooth, event-driven dynamics matching the motor-production signal; "
    "(iii) fewer attention layers means fewer positional-bias assumptions on an "
    "irregularly sampled event sequence. Risks: Mamba blocks are newer and less "
    "validated on small neural datasets (~81 recordings); state-space layers can "
    "underfit rare long-range dependencies, which the periodic attention layers are "
    "designed to mitigate.",
    body))
story.append(Paragraph(
    "<b>Hypothesis B:</b> a hybrid Mamba-2/attention sequence stack matches or "
    "improves CER relative to the Conformer stack at equal parameter budget on "
    "SpanishBCBL, with measurably better compute scaling on longer sequences.",
    body))

# --- 5. Experimental plan -----------------------------------------------------
story.append(Paragraph("5. Experimental Plan and Evaluation Protocol", h1))

story.append(Paragraph("5.1 Staged experiments", h2))
plan_rows = [
    [Paragraph("Stage", cellb), Paragraph("Experiment", cellb), Paragraph("Comparison / decision", cellb)],
    [Paragraph("S0", cell), Paragraph("Run V1 KenLM n-gram rescoring on the reproduced baseline (pipeline exists, not yet executed)", cell),
     Paragraph("Establishes the LM-corrected reference CER; zero training cost", cell)],
    [Paragraph("S1", cell), Paragraph("Architecture A ablations on SpanishBCBL: (a) V2 framework as-is ported to Spanish; (b) + V1 encoder; (c) + keystroke-CE auxiliary loss", cell),
     Paragraph("Isolates the contribution of each merged component against the 0.389 CER baseline", cell)],
    [Paragraph("S2", cell), Paragraph("Architecture B: swap the Conformer for the M-M-M-A Mamba-2 hybrid at matched parameter count and training schedule", cell),
     Paragraph("Hypothesis B test: CER parity or better, plus wall-clock scaling on longer contexts", cell)],
    [Paragraph("S3", cell), Paragraph("Decoder study on the best encoder: KenLM vs LoRA-LLM (Spanish LM, e.g. a Spanish Llama-family model) vs none", cell),
     Paragraph("Quantifies the neural vs. linguistic share of the error reduction", cell)],
    [Paragraph("S4", cell), Paragraph("Per-subject analysis and adaptation (fine-tune on worst subjects, e.g. S20 at CER 0.552)", cell),
     Paragraph("Addresses the dominant error source observed in the reproduction (subject-level spread 0.287–0.552)", cell)],
]
story.append(three_line_table(plan_rows, [1.2 * cm, 8.3 * cm, 6.0 * cm]))
story.append(Paragraph("Table 2 — Staged experimental plan.", caption))

story.append(Paragraph("5.2 Metrics and protocol", h2))
story.append(Paragraph(
    "Primary metric: CER (micro-averaged, matching the paper and the reproduced "
    "baseline), reported both with and without language-model decoding. Secondary "
    "metrics: WER, exact-sentence rate, and V2's SemER (RoBERTa-large semantic error "
    "rate) for the LLM decoding stage; per-subject CER distributions throughout, "
    "since the reproduction showed subject identity is the dominant variance source. "
    "All models use the V1 deterministic subject-disjoint split on SpanishBCBL for "
    "direct comparability with the reproduced baseline, with the same early-stopping "
    "criterion (val CER, patience 30). Each configuration is trained on 4× V100 with "
    "the V2 bf16/grad-accum settings; the 4.5 GB feature cache pipeline is reused, "
    "with cache rebuilt at 64 GB RAM to avoid the OOM corruption observed during the "
    "reproduction.",
    body))

story.append(Paragraph("5.3 Expected outcomes and risks", h2))
story.append(Paragraph(
    "Expected outcome: Architecture A improves over 0.389 CER (target: match or beat "
    "the paper's LM-corrected ~0.32) because each merged component addresses a known, "
    "independently verified weakness; Architecture B provides the first evaluation of "
    "state-space sequence models for MEG typing decoding, publishable whether or not "
    "it wins, provided the comparison is at matched parameter budget. Main risks and "
    "mitigations: small dataset (mitigated by V2 augmentations, per-subject merger, "
    "and the staged-loss schedule which delays the data-hungry LLM stage); Mamba "
    "implementation risk (mitigated by using the reference mamba-ssm kernels and "
    "keeping Architecture A as a fallback contribution); Spanish LLM availability for "
    "the V2-style decoding stage (mitigated by KenLM Spanish n-grams as the primary "
    "linguistic decoder, with the LLM stage optional).",
    body))

# --- 6. Timeline ---------------------------------------------------------------
story.append(Paragraph("6. Indicative Timeline", h1))
tl_rows = [
    [Paragraph("Weeks", cellb), Paragraph("Work package", cellb)],
    [Paragraph("1–2", cell), Paragraph("S0 rescoring; port V2 data pipeline and CTC framework to SpanishBCBL; rebuild feature cache", cell)],
    [Paragraph("3–5", cell), Paragraph("Architecture A training and ablations (S1)", cell)],
    [Paragraph("6–8", cell), Paragraph("Architecture B implementation (Mamba-2 hybrid stack) and matched-budget training (S2)", cell)],
    [Paragraph("9–10", cell), Paragraph("Decoder study and per-subject adaptation (S3, S4)", cell)],
    [Paragraph("11–12", cell), Paragraph("Analysis, figures, and final report writing", cell)],
]
story.append(three_line_table(tl_rows, [2.2 * cm, 13.3 * cm]))
story.append(Paragraph("Table 3 — Twelve-week plan on Kelvin2 HPC.", caption))

# --- References -----------------------------------------------------------------
story.append(Paragraph("References", h1))
refs = [
    "[1] Lévy, J., et al. (2025). Brain2Qwerty: decoding typed sentences from non-invasive brain recordings. Meta AI / Basque Center on Cognition, Brain and Language.",
    "[2] Zhang, Y., et al. (2025). Companion neuroscience study of brain activity during typing (SpanishBCBL dataset).",
    "[3] Gulati, A., et al. (2020). Conformer: Convolution-augmented Transformer for speech recognition. <i>Interspeech 2020</i>.",
    "[4] Graves, A., et al. (2006). Connectionist temporal classification: labelling unsegmented sequence data with recurrent neural networks. <i>ICML 2006</i>.",
    "[5] Gu, A., & Dao, T. (2023). Mamba: Linear-time sequence modeling with selective state spaces. <i>arXiv:2312.00752</i>.",
    "[6] Dao, T., & Gu, A. (2024). Transformers are SSMs: Generalized models and efficient algorithms through structured state space duality. <i>ICML 2024</i>.",
    "[7] NVIDIA (2025). Nemotron-H: A family of accurate and efficient hybrid Mamba-Transformer models. <i>arXiv:2504.03624</i>.",
]
for r in refs:
    story.append(Paragraph(r, ref))

# --- Build ----------------------------------------------------------------------
def header_footer(canvas, doc):
    canvas.saveState()
    width, height = A4
    canvas.setFont("Helvetica", 8.5)
    canvas.setFillColor(HexColor(GRAY))
    canvas.drawCentredString(width / 2, height - 1.4 * cm,
                             "COM865 Proposal — Mamba/Hybrid Architectures for Brain2Qwerty (SpanishBCBL)")
    canvas.line(3 * cm, height - 1.6 * cm, width - 2.5 * cm, height - 1.6 * cm)
    canvas.drawCentredString(width / 2, 1.3 * cm, f"Page {doc.page}")
    canvas.restoreState()


doc = SimpleDocTemplate(
    str(OUT_PDF), pagesize=A4,
    topMargin=2.4 * cm, bottomMargin=2.4 * cm,
    leftMargin=3 * cm, rightMargin=2.5 * cm,
    title="COM865 Proposal — Mamba/Hybrid Architectures for Brain2Qwerty",
    author="Brain2Qwerty project",
)
doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)
print(f"PDF written: {OUT_PDF}")
