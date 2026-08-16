"""COM865 proposal v2: Mamba-hybrid MEG decoder for SpanishBCBL.

Methodology-focused, deep related-work section, no timeline.
Generates Project_Notes/proposal/COM865_Proposal_B2Q_Mamba_SpanishBCBL.pdf
"""

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
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image,
)

ROOT = Path(__file__).resolve().parent
OUT_PDF = ROOT / "COM865_Proposal_B2Q_Mamba_SpanishBCBL.pdf"
FIG_ARCH = ROOT / "fig_mamba_arch.png"

NAVY = "#1f3864"
BLUE = "#2e75b6"
TEAL = "#2a9d8f"
ORANGE = "#e76f51"
GRAY = "#666666"

# ----------------------------------------------------------------------------
# Architecture figure
# ----------------------------------------------------------------------------

def draw_flow(blocks, path, title, subtitle):
    fig, ax = plt.subplots(figsize=(13.0, 2.7))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 26)
    ax.axis("off")

    total_w = sum(w for _, _, w in blocks)
    gap = 2.0
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
        longest = max(len(line) for line in label.split("\n"))
        fs = min(8.2, (bw - 1.6) / (0.072 * max(longest, 1)))
        ax.text(x + bw / 2, 13, label, ha="center", va="center",
                fontsize=fs, color="#222222")
        centers.append((x, x + bw))
        x += bw + gap

    for (l0, l1), (r0, r1) in zip(centers[:-1], centers[1:]):
        ax.add_patch(FancyArrowPatch((l1 + 0.3, 13), (r0 - 0.3, 13),
                                     arrowstyle="-|>", mutation_scale=13,
                                     color=GRAY, linewidth=1.4))

    ax.text(50, 23.4, title, ha="center", va="center",
            fontsize=11.5, fontweight="bold", color=NAVY)
    ax.text(50, 1.5, subtitle, ha="center", va="center",
            fontsize=8, color=GRAY, style="italic")
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


draw_flow(
    [
        ("Whole-sentence\nMEG segment\n(100 Hz, 0.5-45 Hz,\n50 Hz notch)", BLUE, 15),
        ("Per-subject 2D-Fourier\nchannel merger\n(270 virtual ch)", NAVY, 14),
        ("Dilated conv stem\n+ temporal downsampling\n(k16 / s4)", NAVY, 14),
        ("Depthwise conv\nlocal mixer", BLUE, 11),
        ("Hybrid Mamba-2 / attention stack\nM-M-A block pattern\n(selective SSM + sparse global attn)", TEAL, 19),
        ("Aux CTC head\n(softmax blended\nback into stack)", ORANGE, 12),
        ("Char-CTC head +\nword-contrastive\n(SigLIP, DTW-matched)", ORANGE, 14),
        ("KenLM n-gram /\nSpanish LLM\nbeam decoding", TEAL, 12),
    ],
    FIG_ARCH,
    "Proposed architecture — B2Q-Mamba for SpanishBCBL",
    "V2-style whole-sentence CTC framework and staged losses; attention sequence stack replaced by a hybrid Mamba-2 state-space backbone",
)

# ----------------------------------------------------------------------------
# Document
# ----------------------------------------------------------------------------

styles = getSampleStyleSheet()

body = ParagraphStyle(
    "Body", parent=styles["Normal"], fontName="Times-Roman", fontSize=10.5,
    leading=15.5, alignment=TA_JUSTIFY, textColor=HexColor("#333333"),
    spaceAfter=8,
)
h1 = ParagraphStyle(
    "H1", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=14.5,
    leading=18, textColor=HexColor(NAVY), spaceBefore=15, spaceAfter=7,
)
h2 = ParagraphStyle(
    "H2", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=11.5,
    leading=14.5, textColor=HexColor(BLUE), spaceBefore=10, spaceAfter=5,
)
caption = ParagraphStyle(
    "Caption", parent=styles["Normal"], fontName="Helvetica", fontSize=9,
    textColor=HexColor(GRAY), alignment=TA_CENTER, spaceBefore=4, spaceAfter=10,
)
cell = ParagraphStyle(
    "Cell", parent=styles["Normal"], fontName="Helvetica", fontSize=8.8,
    leading=11.8, textColor=HexColor("#333333"),
)
cellb = ParagraphStyle("CellB", parent=cell, fontName="Helvetica-Bold")
ref = ParagraphStyle(
    "Ref", parent=styles["Normal"], fontName="Helvetica", fontSize=9.3,
    leading=13, leftIndent=24, firstLineIndent=-24, spaceAfter=4,
    textColor=HexColor("#333333"),
)


def three_line_table(data, col_widths):
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("LINEABOVE", (0, 0), (-1, 0), 1.5, HexColor("#000000")),
        ("LINEBELOW", (0, 0), (-1, 0), 0.75, HexColor("#000000")),
        ("LINEBELOW", (0, -1), (-1, -1), 1.5, HexColor("#000000")),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    return t


story = []

# --- Title --------------------------------------------------------------------
story.append(Spacer(1, 0.3 * cm))
story.append(Paragraph(
    "State-Space Sequence Models for Non-Invasive Brain-to-Text Decoding:<br/>"
    "A Hybrid Mamba-2 Architecture Trained on the SpanishBCBL MEG Dataset",
    ParagraphStyle("Title", parent=styles["Title"], fontName="Helvetica-Bold",
                   fontSize=16.5, leading=21, textColor=HexColor(NAVY),
                   alignment=TA_CENTER, spaceAfter=12),
))
story.append(Paragraph(
    "COM865 Research Project Proposal — Extending Brain2Qwerty with Selective "
    "State-Space Sequence Modelling",
    ParagraphStyle("Sub", parent=body, alignment=TA_CENTER, fontName="Helvetica",
                   fontSize=11.5, textColor=HexColor(GRAY), spaceAfter=4),
))
story.append(Paragraph(
    "August 2026 — built on a verified reproduction of the published V1 baseline "
    "(test CER 0.389, 4× V100, Kelvin2 HPC)",
    ParagraphStyle("Sub2", parent=body, alignment=TA_CENTER,
                   fontName="Helvetica-Oblique", fontSize=9.5,
                   textColor=HexColor(GRAY), spaceAfter=14),
))

# --- 1. Introduction -----------------------------------------------------------
story.append(Paragraph("1. Introduction and Motivation", h1))
story.append(Paragraph(
    "Decoding produced language from non-invasive brain recordings is one of the "
    "central open problems in applied neuroengineering: it promises communication "
    "neuroprostheses without surgical risk, but non-invasive signals sit orders of "
    "magnitude below invasive recordings in signal-to-noise ratio. The Brain2Qwerty "
    "study<super>1</super> recently demonstrated that this gap is narrower than "
    "previously assumed: a convolution-plus-transformer model trained on MEG "
    "recordings of participants typing memorised sentences reaches a character error "
    "rate (CER) of 32% with language-model rescoring (~38% without it), and as low "
    "as 19% for the best participants. The dataset underlying that result — "
    "SpanishBCBL, 35 native Spanish speakers typing 5–8 word sentences on an "
    "MR-compatible QWERTY keyboard, ~5.1K MEG sentences / ~193K characters — is "
    "publicly available, and Meta has released two reference codebases: V1 "
    "(per-keystroke classification) and V2 (whole-sentence CTC transcription with "
    "contrastive and LLM-based stages, released for the companion English cohort).",
    body))
story.append(Paragraph(
    "Both reference architectures share one structural choice that this proposal "
    "challenges: the sequence-modelling core is built entirely from attention "
    "(a sentence-level transformer in V1, a Conformer in V2). Attention is a "
    "questionable inductive bias for typing signals. Keystroke-related motor events "
    "occur at irregular, subject-specific intervals; the informative structure is a "
    "smooth, event-driven dynamical process rather than an all-pairs token "
    "interaction; and attention's quadratic cost caps the temporal context that can "
    "be modelled at a fixed compute budget. Selective state-space models — Mamba and "
    "its second-generation formulation based on structured state-space duality "
    "(SSD)<super>9,10</super> — offer precisely the complementary bias: linear-time "
    "sequence modelling with input-dependent state transitions that natively "
    "emphasise or skip time steps according to the signal. This proposal introduces "
    "a hybrid Mamba-2 architecture for MEG keystroke decoding, trained on "
    "SpanishBCBL, that replaces the attention core with interleaved Mamba-2 blocks "
    "and sparse global-attention layers while retaining the strongest validated "
    "components of the V2 training framework.",
    body))
story.append(Paragraph(
    "The project stands on a verified foundation: the V1 baseline has been "
    "reproduced end-to-end on the Kelvin2 HPC cluster (feature cache over 81 usable "
    "recordings, DDP training on 4× V100), reaching test CER 0.389 against the "
    "paper's ~0.38 for the same configuration (Conv+Transformer, no language model), "
    "with best validation CER 0.381. Every number proposed below is therefore "
    "anchored to a working pipeline, not a paper plan.",
    body))

# --- 2. Related work -------------------------------------------------------------
story.append(Paragraph("2. State of the Art: Decoding Language from MEG", h1))

story.append(Paragraph("2.1 From perception to production decoding", h2))
story.append(Paragraph(
    "Modern non-invasive language decoding began with perception. Défossez et "
    "al.<super>2</super> showed that perceived speech segments can be retrieved from "
    "3-second MEG/EEG windows with a contrastive model aligned to wav2vec 2.0 "
    "representations (up to 41% top-10 accuracy from MEG), introducing two ideas "
    "that now pervade the field: per-subject input layers and contrastive alignment "
    "with pretrained speech/language models. Tang et al.<super>3</super> "
    "demonstrated semantic reconstruction of perceived and imagined language from "
    "fMRI using a language-model decoder, establishing the "
    "neural-encoder-plus-language-model template. Decoding production is harder and "
    "younger: before Brain2Qwerty<super>1</super>, the strongest non-invasive "
    "production result decoded only 10 letters from EEG at 75.8% CER (Crell and "
    "Müller-Putz, 2024), while Brain2Qwerty's 32% CER over a 29-character alphabet "
    "on MEG — against 67% on EEG for the identical model — remains the reference "
    "point and quantifies why this proposal works on MEG rather than EEG.",
    body))

story.append(Paragraph("2.2 Speech-recognition-style MEG decoders", h2))
story.append(Paragraph(
    "A second line of work treats MEG as a distorted speech signal and imports "
    "automatic-speech-recognition machinery wholesale. NeuSpeech and its successor "
    "MAD (Yang et al., 2024) transform raw MEG and feed it into a pretrained "
    "Whisper model for open-vocabulary text decoding without teacher forcing, "
    "reporting state-of-the-art MEG-to-text results on perception tasks. BrainECHO "
    "(2024)<super>4</super> inserts a vector-quantised spectrogram autoencoder "
    "between brain signals and Whisper to improve generalisation. In the EEG "
    "literature, EEG-to-Text (Wang and Ji, 2022) and DeWave (Duan et al., 2023) "
    "align word-level EEG embeddings with BART via a pretrain-finetune paradigm. "
    "Meta's V2 codebase internalises the same lesson for production decoding: "
    "char-level CTC (Graves et al.<super>13</super>) for alignment-free training, a "
    "Conformer sequence core<super>12</super>, a word-level SigLIP contrastive loss "
    "matched to frozen LLM token embeddings via dynamic time warping, and a "
    "LoRA-adapted LLM decoding stage — a full ASR stack transplanted onto brain "
    "signals. The proposed architecture inherits this framing.",
    body))

story.append(Paragraph("2.3 Benchmarks and the scaling signal", h2))
story.append(Paragraph(
    "The 2025 PNPL competition<super>5,6</super> (LibriBrain: >50 hours of "
    "within-subject MEG during audiobook listening, with speech-detection and "
    "phoneme-classification leaderboards) has given the field its first "
    "ImageNet-style yardstick and produced two findings directly relevant here. "
    "First, within-subject data depth drives decoding gains more than cohort "
    "breadth — supporting a design that extracts maximal per-sentence context. "
    "Second, competition-winning solutions reformulate classification as sequence "
    "reconstruction with auxiliary speech-feature targets, echoing V2's "
    "multi-objective design and validating the staged, auxiliary-loss training "
    "regime adopted below.",
    body))

story.append(Paragraph("2.4 State-space models for neural signals", h2))
story.append(Paragraph(
    "Structured state-space sequence models (S4) and their selective successor "
    "Mamba<super>9</super> model long sequences in linear time with input-dependent "
    "dynamics; Mamba-2's SSD formulation<super>10</super> unifies SSMs and attention "
    "and enables larger state dimensions at transformer-competitive quality. In the "
    "biosignal domain the evidence is accumulating rapidly: EEGMamba (Wang et al., "
    "2025) uses Mamba blocks as the backbone of a multi-task EEG foundation model; "
    "MI-Mamba (2025) shows a CNN+Mamba hybrid beating prior state of the art on "
    "motor-imagery classification with roughly 6× fewer parameters; bidirectional "
    "Mamba variants (BiMamba, EmotionMamba) improve long-sequence EEG emotion "
    "recognition; and State Mamba (Weng et al., AAAI 2026) models spatiotemporal "
    "state transitions explicitly, topping transformer-based EEG foundation models "
    "on three public benchmarks. Closest to this proposal, a NeurIPS 2025 study "
    "demonstrates generalizable, real-time neural decoding with hybrid state-space "
    "models — evidence that SSM-attention hybrids are viable specifically for "
    "neural decoding, not only for classification benchmarks. In large-scale "
    "language modelling, NVIDIA's Nemotron-H family<super>11</super> established "
    "the hybrid pattern this proposal borrows: mostly Mamba-2 layers with a small "
    "fraction of interleaved self-attention layers recovering exact long-range "
    "retrieval at a fraction of the inference cost of a pure transformer.",
    body))

story.append(Paragraph("2.5 The gap this project occupies", h2))
story.append(Paragraph(
    "Three facts define the opening. (i) Production decoding from MEG is "
    "under-explored relative to perception — essentially one published model family "
    "(Brain2Qwerty V1/V2), both attention-based. (ii) SSMs have proven themselves on "
    "EEG classification and foundation-model benchmarks but have never been applied "
    "to sentence-level brain-to-text decoding from MEG. (iii) The V2 training "
    "framework (CTC + word-contrastive + LLM decoding) has not been evaluated on "
    "SpanishBCBL at all — it was released for the English cohort. This project sits "
    "at the intersection: the first state-space sequence model for MEG production "
    "decoding, trained with the strongest available framework, on the dataset for "
    "which a verified reproduction baseline already exists.",
    body))

# --- 3. Methodology --------------------------------------------------------------
story.append(Paragraph("3. Proposed Methodology", h1))
story.append(Image(str(FIG_ARCH), width=15.8 * cm, height=3.28 * cm))
story.append(Paragraph(
    "Figure 1 — Proposed B2Q-Mamba architecture. Grey/blue components are inherited "
    "from the validated V1/V2 pipelines; the Mamba-2 hybrid stack is the new "
    "contribution.", caption))

story.append(Paragraph("3.1 Data pipeline and preprocessing (from V2)", h2))
story.append(Paragraph(
    "The unit of modelling is the whole typed sentence, not the individual "
    "keystroke. Continuous MEG is resampled to 100 Hz, bandpass-filtered at "
    "0.5–45 Hz with a 50 Hz notch, RobustScaler-normalised, and segmented per "
    "sentence. This departs deliberately from V1's 0.5 s per-keystroke windows "
    "(50 Hz, 0.1–20 Hz): the wider band retains beta/low-gamma motor content that "
    "V1 discards by construction, and whole-sentence segments let the sequence "
    "model — rather than the label timestamps — decide where keystroke information "
    "lives. V2's on-device augmentation suite is applied during training only: "
    "per-channel constant offsets (SD 0.3) simulating session drift, SpecAugment "
    "time/frequency masking, and time-stretch, the last being particularly apt for "
    "typing data since it varies inter-keystroke rhythm without changing the "
    "character sequence. With only 81 usable recordings, this augmentation is a "
    "necessity, not a luxury.",
    body))

story.append(Paragraph("3.2 Front-end: subject-conditioned convolutional encoder", h2))
story.append(Paragraph(
    "The front-end retains the component both released versions agree on — the "
    "per-subject 2D-Fourier channel merger projecting the sensor array onto 270 "
    "learned virtual channels — followed by a dilated convolutional stem and "
    "learned temporal downsampling (kernel 16, stride 4) that compresses the "
    "waveform into a frame sequence at ~6 Hz. Per-subject conditioning matters "
    "here because the reproduction confirmed subject identity as the dominant "
    "error source (per-subject CER spanning 0.287–0.552 on the baseline); the "
    "merger is the architecture's primary mechanism for absorbing inter-subject "
    "anatomical and session differences before sequence modelling begins.",
    body))

story.append(Paragraph("3.3 Core contribution: hybrid Mamba-2 sequence stack", h2))
story.append(Paragraph(
    "The Conformer/transformer core is replaced by a hybrid stack in the Nemotron-H "
    "pattern<super>11</super>: groups of three Mamba-2 blocks followed by one "
    "global self-attention layer (M-M-A), repeated twice, with a depthwise "
    "convolution as the local mixer in front of the stack (preserving the local "
    "smoothing that both V1's conv encoder and V2's Conformer rely on). Each "
    "Mamba-2 block uses the SSD parametrisation — d_model matched to the encoder "
    "output, state dimension 64–128, head dimension 64, expansion factor 2 — "
    "implemented with the reference mamba-ssm kernels. The design rests on three "
    "methodological arguments:",
    body))
args = [
    ("<b>Inductive bias.</b> Typing is an event-driven process: motor preparation, "
     "keypress, and inter-key intervals of irregular duration. Mamba's selective "
     "scan makes the state transition input-dependent, so the model can hold motor "
     "state across a long pause and update sharply at a keypress — the continuous-"
     "time analogue of the task's dynamics. Attention must learn this behaviour "
     "from scratch through positional encodings that assume a uniform clock."),
    ("<b>Compute and context.</b> Sequence cost grows linearly in segment length, "
     "so the same 4× V100 budget that trains V2 on single sentences can train the "
     "Mamba variant on longer contexts — multi-sentence spans that let the model "
     "exploit cross-sentence language statistics, the single most predictable "
     "source of CER reduction given that the reproduction's residual errors are "
     "predominantly single-character substitutions in otherwise correct sentences."),
    ("<b>Why not pure Mamba.</b> SSMs are known to underperform at exact long-range "
     "retrieval — precisely what is needed to resolve an ambiguous character using "
     "a word typed two seconds earlier. Interleaving one attention layer per three "
     "Mamba blocks restores retrieval at ~25% of the attention compute of a pure "
     "transformer, a trade validated at scale by Nemotron-H<super>11</super> and, "
     "for neural signals specifically, by hybrid-SSM real-time decoding work at "
     "NeurIPS 2025. The M-M-A ratio is an explicit ablation axis (M-only, M-A, "
     "M-M-M-A) against the Conformer core at matched parameter count."),
]
for a in args:
    story.append(Paragraph(a, ParagraphStyle("Arg", parent=body, leftIndent=14,
                                             spaceAfter=5)))

story.append(Paragraph("3.4 Training objectives (adopted and adapted from V2)", h2))
story.append(Paragraph(
    "Training follows V2's staged multi-objective schedule, which the LibriBrain "
    "competition results independently support as the right regime for MEG "
    "sequence decoding<super>5</super>:",
    body))
obj_rows = [
    [Paragraph("Stage", cellb), Paragraph("Objective", cellb), Paragraph("Rationale for this project", cellb)],
    [Paragraph("From epoch 0", cell),
     Paragraph("Character-level CTC on final logits + auxiliary CTC head whose softmax is blended back into the stack input (blend α=0.7)", cell),
     Paragraph("Removes V1's brittle assumption that the logged keypress timestamp is the neural event time; the aux head deepens supervision of the early stack — important because Mamba blocks have no attention shortcut", cell)],
    [Paragraph("Mid-training", cell),
     Paragraph("Word-level SigLIP contrastive loss: CTC-segmented pseudo-word embeddings, DTW-matched to frozen LLM token embeddings", cell),
     Paragraph("Injects lexical structure the character CTC cannot represent; targets exactly the substitution-type errors dominating the reproduction's residual error mass", cell)],
    [Paragraph("Late stage (optional)", cell),
     Paragraph("LLM decoding loss: LoRA-adapted Spanish LLM conditioned on CTC greedy text + adapted neural word embeddings", cell),
     Paragraph("Replaces V1's offline 9-gram rescoring with a learned decoder; kept optional because Spanish LLM selection is a separate decision and KenLM Spanish n-grams provide the primary linguistic baseline", cell)],
]
story.append(three_line_table(obj_rows, [2.3 * cm, 6.3 * cm, 6.9 * cm]))
story.append(Paragraph("Table 1 — Staged training objectives and their justification.", caption))
story.append(Paragraph(
    "Optimisation uses V2's settings (AdamW lr 8e-4, weight decay 1e-3, "
    "warmup+cosine schedule, bf16 mixed precision, gradient accumulation 2), with "
    "one deliberate exception: the Mamba blocks are initialised with a smaller "
    "effective learning rate or a short SSM-only warmup, since selective-scan "
    "parameters are more sensitive to early-training instability than attention "
    "layers — a standard precaution in hybrid SSM training.",
    body))

# --- 4. Evaluation -----------------------------------------------------------------
story.append(Paragraph("4. Evaluation Protocol", h1))
story.append(Paragraph(
    "Primary metric: sentence-level CER (micro-averaged) on the held-out test "
    "split, directly comparable to the reproduced 0.389 baseline and the paper's "
    "0.38/0.32 figures (without/with language model). Secondary metrics: WER, "
    "exact-sentence rate, per-subject CER distributions, and the semantic error "
    "rate (RoBERTa-large embedding distance) adopted from V2 for the LLM decoding "
    "stage. All comparisons use the V1 deterministic subject-disjoint split and "
    "identical early stopping (val CER, patience 30). The ablation grid is the "
    "scientific core of the project: (a) V2 framework on SpanishBCBL with its "
    "original Conformer core — isolates the dataset/framework transfer; (b) the "
    "same with the hybrid Mamba-2 core at matched parameters — isolates the "
    "architecture variable; (c) M-M-A pattern variants and a bidirectional Mamba "
    "variant (motivated by BiMamba results on EEG) — isolates the design choice; "
    "(d) decoder study (none / KenLM / LLM) — isolates the linguistic "
    "contribution. Each cell is a same-budget training run, so every claim in the "
    "final report is a controlled comparison against the verified baseline.",
    body))

# --- 5. Expected contributions -----------------------------------------------------
story.append(Paragraph("5. Expected Contributions and Risks", h1))
story.append(Paragraph(
    "Contributions: (1) the first state-space sequence model for sentence-level "
    "brain-to-text decoding from MEG production data; (2) the first evaluation of "
    "the V2 CTC/contrastive/LLM framework on SpanishBCBL, extending a framework "
    "released only for English; (3) a controlled, same-budget comparison of "
    "attention, Conformer, and hybrid Mamba-2 cores on an identical pipeline — a "
    "result of value to the MEG decoding community whichever way it resolves; "
    "(4) an open, reproducible extension of Meta's reference codebase.",
    body))
story.append(Paragraph(
    "Risks and mitigations. The dataset is small (~81 recordings): mitigated by the "
    "augmentation suite, per-subject merger, and the staged schedule that delays "
    "data-hungry objectives. Mamba blocks are less battle-tested on small neural "
    "datasets: mitigated by keeping the Conformer-core run (ablation a) as a "
    "guaranteed contribution, and by the hybrid design itself, which degrades "
    "gracefully toward a sparse-attention model. Spanish LLM availability for the "
    "decoding stage: mitigated by KenLM Spanish n-grams as the primary linguistic "
    "decoder, with the LLM stage optional and clearly marked as such.",
    body))

# --- References ----------------------------------------------------------------------
story.append(Paragraph("References", h1))
refs = [
    "[1] Lévy, J., Zhang, M., Pinet, S., Rapin, J., Banville, H., d'Ascoli, S., King, J.-R., et al. (2025). Brain-to-Text Decoding: A Non-invasive Approach via Typing (Brain2Qwerty). <i>arXiv:2502.17480</i>. Companion neuroscience study: Zhang, M., et al. (2025).",
    "[2] Défossez, A., Caucheteux, C., Rapin, J., Kabeli, O., & King, J.-R. (2023). Decoding speech perception from non-invasive brain recordings. <i>Nature Machine Intelligence, 5</i>, 912–921.",
    "[3] Tang, J., LeBel, A., Jain, S., & Huth, A. G. (2023). Semantic reconstruction of continuous language from non-invasive brain recordings. <i>Nature Neuroscience, 26</i>, 858–866.",
    "[4] Yang, C., et al. (2024). NeuSpeech: decoding neural signals as speech with a pretrained Whisper model; and MAD: MEG-to-text decoding on unseen text. <i>arXiv preprints</i>. See also BrainECHO (2024): semantic brain signal decoding via vector-quantized spectrogram reconstruction, <i>arXiv:2410.14971</i>.",
    "[5] Landau, G., Özdogan, M., Elvers, G., et al. (2025). The 2025 PNPL Competition: Speech Detection and Phoneme Classification in the LibriBrain Dataset. <i>arXiv:2506.10165</i>.",
    "[6] Özdogan, M., Elvers, G., Mantegna, F., et al. (2025). LibriBrain: Over 50 hours of within-subject MEG to improve speech decoding methods at scale. <i>arXiv:2506.02098</i>.",
    "[7] Wang, J., Zhao, S., Luo, Z., Zhou, Y., Li, S., & Pan, G. (2025). EEGMamba: an EEG foundation model with Mamba. <i>Neural Networks</i>.",
    "[8] MI-Mamba: a hybrid motor-imagery EEG classification model with Mamba's global scanning (2025). <i>Annals of the New York Academy of Sciences</i>. See also Weng, W., et al. (2026). State Mamba: a spatiotemporal EEG state-space model. <i>AAAI 2026</i>; and Generalizable, real-time neural decoding with hybrid state-space models. <i>NeurIPS 2025</i>.",
    "[9] Gu, A., & Dao, T. (2023). Mamba: linear-time sequence modeling with selective state spaces. <i>arXiv:2312.00752</i>.",
    "[10] Dao, T., & Gu, A. (2024). Transformers are SSMs: generalized models and efficient algorithms through structured state space duality. <i>ICML 2024</i>.",
    "[11] NVIDIA (2025). Nemotron-H: a family of accurate and efficient hybrid Mamba-Transformer models. <i>arXiv:2504.03624</i>.",
    "[12] Gulati, A., et al. (2020). Conformer: convolution-augmented Transformer for speech recognition. <i>Interspeech 2020</i>.",
    "[13] Graves, A., Fernández, S., Gomez, F., & Schmidhuber, J. (2006). Connectionist temporal classification: labelling unsegmented sequence data with recurrent neural networks. <i>ICML 2006</i>.",
]
for r in refs:
    story.append(Paragraph(r, ref))


def header_footer(canvas, doc):
    canvas.saveState()
    width, height = A4
    canvas.setFont("Helvetica", 8.5)
    canvas.setFillColor(HexColor(GRAY))
    canvas.drawCentredString(width / 2, height - 1.4 * cm,
                             "COM865 Proposal — Hybrid Mamba-2 MEG Decoder for SpanishBCBL")
    canvas.line(3 * cm, height - 1.6 * cm, width - 2.5 * cm, height - 1.6 * cm)
    canvas.drawCentredString(width / 2, 1.3 * cm, f"Page {doc.page}")
    canvas.restoreState()


doc = SimpleDocTemplate(
    str(OUT_PDF), pagesize=A4,
    topMargin=2.4 * cm, bottomMargin=2.4 * cm,
    leftMargin=3 * cm, rightMargin=2.5 * cm,
    title="COM865 Proposal — Hybrid Mamba-2 MEG Decoder for SpanishBCBL",
    author="Brain2Qwerty project",
)
doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)
print(f"PDF written: {OUT_PDF}")
