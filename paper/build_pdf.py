# -*- coding: utf-8 -*-
"""Build the IEEE-style two-column PDF paper with reportlab."""
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(sys.executable).parent.parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch, mm
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.colors import HexColor, black
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT
from reportlab.platypus import (BaseDocTemplate, Frame, PageTemplate, Paragraph,
                                Spacer, Image, Table, TableStyle, FrameBreak,
                                NextPageTemplate, KeepTogether)

HERE = Path(__file__).resolve().parent
OUT_PDF = HERE / "Hybrid_Mamba_Transformer_EEG_Brain2Text.pdf"
EQ_DIR = HERE / "eq"
EQ_DIR.mkdir(exist_ok=True)

PAGE_W, PAGE_H = letter
ML = MR = 0.70 * inch
MT, MB = 0.75 * inch, 0.85 * inch
GUTTER = 0.24 * inch
COL_W = (PAGE_W - ML - MR - GUTTER) / 2
BODY_H = PAGE_H - MT - MB

# ------------------------------------------------------------------ styles
S = {}
S["title"] = ParagraphStyle("title", fontName="Times-Bold", fontSize=17.5,
                            leading=21, alignment=TA_CENTER, spaceAfter=10)
S["authors"] = ParagraphStyle("authors", fontName="Times-Roman", fontSize=11,
                              leading=14, alignment=TA_CENTER, spaceAfter=2)
S["affil"] = ParagraphStyle("affil", fontName="Times-Italic", fontSize=9,
                            leading=11.5, alignment=TA_CENTER, spaceAfter=8)
S["abstract"] = ParagraphStyle("abstract", fontName="Times-Italic", fontSize=9,
                               leading=11, alignment=TA_JUSTIFY, spaceAfter=5,
                               leftIndent=6, rightIndent=6)
S["keywords"] = ParagraphStyle("keywords", fontName="Times-Italic", fontSize=9,
                               leading=11, alignment=TA_JUSTIFY, spaceAfter=10,
                               leftIndent=6, rightIndent=6)
S["h1"] = ParagraphStyle("h1", fontName="Times-Bold", fontSize=10.5, leading=13,
                         alignment=TA_CENTER, spaceBefore=10, spaceAfter=5,
                         keepWithNext=1)
S["h2"] = ParagraphStyle("h2", fontName="Times-BoldItalic", fontSize=9.8,
                         leading=12, alignment=TA_LEFT, spaceBefore=6,
                         spaceAfter=3, keepWithNext=1)
S["body"] = ParagraphStyle("body", fontName="Times-Roman", fontSize=9.5,
                           leading=11.6, alignment=TA_JUSTIFY, spaceAfter=4,
                           firstLineIndent=14)
S["bodynoindent"] = ParagraphStyle("bodyni", parent=S["body"], firstLineIndent=0)
S["caption"] = ParagraphStyle("caption", fontName="Times-Roman", fontSize=8,
                              leading=9.6, alignment=TA_JUSTIFY, spaceBefore=3,
                              spaceAfter=8)
S["tablecap"] = ParagraphStyle("tablecap", fontName="Times-Roman", fontSize=8,
                               leading=9.6, alignment=TA_CENTER, spaceBefore=6,
                               spaceAfter=3, keepWithNext=1)
S["tcell"] = ParagraphStyle("tcell", fontName="Times-Roman", fontSize=7.6,
                            leading=9.2, alignment=TA_LEFT)
S["tcellc"] = ParagraphStyle("tcellc", parent=S["tcell"], alignment=TA_CENTER)
S["thead"] = ParagraphStyle("thead", fontName="Times-Bold", fontSize=7.6,
                            leading=9.2, alignment=TA_CENTER)
S["ref"] = ParagraphStyle("ref", fontName="Times-Roman", fontSize=8.2,
                          leading=10, alignment=TA_JUSTIFY, leftIndent=16,
                          firstLineIndent=-16, spaceAfter=2)


def cite(*nums):
    """Superscripted clickable citations."""
    return "<super>" + "".join(
        f'<a href="#ref{n}" color="black">[{n}]</a>' for n in nums) + "</super>"


# ------------------------------------------------------------------ equations
def render_eq(name, latex, fontsize=11):
    path = EQ_DIR / f"{name}.png"
    if not path.exists():
        fig = plt.figure(figsize=(3.2, 0.5))
        fig.text(0.5, 0.5, f"${latex}$", fontsize=fontsize, ha="center", va="center")
        fig.savefig(path, dpi=300, bbox_inches="tight", transparent=True, pad_inches=0.02)
        plt.close(fig)
    return path


def eq_block(latex, number, name, maxw=2.9 * inch):
    path = render_eq(name, latex)
    from PIL import Image as PILImage
    w, h = PILImage.open(path).size
    iw = min(maxw, 2.4 * inch)
    ih = iw * h / w
    img = Image(str(path), width=iw, height=ih)
    tab = Table([[img, Paragraph(f"({number})", S["tcellc"])]],
                colWidths=[COL_W - 0.45 * inch, 0.45 * inch])
    tab.setStyle(TableStyle([
        ("ALIGN", (0, 0), (0, 0), "CENTER"),
        ("ALIGN", (1, 0), (1, 0), "RIGHT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 1),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return tab


def fig_block(path, caption, width=COL_W):
    from PIL import Image as PILImage
    w, h = PILImage.open(path).size
    ih = width * h / w
    return KeepTogether([
        Image(str(path), width=width, height=ih),
        Paragraph(caption, S["caption"]),
    ])


def table_block(caption, header, rows, col_widths):
    cap = Paragraph(caption, S["tablecap"])
    data = [[Paragraph(h, S["thead"]) for h in header]]
    for r in rows:
        data.append([Paragraph(c, S["tcell"]) for c in r])
    tab = Table(data, colWidths=col_widths, repeatRows=1)
    tab.setStyle(TableStyle([
        ("LINEABOVE", (0, 0), (-1, 0), 1.1, black),
        ("LINEBELOW", (0, 0), (-1, 0), 0.6, black),
        ("LINEBELOW", (0, -1), (-1, -1), 1.1, black),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
    ]))
    return KeepTogether([cap, tab, Spacer(1, 8)])


# ------------------------------------------------------------------ tables
TABLE_I = dict(
    caption="TABLE I<br/>Hybrid Mamba-Transformer Architectures (Language Modeling)",
    header=["Model", "Year", "Fusion pattern", "Attention role"],
    rows=[
        ["MambaFormer" + cite(9), "2024", "Interleaved Mamba / attention blocks",
         "In-context recall"],
        ["Jamba" + cite(5), "2024", "Interleaved 1:7 attn:Mamba + MoE MLPs",
         "1/8 of layers; 256K ctx, 8x smaller KV cache"],
        ["Zamba" + cite(6), "2024", "Mamba backbone + one shared attention",
         "Single shared global module"],
        ["Samba" + cite(8), "2024", "Mamba + sliding-window attention repeats",
         "Local (windowed) recall; 1M ctx"],
        ["Hymba" + cite(7), "2024", "Parallel attention + SSM heads per layer",
         "High-resolution recall heads"],
        ["Nemotron-H" + cite(10), "2025", "Mamba-2 majority, ~8% attention blocks",
         "Sparse global recall; production scale"],
    ],
    col_widths=[0.72 * inch, 0.38 * inch, 1.32 * inch, 1.01 * inch],
)

TABLE_II = dict(
    caption="TABLE II<br/>EEG Architectures Relevant to Character-Level Decoding",
    header=["Model", "Year", "Sequence core", "Global model.", "Scaling"],
    rows=[
        ["EEGNet" + cite(11), "2018", "Depthwise/separable CNN", "None", "O(T)"],
        ["DeepConvNet" + cite(12), "2017", "Deep CNN", "None", "O(T)"],
        ["ATCNet" + cite(15), "2023", "Attention + TCN", "Attention over TCN feats.", "O(T^2)"],
        ["EEG Conformer" + cite(13), "2023", "Conv module + Transformer enc.", "Full self-attention", "O(T^2)"],
        ["EEG-Deformer" + cite(14), "2024", "Dense coarse-to-fine conv-Transformer",
         "Full self-attention", "O(T^2)"],
        ["EEGMamba" + cite(16), "2024", "Bi-directional Mamba + MoE",
         "Selective SSM (bi-dir.)", "O(T)"],
        ["Brain2Qwerty" + cite(18), "2025", "Conv + sentence Transformer + 9-gram LM",
         "Full self-attention", "O(T^2)"],
        ["<b>Ours (B2Q V3)</b>", "2026", "Conv + hybrid Mamba-2/attention + LoRA LLM",
         "SSM + sparse attention", "mostly O(T)"],
    ],
    col_widths=[0.78 * inch, 0.34 * inch, 1.10 * inch, 0.75 * inch, 0.46 * inch],
)

TABLE_III = dict(
    caption="TABLE III<br/>Language-Model Correction Strategies for Brain-to-Text",
    header=["Strategy", "Example", "Strength", "Weakness"],
    rows=[
        ["Character n-gram fusion", "Brain2Qwerty 9-gram" + cite(18),
         "Cheap, robust", "No semantics; short context"],
        ["LLM rescoring of n-best", "OPT/GPT-2 rescoring" + cite(22),
         "No retraining", "Limited by n-best quality"],
        ["Seq2seq translation", "DeWave (BART)" + cite(24) + "," + cite(25),
         "Open vocabulary", "Entangles alignment and correction"],
        ["<b>End-to-end LoRA LLM (ours)</b>", "B2Q V3" + cite(27) + "," + cite(32),
         "Sees neural word embeds + transcript", "Heavier; hallucination risk"],
    ],
    col_widths=[0.95 * inch, 0.88 * inch, 0.85 * inch, 0.75 * inch],
)

TABLE_IV = dict(
    caption="TABLE IV<br/>Reported Brain-to-Text / Typing Decoding Performance",
    header=["System", "Modality", "Task", "Reported error"],
    rows=[
        ["Crell and Muller-Putz" + cite(29), "EEG", "10-letter classification",
         "75.8% CER*"],
        ["EEGNet baseline" + cite(11) + "," + cite(18), "EEG", "Sentence typing",
         "~76% CER\u2020"],
        ["Brain2Qwerty" + cite(18), "EEG", "Sentence typing", "67 +/- 1.5% CER"],
        ["Brain2Qwerty" + cite(18), "MEG", "Sentence typing",
         "32 +/- 0.6% CER (best 19%)"],
        ["Willett 2021" + cite(21), "Intracortical", "Handwriting",
         "&lt;1% CER (offline, w/ LM)"],
        ["Willett 2023" + cite(22), "Intracortical", "Speech",
         "9.1% WER (50-word); 23.8% (125k)"],
        ["Metzger 2023" + cite(23), "ECoG", "Speech", "15.2% CER, 79 wpm"],
    ],
    col_widths=[0.98 * inch, 0.62 * inch, 0.83 * inch, 1.00 * inch],
)

FIG1_CAP = ("Fig. 1. Proposed Conv + hybrid Mamba-Transformer brain-to-text pipeline "
            "(Brain2Qwerty V3). A convolutional front-end feeds a Nemotron-H-style "
            "stack of Mamba-2 blocks with periodic attention; a CTC head yields "
            "character sequences; a space-based segmenter pools frames into pseudo-"
            "word embeddings aligned to LLM space by a contrastive loss; a "
            "LoRA-adapted LLM generates the corrected sentence.")
FIG2_CAP = ("Fig. 2. (a) Illustrative scaling of the sequence mixer: softmax "
            "attention grows quadratically in sequence length, while the Mamba-2 "
            "SSD recurrence grows linearly" + cite(3) + "," + cite(4) + ". "
            "(b) Published brain-to-text character error rates" + cite(18) + "," +
            cite(29) + ". B2Q = Brain2Qwerty. *As reported in" + cite(18) +
            ". \u2020Derived from the reported 1.14x CER improvement of the full "
            "Brain2Qwerty model over EEGNet on EEG.")


# ------------------------------------------------------------------ build story
def build_story():
    import content as C

    story = [
        Paragraph(C.TITLE, S["title"]),
        Paragraph(C.AUTHORS, S["authors"]),
        Paragraph(C.AFFIL, S["affil"]),
        Paragraph(C.ABSTRACT, S["abstract"]),
        Paragraph(C.KEYWORDS, S["keywords"]),
        FrameBreak(),
    ]
    for blk in C.STORY:
        kind = blk[0]
        if kind == "h1":
            story.append(Paragraph(blk[1], S["h1"]))
        elif kind == "h2":
            story.append(Paragraph(blk[1], S["h2"]))
        elif kind == "p":
            story.append(Paragraph(blk[1], S["body"]))
        elif kind == "eq":
            story.append(eq_block(blk[1], blk[2], blk[3]))
        elif kind == "fig1":
            story.append(fig_block(str(HERE / "fig_architecture.png"), FIG1_CAP))
        elif kind == "fig2":
            story.append(fig_block(str(HERE / "fig_results.png"), FIG2_CAP))
        elif kind == "tableI":
            t = TABLE_I
            story.append(table_block(t["caption"], t["header"], t["rows"], t["col_widths"]))
        elif kind == "tableII":
            t = TABLE_II
            story.append(table_block(t["caption"], t["header"], t["rows"], t["col_widths"]))
        elif kind == "tableIII":
            t = TABLE_III
            story.append(table_block(t["caption"], t["header"], t["rows"], t["col_widths"]))
        elif kind == "tableIV":
            t = TABLE_IV
            story.append(table_block(t["caption"], t["header"], t["rows"], t["col_widths"]))

    story.append(Paragraph("References", S["h1"]))
    for i, r in enumerate(C.REFS, 1):
        story.append(Paragraph(f'<a name="ref{i}"/>[{i}]&nbsp;&nbsp;{r}', S["ref"]))
    return story


def main():
    title_frame = Frame(ML, PAGE_H - MT - 2.45 * inch, PAGE_W - ML - MR,
                        2.45 * inch, id="title", leftPadding=0, rightPadding=0,
                        topPadding=0, bottomPadding=0)
    col1_first = Frame(ML, MB, COL_W, BODY_H - 2.45 * inch, id="c1",
                       leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    col2_first = Frame(ML + COL_W + GUTTER, MB, COL_W, BODY_H - 2.45 * inch,
                       id="c2", leftPadding=0, rightPadding=0, topPadding=0,
                       bottomPadding=0)
    col1 = Frame(ML, MB, COL_W, BODY_H, id="c1", leftPadding=0, rightPadding=0,
                 topPadding=0, bottomPadding=0)
    col2 = Frame(ML + COL_W + GUTTER, MB, COL_W, BODY_H, id="c2",
                 leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Times-Roman", 8)
        canvas.drawCentredString(PAGE_W / 2, 0.5 * inch, str(doc.page))
        canvas.restoreState()

    doc = BaseDocTemplate(str(OUT_PDF), pagesize=letter,
                          leftMargin=ML, rightMargin=MR, topMargin=MT,
                          bottomMargin=MB,
                          title="Hybrid Mamba-Transformer Architectures for "
                                "EEG-to-Text Decoding",
                          author="A. L. Tito")
    doc.addPageTemplates([
        PageTemplate(id="First", frames=[title_frame, col1_first, col2_first],
                     onPage=footer),
        PageTemplate(id="Later", frames=[col1, col2], onPage=footer),
    ])

    story = build_story()
    story.insert(5, NextPageTemplate("Later"))
    doc.build(story)
    print("wrote", OUT_PDF)


if __name__ == "__main__":
    main()
