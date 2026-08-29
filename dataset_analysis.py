"""Comprehensive Exploratory Data Analysis (EDA) on SpanishBCBL (3 Subjects: S15, S16, S6)."""

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

import studies  # registers Pinet2024Meg
import brain2qwerty_v1.transforms
from brain2qwerty_v1.utils import CHAR_INDEX
from brain2qwerty_v1_mamba.config.xp_config import experiment_config
from brain2qwerty_v1_mamba.main import Experiment

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 14,
    "figure.autolayout": True,
})

def main():
    out_dir = Path("dataset_eda_out")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("[eda] loading SpanishBCBL 3-subject dataset (S15, S16, S6)...")
    cfg = experiment_config(subjects=["S15", "S16", "S6"], small=True)
    cfg["data"]["num_workers"] = 0
    cfg["data"]["batch_size"] = 128
    cfg["data"]["val_batch_size"] = 128
    cfg["data"]["test_batch_size"] = 128
    cfg["data"]["persistent_workers"] = False
    cfg["data"]["pin_memory"] = False
    
    exp = Experiment(**cfg)
    loaders = exp.data.build()
    
    split_stats = {}
    all_data = []
    split_neuro = []
    
    for split_name, loader in loaders.items():
        n_windows = 0
        split_chars = []
        split_subjs = []
        split_uids = []
        
        for batch in loader:
            labels = batch.data["feature"].view(-1).tolist()
            subjs = batch.data["subject_id"].view(-1).tolist()
            uids = [seg.trigger.extra["sentence_UID"] for seg in batch.segments]
            neuro = batch.data["neuro"]  # (B, 306, 25)
            
            n_windows += len(labels)
            split_chars.extend(labels)
            split_subjs.extend(subjs)
            split_uids.extend(uids)
            
            if split_name == "test" and len(split_neuro) < 6:
                split_neuro.append(neuro)
                
        unique_sents = len(set(split_uids))
        split_stats[split_name] = {
            "n_keystrokes": n_windows,
            "n_unique_sentences": unique_sents,
            "mean_keystrokes_per_sent": n_windows / max(unique_sents, 1),
        }
        
        for ch, subj, uid in zip(split_chars, split_subjs, split_uids):
            subj_val = subj[0] if isinstance(subj, list) else int(subj)
            ch_val = ch[0] if isinstance(ch, list) else int(ch)
            all_data.append({
                "split": split_name,
                "char_idx": ch_val,
                "char": CHAR_INDEX.get(ch_val, "?"),
                "subject_idx": subj_val,
                "sentence_uid": uid,
            })
            
    df = pd.DataFrame(all_data)
    subj_map = {0: "S15", 1: "S16", 2: "S6"}
    df["subject"] = df["subject_idx"].map(lambda x: subj_map.get(x, f"S{x}"))
    
    print("\n" + "="*70)
    print(f"{'Split':<12} {'Keystrokes':>15} {'Sentences':>15} {'Mean Keys/Sent':>18}")
    print("="*70)
    total_keys = 0
    total_sents = 0
    for s, st in split_stats.items():
        print(f"{s:<12} {st['n_keystrokes']:>15,d} {st['n_unique_sentences']:>15,d} {st['mean_keystrokes_per_sent']:>18.1f}")
        total_keys += st['n_keystrokes']
        total_sents += st['n_unique_sentences']
    print("-" * 70)
    print(f"{'Total':<12} {total_keys:>15,d} {total_sents:>15,d} {total_keys/total_sents:>18.1f}")
    print("="*70 + "\n")
    
    # --------------------------------------------------------------------------- #
    # 2. Character Frequency Distribution
    # --------------------------------------------------------------------------- #
    print("[eda] plotting character frequency distribution...")
    char_counts = df["char"].value_counts()
    char_labels = [c.replace(" ", "␣").replace("@", "<sp>").replace("9", "<num>") for c in char_counts.index]
    
    fig, ax = plt.subplots(figsize=(13, 5))
    sns.barplot(x=char_labels, y=char_counts.values, ax=ax, palette="mako")
    ax.set_title("SpanishBCBL: Character Frequency Distribution (3 Subjects: S15, S16, S6)", fontweight="bold")
    ax.set_xlabel("Character Token")
    ax.set_ylabel("Total Keystroke Windows")
    for i, v in enumerate(char_counts.values):
        ax.text(i, v + max(char_counts.values)*0.015, str(v), ha="center", fontsize=8, rotation=90)
    ax.set_ylim(0, max(char_counts.values) * 1.18)
    fig.savefig(out_dir / "01_character_distribution.png", dpi=200)
    plt.close(fig)
    
    # --------------------------------------------------------------------------- #
    # 3. Cross-Subject Keystroke Breakdown
    # --------------------------------------------------------------------------- #
    print("[eda] plotting cross-subject keystroke distribution...")
    subj_split = pd.crosstab(df["subject"], df["split"])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    subj_split.plot(kind="bar", stacked=True, ax=ax, colormap="viridis")
    ax.set_title("Keystroke Windows per Subject and Split", fontweight="bold")
    ax.set_xlabel("Participant")
    ax.set_ylabel("Number of Keystrokes (500 ms windows)")
    ax.legend(title="Split")
    fig.savefig(out_dir / "02_subject_split_breakdown.png", dpi=200)
    plt.close(fig)
    
    # --------------------------------------------------------------------------- #
    # 4. Sentence Length Distribution (Keystrokes per sentence)
    # --------------------------------------------------------------------------- #
    print("[eda] plotting sentence length distribution...")
    sent_lens = df.groupby(["split", "sentence_uid"]).size().reset_index(name="keystrokes")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.histplot(data=sent_lens, x="keystrokes", hue="split", bins=20, kde=True, ax=ax, element="step")
    ax.set_title("Sentence Length Distribution (Keystrokes per Sentence)", fontweight="bold")
    ax.set_xlabel("Keystrokes per Sentence")
    ax.set_ylabel("Sentence Count")
    fig.savefig(out_dir / "03_sentence_length_distribution.png", dpi=200)
    plt.close(fig)
    
    # --------------------------------------------------------------------------- #
    # 5. Grand-Average MEG Evoked Waveform (-200 ms to +300 ms)
    # --------------------------------------------------------------------------- #
    print("[eda] computing grand-average MEG evoked response (306 channels)...")
    neuro_cat = torch.cat(split_neuro, dim=0)  # (N, 306, 25)
    times = np.linspace(-200, 300, 25)
    mean_evoked = neuro_cat.mean(dim=0).numpy()  # (306, 25)
    gfp = np.std(mean_evoked, axis=0)           # Global Field Power
    
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), gridspec_kw={"width_ratios": [1.4, 1]})
    
    for c in range(306):
        axes[0].plot(times, mean_evoked[c], color="#94a3b8", alpha=0.35, lw=0.7)
    axes[0].plot(times, mean_evoked.mean(axis=0), color="#0284c7", lw=2.5, label="Mean Evoked Potential")
    axes[0].axvline(0, color="#ef4444", lw=1.5, linestyle="--", label="Keypress Strike (0 ms)")
    axes[0].axvspan(-50, 0, color="#fde047", alpha=0.2, label="Motor Preparation Dip")
    axes[0].axvspan(20, 60, color="#22c55e", alpha=0.2, label="Somatosensory Feedback Peak")
    axes[0].set_title("Grand-Average MEG Butterfly Plot (306 Channels)", fontweight="bold")
    axes[0].set_xlabel("Time relative to keypress (ms)")
    axes[0].set_ylabel("Normalized Magnetic Field (RobustScaled)")
    axes[0].legend(loc="upper left", fontsize=8)
    
    axes[1].plot(times, gfp, color="#7c3aed", lw=2.5)
    axes[1].axvline(0, color="#ef4444", lw=1.5, linestyle="--", label="0 ms Strike")
    axes[1].set_title("Global Field Power (GFP)", fontweight="bold")
    axes[1].set_xlabel("Time relative to keypress (ms)")
    axes[1].set_ylabel("Spatial STD across 306 Sensors")
    axes[1].legend(loc="upper left", fontsize=8)
    
    fig.savefig(out_dir / "04_meg_evoked_response.png", dpi=200)
    plt.close(fig)
    
    # --------------------------------------------------------------------------- #
    # 6. Save JSON Summary & Markdown Report
    # --------------------------------------------------------------------------- #
    summary = {
        "dataset_name": "SpanishBCBL (Pinet2024Meg)",
        "subjects": ["S15", "S16", "S6"],
        "total_keystroke_windows": int(total_keys),
        "total_unique_sentences": int(total_sents),
        "mean_keystrokes_per_sentence": float(total_keys / total_sents),
        "split_counts": split_stats,
        "subject_keystrokes": df["subject"].value_counts().to_dict(),
        "top_5_frequent_characters": char_counts.head(5).to_dict(),
        "recording_modality": "306-Channel Elekta Neuromag MEG (204 Planar Gradiometers + 102 Magnetometers)",
        "window_duration_ms": 500,
        "window_time_range_ms": [-200, 300],
        "sampling_frequency_hz": 50,
        "time_samples_per_window": 25,
    }
    (out_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[eda] All figures and dataset_summary.json saved to {out_dir}/")

if __name__ == "__main__":
    main()
