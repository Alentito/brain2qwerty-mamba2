"""Compute Overall and Per-Subject Word Accuracy, Character Accuracy, WER, CER, SemER.

Identifies the best-performing participant for each model and outputs a comprehensive
comparison table.

Usage:
    python evaluate_accuracy.py [path/to/predictions_test.csv]
    python evaluate_accuracy.py  (auto-scans all results in cache)
"""

import sys
import re
from pathlib import Path
import pandas as pd
import numpy as np


def extract_subject(row) -> str:
    """Extract subject string from sentence_UID or subject column."""
    uid = str(row.get("sentence_UID", ""))
    
    # Try regex matching subject pattern
    match = re.search(r"subject-(S\d+)", uid)
    if match:
        return match.group(1)
        
    match = re.search(r"_(S\d+)_", uid)
    if match:
        return match.group(1)
        
    match = re.search(r"Pinet2024Meg/(S\d+)", uid)
    if match:
        return match.group(1)
    
    # Fallback to subject column
    subj = str(row.get("subject", ""))
    if subj in ["0", "1", "2"]:
        mapping = {"0": "S15", "1": "S16", "2": "S6"}
        return mapping.get(subj, f"Subj_{subj}")
    elif subj:
        return f"Subj_{subj}" if not subj.startswith("S") else subj
    return "Unknown"


def evaluate_file(csv_path: Path):
    if not csv_path.exists():
        print(f"Error: {csv_path} not found.")
        return None
    
    df = pd.read_csv(csv_path)
    df["Extracted_Subject"] = df.apply(extract_subject, axis=1)
    df["Word_Accuracy"] = (1.0 - df["WER"]).clip(lower=0.0) * 100.0
    df["Char_Accuracy"] = (1.0 - df["CER"]).clip(lower=0.0) * 100.0
    if "CTC_CER" in df.columns:
        df["CTC_Char_Accuracy"] = (1.0 - df["CTC_CER"]).clip(lower=0.0) * 100.0
        
    model_name = csv_path.parent.name
    print("\n" + "="*95)
    print(f"🏆 MODEL BENCHMARK REPORT: {model_name}")
    print("="*95)
    
    # 1. Overall Summary
    print(f"Total Test Sentences: {len(df)}")
    print("-" * 95)
    print(f"{'Overall Metric':<40} | {'Mean Value':<15}")
    print("-" * 95)
    print(f"{'Word Accuracy (1 - WER)':<40} | {df['Word_Accuracy'].mean():.2f}%")
    print(f"{'Word Error Rate (WER)':<40} | {df['WER'].mean()*100:.2f}%")
    print(f"{'Character Accuracy (1 - CER)':<40} | {df['Char_Accuracy'].mean():.2f}%")
    print(f"{'Character Error Rate (CER)':<40} | {df['CER'].mean()*100:.2f}%")
    if "CTC_CER" in df.columns:
        print(f"{'CTC Greedy Char Accuracy':<40} | {df['CTC_Char_Accuracy'].mean():.2f}%")
        print(f"{'CTC Greedy CER':<40} | {df['CTC_CER'].mean()*100:.2f}%")
    print(f"{'Semantic Distance (SemER)':<40} | {df['SemER'].mean():.4f}")
    
    # 2. Per-Subject Breakdown Table
    print("\n" + "-"*95)
    print("👥 PER-SUBJECT PERFORMANCE BREAKDOWN")
    print("-" * 95)
    header = f"{'Subject':<12} | {'Sentences':<10} | {'Word Acc':<12} | {'WER':<12} | {'Char Acc':<12} | {'CER':<12} | {'SemER':<10}"
    print(header)
    print("-" * len(header))
    
    subj_stats = []
    for subj, group in df.groupby("Extracted_Subject"):
        w_acc = group["Word_Accuracy"].mean()
        wer = group["WER"].mean() * 100.0
        c_acc = group["Char_Accuracy"].mean()
        cer = group["CER"].mean() * 100.0
        semer = group["SemER"].mean()
        n = len(group)
        subj_stats.append({
            "subject": subj, "n": n, "w_acc": w_acc, "wer": wer,
            "c_acc": c_acc, "cer": cer, "semer": semer
        })
        print(f"{subj:<12} | {n:<10} | {w_acc:6.2f}%     | {wer:6.2f}%    | {c_acc:6.2f}%     | {cer:6.2f}%    | {semer:.4f}")
    print("=" * 95)
    
    # Find best participant
    if subj_stats:
        best_w = max(subj_stats, key=lambda x: x["w_acc"])
        best_c = max(subj_stats, key=lambda x: x["c_acc"])
        print(f"🌟 Best Word Accuracy Participant:      {best_w['subject']} ({best_w['w_acc']:.2f}% Word Acc / {best_w['wer']:.2f}% WER)")
        print(f"🌟 Best Character Accuracy Participant: {best_c['subject']} ({best_c['c_acc']:.2f}% Char Acc / {best_c['cer']:.2f}% CER)")
        print("=" * 95 + "\n")
        
    return {"name": model_name, "overall_w_acc": df['Word_Accuracy'].mean(), "subj_stats": subj_stats}


def scan_all():
    base_dir = Path.home() / "sharedscratch/B2Q/cache_v1mamba/results"
    if not base_dir.exists():
        base_dir = Path(".cache/b2q_v1mamba/results")
    if not base_dir.exists():
        print("No results directory found.")
        return
        
    csvs = sorted(list(base_dir.glob("*/predictions_test.csv")))
    if not csvs:
        print(f"No predictions_test.csv files found in {base_dir}")
        return
        
    print(f"Found {len(csvs)} evaluated models. Generating Per-Participant Breakdown...\n")
    for csv_file in csvs:
        evaluate_file(csv_file)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        evaluate_file(Path(sys.argv[1]))
    else:
        scan_all()
