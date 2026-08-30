"""Compute Overall and Per-Subject Word Accuracy, Character Accuracy, WER, CER, SemER."""

import sys
import re
from pathlib import Path
import pandas as pd
import numpy as np

def extract_subject(row) -> str:
    """Extract subject string from sentence_UID or subject column."""
    uid = str(row.get("sentence_UID", ""))
    match = re.search(r"_(S\d+)_", uid)
    if match:
        return match.group(1)
    
    # Fallback to subject column
    subj = str(row.get("subject", ""))
    if subj in ["0", "1", "2"]:
        mapping = {"0": "S15", "1": "S16", "2": "S6"}
        return mapping.get(subj, f"Subj_{subj}")
    return subj or "Unknown"

def evaluate(csv_path: str):
    p = Path(csv_path)
    if not p.exists():
        print(f"Error: {csv_path} not found.")
        return
    
    df = pd.read_csv(p)
    df["Extracted_Subject"] = df.apply(extract_subject, axis=1)
    df["Word_Accuracy"] = (1.0 - df["WER"]).clip(lower=0.0) * 100.0
    df["Char_Accuracy"] = (1.0 - df["CER"]).clip(lower=0.0) * 100.0
    if "CTC_CER" in df.columns:
        df["CTC_Char_Accuracy"] = (1.0 - df["CTC_CER"]).clip(lower=0.0) * 100.0
        
    model_name = p.parent.name
    print("\n" + "="*85)
    print(f"🏆 BENCHMARK EVALUATION REPORT: {model_name}")
    print("="*85)
    
    # 1. Overall Summary
    print(f"Total Test Sentences: {len(df)}")
    print("-" * 85)
    print(f"{'Metric':<35} | {'Mean Value':<15}")
    print("-" * 85)
    print(f"{'Word Accuracy (1 - WER)':<35} | {df['Word_Accuracy'].mean():.2f}%")
    print(f"{'Word Error Rate (WER)':<35} | {df['WER'].mean()*100:.2f}%")
    print(f"{'Character Accuracy (1 - CER)':<35} | {df['Char_Accuracy'].mean():.2f}%")
    print(f"{'Character Error Rate (CER)':<35} | {df['CER'].mean()*100:.2f}%")
    if "CTC_CER" in df.columns:
        print(f"{'CTC Greedy Char Accuracy':<35} | {df['CTC_Char_Accuracy'].mean():.2f}%")
        print(f"{'CTC Greedy CER':<35} | {df['CTC_CER'].mean()*100:.2f}%")
    print(f"{'Semantic Error Rate (SemER)':<35} | {df['SemER'].mean():.4f}")
    
    # 2. Per-Subject Breakdown Table
    print("\n" + "="*85)
    print("👥 PER-SUBJECT PERFORMANCE BREAKDOWN")
    print("="*85)
    header = f"{'Subject':<10} | {'Sentences':<10} | {'Word Acc':<10} | {'WER':<10} | {'Char Acc':<10} | {'CER':<10} | {'SemER':<10}"
    print(header)
    print("-" * len(header))
    
    for subj, sub_df in df.groupby("Extracted_Subject"):
        s_count = len(sub_df)
        w_acc = sub_df["Word_Accuracy"].mean()
        wer = sub_df["WER"].mean() * 100
        c_acc = sub_df["Char_Accuracy"].mean()
        cer = sub_df["CER"].mean() * 100
        semer = sub_df["SemER"].mean()
        print(f"{subj:<10} | {s_count:<10} | {w_acc:>8.2f}% | {wer:>8.2f}% | {c_acc:>8.2f}% | {cer:>8.2f}% | {semer:>10.4f}")
    print("="*85 + "\n")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            evaluate(arg)
    else:
        # Default scan results folder
        base = Path.home() / "sharedscratch/B2Q/cache_v1mamba/results"
        if not base.exists():
            base = Path(".cache/b2q_v1mamba/results")
        for f in sorted(base.glob("**/predictions_test.csv")):
            evaluate(str(f))
