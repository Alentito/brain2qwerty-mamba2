"""Compute Word Accuracy, Character Accuracy, and Exact Match Rate from predictions_test.csv."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

def evaluate(csv_path: str):
    p = Path(csv_path)
    if not p.exists():
        print(f"Error: {csv_path} not found.")
        return
    
    df = pd.read_csv(p)
    
    # 1. Standard Accuracy = 1 - Error Rate
    df["Word_Accuracy"] = (1.0 - df["WER"]).clip(lower=0.0) * 100.0
    df["Char_Accuracy"] = (1.0 - df["CER"]).clip(lower=0.0) * 100.0
    if "CTC_CER" in df.columns:
        df["CTC_Char_Accuracy"] = (1.0 - df["CTC_CER"]).clip(lower=0.0) * 100.0
        
    # 2. Exact Word Match Rate
    total_words = 0
    matched_words = 0
    exact_sentences = 0
    
    for _, row in df.iterrows():
        true_words = str(row["true_text"]).strip().split()
        pred_words = str(row["pred_text"]).strip().split()
        
        total_words += len(true_words)
        # Count words that appear in the exact position or set
        for tw, pw in zip(true_words, pred_words):
            if tw == pw:
                matched_words += 1
                
        if str(row["true_text"]).strip() == str(row["pred_text"]).strip():
            exact_sentences += 1
            
    exact_word_rate = (matched_words / max(total_words, 1)) * 100.0
    exact_sent_rate = (exact_sentences / len(df)) * 100.0
    
    print("\n" + "="*70)
    print(f"📊 Accuracy Evaluation Report: {p.parent.name}")
    print("="*70)
    print(f"Total Test Sentences:              {len(df):,d}")
    print(f"Total Test Words:                  {total_words:,d}")
    print("-" * 70)
    print(f"Mean Word Accuracy (1 - WER):       {df['Word_Accuracy'].mean():.2f}%")
    print(f"Mean Character Accuracy (1 - CER):  {df['Char_Accuracy'].mean():.2f}%")
    if "CTC_CER" in df.columns:
        print(f"Mean CTC Greedy Char Accuracy:     {df['CTC_Char_Accuracy'].mean():.2f}%")
    print(f"Exact Word Position Match Rate:    {exact_word_rate:.2f}%")
    print(f"Exact Sentence Match Rate:         {exact_sent_rate:.2f}%")
    print(f"Mean Semantic Error (SemER):       {df['SemER'].mean():.4f}")
    print("="*70 + "\n")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            evaluate(arg)
    else:
        # Default check known result folders
        base = Path.home() / "sharedscratch/B2Q/cache_v1mamba/results"
        if not base.exists():
            base = Path(".cache/b2q_v1mamba/results")
        for f in base.glob("**/predictions_test.csv"):
            evaluate(str(f))
