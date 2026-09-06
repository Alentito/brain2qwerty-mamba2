"""Statistical Aggregation & Significance Testing across Multi-Seed Runs.

Computes:
1. Mean +/- Std Dev across seeds for:
   - Word Error Rate (WER) & Word Accuracy (1 - WER)
   - Character Error Rate (CER) & Character Accuracy (1 - CER)
   - CTC Character Error Rate (CTC CER)
   - Semantic Error Rate (SemER)
2. Statistical Significance:
   - Paired Student's t-test and Wilcoxon signed-rank test comparing Mamba vs Conformer.
3. Formats an IEEE/Dissertation ready LaTeX & Markdown table.

Usage:
    python aggregate_multi_seed_results.py [--dir /path/to/results]
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats


def evaluate_single_csv(p: Path) -> dict:
    df = pd.read_csv(p)
    df["Word_Acc"] = (1.0 - df["WER"]).clip(lower=0.0) * 100.0
    df["Char_Acc"] = (1.0 - df["CER"]).clip(lower=0.0) * 100.0
    
    return {
        "wer": float(df["WER"].mean() * 100.0),
        "word_acc": float(df["Word_Acc"].mean()),
        "cer": float(df["CER"].mean() * 100.0),
        "char_acc": float(df["Char_Acc"].mean()),
        "ctc_cer": float(df["CTC_CER"].mean() * 100.0) if "CTC_CER" in df.columns else np.nan,
        "semer": float(df["SemER"].mean()),
        "df": df,
    }


def analyze_multi_seed(results_dir: Path):
    if not results_dir.exists():
        print(f"Directory not found: {results_dir}")
        return
        
    models = {
        "Conformer": "conformer",
        "BiMamba-2 + Gated MLP": "mamba_mlp",
        "Mamba-3 Stabilized Hybrid": "mamba3_hybrid_stabilized",
    }
    
    summary_data = {}
    
    for display_name, core_key in models.items():
        csv_files = list(results_dir.glob(f"v3-{core_key}-*/predictions_test.csv"))
        if not csv_files:
            csv_files = list(results_dir.glob(f"*{core_key}*/predictions_test.csv"))
            
        print(f"Found {len(csv_files)} runs for {display_name}")
        
        runs = []
        for f in sorted(csv_files):
            try:
                res = evaluate_single_csv(f)
                res["file"] = str(f)
                runs.append(res)
            except Exception as e:
                print(f"  Error reading {f}: {e}")
                
        if runs:
            summary_data[display_name] = {
                "runs": runs,
                "n_runs": len(runs),
                "wer_mean": np.mean([r["wer"] for r in runs]),
                "wer_std": np.std([r["wer"] for r in runs]) if len(runs) > 1 else 0.0,
                "w_acc_mean": np.mean([r["word_acc"] for r in runs]),
                "w_acc_std": np.std([r["word_acc"] for r in runs]) if len(runs) > 1 else 0.0,
                "cer_mean": np.mean([r["cer"] for r in runs]),
                "cer_std": np.std([r["cer"] for r in runs]) if len(runs) > 1 else 0.0,
                "c_acc_mean": np.mean([r["char_acc"] for r in runs]),
                "c_acc_std": np.std([r["char_acc"] for r in runs]) if len(runs) > 1 else 0.0,
                "ctc_mean": np.nanmean([r["ctc_cer"] for r in runs]),
                "ctc_std": np.nanstd([r["ctc_cer"] for r in runs]) if len(runs) > 1 else 0.0,
                "semer_mean": np.mean([r["semer"] for r in runs]),
                "semer_std": np.std([r["semer"] for r in runs]) if len(runs) > 1 else 0.0,
            }
            
    # Print Markdown Summary Table
    print("\n" + "=" * 110)
    print("📊 MULTI-SEED STATISTICAL BENCHMARK SUMMARY (Mean +/- Std Dev)")
    print("=" * 110)
    header = f"{'Architecture':<28} | {'Runs':<5} | {'WER (%)':<16} | {'Word Acc (%)':<16} | {'CER (%)':<16} | {'SemER':<16}"
    print(header)
    print("-" * len(header))
    
    for name, s in summary_data.items():
        wer_str = f"{s['wer_mean']:.2f} +/- {s['wer_std']:.2f}"
        wacc_str = f"{s['w_acc_mean']:.2f} +/- {s['w_acc_std']:.2f}"
        cer_str = f"{s['cer_mean']:.2f} +/- {s['cer_std']:.2f}"
        semer_str = f"{s['semer_mean']:.4f} +/- {s['semer_std']:.4f}"
        print(f"{name:<28} | {s['n_runs']:<5} | {wer_str:<16} | {wacc_str:<16} | {cer_str:<16} | {semer_str:<16}")
    print("=" * 110 + "\n")
    
    # Statistical Hypothesis Testing (if Conformer and Mamba runs exist)
    if "Conformer" in summary_data and len(summary_data["Conformer"]["runs"]) > 0:
        conf_df = summary_data["Conformer"]["runs"][0]["df"]
        
        for m_name in ["BiMamba-2 + Gated MLP", "Mamba-3 Stabilized Hybrid"]:
            if m_name in summary_data and len(summary_data[m_name]["runs"]) > 0:
                m_df = summary_data[m_name]["runs"][0]["df"]
                
                # Paired tests over sentence-level errors
                t_wer, p_wer = stats.ttest_rel(conf_df["WER"], m_df["WER"])
                w_wer, wp_wer = stats.wilcoxon(conf_df["WER"], m_df["WER"])
                
                t_cer, p_cer = stats.ttest_rel(conf_df["CER"], m_df["CER"])
                w_cer, wp_cer = stats.wilcoxon(conf_df["CER"], m_df["CER"])
                
                print(f"🔬 Statistical Significance: {m_name} vs. Conformer")
                print(f"   • Word Error Rate (WER) paired t-test:       t = {t_wer:.3f}, p = {p_wer:.4e} {'(Significant ***)' if p_wer < 0.001 else ''}")
                print(f"   • Word Error Rate (WER) Wilcoxon test:      W = {w_wer:.1f}, p = {wp_wer:.4e} {'(Significant ***)' if wp_wer < 0.001 else ''}")
                print(f"   • Character Error Rate (CER) paired t-test:  t = {t_cer:.3f}, p = {p_cer:.4e} {'(Significant ***)' if p_cer < 0.001 else ''}")
                print(f"   • Character Error Rate (CER) Wilcoxon test: W = {w_cer:.1f}, p = {wp_cer:.4e} {'(Significant ***)' if wp_cer < 0.001 else ''}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default=None)
    args = parser.parse_args()
    
    base_dir = Path(args.dir) if args.dir else Path.home() / "sharedscratch/B2Q/cache_v1mamba/results"
    if not base_dir.exists():
        base_dir = Path(".cache/b2q_v1mamba/results")
    analyze_multi_seed(base_dir)
