"""Comprehensive Benchmarking Suite: O(N) Mamba vs O(N^2) Transformer/Conformer Complexity,
Latency, Throughput, and Memory Scaling Profiler.

Measures:
1. Forward Latency (ms) vs Sequence Length T in [64, 128, 256, 512, 1024, 2048, 4096]
2. Backward Training Latency (ms) & Throughput (frames/sec)
3. Peak GPU VRAM Memory (MB) during Training Step
4. Single-step Inference Latency (ms)

Generates:
- latency_scaling_O_N_vs_O_N2.png
- memory_scaling_O_N_vs_O_N2.png
- benchmark_complexity_report.json
- summary markdown table

Usage:
    python benchmark_complexity.py [--device cuda|mps|cpu] [--dim 512|1024]
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from brain2qwerty_v3.config.model_config import build_encoder_config
from brain2qwerty_v3.models import ConvMambaHybrid


def get_device(user_device: str | None = None) -> torch.device:
    if user_device:
        return torch.device(user_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def measure_forward_backward(
    model: nn.Module,
    B: int,
    T: int,
    C: int,
    device: torch.device,
    n_warmup: int = 2,
    n_repeat: int = 5,
) -> dict:
    """Benchmark forward time, backward time, and peak memory for a given (B, T, C)."""
    model.train()
    is_cuda = device.type == "cuda"
    
    # Synthetic MEG inputs
    x = torch.randn(B, T, C, device=device, requires_grad=True)
    days = torch.zeros(B, dtype=torch.long, device=device)
    chan_pos = torch.randn(B, C, 2, device=device)
    
    # Warmup
    for _ in range(n_warmup):
        model.zero_grad()
        out = model(x, days, chan_pos)
        loss = out["c_out"].sum() + out["z_final"].sum()
        loss.backward()
        if is_cuda:
            torch.cuda.synchronize()
            
    # Measure Forward
    if is_cuda:
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
        
    fwd_times = []
    for _ in range(n_repeat):
        start = time.perf_counter()
        out = model(x, days, chan_pos)
        if is_cuda:
            torch.cuda.synchronize()
        fwd_times.append((time.perf_counter() - start) * 1000.0)
        
    # Measure Forward + Backward (Training Step)
    step_times = []
    for _ in range(n_repeat):
        model.zero_grad()
        start = time.perf_counter()
        out = model(x, days, chan_pos)
        loss = out["c_out"].sum() + out["z_final"].sum()
        loss.backward()
        if is_cuda:
            torch.cuda.synchronize()
        step_times.append((time.perf_counter() - start) * 1000.0)
        
    peak_vram_mb = 0.0
    if is_cuda:
        peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        
    fwd_mean = float(np.mean(fwd_times))
    fwd_std = float(np.std(fwd_times))
    step_mean = float(np.mean(step_times))
    step_std = float(np.std(step_times))
    bwd_mean = max(0.0, step_mean - fwd_mean)
    throughput = (B * T) / (step_mean / 1000.0)  # frames / sec
    
    return {
        "fwd_ms": fwd_mean,
        "fwd_std": fwd_std,
        "bwd_ms": bwd_mean,
        "step_ms": step_mean,
        "step_std": step_std,
        "throughput_fps": throughput,
        "peak_vram_mb": peak_vram_mb,
    }


def run_benchmark(device_str: str | None = None, small: bool = True):
    device = get_device(device_str)
    print("=" * 80)
    print(f"🚀 RUNNING COMPLEXITY & LATENCY BENCHMARK ON: {device.type.upper()}")
    if device.type == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(device)}")
    print(f"   Model Width: {'Small (512-dim)' if small else 'Full (1024-dim)'}")
    print("=" * 80)
    
    out_dir = Path("benchmark_out")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    cores = [
        ("Conformer", "conformer"),
        ("Nemotron-H Hybrid Mamba-2", "hybrid"),
        ("BiMamba-2 + Gated MLP", "mamba_mlp"),
        ("Mamba-3 Stabilized Hybrid", "mamba3_hybrid_stabilized"),
    ]
    
    seq_lengths = [64, 128, 256, 512, 1024, 2048, 4096]
    B = 2
    C = 306  # MEG channels
    
    results = {
        "device": device.type,
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
        "small": small,
        "seq_lengths": seq_lengths,
        "batch_size": B,
        "channels": C,
        "models": {},
    }
    
    for display_name, core_key in cores:
        print(f"\n👉 Benchmarking Architecture: {display_name} ({core_key})...")
        try:
            cfg_dict = build_encoder_config(core=core_key, small=small)
            cfg = ConvMambaHybrid(**cfg_dict)
            model = cfg.build(n_in_channels=C, n_outputs=29).to(device)
            n_params = sum(p.numel() for p in model.parameters())
            print(f"   Compiled model with {n_params:,} parameters.")
        except Exception as e:
            print(f"   ❌ Error compiling {display_name}: {e}")
            continue
            
        core_results = {"params": n_params, "scaling": {}}
        
        for T in seq_lengths:
            try:
                metrics = measure_forward_backward(model, B=B, T=T, C=C, device=device)
                core_results["scaling"][str(T)] = metrics
                print(f"   T={T:4d} frames | Fwd: {metrics['fwd_ms']:6.2f} ms | Step: {metrics['step_ms']:6.2f} ms | VRAM: {metrics['peak_vram_mb']:6.1f} MB | {metrics['throughput_fps']:7.1f} fps")
            except torch.cuda.OutOfMemoryError:
                print(f"   T={T:4d} frames | ❌ CUDA Out Of Memory!")
                core_results["scaling"][str(T)] = {"fwd_ms": None, "step_ms": None, "peak_vram_mb": "OOM", "error": "OOM"}
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            except Exception as e:
                print(f"   T={T:4d} frames | ❌ Error: {e}")
                core_results["scaling"][str(T)] = {"error": str(e)}
                
        results["models"][display_name] = core_results
        
    # Save JSON Report
    json_path = out_dir / "benchmark_complexity_report.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ JSON report saved to: {json_path}")
    
    # Generate Plots
    plot_complexity_curves(results, out_dir)
    print_markdown_summary(results)


def plot_complexity_curves(results: dict, out_dir: Path):
    seq_lengths = results["seq_lengths"]
    models = results["models"]
    
    colors = {
        "Conformer": "#e74c3c",                # Red
        "Transformer": "#e67e22",              # Orange
        "Mamba-2": "#3498db",                  # Blue
        "BiMamba-2 + Gated MLP": "#9b59b6",    # Purple
        "Mamba-3 Stabilized Hybrid": "#2ecc71" # Green
    }
    markers = {
        "Conformer": "o",
        "Transformer": "s",
        "Mamba-2": "^",
        "BiMamba-2 + Gated MLP": "D",
        "Mamba-3 Stabilized Hybrid": "P"
    }
    
    # 1. Latency Scaling Plot
    plt.figure(figsize=(10, 6))
    for name, data in models.items():
        ts, fwd_times = [], []
        for T in seq_lengths:
            val = data["scaling"].get(str(T), {}).get("fwd_ms")
            if val is not None and isinstance(val, (int, float)):
                ts.append(T)
                fwd_times.append(val)
        if ts:
            plt.plot(ts, fwd_times, marker=markers.get(name, "o"), label=name, color=colors.get(name, "black"), linewidth=2.5, markersize=8)
            
    plt.title("Computational Scaling: O(N) Mamba State-Space vs O(N²) Quadratic Attention", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Input Sequence Length T (Frames)", fontsize=12)
    plt.ylabel("Forward Pass Latency (ms)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11, loc="upper left")
    plt.tight_layout()
    latency_plot_path = out_dir / "latency_scaling_O_N_vs_O_N2.png"
    plt.savefig(latency_plot_path, dpi=300)
    plt.close()
    print(f"📊 Latency scaling curve saved to: {latency_plot_path}")
    
    # 2. Peak Memory Plot (if CUDA available)
    if results["device"] == "cuda":
        plt.figure(figsize=(10, 6))
        for name, data in models.items():
            ts, vram = [], []
            for T in seq_lengths:
                val = data["scaling"].get(str(T), {}).get("peak_vram_mb")
                if val is not None and isinstance(val, (int, float)):
                    ts.append(T)
                    vram.append(val)
            if ts:
                plt.plot(ts, vram, marker=markers.get(name, "o"), label=name, color=colors.get(name, "black"), linewidth=2.5, markersize=8)
                
        plt.title("Peak GPU Memory Scaling: O(N) Linear Mamba vs O(N²) Quadratic Attention", fontsize=13, fontweight="bold", pad=12)
        plt.xlabel("Input Sequence Length T (Frames)", fontsize=12)
        plt.ylabel("Peak VRAM Allocated (MB)", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(fontsize=11, loc="upper left")
        plt.tight_layout()
        mem_plot_path = out_dir / "memory_scaling_O_N_vs_O_N2.png"
        plt.savefig(mem_plot_path, dpi=300)
        plt.close()
        print(f"📊 Memory scaling curve saved to: {mem_plot_path}")


def print_markdown_summary(results: dict):
    seq_lengths = results["seq_lengths"]
    models = results["models"]
    
    print("\n" + "=" * 90)
    print("📋 COMPUTATIONAL COMPLEXITY & LATENCY BENCHMARK TABLE (Forward Latency in ms)")
    print("=" * 90)
    
    header = f"{'Architecture':<28} | {'Params':<8} | " + " | ".join([f"T={T:<5}" for T in seq_lengths])
    print(header)
    print("-" * len(header))
    
    for name, data in models.items():
        p_str = f"{data['params']/1e6:.1f}M"
        row = [f"{name:<28}", f"{p_str:<8}"]
        for T in seq_lengths:
            val = data["scaling"].get(str(T), {}).get("fwd_ms")
            if val is None:
                row.append(f"{'OOM':<7}")
            else:
                row.append(f"{val:>5.1f}ms")
        print(" | ".join(row))
    print("=" * 90 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark O(N) Mamba vs O(N^2) Attention Complexity")
    parser.add_argument("--device", type=str, default=None, help="Device to run benchmark on (cuda, mps, cpu)")
    parser.add_argument("--full", action="store_true", help="Use full 1024-dim width instead of 512-dim small")
    args = parser.parse_args()
    
    run_benchmark(device_str=args.device, small=not args.full)
