# Brain2Qwerty: 8 Architectural Ablations Flowcharts

This document provides visual Mermaid flowcharts capturing the end-to-end architectural differences across all 8 experimental ablation arms evaluated on the SpanishBCBL 3-subject dataset ($S15, S16, S6$).

---

## 1. Master Architectural Hierarchy & Ablation Decision Tree

```mermaid
flowchart TD
    RAW["Raw MEG Recording (306 Channels @ 1000 Hz)"] --> PREPROC["Preprocessing: Bandpass 0.5-45 Hz, Downsample to 50 Hz"]
    PREPROC --> WIN["Keystroke Windows [-200 ms, +300 ms] (25 time samples x 306 channels)"]
    WIN --> ENC["SimpleConvTimeAgg Front-End<br/>• Subject-Specific 2D-Fourier Channel Merger (270 virtual ch)<br/>• 8-Layer 1D Temporal Dilated Conv + Attn Pooling<br/>• Output: Keystroke Embedding E_t ∈ ℝ^512"]
    
    ENC --> CORE_CHOICE{"Sequence Core Architecture<br/>(4 Pre-Norm Blocks, Width 512)"}
    
    %% FAMILY 1: TRANSFORMER
    CORE_CHOICE -->|Full Softmax Attention| FAM_TF["Transformer Family<br/>(Bidirectional Self-Attention + ALiBi)"]
    FAM_TF --> ARM1["<b>[Arm 1] Transformer Baseline (R1)</b><br/>• LR = 5e-5, WD = 1e-4, Clip = None<br/>• <b>Test CER: 0.285</b> | Test Loss: 1.992"]
    FAM_TF --> ARM8["<b>[Arm 8] Transformer Control (Task 5)</b><br/>• LR = 1e-4, WD = 1e-4, Clip = None<br/>• <b>Test CER: 0.246</b> (Best Overall)"]

    %% FAMILY 2: BIMAMBA-2
    CORE_CHOICE -->|Bidirectional SSD Recurrence| FAM_M2["BiMamba-2 Family<br/>(Forward + Backward SSD Mixers)"]
    FAM_M2 --> ARM2["<b>[Arm 2] Mamba-2 Baseline (R1)</b><br/>• LR = 5e-5, WD = 1e-4, Clip = None<br/>• <b>Test CER: 0.412</b> (Underfit)"]
    FAM_M2 --> ARM3["<b>[Arm 3] Mamba-2 Mild LR (Task 0)</b><br/>• LR = 1e-4, WD = 1e-4, Clip = None<br/>• <b>Test CER: 0.394</b>"]
    FAM_M2 --> ARM4["<b>[Arm 4] Mamba-2 Aggressive LR (Task 1)</b><br/>• LR = 3e-4, WD = 1e-4, Clip = None<br/>• <b>Test CER: 0.304</b> (Large Jump)"]
    FAM_M2 --> ARM5["<b>[Arm 5] Mamba-2 Full Recipe (Task 2)</b><br/>• LR = 1e-4, WD = 0.1, Clip = 1.0<br/>• <b>Test CER: 0.374</b> (Reg. Boost)"]

    %% FAMILY 3: BIMAMBA-3
    CORE_CHOICE -->|BCNorm + Complex RoPE SSD| FAM_M3["BiMamba-3 Family<br/>(BCNorm + B/C Bias + Data-Dep RoPE)"]
    FAM_M3 --> ARM6["<b>[Arm 6] Mamba-3 Mild Recipe (Task 3)</b><br/>• LR = 1e-4, WD = 0.1, Clip = 1.0<br/>• <b>Test CER: 0.357</b> (+1.7% over Arm 5)"]
    FAM_M3 --> ARM7["<b>[Arm 7] Mamba-3 Aggressive (Task 4)</b><br/>• LR = 3e-4, WD = 0.1, Clip = 1.0<br/>• <b>Test CER: 0.310</b>"]

    %% Output Head
    ARM1 & ARM2 & ARM3 & ARM4 & ARM5 & ARM6 & ARM7 & ARM8 --> HEAD["Linear Classifier (29 Character Classes)<br/>Cross-Entropy Loss (Evaluated via Levenshtein Sentence CER)"]
```

---

## 2. Transformer Block (Arms 1 & 8)

```mermaid
flowchart TD
    IN["Keystroke Embeddings: x ∈ ℝ^(B × T × 512)"] --> LN1["LayerNorm(x)"]
    LN1 --> QKV["Q, K, V Linear Projections (2 Attention Heads)"]
    QKV --> ALIBI["Compute ALiBi Relative Slopes Bias:<br/>B_ij = -m · |i - j|"]
    ALIBI --> SOFTMAX["Scaled Dot-Product Attention:<br/>A = Softmax( (Q Kᵀ / √d_k) + B )"]
    SOFTMAX --> ATTN_OUT["Attn_Out = A · V"]
    ATTN_OUT --> PROJ["Output Linear Projection"]
    IN --> RES1["+ (Residual Add)"]
    PROJ --> RES1
    RES1 --> LN2["LayerNorm"]
    LN2 --> MLP["Feedforward Network (MLP):<br/>Linear(512, 2048) ➔ GELU ➔ Linear(2048, 512)"]
    RES1 --> RES2["+ (Residual Add)"]
    MLP --> RES2
    RES2 --> TO_NEXT["Output to Next Block / Final Classifier"]
```

---

## 3. BiMamba-2 Pre-Norm Block (Arms 2, 3, 4, 5)

```mermaid
flowchart TD
    X["Input x ∈ ℝ^(B × T × 512)"] --> NORM["RMSNorm(x)"]
    
    subgraph FWD ["Forward Mamba-2 SSD Mixer"]
        NORM --> INP_F["in_proj ➔ split: [z, xbc, dt_raw]"]
        INP_F --> CONV_F["1D Conv (kernel 4) + SiLU"]
        CONV_F --> SPLIT_F["Split: x_s (16×64), B (1×64), C (1×64)"]
        INP_F --> DT_F["Δt = Softplus(dt_raw + dt_bias)"]
        SPLIT_F & DT_F --> SSD_F["Quadratic SSD Dual Form:<br/>y_fwd = ( (C · Bᵀ) ⊙ exp(cumsum(Δt · A)) · Δt ) x_s + D · x_s"]
        SSD_F --> GNORM_F["RMSNorm(y_fwd, gate=SiLU(z))"]
        GNORM_F --> OUT_F["out_proj(y_fwd)"]
    end

    subgraph BWD ["Backward Mamba-2 SSD Mixer"]
        NORM --> FLIP_IN["torch.flip(time_dim)"]
        FLIP_IN --> INP_B["in_proj ➔ Conv1D ➔ SSD Recurrence"]
        INP_B --> SSD_B["Quadratic SSD Dual Form (Reversed Order)"]
        SSD_B --> GNORM_B["RMSNorm(gate=SiLU(z)) ➔ out_proj"]
        GNORM_B --> FLIP_OUT["torch.flip(time_dim) ➔ y_bwd"]
    end

    OUT_F --> SUM["Sum: y_fwd + y_bwd"]
    FLIP_OUT --> SUM
    SUM --> DROP["Dropout(0.1)"]
    X --> RES["+ (Residual Add)"]
    DROP --> RES
    RES --> BLK_OUT["Output Block Embedding (B, T, 512)"]
```

---

## 4. BiMamba-3 Block Upgrades (Arms 6 & 7)

```mermaid
flowchart TD
    X["Input x ∈ ℝ^(B × T × 512)"] --> INP["in_proj ➔ [z, xbc, dt_raw, θ] (Extra nheads outputs)"]
    
    INP --> CONV["1D Depthwise Conv (kernel 4) + SiLU"]
    CONV --> SPLIT["Split: x_s, Bmat, Cmat"]
    
    subgraph UPGRADE1 ["Upgrade 1: BCNorm + Learnable Biases"]
        SPLIT --> BCNORM["B = RMSNorm(Bmat) + b_bias<br/>C = RMSNorm(Cmat) + c_bias<br/><i>(Prevents unbounded norm growth)</i>"]
    end

    subgraph UPGRADE2 ["Upgrade 2: Complex State via Data-Dependent RoPE"]
        INP --> PHASE["Per-head rotation rate θ_t ∈ ℝ^H<br/>Cumulative angle: φ_t = cumsum(θ_t, dim=1)"]
        BCNORM & PHASE --> ROPE["Rotate State Dimensions:<br/>B_rot = RoPE(B, -φ)<br/>C_rot = RoPE(C, +φ)"]
    end

    ROPE --> SSD["SSD Dual Form Kernel:<br/>(C_rot · B_rotᵀ) ≡ R(φ_t - φ_s) · (C_t · B_sᵀ)<br/>y = ( (C_rot · B_rotᵀ) ⊙ L · Δt ) x_s + D · x_s"]
    
    SSD --> GNORM["Gated RMSNorm(y, gate=SiLU(z))"]
    GNORM --> OUT_P["out_proj(y)"]
```
