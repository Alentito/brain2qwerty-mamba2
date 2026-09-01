"""End-to-end smoke tests for all 3 V3 word-level decoding tasks.

Validates:
1. Architecture build + forward + backward for all 3 cores:
   - conformer (V2 Baseline on SpanishBCBL)
   - mamba_mlp (Round 3 Champion BiMamba-2 + Gated MLP)
   - mamba3_hybrid_stabilized (Mamba-3 Stabilized Hybrid)
2. Full joint 3-loss training step (_run_step: CTC + Word Contrastive + LoRA LLM).

Run with:
    pytest brain2qwerty_v3/tests/test_v3_smoke.py -v -s
"""

import types
import pytest
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model
from neuralset.dataloader import Batch

from brain2qwerty_v3.config.model_config import build_encoder_config
from brain2qwerty_v3.models import ConvMambaHybrid
from brain2qwerty_v3.pl_module import NeuroLLMModule
from brain2qwerty_v3.metrics import SemanticErrorRate
from torchmetrics.text import CharErrorRate, WordErrorRate


# --------------------------------------------------------------------------- #
# 1. Architecture forward pass smoke tests
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("core", ["conformer", "mamba_mlp", "mamba3_hybrid_stabilized", "hybrid"])
def test_architecture_build_and_forward(core):
    """Build the encoder from config and run a forward pass with synthetic data."""
    cfg_dict = build_encoder_config(core=core, small=True)
    cfg = ConvMambaHybrid(**cfg_dict)
    dim = cfg.dim

    model = cfg.build(n_in_channels=306, n_outputs=29)
    model.eval()

    B, T, C = 2, 400, 306
    x = torch.randn(B, T, C)
    days = torch.zeros(B, dtype=torch.long)
    chan_pos = torch.randn(B, C, 2)

    with torch.no_grad():
        out = model(x, days, chan_pos)

    assert "c_out" in out, f"[{core}] missing 'c_out'"
    assert "z_final" in out, f"[{core}] missing 'z_final'"
    ctc_logits = out["c_out"]
    z_final = out["z_final"]

    assert ctc_logits.shape[0] == B
    assert ctc_logits.shape[-1] == 29, f"[{core}] CTC output should be 29, got {ctc_logits.shape[-1]}"
    assert z_final.shape[0] == B
    assert z_final.shape[-1] == dim, f"[{core}] z_final dim should be {dim}, got {z_final.shape[-1]}"
    assert torch.isfinite(ctc_logits).all(), f"[{core}] CTC logits NaN/Inf"
    assert torch.isfinite(z_final).all(), f"[{core}] z_final NaN/Inf"

    print(f"\n  [{core}] CTC logits: {ctc_logits.shape}, z_final: {z_final.shape} ✅")


# --------------------------------------------------------------------------- #
# 2. Architecture backward pass smoke tests
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("core", ["conformer", "mamba_mlp", "mamba3_hybrid_stabilized"])
def test_architecture_backward(core):
    """Ensure gradients flow through the full encoder."""
    cfg_dict = build_encoder_config(core=core, small=True)
    cfg = ConvMambaHybrid(**cfg_dict)
    model = cfg.build(n_in_channels=306, n_outputs=29)
    model.train()

    B, T, C = 2, 400, 306
    x = torch.randn(B, T, C, requires_grad=True)
    days = torch.zeros(B, dtype=torch.long)
    chan_pos = torch.randn(B, C, 2)

    out = model(x, days, chan_pos)
    loss = out["c_out"].sum() + out["z_final"].sum()
    loss.backward()

    assert x.grad is not None and torch.isfinite(x.grad).all(), f"[{core}] input grad bad"

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_graded = sum(1 for p in model.parameters() if p.grad is not None and torch.isfinite(p.grad).all())
    n_total = sum(1 for p in model.parameters() if p.requires_grad)
    print(f"\n  [{core}] {n_graded}/{n_total} params have finite grads, {n_params:,} total params ✅")


# --------------------------------------------------------------------------- #
# 3. Full Joint 3-Loss Pipeline Smoke Test (_run_step)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("core", ["conformer", "mamba_mlp", "mamba3_hybrid_stabilized"])
def test_full_pipeline_run_step(core):
    """Smoke test of the complete NeuroLLMModule with CTC + Word Contrastive + LoRA LLM."""
    cfg_dict = build_encoder_config(core=core, small=True)
    cfg = ConvMambaHybrid(**cfg_dict)
    word_pool_dim = cfg.dim
    network = cfg.build(n_in_channels=306, n_outputs=29)

    llm_name = "Qwen/Qwen3.5-0.8B"
    tokenizer = AutoTokenizer.from_pretrained(llm_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm = AutoModelForCausalLM.from_pretrained(
        llm_name, torch_dtype=torch.float32, trust_remote_code=True
    )
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=2,
        lora_alpha=4,
        lora_dropout=0.0,
        target_modules=["q_proj", "v_proj"],
    )
    llm = get_peft_model(llm, lora_cfg)

    llm_hidden = llm.get_base_model().config.hidden_size
    adapter = nn.Linear(word_pool_dim, llm_hidden)

    # Word embedding lookup table for contrastive target
    word_embed_lookup = {
        "hola mundo": [torch.randn(llm_hidden).numpy(), torch.randn(llm_hidden).numpy()],
        "buenas tardes": [torch.randn(llm_hidden).numpy(), torch.randn(llm_hidden).numpy()],
    }

    llm_metrics = {
        "CER": CharErrorRate(),
        "WER": WordErrorRate(),
        "SemER": SemanticErrorRate(),
    }

    module = NeuroLLMModule(
        network=network,
        llm=llm,
        tokenizer=tokenizer,
        word_proj_adapter=adapter,
        word_embed_lookup=word_embed_lookup,
        word_pool_dim=word_pool_dim,
        llm_metrics=llm_metrics,
        alpha=0.1,
        beta=0.01,
        ctc_start_epoch=0,
        contrastive_start_epoch=0,
        llm_start_epoch=0,
    )

    # Create synthetic Batch
    B, T, C = 2, 400, 306
    batch_data = {
        "neuros": torch.randn(B, T, C),
        "neuro_sizes": torch.tensor([400, 400], dtype=torch.long),
        "phonemes": torch.randint(1, 28, (B, 15), dtype=torch.long),
        "phoneme_sizes": torch.tensor([15, 12], dtype=torch.long),
        "days": torch.zeros(B, dtype=torch.long),
        "chan_pos": torch.randn(B, C, 2),
    }

    # Synthetic segments with true Spanish sentence text
    seg1 = types.SimpleNamespace(start=0.0, duration=4.0, trigger=types.SimpleNamespace(text="hola mundo", extra={"sentence_UID": "s1"}))
    seg2 = types.SimpleNamespace(start=0.0, duration=4.0, trigger=types.SimpleNamespace(text="buenas tardes", extra={"sentence_UID": "s2"}))
    batch = Batch(data=batch_data, segments=[seg1, seg2])

    module.train()
    loss, ctc_logits, phonemes = module._run_step(batch, 0, "train")

    assert torch.isfinite(loss), f"[{core}] Training loss is not finite: {loss}"
    print(f"\n  [{core}] ✅ Full Joint 3-Loss Step Passed! Train loss = {loss.item():.4f}")
