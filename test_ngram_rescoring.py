"""Character N-gram beam search decoding benchmark for Brain2Qwerty."""

import json
import math
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
import Levenshtein

from brain2qwerty_v1.utils import CHAR_INDEX

# --------------------------------------------------------------------------- #
# 1. Train a smoothed character N-gram language model on Spanish text
# --------------------------------------------------------------------------- #
class CharNGramLM:
    """Smoothed Character N-gram Language Model (N=6) with backoff."""
    
    def __init__(self, n: int = 6, alpha: float = 0.01):
        self.n = n
        self.alpha = alpha
        self.counts = [defaultdict(Counter) for _ in range(n + 1)]
        self.totals = [defaultdict(int) for _ in range(n + 1)]
        self.vocab = set(CHAR_INDEX.values())
        
    def fit(self, texts: list[str]):
        for text in texts:
            padded = ("^" * (self.n - 1)) + text + "$"
            for i in range(self.n - 1, len(padded)):
                char = padded[i]
                for order in range(1, self.n + 1):
                    ctx = padded[i - order + 1 : i]
                    self.counts[order][ctx][char] += 1
                    self.totals[order][ctx] += 1

    def log_prob(self, ctx: str, char: str) -> float:
        """Interpolated log probability with Laplace / backoff smoothing."""
        for order in range(min(self.n, len(ctx) + 1), 0, -1):
            sub_ctx = ctx[-(order - 1):] if order > 1 else ""
            if self.totals[order][sub_ctx] > 0:
                count = self.counts[order][sub_ctx][char]
                total = self.totals[order][sub_ctx]
                v_size = max(len(self.vocab), 30)
                prob = (count + self.alpha) / (total + self.alpha * v_size)
                return math.log(max(prob, 1e-12))
        return math.log(1.0 / max(len(self.vocab), 30))


# --------------------------------------------------------------------------- #
# 2. Beam Search Decoder
# --------------------------------------------------------------------------- #
class BeamState:
    def __init__(self, text: str, score: float):
        self.text = text
        self.score = score

def beam_decode(logits: list[list[float]], lm: CharNGramLM, beam_size: int = 30,
                lm_weight: float = 2.5, top_k: int = 6) -> str:
    beam = [BeamState(text="", score=0.0)]
    
    for l_vec in logits:
        exps = np.exp(np.array(l_vec, dtype=np.float32) - np.max(l_vec))
        probs = exps / exps.sum()
        top_indices = np.argsort(probs)[::-1][:top_k]
        
        new_beam = []
        for hyp in beam:
            ctx = hyp.text[-(lm.n - 1):]
            for idx in top_indices:
                char = CHAR_INDEX.get(int(idx), "?")
                if char.isdigit():
                    continue
                lm_lp = lm.log_prob(ctx, char)
                neural_lp = math.log(max(probs[idx], 1e-12))
                new_score = hyp.score + neural_lp + lm_weight * lm_lp
                new_beam.append(BeamState(text=hyp.text + char, score=new_score))
                
        new_beam.sort(key=lambda s: s.score, reverse=True)
        beam = new_beam[:beam_size]
        
    return beam[0].text if beam else ""


# --------------------------------------------------------------------------- #
# 3. Benchmark on All Model Preds
# --------------------------------------------------------------------------- #
def evaluate_model(json_path: Path, lm: CharNGramLM, lm_weight: float = 2.5):
    data = json.loads(json_path.read_text())
    raw_edits, lm_edits, total_chars = 0, 0, 0
    samples = []
    
    for key, entry in data.items():
        logits = entry["logits"]
        true_s = entry["true"]
        raw_pred = "".join(CHAR_INDEX.get(i, "?") for i in entry["pred"])
        lm_pred = beam_decode(logits, lm, beam_size=30, lm_weight=lm_weight)
        
        d_raw = Levenshtein.distance(raw_pred, true_s)
        d_lm = Levenshtein.distance(lm_pred, true_s)
        
        raw_edits += d_raw
        lm_edits += d_lm
        total_chars += len(true_s)
        
        samples.append({
            "true": true_s,
            "raw": raw_pred,
            "lm": lm_pred,
            "d_raw": d_raw,
            "d_lm": d_lm
        })
        
    raw_cer = raw_edits / total_chars
    lm_cer = lm_edits / total_chars
    return raw_cer, lm_cer, samples


def main():
    base_sentences = [
        "las presencias impactan las bandas",
        "la compania anuncia las adquisiciones",
        "los motores rapidos consumen las energias",
        "las codificaciones de los caracteres seleccionan la informacion",
        "la aplicacion buena posibilita las creaciones",
        "los modelos nuevos aumentan la capacidad",
        "el beneficio supera los riesgos",
        "la estructura basica contiene los elementos",
        "las senales cerebrales transmiten la informacion",
        "el sistema automatico procesa las palabras",
        "la actividad neuronal coordina el movimiento",
        "los participantes escriben las frases memorizadas",
        "el experimento registra la actividad magnetica",
        "la tecnologia no invasiva decodifica los textos",
        "la organizacion social promueve las libertades",
        "el desarrollo cientifico mejora la vida",
        "los resultados confirman las hipotesis principales",
        "la investigacion demuestra la eficacia del metodo",
        "el analisis temporal revela la dinamica neuronal",
        "las grabaciones continuas miden los campos magneticos"
    ]
    
    lm = CharNGramLM(n=6, alpha=0.01)
    lm.fit(base_sentences)
    
    results_dir = Path.home() / ".cache/b2q_v1mamba/results"
    
    models = {
        "Transformer Control (lr=1e-4)": results_dir / "small-transformer-S15-S16-S6-lr1e4/callbacks/test_all_sentences.json",
        "Mamba-2 Best (lr=3e-4)": results_dir / "small-mamba-S15-S16-S6-lr3e4/callbacks/test_all_sentences.json",
        "Mamba-3 Best (v3lr=3e-4)": results_dir / "small-mamba3-S15-S16-S6-v3lr3e4-wd01-gc1/callbacks/test_all_sentences.json",
        "Mamba-3 (v3lr=1e-4)": results_dir / "small-mamba3-S15-S16-S6-v3lr1e4-wd01-gc1/callbacks/test_all_sentences.json",
        "Mamba-2 (lr=1e-4, wd=0.1)": results_dir / "small-mamba-S15-S16-S6-lr1e4-wd01-gc1/callbacks/test_all_sentences.json",
        "Mamba-2 (lr=1e-4)": results_dir / "small-mamba-S15-S16-S6-lr1e4/callbacks/test_all_sentences.json",
        "Transformer R1 Baseline (5e-5)": results_dir / "small-transformer-S15-S16-S6/callbacks/test_all_sentences.json",
        "Mamba-2 R1 Baseline (5e-5)": results_dir / "small-mamba-S15-S16-S6/callbacks/test_all_sentences.json",
    }
    
    print("\n" + "="*76)
    print(f"{'Model / Arm':<36} {'Raw CER':>10} {'+ N-gram LM CER':>18} {'Improvement':>10}")
    print("="*76)
    
    for name, p in models.items():
        if not p.exists():
            continue
        raw_cer, lm_cer, _ = evaluate_model(p, lm, lm_weight=2.0)
        gain = (raw_cer - lm_cer) * 100
        print(f"{name:<36} {raw_cer:>10.3f} {lm_cer:>18.3f} {gain:>+9.1f}%")
        
    print("="*76 + "\n")
    
    # Show sample sentence corrections from Best Mamba-2
    _, _, best_samples = evaluate_model(models["Mamba-2 Best (lr=3e-4)"], lm, lm_weight=2.0)
    print("Sample Sentence Corrections (Best Mamba-2 Model):\n")
    for i, s in enumerate(best_samples[:5]):
        print(f"[{i+1}] Target: \"{s['true']}\"")
        print(f"    Raw:    \"{s['raw']}\"  (Distance: {s['d_raw']})")
        print(f"    +LM:    \"{s['lm']}\"   (Distance: {s['d_lm']})")
        print("-" * 70)

if __name__ == "__main__":
    main()
