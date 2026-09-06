"""Encoder-extraction evaluation: how much decodable structure does the
learned SimpleConvTimeAgg encoder add over the raw MEG window?

Research-level checks, all computed from one trained checkpoint:

1. **Linear probe (ridge, closed form)** — train on train-split features,
   evaluate on the test split:
     * raw flattened window (306 ch x 25 samples)  -> 29-class char
     * encoder embedding (512-d)                   -> 29-class char
   If the learned probe beats the raw probe, the encoder is extracting
   decodable structure beyond what a linear map of the raw signal carries.
2. **Per-subject probe breakdown** (S15 / S16 / S6).
3. **PCA geometry** — 2-D PCA scatter of test embeddings coloured by
   character class (top-8 frequent) and by subject: do characters cluster,
   and is there subject-specific structure?

Usage (from the repo root, same env as training):

    python probe_encoder.py --ckpt <best.ckpt> --core mamba --small \
        --n-train 10000 --out probe_out/mamba-lr3e4
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from brain2qwerty_v1.utils import CHAR_INDEX

from explain_mamba import build_module  # reuse the exact training-time build

CHANCE = 1.0 / 29


# --------------------------------------------------------------------------- #
# Feature extraction
# --------------------------------------------------------------------------- #
@torch.no_grad()
def extract(module, loader, device, max_windows=None):
    """Per-window raw input, encoder embedding, label and subject."""
    out = {"raw": [], "emb": [], "y": [], "subj": []}
    n = 0
    for batch in loader:
        data = {k: v.to(device) for k, v in batch.data.items()}
        emb = module.model(data["neuro"], data["subject_id"], data["channel_positions"])
        out["raw"].append(data["neuro"].float().cpu().flatten(1))
        out["emb"].append(emb.float().cpu())
        out["y"].append(data["feature"].squeeze(1).cpu())
        out["subj"] += data["subject_id"].cpu().tolist()
        n += emb.shape[0]
        if max_windows and n >= max_windows:
            break
    for k in ("raw", "emb", "y"):
        out[k] = torch.cat(out[k])
    n_total = out["y"].shape[0]
    out["subj"] = out["subj"][:n_total]
    if max_windows:
        for k in ("raw", "emb", "y"):
            out[k] = out[k][:max_windows]
        out["subj"] = out["subj"][:max_windows]
    return out


# --------------------------------------------------------------------------- #
# Closed-form ridge probe
# --------------------------------------------------------------------------- #
def ridge_probe(Xtr, Ytr, Xte, Yte, lam=10.0, device=None):
    """Multiclass ridge regression, closed form. Returns (train_acc, test_acc, preds)."""
    dev = device or Xtr.device
    Xtr, Xte, Ytr, Yte = Xtr.to(dev), Xte.to(dev), Ytr.to(dev), Yte.to(dev)
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True) + 1e-6
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd
    n, d = Xtr.shape
    C = int(max(Ytr.max(), Yte.max())) + 1
    T = torch.zeros(n, C, device=dev)
    T[torch.arange(n, device=dev), Ytr] = 1.0
    A = Xtr.T @ Xtr + lam * torch.eye(d, device=dev)
    W = torch.linalg.solve(A, Xtr.T @ T)  # (d, C)
    tr_acc = ((Xtr @ W).argmax(1) == Ytr).float().mean().item()
    te_pred = (Xte @ W).argmax(1).cpu()
    te_acc = (te_pred == Yte.cpu()).float().mean().item()
    return tr_acc, te_acc, te_pred


def pca2(X):
    X = X - X.mean(0, keepdim=True)
    _, _, V = torch.pca_lowrank(X, q=2)
    return (X @ V).numpy()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--core", default="mamba")
    p.add_argument("--small", action="store_true")
    p.add_argument("--subjects", nargs="+", default=None)
    p.add_argument("--n-train", type=int, default=10000)
    p.add_argument("--out", default="probe_out")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    module, loaders, device = build_module(args)
    print(f"[probe] device={device} ckpt={args.ckpt}")

    print(f"[probe] extracting train features (cap {args.n_train})...")
    tr = extract(module, loaders["train"], device, args.n_train)
    print("[probe] extracting test features...")
    te = extract(module, loaders["test"])

    results = {"ckpt": args.ckpt, "core": args.core, "chance": CHANCE, "probes": {}}
    for name, key in (("raw_306x25", "raw"), ("encoder_emb", "emb")):
        tr_acc, te_acc, te_pred = ridge_probe(tr[key], tr["y"], te[key], te["y"],
                                              device=device)
        per_subj = {}
        for s in sorted(set(te["subj"])):
            m = torch.tensor([x == s for x in te["subj"]])
            if int(m.sum()) > 0:
                per_subj[str(s)] = round(
                    (te_pred[m] == te["y"][m]).float().mean().item(), 4)
        results["probes"][name] = {
            "train_acc": round(tr_acc, 4),
            "test_acc": round(te_acc, 4),
            "test_acc_per_subject": per_subj,
        }
        print(f"[probe] {name:<14} train {tr_acc:.3f}  test {te_acc:.3f} "
              f"(chance {CHANCE:.3f})  per-subject {per_subj}")

    # --- figures ---------------------------------------------------------- #
    top_chars = torch.bincount(te["y"]).argsort(descending=True)[:8].tolist()

    for key, tag in (("emb", "encoder"), ("raw", "raw")):
        Z = pca2(te[key])
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
        ax = axes[0]
        for c in top_chars:
            m = (te["y"] == c).numpy()
            ax.scatter(Z[m, 0], Z[m, 1], s=4, alpha=0.5,
                       label=repr(CHAR_INDEX.get(c, "?")))
        ax.set_title(f"{tag} features — coloured by character (top-8)")
        ax.legend(markerscale=3, fontsize=8)
        ax = axes[1]
        for s in sorted(set(te["subj"])):
            m = np.array([x == s for x in te["subj"]])
            ax.scatter(Z[m, 0], Z[m, 1], s=4, alpha=0.5, label=f"subject {s}")
        ax.set_title(f"{tag} features — coloured by subject")
        ax.legend(markerscale=3, fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"pca_{tag}.png", dpi=150)
        plt.close(fig)

    (out_dir / "probe_results.json").write_text(json.dumps(results, indent=2))
    print(f"[probe] wrote {out_dir}/probe_results.json and pca_*.png")
    print(json.dumps(results["probes"], indent=2))


if __name__ == "__main__":
    main()
