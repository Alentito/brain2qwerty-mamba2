# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Explainability visualizations for the Brain2Qwerty V1-Mamba ablation.

Loads a trained checkpoint, runs test sentences through the model, and
produces a four-level explainability report:

1. Input level    — grad x input saliency on the raw MEG window
                    (306 channels x 25 samples @ 50 Hz, -200..+300 ms):
                    heatmap, 2D channel topomap, and per-time-sample course.
2. Core level     — Mamba "mixing maps": the quadratic SSD form materializes
                    the exact input->output mixing matrix  m = (C.B) * L * dt
                    (B, H, T, S), i.e. the SSM analogue of an attention map.
                    Plotted per block, forward and backward direction, with
                    the typed characters as axis labels.
3. Selectivity    — Delta_t (dt) profiles per head/block: where along the
                    sentence the SSM opens/closes its state update, plus the
                    aggregate "memory horizon" = how many keystrokes back the
                    mixing weight effectively reaches.
4. Output level   — per-position char probability heatmaps with the true
                    characters marked.

Usage (from the repo root, same env/flags as training):

    python explain_mamba.py --ckpt <path/to/best.ckpt> --core mamba --small \
        --n-sentences 6 --out explain_out/mamba-lr1e4

For --core transformer only levels 1 and 4 are produced (no SSD matrices);
pass the same --core/--small/--subjects flags used to train the checkpoint.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import studies  # noqa: F401  (registers Pinet2024Meg)

import brain2qwerty_v1.transforms  # noqa: F401
from brain2qwerty_v1.metrics import CER
from brain2qwerty_v1.pl_module import BrainModule
from brain2qwerty_v1.utils import CHAR_INDEX, materialize_lazy_params

from brain2qwerty_v1_mamba import mamba_core as _mamba_core  # noqa: F401
from brain2qwerty_v1_mamba import transforms as _transforms  # noqa: F401
from brain2qwerty_v1_mamba.config.xp_config import experiment_config
from brain2qwerty_v1_mamba.mamba_core import Mamba2Mixer, _segsum
from brain2qwerty_v1_mamba.main import Experiment

FS = 50.0  # windows: start=-0.2, duration=0.5 -> 25 samples @ 50 Hz
T0 = -0.2


# --------------------------------------------------------------------------- #
# SSD capture: patch _ssd_expanded to stash the mixing matrix m = (C.B)*L*dt
# --------------------------------------------------------------------------- #
def patch_mixers(core_module: torch.nn.Module) -> list[Mamba2Mixer]:
    """Replace each mixer's _ssd_expanded with a capturing copy (same math).

    Stores on each mixer ``._xai = {"m": (1,H,T,S), "dt": (1,T,H), "L": ...}``
    of the most recent forward pass. Works for Mamba2Mixer and Mamba3Mixer
    (the latter calls _ssd_expanded with the RoPE-rotated B/C, which is exactly
    the mixing we want to visualize).
    """
    mixers = [m for m in core_module.modules() if isinstance(m, Mamba2Mixer)]
    for mixer in mixers:

        def capturing_ssd(x, Bh, Ch, dt, _mixer=mixer):
            b, t, h, p = x.shape
            A = -torch.exp(_mixer.A_log.float())
            dA = dt.float() * A
            L = torch.exp(_segsum(dA.permute(0, 2, 1)))  # (B, H, T, S)
            dt_s = dt.float().permute(0, 2, 1)  # (B, H, S)

            y = torch.empty(b, t, h, p, device=x.device, dtype=torch.float32)
            maps = []
            for h0 in range(0, h, _mixer.head_chunk):
                h1 = min(h0 + _mixer.head_chunk, h)
                cb = torch.einsum("bthn,bshn->bhts", Ch[:, :, h0:h1], Bh[:, :, h0:h1])
                m = cb * L[:, h0:h1] * dt_s[:, h0:h1].unsqueeze(2)
                maps.append(m)
                y[:, :, h0:h1] = torch.einsum(
                    "bhts,bshp->bthp", m, x[:, :, h0:h1].float()
                )
            _mixer._xai = {
                "m": torch.cat(maps, dim=1).detach().cpu(),  # (B, H, T, S)
                "dt": dt.detach().cpu(),  # (B, T, H)
            }
            return y

        mixer._ssd_expanded = capturing_ssd
    return mixers


# --------------------------------------------------------------------------- #
# Data + model
# --------------------------------------------------------------------------- #
def build_module(args) -> tuple[BrainModule, dict, torch.device]:
    cfg = experiment_config(subjects=args.subjects, core=args.core, small=args.small)
    exp = Experiment(**cfg)
    loaders = exp.data.build()
    brain, core = exp._build_modules(loaders["test"])
    module = BrainModule(
        model=brain,
        transformer=core,
        loss=exp.loss.build(),
        metrics={"CER": CER()},
        optimizer=exp.optimizer,
    )
    materialize_lazy_params(module, loaders["test"])
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    module.load_state_dict(state["state_dict"])
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    module = module.to(device).eval()
    return module, loaders, device


@torch.no_grad()
def collect_windows(module: BrainModule, loaders: dict, device) -> dict:
    """One pass over the test split: per-window embeddings + metadata."""
    store = {"emb": [], "neuro": [], "y": [], "uid": [], "subj": [], "chpos": None}
    for batch in loaders["test"]:
        data = {k: v.to(device) for k, v in batch.data.items()}
        emb = module.model(data["neuro"], data["subject_id"], data["channel_positions"])
        store["emb"].append(emb.cpu())
        store["neuro"].append(data["neuro"].cpu())
        store["y"].append(data["feature"].squeeze(1).cpu())
        store["uid"] += [seg.trigger.extra["sentence_UID"] for seg in batch.segments]
        store["subj"] += data["subject_id"].cpu().tolist()
        if store["chpos"] is None:
            store["chpos"] = data["channel_positions"][0].cpu()  # (C, 2), same for all
    for k in ("emb", "neuro", "y"):
        store[k] = torch.cat(store[k])
    return store


def group_sentences(store: dict) -> list[dict]:
    """Group window indices by sentence_UID, preserving order."""
    order, groups = [], {}
    for i, uid in enumerate(store["uid"]):
        if uid not in groups:
            groups[uid] = []
            order.append(uid)
        groups[uid].append(i)
    return [{"uid": u, "idx": groups[u]} for u in order]


# --------------------------------------------------------------------------- #
# Per-sentence forward passes
# --------------------------------------------------------------------------- #
def core_forward(module: BrainModule, emb_s: torch.Tensor, device):
    """Run the sentence core on one sentence (L, D); returns logits (L, 29).

    NOT no_grad internally: the saliency pass needs gradients through the
    core. Wrap with ``torch.no_grad()`` at call sites that don't.
    """
    x = emb_s[None].to(device)
    mask = torch.ones(1, x.shape[1], dtype=torch.bool, device=device)
    out = module.transformer(x, mask=mask)
    return module.linear(out)[0]


def saliency(module: BrainModule, store: dict, idx: list[int], device) -> torch.Tensor:
    """grad x input on the raw MEG windows w.r.t. the true-char log-probs."""
    neuro = store["neuro"][idx].to(device).requires_grad_(True)
    subj = torch.tensor([store["subj"][i] for i in idx], device=device)
    chpos = store["chpos"].to(device).expand(len(idx), -1, -1)
    y = store["y"][idx].to(device)
    emb = module.model(neuro, subj, chpos)
    logits = core_forward(module, emb, device)
    lp = torch.log_softmax(logits.float(), dim=-1)[torch.arange(len(idx)), y].sum()
    module.zero_grad()
    lp.backward()
    return (neuro.grad * neuro).detach().cpu()  # (L, C, T)


def chars_of(y: torch.Tensor) -> list[str]:
    out = []
    for c in y.tolist():
        ch = CHAR_INDEX.get(c, "?")
        out.append({" ": "␣", "@": "<sp>", "9": "<num>"}.get(ch, ch))
    return out


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _char_ticks(ax, chars, font=6):
    ax.set_xticks(range(len(chars)), chars, rotation=90, fontsize=font)
    ax.set_yticks(range(len(chars)), chars, fontsize=font)


def fig_mixing_maps(sent_data, chars, path, title):
    """Grid: n_block rows x (fwd, bwd) cols of head-mean mixing matrices."""
    blocks = sent_data["maps"]  # list per block: {"fwd": (T,S), "bwd": (T,S)}
    n = len(blocks)
    fig, axes = plt.subplots(n, 2, figsize=(11, 2.6 * n), squeeze=False)
    vmax = max(np.abs(b[d]).max() for b in blocks for d in ("fwd", "bwd"))
    for r, blk in enumerate(blocks):
        for c, direction in enumerate(("fwd", "bwd")):
            ax = axes[r][c]
            im = ax.imshow(blk[direction], cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                           aspect="equal", origin="upper")
            ax.set_title(f"block {r} — {direction}", fontsize=9)
            if r == n - 1:
                ax.set_xlabel("source keypress s")
            if c == 0:
                ax.set_ylabel("target keypress t")
            _char_ticks(ax, chars)
            fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_dt_profiles(sent_data, chars, path, title):
    blocks = sent_data["dt"]  # list per block: {"fwd": (T,H), "bwd": (T,H)}
    n = len(blocks)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 2.8), sharey=True, squeeze=False)
    T = len(chars)
    for r, blk in enumerate(blocks):
        ax = axes[0][r]
        for direction, color in (("fwd", "tab:blue"), ("bwd", "tab:orange")):
            dt = blk[direction]
            ax.plot(range(T), dt.mean(1), color=color, label=direction)
            ax.fill_between(range(T), dt.mean(1) - dt.std(1), dt.mean(1) + dt.std(1),
                            color=color, alpha=0.2)
        ax.set_title(f"block {r}", fontsize=9)
        ax.set_xticks(range(T), chars, rotation=90, fontsize=6)
        if r == 0:
            ax.set_ylabel("Δt (softplus)")
            ax.legend(fontsize=8)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_signal_saliency(sal, neuro, chpos, path, title):
    """Raw window + grad×input: heatmap, 2D topomap, time course."""
    times = (T0 + np.arange(neuro.shape[-1]) / FS) * 1000  # ms
    sig = neuro.mean(0).numpy()  # (C, T) mean over keystrokes
    s = sal.mean(0).abs().numpy()  # (C, T)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    im0 = axes[0][0].imshow(sig, aspect="auto", cmap="RdBu_r",
                            extent=[times[0], times[-1], sig.shape[0], 0])
    axes[0][0].set_title("mean MEG window (306 ch × time)")
    axes[0][0].set_ylabel("channel")
    fig.colorbar(im0, ax=axes[0][0], fraction=0.046)

    im1 = axes[0][1].imshow(s, aspect="auto", cmap="magma",
                            extent=[times[0], times[-1], s.shape[0], 0])
    axes[0][1].set_title("mean |grad × input| saliency")
    fig.colorbar(im1, ax=axes[0][1], fraction=0.046)

    ch_imp = s.mean(1)  # (C,)
    xy = chpos.numpy()
    sc = axes[1][0].scatter(xy[:, 0], xy[:, 1], c=ch_imp, cmap="magma", s=14)
    axes[1][0].set_title("channel importance (2D layout)")
    axes[1][0].set_xticks([])
    axes[1][0].set_yticks([])
    fig.colorbar(sc, ax=axes[1][0], fraction=0.046)

    axes[1][1].plot(times, s.mean(0))
    axes[1][1].axvline(0, color="k", ls="--", lw=0.8)
    axes[1][1].set_title("saliency time course (keypress @ 0 ms)")
    axes[1][1].set_xlabel("time (ms)")

    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_char_probs(probs, y, path, title):
    chars = chars_of(y)
    fig, ax = plt.subplots(figsize=(max(6, 0.35 * len(chars)), 4))
    im = ax.imshow(probs.numpy().T, aspect="auto", cmap="viridis", vmin=0, vmax=1,
                   origin="lower")
    ax.set_yticks(range(probs.shape[1]),
                  [CHAR_INDEX.get(c, "?") for c in range(probs.shape[1])], fontsize=7)
    ax.set_xticks(range(len(chars)), chars, rotation=90, fontsize=7)
    for t, c in enumerate(y.tolist()):
        ax.add_patch(plt.Rectangle((t - 0.5, c - 0.5), 1, 1, fill=False,
                                   edgecolor="red", lw=1.2))
    ax.set_ylabel("char class")
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, label="P(char)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_memory_horizon(all_maps: dict, path):
    """Row-normalized |m| as a function of keystroke lag, per block/direction.

    all_maps: list over sentences of per-block {"fwd": (T,S), "bwd": (T,S)}.
    """
    n_blocks = len(all_maps[0])
    max_lag = max(m[0]["fwd"].shape[0] for m in all_maps) - 1
    fig, axes = plt.subplots(1, n_blocks, figsize=(3.2 * n_blocks, 3),
                             sharey=True, squeeze=False)
    stats = {}
    for b in range(n_blocks):
        ax = axes[0][b]
        for direction, color in (("fwd", "tab:blue"), ("bwd", "tab:orange")):
            acc, cnt = np.zeros(max_lag + 1), np.zeros(max_lag + 1)
            for sent in all_maps:
                m = np.abs(sent[b][direction])
                m /= m.sum(axis=1, keepdims=True) + 1e-12
                T = m.shape[0]
                for lag in range(T):
                    diag = np.diag(m, k=-lag)  # m[t, t-lag]
                    acc[lag] += diag.sum()
                    cnt[lag] += len(diag)
            w = acc / np.maximum(cnt, 1)
            w /= w.sum()
            ax.plot(range(len(w)), w, color=color, label=direction)
            cum = np.cumsum(w)
            lag90 = int(np.searchsorted(cum, 0.9))
            stats[f"block{b}-{direction}"] = {"lag90_keystrokes": lag90}
            ax.axvline(lag90, color=color, ls=":", lw=0.8)
        ax.set_title(f"block {b}", fontsize=9)
        ax.set_xlabel("lag (keystrokes)")
        if b == 0:
            ax.set_ylabel("normalized |mixing weight|")
            ax.legend(fontsize=8)
    fig.suptitle("Effective memory horizon (dotted: 90% cumulative weight)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return stats


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--core", choices=["mamba", "mamba3", "transformer"], default="mamba")
    p.add_argument("--small", action="store_true")
    p.add_argument("--subjects", nargs="+", default=None)
    p.add_argument("--n-sentences", type=int, default=6)
    p.add_argument("--out", default="explain_out")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    module, loaders, device = build_module(args)
    print(f"[xai] device={device} ckpt={args.ckpt}")

    mixers = patch_mixers(module.transformer) if args.core != "transformer" else []
    if args.core == "transformer":
        print("[xai] transformer core: no SSD matrices — "
              "producing saliency + char-probability figures only")

    print("[xai] encoding test windows...")
    store = collect_windows(module, loaders, device)
    sentences = group_sentences(store)
    print(f"[xai] {len(sentences)} test sentences, "
          f"explaining the first {min(args.n_sentences, len(sentences))}")

    all_maps, all_sal, all_neuro = [], [], []
    horizon_stats = {}
    for si, sent in enumerate(sentences[: args.n_sentences]):
        idx = sent["idx"]
        emb_s, y_s = store["emb"][idx], store["y"][idx]
        chars = chars_of(y_s)
        text = "".join(c.replace("␣", " ") for c in chars)
        tag = f"s{si:02d}_{sent['uid']}"
        title = f"sentence {si} (uid {sent['uid']}): \"{text}\""

        with torch.no_grad():
            logits = core_forward(module, emb_s, device)
        probs = torch.softmax(logits.float().cpu(), dim=-1)
        pred = probs.argmax(-1)
        fig_char_probs(probs, y_s, out_dir / f"charprobs_{tag}.png",
                       f"{title}\npred: {''.join(chars_of(pred))}")

        sal = saliency(module, store, idx, device)
        fig_signal_saliency(sal, store["neuro"][idx], store["chpos"],
                            out_dir / f"signal_saliency_{tag}.png", title)
        all_sal.append(sal)
        all_neuro.append(store["neuro"][idx])

        if mixers:
            # _xai buffers were filled by the saliency() core pass above
            # (same sentence, values detached to CPU).
            maps, dts = [], []
            for blk in module.transformer.blocks:
                fm, bm = blk.fwd._xai["m"][0].mean(0).numpy(), None
                bwd_raw = blk.bwd._xai["m"][0].mean(0).numpy()
                # backward mixer saw the reversed sequence: flip both axes
                bm = bwd_raw[::-1, ::-1]
                maps.append({"fwd": fm, "bwd": bm})
                dts.append({
                    "fwd": blk.fwd._xai["dt"][0].numpy(),
                    "bwd": blk.bwd._xai["dt"][0].numpy()[::-1],
                })
            fig_mixing_maps({"maps": maps}, chars, out_dir / f"mixmaps_{tag}.png", title)
            fig_dt_profiles({"dt": dts}, chars, out_dir / f"dt_{tag}.png", title)
            all_maps.append(maps)
        print(f"[xai] {tag} done ({len(idx)} keystrokes)")

    if all_maps:
        horizon_stats = fig_memory_horizon(all_maps, out_dir / "memory_horizon.png")
    fig_signal_saliency(torch.cat(all_sal), torch.cat(all_neuro), store["chpos"],
                        out_dir / "signal_saliency_MEAN.png",
                        f"mean over {len(all_sal)} explained sentences")

    summary = {
        "ckpt": args.ckpt, "core": args.core,
        "n_sentences_explained": len(all_sal),
        "memory_horizon": horizon_stats,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[xai] figures written to {out_dir}/")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
