"""Generate animated continuous brain activation video / GIF and interactive viewer."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from scipy.interpolate import griddata
import torch

import studies  # registers Pinet2024Meg
import brain2qwerty_v1.transforms
from brain2qwerty_v1.utils import CHAR_INDEX
from explain_mamba import build_module, collect_windows, group_sentences, core_forward, chars_of, patch_mixers

def create_head_outline():
    """Returns (x, y) coordinates for head circle, nose, and ears."""
    t = np.linspace(0, 2 * np.pi, 100)
    head_x = np.cos(t)
    head_y = np.sin(t)
    # Nose
    nose_x = np.array([np.sin(-np.pi/18), 0, np.sin(np.pi/18)])
    nose_y = np.array([np.cos(-np.pi/18), 1.12, np.cos(np.pi/18)])
    # Left Ear
    ear_t = np.linspace(-np.pi/2, np.pi/2, 30)
    l_ear_x = -1.0 - 0.08 * np.cos(ear_t)
    l_ear_y = 0.25 * np.sin(ear_t)
    # Right Ear
    r_ear_x = 1.0 + 0.08 * np.cos(ear_t)
    r_ear_y = 0.25 * np.sin(ear_t)
    return (head_x, head_y), (nose_x, nose_y), (l_ear_x, l_ear_y), (r_ear_x, r_ear_y)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", default="checkpoints/small-mamba-S15-S16-S6-lr3e4/best.ckpt")
    p.add_argument("--core", default="mamba")
    p.add_argument("--small", action="store_true", default=True)
    p.add_argument("--subjects", nargs="+", default=None)
    p.add_argument("--sentence-idx", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--out-dir", default="explain_out/animation")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[anim] loading model and test data...")
    module, loaders, device = build_module(args)
    mixers = patch_mixers(module.transformer) if args.core != "transformer" else []
    
    store = collect_windows(module, loaders, device)
    sentences = group_sentences(store)
    
    sent = sentences[args.sentence_idx]
    idx = sent["idx"]
    neuro_s = store["neuro"][idx]  # (L, 306, 25)
    y_s = store["y"][idx]          # (L,)
    chpos = store["chpos"]         # (306, 2)
    chars = chars_of(y_s)
    target_text = "".join(c.replace("␣", " ") for c in chars)
    
    L, C, T = neuro_s.shape
    times = np.linspace(-200, 300, T)  # ms
    
    # Normalize chpos to unit circle [-0.85, 0.85]
    xy = chpos.numpy()
    xy_norm = xy - xy.mean(axis=0)
    max_radius = np.max(np.hypot(xy_norm[:, 0], xy_norm[:, 1]))
    xy_norm = (xy_norm / max_radius) * 0.85
    
    # 2D Grid for topographic interpolation
    grid_x, grid_y = np.mgrid[-1.05:1.05:80j, -1.05:1.05:80j]
    head_mask = (grid_x**2 + grid_y**2) <= 1.0

    print(f"[anim] sentence {args.sentence_idx}: \"{target_text}\" ({L} keystrokes, {L * T} total time frames)")
    
    # --------------------------------------------------------------------------- #
    # Export JSON payload for interactive Web Player
    # --------------------------------------------------------------------------- #
    web_payload = {
        "sentence": target_text,
        "chars": chars,
        "n_keystrokes": L,
        "time_samples": times.tolist(),
        "chpos": xy_norm.tolist(),
        "activations": neuro_s.numpy().transpose(0, 2, 1).tolist(),
    }
    (out_dir / "brain_data.json").write_text(json.dumps(web_payload))
    print(f"[anim] saved interactive brain_data.json ({len(web_payload['activations'])} keystrokes)")

    # --------------------------------------------------------------------------- #
    # Generate Animated GIF for keypresses
    # --------------------------------------------------------------------------- #
    fig = plt.figure(figsize=(11, 5.5), facecolor="#0f172a")
    gs = fig.add_gridspec(2, 2, width_ratios=[1.1, 1.4], height_ratios=[1, 1],
                           left=0.05, right=0.95, top=0.90, bottom=0.08, wspace=0.25, hspace=0.35)

    ax_topo = fig.add_subplot(gs[:, 0])
    ax_wave = fig.add_subplot(gs[0, 1])
    ax_text = fig.add_subplot(gs[1, 1])

    # Head geometry
    (hx, hy), (nx, ny), (lx, ly), (rx, ry) = create_head_outline()

    vmax = float(np.percentile(np.abs(neuro_s.numpy()), 98))
    vmin = -vmax

    mean_wave = neuro_s.abs().mean(dim=1).numpy()  # (L, 25)

    def draw_head_base(ax):
        ax.clear()
        ax.set_facecolor("#0b1120")
        ax.plot(hx, hy, color="#64748b", lw=2)
        ax.plot(nx, ny, color="#64748b", lw=2)
        ax.plot(lx, ly, color="#64748b", lw=2)
        ax.plot(rx, ry, color="#64748b", lw=2)
        ax.set_xlim(-1.25, 1.25)
        ax.set_ylim(-1.25, 1.25)
        ax.set_aspect("equal")
        ax.axis("off")

    total_frames = L * T
    frame_indices = np.linspace(0, total_frames - 1, min(60, total_frames), dtype=int)

    def update(frame_idx):
        k_idx = frame_idx // T
        t_idx = frame_idx % T
        curr_t = times[t_idx]
        curr_char = chars[k_idx]
        typed_so_far = "".join(chars[:k_idx + 1]).replace("␣", " ")

        # 1. Topographic Map
        draw_head_base(ax_topo)
        signal_t = neuro_s[k_idx, :, t_idx].numpy()
        grid_z = griddata(xy_norm, signal_t, (grid_x, grid_y), method="cubic", fill_value=0)
        grid_z[~head_mask] = np.nan

        im = ax_topo.imshow(grid_z.T, extent=[-1.05, 1.05, -1.05, 1.05], origin="lower",
                            cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="equal")
        ax_topo.scatter(xy_norm[:, 0], xy_norm[:, 1], c="white", s=6, alpha=0.6, edgecolors="none")
        ax_topo.set_title(f"MEG 306-Channel Topomap\nKeypress #{k_idx+1}: '{curr_char}' (t = {curr_t:+.0f} ms)",
                          color="#f8fafc", fontsize=11, fontweight="bold")

        # 2. Sensor Waveform
        ax_wave.clear()
        ax_wave.set_facecolor("#0b1120")
        ax_wave.plot(times, mean_wave[k_idx], color="#38bdf8", lw=2, label="Mean |MEG| (306 ch)")
        ax_wave.axvline(curr_t, color="#ef4444", lw=2, linestyle="--", label=f"t = {curr_t:+.0f} ms")
        ax_wave.axvline(0, color="#94a3b8", lw=1, linestyle=":", label="Keypress Strike (0 ms)")
        ax_wave.set_xlim(-200, 300)
        ax_wave.set_ylim(0, mean_wave.max() * 1.15)
        ax_wave.set_xlabel("Time relative to keypress (ms)", color="#94a3b8", fontsize=9)
        ax_wave.set_ylabel("Normalized Magnitude", color="#94a3b8", fontsize=9)
        ax_wave.tick_params(colors="#94a3b8", labelsize=8)
        ax_wave.set_title(f"500 ms Window Activity [-200 ms to +300 ms]", color="#f8fafc", fontsize=10)
        ax_wave.legend(loc="upper right", fontsize=8, facecolor="#1e293b", edgecolor="#334155", labelcolor="#f8fafc")

        # 3. Sentence Streaming Text
        ax_text.clear()
        ax_text.set_facecolor("#0b1120")
        ax_text.axis("off")
        ax_text.text(0.05, 0.75, "Target Sentence:", color="#94a3b8", fontsize=10, transform=ax_text.transAxes)
        ax_text.text(0.05, 0.50, f"\"{target_text}\"", color="#f8fafc", fontsize=12, fontweight="bold", transform=ax_text.transAxes)
        ax_text.text(0.05, 0.25, f"Decoded Stream: ", color="#38bdf8", fontsize=10, transform=ax_text.transAxes)
        ax_text.text(0.35, 0.25, f"{typed_so_far} █", color="#22c55e", fontsize=11, fontweight="bold", transform=ax_text.transAxes)

    ani = animation.FuncAnimation(fig, update, frames=frame_indices, interval=100)
    gif_path = out_dir / "brain_typing_continuous.gif"
    ani.save(gif_path, writer="pillow", dpi=100)
    plt.close(fig)
    print(f"[anim] saved GIF animation to {gif_path}")


if __name__ == "__main__":
    main()
