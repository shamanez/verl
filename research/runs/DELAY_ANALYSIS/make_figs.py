#!/usr/bin/env python3
"""Generate the 3 publication figures for the K=5 vs K=20 staleness report.
Data-driven (no EF content). Saves dpi=200 PNGs into figs/.
"""
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
FIGS = HERE / "figs"
FIGS.mkdir(exist_ok=True)

# ---- palette ----
C_DENSE = "#0f6e56"   # teal/green
C_K5 = "#185fa5"      # blue
C_K20 = "#a32d2d"     # red
C_GREY = "#6b6b6b"
C_GRID = "#d7d7d7"

# ---- clean scientific style ----
mpl.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.edgecolor": "#444444",
    "axes.linewidth": 0.9,
    "axes.grid": True,
    "grid.color": C_GRID,
    "grid.linewidth": 0.7,
    "grid.alpha": 0.7,
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12.5,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ---- load data ----
df = pd.read_csv(HERE / "exp37_timeseries.csv")
ref = json.loads((HERE / "reference_data.json").read_text())

dense = {int(k): v for k, v in ref["dense_EXP-36C"].items()}
k5 = {int(k): v for k, v in ref["K5_EXP-36B"].items()}
k20 = {int(k): v for k, v in ref["K20_EXP-37"].items()}
floor = ref["no_merger_floor"]
band = ref["rollout_noise_band"]


# ============================================================
# FIGURE 1 — validation curves (the headline)
# ============================================================
def fig1():
    fig, ax = plt.subplots(figsize=(8.0, 5.2))

    # rollout-noise band around dense (faint shading), flat-extended to 100
    dense_x = sorted(dense)
    dense_y = [dense[x] for x in dense_x]
    # flat extrapolation of dense to step 100 (shows the ceiling)
    ext_x = [50, 100]
    ext_y = [dense[50], dense[50]]
    band_x = np.array([25, 100])
    band_center = np.array([dense[25], dense[50]] if False else [dense[25], dense[50]])
    # build a smooth-ish center for the band across 25..100 (25->50 measured, 50->100 flat)
    bx = np.array([25, 50, 100])
    by = np.array([dense[25], dense[50], dense[50]])
    ax.fill_between(bx, by - band, by + band, color=C_DENSE, alpha=0.12, lw=0,
                    label=f"dense ± rollout noise ({band:.3f})")

    # dense: measured 25->50 thicker, dashed thin extrapolation to 100
    ax.plot(dense_x, dense_y, color=C_DENSE, lw=2.4, marker="o", ms=7,
            label="dense (comm-eff OFF)", zorder=5)
    ax.plot(ext_x, ext_y, color=C_DENSE, lw=1.3, ls="--", alpha=0.8, zorder=4)
    ax.annotate("flat ceiling", xy=(100, dense[50]), xytext=(83, dense[50] + 0.018),
                color=C_DENSE, fontsize=10.5, ha="center")

    # K=5 solid, stops @50
    k5_x = sorted(k5)
    k5_y = [k5[x] for x in k5_x]
    ax.plot(k5_x, k5_y, color=C_K5, lw=2.4, marker="s", ms=7,
            label="K=5 EMA merger (stable)", zorder=5)
    ax.annotate("stops @50\n(pre-spiral)", xy=(50, k5[50]),
                xytext=(54, k5[50] - 0.005), color=C_K5, fontsize=10,
                ha="left", va="top")

    # K=20 solid, the collapse (4 pts)
    k20_x = sorted(k20)
    k20_y = [k20[x] for x in k20_x]
    ax.plot(k20_x, k20_y, color=C_K20, lw=2.6, marker="D", ms=7.5,
            label="K=20 EMA merger (collapses)", zorder=6)
    ax.annotate("collapse", xy=(100, k20[100]), xytext=(92, k20[100] - 0.028),
                color=C_K20, fontsize=11, fontweight="bold", ha="center",
                arrowprops=dict(arrowstyle="->", color=C_K20, lw=1.4))

    # no-merger floor
    ax.axhline(floor, color=C_GREY, lw=1.4, ls=":")
    ax.annotate(f"no-merger floor ({floor:.4f})", xy=(7, floor),
                xytext=(7, floor + 0.006), color=C_GREY, fontsize=10, va="bottom")

    # length-spiral ignition marker (~step 93)
    ax.axvline(93, color=C_K20, lw=1.1, ls="--", alpha=0.55)
    ax.annotate("length spiral\nignites (~step 93)", xy=(93, 0.43),
                xytext=(93, 0.435), color=C_K20, fontsize=9.5, ha="center",
                va="bottom", alpha=0.9)

    ax.set_xlim(0, 104)
    ax.set_ylim(0.40, 0.80)
    ax.set_xlabel("training step")
    ax.set_ylabel("GSM8K validation accuracy (greedy mean@1)")
    ax.set_title("Same EMA method: stable at K=5, collapses at K=20",
                 fontweight="bold")
    ax.legend(loc="lower left", frameon=True, framealpha=0.95, edgecolor="#cccccc")
    fig.tight_layout()
    out = FIGS / "fig1_val_curves.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


# ============================================================
# FIGURE 2 — collapse mechanism (3 stacked panels, K=20 only)
# ============================================================
def fig2():
    step = df["step"].values
    rlen = df["response_length/mean"].values
    ent = df["actor/entropy"].values
    cf = df["actor/pg_clipfrac"].values

    fig, axes = plt.subplots(3, 1, figsize=(8.0, 9.4), sharex=True,
                             constrained_layout=True)
    axT, axM, axB = axes

    # --- Top: response length ---
    axT.plot(step, rlen, color=C_K20, lw=1.8)
    axT.fill_between(step, rlen, rlen.min(), color=C_K20, alpha=0.07, lw=0)
    axT.set_ylabel("response length\n(mean tokens)")
    axT.annotate(
        "back-half ratchet:\n251 → 683 tokens over steps ~93–100",
        xy=(100, rlen[-1]), xytext=(62, rlen.max() * 0.78),
        color=C_K20, fontsize=10.5, ha="left",
        arrowprops=dict(arrowstyle="->", color=C_K20, lw=1.3))
    axT.set_ylim(rlen.min() - 20, rlen.max() * 1.10)

    # --- Middle: entropy ---
    axM.plot(step, ent, color="#7a4fb0", lw=1.8)
    axM.set_ylabel("policy entropy\n(nats)")
    axM.annotate(
        "entropy follows length —\na follower, not the trigger\n(5.78 → 0.42)",
        xy=(96, ent[95]), xytext=(40, 3.6),
        color="#5a3a86", fontsize=10.5, ha="left",
        arrowprops=dict(arrowstyle="->", color="#7a4fb0", lw=1.3))
    axM.set_ylim(0, 6.4)

    # --- Bottom: clipfrac with Q-refresh markers ---
    refresh = list(range(10, 101, 10))
    for rs in refresh:
        axB.axvline(rs, color=C_GREY, lw=0.8, ls=":", alpha=0.55, zorder=1)
    axB.plot(step, cf, color="#c8771f", lw=1.6, zorder=3)
    # highlight refresh-step values
    rmask = np.isin(step, refresh)
    axB.scatter(step[rmask], cf[rmask], color="#a32d2d", s=34, zorder=5,
                label="Q-refresh step (every 10)")
    axB.set_ylabel("PG clip fraction")
    axB.set_xlabel("training step (K=20 run, EXP-37)")
    axB.annotate(
        "spikes at every Q-refresh\n(cadence fingerprint:"
        " ~0.10–0.19 vs ~0.004–0.05 between)",
        xy=(60, cf[59]), xytext=(30, 0.205),
        color="#8a3d10", fontsize=10, ha="left",
        arrowprops=dict(arrowstyle="->", color="#c8771f", lw=1.2))
    axB.set_ylim(0, 0.31)
    axB.legend(loc="upper left", frameon=True, framealpha=0.95,
               edgecolor="#cccccc", fontsize=9.5)

    axB.set_xlim(1, 100)
    fig.suptitle(
        "The K=20 back-half collapse: length ratchet → entropy follows;\n"
        "clipfrac spikes mark Q refreshes",
        fontsize=13.5, fontweight="bold")
    out = FIGS / "fig2_collapse_mechanism.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


# ============================================================
# FIGURE 3 — staleness schematic (conceptual / illustrative)
# ============================================================
def fig3():
    fig, ax = plt.subplots(figsize=(8.0, 5.2))

    K = np.linspace(0, 26, 400)
    # illustrative components (arbitrary units)
    a = 0.10       # weight-drift slope
    b = 0.012      # off-policy scale
    c = 0.16       # off-policy exponential rate
    drift = a * K                          # linear weight-drift
    offpol = b * K * np.exp(c * K)         # super-linear distribution-shift
    total = drift + offpol

    # ignition threshold (illustrative)
    thresh = 2.2

    # stable / ignition shaded regions
    ax.axvspan(3.5, 6.5, color="#2e8b57", alpha=0.10, lw=0)
    ax.axvspan(17, 23, color="#a32d2d", alpha=0.10, lw=0)

    ax.plot(K, drift, color=C_GREY, lw=2.0, ls="--",
            label=r"weight-drift  $\propto K$  (linear)")
    ax.plot(K, offpol, color="#c8771f", lw=2.0, ls="-.",
            label=r"off-policy shift  $\propto K\,e^{cK}$  (super-linear)")
    ax.plot(K, total, color="#222222", lw=3.0,
            label=r"total staleness error  $\Vert e_K\Vert$")

    # ignition threshold line
    ax.axhline(thresh, color=C_K20, lw=1.3, ls=":")
    ax.annotate("ignition threshold", xy=(0.6, thresh), xytext=(0.6, thresh + 0.12),
                color=C_K20, fontsize=10, va="bottom")

    # K=5 and K=20 markers
    for kk, col, lab in [(5, C_K5, "K=5"), (20, C_K20, "K=20")]:
        yk = a * kk + b * kk * np.exp(c * kk)
        ax.axvline(kk, color=col, lw=1.3, ls="-", alpha=0.7)
        ax.scatter([kk], [yk], color=col, s=55, zorder=6, edgecolor="white", lw=1.0)
        ax.annotate(lab, xy=(kk, yk), xytext=(kk + 0.4, yk + 0.18),
                    color=col, fontsize=11.5, fontweight="bold")

    ax.set_xlim(0, 25)
    # cap just above the K=20 total so K=5 / threshold stay legible
    y_k20 = a * 20 + b * 20 * np.exp(c * 20)
    ytop = y_k20 * 1.55
    ax.set_ylim(0, ytop)

    # region labels (placed relative to new ylim)
    ax.text(5, ytop * 0.50, "sub-critical\n(stable)", color="#1f6b3b",
            fontsize=10.5, ha="center", va="center", fontweight="bold")
    ax.text(20, ytop * 0.40, "super-critical\n(ignition)", color=C_K20,
            fontsize=10.5, ha="center", va="center", fontweight="bold")
    ax.set_xlabel("anchor latency  K  (optimizer ticks between Q refreshes)")
    ax.set_ylabel(r"staleness error $\Vert e_K\Vert$  (arbitrary units)")
    ax.set_title(
        "Why K=20 crosses the ignition threshold:\n"
        "staleness error grows super-linearly in K",
        fontweight="bold")
    ax.legend(loc="upper left", frameon=True, framealpha=0.95, edgecolor="#cccccc")
    # schematic disclaimer
    ax.text(0.985, 0.02,
            "Schematic / illustrative — axes are conceptual, not measured",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, style="italic", color="#666666",
            bbox=dict(boxstyle="round,pad=0.3", fc="#f4f4f4", ec="#cccccc", lw=0.8))
    fig.tight_layout()
    out = FIGS / "fig3_staleness_schematic.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    o1 = fig1()
    o2 = fig2()
    o3 = fig3()
    for o in (o1, o2, o3):
        print(o, o.stat().st_size)
