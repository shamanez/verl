"""Plots for the gap-sensitivity sweep (gap_sweep.json).

  1. skill_vs_gap.png    - pooled forecast skill vs source gap G, per window W.
  2. persist_vs_gap.png  - update-direction persistence (global cosine of successive
                           gap-spaced deltas) and mean per-tensor direction cosine vs G.
                           This is the mechanism: skill tracks persistence.

Usage: python make_gap_plots.py --in_dir <outputs> --out_dir <plots>
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CW = {2: "#71b7ff", 4: "#c4a7ff", 3: "#6ee7a2", 6: "#ffd166"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    S = json.load(open(os.path.join(args.in_dir, "gap_sweep.json")))["summary"]

    byW = defaultdict(list)
    for r in S:
        byW[r["W"]].append(r)
    for W in byW:
        byW[W].sort(key=lambda r: r["gap"])

    # 1. skill vs gap
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for W in sorted(byW):
        rs = byW[W]
        gaps = [r["gap"] for r in rs]
        sk = [r["pooled_skill"] for r in rs]
        er = [r["pooled_skill_std"] for r in rs]
        ax.errorbar(gaps, sk, yerr=er, marker="o", capsize=3, label=f"W={W}", color=CW.get(W))
    ax.axhline(0, color="#888", lw=1, ls=":")
    ax.set_xlabel("source gap G (steps between checkpoints)")
    ax.set_ylabel("forecast skill (pooled)")
    ax.set_title("Skill depends on gap, not window: pooled skill vs G")
    ax.set_xscale("log")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "skill_vs_gap.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("wrote skill_vs_gap.png")

    # 2. persistence + direction cosine vs gap (use W=2 secant)
    rs = byW.get(2) or next(iter(byW.values()))
    # persistence is only defined where the pre-anchor step exists; drop None points
    pgaps = [r["gap"] for r in rs if r.get("persist_cos") is not None]
    persist = [r["persist_cos"] for r in rs if r.get("persist_cos") is not None]
    gaps = [r["gap"] for r in rs]
    dcos = [r["macro_cos"] for r in rs]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(pgaps, persist, "o-", color="#71b7ff", label="delta persistence (global cos of successive deltas)")
    ax.plot(gaps, dcos, "s--", color="#6ee7a2", label="mean per-tensor update direction cosine")
    ax.axhline(0, color="#888", lw=1, ls=":")
    ax.axhline(0.906, color="#ffd166", lw=1.2, ls="-.", label="live compressed run (cos 0.91)")
    ax.set_xlabel("source gap G (steps between checkpoints)")
    ax.set_ylabel("cosine")
    ax.set_ylim(-0.08, 1.0)
    ax.set_title("Dense updates stay near-orthogonal at every gap (vs live cos 0.91)")
    ax.set_xscale("log")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "persist_vs_gap.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("wrote persist_vs_gap.png")


if __name__ == "__main__":
    main()
