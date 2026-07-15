"""Plot for the consecutive-checkpoint COUNT sweep (does adding more consecutive
checkpoints help our pinned rank-1 projector?).

Reads report_digest.json produced by summarize_results.py on the outputs_consec run
(rank1_relex, gap=1, horizon=1, windows 2/4/8/10) and plots pooled forecast skill vs
window W, with the mean per-tensor direction cosine on a twin axis.

Usage: python make_count_plot.py --digest <outputs_consec>/report_digest.json --out <plots>/count_sweep.png
"""
from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--digest", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    S = json.load(open(args.digest))["scalars"]

    Ws, sk, err, cos = [], [], [], []
    for W in sorted({int(k.split("|")[1][1:]) for k in S if k.startswith("rank1_relex|")}):
        k = f"rank1_relex|W{W}|h1|r1"
        if k not in S:
            continue
        Ws.append(W)
        sk.append(S[k]["pooled_skill"])
        err.append(S[k].get("pooled_skill_std") or 0.0)
        cos.append(S[k]["macro_cos"])

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.errorbar(Ws, sk, yerr=err, marker="o", capsize=3, color="#71b7ff", label="forecast skill (pooled)")
    ax.axhline(0, color="#888", lw=1, ls=":")
    ax.set_xlabel("window W (number of CONSECUTIVE checkpoints, stride 1)")
    ax.set_ylabel("forecast skill (pooled)", color="#71b7ff")
    ax.tick_params(axis="y", labelcolor="#71b7ff")
    ax.set_xticks(Ws)
    ax2 = ax.twinx()
    ax2.plot(Ws, cos, marker="s", ls="--", color="#6ee7a2", label="update direction cosine (macro)")
    ax2.set_ylabel("mean per-tensor direction cosine", color="#6ee7a2")
    ax2.tick_params(axis="y", labelcolor="#6ee7a2")
    ax.set_title("Does adding consecutive checkpoints help our rank-1 projector? (gap 1, horizon 1)")
    ax.grid(alpha=0.3)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="best")
    fig.tight_layout()
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
