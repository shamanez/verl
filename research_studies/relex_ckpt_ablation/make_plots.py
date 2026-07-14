"""Turn the ablation outputs into the study's result plots + a results HTML.

Reads summary.json (per-combo aggregates, one per anchor instance) and
forecast_rows.csv (per-tensor rows), averages over anchor positions (mean +/- std),
and renders the plots the plan asks for:

  1. skill_vs_W.png        - forecast skill vs window W (per method), at horizon=1.
                             THE "how many checkpoints" answer.
  2. skill_vs_horizon.png  - skill vs horizon (current-fast=1 .. twice-fast=2 ..), per W.
  3. fit_vs_skill.png      - EVR & R^2 (in-window fit) vs forecast skill, per W:
                             shows "great fit, poor forecast".
  4. skill_by_type.png     - per tensor-type skill at a chosen combo (norms vs q_proj ..).
  5. method_compare.png    - rank1_relex(pin-to-latest) vs relex_from_base vs fixed_linear vs stale.
  6. rank_ablation.png     - skill vs SVD rank r (W>=3): is rank-1 enough?

Usage:  python make_plots.py --in_dir <out_from_runner> --out_dir <plots_dir>
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics as st
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

C = {"rank1_relex": "#71b7ff", "relex_from_base": "#c4a7ff", "fixed_linear": "#ffd166", "stale": "#ff8585"}


def load(in_dir):
    summ = json.load(open(os.path.join(in_dir, "summary.json")))["summaries"]
    rows = list(csv.DictReader(open(os.path.join(in_dir, "forecast_rows.csv"))))
    return summ, rows


def key(s, *fields):
    return tuple(s[f] for f in fields)


def agg_over_anchors(summ, metric="pooled_skill", scope="overall"):
    """(method,W,horizon,rank,strength) -> (mean, std, n) of overall[metric]."""
    buck = defaultdict(list)
    for s in summ:
        val = s[scope][metric] if scope == "overall" else None
        if val is None:
            continue
        buck[key(s, "method", "W", "horizon", "rank", "strength")].append(val)
    return {k: (st.mean(v), (st.pstdev(v) if len(v) > 1 else 0.0), len(v)) for k, v in buck.items()}


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def plot_skill_vs_W(summ, out, horizon=1, strength=1.0):
    for metric, tag in [("pooled_skill", "pooled"), ("macro_skill", "macro")]:
        a = agg_over_anchors(summ, metric)
        fig, ax = plt.subplots(figsize=(7.5, 5))
        for method in sorted({k[0] for k in a}):
            pts = sorted(
                (int(k[1]), m, s)
                for k, (m, s, _n) in a.items()
                if k[0] == method and int(k[2]) == horizon and float(k[4]) == strength and int(k[3]) == 1
            )
            if not pts:
                continue
            Ws = [p[0] for p in pts]
            ms = [p[1] for p in pts]
            ss = [p[2] for p in pts]
            ax.errorbar(Ws, ms, yerr=ss, marker="o", capsize=3, label=method, color=C.get(method))
        ax.axhline(0, color="#888", lw=1, ls=":")
        ax.set_xlabel("window W (number of source checkpoints)")
        ax.set_ylabel(f"forecast skill ({tag})   1 - proj_SSE/stale_SSE")
        ax.set_title(f"How many checkpoints? skill vs W  (horizon={horizon} gap, alpha={strength})")
        ax.legend()
        ax.grid(alpha=0.3)
        _save(fig, os.path.join(out, f"skill_vs_W_{tag}.png"))


def plot_skill_vs_horizon(summ, out, method="rank1_relex", strength=1.0):
    a = agg_over_anchors(summ, "pooled_skill")
    fig, ax = plt.subplots(figsize=(7.5, 5))
    Wset = sorted({int(k[1]) for k in a if k[0] == method})
    for W in Wset:
        pts = sorted(
            (int(k[2]), m, s)
            for k, (m, s, _n) in a.items()
            if k[0] == method and int(k[1]) == W and float(k[4]) == strength and int(k[3]) == 1
        )
        if not pts:
            continue
        hs = [p[0] for p in pts]
        ms = [p[1] for p in pts]
        ss = [p[2] for p in pts]
        ax.errorbar(hs, ms, yerr=ss, marker="o", capsize=3, label=f"W={W}")
    ax.axhline(0, color="#888", lw=1, ls=":")
    ax.set_xlabel("horizon (gaps ahead:  1 = current-fast,  2 = twice-a-fast, ...)")
    ax.set_ylabel("forecast skill (pooled)")
    ax.set_title(f"Firing the anchor less often: skill vs horizon  ({method})")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, os.path.join(out, "skill_vs_horizon.png"))


def plot_fit_vs_skill(summ, out, method="rank1_relex", horizon=1, strength=1.0):
    a_sk = agg_over_anchors(summ, "pooled_skill")
    a_evr = agg_over_anchors(summ, "evr_mean")
    a_r2 = agg_over_anchors(summ, "r2_mean")
    Ws, sk, evr, r2 = [], [], [], []
    for k, (m, _s, _n) in sorted(a_sk.items(), key=lambda kv: int(kv[0][1])):
        if k[0] != method or int(k[2]) != horizon or float(k[4]) != strength or int(k[3]) != 1:
            continue
        Ws.append(int(k[1]))
        sk.append(m)
        evr.append(a_evr.get(k, (None,))[0])
        r2.append(a_r2.get(k, (None,))[0])
    if not Ws:
        return
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(Ws, sk, "o-", color=C[method], label="forecast skill")
    if any(e is not None for e in evr):
        ax.plot(Ws, [e if e is not None else float("nan") for e in evr], "s--", color="#6ee7a2", label="in-window EVR")
    if any(x is not None for x in r2):
        ax.plot(Ws, [x if x is not None else float("nan") for x in r2], "^--", color="#ffd166", label="in-window R^2")
    ax.axhline(0, color="#888", lw=1, ls=":")
    ax.set_xlabel("window W")
    ax.set_ylabel("value")
    ax.set_title("The paradox: in-window fit (EVR/R^2) high, forecast skill can collapse")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, os.path.join(out, "fit_vs_skill.png"))


def plot_skill_by_type(summ, out, method="rank1_relex", W=2, horizon=1, rank=1, strength=1.0):
    """Energy-pooled per-type skill, averaged over anchors. Pooled (not the macro
    mean) so tiny-motion tensors like q/k/v biases - whose per-tensor skill explodes
    because their stale error is almost zero - do not swamp the plot."""
    buck = defaultdict(list)
    for s in summ:
        if (
            s["method"] == method
            and int(s["W"]) == W
            and int(s["horizon"]) == horizon
            and int(s["rank"]) == rank
            and abs(float(s["strength"]) - strength) < 1e-9
        ):
            for ttype, agg in s.get("by_type", {}).items():
                if agg and agg.get("pooled_skill") is not None:
                    buck[ttype].append(agg["pooled_skill"])
    if not buck:
        return
    types = sorted(buck, key=lambda t: -st.mean(buck[t]))
    means = [st.mean(buck[t]) for t in types]
    errs = [st.pstdev(buck[t]) if len(buck[t]) > 1 else 0.0 for t in types]
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#6ee7a2" if m > 0 else "#71b7ff" for m in means]
    ax.bar(types, means, yerr=errs, capsize=3, color=colors)
    ax.axhline(0, color="#888", lw=1)
    # focus the axis on the informative band; unstable tiny-motion types (post_ln)
    # can have huge error bars that would otherwise squash every other bar.
    lo = min(-1.6, min(means) - 0.2)
    ax.set_ylim(max(lo, -2.2), max(0.5, max(means) + 0.15))
    ax.set_ylabel("energy-pooled skill (per tensor type)")
    ax.set_title(f"Which tensor types does projection help? ({method}, W={W}, horizon={horizon})")
    plt.xticks(rotation=40, ha="right")
    ax.grid(alpha=0.3, axis="y")
    _save(fig, os.path.join(out, "skill_by_type.png"))


def plot_method_compare(summ, out, W=2, horizon=1, strength=1.0):
    a = agg_over_anchors(summ, "pooled_skill")
    methods, means, errs = [], [], []
    for method in ["rank1_relex", "relex_from_base", "fixed_linear"]:
        hits = [
            (m, s)
            for k, (m, s, _n) in a.items()
            if k[0] == method and int(k[1]) == W and int(k[2]) == horizon and float(k[4]) == strength and int(k[3]) == 1
        ]
        if hits:
            methods.append(method)
            means.append(hits[0][0])
            errs.append(hits[0][1])
    if not methods:
        return
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.bar(methods, means, yerr=errs, capsize=3, color=[C.get(m) for m in methods])
    ax.axhline(0, color="#888", lw=1)
    ax.set_ylabel("forecast skill (pooled)")
    ax.set_title(f"Method comparison at W={W}, horizon={horizon}  (stale baseline = skill 0)")
    plt.xticks(rotation=15)
    ax.grid(alpha=0.3, axis="y")
    _save(fig, os.path.join(out, "method_compare.png"))


def plot_rank_ablation(summ, out, method="rank1_relex", horizon=1, strength=1.0):
    a = agg_over_anchors(summ, "pooled_skill")
    ranks_present = sorted({int(k[3]) for k in a if k[0] == method})
    if len(ranks_present) < 2:
        return
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for W in sorted({int(k[1]) for k in a if k[0] == method and int(k[1]) >= 3}):
        pts = sorted(
            (int(k[3]), m, s)
            for k, (m, s, _n) in a.items()
            if k[0] == method and int(k[1]) == W and int(k[2]) == horizon and float(k[4]) == strength
        )
        if len(pts) < 2:
            continue
        rs = [p[0] for p in pts]
        ms = [p[1] for p in pts]
        ss = [p[2] for p in pts]
        ax.errorbar(rs, ms, yerr=ss, marker="o", capsize=3, label=f"W={W}")
    ax.set_xlabel("SVD rank r")
    ax.set_ylabel("forecast skill (pooled)")
    ax.set_title("Is rank-1 enough? skill vs rank r (W>=3)")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, os.path.join(out, "rank_ablation.png"))


def write_results_html(out):
    imgs = [
        f
        for f in (
            "skill_vs_W_pooled.png",
            "skill_vs_W_macro.png",
            "skill_vs_horizon.png",
            "fit_vs_skill.png",
            "method_compare.png",
            "skill_by_type.png",
            "rank_ablation.png",
        )
        if os.path.exists(os.path.join(out, f))
    ]
    body = "\n".join(
        f'<figure><img src="{f}" style="max-width:100%"><figcaption>{f}</figcaption></figure>' for f in imgs
    )
    html = (
        "<!doctype html><meta charset=utf-8><title>RELEX ckpt-ablation results</title>"
        "<style>body{background:#09111f;color:#eef4ff;font-family:system-ui;max-width:1000px;margin:2rem auto}"
        "figure{margin:2rem 0}figcaption{color:#aebbd0}</style>"
        "<h1>RELEX checkpoint-count ablation — results</h1>" + body
    )
    open(os.path.join(out, "results.html"), "w").write(html)
    print("wrote", os.path.join(out, "results.html"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    summ, rows = load(args.in_dir)
    plot_skill_vs_W(summ, args.out_dir)
    plot_skill_vs_horizon(summ, args.out_dir)
    plot_fit_vs_skill(summ, args.out_dir)
    plot_method_compare(summ, args.out_dir)
    plot_skill_by_type(summ, args.out_dir)
    plot_rank_ablation(summ, args.out_dir)
    write_results_html(args.out_dir)


if __name__ == "__main__":
    main()
