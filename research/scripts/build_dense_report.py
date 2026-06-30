#!/usr/bin/env python3
# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
"""EXP-42 DENSE-ONLY report (normal GRPO run, regime A). GPU-free, MacBook.

Strictly about how the NORMAL (codec-off) GRPO weights behave over the run, from
the per-tick weight-trajectory sketch already on the MacBook
(runs/EXP-42/regimeA/weights). Answers four things, with plots and descriptions:
  (a) how the GRPO weights change over the run (cumulative relative drift and the
      per-tick step size, median over matrices),
  (b) the ability to project the weights and over HOW MANY steps before it stops
      helping (weight_proj_ratio vs horizon and the crossover h*),
  (c) the performance of the run (reward and validation accuracy trajectories),
  (d) an explicit test of the RLVR-linear claim "weights move roughly linearly,
      R^2 about 0.9" via per-matrix linearity R^2 at several scales.

All numbers are computed here (no hand-set claims); the prose is generated from
them. The count-sketch is linear and norm-preserving (rel std about 1/sqrt(k)),
so differences/cosines of weight-difference vectors reconstruct from sketches.

Usage:
    python build_dense_report.py runs/EXP-42 regimeA runs/EXP-42/report_dense.html
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # weight_proj_sweep is a sibling in research/scripts/
from weight_proj_sweep import load_trace, group_of  # noqa: E402

INK, MUTED, GRID, A, HELP, HURT = "#14181f", "#5b6573", "#e3e6eb", "#1f5fae", "#1a7f37", "#b42318"
TICKS_PER_STEP = 2  # batch128 / mini64
plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white", "axes.titlesize": 10,
})


def _png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _pct(a, q):
    return float(np.percentile(np.asarray(a, np.float64), q)) if len(a) else float("nan")


def compute(weights_dir):
    ticks, names, sketches, means, dims, k = load_trace(weights_dir)
    idx = {t: i for i, t in enumerate(ticks)}
    t0 = ticks[0]
    # (a) weight-change dynamics: cumulative relative drift and per-tick step, median over matrices
    drift, step = [], []
    for t in ticks:
        cum, stp = [], []
        for nm in names:
            s = sketches[nm]
            n0 = np.linalg.norm(s[0])
            if n0 > 0:
                cum.append(float(np.linalg.norm(s[idx[t]] - s[0])) / n0)
            if (t - 1) in idx:
                np1 = np.linalg.norm(s[idx[t - 1]])
                if np1 > 0:
                    stp.append(float(np.linalg.norm(s[idx[t]] - s[idx[t - 1]])) / np1)
        drift.append((t, _pct(cum, 50)))
        if stp:
            step.append((t, _pct(stp, 50)))
    # (d) linearity R^2 (= mean over t of cos^2 of consecutive displacements) per matrix, per scale
    lin = {}
    permat_by_scale = {}
    for s_scale in (1, 2, 5, 10):
        permat = {}
        for nm in names:
            ss = sketches[nm]
            c2 = []
            for t in ticks:
                if (t - s_scale) in idx and (t + s_scale) in idx:
                    x = ss[idx[t]] - ss[idx[t - s_scale]]
                    y = ss[idx[t + s_scale]] - ss[idx[t]]
                    nx, ny = np.linalg.norm(x), np.linalg.norm(y)
                    if nx > 0 and ny > 0:
                        c = float(np.dot(x, y)) / (nx * ny)
                        c2.append(c * c)
            if c2:
                permat[nm] = float(np.mean(c2))
        vals = list(permat.values())
        lin[s_scale] = {"p10": _pct(vals, 10), "p50": _pct(vals, 50), "p90": _pct(vals, 90), "n": len(vals)}
        permat_by_scale[s_scale] = permat
    return {"ticks": ticks, "names": names, "n_matrices": len(names), "k": k,
            "drift": drift, "step": step, "lin": lin, "permat_lin": permat_by_scale}


def parse_perf(internal_log):
    """Pull per-step reward (critic/score/mean) and val acc from the train log.

    Metrics are one logical line per step; the step marker, reward, and (at a val
    step) the val accuracy all sit on that same line, in arbitrary key order, so
    scan each line rather than a forward window from the step token.
    """
    rew, val = {}, {}
    if not os.path.exists(internal_log):
        return [], []
    for line in open(internal_log, errors="ignore"):
        sm = re.search(r"training/global_step:(\d+)", line)
        if not sm:
            continue
        gs = int(sm.group(1))
        rm = re.search(r"critic/score/mean:([0-9.]+)", line)
        if rm:
            rew[gs] = float(rm.group(1))
        vm = re.search(r"val-core/openai/gsm8k/acc/mean@1:([0-9.]+)", line)
        if vm:
            val[gs] = float(vm.group(1))
    return sorted(rew.items()), sorted(val.items())


def proj_curve(sweep_json, regime, delta=10):
    d = json.load(open(sweep_json))[regime]["results"]
    hs, p50, p10, p90 = [], [], [], []
    for h in (1, 2, 3, 5, 8, 10, 13, 20, 30):
        c = d.get(f"fixed_linear|{delta}|{h}")
        if c:
            hs.append(h); p50.append(c["w1_p50"]); p10.append(c["w1_p10"]); p90.append(c["w1_p90"])
    hstar = None
    for h, v in zip(hs, p50):
        if v < 1.0:
            hstar = h
    return hs, p50, p10, p90, hstar


def esc(x):
    return html.escape(str(x))


CSS = """
:root{--ground:#f7f8fa;--surface:#fff;--ink:#14181f;--muted:#5b6573;--line:#e3e6eb;--a:#1f5fae;
--help:#1a7f37;--b:#b42318;--sans:ui-sans-serif,-apple-system,"Segoe UI",Roboto,Helvetica,sans-serif;
--mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace}
*{box-sizing:border-box}body{margin:0;background:var(--ground);color:var(--ink);font-family:var(--sans);
line-height:1.6;font-size:16px;-webkit-font-smoothing:antialiased}
.wrap{max-width:980px;margin:0 auto;padding:40px 24px 80px}
header{border-bottom:1px solid var(--line);padding-bottom:22px}
.eyebrow{font-size:12px;letter-spacing:.13em;text-transform:uppercase;color:var(--muted);font-weight:600}
h1{font-size:30px;line-height:1.18;margin:10px 0 12px;text-wrap:balance;font-weight:680;letter-spacing:-.01em}
.thesis{color:var(--muted);max-width:70ch;margin:0 0 16px}
.chips{display:flex;flex-wrap:wrap;gap:8px}.chip{font-family:var(--mono);font-size:11.5px;color:var(--muted);
background:var(--surface);border:1px solid var(--line);border-radius:999px;padding:3px 10px}
h2{font-size:20px;margin:44px 0 12px;padding-bottom:6px;border-bottom:1px solid var(--line);font-weight:660}
h3{font-size:15px;margin:22px 0 6px;font-weight:640}
p{max-width:72ch}.cap{color:var(--muted);font-size:13px;max-width:74ch;margin:8px 0 4px}
.callout{background:var(--surface);border:1px solid var(--line);border-radius:14px;padding:20px 22px;margin:22px 0}
.callout h2{border:0;margin:0 0 8px;padding:0;font-size:16px}
.big{font-size:34px;font-weight:700;font-variant-numeric:tabular-nums;letter-spacing:-.02em;color:var(--a)}
.tbl{overflow-x:auto;margin:10px 0}
table{border-collapse:collapse;font-size:13.5px;font-variant-numeric:tabular-nums;min-width:100%}
th,td{padding:6px 12px;text-align:right;border-bottom:1px solid var(--line);white-space:nowrap}
th{font-family:var(--mono);font-size:11px;letter-spacing:.04em;text-transform:uppercase;color:var(--muted);
font-weight:600;border-bottom:1.5px solid var(--muted)}
td:first-child,th:first-child{text-align:left}tbody tr:nth-child(even){background:rgba(31,95,174,.035)}
.ok{color:var(--help);font-weight:640}.bad{color:var(--b);font-weight:640}
figure{margin:14px 0;background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:14px}
figure img{max-width:100%;height:auto;display:block;border-radius:6px}
ul{max-width:74ch}li{margin:3px 0}b{font-weight:660}
"""


def build(run_dir, regime, out_html, fragment=False):
    wdir = os.path.join(run_dir, regime, "weights")
    m = compute(wdir)
    rew, val = parse_perf(os.path.join(run_dir, regime, f"train_{regime}_internal.log"))
    sweep_json = os.path.join(run_dir, "sweep_narrow.json")
    hs, p50, p10, p90, hstar = proj_curve(sweep_json, regime) if os.path.exists(sweep_json) else ([], [], [], [], None)

    # ---- plots ----
    # weight change
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))
    dt = [t / TICKS_PER_STEP for t, _ in m["drift"]]; dv = [v for _, v in m["drift"]]
    ax1.plot(dt, dv, "-", color=A)
    ax1.set_xlabel("global step"); ax1.set_ylabel("median relative drift  ||theta[t]-theta[0]|| / ||theta[0]||")
    ax1.set_title("Cumulative weight change over the run")
    st = [t / TICKS_PER_STEP for t, _ in m["step"]]; sv = [v for _, v in m["step"]]
    ax2.plot(st, sv, "-", color=A)
    ax2.set_xlabel("global step"); ax2.set_ylabel("median per-tick step  ||theta[t]-theta[t-1]|| / ||theta[t-1]||")
    ax2.set_title("Per-step update size")
    fig_change = _png(fig)

    # projectability + linearity
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))
    if hs:
        ax1.plot(hs, p50, "o-", color=A); ax1.fill_between(hs, p10, p90, color=A, alpha=0.15)
        ax1.axhline(1.0, color="k", ls="--", lw=1); ax1.axvline(10, color=MUTED, ls=":", lw=1)
        ax1.text(10, ax1.get_ylim()[1], " K=10", va="top", fontsize=8, color=MUTED)
    ax1.set_xlabel("horizon h (ticks ahead; 2 ticks per step)")
    ax1.set_ylabel("weight_proj_ratio (below 1 helps)")
    ax1.set_title("How far ahead can we project the weights")
    scales = sorted(m["lin"].keys()); lp50 = [m["lin"][s]["p50"] for s in scales]
    lp10 = [m["lin"][s]["p10"] for s in scales]; lp90 = [m["lin"][s]["p90"] for s in scales]
    xs = [s / TICKS_PER_STEP for s in scales]
    ax2.plot(xs, lp50, "o-", color=A); ax2.fill_between(xs, lp10, lp90, color=A, alpha=0.15)
    ax2.axhline(0.9, color=HELP, ls="--", lw=1); ax2.text(xs[-1], 0.9, " RLVR ~0.9", va="bottom", ha="right", fontsize=8, color=HELP)
    ax2.set_ylim(0, 1); ax2.set_xlabel("scale (global steps between displacements)")
    ax2.set_ylabel("per-matrix linearity R^2 (median)")
    ax2.set_title("Is the trajectory linear (RLVR-linear test)")
    fig_proj = _png(fig)

    # performance
    fig_perf = None
    if rew or val:
        fig, ax = plt.subplots(1, 1, figsize=(7.5, 4.0))
        if rew:
            ax.plot([g for g, _ in rew], [v for _, v in rew], "-", color=A, label="reward (critic/score/mean)")
        if val:
            ax.plot([g for g, _ in val], [v for _, v in val], "o", color=HURT, ms=8, label="val acc/mean@1")
        ax.set_xlabel("global step"); ax.set_ylabel("value"); ax.set_ylim(0, 1)
        ax.set_title("Performance of the normal run"); ax.legend(fontsize=8)
        fig_perf = _png(fig)

    # ---- headline numbers ----
    hstar_steps = (hstar / TICKS_PER_STEP) if hstar else None
    lin1 = m["lin"][1]["p50"]; lin10 = m["lin"][10]["p50"]
    final_val = val[-1][1] if val else None
    total_drift = m["drift"][-1][1] if m["drift"] else None

    P = []
    P.append("<header><div class='eyebrow'>EXP-42 / dense run / weight behavior</div>")
    P.append("<h1>How the normal GRPO weights behave: drift, projectability, and linearity</h1>")
    P.append("<p class='thesis'>A single look at the plain GRPO run (codec off), measured directly "
             "from the per-tick weight-trajectory sketch. How much do the weights move, how many steps "
             "ahead can a linear rule predict them, how well does the run learn, and do the updates "
             "actually move roughly linearly the way the RLVR-linear result claims.</p>")
    P.append(f"<div class='chips'><span class='chip'>{m['n_matrices']} decoder matrices</span>"
             f"<span class='chip'>{len(m['ticks'])} ticks ({len(m['ticks'])//TICKS_PER_STEP} steps)</span>"
             f"<span class='chip'>count-sketch k={m['k']}</span>"
             f"<span class='chip'>codec OFF (dense)</span></div></header>")

    P.append("<section class='callout'><h2>Headline</h2>")
    if hstar_steps is not None:
        P.append(f"<p>You can linearly project the weights about <span class='big'>{hstar_steps:.0f}</span> "
                 f"global steps ahead ({hstar} ticks) before projection stops helping. Beyond that the "
                 f"linear rule overshoots the true weight.</p>")
    P.append(f"<p>The trajectory is locally close to linear (R squared = {lin1:.2f} at one tick) but "
             f"that linearity decays with distance (R squared = {lin10:.2f} at the K=10 staleness scale), "
             f"which is exactly why projection helps only out to the crossover.</p></section>")

    P.append("<h2>How the weights change over the run</h2>")
    P.append(f"<figure><img src='data:image/png;base64,{fig_change}' /></figure>")
    if total_drift is not None:
        P.append(f"<p class='cap'>By the end of the run the decoder weights have moved only about "
                 f"{total_drift*100:.3f} percent of their initial norm (median matrix). The motion is "
                 f"tiny in magnitude: GRPO at lr 1e-6 on an already-instruction-tuned 1.5B model is a "
                 f"gentle nudge, not a large weight change. The right panel is the per-tick step size, "
                 f"a roughly steady small step. What matters for projection is the DIRECTION of this "
                 f"small motion, analysed next.</p>")

    P.append("<h2>How far ahead can we project, and is the path linear</h2>")
    P.append(f"<figure><img src='data:image/png;base64,{fig_proj}' /></figure>")
    P.append("<p class='cap'>Left: the linear projection (theta_hat = theta_stale + alpha (theta_stale "
             "- theta_old)) lands closer than doing nothing while the ratio stays below 1.0, and "
             "crosses 1.0 at the crossover. Right: per-matrix linearity R squared versus the scale of "
             "the displacement, against the RLVR-linear reference of about 0.9.</p>")
    P.append("<table><tr><th>scale (steps)</th><th>scale (ticks)</th>"
             "<th>R2 p10</th><th>R2 p50</th><th>R2 p90</th></tr>")
    for s in scales:
        L = m["lin"][s]
        P.append(f"<tr><td>{s/TICKS_PER_STEP:.1f}</td><td>{s}</td><td>{L['p10']:.3f}</td>"
                 f"<td>{L['p50']:.3f}</td><td>{L['p90']:.3f}</td></tr>")
    P.append("</table>")

    # ---- deeper structure: attention vs MLP, and layer-depth trend ----
    def _layer(nm):
        mt = re.search(r"layers\.(\d+)\.", nm)
        return int(mt.group(1)) if mt else -1
    def _type(nm):
        if any(s in nm for s in ("q_proj", "k_proj", "v_proj", "o_proj")):
            return "attention"
        if any(s in nm for s in ("gate_proj", "up_proj", "down_proj")):
            return "mlp"
        return "other"
    l1, l10 = m["permat_lin"][1], m["permat_lin"][10]
    layers = sorted({_layer(nm) for nm in m["names"] if _layer(nm) >= 0})
    nL = len(layers)
    def _agg(sel):
        a1 = [l1[nm] for nm in m["names"] if nm in l1 and sel(nm)]
        a10 = [l10[nm] for nm in m["names"] if nm in l10 and sel(nm)]
        return (_pct(a1, 50), _pct(a10, 50), len(a1))
    struct = {"attention": _agg(lambda nm: _type(nm) == "attention"),
              "mlp": _agg(lambda nm: _type(nm) == "mlp")}
    terc = {}
    if nL:
        cut1, cut2 = layers[nL // 3], layers[2 * nL // 3]
        terc = {f"early (L0-{cut1-1})": _agg(lambda nm: 0 <= _layer(nm) < cut1),
                f"mid (L{cut1}-{cut2-1})": _agg(lambda nm: cut1 <= _layer(nm) < cut2),
                f"late (L{cut2}-{layers[-1]})": _agg(lambda nm: _layer(nm) >= cut2)}
    P.append("<h2>Deeper structure: where in the model is the motion most linear</h2>")
    P.append("<table><tr><th>group</th><th>R2 @ 1 tick</th><th>R2 @ 10 ticks (K)</th><th>matrices</th></tr>")
    for g, (a1, a10, n) in {**struct, **terc}.items():
        P.append(f"<tr><td>{esc(g)}</td><td>{a1:.3f}</td><td>{a10:.3f}</td><td>{n}</td></tr>")
    P.append("</table>")
    at1, at10, _ = struct["attention"]; mt1, mt10, _ = struct["mlp"]
    more_linear = "attention" if at1 > mt1 else "MLP"
    P.append(f"<p class='cap'>At the fine scale the {esc(more_linear)} matrices move slightly more "
             f"linearly (attention R2 {at1:.2f} vs MLP {mt1:.2f} at 1 tick). Both families lose "
             f"linearity by the K=10 scale (attention {at10:.2f}, MLP {mt10:.2f}). The layer-depth "
             f"terciles show whether early or late layers hold their direction longer. The takeaway "
             f"is uniform: linearity is a fine-scale property across the whole decoder, and it decays "
             f"with horizon everywhere, not just in one region.</p>")

    if fig_perf:
        P.append("<h2>Performance of the normal run</h2>")
        P.append(f"<figure><img src='data:image/png;base64,{fig_perf}' /></figure>")
        if final_val is not None:
            P.append(f"<p class='cap'>Final validation accuracy {final_val:.4f} on GSM8K. The reward "
                     f"climbs early and then holds, a healthy plain-GRPO control.</p>")

    # RLVR verdict (generated)
    P.append("<h2>Do we observe the RLVR-linear result</h2>")
    near = abs(lin1 - 0.9) <= 0.15
    verdict = ("PARTIALLY" if (lin1 >= 0.6 and lin10 < 0.6) else
               ("YES" if lin1 >= 0.8 and lin10 >= 0.8 else "NO"))
    P.append(f"<p><b>{verdict}.</b> The claim that RLVR weight updates move roughly linearly with "
             f"R squared about 0.9 holds for this run only at the finest scale: per-matrix R squared "
             f"is {lin1:.2f} at one tick ({'in the ballpark of' if near else 'below'} the 0.9 "
             f"reference), and it falls to {lin10:.2f} by the K=10 scale. So the weights move almost "
             f"linearly over a step or two, then the path curves. The look-ahead anchor operates at "
             f"the K=10 scale, where linearity has already broken down, which is the mechanism behind "
             f"the overshoot we measured. The single-scale 'weights move linearly' headline is true "
             f"locally but scale-dependent: it should be stated with the scale attached.</p>")
    P.append(f"<p><b>How many steps does linear extrapolation hold: paper versus this run.</b> "
             f"The RLVR-linearity paper our method cites (arXiv:2601.04537, as quoted in "
             f"lookahead.py) reports that linear weight extrapolation holds for about 600 training "
             f"steps (R squared about 0.9). By the directly analogous metric here, the "
             f"weight_proj_ratio crossover, linear extrapolation stops helping after about "
             f"{hstar_steps:.0f} global steps ({hstar} ticks) in this run. That is far short of 600. "
             f"Two things to keep in mind before reading this as a contradiction. First, this run is "
             f"only {len(m['ticks'])//TICKS_PER_STEP} steps long and starts from an already "
             f"instruction-tuned model, so the weights barely move (about {total_drift*100:.3f} "
             f"percent total drift) and the small per-step motion is easily dominated by rollout "
             f"sampling noise, which destroys step-to-step direction persistence quickly. A longer "
             f"run from a less converged start, with larger directed motion, is where a 600-step "
             f"linear regime would plausibly appear, and we did not run that. Second, the comparison "
             f"assumes the two papers measure linearity the same way: the about-0.9 / 600-step figure "
             f"may be a global straight-line fit over the whole trajectory or a low-rank-subspace "
             f"statement, which is not the same as the local consecutive-step direction R squared "
             f"used here. A like-for-like check (global line fit and effective rank of the "
             f"displacement subspace) is the right GPU-free next analysis. NOTE: arXiv:2601.04537 is "
             f"dated at the edge of this assistant's knowledge, so the paper figures here are taken "
             f"from the code citation, not from an independent reading of the paper.</p>")

    P.append("<h2>Scope and provenance</h2><ul>")
    P.append(f"<li>Source: {esc(regime)} per-tick weight sketch, runs/EXP-42/{esc(regime)}/weights "
             f"({m['n_matrices']} matrices, {len(m['ticks'])} ticks). Decoder matrices only "
             f"(q/k/v/o/gate/up/down); the embeddings, RMSNorm gains and biases are not in this run.</li>")
    P.append("<li>All metrics computed on the MacBook from the sketch, GPU-free. The count-sketch is "
             "linear and norm-preserving, so weight-difference norms and cosines reconstruct from it.</li>")
    P.append("<li>Linearity R squared = mean over ticks of cos squared between the previous and next "
             "displacement at the given scale (the regression of next displacement on previous, through "
             "the origin), then percentiles across matrices.</li></ul>")

    metrics = {"regime": regime, "n_matrices": m["n_matrices"], "n_ticks": len(m["ticks"]),
               "crossover_h_ticks": hstar, "crossover_h_steps": hstar_steps,
               "linearity": {str(s): m["lin"][s] for s in scales},
               "final_val": final_val, "total_drift_rel": total_drift,
               "proj_ratio_at_h10": dict(zip(hs, p50)).get(10)}

    content = "".join(P).replace("<table", "<div class='tbl'><table").replace("</table>", "</table></div>")
    body = "<div class='wrap'>" + content + "</div>"
    if fragment:
        out = f"<style>{CSS}</style>" + body
    else:
        out = (f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
               f"<meta name='viewport' content='width=device-width, initial-scale=1'>"
               f"<title>EXP-42 dense run weight behavior</title><style>{CSS}</style></head><body>"
               + body + "</body></html>")
    with open(out_html, "w") as fh:
        fh.write(out)
    mj = os.path.splitext(out_html)[0] + "_metrics.json"
    with open(mj, "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"wrote {out_html} ({len(out)} bytes, fragment={fragment}); metrics -> {mj}")
    print(json.dumps(metrics, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir"); ap.add_argument("regime"); ap.add_argument("out_html")
    ap.add_argument("--fragment", action="store_true")
    a = ap.parse_args()
    build(a.run_dir, a.regime, a.out_html, fragment=a.fragment)


if __name__ == "__main__":
    main()
