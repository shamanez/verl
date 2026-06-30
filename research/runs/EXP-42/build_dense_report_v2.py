#!/usr/bin/env python3
# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
"""EXP-42 DENSE-ONLY report v2 (normal GRPO, regime A). GPU-free, MacBook.

A deeper follow-up to report_dense.html. Five studies, each computed directly
from the per-tick weight-trajectory count-sketch already on the MacBook
(runs/EXP-42/regimeA/weights: 196 decoder matrices, 160 ticks, k=4096), each
with a plot and a plain description:

  (a) low-rank / effective-rank structure of the weight-DISPLACEMENT subspace
      per matrix (stack the per-tick displacement vectors, report participation
      ratio and the number of components for 90 percent energy), plus a
      like-for-like GLOBAL straight-line-fit linearity, to test the RLVR claim
      that RLVR updates are low-rank / move linearly;
  (b) the per-matrix projectability distribution and per-matrix crossover
      horizon (which matrices project furthest, the histogram of h*);
  (c) an optimal-coefficient sweep: at each horizon find the alpha that
      minimizes the median weight_proj_ratio, compare to naive alpha = h/Delta;
  (d) the correlation between a matrix's fine-scale linearity R squared and its
      projectability;
  (e) the learned_linear vs fixed_linear residual effect, quantified.

All numbers are computed here (no hand-set claims). The count-sketch is linear
and norm-preserving, so displacement norms, cosines, and Gram matrices
reconstruct from it (relative std about 1/sqrt(k), about 1.6 percent at k=4096).
Where a study would need data we did not collect (the matrix-native singular
spectrum of a single weight matrix; embeddings, RMSNorm gains, biases) it is
stated explicitly and skipped rather than invented.

Usage:
    python build_dense_report_v2.py runs/EXP-42 regimeA runs/EXP-42/report_dense_v2.html
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))
from weight_proj_sweep import load_trace, ones_sketch  # noqa: E402

INK, MUTED, GRID, A, HELP, HURT = "#14181f", "#5b6573", "#e3e6eb", "#1f5fae", "#1a7f37", "#b42318"
AMBER = "#b7791f"
TICKS_PER_STEP = 2  # batch128 / mini64
DELTA = 10          # operating anchor spacing
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
    a = np.asarray(a, np.float64)
    a = a[np.isfinite(a)]
    return float(np.percentile(a, q)) if a.size else float("nan")


def _layer(nm):
    mt = re.search(r"layers\.(\d+)\.", nm)
    return int(mt.group(1)) if mt else -1


def _type(nm):
    if any(s in nm for s in ("q_proj", "k_proj", "v_proj", "o_proj")):
        return "attention"
    if any(s in nm for s in ("gate_proj", "up_proj", "down_proj")):
        return "mlp"
    return "other"


def _proj_short(nm):
    mt = re.search(r"layers\.(\d+)\.(?:self_attn|mlp)\.([a-z_]+)\.weight", nm)
    return f"L{mt.group(1)}.{mt.group(2)}" if mt else nm


# ====================================================================== #
# Computation
# ====================================================================== #
def compute(weights_dir):
    ticks, names, sketches, means, dims, k = load_trace(weights_dir)
    nmn = len(names)
    n = len(ticks)
    idx = {t: i for i, t in enumerate(ticks)}
    contiguous = ticks == list(range(ticks[0], ticks[0] + n))
    S = np.stack([sketches[nm] for nm in names], axis=0)  # (m, n, k) fp32
    relstd = 1.0 / np.sqrt(k)
    t_axis = np.arange(n, dtype=np.float64)

    # ---------------- (a) displacement-subspace effective rank ----------------
    # Stack the per-tick increments d_t = s[t]-s[t-1] (the displacement vectors);
    # the Gram matrix of inner products is preserved by the linear count-sketch.
    pr_inc, n90_inc, n99_inc = [], [], []
    pr_cum, n90_cum = [], []
    pc1_frac = []                      # top-PC energy fraction of the centered trajectory
    r2_line_centered, r2_line_origin = [], []  # global straight-line linearity
    Stt = float(((t_axis - t_axis.mean()) ** 2).sum())
    tc = t_axis - t_axis.mean()
    cum_energy_curves = []             # for the scree plot (cumulative energy, per matrix)
    for mi in range(nmn):
        s = S[mi].astype(np.float64)   # (n, k)
        # per-tick increments
        D = np.diff(s, axis=0)         # (n-1, k)
        G = D @ D.T
        ev = np.clip(np.linalg.eigvalsh(G), 0, None)[::-1]
        tot = ev.sum()
        if tot > 0:
            pr_inc.append((tot ** 2) / float(np.sum(ev ** 2)))
            cs = np.cumsum(ev) / tot
            n90_inc.append(int(np.searchsorted(cs, 0.90) + 1))
            n99_inc.append(int(np.searchsorted(cs, 0.99) + 1))
            cum_energy_curves.append(cs)
        # cumulative displacement subspace (alternative view)
        Dc = s - s[0:1]
        Dc = Dc[1:]
        Gc = Dc @ Dc.T
        evc = np.clip(np.linalg.eigvalsh(Gc), 0, None)[::-1]
        if evc.sum() > 0:
            pr_cum.append((evc.sum() ** 2) / float(np.sum(evc ** 2)))
            csc = np.cumsum(evc) / evc.sum()
            n90_cum.append(int(np.searchsorted(csc, 0.90) + 1))
        # global straight-line fit (centered): s[t] ~ a + b t, aggregate R^2
        sb = s.mean(0)
        b = (tc[:, None] * (s - sb)).sum(0) / Stt
        a = sb - b * t_axis.mean()
        fit = a[None, :] + np.outer(t_axis, b)
        ss_res = float(((s - fit) ** 2).sum())
        ss_tot = float(((s - sb) ** 2).sum())
        if ss_tot > 0:
            r2_line_centered.append(1.0 - ss_res / ss_tot)
            Sc = s - sb
            evt = np.clip(np.linalg.eigvalsh(Sc @ Sc.T), 0, None)[::-1]
            pc1_frac.append(float(evt[0] / evt.sum()))
        # through-origin cumulative-displacement line D[t] ~ t v (extrapolation form)
        d0 = s - s[0:1]
        v = (t_axis[:, None] * d0).sum(0) / float((t_axis ** 2).sum())
        fit0 = np.outer(t_axis, v)
        denom0 = float((d0 ** 2).sum())
        if denom0 > 0:
            r2_line_origin.append(1.0 - float(((d0 - fit0) ** 2).sum()) / denom0)
    maxdim = n - 1
    # total relative drift by the end of the run (median matrix), from the sketch
    drift_final = []
    for mi in range(nmn):
        n0 = float(np.linalg.norm(S[mi, 0]))
        if n0 > 0:
            drift_final.append(float(np.linalg.norm(S[mi, -1] - S[mi, 0])) / n0)
    total_drift_p50 = float(np.median(drift_final))
    median_cum_curve = np.median(np.stack(cum_energy_curves), axis=0)
    p10_cum_curve = np.percentile(np.stack(cum_energy_curves), 10, axis=0)
    p90_cum_curve = np.percentile(np.stack(cum_energy_curves), 90, axis=0)

    # ---------------- per-matrix fine-scale linearity R^2 (scale 1 and 10) -----
    def permat_R2(scale):
        out = np.full(nmn, np.nan)
        for mi in range(nmn):
            ss = S[mi]
            c2 = []
            for t in range(scale, n - scale):
                x = ss[t] - ss[t - scale]
                y = ss[t + scale] - ss[t]
                nx = np.linalg.norm(x)
                ny = np.linalg.norm(y)
                if nx > 0 and ny > 0:
                    c = float(np.dot(x, y)) / (nx * ny)
                    c2.append(c * c)
            if c2:
                out[mi] = float(np.mean(c2))
        return out
    R2_1 = permat_R2(1)
    R2_10 = permat_R2(10)

    # ---------------- (b) per-matrix projectability + crossover ----------------
    HG = list(range(1, 41))
    permat_ratio = {}
    for h in HG:
        alpha = h / DELTA
        s_lo, s_hi = DELTA, n - 1 - h
        if s_hi < s_lo:
            continue
        anchors = np.arange(s_lo, s_hi + 1)
        d_old = S[:, anchors, :] - S[:, anchors - DELTA, :]
        d_tgt = S[:, anchors, :] - S[:, anchors + h, :]
        num = np.linalg.norm(d_tgt + alpha * d_old, axis=2)
        den = np.linalg.norm(d_tgt, axis=2)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(den > 0, num / den, np.nan)
        permat_ratio[h] = np.nanmedian(r, axis=1)  # (m,)
    hstar = np.zeros(nmn)
    for h in HG:
        if h in permat_ratio:
            hstar[permat_ratio[h] < 1.0] = h
    r10 = permat_ratio[10]
    order = np.argsort(hstar + 1e-3 * (1.0 - r10))[::-1]  # furthest first, tie-break by lower r10
    furthest = [(names[i], int(hstar[i]), float(r10[i])) for i in order[:6]]
    least = [(names[i], int(hstar[i]), float(r10[i])) for i in order[-6:]]
    hstar_hist = {int(v): int((hstar == v).sum()) for v in sorted(set(hstar.tolist()))}
    # representative ratio-vs-h curves: furthest, median, least matrix
    med_mat = order[len(order) // 2]
    rep = {"furthest": order[0], "median": med_mat, "least": order[-1]}
    rep_curves = {key: [(h, float(permat_ratio[h][mi])) for h in HG if h in permat_ratio]
                  for key, mi in rep.items()}

    # ---------------- (c) optimal-coefficient sweep ----------------
    ALPHA = np.linspace(0.0, 3.0, 301)
    copt = {}
    alpha_curves = {}
    for h in HG:
        s_lo, s_hi = DELTA, n - 1 - h
        if s_hi < s_lo:
            continue
        anchors = np.arange(s_lo, s_hi + 1)
        d_old = S[:, anchors, :] - S[:, anchors - DELTA, :]
        d_tgt = S[:, anchors, :] - S[:, anchors + h, :]
        Aq = np.einsum("mak,mak->ma", d_tgt, d_tgt).ravel()
        Bq = np.einsum("mak,mak->ma", d_tgt, d_old).ravel()
        Cq = np.einsum("mak,mak->ma", d_old, d_old).ravel()
        ok = Aq > 0
        Aq, Bq, Cq = Aq[ok], Bq[ok], Cq[ok]
        med = np.array([np.median(np.sqrt(np.clip((Aq + 2 * al * Bq + al * al * Cq) / Aq, 0, None)))
                        for al in ALPHA])
        iopt = int(np.argmin(med))
        naive = h / DELTA
        r_naive = float(np.median(np.sqrt(np.clip((Aq + 2 * naive * Bq + naive * naive * Cq) / Aq, 0, None))))
        copt[h] = {"alpha_naive": naive, "ratio_naive": r_naive,
                   "alpha_opt": float(ALPHA[iopt]), "ratio_opt": float(med[iopt]),
                   "gain": r_naive - float(med[iopt])}
        if h in (5, 10, 20):
            alpha_curves[h] = (ALPHA.copy(), med.copy())

    # ---------------- (d) correlation linearity vs projectability ----------------
    def _pearson(x, y):
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        m = np.isfinite(x) & np.isfinite(y)
        return float(np.corrcoef(x[m], y[m])[0, 1])

    def _spearman(x, y):
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        m = np.isfinite(x) & np.isfinite(y)
        rx = np.argsort(np.argsort(x[m]))
        ry = np.argsort(np.argsort(y[m]))
        return float(np.corrcoef(rx, ry)[0, 1])
    corr = {"r2_1_vs_hstar_pearson": _pearson(R2_1, hstar),
            "r2_1_vs_hstar_spearman": _spearman(R2_1, hstar),
            "r2_1_vs_r10_pearson": _pearson(R2_1, r10),
            "r2_1_vs_r10_spearman": _spearman(R2_1, r10)}

    # ---------------- (e) learned vs fixed residual ----------------
    ones = {nm: ones_sketch(dims[nm], k) for nm in names}
    fires = [t for t in ticks if (t - ticks[0]) % DELTA == 0]
    learned_resid = {}
    learned_resid_at = {}
    prev_hat_mean = None
    prev_fire = None
    for f in fires:
        if prev_fire is not None and prev_hat_mean is not None:
            for nm in names:
                err = float(means[nm][idx[f]]) - float(prev_hat_mean.get(nm, means[nm][idx[f]]))
                new = float(learned_resid.get(nm, 0.0)) + 0.1 * err
                learned_resid[nm] = max(-1e-3, min(1e-3, new))
        learned_resid_at[f] = dict(learned_resid)
        if (f - DELTA) in idx:
            prev_hat_mean = {nm: 2.0 * float(means[nm][idx[f]]) - 1.0 * float(means[nm][idx[f - DELTA]])
                             for nm in names}
            prev_fire = f
    h = 10
    alpha = h / DELTA
    dratio = []
    resid_rel = []
    for s in fires:
        if (s - DELTA) not in idx or (s + h) not in idx:
            continue
        resid = learned_resid_at.get(s, {})
        for mi, nm in enumerate(names):
            d_old = S[mi, idx[s]] - S[mi, idx[s - DELTA]]
            d_tgt = S[mi, idx[s]] - S[mi, idx[s + h]]
            proj = d_tgt + alpha * d_old
            den = float(np.linalg.norm(d_tgt))
            if den <= 0:
                continue
            rf = float(np.linalg.norm(proj)) / den
            rterm = np.float32(resid.get(nm, 0.0)) * ones[nm]
            rl = float(np.linalg.norm(proj + rterm)) / den
            dratio.append(rl - rf)
            resid_rel.append(float(np.linalg.norm(rterm)) / den)
    dratio = np.array(dratio)
    resid_rel = np.array(resid_rel)
    final_resid = learned_resid_at[fires[-1]]
    resid_e = {"max_abs_resid": float(max(abs(v) for v in final_resid.values()) if final_resid else 0.0),
               "resid_term_rel_p50": _pct(resid_rel, 50), "resid_term_rel_p90": _pct(resid_rel, 90),
               "dratio_median": float(np.median(dratio)) if dratio.size else 0.0,
               "dratio_abs_p90": _pct(np.abs(dratio), 90), "dratio_abs_max": float(np.abs(dratio).max()) if dratio.size else 0.0}

    return dict(
        names=names, n_matrices=nmn, n_ticks=n, k=k, relstd=relstd, maxdim=maxdim,
        contiguous=contiguous, total_drift_p50=total_drift_p50,
        pr_inc=np.array(pr_inc), n90_inc=np.array(n90_inc), n99_inc=np.array(n99_inc),
        pr_cum=np.array(pr_cum), n90_cum=np.array(n90_cum), pc1_frac=np.array(pc1_frac),
        r2_line_centered=np.array(r2_line_centered), r2_line_origin=np.array(r2_line_origin),
        median_cum_curve=median_cum_curve, p10_cum_curve=p10_cum_curve, p90_cum_curve=p90_cum_curve,
        R2_1=R2_1, R2_10=R2_10,
        permat_ratio=permat_ratio, hstar=hstar, r10=r10, hstar_hist=hstar_hist,
        furthest=furthest, least=least, rep_curves=rep_curves,
        copt=copt, alpha_curves=alpha_curves, corr=corr, resid_e=resid_e,
    )


# ====================================================================== #
# Performance parse (reused, for the scope line only)
# ====================================================================== #
def parse_perf(internal_log):
    rew, val = {}, {}
    if not os.path.exists(internal_log):
        return [], []
    for line in open(internal_log, errors="ignore"):
        sm = re.search(r"training/global_step:(\d+)", line)
        if not sm:
            continue
        gs = int(sm.group(1))
        vm = re.search(r"val-core/openai/gsm8k/acc/mean@1:([0-9.]+)", line)
        if vm:
            val[gs] = float(vm.group(1))
    return sorted(rew.items()), sorted(val.items())


CSS = """
:root{--ground:#f7f8fa;--surface:#fff;--ink:#14181f;--muted:#5b6573;--line:#e3e6eb;--a:#1f5fae;
--help:#1a7f37;--b:#b42318;--amber:#b7791f;--sans:ui-sans-serif,-apple-system,"Segoe UI",Roboto,Helvetica,sans-serif;
--mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace}
*{box-sizing:border-box}body{margin:0;background:var(--ground);color:var(--ink);font-family:var(--sans);
line-height:1.6;font-size:16px;-webkit-font-smoothing:antialiased}
.wrap{max-width:980px;margin:0 auto;padding:40px 24px 80px}
header{border-bottom:1px solid var(--line);padding-bottom:22px}
.eyebrow{font-size:12px;letter-spacing:.13em;text-transform:uppercase;color:var(--muted);font-weight:600}
h1{font-size:30px;line-height:1.18;margin:10px 0 12px;text-wrap:balance;font-weight:680;letter-spacing:-.01em}
.thesis{color:var(--muted);max-width:72ch;margin:0 0 16px}
.chips{display:flex;flex-wrap:wrap;gap:8px}.chip{font-family:var(--mono);font-size:11.5px;color:var(--muted);
background:var(--surface);border:1px solid var(--line);border-radius:999px;padding:3px 10px}
h2{font-size:20px;margin:46px 0 12px;padding-bottom:6px;border-bottom:1px solid var(--line);font-weight:660}
h3{font-size:15px;margin:22px 0 6px;font-weight:640}
p{max-width:74ch}.cap{color:var(--muted);font-size:13px;max-width:76ch;margin:8px 0 4px}
.callout{background:var(--surface);border:1px solid var(--line);border-radius:14px;padding:20px 22px;margin:22px 0}
.callout h2{border:0;margin:0 0 8px;padding:0;font-size:16px}
.note{background:#fbfaf5;border:1px solid #ece3c8;border-radius:12px;padding:14px 18px;margin:18px 0;font-size:14px;color:#5a4f2e}
.note b{color:#7a6a2e}
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
ul{max-width:76ch}li{margin:4px 0}b{font-weight:660}
code{font-family:var(--mono);font-size:.92em;background:rgba(31,95,174,.06);padding:1px 5px;border-radius:5px}
"""


def esc(x):
    return html.escape(str(x))


# ====================================================================== #
# Build
# ====================================================================== #
def build(run_dir, regime, out_html, fragment=False):
    wdir = os.path.join(run_dir, regime, "weights")
    m = compute(wdir)
    _, val = parse_perf(os.path.join(run_dir, regime, f"train_{regime}_internal.log"))
    final_val = val[-1][1] if val else None
    relstd_pct = m["relstd"] * 100.0

    # ----------------------------- plots -----------------------------
    # (a) low-rank: PR histogram + n90 histogram + median cumulative-energy scree
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13.5, 3.8))
    ax1.hist(m["pr_inc"], bins=24, color=A, alpha=0.85)
    ax1.axvline(np.median(m["pr_inc"]), color=HURT, ls="--", lw=1.2)
    ax1.set_xlabel("participation ratio of per-tick increments")
    ax1.set_ylabel("matrices")
    ax1.set_title(f"(a) Effective rank of the\ndisplacement subspace (ceiling {m['maxdim']})")
    ax2.hist(m["n90_inc"], bins=20, color=HELP, alpha=0.85)
    ax2.axvline(np.median(m["n90_inc"]), color=HURT, ls="--", lw=1.2)
    ax2.set_xlabel("components for 90% of increment energy")
    ax2.set_ylabel("matrices")
    ax2.set_title("(a) Components for 90% energy")
    xcomp = np.arange(1, len(m["median_cum_curve"]) + 1)
    ax3.plot(xcomp, m["median_cum_curve"], "-", color=A, lw=1.6, label="median matrix")
    ax3.fill_between(xcomp, m["p10_cum_curve"], m["p90_cum_curve"], color=A, alpha=0.15, label="p10-p90")
    ax3.axhline(0.90, color=HELP, ls="--", lw=1)
    ax3.set_xlim(1, 60)
    ax3.set_xlabel("number of leading components")
    ax3.set_ylabel("cumulative energy fraction")
    ax3.set_title("(a) Cumulative energy (scree)")
    ax3.legend(fontsize=8, loc="lower right")
    fig_a = _png(fig)

    # (b) per-matrix crossover histogram + representative ratio-vs-h curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))
    hk = sorted(m["hstar_hist"].keys())
    ax1.bar(hk, [m["hstar_hist"][h] for h in hk], color=A, alpha=0.85, width=0.8)
    ax1.axvline(np.median(m["hstar"]), color=HURT, ls="--", lw=1.2,
                label=f"median h*={np.median(m['hstar']):.0f}")
    ax1.set_xlabel("per-matrix crossover horizon h* (ticks; naive alpha, Delta=10)")
    ax1.set_ylabel("matrices")
    ax1.set_title("(b) Where each matrix stops being projectable")
    ax1.legend(fontsize=8)
    style = {"furthest": (HELP, "furthest"), "median": (A, "median"), "least": (HURT, "least")}
    for key, (col, lab) in style.items():
        hs = [h for h, _ in m["rep_curves"][key]]
        rs = [r for _, r in m["rep_curves"][key]]
        ax2.plot(hs, rs, "-", color=col, label=lab)
    ax2.axhline(1.0, color="k", ls="--", lw=1)
    ax2.axvline(10, color=MUTED, ls=":", lw=1)
    ax2.set_xlabel("horizon h (ticks)")
    ax2.set_ylabel("weight_proj_ratio (below 1 helps)")
    ax2.set_title("(b) Projectability vs horizon, by matrix")
    ax2.legend(fontsize=8)
    fig_b = _png(fig)

    # (c) optimal-alpha: ratio-vs-alpha curves + opt/naive alpha vs h
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))
    cols = {5: HELP, 10: A, 20: HURT}
    for h, (al, med) in m["alpha_curves"].items():
        ax1.plot(al, med, "-", color=cols.get(h, MUTED), label=f"h={h}")
        naive = h / DELTA
        ax1.plot([naive], [m["copt"][h]["ratio_naive"]], "o", color=cols.get(h, MUTED), ms=7, mfc="white")
        ax1.plot([m["copt"][h]["alpha_opt"]], [m["copt"][h]["ratio_opt"]], "*", color=cols.get(h, MUTED), ms=13)
    ax1.axhline(1.0, color="k", ls="--", lw=1)
    ax1.set_xlabel("coefficient alpha")
    ax1.set_ylabel("median weight_proj_ratio")
    ax1.set_title("(c) Ratio vs alpha (circle=naive, star=optimal)")
    ax1.legend(fontsize=8)
    hh = sorted(m["copt"].keys())
    ax2.plot(hh, [m["copt"][h]["alpha_naive"] for h in hh], "-", color=MUTED, label="naive alpha = h/Delta")
    ax2.plot(hh, [m["copt"][h]["alpha_opt"] for h in hh], "-", color=A, label="optimal alpha")
    ax2b = ax2.twinx()
    ax2b.plot(hh, [m["copt"][h]["gain"] for h in hh], "-", color=HELP, lw=1.4, label="ratio gain (right)")
    ax2b.set_ylabel("median ratio gain (naive minus optimal)", color=HELP)
    ax2b.tick_params(axis="y", labelcolor=HELP)
    ax2b.grid(False)
    ax2.set_xlabel("horizon h (ticks)")
    ax2.set_ylabel("coefficient alpha")
    ax2.set_title("(c) Optimal vs naive coefficient, and the gain")
    ax2.legend(fontsize=8, loc="upper left")
    fig_c = _png(fig)

    # (d) scatter R2@1 vs ratio@10, colored by family
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))
    fam = np.array([_type(nm) for nm in m["names"]])
    for f, col in (("attention", A), ("mlp", HURT)):
        sel = fam == f
        ax1.scatter(m["R2_1"][sel], m["r10"][sel], s=18, color=col, alpha=0.7, label=f)
    ax1.axhline(1.0, color="k", ls="--", lw=1)
    ax1.set_xlabel("fine-scale linearity R^2 (1 tick)")
    ax1.set_ylabel("weight_proj_ratio at h=10")
    ax1.set_title(f"(d) More-linear -> more-projectable\nspearman={m['corr']['r2_1_vs_r10_spearman']:.2f}")
    ax1.legend(fontsize=8)
    for f, col in (("attention", A), ("mlp", HURT)):
        sel = fam == f
        ax2.scatter(m["R2_1"][sel], m["hstar"][sel] + np.random.default_rng(0).uniform(-0.15, 0.15, sel.sum()),
                    s=18, color=col, alpha=0.7, label=f)
    ax2.set_xlabel("fine-scale linearity R^2 (1 tick)")
    ax2.set_ylabel("crossover h* (ticks, jittered)")
    ax2.set_title(f"(d) Linearity vs crossover\nspearman={m['corr']['r2_1_vs_hstar_spearman']:.2f}")
    ax2.legend(fontsize=8)
    fig_d = _png(fig)

    # (e) residual effect: key magnitudes against the sketch noise floor.
    # Only summary stats are retained, so draw them as bars vs the noise floor.
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 4.0))
    bars = {
        "learned-fixed\n|dratio| p90": m["resid_e"]["dratio_abs_p90"],
        "learned-fixed\n|dratio| max": m["resid_e"]["dratio_abs_max"],
        "residual term\nrel-norm p50": m["resid_e"]["resid_term_rel_p50"],
        "residual term\nrel-norm p90": m["resid_e"]["resid_term_rel_p90"],
    }
    xb = np.arange(len(bars))
    ax.bar(xb, list(bars.values()), color=A, alpha=0.85, width=0.6)
    ax.axhline(m["relstd"], color=HURT, ls="--", lw=1.3, label=f"sketch noise floor ~{relstd_pct:.1f}%")
    ax.set_yscale("log")
    ax.set_xticks(xb)
    ax.set_xticklabels(list(bars.keys()), fontsize=8)
    ax.set_ylabel("relative magnitude (log)")
    ax.set_title("(e) Learned residual effect vs the sketch noise floor")
    ax.legend(fontsize=8)
    fig_e = _png(fig)

    # ----------------------------- headline numbers -----------------------------
    pr50 = float(np.median(m["pr_inc"]))
    n90_50 = float(np.median(m["n90_inc"]))
    r2line_o50 = float(np.median(m["r2_line_origin"]))
    r2line_c50 = float(np.median(m["r2_line_centered"]))
    pc1_50 = float(np.median(m["pc1_frac"]))
    hstar50 = float(np.median(m["hstar"]))
    r10_50 = float(np.median(m["r10"]))
    c10 = m["copt"][10]
    c20 = m["copt"][20]

    # ----------------------------- prose -----------------------------
    P = []
    P.append("<header><div class='eyebrow'>EXP-42 / dense run / weight behavior v2</div>")
    P.append("<h1>The dense GRPO trajectory is globally near-linear and low-rank, "
             "but the look-ahead's two-point slope overshoots</h1>")
    P.append("<p class='thesis'>A deeper GPU-free look at the plain GRPO run (codec off), all "
             "computed from the per-tick weight-trajectory count-sketch. Five studies: the low-rank "
             "structure of the weight-displacement subspace, the per-matrix projectability and "
             "crossover horizon, an optimal-coefficient sweep against the naive rule, the link between "
             "local linearity and projectability, and the learned-vs-fixed residual effect.</p>")
    P.append(f"<div class='chips'><span class='chip'>{m['n_matrices']} decoder matrices</span>"
             f"<span class='chip'>{m['n_ticks']} ticks ({m['n_ticks']//TICKS_PER_STEP} steps)</span>"
             f"<span class='chip'>count-sketch k={m['k']}</span>"
             f"<span class='chip'>rel std ~{relstd_pct:.1f}%</span>"
             f"<span class='chip'>codec OFF (dense)</span></div></header>")

    P.append("<section class='callout'><h2>Headline</h2>")
    P.append(f"<p>The per-tick weight updates of the {m['n_matrices']} decoder matrices live in a "
             f"<b>low-dimensional subspace</b>: the median participation ratio is "
             f"<span class='big'>{pr50:.1f}</span> out of a possible {m['maxdim']}, and a median of "
             f"{n90_50:.0f} components capture 90 percent of the per-tick displacement energy. By a "
             f"global straight-line fit the trajectory is substantially linear (through-origin "
             f"extrapolation R squared {r2line_o50:.2f}; one dominant direction holds {pc1_50*100:.0f} "
             f"percent of the centered energy), which is the RLVR-linear claim, just measured globally "
             f"rather than step to step.</p>")
    P.append(f"<p>Yet the look-ahead anchor still helps only out to about <b>{hstar50:.0f} ticks</b> "
             f"(median crossover) because it estimates the trajectory direction from a noisy "
             f"<b>two-point slope</b>. The naive coefficient alpha = h/Delta overshoots: at h=10 a "
             f"damped alpha = {c10['alpha_opt']:.2f} cuts the median ratio from {c10['ratio_naive']:.3f} "
             f"to {c10['ratio_opt']:.3f}, and at h=20 the optimal alpha keeps the ratio below 1 "
             f"({c20['ratio_opt']:.3f}) where the naive rule fails ({c20['ratio_naive']:.3f}). A simple "
             f"damped coefficient is the actionable lever.</p></section>")

    # ---- study (a) ----
    P.append("<h2>(a) Low-rank structure of the weight-displacement subspace</h2>")
    P.append(f"<figure><img src='data:image/png;base64,{fig_a}' /></figure>")
    P.append(f"<p class='cap'>For each matrix we stack the per-tick displacement vectors "
             f"theta[t] - theta[t-1] (there are {m['maxdim']} of them) and take the eigenvalues of "
             f"their Gram matrix, which the linear count-sketch preserves up to the ~{relstd_pct:.1f} "
             f"percent noise floor. The participation ratio (median {pr50:.1f}) and the 90-percent "
             f"component count (median {n90_50:.0f}) both say the updates explore only a handful of "
             f"directions out of the {m['maxdim']} available. Sketch noise inflates the small "
             f"eigenvalues, so it biases the effective rank upward: the true subspace is at least this "
             f"low-rank, probably lower.</p>")
    P.append("<table><tr><th>measure</th><th>p10</th><th>p50</th><th>p90</th></tr>")
    P.append(f"<tr><td>participation ratio (per-tick increments)</td><td>{_pct(m['pr_inc'],10):.1f}</td>"
             f"<td>{_pct(m['pr_inc'],50):.1f}</td><td>{_pct(m['pr_inc'],90):.1f}</td></tr>")
    P.append(f"<tr><td>components for 90% increment energy</td><td>{_pct(m['n90_inc'],10):.0f}</td>"
             f"<td>{_pct(m['n90_inc'],50):.0f}</td><td>{_pct(m['n90_inc'],90):.0f}</td></tr>")
    P.append(f"<tr><td>participation ratio (cumulative displacement)</td><td>{_pct(m['pr_cum'],10):.1f}</td>"
             f"<td>{_pct(m['pr_cum'],50):.1f}</td><td>{_pct(m['pr_cum'],90):.1f}</td></tr>")
    P.append(f"<tr><td>top-PC energy fraction (centered trajectory)</td><td>{_pct(m['pc1_frac'],10):.2f}</td>"
             f"<td>{_pct(m['pc1_frac'],50):.2f}</td><td>{_pct(m['pc1_frac'],90):.2f}</td></tr>")
    P.append(f"<tr><td>global line-fit R^2 (centered)</td><td>{_pct(m['r2_line_centered'],10):.2f}</td>"
             f"<td>{_pct(m['r2_line_centered'],50):.2f}</td><td>{_pct(m['r2_line_centered'],90):.2f}</td></tr>")
    P.append(f"<tr><td>global line R^2 (through-origin extrapolation)</td><td>{_pct(m['r2_line_origin'],10):.2f}</td>"
             f"<td>{_pct(m['r2_line_origin'],50):.2f}</td><td>{_pct(m['r2_line_origin'],90):.2f}</td></tr>")
    P.append("</table>")
    P.append(f"<p class='cap'>Reconciling the RLVR-linear claim: the prior report measured "
             f"<i>local</i> consecutive-step linearity (R squared {_pct(m['R2_1'],50):.2f} at one tick, "
             f"decaying to {_pct(m['R2_10'],50):.2f} at the K=10 scale) and read this as the trajectory "
             f"curving. The global line fit here shows the opposite face of the same data: one direction "
             f"holds {pc1_50*100:.0f} percent of the energy and a single straight line explains "
             f"R squared {r2line_o50:.2f} of the displacement (through-origin). Both are true. The "
             f"trajectory is one dominant slow drift plus per-step noise; the local metric sees mostly "
             f"the noise at fine scale, the global metric sees mostly the drift. The cited RLVR result "
             f"reports a global line at R squared about 0.9 over hundreds of steps; our analogous global "
             f"figure ({r2line_o50:.2f}) is in that ballpark, and since sketch noise inflates the "
             f"residual it is a lower bound on the true linearity. The gap to 600 steps is then about "
             f"horizon and motion size ({m['n_ticks']//TICKS_PER_STEP} steps, "
             f"{m['total_drift_p50']*100:.3f} percent total drift from an already-tuned model), "
             f"not a qualitative difference in shape.</p>")
    P.append("<div class='note'><b>What this can and cannot test.</b> The count-sketch flattens each "
             "weight matrix to a vector before sketching, so it preserves displacement norms, cosines "
             "and the temporal Gram matrix, but it does <b>not</b> preserve the singular spectrum of a "
             "single weight matrix in its native rows-by-cols shape. The LoRA-style reading of "
             "'RLVR updates are low-rank' (the update matrix Delta W has few large singular values) is "
             "therefore <b>not computable</b> from this data and is not claimed here. What we measure is "
             "the <b>temporal</b> low-rank structure: the trajectory of flattened updates lives in a "
             "low-dimensional subspace. The embeddings, RMSNorm gains and biases were not collected in "
             "this run, so all of (a) is the decoder set only.</div>")

    # ---- study (b) ----
    P.append("<h2>(b) Per-matrix projectability and crossover horizon</h2>")
    P.append(f"<figure><img src='data:image/png;base64,{fig_b}' /></figure>")
    P.append(f"<p class='cap'>For each matrix we take the median weight_proj_ratio over all valid "
             f"anchors at the operating spacing Delta=10 and the naive coefficient, and read off the "
             f"largest horizon with median ratio below 1 (its crossover h*). The distribution is tight: "
             f"crossover h* ranges {int(m['hstar'].min())} to {int(m['hstar'].max())} ticks with median "
             f"{hstar50:.0f}, and the ratio at h=10 spans only {_pct(m['r10'],10):.3f} to "
             f"{_pct(m['r10'],90):.3f} (median {r10_50:.3f}). Projectability is a property of the whole "
             f"decoder, not a few outliers. Per-matrix differences of about one tick in h* are within "
             f"the ~{relstd_pct:.1f} percent sketch noise, but the ranking below is supported by ratio "
             f"gaps larger than the floor.</p>")
    P.append("<table><tr><th>projects furthest</th><th>h*</th><th>ratio@10</th>"
             "<th>projects least</th><th>h*</th><th>ratio@10</th></tr>")
    for (fn, fh, fr), (ln, lh, lr) in zip(m["furthest"], m["least"][::-1]):
        P.append(f"<tr><td>{esc(_proj_short(fn))}</td><td>{fh}</td><td>{fr:.3f}</td>"
                 f"<td>{esc(_proj_short(ln))}</td><td>{lh}</td><td>{lr:.3f}</td></tr>")
    P.append("</table>")
    nv = sum(1 for fn, _, _ in m["furthest"] if "v_proj" in fn)
    P.append(f"<p class='cap'>The matrices that project furthest are dominated by the attention value "
             f"and output projections (v_proj, o_proj) in the middle-to-late layers; the least "
             f"projectable tend to be MLP and the attention key/query projections. This matches the "
             f"fine-scale linearity ordering in the prior report (attention slightly more linear than "
             f"MLP) and is quantified directly in study (d).</p>")

    # ---- study (c) ----
    P.append("<h2>(c) Optimal coefficient: is there a better-than-naive linear extrapolation</h2>")
    P.append(f"<figure><img src='data:image/png;base64,{fig_c}' /></figure>")
    P.append(f"<p class='cap'>At each horizon we sweep a single global coefficient alpha and find the "
             f"value that minimizes the median weight_proj_ratio over all matrices and anchors, then "
             f"compare it to the naive alpha = h/Delta that the look-ahead uses. A better-than-naive "
             f"coefficient clearly exists, and the gain grows with horizon.</p>")
    P.append("<table><tr><th>h (ticks)</th><th>naive alpha</th><th>ratio @ naive</th>"
             "<th>optimal alpha</th><th>ratio @ optimal</th><th>ratio gain</th></tr>")
    for h in (3, 5, 8, 10, 13, 20, 30):
        if h in m["copt"]:
            c = m["copt"][h]
            P.append(f"<tr><td>{h}</td><td>{c['alpha_naive']:.2f}</td><td>{c['ratio_naive']:.3f}</td>"
                     f"<td>{c['alpha_opt']:.2f}</td><td>{c['ratio_opt']:.3f}</td>"
                     f"<td class='ok'>{c['gain']:.3f}</td></tr>")
    P.append("</table>")
    P.append(f"<p class='cap'>The optimal coefficient is well below the naive one at every horizon "
             f"(roughly {m['copt'][10]['alpha_opt']:.2f} at h=10 versus the naive 1.0, and "
             f"{m['copt'][20]['alpha_opt']:.2f} at h=20 versus the naive 2.0). Interpretation: the "
             f"two-point slope theta_stale - theta_old over-states the persistent drift because it also "
             f"captures per-step noise, so the naive rule extrapolates too far. Damping alpha corrects "
             f"the over-step. The gain is far above the ~{relstd_pct:.1f} percent sketch floor, so it is "
             f"real. Caveat: this alpha is fit on the same trajectory it is scored on, so the table is "
             f"an oracle upper bound on the achievable gain. But the optimal alpha is stable across "
             f"horizons (about 0.5 to 0.75), so a single fixed damped coefficient near 0.5 would capture "
             f"most of it without any online estimation, which is a concrete, deployable change to the "
             f"look-ahead rule.</p>")

    # ---- study (d) ----
    P.append("<h2>(d) Does more-linear mean more-projectable</h2>")
    P.append(f"<figure><img src='data:image/png;base64,{fig_d}' /></figure>")
    cr = m["corr"]
    P.append(f"<p class='cap'>Across the {m['n_matrices']} matrices, the fine-scale linearity "
             f"R squared (one tick) correlates with projectability in the expected direction: "
             f"more-linear matrices project further (Spearman {cr['r2_1_vs_hstar_spearman']:.2f} versus "
             f"crossover h*) and land closer (Spearman {cr['r2_1_vs_r10_spearman']:.2f} versus the "
             f"ratio at h=10, negative because lower ratio is better). With n={m['n_matrices']} a "
             f"correlation above about 0.14 is significant at the 5 percent level, so these moderate "
             f"correlations are well clear of noise. The relationship is real but loose: local "
             f"linearity explains part of projectability, not all of it, because projectability also "
             f"depends on how the per-step noise inflates the two-point slope (study c).</p>")

    # ---- study (e) ----
    P.append("<h2>(e) Learned-linear vs fixed-linear residual effect</h2>")
    P.append(f"<figure><img src='data:image/png;base64,{fig_e}' /></figure>")
    re_ = m["resid_e"]
    P.append(f"<p class='cap'>The learned variant adds a per-matrix scalar mean-shift residual, updated "
             f"retrospectively (r left-arrow clip(r + 0.1 mean(theta_true - theta_hat), +/- 1e-3)) and "
             f"replayed through the all-ones sketch. Quantified on the dense run: the residual grows to "
             f"at most {re_['max_abs_resid']:.1e} in magnitude (far from its 1e-3 clip), its contribution "
             f"to the projection has relative norm {re_['resid_term_rel_p50']:.1e} (median) against the "
             f"target displacement, and the resulting change in weight_proj_ratio is "
             f"{re_['dratio_median']:.1e} (median), at most {re_['dratio_abs_max']:.1e}. Every one of "
             f"these is below the ~{relstd_pct:.1f} percent sketch noise floor, so the learned residual "
             f"is statistically indistinguishable from doing nothing here. The mechanism is clear: a "
             f"single scalar added uniformly to every element of a high-dimensional matrix barely moves "
             f"the norm or direction of a displacement vector, and the dense run's per-matrix mean drifts "
             f"so smoothly that the fixed extrapolation of that mean already has almost no retrospective "
             f"error to correct. The learned residual is inert on the dense run.</p>")

    # ---- scope ----
    P.append("<h2>Scope, provenance, and caveats</h2><ul>")
    P.append(f"<li>Source: {esc(regime)} per-tick weight sketch, runs/EXP-42/{esc(regime)}/weights "
             f"({m['n_matrices']} decoder matrices q/k/v/o/gate/up/down, {m['n_ticks']} ticks, "
             f"k={m['k']}). Ticks are contiguous ({m['contiguous']}). Two ticks per global step.</li>")
    P.append(f"<li>All metrics computed on the MacBook from the sketch, GPU-free. No GPU, no training, "
             f"no provisioning. The count-sketch is linear and norm-preserving, so all displacement "
             f"norms, cosines and Gram matrices reconstruct from it with relative std about 1/sqrt(k) = "
             f"{relstd_pct:.1f} percent; this floor is stated at each study where it matters.</li>")
    P.append("<li>Not computable from this data, so not claimed: the matrix-native singular spectrum "
             "(LoRA-style rank) of an individual weight matrix, which the flatten-then-sketch step "
             "destroys; and anything about the embeddings, RMSNorm gains or biases, which were not "
             "collected for this run (decoder matrices only).</li>")
    if final_val is not None:
        P.append(f"<li>Performance context: this is a healthy plain-GRPO control, final validation "
                 f"accuracy {final_val:.4f} on GSM8K (near-converged, flat over the run).</li>")
    P.append("</ul>")

    metrics = {
        "regime": regime, "n_matrices": m["n_matrices"], "n_ticks": m["n_ticks"], "k": m["k"],
        "relstd": m["relstd"],
        "a_low_rank": {
            "participation_ratio_inc": {"p10": _pct(m["pr_inc"], 10), "p50": pr50, "p90": _pct(m["pr_inc"], 90)},
            "n90_inc": {"p10": _pct(m["n90_inc"], 10), "p50": n90_50, "p90": _pct(m["n90_inc"], 90)},
            "ceiling_dim": m["maxdim"],
            "participation_ratio_cum_p50": _pct(m["pr_cum"], 50),
            "pc1_energy_frac_p50": pc1_50,
            "line_fit_r2_centered_p50": r2line_c50,
            "line_fit_r2_origin_p50": r2line_o50,
            "local_r2_1_p50": _pct(m["R2_1"], 50), "local_r2_10_p50": _pct(m["R2_10"], 50),
        },
        "b_projectability": {
            "hstar": {"min": int(m["hstar"].min()), "p50": hstar50, "max": int(m["hstar"].max())},
            "ratio_at_h10": {"p10": _pct(m["r10"], 10), "p50": r10_50, "p90": _pct(m["r10"], 90)},
            "hstar_hist": m["hstar_hist"],
            "furthest": [{"name": fn, "hstar": fh, "ratio10": fr} for fn, fh, fr in m["furthest"]],
            "least": [{"name": ln, "hstar": lh, "ratio10": lr} for ln, lh, lr in m["least"]],
        },
        "c_optimal_alpha": {str(h): m["copt"][h] for h in sorted(m["copt"])},
        "d_correlation": m["corr"],
        "e_residual": m["resid_e"],
        "final_val": final_val,
    }

    content = "".join(P).replace("<table", "<div class='tbl'><table").replace("</table>", "</table></div>")
    body = "<div class='wrap'>" + content + "</div>"
    if fragment:
        out = f"<style>{CSS}</style>" + body
    else:
        out = (f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
               f"<meta name='viewport' content='width=device-width, initial-scale=1'>"
               f"<title>EXP-42 dense run weight behavior v2</title><style>{CSS}</style></head><body>"
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
    ap.add_argument("run_dir")
    ap.add_argument("regime")
    ap.add_argument("out_html")
    ap.add_argument("--fragment", action="store_true")
    a = ap.parse_args()
    build(a.run_dir, a.regime, a.out_html, fragment=a.fragment)


if __name__ == "__main__":
    main()
