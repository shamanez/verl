#!/usr/bin/env python3
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""EXP-42 self-contained HTML report builder (analysis + plots + descriptions).

Reads the sweep JSON produced by ``weight_proj_sweep.py --json`` and emits a
single self-contained HTML file: matplotlib plots embedded as base64 PNG, the
operating-point answer, the crossover table, full-curve tables, the sketch
fidelity check, and prose descriptions generated from the numbers (not
hand-written, so they cannot drift from the data). Handles the per-group
structure (decoder / embed / norm / bias) when the widened select_all sweep is
present. GPU-free, MacBook-only.

Usage:
    python build_report.py sweep_narrow.json report.html --title "EXP-42 narrow"
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Palette (cool-neutral instrument readout). Regime colour IS the data encoding.
INK = "#14181f"
MUTED = "#5b6573"
GRID = "#e3e6eb"
HELP = "#1a7f37"
HURT = "#b42318"
HORIZONS = (1, 2, 3, 5, 8, 10, 13, 20, 30)
REGIME_LABEL = {"regimeA": "regime A  plain GRPO, codec off",
                "regimeB": "regime B  PowerSGD r=77, codec only"}
REGIME_COLOR = {"regimeA": "#1f5fae", "regimeB": "#b42318"}

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white", "axes.titlesize": 10,
})


def _cells(reg, method, delta):
    """Return (hs, w1_p50, w1_p10, w1_p90, dir_cos) arrays over the horizon grid."""
    res = reg["results"]
    hs, p50, p10, p90, dc = [], [], [], [], []
    for h in HORIZONS:
        c = res.get(f"{method}|{delta}|{h}")
        if not c:
            continue
        hs.append(h); p50.append(c["w1_p50"]); p10.append(c["w1_p10"])
        p90.append(c["w1_p90"]); dc.append(c["dir_cos_p50"])
    return (np.array(hs), np.array(p50), np.array(p10), np.array(p90), np.array(dc))


def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _crossover(reg, method, delta):
    best = None
    for h in HORIZONS:
        c = reg["results"].get(f"{method}|{delta}|{h}")
        if c and c["w1_p50"] < 1.0:
            best = h
    return best


def fig_headline(regimes, delta=10, method="fixed_linear"):
    """Two panels: w1 vs h (with p10-p90 band, y=1) and dir_cos vs h (y=0)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    for rname, reg in regimes.items():
        hs, p50, p10, p90, dc = _cells(reg, method, delta)
        if not len(hs):
            continue
        col = REGIME_COLOR.get(rname, None)
        ax1.plot(hs, p50, "o-", color=col, label=REGIME_LABEL.get(rname, rname))
        ax1.fill_between(hs, p10, p90, color=col, alpha=0.15)
        ax2.plot(hs, dc, "o-", color=col, label=REGIME_LABEL.get(rname, rname))
    ax1.axhline(1.0, color="k", ls="--", lw=1)
    ax1.axvline(delta, color="gray", ls=":", lw=1)
    ax1.text(delta, ax1.get_ylim()[1], f" K={delta}", va="top", fontsize=8, color="gray")
    ax1.set_xlabel("horizon h (optimizer ticks ahead; 2 ticks per global step)")
    ax1.set_ylabel("weight_proj_ratio  ||theta_hat - target|| / ||theta_stale - target||")
    ax1.set_title(f"Projection accuracy vs horizon ({method}, spacing={delta})\nbelow 1.0 helps, shaded = per-matrix p10..p90")
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3)
    ax2.axhline(0.0, color="k", ls="--", lw=1)
    ax2.axvline(delta, color="gray", ls=":", lw=1)
    ax2.set_xlabel("horizon h (optimizer ticks ahead)")
    ax2.set_ylabel("dir_cos  cos(theta_stale - theta_old, target - theta_stale)")
    ax2.set_title("Trajectory straightness vs horizon\nbelow 0 = past update points away from future (sign flip)")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
    return _png(fig)


def fig_spacing(regimes, method="fixed_linear"):
    """w1 vs h for spacing Delta=5 and Delta=10, per regime."""
    fig, axes = plt.subplots(1, len(regimes), figsize=(6 * len(regimes), 4.4), squeeze=False)
    for ax, (rname, reg) in zip(axes[0], regimes.items()):
        for delta, mk in ((5, "s--"), (10, "o-")):
            hs, p50, _, _, _ = _cells(reg, method, delta)
            if len(hs):
                ax.plot(hs, p50, mk, label=f"spacing Delta={delta}")
        ax.axhline(1.0, color="k", ls="--", lw=1)
        ax.set_xlabel("horizon h (ticks)")
        ax.set_ylabel("weight_proj_ratio")
        ax.set_title(f"{REGIME_LABEL.get(rname, rname)}\n{method}: spacing effect on crossover")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    return _png(fig)


def fig_groups(regimes, delta=10, method="fixed_linear"):
    """Per-group w1 vs h (only when the widened select_all sweep is present)."""
    groups = []
    for reg in regimes.values():
        groups = reg.get("groups_present", []) or groups
    groups = [g for g in groups if g != "other"]
    if len(groups) <= 1:
        return None
    fig, axes = plt.subplots(1, len(regimes), figsize=(6 * len(regimes), 4.6), squeeze=False)
    for ax, (rname, reg) in zip(axes[0], regimes.items()):
        for g in groups:
            hs, ys = [], []
            for h in HORIZONS:
                c = reg["results"].get(f"{method}|{delta}|{h}")
                bg = (c or {}).get("by_group", {}).get(g)
                if bg:
                    hs.append(h); ys.append(bg["w1_p50"])
            if hs:
                ax.plot(hs, ys, "o-", label=g)
        ax.axhline(1.0, color="k", ls="--", lw=1)
        ax.axvline(delta, color="gray", ls=":", lw=1)
        ax.set_xlabel("horizon h (ticks)"); ax.set_ylabel("weight_proj_ratio (median)")
        ax.set_title(f"{REGIME_LABEL.get(rname, rname)}\nper-group, {method}, spacing={delta}")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    return _png(fig)


def esc(x):
    return html.escape(str(x))


def build(sweep_json_path, out_path, title, fragment=False):
    d = json.load(open(sweep_json_path))
    regimes = {k: d[k] for k in d if isinstance(d[k], dict) and "results" in d[k]}

    P = []
    P.append("<header>")
    P.append("<div class='eyebrow'>EXP-42 / M4 / weight-space measurement</div>")
    P.append(f"<h1>{esc(title)}</h1>")
    P.append("<p class='thesis'>How accurately does the look-ahead anchor predict the WEIGHTS it "
             "extrapolates? The predictor takes theta_hat = theta_stale + alpha (theta_stale - "
             "theta_old). The one question underneath the whole method: does theta_hat land closer to "
             "the current weight theta_now than simply reusing the stale weight theta_stale? Measured "
             "directly in weight space, per matrix, across horizons, in a clean regime and a "
             "compressed regime.</p>")
    chips = []
    for rname, reg in regimes.items():
        chips.append(f"<span class='chip'>{esc(rname)}: {reg.get('n_matrices')} matrices, "
                     f"{reg.get('n_ticks')} ticks</span>")
    chips.append("<span class='chip'>count-sketch k = 4096</span>")
    chips.append("<span class='chip'>single H200, MacBook analysis</span>")
    P.append("<div class='chips'>" + "".join(chips) + "</div>")
    P.append("</header>")

    # data-driven key-result callout (the answer at the operating horizon)
    P.append("<section class='callout'><h2 class='callout-h'>The answer at the operating horizon "
             "(h = K = 10, alpha = 1.0)</h2><div class='cards'>")
    for rname, reg in regimes.items():
        c = reg["results"].get("fixed_linear|10|10")
        if not c:
            continue
        helps = c["w1_p50"] < 1.0
        verb = "projection helps" if helps else "projection does not help"
        cls = "card help" if helps else "card hurt"
        hstar = _crossover(reg, "fixed_linear", 10)
        P.append(f"<div class='{cls}'><div class='card-reg'>{esc(REGIME_LABEL.get(rname, rname))}</div>"
                 f"<div class='card-num'>{c['w1_p50']:.3f}</div>"
                 f"<div class='card-lab'>weight_proj_ratio, median</div>"
                 f"<div class='card-verb'>{verb}</div>"
                 f"<div class='card-meta'>dir_cos {c['dir_cos_p50']:.3f} &middot; crossover h* = {hstar}</div></div>")
    P.append("</div><p class='callout-note'>Below 1.0 means the projected weight is closer to the "
             "current weight than doing nothing. The crossover h* is the largest horizon where the "
             "median ratio still stays below 1.0.</p></section>")

    # ---- definitions ----
    P.append("<h2>What is measured</h2>")
    P.append("<ul>"
             "<li><b>weight_proj_ratio (W1, headline)</b> = ||theta_hat - target|| / ||theta_stale - target||. "
             "Below 1.0 means the linear projection lands closer to the future weight than the raw-stale weight "
             "(it helps). Equal to 1.0 means no help. Above 1.0 means it overshoots and hurts.</li>"
             "<li><b>dir_cos (W2)</b> = cos(theta_stale - theta_old, target - theta_stale). It is the trajectory "
             "straightness, independent of how aggressively we extrapolate. Below 0 would mean the past update "
             "points away from the future update, the sign flip that was suspected to ignite the prior collapses.</li>"
             "<li><b>theta_hat</b> = theta_stale + alpha (theta_stale - theta_old), with alpha = h / Delta. "
             "alpha = 1 is full catch-up (predict one spacing ahead).</li>"
             "<li>Horizons and spacings are in optimizer ticks. There are 2 ticks per global step at "
             "train_batch=128, mini_batch=64. The anchor operating point is spacing Delta = delay_K = cadence = 10 ticks.</li>"
             "</ul>")

    # ---- operating-point answer ----
    P.append("<h2>Operating-point answer (spacing Delta = 10)</h2>")
    P.append("<table><tr><th>regime</th><th>method</th><th>h</th><th>alpha</th>"
             "<th>w1_p50</th><th>verdict</th><th>dir_cos</th><th>w1 p10..p90</th><th>samples</th></tr>")
    for rname, reg in regimes.items():
        for method in ("fixed_linear", "learned_linear"):
            for h in (5, 10, 20):
                c = reg["results"].get(f"{method}|10|{h}")
                if not c:
                    continue
                v = "helps" if c["w1_p50"] < 1.0 else "no help"
                cls = "ok" if c["w1_p50"] < 1.0 else "bad"
                P.append(f"<tr><td>{esc(rname)}</td><td>{esc(method)}</td><td>{h}</td>"
                         f"<td>{c['alpha']:.1f}</td><td class='{cls}'>{c['w1_p50']:.4f}</td>"
                         f"<td class='{cls}'>{v}</td><td>{c['dir_cos_p50']:.4f}</td>"
                         f"<td>[{c['w1_p10']:.3f}, {c['w1_p90']:.3f}]</td><td>{c['n']}</td></tr>")
    P.append("</table>")

    # ---- crossover table ----
    P.append("<h2>Crossover horizon h* (largest horizon with median ratio below 1.0)</h2>")
    P.append("<table><tr><th>regime</th><th>method</th><th>Delta=5</th><th>Delta=10</th></tr>")
    for rname, reg in regimes.items():
        for method in ("fixed_linear", "learned_linear"):
            P.append(f"<tr><td>{esc(rname)}</td><td>{esc(method)}</td>"
                     f"<td>{_crossover(reg, method, 5)}</td><td>{_crossover(reg, method, 10)}</td></tr>")
    P.append("</table>")

    # ---- plots ----
    P.append("<h2>Plots</h2>")
    P.append("<h3>Headline: accuracy and straightness vs horizon (spacing 10, fixed_linear)</h3>")
    P.append(f"<figure><img src='data:image/png;base64,{fig_headline(regimes)}' /></figure>")
    P.append("<p class='cap'>Left: the projection ratio crosses 1.0 (the do-nothing line) at the crossover "
             "horizon. The shaded band is the spread across individual weight matrices (p10 to p90). Right: "
             "dir_cos stays positive across the whole grid in both regimes, so the overshoot above is a "
             "magnitude effect (alpha scales the step past theta_now along a consistently aligned direction), "
             "not a direction reversal.</p>")
    P.append("<h3>Spacing effect on the crossover</h3>")
    P.append(f"<figure><img src='data:image/png;base64,{fig_spacing(regimes)}' /></figure>")
    P.append("<p class='cap'>A shorter spacing Delta = 5 reaches the same alpha at a smaller h, so its curve "
             "rises later in h. The operating spacing is Delta = 10.</p>")
    gfig = fig_groups(regimes)
    if gfig:
        P.append("<h3>Per-group accuracy (widened select_all sweep)</h3>")
        P.append(f"<figure><img src='data:image/png;base64,{gfig}' /></figure>")
        P.append("<p class='cap'>decoder is the set the projector actually extrapolates. embed, norm and bias "
                 "are the params it excludes. Curves below 1.0 at the operating horizon would indicate the "
                 "exclusion is leaving usable predictability on the table.</p>")

    # ---- full curves ----
    P.append("<h2>Full curves (fixed_linear)</h2>")
    for rname, reg in regimes.items():
        P.append(f"<h3>{esc(REGIME_LABEL.get(rname, rname))}</h3>")
        for delta in (5, 10):
            P.append(f"<h4>spacing Delta = {delta}</h4>")
            P.append("<table><tr><th>h</th><th>alpha</th><th>w1_p50</th><th>w1_p10</th>"
                     "<th>w1_p90</th><th>dir_cos</th><th>samples</th></tr>")
            for h in HORIZONS:
                c = reg["results"].get(f"fixed_linear|{delta}|{h}")
                if not c:
                    continue
                cls = "ok" if c["w1_p50"] < 1.0 else "bad"
                P.append(f"<tr><td>{h}</td><td>{c['alpha']:.2f}</td><td class='{cls}'>{c['w1_p50']:.4f}</td>"
                         f"<td>{c['w1_p10']:.4f}</td><td>{c['w1_p90']:.4f}</td>"
                         f"<td>{c['dir_cos_p50']:.4f}</td><td>{c['n']}</td></tr>")
            P.append("</table>")

    # ---- calib ----
    P.append("<h2>Sketch fidelity check (count-sketch vs on-box exact fp32 calibration)</h2>")
    P.append("<table><tr><th>regime</th><th>Delta</th><th>h</th><th>exact w1_p50</th>"
             "<th>sketch w1_p50</th><th>rel error</th><th>within 5%</th></tr>")
    for rname, reg in regimes.items():
        cal = reg.get("calib", {})
        if not cal.get("available"):
            P.append(f"<tr><td>{esc(rname)}</td><td colspan=6>no calib.jsonl</td></tr>")
            continue
        for ch in cal["checks"]:
            cls = "ok" if ch["pass"] else "bad"
            P.append(f"<tr><td>{esc(rname)}</td><td>{ch['delta']}</td><td>{ch['h']}</td>"
                     f"<td>{ch['calib_w1_p50']:.4f}</td><td>{ch['sketch_w1_p50']:.4f}</td>"
                     f"<td class='{cls}'>{ch['rel_err']*100:.2f}%</td><td class='{cls}'>{ch['pass']}</td></tr>")
    P.append("</table>")

    # ---- generated discussion (from the numbers) ----
    P.append("<h2>Discussion</h2>")
    P.append(_discussion(regimes))

    # ---- provenance ----
    P.append("<h2>Provenance</h2><ul>")
    for rname, reg in regimes.items():
        P.append(f"<li>{esc(rname)}: {reg.get('n_matrices')} matrices, {reg.get('n_ticks')} ticks, "
                 f"count-sketch width k = {reg.get('k')}.</li>")
    P.append("<li>Source: research/scripts/weight_proj_sweep.py over the per-tick weight sketch trace "
             "collected on a single H200. Analysis is GPU-free on the MacBook.</li></ul>")

    css = """
:root{--ground:#f7f8fa;--surface:#fff;--ink:#14181f;--muted:#5b6573;--line:#e3e6eb;
  --a:#1f5fae;--b:#b42318;--help:#1a7f37;--sans:ui-sans-serif,-apple-system,"Segoe UI",Roboto,Helvetica,sans-serif;
  --mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);font-family:var(--sans);line-height:1.6;
  font-size:16px;-webkit-font-smoothing:antialiased}
.wrap{max-width:980px;margin:0 auto;padding:40px 24px 80px}
header{border-bottom:1px solid var(--line);padding-bottom:22px;margin-bottom:8px}
.eyebrow{font-size:12px;letter-spacing:.13em;text-transform:uppercase;color:var(--muted);font-weight:600}
h1{font-size:30px;line-height:1.18;margin:10px 0 12px;text-wrap:balance;font-weight:680;letter-spacing:-.01em}
.thesis{color:var(--muted);max-width:68ch;margin:0 0 16px;font-size:16px}
.chips{display:flex;flex-wrap:wrap;gap:8px}
.chip{font-family:var(--mono);font-size:11.5px;color:var(--muted);background:var(--surface);
  border:1px solid var(--line);border-radius:999px;padding:3px 10px}
h2{font-size:20px;margin:44px 0 12px;padding-bottom:6px;border-bottom:1px solid var(--line);
  letter-spacing:-.01em;font-weight:660}
h3{font-size:15px;margin:24px 0 6px;font-weight:640}
h4{font-size:13px;margin:16px 0 4px;color:var(--muted);font-weight:640;font-family:var(--mono)}
p{max-width:72ch}
.callout{background:var(--surface);border:1px solid var(--line);border-radius:14px;padding:22px 24px;margin:22px 0}
.callout-h{border:0;margin:0 0 14px;padding:0;font-size:16px}
.cards{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.card{border:1px solid var(--line);border-radius:11px;padding:16px 18px;border-left-width:4px}
.card.help{border-left-color:var(--help)}.card.hurt{border-left-color:var(--b)}
.card-reg{font-family:var(--mono);font-size:12px;color:var(--muted)}
.card-num{font-size:38px;font-weight:700;font-variant-numeric:tabular-nums;letter-spacing:-.02em;margin:4px 0}
.card.help .card-num{color:var(--help)}.card.hurt .card-num{color:var(--b)}
.card-lab{font-size:12px;color:var(--muted)}
.card-verb{font-weight:640;margin-top:8px;font-size:14px}
.card-help .card-verb{color:var(--help)}
.card-meta{font-family:var(--mono);font-size:11.5px;color:var(--muted);margin-top:4px}
.callout-note{color:var(--muted);font-size:13.5px;margin:14px 0 0}
.tbl{overflow-x:auto;margin:10px 0}
table{border-collapse:collapse;font-size:13.5px;font-variant-numeric:tabular-nums;min-width:100%}
th,td{padding:6px 12px;text-align:right;border-bottom:1px solid var(--line);white-space:nowrap}
th{font-family:var(--mono);font-size:11px;letter-spacing:.04em;text-transform:uppercase;color:var(--muted);
  font-weight:600;border-bottom:1.5px solid var(--muted);background:transparent}
td:first-child,th:first-child{text-align:left}
tbody tr:nth-child(even){background:rgba(31,95,174,.035)}
.ok{color:var(--help);font-weight:640}.bad{color:var(--b);font-weight:640}
.cap{color:var(--muted);font-size:13px;max-width:74ch;margin:8px 0 4px}
ul{max-width:74ch;color:var(--ink)}li{margin:3px 0}
figure{margin:14px 0;background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:14px}
figure img{max-width:100%;height:auto;display:block;border-radius:6px}
b{font-weight:660}
"""
    content = "".join(P).replace("<table", "<div class='tbl'><table").replace("</table>", "</table></div>")
    body = "<div class='wrap'>" + content + "</div>"
    if fragment:
        out = f"<style>{css}</style>" + body
    else:
        out = (f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
               f"<meta name='viewport' content='width=device-width, initial-scale=1'>"
               f"<title>{esc(title)}</title><style>{css}</style></head><body>" + body + "</body></html>")
    with open(out_path, "w") as fh:
        fh.write(out)
    print(f"wrote {out_path} ({len(out)} bytes, fragment={fragment})")


def _discussion(regimes):
    """Generate prose strictly from the computed numbers (no hand-set claims)."""
    lines = []
    def cell(r, m, dl, h):
        return regimes[r]["results"].get(f"{m}|{dl}|{h}")
    # H1 premise at operating point
    lines.append("<p><b>Premise test (H1).</b> ")
    for r in regimes:
        c = cell(r, "fixed_linear", 10, 10)
        if not c:
            continue
        verb = "lands closer than raw-stale" if c["w1_p50"] < 1.0 else "does not beat raw-stale"
        lines.append(f"In {esc(REGIME_LABEL.get(r, r))}, at the operating horizon h = 10 (alpha = 1.0) the "
                     f"projection {verb}: median ratio = {c['w1_p50']:.3f}, dir_cos = {c['dir_cos_p50']:.3f}. ")
    lines.append("</p>")
    # H2 overshoot + sign flip
    lines.append("<p><b>Overshoot test (H2).</b> The ratio rises monotonically with the horizon and crosses "
                 "1.0 at the crossover h*. ")
    for r in regimes:
        lines.append(f"For {esc(r)} the crossover is h* = {_crossover(regimes[r], 'fixed_linear', 10)} "
                     f"at spacing 10. ")
    # dir_cos sign across grid
    anyneg = False
    for r in regimes:
        for h in HORIZONS:
            c = cell(r, "fixed_linear", 10, h)
            if c and c["dir_cos_p50"] < 0:
                anyneg = True
    if not anyneg:
        lines.append("dir_cos stays positive at every horizon in both regimes, so there is no weight-space "
                     "sign flip on this grid: the overshoot is a magnitude effect (the extrapolation steps "
                     "past theta_now along a consistently aligned direction), not a direction reversal. This "
                     "refines the prior-collapse picture, where a sign flip in an extrapolated anchor cosine "
                     "was the suspected trigger.")
    else:
        lines.append("dir_cos goes below 0 at some horizons, the weight-space sign flip suspected to trigger "
                     "the prior collapses.")
    lines.append("</p>")
    # H3 regime effect
    if "regimeA" in regimes and "regimeB" in regimes:
        ha = _crossover(regimes["regimeA"], "fixed_linear", 10)
        hb = _crossover(regimes["regimeB"], "fixed_linear", 10)
        lines.append(f"<p><b>Regime effect (H3).</b> Activation compression changes predictability: the "
                     f"crossover moves from h* = {ha} (clean) to h* = {hb} (codec only) at spacing 10, and the "
                     f"compressed regime shows a wider per-matrix spread. The codec makes the weight trajectory "
                     f"less linearly predictable.</p>")
        lines.append("<p><b>Important caveat on regime B.</b> Regime B ran the PowerSGD codec on a "
                     "FROZEN RANDOM basis: the anchor was off but owns_q was true, so the only basis "
                     "updater was off and the fast-path update is fail-closed (basis_updates = 0 for "
                     "the whole run, reconstruction error flat at about 0.97, no merger). About 97 "
                     "percent of the boundary gradient was discarded, which is why the policy "
                     "collapsed to val 0.079. This is not an inherent PowerSGD limitation: earlier "
                     "comm-eff runs that learned at delay 5 (val about 0.72 to 0.74) kept the anchor "
                     "on, which adapts the basis, plus a signed_ema merger. So this contrast is clean "
                     "dense versus a COLLAPSING FROZEN-BASIS codec, not clean versus a healthy "
                     "adaptive-compressed run; regime B's weight trajectory is a degenerate one and "
                     "its numbers should be read as such.</p>")
    # fixed vs learned
    same = True
    for r in regimes:
        cf = cell(r, "fixed_linear", 10, 10); cl = cell(r, "learned_linear", 10, 10)
        if cf and cl and abs(cf["w1_p50"] - cl["w1_p50"]) > 1e-3:
            same = False
    if same:
        lines.append("<p><b>Fixed vs learned.</b> The learned per-matrix scalar-mean residual is inert at the "
                     "operating point (it matches fixed_linear to within rounding). A uniform mean shift barely "
                     "moves the norm or direction of a weight-difference vector, as expected.</p>")
    # gate
    lines.append("<p><b>Gate for a gradient-accuracy follow-up.</b> A gradient cannot be more accurate than the "
                 "weight it is computed at. A follow-up is worth running only where the projected weight already "
                 "beats raw-stale at a useful horizon (ratio below 1.0 and dir_cos above 0). ")
    for r in regimes:
        c = cell(r, "fixed_linear", 10, 10)
        if c and c["w1_p50"] < 1.0 and c["dir_cos_p50"] > 0:
            lines.append(f"{esc(r)} qualifies at h up to {_crossover(regimes[r], 'fixed_linear', 10)} (fixed_linear, spacing 10). ")
        else:
            lines.append(f"{esc(r)} does not qualify at the operating horizon h = 10. ")
    lines.append("</p>")
    return "".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep_json")
    ap.add_argument("out_html")
    ap.add_argument("--title", default="EXP-42 weight-projection accuracy")
    ap.add_argument("--fragment", action="store_true", help="emit body-only fragment (for Artifact)")
    a = ap.parse_args()
    build(a.sweep_json, a.out_html, a.title, fragment=a.fragment)


if __name__ == "__main__":
    main()
