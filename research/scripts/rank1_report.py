#!/usr/bin/env python3
"""rank1_report.py — render runs/RANK1-ANALYSIS/report.html from the two
rank1_scorecard runs (late anchors = scorecard/, early anchors =
scorecard-early/). Output is a single self-contained HTML file: inline SVG
plots, no JS, no external assets. Companion to rank1_scorecard.py; findings
prose mirrors runs/RANK1-ANALYSIS/verdict.md (single source: the verdict file
is canonical, this report visualizes it).

Usage:  python3 scripts/rank1_report.py            (from research/)
"""
from __future__ import annotations

import html
import json
import math
import os
import sys

import numpy as np

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                    "runs", "RANK1-ANALYSIS")
LATE = os.path.join(BASE, "scorecard")
EARLY = os.path.join(BASE, "scorecard-early")
OUT = os.path.join(BASE, "report.html")

H_GRID = [1, 2, 5, 10, 20, 40]
WSPECS = ["8", "16", "32", "prefix"]

C = {"hold": "#8a8f85", "naive": "#B23A2B", "two": "#D98E2B",
     "r1t": "#3441C9", "r1a": "#148A56", "r2a": "#7A2BD9", "r2t": "#7fa8dd",
     "ink": "#22262a", "mut": "#5b6167", "grid": "#e6e7e3"}


def load_rows(d):
    with open(os.path.join(d, "rows.jsonl")) as f:
        return [json.loads(l) for l in f]


def med(vals):
    v = [x for x in vals if x is not None and np.isfinite(x)]
    return float(np.median(v)) if v else float("nan")


def gratio(rows, method, wspec, anchor=None):
    """h -> ratio for global rows (median over anchors unless one is given)."""
    out = {}
    for h in H_GRID:
        vs = [r["weight_proj_ratio"] for r in rows
              if r["group_kind"] == "global" and r["method"] == method
              and (r.get("window_spec") or "-") == wspec and r["h_ticks"] == h
              and (anchor is None or r["anchor_tick"] == anchor)]
        out[h] = med(vs)
    return out


# =============================================================================
# tiny SVG plot kit
# =============================================================================
def _yt(v, ylog, y0, y1, top, bot):
    t = (math.log10(v) - math.log10(y0)) / (math.log10(y1) - math.log10(y0)) \
        if ylog else (v - y0) / (y1 - y0)
    return bot - t * (bot - top)


def line_plot(series, *, w=880, h=380, ylog=False, yrange, yticks, title,
              ylab="weight_proj_ratio", xlab="horizon h (ticks ahead of anchor)",
              refline=1.0, xvals=None, xlabels=None, xlog=None,
              ref_label="ratio = 1 (hold-stale / do nothing)",
              better_note="↓ lower = better", shade=False):
    """series: [{name,color,pts:{x:y},dash?}] with shared log2 or categorical x.

    Every plot states its polarity: `better_note` is a top-right badge (set the
    arrow/direction per metric); `shade=True` tints the regions on either side
    of `refline` (green = beats the do-nothing reference, red = harmful)."""
    xvals = xvals or H_GRID
    xlabels = xlabels or [str(x) for x in xvals]
    if xlog is None:
        xlog = all(isinstance(v, (int, float)) and v > 0 for v in xvals)
    ml, mr, mt, mb = 62, 18, 34, 48
    left, right, top, bot = ml, w - mr, mt, h - mb
    lx = [math.log2(v) for v in xvals] if xlog else list(range(len(xvals)))
    x0, x1 = min(lx), max(lx)

    def X(v):
        i = xvals.index(v)
        return left + (lx[i] - x0) / (x1 - x0) * (right - left)

    y0, y1 = yrange
    parts = [f'<svg viewBox="0 0 {w} {h}" role="img" '
             f'style="width:100%;height:auto;display:block">']
    parts.append(f'<text x="{ml}" y="20" font-size="13.5" font-weight="600" '
                 f'fill="{C["ink"]}">{html.escape(title)}</text>')
    if better_note:
        parts.append(f'<text x="{right}" y="20" font-size="11.5" '
                     f'font-weight="600" fill="#14532d" text-anchor="end">'
                     f'{html.escape(better_note)}</text>')
    if shade and refline is not None and y0 < refline < y1:
        yref = _yt(refline, ylog, y0, y1, top, bot)
        parts.append(f'<rect x="{left}" y="{top}" width="{right - left}" '
                     f'height="{yref - top:.1f}" fill="#B23A2B" opacity="0.05"/>')
        parts.append(f'<rect x="{left}" y="{yref:.1f}" width="{right - left}" '
                     f'height="{bot - yref:.1f}" fill="#2e8b57" opacity="0.06"/>')
        parts.append(f'<text x="{right - 8}" y="{top + 14}" font-size="10.5" '
                     f'fill="#B23A2B" text-anchor="end">above the line = HARMFUL '
                     f'(worse than doing nothing)</text>')
        parts.append(f'<text x="{right - 8}" y="{bot - 8}" font-size="10.5" '
                     f'fill="#14532d" text-anchor="end">below the line = GOOD '
                     f'(beats holding stale weights)</text>')
    for tv in yticks:
        y = _yt(tv, ylog, y0, y1, top, bot)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" '
                     f'stroke="{C["grid"]}" stroke-width="1"/>')
        parts.append(f'<text x="{left - 8}" y="{y + 3.5:.1f}" font-size="11" '
                     f'fill="{C["mut"]}" text-anchor="end">{tv:g}</text>')
    for v, lab in zip(xvals, xlabels):
        x = X(v)
        parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{bot}" '
                     f'stroke="{C["grid"]}" stroke-width="1"/>')
        parts.append(f'<text x="{x:.1f}" y="{bot + 18}" font-size="11" '
                     f'fill="{C["mut"]}" text-anchor="middle">{lab}</text>')
    if refline is not None and y0 < refline < y1:
        y = _yt(refline, ylog, y0, y1, top, bot)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" '
                     f'stroke="{C["ink"]}" stroke-width="1.3" '
                     f'stroke-dasharray="5 5" opacity="0.75"/>')
        parts.append(f'<text x="{right}" y="{y - 6:.1f}" font-size="10.5" '
                     f'fill="{C["ink"]}" text-anchor="end" opacity="0.8">'
                     f'{html.escape(ref_label)}</text>')
    parts.append(f'<line x1="{left}" y1="{bot}" x2="{right}" y2="{bot}" '
                 f'stroke="#aaa" stroke-width="1.2"/>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bot}" '
                 f'stroke="#aaa" stroke-width="1.2"/>')
    parts.append(f'<text x="{(left + right) / 2:.0f}" y="{h - 10}" font-size="11.5" '
                 f'fill="{C["mut"]}" text-anchor="middle">{html.escape(xlab)}</text>')
    parts.append(f'<text x="16" y="{(top + bot) / 2:.0f}" font-size="11.5" '
                 f'fill="{C["mut"]}" text-anchor="middle" '
                 f'transform="rotate(-90 16 {(top + bot) / 2:.0f})">'
                 f'{html.escape(ylab)}</text>')
    for s in series:
        pts = [(X(x), _yt(min(max(s["pts"][x], y0), y1), ylog, y0, y1, top, bot))
               for x in xvals if x in s["pts"] and np.isfinite(s["pts"][x])]
        if not pts:
            continue
        pl = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        dash = f' stroke-dasharray="{s["dash"]}"' if s.get("dash") else ""
        parts.append(f'<polyline points="{pl}" fill="none" '
                     f'stroke="{s["color"]}" stroke-width="2.4"{dash}/>')
        for x, y in pts:
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.1" '
                         f'fill="{s["color"]}"/>')
        # clip marker: value above y1 gets an up-arrow annotation at first clip
        for x in xvals:
            if x in s["pts"] and np.isfinite(s["pts"][x]) and s["pts"][x] > y1:
                parts.append(f'<text x="{X(x):.1f}" y="{top + 12}" font-size="10" '
                             f'fill="{s["color"]}" text-anchor="middle">'
                             f'{s["pts"][x]:.1f}&#8593;</text>')
    parts.append("</svg>")
    return "".join(parts)


def legend(items):
    row = "".join(
        f'<span style="display:inline-flex;align-items:center;gap:6px;'
        f'margin-right:16px;white-space:nowrap"><span style="width:16px;'
        f'height:3px;background:{c};display:inline-block;border-radius:2px">'
        f'</span><span>{html.escape(n)}</span></span>'
        for n, c in items)
    return (f'<div style="font-family:ui-monospace,Menlo,monospace;font-size:'
            f'11.5px;color:{C["mut"]};padding:6px 2px 0">{row}</div>')


def hist_plot(vals, *, bins, rng, w=430, h=250, title, vline=None,
              vline_label="", xlab, better_note="→ higher = better"):
    counts, edges = np.histogram(np.clip(vals, rng[0], rng[1]), bins=bins,
                                 range=rng)
    ml, mr, mt, mb = 40, 12, 44, 40
    left, right, top, bot = ml, w - mr, mt, h - mb
    cmax = max(int(counts.max()), 1)
    bw = (right - left) / bins
    parts = [f'<svg viewBox="0 0 {w} {h}" role="img" '
             f'style="width:100%;height:auto;display:block">']
    parts.append(f'<text x="{ml}" y="18" font-size="12.5" font-weight="600" '
                 f'fill="{C["ink"]}">{html.escape(title)}</text>')
    if better_note:
        parts.append(f'<text x="{right}" y="{mt - 10}" font-size="11" '
                     f'font-weight="600" fill="#14532d" text-anchor="end">'
                     f'{html.escape(better_note)}</text>')
    for i, c in enumerate(counts):
        if c == 0:
            continue
        bh = (bot - top) * c / cmax
        parts.append(f'<rect x="{left + i * bw + 0.5:.1f}" y="{bot - bh:.1f}" '
                     f'width="{bw - 1:.1f}" height="{bh:.1f}" fill="{C["r1t"]}" '
                     f'opacity="0.75"/>')
    parts.append(f'<line x1="{left}" y1="{bot}" x2="{right}" y2="{bot}" '
                 f'stroke="#aaa" stroke-width="1.2"/>')
    for tv in np.linspace(rng[0], rng[1], 6):
        x = left + (tv - rng[0]) / (rng[1] - rng[0]) * (right - left)
        parts.append(f'<text x="{x:.1f}" y="{bot + 16}" font-size="10.5" '
                     f'fill="{C["mut"]}" text-anchor="middle">{tv:.2f}</text>')
    if vline is not None:
        x = left + (vline - rng[0]) / (rng[1] - rng[0]) * (right - left)
        parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{bot}" '
                     f'stroke="{C["naive"]}" stroke-width="1.6" '
                     f'stroke-dasharray="4 4"/>')
        parts.append(f'<text x="{x - 4:.1f}" y="{top + 10}" font-size="10.5" '
                     f'fill="{C["naive"]}" text-anchor="end">'
                     f'{html.escape(vline_label)}</text>')
    parts.append(f'<text x="{(left + right) / 2:.0f}" y="{h - 8}" '
                 f'font-size="11" fill="{C["mut"]}" text-anchor="middle">'
                 f'{html.escape(xlab)}</text>')
    parts.append("</svg>")
    return "".join(parts)


# =============================================================================
# build the report
# =============================================================================
def main() -> int:
    late = load_rows(LATE)
    early = load_rows(EARLY)
    lm = [r for r in late if r["group_kind"] == "matrix"]

    # ---- plot A: ratio vs h, key arms (late, median over anchors) ----------
    plot_a = line_plot([
        {"name": "hold_stale", "color": C["hold"],
         "pts": gratio(late, "hold_stale", "-"), "dash": "4 4"},
        {"name": "naive_last2", "color": C["naive"],
         "pts": gratio(late, "naive_last2", "-")},
        {"name": "two_point_window[8]", "color": C["two"],
         "pts": gratio(late, "two_point_window", "8")},
        {"name": "rank1_traj[8] (paper form)", "color": C["r1t"],
         "pts": gratio(late, "rank1_traj", "8")},
        {"name": "rank2_anchored[8]", "color": C["r2a"],
         "pts": gratio(late, "rank2_anchored", "8")},
        {"name": "rank1_anchored[8]", "color": C["r1a"],
         "pts": gratio(late, "rank1_anchored", "8")},
    ], ylog=True, yrange=(0.4, 4.2), yticks=[0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0],
        title="A · Staleness skill by horizon — global pooled ratio, late anchors "
              "(79, 119), W = 8, log axes",
        better_note="↓ lower = better", shade=True)
    leg_a = legend([("hold_stale (=1)", C["hold"]), ("naive_last2", C["naive"]),
                    ("two_point_window[8]", C["two"]),
                    ("rank1_traj[8] · RELEX form", C["r1t"]),
                    ("rank2_anchored[8]", C["r2a"]),
                    ("rank1_anchored[8]", C["r1a"])])

    # ---- plot B: window-size knob ------------------------------------------
    def by_window(method, h):
        return {i: med([r["weight_proj_ratio"] for r in late
                        if r["group_kind"] == "global" and r["method"] == method
                        and r.get("window_spec") == w and r["h_ticks"] == h])
                for i, w in enumerate(WSPECS)}

    off = {i: med([r["anchor_offline_frac"] for r in lm
                   if r["method"] == "rank1_traj" and r["h_ticks"] == 1
                   and r.get("window_spec") == w])
           for i, w in enumerate(WSPECS)}
    plot_b1 = line_plot([
        {"name": "rank1_anchored h=10", "color": C["r1a"],
         "pts": by_window("rank1_anchored", 10)},
        {"name": "rank1_anchored h=40", "color": "#0b5e39",
         "pts": by_window("rank1_anchored", 40), "dash": "6 4"},
        {"name": "rank1_traj h=40", "color": C["r1t"],
         "pts": by_window("rank1_traj", 40)},
    ], w=430, h=280, yrange=(0.9, 2.0), yticks=[1.0, 1.2, 1.5, 1.8],
        title="B1 · Ratio vs window size (late)", xvals=[0, 1, 2, 3],
        xlabels=WSPECS, xlab="checkpoints feeding the SVD (W)",
        better_note="↓ lower = better", shade=True)
    plot_b2 = line_plot([
        {"name": "off-line share", "color": C["naive"], "pts": off},
    ], w=430, h=280, yrange=(0.0, 0.4), yticks=[0.0, 0.1, 0.2, 0.3, 0.4],
        title="B2 · Off-v₁ share of accumulated Δθ at anchor",
        xvals=[0, 1, 2, 3], xlabels=WSPECS, refline=None,
        ylab="fraction of ||Δθ_anchor|| off the line",
        xlab="checkpoints feeding the SVD (W)",
        better_note="↓ lower = smaller unavoidable residual")
    leg_b = legend([("rank1_anchored h=10", C["r1a"]),
                    ("rank1_anchored h=40 (dashed)", "#0b5e39"),
                    ("rank1_traj h=40", C["r1t"]),
                    ("B2: off-line residual share (v₁ rotates ⇒ grows with W)",
                     C["naive"])])

    # ---- plot C: paper replication histograms (prefix window, anchor 119) --
    pref = [r for r in lm if r["method"] == "rank1_traj"
            and r.get("window_spec") == "prefix" and r["h_ticks"] == 1
            and r["anchor_tick"] == 119]
    coef = [r["coef_r2_1"] for r in pref if r.get("coef_r2_1") is not None]
    evr = [r["evr1"] for r in pref if r.get("evr1") is not None]
    plot_c1 = hist_plot(coef, bins=30, rng=(0.4, 1.0),
                        title=f"C1 · coef-linearity R² per matrix "
                              f"(median {np.median(coef):.3f})",
                        vline=0.98, vline_label="paper bar: 0.98",
                        xlab="R² of c(t)=at+b, prefix window, anchor 119",
                        better_note="→ higher = more linear (1 = perfect)")
    plot_c2 = hist_plot(evr, bins=25, rng=(0.9, 1.0),
                        title=f"C2 · rank-1 energy share EVR₁ "
                              f"(median {np.median(evr):.4f})",
                        xlab="λ₁/trace(G) per matrix (paper: ~0.81 "
                             "of a rank-5 window)",
                        better_note="→ higher = more rank-1 (1 = pure line)")

    # ---- plot D: early vs late anchors, anchored form ------------------------
    shades = {39: "#8fcdb0", 59: "#4aab7f", 79: "#148A56", 119: "#0b4b30"}
    series_d = ([{"name": f"anchor {a}", "color": shades[a],
                  "pts": gratio(early, "rank1_anchored", "8", a)}
                 for a in (39, 59)] +
                [{"name": f"anchor {a}", "color": shades[a],
                  "pts": gratio(late, "rank1_anchored", "8", a)}
                 for a in (79, 119)])
    plot_d = line_plot(series_d, w=430, h=280, yrange=(0.94, 1.12),
                       yticks=[0.95, 1.0, 1.05, 1.10],
                       title="D · rank1_anchored[8] at four anchors",
                       better_note="↓ lower = better", shade=True)
    leg_d = legend([(f"anchor {a}", shades[a]) for a in (39, 59, 79, 119)])

    # ---- plot E: residual geometry -------------------------------------------
    ra = [r for r in lm if r["method"] == "rank1_anchored"
          and r.get("window_spec") == "8"]
    rad = {h: med([r["radial"] / r["base_norm"] for r in ra
                   if r["h_ticks"] == h and r["base_norm"]]) for h in H_GRID}
    tan = {h: med([r["tangential"] / r["base_norm"] for r in ra
                   if r["h_ticks"] == h and r["base_norm"]]) for h in H_GRID}
    plot_e = line_plot([
        {"name": "radial", "color": C["r1t"], "pts": rad},
        {"name": "tangential", "color": C["two"], "pts": tan},
    ], w=430, h=280, yrange=(0.0, 1.1), yticks=[0.0, 0.25, 0.5, 0.75, 1.0],
        title="E · Where the anchored residual lives", refline=1.0,
        ylab="residual component / ||true move||",
        ref_label="1.0 = residual as large as the whole move (zero skill)",
        better_note="↓ lower = better (0 = move fully predicted)")
    leg_e = legend([("radial: along the true move (≈0.97 ⇒ the line "
                     "predicted ~none of it)", C["r1t"]),
                    ("tangential: sideways error added", C["two"])])

    # ---- ratio table (late) ---------------------------------------------------
    tbl_rows = [("hold_stale", "-"), ("naive_last2", "-"),
                ("two_point_window", "8"), ("rank1_traj", "8"),
                ("rank1_traj", "prefix"), ("rank1_anchored", "8"),
                ("rank1_anchored", "16"), ("rank2_anchored", "8"),
                ("rank2_traj", "8")]

    def cell(v):
        if not np.isfinite(v):
            return "<td>-</td>"
        bg = ("#e7f3ea" if v < 0.98 else
              "#fdeeea" if v > 1.05 else "#f7f7f2")
        return f'<td style="background:{bg}">{v:.3f}</td>'

    table = ['<div style="font-family:ui-monospace,Menlo,monospace;font-size:12px;'
             'color:#14532d;font-weight:600;padding:2px 0 6px">'
             'weight_proj_ratio — ↓ LOWER = BETTER · 1.000 = break-even with '
             'doing nothing (hold-stale) · &lt;1 beats it · &gt;1 harmful</div>',
             '<table><thead><tr><th>method [window]</th>' +
             "".join(f"<th>h={h} ↓</th>" for h in H_GRID) + "</tr></thead><tbody>"]
    for m, w in tbl_rows:
        g = gratio(late, m, w)
        table.append(f'<tr><td class="rh">{m}[{w}]</td>' +
                     "".join(cell(g[h]) for h in H_GRID) + "</tr>")
    table.append("</tbody></table>")
    table_html = "".join(table)

    doc = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>RANK1-ANALYSIS — RELEX rank-1 trajectory family on the EXP-57 trace</title>
<style>
body{{background:#f4f4ef;color:{C['ink']};font-family:'Helvetica Neue',Arial,
sans-serif;font-size:15.5px;line-height:1.58;margin:0}}
.wrap{{max-width:960px;margin:0 auto;padding:36px 22px 80px}}
h1{{font-family:Georgia,'Times New Roman',serif;font-size:34px;line-height:1.15;
letter-spacing:-.01em;margin:6px 0 10px}}
h2{{font-family:Georgia,serif;font-size:22px;margin:40px 0 10px}}
p{{max-width:76ch;margin:0 0 14px}}
.k{{font-family:ui-monospace,Menlo,monospace;font-size:11.5px;letter-spacing:.08em;
text-transform:uppercase;color:{C['mut']}}}
.chips span{{display:inline-block;font-family:ui-monospace,Menlo,monospace;
font-size:11.5px;padding:3px 10px;border-radius:999px;margin:0 6px 6px 0;
background:#e7f3ea;color:#14532d;border:1px solid #bcdcc6}}
.chips span.n{{background:#eef0f8;color:#2b3a8c;border-color:#c9cfec}}
.panel{{background:#fdfdf9;border:1px solid #ddded6;border-radius:8px;
padding:18px 18px 12px;margin:16px 0}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
@media(max-width:760px){{.grid2{{grid-template-columns:1fr}}}}
code{{font-family:ui-monospace,Menlo,monospace;font-size:.88em;background:#ecece4;
border-radius:3px;padding:1px 5px}}
pre{{background:#22261f;color:#e8eae0;border-radius:6px;padding:14px 16px;
font-size:12.5px;line-height:1.6;overflow-x:auto}}
table{{border-collapse:collapse;width:100%;font-size:13.5px;
font-family:ui-monospace,Menlo,monospace}}
th{{font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:{C['mut']};
text-align:right;padding:7px 10px;border-bottom:2px solid #ddded6}}
th:first-child{{text-align:left}}
td{{padding:6px 10px;text-align:right;border-bottom:1px solid #e8e8e0}}
td.rh{{text-align:left;font-weight:600}}
.note{{background:#fdfdf9;border-left:3px solid {C['r1a']};border-radius:4px;
padding:12px 16px;margin:16px 0;max-width:80ch}}
li{{margin-bottom:7px}} ul{{max-width:78ch}}
a{{color:#2b3a8c}}
</style></head><body><div class="wrap">

<div class="k">verl comm-eff research &middot; offline weight-projection analysis &middot; 2026-07-03</div>
<h1>The rank-1 trajectory family, tested on our own trace</h1>
<p>RELEX (Wei et&nbsp;al. 2026, arXiv:2605.21468) predicts future RLVR checkpoints from a
short observed prefix: per-tensor SVD of the accumulated weight deltas &rarr; a rank-1
direction v&#8321; whose coefficient grows linearly &rarr; closed-form extrapolation
&theta;&#770;<sub>T</sub>&nbsp;=&nbsp;&theta;&#8320;&nbsp;+&nbsp;(aT+b)&middot;v&#8321;. We implemented it as a NEW offline
predictor family beside the CORE-4 raw-space arms (Paper&nbsp;A&rsquo;s two-point Weight
Extrapolation = our existing <code>naive_linear</code>), swept its two design knobs —
<b>how many checkpoints feed the SVD</b> (W&nbsp;=&nbsp;8/16/32/full-prefix) and <b>how far
ahead we project</b> (h&nbsp;=&nbsp;1&hellip;40 ticks) — on the EXP-57 fp32 trace
(Qwen2.5-1.5B GRPO/GSM8K, 160 optimizer ticks, 61-matrix panel), at early (39/59)
and late (79/119) anchors.</p>
<div class="chips">
<span>26/26 self-tests</span><span>hold-stale identity 2.2e-16</span>
<span>raw-tensor audit PASS &times;2 runs</span><span>independent verifier PASS @ ~1e-14</span>
<span class="n">metric contract: weight-proj-metrics-v1</span>
<span class="n">rows: 2&times;19,800</span>
</div>

<h2>1 &middot; The headline: who compensates staleness at which horizon</h2>
<div class="panel">{plot_a}{leg_a}</div>
<p><b>Reading it:</b> below the dashed line beats doing nothing (holding the stale
anchor weights); above it is actively harmful. <code>naive_last2</code> (consecutive-tick
momentum) owns h&nbsp;&le;&nbsp;5 but is harmful past h&asymp;15. The paper&rsquo;s own form
<code>rank1_traj</code> starts at 3.9 (clipped &uarr;) because it pays a constant absolute
residual (&sect;2). The anchor-pinned variant we added, <code>rank1_anchored</code>, is the
<b>only arm that never goes harmful</b> out to h=40 — but its skill is ~1&ndash;2%.</p>
{table_html}
<p style="font-size:12.5px;color:{C['mut']};margin-top:6px">Global pooled ratio,
median over anchors 79/119. Cell tint: <span style="background:#e7f3ea;padding:1px 6px">
green &lt; 0.98 = beats hold-stale</span> · <span style="background:#f7f7f2;
padding:1px 6px">grey ≈ 1 = neutral</span> · <span style="background:#fdeeea;
padding:1px 6px">red &gt; 1.05 = harmful</span>. Lower is better in every cell.</p>

<h2>2 &middot; Why the paper form fails this metric (and what it&rsquo;s actually for)</h2>
<div class="grid2">
<div class="panel">{plot_b1}</div>
<div class="panel">{plot_b2}</div>
</div>
{leg_b}
<p>RELEX optimizes <i>absolute end-checkpoint reconstruction</i>; our ratio measures
<i>improvement over the stale anchor for the next h ticks</i>. The accumulated delta
carries a persistent off-v&#8321; component (B2): only 4.5% with a recent 8-tick window,
but 28&ndash;33% with the paper-faithful full-prefix window — <b>the rank-1 direction
rotates over training</b>. The paper form re-pays that residual at every horizon
(so bigger windows hurt, B1 blue), while for RELEX&rsquo;s own objective it is a
negligible fraction of ||&theta;&#8320;&nbsp;&minus;&nbsp;&theta;<sub>T</sub>||.</p>
<div class="panel">{plot_e}{leg_e}</div>
<p>The residual geometry pins the mechanism: even anchored, the leftover error is
&asymp;97% <i>radial</i> — still pointing along the true move — at every h. The fitted
line&rsquo;s step is nearly orthogonal to where the weights actually go over the next
&le;40 ticks: per-tick displacement is dominated by a high-frequency component the
window-averaged line cannot see (but consecutive-tick momentum partially can).</p>

<h2>3 &middot; The paper&rsquo;s structural claims DO replicate here</h2>
<div class="grid2">
<div class="panel">{plot_c1}</div>
<div class="panel">{plot_c2}</div>
</div>
<p>Per-matrix rank-1 coefficient linearity R&sup2; median <b>0.979</b> (49% of matrices
clear the paper&rsquo;s 0.98 bar; early anchors: 0.968/37%) and rank-1 energy share
<b>EVR&#8321; &asymp; 99.3%</b> — our accumulated GRPO deltas are at least as rank-1 as the
paper&rsquo;s. The structure is real; only its <i>short-horizon predictive power</i> is not.</p>

<h2>4 &middot; Not a late-training artifact</h2>
<div class="grid2">
<div class="panel">{plot_d}{leg_d}</div>
<div class="panel"><p style="margin-top:26px">The saturation hypothesis — &ldquo;the
line stops predicting because training has converged&rdquo; — fails: anchors at ticks
39 and 59 (drift still strong) show the <i>same</i> neutral profile as 79/119, and
the prefix window is <i>worse</i> early (1.25 at h=1) because it averages over the
fast-rotating warm-up phase. On this trace the rank-1 line is a
<b>checkpoint-scale object</b>, not a per-tick-increment predictor, at every
training stage we probed.</p></div>
</div>

<h2>5 &middot; What this means for the comm-eff design</h2>
<div class="note"><b>Use the line for position, not velocity.</b> A rank-1
&ldquo;trajectory clock&rdquo; state (v&#8321; + two scalars per tensor) is cheap and
well-defined, and RELEX shows it reconstructs <i>distant checkpoints</i> well. But
short-horizon (&le;40 optimizer-tick) staleness repair — the regime of our
pipeline-parallel anchor design — should keep <b>anchor + recent-delta (momentum)</b>
signals: <code>naive_last2</code> 0.43&ndash;0.46 at h=1, <code>rank2_anchored</code> 0.69 at
h=1 (component 2 = real local dynamics, harmful by h=20), with
<code>rank1_anchored</code> as the never-harmful fallback (0.98&ndash;1.06 across all
h&nbsp;&le;&nbsp;40, both regimes).</div>

<h2>Provenance &amp; repro</h2>
<pre>python3 scripts/rank1_scorecard.py --self-test
python3 scripts/rank1_scorecard.py \\
  --trace-root /workspace/trace/EXP-57 \\
  --manifest runs/EXP-57/regimeA/weights/full_manifest.jsonl \\
  --out runs/RANK1-ANALYSIS/scorecard --scope panel        # late anchors (79,119)
# + --anchors 39,59 --out .../scorecard-early              # early anchors
python3 scripts/rank1_report.py                            # this page</pre>
<p style="font-size:13px;color:{C['mut']}">Implementation:
<code>scripts/weight_proj/rank1_traj.py</code> + <code>scripts/rank1_scorecard.py</code>
(new lane; <code>moat_scorecard.py</code> and the online comm-eff path untouched).
Canonical findings: <code>runs/RANK1-ANALYSIS/verdict.md</code> &middot; independent
verification: <code>runs/RANK1-ANALYSIS/verify_verdict.md</code> &middot; papers: Paper&nbsp;A
arXiv:2601.04537 (linear dynamics / weight extrapolation), Paper&nbsp;B
arXiv:2605.21468 (RELEX). Trace: EXP-57 regimeA fp32, 160 ticks (2 optimizer ticks
per GRPO step), 61-matrix panel = layers {{0,7,13,20,27}} + final norm.</p>

</div></body></html>"""
    os.makedirs(BASE, exist_ok=True)
    with open(OUT, "w") as f:
        f.write(doc)
    print(f"[rank1-report] wrote {OUT} ({len(doc):,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
