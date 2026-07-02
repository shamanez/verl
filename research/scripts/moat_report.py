#!/usr/bin/env python3
"""moat_report.py — #47 self-contained HTML report for the ANCHOR linear/damped lane.

Reads BOTH emitted scorecard dirs (regime S per-step / regime T per-tick) and renders
one offline HTML report:

  python scripts/moat_report.py \
      --perstep runs/MOAT-47-ANALYSIS/scorecard-perstep/ \
      --pertick runs/MOAT-47-ANALYSIS/scorecard-pertick/ \
      --out     runs/MOAT-47-ANALYSIS/report.html

The page is STRICTLY self-contained: inline <svg> only, ZERO <script>/<img>/external
refs (re-openable offline). It LEADS with the per-scalar linearity R² readout vs the
Wang et al. 2026 anchors, THEN the ratio/projection findings (PRIMARY per-step, then
the per-tick extended-Δ sweep), the paper-protocol equivalence panel, and the lane
verdict. Copies the inline-SVG pattern from weight_proj/report.py (does NOT edit it).

Direction-agnostic: a decisive NEGATIVE result (damped never beats hold-stale, wider Δ
never helps, low R²) is a VALID finding and is reported as such, not as a failure.
"""
from __future__ import annotations

import argparse
import html
import json
import math
import os

# Wang et al. 2026 (arXiv:2601.04537) per-scalar linearity R² anchors (median / Pr>0.7)
WANG_RL = (0.845, 0.794)     # nearest RL analog: R1-Distill-Qwen-1.5B GRPO / DeepScaleR
WANG_SFT = (0.426, 0.259)    # SFT Qwen2.5-1.5B + GSM8K
ESC = html.escape

PALETTE = {"hold_stale": "#7f7f7f", "naive_linear": "#1f77b4",
           "damped_linear": "#d62728", "paper_linear": "#2ca02c"}


def _load(d: str) -> dict:
    if not d:
        return {}
    rows = [json.loads(l) for l in open(os.path.join(d, "scorecard.jsonl")) if l.strip()]
    meta = json.load(open(os.path.join(d, "meta.json")))
    vis = json.load(open(os.path.join(d, "visuals.json")))
    idx = {(r["method"], r["delta_ticks"], r["h_ticks"], r["group_kind"],
            r["group_key"]): r for r in rows}
    return {"rows": rows, "meta": meta, "vis": vis, "idx": idx, "dir": d}


def _g(reg, method, d, h):
    return reg["idx"].get((method, d, h, "global", "all"), {}) if reg else {}


def _fmt(v, nd=3):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


# =============================================================================
# Inline-SVG primitives (self-contained; copied idiom from weight_proj/report.py)
# =============================================================================
def svg_line(series: dict, title: str, xlabel="horizon h", w=520, h=300, y_max=1.6,
             yref=1.0, xvals=None) -> str:
    """series: name -> [(x, y)]. Dashed y=yref reference line."""
    pad = 46
    pts_all = [p for s in series.values() for p in s if p[1] is not None and p[1] == p[1]]
    if not pts_all:
        return f"<p><em>no data for {ESC(title)}</em></p>"
    xs = xvals or sorted({x for x, _ in pts_all})
    x_min, x_max = min(xs), max(xs)
    ys = [y for _, y in pts_all]
    ymax = max(y_max, min(max(ys) * 1.1, 5.0)) if ys else y_max

    def sx(x):
        return pad + (0 if x_max == x_min else (x - x_min) / (x_max - x_min)) * (w - 2 * pad)

    def sy(y):
        y = max(0.0, min(ymax, y))
        return (h - pad) - (y / ymax) * (h - 2 * pad)

    P = [f'<svg width="{w}" height="{h}" style="background:#fff;border:1px solid #ddd">']
    P.append(f'<line x1="{pad}" y1="{sy(yref):.1f}" x2="{w-pad}" y2="{sy(yref):.1f}" '
             f'stroke="#bbb" stroke-dasharray="4 3"/>')
    P.append(f'<text x="{w-pad}" y="{sy(yref)-4:.1f}" font-size="10" text-anchor="end" '
             f'fill="#888">ratio={yref:g} (no skill)</text>')
    P.append(f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{h-pad}" stroke="#333"/>')
    P.append(f'<line x1="{pad}" y1="{h-pad}" x2="{w-pad}" y2="{h-pad}" stroke="#333"/>')
    P.append(f'<text x="{pad-6}" y="{sy(0):.1f}" font-size="10" text-anchor="end">0</text>')
    P.append(f'<text x="{pad-6}" y="{sy(ymax)+8:.1f}" font-size="10" text-anchor="end">{ymax:.2g}</text>')
    P.append(f'<text x="{w/2}" y="{h-8}" font-size="11" text-anchor="middle">{ESC(xlabel)}</text>')
    P.append(f'<text x="{w/2}" y="15" font-size="13" text-anchor="middle" font-weight="bold">{ESC(title)}</text>')
    for i, (name, pts) in enumerate(series.items()):
        col = PALETTE.get(name, ["#9467bd", "#ff7f0e", "#17becf", "#8c564b"][i % 4])
        pts = sorted((x, y) for x, y in pts if y is not None and y == y)
        if not pts:
            continue
        d = " ".join(f"{'M' if k==0 else 'L'}{sx(x):.1f},{sy(y):.1f}"
                     for k, (x, y) in enumerate(pts))
        P.append(f'<path d="{d}" fill="none" stroke="{col}" stroke-width="2"/>')
        for x, y in pts:
            P.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="2.5" fill="{col}"/>')
        ly = 30 + i * 14
        P.append(f'<rect x="{w-pad-150}" y="{ly-8}" width="10" height="10" fill="{col}"/>')
        P.append(f'<text x="{w-pad-136}" y="{ly}" font-size="10">{ESC(name)}</text>')
    P.append("</svg>")
    return "".join(P)


def svg_bars(counts: list, title: str, x0=0.0, x1=1.0, w=520, h=240, vmark=None) -> str:
    """Histogram bars over [x0,x1]; optional vertical marker(s) [(x,label,color)]."""
    if not counts or sum(counts) == 0:
        return f"<p><em>no data for {ESC(title)}</em></p>"
    pad = 44
    n = len(counts)
    cmax = max(counts)
    bw = (w - 2 * pad) / n
    P = [f'<svg width="{w}" height="{h}" style="background:#fff;border:1px solid #ddd">']
    P.append(f'<text x="{w/2}" y="15" font-size="13" text-anchor="middle" font-weight="bold">{ESC(title)}</text>')
    P.append(f'<line x1="{pad}" y1="{h-pad}" x2="{w-pad}" y2="{h-pad}" stroke="#333"/>')
    for i, c in enumerate(counts):
        bh = 0 if cmax == 0 else (c / cmax) * (h - 2 * pad)
        x = pad + i * bw
        P.append(f'<rect x="{x:.1f}" y="{h-pad-bh:.1f}" width="{max(bw-0.5,0.5):.1f}" '
                 f'height="{bh:.1f}" fill="#1f77b4"/>')
    for frac, lab in ((0.0, f"{x0:g}"), (0.5, f"{(x0+x1)/2:g}"), (1.0, f"{x1:g}")):
        xx = pad + frac * (w - 2 * pad)
        P.append(f'<text x="{xx:.1f}" y="{h-pad+14}" font-size="10" text-anchor="middle">{lab}</text>')
    for (mx, lab, col) in (vmark or []):
        if mx is None or not math.isfinite(mx):
            continue
        xx = pad + ((mx - x0) / (x1 - x0)) * (w - 2 * pad)
        P.append(f'<line x1="{xx:.1f}" y1="{pad-6}" x2="{xx:.1f}" y2="{h-pad}" '
                 f'stroke="{col}" stroke-width="1.5" stroke-dasharray="3 2"/>')
        P.append(f'<text x="{xx:.1f}" y="{pad-8}" font-size="9" text-anchor="middle" fill="{col}">{ESC(lab)}</text>')
    P.append("</svg>")
    return "".join(P)


def _heat_color(v, vmin, vmax, reverse=False):
    if v is None or not (isinstance(v, (int, float)) and math.isfinite(v)):
        return "#eee"
    t = 0.0 if vmax == vmin else (v - vmin) / (vmax - vmin)
    t = max(0.0, min(1.0, t))
    if reverse:
        t = 1.0 - t
    # blue(low)->white->red(high)
    if t < 0.5:
        u = t / 0.5
        r, g, b = int(31 + u * 224), int(119 + u * 136), int(180 + u * 75)
    else:
        u = (t - 0.5) / 0.5
        r, g, b = int(255 - u * 41), int(255 - u * 216), int(255 - u * 216)
    return f"rgb({r},{g},{b})"


def svg_heatmap(matrix, rowlabels, collabels, title, vmin, vmax, reverse=False,
                cell=15) -> str:
    if not matrix or not matrix[0]:
        return f"<p><em>no data for {ESC(title)}</em></p>"
    nr, nc = len(matrix), len(matrix[0])
    lpad, tpad = 40, 34
    w = lpad + nc * cell + 8
    hgt = tpad + nr * cell + 30
    P = [f'<svg width="{w}" height="{hgt}" style="background:#fff;border:1px solid #ddd">']
    P.append(f'<text x="{w/2}" y="14" font-size="12" text-anchor="middle" font-weight="bold">{ESC(title)}</text>')
    for j, cl in enumerate(collabels):
        x = lpad + j * cell + cell / 2
        P.append(f'<text x="{x:.1f}" y="{tpad-3}" font-size="7" text-anchor="end" '
                 f'transform="rotate(-60 {x:.1f} {tpad-3})">{ESC(str(cl))}</text>')
    for i, row in enumerate(matrix):
        y = tpad + i * cell
        if i % 4 == 0:
            P.append(f'<text x="{lpad-3}" y="{y+cell-3:.1f}" font-size="8" text-anchor="end">{ESC(str(rowlabels[i]))}</text>')
        for j, v in enumerate(row):
            x = lpad + j * cell
            P.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" '
                     f'fill="{_heat_color(v, vmin, vmax, reverse)}" stroke="#fff" stroke-width="0.3"/>')
    P.append(f'<text x="{lpad}" y="{hgt-8}" font-size="9" fill="#888">'
             f'{ESC(title.split("—")[0])}: {vmin:g}…{vmax:g}</text>')
    P.append("</svg>")
    return "".join(P)


def svg_scatter(points, xlab, ylab, title, w=460, h=320, spearman=None) -> str:
    pts = [(p.get("r2_median"), p.get("ratio_median"), p.get("group_key"))
           for p in points]
    pts = [(x, y, k) for x, y, k in pts
           if x is not None and y is not None and math.isfinite(x) and math.isfinite(y)]
    if not pts:
        return f"<p><em>no data for {ESC(title)}</em></p>"
    pad = 50
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    x0, x1 = min(xs + [0.0]), max(xs + [1.0]); y0, y1 = min(ys + [0.0]), max(ys + [1.2])

    def sx(x):
        return pad + (0 if x1 == x0 else (x - x0) / (x1 - x0)) * (w - 2 * pad)

    def sy(y):
        return (h - pad) - (0 if y1 == y0 else (y - y0) / (y1 - y0)) * (h - 2 * pad)

    P = [f'<svg width="{w}" height="{h}" style="background:#fff;border:1px solid #ddd">']
    P.append(f'<text x="{w/2}" y="15" font-size="13" text-anchor="middle" font-weight="bold">{ESC(title)}</text>')
    P.append(f'<line x1="{pad}" y1="{sy(1.0):.1f}" x2="{w-pad}" y2="{sy(1.0):.1f}" stroke="#bbb" stroke-dasharray="4 3"/>')
    P.append(f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{h-pad}" stroke="#333"/>')
    P.append(f'<line x1="{pad}" y1="{h-pad}" x2="{w-pad}" y2="{h-pad}" stroke="#333"/>')
    for x, y, k in pts:
        P.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="3" fill="#d62728" opacity="0.7"/>')
    P.append(f'<text x="{w/2}" y="{h-6}" font-size="11" text-anchor="middle">{ESC(xlab)}</text>')
    P.append(f'<text x="14" y="{h/2}" font-size="11" text-anchor="middle" transform="rotate(-90 14 {h/2})">{ESC(ylab)}</text>')
    if spearman is not None and math.isfinite(spearman):
        P.append(f'<text x="{w-pad}" y="{pad}" font-size="11" text-anchor="end" fill="#333">Spearman ρ = {spearman:.3f}</text>')
    P.append("</svg>")
    return "".join(P)


# =============================================================================
# Verdict computation (direction-agnostic; reads the emitted rows)
# =============================================================================
def _h_safe(reg, method):
    """Max h at which the GLOBAL median ratio for `method` is < 1.0 (0 if none)."""
    if not reg:
        return 0
    d = reg["meta"]["operating_point"][0]
    hs = reg["meta"]["h_ticks"]
    best = 0
    for h in hs:
        r = _g(reg, method, d, h).get("weight_proj_ratio_median")
        if r is not None and math.isfinite(r) and r < 1.0:
            best = max(best, h)
    return best


def _best_delta(reg, method):
    """(best_delta, ratio) minimizing GLOBAL median ratio at the operating h."""
    if not reg:
        return None, None
    op_h = reg["meta"]["operating_point"][1]
    cand = []
    for d in reg["meta"]["delta_ticks"]:
        r = _g(reg, method, d, op_h).get("weight_proj_ratio_median")
        if r is not None and math.isfinite(r):
            cand.append((r, d))
    if not cand:
        return None, None
    r, d = min(cand)
    return d, r


def _breakers(reg, method, op_d, op_h):
    """Groups (block_type/super_block/layer/special) with median ratio >= 1.0 at op."""
    out = []
    for r in (reg["rows"] if reg else []):
        if (r["method"] == method and r["delta_ticks"] == op_d and r["h_ticks"] == op_h
                and r["group_kind"] in ("block_type", "super_block", "layer", "special")):
            v = r["weight_proj_ratio_median"]
            if v is not None and math.isfinite(v) and v >= 1.0:
                out.append((f"{r['group_kind']}:{r['group_key']}", v))
    return sorted(out, key=lambda x: -x[1])


def build_verdict(S, T) -> list[str]:
    L = []
    # (5/lead) linearity R²
    for reg, tag in ((S, "regime S / per-step (PAPER-COMPARABLE)"), (T, "regime T / per-tick")):
        if not reg:
            continue
        lr = reg["meta"].get("linearity_r2", {})
        L.append(f"<b>Linearity R² [{tag}]</b>: global median = {_fmt(lr.get('r2_median'))}, "
                 f"Pr(R²&gt;0.7) = {_fmt(lr.get('r2_frac_gt_0.7'))}, "
                 f"excluded-constant scalars = {_fmt(lr.get('n_excluded_const'))}. "
                 f"vs Wang anchors — RL analog {WANG_RL[0]}/{WANG_RL[1]}, SFT-GSM8K {WANG_SFT[0]}/{WANG_SFT[1]}.")
    # (1) operating point (regime S primary)
    if S:
        od, oh = S["meta"]["operating_point"]
        dr = _g(S, "damped_linear", od, oh).get("weight_proj_ratio_median")
        nr = _g(S, "naive_linear", od, oh).get("weight_proj_ratio_median")
        L.append(f"<b>Operating point (per-step, Δ={od},h={oh} global steps)</b>: "
                 f"OOS-damped median ratio = {_fmt(dr)} "
                 f"({'BEATS' if (dr is not None and nr is not None and dr < nr) else 'does NOT beat'} "
                 f"naive_linear {_fmt(nr)}; "
                 f"{'BEATS' if (dr is not None and math.isfinite(dr) and dr < 1.0) else 'does NOT beat'} "
                 f"hold-stale 1.0 — projection {'helps' if (dr is not None and math.isfinite(dr) and dr<1.0) else 'does not help'}).")
    if T:
        od, oh = T["meta"]["operating_point"]
        dr = _g(T, "damped_linear", od, oh).get("weight_proj_ratio_median")
        nr = _g(T, "naive_linear", od, oh).get("weight_proj_ratio_median")
        L.append(f"<b>Per-tick cell (Δ={od},h={oh} ticks)</b>: OOS-damped ratio = {_fmt(dr)}, "
                 f"naive_linear = {_fmt(nr)} (vs #45 naive reference 1.158).")
    # (2) does increasing delta help (regime T extended sweep)
    if T:
        bd, br = _best_delta(T, "damped_linear")
        deltas = T["meta"]["delta_ticks"]
        L.append(f"<b>Does wider Δ help? (per-tick extended sweep Δ∈{deltas})</b>: "
                 f"best_delta = {_fmt(bd)} (min OOS-damped ratio {_fmt(br)} at operating h).")
    # (3) h_safe both regimes
    if S:
        L.append(f"<b>h_safe (per-step, global steps)</b>: {_h_safe(S, 'damped_linear')} "
                 f"(max h with global OOS-damped median &lt; 1.0; 0 = never).")
    if T:
        L.append(f"<b>h_safe (per-tick, ticks)</b>: {_h_safe(T, 'damped_linear')}.")
    # (4) breakers (regime S)
    if S:
        od, oh = S["meta"]["operating_point"]
        bk = _breakers(S, "damped_linear", od, oh)
        L.append(f"<b>Breakers (per-step, groups with ratio≥1 at op)</b>: "
                 + (", ".join(f"{g} ({v:.3f})" for g, v in bk[:8]) if bk else "none"))
    # (6) coupling + paper
    if S:
        cp = S["vis"].get("k_r2_ratio_coupling", {})
        L.append(f"<b>R²-vs-ratio coupling (per-step)</b>: Spearman ρ = {_fmt(cp.get('spearman'))} "
                 f"over {len(cp.get('points', []))} groups (do high-R² groups project better?).")
        panel = S["vis"].get("m_paper_equivalence", {})
        oh = panel.get("operating_h")
        pr = _g(S, "paper_linear", S["meta"].get("paper_sentinel_delta", 0), oh).get("weight_proj_ratio_median") if oh else None
        nr = _g(S, "naive_linear", S["meta"]["operating_point"][0], oh).get("weight_proj_ratio_median") if oh else None
        dr = _g(S, "damped_linear", S["meta"]["operating_point"][0], oh).get("weight_proj_ratio_median") if oh else None
        L.append(f"<b>Paper-protocol (paper_linear, per-step, h={oh})</b>: wide proportional-window "
                 f"ratio = {_fmt(pr)} vs fixed-Δ naive {_fmt(nr)} vs OOS-damped {_fmt(dr)} "
                 f"(β = 1 + h/Δ_resolved; anchor selection is the ONLY difference from naive_linear).")
    return L


# =============================================================================
# Render
# =============================================================================
def _accuracy_series(reg, methods):
    a = reg["vis"].get("a_accuracy_vs_horizon", {})
    out = {}
    for m in methods:
        s = a.get(m)
        if s:
            out[m] = list(zip(s["h"], s["ratio_median"]))
    return out


def _delta_series(reg, methods):
    b = reg["vis"].get("b_delta_sensitivity", {})
    out = {}
    for m in methods:
        s = b.get(m)
        if s:
            out[m] = list(zip(s["delta"], s["ratio_median"]))
    return out


def _table(headers, rows_html):
    return ("<table><tr>" + "".join(f'<th class="l">{ESC(h)}</th>' for h in headers)
            + "</tr>" + "".join(rows_html) + "</table>")


def render(S, T, out_path):
    P = ['<meta charset="utf-8"><title>EXP-47 — ANCHOR linear/damped projection lane</title>',
         '<style>body{font-family:-apple-system,Helvetica,Arial,sans-serif;margin:24px;color:#222;max-width:1180px}',
         'h1{font-size:23px}h2{font-size:17px;margin-top:30px;border-bottom:2px solid #333;padding-bottom:4px}',
         'h3{font-size:14px;margin-top:18px;color:#444}',
         'table{border-collapse:collapse;font-size:12px;margin:8px 0}td,th{border:1px solid #ccc;padding:4px 8px;text-align:right}',
         'th{background:#f5f5f5}td.l,th.l{text-align:left}.mono{font-family:ui-monospace,Menlo,monospace}',
         '.small{color:#777;font-size:11px}.ok{color:#127a12;font-weight:bold}.bad{color:#c00;font-weight:bold}',
         '.lead{background:#f0f6ff;border:1px solid #cfe0ff;border-radius:6px;padding:12px 18px;margin:14px 0}',
         '.side{display:flex;flex-wrap:wrap;gap:14px}.col{flex:1;min-width:420px}',
         '.verdict{background:#fffbe6;border:1px solid #f0e0a0;border-radius:6px;padding:10px 18px}</style>']
    ref = S or T
    op_s = S["meta"]["operating_point"] if S else None
    op_t = T["meta"]["operating_point"] if T else None
    P.append("<h1>EXP-47 — MOAT ANCHOR linear / damped-linear projection lane</h1>")
    P.append(f'<p class="small">Weight-geometry replay over the EXP-57 fp32 trace (338 matrices). '
             f'Two cadence regimes: <b>S = per-step</b> (global steps, PRIMARY, paper-comparable) '
             f'and <b>T = per-tick</b> (optimizer ticks, extended-Δ sweep). '
             f'Metric contract: {ESC(str((ref or {}).get("meta", {}).get("metric_contract", "")))}. '
             f'Ratio &lt; 1 ⇒ projection beats holding the stale weights.</p>')

    # ---- LEAD: per-scalar linearity R² -----------------------------------------
    P.append("<h2>1 — Per-scalar linearity R² (the MUST metric; Wang et al. 2026)</h2>")
    P.append('<div class="lead">')
    P.append(f'<p>Per-individual-scalar OLS R² of the weight value vs the training-step index '
             f'(constant scalars excluded &amp; counted). Anchors: nearest RL analog '
             f'R1-Distill-Qwen-1.5B GRPO/DeepScaleR <b>{WANG_RL[0]}/{WANG_RL[1]}</b> (median/Pr&gt;0.7); '
             f'SFT Qwen2.5-1.5B+GSM8K <b>{WANG_SFT[0]}/{WANG_SFT[1]}</b>. '
             f'<b>Regime S (per-step) is the paper-comparable cadence.</b> '
             f'A LOW R² is a valid, decisive finding.</p>')
    rowsh = []
    for reg, tag in ((S, "S / per-step"), (T, "T / per-tick")):
        if not reg:
            continue
        lr = reg["meta"].get("linearity_r2", {})
        rowsh.append(f'<tr><td class="l">{ESC(tag)}</td><td class="mono">{_fmt(lr.get("r2_median"))}</td>'
                     f'<td class="mono">{_fmt(lr.get("r2_frac_gt_0.7"))}</td>'
                     f'<td class="mono">{_fmt(lr.get("n_excluded_const"),0)}</td></tr>')
    P.append(_table(["regime", "R² median", "Pr(R²>0.7)", "n_excluded_const"], rowsh))
    # R² histogram + depth×block heatmap + coupling (prefer regime S)
    P.append('<div class="side">')
    if S and S["vis"].get("i_r2_histogram", {}).get("global"):
        g = S["vis"]["i_r2_histogram"]["global"]
        lr = S["meta"].get("linearity_r2", {})
        P.append('<div class="col">' + svg_bars(
            g["counts"], "Per-scalar R² histogram (per-step, global)", 0.0, 1.0,
            vmark=[(lr.get("r2_median"), "median", "#d62728"),
                   (WANG_RL[0], "RL", "#2ca02c"), (WANG_SFT[0], "SFT", "#ff7f0e")]) + '</div>')
    if S and S["vis"].get("k_r2_ratio_coupling"):
        cp = S["vis"]["k_r2_ratio_coupling"]
        P.append('<div class="col">' + svg_scatter(
            cp.get("points", []), "group median R²", "OOS-damped ratio @op",
            "R²-vs-ratio coupling (per-step)", spearman=cp.get("spearman")) + '</div>')
    P.append('</div>')
    if S and S["vis"].get("j_r2_depth_block_heatmap"):
        hm = S["vis"]["j_r2_depth_block_heatmap"]
        P.append(svg_heatmap(hm["r2_median"], hm["layers"], hm["block_types"],
                             "Depth × block per-scalar R² (per-step) — distinct from traj_r2",
                             0.0, 1.0))
    P.append('</div>')

    # ---- ratio/projection findings, both regimes -------------------------------
    P.append("<h2>2 — Projection accuracy vs horizon (both regimes)</h2>")
    P.append('<div class="side">')
    if S:
        methods = [m for m in S["meta"]["methods"] if m != "paper_linear"]
        P.append('<div class="col"><h3>Regime S — per-step (global steps), Δ=%d</h3>%s</div>'
                 % (op_s[0], svg_line(_accuracy_series(S, methods),
                                      "median ratio vs h (global steps)", "horizon h (global steps)")))
    if T:
        methods = [m for m in T["meta"]["methods"] if m != "paper_linear"]
        P.append('<div class="col"><h3>Regime T — per-tick, Δ=%d</h3>%s</div>'
                 % (op_t[0], svg_line(_accuracy_series(T, methods),
                                      "median ratio vs h (ticks)", "horizon h (ticks)")))
    P.append('</div>')

    P.append("<h2>3 — Δ-sensitivity (extended to Δ=40, per-tick) &amp; λ-selection</h2>")
    P.append('<div class="side">')
    if T:
        methods = [m for m in T["meta"]["methods"] if m != "paper_linear"]
        P.append('<div class="col"><h3>Δ-sensitivity at operating h=%d (per-tick)</h3>%s'
                 '<p class="small">Δ∈%s — does a wider anchor help past Δ=20?</p></div>'
                 % (op_t[1], svg_line(_delta_series(T, methods),
                                      "median ratio vs Δ", "Δ (ticks)", xvals=T["meta"]["delta_ticks"]),
                    T["meta"]["delta_ticks"]))
    if S:
        lam = S["vis"].get("h_lambda_selection", {}).get("cells", {})
        key = f"{op_s[0]},{op_s[1]}"
        cell = lam.get(key) or (next(iter(lam.values())) if lam else None)
        if cell:
            ser = {"in-sample ratio": list(zip(cell["lambda"], cell["ratio_median"]))}
            P.append('<div class="col"><h3>λ-selection at (Δ=%d,h=%d), per-step</h3>%s'
                     '<p class="small">in-sample median ratio vs λ (OOS picks per-window on '
                     'strictly-earlier data). λ=0 ⇒ hold-stale, λ=1 ⇒ naive.</p></div>'
                     % (op_s[0], op_s[1], svg_line(ser, "ratio vs λ", "λ",
                                                   xvals=cell["lambda"])))
    P.append('</div>')

    # ratio heatmap by layer×block (regime S, damped)
    if S and S["vis"].get("e_ratio_heatmap", {}).get("damped_linear"):
        e = S["vis"]["e_ratio_heatmap"]["damped_linear"]
        P.append("<h3>OOS-damped ratio by layer × block (per-step, at operating point)</h3>")
        P.append(svg_heatmap(e["ratio_median"], e["layers"], e["block_types"],
                             "damped ratio (blue<1 good, red>1 harmful)", 0.5, 1.5))

    # special groups table (regime S)
    if S and S["vis"].get("g_special_groups", {}).get("damped_linear"):
        P.append("<h3>Special groups — OOS-damped ratio at operating point (per-step)</h3>")
        rowsh = []
        for grp in S["vis"]["g_special_groups"]["damped_linear"]:
            rowsh.append(f'<tr><td class="l">{ESC(str(grp.get("group","")))}</td>'
                         f'<td class="mono">{_fmt(grp.get("weight_proj_ratio_median"))}</td>'
                         f'<td class="mono">{_fmt(grp.get("r2_median"))}</td>'
                         f'<td class="mono">{_fmt(grp.get("lam_star"))}</td></tr>')
        P.append(_table(["special group", "damped ratio", "R² median", "λ* (med)"], rowsh))

    # ---- paper-protocol equivalence panel (regime S) ---------------------------
    P.append("<h2>4 — Paper-protocol equivalence panel (Wang et al. §6.2, regime S)</h2>")
    if S and "paper_linear" in S["meta"].get("methods", []):
        pd = S["meta"].get("paper_sentinel_delta", 0)
        P.append('<div class="lead"><p><b>The algebra.</b> The paper\'s weight-space extrapolation '
                 'W<sub>t\'</sub> = W<sub>t0</sub> + β·(W<sub>t1</sub> − W<sub>t0</sub>) with '
                 'β = (t\'−t0)/(t1−t0) IS the first-order secant — with t0 = t−Δ, t1 = t, t\' = t+h it '
                 'equals <b>naive_linear with β = 1 + h/Δ</b>. Nothing is fitted; β is an '
                 'extrapolation ratio, not a regression coefficient. <b>The ONLY difference is the '
                 'anchor protocol</b>: paper_linear uses a WIDE proportional window (t0 = ⌊0.25·t⌋, so '
                 'Δ_resolved ≈ 0.75·t grows with t); naive/damped use FIXED SHORT lags. '
                 'Baseline differs too: our ratio denominator is hold-stale (comm-substitution — can a '
                 'worker\'s stale copy be beaten by local prediction), NOT more RL training '
                 '(the paper\'s compute-substitution comparator).</p></div>')
        # matched-(t,h) comparison + beta distribution
        oh = S["vis"].get("m_paper_equivalence", {}).get("operating_h")
        hs = S["meta"]["h_ticks"]
        rowsh = []
        for h in hs:
            pr = _g(S, "paper_linear", pd, h).get("weight_proj_ratio_median")
            prow = S["idx"].get(("paper_linear", pd, h, "global", "all"), {})
            nr = _g(S, "naive_linear", op_s[0], h).get("weight_proj_ratio_median")
            dr = _g(S, "damped_linear", op_s[0], h).get("weight_proj_ratio_median")
            rowsh.append(f'<tr><td class="mono">{h}</td><td class="mono">{_fmt(pr)}</td>'
                         f'<td class="mono">{_fmt(nr)}</td><td class="mono">{_fmt(dr)}</td>'
                         f'<td class="mono">{_fmt(prow.get("delta_resolved"),1)}</td>'
                         f'<td class="mono">{_fmt(prow.get("beta"),2)} '
                         f'[{_fmt(prow.get("beta_min"),2)},{_fmt(prow.get("beta_max"),2)}]</td>'
                         f'<td class="mono">{_fmt(prow.get("n_windows"),0)}</td></tr>')
        P.append(_table(["h (steps)", "paper ratio", "naive(fixed Δ)", "OOS-damped",
                         "Δ_resolved", "β [min,max]", "n_win"], rowsh))
        P.append('<p class="small">paper_linear is regime-S ONLY (the protocol is checkpoint/per-step-like; '
                 'per-tick is the catastrophic-cancellation regime). Map β onto the paper\'s Fig. 5 '
                 'inverted-U: moderate β helps, excessive β amplifies slope-estimation error.</p>')
    else:
        P.append('<p class="small">paper_linear not present in the per-step table.</p>')

    # ---- verdict ---------------------------------------------------------------
    P.append("<h2>5 — Lane verdict (direction-agnostic)</h2>")
    P.append('<div class="verdict"><ul>')
    for line in build_verdict(S, T):
        P.append(f"<li>{line}</li>")
    P.append('</ul><p class="small">A decisive negative result (damped never beats hold-stale; '
             'wider Δ never helps; ratio &gt; 1 at long h; low R²) is a VALID finding — the lane '
             'PASSES by producing a correct, schema-verified, machine-readable answer either way.</p></div>')

    # ---- provenance ------------------------------------------------------------
    P.append("<h2>Provenance</h2>")
    for reg, tag in ((S, "per-step"), (T, "per-tick")):
        if not reg:
            continue
        m = reg["meta"]
        P.append(f'<p class="small mono">{ESC(tag)}: cadence={m.get("cadence")} unit={m.get("unit")} '
                 f'n_ticks={m.get("n_ticks")} band={m.get("band")} n_rows={m.get("n_rows")} '
                 f'lam_grid={len(m.get("lam_grid", []))}pts fingerprint={m.get("stats_cache_fingerprint")} '
                 f'gates={ {k: v.get("pass") for k, v in m.get("gates", {}).items()} }</p>')
    with open(out_path, "w") as f:
        f.write("".join(P))
    print(f"report written: {out_path} ({os.path.getsize(out_path)} bytes)", flush=True)


def main():
    ap = argparse.ArgumentParser(description="#47 self-contained HTML report (both regimes)")
    ap.add_argument("--perstep", default="", help="regime S scorecard dir")
    ap.add_argument("--pertick", default="", help="regime T scorecard dir")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    S = _load(args.perstep)
    T = _load(args.pertick)
    if not S and not T:
        raise SystemExit("need at least one of --perstep / --pertick")
    render(S, T, args.out)


if __name__ == "__main__":
    main()
