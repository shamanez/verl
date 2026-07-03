#!/usr/bin/env python3
"""moat_report.py — #47 self-contained HTML report for the ANCHOR linear/damped lane.

Reads BOTH emitted scorecard dirs (regime S per-step / regime T per-tick) and renders
one offline HTML report:

  python scripts/moat_report.py \
      --perstep runs/MOAT-47-ANALYSIS/scorecard-perstep/ \
      --pertick runs/MOAT-47-ANALYSIS/scorecard-pertick/ \
      --out     runs/MOAT-47-ANALYSIS/report.html

The page is STRICTLY self-contained: inline <svg> only, ZERO <script>/<img>/external
refs/fonts/url()/gradients (re-openable offline). It LEADS with the per-scalar
linearity R² readout vs the Wang et al. 2026 anchors, THEN the ratio/projection
findings (PRIMARY per-step, then the per-tick extended-Δ sweep) and the lane verdict.

Prose is embedded from a sibling `report_explainer.md` (a RENDER INPUT): the "how to
read", "what was verified", per-figure captions, glossary and one-paragraph findings
are converted from that markdown at render time. If the explainer is absent the report
still renders (those sections are skipped and a warning is printed) — graceful
degradation, single source of truth for the prose.

Direction-agnostic: a decisive NEGATIVE result (damped never beats hold-stale, wider Δ
never helps, low R²) is a VALID finding and is reported as such, not as a failure.
"""
from __future__ import annotations

import argparse
import html
import json
import math
import os
import re

# Wang et al. 2026 (arXiv:2601.04537) per-scalar linearity R² anchors (median / Pr>0.7)
WANG_RL = (0.845, 0.794)     # nearest RL analog: R1-Distill-Qwen-1.5B GRPO / DeepScaleR
WANG_SFT = (0.426, 0.259)    # SFT Qwen2.5-1.5B + GSM8K
ESC = html.escape

PALETTE = {"hold_stale": "#7f7f7f", "naive_linear": "#1f77b4",
           "damped_linear": "#d62728", "naive_second_order": "#2ca02c",
           "damped_second_order": "#ff7f0e", "adaptive_linear": "#9467bd",
           "adaptive_second_order": "#8c564b"}

CSS = (
    "body{font-family:-apple-system,Helvetica,Arial,sans-serif;margin:24px;color:#222;"
    "max-width:1180px;line-height:1.5}"
    "h1{font-size:23px}"
    "h2{font-size:17px;margin-top:34px;border-bottom:2px solid #333;padding-bottom:4px}"
    "h3{font-size:14px;margin-top:18px;color:#444}"
    "p{margin:8px 0}"
    "table{border-collapse:collapse;font-size:12px;margin:8px 0}"
    "td,th{border:1px solid #ccc;padding:4px 8px;text-align:right}"
    "th{background:#f5f5f5}td.l,th.l{text-align:left}"
    ".mono{font-family:ui-monospace,Menlo,monospace}"
    ".small{color:#777;font-size:11px}.ok{color:#127a12;font-weight:bold}.bad{color:#c00;font-weight:bold}"
    ".lead{background:#f0f6ff;border:1px solid #cfe0ff;border-radius:6px;padding:12px 18px;margin:14px 0}"
    ".fastbanner{background:#fff3e6;border:1px solid #f0c080;border-radius:6px;"
    "padding:8px 16px;margin:10px 0;font-size:13px}"
    ".fullbanner{background:#eefaee;border:1px solid #b0d8b0;border-radius:6px;"
    "padding:8px 16px;margin:10px 0;font-size:13px}"
    ".explain{background:#f7f9fc;border:1px solid #dbe4f0;border-radius:6px;padding:4px 20px;margin:14px 0}"
    ".side{display:flex;flex-wrap:wrap;gap:14px}.col{flex:1;min-width:420px}"
    ".verdict{background:#fffbe6;border:1px solid #f0e0a0;border-radius:6px;padding:10px 18px}"
    ".figtitle{font-size:13px;font-weight:bold;margin:16px 0 4px;color:#333}"
    ".cap{color:#555;font-size:11.5px;line-height:1.5;margin:5px 0 2px;max-width:900px}"
    ".cbcap{color:#555;font-size:10.5px;margin:2px 0 8px;max-width:640px}"
    ".figscroll{overflow-x:auto;max-width:100%}"
    ".tblwrap{overflow-x:auto;max-width:100%}"
    ".gloss{font-size:12px}.gloss td{vertical-align:top}"
    ".gloss td.t{white-space:nowrap;font-weight:600;color:#1a1a1a}"
    ".gloss td.d{text-align:left;max-width:860px}"
    "pre{background:#f6f8fa;border:1px solid #e0e0e0;border-radius:5px;padding:10px 12px;"
    "font-size:11.5px;line-height:1.4;overflow-x:auto}"
    "code{background:#eef0f3;border-radius:3px;padding:0 3px;"
    "font-family:ui-monospace,Menlo,monospace;font-size:12px}"
    "ol,ul{margin:6px 0;padding-left:24px}li{margin:4px 0}"
    ".foot{color:#888;font-size:11px;margin-top:30px;border-top:1px solid #ddd;padding-top:8px}"
)


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


def fidelity_banner(reg, tag) -> str:
    """FAST/FULL provenance banner (meta.fidelity/meta.sampling; pre-rows-v48 dirs
    were emitted by the exact path only, so a missing tag reads as FULL)."""
    if not reg:
        return ""
    fid = reg["meta"].get("fidelity", "full")
    if fid == "fast":
        s = reg["meta"].get("sampling") or {}
        return (f'<div class="fastbanner"><b>{ESC(tag)}: FAST — sampled estimate</b> '
                f'(frac={s.get("frac")}, seed={s.get("seed")}, '
                f'strips={s.get("strip_elems")}; small tensors exact; '
                f'{_fmt(s.get("n_elems_sampled_total"), 0)} of '
                f'{_fmt(s.get("n_elems_total"), 0)} scalars); '
                f'verdict-grade = <code>--fidelity full</code></div>')
    return f'<div class="fullbanner"><b>{ESC(tag)}: FULL — exact</b> (all scalars, all ticks)</div>'


# built-in fallback prose for the regression section + fast-mode caveats — used
# when the run's report_explainer.md lacks a matching "## Absolute prediction
# accuracy" heading; a future explainer overrides it.
FALLBACK_REGRESSION_PROSE = (
    "<p><b>Reading this section.</b> <code>pred_evr_pooled</code> is the pooled "
    "explained-variance ratio vs the stale baseline over all scored windows: "
    "1 − Σ‖e‖²/Σ‖b‖². <code>hold_stale</code> scores EXACTLY 0 (the doing-nothing "
    "baseline); 1 = perfect prediction; negative = worse than holding stale. "
    "<code>pred R² (scalar)</code> is the classical per-coordinate R² of predicted "
    "vs actual FUTURE weights over the scored windows, computed on the sampled "
    "panel at the operating cell; for <code>damped_linear</code> it deploys the "
    "GLOBAL group's per-window OOS λ path (a documented modeling choice that can "
    "slightly understate damped accuracy on groups whose optimal λ differs from "
    "global).</p>"
    "<p><b>Fast-mode caveats.</b> Under <code>--fidelity fast</code> all ratio/EVR "
    "numbers are sampled estimates (~0.1%/matrix, deterministic seed; tensors "
    "≤ 8192 elems exact); per-matrix tails (p10/p90, h*) can wobble. The linearity "
    "R² population also differs: fast applies the paper's range/unique trajectory "
    "filters (<code>r2_population=sampled_paper_filtered</code>), full excludes "
    "only constant scalars — do not compare r2_median across fidelities. Strip "
    "sampling (1024-contiguous runs) trades paper-faithful uniform scatter for "
    "page locality; <code>--sample-strip-elems 1</code> restores pure scatter.</p>")


# =============================================================================
# Explainer (report_explainer.md) → HTML. Single source of truth for the prose;
# a RENDER INPUT that degrades gracefully (absent ⇒ warn + skip those sections).
# =============================================================================
def load_explainer(path):
    """Parse report_explainer.md into {h2_title: [body_lines]}. Returns {} if absent."""
    if not path or not os.path.exists(path):
        print(f"WARNING: explainer not found ({path!r}); rendering without embedded prose.",
              flush=True)
        return {}
    secs, cur = {}, None
    for line in open(path):
        line = line.rstrip("\n")
        m = re.match(r"^## (.+)$", line)
        if m:
            cur = m.group(1).strip()
            secs[cur] = []
        elif cur is not None:
            secs[cur].append(line)
    return secs


def _md_inline(s):
    """Inline markdown → HTML on a raw string. Handles `code`, **bold**, *italic*,
    and backslash-escaped * / _. Everything is HTML-escaped first."""
    s = ESC(s)
    s = s.replace(r"\*", "\x00").replace(r"\_", "\x01")   # protect escaped punctuation
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"\*([^*]+?)\*", r"<em>\1</em>", s)
    return s.replace("\x00", "*").replace("\x01", "_")


def _md_block(lines):
    """Convert a list of markdown lines → HTML: headers, bold/italic/code, ordered &
    unordered lists (with indented continuations), fenced ``` code → <pre>, tables,
    and paragraphs."""
    out, para, ul, ol, tbl = [], [], [], [], []
    pre = None

    def flush_para():
        if para:
            out.append("<p>" + _md_inline(" ".join(para)) + "</p>")
            para.clear()

    def flush_ul():
        if ul:
            out.append("<ul>" + "".join(f"<li>{_md_inline(x)}</li>" for x in ul) + "</ul>")
            ul.clear()

    def flush_ol():
        if ol:
            out.append("<ol>" + "".join(f"<li>{_md_inline(x)}</li>" for x in ol) + "</ol>")
            ol.clear()

    def flush_tbl():
        if tbl:
            body = []
            for r, cells in enumerate(tbl):
                tag = "th" if r == 0 else "td"
                body.append("<tr>" + "".join(f'<{tag} class="l">{_md_inline(c)}</{tag}>'
                                              for c in cells) + "</tr>")
            out.append('<div class="tblwrap"><table>' + "".join(body) + "</table></div>")
            tbl.clear()

    def flush_all():
        flush_para(); flush_ul(); flush_ol(); flush_tbl()

    for raw in lines:
        line = raw.rstrip()
        if line.strip().startswith("```"):
            if pre is None:
                flush_all(); pre = []
            else:
                out.append("<pre>" + ESC("\n".join(pre)) + "</pre>"); pre = None
            continue
        if pre is not None:
            pre.append(raw); continue
        if not line.strip():
            flush_all(); continue
        m_h3 = re.match(r"^###\s+(.*)$", line)
        m_ul = re.match(r"^\s*-\s+(.*)$", line)
        m_ol = re.match(r"^\s*\d+\.\s+(.*)$", line)
        is_tbl = line.lstrip().startswith("|") and line.rstrip().endswith("|")
        if m_h3:
            flush_all(); out.append(f"<h3>{_md_inline(m_h3.group(1))}</h3>"); continue
        if is_tbl:
            flush_para(); flush_ul(); flush_ol()
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if all(set(c) <= set("-: ") for c in cells):   # separator row
                continue
            tbl.append(cells); continue
        if m_ul:
            flush_para(); flush_ol(); flush_tbl(); ul.append(m_ul.group(1)); continue
        if m_ol:
            flush_para(); flush_ul(); flush_tbl(); ol.append(m_ol.group(1)); continue
        if (ul or ol) and (raw.startswith("  ") or raw.startswith("\t")):
            (ul if ul else ol)[-1] += " " + line.strip(); continue
        flush_ul(); flush_ol(); flush_tbl(); para.append(line.strip())

    if pre is not None:
        out.append("<pre>" + ESC("\n".join(pre)) + "</pre>")
    flush_all()
    return "".join(out)


def parse_glossary(lines):
    """`- **term** — definition` bullets (with indented continuations) → [(term, def)]."""
    items, cur = [], None
    for raw in lines:
        line = raw.rstrip()
        m = re.match(r"^\s*-\s+\*\*(.+?)\*\*\s*—\s*(.*)$", line)
        if m:
            if cur:
                items.append(cur)
            cur = [m.group(1), m.group(2)]
        elif cur is not None and line.strip():
            cur[1] += " " + line.strip()
    if cur:
        items.append(cur)
    return [(_md_inline(t), _md_inline(d)) for t, d in items]


def parse_figure_caps(lines):
    """`- **Figure name** — text` bullets → {figure_name: rendered_caption_html}."""
    caps, key, buf = {}, None, []

    def commit():
        if key is not None:
            caps[key] = _md_inline(" ".join(buf).strip())

    for raw in lines:
        line = raw.rstrip()
        m = re.match(r"^\s*-\s+\*\*(.+?)\*\*(.*)$", line)
        if m:
            commit()
            key = m.group(1).strip()
            buf = ["**" + m.group(1) + "**" + m.group(2)]
        elif key is not None and line.strip():
            buf.append(line.strip())
    commit()
    return caps


# =============================================================================
# Inline-SVG primitives (self-contained; no <img>/url()/gradient/<script>)
# =============================================================================
def svg_line(series: dict, title: str, xlabel="horizon h", w=520, h=300, y_max=1.6,
             yref=1.0, xvals=None, y_min=0.0) -> str:
    """series: name -> [(x, y)]. Dashed y=yref reference line. Legend sits on a white
    card (drawn last) so it never disappears behind a curve."""
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
        y = max(y_min, min(ymax, y))
        return (h - pad) - ((y - y_min) / (ymax - y_min)) * (h - 2 * pad)

    P = [f'<svg width="{w}" height="{h}" style="background:#fff;border:1px solid #ddd">']
    P.append(f'<line x1="{pad}" y1="{sy(yref):.1f}" x2="{w-pad}" y2="{sy(yref):.1f}" '
             f'stroke="#bbb" stroke-dasharray="4 3"/>')
    P.append(f'<text x="{w-pad}" y="{sy(yref)-4:.1f}" font-size="10" text-anchor="end" '
             f'fill="#888">ratio={yref:g} (no skill)</text>')
    P.append(f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{h-pad}" stroke="#333"/>')
    P.append(f'<line x1="{pad}" y1="{h-pad}" x2="{w-pad}" y2="{h-pad}" stroke="#333"/>')
    P.append(f'<text x="{pad-6}" y="{sy(y_min):.1f}" font-size="10" text-anchor="end">{y_min:g}</text>')
    P.append(f'<text x="{pad-6}" y="{sy(ymax)+8:.1f}" font-size="10" text-anchor="end">{ymax:.2g}</text>')
    P.append(f'<text x="{w/2:.1f}" y="{h-8}" font-size="11" text-anchor="middle">{ESC(xlabel)}</text>')
    P.append(f'<text x="{w/2:.1f}" y="15" font-size="13" text-anchor="middle" font-weight="bold">{ESC(title)}</text>')
    legend = []
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
        legend.append((name, col))
    # legend card (top-right), drawn last so it stays readable over the curves
    if legend:
        lx = w - pad - 152
        ly0 = pad + 6
        box_h = len(legend) * 15 + 8
        P.append(f'<rect x="{lx-6:.1f}" y="{ly0-4:.1f}" width="152" height="{box_h}" '
                 f'rx="4" fill="#fff" opacity="0.9" stroke="#ddd"/>')
        for i, (name, col) in enumerate(legend):
            ly = ly0 + 6 + i * 15
            P.append(f'<rect x="{lx:.1f}" y="{ly-8:.1f}" width="10" height="10" fill="{col}"/>')
            P.append(f'<text x="{lx+15:.1f}" y="{ly:.1f}" font-size="10">{ESC(name)}</text>')
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
    P.append(f'<text x="{w/2:.1f}" y="15" font-size="13" text-anchor="middle" font-weight="bold">{ESC(title)}</text>')
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


def centered_vlim(matrix, cap=0.5):
    """Symmetric-about-1.0 color limits from the actual finite data (bug-fix for the
    washed-out 0.5..1.5 scale). Returns (vmin, vmax, m) with vmin=1-m, vmax=1+m."""
    vals = [v for row in matrix for v in row
            if isinstance(v, (int, float)) and math.isfinite(v)]
    if not vals:
        return 0.5, 1.5, 0.5
    m = min(cap, max(abs(v - 1.0) for v in vals))
    m = max(m, 0.02)
    return 1.0 - m, 1.0 + m, m


def svg_heatmap(matrix, rowlabels, collabels, vmin, vmax, reverse=False, cell=17,
                minw=200) -> str:
    """Bare heatmap SVG (NO title/caption inside — those are HTML, drawn by the caller
    so they cannot be clipped by a narrow canvas). Column labels tilt up-and-right at
    9px with top/right padding sized to hold them; rows labelled every 4."""
    if not matrix or not matrix[0]:
        return "<p><em>no data</em></p>"
    nr, nc = len(matrix), len(matrix[0])
    lpad, fs, charw = 44, 9, 0.62
    lmax = max((len(str(c)) for c in collabels), default=1) * charw * fs
    tpad = int(math.ceil(0.866 * lmax)) + 10
    rpad = int(math.ceil(0.5 * lmax)) + 8
    w = max(minw, lpad + nc * cell + rpad)
    hgt = tpad + nr * cell + 10
    P = [f'<svg width="{w}" height="{hgt}" style="background:#fff;border:1px solid #ddd">']
    for j, cl in enumerate(collabels):
        cx = lpad + j * cell + cell / 2
        cy = tpad - 4
        P.append(f'<text x="{cx:.1f}" y="{cy:.1f}" font-size="{fs}" text-anchor="start" '
                 f'transform="rotate(-60 {cx:.1f} {cy:.1f})">{ESC(str(cl))}</text>')
    for i, row in enumerate(matrix):
        y = tpad + i * cell
        if i % 4 == 0:
            P.append(f'<text x="{lpad-4}" y="{y+cell-4:.1f}" font-size="{fs}" '
                     f'text-anchor="end">{ESC(str(rowlabels[i]))}</text>')
        for j, v in enumerate(row):
            x = lpad + j * cell
            P.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" '
                     f'fill="{_heat_color(v, vmin, vmax, reverse)}" stroke="#fff" stroke-width="0.4"/>')
    P.append("</svg>")
    return "".join(P)


def svg_colorbar(vmin, vmax, ticks, reverse=False, w=300, h=48) -> str:
    """Horizontal gradient legend built from solid <rect> stripes (no url()/gradient).
    Short numeric ticks only; the wordy caption is a sibling HTML element."""
    pad_l, pad_r, bar_y, bar_h, nst = 8, 8, 10, 16, 120
    bw = w - pad_l - pad_r
    P = [f'<svg width="{w}" height="{h}" style="background:#fff">']
    for i in range(nst):
        t = i / (nst - 1)
        v = vmin + t * (vmax - vmin)
        x = pad_l + t * bw
        P.append(f'<rect x="{x:.2f}" y="{bar_y}" width="{bw/nst+1.0:.2f}" height="{bar_h}" '
                 f'fill="{_heat_color(v, vmin, vmax, reverse)}"/>')
    P.append(f'<rect x="{pad_l}" y="{bar_y}" width="{bw:.1f}" height="{bar_h}" '
             f'fill="none" stroke="#999" stroke-width="0.6"/>')
    for k, tv in enumerate(ticks):
        t = 0.0 if vmax == vmin else (tv - vmin) / (vmax - vmin)
        t = max(0.0, min(1.0, t))
        x = pad_l + t * bw
        P.append(f'<line x1="{x:.1f}" y1="{bar_y+bar_h}" x2="{x:.1f}" y2="{bar_y+bar_h+3}" stroke="#333"/>')
        anc = "start" if k == 0 else "end" if k == len(ticks) - 1 else "middle"
        P.append(f'<text x="{x:.1f}" y="{bar_y+bar_h+15}" font-size="9" '
                 f'text-anchor="{anc}">{tv:g}</text>')
    P.append("</svg>")
    return "".join(P)


def heatmap_figure(title, matrix, rowlabels, collabels, vmin, vmax, cb_ticks, cb_caption,
                   reverse=False, caption_html=""):
    """HTML title + scrollable heatmap SVG + colorbar SVG + colorbar caption + optional
    how-to-read caption. Title/caption live OUTSIDE the SVG so they never clip."""
    parts = [f'<div class="figtitle">{ESC(title)}</div>',
             f'<div class="figscroll">{svg_heatmap(matrix, rowlabels, collabels, vmin, vmax, reverse)}</div>',
             f'<div class="figscroll">{svg_colorbar(vmin, vmax, cb_ticks, reverse)}</div>',
             f'<div class="cbcap">{ESC(cb_caption)}</div>']
    if caption_html:
        parts.append(caption_html)
    return "".join(parts)


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
    P.append(f'<text x="{w/2:.1f}" y="15" font-size="13" text-anchor="middle" font-weight="bold">{ESC(title)}</text>')
    P.append(f'<line x1="{pad}" y1="{sy(1.0):.1f}" x2="{w-pad}" y2="{sy(1.0):.1f}" stroke="#bbb" stroke-dasharray="4 3"/>')
    P.append(f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{h-pad}" stroke="#333"/>')
    P.append(f'<line x1="{pad}" y1="{h-pad}" x2="{w-pad}" y2="{h-pad}" stroke="#333"/>')
    for x, y, k in pts:
        P.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="3" fill="#d62728" opacity="0.7"/>')
    P.append(f'<text x="{w/2:.1f}" y="{h-6}" font-size="11" text-anchor="middle">{ESC(xlab)}</text>')
    P.append(f'<text x="14" y="{h/2:.1f}" font-size="11" text-anchor="middle" transform="rotate(-90 14 {h/2:.1f})">{ESC(ylab)}</text>')
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
    # (6) coupling
    if S:
        cp = S["vis"].get("k_r2_ratio_coupling", {})
        L.append(f"<b>R²-vs-ratio coupling (per-step)</b>: Spearman ρ = {_fmt(cp.get('spearman'))} "
                 f"over {len(cp.get('points', []))} groups (do high-R² groups project better?).")
    # (7) predictor ladder at the op cell (per regime)
    coef_arms = ("damped_linear", "damped_second_order",
                 "adaptive_linear", "adaptive_second_order")
    for reg, tag in ((S, "per-step"), (T, "per-tick")):
        if not reg:
            continue
        od, oh = reg["meta"]["operating_point"]
        parts = []
        best = None
        for m in reg["meta"].get("methods", []):
            r = _g(reg, m, od, oh)
            if not r:
                continue
            rm = r.get("weight_proj_ratio_median")
            coef = ""
            if r.get("lam_star") is not None:
                coef = f", λ*={_fmt(r.get('lam_star'))}"
                if r.get("lam2_star") is not None:
                    coef += f"/λc*={_fmt(r.get('lam2_star'))}"
                coef += f", warmup {_fmt(r.get('n_warmup'), 0)}"
            parts.append(f"{ESC(m)}: ratio {_fmt(rm)}, EVR {_fmt(r.get('pred_evr_pooled'))}, "
                         f"pred R² {_fmt(r.get('pred_r2_scalar_median'))}{coef}")
            if (m in coef_arms and rm is not None and math.isfinite(rm)
                    and (best is None or rm < best[1])):
                best = (m, rm)
        if parts:
            L.append(f"<b>Predictor ladder ({tag}, Δ={od},h={oh})</b>: "
                     + "; ".join(parts)
                     + (f". Best OOS arm: <b>{ESC(best[0])}</b> ({_fmt(best[1])})."
                        if best else "."))
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


def render(S, T, out_path, explainer_path=None):
    secs = load_explainer(explainer_path)
    figcaps = parse_figure_caps(secs.get("Reading each figure", []))

    def cap(needle):
        for k, v in figcaps.items():
            if needle.lower() in k.lower():
                return f'<p class="cap">{v}</p>'
        return ""

    def prose(*titles):
        return "".join(_md_block(secs[t]) for t in titles if secs.get(t))

    def prose_titled(*titles):
        out = []
        for t in titles:
            if secs.get(t):
                out.append(f"<h3>{ESC(t)}</h3>" + _md_block(secs[t]))
        return "".join(out)

    n = [0]

    def H2(title):
        n[0] += 1
        return f"<h2>{n[0]} — {ESC(title)}</h2>"

    P = ['<meta charset="utf-8"><title>EXP-47 — ANCHOR linear/damped projection lane</title>',
         '<style>' + CSS + '</style>']
    ref = S or T
    op_s = S["meta"]["operating_point"] if S else None
    op_t = T["meta"]["operating_point"] if T else None
    P.append("<h1>EXP-47 — MOAT ANCHOR linear / damped-linear projection lane</h1>")
    P.append(f'<p class="small">Weight-geometry replay over the EXP-57 fp32 trace (338 matrices). '
             f'Two cadence regimes: <b>S = per-step</b> (global steps, PRIMARY, paper-comparable) '
             f'and <b>T = per-tick</b> (optimizer ticks, extended-Δ sweep). '
             f'Metric contract: {ESC(str((ref or {}).get("meta", {}).get("metric_contract", "")))}. '
             f'Ratio &lt; 1 ⇒ projection beats holding the stale weights.</p>')
    for reg, tag in ((S, "regime S / per-step"), (T, "regime T / per-tick")):
        P.append(fidelity_banner(reg, tag))

    # ---- How to read this report (embedded prose) ------------------------------
    intro = prose_titled("What this analysis is",
                         "Where Δ comes from: the sliding-window protocol",
                         "Which parameters were used")
    if intro:
        P.append(H2("How to read this report"))
        P.append('<div class="explain">' + intro + '</div>')

    # ---- LEAD SCIENCE: per-scalar linearity R² ---------------------------------
    P.append(H2("Per-scalar linearity R² (the MUST metric; Wang et al. 2026)"))
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
    P.append('</div>')
    # R² histogram + coupling scatter
    P.append('<div class="side">')
    if S and S["vis"].get("i_r2_histogram", {}).get("global"):
        g = S["vis"]["i_r2_histogram"]["global"]
        lr = S["meta"].get("linearity_r2", {})
        P.append('<div class="col">' + svg_bars(
            g["counts"], "Per-scalar R² histogram (per-step, global)", 0.0, 1.0,
            vmark=[(lr.get("r2_median"), "median", "#d62728"),
                   (WANG_RL[0], "RL", "#2ca02c"), (WANG_SFT[0], "SFT", "#ff7f0e")])
            + cap("histogram") + '</div>')
    if S and S["vis"].get("k_r2_ratio_coupling"):
        cp = S["vis"]["k_r2_ratio_coupling"]
        P.append('<div class="col">' + svg_scatter(
            cp.get("points", []), "group median R²", "OOS-damped ratio @op",
            "R²-vs-ratio coupling (per-step)", spearman=cp.get("spearman"))
            + cap("coupling") + '</div>')
    P.append('</div>')
    # depth×block R² heatmap (colorbar + caption)
    if S and S["vis"].get("j_r2_depth_block_heatmap"):
        hm = S["vis"]["j_r2_depth_block_heatmap"]
        P.append(heatmap_figure(
            "Depth × block per-scalar R² (per-step) — distinct from traj_r2",
            hm["r2_median"], hm["layers"], hm["block_types"], 0.0, 1.0,
            cb_ticks=[0.0, 0.25, 0.5, 0.75, 1.0],
            cb_caption="per-scalar R² median · blue = low (less linear) → red = high (more linear)",
            reverse=False, caption_html=cap("depth")))

    # legacy per-matrix trajectory R² (traj_r2): distribution + depth×block heatmap
    if S and S["vis"].get("d_traj_r2"):
        dtr = S["vis"]["d_traj_r2"]
        pm = [d.get("traj_r2") for d in dtr.get("per_matrix", [])
              if isinstance(d.get("traj_r2"), (int, float)) and math.isfinite(d["traj_r2"])]
        if pm:
            nb = 20
            counts = [0] * nb
            for v in pm:
                counts[min(nb - 1, max(0, int(min(max(v, 0.0), 1.0) * nb)))] += 1
            P.append('<div class="figscroll">' + svg_bars(
                counts, "Legacy per-matrix trajectory R² (traj_r2) distribution",
                0.0, 1.0) + '</div>')
        dbh = dtr.get("depth_block_heatmap", {})
        if dbh.get("traj_r2"):
            P.append(heatmap_figure(
                "Depth × block trajectory R² (traj_r2, legacy per-matrix proxy)",
                dbh["traj_r2"], dbh["layers"], dbh["block_types"], 0.0, 1.0,
                cb_ticks=[0.0, 0.25, 0.5, 0.75, 1.0],
                cb_caption="traj_r2 · blue = low (less linear) → red = high (more linear)",
                reverse=False, caption_html=""))

    # ---- ratio/projection findings, both regimes -------------------------------
    P.append(H2("Projection accuracy vs horizon (both regimes)"))
    P.append('<div class="side">')
    if S:
        methods = S["meta"]["methods"]
        P.append('<div class="col"><h3>Regime S — per-step (global steps), Δ=%d</h3>%s</div>'
                 % (op_s[0], svg_line(_accuracy_series(S, methods),
                                      "median ratio vs h (global steps)", "horizon h (global steps)")))
    if T:
        methods = T["meta"]["methods"]
        P.append('<div class="col"><h3>Regime T — per-tick, Δ=%d</h3>%s</div>'
                 % (op_t[0], svg_line(_accuracy_series(T, methods),
                                      "median ratio vs h (ticks)", "horizon h (ticks)")))
    P.append('</div>')
    P.append(cap("accuracy"))

    # target-horizon sweep: ratio-vs-h with one line per Δ (per anchor spacing)
    if S and S["vis"].get("c_target_horizon_sweep"):
        ths = S["vis"]["c_target_horizon_sweep"]
        P.append(H2("Target-horizon sweep per anchor spacing Δ (per-step)"))
        P.append('<div class="side">')
        for m in S["meta"].get("methods", []):
            if m == "hold_stale":
                continue                       # flat at ratio=1 by construction
            per_d = ths.get(m)
            if not per_d:
                continue
            ser = {}
            for dstr, cd in sorted(per_d.items(), key=lambda kv: int(kv[0])):
                pts = list(zip(cd.get("h", []), cd.get("ratio_median", [])))
                if any(y is not None and y == y for _, y in pts):
                    ser[f"Δ={dstr}"] = pts
            if ser:
                P.append('<div class="col">' + svg_line(
                    ser, f"{m}: ratio vs h per Δ", "horizon h") + '</div>')
        P.append('</div>')
        P.append('<p class="small">One line per anchor spacing Δ; where each line crosses '
                 'ratio=1 is that Δ\'s safe horizon (h_star). Later crossing = projects further.</p>')

    P.append(H2("Δ-sensitivity (extended to Δ=40, per-tick) & λ-selection"))
    P.append('<div class="side">')
    if T:
        methods = T["meta"]["methods"]
        P.append('<div class="col"><h3>Δ-sensitivity at operating h=%d (per-tick)</h3>%s'
                 '<p class="small">Δ∈%s — does a wider anchor help past Δ=20?</p>%s</div>'
                 % (op_t[1], svg_line(_delta_series(T, methods),
                                      "median ratio vs Δ", "Δ (ticks)", xvals=T["meta"]["delta_ticks"]),
                    T["meta"]["delta_ticks"], cap("sensitivity")))
    if S:
        lam = S["vis"].get("h_lambda_selection", {}).get("cells", {})
        key = f"{op_s[0]},{op_s[1]}"
        cell = lam.get(key) or (next(iter(lam.values())) if lam else None)
        if cell:
            ser = {"in-sample ratio": list(zip(cell["lambda"], cell["ratio_median"]))}
            P.append('<div class="col"><h3>λ-selection at (Δ=%d,h=%d), per-step</h3>%s'
                     '<p class="small">in-sample median ratio vs λ (OOS picks per-window on '
                     'strictly-earlier data). λ=0 ⇒ hold-stale, λ=1 ⇒ naive.</p>%s</div>'
                     % (op_s[0], op_s[1], svg_line(ser, "ratio vs λ", "λ",
                                                   xvals=cell["lambda"]), cap("λ-selection")))
        mu = S["vis"].get("h_lambda_selection", {}).get("mu_cells", {})
        mcell = mu.get(key) or (next(iter(mu.values())) if mu else None)
        if mcell:
            mser = {"in-sample ratio": list(zip(mcell["mu"], mcell["ratio_median"]))}
            P.append('<div class="col"><h3>μ-selection (damped_second_order) at '
                     '(Δ=%d,h=%d), per-step</h3>%s'
                     '<p class="small">in-sample median ratio vs curvature damp μ. '
                     'μ=0 ⇒ naive_linear, μ=1 ⇒ naive_second_order.</p></div>'
                     % (op_s[0], op_s[1], svg_line(mser, "ratio vs μ", "μ",
                                                   xvals=mcell["mu"])))
    P.append('</div>')

    # ratio heatmap by layer×block (regime S, damped) — colorbar + caption
    if S and S["vis"].get("e_ratio_heatmap", {}).get("damped_linear"):
        e = S["vis"]["e_ratio_heatmap"]["damped_linear"]
        vmin, vmax, m = centered_vlim(e["ratio_median"])
        P.append(heatmap_figure(
            "OOS-damped ratio by layer × block (per-step, at operating point)",
            e["ratio_median"], e["layers"], e["block_types"], vmin, vmax,
            cb_ticks=[round(vmin, 3), round(1.0 - m / 2, 3), 1.0,
                      round(1.0 + m / 2, 3), round(vmax, 3)],
            cb_caption="ratio: blue < 1 = projection beats stale · red > 1 = worse than stale",
            reverse=False, caption_html=cap("block ratio")))

    # h_star heatmap by layer×block (regime S) — the figure the shared "Layer×block
    # ratio / h_star heatmaps" caption promises; keyed on the ratio heatmap's method.
    if S and S["vis"].get("f_hstar_heatmap"):
        hsv = S["vis"]["f_hstar_heatmap"]
        # prefer damped_linear, else the first informative (non-hold_stale) method
        # (hold_stale's h_star is 0 everywhere by construction)
        hm_m = ("damped_linear" if "damped_linear" in hsv
                else next((k for k in hsv if k != "hold_stale"), None)
                or next(iter(hsv), None))
        hh = hsv.get(hm_m) if hm_m else None
        if hh and hh.get("h_star"):
            hvals = [v for row in hh["h_star"] for v in row
                     if isinstance(v, (int, float)) and math.isfinite(v)]
            hmax = max(hvals) if hvals else 1.0
            P.append(heatmap_figure(
                f"Safe horizon h_star by layer × block (per-step, {hm_m}, Δ={hh.get('delta')})",
                hh["h_star"], hh["layers"], hh["block_types"], 0.0, hmax,
                cb_ticks=[0, round(hmax / 2), round(hmax)],
                cb_caption="h_star = furthest horizon with median ratio < 1 · blue = "
                           "projects further (good) · red = short/zero safe horizon",
                reverse=True, caption_html=cap("h_star heatmaps")))

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
        P.append(cap("special-groups"))

    # ---- absolute prediction accuracy (regression view) ------------------------
    P.append(H2("Absolute prediction accuracy (regression view)"))
    reg_prose = prose("Absolute prediction accuracy (regression view)",
                      "Absolute prediction accuracy")
    P.append('<div class="explain">' + (reg_prose or FALLBACK_REGRESSION_PROSE)
             + '</div>')
    for reg, tag in ((S, "S / per-step"), (T, "T / per-tick")):
        if not reg:
            continue
        od, oh = reg["meta"]["operating_point"]
        P.append(f"<h3>Regime {ESC(tag)} — operating cell (Δ={od}, h={oh})</h3>")
        rowsh = []
        for m in reg["meta"].get("methods", []):
            r = _g(reg, m, od, oh)
            rowsh.append(
                f'<tr><td class="l">{ESC(m)}</td>'
                f'<td class="mono">{_fmt(r.get("pred_evr_pooled"))}</td>'
                f'<td class="mono">{_fmt(r.get("pred_r2_scalar_median"))}</td>'
                f'<td class="mono">{_fmt(r.get("pred_r2_scalar_frac_gt_0.7"))}</td>'
                f'<td class="mono">{_fmt(r.get("pred_r2_scalar_frac_lt_0"))}</td>'
                f'<td class="mono">{_fmt(r.get("pred_r2_scalar_n"), 0)}</td>'
                f'<td class="mono">{_fmt(r.get("pred_r2_scalar_n_excluded"), 0)}</td></tr>')
        P.append('<div class="tblwrap">' + _table(
            ["method", "pred_evr_pooled", "pred R² median (scalar)", "frac > 0.7",
             "frac < 0", "n", "n_excluded"], rowsh) + '</div>')
        evr = reg["vis"].get("o_pred_evr_vs_h", {})
        ser = {m: list(zip(v.get("h", []), v.get("pred_evr_pooled", [])))
               for m, v in evr.items()}
        if any(s for s in ser.values()):
            P.append('<div class="figscroll">' + svg_line(
                ser, f"pooled EVR vs h at operating Δ ({tag})",
                "horizon h", y_max=1.0, yref=0.0)
                + '</div><p class="small">EVR = 1 − Σ‖e‖²/Σ‖b‖² (global group); '
                  '0 = hold-stale, 1 = perfect; negative values clip to the axis.</p>')
        ph = reg["vis"].get("n_pred_r2_scalar_hist")
        if ph and ph.get("methods"):
            P.append('<div class="side">')
            for m, counts in ph["methods"].items():
                P.append('<div class="col">' + svg_bars(
                    counts, f"per-scalar pred R² — {m} ({tag})", -1.0, 1.0,
                    vmark=[(0.7, "strong", "#2ca02c"), (0.0, "0", "#888")]) + '</div>')
            P.append('</div>')
            P.append(cap("pred R²"))

    # ---- adaptive coefficient trajectories (per-emit visual; degrade if absent) --
    if any(reg and reg["vis"].get("l_adaptive_coef_traj") for reg in (S, T)):
        P.append(H2("Adaptive coefficient trajectories (GLOBAL group, OOS per window)"))
        for reg, tag in ((S, "per-step"), (T, "per-tick")):
            ct = (reg or {}).get("vis", {}).get("l_adaptive_coef_traj") if reg else None
            if not ct:
                continue
            for m, cellsd in ct.items():
                P.append(f'<h3>{ESC(m)} ({ESC(tag)})</h3><div class="side">')
                for ck, cd in sorted(cellsd.items()):
                    lam = cd.get("lam") or []
                    ser = {"λ": [(i, v) for i, v in enumerate(lam)]}
                    if cd.get("lam2"):
                        ser["λc"] = [(i, v) for i, v in enumerate(cd["lam2"])]
                    ys = [v for pts in ser.values() for _, v in pts
                          if v is not None and v == v]
                    lo = min([0.0] + ys) if ys else 0.0
                    P.append('<div class="col">'
                             + svg_line(ser, f"(Δ,h)=({ck}) coefficient vs window",
                                        "window index", y_max=1.6, yref=1.0,
                                        y_min=min(0.0, lo))
                             + f'<p class="small">warm-up windows (NaN, dropped): '
                               f'{_fmt(cd.get("n_warmup"), 0)}</p></div>')
                P.append('</div>')

    # ---- what was verified (embedded prose, BEFORE the verdict) -----------------
    verified = prose("What was verified before trusting these numbers")
    if verified:
        P.append(H2("What was verified before trusting these numbers"))
        P.append('<div class="explain">' + verified + '</div>')

    # ---- verdict ---------------------------------------------------------------
    P.append(H2("Lane verdict (direction-agnostic)"))
    findings = prose("The findings in one paragraph")
    if findings:
        P.append('<div class="lead">' + findings + '</div>')
    P.append('<div class="verdict"><ul>')
    for line in build_verdict(S, T):
        P.append(f"<li>{line}</li>")
    P.append('</ul><p class="small">A decisive negative result (damped never beats hold-stale; '
             'wider Δ never helps; ratio &gt; 1 at long h; low R²) is a VALID finding — the lane '
             'PASSES by producing a correct, schema-verified, machine-readable answer either way.</p></div>')

    # ---- glossary (embedded prose) ---------------------------------------------
    gl = parse_glossary(secs.get("Glossary", []))
    if gl:
        P.append(H2("Glossary"))
        rows = "".join(f'<tr><td class="l t">{t}</td><td class="l d">{d}</td></tr>' for t, d in gl)
        P.append('<div class="tblwrap"><table class="gloss">'
                 '<tr><th class="l">term</th><th class="l">definition</th></tr>'
                 + rows + '</table></div>')

    # ---- provenance (technical footer) -----------------------------------------
    P.append('<div class="foot"><b>Provenance</b>')
    for reg, tag in ((S, "per-step"), (T, "per-tick")):
        if not reg:
            continue
        m = reg["meta"]
        P.append(f'<p class="small mono">{ESC(tag)}: fidelity={m.get("fidelity", "full")} '
                 f'cadence={m.get("cadence")} unit={m.get("unit")} '
                 f'n_ticks={m.get("n_ticks")} band={m.get("band")} n_rows={m.get("n_rows")} '
                 f'lam_grid={len(m.get("lam_grid", []))}pts fingerprint={m.get("stats_cache_fingerprint")} '
                 f'panel={m.get("panel_cache_fingerprint")} '
                 f'gates={ {k: v.get("pass") for k, v in m.get("gates", {}).items()} }</p>')
    P.append('</div>')

    with open(out_path, "w") as f:
        f.write("".join(P))
    print(f"report written: {out_path} ({os.path.getsize(out_path)} bytes)", flush=True)


def main():
    ap = argparse.ArgumentParser(description="#47 self-contained HTML report (both regimes)")
    ap.add_argument("--perstep", default="", help="regime S scorecard dir")
    ap.add_argument("--pertick", default="", help="regime T scorecard dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--explainer", default="",
                    help="report_explainer.md (default: sibling of --out); prose degrades if absent")
    args = ap.parse_args()
    S = _load(args.perstep)
    T = _load(args.pertick)
    if not S and not T:
        raise SystemExit("need at least one of --perstep / --pertick")
    explainer = args.explainer or os.path.join(os.path.dirname(os.path.abspath(args.out)),
                                               "report_explainer.md")
    render(S, T, args.out, explainer)


if __name__ == "__main__":
    main()
