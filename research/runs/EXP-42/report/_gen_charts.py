#!/usr/bin/env python3
"""Generate runs/EXP-42/report/charts.html — 5 inline-SVG charts from series.json.

No external libs / no CDN: pure hand-built SVG so the report renders offline.
Every plotted coordinate is computed directly from series.json (single source of truth).
"""
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SERIES = os.path.join(HERE, "series.json")
OUT = os.path.join(HERE, "charts.html")

with open(SERIES) as f:
    D = json.load(f)

CELLS = D["cells"]
META = D["meta"]

# ---- consistent per-cell color (light + dark friendly mid-ramp hexes) ----
COLOR = {
    "A25":              "#378ADD",  # blue
    "A50":              "#1D9E75",  # teal
    "A75":              "#EF9F27",  # amber
    "L":                "#D4537E",  # pink
    "EXP41_ref_5over5": "#888780",  # gray (reference)
    "EXP41_alpha1p0":   "#E24B4A",  # red
}
LABEL = {
    "A25":              "A25  (fixed_linear, alpha=0.25)",
    "A50":              "A50  (fixed_linear, alpha=0.50)",
    "A75":              "A75  (fixed_linear, alpha=0.75)",
    "L":                "L  (learned_linear, alpha=1.0)",
    "EXP41_ref_5over5": "EXP-41 5/5-ref  (lookahead OFF, stable)",
    "EXP41_alpha1p0":   "EXP-41 alpha=1.0  (full catch-up)",
}
# axis / grid / text colors that survive light & dark mode
AX = "var(--color-text-secondary)"
GRID = "var(--color-border-tertiary)"
TXT = "var(--color-text-primary)"
MUT = "var(--color-text-secondary)"

W, H = 720, 360
# default plot box
ML, MR, MT, MB = 64, 150, 30, 46


SHORT = {
    "A25": "A25", "A50": "A50", "A75": "A75", "L": "L",
    "EXP41_ref_5over5": "EXP-41 5/5-ref", "EXP41_alpha1p0": "EXP-41 a=1.0",
}


def esc(s):
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def fmt(v, n=2):
    return f"{v:.{n}f}".rstrip("0").rstrip(".") if "." in f"{v:.{n}f}" else f"{v:.{n}f}"


class Plot:
    """Linear-mapped SVG plot area."""

    def __init__(self, x0, x1, y0, y1, ml=ML, mr=MR, mt=MT, mb=MB):
        self.x0, self.x1, self.y0, self.y1 = x0, x1, y0, y1
        self.ml, self.mr, self.mt, self.mb = ml, mr, mt, mb
        self.pw = W - ml - mr
        self.ph = H - mt - mb

    def sx(self, x):
        if self.x1 == self.x0:
            return self.ml + self.pw / 2
        return self.ml + (x - self.x0) / (self.x1 - self.x0) * self.pw

    def sy(self, y):
        if self.y1 == self.y0:
            return self.mt + self.ph / 2
        return self.mt + (self.y1 - y) / (self.y1 - self.y0) * self.ph

    def frame(self):
        s = []
        s.append(f'<rect x="{self.ml}" y="{self.mt}" width="{self.pw}" height="{self.ph}" '
                 f'fill="none" stroke="{GRID}" stroke-width="1"/>')
        return "".join(s)

    def xticks(self, ticks, label=None, fmtfn=lambda v: f"{v:g}"):
        s = []
        for t in ticks:
            x = self.sx(t)
            s.append(f'<line x1="{x:.1f}" y1="{self.mt}" x2="{x:.1f}" y2="{self.mt+self.ph}" '
                     f'stroke="{GRID}" stroke-width="0.5"/>')
            s.append(f'<text x="{x:.1f}" y="{self.mt+self.ph+16}" text-anchor="middle" '
                     f'font-size="11" fill="{AX}">{fmtfn(t)}</text>')
        if label:
            s.append(f'<text x="{self.ml+self.pw/2:.1f}" y="{H-6}" text-anchor="middle" '
                     f'font-size="12" fill="{MUT}">{esc(label)}</text>')
        return "".join(s)

    def yticks(self, ticks, label=None, fmtfn=lambda v: f"{v:g}"):
        s = []
        for t in ticks:
            y = self.sy(t)
            s.append(f'<line x1="{self.ml}" y1="{y:.1f}" x2="{self.ml+self.pw}" y2="{y:.1f}" '
                     f'stroke="{GRID}" stroke-width="0.5"/>')
            s.append(f'<text x="{self.ml-8}" y="{y+4:.1f}" text-anchor="end" '
                     f'font-size="11" fill="{AX}">{fmtfn(t)}</text>')
        if label:
            yc = self.mt + self.ph / 2
            s.append(f'<text x="14" y="{yc:.1f}" text-anchor="middle" font-size="12" '
                     f'fill="{MUT}" transform="rotate(-90 14 {yc:.1f})">{esc(label)}</text>')
        return "".join(s)


def legend(plot, items, x=None, y=None):
    """items: list of (color, text, dash, marker) — marker in {'line','filled','hollow','bar','band'}"""
    x = (plot.ml + plot.pw + 14) if x is None else x
    y = (plot.mt + 4) if y is None else y
    s = []
    dy = 19
    for i, (col, txt, dash, marker) in enumerate(items):
        yy = y + i * dy
        if marker == "band":
            s.append(f'<rect x="{x}" y="{yy-7}" width="22" height="11" fill="{col}" '
                     f'fill-opacity="0.18" stroke="{col}" stroke-width="1" stroke-dasharray="3 2"/>')
        elif marker == "bar":
            s.append(f'<rect x="{x}" y="{yy-7}" width="22" height="11" fill="{col}"/>')
        else:
            da = f' stroke-dasharray="{dash}"' if dash else ""
            s.append(f'<line x1="{x}" y1="{yy-2}" x2="{x+22}" y2="{yy-2}" stroke="{col}" '
                     f'stroke-width="2"{da}/>')
            cx = x + 11
            if marker == "filled":
                s.append(f'<circle cx="{cx}" cy="{yy-2}" r="3.2" fill="{col}"/>')
            elif marker == "hollow":
                s.append(f'<circle cx="{cx}" cy="{yy-2}" r="3.2" fill="var(--color-background-primary)" '
                         f'stroke="{col}" stroke-width="1.5"/>')
        s.append(f'<text x="{x+28}" y="{yy+2}" font-size="11" fill="{MUT}">{esc(txt)}</text>')
    return "".join(s)


def svg_open(aria):
    return (f'<svg viewBox="0 0 {W} {H}" width="100%" style="height:auto;font-family:var(--font-sans)" '
            f'role="img" aria-label="{esc(aria)}" xmlns="http://www.w3.org/2000/svg">')


def polyline(plot, pts, col, dash=None, w=2):
    if not pts:
        return ""
    d = " ".join(f"{plot.sx(x):.1f},{plot.sy(y):.1f}" for x, y in pts)
    da = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<polyline points="{d}" fill="none" stroke="{col}" stroke-width="{w}" '
            f'stroke-linejoin="round" stroke-linecap="round"{da}/>')


# =====================================================================
# CHART 1 — alpha -> collapse-onset step (bar + EXP-41 alpha=1.0 marker)
# =====================================================================
def chart1():
    # bars for A25/A50/A75/L ; marker for EXP41_alpha1p0 ; ref line for 5/5
    order = ["A25", "A50", "A75", "L"]
    alphas = {"A25": 0.25, "A50": 0.50, "A75": 0.75, "L": 1.0}
    onset = {k: CELLS[k]["collapse_onset_step"] for k in order}
    ymax = 90
    p = Plot(0, 1, 0, ymax, ml=64, mr=150, mt=30, mb=58)
    s = [svg_open("Bar chart: projection strength alpha versus collapse-onset global step "
                  "for cells A25, A50, A75, L, with EXP-41 alpha=1.0 marker and 5/5 stable reference.")]
    s.append(p.frame())
    s.append(p.yticks(range(0, 91, 15), "collapse-onset global step"))
    # x ticks at the alpha positions
    xs = [0.25, 0.50, 0.75, 1.0]
    s.append('<g>')
    for a in xs:
        x = p.sx(a)
        s.append(f'<text x="{x:.1f}" y="{p.mt+p.ph+16}" text-anchor="middle" font-size="11" '
                 f'fill="{AX}">{a:.2f}</text>')
    s.append(f'<text x="{p.ml+p.pw/2:.1f}" y="{H-6}" text-anchor="middle" font-size="12" '
             f'fill="{MUT}">projection strength alpha</text>')
    s.append('</g>')
    # 5/5 ref: no collapse (stable to 100) -> dashed line near top + note
    yref = p.sy(ymax)  # top of plot
    s.append(f'<line x1="{p.ml}" y1="{p.mt+4}" x2="{p.ml+p.pw}" y2="{p.mt+4}" '
             f'stroke="{COLOR["EXP41_ref_5over5"]}" stroke-width="1.5" stroke-dasharray="5 3"/>')
    s.append(f'<text x="{p.ml+p.pw-4}" y="{p.mt+18}" text-anchor="end" font-size="10.5" '
             f'fill="{MUT}">EXP-41 5/5-ref: no collapse (stable to 100)</text>')
    bw = 34
    for a in xs:
        k = [kk for kk in order if alphas[kk] == a][0]
        x = p.sx(a)
        col = COLOR[k]
        ov = onset[k]
        if ov is None:
            # A75 truncated: hatched 'no verdict' stub from baseline
            stub = 27  # last observed step (truncation)
            s.append(f'<rect x="{x-bw/2:.1f}" y="{p.sy(stub):.1f}" width="{bw}" '
                     f'height="{p.sy(0)-p.sy(stub):.1f}" fill="{col}" fill-opacity="0.22" '
                     f'stroke="{col}" stroke-width="1" stroke-dasharray="3 2"/>')
            s.append(f'<text x="{x:.1f}" y="{p.sy(stub)-6:.1f}" text-anchor="middle" '
                     f'font-size="10.5" fill="{MUT}">log cut @27</text>')
            s.append(f'<text x="{x:.1f}" y="{p.sy(stub)-19:.1f}" text-anchor="middle" '
                     f'font-size="10.5" fill="{MUT}">(no verdict)</text>')
        else:
            s.append(f'<rect x="{x-bw/2:.1f}" y="{p.sy(ov):.1f}" width="{bw}" '
                     f'height="{p.sy(0)-p.sy(ov):.1f}" fill="{col}" rx="2"/>')
            s.append(f'<text x="{x:.1f}" y="{p.sy(ov)-6:.1f}" text-anchor="middle" '
                     f'font-size="11" fill="{TXT}">{ov}</text>')
            s.append(f'<text x="{x:.1f}" y="{p.sy(0)-7:.1f}" text-anchor="middle" '
                     f'font-size="10.5" fill="#ffffff">{k}</text>')
    # EXP41 alpha=1.0 distinct diamond marker at alpha=1.0
    ka = "EXP41_alpha1p0"
    ov = CELLS[ka]["collapse_onset_step"]
    xa = p.sx(1.0) + 10  # offset right so it doesn't sit on the L bar
    ya = p.sy(ov)
    s.append(f'<line x1="{xa:.1f}" y1="{p.sy(0):.1f}" x2="{xa:.1f}" y2="{ya:.1f}" '
             f'stroke="{COLOR[ka]}" stroke-width="1" stroke-dasharray="2 2"/>')
    s.append(f'<path d="M {xa:.1f} {ya-6:.1f} L {xa+6:.1f} {ya:.1f} L {xa:.1f} {ya+6:.1f} '
             f'L {xa-6:.1f} {ya:.1f} Z" fill="{COLOR[ka]}"/>')
    s.append(f'<text x="{xa+9:.1f}" y="{ya-6:.1f}" font-size="10.5" fill="{MUT}">a=1.0 -&gt; {ov}</text>')
    # non-monotonicity annotation
    s.append(f'<text x="{p.ml+8}" y="{p.mt+16}" font-size="11" fill="{MUT}">'
             f'non-monotone: A50 (alpha=0.50) collapses latest</text>')
    # legend
    items = [(COLOR[k], LABEL[k].split("  ")[0] + f"  (a={alphas[k]:g})", None, "bar") for k in order]
    items.append((COLOR["EXP41_alpha1p0"], "EXP-41 a=1.0", None, "filled"))
    s.append(legend(p, items, x=p.ml + p.pw + 14, y=p.mt + 40))
    s.append("</svg>")
    return "".join(s)


# =====================================================================
# CHART 2 — anchor_align_cos per fire (mechanism chart)
#   x = fire tick, y = cos ; hollow=raw_stale, filled=extrapolated ; y=0 line
# =====================================================================
def chart2():
    order = ["A25", "A50", "A75", "L", "EXP41_ref_5over5", "EXP41_alpha1p0"]
    allv = [pt["value"] for k in order for pt in CELLS[k]["anchor_align_cos"]]
    allt = [pt["tick"] for k in order for pt in CELLS[k]["anchor_align_cos"]]
    ymin, ymax = -0.10, 0.16
    xmax = max(allt)
    p = Plot(0, xmax, ymin, ymax, ml=64, mr=176, mt=30, mb=46)
    s = [svg_open("Line chart: per-fire anchor_align_cos versus optimizer tick for all cells; "
                  "hollow markers are raw-stale fires, filled markers are extrapolated fires; "
                  "a zero reference line shows the near-zero, sign-oscillating regime.")]
    s.append(p.frame())
    yt = [-0.10, -0.05, 0.0, 0.05, 0.10, 0.15]
    s.append(p.yticks(yt, "anchor_align_cos  (cos with g_live)", fmtfn=lambda v: f"{v:.2f}"))
    s.append(p.xticks(range(0, xmax + 1, 20), "anchor fire (optimizer tick;  ~2 ticks / global step)"))
    # emphasized y=0 line
    y0 = p.sy(0.0)
    s.append(f'<line x1="{p.ml}" y1="{y0:.1f}" x2="{p.ml+p.pw}" y2="{y0:.1f}" '
             f'stroke="{AX}" stroke-width="1.3"/>')
    for k in order:
        col = COLOR[k]
        fires = CELLS[k]["anchor_align_cos"]
        pts = [(pt["tick"], pt["value"]) for pt in fires]
        dash = "5 3" if k == "EXP41_ref_5over5" else None
        s.append(polyline(p, pts, col, dash=dash, w=1.6))
        for pt in fires:
            cx, cy = p.sx(pt["tick"]), p.sy(pt["value"])
            if pt["phase"] == "extrapolated":
                s.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="3.4" fill="{col}"/>')
            else:
                s.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="3.4" '
                         f'fill="var(--color-background-primary)" stroke="{col}" stroke-width="1.5"/>')
    # legend (cells) + marker key
    items = [(COLOR[k], SHORT[k],
              ("5 3" if k == "EXP41_ref_5over5" else None), "line") for k in order]
    s.append(legend(p, items, x=p.ml + p.pw + 12, y=p.mt + 6))
    ky = p.mt + 6 + len(items) * 19 + 8
    s.append(f'<circle cx="{p.ml+p.pw+23}" cy="{ky}" r="3.4" '
             f'fill="var(--color-background-primary)" stroke="{MUT}" stroke-width="1.5"/>')
    s.append(f'<text x="{p.ml+p.pw+40}" y="{ky+4}" font-size="10.5" fill="{MUT}">raw-stale fire</text>')
    s.append(f'<circle cx="{p.ml+p.pw+23}" cy="{ky+18}" r="3.4" fill="{MUT}"/>')
    s.append(f'<text x="{p.ml+p.pw+40}" y="{ky+22}" font-size="10.5" fill="{MUT}">extrapolated fire</text>')
    s.append("</svg>")
    return "".join(s)


# =====================================================================
# CHART 3 — response_length/mean vs step, per cell + 2x collapse threshold
# =====================================================================
def chart3():
    order = ["A25", "A50", "A75", "L", "EXP41_ref_5over5", "EXP41_alpha1p0"]
    series = {k: CELLS[k]["series"]["response_length_mean"] for k in order}
    xmax = max(pt[0] for k in order for pt in series[k])
    ymax = 900
    p = Plot(0, xmax, 0, ymax, ml=64, mr=176, mt=30, mb=46)
    s = [svg_open("Line chart: mean response length versus global step per cell, showing the length "
                  "explosion; each cell's 2x first-25-step collapse threshold is a dashed horizontal line "
                  "and its collapse-onset step is marked.")]
    s.append(p.frame())
    s.append(p.yticks(range(0, 901, 150), "response_length / mean (tokens)"))
    s.append(p.xticks(range(0, xmax + 1, 10), "global step"))
    for k in order:
        col = COLOR[k]
        dash = "5 3" if k == "EXP41_ref_5over5" else None
        s.append(polyline(p, [(x, y) for x, y in series[k]], col, dash=dash, w=1.7))
    # thresholds + onset markers (only the collapsing EXP-42 cells + EXP41 alpha1)
    for k in ["A25", "A50", "L", "EXP41_alpha1p0"]:
        col = COLOR[k]
        thr = CELLS[k]["collapse_threshold"]
        yt = p.sy(thr)
        s.append(f'<line x1="{p.ml}" y1="{yt:.1f}" x2="{p.ml+p.pw}" y2="{yt:.1f}" '
                 f'stroke="{col}" stroke-width="1" stroke-dasharray="4 3" opacity="0.8"/>')
        onset = CELLS[k]["collapse_onset_step"]
        if onset is not None:
            # value at onset
            ov = next((y for x, y in series[k] if x == onset), thr)
            cx, cy = p.sx(onset), p.sy(ov)
            s.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="4" fill="{col}" '
                     f'stroke="var(--color-background-primary)" stroke-width="1"/>')
            s.append(f'<text x="{cx:.1f}" y="{cy-8:.1f}" text-anchor="middle" font-size="10" '
                     f'fill="{col}">@{onset}</text>')
    s.append(f'<text x="{p.ml+8}" y="{p.mt+14}" font-size="10.5" fill="{MUT}">'
             f'dashed = each cell 2x-first-25-step threshold; dot = collapse onset</text>')
    items = [(COLOR[k], SHORT[k],
              ("5 3" if k == "EXP41_ref_5over5" else None), "line") for k in order]
    s.append(legend(p, items, x=p.ml + p.pw + 12, y=p.mt + 30))
    s.append("</svg>")
    return "".join(s)


# =====================================================================
# CHART 4 — val-core acc trajectories + EXP-41 5/5 band + alpha=1.0
# =====================================================================
def chart4():
    order = ["A25", "A50", "A75", "L", "EXP41_alpha1p0"]
    p = Plot(25, 100, 0.0, 0.8, ml=64, mr=176, mt=30, mb=46)
    s = [svg_open("Line chart: validation core accuracy at steps 25/50/75/100 for cells A25, A50, A75, L "
                  "and EXP-41 alpha=1.0, against a shaded EXP-41 5/5 reference band around 0.70-0.73; "
                  "no cell reaches the band.")]
    s.append(p.frame())
    yt = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    s.append(p.yticks(yt, "val-core accuracy", fmtfn=lambda v: f"{v:.1f}"))
    s.append(p.xticks([25, 50, 75, 100], "validation global step"))
    # EXP-41 5/5 reference band = data-true min/max of its val (0.6998..0.7255)
    ref = CELLS["EXP41_ref_5over5"]["val"]
    refmin = min(y for _, y in ref)
    refmax = max(y for _, y in ref)
    yb0, yb1 = p.sy(refmin), p.sy(refmax)
    s.append(f'<rect x="{p.ml}" y="{yb1:.1f}" width="{p.pw}" height="{yb0-yb1:.1f}" '
             f'fill="{COLOR["EXP41_ref_5over5"]}" fill-opacity="0.16"/>')
    s.append(polyline(p, [(x, y) for x, y in ref], COLOR["EXP41_ref_5over5"], dash="5 3", w=1.8))
    for x, y in ref:
        s.append(f'<circle cx="{p.sx(x):.1f}" cy="{p.sy(y):.1f}" r="3" '
                 f'fill="{COLOR["EXP41_ref_5over5"]}"/>')
    # the sweep cells + alpha=1.0
    for k in order:
        col = COLOR[k]
        v = CELLS[k]["val"]
        pts = [(x, y) for x, y in v]
        s.append(polyline(p, pts, col, w=2))
        for x, y in pts:
            s.append(f'<circle cx="{p.sx(x):.1f}" cy="{p.sy(y):.1f}" r="3.4" fill="{col}"/>')
        # label last point for the short single-point cells
        if len(pts) == 1:
            x, y = pts[0]
            s.append(f'<text x="{p.sx(x)+6:.1f}" y="{p.sy(y)+4:.1f}" font-size="10" '
                     f'fill="{col}">{k} {y:.3f}</text>')
    s.append(f'<text x="{p.ml+p.pw-4}" y="{(yb0+yb1)/2+4:.1f}" text-anchor="end" font-size="10.5" '
             f'fill="{MUT}">EXP-41 5/5-ref band ({refmin:.3f}-{refmax:.3f}, val@100=0.7066)</text>')
    s.append(f'<text x="{p.ml+8}" y="{p.mt+14}" font-size="10.5" fill="{MUT}">'
             f'no look-ahead cell reaches the reference band</text>')
    items = [(COLOR[k], SHORT[k], None, "line") for k in order]
    items.append((COLOR["EXP41_ref_5over5"], "EXP-41 5/5-ref (band)", "5 3", "band"))
    s.append(legend(p, items, x=p.ml + p.pw + 12, y=p.mt + 30))
    s.append("</svg>")
    return "".join(s)


# =====================================================================
# CHART 5 — score & entropy vs step (two stacked panels)
# =====================================================================
def chart5():
    order = ["A25", "A50", "A75", "L", "EXP41_ref_5over5", "EXP41_alpha1p0"]
    xmax = 100
    # two panels stacked inside one svg, taller H
    Hh = 470
    pw = W - 64 - 176
    panelh = 185
    gap = 26
    s = [f'<svg viewBox="0 0 {W} {Hh}" width="100%" style="height:auto;font-family:var(--font-sans)" '
         f'role="img" aria-label="Two stacked line charts: top panel critic score mean versus global '
         f'step per cell, bottom panel actor entropy versus global step per cell, showing the '
         f'learning-then-degradation arc." xmlns="http://www.w3.org/2000/svg">']

    def panel(metric, y_top, ylab, ymax, yticks, title):
        ml, mt = 64, y_top
        ph = panelh
        out = []
        out.append(f'<rect x="{ml}" y="{mt}" width="{pw}" height="{ph}" fill="none" '
                   f'stroke="{GRID}" stroke-width="1"/>')
        out.append(f'<text x="{ml+8}" y="{mt+14}" font-size="11" fill="{MUT}">{esc(title)}</text>')

        def sx(x):
            return ml + x / xmax * pw

        def sy(v):
            return mt + (ymax - v) / ymax * ph
        for t in yticks:
            yy = sy(t)
            out.append(f'<line x1="{ml}" y1="{yy:.1f}" x2="{ml+pw}" y2="{yy:.1f}" '
                       f'stroke="{GRID}" stroke-width="0.5"/>')
            out.append(f'<text x="{ml-8}" y="{yy+4:.1f}" text-anchor="end" font-size="11" '
                       f'fill="{AX}">{t:g}</text>')
        for t in range(0, xmax + 1, 10):
            xx = sx(t)
            out.append(f'<line x1="{xx:.1f}" y1="{mt}" x2="{xx:.1f}" y2="{mt+ph}" '
                       f'stroke="{GRID}" stroke-width="0.5"/>')
            out.append(f'<text x="{xx:.1f}" y="{mt+ph+15}" text-anchor="middle" font-size="11" '
                       f'fill="{AX}">{t}</text>')
        yc = mt + ph / 2
        out.append(f'<text x="14" y="{yc:.1f}" text-anchor="middle" font-size="12" fill="{MUT}" '
                   f'transform="rotate(-90 14 {yc:.1f})">{esc(ylab)}</text>')
        for k in order:
            col = COLOR[k]
            dash = "5 3" if k == "EXP41_ref_5over5" else None
            data = CELLS[k]["series"][metric]
            d = " ".join(f"{sx(x):.1f},{sy(min(y,ymax)):.1f}" for x, y in data)
            da = f' stroke-dasharray="{dash}"' if dash else ""
            out.append(f'<polyline points="{d}" fill="none" stroke="{col}" stroke-width="1.6" '
                       f'stroke-linejoin="round"{da}/>')
        return "".join(out), sx, mt + ph

    # top: critic/score/mean
    out1, _, _ = panel("score_mean", 24, "critic / score / mean", 0.9,
                       [0, 0.2, 0.4, 0.6, 0.8], "learning then degradation: reward")
    s.append(out1)
    # bottom: actor/entropy
    out2, _, b2 = panel("entropy", 24 + panelh + gap, "actor / entropy", 6.5,
                        [0, 1.5, 3.0, 4.5, 6.0], "entropy")
    s.append(out2)
    s.append(f'<text x="{64+pw/2:.1f}" y="{Hh-6}" text-anchor="middle" font-size="12" '
             f'fill="{MUT}">global step</text>')
    # legend on the right spanning both panels
    items = [(COLOR[k], SHORT[k],
              ("5 3" if k == "EXP41_ref_5over5" else None), "line") for k in order]
    lx = 64 + pw + 12
    ly = 24 + 6
    for i, (col, txt, dash, _) in enumerate(items):
        yy = ly + i * 19
        da = f' stroke-dasharray="{dash}"' if dash else ""
        s.append(f'<line x1="{lx}" y1="{yy-2}" x2="{lx+22}" y2="{yy-2}" stroke="{col}" '
                 f'stroke-width="2"{da}/>')
        s.append(f'<text x="{lx+28}" y="{yy+2}" font-size="11" fill="{MUT}">{esc(txt)}</text>')
    s.append("</svg>")
    return "".join(s)


# ---------------------------------------------------------------------
figs = [
    ("chart1",
     "alpha -> collapse-onset step. Bars give the collapse-onset global step for each "
     "look-ahead-ON fixed_linear / learned cell (A25=38, A50=83, L=43); the diamond marks "
     "EXP-41 alpha=1.0 (onset 57). A75 truncated at step 27 (hatched, no verdict). The "
     "EXP-41 5/5-ref dashed line denotes no collapse (stable to step 100). Onset is "
     "non-monotone in alpha: A50 (alpha=0.50) survives longest.",
     chart1),
    ("chart2",
     "anchor_align_cos per anchor fire (the mechanism chart). x = optimizer tick, y = cosine "
     "of the anchor update with the live gradient. Hollow markers = raw-stale (WARMING) fires; "
     "filled markers = extrapolated look-ahead fires; the bold line is y=0. Across every cell the "
     "cosine sits in a near-zero, sign-oscillating band (about -0.07 to +0.13) — extrapolation "
     "does not lift alignment above the EXP-41 5/5-ref raw-stale baseline (dashed gray).",
     chart2),
    ("chart3",
     "response_length / mean vs global step, per cell. Shows the terminal length explosion. Each "
     "dashed horizontal line is that cell's collapse threshold (2x its own first-25-step mean); the "
     "dot marks the collapse-onset step (A25@38, A50@83, L@43, EXP-41 alpha=1.0@57). The EXP-41 "
     "5/5-ref (dashed gray) never breaches. A75 (amber) ends at step 27 with length still declining "
     "(log truncated, no collapse verdict).",
     chart3),
    ("chart4",
     "val-core accuracy trajectories at steps 25/50/75/100. Sweep cells A25/A50/A75/L and EXP-41 "
     "alpha=1.0 are plotted against the shaded EXP-41 5/5-ref band (~0.70-0.73, val@100=0.7066). "
     "No look-ahead-ON cell reaches the band; A75 and EXP-41 alpha=1.0 end near zero. A25/A75/L "
     "have only val@25 (collapsed/truncated before later vals).",
     chart4),
    ("chart5",
     "critic/score/mean (top) and actor/entropy (bottom) vs global step, per cell. Captures the "
     "learning-then-degradation arc: reward climbs then decays as each look-ahead cell collapses, "
     "while the EXP-41 5/5-ref (dashed gray) holds. Entropy drops sharply at the step-10/11 "
     "anchor-engagement boundary and then bleeds down. Short cells (A25/A75/L) stop where their "
     "logs end.",
     chart5),
]

parts = ['<div style="font-family:var(--font-sans)">']
for cid, cap, fn in figs:
    parts.append(f'<figure id="{cid}" style="margin:0 0 2rem;">')
    parts.append(fn())
    parts.append(f'<figcaption style="font-size:13px;color:var(--color-text-secondary);'
                 f'line-height:1.6;margin-top:8px;">{esc(cap)}</figcaption>')
    parts.append('</figure>')
parts.append('</div>')

with open(OUT, "w") as f:
    f.write("\n".join(parts))
print("wrote", OUT)
print("figures:", ", ".join(c for c, _, _ in figs))
