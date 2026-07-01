#!/usr/bin/env python3
"""weight_proj/report.py — self-contained smoke report builder.

Renders `research/reports/infra-b-sweep-engine-selftest.html`: one curve per new
predictor family (weight_proj_ratio vs horizon h), the bf16 DIFFERENCED-noise-floor
gate table (per (block,h): floor, residual, SNR, and the `bf16-unreliable` flag), and
the SPARSE-SUBSET (PuLSE) characterization (changed-element fraction + ULP-multiple
distribution so a dense L2 can never hide a sparse signal). Pure Python string
templating — no matplotlib/JS deps; curves rendered as inline SVG so the file is
fully self-contained and re-openable offline.
"""
from __future__ import annotations

import html
import json


def _svg_curve(series: dict[str, list[tuple[int, float]]], title: str,
               w=520, h=300, y_max=1.6) -> str:
    """Inline SVG line chart: series name -> [(x=h, y=ratio)]. y=1.0 reference line."""
    pad = 44
    xs = sorted({x for pts in series.values() for x, _ in pts})
    if not xs:
        return f"<p><em>no data for {html.escape(title)}</em></p>"
    x_min, x_max = min(xs), max(xs)
    def sx(x):
        return pad + (0 if x_max == x_min else (x - x_min) / (x_max - x_min)) * (w - 2 * pad)
    def sy(y):
        y = max(0.0, min(y_max, y))
        return (h - pad) - (y / y_max) * (h - 2 * pad)
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf",
              "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#393b79", "#637939",
              "#8c6d31", "#843c39", "#7b4173"]
    parts = [f'<svg width="{w}" height="{h}" style="background:#fff;border:1px solid #ddd">']
    # axes + y=1 reference
    parts.append(f'<line x1="{pad}" y1="{sy(1.0)}" x2="{w-pad}" y2="{sy(1.0)}" stroke="#bbb" stroke-dasharray="4 3"/>')
    parts.append(f'<text x="{w-pad}" y="{sy(1.0)-4}" font-size="10" text-anchor="end" fill="#888">ratio=1 (no skill)</text>')
    parts.append(f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{h-pad}" stroke="#333"/>')
    parts.append(f'<line x1="{pad}" y1="{h-pad}" x2="{w-pad}" y2="{h-pad}" stroke="#333"/>')
    parts.append(f'<text x="{pad-6}" y="{sy(0)}" font-size="10" text-anchor="end">0</text>')
    parts.append(f'<text x="{pad-6}" y="{sy(y_max)+8}" font-size="10" text-anchor="end">{y_max:g}</text>')
    parts.append(f'<text x="{(w)/2}" y="{h-8}" font-size="11" text-anchor="middle">horizon h</text>')
    parts.append(f'<text x="{w/2}" y="16" font-size="13" text-anchor="middle" font-weight="bold">{html.escape(title)}</text>')
    for i, (name, pts) in enumerate(sorted(series.items())):
        col = colors[i % len(colors)]
        pts = sorted(p for p in pts if p[1] == p[1])  # drop NaN
        if not pts:
            continue
        d = " ".join(f"{'M' if k==0 else 'L'}{sx(x):.1f},{sy(y):.1f}" for k, (x, y) in enumerate(pts))
        parts.append(f'<path d="{d}" fill="none" stroke="{col}" stroke-width="2"/>')
        for x, y in pts:
            parts.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="2.5" fill="{col}"/>')
        ly = 30 + i * 14
        parts.append(f'<rect x="{w-pad-140}" y="{ly-8}" width="10" height="10" fill="{col}"/>')
        parts.append(f'<text x="{w-pad-126}" y="{ly}" font-size="10">{html.escape(name)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def render_html(report: dict, out_path: str) -> None:
    """report keys: meta, family_curves{family:[(h,median_ratio)]}, floor_table[rows],
    sparse_subset{block:{...}}, invariants[rows], families[rows], grouping{...}.
    Writes a self-contained HTML."""
    meta = report.get("meta", {})
    esc = html.escape
    P = ['<!doctype html><html><head><meta charset="utf-8">',
         '<title>Infra-B sweep engine self-test (EXP-44)</title>',
         '<style>body{font-family:-apple-system,Helvetica,Arial,sans-serif;margin:24px;color:#222;max-width:1100px}',
         'h1{font-size:22px}h2{font-size:16px;margin-top:28px;border-bottom:1px solid #eee;padding-bottom:4px}',
         'table{border-collapse:collapse;font-size:12px;margin:8px 0}td,th{border:1px solid #ccc;padding:4px 8px;text-align:right}',
         'th{background:#f5f5f5}.ok{color:#127a12;font-weight:bold}.bad{color:#c00;font-weight:bold}',
         '.flag{background:#fff3cd;color:#8a6d3b;font-weight:bold}.mono{font-family:ui-monospace,Menlo,monospace}',
         'td.l,th.l{text-align:left}.small{color:#888;font-size:11px}</style></head><body>']
    P.append(f"<h1>Infra-B weight-projection sweep engine — self-test</h1>")
    P.append(f'<p class="small">EXP-44 · trace: {esc(str(meta.get("manifest","")))} · '
             f'ticks sampled: {esc(str(meta.get("ticks","")))} · '
             f'generated: {esc(str(meta.get("generated","")))}</p>')

    # verdict banner
    verdict = report.get("verdict", "")
    cls = "ok" if verdict == "PASS" else ("flag" if verdict == "REVISE" else "bad")
    P.append(f'<p>Self-test gate: <span class="{cls}">{esc(verdict)}</span></p>')

    # invariants
    P.append("<h2>Pre-run gate — correctness invariants</h2><table>")
    P.append('<tr><th class="l">invariant</th><th class="l">gate</th><th class="l">result</th><th class="l">detail</th></tr>')
    for r in report.get("invariants", []):
        rc = "ok" if r["pass"] else ("flag" if r.get("gate") == "soft" else "bad")
        P.append(f'<tr><td class="l">{esc(r["name"])}</td><td class="l">{esc(r.get("gate",""))}</td>'
                 f'<td class="l {rc}">{"PASS" if r["pass"] else "FAIL"}</td>'
                 f'<td class="l mono small">{esc(str(r.get("detail","")))}</td></tr>')
    P.append("</table>")

    # families / reconstruction
    P.append("<h2>Predictor families — presence + reconstruction (err &lt;= 1e-5 rel)</h2><table>")
    P.append('<tr><th class="l">family</th><th class="l">coeff_source</th><th>order</th>'
             '<th>recon rel-err</th><th class="l">reconstructable</th></tr>')
    for r in report.get("families", []):
        rc = "ok" if r["reconstructable"] else "bad"
        P.append(f'<tr><td class="l">{esc(r["family"])}</td><td class="l">{esc(r["coeff_source"])}</td>'
                 f'<td>{esc(str(r["order"]))}</td><td class="mono">{r["recon_rel_err"]:.2e}</td>'
                 f'<td class="l {rc}">{"YES" if r["reconstructable"] else "NO"}</td></tr>')
    P.append("</table>")

    # family curves (one per family)
    P.append("<h2>weight_proj_ratio vs horizon — one curve per family</h2>")
    curves = {k: v for k, v in report.get("family_curves", {}).items()}
    P.append(_svg_curve(curves, "median weight_proj_ratio(h)  (per-block group, sampled)"))
    P.append('<p class="small">A ratio &gt; 1 / h* = 0 (predictor no better than the stale '
             'anchor) is a VALID SCIENTIFIC FINDING about bf16 RLVR weight geometry for '
             '#52-#56 to interpret through the sparsity lens — NOT an engine-acceptance '
             'failure. Engine acceptance = families reconstruct + fro-norm OK + no MOVING '
             'core block noise-dominated at h&gt;=5.</p>')

    # noise-floor gate table (CORRECTED differenced floor)
    P.append("<h2>bf16 DIFFERENCED-noise-floor gate (replaces on-box parity)</h2>")
    P.append('<p class="small">floor = bf16 quantization noise of the DIFFERENCE '
             'e = (&Sigma; c<sub>j</sub>&theta;<sub>j</sub>) &minus; &theta;<sub>now</sub> of two '
             'CORRELATED snapshots (per-element ULP-of-the-difference, propagated through the '
             'predictor coeffs) — NOT the &#124;&#124;&theta;&#124;&#124;-scaled STORAGE floor '
             '(the prior category error, which over-estimated the true floor by ~600&ndash;2200&times;). '
             'A HELD-CONSTANT tensor differences to EXACTLY 0.0 (empirical null). The '
             'true correlated floor is the zero-motion null (~0); the reported floor is an '
             'honest UPPER-BOUND (0.5-ULP-per-changed-element). DISCRIMINATOR: cumulative '
             'displacement scales as h<sup>p</sup>; p &gt;= 0.8 =&gt; DIRECTED drift (real signal, '
             'not rounding random-walk p~0.5). A moving block that is directed CLEARS the floor '
             'regardless of the per-element ULP multiple. manifest fro-norm cross-check tol = 1e-2 rel.</p>')
    P.append("<table><tr><th class='l'>block</th><th>h</th><th>floor(ub)</th><th>||disp||</th>"
             "<th>SNR(ub)</th><th>p</th><th>ratio</th><th class='l'>status</th></tr>")
    for r in report.get("floor_table", []):
        moves = r.get("moves", r["err_norm"] > 0.0)
        p = r.get("directed_p", float("nan"))
        if not moves:
            status = '<span class="small">zero-motion (unchanging; floor~0, signal~0)</span>'
            ratio_cell = "—"
        elif not r["bf16_unreliable"]:
            status = '<span class="ok">clears floor (directed signal)</span>'
            ratio_cell = f'{r["ratio"]:.4f}' if r["ratio"] == r["ratio"] else "nan"
        else:
            status = '<span class="flag">bf16-unreliable (not directed)</span>'
            ratio_cell = "—"
        snr_cell = f'{r["snr"]:.2f}' if r["snr"] == r["snr"] else "nan"
        p_cell = f'{p:.2f}' if p == p else "—"
        P.append(f'<tr><td class="l">{esc(r["block"])}</td><td>{r["h"]}</td>'
                 f'<td class="mono">{r["floor"]:.4e}</td><td class="mono">{r["err_norm"]:.4e}</td>'
                 f'<td class="mono">{snr_cell}</td><td class="mono">{p_cell}</td>'
                 f'<td class="mono">{ratio_cell}</td>'
                 f'<td class="l">{status}</td></tr>')
    P.append("</table>")

    # sparse-subset (PuLSE) characterization
    P.append("<h2>Sparse-subset (PuLSE) characterization</h2>")
    P.append('<p class="small">RLVR updates are intrinsically SPARSE; a dense L2 ratio can hide a '
             'sparse signal. Per block, the changed-element fraction (bf16 stored-bit inequality) '
             'and the ULP-multiple distribution of the motion: % &lt;=1 ULP is jitter, % &gt;=3 ULP '
             'is real directed motion resolved in the bf16 bits.</p>')
    P.append("<table><tr><th class='l'>block</th><th class='l'>pair</th><th>changed frac</th>"
             "<th>n_changed</th><th>median ULP</th><th>mean ULP</th><th>p90 ULP</th>"
             "<th>max ULP</th><th>% &lt;=1ULP</th><th>% &gt;=3ULP</th></tr>")
    for block, s in sorted(report.get("sparse_subset", {}).items()):
        u = s.get("ulp", {})
        P.append(f'<tr><td class="l">{esc(block)}</td><td class="l mono small">{esc(str(s.get("pair","")))}</td>'
                 f'<td class="mono">{s.get("changed_element_fraction",0.0)*100:.3f}%</td>'
                 f'<td class="mono">{s.get("n_changed",0)}</td>'
                 f'<td class="mono">{u.get("median_ulp_mult",0.0):.1f}</td>'
                 f'<td class="mono">{u.get("mean_ulp_mult",0.0):.2f}</td>'
                 f'<td class="mono">{u.get("p90_ulp_mult",0.0):.1f}</td>'
                 f'<td class="mono">{u.get("max_ulp_mult",0.0):.0f}</td>'
                 f'<td class="mono">{u.get("frac_le_1ulp",0.0)*100:.0f}%</td>'
                 f'<td class="mono">{u.get("frac_ge_3ulp",0.0)*100:.0f}%</td></tr>')
    P.append("</table>")

    # grouping integrity
    g = report.get("grouping", {})
    P.append("<h2>Grouping integrity</h2><table>")
    for k in ("n_matrices", "n_blocks", "n_layers", "matrix_partition_ok",
              "block_partition_ok", "layer_partition_ok"):
        v = g.get(k)
        cls = "" if not isinstance(v, bool) else ("ok" if v else "bad")
        P.append(f'<tr><td class="l">{esc(k)}</td><td class="l {cls}">{esc(str(v))}</td></tr>')
    P.append("</table>")
    P.append(f'<p class="small">block families: {esc(", ".join(g.get("block_families", [])))}</p>')

    # raw json for machine consumers (#45)
    P.append("<h2>Machine-readable self-test record</h2>")
    P.append(f'<pre class="mono small">{esc(json.dumps({k:v for k,v in report.items() if k!="family_curves"}, indent=2, default=str)[:8000])}</pre>')
    P.append("</body></html>")
    with open(out_path, "w") as f:
        f.write("".join(P))
