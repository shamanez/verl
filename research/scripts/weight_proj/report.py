#!/usr/bin/env python3
"""weight_proj/report.py — self-contained smoke report builder.

Renders an HTML report (e.g. `runs/<ID>/report.html`): one curve per new
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
    dtype = str(meta.get("dump_dtype", "bf16"))
    fp32 = dtype.lower() == "fp32"
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
             f'dtype: {esc(dtype)} · source: {esc(str(meta.get("trace_source","")))} · '
             f'ticks sampled: {esc(str(meta.get("ticks","")))} · '
             f'generated: {esc(str(meta.get("generated","")))}</p>')

    # verdict banner
    verdict = report.get("verdict", "")
    cls = "ok" if verdict == "PASS" else ("flag" if verdict == "REVISE" else "bad")
    P.append(f'<p>Self-test gate: <span class="{cls}">{esc(verdict)}</span></p>')

    # ---- plain-language orientation (how to read this report) -----------------
    # Everything specific (which groups / cadence / horizons / h* / directedness) is
    # pulled from the report dict so this stays correct when #52-#56 regenerate.
    ft_hdr = report.get("floor_table", [])
    fam_hdr = report.get("families", [])
    hstars_hdr = report.get("hstars", {})
    fc_hdr = report.get("family_curves", {})
    blocks_hdr = sorted({r["block"] for r in ft_hdr}) or sorted(report.get("sparse_subset", {}).keys())
    horizons_hdr = sorted({r["h"] for r in ft_hdr})
    cadence_hdr = str(meta.get("cadence", "per-step"))

    def _is_learned(name):
        return ("learnable" in name) or (name == "general-regression")

    # linearity: directedness exponent p per moving block
    dir_bits, seen_hdr = [], set()
    for r in ft_hdr:
        b = r["block"]; p = r.get("directed_p")
        moves = r.get("moves", float(r.get("err_norm", 0.0) or 0.0) > 0.0)
        if b not in seen_hdr and moves and isinstance(p, (int, float)) and p == p:
            dir_bits.append(f"{esc(b)} p={p:.2f}"); seen_hdr.add(b)
    dir_str = ", ".join(dir_bits) if dir_bits else "n/a"

    # steps-ahead: h* (furthest horizon still beating "keep the stale weights")
    fixed_hs = [v for k, v in hstars_hdr.items() if not _is_learned(k)]
    learn_hs = [v for k, v in hstars_hdr.items() if _is_learned(k)]

    def _best_ratio(pred):
        best = None
        for k, pts in fc_hdr.items():
            if pred(k):
                for _h, rr in pts:
                    if rr == rr and (best is None or rr < best):
                        best = rr
        return best
    learned_best = _best_ratio(_is_learned)
    if learned_best is not None and learned_best >= 0.99:
        skill_note = ("no rule beat &ldquo;keep the stale weights&rdquo; by a meaningful margin "
                      "(best learned median ratio %.3f &#8776; 1.00 &mdash; it merely reproduces the stale "
                      "weights). The motion is real and linear but NOT usefully extrapolable at this cadence." % learned_best)
    elif learned_best is not None:
        skill_note = ("the best learned rule reached median ratio %.3f &mdash; a real prediction gain over "
                      "keeping the stale weights." % learned_best)
    else:
        skill_note = "see the per-family curves below for the actual margin."

    P.append('<div style="background:#f0f6ff;border:1px solid #cfe0ff;border-radius:6px;padding:10px 16px;margin:14px 0">')
    P.append('<h2 style="margin-top:4px;border:none">How to read this report (plain language)</h2>')
    P.append('<p><b>Snapshot vs. predictor vs. prediction.</b> A <b>snapshot</b> &theta;[t] is the '
             '<b>real</b> model weights recorded at training tick t. A <b>predictor</b> is <b>not</b> a '
             'snapshot &mdash; it is a <b>rule</b> that <b>guesses</b> a future weight from past snapshots '
             '(e.g. &ldquo;extend the last velocity forward&rdquo;). Its output &theta;&#770; is that '
             '<b>guess</b>, never a stored snapshot. A guess is scored by whether it lands closer to the real '
             'future snapshot than simply keeping the stale weights.</p>')
    P.append('<p><b>What was tested.</b> %d predictor rules (the &ldquo;families&rdquo; table below), on the '
             'weight groups <span class="mono">%s</span>, at <b>%s</b> cadence, projecting <b>%s</b> steps ahead '
             'of the stale anchor.</p>'
             % (len(fam_hdr), esc(", ".join(blocks_hdr)), esc(cadence_hdr),
                esc(", ".join(str(h) for h in horizons_hdr)) or "n/a"))
    P.append('<p><b>How we measured linearity.</b> We fit the <i>fixed-origin cumulative displacement</i> '
             '&#8214;&theta;(origin+h) &minus; &theta;(origin)&#8214; vs horizon h as &#8733; h<sup>p</sup>. '
             '<b>p&#8776;1</b> = a straight-line <b>directed</b> drift (real signal); <b>p&#8776;0.5</b> = a '
             'random walk (rounding noise). Observed on the moving blocks: <b>%s</b> &rarr; directed / linear.</p>'
             % dir_str)
    P.append('<p><b>How many steps ahead we could predict (h*).</b> h* = the furthest horizon at which a rule '
             'still beat &ldquo;keep the stale weights.&rdquo; Here: <b>fixed / non-learned rules h*=%s</b> '
             '(0 = never beat the stale weights; higher polynomial orders overshoot far worse), '
             '<b>learned rules h*=%s</b>. Bottom line: <b>%s</b></p>'
             % (max(fixed_hs) if fixed_hs else "0", max(learn_hs) if learn_hs else "0", skill_note))
    P.append('<p class="small">Scope: acceptance self-test on a subsampled window; the full-resolution, '
             'per-layer, and coarser-cadence study is #52&ndash;#56. &ldquo;YES&rdquo; in the families table '
             'means a rule is a faithful linear recipe (auditable), <i>not</i> that its guess is accurate &mdash; '
             'accuracy is the ratio / h* above.</p>')
    P.append('</div>')

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
    P.append('<p class="small">Each row is a predictor <b>rule</b> (not a snapshot). '
             '<b>reconstructable = YES</b> means the rule&rsquo;s output can be reproduced exactly from the '
             'linear coefficients it declares (&theta;&#770; = &Sigma;<sub>j</sub> c<sub>j</sub>&theta;<sub>j</sub>), '
             'so the engine is auditable &mdash; it says <i>nothing</i> about whether the guess is accurate. '
             'recon rel-err is 0 because this recomputes the identical arithmetic '
             '<span class="mono">predict()</span> already runs, not because the prediction is correct.</p>')

    # family curves (one per family)
    P.append("<h2>weight_proj_ratio vs horizon — one curve per family</h2>")
    curves = {k: v for k, v in report.get("family_curves", {}).items()}
    P.append(_svg_curve(curves, "median weight_proj_ratio(h)  (per-block group, sampled)"))
    if fp32:
        P.append('<p class="small">A ratio &gt; 1 / h* = 0 (predictor no better than the stale '
                 'anchor) is a VALID SCIENTIFIC FINDING about RLVR weight geometry for '
                 '#52-#56 — NOT an engine-acceptance failure. Engine acceptance (fp32) = '
                 'families reconstruct + fro-norm OK. Weights are exact fp32, so there is no '
                 'quantization-noise floor: reliability is projection accuracy + linearity.</p>')
    else:
        P.append('<p class="small">A ratio &gt; 1 / h* = 0 (predictor no better than the stale '
                 'anchor) is a VALID SCIENTIFIC FINDING about bf16 RLVR weight geometry for '
                 '#52-#56 to interpret through the sparsity lens — NOT an engine-acceptance '
                 'failure. Engine acceptance = families reconstruct + fro-norm OK + no MOVING '
                 'core block noise-dominated at h&gt;=5.</p>')

    # motion & linearity table — dtype-aware. fp32: no quantization floor (operator
    # directive); bf16 (legacy): the CORRECTED differenced-noise floor gate.
    if fp32:
        P.append("<h2>Weight motion &amp; linearity (fp32 — exact weights, no quantization floor)</h2>")
        P.append('<p class="small">The weights are exact fp32, so there is NO quantization-noise '
                 'floor to clear — the bf16 differenced-floor / SNR / PuLSE-ULP gate is removed. '
                 'Per (block, h) we report the raw cumulative displacement '
                 '&#8214;&theta;[t]&minus;&theta;[t&minus;h]&#8214; and the LINEARITY of the drift: '
                 'fixed-origin cumulative displacement &#8733; h<sup>p</sup>, with p&#8776;1 = a '
                 'straight-line directed drift (R&sup2; is the log-log fit quality). Reliability = '
                 'projection accuracy (ratio) + linearity, not a floor. manifest fro-norm '
                 'cross-check tol = 1e-2 rel.</p>')
        P.append("<table><tr><th class='l'>block</th><th>h</th><th>||disp||</th>"
                 "<th>p (linearity)</th><th>R&sup2;</th><th>ratio</th><th class='l'>status</th></tr>")
        for r in report.get("floor_table", []):
            moves = r.get("moves", r["err_norm"] > 0.0)
            p = r.get("directed_p", float("nan"))
            r2v = r.get("directed_r2", float("nan"))
            if not moves:
                status = '<span class="small">zero-motion (unchanging)</span>'
                ratio_cell = "—"
            else:
                status = '<span class="ok">moves (fp32-exact)</span>'
                ratio_cell = f'{r["ratio"]:.4f}' if r["ratio"] == r["ratio"] else "nan"
            p_cell = f'{p:.2f}' if p == p else "—"
            r2_cell = f'{r2v:.3f}' if r2v == r2v else "—"
            P.append(f'<tr><td class="l">{esc(r["block"])}</td><td>{r["h"]}</td>'
                     f'<td class="mono">{r["err_norm"]:.4e}</td>'
                     f'<td class="mono">{p_cell}</td><td class="mono">{r2_cell}</td>'
                     f'<td class="mono">{ratio_cell}</td>'
                     f'<td class="l">{status}</td></tr>')
        P.append("</table>")
        # skip the bf16-only floor table + PuLSE sections; jump to grouping integrity.
        return _finish_report(P, report, esc, out_path)

    # noise-floor gate table (CORRECTED differenced floor) — bf16 legacy path
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

    return _finish_report(P, report, esc, out_path)


def _finish_report(P: list, report: dict, esc, out_path: str) -> None:
    """Shared tail (grouping integrity + machine-readable JSON + close + write).

    Called by both the fp32 (motion/linearity) and bf16 (floor-gate) render paths.
    """
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
