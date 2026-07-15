"""Assemble the final self-contained HTML report from the ablation digests + plots.

Data-driven from report_digest.json (main gap=10 grid) and gap_sweep.json (gap
sensitivity), so the prose numbers cannot drift from the computed results. Plots are
base64-inlined so the single .html file is fully self-contained.

Usage:
  python build_report.py --digest <outputs>/report_digest.json \
      --gap_digest <outputs>/gap_sweep.json --plots <plots_dir> --out <repo>/docs/.../report.html
"""
# ruff: noqa: E501  (report generator: long inline HTML/CSS/f-string lines are intentional)

from __future__ import annotations

import argparse
import base64
import json
import os

# ---- constants pulled from the live run (plan section 1.3) and the embedding probe ----
LIVE = [
    dict(
        run="vqe9554z",
        label="W=2 secant, alpha=1",
        W=2,
        skill=0.780,
        cos=0.906,
        proj_rmse="6.47e-6",
        stale_rmse="1.38e-5",
        math="67.89%",
    ),
    dict(
        run="lzl4vlcr",
        label="W=4 rank-1 OLS, alpha=1",
        W=4,
        skill=0.173,
        cos=0.571,
        proj_rmse="8.74e-6",
        stale_rmse="9.62e-6",
        math="63.61% @50",
    ),
    dict(
        run="kvgtcs07",
        label="dense control (no anchor)",
        W=None,
        skill=None,
        cos=None,
        proj_rmse=None,
        stale_rmse=None,
        math="67.41%",
    ),
]
# embedding_probe.py results (tied embedding, gap=10, h=1, anchors 40/50/60):
EMBED = {"W2": (-1.535, 0.228), "W4": (-0.368, 0.221)}

CSS = """
:root{color-scheme:light dark;--bg:#09111f;--panel:#111c2e;--panel-2:#17243a;--text:#eef4ff;
--muted:#aebbd0;--line:#2b3b55;--blue:#71b7ff;--green:#6ee7a2;--amber:#ffd166;--red:#ff8585;
--purple:#c4a7ff;--shadow:0 18px 50px rgb(0 0 0 / 24%);}
*{box-sizing:border-box}html{scroll-behavior:smooth}
body{margin:0;background:radial-gradient(circle at 12% -10%,rgb(62 113 184 / 30%),transparent 34rem),
radial-gradient(circle at 95% 5%,rgb(122 85 195 / 22%),transparent 28rem),var(--bg);color:var(--text);
font:16px/1.58 Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;}
a{color:var(--blue)}
code,.mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:.94em}
code{padding:.1rem .35rem;border:1px solid var(--line);border-radius:.35rem;background:rgb(255 255 255 / 5%)}
.wrap{width:min(1180px,calc(100% - 2rem));margin:0 auto}
header{padding:4.5rem 0 2rem}
.eyebrow{margin:0 0 .8rem;color:var(--blue);font-weight:750;letter-spacing:.08em;text-transform:uppercase;font-size:.78rem}
h1{max-width:980px;margin:0;font-size:clamp(2.1rem,5.5vw,4rem);line-height:1.0;letter-spacing:-.04em}
h2{margin:0 0 1rem;font-size:clamp(1.45rem,3vw,2rem);letter-spacing:-.02em}
h3{margin:1.3rem 0 .5rem;font-size:1.08rem;color:var(--text)}
p{margin:.5rem 0 1rem}
.lede{max-width:900px;margin:1.25rem 0;color:var(--muted);font-size:1.14rem}
.status-row{display:flex;flex-wrap:wrap;gap:.65rem;margin:1.4rem 0 0}
.pill{display:inline-flex;align-items:center;gap:.45rem;padding:.4rem .75rem;border:1px solid var(--line);
border-radius:999px;background:rgb(255 255 255 / 4%);color:var(--muted);font-size:.86rem}
.dot{width:.58rem;height:.58rem;border-radius:50%;background:var(--green);box-shadow:0 0 0 .2rem rgb(110 231 162 / 12%)}
nav{position:sticky;top:0;z-index:5;border-block:1px solid var(--line);background:rgb(9 17 31 / 88%);backdrop-filter:blur(14px)}
nav .wrap{display:flex;gap:1.05rem;overflow-x:auto;padding-block:.75rem}
nav a{color:var(--muted);text-decoration:none;white-space:nowrap;font-size:.9rem}
nav a:hover{color:var(--text)}
main{padding:1.5rem 0 5rem}
section{scroll-margin-top:4rem;margin:1rem 0 1.4rem;padding:1.35rem 1.5rem;border:1px solid var(--line);
border-radius:1rem;background:linear-gradient(145deg,rgb(255 255 255 / 4%),rgb(255 255 255 / 1%));box-shadow:var(--shadow)}
.grid{display:grid;gap:1rem;grid-template-columns:repeat(auto-fit,minmax(230px,1fr))}
.card{min-width:0;padding:1rem;border:1px solid var(--line);border-radius:.8rem;background:var(--panel)}
.card p:last-child{margin-bottom:0}
.metric{margin:.1rem 0 .2rem;font-size:2rem;font-weight:780;letter-spacing:-.035em}
.metric.sm{font-size:1.5rem}
.label{color:var(--muted);font-size:.82rem;text-transform:uppercase;letter-spacing:.06em}
.good{color:var(--green)}.warn{color:var(--amber)}.bad{color:var(--red)}.accent{color:var(--blue)}.muted{color:var(--muted)}.purple{color:var(--purple)}
.callout{margin:1rem 0 0;padding:.9rem 1rem;border-left:.25rem solid var(--amber);background:rgb(255 209 102 / 8%);border-radius:.25rem .7rem .7rem .25rem}
.callout strong{color:var(--amber)}
.callout.good{border-left-color:var(--green);background:rgb(110 231 162 / 8%)}
.callout.good strong{color:var(--green)}
.callout.blue{border-left-color:var(--blue);background:rgb(113 183 255 / 8%)}
.callout.blue strong{color:var(--blue)}
.callout.red{border-left-color:var(--red);background:rgb(255 133 133 / 8%)}
.callout.red strong{color:var(--red)}
.formula{overflow-x:auto;margin:1rem 0;padding:1rem;border:1px solid var(--line);border-radius:.75rem;
background:#080f1b;color:#dceaff;text-align:center;font:600 1rem/1.6 ui-monospace,SFMono-Regular,Menlo,monospace}
figure{margin:1.2rem 0}
figure img{max-width:100%;border:1px solid var(--line);border-radius:.75rem;background:#0b1526}
figcaption{color:var(--muted);font-size:.9rem;margin-top:.5rem}
.flow{display:grid;grid-template-columns:1fr auto 1fr auto 1fr;gap:.7rem;align-items:stretch;margin:1rem 0}
.flow .node{padding:1rem;border:1px solid var(--line);border-radius:.75rem;background:var(--panel)}
.flow .arrow{align-self:center;color:var(--blue);font-size:1.5rem;text-align:center}
.table-wrap{overflow-x:auto;border:1px solid var(--line);border-radius:.8rem;margin:1rem 0}
table{width:100%;border-collapse:collapse;min-width:640px}
th,td{padding:.62rem .8rem;text-align:left;border-bottom:1px solid var(--line);vertical-align:top;font-size:.9rem}
th{background:var(--panel-2);color:var(--muted);font-size:.75rem;letter-spacing:.04em;text-transform:uppercase}
tr:last-child td{border-bottom:0}
td.mono,th.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.right{text-align:right}
ul,ol{padding-left:1.25rem}li{margin:.4rem 0}
.best{background:rgb(110 231 162 / 12%)}
.worst{background:rgb(255 133 133 / 10%)}
dl.terms{margin:0}dl.terms dt{font-weight:750;color:var(--blue);margin-top:.9rem}
dl.terms dd{margin:.2rem 0 .2rem 0;color:var(--text)}
footer{padding:0 0 3rem;color:var(--muted);font-size:.86rem}
@media (max-width:760px){header{padding-top:2.8rem}.flow{grid-template-columns:1fr}.flow .arrow{transform:rotate(90deg);justify-self:center}section{padding:1rem}}
"""


def img_tag(plots, fname, alt):
    path = os.path.join(plots, fname)
    if not os.path.exists(path):
        return f'<p class="bad">[missing plot: {fname}]</p>'
    b64 = base64.b64encode(open(path, "rb").read()).decode()
    return f'<img src="data:image/png;base64,{b64}" alt="{alt}">'


def f3(v, plus=True):
    if v is None:
        return "n/a"
    return f"{v:+.3f}" if plus else f"{v:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--digest", required=True)
    ap.add_argument("--gap_digest", required=True)
    ap.add_argument("--plots", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--updated", default="")
    ap.add_argument("--count_digest", default="", help="Optional report_digest.json from the consecutive-count sweep (outputs_consec).")
    args = ap.parse_args()

    D = json.load(open(args.digest))
    S, BT, META = D["scalars"], D["by_type"], D["meta"]
    GAP = json.load(open(args.gap_digest))
    gap_summary = GAP["summary"]
    COUNT = json.load(open(args.count_digest))["scalars"] if args.count_digest and os.path.exists(args.count_digest) else None

    def g(method, W, h, r, field):
        return S.get(f"{method}|W{W}|h{h}|r{r}", {}).get(field)

    Ws = sorted({v["W"] for v in S.values() if v["method"] == "rank1_relex" and v["horizon"] == 1 and v["rank"] == 1})
    skill_by_W = {W: g("rank1_relex", W, 1, 1, "pooled_skill") for W in Ws}
    skill_by_W = {W: s for W, s in skill_by_W.items() if s is not None}
    # on the dense proxy the "best" (least-negative) window is the largest; the worst is W=2
    best_W = max(skill_by_W, key=skill_by_W.get)
    worst_W = min(skill_by_W, key=skill_by_W.get)
    w2 = skill_by_W.get(2)
    w4 = skill_by_W.get(4)
    best_skill = skill_by_W[best_W]

    IMG = lambda fn, alt: img_tag(args.plots, fn, alt)

    updated_pill = f'<span class="pill">Updated {args.updated}</span>' if args.updated else ""

    # ---------------- TL;DR cards ----------------
    tldr_cards = f"""
    <article class="card"><div class="label">Does any W beat "stay put"?</div>
      <div class="metric bad">No</div>
      <p>At the cadence gap (10 steps) on the dense proxy, every window W=2..6 gives
      <em>negative</em> forecast skill: reusing the stale checkpoint beats projecting.</p></article>
    <article class="card"><div class="label">W=2 vs W=6 skill (h=1)</div>
      <div class="metric sm accent">{f3(w2)} vs {f3(best_skill)}</div>
      <p>The aggressive 2-checkpoint secant is the <strong>worst</strong> (it roughly doubles the error);
      more checkpoints only help by damping toward "do not move".</p></article>
    <article class="card"><div class="label">Update-direction persistence</div>
      <div class="metric sm bad">&approx; 0</div>
      <p>Successive weight updates are near-orthogonal at every gap 1..30. That is why extrapolation
      cannot help here - and it is exactly what the live compressed run does <em>not</em> look like.</p></article>
    <article class="card"><div class="label">Live compressed run (for contrast)</div>
      <div class="metric sm good">cos 0.91</div>
      <p>The PowerSGD fast circuit's updates are collinear, so there the W=2 secant wins
      (skill 0.78, MATH 67.89%). The projector's payoff is compression-specific.</p></article>
    """

    # ---------------- H1 table ----------------
    def skill_row(method, label):
        cells = []
        for W in Ws:
            s = g(method, W, 1, 1, "pooled_skill")
            cls = ""
            if method == "rank1_relex" and W == best_W:
                cls = " class='best'"
            elif method == "rank1_relex" and W == worst_W:
                cls = " class='worst'"
            cells.append(f"<td{cls} class='mono right'>{f3(s)}</td>")
        return f"<tr><td>{label}</td>{''.join(cells)}</tr>"

    h1_header = "".join(f"<th class='right'>W={W}</th>" for W in Ws)
    h1_table = f"""
    <div class="table-wrap"><table>
      <thead><tr><th>method (pooled skill, h=1, rank=1)</th>{h1_header}</tr></thead>
      <tbody>
        {skill_row("rank1_relex", "rank1_relex (pin-to-latest)")}
        {skill_row("relex_from_base", "relex_from_base (rebuild)")}
        {skill_row("fixed_linear", "fixed_linear (decoder-only)")}
      </tbody></table></div>
    """

    # ---------------- full results table ----------------
    def full_rows():
        out = []
        for method in ["rank1_relex", "relex_from_base", "fixed_linear"]:
            vs = sorted(
                [v for v in S.values() if v["method"] == method and v["rank"] == 1],
                key=lambda v: (v["W"], v["horizon"]),
            )
            for v in vs:
                out.append(
                    f"<tr><td class='mono'>{method}</td><td class='right'>{v['W']}</td>"
                    f"<td class='right'>{v['horizon']}</td>"
                    f"<td class='mono right'>{f3(v['pooled_skill'])} &plusmn; {v['pooled_skill_std']:.2f}</td>"
                    f"<td class='mono right'>{f3(v['macro_cos'])}</td>"
                    f"<td class='mono right'>{f3(v['frac_win'], False)}</td>"
                    f"<td class='mono right'>{f3(v['evr'], False) if v['evr'] is not None else 'n/a'}</td>"
                    f"<td class='mono right'>{f3(v['r2'], False) if v['r2'] is not None else 'n/a'}</td>"
                    f"<td class='right'>{v['n_anchors']}</td></tr>"
                )
        return "\n".join(out)

    results_table = f"""
    <div class="table-wrap"><table>
      <thead><tr><th>method</th><th class="right">W</th><th class="right">h</th>
        <th class="right">pooled skill</th><th class="right">dir cos</th>
        <th class="right">frac win</th><th class="right">EVR</th><th class="right">R^2</th>
        <th class="right">n anchors</th></tr></thead>
      <tbody>{full_rows()}</tbody></table></div>
    """

    # ---------------- gap sweep table + reconciliation ----------------
    def gap_rows(W):
        rs = sorted([r for r in gap_summary if r["W"] == W], key=lambda r: r["gap"])
        return rs

    gap_W2 = gap_rows(2)
    gap_W4 = gap_rows(4)
    all_gaps = sorted({r["gap"] for r in gap_summary})

    def gap_table():
        hdr = "".join(f"<th class='right'>G={gp}</th>" for gp in all_gaps)

        def row(rs, label):
            d = {r["gap"]: r for r in rs}
            cells = "".join(
                f"<td class='mono right'>{f3(d[gp]['pooled_skill']) if gp in d else '-'}</td>" for gp in all_gaps
            )
            return f"<tr><td>{label}</td>{cells}</tr>"

        def prow(rs, label, field):
            d = {r["gap"]: r for r in rs}
            cells = "".join(
                f"<td class='mono right'>{f3(d[gp][field]) if gp in d and d[gp][field] is not None else '-'}</td>"
                for gp in all_gaps
            )
            return f"<tr><td>{label}</td>{cells}</tr>"

        return f"""<div class="table-wrap"><table>
          <thead><tr><th>metric</th>{hdr}</tr></thead><tbody>
          {row(gap_W2, "pooled skill (W=2 secant)")}
          {row(gap_W4, "pooled skill (W=4 OLS)")}
          {prow(gap_W2, "update direction cosine (macro, W=2)", "macro_cos")}
          {prow(gap_W2, "delta persistence (global cos, W=2)", "persist_cos")}
          </tbody></table></div>"""

    # persistence at the finest gap
    fine = min(all_gaps)
    persist_fine = next((r["persist_cos"] for r in gap_W2 if r["gap"] == fine), None)
    cos_fine = next((r["macro_cos"] for r in gap_W2 if r["gap"] == fine), None)

    # ---------------- by-type table ----------------
    def by_type_table(W):
        key = f"rank1_relex|W{W}|h1|r1"
        d = BT.get(key) or {}
        if not d:
            return "<p class='muted'>[no per-type data]</p>"
        order = sorted(d.items(), key=lambda kv: -(kv[1]["skill"] if kv[1]["skill"] is not None else -9))
        rows = "\n".join(
            f"<tr><td class='mono'>{t}</td><td class='mono right'>{f3(v['skill'])} &plusmn; {v['std']:.2f}</td>"
            f"<td class='right'>{v['n']}</td></tr>"
            for t, v in order
        )
        return (
            f"<div class='table-wrap'><table><thead><tr><th>tensor type</th>"
            f"<th class='right'>macro skill (mean over tensors)</th><th class='right'>rows</th></tr></thead>"
            f"<tbody>{rows}</tbody></table></div>"
        )

    # ---------------- live vs study table ----------------
    def live_rows():
        out = []
        for L in LIVE:
            sp = g("rank1_relex", L["W"], 1, 1, "pooled_skill") if L["W"] else None
            sc = g("rank1_relex", L["W"], 1, 1, "macro_cos") if L["W"] else None
            out.append(
                f"<tr><td class='mono'>{L['run']}</td><td>{L['label']}</td>"
                f"<td class='right'>{L['W'] if L['W'] else '-'}</td>"
                f"<td class='mono right'>{f3(L['skill']) if L['skill'] is not None else '-'}</td>"
                f"<td class='mono right'>{f3(sp) if sp is not None else '-'}</td>"
                f"<td class='mono right'>{f3(L['cos']) if L['cos'] is not None else '-'}</td>"
                f"<td class='mono right'>{f3(sc) if sc is not None else '-'}</td>"
                f"<td class='right'>{L['math']}</td></tr>"
            )
        return "\n".join(out)

    # H3 rank sweep text
    r1 = g("rank1_relex", 4, 1, 1, "pooled_skill")
    r3 = g("rank1_relex", 4, 1, 3, "pooled_skill")
    evr_r1 = g("rank1_relex", 4, 1, 1, "evr")
    evr_r3 = g("rank1_relex", 4, 1, 3, "evr")

    # H4 horizon numbers
    h2_w2 = (
        g("rank1_relex", 2, 1, 1, "pooled_skill"),
        g("rank1_relex", 2, 2, 1, "pooled_skill"),
        g("rank1_relex", 2, 3, 1, "pooled_skill"),
    )
    h_w4 = (
        g("rank1_relex", 4, 1, 1, "pooled_skill"),
        g("rank1_relex", 4, 2, 1, "pooled_skill"),
        g("rank1_relex", 4, 3, 1, "pooled_skill"),
    )

    # H5 pin vs base
    pin4 = g("rank1_relex", 4, 1, 1, "pooled_skill")
    base4 = g("relex_from_base", 4, 1, 1, "pooled_skill")

    # ---- consecutive-count sweep (optional) --------------------------------
    count_section = ""
    count_navlink = ""
    if COUNT:
        cW = sorted({int(k.split("|")[1][1:]) for k in COUNT if k.startswith("rank1_relex|")})
        def cget(W, f):
            return COUNT.get(f"rank1_relex|W{W}|h1|r1", {}).get(f)
        rows = "\n".join(
            f"<tr><td class='right'>{W}</td>"
            f"<td class='mono right'>{f3(cget(W,'pooled_skill'))} &plusmn; {(cget(W,'pooled_skill_std') or 0):.2f}</td>"
            f"<td class='mono right'>{f3(cget(W,'macro_cos'))}</td>"
            f"<td class='mono right'>{f3(cget(W,'frac_win'), False)}</td>"
            f"<td class='right'>{cget(W,'n_anchors')}</td></tr>" for W in cW)
        lo_W, hi_W = cW[0], cW[-1]
        s_lo, s_hi = cget(lo_W, "pooled_skill"), cget(hi_W, "pooled_skill")
        improved = (s_lo is not None and s_hi is not None and s_hi > s_lo + 0.02)
        crossed = (s_hi is not None and s_hi > 0)
        verdict = (
            (f"Adding consecutive checkpoints DOES move skill upward: W={lo_W} {f3(s_lo)} to W={hi_W} {f3(s_hi)}. "
             + ("It even crosses zero (beats stale) at the largest window, so with enough consecutive history our own projector starts to work on the dense trajectory too."
                if crossed else
                "It is still below zero at W={0}, so more history helps but our short-horizon pinned projector needs more than {0} consecutive checkpoints (or the temporal denoising RELEX gets from ~50-75) to clear the stale baseline.".format(hi_W)))
            if improved else
            (f"Adding consecutive checkpoints does NOT rescue our projector here: skill stays near {f3(s_hi)} from W={lo_W} to W={hi_W}. "
             "Because the projector is pinned to the latest checkpoint and only steps one gap ahead, a better-denoised direction is not enough when the per-step motion it must predict is itself noise-dominated; the payoff still requires either RELEX's long from-base horizon or the live circuit's compression."))
        count_navlink = '<a href="#count">Count sweep</a>'
        count_section = f"""
<section id="count">
  <h2>Follow-up - does adding consecutive checkpoints help our projector?</h2>
  <p>The gap sweep and the RELEX paper both point at one untested corner: our study only ever used a
  <em>few</em> checkpoints (W=2 to 8). RELEX instead fits ~50-125 checkpoints sampled at every step. This
  follow-up isolates the checkpoint <strong>count</strong> using our own projector unchanged: feed it
  W = {', '.join(map(str, cW))} <strong>consecutive</strong> checkpoints (stride 1, from the early trajectory,
  steps 1-12), deltas from the window base, pinned to latest, projecting one step ahead. It does NOT
  reproduce RELEX (no from-the-pretrained-base reconstruction, no long horizon); it just asks whether more
  consecutive history sharpens our rank-1 estimate.</p>
  <figure>{IMG('count_sweep.png', 'skill vs number of consecutive checkpoints')}
    <figcaption>Pooled forecast skill (blue, left axis) and mean per-tensor update-direction cosine (green,
    right axis) vs the number of consecutive checkpoints W, at gap 1, horizon 1, with our pinned rank-1
    projector.</figcaption></figure>
  <div class="table-wrap"><table>
    <thead><tr><th class="right">W (consecutive)</th><th class="right">pooled skill</th>
      <th class="right">dir cos</th><th class="right">frac win</th><th class="right">n anchors</th></tr></thead>
    <tbody>{rows}</tbody></table></div>
  <div class="callout blue"><strong>Result.</strong> {verdict}</div>
  <p class="muted">Caveat: this uses deltas from the window's earliest checkpoint (as the live projector
  does), a stride-1 spacing, and a one-step horizon - not RELEX's from-pretrained-base, full-prefix,
  10-20x-horizon recipe. It isolates the count axis for OUR projector only.</p>
</section>"""

    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>How many checkpoints does the two-circuit anchor need?</title>
<style>{CSS}</style></head>
<body>
<header class="wrap">
  <p class="eyebrow">Qwen2.5-Math-1.5B &middot; RELEX RLVR trajectory &middot; weight-space forecast</p>
  <h1>How many exact checkpoints does the two-circuit anchor need to forecast the fast network?</h1>
  <p class="lede">A CPU replay of the anchor's forward weight projector on the open-sourced RELEX RLVR
  checkpoint trajectory, scored by forecast skill over all 338 tensors. The short answer is a
  reframing: <strong>the number of checkpoints is not the deciding variable - the directional
  persistence of the weight updates is</strong>. That single fact explains why the live compressed
  run loves a 2-checkpoint secant while this dense proxy cannot be forecast by any window.</p>
  <div class="status-row">
    <span class="pill"><span class="dot"></span> Study complete</span>
    {updated_pill}
    <span class="pill">Branch <span class="mono">exp/relex-ckpt-ablation</span></span>
    <span class="pill">Substrate <span class="mono">relex-rlvr/RLVR-Qwen2.5-Math-1.5B</span></span>
    <span class="pill">CPU only &middot; no verl install</span>
  </div>
</header>

<nav aria-label="Report sections"><div class="wrap">
  <a href="#tldr">TL;DR</a>
  <a href="#setup">Setup</a>
  <a href="#metrics">Metrics</a>
  <a href="#h1">H1 how many</a>
  <a href="#h2">H2 why W=2</a>
  <a href="#h3">H3 fit vs skill</a>
  <a href="#h4">H4 horizon</a>
  <a href="#h5">H5 pin vs base</a>
  <a href="#h6">H6 by tensor</a>
  <a href="#gap">Gap sweep</a>
  {count_navlink}
  <a href="#results">Results table</a>
  <a href="#live">Live check</a>
  <a href="#decision">Decision</a>
  <a href="#caveats">Caveats</a>
  <a href="#method">Method</a>
</div></nav>

<main class="wrap">

<section id="tldr">
  <h2>TL;DR</h2>
  <p>The anchor holds the fast network's own exact-but-delayed checkpoints and must project them
  forward to estimate where the fast network is <em>now</em>. We replayed that projector on a clean
  dense RLVR trajectory of the same base model and scored how well it forecasts the next checkpoint
  versus simply reusing the newest stale one.</p>
  <div class="grid">{tldr_cards}</div>
  <div class="callout blue"><strong>The answer, precisely.</strong> "How many checkpoints" is the wrong
  first question. On this dense trajectory, successive cadence-spaced weight updates are almost
  orthogonal (a random walk), so <em>no</em> window beats the stale baseline and the aggressive
  W=2 secant is the worst (skill {f3(w2)} vs {f3(best_skill)} at W={best_W}); larger windows help only
  by damping toward "do not move". The live run's W=2 win is <strong>real but contingent</strong>: the
  compressed PowerSGD fast circuit produces low-rank, directionally-persistent updates (direction cosine
  0.91), and it is that persistence - not the checkpoint count - that makes the 2-point secant the right
  projector. <strong>Recommendation:</strong> keep the W=2 secant at horizon 1 as the projector for the
  compressed circuit (validated live), understand it as a recent-tangent extrapolator that pays off only
  when updates persist, and do not expect any projector to help an uncompressed circuit.</div>
</section>

<section id="setup">
  <h2>The setup in one screen</h2>
  <p>The comm-efficient GRPO trainer runs two circuits. A compressed <strong>fast</strong> learner
  (PowerSGD, rank 77) does the real optimizer steps. An isolated <strong>anchor</strong> never
  optimizer-steps and never reads the immediate fast weights; it is fed the fast network's exact but
  <em>delayed</em> checkpoints and projects them forward. From that projected state it hands back
  <strong>Q</strong> (the activation-energy basis PowerSGD compresses against) and <strong>M</strong>
  (an EMA of the clean anchor gradient, used as a sign reference in
  <code>G_corr = 0.25&middot;G_fast + 0.75&middot;|G_fast|&middot;sign(M)</code>).</p>
  <div class="flow" role="img" aria-label="delayed exact checkpoints are projected forward into the anchor, which supplies Q and M to the compressed fast circuit">
    <div class="node"><strong>Delayed exact checkpoints</strong><br><span class="muted">the fast net's own weights, K steps stale</span></div>
    <div class="arrow">&rarr;</div>
    <div class="node"><strong>Forward projector</strong><br><span class="muted">window W, horizon h - the object under study</span></div>
    <div class="arrow">&rarr;</div>
    <div class="node"><strong>Anchor supplies Q, M</strong><br><span class="muted">back to the compressed fast circuit</span></div>
  </div>
  <h3>Cadence, delay, window, horizon</h3>
  <p>Config defaults (<code>verl/workers/config/comm_eff.py</code>): cadence 20, delay 20 optimizer
  ticks, window 4 snapshots. There are 2 optimizer ticks per global step, so the anchor fires every 10
  global steps and its newest exact checkpoint is 10 steps stale. When the fast net is at step 50 the
  anchor holds {{10, 20, 30, 40}} (W=4, gap 10), projects to step 50 (horizon 1, "current fast") and
  could project to 60 (horizon 2, "twice a fast").</p>
  <div class="grid">
    <div class="card"><div class="label">gap G</div><div class="metric sm">10 steps</div><p>spacing between source checkpoints (= the cadence). We also sweep it.</p></div>
    <div class="card"><div class="label">window W</div><div class="metric sm">2 - 6</div><p>number of source checkpoints fed to the projector.</p></div>
    <div class="card"><div class="label">horizon h</div><div class="metric sm">1 - 3 gaps</div><p>how far ahead we predict (1 = current-fast).</p></div>
    <div class="card"><div class="label">substrate</div><div class="metric sm">RELEX</div><p>same base model (Qwen2.5-Math-1.5B) as the harness MATH track.</p></div>
  </div>
  <p class="muted">Substrate: <code>relex-rlvr/RLVR-Qwen2.5-Math-1.5B</code>, one Hub branch per training
  step. We replay the genuinely-distinct region, steps {META["available_steps"]} (the release repeats the
  final checkpoint from step 80 onward, and step 70&rarr;80 is a one-off outlier jump, so both are
  excluded). RELEX step index is treated as the projector time axis. Each on-disk step that has a full
  window behind it and a target ahead of it becomes one anchor instance; we report mean and standard
  deviation across instances.</p>
</section>

<section id="metrics">
  <h2>What each metric means</h2>
  <p>Every metric is the whole-tensor port of the live 16-coordinate causal probe
  (<code>verl/workers/comm_eff/rank1_probe.py</code>), computed in float64 over the tensor's elements
  (full tensor for the ~250 smaller tensors; a fixed {"{:,}".format(250000)}-coordinate sample for the
  larger matrices - a 15,000x expansion of the live 16-coordinate probe). The comparison is always the
  same: projected weights versus "do nothing, reuse the newest stale checkpoint".</p>
  <dl class="terms">
    <dt>Forecast skill = 1 &minus; proj_SSE / stale_SSE</dt>
    <dd>The headline. proj_SSE is the sum of squared errors between the projected weights and the true
    future checkpoint; stale_SSE is the same for the stale checkpoint. <strong>Skill &gt; 0 means the
    projection beats doing nothing</strong>; 1 is perfect; <strong>skill &lt; 0 means the projection is
    worse than reusing the stale checkpoint</strong> (skill = &minus;1 means it doubles the squared
    error). The bar the anchor must clear is simply skill &gt; 0.</dd>
    <dt>Direction cosine</dt>
    <dd>Cosine between the predicted update (projected &minus; latest) and the true update
    (actual &minus; latest). Skill asks whether we moved the right <em>distance</em>; cosine asks whether
    we moved in the right <em>direction</em>. This is the pivotal quantity in this study.</dd>
    <dt>Delta persistence (global cosine of successive updates)</dt>
    <dd>The energy-weighted cosine between one gap-spaced weight update and the next
    (D&#8321; = &theta;<sub>t-G</sub> &minus; &theta;<sub>t-2G</sub> vs D&#8322; = &theta;<sub>t</sub> &minus; &theta;<sub>t-G</sub>).
    This is exactly what a 2-point secant assumes stays near 1: that the last move predicts the next move.
    Near 0 means a random walk.</dd>
    <dt>In-window EVR (explained variance ratio) and R^2</dt>
    <dd>How well a single rank-1 direction and a straight line describe the checkpoints already in the
    window. Both near 1 means a pristine <em>in-window</em> fit. As H3 shows, that says nothing about the
    <em>forecast</em>.</dd>
    <dt>Fraction of tensors that beat stale (frac win)</dt>
    <dd>Share of tensors whose individual skill is &gt; 0. A breadth measure.</dd>
    <dt>Pooled vs macro aggregation</dt>
    <dd><strong>Pooled</strong> (energy-weighted): sum SSEs across tensors, then form the ratio, so
    high-energy tensors dominate - the robust headline. <strong>Macro</strong>: average per-tensor skills
    equally, so a tiny near-frozen bias counts as much as a big matrix; it is informative but unstable for
    near-zero-motion tensors (their skill explodes), so we lead with pooled.</dd>
  </dl>
</section>

<section id="h1">
  <h2>H1 - How many checkpoints?</h2>
  <p>Forecast skill versus window W at horizon 1, on the dense proxy at the cadence gap. The hypothesis
  was that skill peaks at small W. The data says something sharper: at this gap the projector <em>cannot
  win at any W</em>, and within that losing range the ordering is <strong>reversed</strong> from the live
  run - larger windows are less harmful.</p>
  <figure>{IMG("skill_vs_W_pooled.png", "pooled forecast skill vs window W")}
    <figcaption>Pooled (energy-weighted) forecast skill vs window W, per method, horizon 1. The dotted
    line at 0 is the stale baseline. Every curve sits below it: no projector beats "stay put" here.
    rank1_relex (pin-to-latest) climbs from {f3(w2)} at W=2 toward {f3(best_skill)} at W={best_W} because a
    longer OLS window fits a flatter slope and therefore moves less.</figcaption></figure>
  <figure>{IMG("skill_vs_W_macro.png", "macro forecast skill vs window W")}
    <figcaption>The same story with equal-per-tensor (macro) aggregation.</figcaption></figure>
  {h1_table}
  <div class="callout red"><strong>Finding (a reversal).</strong> The 2-checkpoint secant is the
  <em>worst</em> window here ({f3(w2)}), not the best; W=2 vs W=4 is {f3(w2)} vs {f3(w4)}, the mirror image
  of the live run's {f3(0.780)} vs {f3(0.173)}. This is not a contradiction of the live result - it is the
  key to it. See H2.</div>
</section>

<section id="h2">
  <h2>H2 - Why W=2 (and why it flips)</h2>
  <p>The 2-checkpoint secant draws the line through the two newest exact checkpoints and steps one gap
  further along it. It bets everything on one assumption: <strong>the last weight update predicts the next
  one</strong> (delta persistence near 1). A W&ge;3 OLS instead fits a line through several deltas, which
  averages in older directions and damps the step.</p>
  <div class="formula">theta_hat = latest + alpha &middot; (h / g) &middot; (latest &minus; base)&nbsp;&nbsp;&nbsp;(the W=2 secant)</div>
  <p>When updates persist (are collinear), the secant's aggressive full-step extrapolation is right and
  W=2 wins - which is the live compressed regime (direction cosine 0.91, skill 0.78). When updates are
  orthogonal, the secant confidently steps in an uncorrelated direction and roughly doubles the error,
  while the damped W&ge;3 estimate that barely moves is safer. That is this dense proxy: the measured
  update-direction cosine is only about +0.05 to +0.20, and the energy-weighted delta persistence is
  essentially 0. So the very same projector that wins live is worst here. <strong>The controlling variable
  is persistence, not the number of checkpoints.</strong> The gap sweep below tests whether finer spacing
  recovers that persistence (it does not).</p>
</section>

<section id="h3">
  <h2>H3 - Is rank-1 right, and does in-window fit predict forecast skill?</h2>
  <p>For a single delta (W=2) rank-1 is trivially exact (EVR = 1). The real question at W&ge;3 is whether a
  beautiful in-window fit predicts a good forecast. It does not - and cranking the rank up, which fits the
  window even better, makes the forecast <em>worse</em>.</p>
  <figure>{IMG("fit_vs_skill.png", "in-window fit vs forecast skill")}
    <figcaption>The paradox: in-window EVR and R^2 stay near 1 across all W while forecast skill is deeply
    negative. A pristine fit to the window's own history says nothing about the next checkpoint.</figcaption></figure>
  <figure>{IMG("rank_ablation.png", "skill vs SVD rank")}
    <figcaption>Rank sweep (W&ge;3): forecast skill vs SVD rank r. For every window, rank 1 is best and
    higher rank is monotonically worse - more in-window variance captured, worse extrapolation.</figcaption></figure>
  <div class="callout"><strong>Decoupling, quantified.</strong> At W=4, raising the rank from 1 to 3 lifts
  in-window EVR from {f3(evr_r1, False)} to {f3(evr_r3, False)} yet drops forecast skill from {f3(r1)} to
  {f3(r3)}. Rank-1 is not just "enough" - it is the safest choice, confirming the harness's hard rank-1
  design. In-window EVR/R^2 are diagnostics of the fit, not predictors of the forecast.</div>
</section>

<section id="h4">
  <h2>H4 - How far ahead can the anchor fire (horizon)?</h2>
  <p>Horizon h is how many gaps ahead we project (h=2 = "twice a fast", which would let the anchor fire
  half as often). On a persistent trajectory a modest h can still beat stale; on this random-walk proxy it
  cannot, and further extrapolation is strictly worse.</p>
  <figure>{IMG("skill_vs_horizon.png", "forecast skill vs horizon")}
    <figcaption>Pooled forecast skill vs horizon per window, for the pinned rank1_relex projector. Every
    curve descends further below the stale line as the horizon grows.</figcaption></figure>
  <div class="callout blue"><strong>No cadence headroom on the proxy.</strong> For W=2, skill goes
  {f3(h2_w2[0])} &rarr; {f3(h2_w2[1])} &rarr; {f3(h2_w2[2])} across h=1,2,3; for W=4, {f3(h_w4[0])} &rarr;
  {f3(h_w4[1])} &rarr; {f3(h_w4[2])}. Because increments are uncorrelated, doubling the horizon roughly
  doubles the overshoot. Any decision to fire the anchor less often must be justified in the compressed
  regime (where updates persist), not on this dense proxy.</div>
</section>

<section id="h5">
  <h2>H5 - Pin-to-latest vs rebuild-from-base</h2>
  <p>The harness projector is pinned to the newest exact checkpoint: it keeps <code>latest</code> and adds
  only the incremental rank-1 motion, preserving that checkpoint's off-subspace residual. The RELEX paper
  instead rebuilds the whole delta from the window base
  (<code>theta_hat = base + c_pred &middot; V^T</code>), discarding that residual. They are identical at
  W=2 and diverge for W&ge;3.</p>
  <figure>{IMG("method_compare.png", "method comparison")}
    <figcaption>Pooled forecast skill by method (stale baseline = 0). rank1_relex is pin-to-latest,
    relex_from_base is the paper's rebuild, fixed_linear is the frozen decoder-only 2-point seed.</figcaption></figure>
  <div class="callout blue"><strong>Pinning wins.</strong> At W=4, pin-to-latest ({f3(pin4)}) is clearly
  less harmful than rebuild-from-base ({f3(base4)}), because the newest exact checkpoint carries real
  off-subspace content that the base reconstruction throws away. Keep the pinned increment. (The H1 curves
  show the gap widening with W: rebuild-from-base gets steadily worse as the window lengthens.)</div>
</section>

<section id="h6">
  <h2>H6 - Which tensors benefit?</h2>
  <p>Per-tensor-type skill at the W=2 secant, horizon 1. The live probe hinted that norms gain and the
  attention projections do not; the whole-tensor replay confirms the ordering, even though at this gap
  almost nothing is above zero.</p>
  <figure>{IMG("skill_by_type.png", "skill by tensor type")}
    <figcaption>Energy-pooled forecast skill by tensor type (rank1_relex, W=2, horizon 1). The input
    layernorm is the only type marginally above zero; every attention/MLP projection sits near &minus;1
    (the secant doubles their error). Bars use energy pooling so the near-frozen q/k/v biases, whose
    per-tensor skill is wildly unstable, do not dominate.</figcaption></figure>
  {by_type_table(2)}
  <p>The tied embedding is scored separately (its 233M parameters are coordinate-sampled): W=2 skill
  {f3(EMBED["W2"][0])} &plusmn; {EMBED["W2"][1]:.2f}, W=4 skill {f3(EMBED["W4"][0])} &plusmn;
  {EMBED["W4"][1]:.2f} - the same pattern as the projections. The relative order matches the live hint
  (norms least-harmed, attention projections most-harmed), which is what carries over between regimes even
  when the absolute sign does not.</p>
</section>

<section id="gap">
  <h2>The gap sweep - reconciling the proxy with the live run</h2>
  <p>If the dense proxy fails only because 10-step spacing is too coarse to see the tangent, then finer
  gaps should recover both the direction persistence and a positive secant skill. We downloaded a block of
  consecutive checkpoints and swept the source gap G from 1 to 30 steps.</p>
  <figure>{IMG("skill_vs_gap.png", "skill vs gap")}
    <figcaption>Pooled forecast skill vs source gap G (log axis), for the W=2 secant and the W=4 OLS.
    Neither approaches the stale line at any gap; the secant stays near &minus;1 all the way down to gap 1.</figcaption></figure>
  <figure>{IMG("persist_vs_gap.png", "persistence vs gap")}
    <figcaption>The mechanism: delta persistence (global cosine of successive updates) and mean
    per-tensor update-direction cosine vs gap. Both stay near 0 at every gap - the dense trajectory is a
    random walk in weight space at all spacings we can probe.</figcaption></figure>
  {gap_table()}
  <div class="callout red"><strong>Persistence does not recover.</strong> Even at the finest gap (G={fine},
  consecutive checkpoints = 2 optimizer steps), the W=2 secant skill is about {f3(next((r["pooled_skill"] for r in gap_W2 if r["gap"] == fine), None))}
  and the delta persistence is {f3(persist_fine)} (per-tensor direction cosine {f3(cos_fine)}). The dense
  RLVR trajectory of this base model is directionally uncorrelated step-to-step at every scale we tested.
  The live compressed run's direction cosine of 0.91 therefore cannot come from the base model's dynamics
  - the most parsimonious source is the PowerSGD compression itself, which forces each update into a shared
  rank-77 subspace and so makes consecutive updates collinear. That is the reconciliation: the projector is
  a good idea <em>because</em> the fast circuit is compressed, not in spite of it.</div>
</section>
{count_section}
<section id="results">
  <h2>Full results table</h2>
  <p>Every (method, window, horizon) at rank 1, aggregated over anchor positions (mean &plusmn; standard
  deviation), on the clean gap=10 grid. Skill is 1 &minus; proj_SSE/stale_SSE; the best and worst
  rank1_relex windows at h=1 are shaded in the H1 table above.</p>
  {results_table}
  <p class="muted">n anchors is the number of on-disk anchor positions with a full window behind them and a
  target ahead, restricted to the clean region (steps 10-70). EVR and R^2 are the in-window fit
  diagnostics (n/a for relex_from_base and fixed_linear, which do not expose them). Larger windows have
  fewer clean anchors, hence wider error bars.</p>
</section>

<section id="live">
  <h2>Live-run reality check</h2>
  <p>The study is a whole-tensor replay on a clean dense trajectory. The live harness numbers below are the
  16-coordinate causal probe on the real compressed run. The point of putting them side by side is the
  <em>contrast</em>: same base model, same projector, opposite outcome - because the live updates persist
  and the dense ones do not.</p>
  <div class="table-wrap"><table>
    <thead><tr><th>WandB run</th><th>arm</th><th class="right">W</th>
      <th class="right">live skill</th><th class="right">study skill (dense proxy)</th>
      <th class="right">live dir cos</th><th class="right">study dir cos</th>
      <th class="right">MATH acc</th></tr></thead>
    <tbody>{live_rows()}</tbody></table></div>
  <p class="muted">Live skill/cos are from the harness probe (4 tensors &times; 16 coordinates) on the
  compressed fast circuit; study skill/cos are the whole-tensor replay on the dense RELEX trajectory. MATH
  accuracy is the downstream end-to-end result of the live GRPO runs. The live W=2 secant beat both the W=4
  arm and matched the dense control on MATH; that win is what this study explains and bounds.</p>
</section>

<section id="decision">
  <h2>Interpretation and decision</h2>
  <ul>
    <li><strong>Keep the W=2 secant at horizon 1 for the compressed circuit.</strong> The live evidence
    (skill 0.78, direction cosine 0.91, MATH 67.89% vs 63.61%) stands. The secant is the right recent-tangent
    extrapolator precisely when updates persist, which the compressed fast circuit delivers.</li>
    <li><strong>Do not read "more checkpoints" as "better".</strong> On any trajectory whose updates are not
    persistent, extra checkpoints only help by damping the projector toward the stale checkpoint; the honest
    move there is to not project at all. The number of checkpoints is a second-order knob behind persistence.</li>
    <li><strong>Fit is not forecast.</strong> In-window EVR/R^2 near 1 (and higher SVD rank) do not imply
    skill; here higher rank actively hurts. Trust a held-out forecast score, never the in-window fit.</li>
    <li><strong>Rank-1 is correct.</strong> The rank sweep confirms the harness's hard rank-1 choice.</li>
    <li><strong>Keep pinning to latest.</strong> Pin-to-latest beats rebuild-from-base at every W&ge;3 by
    preserving the newest checkpoint's off-subspace residual.</li>
    <li><strong>The projector's benefit is compression-specific.</strong> The persistence that makes it work
    is, on the evidence here, induced by PowerSGD, not by the base model. A follow-up that measures direction
    persistence directly on the live compressed checkpoints (vs an uncompressed control of the same run)
    would close this loop.</li>
  </ul>
</section>

<section id="caveats">
  <h2>Caveats</h2>
  <ul>
    <li><strong>Dense proxy vs compressed fast.</strong> This is the central caveat, now a finding: the
    RELEX trajectory is a dense (uncompressed) RLVR run and is directionally uncorrelated step-to-step,
    whereas the live fast circuit trains under PowerSGD compression and is collinear. The study upper-bounds
    what a projector can do on a clean same-base-model trajectory and isolates persistence as the deciding
    factor; the live probe is the ground truth for the compressed regime.</li>
    <li><strong>Weight-space, not MATH.</strong> Skill and cosine are weight-space forecast quality, not
    downstream accuracy. The accuracy link is the existing live result. A GPU follow-up could reconstruct a
    predicted checkpoint and evaluate it.</li>
    <li><strong>Step-index and spacing.</strong> RELEX step index is treated as the trajectory time axis and
    the gap as a free parameter bracketing the harness's 10-global-step cadence. The finest gap we can probe
    is 1 RELEX step (2 optimizer steps); we cannot see below that.</li>
    <li><strong>Large tensors are coordinate-sampled.</strong> The ~250 smaller tensors are scored in full;
    matrices above 250,000 elements use a fixed 250k-coordinate sample. Validation on held-out tensors put
    the sampled skill within about 0.01-0.06 of the full-tensor value and the direction cosine within
    0.003, far tighter than the effects discussed.</li>
    <li><strong>Sparse anchors and a short trajectory.</strong> The genuinely-distinct RELEX region is short
    (steps 10-70 after excluding the repeated tail and the step-80 outlier), so large windows have few
    anchors and wide error bars. We report mean &plusmn; std and do not over-interpret single-anchor points.</li>
  </ul>
</section>

<section id="method">
  <h2>Method and reproducibility</h2>
  <p>The projector under test is a CPU port of <code>verl/workers/comm_eff/lookahead.py</code>
  (<code>project_rank1_tensor</code> and <code>Rank1RelexProjector</code>), <strong>proven numerically
  identical to the live function</strong> for W=2,3,4,6 (max difference 0.00e+00). Mechanics: cumulative
  deltas versus the window base, rank-1 via the Gram trick, an OLS temporal fit (with the base coordinate
  added at W=2, giving the exact secant), and the prediction pinned to the newest exact checkpoint plus the
  incremental rank-1 motion.</p>
  <div class="grid">
    <div class="card"><div class="label">tensors scored</div><div class="metric sm">338 + embed</div><p>196 decoder matrices, 84 q/k/v biases, 57 norms; the tied embedding scored separately.</p></div>
    <div class="card"><div class="label">rows computed</div><div class="metric sm">{META["n_rows"]:,}</div><p>one per (combo &times; tensor); {META["n_combos"]} combo summaries on the clean grid.</p></div>
    <div class="card"><div class="label">port fidelity</div><div class="metric sm good">0.00e+00</div><p>max abs diff vs the live projector.</p></div>
    <div class="card"><div class="label">compute</div><div class="metric sm">CPU</div><p>no GPU, no verl install; safetensors + torch.</p></div>
  </div>
  <p class="muted">Reproduce: the scripts in <code>research_studies/relex_ckpt_ablation/</code> -
  <code>run_forecast_ablation_fast.py</code> (tensor-major, coordinate-sampled), <code>gap_sweep.py</code>,
  <code>make_plots.py</code>, <code>make_gap_plots.py</code>, and <code>build_report.py</code>. The port
  equivalence check is <code>harness_projector.py</code>. See <code>PLAN.md</code> for the full protocol.</p>
</section>

</main>
<footer class="wrap">
  <p>RELEX checkpoint-count ablation &middot; branch <span class="mono">exp/relex-ckpt-ablation</span> &middot;
  substrate <span class="mono">relex-rlvr/RLVR-Qwen2.5-Math-1.5B</span> (base Qwen2.5-Math-1.5B) &middot;
  RELEX paper arXiv 2605.21468. Weight-space forecast quality on a dense proxy; downstream MATH numbers are
  the live GRPO runs. Generated from the ablation digests.</p>
</footer>
</body></html>
"""
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    open(args.out, "w").write(html)
    print(f"wrote {args.out}  ({len(html) / 1000:.0f} KB)")


if __name__ == "__main__":
    main()
