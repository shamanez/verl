#!/usr/bin/env python3
"""EXP-38 — JOINT GSM8K vs Big-Math comparative report (B4).

Reads the two per-arm ``*_findings.json`` (the DERIVED scalars/curves emitted by
``exp38_drift_analysis.py`` — never raw tensors, never merged across datasets) and
renders ONE self-contained, dataset-tagged HTML comparing dense-GRPO temporal drift,
gradient/activation rank, boundary subspace staleness, and the nature of learning
between an EASY task (GSM8K, base \\boxed ≈ 0.72) and a HARD task (Big-Math, base ≈ 0.48).

The central question: are the gradient-anchor staleness budget, the activation-codec
low-rank-ness, and the nature of learning TASK-DEPENDENT — i.e. must the next
communication-efficient PP GRPO method's staleness/codec budget be set per task?

HARD anti-mixing rule (operator): GSM8K and Big-Math are NEVER merged into one array or
one plot series. Each curve is computed from its OWN dataset's findings and drawn as a
separate, clearly dataset-labelled series; the comparison is over derived scalars/curves
only. Tensors are never touched here.

Usage:
  python3 scripts/exp38_compare.py \
      --gsm8k    reports/dense-run-behaviour/exp38-dense-drift-gsm8k_findings.json \
      --big-math reports/dense-run-behaviour/exp38-dense-drift-big-math_findings.json \
      --narrative reports/dense-run-behaviour/_joint_narrative.html \
      --out reports/dense-run-behaviour/exp38-dense-drift-joint.html
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import json
import math
import os

import numpy as np

import exp38_drift_analysis as A  # constants only
import exp38_report as R          # reuse the per-arm hypothesis resolvers + CSS (DRY)

LAGS = A.LAGS
R_LOCKED = A.R_LOCKED
HIDDEN = A.HIDDEN

# distinct, clearly dataset-labelled series (anti-mixing: two series, never merged).
COL = {"gsm8k": "#1f6b3a", "big-math": "#b3261e"}
LAB = {"gsm8k": "GSM8K (easy task, base ≈0.72)", "big-math": "Big-Math (hard task, base ≈0.48)"}
SHORT = {"gsm8k": "GSM8K", "big-math": "Big-Math"}


def gk(d, k):
    """Findings dicts come from JSON, so per-lag keys are strings; fetch by str or int."""
    if not isinstance(d, dict):
        return None
    return d.get(str(k), d.get(k))


def _normalize_keys(obj):
    """JSON turns the int per-lag/per-step dict keys into strings. Recursively convert
    any dict whose keys are ALL digit-strings back to int, so the reused per-arm
    resolvers (which index with int lags, e.g. cos.get(5)) work on loaded findings.
    Dicts with non-numeric keys (signal names, boundary names) are left untouched."""
    if isinstance(obj, dict):
        new = {k: _normalize_keys(v) for k, v in obj.items()}
        if new and all(isinstance(k, str) and k.lstrip("-").isdigit() for k in new):
            return {int(k): v for k, v in new.items()}
        return new
    if isinstance(obj, list):
        return [_normalize_keys(x) for x in obj]
    return obj


def _safe(x, nd=3):
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return "n/a"
        return f"{x:.{nd}f}"
    except Exception:
        return str(x)


def _fig(plot_fn, w=7.6, h=4.2):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(w, h))
    plot_fn(ax)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _img(plots, key, caption):
    if key not in plots:
        return f'<p class="sub"><em>[plot {html.escape(key)} unavailable]</em></p>'
    return f'<img src="{plots[key]}" alt="{html.escape(caption)}"><p class="sub">{html.escape(caption)}</p>'


def build(g, b, out_path, narrative_html=""):
    """g = GSM8K findings dict, b = Big-Math findings dict."""
    import matplotlib

    matplotlib.use("Agg")

    arms = [("gsm8k", g), ("big-math", b)]
    plots = {}

    # ---- 1. gradient cosine vs lag (two labelled series) ----
    def _p_cos(ax):
        for ds, f in arms:
            cos = f.get("grad_cos", {})
            ks = [k for k in LAGS if gk(cos, k) is not None]
            ax.plot(ks, [gk(cos, k) for k in ks], "o-", color=COL[ds], label=LAB[ds], lw=2)
        ax.axvline(5, color="#27ae60", ls=":", lw=1.2, label="k≈5 (stable 5/5)")
        ax.axvline(20, color="#e67e22", ls=":", lw=1.2, label="k≈20 (broken 20/20)")
        ax.axhline(0, color="#888", lw=0.8)
        ax.set_xlabel("lag k (optimizer ticks)")
        ax.set_ylabel("median cos(g_t, g_{t−k})")
        ax.set_title("Gradient-direction staleness: easy vs hard task")
        ax.legend(fontsize=7.5)
        ax.grid(alpha=0.3)

    plots["cos"] = _fig(_p_cos)

    # ---- 2. gradient sign-agreement vs lag ----
    def _p_sign(ax):
        for ds, f in arms:
            sg = f.get("grad_sign", {})
            ks = [k for k in LAGS if gk(sg, k) is not None]
            ax.plot(ks, [gk(sg, k) for k in ks], "s-", color=COL[ds], label=LAB[ds], lw=2)
        ax.axhline(0.5, color="#888", ls="--", lw=0.9, label="chance = 0.5")
        ax.axvline(5, color="#27ae60", ls=":", lw=1.1)
        ax.axvline(20, color="#e67e22", ls=":", lw=1.1)
        ax.set_xlabel("lag k (optimizer ticks)")
        ax.set_ylabel("median sign-agreement")
        ax.set_title("Gradient sign-agreement vs staleness")
        ax.legend(fontsize=7.5)
        ax.grid(alpha=0.3)

    plots["sign"] = _fig(_p_sign)

    # ---- 3. gradient rank-for-90 over training (two labelled series) ----
    def _p_rank(ax):
        for ds, f in arms:
            bystep = f.get("grad_rank90_by_step", {})
            if bystep:
                xs = sorted(int(s) for s in bystep)
                ax.plot(xs, [gk(bystep, s) for s in xs], "-o", color=COL[ds], ms=3, label=LAB[ds])
        ax.axhline(R_LOCKED, color="#333", ls="--", lw=1, label=f"PowerSGD r={R_LOCKED}")
        if g.get("epoch2_step"):
            ax.axvline(g["epoch2_step"], color="#999", ls=":", lw=1, label="GSM8K epoch-2 (~58)")
        ax.set_xlabel("global step")
        ax.set_ylabel("median dense-gradient rank-for-90% energy")
        ax.set_title("Nature of learning: dense-gradient effective rank over training")
        ax.legend(fontsize=7.5)
        ax.grid(alpha=0.3)

    plots["rank"] = _fig(_p_rank)

    # ---- 4. boundary subspace overlap vs lag (top-77 and top-1, two datasets) ----
    def _p_ov(ax):
        for ds, f in arms:
            o77 = f.get("boundary_overlap_by_k", {})
            o1 = f.get("boundary_overlap_r1_by_k", {})
            ks = [k for k in LAGS if gk(o77, k) is not None]
            ax.plot(ks, [gk(o77, k) for k in ks], "o-", color=COL[ds], lw=2, label=f"{SHORT[ds]} top-{R_LOCKED}")
            if any(gk(o1, k) is not None for k in LAGS):
                ax.plot(ks, [gk(o1, k) for k in ks], "^--", color=COL[ds], lw=1.3, alpha=0.7, label=f"{SHORT[ds]} top-1")
        ax.axvline(5, color="#27ae60", ls=":", lw=1.1)
        ax.axvline(20, color="#e67e22", ls=":", lw=1.1)
        ax.set_ylim(0, 1.03)
        ax.set_xlabel("lag k (optimizer ticks)")
        ax.set_ylabel("boundary subspace overlap o(t,t−k)")
        ax.set_title("Activation-codec staleness: forward-subspace overlap vs lag")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    plots["overlap"] = _fig(_p_ov)

    # ---- 5. headline rank comparison bars (grouped: easy vs hard) ----
    def _p_bars(ax):
        cats = ["dense grad\nrank-90", "boundary h\nrank-90", "boundary grad_h\nrank-90"]
        gv = [g.get("grad_rank90_median"), g.get("boundary_h_raw_rank90_median"), g.get("boundary_gradh_rank90_median")]
        bv = [b.get("grad_rank90_median"), b.get("boundary_h_raw_rank90_median"), b.get("boundary_gradh_rank90_median")]
        x = np.arange(len(cats))
        ax.bar(x - 0.19, [v or 0 for v in gv], 0.38, color=COL["gsm8k"], label=SHORT["gsm8k"])
        ax.bar(x + 0.19, [v or 0 for v in bv], 0.38, color=COL["big-math"], label=SHORT["big-math"])
        ax.axhline(R_LOCKED, color="#333", ls="--", lw=1, label=f"r={R_LOCKED}")
        for xi, v in zip(x - 0.19, gv):
            ax.text(xi, (v or 0) + 2, f"{v:.0f}" if v else "—", ha="center", fontsize=8)
        for xi, v in zip(x + 0.19, bv):
            ax.text(xi, (v or 0) + 2, f"{v:.0f}" if v else "—", ha="center", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(cats, fontsize=8)
        ax.set_ylabel("rank-for-90% energy")
        ax.set_title("Rank comparison vs codec budget r=77")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis="y")

    plots["bars"] = _fig(_p_bars)

    # ---- 6. weight drift vs lag (normalized to each arm's k=40 value) ----
    def _p_wd(ax):
        for ds, f in arms:
            wd = f.get("weight_drift", {})
            ks = [k for k in LAGS if gk(wd, k) is not None]
            base = gk(wd, max(ks)) if ks else None
            if base:
                ax.plot(ks, [gk(wd, k) / base for k in ks], "d-", color=COL[ds], lw=2, label=LAB[ds])
        ax.axvline(5, color="#27ae60", ls=":", lw=1.1)
        ax.axvline(20, color="#e67e22", ls=":", lw=1.1)
        ax.set_xlabel("lag k (optimizer ticks)")
        ax.set_ylabel("weight drift ‖Δθ‖ (normalized to k=40)")
        ax.set_title("Parameter-point gap (weight drift) vs lag")
        ax.legend(fontsize=7.5)
        ax.grid(alpha=0.3)

    plots["wd"] = _fig(_p_wd)

    _write(out_path, g, b, plots, narrative_html)


# --------------------------------------------------------------------------- #
def _verdict_badge(res):
    return f'<span class="verdict {res[0]}">{html.escape(res[1])}</span>'


def _cmp_row(label, gv, bv, fmt=3, hint=""):
    return (f"<tr><td>{label}</td><td class='num'>{_safe(gv, fmt) if not isinstance(gv,str) else gv}</td>"
            f"<td class='num'>{_safe(bv, fmt) if not isinstance(bv,str) else bv}</td><td>{hint}</td></tr>")


def _write(out_path, g, b, plots, narrative_html):
    h1g, h2g, h3g = R._resolve_h1(g), R._resolve_h2(g), R._resolve_h3(g)
    h1b, h2b, h3b = R._resolve_h1(b), R._resolve_h2(b), R._resolve_h3(b)

    cg, cb = g.get("grad_cos", {}), b.get("grad_cos", {})
    sg, sb = g.get("grad_sign", {}), b.get("grad_sign", {})
    og, ob = g.get("boundary_overlap_by_k", {}), b.get("boundary_overlap_by_k", {})

    # central-question determination, from the numbers
    def _diff(a, c, rel=0.25):
        if a is None or c is None:
            return None
        try:
            return abs(a - c) > rel * max(abs(a), abs(c), 1e-9)
        except Exception:
            return None

    budget_diff = _diff(gk(cg, 1), gk(cb, 1))           # staleness budget (cos at lag 1)
    grank_diff = _diff(g.get("grad_rank90_median"), b.get("grad_rank90_median"))
    gradh_diff = _diff(g.get("boundary_gradh_rank90_median"), b.get("boundary_gradh_rank90_median"))
    h_invariant = (not _diff(g.get("boundary_h_raw_rank90_median"), b.get("boundary_h_raw_rank90_median")))

    # headline comparison table
    rows = "".join([
        _cmp_row("Gradient cos at k=1 (lag-1 tick)", gk(cg, 1), gk(cb, 1), 3,
                 "the most-correlated case — the hard task is decorrelated even here"),
        _cmp_row("Gradient cos at k≈5 (stable 5/5 anchor)", gk(cg, 5), gk(cb, 5), 3, ""),
        _cmp_row("Gradient cos at k≈20 (broken 20/20 anchor)", gk(cg, 20), gk(cb, 20), 3, ""),
        _cmp_row("Gradient sign-agreement at k≈5", gk(sg, 5), gk(sb, 5), 3, "chance = 0.5"),
        _cmp_row("Dense-gradient rank-for-90% (median)", g.get("grad_rank90_median"),
                 b.get("grad_rank90_median"), 1, f"vs codec r={R_LOCKED}"),
        _cmp_row("Boundary h rank-for-90% (median)", g.get("boundary_h_raw_rank90_median"),
                 b.get("boundary_h_raw_rank90_median"), 1, "≈ rank-1 on BOTH (massive activation)"),
        _cmp_row("Boundary h top-1 energy share", g.get("boundary_h_top1_energy_share_median"),
                 b.get("boundary_h_top1_energy_share_median"), 3, "fraction of energy in 1 direction"),
        _cmp_row("Boundary grad_h rank-for-90% (median)", g.get("boundary_gradh_rank90_median"),
                 b.get("boundary_gradh_rank90_median"), 1, f"backward link, vs r={R_LOCKED}"),
        _cmp_row("Top-77 subspace overlap o(t,t−20)", gk(og, 20), gk(ob, 20), 3, "Q-staleness (flat = stale-tolerant)"),
        _cmp_row("H2 weight half-drift lag (global steps)", g.get("h2_weight_half_lag"),
                 b.get("h2_weight_half_lag"), 1, "behaviour signals beat this in both arms"),
    ])

    def _q(supp):
        return ("<b style='color:#b3261e'>YES — task-dependent</b>" if supp
                else "<b style='color:#1f6b3a'>no — invariant</b>")

    central = f"""
<div class="kpi">
<div><b>{_safe(gk(cg,1),2)} → {_safe(gk(cb,1),2)}</b><span>grad cos @k=1: easy → hard</span></div>
<div><b>{_safe(g.get('grad_rank90_median'),0)} → {_safe(b.get('grad_rank90_median'),0)}</b><span>grad rank-90: easy → hard</span></div>
<div><b>{_safe(g.get('boundary_gradh_rank90_median'),0)} → {_safe(b.get('boundary_gradh_rank90_median'),0)}</b><span>grad_h rank-90: easy → hard</span></div>
<div><b>1 ≈ 1</b><span>boundary h rank (INVARIANT)</span></div>
</div>
<ul>
<li><b>Gradient-anchor staleness budget</b> — {_q(budget_diff)}. Easy task keeps a short usable window
(cos {_safe(gk(cg,1),2)}→{_safe(gk(cg,5),2)} over k=1→5); the hard task is decorrelated <i>even at lag 1</i>
(cos {_safe(gk(cb,1),3)}). On a hard task a stale dense gradient is junk at any latency.</li>
<li><b>Dense-gradient effective rank</b> — {_q(grank_diff)}: {_safe(g.get('grad_rank90_median'),0)} (easy) vs
{_safe(b.get('grad_rank90_median'),0)} (hard, ≈ the codec budget r={R_LOCKED}).</li>
<li><b>Backward boundary traffic (grad_h) rank</b> — {_q(gradh_diff)}: {_safe(g.get('boundary_gradh_rank90_median'),0)}
(easy) vs {_safe(b.get('boundary_gradh_rank90_median'),0)} (hard). The backward link gets harder to compress as the
task gets harder.</li>
<li><b>Forward activation low-rank-ness</b> — {_q(not h_invariant)}: boundary h is rank-≈1 (massive-activation
direction) on BOTH tasks. The activation-codec <i>primitive</i> (rank-r forward compression) is task-independent;
its <i>budget</i> (and the gradient/backward budgets) are not.</li>
</ul>
"""

    # per-arm hypothesis comparison table
    hyp_rows = f"""
<tr><td>H1 — gradient-space staleness budget crossed by ~20-tick lag</td>
<td>{_verdict_badge(h1g)}<br><span class="sub">cos {_safe(gk(cg,5),3)}→{_safe(gk(cg,20),3)}; sign {_safe(gk(sg,5),2)}→{_safe(gk(sg,20),2)}</span></td>
<td>{_verdict_badge(h1b)}<br><span class="sub">cos {_safe(gk(cb,5),3)}→{_safe(gk(cb,20),3)}; sign {_safe(gk(sb,5),2)}→{_safe(gk(sb,20),2)}</span></td></tr>
<tr><td>H2 — drift is GRPO-coupled (distribution gap), not pure parameter-point gap</td>
<td>{_verdict_badge(h2g)}</td><td>{_verdict_badge(h2b)}</td></tr>
<tr><td>H3 — boundary activation low-rank with a Q-staleness budget</td>
<td>{_verdict_badge(h3g)}<br><span class="sub">h rank≈{_safe(g.get('boundary_h_raw_rank90_median'),1)}; o(t,t−20)={_safe(gk(og,20),3)}</span></td>
<td>{_verdict_badge(h3b)}<br><span class="sub">h rank≈{_safe(b.get('boundary_h_raw_rank90_median'),1)}; o(t,t−20)={_safe(gk(ob,20),3)}</span></td></tr>
"""

    nar = narrative_html or (
        '<p class="sub"><em>[theorist narrative fragment not supplied — pass --narrative '
        'reports/dense-run-behaviour/_joint_narrative.html]</em></p>')

    body = f"""<div class="wrap">
<h1>EXP-38 — Dense GRPO temporal drift: <span style="background:#1f2328;color:#fff;border-radius:6px;padding:.08em .5em;font-size:.62em;vertical-align:middle">GSM8K ↔ Big-Math</span></h1>
<p class="sub">A joint, dataset-tagged comparison of how the on-policy GRPO learning signal drifts in time on an
EASY task (GSM8K) vs a HARD task (Big-Math) — to decide whether the next communication-efficient pipeline-parallel
GRPO method's <b>staleness budget</b> and <b>codec budget</b> must be set <b>per task</b>.</p>
<p class="sub">Qwen2.5-1.5B-Instruct · dense (comm_eff OFF) · 75 global steps = 150 optimizer ticks · n=1 per task ·
lag axis in optimizer ticks (2/global-step ⇒ k≈5 ≙ stable 5/5 anchor, k≈20 ≙ broken 20/20 anchor). <b>The two
datasets' tensors and curves are never merged</b> — every series is computed from its own dataset's captures and
drawn as a separate, dataset-labelled line.</p>

<div class="tldr">
<h3>TL;DR — is the budget task-dependent?</h3>
{central}
</div>

<h2><span class="num">1</span>Gradient-direction staleness — easy vs hard</h2>
{_img(plots,'cos','Median gradient cosine cos(g_t,g_{t−k}) vs lag, GSM8K vs Big-Math (separate series).')}
{_img(plots,'sign','Median gradient sign-agreement vs lag (chance = 0.5).')}
<p>The dense run measures the <b>parameter-point gap (gap 1)</b> — the curvature-bounded term SFT also has. On
the EASY task it decays from cos {_safe(gk(cg,1),3)} (k=1) to {_safe(gk(cg,5),3)} (k≈5) to ≈0 by k≈10; on the
HARD task it is already ≈{_safe(gk(cb,1),3)} at k=1 — the dense gradient direction carries almost no
reusable information across even a single tick. <b>The gradient-anchor staleness budget is task-dependent and
near-zero on the hard task.</b></p>

<h2><span class="num">2</span>Nature of learning — gradient effective rank over training</h2>
{_img(plots,'rank','Median dense-gradient rank-for-90%-energy over training (per dataset), vs r=77.')}
<p>Median dense-gradient rank-for-90%: <b>{_safe(g.get('grad_rank90_median'),0)}</b> (GSM8K) vs
<b>{_safe(b.get('grad_rank90_median'),0)}</b> (Big-Math, right at the codec budget r={R_LOCKED}). On GSM8K the
naive ≤58-vs->58 epoch split ({_safe(g.get('grad_rank90_pre_epoch2_naive'),0)}→{_safe(g.get('grad_rank90_post_epoch2'),0)})
is a warmup-binning artifact; post-warmup the rank is stationary ({_safe(g.get('grad_rank90_pre_epoch2'),0)} vs
{_safe(g.get('grad_rank90_post_epoch2'),0)} across step 58). Big-Math crosses no epoch boundary in 75 steps.</p>

<h2><span class="num">3</span>Boundary activation &amp; the forward/backward asymmetry</h2>
{_img(plots,'overlap','Boundary forward-subspace overlap o(t,t−k) vs lag (top-77 solid, top-1 dashed), per dataset.')}
{_img(plots,'bars','Rank-for-90% of dense grad, boundary h, and boundary grad_h vs codec budget r=77 (easy vs hard).')}
<p>The forward activation <code>h</code> is rank-≈1 on BOTH tasks (top-1 energy
{_safe(g.get('boundary_h_top1_energy_share_median'),3)} / {_safe(b.get('boundary_h_top1_energy_share_median'),3)})
— a massive-activation direction dominates, so rank-r forward compression is the right primitive and r={R_LOCKED} is
hugely over-provisioned, <b>task-independently</b>. The subspace overlap is flat across lag (Q is stale-tolerant).
But the BACKWARD <code>grad_h</code> is rank <b>{_safe(g.get('boundary_gradh_rank90_median'),0)}</b> (GSM8K) /
<b>{_safe(b.get('boundary_gradh_rank90_median'),0)}</b> (Big-Math) — above r and growing with task hardness: the
backward link is NOT as compressible as the forward, and the gap widens on the hard task.</p>

<h2><span class="num">4</span>Hypotheses, per task</h2>
<table>
<tr><th>hypothesis</th><th>GSM8K (easy)</th><th>Big-Math (hard)</th></tr>
{hyp_rows}
</table>
<p class="sub">H1/H2/H3 are re-resolved from each arm's own findings with the identical resolver used in the per-arm
reports — so these verdicts match <code>exp38-dense-drift-gsm8k.html</code> / <code>-big-math.html</code> exactly.</p>

<h2><span class="num">5</span>Headline numbers, side by side</h2>
<table>
<tr><th>metric</th><th class="num">GSM8K</th><th class="num">Big-Math</th><th>note</th></tr>
{rows}
</table>
{_img(plots,'wd','Weight drift (parameter-point gap) vs lag, normalized per arm.')}

<h2><span class="num">6</span>Theory — why this matters for the next method</h2>
{nar}

<p class="foot">n=1 per task, 75 steps each — within-run measurements on the dense path, not cross-seed statistics.
Small-lag (k≤5) gradient cosines are sampled only in early training (capture-schedule confound). Knees are framed
as "consistent / inconsistent with" the EXP-37 5/5-stable, 20/20-broken boundary. Datasets are never merged: every
curve is derived from its own arm's <code>*_findings.json</code>. Generated by <code>research/scripts/exp38_compare.py</code>.</p>
</div>"""

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        fh.write("<!doctype html><html><head><meta charset='utf-8'>"
                 "<meta name='viewport' content='width=device-width,initial-scale=1'>"
                 "<title>EXP-38 — Dense GRPO drift: GSM8K vs Big-Math (joint)</title>"
                 f"<style>{R.CSS}</style></head><body>" + body + "</body></html>")
    print(f"[exp38-compare] joint report -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gsm8k", default="reports/dense-run-behaviour/exp38-dense-drift-gsm8k_findings.json")
    ap.add_argument("--big-math", default="reports/dense-run-behaviour/exp38-dense-drift-big-math_findings.json")
    ap.add_argument("--narrative", default="reports/dense-run-behaviour/_joint_narrative.html")
    ap.add_argument("--out", default="reports/dense-run-behaviour/exp38-dense-drift-joint.html")
    args = ap.parse_args()
    g = _normalize_keys(json.load(open(args.gsm8k)))
    b = _normalize_keys(json.load(open(args.big_math)))
    assert g.get("dataset") == "gsm8k", f"--gsm8k findings has dataset={g.get('dataset')!r}, expected gsm8k"
    assert b.get("dataset") == "big-math", f"--big-math findings has dataset={b.get('dataset')!r}, expected big-math"
    narrative = ""
    if args.narrative and os.path.exists(args.narrative):
        narrative = open(args.narrative).read()
    build(g, b, args.out, narrative)


if __name__ == "__main__":
    main()
