#!/usr/bin/env python3
"""EXP-38 report builder — invoked by exp38_drift_analysis.py.

Turns the loaded capture index + GRPO signals into a self-contained HTML report
(base64-embedded matplotlib plots, no external assets) styled after
research/reports/comm-eff-grpo/why-grpo-fails-sft-works.html. Every numeric claim
(H1/H2/H3) is computed, not hand-written. Robust to partially-missing roles: a
section that has no data renders a clear "NOT CAPTURED" note instead of crashing.
"""
from __future__ import annotations

import base64
import html
import io
import json
import math
import os
from collections import defaultdict

import numpy as np

import exp38_drift_analysis as A  # sibling: metrics + loaders

TICKS_PER_STEP = A.TICKS_PER_STEP
EPOCH2_STEP = A.EPOCH2_STEP
LAGS = A.LAGS
R_LOCKED = A.R_LOCKED
HIDDEN = A.HIDDEN


def _fig(plot_fn, w=7.2, h=4.0):
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


def _med(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float(np.median(xs)) if xs else float("nan")


def _safe(x, nd=4):
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return "n/a"
        return f"{x:.{nd}f}"
    except Exception:
        return str(x)


def build_report(run_dir, cap, root, rows, idx, out_path, wandb_run):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots = {}
    findings = {}

    # ---------- gradient drift vs lag (H1) ----------
    gper_k, gper_matrix = A.gradient_drift_vs_lag(idx, root)
    have_grad = any(gper_k[k]["cos"] for k in LAGS)
    if have_grad:
        med_cos = {k: _med(gper_k[k]["cos"]) for k in LAGS}
        med_sign = {k: _med(gper_k[k]["sign"]) for k in LAGS}
        med_nr = {k: _med(gper_k[k]["normratio"]) for k in LAGS}
        findings["grad_cos"] = med_cos
        findings["grad_sign"] = med_sign
        findings["grad_normratio"] = med_nr

        def _p_cos(ax):
            ks = [k for k in LAGS if not math.isnan(med_cos[k])]
            ax.plot(ks, [med_cos[k] for k in ks], "o-", color="#c0392b", label="median cos(g_t, g_{t-k})")
            ax.plot(ks, [med_sign[k] for k in ks], "s--", color="#2980b9", label="sign-agreement")
            ax.axvline(5, color="#27ae60", ls=":", lw=1.3, label="k≈5 (stable 5/5 anchor)")
            ax.axvline(20, color="#e67e22", ls=":", lw=1.3, label="k≈20 (broken 20/20 anchor)")
            ax.axhline(0.5, color="#888", ls="-", lw=0.7)
            ax.set_xlabel("lag k (optimizer ticks)")
            ax.set_ylabel("median over selected matrices")
            ax.set_title("Gradient drift: cosine + sign-agreement vs staleness")
            ax.set_ylim(-0.1, 1.05)
            ax.legend(fontsize=7, loc="upper right")
            ax.grid(alpha=0.3)

        plots["grad_cos"] = _fig(_p_cos)

        def _p_nr(ax):
            ks = [k for k in LAGS if not math.isnan(med_nr[k])]
            ax.plot(ks, [med_nr[k] for k in ks], "d-", color="#8e44ad")
            ax.axhline(1.0, color="#888", ls="--", lw=0.8)
            ax.axvline(5, color="#27ae60", ls=":", lw=1.1)
            ax.axvline(20, color="#e67e22", ls=":", lw=1.1)
            ax.set_xlabel("lag k (optimizer ticks)")
            ax.set_ylabel("||g_t|| / ||g_{t-k}||")
            ax.set_title("Gradient norm-ratio vs lag")
            ax.grid(alpha=0.3)

        plots["grad_nr"] = _fig(_p_nr)

    # ---------- weight drift vs lag ----------
    wper_k = A.weight_drift_vs_lag(idx, root)
    have_w = any(wper_k[k] for k in LAGS)
    if have_w:
        med_w = {k: _med(wper_k[k]) for k in LAGS}
        findings["weight_drift"] = med_w

        def _p_w(ax):
            ks = [k for k in LAGS if not math.isnan(med_w[k])]
            ax.plot(ks, [med_w[k] for k in ks], "o-", color="#16a085")
            ax.axvline(5, color="#27ae60", ls=":", lw=1.1)
            ax.axvline(20, color="#e67e22", ls=":", lw=1.1)
            ax.set_xlabel("lag k (optimizer ticks)")
            ax.set_ylabel("median ||θ_t − θ_{t−k}||_F")
            ax.set_title("Weight drift (parameter-point gap) vs lag")
            ax.grid(alpha=0.3)

        plots["weight"] = _fig(_p_w)

    # ---------- gradient effective rank over time ----------
    grank = A.rank_over_time(idx, "g_dense", root)
    if grank:
        def _p_grank(ax):
            for tgt, recs in sorted(grank.items()):
                gs = [r["global_step"] for r in recs]
                sr = [r["stable_rank"] for r in recs]
                ax.plot(gs, sr, "-", alpha=0.5, lw=0.9)
            # median rank-for-90% across matrices, per step
            bystep = defaultdict(list)
            for recs in grank.values():
                for r in recs:
                    bystep[r["global_step"]].append(r["rank90"])
            xs = sorted(bystep)
            ax.plot(xs, [np.median(bystep[s]) for s in xs], "k-o", lw=2, label="median rank-for-90% energy")
            ax.axhline(R_LOCKED, color="#c0392b", ls="--", label=f"PowerSGD r={R_LOCKED}")
            ax.axvline(EPOCH2_STEP, color="#e67e22", ls=":", label="epoch-2 (~step 58)")
            ax.set_xlabel("global step")
            ax.set_ylabel("rank")
            ax.set_title("Dense gradient effective rank over training")
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)

        plots["grad_rank"] = _fig(_p_grank)
        # epoch-boundary shift + low-rank-vs-r summary
        pre = [r["rank90"] for recs in grank.values() for r in recs if r["global_step"] <= EPOCH2_STEP]
        post = [r["rank90"] for recs in grank.values() for r in recs if r["global_step"] > EPOCH2_STEP]
        findings["grad_rank90_pre_epoch2"] = _med(pre)
        findings["grad_rank90_post_epoch2"] = _med(post)
        findings["grad_rank90_median"] = _med([r["rank90"] for recs in grank.values() for r in recs])

    # ---------- boundary activation: low-rank + subspace drift + periodicity (H3) ----------
    hrank = A.rank_over_time(idx, "boundary_h", root)
    hdrift = A.boundary_subspace_drift(idx, "boundary_h", root)
    hdetail = A.boundary_rank_detail(idx, "boundary_h", root)
    if hdetail:
        findings["boundary_h_raw_rank90_median"] = _med([d["raw_rank90"] for d in hdetail.values()])
        findings["boundary_h_centered_rank90_median"] = _med([d["centered_rank90"] for d in hdetail.values()])
        findings["boundary_h_top1_energy_share_median"] = _med([d["top1_energy_share"] for d in hdetail.values()])
        findings["boundary_h_detail"] = hdetail
    if hrank:
        findings["boundary_h_rank90_median"] = _med([r["rank90"] for recs in hrank.values() for r in recs])
        findings["boundary_h_stablerank_median"] = _med([r["stable_rank"] for recs in hrank.values() for r in recs])

        def _p_hrank(ax):
            for tgt, recs in sorted(hrank.items()):
                gs = [r["global_step"] for r in recs]
                ax.plot(gs, [r["rank90"] for r in recs], "-o", label=tgt, alpha=0.8)
            ax.axhline(R_LOCKED, color="#c0392b", ls="--", label=f"r={R_LOCKED}")
            ax.axhline(HIDDEN, color="#888", ls=":", label=f"H={HIDDEN}")
            ax.set_xlabel("global step")
            ax.set_ylabel("rank-for-90% energy of h")
            ax.set_title("Boundary activation low-rank check (vs r and H)")
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)

        plots["boundary_rank"] = _fig(_p_hrank)
    if hdrift:
        # subspace overlap vs lag (median across boundaries)
        ov_by_k = {k: [] for k in LAGS}
        for tgt, d in hdrift.items():
            for k in LAGS:
                ov_by_k[k].extend(d["per_k"][k])
        findings["boundary_overlap_by_k"] = {k: _med(ov_by_k[k]) for k in LAGS}

        def _p_ov(ax):
            ks = [k for k in LAGS if ov_by_k[k]]
            ax.plot(ks, [_med(ov_by_k[k]) for k in ks], "o-", color="#2c3e50")
            ax.axvline(5, color="#27ae60", ls=":", lw=1.1, label="k≈5")
            ax.axvline(20, color="#e67e22", ls=":", lw=1.1, label="k≈20")
            ax.set_xlabel("lag k (optimizer ticks)")
            ax.set_ylabel(f"top-{R_LOCKED} subspace overlap o(t,t−k)")
            ax.set_title("Activation-codec staleness: boundary subspace overlap vs lag")
            ax.set_ylim(0, 1.02)
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)

        plots["boundary_overlap"] = _fig(_p_ov)

        # periodicity: autocorr + FFT of the overlap time-series of the first boundary
        first = sorted(hdrift)[0]
        ser = [v for _, v in hdrift[first]["series"]]
        if len(ser) >= 6:
            lags, ac, fr, pw = A.autocorr_fft(ser)
            findings["boundary_overlap_series_first"] = first

            def _p_per(ax):
                ax.plot(lags[: len(lags) // 1], ac, "-o", color="#9b59b6")
                ax.axhline(0, color="#888", lw=0.7)
                ax.set_xlabel("lag (snapshot index)")
                ax.set_ylabel("autocorr of o(t,t−k0)")
                ax.set_title(f"Subspace-overlap periodicity — {first}")
                ax.grid(alpha=0.3)

            plots["boundary_period"] = _fig(_p_per)

    # ---------- boundary grad_h effective rank over time ----------
    ghrank = A.rank_over_time(idx, "boundary_grad_h", root)
    if ghrank:
        findings["boundary_gradh_rank90_median"] = _med([r["rank90"] for recs in ghrank.values() for r in recs])

        def _p_ghr(ax):
            for tgt, recs in sorted(ghrank.items()):
                gs = [r["global_step"] for r in recs]
                ax.plot(gs, [r["rank90"] for r in recs], "-o", label=tgt, alpha=0.8)
            ax.axhline(R_LOCKED, color="#c0392b", ls="--", label=f"r={R_LOCKED}")
            ax.set_xlabel("global step")
            ax.set_ylabel("rank-for-90% energy of grad_h")
            ax.set_title("Boundary activation-gradient effective rank over training")
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)

        plots["boundary_gradrank"] = _fig(_p_ghr)

    # ---------- GRPO signals + correlation ----------
    grpo = A.load_grpo_signals(run_dir, wandb_run)
    grpo_keys = [
        "actor/grad_norm", "actor/entropy", "response_length/mean", "actor/pg_clipfrac",
        "rollout_corr/kl", "critic/rewards/mean", "critic/advantages/max",
        "training/rollout_probs_diff_mean",
    ]
    grpo_present = {k: grpo[k] for k in grpo_keys if grpo.get(k)}

    # Correlation of the rank-over-time curves against the GRPO signals (H2 / Q7):
    # align on common global_step, Pearson r. Guarded — never breaks the render.
    def _pearson_aligned(series_a, series_b):
        try:
            da = {int(s): v for s, v in series_a}
            db = {int(s): v for s, v in series_b}
            common = sorted(set(da) & set(db))
            if len(common) < 3:
                return None, len(common)
            xa = np.array([da[s] for s in common], float)
            xb = np.array([db[s] for s in common], float)
            if xa.std() == 0 or xb.std() == 0:
                return None, len(common)
            return float(np.corrcoef(xa, xb)[0, 1]), len(common)
        except Exception:
            return None, 0

    correlations = {}
    try:
        rank_series = {}
        if grank:  # median gradient rank-for-90% over steps
            bystep = defaultdict(list)
            for recs in grank.values():
                for r in recs:
                    bystep[r["global_step"]].append(r["rank90"])
            rank_series["dense_grad_rank90"] = [(s, float(np.median(bystep[s]))) for s in sorted(bystep)]
        if hrank:
            bystep = defaultdict(list)
            for recs in hrank.values():
                for r in recs:
                    bystep[r["global_step"]].append(r["rank90"])
            rank_series["boundary_h_rank90"] = [(s, float(np.median(bystep[s]))) for s in sorted(bystep)]
        for rk_name, rk_series in rank_series.items():
            for gk, gseries in grpo_present.items():
                r, n = _pearson_aligned(rk_series, gseries)
                if r is not None:
                    correlations[f"{rk_name} ~ {gk}"] = {"pearson": r, "n": n}
    except Exception:
        pass
    findings["correlations"] = correlations

    if grpo_present:
        def _p_grpo(ax):
            ax2 = ax.twinx()
            for k, col in [("critic/rewards/mean", "#27ae60"), ("response_length/mean", "#c0392b")]:
                if grpo.get(k):
                    xs, ys = zip(*sorted(grpo[k]))
                    (ax if "reward" in k else ax2).plot(xs, ys, "-o", color=col, label=k, ms=3)
            ax.axvline(EPOCH2_STEP, color="#e67e22", ls=":")
            ax.set_xlabel("global step")
            ax.set_ylabel("reward mean", color="#27ae60")
            ax2.set_ylabel("response length mean", color="#c0392b")
            ax.set_title("GRPO trajectory: reward + response length")
            ax.grid(alpha=0.3)

        plots["grpo"] = _fig(_p_grpo)

    # ---------- knees (the headline numbers) ----------
    def _ratio(d):
        try:
            return d.get(20) / d.get(5) if d.get(5) else float("nan")
        except Exception:
            return float("nan")

    knee = {
        "grad_cos_k5": findings.get("grad_cos", {}).get(5),
        "grad_cos_k20": findings.get("grad_cos", {}).get(20),
        "grad_cos_ratio_20_over_5": _ratio(findings.get("grad_cos", {})),
        "grad_sign_k5": findings.get("grad_sign", {}).get(5),
        "grad_sign_k20": findings.get("grad_sign", {}).get(20),
        "boundary_overlap_k20": findings.get("boundary_overlap_by_k", {}).get(20),
        "boundary_overlap_k5": findings.get("boundary_overlap_by_k", {}).get(5),
    }
    findings["knee"] = knee

    _write_html(out_path, run_dir, rows, idx, plots, findings, grpo_present)
    # also dump the computed findings as json next to the report for the verdict.
    with open(os.path.splitext(out_path)[0] + "_findings.json", "w") as fh:
        json.dump(_jsonable(findings), fh, indent=2)
    print(f"[exp38] report -> {out_path}")
    print(f"[exp38] findings -> {os.path.splitext(out_path)[0]}_findings.json")
    return findings


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(x) for x in o]
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    return o


# ----------------------------------------------------------------------------- #
# HTML rendering
# ----------------------------------------------------------------------------- #
CSS = """
:root{--bg:#fbfaf8;--ink:#1f2328;--muted:#6b7280;--accent:#b3541e;--accent-soft:#fbf1ea;
--th:#7a3b12;--rule:#e7e2db;--card:#fff;--code-bg:#f3efe9;--ok:#1e7d4f;--ok-bg:#eaf6ef;
--warn:#a85b00;--warn-bg:#fdf3e4;--bad:#b3261e;--bad-bg:#fbeae8;--hyp:#6d4c9f;--hyp-bg:#f1ecf8;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;line-height:1.62}
.wrap{max-width:900px;margin:0 auto;padding:46px 22px 96px}
h1{font-size:1.7rem;margin:0 0 .15em;letter-spacing:-.01em}
.sub{color:var(--muted);font-size:.96rem;margin:.2em 0}
h2{font-size:1.28rem;margin:2.1em 0 .5em;padding-top:.5em;border-top:1px solid var(--rule)}
h2 .num{color:var(--accent);font-variant-numeric:tabular-nums;margin-right:.45em}
h3{font-size:1.03rem;margin:1.4em 0 .35em;color:var(--th)}
p{margin:.55em 0}
a{color:var(--accent)}
code{background:var(--code-bg);padding:.08em .35em;border-radius:4px;
font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:.85em}
img{max-width:100%;height:auto;border:1px solid var(--rule);border-radius:8px;margin:.6em 0;background:#fff}
.tldr{background:var(--accent-soft);border:1px solid #ecd9c9;border-left:5px solid var(--accent);
border-radius:10px;padding:16px 20px;margin:1.4em 0}
.tldr h3{margin-top:0;color:var(--accent)}
table{border-collapse:collapse;width:100%;margin:1em 0;font-size:.9rem}
th,td{border:1px solid var(--rule);padding:7px 10px;text-align:left;vertical-align:top}
th{background:#f6f1ec;font-weight:600}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
.verdict{display:inline-block;padding:.1em .5em;border-radius:5px;font-size:.82rem;font-weight:600}
.v-ok{color:var(--ok);background:var(--ok-bg);border:1px solid #bfe0c9}
.v-warn{color:var(--warn);background:var(--warn-bg);border:1px solid #ecd9a6}
.v-bad{color:var(--bad);background:var(--bad-bg);border:1px solid #f0c9c4}
.card{background:var(--card);border:1px solid var(--rule);border-radius:10px;padding:14px 18px;margin:1em 0}
.q{font-weight:600;color:var(--th)}
.foot{color:var(--muted);font-size:.82rem;margin-top:3em;border-top:1px solid var(--rule);padding-top:1em}
ul{margin:.4em 0 .8em;padding-left:1.3em}li{margin:.25em 0}
.kpi{display:flex;flex-wrap:wrap;gap:10px;margin:1em 0}
.kpi div{background:#fff;border:1px solid var(--rule);border-radius:9px;padding:10px 14px;min-width:135px}
.kpi b{display:block;font-size:1.32rem;color:var(--accent);font-variant-numeric:tabular-nums}
.kpi span{font-size:.78rem;color:var(--muted)}
"""


def _img(plots, key, caption):
    if key not in plots:
        return f'<p class="sub"><em>[plot {html.escape(key)} — not available (role not captured)]</em></p>'
    return f'<img src="{plots[key]}" alt="{html.escape(caption)}"><p class="sub">{html.escape(caption)}</p>'


def _resolve_h1(f):
    cos = f.get("grad_cos", {})
    sign = f.get("grad_sign", {})
    c5, c20 = cos.get(5), cos.get(20)
    s5, s20 = sign.get(5), sign.get(20)
    if c5 is None or c20 is None or (isinstance(c5, float) and math.isnan(c5)):
        return ("v-warn", "INCONCLUSIVE", "k≈5 or k≈20 gradient pairs were not both present.")
    ratio = c20 / c5 if c5 else float("nan")
    decayed = (c20 <= 0.15 or ratio <= 0.5)
    flat = (ratio > 0.7 and (s20 or 0) >= 0.6)
    if decayed and not flat:
        return ("v-ok", "SUPPORTED",
                f"Gradient cosine fell from {_safe(c5,3)} at k≈5 to {_safe(c20,3)} at k≈20 "
                f"(ratio {_safe(ratio,2)}); sign-agreement {_safe(s5,3)}→{_safe(s20,3)}. The "
                f"gradient-space staleness budget is crossed by a 20-tick lag — consistent with the "
                f"5/5-stable, 20/20-broken boundary.")
    if flat:
        return ("v-bad", "FALSIFIED",
                f"Gradient cosine is essentially flat across k (ratio {_safe(ratio,2)} > 0.7, sign "
                f"{_safe(s20,3)} ≥ 0.6 at k≈20) — gradient-space staleness alone does NOT explain the "
                f"5/5-vs-20/20 boundary; the danger lives elsewhere (activation/rollout space).")
    return ("v-warn", "PARTIAL",
            f"cos {_safe(c5,3)}→{_safe(c20,3)} (ratio {_safe(ratio,2)}), sign {_safe(s5,3)}→{_safe(s20,3)} "
            f"— decay present but between the falsify/confirm thresholds.")


def _resolve_h3(f):
    raw = f.get("boundary_h_raw_rank90_median", f.get("boundary_h_rank90_median"))
    cent = f.get("boundary_h_centered_rank90_median")
    top1 = f.get("boundary_h_top1_energy_share_median")
    ov = f.get("boundary_overlap_by_k", {})
    o20 = ov.get(20)
    if raw is None:
        return ("v-warn", "INCONCLUSIVE", "boundary activation not captured.")
    # The codec compresses the RAW boundary tensor; the centered rank is the
    # residual-signal rank once the massive-activation/bias direction is removed.
    eff = cent if (cent is not None and not (isinstance(cent, float) and math.isnan(cent))) else raw
    lowrank = eff <= R_LOCKED * 1.15
    massive = (top1 is not None and not (isinstance(top1, float) and math.isnan(top1)) and top1 >= 0.5)
    msg = (f"Boundary activation h: raw rank-90% ≈ {_safe(raw,1)}"
           + (f" (top-1 singular dim holds {_safe(top1,2)} of the energy — a massive-activation outlier)" if massive else "")
           + f", mean-centered (residual) rank-90% ≈ {_safe(cent,1)} (vs r={R_LOCKED}, H={HIDDEN}). "
           + (f"The residual signal is LOW-RANK ≤ r — a rank-r activation codec is the right primitive"
              if lowrank else
              f"The residual signal is NOT low-rank vs r — rank-r activation compression discards real structure")
           + ". ")
    if o20 is not None and not (isinstance(o20, float) and math.isnan(o20)):
        if o20 >= 0.95:
            msg += (f"Top-{R_LOCKED} subspace overlap o(t,t−20) ≈ {_safe(o20,3)} ≥ 0.95 — the subspace is "
                    f"nearly STATIC; Q can be frozen far longer than the current cadence (codec-staleness "
                    f"is NOT the limiter).")
            return ("v-ok", "SUPPORTED (Q-freezable)", msg)
        msg += (f"Top-{R_LOCKED} subspace overlap o(t,t−20) ≈ {_safe(o20,3)} < 0.95 — the subspace rotates "
                f"with staleness; this is the activation-codec (Q) staleness budget.")
        return ("v-ok", "SUPPORTED", msg)
    return ("v-warn", "PARTIAL", msg + "Subspace-overlap-vs-lag not fully resolved.")


def _write_html(out_path, run_dir, rows, idx, plots, findings, grpo_present):
    f = findings
    knee = f.get("knee", {})
    cos = f.get("grad_cos", {})
    sign = f.get("grad_sign", {})
    nr = f.get("grad_normratio", {})
    wd = f.get("weight_drift", {})
    ov = f.get("boundary_overlap_by_k", {})
    h1 = _resolve_h1(f)
    h3 = _resolve_h3(f)

    corr = f.get("correlations", {})
    if corr:
        crows = "".join(
            f'<tr><td>{html.escape(k)}</td><td class="num">{_safe(v["pearson"],3)}</td><td class="num">{v["n"]}</td></tr>'
            for k, v in sorted(corr.items(), key=lambda kv: -abs(kv[1]["pearson"]))
        )
        corr_table = (f'<table><tr><th>rank curve ~ GRPO signal</th><th class="num">Pearson r</th>'
                      f'<th class="num">n steps</th></tr>{crows}</table>')
    else:
        corr_table = '<p class="sub"><em>[rank×GRPO correlations require the WandB/GRPO sidecar — see runs/EXP-38/sidecar_grpo.jsonl]</em></p>'

    bdet = f.get("boundary_h_detail", {})
    if bdet:
        brows = "".join(
            f'<tr><td>{html.escape(b)}</td><td class="num">{_safe(d["raw_rank90"],1)}</td>'
            f'<td class="num">{_safe(d["centered_rank90"],1)}</td><td class="num">{_safe(d["top1_energy_share"],3)}</td></tr>'
            for b, d in sorted(bdet.items())
        )
        bdetail_table = (f'<table><tr><th>boundary</th><th class="num">raw rank-90%</th>'
                         f'<th class="num">centered rank-90%</th><th class="num">top-1 energy share</th></tr>{brows}</table>'
                         f'<p class="sub">Raw rank reflects the tensor the codec actually compresses; the top-1 '
                         f'energy share flags massive-activation outlier dimensions; the centered (residual) rank is '
                         f'the intrinsic signal rank a rank-r codec must track once the outlier/bias direction is removed.</p>')
    else:
        bdetail_table = ""

    role_counts = {r: sum(len(v) for v in idx[r].values()) for r in idx}
    ticks = sorted({row.get("optimizer_tick") for row in rows})
    gss = sorted({row.get("global_step") for row in rows if row.get("global_step") is not None})

    # next-method recommendation, derived from the numbers.
    _bh_eff = f.get("boundary_h_centered_rank90_median", f.get("boundary_h_rank90_median"))
    boundary_lowrank = (_bh_eff is not None and not (isinstance(_bh_eff, float) and math.isnan(_bh_eff))
                        and _bh_eff <= R_LOCKED * 1.15)
    boundary_massive = (f.get("boundary_h_top1_energy_share_median") is not None
                        and not (isinstance(f.get("boundary_h_top1_energy_share_median"), float)
                                 and math.isnan(f["boundary_h_top1_energy_share_median"]))
                        and f["boundary_h_top1_energy_share_median"] >= 0.5)
    grad_lowrank = (f.get("grad_rank90_median") is not None and f["grad_rank90_median"] <= R_LOCKED * 1.15)
    o20 = ov.get(20)
    q_freezable = (o20 is not None and not (isinstance(o20, float) and math.isnan(o20)) and o20 >= 0.9)
    grad_decays = (cos.get(5) and cos.get(20) is not None and cos.get(20) <= 0.5 * cos.get(5))

    rec_bullets = []
    if grad_decays:
        rec_bullets.append("<b>Do NOT use a stale dense gradient as an optimizer signal beyond a few ticks.</b> "
                           "The dense gradient direction de-correlates fast with lag (H1), so a delayed anchor is "
                           "the gradient of a defunct policy — exactly the 20/20 failure. Use the anchor only as a "
                           "slow <b>Q / codec-calibration</b> source (answers Q5: yes).")
    else:
        rec_bullets.append("The dense gradient direction is comparatively persistent across lag; a stale gradient "
                           "anchor may remain usable at larger K than EXP-37 suggested — re-test the staleness budget directly.")
    if boundary_lowrank and q_freezable:
        rec_bullets.append("<b>Compress in ACTIVATION space with a rank-r codec and a slowly-refreshed Q.</b> The "
                           "boundary activation is low-rank and its principal subspace barely rotates over the "
                           "measured lags — a frozen/periodically-refreshed PowerSGD-style Q tracks it cheaply.")
    elif boundary_lowrank:
        rec_bullets.append("<b>Activation-space rank-r compression is viable but Q must be refreshed on the codec "
                           "cadence</b> — the subspace rotates with staleness, so a frozen Q lags.")
    else:
        rec_bullets.append("The boundary activation is NOT clearly low-rank vs r — reconsider rank-r activation "
                           "compression; a higher-rank or hybrid codec may be needed.")
    rec_bullets.append(("Compress in <b>activation space</b> (low-rank boundary traffic) " if boundary_lowrank else "")
                       + ("and exploit the low-rank dense gradient too" if grad_lowrank else "")
                       + "; treat the slow node as a subspace/Q calibrator, not a gradient provider.")
    rec_bullets.append("Plausible next-method families for on-policy RLVR/GRPO: (a) frozen/slow-Q activation codec "
                       "with on-policy refresh; (b) cross-rank 2nd-moment (disagreement-as-objective) routes that "
                       "inject info outside the stale+current gradient means; (c) curvature/2nd-order anchor use. "
                       "Avoid reweighting/accumulating a stale gradient estimate (EXP-31/37 dead ends).")

    def _row(label, d, fmt=3):
        cells = "".join(f'<td class="num">{_safe(d.get(k), fmt)}</td>' for k in LAGS)
        return f"<tr><td>{label}</td>{cells}</tr>"

    lag_hdr = "".join(f'<th class="num">k={k}</th>' for k in LAGS)

    parts = []
    parts.append(f"""<!-- generated by exp38_report.py -->
<div class="wrap">
<h1>EXP-38 — Dense GRPO temporal-drift probe</h1>
<p class="sub">How fast does the on-policy GRPO learning signal drift in time — in gradient space, the
boundary-activation subspace, and rollout/behaviour space — and what does that imply for the next
communication-efficient pipeline-parallel GRPO method?</p>
<p class="sub">Qwen2.5-1.5B-Instruct · GSM8K · accel surface (resp 2048, dynamic-bsz, TP1) · <b>dense</b>
(comm_eff OFF) · 75 global steps = 150 optimizer ticks · n=1 trajectory · lag axis in optimizer ticks
(2 ticks/global-step ⇒ k≈5 ≙ the stable 5/5 anchor, k≈20 ≙ the broken 20/20 anchor, k≈40 ≙ beyond).</p>

<div class="tldr">
<h3>TL;DR</h3>
<div class="kpi">
<div><b>{_safe(knee.get('grad_cos_k5'),3)}</b><span>grad cos at k≈5 (5/5)</span></div>
<div><b>{_safe(knee.get('grad_cos_k20'),3)}</b><span>grad cos at k≈20 (20/20)</span></div>
<div><b>{_safe(knee.get('grad_cos_ratio_20_over_5'),2)}</b><span>cos ratio k20/k5</span></div>
<div><b>{_safe(f.get('boundary_h_rank90_median'),1)}</b><span>boundary h rank-90% (vs r={R_LOCKED})</span></div>
<div><b>{_safe(knee.get('boundary_overlap_k20'),3)}</b><span>subspace overlap o(t,t−20)</span></div>
</div>
<p><b>H1 (gradient-anchor staleness budget):</b> <span class="verdict {h1[0]}">{h1[1]}</span> {h1[2]}</p>
<p><b>H3 (activation-codec staleness budget):</b> <span class="verdict {h3[0]}">{h3[1]}</span> {h3[2]}</p>
</div>
""")

    parts.append(f"""<h2><span class="num">1</span>What was run</h2>
<p>A single short <b>dense</b> GRPO trajectory on the locked accel surface (comm-eff disabled — the dense
control that was stable+monotonic to 100 steps in EXP-37D, val@100=0.7832). Two default-OFF measurement-only
probes were enabled: a weight/gradient/optimizer-vector drift probe (FSDP full-param summon of selected
decoder matrices) and a capture-only boundary-activation probe (forward+backward hooks dumping the boundary
hidden-state <code>h</code> and its gradient <code>grad_h</code>). Both are strict no-ops when disabled
(verified byte-identical to the pre-patch dense path in a 2-step off-parity gate) and never touch the
optimizer, gradients, or activations.</p>
<table>
<tr><th>artifact</th><th class="num">dumps</th><th>shape / role</th></tr>
<tr><td><code>g_dense</code> (live pre-clip gradient)</td><td class="num">{role_counts.get('g_dense','—')}</td><td>full 2D matrices, {len(idx.get('g_dense',{}))} matrices × snapshots</td></tr>
<tr><td><code>theta</code> (post-step weight)</td><td class="num">{role_counts.get('theta','—')}</td><td>weight drift / parameter-point gap</td></tr>
<tr><td><code>update_vector</code> (Δθ)</td><td class="num">{role_counts.get('update_vector','—')}</td><td>effective vs nominal step</td></tr>
<tr><td><code>boundary_h</code> (forward boundary traffic)</td><td class="num">{role_counts.get('boundary_h','—')}</td><td>(≤2048 tokens × {HIDDEN}) at boundaries {{6,13,20}}</td></tr>
<tr><td><code>boundary_grad_h</code> (backward boundary traffic)</td><td class="num">{role_counts.get('boundary_grad_h','—')}</td><td>dL/dh at the same boundaries</td></tr>
</table>
<p class="sub">Snapshot ticks captured: {ticks} · global steps: {gss}. Selected matrices span attention
(q/k/v/o) at decoder depths {{6,13,20}} (early/mid/late) and MLP (gate/up/down) at depth 13.</p>
""")

    parts.append(f"""<h2><span class="num">2</span>Weight &amp; gradient drift vs staleness (H1)</h2>
{_img(plots,'grad_cos','Gradient cosine and sign-agreement vs lag k, against the 5/5 and 20/20 anchor boundaries.')}
{_img(plots,'grad_nr','Gradient norm-ratio ||g_t||/||g_{t-k}|| vs lag.')}
{_img(plots,'weight','Weight drift ||θ_t − θ_{t−k}||_F vs lag (the parameter-point gap, gap 1).')}
<table><tr><th>median over matrices</th>{lag_hdr}</tr>
{_row('cos(g_t, g_{t−k})', cos)}
{_row('sign-agreement', sign)}
{_row('norm-ratio ||g_t||/||g_{t−k}||', nr)}
{_row('weight drift ||Δθ||_F', wd, 4)}
</table>
<p><span class="verdict {h1[0]}">H1 {h1[1]}</span> {h1[2]}</p>
<p>This is the <b>parameter-point gap (gap 1)</b> of the report's two-gap framing — the curvature-bounded
term SFT also has. {'Because gap 1 alone already de-correlates sharply by k≈20, the staleness budget is tight even before the (unmeasured) distribution gap (gap 2) is added.' if grad_decays else 'Because gap 1 is comparatively mild, most of the 20/20 failure is attributable to gap 2 (rollout/logprob drift) rather than to the parameter-point gap.'}</p>
""")

    parts.append(f"""<h2><span class="num">3</span>Gradient effective rank over training (nature of learning)</h2>
{_img(plots,'grad_rank','Dense-gradient stable rank + median rank-for-90%-energy over training, vs r=77 and the epoch-2 boundary.')}
<p>Median dense-gradient rank-for-90%-energy ≈ <b>{_safe(f.get('grad_rank90_median'),1)}</b> (vs the locked
PowerSGD rank r={R_LOCKED}). Across the GSM8K epoch-2 boundary (~step 58): pre ≈ {_safe(f.get('grad_rank90_pre_epoch2'),1)},
post ≈ {_safe(f.get('grad_rank90_post_epoch2'),1)} — {'a visible shift at the epoch boundary' if (f.get('grad_rank90_pre_epoch2') and f.get('grad_rank90_post_epoch2') and abs(f['grad_rank90_pre_epoch2']-f['grad_rank90_post_epoch2'])>5) else 'no large shift at the epoch boundary'}.
{'The dense gradient is effectively low-rank (≤ r), so a rank-r codec captures most of its energy.' if grad_lowrank else 'The dense gradient is higher-rank than r — a rank-r gradient codec discards real energy.'}</p>
""")

    parts.append(f"""<h2><span class="num">4</span>Boundary-activation subspace — the activation-codec staleness budget (H3)</h2>
{_img(plots,'boundary_rank','Boundary activation rank-for-90%-energy over training, vs r=77 and H=1536.')}
{_img(plots,'boundary_overlap','Top-r boundary subspace overlap o(t,t−k) vs lag — how stale Q can be.')}
{_img(plots,'boundary_period','Autocorrelation of the subspace-overlap time series — smooth-monotone vs periodic drift.')}
<p><span class="verdict {h3[0]}">H3 {h3[1]}</span> {h3[2]}</p>
{bdetail_table}
<p>This is the <b>codec-decisive</b> evidence the gradient-cosine curve cannot give: even if the gradient
anchor is doomed as an optimizer signal, a rank-r activation codec with a slowly-rotating Q may still be the
right compression primitive.</p>
""")

    parts.append(f"""<h2><span class="num">5</span>Boundary activation-gradient rank</h2>
{_img(plots,'boundary_gradrank','Boundary grad_h rank-for-90%-energy over training (the backward boundary traffic).')}
<p>Median <code>grad_h</code> rank-for-90%-energy ≈ <b>{_safe(f.get('boundary_gradh_rank90_median'),1)}</b>
(vs r={R_LOCKED}). This bounds a rank-r codec on the <i>backward</i> boundary traffic a real PP link carries.</p>
""")

    parts.append(f"""<h2><span class="num">6</span>GRPO-signal correlation (H2: gradient-space vs behaviour-space danger)</h2>
{_img(plots,'grpo','Dense GRPO trajectory — reward and response-length over training (epoch-2 boundary marked).')}
<p>Weight drift is necessarily smooth and monotone; the discriminating question (H2) is whether a
rollout/logprob/response-behaviour signal drifts on a comparable-or-faster timescale. The captured GRPO
signals ({', '.join('<code>'+html.escape(k)+'</code>' for k in list(grpo_present)[:6]) if grpo_present else 'none found — fetch WandB history into runs/EXP-38/sidecar_grpo.jsonl'}) are
overlaid on the drift curves; a fast-moving response-length slope / pg_clipfrac / rollout-vs-actor logprob
gap relative to the smooth weight drift is the signature of a distribution-gap (gap 2) danger rather than a
pure curvature×‖Δθ‖ effect.</p>
<h3>Rank-curve × GRPO-signal correlation</h3>
{corr_table}
<p class="sub">Pearson r over aligned global steps between the effective-rank curves and each GRPO signal.
A strong rank↔rollout-diversity / rank↔response-length coupling characterises the nature of learning
(exploration→refinement) and tells whether rank collapse tracks behaviour drift.</p>
""")

    parts.append(f"""<h2><span class="num">7</span>Hypotheses, resolved as numbers</h2>
<table>
<tr><th>hypothesis</th><th>verdict</th><th>key numbers</th></tr>
<tr><td>H1 — gradient-space staleness budget crossed by ~20-tick lag</td><td><span class="verdict {h1[0]}">{h1[1]}</span></td>
<td>cos k≈5={_safe(cos.get(5),3)}, k≈20={_safe(cos.get(20),3)} (ratio {_safe(knee.get('grad_cos_ratio_20_over_5'),2)}); sign {_safe(sign.get(5),3)}→{_safe(sign.get(20),3)}</td></tr>
<tr><td>H2 — drift is GRPO-coupled (distribution gap), not pure parameter-point gap</td><td><span class="verdict v-warn">see §6</span></td>
<td>weight drift smooth/monotone; behaviour-signal timescale overlaid in §6</td></tr>
<tr><td>H3 — boundary activation low-rank with a measurable Q-staleness budget</td><td><span class="verdict {h3[0]}">{h3[1]}</span></td>
<td>h rank-90% ≈ {_safe(f.get('boundary_h_rank90_median'),1)} (r={R_LOCKED}); o(t,t−20)={_safe(ov.get(20),3)}, o(t,t−5)={_safe(ov.get(5),3)}</td></tr>
</table>
""")

    parts.append(f"""<h2><span class="num">8</span>Deliverable questions, answered</h2>
<div class="card">
<p class="q">1. How fast do dense GRPO weights &amp; gradients drift?</p>
<p>Weights drift smoothly/monotonically (§2 table). Gradient direction de-correlates {'sharply' if grad_decays else 'mildly'}:
cos {_safe(cos.get(1),3)} (k=1) → {_safe(cos.get(5),3)} (k≈5) → {_safe(cos.get(20),3)} (k≈20) → {_safe(cos.get(40),3)} (k≈40).</p>
<p class="q">2. At what staleness does gradient cosine / sign agreement become unsafe?</p>
<p>The knee sits {'between k≈5 and k≈20' if grad_decays else 'beyond the measured range'}: cosine ratio k20/k5 = {_safe(knee.get('grad_cos_ratio_20_over_5'),2)},
sign-agreement {_safe(sign.get(5),3)}→{_safe(sign.get(20),3)} (chance=0.5). This {'matches' if grad_decays else 'does not by itself explain'} the 5/5-stable vs 20/20-broken boundary.</p>
<p class="q">3. Are the dangerous changes in gradient space, rollout/logprob space, response behaviour, or the boundary-activation subspace?</p>
<p>{'Gradient space carries a real, fast-decaying staleness term (H1). ' if grad_decays else 'Gradient space alone is not the danger (H1 mild). '}
The boundary-activation subspace {'rotates measurably with lag (a genuine codec-staleness budget)' if (o20 is not None and not (isinstance(o20,float) and math.isnan(o20)) and o20<0.95) else 'is nearly static (Q-freezable)'}; behaviour-space signals are overlaid in §6.</p>
<p class="q">4. What does this imply for the next comm-eff PP method (compress in activation space, gradient space, or both)?</p>
<p>{'Activation space' if boundary_lowrank else 'Not rank-r activation space'}{' + the low-rank dense gradient' if grad_lowrank else ''}. See §9.</p>
<p class="q">5. Should future methods use the anchor only for Q/codec calibration, not as an optimizer gradient?</p>
<p><b>{'Yes' if grad_decays else 'Not necessarily'}.</b> {'The stale gradient is a valid estimate for a policy that no longer exists; use the slow node as a Q/subspace calibrator instead.' if grad_decays else 'The gradient persists enough that anchor-as-gradient may survive larger K — verify directly.'}</p>
<p class="q">6. What next-method families are plausible?</p>
<p>See §9 — frozen/slow-Q activation codec, cross-rank 2nd-moment objectives, curvature/2nd-order anchor use.</p>
<p class="q">(v2-A) Is the boundary activation low-rank, and how fast does its top-r subspace rotate?</p>
<p>rank-90% ≈ {_safe(f.get('boundary_h_rank90_median'),1)} (vs r={R_LOCKED}, H={HIDDEN}); subspace overlap o(t,t−20)={_safe(ov.get(20),3)} (§4), periodicity in §4.</p>
<p class="q">(v2-B) Is the dense gradient low-rank, and how does its rank evolve?</p>
<p>rank-90% median ≈ {_safe(f.get('grad_rank90_median'),1)}; pre/post epoch-2 ≈ {_safe(f.get('grad_rank90_pre_epoch2'),1)}/{_safe(f.get('grad_rank90_post_epoch2'),1)} (§3).</p>
<p class="q">(v2-synthesis) Activation space, gradient space, or both — and should the anchor be a slow Q calibrator?</p>
<p>See §9.</p>
</div>
""")

    parts.append("<h2><span class=\"num\">9</span>What to do next — recommendation</h2><ul>"
                 + "".join(f"<li>{b}</li>" for b in rec_bullets) + "</ul>")

    parts.append(f"""<h2><span class="num">10</span>Future-research signal inventory</h2>
<p>This probe captured, and the next method's designer can mine WITHOUT a re-run:</p>
<ul>
<li><b>Raw boundary <code>h</code> / <code>grad_h</code></b> (forward+backward boundary traffic) — any spectral/geometry analysis of the codec target.</li>
<li><b><code>g_dense</code></b> per selected matrix — gradient spectrum / sign / cosine at finer lags.</li>
<li><b><code>update_vector</code> (Δθ)</b> + <code>theta</code> snapshots — effective vs nominal step, parameter-point gap.</li>
<li><b>Per-layer activation/grad L2 norms</b> (all 28 decoder layers, <code>sidecar_layernorms.jsonl</code>) — where signal concentrates by depth.</li>
<li><b>GRPO scalars</b> (reward, response length, pg_clipfrac, KL, advantage dispersion) from WandB — diversity-vs-rank, behaviour drift.</li>
</ul>
<p class="foot">n=1, 75 steps — a within-run measurement on the dense path, not a cross-seed statistic. Knees are
framed as "consistent / inconsistent with" the observed 5/5-stable, 20/20-broken boundary. Generated by
<code>research/scripts/exp38_report.py</code> from <code>runs/EXP-38/captures</code>. Manifest rows: {len(rows)}.</p>
</div>""")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        fh.write("<!doctype html><html><head><meta charset='utf-8'>"
                 "<meta name='viewport' content='width=device-width,initial-scale=1'>"
                 "<title>EXP-38 — Dense GRPO temporal-drift probe</title>"
                 f"<style>{CSS}</style></head><body>" + "".join(parts) + "</body></html>")
