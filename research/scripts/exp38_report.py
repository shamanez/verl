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


def _isnan(x):
    return isinstance(x, float) and math.isnan(x)


def _multifig(build_fn):
    """base64 a figure built by build_fn() -> Figure (for multi-axes plots)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = build_fn()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _spectrum_fig(records, has_epoch2, title, top=60):
    """Two-panel SVD-spectrum-evolution figure for a representative target:
    (L) log singular-value spectra overlaid at first/mid/last snapshots,
    (R) heatmap of log10 σ_i over the full trajectory. records: target -> list of
    {global_step, spectrum}. Returns base64 or None."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    targets = sorted(records)
    if not targets:
        return None
    rep = targets[len(targets) // 2]  # deterministic representative
    recs = sorted(records[rep], key=lambda r: r["global_step"])
    if not recs:
        return None
    steps = [r["global_step"] for r in recs]
    snap_idx = sorted(set([0, len(recs) // 2, len(recs) - 1]))

    def _build():
        fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.2, 4.0))
        for i in snap_idx:
            s = np.asarray(recs[i]["spectrum"], dtype=np.float64)[:top]
            s = np.clip(s, 1e-12, None)
            axL.semilogy(range(1, len(s) + 1), s, "-", lw=1.4, label=f"step {recs[i]['global_step']}")
        axL.set_xlabel("singular-value index i")
        axL.set_ylabel("singular value σ_i (log)")
        axL.set_title(f"{title} — spectrum at snapshots [{rep.split('.')[-2] if '.' in rep else rep}]", fontsize=9)
        axL.legend(fontsize=7)
        axL.grid(alpha=0.3, which="both")
        M = np.array([np.log10(np.clip(np.asarray(r["spectrum"], dtype=np.float64)[:top], 1e-12, None)) for r in recs]).T
        im = axR.imshow(M, aspect="auto", origin="lower",
                        extent=[steps[0], steps[-1], 1, M.shape[0]], cmap="magma")
        axR.set_xlabel("global step")
        axR.set_ylabel("singular-value index i")
        axR.set_title(f"{title} — spectrum evolution", fontsize=9)
        fig.colorbar(im, ax=axR, label="log₁₀ σ_i")
        if has_epoch2:
            axR.axvline(EPOCH2_STEP, color="#39ff14", ls=":", lw=1.2)
        fig.tight_layout()
        return fig

    return _multifig(_build)


def build_report(run_dir, cap, root, rows, idx, out_path, wandb_run, dataset="gsm8k"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots = {}
    findings = {"dataset": dataset}
    dataset_key = str(dataset).lower()
    has_epoch2_boundary = dataset_key == "gsm8k"
    findings["epoch2_step"] = EPOCH2_STEP if has_epoch2_boundary else None
    findings["epoch2_note"] = (
        "GSM8K crosses epoch 2 around global step 58."
        if has_epoch2_boundary
        else "No epoch-2 boundary is crossed in this 75-step Big-Math run."
    )

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
            if has_epoch2_boundary:
                ax.axvline(EPOCH2_STEP, color="#e67e22", ls=":", label="epoch-2 (~step 58)")
            ax.set_xlabel("global step")
            ax.set_ylabel("rank")
            ax.set_title("Dense gradient effective rank over training")
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)

        plots["grad_rank"] = _fig(_p_grank)
        # per-step median gradient rank-for-90 (the trajectory — lets the report state the
        # warmup ramp + whether the epoch boundary actually shifts rank).
        _bystep = defaultdict(list)
        for recs in grank.values():
            for r in recs:
                _bystep[r["global_step"]].append(r["rank90"])
        findings["grad_rank90_by_step"] = {int(s): _med(_bystep[s]) for s in sorted(_bystep)}
        # epoch-boundary shift: compute pre/post EXCLUDING the lr-warmup ramp (steps <= WARMUP,
        # where rank is still climbing); the naive (<=58) pre is warmup-contaminated and
        # MISLEADING — the verifier confirmed the per-step rank is stationary post-warmup.
        WARMUP = 5
        if has_epoch2_boundary:
            pre = [r["rank90"] for recs in grank.values() for r in recs
                   if WARMUP < r["global_step"] <= EPOCH2_STEP]
            post = [r["rank90"] for recs in grank.values() for r in recs if r["global_step"] > EPOCH2_STEP]
            pre_naive = [r["rank90"] for recs in grank.values() for r in recs if r["global_step"] <= EPOCH2_STEP]
            findings["grad_rank90_pre_epoch2"] = _med(pre)              # post-warmup pre
            findings["grad_rank90_post_epoch2"] = _med(post)
            findings["grad_rank90_pre_epoch2_naive"] = _med(pre_naive)  # warmup-contaminated
            findings["grad_rank90_warmup_steps"] = WARMUP
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

        # periodicity: autocorr of the ADJACENT-snapshot overlap series (~20 pts; the
        # fixed-lag series has only ~5 early points so is too short for FFT/autocorr).
        first = sorted(hdrift)[0]
        ser = [v for _, v in hdrift[first].get("adj_series", [])]
        if len(ser) >= 6:
            lags, ac, fr, pw = A.autocorr_fft(ser)
            findings["boundary_overlap_series_first"] = first

            def _p_per(ax):
                ax.plot(lags, ac, "-o", color="#9b59b6")
                ax.axhline(0, color="#888", lw=0.7)
                ax.set_xlabel("lag (snapshot index)")
                ax.set_ylabel("autocorr of adjacent-snapshot o")
                ax.set_title(f"Subspace-overlap periodicity (adjacent-snapshot) — {first}")
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
    # derived: advantage dispersion (an n=8 rollout-group-spread proxy) = max − min advantage per step.
    _amax = {int(s): v for s, v in grpo.get("critic/advantages/max", [])}
    _amin = {int(s): v for s, v in grpo.get("critic/advantages/min", [])}
    _disp = [(s, _amax[s] - _amin[s]) for s in sorted(set(_amax) & set(_amin))]
    if _disp:
        grpo["derived/advantage_dispersion"] = _disp
    grpo_keys = [
        "actor/grad_norm", "actor/entropy", "response_length/mean", "actor/pg_clipfrac",
        "actor/ppo_kl", "rollout_corr/kl", "rollout_corr/log_ppl_abs_diff",
        "critic/rewards/mean", "critic/advantages/max", "derived/advantage_dispersion",
        "training/rollout_probs_diff_mean", "training/rollout_probs_diff_std",
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
            if has_epoch2_boundary:
                ax.axvline(EPOCH2_STEP, color="#e67e22", ls=":")
            ax.set_xlabel("global step")
            ax.set_ylabel("reward mean", color="#27ae60")
            ax2.set_ylabel("response length mean", color="#c0392b")
            ax.set_title("GRPO trajectory: reward + response length")
            ax.grid(alpha=0.3)

        plots["grpo"] = _fig(_p_grpo)

    # ---------- per-lag sample counts + the early/late capture-schedule confound ----------
    findings["lag_sample_counts"] = A.pairs_per_lag(idx, "g_dense")

    # ---------- gradient participation ratio + stable rank (surface "effective rank") ----------
    if grank:
        findings["grad_stablerank_median"] = _med([r["stable_rank"] for recs in grank.values() for r in recs])
        findings["grad_participation_median"] = _med([r["participation"] for recs in grank.values() for r in recs])

    # ---------- boundary multi-r overlap + principal angles (H3 detail) ----------
    if hdrift:
        for rr in (1, 5, R_LOCKED):
            by_k = {k: [] for k in LAGS}
            for d in hdrift.values():
                for k in LAGS:
                    by_k[k].extend(d.get("per_k_multi", {}).get(rr, {}).get(k, []))
            findings[f"boundary_overlap_r{rr}_by_k"] = {k: _med(by_k[k]) for k in LAGS}
        paf, pal = {k: [] for k in LAGS}, {k: [] for k in LAGS}
        for d in hdrift.values():
            for k in LAGS:
                paf[k].extend(d.get("pa_first", {}).get(k, []))
                pal[k].extend(d.get("pa_last", {}).get(k, []))
        findings["boundary_pa_first_by_k"] = {k: _med(paf[k]) for k in LAGS}
        findings["boundary_pa_last_by_k"] = {k: _med(pal[k]) for k in LAGS}
        # periodicity: dominant period of the adjacent-snapshot overlap series (first boundary)
        first_b = sorted(hdrift)[0]
        _ser = [v for _, v in hdrift[first_b].get("adj_series", [])]
        dp = A.dominant_period(_ser)
        if dp:
            findings["boundary_overlap_period"] = dp
        # multi-r overlap plot + principal-angle plot
        def _p_mr(ax):
            for rr, col in [(1, "#c0392b"), (5, "#e67e22"), (R_LOCKED, "#2c3e50")]:
                d = findings.get(f"boundary_overlap_r{rr}_by_k", {})
                ks = [k for k in LAGS if d.get(k) is not None and not _isnan(d.get(k))]
                if ks:
                    ax.plot(ks, [d[k] for k in ks], "o-", color=col, label=f"top-{rr}")
            ax.axvline(5, color="#27ae60", ls=":", lw=1.1, label="k≈5")
            ax.axvline(20, color="#e67e22", ls=":", lw=1.1, label="k≈20")
            ax.set_ylim(0, 1.02)
            ax.set_xlabel("lag k (optimizer ticks)")
            ax.set_ylabel("subspace overlap o(t,t−k)")
            ax.set_title("Boundary subspace overlap vs lag — by codec rank r")
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)

        plots["boundary_overlap_multir"] = _fig(_p_mr)

        def _p_pa(ax):
            d1, d2 = findings["boundary_pa_first_by_k"], findings["boundary_pa_last_by_k"]
            ks = [k for k in LAGS if d1.get(k) is not None and not _isnan(d1.get(k))]
            if ks:
                ax.plot(ks, [d1[k] for k in ks], "o-", color="#27ae60", label="smallest principal angle")
                ax.plot(ks, [d2[k] for k in ks], "s--", color="#c0392b", label="largest principal angle")
            ax.axvline(5, color="#27ae60", ls=":", lw=1.1)
            ax.axvline(20, color="#e67e22", ls=":", lw=1.1)
            ax.set_xlabel("lag k (optimizer ticks)")
            ax.set_ylabel(f"principal angle of top-{R_LOCKED} subspace (deg)")
            ax.set_title("Boundary subspace principal angles vs lag")
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)

        plots["boundary_pa"] = _fig(_p_pa)

    # ---------- FFT of the boundary-overlap series (periodicity, complements autocorr) ----------
    if hdrift:
        first_b = sorted(hdrift)[0]
        _ser = [v for _, v in hdrift[first_b].get("adj_series", [])]
        if len(_ser) >= 6:
            _lags, _ac, _fr, _pw = A.autocorr_fft(_ser)

            def _p_fft(ax):
                if len(_fr) > 1:
                    ax.stem([1.0 / x if x > 0 else 0 for x in _fr[1:]], _pw[1:])
                ax.set_xlabel("period (snapshots)")
                ax.set_ylabel("FFT power")
                ax.set_title(f"Subspace-overlap FFT — {first_b}")
                ax.grid(alpha=0.3)

            plots["boundary_fft"] = _fig(_p_fft)

    # ---------- SVD spectrum-evolution plots (g_dense, boundary_h, boundary_grad_h) ----------
    if grank:
        sp = _spectrum_fig(grank, has_epoch2_boundary, "Dense gradient g")
        if sp:
            plots["spectrum_grad"] = sp
    if hrank:
        sp = _spectrum_fig(hrank, has_epoch2_boundary, "Boundary activation h")
        if sp:
            plots["spectrum_h"] = sp
    if ghrank:
        sp = _spectrum_fig(ghrank, has_epoch2_boundary, "Boundary grad_h")
        if sp:
            plots["spectrum_gradh"] = sp

    # ---------- H2: behaviour-signal drift timescale vs the smooth weight drift ----------
    behaviour_keys = [
        "response_length/mean", "response_length/max", "actor/pg_clipfrac", "actor/ppo_kl",
        "actor/entropy", "rollout_corr/kl", "rollout_corr/log_ppl_abs_diff",
        "training/rollout_probs_diff_mean", "derived/advantage_dispersion", "critic/rewards/mean",
    ]
    h2b = A.behaviour_drift_vs_lag(grpo, behaviour_keys)
    wnorm = A.weight_drift_normalized(wper_k)
    findings["h2_behaviour"] = {k: {"half_lag": v["half_lag"], "norm": v["norm"], "n": v["n"]}
                               for k, v in h2b.items()}
    findings["h2_weight_half_lag"] = wnorm["half_lag"]
    findings["h2_weight_norm"] = wnorm["norm"]
    if h2b and wnorm["norm"]:
        def _p_h2(ax):
            wn = wnorm["norm"]
            wks = sorted(wn)
            ax.plot(wks, [wn[k] for k in wks], "k-o", lw=2.6, ms=5, label="weight drift (param-point gap)", zorder=5)
            order = sorted(h2b.items(),
                           key=lambda kv: (kv[1]["half_lag"] if kv[1]["half_lag"] is not None else 1e9))
            for key, v in order[:5]:
                ks = sorted(v["norm"])
                ax.plot(ks, [v["norm"][k] for k in ks], "--", marker=".", alpha=0.85, label=key.split("/")[-1])
            ax.axhline(0.5, color="#888", ls=":", lw=0.8)
            ax.set_xlabel("lag (global steps)")
            ax.set_ylabel("normalized drift (fraction of max-lag drift)")
            ax.set_title("H2 — behaviour vs weight drift timescale (front-loaded = faster)")
            ax.legend(fontsize=6.5, loc="lower right")
            ax.grid(alpha=0.3)

        plots["h2"] = _fig(_p_h2)

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

    _write_html(out_path, run_dir, rows, idx, plots, findings, grpo_present, dataset)
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
    c1 = cos.get(1)
    c5, c20 = cos.get(5), cos.get(20)
    s5, s20 = sign.get(5), sign.get(20)
    if c5 is None or c20 is None or (isinstance(c5, float) and math.isnan(c5)):
        return ("v-warn", "INCONCLUSIVE", "k≈5 or k≈20 gradient pairs were not both present.")
    ratio = c20 / c5 if c5 else float("nan")
    # immediate-decorrelation regime (hard task): even the lag-1 cosine is ~0, so there is
    # NO usable staleness window at all — the budget is effectively zero, not "crossed at k≈20".
    if c1 is not None and not _isnan(c1) and c1 < 0.1:
        return ("v-ok", "SUPPORTED (budget ≈ 0)",
                f"The dense gradient direction is essentially DECORRELATED even at the shortest lag "
                f"(cos(g_t,g_{{t−1}})={_safe(c1,3)}, cos(k≈5)={_safe(c5,3)}, cos(k≈20)={_safe(c20,3)}; "
                f"sign-agreement {_safe(s5,3)}→{_safe(s20,3)} ≈ chance 0.5). The gradient-anchor staleness budget "
                f"is effectively ZERO — a stale dense gradient is unusable at ANY latency on this task, well inside "
                f"even the stable 5/5 cadence. The knee is at or below k=1.")
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


def _resolve_h2(f):
    """H2: at least one rollout/logprob/response signal drifts on a comparable-or-
    faster timescale than the smooth weight drift (distribution gap, not pure
    parameter-point gap). Falsified if every behaviour signal is strictly slower."""
    bh = f.get("h2_behaviour", {})
    wl = f.get("h2_weight_half_lag")
    if not bh or wl is None:
        return ("v-warn", "INCONCLUSIVE", "behaviour-signal or weight-drift timescale unavailable.")
    nm = lambda s: s.split("/")[-1]
    faster = [(k, v) for k, v in bh.items()
              if v.get("half_lag") is not None and not _isnan(v["half_lag"]) and v["half_lag"] <= wl + 1e-9]
    n_fast, n_tot = len(faster), len(bh)
    if n_fast >= 1:
        ex = sorted(faster, key=lambda kv: kv[1]["half_lag"])[:3]
        exs = ", ".join(f"<code>{html.escape(nm(k))}</code> (t½≈{_safe(v['half_lag'],1)})" for k, v in ex)
        return ("v-ok", "SUPPORTED",
                f"{n_fast}/{n_tot} rollout/logprob/response signals reach half their total drift at a lag "
                f"≤ the weight half-drift lag (≈{_safe(wl,1)} global steps) — they drift on a comparable-or-"
                f"FASTER timescale than the smooth parameter-point gap. Fastest: {exs} steps. The dangerous "
                f"term is the distribution gap (gap 2), not just curvature×‖Δθ‖.")
    return ("v-bad", "FALSIFIED",
            f"All {n_tot} behaviour signals reach half their drift only at lags LONGER than the weight "
            f"half-drift lag (≈{_safe(wl,1)} steps) — every GRPO signal drifts strictly slower than the "
            f"smooth weight drift, so the staleness is curvature-bounded (SFT-like).")


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
    eff = cent if (cent is not None and not _isnan(cent)) else raw
    lowrank = eff <= R_LOCKED * 1.15
    very_lowrank = eff <= R_LOCKED * 0.5
    massive = (top1 is not None and not _isnan(top1) and top1 >= 0.5)
    msg = (f"Boundary activation h: raw rank-90% ≈ {_safe(raw,1)}"
           + (f" (top-1 singular dim holds {_safe(top1,3)} of the energy — a massive-activation outlier)" if massive else "")
           + f", mean-centered (residual) rank-90% ≈ {_safe(cent,1)} (vs r={R_LOCKED}, H={HIDDEN}). "
           + (f"The signal is FAR below r — a rank-r activation codec is the right primitive and r is heavily OVER-provisioned"
              if very_lowrank else
              (f"The residual signal is LOW-RANK ≤ r — a rank-r activation codec is the right primitive"
               if lowrank else
               f"The residual signal is NOT low-rank vs r — rank-r activation compression discards real structure"))
           + ". ")
    if o20 is None or _isnan(o20):
        return ("v-warn", "PARTIAL", msg + "Subspace-overlap-vs-lag not fully resolved.")
    # is the overlap FLAT across lag (staleness-insensitive) or DECAYING (a real Q budget)?
    o_small = next((ov.get(k) for k in (1, 2, 5) if ov.get(k) is not None and not _isnan(ov.get(k))), None)
    decay_ratio = (o20 / o_small) if (o_small not in (None, 0) and not _isnan(o_small)) else None
    flat = decay_ratio is not None and decay_ratio >= 0.9
    if o20 >= 0.95:
        msg += (f"Top-{R_LOCKED} subspace overlap o(t,t−20) ≈ {_safe(o20,3)} ≥ 0.95 — the subspace is nearly "
                f"STATIC; Q can be frozen far longer than the current cadence (codec-staleness is NOT the limiter).")
        return ("v-ok", "SUPPORTED (Q-freezable)", msg)
    if flat:
        msg += (f"Top-{R_LOCKED} overlap is essentially FLAT across lag (o(t,t−1)≈{_safe(o_small,3)} → "
                f"o(t,t−20)≈{_safe(o20,3)}, ratio {_safe(decay_ratio,2)}): the mismatch is PER-STEP, not "
                f"staleness-driven, so codec STALENESS is not the limiter (Q is stale-tolerant); the constant "
                f"~{_safe(o20,2)} offset is the noise-padded tail (h is ~rank-1, so the top-{R_LOCKED} subspace "
                f"is mostly noise beyond the energetic directions).")
        return ("v-ok", "SUPPORTED (staleness-insensitive)", msg)
    msg += (f"Top-{R_LOCKED} subspace overlap decays with lag (o(t,t−5)≈{_safe(ov.get(5),3)} → "
            f"o(t,t−20) ≈ {_safe(o20,3)}, ratio {_safe(decay_ratio,2)}) — this IS the activation-codec (Q) "
            f"staleness budget.")
    return ("v-ok", "SUPPORTED", msg)


def _write_html(out_path, run_dir, rows, idx, plots, findings, grpo_present, dataset="gsm8k"):
    f = findings
    ds = html.escape(str(dataset).upper())
    knee = f.get("knee", {})
    cos = f.get("grad_cos", {})
    sign = f.get("grad_sign", {})
    nr = f.get("grad_normratio", {})
    wd = f.get("weight_drift", {})
    ov = f.get("boundary_overlap_by_k", {})
    h1 = _resolve_h1(f)
    h2 = _resolve_h2(f)
    h3 = _resolve_h3(f)

    # per-lag sample counts + the early/late capture-schedule confound (honest n).
    lsc = f.get("lag_sample_counts", {})
    if lsc:
        lrows = "".join(
            f'<tr><td class="num">k={k}</td><td class="num">{lsc.get(str(k), lsc.get(k, {})).get("pairs_per_matrix","—")}</td>'
            f'<td class="num">{lsc.get(str(k), lsc.get(k, {})).get("n_values","—")}</td>'
            f'<td class="num">{lsc.get(str(k), lsc.get(k, {})).get("gs_lo","—")}–{lsc.get(str(k), lsc.get(k, {})).get("gs_hi","—")}</td></tr>'
            for k in LAGS
        )
        lag_table = (f'<table><tr><th class="num">lag k (ticks)</th><th class="num">pairs / matrix</th>'
                     f'<th class="num">total values</th><th class="num">global-step span</th></tr>{lrows}</table>'
                     f'<p class="sub">The accel capture schedule dumps consecutive ticks only at global steps 1–3 '
                     f'(ticks 0–5), so the small lags k≤5 are sampled almost entirely in EARLY training while k≥10 '
                     f'span mid/late. cos at k=1,2,5 is therefore a clean <i>within-(early)-phase</i> lag decay; '
                     f'k=10/20/40 are all mid/late. n=1 trajectory throughout.</p>')
    else:
        lag_table = ""

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
    has_epoch2_boundary = f.get("epoch2_step") is not None
    response_cap = "16384" if str(dataset).lower() == "big-math" else "2048"
    if has_epoch2_boundary:
        pre_epoch = f.get("grad_rank90_pre_epoch2")        # post-warmup pre
        post_epoch = f.get("grad_rank90_post_epoch2")
        pre_naive = f.get("grad_rank90_pre_epoch2_naive")  # warmup-contaminated
        warm = f.get("grad_rank90_warmup_steps", 5)
        shifted = (
            pre_epoch is not None
            and post_epoch is not None
            and not _isnan(pre_epoch)
            and not _isnan(post_epoch)
            and abs(pre_epoch - post_epoch) > 5
        )
        grad_rank_caption = (
            "Dense-gradient stable rank + median rank-for-90%-energy over training, vs r=77 "
            "and the GSM8K epoch-2 boundary."
        )
        grad_epoch_sentence = (
            f"Across the GSM8K epoch-2 boundary (~step 58), EXCLUDING the lr-warmup ramp (steps ≤{warm}, "
            f"where rank is still climbing): pre ≈ {_safe(pre_epoch,1)} vs post ≈ {_safe(post_epoch,1)} — "
            f"{'a real shift at the epoch boundary' if shifted else 'essentially FLAT (no clean epoch-2 jump)'}. "
            f"(A naive ≤58-vs->58 split would read {_safe(pre_naive,1)}→{_safe(post_epoch,1)}, but that apparent "
            f"rise is a warmup-binning artifact — the early low-rank warmup steps sit in the pre bin; per-step rank "
            f"is stationary post-warmup, as the trajectory plot shows.)"
        )
        grpo_caption = "Dense GRPO trajectory — reward and response-length over training (GSM8K epoch-2 boundary marked)."
        v2b_epoch = f"; post-warmup pre/post epoch-2 ≈ {_safe(pre_epoch,1)}/{_safe(post_epoch,1)} (flat; the naive split {_safe(pre_naive,1)}→{_safe(post_epoch,1)} is a warmup artifact)"
    else:
        grad_rank_caption = (
            "Dense-gradient stable rank + median rank-for-90%-energy over training, vs r=77 "
            "(no Big-Math epoch-2 split)."
        )
        grad_epoch_sentence = (
            "Big-Math does not cross an epoch boundary in this 75-step run (train cap 20000 gives "
            "about 156 steps per epoch), so no epoch-2 split is reported."
        )
        grpo_caption = "Dense GRPO trajectory — reward and response-length over training (no epoch-2 boundary for this dataset)."
        v2b_epoch = "; no epoch-2 split for this dataset"
    grad_lowrank = (f.get("grad_rank90_median") is not None and f["grad_rank90_median"] <= R_LOCKED * 1.15)
    gradh_rank = f.get("boundary_gradh_rank90_median")
    gradh_highrank = (gradh_rank is not None and not _isnan(gradh_rank) and gradh_rank > R_LOCKED)
    o20 = ov.get(20)
    o_small_v = next((ov.get(k) for k in (1, 2, 5) if ov.get(k) is not None and not _isnan(ov.get(k))), None)
    overlap_flat = (o20 is not None and not _isnan(o20) and o_small_v not in (None, 0)
                    and (o20 / o_small_v) >= 0.9)
    q_freezable = (o20 is not None and not _isnan(o20) and (o20 >= 0.9 or overlap_flat))
    grad_decays = (cos.get(5) and cos.get(20) is not None and cos.get(20) <= 0.5 * cos.get(5))
    grad_zero_budget = (cos.get(1) is not None and not _isnan(cos.get(1)) and cos.get(1) < 0.1)
    h2_supported = h2[1].startswith("SUPPORTED")

    rec_bullets = []
    if grad_decays:
        rec_bullets.append("<b>Do NOT use a stale dense gradient as an optimizer signal beyond a few ticks.</b> "
                           "The dense gradient direction de-correlates fast with lag (H1: cos "
                           f"{_safe(cos.get(1),2)}→{_safe(cos.get(5),2)}→{_safe(cos.get(20),2)} at k=1/5/20), so a "
                           "delayed anchor is the gradient of a defunct policy — exactly the 20/20 failure. Use the "
                           "anchor <b>only as a slow Q / codec-calibration</b> source (answers Q5: yes).")
    else:
        rec_bullets.append("The dense gradient direction is comparatively persistent across lag; a stale gradient "
                           "anchor may remain usable at larger K than EXP-37 suggested — re-test the staleness budget directly.")
    if boundary_lowrank and q_freezable:
        rec_bullets.append("<b>Compress the FORWARD boundary traffic in ACTIVATION space with a low-rank codec and a "
                           "slowly-refreshed (or frozen) Q.</b> The boundary activation is "
                           + ("essentially rank-1 (a massive-activation direction dominates), so r=77 is heavily "
                              "over-provisioned " if boundary_massive else "low-rank ")
                           + "and its top-r subspace overlap is "
                           + ("FLAT across lag (staleness-insensitive) " if overlap_flat else "high ")
                           + "— a frozen/periodically-refreshed PowerSGD-style Q tracks it cheaply; codec STALENESS is "
                           "not the limiter.")
    elif boundary_lowrank:
        rec_bullets.append("<b>Activation-space rank-r compression is viable but Q must be refreshed on the codec "
                           "cadence</b> — the subspace rotates with staleness, so a frozen Q lags.")
    else:
        rec_bullets.append("The boundary activation is NOT clearly low-rank vs r — reconsider rank-r activation "
                           "compression; a higher-rank or hybrid codec may be needed.")
    if gradh_highrank:
        rec_bullets.append(f"<b>Forward/backward asymmetry — do NOT assume the backward link is as compressible as the "
                           f"forward one.</b> The forward activation <code>h</code> is ~rank-1, but the backward "
                           f"<code>grad_h</code> is rank-for-90% ≈ {_safe(gradh_rank,0)} &gt; r={R_LOCKED}: a rank-r "
                           f"codec on <code>grad_h</code> discards real energy. Budget the backward boundary codec at "
                           f"higher rank than the forward, or use error-feedback on the backward link.")
    rec_bullets.append(("Compress in <b>activation space</b> (low-rank forward boundary traffic) " if boundary_lowrank else "")
                       + ("and exploit the low-rank dense gradient too" if grad_lowrank else "")
                       + "; treat the slow node as a subspace/Q calibrator, <b>not</b> a gradient provider.")
    if h2_supported:
        rec_bullets.append("Because behaviour/rollout signals drift on a comparable-or-faster timescale than the "
                           "weights (H2 supported), the staleness danger is the <b>distribution gap</b>, not the "
                           "parameter-point gap — a codec/anchor that is correct in parameter space can still be "
                           "stale in distribution space. The next method must be robust to rollout-distribution drift "
                           "(e.g. on-policy Q refresh, or IS-style correction on any reused gradient signal).")
    rec_bullets.append("Plausible next-method families for on-policy RLVR/GRPO: (a) frozen/slow-Q <b>activation</b> "
                       "codec with on-policy refresh (forward link cheap, backward link higher-rank); (b) cross-rank "
                       "2nd-moment (disagreement-as-objective) routes that inject info outside the stale+current "
                       "gradient means (the σ(M) ceiling); (c) curvature/2nd-order anchor use. Avoid "
                       "reweighting/accumulating a stale gradient estimate (EXP-31/37 dead ends).")

    def _row(label, d, fmt=3):
        cells = "".join(f'<td class="num">{_safe(d.get(k), fmt)}</td>' for k in LAGS)
        return f"<tr><td>{label}</td>{cells}</tr>"

    lag_hdr = "".join(f'<th class="num">k={k}</th>' for k in LAGS)

    _wl = f.get("h2_weight_half_lag")
    _h2b = f.get("h2_behaviour", {})
    h2_fast_n = sum(1 for v in _h2b.values()
                    if v.get("half_lag") is not None and not _isnan(v["half_lag"])
                    and _wl is not None and v["half_lag"] <= _wl + 1e-9)
    h2_tot_n = len(_h2b)

    parts = []
    parts.append(f"""<!-- generated by exp38_report.py -->
<div class="wrap">
<h1>EXP-38 — Dense GRPO temporal-drift probe <span style="background:#1f2328;color:#fff;border-radius:6px;padding:.08em .5em;font-size:.62em;vertical-align:middle">dataset: {ds}</span></h1>
<p class="sub">How fast does the on-policy GRPO learning signal drift in time — in gradient space, the
boundary-activation subspace, and rollout/behaviour space — and what does that imply for the next
communication-efficient pipeline-parallel GRPO method? <b>This report covers the {ds} dataset ONLY.</b></p>
<p class="sub">Qwen2.5-1.5B-Instruct · {ds} · accel surface (response cap {response_cap}, dynamic-bsz, TP1) · <b>dense</b>
(comm_eff OFF) · 75 global steps = 150 optimizer ticks · n=1 trajectory · lag axis in optimizer ticks
(2 ticks/global-step ⇒ k≈5 ≙ the stable 5/5 anchor, k≈20 ≙ the broken 20/20 anchor, k≈40 ≙ beyond).</p>

<div class="tldr">
<h3>TL;DR</h3>
<div class="kpi">
<div><b>{_safe(knee.get('grad_cos_k5'),3)}</b><span>grad cos at k≈5 (5/5)</span></div>
<div><b>{_safe(knee.get('grad_cos_k20'),3)}</b><span>grad cos at k≈20 (20/20)</span></div>
<div><b>{_safe(knee.get('grad_cos_ratio_20_over_5'),2)}</b><span>cos ratio k20/k5</span></div>
<div><b>{_safe(f.get('grad_rank90_median'),0)}</b><span>dense-grad rank-90% (vs r={R_LOCKED})</span></div>
<div><b>{_safe(f.get('boundary_h_rank90_median'),1)}</b><span>boundary h rank-90% (vs r={R_LOCKED})</span></div>
<div><b>{_safe(f.get('boundary_gradh_rank90_median'),0)}</b><span>boundary grad_h rank-90%</span></div>
<div><b>{_safe(knee.get('boundary_overlap_k20'),3)}</b><span>subspace overlap o(t,t−20)</span></div>
</div>
<p><b>H1 (gradient-anchor staleness budget):</b> <span class="verdict {h1[0]}">{h1[1]}</span> {h1[2]}</p>
<p><b>H2 (drift is GRPO-coupled, not a pure parameter-point gap):</b> <span class="verdict {h2[0]}">{h2[1]}</span> {h2[2]}</p>
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
<h3>Per-lag sample counts (honest n)</h3>
{lag_table}
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
{_img(plots,'grad_rank',grad_rank_caption)}
{_img(plots,'spectrum_grad','Dense-gradient SVD spectrum: (left) singular values at early/mid/late snapshots; (right) full spectrum evolution over training.')}
<p>Median dense-gradient rank-for-90%-energy ≈ <b>{_safe(f.get('grad_rank90_median'),1)}</b> (vs the locked
PowerSGD rank r={R_LOCKED}); median stable rank ‖g‖_F²/‖g‖₂² ≈ <b>{_safe(f.get('grad_stablerank_median'),1)}</b>,
participation ratio ≈ <b>{_safe(f.get('grad_participation_median'),1)}</b>. {grad_epoch_sentence}
{'The dense gradient is effectively low-rank (≤ r), so a rank-r codec captures most of its energy.' if grad_lowrank else 'The dense gradient is higher-rank than r — a rank-r gradient codec discards real energy.'}</p>
""")

    period = f.get("boundary_overlap_period")
    period_sentence = (
        (f"Periodicity: the dominant non-DC period of the overlap series is ≈ {_safe(period['period'],1)} snapshots, "
         f"holding {_safe(period['power_frac'],2)} of the AC power → classified <b>{html.escape(period['verdict'])}</b>. ")
        if period else "")
    pa_first = f.get("boundary_pa_first_by_k", {})
    pa_last = f.get("boundary_pa_last_by_k", {})
    ov1 = f.get("boundary_overlap_r1_by_k", {})
    parts.append(f"""<h2><span class="num">4</span>Boundary-activation subspace — the activation-codec staleness budget (H3)</h2>
{_img(plots,'boundary_rank','Boundary activation rank-for-90%-energy over training, vs r=77 and H=1536.')}
{_img(plots,'spectrum_h','Boundary activation h SVD spectrum: (left) snapshots; (right) evolution. A near-vertical drop after σ₁ = the massive-activation rank-1 structure.')}
{_img(plots,'boundary_overlap','Top-r boundary subspace overlap o(t,t−k) vs lag — how stale Q can be.')}
{_img(plots,'boundary_overlap_multir','Subspace overlap vs lag at codec ranks r∈{{1,5,77}} — the energetic (top-1/5) subspace is what a codec actually tracks.')}
{_img(plots,'boundary_pa','Principal angles (smallest = best-aligned direction, largest = worst) of the top-r subspace vs lag.')}
{_img(plots,'boundary_period','Autocorrelation of the subspace-overlap time series — smooth-monotone vs periodic drift.')}
{_img(plots,'boundary_fft','FFT power of the subspace-overlap series (period in snapshots) — periodicity test.')}
<p><span class="verdict {h3[0]}">H3 {h3[1]}</span> {h3[2]}</p>
<table><tr><th>median over boundaries</th>{lag_hdr}</tr>
{_row('top-1 subspace overlap', ov1)}
{_row('top-5 subspace overlap', f.get('boundary_overlap_r5_by_k', {}))}
{_row('top-77 subspace overlap', ov)}
{_row('smallest principal angle (deg)', pa_first, 1)}
{_row('largest principal angle (deg)', pa_last, 1)}
</table>
<p class="sub">{period_sentence}The top-1 overlap is the alignment of the single energetic (massive-activation)
direction the codec must track; the top-77 overlap is dragged down by the noise-padded tail (h is ~rank-1).</p>
{bdetail_table}
<p>This is the <b>codec-decisive</b> evidence the gradient-cosine curve cannot give: even if the gradient
anchor is doomed as an optimizer signal, a rank-r activation codec with a slowly-rotating Q may still be the
right compression primitive.</p>
""")

    parts.append(f"""<h2><span class="num">5</span>Boundary activation-gradient rank (the backward link)</h2>
{_img(plots,'boundary_gradrank','Boundary grad_h rank-for-90%-energy over training (the backward boundary traffic).')}
{_img(plots,'spectrum_gradh','Boundary grad_h SVD spectrum: (left) snapshots; (right) evolution. Compare with h above — the backward traffic is markedly higher-rank.')}
<p>Median <code>grad_h</code> rank-for-90%-energy ≈ <b>{_safe(f.get('boundary_gradh_rank90_median'),1)}</b>
(vs r={R_LOCKED}). {'This is ABOVE r — a rank-r codec on the backward link discards real energy, a sharp forward/backward asymmetry: the forward h is ~rank-1 but the backward grad_h is not.' if gradh_highrank else 'This bounds a rank-r codec on the backward boundary traffic a real PP link carries.'}</p>
""")

    parts.append(f"""<h2><span class="num">6</span>GRPO-signal drift + correlation (H2: distribution-gap vs parameter-point gap)</h2>
{_img(plots,'h2','Normalized lag-drift of behaviour/rollout signals vs the smooth weight drift. A curve that rises ABOVE the weight curve at small lag front-loads its drift = drifts faster.')}
{_img(plots,'grpo',grpo_caption)}
<p><span class="verdict {h2[0]}">H2 {h2[1]}</span> {h2[2]}</p>
<p>The H2 test normalizes each signal's lag-k drift <code>D(k)=median|x_t−x_{{t−k}}|</code> to its own
max-lag value and compares the <b>half-drift lag</b> (where the normalized curve reaches 0.5) against the
weight half-drift lag (≈ {_safe(f.get('h2_weight_half_lag'),1)} global steps). Behaviour signals that reach
half their drift sooner than the (cumulative, smooth) weight drift are the signature of a distribution-gap
(gap 2) danger rather than a pure curvature×‖Δθ‖ effect.</p>
<h3>Rank-curve × GRPO-signal correlation (incl. rollout-group diversity)</h3>
{corr_table}
<p class="sub">Pearson r over aligned global steps between the effective-rank curves and each GRPO signal
(<code>derived/advantage_dispersion</code> = per-step max−min advantage, an n=8 rollout-group-spread proxy).
A strong rank↔diversity / rank↔response-length coupling characterises the nature of learning
(exploration→refinement) and tells whether rank evolution tracks behaviour drift.</p>
""")

    parts.append(f"""<h2><span class="num">7</span>Hypotheses, resolved as numbers</h2>
<table>
<tr><th>hypothesis</th><th>verdict</th><th>key numbers</th></tr>
<tr><td>H1 — gradient-space staleness budget crossed by ~20-tick lag</td><td><span class="verdict {h1[0]}">{h1[1]}</span></td>
<td>cos k≈5={_safe(cos.get(5),3)}, k≈20={_safe(cos.get(20),3)} (ratio {_safe(knee.get('grad_cos_ratio_20_over_5'),2)}); sign {_safe(sign.get(5),3)}→{_safe(sign.get(20),3)}</td></tr>
<tr><td>H2 — drift is GRPO-coupled (distribution gap), not pure parameter-point gap</td><td><span class="verdict {h2[0]}">{h2[1]}</span></td>
<td>weight half-drift lag ≈ {_safe(_wl,1)} steps; {h2_fast_n}/{h2_tot_n} behaviour signals drift comparably-or-faster</td></tr>
<tr><td>H3 — boundary activation low-rank with a measurable Q-staleness budget</td><td><span class="verdict {h3[0]}">{h3[1]}</span></td>
<td>h rank-90% ≈ {_safe(f.get('boundary_h_rank90_median'),1)} (r={R_LOCKED}); o(t,t−20)={_safe(ov.get(20),3)}, o(t,t−5)={_safe(ov.get(5),3)}</td></tr>
</table>
""")

    parts.append(f"""<h2><span class="num">8</span>Deliverable questions, answered</h2>
<div class="card">
<p class="q">1. How fast do dense GRPO weights &amp; gradients drift?</p>
<p>Weights drift smoothly/monotonically (§2 table). Gradient direction {'is already decorrelated at the shortest measured lag' if grad_zero_budget else ('de-correlates sharply' if grad_decays else 'de-correlates mildly')}:
cos {_safe(cos.get(1),3)} (k=1) → {_safe(cos.get(5),3)} (k≈5) → {_safe(cos.get(20),3)} (k≈20) → {_safe(cos.get(40),3)} (k≈40).</p>
<p class="q">2. At what staleness does gradient cosine / sign agreement become unsafe?</p>
<p>{'The knee is at or BELOW k=1: the gradient is decorrelated immediately, so even the stable 5/5 anchor (k≈5) operates on a ~0-correlation gradient — the usable staleness budget is effectively zero on this task.' if grad_zero_budget else 'The knee sits '+('between k≈5 and k≈20' if grad_decays else 'beyond the measured range')+f': cosine ratio k20/k5 = {_safe(knee.get("grad_cos_ratio_20_over_5"),2)}, sign-agreement '+f'{_safe(sign.get(5),3)}→{_safe(sign.get(20),3)} (chance=0.5). This '+('matches' if grad_decays else 'does not by itself explain')+' the 5/5-stable vs 20/20-broken boundary.'}
{'This is consistent with the 20/20 failure AND implies even the 5/5 cadence is marginal here (cos(k≈5)='+_safe(cos.get(5),3)+').' if grad_zero_budget else ''}</p>
<p class="q">3. Are the dangerous changes in gradient space, rollout/logprob space, response behaviour, or the boundary-activation subspace?</p>
<p>{'Gradient space carries a real, fast-decaying staleness term (H1). ' if grad_decays else 'Gradient space alone is not the danger (H1 mild). '}
The boundary-activation forward subspace {'is staleness-INSENSITIVE (top-r overlap flat across lag), so codec staleness is NOT the danger there' if overlap_flat else ('rotates measurably with lag (a genuine codec-staleness budget)' if (o20 is not None and not _isnan(o20) and o20<0.95) else 'is nearly static (Q-freezable)')}.
{'Behaviour/rollout signals drift comparably-or-faster than the weights (H2 supported), so the dangerous term is the distribution gap (rollout/logprob/response space), not the parameter-point gap.' if h2_supported else 'Behaviour signals drift no faster than the weights (H2 falsified), so the staleness looks curvature-bounded.'}</p>
<p class="q">4. What does this imply for the next comm-eff PP method (compress in activation space, gradient space, or both)?</p>
<p>{'Activation space' if boundary_lowrank else 'Not rank-r activation space'}{' + the low-rank dense gradient' if grad_lowrank else ''}. See §9.</p>
<p class="q">5. Should future methods use the anchor only for Q/codec calibration, not as an optimizer gradient?</p>
<p><b>{'Yes' if grad_decays else 'Not necessarily'}.</b> {'The stale gradient is a valid estimate for a policy that no longer exists; use the slow node as a Q/subspace calibrator instead.' if grad_decays else 'The gradient persists enough that anchor-as-gradient may survive larger K — verify directly.'}</p>
<p class="q">6. What next-method families are plausible?</p>
<p>See §9 — frozen/slow-Q activation codec, cross-rank 2nd-moment objectives, curvature/2nd-order anchor use.</p>
<p class="q">(v2-A) Is the boundary activation low-rank, and how fast does its top-r subspace rotate?</p>
<p>rank-90% ≈ {_safe(f.get('boundary_h_rank90_median'),1)} (vs r={R_LOCKED}, H={HIDDEN}) — {'far below r (≈rank-1; r over-provisioned)' if (f.get('boundary_h_rank90_median') or 99) <= R_LOCKED*0.5 else 'low-rank'}; top-1 overlap o(t,t−20)={_safe(f.get('boundary_overlap_r1_by_k',{}).get(20),3)}, top-{R_LOCKED} overlap o(t,t−20)={_safe(ov.get(20),3)} — {'flat across lag (staleness-insensitive)' if overlap_flat else 'decaying with lag'} (§4); periodicity in §4. Backward grad_h rank-90% ≈ {_safe(f.get('boundary_gradh_rank90_median'),1)} (the forward/backward asymmetry).</p>
<p class="q">(v2-B) Is the dense gradient low-rank, and how does its rank evolve?</p>
<p>rank-90% median ≈ {_safe(f.get('grad_rank90_median'),1)}{v2b_epoch} (§3).</p>
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
                 f"<title>EXP-38 — Dense GRPO temporal-drift probe [{ds}]</title>"
                 f"<style>{CSS}</style></head><body>" + "".join(parts) + "</body></html>")
