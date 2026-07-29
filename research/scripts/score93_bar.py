#!/usr/bin/env python3
"""score93_bar.py - score one issue #93 probe cell against its REGISTERED bar.

One command, identical for a5b and a6, so neither cell can be scored on a
different recipe than the other. Computes numbers and prints PASS/FAIL per
registered gate; it does NOT write the verdict. The analyst does that.

The bar comes from research/runs/93-long-horizon-stability/PREREG_a6.md and its
two logged amendments. Thresholds are hardcoded here on purpose: a bar that can
be passed on the command line is not a pre-registered bar.

  G1 learning   critic/score/mean  LEVEL  over 100-120  >= 0.6248
  G2 gap        rollout_corr/kl    LEVEL  over 100-120  <  14.2458 (strict)
                rollout_corr/kl    SLOPE  over 100-120  <= +5.0e-4
  G3 drift      actor/kl_loss      SLOPE  over 100-120  <= 3.264e-3
  G4 wire       bits/token/boundary              ==     1232 (by construction)

Windows: 100-120 is PRIMARY. Amendment 1 established that 3.264e-3 is 1.5x the
incumbent's slope over 100-120 (0.002176, exact), so 100-120 is the window the
threshold was derived from and the one round A used. 61-120 is reported as
SECONDARY for both cells so the window choice conceals nothing.

The dense channel is reported but NOT gated. Amendment 2: the V1 clause compares
a dense-channel slope against a codec-view-derived threshold, and the codec
inflates the drift reading 13.8x to 34.7x, so the clause passes by ~77x on a unit
mismatch and discriminates nothing.

Usage:
  score93_bar.py --cell a5b-frlr-bnorm-200
  score93_bar.py --cell a6-prf-exactk-tis-bnorm-200 --compare
"""
import argparse
import sys

import numpy as np

ENTITY = "shamanework-pl"
PROJECT = "93-long-horizon-stability"
INCUMBENT = ("90-prf-exactk-600", "90-prf-exactk-600")  # (run, project)

# --- REGISTERED BAR. Do not parameterise. ---
G1_SCORE_MIN = 0.6248
G2_GAP_LEVEL_MAX = 14.2458
G2_GAP_SLOPE_MAX = 5.0e-4
G3_DRIFT_SLOPE_MAX = 3.264e-3
G4_WIRE_BITS = 1232
WIN_PRIMARY = (100, 120)
WIN_SECONDARY = (61, 120)

STEP = "training/global_step"


def series(run, key):
    """Unsampled single-key pull. history() samples and drops rows missing any
    key; scan_history with one metric avoids both."""
    rows = []
    try:
        for row in run.scan_history(keys=[STEP, key]):
            s, v = row.get(STEP), row.get(key)
            if s is None or v is None:
                continue
            rows.append((float(s), float(v)))
    except Exception:
        return None, None
    if not rows:
        return None, None
    a = np.array(sorted(rows))
    return a[:, 0], a[:, 1]


def level(x, y, lo, hi):
    if x is None:
        return float("nan"), 0
    m = (x >= lo) & (x <= hi) & np.isfinite(y)
    return (float(np.mean(y[m])) if m.sum() else float("nan")), int(m.sum())


def slope(x, y, lo, hi):
    if x is None:
        return float("nan"), 0
    m = (x >= lo) & (x <= hi) & np.isfinite(y)
    if m.sum() < 3:
        return float("nan"), int(m.sum())
    return float(np.polyfit(x[m], y[m], 1)[0]), int(m.sum())


def verdict(ok):
    return "PASS" if ok else ("FAIL" if ok is False else "n/a ")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--project", default=PROJECT)
    ap.add_argument("--entity", default=ENTITY)
    ap.add_argument("--compare", action="store_true",
                    help="also print the incumbent's matched-window values")
    args = ap.parse_args()

    import wandb
    api = wandb.Api()

    def resolve(name, project):
        try:
            return api.run(f"{args.entity}/{project}/{name}")
        except Exception:
            for r in api.runs(f"{args.entity}/{project}"):
                if r.name == name:
                    return r
        return None

    run = resolve(args.cell, args.project)
    if run is None:
        print(f"ERROR: {args.cell!r} not found in {args.entity}/{args.project}", file=sys.stderr)
        sys.exit(2)

    xs, sc = series(run, "critic/score/mean")
    xg, gp = series(run, "rollout_corr/kl")
    xd, dr = series(run, "actor/kl_loss")
    max_step = int(np.nanmax(xs)) if xs is not None else -1

    lo, hi = WIN_PRIMARY
    slo, shi = WIN_SECONDARY

    g1, n1 = level(xs, sc, lo, hi)
    g2l, n2 = level(xg, gp, lo, hi)
    g2s, _ = slope(xg, gp, lo, hi)
    g2s2, _ = slope(xg, gp, slo, shi)
    g3s, _ = slope(xd, dr, lo, hi)
    g3s2, _ = slope(xd, dr, slo, shi)
    g3l, _ = level(xd, dr, lo, hi)

    f = np.isfinite
    ok1 = bool(g1 >= G1_SCORE_MIN) if f(g1) else None
    ok2l = bool(g2l < G2_GAP_LEVEL_MAX) if f(g2l) else None
    ok2s = bool(g2s <= G2_GAP_SLOPE_MAX) if f(g2s) else None
    ok3 = bool(g3s <= G3_DRIFT_SLOPE_MAX) if f(g3s) else None

    print(f"=== score93_bar: {run.name} ({run.id}) state={run.state} max_step={max_step} ===")
    if max_step < hi:
        print(f"*** INCOMPLETE: window {lo}-{hi} needs step {hi}; run is at {max_step}. "
              f"Numbers below are PARTIAL and are not a verdict. ***")
    print(f"primary window {lo}-{hi} ({n1} score rows, {n2} gap rows)\n")

    print("REGISTERED GATES (primary window)")
    print(f"  G1 learning  score level   {g1:9.4f}   bar >= {G1_SCORE_MIN}          {verdict(ok1)}")
    print(f"  G2 gap       gap level     {g2l:9.4f}   bar <  {G2_GAP_LEVEL_MAX}        {verdict(ok2l)}")
    print(f"     gap       gap slope     {g2s:+9.6f}   bar <= {G2_GAP_SLOPE_MAX:+.1e}      {verdict(ok2s)}")
    print(f"  G3 drift     drift slope   {g3s:+9.6f}   bar <= {G3_DRIFT_SLOPE_MAX:.3e}     {verdict(ok3)}")
    print(f"  G4 wire      1232 bits by construction (77 coords x 16)         PASS (automatic, no information)")
    print(f"     drift level (context)   {g3l:9.6f}   not gated")
    print()
    print(f"SECONDARY window {slo}-{shi} (reported so the window choice hides nothing)")
    print(f"  gap slope   {g2s2:+9.6f}      drift slope {g3s2:+9.6f}")
    print()

    print("DENSE CHANNEL (codec-free; reported, NOT gated per amendment 2)")
    xk, kd = series(run, "probe/kl_dense")
    xgd, gd = series(run, "probe/gap_dense")
    xkg, kg = series(run, "probe/kl_gain")
    if xk is None:
        print("  no probe/* keys on this run")
    else:
        ds, dn = slope(xk, kd, 0, 10**9)
        print(f"  probe/kl_dense   n={len(xk):3d}  last={kd[-1]:.6f}  full-run slope={ds:+.3e}"
              f"   (codec-view bar {G3_DRIFT_SLOPE_MAX:.3e} is a UNIT MISMATCH)")
        if xgd is not None:
            gs, _ = slope(xgd, gd, 0, 10**9)
            print(f"  probe/gap_dense  n={len(xgd):3d}  mean={np.nanmean(gd):.6f} nats  slope={gs:+.3e}")
            if f(g2l) and np.nanmean(gd) > 0:
                print(f"      -> codec accounts for a factor of {g2l/np.nanmean(gd):.0f} in the measured gap")
        if xkg is not None:
            print(f"  probe/kl_gain    first={kg[0]:.1f}x  last={kg[-1]:.1f}x"
                  f"   ({'GROWING: view offset is time-varying' if kg[-1] > kg[0] else 'stable'})")
        xb, br = series(run, "probe/lr_brake_triggered")
        if xb is not None:
            print(f"  lr_brake fired   {int(np.nansum(br))} of {len(xb)} probes (detection only, never mutates LR)")
    print()

    print("IS HEALTH (reported, not gated; all metrics are PRE-normalization)")
    for key, lab in [("rollout_corr/rollout_is_mean", "mean weight (raw)"),
                     ("rollout_corr/rollout_is_eff_sample_size", "ESS (scale-INVARIANT)"),
                     ("rollout_corr/rollout_is_batch_norm_factor", "batch_norm_factor"),
                     ("rollout_corr/rollout_is_ratio_fraction_low", "fraction low tail"),
                     ("rollout_corr/rollout_is_ratio_fraction_high", "fraction at cap")]:
        x, y = series(run, key)
        v, _ = level(x, y, lo, hi)
        extra = ""
        if key.endswith("batch_norm_factor") and f(v) and v > 0:
            extra = f"   -> update scaled up {1.0/v:.2f}x"
        print(f"  {lab:24s} {v:9.4f}{extra}")
    print()

    print("DEGENERACY SCREEN (codec-FREE observables only)")
    for key, lab in [("critic/score/max", "score max"), ("critic/score/min", "score min"),
                     ("response_length/mean", "response length"),
                     ("actor/grad_norm", "grad_norm"),
                     ("rollout_corr/rollout_log_ppl", "rollout log ppl (sampler-side)")]:
        x, y = series(run, key)
        v, _ = level(x, y, lo, hi)
        print(f"  {lab:32s} {v:12.4f}")
    x, y = series(run, "actor/grad_norm")
    if x is not None:
        m = (x >= 10) & np.isfinite(y)
        if m.sum():
            print(f"  grad_norm max at step>=10        {np.nanmax(y[m]):12.4f}   (step 1-3 transient excluded)")
    x, y = series(run, "actor/entropy")
    v, _ = level(x, y, lo, hi)
    print(f"  entropy (CODEC VIEW, ~34x inflated: NOT a health signal)  {v:.4f}")
    print()

    if args.compare:
        inc = resolve(*INCUMBENT)
        if inc is None:
            print("incumbent not resolvable")
        else:
            ixs, isc = series(inc, "critic/score/mean")
            ixg, igp = series(inc, "rollout_corr/kl")
            ixd, idr = series(inc, "actor/kl_loss")
            print(f"INCUMBENT {INCUMBENT[0]} over the SAME window {lo}-{hi}")
            print(f"  score level {level(ixs, isc, lo, hi)[0]:9.4f}"
                  f"   gap level {level(ixg, igp, lo, hi)[0]:9.4f}"
                  f"   drift slope {slope(ixd, idr, lo, hi)[0]:+9.6f}")
            print(f"  (its 61-120 drift slope is {slope(ixd, idr, slo, shi)[0]:+.6f}; the bar was "
                  f"derived from the 100-120 figure, see amendment 1)")


if __name__ == "__main__":
    main()
