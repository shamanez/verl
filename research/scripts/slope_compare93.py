#!/usr/bin/env python3
"""slope_compare93.py - matched-window slope comparison between two #93 cells.

Built for the issue #93 round-A kill gates, where a rule of the form "kill arm B
if its reference-KL slope is at least 2x arm A's" is only well posed on a FIXED
step window, because these slopes accelerate and a full-run fit can differ from
a matched-window fit by more than the factor being tested.

It fits OLS slopes for both runs over the SAME step window and reports:
  - the point slopes and their ratio
  - textbook iid standard errors
  - Newey-West HAC standard errors (residuals here are strongly autocorrelated,
    lag-1 ACF around 0.83 on these series, so the iid SE understates by ~2x)
  - a moving-block bootstrap interval, which does not assume a noise model
  - the kill/acquit/inconclusive call against a supplied multiplier

Usage:
  slope_compare93.py --ref a1-srq-b1-sr --test a2-srq-b1-rn \\
      --metric actor/kl_loss --lo 2 --hi 60 --multiplier 2.0

WANDB_API_KEY must be in the environment (source the laptop secrets first).
Never prints secret values.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

DEFAULT_ENTITY = "shamanework-pl"
DEFAULT_PROJECT = "93-long-horizon-stability"
STEP_KEY = "training/global_step"


def _pull(entity: str, project: str, name: str, metric: str):
    """Full (unsampled) history for one run, resolved by name or id."""
    import wandb

    api = wandb.Api()
    run = None
    for candidate in api.runs(f"{entity}/{project}"):
        if name in (candidate.name, candidate.id):
            run = candidate
            break
    if run is None:
        raise SystemExit(f"run {name!r} not found in {entity}/{project}")

    steps, vals = [], []
    for row in run.scan_history(keys=[STEP_KEY, metric]):
        s, v = row.get(STEP_KEY), row.get(metric)
        if s is None or v is None:
            continue
        steps.append(float(s))
        vals.append(float(v))
    x = np.asarray(steps, float)
    y = np.asarray(vals, float)
    order = np.argsort(x)
    return run, x[order], y[order]


def _ols(x, y):
    """Slope, intercept, residuals, iid slope SE."""
    n = len(x)
    xb, yb = x.mean(), y.mean()
    sxx = ((x - xb) ** 2).sum()
    slope = ((x - xb) * (y - yb)).sum() / sxx
    intercept = yb - slope * xb
    resid = y - (intercept + slope * x)
    dof = n - 2
    s2 = (resid**2).sum() / dof
    return slope, intercept, resid, float(np.sqrt(s2 / sxx))


def _hac_se(x, resid, lags: int):
    """Newey-West HAC standard error for the OLS slope, Bartlett kernel.

    Sandwich on the demeaned regressor, which is the slope-only equivalent of the
    full 2x2 form and avoids inverting X'X explicitly.
    """
    xc = x - x.mean()
    sxx = (xc**2).sum()
    u = xc * resid
    s = (u**2).sum()
    n = len(x)
    for lag in range(1, min(lags, n - 1) + 1):
        w = 1.0 - lag / (lags + 1.0)
        s += 2.0 * w * (u[lag:] * u[:-lag]).sum()
    return float(np.sqrt(max(s, 0.0)) / sxx)


def _block_bootstrap(x, y, block: int, draws: int, seed: int = 0):
    """Moving-block bootstrap of the slope. Preserves local autocorrelation."""
    rng = np.random.default_rng(seed)
    n = len(x)
    block = max(2, min(block, n))
    nblocks = int(np.ceil(n / block))
    starts_hi = n - block + 1
    out = np.empty(draws)
    for d in range(draws):
        starts = rng.integers(0, starts_hi, size=nblocks)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        out[d] = _ols(x[idx], y[idx])[0]
    return out


def _fit(entity, project, name, metric, lo, hi, lags, block, draws, label):
    run, x, y = _pull(entity, project, name, metric)
    m = (x >= lo) & (x <= hi)
    if m.sum() < 5:
        raise SystemExit(
            f"{label} {name}: only {int(m.sum())} points in window {lo}-{hi} "
            f"(run max_step={x.max() if len(x) else 'n/a'}); too few to fit"
        )
    xw, yw = x[m], y[m]
    slope, intercept, resid, se_iid = _ols(xw, yw)
    se_hac = _hac_se(xw, resid, lags)
    boot = _block_bootstrap(xw, yw, block, draws)
    lag1 = (
        float(np.corrcoef(resid[:-1], resid[1:])[0, 1])
        if len(resid) > 2 and resid.std() > 0
        else float("nan")
    )
    return {
        "label": label,
        "run": name,
        "wandb_id": run.id,
        "state": run.state,
        "max_step": float(x.max()),
        "n_in_window": int(m.sum()),
        "window": [lo, hi],
        "first": float(yw[0]),
        "last": float(yw[-1]),
        "slope": float(slope),
        "intercept": float(intercept),
        "se_iid": se_iid,
        "se_hac": se_hac,
        "resid_sd": float(resid.std(ddof=2)),
        "resid_lag1_acf": lag1,
        "boot_lo": float(np.percentile(boot, 2.5)),
        "boot_hi": float(np.percentile(boot, 97.5)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="reference cell (the incumbent arm)")
    ap.add_argument("--test", required=True, help="cell being judged")
    ap.add_argument("--metric", default="actor/kl_loss")
    ap.add_argument("--lo", type=int, default=2, help="window start (step 1 is 0.0 by construction)")
    ap.add_argument("--hi", type=int, default=60)
    ap.add_argument("--multiplier", type=float, default=2.0, help="kill if test slope >= mult x ref slope")
    ap.add_argument("--entity", default=DEFAULT_ENTITY)
    ap.add_argument("--project", default=DEFAULT_PROJECT)
    ap.add_argument("--lags", type=int, default=3, help="Newey-West bandwidth")
    ap.add_argument("--block", type=int, default=8, help="bootstrap block length")
    ap.add_argument("--draws", type=int, default=20000)
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    kw = dict(
        entity=args.entity,
        project=args.project,
        metric=args.metric,
        lo=args.lo,
        hi=args.hi,
        lags=args.lags,
        block=args.block,
        draws=args.draws,
    )
    ref = _fit(name=args.ref, label="REF", **kw)
    test = _fit(name=args.test, label="TEST", **kw)

    thresh = args.multiplier * ref["slope"]
    # Uncertainty on the threshold comparison: the test slope and the threshold
    # each carry error, so combine them. The threshold inherits mult x ref HAC SE.
    se_comb = float(np.sqrt(test["se_hac"] ** 2 + (args.multiplier * ref["se_hac"]) ** 2))
    margin = test["slope"] - thresh
    z = margin / se_comb if se_comb > 0 else float("nan")

    if z >= 1.0:
        call = "KILL"
        why = "test slope exceeds the threshold by more than the combined standard error"
    elif z <= -1.0:
        call = "ACQUIT"
        why = "test slope sits below the threshold by more than the combined standard error"
    else:
        call = "INCONCLUSIVE"
        why = "test slope is within one combined standard error of the threshold; decide on reward slope and gap corroboration, and prefer running to completion over killing on noise"

    print(f"=== slope_compare93: {args.metric} over steps {args.lo}-{args.hi} ===")
    for r in (ref, test):
        print(
            f"\n  [{r['label']}] {r['run']} ({r['wandb_id']}) state={r['state']} "
            f"max_step={r['max_step']:.0f} n={r['n_in_window']}"
        )
        print(f"    {r['metric'] if 'metric' in r else args.metric}: {r['first']:.6f} -> {r['last']:.6f}")
        print(f"    slope   = {r['slope']:+.6f}/step")
        print(f"    se      = {r['se_iid']:.6f} (iid)   {r['se_hac']:.6f} (Newey-West L={args.lags})")
        print(f"    boot95  = [{r['boot_lo']:+.6f}, {r['boot_hi']:+.6f}]  (moving block, {args.draws} draws)")
        print(f"    resid   = sd {r['resid_sd']:.6f}, lag-1 ACF {r['resid_lag1_acf']:+.3f}")

    ratio = test["slope"] / ref["slope"] if ref["slope"] != 0 else float("nan")
    print(f"\n  ratio test/ref            = {ratio:.3f}x")
    print(f"  threshold ({args.multiplier}x ref)      = {thresh:+.6f}/step")
    print(f"  test - threshold          = {margin:+.6f}  (combined se {se_comb:.6f}, z {z:+.2f})")
    print(f"\n  CALL: {call}")
    print(f"        {why}")
    print(
        "\n  NOTE: both arms are fitted over the IDENTICAL window on purpose. These\n"
        "        slopes accelerate, so a full-run reference fit would move the gate."
    )

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(
                {
                    "metric": args.metric,
                    "window": [args.lo, args.hi],
                    "multiplier": args.multiplier,
                    "ref": ref,
                    "test": test,
                    "ratio": ratio,
                    "threshold": thresh,
                    "margin": margin,
                    "se_combined": se_comb,
                    "z": z,
                    "call": call,
                },
                fh,
                indent=2,
            )
        print(f"\n  wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
