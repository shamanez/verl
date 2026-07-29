#!/usr/bin/env python3
"""a5_tripwire.py - in-flight collapse tripwire for issue #93 cell a5.

Why this exists: the blind-committed uniformization guard (U1-U4) watches
`actor/entropy` going UP, because flattening was the predicted risk for a4's CVC
arm. a5 does the opposite: its codec-view entropy DECLINES, and that decline is
provably benign. With a5's reference KL at 0.0025 nats, Pinsker bounds total
variation at sqrt(KL/2) = 0.035, and Fannes-Audenaert over a 151936-token vocab
then caps any true policy-entropy change at about 0.575 nats. The measured drop
is 2.24 nats, i.e. 3.9x that ceiling, so it cannot be policy movement. The dense
uncompressed control confirms it directly: dense entropy reads 0.324 nats, so the
7.8 to 7.9 readings on every other arm are a saturated view-noise ceiling and
a5's 4.0 is simply the least corrupted readout in the matrix.

So codec-view entropy must NOT be a kill trigger here. The tripwire is pointed at
codec-free observables instead, and entropy alone is only a warning.

Rule (evaluated from step 30, rolling 6-step and 10-step windows):
  T1 entropy   6-step mean < 3.3, OR trailing 10-step slope < -0.05/step
  T2 reward    6-step mean score < 0.30, OR 10-step score slope < -0.004/step
  T3 behaviour 6-step mean response_length outside [600, 950]
  KILL on T1 AND (T2 OR T3).  T1 alone is a warning.
  HARD FLOOR   any step with entropy < 1.5 AND score < 0.25 -> immediate kill.
  DEGENERACY   E[rho] 6-step median > 0.9 concurrent with T1 corroborates a real
               collapse, since sharpening mechanically drives view agreement to 1.
  WATCH-ONLY   score 6-step mean < 0.42 at step >= 60 means stable but
               learning-impaired: changes the promotion story, not the run.

Prints one line per invocation. Exits 0 always; the caller greps for KILL/WARN.
"""

from __future__ import annotations

import sys

import numpy as np

ENTITY = "shamanework-pl"
PROJECT = "93-long-horizon-stability"
RUN = "a5-frlr-r48k28-tis"
KEYS = [
    "actor/entropy",
    "critic/score/mean",
    "response_length/mean",
    "rollout_corr/kl",
    "rollout_corr/k3_kl",
    "rollout_corr/rollout_log_ppl",
]


def _slope(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 4:
        return float("nan")
    xx, yy = x[m], y[m]
    xb = xx.mean()
    sxx = ((xx - xb) ** 2).sum()
    if sxx == 0:
        return float("nan")
    return float(((xx - xb) * (yy - yy.mean())).sum() / sxx)


def main() -> int:
    import argparse

    import wandb

    # There was NO argument parsing here. The script matched a module constant
    # and never read sys.argv, so `--run <anything>` was silently discarded and
    # every invocation reported on RUN below. That made it print a confident
    # tripwire line about a finished, unrelated run while appearing to watch the
    # cell it was asked about.
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=RUN)
    ap.add_argument("--project", default=PROJECT)
    ap.add_argument("--entity", default=ENTITY)
    args = ap.parse_args()

    api = wandb.Api()
    run = None
    try:
        run = api.run(f"{args.entity}/{args.project}/{args.run}")
    except Exception:
        for c in api.runs(f"{args.entity}/{args.project}"):
            if c.name == args.run:
                run = c
                break
    if run is None:
        print(f"TRIPWIRE: run {args.run!r} not found in {args.entity}/{args.project}")
        return 0
    print(f"TRIPWIRE target: {run.name} ({run.id}) state={run.state}")

    # Pull each key SEPARATELY and merge on global_step. A single
    # scan_history(keys=[...many...]) returns only rows where every requested key
    # is present, and on a live run that intersection lags badly: with all six
    # metrics requested together this returned rows only to step 120 while each
    # key on its own reached 158. A tripwire that is 38 steps stale protects
    # nothing, and it reports a confident-looking number while doing it.
    per_key = {}
    for k in KEYS:
        m = {}
        try:
            for r in run.scan_history(keys=["training/global_step", k]):
                s, v = r.get("training/global_step"), r.get(k)
                if s is None or v is None:
                    continue
                m[float(s)] = float(v)
        except Exception:
            pass
        per_key[k] = m

    steps = sorted({s for m in per_key.values() for s in m})
    if not steps:
        print("TRIPWIRE: no history")
        return 0
    arr = {"step": np.array(steps, float)}
    for k in KEYS:
        arr[k] = np.array([per_key[k].get(s, np.nan) for s in steps], float)
    x = arr["step"]
    if len(x) == 0:
        print("TRIPWIRE: no history")
        return 0
    hi = int(np.nanmax(x))
    if hi < 30:
        print(f"TRIPWIRE: step {hi}, below the step-30 activation threshold, no evaluation")
        return 0

    w6 = x > hi - 6
    w10 = x > hi - 10
    ent6 = float(np.nanmean(arr["actor/entropy"][w6]))
    sc6 = float(np.nanmean(arr["critic/score/mean"][w6]))
    rl6 = float(np.nanmean(arr["response_length/mean"][w6]))
    ent_sl = _slope(x[w10], arr["actor/entropy"][w10])
    sc_sl = _slope(x[w10], arr["critic/score/mean"][w10])
    erho = arr["rollout_corr/k3_kl"] - arr["rollout_corr/kl"] + 1
    er6 = float(np.nanmedian(erho[w6]))
    rlp6 = float(np.nanmean(arr["rollout_corr/rollout_log_ppl"][w6]))

    t1 = (ent6 < 3.3) or (np.isfinite(ent_sl) and ent_sl < -0.05)
    # CALIBRATION WARNING measured 2026-07-25 on a5b at step 160: the -0.004
    # 10-step slope condition fires on 20 percent of 10-step windows in the
    # incumbent's own healthy 600-step run, because step-to-step score sd is
    # 0.0453 and a 10-step OLS slope of that size is ordinary noise. T2's slope
    # arm therefore carries almost no information on its own. It is only safe
    # because the kill rule is a CONJUNCTION, T1 AND (T2 OR T3); never promote T2
    # to a standalone trigger, and never read a lone T2 flag as a collapse onset.
    t2 = (sc6 < 0.30) or (np.isfinite(sc_sl) and sc_sl < -0.004)
    t3 = not (600.0 <= rl6 <= 950.0)
    degen = er6 > 0.9
    floor = bool(np.any((arr["actor/entropy"] < 1.5) & (arr["critic/score/mean"] < 0.25)))

    state = "OK"
    if floor:
        state = "KILL"
    elif t1 and (t2 or t3):
        state = "KILL"
    elif t1:
        state = "WARN"

    flags = "".join(
        [
            "T1" if t1 else "--",
            "/T2" if t2 else "/--",
            "/T3" if t3 else "/--",
            "/DEGEN" if degen else "",
            "/FLOOR" if floor else "",
        ]
    )
    extra = ""
    if hi >= 60 and sc6 < 0.42:
        extra = "  [WATCH: learning-impaired, score below the 0.42 interpolated incumbent trajectory]"

    print(
        f"TRIPWIRE {state} step={hi} flags={flags} | ent6={ent6:.3f} entslope={ent_sl:+.4f} "
        f"| score6={sc6:.4f} scslope={sc_sl:+.5f} | rl6={rl6:.0f} | Erho6={er6:.3f} "
        f"| rollout_log_ppl6={rlp6:.4f}{extra}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
