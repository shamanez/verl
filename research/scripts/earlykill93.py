#!/usr/bin/env python3
"""Evaluate an issue #93 cell's PRE-REGISTERED early-kill triggers over one window.

The registered bar is scored at 100-120 by `score93_bar.py`. This is the separate,
narrower question the operator's "do not wait for 200 steps" instruction creates:
may this cell be killed EARLY? The triggers are fixed in each cell's PREREG and are
hardcoded here so they cannot drift once data exists.

Two deliberate design choices, both from defects this program already paid for:

1. **Per-key `scan_history`, merged on the step axis.** `history()` SAMPLES, and
   `scan_history(keys=[a, b])` returns only rows where EVERY key exists, which
   silently returned zero rows for one cell and 38-step-stale rows for another.
   Five wrong answers in one session came from this.
2. **Refuses to evaluate a window it does not fully have.** A trigger read before
   its window is complete is not evidence. Three short-window over-reads happened
   in this session alone, and one nearly justified killing a8, the cell that
   identified the gap mechanism.

The gap trigger is read at the SINGLE step `hi`, not as a window mean, because the
PREREG says "gap > 12 at step 60". That is noisier than a mean and is honored as
registered rather than quietly reinterpreted; the window level is printed alongside
so both are visible.

Validated against three cells before first use: it refuses a9's incomplete 41-60,
returns CONTINUE on a8's complete 41-60 (score 0.5361, gap 9.5067), and returns
KILL on a7's 61-80 at +0.016365 against the +0.016 ceiling. The a7 case matters
because the ceiling was DERIVED from a7's slope, so reproducing it confirms the
arithmetic matches the historical figure rather than merely running.

Usage:
    python3 earlykill93.py --cell a9-frlr-anchorq-200 --lo 41 --hi 60
"""

from __future__ import annotations

import argparse
import sys

# Pre-registered early-kill triggers (PREREG_a9.md, PREREG_a10.md). Any ONE fires.
SCORE_FLOOR = 0.40  # score level over 41-60; a6's failure signature
GAP_CEILING = 12.0  # gap level at step 60; above this it cannot beat PRF on level
GAP_SLOPE_CEILING = 0.016  # gap slope over 61-80; a7's failing value

ENTITY = "shamanework-pl"
PROJECT = "93-long-horizon-stability"
# NOT "global_step": the run's step axis is logged under this key. Using the bare
# name silently returns ZERO rows, which a first version of this script did even on
# a finished run. Same family as the scan_history pitfall documented below.
STEP = "training/global_step"


def series(run, key):
    """One key at a time, so a row missing another key cannot drop this one."""
    out = {}
    for row in run.scan_history(keys=[STEP, key]):
        gs, v = row.get(STEP), row.get(key)
        if gs is None or v is None:
            continue
        out[int(gs)] = float(v)
    return out


def ols_slope(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    if den == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / den


def window(d, lo, hi):
    ks = sorted(k for k in d if lo <= k <= hi)
    return ks, [d[k] for k in ks]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--lo", type=int, required=True)
    ap.add_argument("--hi", type=int, required=True)
    ap.add_argument("--project", default=PROJECT)
    ap.add_argument("--entity", default=ENTITY)
    a = ap.parse_args()

    import wandb

    api = wandb.Api()
    runs = [r for r in api.runs(f"{a.entity}/{a.project}") if r.name == a.cell]
    if not runs:
        sys.exit(f"no run named {a.cell} in {a.entity}/{a.project}")
    run = runs[0]

    score = series(run, "critic/score/mean")
    gap = series(run, "rollout_corr/kl")
    max_step = max(max(score, default=0), max(gap, default=0))
    print(f"=== earlykill93: {a.cell} ({run.id}) state={run.state} max_step={max_step} ===")
    print(f"window {a.lo}-{a.hi}")

    expected = a.hi - a.lo + 1
    sk, sv = window(score, a.lo, a.hi)
    gk, gv = window(gap, a.lo, a.hi)
    if len(sk) < expected or len(gk) < expected:
        print(f"\nINCOMPLETE: have {len(sk)} score / {len(gk)} gap rows, need {expected}.")
        print("A trigger read before its window is complete is NOT evidence. Refusing.")
        return

    score_level = sum(sv) / len(sv)
    gap_level = sum(gv) / len(gv)
    gap_slope = ols_slope(gk, gv)
    gap_at_hi = gap.get(a.hi)

    print(f"\n  score level      {score_level:.4f}   floor {SCORE_FLOOR}")
    print(f"  gap level        {gap_level:.4f}")
    shown = "n/a" if gap_at_hi is None else f"{gap_at_hi:.4f}"
    print(f"  gap at step {a.hi}   {shown}   ceiling {GAP_CEILING}")
    if gap_slope is None:
        print("  gap slope        n/a")
    else:
        print(f"  gap slope        {gap_slope:+.6f}   ceiling {GAP_SLOPE_CEILING:+.6f}")

    fired = []
    if a.lo == 41 and a.hi == 60 and score_level < SCORE_FLOOR:
        fired.append(f"score level {score_level:.4f} < {SCORE_FLOOR}")
    if gap_at_hi is not None and a.hi == 60 and gap_at_hi > GAP_CEILING:
        fired.append(f"gap at 60 {gap_at_hi:.4f} > {GAP_CEILING}")
    if gap_slope is not None and a.lo == 61 and a.hi == 80 and gap_slope > GAP_SLOPE_CEILING:
        fired.append(f"gap slope {gap_slope:+.6f} > {GAP_SLOPE_CEILING:+.6f}")

    print()
    if fired:
        print("KILL: " + "; ".join(fired))
    else:
        print("CONTINUE: no registered early-kill trigger fired for this window.")
    print("\nNote: this is the early-kill question ONLY. The registered bar is scored")
    print("at 100-120 by score93_bar.py and nothing here anticipates it.")


if __name__ == "__main__":
    main()
