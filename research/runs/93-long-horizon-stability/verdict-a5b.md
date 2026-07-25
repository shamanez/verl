# Verdict: a5b-frlr-bnorm-200. FAILS the registered bar. Decisive clause is G3.

Scored 2026-07-25T17:35Z at the registered primary window **100-120**, window
**complete at 21 rows**. Cell: FRLR r48/k28 + token-IS 2.0 +
`rollout_is_batch_normalize=true`. Scored with
`research/scripts/score93_bar.py`, thresholds hardcoded from `PREREG_a6.md` and
its two amendments. The run continues to 200.

## The registered bar

| gate | measured | bar | call |
|---|---|---|---|
| **G1** learning, `critic/score/mean` level | **0.6277** | >= 0.6248 | **COIN FLIP** |
| **G2** gap, `rollout_corr/kl` level | **4.4486** | < 14.2458 | **PASS**, 0.31x, decisive |
| **G2** gap slope | **+0.000358** | <= +5.0e-4 | pass, but **fragile**, see below |
| **G3** drift, `actor/kl_loss` slope | **+0.008091** | <= 3.264e-3 | **FAIL, 2.48x, unambiguous** |
| **G4** wire | 1232 bits | = 1232 | PASS, automatic, no information |

So a5b fails **exactly one** clause. That clause is decisive and is the one that
cannot be argued with.

## G1 is a coin flip, not a win

| quantity | value |
|---|---|
| window mean | 0.6277 |
| margin over bar | **+0.0029** |
| SE, iid | 0.0049 |
| SE, Newey-West L=3 | 0.0056, so the margin is **+0.52 SE** |
| moving-block bootstrap 95 percent CI | **[0.6173, 0.6389]**, which **contains the bar** |
| bootstrap P(mean > bar) | **0.629** |

a5b is statistically indistinguishable from the learning bar. It must not be
reported as clearing it. What is solid is the comparison against a5 at the last
complete matched window 81-100: **0.5733 against 0.5312, +7.9 percent**, so batch
normalisation did move learning, and it moved the cell from a clear miss to a
coin flip. Against the incumbent's 0.6577 at the same window, a5b is at 0.95x.

## G3 is the decisive failure, and it is not noise

Drift accelerates monotonically across every window in the run:

| window | drift level | drift slope |
|---|---|---|
| 21-40 | 0.005638 | +0.000357 |
| 41-60 | 0.025834 | +0.001548 |
| 61-80 | 0.068307 | +0.003234 |
| 81-100 | 0.149706 | +0.004583 |
| **100-120** | **0.268114** | **+0.008091** |

Five windows, monotone in both level and slope, ending at 2.48x the bar. There is
no window choice that rescues it: the secondary 61-120 window gives +0.005054,
still 1.55x the bar.

And the **level has crossed the incumbent**. a5b sits at 0.268114 against the
incumbent's 0.203385 at the identical window, **1.32x**. The FRLR arms had
previously been below the incumbent on drift level at every measured window,
which was the strongest thing in their favour. That advantage is gone. The
round-A correction memo fitted a5's crossing at about step 122; a5b crossed
inside 100-120, so the direction of that prediction is confirmed on a related arm.

## G2's slope clause passes, and it is not a well-posed test. Both are true and I am reporting both.

The gap **level** passes decisively and that stands on its own.

The gap **slope** clause passes at +0.000358 against a +5.0e-4 bar, and that pass
should not be banked, because **one data point flipped it**. At 20 rows the same
window read **+0.001408, a FAIL at 2.8x the bar**. Adding step 120 moved it to
+0.000358, a pass. A registered clause whose sign is decided by the 21st sample is
not measuring a trend.

The reason is a scale mismatch that was visible in advance:

- within 100-120 the gap oscillates between 4.312 and 4.606, a range of **0.295 nats**
- the +5.0e-4 bar is trying to detect **0.010 nats** of movement over those 20 steps

The oscillation is **30x** the signal the gate targets, so a 20-step gap slope
reports oscillation phase rather than trend. On every window of 80 steps or more
the trend is **negative**: -0.009490 over 2-120, -0.005621 over 41-120, -0.001996
over 21-120, with the level going 11.621 to 4.386. The gap really is falling,
which is the a5 line's signature property and it reproduces here.

So this clause is recorded as **passed but uninformative**, and the same fragility
would have applied had it failed. It is not counted for the cell and it would not
have been counted against it. **G3 is decisive on its own and no reading of G2
changes the outcome**, which is the only reason this ambiguity is tolerable here.
Round B and C should either lengthen the gap-slope window to 80 steps or widen the
bar to exceed the oscillation amplitude; as written it is a coin toss.

## Verdict and program action

**Cell verdict: FAIL.** a5b does not displace `90-prf-exactk-600`. It buys a 3.2x
better gap level and roughly equal learning in exchange for **1.32x the drift
level and 2.48x the drift slope**, on an accelerating trend, against an incumbent
whose own drift saturates and reaches only 0.91 nats by step 600. That is a bad
trade under this program's cardinal rule, which is not to damage the base model.

**Program action: proceed to a6 exactly as registered. Do NOT promote a5b to
round B or C.** a6 is already chained and launches on a5b's exit. Its value is now
higher than when it was queued, because a5b has produced a clean, unambiguous
drift failure and a6 is the cell that attributes it:

- if a6 (PRF exact-k + the same token-IS + normalize) shows the same accelerating
  drift, **the weighting causes it** and token-IS closes as a line for this program
- if a6 shows the incumbent's benign drift, **FRLR causes it**, and the gap win is
  purchasable only with a codec that damages the model

Either answer is publishable and neither is available without a6.

## Riders

- The run continues to 200, which is deliberate and should not be cut short: 80
  more steps is the only evidence that will settle whether this acceleration
  saturates the way the incumbent's did after step 80 or keeps compounding. That
  reads as a secondary at termination and does not alter the registered verdict.
- Two of the three registered falsifiers were evaluable. The first was withdrawn
  as vacuous before scoring (Kish ESS is scale-invariant, so batch normalisation
  cannot move it by construction; measured 0.2385 to 0.2756). See `PREREAD_a5b.md`.
- The V1 dense clause passes by 77x on a unit mismatch and is reported as **not
  discriminating**, per amendment 2, not as a pass earned on the merits.
- Health is clean: zero error markers, `grad_norm` max 2.0774 excluding the step
  1-3 transient, `actor/ppo_kl` exactly 0 by construction, score max/min 1.0/0.0
  so the reward is not degenerate, response length 688.
- Checkpoint `global_step_100` exists on the box (19 G). It is **local only**, the
  R2 sink is off for this cell, so post-hoc geometry on a5b is available while the
  box lives and is lost at teardown. Flagged for the operator, not acted on.
