# Pre-registration: a9-frlr-anchorq-200. Anchor-owned FRLR basis Q.

Written 2026-07-26T13:30Z, **before the cell exists**. a8 is at step 144/200 and
a9 launches when it ends. Nothing below may be edited after a9 logs its first
step; amendments get appended and dated, as with `PREREG_a6.md`.

## What changes, and only this

a7's exact codec (FRLR r48/k28, no token-IS, probe cadence 5), with the fast path
removed as a `Q` writer. The basis is harvested from the anchor's clean
stale-weight forward and refreshed **only when the anchor fires**, at cadence 20
optimizer ticks. This is the operator's instruction of 2026-07-26, "Q update only
in the anchor and only when it fires, like in normal powerSGD Q", and it is the
governance PowerSGD has always had.

It is **not** a re-run of a8. a8 raised `frlr_q_cadence` from 1 to 20, which
slowed the FAST refresh but kept it fitted to the policy being compressed. Two
things are different here:

1. `Q` is fitted to the **slow, stale-weight net**, so it cannot chase the policy
   at all. a8 could still chase, only less often.
2. the `Q` broadcast rides the **slow circuit**, which this program does not
   charge to the boundary wire budget (the anchor's dense replay is not charged
   either). That erases the 1233.4-against-1232 discrepancy from the wire-budget
   correction, so FRLR regains **exact parity** with the PRF incumbent.

Code landed in `1ff5e775` and `f0f4a167` (25 new tests, 159 pass, ruff clean).
The `comm_eff.py` validator previously rejected this combination outright on the
premise that the codec "has no PowerSGD basis Q for the anchor to own", which is
true of the plain PRF mask and **false** of FRLR.

## The registered bar, unchanged from a6/a7

| gate | threshold | window |
|---|---|---|
| G1 learning, score level | >= 0.6248 | 100-120 |
| G2 gap level | < 14.2458 | 100-120 |
| G2 gap slope | <= +5.0e-4 | 100-120 |
| G3 drift slope | <= 3.264e-3 | 100-120 |
| G4 wire | 1232 bits/token/boundary | n/a |

G3 is scored for bar-compliance and carries **no physical content**: per
`FINDING_drift_metric_invalid.md`, `actor/kl_loss` is the true drift times a
codec-specific, time-varying factor observed at 10.1x to 710.2x, and it ranks the
three probe cells the **wrong way** against capability (Spearman +1.00, where
`probe/kl_dense` gives the correct -1.00). The physical read is
`probe/kl_dense`.

## Predictions

Registered so that a miss is a miss, not a re-interpretation.

- **P1. The gap slope at 100-120 comes in at or below a8's +0.001262.** This is
  the arm's whole point. Anchor ownership is the limit case of what a8 validated:
  a8 froze `Q` for 20 steps and the slope flattened 13x from a7's +0.016351, with
  the gap falling monotonically 11.7151 -> 8.0304 -> 6.8293. Fitting `Q` to the
  slow net should flatten it at least as much, because the estimator no longer
  tracks the policy at all.
- **P2. The gap LEVEL lands between a7's 5.0849 and a8's 6.8293.** a8's higher
  level is under-fitting: only 10 refreshes in 200 steps, converging slowly from a
  random start. a9 gets the same 10 refreshes, so it inherits that handicap, but
  each refresh consumes a sketch accumulated over a full anchor window rather
  than 20 fast steps of the same batch stream. Prediction is deliberately a
  range, not a point.
- **P3. Learning at 100-120 is within 0.02 of a7's 0.6559**, i.e. it passes G1.
  Nothing here touches the loss, the weighting or the optimizer; a codec whose
  basis is slightly staler should not move capability much. **If a9 learns
  materially worse than a7, the every-step-fitted `Q` was load-bearing for
  capability**, which would be the most interesting negative result available.
- **P4. `probe/kl_gain` (the codec-view inflation) FALLS relative to a7's 710.2x.**
  Mechanism, and this is the one I would most like tested: a7's inflation rises
  because `Q` is refit to the current policy every step while the FROZEN reference
  is reconstructed ever worse. A `Q` fitted to the slow net is closer to the
  reference's own geometry, so the two views should diverge less. If P4 fails,
  the inflation mechanism in `FINDING_drift_metric_invalid.md` is incomplete.
- **P5. Wire cost is exactly 1232 bits/token/boundary**, since the basis moves on
  the uncharged slow circuit. Verified from config, not measured.

## Early-kill triggers, per the operator's "do not wait for 200 steps"

Any ONE of these ends the cell:

- score at 41-60 below **0.40** (a6's failure signature)
- gap above **12** at step 60, since it then cannot beat PRF on level
- gap slope at 61-80 above **+0.016**, a7's failing value, since a flatter trend
  is the entire hypothesis

A trigger evaluated **before** its window is non-evidence. Four early-window
over-reads happened in this program, including one that nearly justified killing
a8, the cell that identified the mechanism.

## The must-fire invariant

The engine asserts that the anchor's clone forward harvested a **non-empty**
sketch before refreshing `Q`, and that the refresh produced a basis. Without it a
silently no-op'd `register()` would leave `Q` frozen at its random bootstrap for
the whole run while the arm still reported as anchor-owned. Expect
`[comm_eff][frlr-anchor-q] refreshed` at the first anchor fire, which at
train_batch 128 / mini 128 (one tick per step) is **step 20**. If the assert
trips, it trips there and loudly, roughly 35 minutes in.
