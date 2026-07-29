# Pre-registration: a10-frlr-anchorq-unbiased-200. Does the codec's directional bias matter?

Written 2026-07-26T13:30Z, **before a9 exists, let alone a10**. a10 is chained
behind a9. Nothing below may be edited once a10 logs its first step.

## What changes, and only this

a9 plus `frlr_unbiased=true`. The residual gain becomes the **constant `H/k`**
instead of the capped, detached, data-dependent
`gamma = ||res|| / max(||scatter_J(res_J)||, eps)`. That makes
`E[h_hat | h, Q] = h` exactly, at **zero extra wire** (the per-token norm scalar,
the `+1` of "rank + k + 1", stops being sent, so the payload is 76 numbers rather
than 77). One environment variable from a9.

## Why this arm exists: I was wrong to drop it

On 2026-07-26 I demoted this test to unscheduled, arguing that a8 flattened the
gap trend **while still being the biased variant**, so bias is not the driver.
The operator pushed back. The pushback is correct and my reasoning was bad:

- a8 shows estimator variance is **sufficient** to explain much of the gap trend.
  It does **not** show bias is excluded. Two mechanisms can both contribute, and I
  treated "sufficient" as "exclusive".
- The operator was asking about a different quantity than I was reasoning about. I
  had the 200-step **gap trend** in mind; they asked about **divergence and
  collapse**. On that question the program's strongest evidence is theirs: the
  a1/a2 factorial killed the **biased** round-to-nearest arm at step 60 with
  **6.9x worse drift, z = +15**, while the unbiased stochastic-rounding arm
  survived. One env var apart.
- FRLR as run is **biased**. PRF exact-k, the incumbent, is **unbiased** (constant
  `1/(1-p)` gain, exact to 0.26%). So the program's two codecs differ on exactly
  this axis, and it has never been isolated within FRLR.

**The caveat that keeps the question open rather than settled.** The a1/a2 result
was measured on `actor/kl_loss`, the channel this program has since shown ranks
the wrong way against capability, and neither a1 nor a2 carries a probe. So the
bias-causes-drift law is **open**, not established. That cuts both ways, and it is
cheapness that settles the argument: one variable, 6.5 GPU-h, ahead of a 20-h
600-step commitment.

## The registered bar

Identical to a9 (G1 >= 0.6248, G2 level < 14.2458 and slope <= +5.0e-4, G3 slope
<= 3.264e-3 as a labelled non-physical gate, window 100-120). Wire is **1216
bits/token/boundary** here, not 1232: the norm scalar is no longer sent. That is
*below* parity, so G4 passes with room.

## Predictions

- **P1. The gap LEVEL is WORSE than a9's** (higher). The constant `H/k` gain is
  unbiased in expectation but has higher variance per token than norm matching,
  which is what the capped gamma buys. The mismatch is a per-token quantity, so
  variance should show up in it directly. Registering the direction against my
  own interest in the arm.
- **P2. The gap SLOPE is within 2x of a9's** either way. Bias is a property of the
  reconstruction, not of how `Q` moves, and the slope is now governed by anchor
  ownership. If the slope changes a lot, bias and basis dynamics are coupled in a
  way nothing in this program predicts.
- **P3. Codec-free drift (`probe/kl_dense`) at 200 comes in at or below a9's.**
  This is the arm's actual hypothesis: if directional bias accumulates into real
  reference drift the way the a1/a2 factorial says it does, removing it should
  show up here and nowhere else.
- **P4. Terminal val is within the reference's own 0.0120 checkpoint-to-checkpoint
  spread of a9's.** I do not expect capability to move. If it moves *up*, bias was
  costing capability, which would be the strongest result in the program.

**What each outcome licenses**, fixed now so it cannot be chosen later:

| a10 vs a9 | reading |
|---|---|
| drift lower AND val equal-or-better | bias contributes; unbiased goes into the 600-step run |
| drift equal, gap worse | bias is not the mechanism at this horizon; a9 goes forward, and a8's "variance not bias" reading is confirmed rather than assumed |
| drift higher | the capped gamma is doing real work beyond bias correction; record it and stop asking |

## Early-kill triggers

Same three as a9: score at 41-60 below 0.40, gap above 12 at step 60, gap slope at
61-80 above +0.016. Plus one specific to this arm: **kill if the gap level at
41-60 exceeds 1.5x a9's at the same window**, since P1 already concedes some
degradation and 1.5x would mean the unbiased gain is simply a worse codec, which
is answerable at step 60 rather than 200.
