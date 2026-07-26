# Verdict: a10-frlr-anchorq-unbiased-200. KILLED at step 62 on two registered triggers. The codec's bias is what buys the entire gap advantage, and removing it barely touches real drift.

Killed 2026-07-26T23:40Z at step 62 of 200 under the pre-registered early-kill
triggers in `PREREG_a10.md`, per the operator's instruction not to wait for 200 steps
when the signal is clear. Cell: a9's exact configuration plus
`frlr_unbiased=true`, so the residual gain becomes the constant `H/k` and
`E[h_hat | h, Q] = h` exactly. One environment variable from a9.

## The triggers that fired

| trigger | measured 41-60 | threshold | result |
|---|---|---|---|
| a10-specific gap level | **14.8751** | <= 8.7078 (1.5x a9's 5.8052) | **KILL, 2.56x a9** |
| gap at step 60 | **14.9201** | <= 12.0 | **KILL** |
| score level | 0.5179 | >= 0.40 | pass |
| gap slope | +0.008671 | <= +0.016 | pass |

The gap ceiling was registered as a **ratio** before a9 existed and pinned to an
absolute the moment a9's window closed, so it was not chosen after seeing a10.

## The result: bias buys the gap, and its removal buys almost no drift

**a10's gap is 14.8751, which is the INCUMBENT's operating point** (PRF exact-k sits
at 14.2458). The entire 2.4x gap advantage that FRLR has over the incumbent comes
from the biased, capped, per-token norm-matching gain. Make the codec unbiased and
FRLR is no better than the mask it was built to beat.

And the thing the arm existed to test:

| `probe/kl_dense`, codec-FREE | a9 **biased** | a10 **unbiased** | ratio |
|---|---|---|---|
| step 25 | 0.000285 | 0.000256 | 0.90x |
| step 50 | 0.001498 | 0.001471 | 0.98x |
| step 60 | 0.002511 | 0.002419 | 0.96x |

Removing the bias lowers real drift by **2 to 10 percent**. That is consistently in
the predicted direction at all three probe points, so the effect is probably real,
and it is the **same order as the arm-to-arm noise** already established across
a7/a8/a9 (4 to 8 percent at matched steps). Against a gap penalty of **2.56x**, the
trade is not close.

| at 41-60 | a9 biased | a10 unbiased |
|---|---|---|
| gap | **5.8052** | 14.8751 |
| score | 0.5385 | 0.5179 |
| `actor/kl_loss` | 0.0637 | 0.1775 |
| `probe/kl_gain` | 106.5x | 322.7x |
| wire | 1232 bits | **1216 bits** |

a10 is the only arm in the program **below** the wire parity budget, because the
unbiased gain makes the per-token norm scalar unnecessary (76 coordinates rather than
77, verified from the runtime `mask_ratio` of 0.9505208 = 1 - 76/1536, not merely from
the config). It bought 16 bits and paid 9 nats.

## Prediction scorecard

| # | prediction | outcome |
|---|---|---|
| P1 | gap level WORSE than a9's | **CONFIRMED**, and far beyond the registered margin: 2.56x, not the 1.5x kill line |
| P2 | gap slope within 2x of a9's either way | **CONFIRMED**: +0.008671 vs +0.009262, a ratio of 0.94 |
| P3 | codec-free drift at 200 at or below a9's | **trending CONFIRMED** at 0.90-0.96x through step 60; not scored at 200 because the cell was killed |
| P4 | terminal val within the reference's 0.0120 spread of a9's | **not scored**, cell killed |

P1 was registered against my own interest in the arm and it is confirmed. The
mechanism is the one stated in the pre-registration: the constant `H/k` gain is
unbiased in expectation but has **higher per-token variance**, and the gap is a
per-token KL, so it is exactly the quantity that variance inflates. The capped gamma
trades a little bias for a lot of variance reduction, and for this observable that is
a very favourable trade.

## What this settles, and what it does NOT

**Settles, for FRLR:** directional bias in the residual gain is **not** what drives
reference drift. The unbiased arm drifts 2-10 percent less while its mismatch is
2.56x worse. If bias were the mechanism behind divergence, this arm should have shown
a large drift improvement, and it did not.

**Does NOT overturn the a1/a2 factorial.** That result (biased round-to-nearest killed
at step 60 with 6.9x worse drift at z=+15, unbiased stochastic rounding surviving) was
measured on **sr_quant, a 1-bit quantizer**, where the bias is a rounding bias on every
coordinate. a10's bias is a **norm-matching scalar on a low-rank residual**, a
structurally different object: it is a single detached per-token multiplier, not a
per-coordinate systematic error. So the honest statement is that **bias matters for
some codecs and not for this one**, and the a1/a2 caveat still stands on its own terms
(it was measured on the discredited `actor/kl_loss` channel with no probe on either
arm).

**The operator was right to insist this be tested.** I had demoted it on the wrong
reasoning, that a8 flattening while biased showed bias was irrelevant. The correct
reason to run it was that the question was open and cheap, and the answer turned out
to be worth having: it is now established that FRLR's advantage over the incumbent
rests entirely on a **biased** estimator, which is a real and previously unstated
property of the method being proposed.

## Cost and consequence

Killed at step 62 of 200, so about **2 GPU-h spent of 6.5 budgeted**, and the 4.5 h
saved went straight into run 3. The pre-registered selection rule for run 3 was
"flattest gap slope at 100-120 among {a9, a10} at G1-passing capability"; a10 is
disqualified, so **a9 wins by the rule**, and a9 also satisfies the operator's
architectural constraint.

`c600-a9-anchorq-val600` launched at 23:45:53Z: a9's configuration at **600 steps**,
val at 300 and 600, probe cadence 5, **R2 sink deliberately OFF** because a9 showed
the in-training sink idles the GPU at teardown (three saves would cost about 2.5 h).
GPU idle across the a10-to-run-3 handoff was about **6 minutes**.
