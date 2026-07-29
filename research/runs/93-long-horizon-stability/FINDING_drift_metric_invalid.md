# Program-level finding: `actor/kl_loss` is not a drift measurement, and the drift veto is not valid across codecs

Established 2026-07-25T22:15Z from a5b's completed probe series and a6's first 100
steps. This is not a cell verdict. It concerns the criterion the whole matrix was
gated on, so it is written separately and flagged for the operator.

## 1. a5b's "2.23 nats of drift" is 0.0168 nats of drift and a 133x measurement offset

a5b's step-200 probe fired, so both channels exist at every 25 steps:

| step | `actor/kl_loss` (codec view) | `probe/kl_dense` (codec-FREE) | inflation |
|---|---|---|---|
| 25 | 0.00348 | 0.000252 | 13.8x |
| 50 | 0.02499 | 0.000752 | 33.2x |
| 75 | 0.08262 | 0.002383 | 34.7x |
| 100 | 0.19011 | 0.003857 | 49.3x |
| 125 | 0.41356 | 0.006258 | 66.1x |
| 150 | 0.80699 | 0.008710 | 92.6x |
| 175 | 1.40565 | 0.012904 | 108.9x |
| **200** | **2.22616** | **0.016754** | **132.9x** |

The probe runs the same weights through the same forward pass with the codec
silent, no backward and no weight change, so `probe/kl_dense` is the actual
policy's KL to the reference. **a5b's true drift at step 200 is 0.0168 nats.** The
2.226 figure is that number multiplied by an offset which itself grew **9.6x** over
the run.

This dissolves the puzzle from the terminal addendum. a5b validating at parity was
never surprising: **the base model was barely moved.** 0.0168 nats is not a
capability threat, and no explanation involving "damage is collapse-only" is
required.

It also means the a5b verdict's central quantity was mostly artifact. The FAIL
stands as bar compliance, because G3 was registered on `actor/kl_loss` and
registered bars do not move after the data. But what it measured was not drift.

## 2. The two channels rank the arms in OPPOSITE order

a6 carries the same weighting on the incumbent's codec, and it has a dense channel
at cadence 5. At matched step 100:

| channel | a6 (PRF) | a5b (FRLR) | who looks worse |
|---|---|---|---|
| codec view, `actor/kl_loss` 81-100 | 0.08037 | 0.14971 | **a5b, by 1.86x** |
| codec FREE, `probe/kl_dense` at 100 | **0.006561** | 0.003857 | **a6, by 1.70x** |

**The ordering reverses.** In codec view a6 is the best arm in the matrix on drift,
with a slope of +0.000712 against the incumbent's +0.002381. In truth it is
drifting **more** than a5b, and the ratio is widening: 1.22x at step 50, 1.44x at
75, **1.70x** at 100.

The mechanism is the opposite motion of the two offsets, documented in
`PREREAD_a6.md`: PRF exact-k draws from a fixed stationary mask so its inflation
**falls** (134.6x to 14.3x over 100 steps) while FRLR refreshes its basis every step
so its inflation **rises** (13.8x to 132.9x over 200). Codec-view drift is the
product of a real quantity and a codec-specific, time-varying, non-monotonic factor
spanning an order of magnitude in each direction. **A cross-codec comparison of that
product carries no information about drift.**

## 3. What this does and does not invalidate

**Invalidated: any cross-codec comparison of `actor/kl_loss`.** That includes the
drift column of the round-A matrix, the V1 veto as applied across arms, and the a5b
G3 result read as a physical claim. Round A had **no probes at all**, so none of its
drift numbers can be corrected retrospectively.

**Not necessarily invalidated: the a1/a2 factorial**, which is the source of the
"coherence not magnitude" law. That was a within-codec-family comparison, sr_quant
bits=1 block=32 differing only in rounding mode, at identical wire. If both arms
carried a similar offset, the 6.9x difference at z=+15 survives. **But this is not
established**, and there is a specific reason for doubt: rounding mode is exactly
the kind of change that alters a view offset, since a biased codec displaces the
view systematically where an unbiased one does not. Neither arm has probe data, so
this is an open question, not a refutation. It should be listed as such rather than
assumed safe.

**Unaffected:** the gap findings, which were established structurally at step 1 and
cross-checked against the dense channel; the wire budgets, which are computed from
source; and a5b's val result, which is a direct capability measurement.

## 4. Recommendation, which is the operator's call

This is now the **second independent reason** the drift veto should not gate this
program. The first was that it fails to predict capability: a5b missed G3 by 2.48x
and validated at parity. The second is this one, that it is not even a valid
comparison between codecs.

Recommended change to the registered procedure:

1. **Demote `actor/kl_loss` from veto to reported diagnostic**, labelled as a
   codec-view quantity, never compared across codecs.
2. **Promote `probe/kl_dense` to the drift criterion.** It is codec-free by
   construction and is the quantity the cardinal rule actually cares about.
3. **Require a probe on every future cell**, at cadence 5. Measured cost is about
   20 s per probe, 13 min on a 200-step run, roughly 3 percent. Round A's arms are
   permanently unanalysable on drift because they lacked this, which is a far larger
   cost than 3 percent.
4. **Gate promotion on capability directly**, val and OOD, which round C already runs.

## 5. The gap this leaves, and a cheap way to close it

There is one structural hole. **The incumbent has no probe**, so its codec-free
drift is unknown, and the comparison that actually matters, whether token-IS adds
true drift, cannot be made: a6 versus a5b isolates the codec because both carry the
weighting, but there is no no-weighting arm with a dense channel.

Closing it needs one cell: the incumbent config, probe at cadence 5, no token-IS.
That is the missing corner of the 2x2 on the codec-free channel. It would also give
the first trustworthy drift number for the arm this program has been treating as its
reference for the whole matrix. **New spend, so it is not being started.**
