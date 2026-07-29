# Verdict: a6-prf-exactk-tis-bnorm-200. FAILS on G1, and the mechanism reverses my earlier reading of a5b

## STABILITY VERDICT (2026-07-28 re-scoring)

> **Re-scored against stability, not reward.** The body below this section was
> written against a bar that leads with capability. Capability turned out to be
> a tie across the field, so it cannot carry a conclusion. What follows
> supersedes the original ranking claims; the original text is kept in place.

**Stability rank: 12 of 12.** a6 is PRF exact-k plus token-IS plus batch
normalisation, and it is the most important counterexample in the program: it
produced the flattest gap of any 200-step arm and it collapsed anyway, so it is
the arm that proves gap stationarity alone cannot be the bar.

| axis | this arm | reference | read |
|---|---|---|---|
| gap slope | +0.000413 over window 100-199 (n=100); +0.000383 over the matched window 100-120 | incumbent +0.000848 over 100-599 (n=500); +0.000838 over 100-120 | flatter than the incumbent on both windows, and the flattest slope of any arm that reached 200 |
| gap drift ratio | 1.002 | incumbent 1.029 | nominally 14x closer to perfectly stationary, and it means nothing, see below |
| grad_norm drift | 2.27x, p50 17.7015 over the first 20 percent rising to 40.2098 over the last 20 percent, 200 steps | incumbent 0.85x, p50 1.7866 falling to 1.5259, 600 steps | second-worst drift in the program, beaten only by c600's 9.25x, and the only arm that combines a rising optimizer with a flat gap |
| grad_norm max | 608.81, max/p50 18.3x | incumbent 4.645, max/p50 2.9x | the largest run maximum measured anywhere in the program, 131x the incumbent's |
| collapse / kill | ran to its scheduled 200 and collapsed capability: val 0.5391 against a field of 0.6593-0.6713 | incumbent none in 600 | the only collapse in the program, and it was not a kill, nothing tripped |
| capability | 0.5391 @200 | incumbent 0.6613 @150, 0.6633 @300, 0.6733 @450, 0.6613 @600 | does not separate the arms; a6 sits outside the tie only because it collapsed, which is already carried by the collapse row |

What a6 proves is a decoupling. Gap stationarity measures the sampler/trainer
wedge, and on this arm that wedge was as still as anything in the program: level
14.132 at step 100 and 14.131 at step 120, ratio 1.002, slope +0.000413 across
100-199. Underneath that still surface the optimizer was doing the opposite. The
grad_norm median went 17.7015 to 40.2098 across the run, the run maximum reached
608.81, and the codec-free drift on `probe/kl_dense` went 0.006561 at step 100 to
0.010755 at 150 to 0.026793 at 200, which is 3.27x c600's 0.008186 and 1.60x
a5b's 0.016754 at the same step 200. Training reward confirms it: a6's 101-200
block mean is 0.5185, the lowest 101-200 figure in the whole table, against
0.6726 for the incumbent and 0.6833 for c600. So the flat gap here is flat for a
bad reason. It is the codec's own high plateau being reported back, not evidence
that anything was under control.

That is the disciplining result for the program's bar. If the bar had been "gap
slope at the matched window", a6 would have ranked third at 100-120, ahead of
the incumbent, and it would have been promoted. The bar has to be conjunctive:
flat gap AND a stationary optimizer AND no collapse. a6 clears the first and
fails the other two as hard as any arm in the field.

What a6 does NOT prove is an indictment of PRF exact-k. a6 runs the incumbent's
codec at the incumbent's 1232-bit wire, and the incumbent ran 600 steps with a
block median flat at 1.50-1.82 and a block max never above 4.645. The failure
therefore belongs to the token-IS plus batch-normalisation layer stacked on top,
not to the codec underneath. The fact sheet does not contain an arm that is PRF
exact-k plus token-IS with batch normalisation switched off, so the split of
blame between the weighting and the normalisation is not measured here. It is
also worth pairing a6 with a5, which the fact sheet records as the second
counterexample: a5's grad_norm drift is exactly 1.00x, the most stationary number
in the program, achieved by token-IS suppressing the update to a 101-200 reward
of 0.5895. Token-IS produces either a suppressed run or an exploded one on this
evidence, and neither is stability.

Three things in the body below now read as wrong, and they are corrected here
rather than edited away. First, the mid-run health section says a6 "is not
collapsing"; that was written before step 200 and the terminal val of 0.5391
against a field of 0.6593-0.6713 makes a6 the one collapse recorded in the
program. The terminal addendum's softer reading, that a6 improved on base and so
this is not damage, is not the frame used now: whether or not it fell below base,
it is the only arm outside the capability tie and that is what "collapse" names
here. Second, the body quotes a run max of 248.79 excluding the early transient;
the authoritative run max is 608.81 and that is the number to quote. Third, the
body ranks and grades a6 on `actor/kl_loss` slopes, quoting +0.001404 against the
incumbent's +0.002176. Those numbers are disqualified for ranking: `actor/kl_loss`
is real drift multiplied by a codec-view inflation factor that itself moves by
50x between arms and within a run, so it confounds the thing being measured with
the instrument. The body's own conclusion from that comparison survives and is
in fact strengthened, because the gate it describes would have passed the one arm
that collapsed. Finally, on codec-free drift the axis still cannot compare the
shipping PRF configuration against FRLR at all: the incumbent and the dense
control have no probe, so a6's probe characterises the token-IS plus batch-norm
variant and nothing more.

Scored 2026-07-25T23:30Z at the registered primary window **100-120**, complete at
21 rows. Cell: incumbent PRF exact-k codec + token-IS 2.0 +
`rollout_is_batch_normalize=true`. Run continues to 200 with a terminal val.

## The registered bar

| gate | measured | bar | call |
|---|---|---|---|
| **G1** learning, score level | **0.4854** | >= 0.6248 | **FAIL, and not marginally** |
| **G2** gap level | 14.1366 | < 14.2458 | pass by **0.8 percent**, meaningless |
| **G2** gap slope | +0.000383 | <= +5.0e-4 | pass |
| **G3** drift slope | +0.001404 | <= 3.264e-3 | pass, **but see the finding doc** |
| **G4** wire | 1232 bits | = 1232 | pass, automatic |

G1 fails at **0.74x the incumbent's 0.6577**, against a5b's 0.6277 at the identical
window. This is the pre-registered branch: *"G1 fails -> token-IS costs learning
even with the shrinkage removed, on a codec that is known to learn fine. That
indicts the weighting, not FRLR."*

## It is a real deficit, not an onset delay, and a5b is what proves it

The a5b lesson was that TIS arms start late and catch up, so a shortfall before step
100 means nothing. a6 has now had 150 steps and it is **not** catching up:

| window | incumbent | a5b | a6 | a6 / incumbent |
|---|---|---|---|---|
| 41-60 | 0.5246 | 0.3728 | 0.4148 | 0.79x |
| 61-80 | 0.6068 | 0.4609 | 0.4605 | 0.76x |
| 81-100 | 0.6240 | 0.5733 | 0.4773 | 0.76x |
| 101-120 | 0.6568 | 0.6270 | 0.4828 | 0.74x |
| 121-150 | 0.6602 | 0.6454 | 0.5144 | **0.78x** |

a5b closed from **0.71x to 0.98x** across this span. a6 sits flat at **0.74x to
0.79x** and does not move. Same weighting, same normalisation, opposite trajectory.
So the deficit belongs to the interaction, not to the weighting alone.

## The mechanism: `batch_normalize` amplifies by 1/mean_weight, and mean_weight collapses when the gap is large

| window | a5b ESS | a6 ESS | a5b IS mean | a6 IS mean | a5b grad_norm | a6 grad_norm |
|---|---|---|---|---|---|---|
| 41-60 | 0.2357 | 0.0019 | 0.1772 | 0.0024 | 0.6692 | 29.79 |
| 81-100 | 0.2696 | 0.0007 | 0.1732 | 0.0007 | 0.9647 | 41.65 |
| 121-150 | 0.2644 | **0.0006** | 0.1586 | **0.0005** | 0.6987 | **57.12** |

a6's effective sample size is **0.0006**, roughly **440x worse** than a5b's, meaning
a 128-sequence batch does the statistical work of about **0.08 sequences**. 99.97
percent of tokens sit in the low tail. `rollout_is_batch_normalize` divides by the
mean weight, so at a mean of 0.0005 it amplifies the surviving mass by about
**1600x to 2000x**, and that mass rests on a handful of tokens. Gradient norm
reaches **57.1 against the incumbent's 1.73**, a factor of **33**, and it is still
climbing.

The cause is the gap itself. The IS weight is `exp(log pi_trainer - log pi_rollout)`.
On PRF exact-k the codec-view gap is **14.1 nats**, so that exponential is
essentially zero for nearly every token; on FRLR it is **4.4 nats** and the mean
weight lands at a workable 0.17. **The amplification factor is 1/mean_weight and
mean_weight shrinks as the gap grows, so batch normalisation turns a large gap into
a gradient explosion.**

## This reverses what I wrote in the a6 pre-read

The pre-read concluded, from a6 matching the incumbent's gap: *"token-IS buys
nothing on the gap ... the weighting can be dropped from future FRLR cells unless
it earns its place on some other axis."*

The first half stands: the weighting does not improve the gap. **The inference does
not.** The correct statement is the converse. FRLR's low gap is precisely **what
makes the weighting usable at all**. They are complementary rather than independent:
reduce the gap and the IS estimator becomes well conditioned; leave the gap at 14
nats and the same estimator degenerates to an 0.0006 ESS and a 33x gradient
explosion. Dropping the weighting from FRLR cells might be right for other reasons,
but not because it is idle.

## G3 passing here is exactly the trap the finding doc describes

a6's codec-view drift slope is **+0.001404**, better than the incumbent's +0.002176,
so G3 passes and a6 reads as the safest arm in the matrix. On the codec-free channel
a6's drift at step 150 is **0.010755** against a5b's **0.008710**, so it is in fact
drifting **more**. This is the ranking reversal documented in
`FINDING_drift_metric_invalid.md`, and a6's G3 pass should be read as evidence
against the gate rather than in favour of the cell.

## Health: impaired, not degenerate

Entropy is flat at 7.8086, `response/aborted_ratio` is 0.0000, score max and min are
1.0 and 0.0, and score is still slowly rising. Response length fell from 674.7 to
592.7. So a6 is not collapsing; it is learning at three quarters speed with wildly
mis-scaled gradients. `grad_norm` max excluding the step 1-3 transient is **248.79**,
which is not a healthy trajectory even though nothing has broken.

## Verdict and action

**Cell verdict: FAIL on G1.** Token-IS with batch normalisation must not be applied
on a high-gap codec. Combined with a5b, the pair gives a clean and useful pair of
statements:

- **the 3.2x gap reduction is FRLR's alone** (a6 reproduces the incumbent's gap to
  0.8 percent while carrying the same weighting), established structurally at step 1
- **the weighting is only viable because of that gap reduction**, established here by
  its failure at 14 nats

**a6 runs to 200 as registered, and it should.** It has `TEST_FREQ=200`, so it will
produce a terminal val, which answers a question worth having: does 200 steps at 33x
gradient norm damage capability? a5b answered the analogous question at its own
operating point. Stopping early would also idle the GPU, since nothing is chained
after a6 and rounds B and C need operator approval.

## Rider

`batch_normalize` should be treated as **gap-conditional** in any future cell: safe
where the codec-view gap is small, actively harmful where it is large. A cheap guard
is to refuse the knob, or fall back to unnormalised truncated IS, whenever the
measured mean IS weight falls below some floor. On this evidence a floor near 0.05,
which a5b clears at 0.17 and a6 misses by 100x, would have caught it before the run.

---

## TERMINAL ADDENDUM at step 200: the codec-view drift gate does not merely fail to predict capability, it ANTI-predicts it

a6 finished 200/200 at 02:05Z with 2 error markers, both shutdown-path (atexit,
teardown, workers). Both checkpoints saved. Its terminal val is the number that
closes the gating argument.

| cell | codec-view drift @200 | codec-FREE drift @200 | terminal val | val vs incumbent |
|---|---|---|---|---|
| incumbent | 0.9085 at step 600 | no probe exists | 0.6613 @150 | 1.000x |
| **a5b** FRLR+TIS | **2.2262** | **0.016754** | **0.6593** | **0.997x** |
| **a6** PRF+TIS | **0.2918** | **0.026793** | **0.5391** | **0.815x** |

Read the first and third columns together. **a5b carries 7.63x a6's codec-view
drift and has 1.22x its capability.** The registered gate, computed on
`actor/kl_loss`, **passed a6** (slope +0.001404, better than the incumbent's
+0.002176) and **failed a5b** at 2.48x over threshold. The capability outcome is
the exact reverse of both calls.

So the earlier claim, that the gate fails to predict capability, was too gentle. On
this pair it is **anti-correlated with capability**. A veto that reliably points the
wrong way is worse than no veto.

**The codec-free channel gets the ordering right.** a6's true drift is **1.60x**
a5b's (0.026793 against 0.016754) and its val is **0.82x**. That is the correct
sign, on the only two cells in the program that both have a dense probe.

## What a6's val does and does not show

**It does not show damage below baseline.** a6's val of 0.5391 tracks its own
training score of 0.4932 at step 200, and both arms started near 0.357. a6 improved
on the base model; it simply improved less. Calling this "capability damage" would
overstate it.

**What it does show is movement without benefit.** Taking the common starting score
of about 0.357 as a proxy for base val, which is a proxy because these cells ran
with `VAL_BEFORE_TRAIN=False`:

| cell | true drift | capability gained | drift per unit gain |
|---|---|---|---|
| a5b | 0.016754 | 0.302 | **0.0554** |
| a6 | 0.026793 | 0.182 | **0.1471** |

a6 moved **2.66x further from the base model per unit of capability acquired**. That
is the real cost of the gradient explosion: ESS 0.00067 and grad_norm 64.05 at step
200 do not produce a broken model, they produce an inefficient one that spends
weight movement without buying accuracy.

## Consequence for the registered procedure

This is the third and strongest piece of evidence, and the three are independent:

1. a5b failed the drift gate by 2.48x and validated at **parity**
2. the two channels **rank a6 and a5b in opposite orders**, so codec-view drift is
   not a valid cross-codec comparison at all
3. across these two cells the gate is **anti-correlated with capability**, passing
   the arm that lost 18.5 percent and failing the arm that lost nothing

`actor/kl_loss` should be demoted to a labelled codec-view diagnostic,
`probe/kl_dense` promoted to the drift criterion, a cadence-5 probe required on
every cell, and promotion gated on val and OOD. That remains the operator's call
because it changes the registered procedure, and it is now flagged with three
independent supports rather than one.

**Neither a5b's nor a6's recorded verdict changes.** Both were scored against bars
registered before their data existed, and that is exactly why this evidence is
worth anything.
