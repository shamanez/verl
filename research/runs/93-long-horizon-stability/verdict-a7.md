# Verdict: a7-frlr-r48k28-notis-200. The best cell in the program, and the first with a REAL gap-slope failure.

## STABILITY VERDICT (2026-07-28 re-scoring)

> **Re-scored against stability, not reward.** The body below this section was
> written against a bar that leads with capability. Capability turned out to be
> a tie across the field, so it cannot carry a conclusion. What follows
> supersedes the original ranking claims; the original text is kept in place.

**Stability rank: 6 of 12.** a7 is FRLR r48/k28 with the fast `Q` refreshed every
step (`frlr_q_cadence=1`) and no token-IS, and it produced the worst gap trend and
the worst gradient trend of any arm in the program that was not run to horizon.

| axis | this arm | reference | read |
|---|---|---|---|
| gap slope | +0.016351 over 100-120; +0.028329 over 100-199 (n=100) | incumbent +0.000838 over 100-120, +0.000848 over 100-599 (n=500) | last of the eleven arms that reached the matched 100-120 window, and the steepest 100-199 slope of any arm |
| gap drift ratio | 1.551 | incumbent 1.029 | the gap ends half again above its own step-100 level; only c600 (5.122) is worse |
| grad_norm drift | 1.86x over 200 steps, p50 1.9744 to 3.6814 | incumbent 0.85x over 600 steps, p50 1.7866 to 1.5259 | the optimizer walks up rather than settling; the incumbent's twelve 50-step block medians never leave 1.50-1.82 |
| grad_norm max | 68.01, max/p50 30.1x | incumbent 4.645, max/p50 2.9x | large excursions sit on top of the upward walk |
| collapse / kill | none, ran 200/200 to its scheduled end | incumbent none in 600 | a7 did not fail, its trends did |
| capability | val 0.6713426853707415 @200, joint-best of the twelve program arms and identical to a9 to the digit (= 335/499); dense and the incumbent, both references, read higher | incumbent 0.6613 @150, 0.6633 @300, 0.6733 @450, 0.6613 @600; dense 0.6874 @450, 0.6774 @600 | does not separate the arms |

What a7 proves is that the every-step `Q` refit buys a low gap level and pays for it
with the worst trend in the field. Its level is genuinely low, 4.945 at step 100
against the incumbent's 14.243, and the original body was right about that. But the
level is not held: over 100-199 the gap climbs at +0.028329 nats per step and the
drift ratio reaches 1.551, against the incumbent's +0.000848 over 100-599 at a ratio
of 1.029. The gradient side tells the same story on the same run: median 1.9744 in
the first 20 percent rising to 3.6814 in the last 20 percent, drift 1.86x, run max
68.01. Both axes point the same way, which is why this arm ranks below a8 (gap ratio
0.986) and a9 (1.192) despite the three sharing one codec.

The cause is `Q` governance, not physical drift, and the fact sheet pins that down
twice. At the matched step 200 a7's codec-free `probe/kl_dense` is 0.008201 against
c600's 0.008186, identical to 0.2 percent, so the arms that differ only in how `Q` is
governed have the same real drift. On the identical 100-199 window a8, which is a7's
codec with cadence raised from 1 to 20, has a gap slope of +0.001485 against a7's
+0.028329. The every-step refit is therefore the thing being measured, and slowing it
costs nothing physical. That vindicates the mechanism hypothesis the original body
offered in the body's section "The one genuine concern: the gap has stopped
settling", and it removes a7's own configuration from consideration: if
cadence 20 is available at the same real drift, cadence 1 has no argument left.

What a7 does NOT prove is where its trend ends. It stopped at 200 steps, so no
crossing against the incumbent was ever observed for this arm. The relevant horizon
evidence belongs to c600, the only FRLR arm run to 600: gap slope +0.045972 over
100-599, drift ratio 5.122, gradient drift 9.25x with max 176.367, first exceeding
the incumbent's gap at step 417 and staying above from 424. a7 is the FRLR governance
variant whose 200-step signature most resembles that arm's early signature, higher
ratio than both a8 and a9, so the natural expectation is that it would have crossed
sooner. That is an extrapolation from an unrun horizon and is stated as one, not as a
result. Axis 4 also cannot be used to argue PRF against FRLR at all here, because the
incumbent has no probe: it has 0 probe points.

Three claims in the body below are wrong under the stability bar and are corrected
here rather than edited away. First, the title's "best cell in the program" and the
addendum's "best capability in the program" do not hold: a7's terminal val
0.6713426853707415 is identical to a9's to sixteen digits and both sit below dense's
0.6874 @450 and 0.6774 @600, which is the cleanest demonstration in the program that
499 held-out problems cannot resolve these arms. Second, the addendum's headline
number, 5.8246 nats of `actor/kl_loss`, cannot rank or defend this arm: `actor/kl_loss`
is real drift multiplied by a codec-view inflation factor that itself moves 50x
between arms and across a run, and it spans 55x across a7, a8 and a9, which differ
only in `Q` governance and have identical physical drift. The same disqualification
applies to the codec-view entropy readings quoted alongside it: the dense control
sharpens the same way, and sampler-side `rollout_log_ppl` at step 599 is near
identical across c600, incumbent and dense at 0.091, 0.093 and 0.108. That also voids
the n=3 Spearman table, whose one axis is a disqualified metric and whose other axis
is a tie. Third, "grad_norm also rose to 4.64 at step 200 ... neither is alarming"
understates the run: the 200-step maximum is 68.01 at 30.1x the median, and the
directional worry the sentence raises is exactly the one that ranks this arm sixth.

Scored 2026-07-26T05:10Z at the registered primary window **100-120**, complete at
21 rows. Cell: FRLR r48/k28, **no token-IS, no batch normalisation**, probe cadence
5. Run continues to 200 with a terminal val.

## The registered bar

| gate | measured | bar | call |
|---|---|---|---|
| **G1** learning, score level | **0.6559** | >= 0.6248 | **PASS**, and 0.997x the incumbent's 0.6577 |
| **G2** gap level | **5.0849** | < 14.2458 | **PASS**, 0.36x |
| **G2** gap slope | **+0.016351** | <= +5.0e-4 | **FAIL, 33x, and this one is REAL** |
| **G3** drift slope | +0.043919 | <= 3.264e-3 | FAIL 13.5x, but 641x inflated, see below |
| **G4** wire | 1233.4 bits | 1232 | parity to 0.1 percent (corrected: includes the Q broadcast) |

## G1 is the headline: learning parity with the dense-grade incumbent

| | score 100-120 | grad_norm | mechanism |
|---|---|---|---|
| incumbent PRF | 0.6577 | 1.808 | none |
| a5b FRLR + IS + bnorm | 0.6277 | 0.856 | IS shrinkage |
| a6 PRF + IS + bnorm | 0.4854 | ~55 | bnorm amplification |
| **a7 FRLR, neither** | **0.6559** | **2.243** | **none** |

a7 reaches **99.7 percent of the incumbent's learning** at a **2.8x smaller gap**,
with gradient norm 1.24x the incumbent's, which is unremarkable. No arm in this
program has previously combined those. The two IS variants sit either side of it,
one suppressed and one exploded, which is the cleanest possible demonstration that
the weighting was never the useful part.

## Correction: my P3 note was right in reasoning and WRONG in its prediction

At step 51 I recorded that registered prediction P3 (a7's codec-free drift at 200
at or below a5b's 0.016754) was "off track and likely to fail", and built a careful
argument that its failure should not count against a7. The argument about absolute
versus harmful drift stands. **The empirical claim does not.**

| probe step | a7 codec-free | a5b codec-free | ratio |
|---|---|---|---|
| 25 | 0.000293 | 0.000252 | 1.16x |
| 50 | 0.001576 | 0.000752 | **2.10x** |
| 75 | 0.002804 | 0.002383 | 1.18x |
| 100 | 0.003992 | 0.003857 | 1.04x |
| 125 | 0.004867 | 0.006258 | 0.78x |
| **150** | **0.005214** | **0.008710** | **0.60x** |

a7's real drift started higher, crossed **below** a5b's around step 110, and is now
**40 percent lower**. On this trajectory **P3 will pass**, and the caveat I
constructed is unnecessary.

I extrapolated from a single step-50 ratio that then reversed. That is the third
time in this session I over-read an early window, and I did it inside the very note
whose purpose was to be careful about early windows. The lesson is not "be careful
with early windows", which I already knew and wrote down; it is that **stating a
directional prediction from one ratio is the same error regardless of how much
hedging surrounds it.**

## G3 fails on a 641x-inflated number and should not be read as drift

a7's codec-view drift level at 100-120 is **1.8417**, which is 6.9x a5b's at the
same window. Its `probe/kl_gain` has reached **641.1x**, so its real drift is
**0.005386**, which as the table above shows is now the **lowest of the two FRLR
arms**.

This is `FINDING_drift_metric_invalid.md` at its most extreme. The inflation factor
in this program has now been observed at 10.1x, 14.3x, 132.9x, 352.9x and 641.1x
depending on codec, cadence, run and step. G3 is recorded as failed for
bar-compliance and carries **no physical content**.

## The one genuine concern: the gap has stopped settling

This is the first failure in the program that survives every deflationary check.

| window | gap level | gap slope |
|---|---|---|
| 21-40 | 4.5207 | -0.00037 |
| 41-60 | **4.4602** | -0.00267 |
| 61-80 | 4.5858 | **+0.01637** |
| 81-100 | 4.8033 | +0.01468 |
| 100-120 | 5.0849 | +0.01635 |
| 121-149 | **5.6737** | **+0.02624** |

The gap **turns around at about step 60** and rises monotonically thereafter, from
4.4602 to 5.6737, a **1.21 nat rise** with an **accelerating** slope. That is 4x
the within-window oscillation amplitude of about 0.3 nats which invalidated a5b's
slope clause, so the ill-posedness argument does **not** rescue it here. It is a
real trend.

It matters because a settling gap is the program's own registered success criterion.
a7 is still at 5.67 against the incumbent's 14.25, so it is far ahead on level, but
the direction is wrong and getting worse.

The likely mechanism, stated as a hypothesis: Q is refreshed every step
(`frlr_q_cadence=1`) and chases a policy that a7 moves faster than any previous
arm, so the basis lags further behind as learning accelerates. That predicts
`frlr_q_cadence` would trade gap growth against reconstruction freshness, and it is
the same knob that the wire-budget correction identified as reducing the codec-view
inflation. **Testable with an existing knob, no new code.**

## Health

Clean, with one first. `grad_norm` 2.243 at the window (max 12.25 excluding the
step 1-3 transient) against the incumbent's 1.808. Sampler-side
`rollout_log_ppl` is **0.1779**, normal, which is the cross-check that matters
because the codec-view entropy reads **1.9336** and would look like collapse to
anyone reading it directly. `probe/lr_brake_triggered` fired **once, at probe step
140**, the first time in the program; it is detection-only and never mutates the LR,
and by its own docstring measuring how often it *would* fire is the point.

## Verdict

**Two gates fail and one of them matters.** a7 is the best configuration this
program has produced: learning parity with the incumbent, a 2.8x smaller gap, real
drift below the other FRLR arm and falling relative to it, normal gradients, no
importance-sampling machinery, at the incumbent's wire budget to within 0.1 percent.

**But its gap no longer settles**, and unlike every other slope failure tonight that
one is not an artifact of window length, oscillation or codec view. It is the
program's registered criterion and a7 misses it.

The terminal val at step 200 is pending and is what decides whether the rising gap
has cost anything. a5b reached parity at 0.0168 nats of real drift; a7 is at 0.0054
and learning far better, so the prior expectation is that a7 validates at or above
the incumbent. If it does, a7 becomes the round-B/C candidate with the Q-cadence
question attached rather than resolved.

---

## TERMINAL ADDENDUM at step 200: a7 has the best capability in the program, at 5.82 nats of "drift"

a7 finished 200/200 with 2 shutdown-path error markers (atexit, teardown, workers)
and both checkpoints saved. Its terminal validation is the highest number this
program has measured.

| cell | `actor/kl_loss` @200 | codec-FREE drift @200 | terminal val |
|---|---|---|---|
| a6 PRF + IS + bnorm | **0.2918** | 0.026793 | **0.5391** |
| a5b FRLR + IS + bnorm | 2.2262 | 0.016754 | 0.6593 |
| **a7 FRLR, no IS** | **5.8246** | **0.008200** | **0.6713** |
| incumbent PRF | 0.9085 @600 | no probe exists | 0.6613 @150, 0.6633 @300, 0.6733 @450, 0.6613 @600 |

### The metric orderings across three cells

| relationship | Spearman | meaning |
|---|---|---|
| `actor/kl_loss` vs val | **+1.00** | higher "drift" reading goes with **BETTER** capability |
| `probe/kl_dense` vs val | **-1.00** | higher real drift goes with **worse** capability, the correct direction |

The codec-view metric is not merely uninformative, it is **ordered the wrong way**
across all three cells. The codec-free metric is ordered correctly across all three.
With n=3 a perfect ordering arises by chance with probability 1/6, so this is
**consistent evidence rather than proof**, but the effect sizes are large and it
matches the mechanism established in `FINDING_drift_metric_invalid.md`.

### The single most striking number in the program

**a7 carries 5.8246 nats of `actor/kl_loss`, which sits inside the historical 3-to-8
nat "collapse band" this program has used as its danger zone, while holding the best
capability ever measured here.** Its real drift is 0.0082 nats and its inflation
factor is **710.2x**. Any gate defined on that band would have killed the winning
arm.

### How good is the val, stated carefully

a7's 0.6713 against the incumbent's **0.6623** interpolated at step 200 is
**+0.0090**. The incumbent's own checkpoint-to-checkpoint spread is **0.0120**, so
a7's advantage is smaller than the reference's own variability and should **not** be
called a clear win over it.

The fair framing is the step count: **a7 reaches at 200 steps what the incumbent
needed about 450 steps to reach** (its 0.6733 peak), and it exceeds three of the
incumbent's four checkpoints. On capability a7 is at least equal to the incumbent
and got there faster.

### What survives against a7

Only the gap trend, and it continued: **4.4602 at 41-60, 5.0849 at 100-120, 5.6737
at 121-149, 8.1849 at step 200.** The rise is accelerating, roughly +0.034/step over
the last 90 steps against +0.024 earlier. At step 200 it remains **1.75x better**
than the incumbent's ~14.3, but the direction is wrong and worsening, and a settling
gap is the program's registered criterion.

`grad_norm` also rose to 4.64 at step 200 from 2.24 at the window, and codec-view
entropy fell to 1.059. Neither is alarming on its own and the sampler-side
`rollout_log_ppl` stayed normal, but both track the same underlying story: the codec
view is drifting away from the sampler view as Q chases the policy.

### Consequence, and the cell now running

The gap trend is the one real defect, and its mechanism points at a single existing
knob. **Cell a8 launched at 08:00Z: a7's exact codec with `frlr_q_cadence` raised
from 1 to 20**, mirroring the PowerSGD governance in which Q moved only at anchor
fires. GPU idle across the handoff was about 5 minutes.

If freezing Q flattens the gap while preserving a7's capability, the combination is
the round-C candidate. If it flattens the gap but costs capability, the every-step
refresh was load-bearing and a7 stands as the best available with a known long-run
limitation.
