# Verdict: a9-frlr-anchorq-200. Q governance does not touch real drift at all, and on the full trajectory a9 may simply be the best arm.

> **Read the terminal addendum before acting on this section.** This body was
> written at the registered window 100-120, where a8's gap slope beats a9's 7.3x,
> and it concludes that anchor ownership is a trade-off carrying a measurable cost.
> The step-200 trajectory shows **a8 was still falling until step 143**, so its
> registered-window slope is a turning point rather than a settled trend. In the
> late window a9 is the flattest of the three and it holds the lowest terminal gap
> and the joint-best terminal val. The registered criterion still belongs to a8 and
> is not rescored; what changes is that **the cost of the anchor-Q constraint is not
> established and may be zero.** Corrected in the addendum rather than edited away.

Scored 2026-07-26T19:00Z at the registered primary window **100-120**, complete at
21 rows. Cell: a7's codec with `Q` harvested from the anchor's clean stale-weight
forward and refreshed **only when the anchor fires**. Run continues to 200 for its
terminal val.

## The registered bar

| gate | measured | bar | call |
|---|---|---|---|
| **G1** learning, score level | **0.6612** | >= 0.6248 | **PASS**, the highest window score in the program |
| **G2** gap level | **5.9232** | < 14.2458 | **PASS**, 0.42x |
| **G2** gap slope | **+0.009262** | <= +5.0e-4 | **FAIL 18.5x** |
| **G3** drift slope | +0.024098 | <= 3.264e-3 | FAIL, codec-view, no physical content |
| **G4** wire | 1232 bits | 1232 | **PASS, and now literally rather than approximately** |

The mechanism worked exactly as designed: five anchor fires at steps 20/40/60/80/100,
`refreshes` 7/14/21/28/35, precisely 7 per fire, zero errors. `refreshes=35` after
100 steps is the proof the fast path never wrote `Q`; at cadence 1 it would read
about 700.

## The prediction scorecard: 4 of 5, and the one that failed is the important one

| # | prediction | outcome |
|---|---|---|
| P1 | gap slope at or below a8's +0.001262 | **FAILED.** +0.009262, 7.3x higher |
| P2 | gap level between a7's 5.0849 and a8's 6.8293 | **CONFIRMED.** 5.9232 |
| P3 | learning within 0.02 of a7's 0.6559 | **CONFIRMED.** 0.6612, +0.0053 |
| P4 | `probe/kl_gain` falls relative to a7's 710.2x rising | **CONFIRMED.** 5852.2x to 285.9x, falling |
| P5 | wire exactly 1232 bits | **CONFIRMED** by construction |

P1's failure is the result. I predicted anchor ownership would be at least as flat as
a8 because it is "the limit case" of a8's mechanism. It is not.

## The pre-committed middle outcome fires: a genuine trade-off

`PREREAD_a9.md` addendum 2 registered three readings before this data existed. The
one that applies:

> **between a8 and a7** -> a genuine trade-off; the 600-step run needs a
> level-vs-slope decision, and a cadence-20 anchor-owned hybrid becomes the obvious
> untested cell

| at 100-120 | score | gap level | gap slope | `actor/kl_loss` | `kl_gain` |
|---|---|---|---|---|---|
| incumbent PRF | 0.6577 | 14.2458 | (no probe) | 0.9085 @600 | 10.9x falling |
| a7 fast Q, cadence 1 | 0.6559 | **5.0849** | +0.016351 | 5.8246 | 710.2x **rising** |
| a8 fast Q, cadence 20 | 0.6602 | 6.8293 | **+0.001262** | 0.1064 | 157.4x falling |
| **a9 anchor-owned Q** | **0.6612** | 5.9232 | +0.009262 | 0.7003 | 285.9x falling |

Neither FRLR arm dominates. Ordering, best first:

- **gap level:** a7 (5.08) then **a9 (5.92)** then a8 (6.83)
- **gap slope:** a8 (+0.0013) then **a9 (+0.0093)** then a7 (+0.0164)
- **learning:** a9 (0.6612) then a8 (0.6602) then incumbent (0.6577) then a7 (0.6559),
  a spread of 0.0053 that is well inside the incumbent's own 0.0120 checkpoint spread,
  so this ordering carries no weight

**a9 sits second on both gap measures and its learning advantage is noise.** So the
honest summary is that anchor ownership buys a better level than a8 and a flatter
slope than a7, at the cost of being worse than each on the other axis.

## THE FINDING: Q governance does not change real drift at all

This is what the matched-step comparison shows, and it is the reason to be careful
about which step a number comes from. I nearly wrote that a9 has the lowest
codec-free drift in the program, which would have been an artifact of comparing
a9 at step 120 against a8 at 150 and a7 at 200.

| `probe/kl_dense`, codec-FREE | a5b | a7 | a8 | a9 |
|---|---|---|---|---|
| step 25 | 0.000252 | 0.000293 | 0.000257 | 0.000285 |
| step 50 | 0.000752 | 0.001576 | 0.001520 | 0.001498 |
| step 75 | 0.002383 | 0.002804 | 0.002754 | 0.002807 |
| step 100 | 0.003857 | 0.003992 | 0.004282 | 0.004319 |
| **step 120** | - | **0.005095** | **0.005329** | **0.005265** |

**a7, a8 and a9 are within 4 percent of each other at step 120, and within 8 percent
at every measured step.** Three arms whose only difference is how `Q` is
governed - refreshed every step, every 20 steps, or only by the anchor from
stale weights - and their policies drift from the reference **identically**.

Meanwhile their codec-view `actor/kl_loss` reads **5.8246, 0.1064 and 0.7003**, a
**55x** spread, and their `kl_gain` inflation runs 710x rising, 157x falling and
286x falling.

So `Q` governance moves the gap and it moves the codec view, and it moves **neither
capability nor real drift**. That is the strongest form yet of
`FINDING_drift_metric_invalid.md`: not merely that the gated metric is
mis-calibrated, but that a knob can swing it 55x while the physical quantity it
claims to measure does not move.

a5b sits lower at steps 50 to 100 for the reason already established: token-IS
suppressed its learning, and a model that learns less moves less.

## The question this forces, stated plainly

Within the FRLR family the gap varies from **5.08 to 6.83** and the slope from
**+0.0013 to +0.0164**, while capability and real drift are **flat**. Nothing
measured in 200 steps distinguishes these arms on any quantity that matters to the
deployment.

That does not make the gap unimportant. It means **the gap's importance is
unestablished at this horizon**, which is precisely what the 600-step run exists to
test, and it is the same observation a7's verdict raised about the incumbent running
600 steps at 14.6 nats and finishing where it stood at step 150.

## Consequence: the run-3 selection rule was built on a false premise

> **Partly SUPERSEDED by the terminal addendum.** The premise critique stands: the
> rule's candidate set assumed a9 would dominate a8. The 7.3x cost figure below does
> not, because it is measured at a8's turning point.

`NEXT_RUNS.md` fixed the rule as "flattest gap slope at 100-120 among {a9, a10} at
G1-passing capability". That candidate set presumed a9 would dominate a8. It does
not, so the rule as written would send a **7.3x worse slope** into the 20-hour run
than a8 offers.

The decision now has a component that is not mine to make:

- **a8 is the best arm on the program's registered criterion** (gap slope
  +0.001262), but its `Q` is refreshed on the **fast path**, which the operator's
  architectural constraint excludes.
- **a9 is the best arm that satisfies the constraint**, at a 7.3x worse slope.

So **the operator's constraint has a measurable cost, and it should be reported as a
cost rather than absorbed silently.** If the constraint is hard, a9 is the candidate.
If it is negotiable, a8 is better on the registered criterion. Since capability and
real drift are identical across all three, that choice rests entirely on which
codec-view quantity one believes matters, and on the deployment requirement.

**The obvious untested cell**, and it is one variable from a9: anchor-owned `Q`
harvesting over **multiple** anchor fires before orthonormalising, rather than one
minibatch per refresh. a9's level beats a8's on a **single minibatch** of sketch data
against a8's 20 steps' worth, which says alignment to the slow net is what buys the
level; a8's flatter slope says sample size is what buys the slope. Nothing in the
matrix has both. Not scheduled, and it is new spend.

## Health

`grad_norm` 3.516 at the window, max 24.7 excluding the step 1-3 transient, against
a8's 2.931 and a7's 2.243. Rising across the three arms and worth watching, but no
arm destabilised. Sampler-side `rollout_log_ppl` **0.1861**, normal. `lr_brake`
fired **0 of 24** probes. Codec-view entropy 2.4264 is ~34x inflated and is not a
health signal. Score min 0.0 / max 1.0, so no reward degeneracy.
`probe/gap_dense` averages 0.000304 nats, so the codec accounts for a factor of
**19464** in the measured gap: essentially all of the 5.92 nats is codec view.

## Verdict

**PASS G1, G2-level and G4; FAIL both slope clauses.** The mechanism the operator
asked for works and is verified by counter arithmetic. It delivers the best learning
in the program, a gap level 2.4x better than the incumbent's, exact wire parity, and
a codec-view inflation that falls rather than rises.

**But it is a trade-off rather than the improvement I framed it as, and my P1 was
wrong.** a8's cadence-20 fast `Q` remains 7.3x flatter on the program's registered
criterion. The terminal val at step 200 is pending and, given that all three arms'
capability sits inside the reference's noise, is unlikely to separate them.

---

## TERMINAL ADDENDUM at step 200: the trajectory reverses my a8-vs-a9 conclusion, and I have to be careful about how I say so

a9 finished 200/200 with **zero errors** and **10 anchor fires**, `refreshes=70`,
exactly 7 per fire across the whole run. Terminal val, from the on-box log because
WandB drops the final step:

**`val-core/.../acc/mean@1 = 0.6713426853707415`**

That is **identical to a7's, to the digit**. Both are 335 of 499 problems. A
coincidence at the problem level, but it puts the capability question beyond
argument: the two arms are the same model as far as this benchmark can see.

| cell | terminal val | gap @199 | codec-free drift @200 | `actor/kl_loss` @200 |
|---|---|---|---|---|
| incumbent PRF | 0.6613 / 0.6633 / 0.6733 / 0.6613 | ~14.3 | no probe | 0.9085 @600 |
| a6 PRF + IS + bnorm | 0.5391 | 14.13 | 0.026793 | 0.2918 |
| a5b FRLR + IS + bnorm | 0.6593 | 5.37 | 0.016754 | 2.2262 |
| a7 FRLR, fast Q cad 1 | **0.6713** | 7.7618 | 0.008200 | 5.8246 |
| a8 FRLR, fast Q cad 20 | 0.6613 | 7.2249 | 0.007006 @150 | 0.1064 |
| **a9 FRLR, anchor-owned Q** | **0.6713** | **7.0031** | 0.008594 | 4.1330 |

**The drift finding holds at the terminal point too:** a9's codec-free drift is
0.008594 against a7's 0.008200, within **5 percent**, while their `actor/kl_loss`
reads 4.1330 against 5.8246. Same policy movement, different view.

## The gap trajectory, which is what changes the reading

| step | a7 | a8 | a9 |
|---|---|---|---|
| 60 | 4.4546 | 9.5067 | 5.7190 |
| 100 | 4.9447 | 7.2417 | 5.8213 |
| 120 | 5.1976 | 6.8292 | 5.9989 |
| 150 | 6.0681 | **6.4195** | 6.5184 |
| 180 | 6.7929 | 6.5946 | 6.6525 |
| **199** | 7.7618 | 7.2249 | **7.0031** |

| OLS gap slope | a7 | a8 | a9 |
|---|---|---|---|
| 41-60 | -0.002670 | +0.001988 | -0.004384 |
| 61-80 | +0.016365 | +0.004549 | +0.006267 |
| **100-120 (REGISTERED)** | +0.016351 | **+0.001262** | +0.009262 |
| 121-150 | +0.026249 | **-0.007288** | +0.016096 |
| **150-199 (late)** | +0.038535 | +0.018366 | **+0.012172** |
| 100-199 | +0.028329 | +0.001485 | +0.012203 |

Minimum gap and where it occurs: a7 **4.2435 at step 57**, a8 **6.1173 at step 143**,
a9 **5.4281 at step 62**.

## a8's registered-window flatness is a turning point, not a settled trend

**a8 was still falling until step 143.** Its registered window 100-120 therefore
sits inside the descending arm of a U, and its 121-150 slope is outright **negative**
(-0.007288). Its +0.001262 is a real measurement of that window and it is also the
bottom of a curve that then rises at +0.018366 through 150-199.

In the late window the ordering **inverts**: a9 is the flattest of the three, and at
step 199 a9 also holds the lowest gap.

## Why I am NOT simply switching to the window that favours a9

The registered criterion is **100-120**, a8 wins it, and choosing the late window
because it ranks arms the way I now prefer would be exactly the goalpost move this
program has been disciplined about. So, stated precisely:

- **On the registered criterion, a8 is the best arm.** That stands.
- **What the late data shows is a reliability problem with using that number to
  predict 600-step behaviour**, not a reason to rescore. A slope measured at a
  curve's turning point extrapolates badly by construction, and 100-199 gives a8
  +0.001485 only because its fall cancels its rise.
- **The one thing true of all three: none settles.** Every arm accelerates by
  150-199 (a7 +0.0385, a8 +0.0184, a9 +0.0122). The program's registered success
  criterion is a settling gap and **no FRLR arm achieves it at 200 steps.**

## What this does to the constraint-cost claim I made earlier

Earlier today I wrote that the operator's anchor-Q constraint "has a measurable cost"
of 7.3x on the registered criterion. **That was measured at a8's turning point and
overstates the case.** The fuller picture:

| measure | best arm |
|---|---|
| registered gap slope, 100-120 | a8 |
| late gap slope, 150-199 | **a9** |
| gap level at step 199 | **a9** |
| terminal val | **a9 = a7** |
| codec-free drift | all three identical |

On three of five measures a9 is at least tied best, and on the two where a8 leads,
one is the turning-point artifact above. **So the honest revision is that the
constraint's cost is not established, and may be zero.** I am recording this as a
correction to my own earlier framing rather than editing that framing away.

## An operational finding that matters for the 600-step run

a9 is the **first cell with the in-training R2 sink enabled**, and it changed
teardown from a8's ~2 minutes to a bandwidth-bound wait. The log says the upload is
queued "async", and it is async with respect to *training* but **not with respect to
process exit**: a9 sat at step 200 with the GPU at 0 percent while 19G uploaded, and
it exited within a minute of those uploads being killed. Measured progress was 16 of
74 parts in 7 minutes, so the full wait would have been about **55 minutes of idle
GPU**; killing it cut the handoff gap to **11 minutes**, and a9's checkpoint was
never at risk because it is on local disk with deletion disabled.

**For a 600-step run with three saves that is roughly 2.5 hours of idle GPU.** Either
set the sink genuinely asynchronous, or leave it off and back-fill, which is what a5b
through a8 did at no cost to occupancy.
