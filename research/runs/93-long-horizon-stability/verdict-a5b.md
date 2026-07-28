# Verdict: a5b-frlr-bnorm-200. FAILS the registered bar. Decisive clause is G3.

## STABILITY VERDICT (2026-07-28 re-scoring)

> **Re-scored against stability, not reward.** The body below this section was
> written against a bar that leads with capability. Capability turned out to be
> a tie across the field, so it cannot carry a conclusion. What follows
> supersedes the original ranking claims; the original text is kept in place.

**Stability rank: 7 of 12.** a5b is FRLR r48/k28 plus token-IS 2.0 plus
`rollout_is_batch_normalize=true`, and it bought a calm-looking gradient trace by
damping the update rather than by stabilising it, while producing the largest
gradient excursion ever recorded in this program.

| axis | this arm | reference | read |
|---|---|---|---|
| gap slope | +0.000358 over 100-120 (n=21); +0.006673 over 100-199 (n=100), its longest window | incumbent +0.000838 over 100-120, +0.000848 over 100-599 (n=500) | the 100-120 flatness that ranked a5b 2nd does not survive to 200 steps; the incumbent's flatness survives to 600 |
| gap drift ratio | 1.150 | incumbent 1.029 | a5b ends 15 percent above its step-100 level, 9th of the 13 rows in the fact sheet |
| grad_norm drift | 0.86x over 200 steps (p50 0.7236 to 0.6255) | incumbent 0.85x over 600 steps (p50 1.7866 to 1.5259) | the same shape on paper, but read next to reward: a5b's update was suppressed, so the flat ratio is not earned |
| grad_norm max | **204.39**, max/p50 **284.4x** | incumbent 4.645, max/p50 2.9x | worst spike ratio in the program, above a7 30.1x, a5 20.9x, a8 18.6x, a6 18.3x and c600 12.1x |
| collapse / kill | none, ran to its scheduled 200 steps | incumbent none in 600 | clean exit, but only a third of the incumbent's horizon |
| capability | val 0.6593 @200; reward 0.6606 over 101-200 | incumbent val 0.6613 @150, 0.6633 @300, 0.6733 @450, 0.6613 @600; reward 0.6726 over 101-200 | does not separate the arms |

What a5b proves is that batch normalisation partially undoes token-IS
suppression. Against a5 on matched blocks, reward over 101-200 goes 0.5895 to
0.6606 and the terminal val lands at 0.6593, inside the non-collapsed field of
0.6593 to 0.6713. That is a real fix and it is why a5b sits above a5 in the
ranking. What it does NOT prove is that the training got calmer. The 0.86x
gradient drift ratio is measured on an optimizer that is still moving less than
the incumbent's: reward 0.4234 against the incumbent's 0.5015 over 1-100, and
0.6606 against 0.6726 over 101-200. The fact sheet's second counterexample
applies here in weakened form, and axis 2 must be read next to reward.

The number that decides the rank is the run maximum. a5b's p50 sits near 0.63
and its run max is 204.39, a ratio of 284.4x, the worst in the program by an
order of magnitude over the next arm. The incumbent's twelve 50-step blocks
never produce a block max above 4.645 in 600 steps. On an internet-split
pipeline a single excursion of that size is precisely the event a stability
program exists to exclude, and no evidence in the fact sheet shows it was
contained rather than merely unrepeated inside 200 steps. a5b ranks below a7
despite a7's much worse gap trend (+0.028329 over 100-199, ratio 1.551) because
a7's worst excursion is 68.01 at 30.1x against a5b's 204.39 at 284.4x.
Capability does not enter: a7's 0.6713 and a5b's 0.6593 are inside the tie.

On the gap axis the original body leads with LEVEL, 4.45 against 14.25, and
level is not stationarity. On stationarity a5b is 1.150 against the incumbent's
1.029, and its own longest-window slope is +0.006673 over 100-199 against the
incumbent's +0.000848 over 100-599. That is the FRLR family signature: a large
early level advantage that is not stationary. c600 is the same story with 400
more steps of evidence, crossing the incumbent at 417 and staying above from
424, ending 2.12x worse with a 9.25x gradient drift. a5b has no horizon evidence
at all. The codec-free channel points the same way: `probe/kl_dense` at the
matched step 200 reads 0.016754 for a5b against 0.008186 for c600, 0.008201 for
a7 and 0.010872 for a8, second only to a6's 0.026793, and a6 is the one arm that
collapsed. Note the limit: the incumbent has no probe, so axis 4 cannot rank a5b
against PRF at all.

Three things in the body below are corrected here rather than edited away.
First, the whole G3 apparatus, the FAIL, the acceleration tables and the
terminal "indictment of the gate", is built on `actor/kl_loss`, which is
disqualified for ranking: it is real drift multiplied by a codec-view inflation
factor that itself moves by 50x between arms and across a run, so it confounds
the thing being measured with the instrument. The body's own probe data, which
survives, was the right instrument. Second, the terminal addendum recommends
making val and OOD the promotion gate; under the reframe that is wrong, because
every non-collapsed arm lands in 0.6593 to 0.6713 and the incumbent ends 600
steps at 0.6613, so capability promotes nothing. Third, the rider's health line
"`grad_norm` max 2.0774 excluding the step 1-3 transient" was scored around step
120 and is superseded by the 200-step run max of 204.39.

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

---

## ADDENDUM at step 152: the acceleration does NOT saturate, and the drift is real in the codec-free channel

The verdict above left one rider open: whether a5b's drift acceleration saturates
the way the incumbent's did after step 80. It does not. Matched windows:

| window | incumbent slope | a5b slope | a5b/inc drift LEVEL |
|---|---|---|---|
| 61-80 | **+0.002848** (its peak) | +0.003234 | 0.79x |
| 81-100 | +0.002065 | +0.004583 | 1.12x |
| 101-120 | +0.002158 | +0.008137 | 1.51x |
| 121-140 | +0.001300 | +0.014008 | **2.34x** |
| 141-152 | **+0.001348** | **+0.017768** | **3.11x** |

The incumbent peaks at 61-80 and then **decays to less than half its peak**, which
is the saturation the round-A correction memo described and the reason it reaches
only 0.91 nats by step 600. a5b's slope rises monotonically through every window,
**5.5x from 61-80 to 141-152**, with no inflection.

In level terms a5b is at **0.727 nats by step 152** against the incumbent's 0.234.
**a5b will exceed the incumbent's entire 600-step drift budget within roughly 10 to
20 more steps.** Naive continuation at the current slope puts it near 1.7 nats by
step 200, and the slope is still climbing, so that is a floor rather than an
estimate.

### The drift is not a measurement artifact, and that question is now closed

This is what the dense probe was added for. Codec-free drift, `probe/kl_dense`:

| step | 25 | 50 | 75 | 100 | 125 | 150 |
|---|---|---|---|---|---|---|
| dense drift | 0.000252 | 0.000752 | 0.002383 | 0.003857 | 0.006258 | 0.008710 |

Monotone, and **itself accelerating**: the codec-free slope runs +6.8e-5/step over
25-150 and +9.7e-5/step over 100-150. So the run really is moving away from the
base model, and the G3 failure is not an artifact of a time-varying view offset.

The offset **is** time-varying, dramatically so. `probe/kl_gain` runs 13.8x, 33.2x,
34.7x, 49.3x, 66.1x, **92.6x** across those same probes, a **6.7x growth in the
codec's own inflation factor** over 150 steps. Both things are true at once: the
codec-view drift metric is badly and increasingly inflated, **and** the underlying
drift is real and accelerating. Had only the first been true, the veto would have
been contaminated and a5b would deserve acquittal. It is not.

Meanwhile the codec-free **gap** stays flat and tiny across all six probes
(0.000362, 0.000439, 0.000355, 0.000297, 0.000167, 0.000260) with no trend,
confirming it as a constant engine-mismatch floor rather than a policy quantity.

### The learning objection is now withdrawn, which makes the trade cleaner

a5b's score converges on the incumbent: **0.760, 0.919, 0.955, 0.978, 0.980** of
the incumbent across 61-80 to 141-152. By 141-152 a5b is at **98 percent** of the
incumbent's score. The G1 coin flip at 100-120 was the tail of the onset delay,
not a learning ceiling.

So the trade is no longer "marginal learning plus a drift failure". It is:

> **equal learning, at 3.1x the drift level and 13x the drift slope, on a
> trajectory with no saturation while the incumbent's has already saturated.**

That is a sharper and stronger reason to keep plain PRF, and it does not depend on
the marginal G1 reading at all.

### What is NOT claimed

**Capability is not damaged yet.** Score is still rising at 141-152 (0.6514, up
from 0.6270 at 101-120), so 0.727 nats of reference KL has not cost accuracy. That
is consistent with this program's earlier OOD result, where a compressed arm matched
dense on all 10 benchmarks despite roughly 1000x reference KL, with damage appearing
only at collapse. The claim here is about a **trajectory with no saturation**, not
about realised damage. Confirming damage would need validation and OOD, which this
cell does not run.

### Decision unchanged, and the run is not being truncated

Verdict stays **FAIL**, now on stronger evidence. a5b runs to 200 as registered:
the remaining 48 steps are the highest-value drift-trajectory data in the program
and the step-200 checkpoint enables post-hoc geometry. Truncating would not save
GPU time either, since the chain would simply start a6 earlier, and a6 gets its
full 200 steps regardless.

---

## TERMINAL ADDENDUM at step 200: a5b validates at PARITY, and the decisive gate does not track capability

a5b ran `TEST_FREQ=200`, so it produced a terminal validation. **This is the first
val number for any cell in issue #93** (round A ran with validation off), and it
speaks directly to the one thing the verdict above said it could not settle.

| run | step | val, MATH acc mean@1 | drift `actor/kl_loss` |
|---|---|---|---|
| incumbent | 150 | 0.6613 | 0.2345 |
| incumbent | 300 | 0.6633 | 0.4538 |
| incumbent | 450 | 0.6733 | 0.6926 |
| incumbent | 600 | 0.6613 | 0.9085 |
| **a5b** | **200** | **0.6593** | **2.2262** |

a5b's terminal drift is **2.2262 nats**: 9.5x the incumbent's at a comparable step
and **2.45x the incumbent's entire 600-step endpoint**. Its validation accuracy is
**0.6593**, which is 0.0020 below the incumbent's step-150 value and 0.0040 below
its step-300 value.

**That difference is smaller than the incumbent's own val-to-val variation.** The
incumbent wobbles 0.6613, 0.6633, 0.6733, 0.6613 across its four checkpoints, a
range of 0.0120, and it **ends exactly where it started**. a5b sits inside that
band. No knowledge of the val set size is needed for this comparison: the
incumbent's own series supplies the empirical noise floor and a5b is well within it.

Training score agrees: 0.6660 at step 200 against the incumbent's 0.6568 at
101-120.

### What this does to the verdict

**The FAIL stands as a bar-compliance fact.** G3 was pre-registered, a5b missed it
by 2.48x, and pre-registered bars do not move after the data arrives. That is the
whole point of registering them.

**But the verdict's reasoning above is now contradicted by direct evidence.** It
said the trade was "a bad trade under this program's cardinal rule, which is not to
damage the base model." At 2.23 nats of reference KL there is **no measurable
damage** to in-domain capability. The sentence was an inference from the drift
metric, and the drift metric has just been shown not to carry it.

So the honest position is: **G3 measured what it was defined to measure and failed
to proxy what it was chosen for.** This is not a defence of a5b so much as an
indictment of the gate.

### Stated fairly, the case FOR a5b is now non-trivial

Having argued against this cell, I should put its case at full strength:

- **equal in-domain capability** at step 200, inside the incumbent's own val noise
- **3.2x better train-inference gap** (4.45 against 14.25 nats)
- **identical wire budget**, 1232 bits, no cost paid for the gap improvement
- equal training score

### And the case against it, which is now narrower but not empty

- **The drift is still accelerating and did not saturate.** 0.727 nats at step 152
  to 2.226 at step 200 is +0.031/step averaged, above the +0.0178 measured at
  141-152, so it is still climbing. The incumbent's slope had already decayed to
  +0.0013 by then. Naive continuation puts a5b far into the historical 3-8 nat
  collapse band well before step 600.
- **A val at 200 says nothing about 600.** This program has previously seen a
  comm-eff arm crash at around 200 steps from drift, so the hazard is real and
  merely unrealised here.
- One benchmark, one checkpoint, `mean@1`, in-domain only. No OOD.

### The consequence for the program, which is an operator-level decision

**Rounds B and C should not gate primarily on reference-KL drift.** The evidence is
now two-fold and consistent: this program's earlier OOD work found a compressed arm
matching dense on all 10 benchmarks at roughly 1000x reference KL with damage only
at collapse, and a5b now reproduces that in-domain with an actual val at 2.23 nats.
A drift gate that fires 2.48x over threshold on a cell with intact capability is
generating false positives, and round A killed arms on it.

The criterion that survives this is **capability measured directly**, which is
exactly what round C runs (val at 0/300/600 plus the OOD suite). The recommendation
is to demote drift slope from a veto to a reported diagnostic, and to make val and
OOD the promotion gate. **That changes the registered decision procedure and is
therefore the operator's call, not mine.**

Flagged for the operator rather than acted on. Nothing in this addendum changes
a5b's recorded FAIL or a6's already-registered bar.
