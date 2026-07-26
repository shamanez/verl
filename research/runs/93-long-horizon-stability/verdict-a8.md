# Verdict: a8-frlr-qcad20-200. The best cell in the program, and it beats the dense-grade incumbent on learning.

Scored 2026-07-26T13:35Z at the registered primary window **100-120**, complete at
21 rows. Cell: a7's exact codec with `frlr_q_cadence` raised from 1 to 20. Run
continues to 200 for its terminal val; a terminal addendum follows.

## The registered bar

| gate | measured | bar | call |
|---|---|---|---|
| **G1** learning, score level | **0.6602** | >= 0.6248 | **PASS, and 1.004x the incumbent's 0.6577** |
| **G2** gap level | **6.8293** | < 14.2458 | **PASS, 0.48x** |
| **G2** gap slope | **+0.001262** | <= +5.0e-4 | FAIL 2.5x, but **13x flatter than a7** and negative over 61-120 |
| **G3** drift slope | +0.006967 | <= 3.264e-3 | FAIL 2.1x, codec-view, no physical content |
| **G4** wire | 1233.4 bits | 1232 | parity to 0.1 percent |

## G1: the first compressed arm to out-learn the incumbent

| | score 100-120 | gap 100-120 | grad_norm |
|---|---|---|---|
| incumbent PRF exact-k | 0.6577 | 14.2458 | 1.808 |
| a5b FRLR + IS + bnorm | 0.6277 | 5.37 | 0.856 |
| a6 PRF + IS + bnorm | 0.4854 | 14.13 | ~55 |
| a7 FRLR, neither | 0.6559 | 5.0849 | 2.243 |
| **a8 FRLR, Q cadence 20** | **0.6602** | 6.8293 | 2.931 |

a8 is at **1.004x the incumbent's learning at 0.48x its gap**, with a gradient
norm 1.6x the incumbent's, which is unremarkable. a7 got to 0.997x; a8 clears it.
The difference is inside normal run-to-run variation and I am **not** claiming a8
learns better than the incumbent. The claim that holds is that **slowing the Q
refresh cost nothing in capability**, which was the open question a7's verdict
left, and it is the answer that matters: the every-step refit was **not**
load-bearing.

## The gap trend: the defect that killed a7 is largely fixed

| window | a7 gap slope | a8 gap slope |
|---|---|---|
| 61-80 | +0.01637 | (see below) |
| 100-120 | **+0.016351** | **+0.001262** |
| 61-120 | accelerating | **-0.026533** |

Over the secondary window the a8 gap is **falling**, and its level fell
monotonically 11.7151 -> 8.0304 -> 6.8293 across the run. The registered slope
clause still fails at 100-120 by 2.5x, so a8 does **not** clear the bar, but the
"gap no longer settles" finding that was a7's one real defect is **13x smaller
here and pointing the other way** over the longer window.

This is the measurement that refuted my own staleness explanation. I had
attributed a7's climb to `Q` lagging a policy it was chasing. Freezing `Q` for 20
steps should then have made it *worse*. It made it 13x better, so the mechanism is
**Q estimator variance**, not staleness: at cadence 1 the basis is re-derived every
step from ONE batch with ONE power iteration and jitters; at cadence 20 the sketch
accumulates over 20 batches before `orth`.

## The new fact: the codec-view inflation FALLS here

| | `probe/kl_gain` first | last | direction |
|---|---|---|---|
| a7 (cadence 1) | 71.6x @50 | **710.2x** @200 | rising |
| **a8 (cadence 20)** | 5641.4x | **157.4x** | **falling** |

a7's inflation rose because `Q` was refit to the current policy every step while
the FROZEN reference was reconstructed ever worse. Slow the refresh and the
inflation behaves like PRF's, which also falls (134.6x -> 10.9x) because its mask
is policy-independent. That is direct support for the mechanism in
`FINDING_drift_metric_invalid.md` from a third codec configuration, and it is the
basis of prediction P4 for a9.

The consequence for the drift column: a8's codec-view `actor/kl_loss` reads
**0.106405** against a7's 5.8246, a 55x difference between two arms whose only
distinction is how often `Q` moves. **Nothing about the policy changed by 55x.**
Any gate on that channel would rank these two arms almost arbitrarily.

## Codec-free drift is the lowest in the program

`probe/kl_dense` last = **0.007006** at step 150, full-run slope **+5.1e-05**.
`probe/gap_dense` averages **0.000302 nats**, so the codec accounts for a factor
of **22619** in the measured gap: essentially all of the 6.83 nats is codec view,
not policy divergence. `lr_brake` fired **0 of 30** probes, against a7's 1.

## Health

`grad_norm` 2.931 at the window (max 53.8 excluding the step 1-3 transient, which
is high but a7 also spiked and neither cell destabilised). Sampler-side
`rollout_log_ppl` **0.1817**, normal. Codec-view entropy reads 4.3549 and is
~34x inflated, so it is not a health signal. Score min 0.0 / max 1.0 across the
window, so no reward degeneracy.

## Verdict

**PASS on G1 and G2-level, FAIL on both slope clauses, and it is the arm to
build on.** a8 has the best learning in the program, less than half the
incumbent's gap, the flattest gap trend of any FRLR arm by 13x, the lowest
codec-free drift, no importance-sampling machinery, and wire parity to 0.1
percent. Its two failures are a slope that misses by 2.5x while falling over the
longer window, and a codec-view gate with no physical content.

It also has a **known confound I built in**: at cadence 20 over 200 steps `Q` gets
only 10 power iterations against a7's 200, so a8 varies both view stationarity
(intended) and total `Q` fitting (not). Its higher gap *level* against a7 (6.83 vs
5.08) is most likely that under-fitting rather than a cost of the slow cadence.
Cadence 5 would have separated them; I am not spending a cell on it, because the
operator's instruction supersedes the question.

## Consequence: a9 is the limit case, and it is running next

The operator's instruction of 2026-07-26 is that `Q` should move **only in the
anchor and only when it fires**, as PowerSGD has always done. That is the limit of
what a8 validates, and it adds two things a cadence knob cannot: `Q` fitted to the
**slow stale-weight net** so it cannot chase the policy at all, and the `Q`
broadcast on the **uncharged slow circuit**, restoring exact 1232-bit parity.
`a9-frlr-anchorq-200` is pre-registered in `PREREG_a9.md` and chained to launch
when a8 ends. `a10` follows with `frlr_unbiased=true`.
