# Verdict: a8-frlr-qcad20-200. Capability tied with a7 and the incumbent, and the gap trend 13x flatter.

## STABILITY VERDICT (2026-07-28 re-scoring)

> **Re-scored against stability, not reward.** The body below this section was
> written against a bar that leads with capability. Capability turned out to be
> a tie across the field, so it cannot carry a conclusion. What follows
> supersedes the original ranking claims; the original text is kept in place.

**Stability rank: 4 of 12.** a8 is FRLR with `frlr_q_cadence` raised from 1 to 20, and it is the best-behaved gap in the program on the drift-ratio axis: it is the only arm whose gap ends BELOW its step-100 level, but it bought its flattering gradient number with a large startup transient, and the FRLR family's only horizon test fails.

| axis | this arm | reference | read |
|---|---|---|---|
| gap slope | +0.001262 over 100-120; +0.001485 over 100-199, n=100 | incumbent +0.000838 over 100-120; +0.000848 over 100-599, n=500 | 5th on the matched 100-120 window, roughly 1.5x the incumbent's slope, and a8 has no window past 199 to test with |
| gap drift ratio | 0.986, level 7.242 at 100 falling to 6.829 at 120 | incumbent 1.029 | best in the entire field, the only arm ending below its step-100 level |
| grad_norm drift | 0.05x, p50 42.9403 over the first 20 percent falling to 2.1538 over the last 20 percent | incumbent 0.85x | not calm, this is a large startup transient decaying; the fact sheet says to read a sub-1.0 ratio next to the run max |
| grad_norm max | 53.82, max/p50 18.6x | incumbent 4.645, max/p50 2.9x | an order of magnitude looser than the incumbent on both the level and the peak-to-median shape |
| collapse / kill | none, ran 200/200 to its scheduled end | incumbent none in 600 | clean at 200, but a8 has one third of the incumbent's horizon and its corrected family continuation c600 fails at 600 |
| capability | val 0.6613 at 200; training reward 0.6837 over 101-200 | incumbent 0.6613 at 150 and 0.6613 at 600; a7 and a9 both 0.6713 at 200; dense 0.6774 at 600 | does not separate the arms |

What a8 proves about stability is narrow and real: freezing `Q` for 20 steps produces the only gap in the program that is genuinely settling rather than merely flat. Drift ratio 0.986 beats a3's 1.001 and the incumbent's 1.029, and it is the sole sub-1.0 entry in the field. That is a mechanism result, not a horizon result. It says `Q` estimator variance, not `Q` staleness, was driving a7's climb, and the matched-step evidence backs it: a7's slope over 100-199 is +0.028329 against a8's +0.001485 on the same window, a 19x difference from the cadence knob alone.

What a8 does not prove is that FRLR is stable. Its 0.05x gradient drift ratio is the single most misleading number in this document. A median of 42.9403 over the first 20 percent of a 200-step run means the elevated regime covers roughly the first forty steps, not the "step 1-3 transient" the body below calls it. The ratio is low because the run started badly and recovered, not because the optimizer sat still, and the run max of 53.82 against the incumbent's 4.645 is the check that catches it. Corrected here rather than edited away: the body's line that grad_norm is "unremarkable" at 1.6x the incumbent is scoped to the window value 2.931 and does not describe the run.

Two further body claims now read wrong and are corrected here rather than edited away. First, "codec-free drift is the lowest in the program" is an unmatched-step artifact of the same kind the fact sheet already flags for a9. At the matched step 200 a8's `probe/kl_dense` is 0.010872 against a7's 0.008201 and c600's 0.008186, the HIGHEST of the three, and at matched step 150 a7's 0.005214 is below a8's 0.007006. Second, "it is the arm to build on" does not survive. The FRLR family's only 600-step test is c600, which crosses the incumbent at step 417 and permanently at 424, ends at gap ratio 5.122 with grad_norm drift 9.25x and a run max of 176.367, and a10 showed that removing the estimator bias removes FRLR's early gap advantage entirely. The upgrade candidates worth a horizon run are a4 and a3, both PRF-side.

Against the incumbent, a8 wins one axis and loses the rest. It wins gap drift ratio 0.986 to 1.029 and it wins gap level, 6.829 at step 120 against 14.239. It loses gap slope on the matched window, it loses gradient shape by roughly an order of magnitude on both max and max/p50, and it has 200 steps of evidence against 600. Note also that the body's 55x `actor/kl_loss` spread between a7 and a8 is quoted here only as evidence that the channel is an instrument artifact, never as a ranking input: `actor/kl_loss` is disqualified because it is real drift multiplied by a codec-view inflation factor that itself moves by 50x between arms, so it confounds the measurement with the thing measured. The same disqualification applies to the body's `entropy` line: the dense control sharpens the same way, so entropy decline is normal GRPO on math and not compression damage.

> **Header corrected after the terminal val landed.** This document was first
> written at the training window, where a8 leads, and titled "the best cell in the
> program". Its terminal val (0.6613) came in **below** a7's 0.6713, so the two
> channels rank the arms oppositely and both differences are inside the
> reference's own noise. The terminal addendum has the arithmetic. Corrected here
> rather than edited away.

Scored 2026-07-26T13:35Z at the registered primary window **100-120**, complete at
21 rows. Cell: a7's exact codec with `frlr_q_cadence` raised from 1 to 20. Run
continues to 200 for its terminal val; a terminal addendum follows.

## The registered bar

| gate | measured | bar | call |
|---|---|---|---|
| **G1** learning, score level | **0.6602** | >= 0.6248 | **PASS**, 1.004x the incumbent's 0.6577 at the window; the held-out val disagrees, see addendum |
| **G2** gap level | **6.8293** | < 14.2458 | **PASS, 0.48x** |
| **G2** gap slope | **+0.001262** | <= +5.0e-4 | FAIL 2.5x, but **13x flatter than a7** and negative over 61-120 |
| **G3** drift slope | +0.006967 | <= 3.264e-3 | FAIL 2.1x, codec-view, no physical content |
| **G4** wire | 1233.4 bits | 1232 | parity to 0.1 percent |

## G1 at the training window (see the terminal addendum: this does NOT survive as a capability claim)

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
build on.** a8 has capability tied with a7 and the incumbent, less than half the
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

---

## TERMINAL ADDENDUM at step 200: the training window and the held-out val disagree, and the honest reading is a tie

a8 finished 200/200. Terminal val, captured from the on-box log because WandB drops
the final step: **0.6613**.

| cell | score 100-120 (training) | terminal val (held-out MATH) |
|---|---|---|
| incumbent PRF | 0.6577 | 0.6613 @150, 0.6633 @300, 0.6733 @450, 0.6613 @600 |
| a6 PRF + IS + bnorm | 0.4854 | 0.5391 |
| a5b FRLR + IS + bnorm | 0.6277 | 0.6593 |
| a7 FRLR, no IS | 0.6559 | **0.6713** |
| **a8 FRLR, Q cadence 20** | **0.6602** | 0.6613 |

**The two channels rank a7 and a8 oppositely.** a8 wins the training window by
0.0043; a7 wins the held-out val by 0.0100. I am not going to pick whichever
favours the arm I prefer, so:

- The val set is MATH-lighteval at **499 problems**, so one problem is 0.20 percent.
  a7's 0.6713 is **335/499** and a8's 0.6613 is **330/499**: a **five-problem**
  difference.
- The incumbent's own checkpoint-to-checkpoint spread is **0.0120**, larger than
  the a7-a8 gap of 0.0100.
- a8's 0.6613 is **exactly** the incumbent's value at both step 150 and step 600.

**So a7 and a8 are tied on capability, inside the reference's own noise, and the
earlier "best learning in the program" phrasing overstates it.** What survives is
the weaker but sufficient claim: **slowing the Q refresh cost no measurable
capability.** That was the open question a7's verdict left, and it is answered.

### Which makes the gap trend the discriminator, and there a8 wins decisively

| | gap @200 | gap slope @100-120 | gap slope 61-120 | direction |
|---|---|---|---|---|
| a7 | 8.1849 | +0.016351 | rising, accelerating | **wrong** |
| **a8** | **~6.83** | **+0.001262** | **-0.026533** | **falling** |

On the program's registered criterion, a settling gap, a7 fails clearly and a8
nearly passes while falling over the longer window. With capability tied, that is
the whole comparison, and it is why a9 (the limit case of a8's mechanism) is the
right next cell rather than a7 at 600.

### Instrument check for a9

a8's harvested `anchor_q_refreshes.txt` snapshot is **empty, 0 lines**, exactly as
it must be: a8 refreshes `Q` on the fast path, so no `[comm_eff][frlr-anchor-q]`
line can exist. That makes the same file a positive control for a9, where it must
be **non-empty** from the first anchor fire at step 20.
