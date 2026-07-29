# Verdict: c600-a9-anchorq-val600. FRLR survives 600 steps and fails them.

## STABILITY VERDICT (2026-07-28 re-scoring)

> Written directly against the stability bar, not against reward. Capability is
> a tie across every arm in this program that did not collapse, so it cannot
> carry a conclusion. The three axes that decide the ranking are gap
> stationarity, gradient-norm behaviour, and non-collapse. Every number in this
> file comes from the single authoritative WandB snapshot of 2026-07-28
> (`scratchpad/wandb93.json`, entity `shamanework-pl`, step axis
> `training/global_step`), run id `5v3hrpef`.

**Stability rank: 10 of 12.** c600 is a9's configuration, corrected and carried
to 600 steps. It is the only FRLR arm in the program ever tested at horizon. It
did not collapse, it kept its capability to the last validation, and its
optimizer left the steady state anyway: the gap ends 5.12x its step-100 level,
the gradient-norm median rises 9.25x, the run maximum reaches 176.367, and it
crosses the incumbent permanently at step 424.

| axis | c600 | reference (incumbent, PRF exact-k 600) | read |
|---|---|---|---|
| gap slope | **+0.045972** over 100-599 (its full window, n=500); +0.008696 over the matched 100-120, 9th of the 11 arms that reached 120 | **+0.000848** over 100-599 (n=500); +0.000838 over 100-120 | 54x steeper than the incumbent over the identical 500-step window. The one comparison in the program where both arms have the same horizon, and it is not close |
| gap drift ratio | **5.122** | **1.029** | the gap ends five times its step-100 level. The worst value in the program by 3.3x over the next worst (a7, 1.551) |
| grad_norm drift | **9.25x**, p50 3.5370 over the first 20 percent of steps rising to 32.7205 over the last 20 percent | 0.85x, 1.7866 falling to 1.5259 | the only arm in the program above 2.3x. This is a rising median over 600 steps, not a startup transient, and the direction is the opposite of a8's 0.05x or a9's 0.40x decay |
| grad_norm max | **176.367**, max over its own p50 12.1x; run min 1.531, median 14.551, last 35.534, last-50-step max 131.317 | 4.645, max over its own p50 2.9x; min 1.181, median 1.594, last 1.337 | absolute level is not comparable across codecs, but the shape is: the incumbent's envelope is bounded for 600 steps and c600's is not |
| collapse / kill | none. 600/600 steps, no kill trigger fired, `pg_clipfrac` exactly 0 at every one of its 600 steps | none in 600 | c600 passes the non-collapse axis outright. This is what makes it the interesting failure rather than a boring one |
| capability | 0.6573 @300, **0.6633 @600** (log-confirmed 0.657314629258517 and 0.6633266533066132) | 0.6613 @150, 0.6633 @300, 0.6733 @450, **0.6613 @600**; dense 0.6774 @600 | does not separate the arms. c600 ends 0.0020 ABOVE the incumbent while its optimizer diverges |

c600 sits at 10 because it is the only arm that failed with a complete horizon
dataset. Every arm ranked above it either has flatter numbers on a shorter
window (a3 and a4 at 120 steps, a8 and a9 at 200) or was stopped before it could
prove anything (a10, killed at 62 for futility with a clean grad_norm max of
2.285). It sits above a2, killed at step 60 with a run-minimum grad_norm of
6.153, and above a6, which combined the flattest gap of any 200-step arm
(+0.000413 over 100-199) with a collapsed model (val 0.5391, grad_norm max
608.81). c600 is worse than everything with less evidence and better than
everything that broke.

## 1. What the run was, and why it is the first clean FRLR cell

Cell `c600-a9-anchorq-val600`, run id `5v3hrpef`. a9's exact configuration:
anchor-owned FRLR at rank 48 with k=28 residual coordinates, the biased capped
per-token norm-matching gain, no token-IS, probe cadence 5, `Q` harvested from
the anchor's clean stale-weight forward and refreshed only when the anchor fires
at cadence 20 optimizer ticks. Val at 300 and 600. 600 steps, completed.

The cell exists because a9 itself was not clean. The claim that `pg_clipfrac` is
zero by construction was FALSE for anchor-owned `Q`: `anchor_update_basis`
published `Q` immediately, but the anchor fires inside `train_batch` AFTER
`old_log_probs` were recomputed, so the two forwards of one step used different
bases and the PPO ratio deviated from 1 for a measurement reason rather than a
policy one. The staging half of the PowerSGD anchor contract was ported in
commit `f2ac3c64`, and that landed before c600 launched.

The snapshot verifies the fix held for the whole 600 steps:

| run | nonzero `pg_clipfrac` steps | max |
|---|---|---|
| a9 | **9** (first at step 20) | **0.352534** |
| a10 | 3 | 0.374120 |
| **c600** | **0** | **0** |
| every other run in the program | 0 | 0 |

So c600 is not just the horizon run for FRLR, it is the first and only clean
measurement of the anchor-owned governance variant. Nothing in what follows can
be blamed on the staging bug. The bug was also not what made a9's gap climb: over
the matched 100-120 window c600 reads +0.008696 against a9's +0.009262, so the
corrected run is marginally flatter early and then goes on to a 100-599 slope of
+0.045972.

## 2. The crossover: FRLR's advantage is real, and then it inverts

The whole case for FRLR over the incumbent was a lower sampler-trainer mismatch
gap. At step 100 that case is strong. By step 599 it is reversed.

`rollout_corr/kl` level, c600 against the incumbent:

| step | incumbent | c600 | ratio |
|---|---|---|---|
| 100 | 14.243 | **5.580** | 0.39 (c600 2.55x better) |
| 200 | 14.318 | 6.846 | 0.48 |
| 300 | 14.429 | 9.199 | 0.64 |
| 400 | 14.479 | 13.190 | 0.91 |
| 450 | 14.542 | 17.805 | 1.22 |
| 599 | 14.659 | **31.104** | **2.12 (c600 2.12x worse)** |

**First step where c600 exceeds the incumbent at all: 417. First step after
which it stays above for the rest of the run: 424.** An earlier draft of this
analysis said the crossing was at 413. That was wrong and is corrected here
rather than edited away: use 417 first, 424 permanently.

Over the identical 100-599 window the incumbent's gap moved 0.42 nats and
c600's moved 25.52 nats, computed from the level table above. The slopes are
+0.000848 and +0.045972 over that same 100-599 window, n=500 rows each.

Two things follow. First, an early-window gap advantage is not evidence about
horizon behaviour, and the registered gate window 100-120 would have scored
c600's gap level as a decisive pass (5.580 against 14.243) at the exact moment
the arm was already on the trajectory that ends at 31.104. Second, the
inversion was predicted in advance by the section-20 theory: PRF's error is
rotation-invariant so its gap is stationary, while FRLR's error is
alignment-dependent so its gap chases a moving subspace. The measurement matches
the mechanism, which is the reason to trust it.

For the record, the run's registered kill triggers (from the launcher record in
`COMPACT_PROMPT_4.md`) were the gap crossing the incumbent's 14.3 or val@300
falling below 0.65. Val@300 was 0.6573 and passed. The gap trigger would have
fired somewhere between step 400 (13.190) and step 450 (17.805), and the run was
allowed to complete instead. Completing it was the right call, because the
600-step tail is the entire finding.

## 3. The optimizer walks away

This is the single clearest stability figure in the program, because it is the
only place where two codecs are compared over the same 600 steps with the same
statistic. `actor/grad_norm` in 50-step blocks, both 600-step arms, in full:

| block | inc max | inc p50 | c600 max | c600 p50 |
|---|---|---|---|---|
| 1-50 | 4.645 | 1.815 | 28.551 | 4.970 |
| 51-100 | 4.142 | 1.802 | 4.789 | 2.537 |
| 101-150 | 2.796 | 1.713 | 9.049 | 4.903 |
| 151-200 | 3.835 | 1.681 | 56.038 | 5.709 |
| 201-250 | 3.764 | 1.627 | 19.720 | 6.784 |
| 251-300 | 2.683 | 1.554 | 22.057 | 9.632 |
| 301-350 | 4.166 | 1.554 | 41.602 | 13.623 |
| 351-400 | 2.363 | 1.522 | 58.687 | 21.379 |
| 401-450 | 2.865 | 1.500 | 48.462 | 25.212 |
| 451-500 | 2.054 | 1.513 | 74.714 | 31.945 |
| 501-550 | 2.455 | 1.528 | **176.367** | 27.966 |
| 551-600 | 2.559 | 1.534 | 131.317 | 35.534 |

The incumbent's block median is flat across all twelve blocks, spanning 1.500 at
401-450 to 1.815 at 1-50, and its block maximum never exceeds 4.645, which
occurs in the very first block. That is a bounded envelope with a gently
declining centre over 600 steps, and it is what a stationary optimizer under
compression looks like.

c600's block median climbs from 2.537 at 51-100 to 35.534 at 551-600, rising in
every successive block except a single dip from 31.945 at 451-500 to 27.966 at
501-550. Its block maximum exceeds the incumbent's whole-run maximum of 4.645 in
every one of the twelve blocks, and from block 3 onward it is never below 9.049.
The largest single value,
176.367, lands at 501-550, and the final block still carries a maximum of
131.317, so the excursions are not a one-off spike that the run recovered from.
The whole-run summary reads min 1.531, median 14.551, max 176.367, last 35.534.

Note what this is not. Absolute grad_norm level is not comparable across codecs,
because different codecs put the optimizer at genuinely different scales, which
is why the drift ratio and max over p50 are the statistics used. On both
scale-free statistics c600 is the worst 600-step behaviour measured here: drift
9.25x against the incumbent's 0.85x, max over p50 12.1x against 2.9x. It is also
worth reading against reward, because the a5 lesson is that a quiet optimizer on
a run that is not learning is worthless. The inverse holds too: c600's optimizer
is not quiet and c600 was learning perfectly well, which is what makes the
divergence hard to dismiss as a suppressed update or a broken step.

## 4. Capability held, and that is the point

Held-out val, `val-core/DigitalLearningGmbH/MATH-lighteval/acc/mean@1`:

| step | dense | incumbent | c600 |
|---|---|---|---|
| 150 | 0.6593 | 0.6613 | - |
| 300 | 0.6653 | 0.6633 | 0.6573 |
| 450 | 0.6874 | 0.6733 | - |
| 600 | **0.6774** | **0.6613** | **0.6633** |

At step 600 c600 scores 0.6633 and the incumbent scores 0.6613. The arm whose
gradient norms rose 9.25x and whose gap rose 5.12x ends 0.0020 ABOVE the arm
that stayed stationary, and 0.0141 below dense. Compression costs about 1.4
points of val against dense at 600 steps and both codecs pay it about equally.

Training reward tells the same story. `critic/score/mean` in 100-step block
means:

| arm | 1-100 | 101-200 | 201-300 | 301-400 | 401-500 | 501-600 |
|---|---|---|---|---|---|---|
| dense | 0.5764 | 0.6729 | 0.6999 | 0.7150 | 0.7291 | **0.7437** |
| incumbent | 0.5015 | 0.6726 | 0.6996 | 0.7216 | 0.7344 | **0.7406** |
| c600 | 0.5081 | 0.6833 | 0.7092 | 0.7271 | 0.7351 | **0.7380** |

c600's final block is 0.0026 below the incumbent and 0.0057 below dense, so all
three 600-step runs finish within 0.006 of one another, and c600 is actually
ahead of both on the 101-200, 201-300 and 301-400 blocks. Its reward curve is
monotone increasing through the end of the run.

This is the reason capability cannot be the stability bar. A run can hold its
score, hold its reward curve, and finish its schedule while its optimizer walks
away from the steady state. If the program had scored c600 on val alone it would
have declared FRLR at horizon a success. The counterexample in the other
direction is a6, which had the flattest gap of any 200-step arm (+0.000413 over
100-199) and collapsed to 0.5391. Capability and gap flatness each fail
independently as a sole criterion, and the bar has to be all three at once: flat
gap, stationary optimizer, no collapse. c600 clears only the third.

## 5. Real drift keeps accelerating while the codec view deflates

`probe/kl_dense` is the codec-free channel. For c600 it rises throughout:

| step | 60 | 100 | 120 | 150 | 195 | 200 | 300 | 450 | 600 |
|---|---|---|---|---|---|---|---|---|---|
| c600 | 0.002647 | 0.004441 | 0.005216 | 0.006209 | 0.007405 | 0.008186 | 0.011176 | 0.016189 | **0.027622** |

Secant rates computed from those levels: 2.990e-5 per step over 200-300,
3.342e-5 per step over 300-450, and 7.622e-5 per step over 450-600. The last
window is 2.3x the rate of the 300-450 window and 2.5x the rate of 200-300, so
physical drift is not settling, it is accelerating in the final third of the
run. That coincides exactly with the region where the gap crosses the incumbent
(417/424) and the grad_norm block median passes 25.

Meanwhile the codec-view inflation factor `probe/kl_gain` FALLS from 589.7x at
step 300 to 354.4x at step 600.

**`actor/kl_loss` appearing to settle over the back half of c600 is the product
of a rising term (real drift, 0.011176 to 0.027622 over 300-600) and a falling
term (inflation, 589.7x to 354.4x over 300-600), so it is disqualified as a
ranking metric: it is real drift multiplied by a codec-view instrument factor
that itself moves by 50x between arms and across a run.** The same
disqualification applies to `actor/entropy`, which is also a codec-view
quantity: c600 reads 6.14 falling to 0.053 while the incumbent reads a flat 7.81
to 7.85, and yet sampler-side `rollout_log_ppl` at step 599 is near-identical
across c600, the incumbent and dense (0.091 / 0.093 / 0.108). Neither metric may
order arms in this program.

One clean cross-check that c600 buys with its probe: at the matched step 200,
c600 reads 0.008186 and a7 reads 0.008201, identical to 0.2 percent. a7 is fast
`Q` at cadence 1 and c600 is anchor-owned `Q`, so this re-confirms with clean
post-fix data that `Q` governance moves the gap and the codec view and does not
touch physical drift at all.

## 6. What this run does not prove

Say this plainly: **the incumbent has no probe and neither does dense, zero
points on `probe/kl_dense`, so codec-free drift cannot be compared between PRF
and FRLR at any step, including step 600.** Everything in section 5 is a
statement about c600's own trajectory. It is not a statement that FRLR drifts
more than PRF in physical terms, because that measurement was never taken on the
incumbent. The comparison that does exist between the two codecs is axis 1 (the
mismatch gap), axis 2 (gradient-norm shape), axis 3 (collapse) and axis 5
(capability), and c600 loses on the first two, ties on the third and ties on the
fourth.

Three further limits.

First, the horizon evidence for FRLR is n=1. c600 is one run of one governance
variant. It is corroborated by the 200-step trends of the same family (a7
+0.028329 over 100-199 with drift ratio 1.551 and grad_norm drift 1.86x to a max
of 68.01; a9 +0.012203 over 100-199 with ratio 1.192), but corroboration by
shorter runs is not the same as a second horizon run.

Second, a8 is the genuine gap in the argument. a8 is the only arm in the entire
program whose gap ENDS below its step-100 level (drift ratio 0.986), with a
100-199 slope of +0.001485, and it has no horizon evidence at all. c600 was
already visibly climbing at the same point (its gap goes 5.580 at step 100 to
6.846 at step 200, a ratio of 1.23), so a8's 200-step trajectory is not c600's
trajectory. Whether a8's fast-`Q`-cadence-20 governance would also invert by 600
is untested. a8 is additionally undercut by a 42.9403 startup transient in its
first 20 percent of steps and by its registered-window slope sitting on the
descending arm of a U rather than on a settled trend, but neither of those is a
horizon result.

Third, this run says nothing about out-of-domain behaviour. No OOD suite was run
on c600, so the only capability statement available is in-domain MATH val.

## 7. Verdict

**FRLR is out for long-horizon work in every governance variant tested.**

The family was run seven ways in this program: a5 (token-IS), a5b (token-IS plus
bnorm), a7 (fast `Q`, cadence 1), a8 (fast `Q`, cadence 20), a9 (anchor-owned
`Q`), a10 (anchor-owned `Q`, unbiased) and c600 (anchor-owned `Q`, clean, 600
steps). The results compose into a closed case:

- Carried to horizon, it inverts. c600's gap slope is +0.045972 over 100-599
  against the incumbent's +0.000848 over the same 100-599 window, drift ratio
  5.122 against 1.029, gradient drift 9.25x against 0.85x with a run max of
  176.367 against 4.645, and it crosses the incumbent at 417 and permanently at
  424.
- Its early advantage requires a biased estimator. a10 removed the bias and the
  gap went to 14.93, which is the incumbent's own operating point (14.66 at step
  599), so a10 was killed at 62 for futility rather than instability.
- Its governance knob does not touch the thing that matters. c600 and a7 read
  `probe/kl_dense` 0.008186 and 0.008201 at the matched step 200, so `Q`
  governance moves only the gap and the codec view.
- It carries machinery the winner does not need: a basis, a `Q` to govern, an
  anchor coupling and a broadcast, in exchange for an advantage that expires at
  step 417.

The incumbent, PRF exact-k with 77 of 1536 coordinates, a constant 1/(1-p)
rescale and a mask that is a PRF of seed/step/layer, remains the only
configuration with 600 steps of evidence that the optimizer stays in a steady
state. It is unbiased, has no side channel, needs no basis broadcast and no
anchor coupling. Its win is narrower than it looks: over the short matched
100-120 window it ranks 4th (+0.000838) behind a3 (+0.000101), a5b (+0.000358)
and a6 (+0.000383), and its claim rests on being the only arm whose flatness was
tested to 600 steps. The two upgrades worth a horizon run, in order, are a4 (PRF
exact-k plus CVC cross-entropy: gap +0.001542 with ratio 1.002, grad_norm drift
0.91x, max over p50 2.0x at 120 steps) and a3 (sr_quant 2-bit byte-parity: gap
+0.000101 with ratio 1.001, grad_norm drift 0.71x, max over p50 2.5x at 120
steps). Neither is FRLR, and neither should borrow FRLR's early gap numbers as
encouragement.
