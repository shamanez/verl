# Verdict: issue 93 round A, cell `a1-srq-b1-sr`

VERDICT: REVISE

Cell `a1-srq-b1-sr` (WandB `h0n67q3a`, project `93-long-horizon-stability`) completed
120/120 steps cleanly in 3h58m with no collapse signature, no NaN, no gaps and no
tripwire, and it is informative: it answers the round-A mechanism question in the
negative. On the gate that matters it misses. The train-inference gap at the 100 to 120
window is **13.751 nats** against a `< 10` gate and a `< 3` target, only 3.4 percent
below the incumbent's 14.24, bought at **1.8701x the incumbent's wire budget** (2304
bits/token/boundary logged against 1232). Its reference-KL slope, the only part of its
reference KL that is comparable to the baseline, is **+0.003887/step against the
baseline's +0.0015/step, 2.6x worse**. Two of five gate items pass cleanly, one passes
as literally written but fails on every window overlapping the gate read, and two fail.
That is a fixable-miss-with-a-finding, not a falsification and not a divergence:
`revise_depth` 0 of `iterations` 7, the round-level stage-1 gate is judged at the end of
round A across all five arms, a2 is already running, and nothing here justifies stopping
the matrix. REVISE.

Judged against: issue 93 section 1 baseline card (plan resolved live from the issue
body), cross-checked against `run.json` `success_criteria[0]`. Both agree.

---

## 1. Gate table (issue 93 section 1, read at steps 100 to 120, n = 21)

All observed values from `runs/93-long-horizon-stability/analysis-a1.log`, which is a
single offline parse of `runs/93-long-horizon-stability/metrics/incoming.log` filtered
to a1's Ray driver (`TaskRunner pid=6382`, `experiment_name: a1-srq-b1-sr`).

| # | gate item | baseline | a1 observed | source | flag |
|---|---|---|---|---|---|
| 1a | reference KL `actor/kl_loss`, absolute at matched step | 0.156 to 0.203 | mean **2.251837**, last 2.304283 | analysis-a1.log GATE WINDOW | **NOT MEANINGFUL** |
| 1b | reference KL, slope (the comparable component) | +0.0015/step | **+0.003887/step**, ols_se 0.000234, hac_se 0.000251, r2 0.935 | analysis-a1.log GATE WINDOW | ✗ FAIL, 2.59x |
| 2 | train-inference gap `rollout_corr/kl` < 10 nats, target < 3 | 14.24 | mean **13.751145**, last 13.774802 | analysis-a1.log GATE WINDOW | ✗ FAIL, 1.38x the gate |
| 3 | training reward slope >= 0.00288 (90 percent of 0.0032) | 0.0032 | full run **+0.003235/step**; gate window **-0.001074/step** | analysis-a1.log FULL RUN / GATE WINDOW | ✓ PASS as written, with a documented caution (section 4) |
| 4 | `actor/ppo_kl == 0` (within-step view identity) | about 0 | **exactly 0.0** at all 120 steps; `pg_clipfrac` likewise exactly 0.0 | analysis-a1.log FULL RUN | ✓ PASS |
| 5 | codec confinement counters clean | n/a | 78358 / 34138 / 6 / 19, all cross-checks consistent | analysis-a1.log CONFINEMENT | ✓ PASS |
| - | no collapse, no capability damage (standing cardinal rule) | n/a | entropy 7.899606 to 7.938554 (rising), grad_norm max 0.898240 at step 11 and falling, `kl_coef` pinned 0.001 | analysis-a1.log | ✓ PASS |

Score: 2 clean passes plus 1 conditional pass plus 1 no-collapse pass, against 2 fails.
Item 2 is the gate the program exists to move, and it is the one that misses hardest.

**On item 1a.** a1's absolute reference KL is not comparable to the baseline's and is not
scored. The quadratic fit over steps 2 to 120 gives an intercept of **1.87260** and the
step-76 pre-read's linear fit gave about 1.86; that constant is the 1-bit stochastic
rounding view offset, the codec's distortion of the measurement rather than policy
movement. The detrended within-half noise amplitude is constant (0.0126 versus 0.0122,
`preread-a1.md`) while the level moves, which is exactly the signature of a stable
measurement offset with a real drift on top. Reporting "2.25 versus 0.18" as a 12x
capability failure would be wrong. Only the slope is a like-for-like comparison, and the
slope fails on its own.

**On item 1b, window dependence.** a1's reference-KL slope is not a single number and
the verdict should not pretend it is. From `analysis-a1.log` SEGMENT SLOPES:
steps 2 to 39 **+0.001743**, 40 to 77 **+0.004592**, 78 to 120 **+0.003684**, gate window
**+0.003887**, honest full run (steps 2 to 120, excluding the step-1 structural zero)
**+0.003581**. The dispatch's +0.00435 is the steps 1 to 120 fit, which includes the
step-1 zero and is therefore the least honest of the set. Every one of these is 1.2x to
2.6x the baseline's +0.0015, so the FAIL does not depend on the choice, but note that the
pre-read's "accelerating" reading was true through step 77 and then stops: the slope
levels off near +0.0037 to +0.0039 rather than running away. a1's reference KL drifts
faster than the incumbent, and it does not blow up.

**On item 5, why the counters are clean and not merely non-zero.** `anchor_replay_fires`
= 6 is exactly `floor(120/20)`, which is what cadence 20 in optimizer ticks predicts when
`train_batch_size` equals `ppo_mini_batch_size` and there is one tick per global step.
`merger_coldM_fallbacks` = 0, `rank1_zero_motion_tensors` = 0, `rank1_m_ready` = 1,
`rank1_r2_mean` = 1.0, `rank1_evr_mean` = 1.0, `rank1_fires` = 5 with
`rank1_prediction_horizon` = 20 and `rank1_window_span` = 20. `mask_applications`
decomposes as 19691 old_logprob + 19691 ref_logprob + 38976 train, with 0 on the
checkpoint path: the old and ref passes are applied an identical number of times, which
is the PRF-keyed bit-identity that makes item 4 hold.

**On item 4, deliberately not flagged as suspicious.** `actor/ppo_kl` and `pg_clipfrac`
being exactly 0.0 at every one of 120 steps is the correct and expected value here, not
an unpopulated metric. `train_batch_size` 128 equals `ppo_mini_batch_size` 128, so there
is one inner update per step and the PPO ratio is identically 1. The independently logged
`anchor_replay_fires` = 6 confirms the one-tick-per-step structure from the metrics side.
Scored as a genuine pass.

---

## 2. Mechanism reading: what 1-bit stochastic rounding bought

**What it was supposed to buy.** Per issue section 4.1 and 4.2: the incumbent PRF exact-k
mask is unbiased but violent, noise energy about **19 ||h||^2** with heavy tails (each kept
channel scaled x20), at 1232 bits/token/boundary. 1-bit sr_quant at block 32 is also
unbiased but bounded, noise energy about **7 ||h||^2** with no heavy tails, at
1.87x the bytes. The hypothesis was that a cleaner unbiased estimator at the source shrinks
the train-inference gap.

**What it actually bought, quantified.**

| quantity | incumbent PRF exact-k | a1 1-bit SR | change |
|---|---|---|---|
| wire budget, bits/token/boundary | 1232 | **2304** (288 bytes, logged) | **1.8701x**, +1072 bits |
| activation noise energy | about 19 ‖h‖² | about 7 ‖h‖² | 2.7x lower, tails bounded |
| train-inference gap at the gate | 14.24 nats | **13.751 nats** | **-0.489 nats, -3.4 percent** |
| reference-KL slope at the gate | +0.0015/step | **+0.003887/step** | **2.59x worse** |
| E[rho] = k3_kl - kl + 1 | 0.0014 | **0.005479** (median 0.002527, last 0.002017) | 3.9x higher, still 9.1x below the program's 0.05 threshold and about 180x below 1.0 |

**The exchange rate.** +1072 bits/token/boundary bought 0.489 nats, i.e. about **2192
extra bits per nat of gap**. Expressed as an elasticity in noise energy, the two anchor
points (19 → 7 ‖h‖², 14.24 → 13.751 nats) give **0.494 nats of gap per e-fold of noise
energy**. Since noise energy falls about 7.8x per extra bit per coordinate (7 → 0.9 ‖h‖²
from 1 bit to 2 bits, issue 4.2), that is a memorable and unwelcome exchange rate of
**about 1 nat of gap per extra bit per coordinate**. Closing the remaining 3.751 nats to
reach the `< 10` gate therefore needs roughly 3.7 more bits per coordinate, about
4.7 bits/coordinate total, about **7990 bits/token/boundary or 6.5x the incumbent's wire
budget**. For a pipeline whose entire premise is that stage boundaries cross the ordinary
internet, that is a nonstarter. This is an elasticity extrapolation from two anchor points,
not a fit, and it should be treated as an order-of-magnitude argument. It makes a
falsifiable prediction, which is the point: see section 5.

**What it did not buy, and why that is the finding.** The gap barely moved and the drift
got worse. The sharpest reading is this: **PRF exact-k and 1-bit SR are BOTH unbiased**
(E[q] = h for each), so unbiasedness is not what separates them. The only thing that
separates them is noise magnitude and tail shape, and a1 has 2.7x less noise energy with
bounded tails and still drifts 2.6x faster and sits at a statistically indistinguishable
gap level. Within the unbiased family, lower noise energy did not reduce mismatch; it
moved the wrong way. That is consistent with the standing finding that bias times
coherence gates reference KL rather than noise magnitude, and it means the "make the
estimator cleaner at the source" branch of the program has now been probed once and
returned a null with a cost penalty.

**The one genuine positive, new at 120 and not visible at the step-76 pre-read.** a1's gap
is **decelerating**, not accelerating. The quadratic fit over steps 1 to 120 has a negative
curvature term (**-0.00003792 x²**) and the half-slopes fall from **+0.008805/step**
(steps 2 to 60) to **+0.003265/step** (steps 61 to 120), with the gate window at
**+0.002825/step**: a 2.7x deceleration. The pre-read's "widening and accelerating"
conclusion was correct for the data it had and is superseded here. Do not overread it: the
level is plateauing in the **13.8 to 14.0** band, not settling below the baseline, and
+0.002825/step is still about **2.2x** the baseline's whole-run total-view gap creep
(13.88 to 14.66 over 600 steps, about +0.0013/step). a1's decomposition into sampler
sharpening versus training-view components is not available in the run dir, so the
baseline's training-view-only +0.000467/step figure is deliberately not used as the
comparator here. Read a1's gap as "settles at the wrong level", which fails program
success criterion 1 on the `< 3` nats clause and on the E[rho] > 0.05 clause both.

---

## 3. Metrics summary and baseline comparison

| metric | a1 gate window (100 to 120, n=21) | a1 full run (1 to 120) | baseline #90 | source |
|---|---|---|---|---|
| `actor/kl_loss` | mean 2.251837, last 2.304283, slope +0.003887 (hac_se 0.000251) | 0.0 to 2.304283, slope +0.003581 over steps 2 to 120 | 0.156 to 0.203 at matched step, about +0.0015/step, reaching 0.899 at 600 | analysis-a1.log |
| `rollout_corr/kl` | mean 13.751145, slope +0.002825 (hac_se 0.000219, r2 0.773) | 13.182191 to 13.774802, slope +0.006583, curvature -3.79e-5 | 14.24 at gate, 13.88 to 14.66 over 600 | analysis-a1.log |
| `rollout_corr/k3_kl` | mean 12.756624, slope +0.002311 | - | - | analysis-a1.log |
| E[rho] | mean 0.005479, median 0.002527, last 0.002017, max 0.027912 | - | 0.0014 (0.052 to 0.0003 over the run) | analysis-a1.log |
| `critic/rewards/mean` | mean 0.652948, slope -0.001074 (ols_se 0.001003) | 0.366211 to 0.631836, slope +0.003235 (ols_se 0.000125, hac_se 0.000210), max 0.735352 at step 107 | slope 0.0032/step; val plateau 0.661@150, 0.663@300, 0.673@450, 0.661@600 | analysis-a1.log, issue section 1 |
| `actor/ppo_kl` | exactly 0.0 | exactly 0.0 at all 120 steps | about 0 | analysis-a1.log |
| `actor/entropy` | mean 7.934806, slope +0.000309 | 7.899606 to 7.938554, slope +0.000365, r2 0.986 | - | analysis-a1.log |
| `actor/grad_norm` | mean 0.648968 | max 0.898240 at step 11, slope -0.001545 | - | analysis-a1.log |
| `actor/kl_coef` | 0.001 | 0.001 at all 120 steps | 0.001 | analysis-a1.log |
| wire budget | 2304 bits = 288 bytes/token/boundary | - | 1232 bits | analysis-a1.log |
| `response_length/mean` | - | 748.4 at step 1, 659.9 at step 120 | - | analysis-a1.log |

No divergence signature anywhere: zero NaN or Inf across 120 steps and every logged
metric, grad_norm peaks at 0.898 at step 11 and declines, entropy rises slightly, and the
KL coefficient never moves off 0.001 (the controller is off in round A by design).

---

## 4. The reward-slope tension, treated explicitly

The gate asks for a training reward slope at or above 0.00288/step. a1's full-run slope is
**+0.003235/step** and its gate-window slope is **-0.001074/step**. Both are real. Here is
what each can and cannot support.

**The full-run pass is real but has no margin.** +0.003235 is **1.011x** the baseline's
0.0032 and **1.123x** the 0.00288 bar. Under iid OLS (se 0.000125) the test against the
bar gives t = +2.84. Under Newey-West HAC (se 0.000210), which is the right standard error
for an autocorrelated training curve, t = **+1.69**, i.e. above the bar on the point
estimate but not significantly above it. Scored PASS on the criterion as literally written,
with no claim of a comfortable margin.

**The negative window slope is NOT a turnover finding, and 21 points cannot make it one.**
Window residual SD is 0.02782 on a metric whose level is 0.653, and `sxx` for 21
consecutive steps is 770, giving se 0.001003. Consequences:

- t against zero is **-1.07**, p about 0.30. The 95 percent CI is
  **[-0.00317, +0.00102]** and straddles zero comfortably.
- A moving-block bootstrap (L=4, B=20000) puts P(fitted slope < 0) at **0.881**, i.e. about
  a 1-in-8 chance of a positive fit from the same process. Not a finding.
- The minimum detectable slope at 80 percent power and alpha 0.05 is **0.00281**, which is
  essentially the size of the parity bar itself. **A 21-point window on this metric has no
  power to resolve anything smaller than the bar it is being tested against.** Any verdict
  that leaned on the window sign would be reading noise.

Read the window sign as a **caution, not a finding**.

**But there is a well-powered claim in the neighbourhood, and it is the substantive one.**
The reward slope decays monotonically across the run, and the decay is decisive on windows
that do have power (`analysis-a1.log` SEGMENT SLOPES):

| window | n | slope | t vs 0 | t vs the 0.00288 bar |
|---|---|---|---|---|
| 1 to 30 | 30 | +0.000210 | +0.40 | -5.08 |
| 31 to 60 | 30 | **+0.007261** | +12.03 | +7.26 |
| 61 to 90 | 30 | +0.001405 | +2.10 | -2.20 |
| 91 to 120 | 30 | +0.000762 | +1.06 | -2.93 |
| **61 to 120** | **60** | **+0.001077** | **+4.44** | **-7.43** |
| 100 to 120 | 21 | -0.001074 | -1.07 | -3.94 |

Over the entire second half, n = 60 and well powered, a1's reward slope is **+0.001077,
significantly positive and 2.7x BELOW the parity bar at t = -7.43**. The full-run pass is
carried almost entirely by the steps 31 to 60 learning burst at +0.007261. So the honest
statement is not "reward turned over in the last 21 steps" but **"reward is still
improving and has decelerated to about a third of the parity bar by the gate window, and
the criterion passes only because it is computed over the full run"**. That is a
criterion-definition artifact, not noise, and it is load-bearing for how round A picks a
winner. It is next_action 1.

**A mitigating read the operator should weigh.** The window reward level is 0.652948
against 0.352332 over steps 1 to 20, a 1.85x improvement, with a run maximum of 0.735352
at step 107. The baseline's own validation accuracy plateaued at 0.661, 0.663, 0.673, 0.661
across steps 150 to 600 (issue section 1), so MATH at this model scale saturates near 0.66
to 0.67 and a decaying train-reward slope at 0.65 is what saturation looks like. Entropy is
flat to rising (+0.000309/step in the window) rather than collapsing, and grad_norm is
falling. The deceleration is saturation-shaped, not damage-shaped. This is why it is a
caution rather than a STOP trigger.

---

## 5. next_actions

```yaml
next_actions:
  - knob: round-A reward-slope criterion window
    from: "full-run OLS slope of critic/rewards/mean, window unstated"
    to: "declare the window: report full run AND steps 61-120 with HAC standard errors, and require both for a reward PASS"
    rationale: "a1 passes the bar at 1.123x on the full run (HAC t=+1.69 vs the bar) and fails it at t=-7.43 on the well-powered second half. The criterion's window, not the data, currently decides the verdict. Fix it before it picks a winner. The 21-point gate window itself must never be used alone: its minimum detectable slope, 0.00281, is the size of the bar."
  - knob: a2 kill-gate reference slope and window
    from: "kill a2 at step 60 if its reference-KL slope is >= 2x a1's (window unspecified)"
    to: "fit a2 over steps 2-60 against a1's matched steps 2-60 slope of +0.002707/step; threshold 0.005414/step; confident kill above 0.0063, confident acquittal below 0.0045, inside the band run to 120 and decide on reward slope plus gap corroboration"
    rationale: "confirmed on the full 120-step log: a1's slope varies 2.3x by window (+0.001743 over 2-39, +0.004592 over 40-77, +0.003887 at the gate). An unwindowed 2x rule is not well posed, and using a1's full-run +0.004349 would set a 26 percent more permissive gate that lets a worse a2 survive. Already pre-registered in PROGRAM_STATE.md; this verdict supplies the final matched number."
  - knob: which arm the program bets on for gap-at-source
    from: "precision allocation (a1, a3) is the primary gap-at-source route"
    to: "treat a4 (PRF exact-k + CVC) and a5 (FRLR + token-IS) as the primary route; demote a3 to a byte-parity control with a pre-registered prediction of about 13.6 nats"
    rationale: "the measured noise-energy elasticity, 0.494 nats per e-fold, means reaching the sub-10 gate by precision alone needs about 4.7 bits/coordinate, about 6.5x the incumbent's wire budget, which contradicts the deployment premise. Coherence and correction, not quantizer quality, are the only remaining levers with a path to the gate."
```

### What each remaining round-A arm must now establish

- **a2 `srq-b1-rn` (running).** The mechanism call, and a1's result has sharpened it. a1 and
  the incumbent are BOTH unbiased, so a1's 2.6x worse drift cannot be attributed to bias
  and unbiasedness alone does not predict drift. a2 (round-to-nearest: biased, coherent,
  lower MSE) is now the decisive discriminator. Fit steps 2 to 60, threshold 0.005414. If
  a2's slope lands **below 0.0045** the SR/unbiasedness mechanism model is falsified and the
  pre-registered `a1-prime` contingency (sr_quant + PRF Hadamard pre-rotation) fires as an
  in-round REVISE. If **above 0.0063**, confident kill at step 60, SR upheld, carry the
  better of a1 and a3.
- **a3 `srq-parity-k493`.** The fair fight at 1233 bits, one bit above the incumbent's 1232.
  The elasticity above predicts **13.58 nats** for its 5 ‖h‖² noise energy. If a3 lands at
  or below 13.751 at parity bytes it **dominates a1 outright** on the winner rule and makes
  a1 irrelevant to round B. If it lands near 14.2, precision allocation is dead as a route
  and the elasticity model is confirmed on a third point.
- **a4 `prf-exactk-cvc-ce`.** The only arm whose success criterion is a negative gap slope.
  a1 has now raised its bar: an untreated codec already decelerates from +0.008805 to
  +0.002825/step on its own, so a4 must beat "decelerating toward +0.0028", not "flat", and
  a merely-decelerating a4 must not be scored as CVC working. Watch the uniformization
  guard (rollout perplexity, reward slope, val proxy).
- **a5 `frlr-r48k28-tis`.** The only arm that can move E[rho] off the floor. a1's 0.005479 is
  **9.1x below** the program's 0.05 importance-sampling safety-net threshold and about 180x
  below 1.0, so token-IS is numerically dead on a1 and dead on the incumbent. If a5 lands
  E[rho] in [0.2, 2] as issue 89 suggests, it is the only configuration that keeps the
  standing token-IS fix available at all, which weighs heavily beyond its gap number.

### What would make the round-A winner rule land on something other than a1

The rule is: lowest train-inference gap **subject to** reference KL at or below baseline and
reward-slope parity, tie-break on higher E[rho]. Under the literal rule **a1 cannot be the
winner**: its reference-KL slope is 2.59x the baseline, so it fails the subject-to clause,
and its absolute reference KL is unscorable. a1 becomes the winner only by default, if every
other arm fails the same clause and the tie is then broken on the gap. Concretely, a1 loses to:

- any arm with reference-KL slope at or below +0.0015/step and gap at or below 13.751;
- **a3 at 1233 bits with a gap at or below 13.751**, which additionally satisfies the
  final winner's "wire budget at or below incumbent" requirement (issue section 7 item 4)
  that a1 structurally cannot satisfy at 1.8701x;
- **a5 with E[rho] in [0.2, 2]** even at an equal gap, on the pre-registered tie-break;
- **a4 with a negative and stabilised gap slope**, which satisfies program success criterion
  1's "settles" clause that a1 misses.

a1's honest role in round B is as the A1 half of the A1-versus-A2 mechanism comparison and
as the elasticity anchor, not as a carry-forward configuration.

---

## 6. Risk to the program (needs an operator decision at the A-to-B boundary)

The stage-1 gate is "at least one arm clears the section-1 gate at steps 100 to 120 AND the
decision tree names a single winner", with `on_fail: stop`. The section-1 gate contains
`rollout_corr/kl < 10` nats. **The incumbent baseline itself sits at 14.24, so the gate was
never calibrated as a beat-the-incumbent test; it is an absolute target that the current
best-known configuration also fails.** a1 lands at 13.751 and the noise-energy elasticity
predicts about 13.6 for a3 and about 12.7 for a hypothetical full 2-bit arm. If a4 and a5
also land in the 13 to 14 band, then on the strict reading **round A fails, `on_fail: stop`
halts the program at stage 1, and rounds B and C never run, even though the winner rule
would happily name a relative winner** (lowest gap subject to the clauses). Two defensible
readings of the same pre-registered text point in opposite directions, and roughly 26.8
GPU-h of rounds B and C on a $3.344/h box hang on which one is applied.

I am not resolving that here and this verdict does not reinterpret the threshold in either
direction. `PLAN_OF_EXECUTION.md` section 4 already pre-commits to the right handling:
compute the full five-arm gate table, post it, and flag `needs:human` with the cost of each
option, with nothing launching while the flag is up. This verdict endorses that and adds one
constraint: **the A-to-B boundary is the decision point, not the a1 verdict and not a later
convenience.** If a4 or a5 clears sub-10, the tension evaporates and nothing is needed. If
neither does, an operator decision is required before any round-B spend, because a unilateral
"the gate obviously meant relative to the incumbent" reading would retroactively rewrite the
program's own falsification condition, and that is precisely the move a pre-registered plan
exists to prevent.

---

## Notes

1. **`RESOLVED_CONFIG_MISSING`.** `capture_resolved_config.py` could not run: there is no
   `train.log` in the run dir, and the synced `metrics/incoming.log` contains no launcher
   `set -x` trace of `python3 -m verl.trainer.main_ppo`, so no expanded command line exists
   to recover. `resolved_params.txt` was written from the hydra config echo that IS present,
   and it marks clearly which knobs are verified and which are not. Verified from the run's
   own output: `experiment_name a1-srq-b1-sr`, `project_name 93-long-horizon-stability`,
   `total_training_steps 120`, `save_freq -1`, `test_freq -1`, `val_before_train False`,
   rollout `prompt_length 1024` / `response_length 2048`. **Not** verifiable from the run
   dir: `sr_quant bits/block/rounding`, `comm_eff.pp_size 8`, `train_batch_size 128`,
   `ppo_mini_batch_size 128`, anchor cadence/delay, signed-EMA betas, LR. Circumstantial
   support does exist for two of them: `logical_pp_bits_sr_quant` = 2304 is exactly
   1536 payload bits plus 48 bf16 block scales, which is 1-bit sr_quant at block 32 over
   H = 1536; and `anchor_replay_fires` = 6 = floor(120/20) requires one optimizer tick per
   global step, i.e. `train_batch_size` equal to `ppo_mini_batch_size`. **No plan-versus-ran
   divergence was found in anything that is verifiable.** One near-miss worth recording so
   nobody re-raises it: the config echo also contains `pipeline_model_parallel_size: 1` and
   `prompt_length: 2048` inside a block whose `name` is `'???'`, which is an unset megatron
   stub, inert under FSDP. The live vLLM rollout block is the 1024/2048 one. That is not a
   pp_size divergence.
2. **Log hygiene defect, found while parsing, worth fixing before a3.**
   `metrics/incoming.log` is a live tail-resync of the shared `/workspace/train.log`
   heartbeat symlink, and it now carries rows from **two** cells: a1 (`TaskRunner pid=6382`,
   1070 metric lines, 120 unique steps, zero gaps) and **a2 (`pid=93602`, 2 metric lines,
   steps 1 to 2)**. a2's step-1 row (reward 0.3525390625, gap 11.496414184570312) is a
   different run from a1's step-1 row (reward 0.3662109375, gap 13.182190895080566), so any
   parse that keys on `global_step` alone will silently substitute a2's values for a1's early
   steps, and the contamination grows as a2 advances. It also defeats a naive
   duplicate-consistency check, because reference KL is exactly 0.0 at step 1 in both runs.
   Every number in this verdict is pid-filtered to a1. Recommend per-cell metric sync targets
   for a3 onward.
3. **Completion evidence.** No `done.flag` exists in the run dir. Completion was established
   from the artifacts instead: 120 of 120 unique steps present with zero gaps, zero NaN and
   zero Inf, matching `run.json` `cells[0].steps` = 120; `PROGRAM_STATE.md` records
   WandB `state=finished` and a 3h58m39s runtime at 119.33 s/step; and a2 is already running
   on the same one-cell-at-a-time box, which requires a1 to have gone terminal. The 5 error
   markers recorded at teardown are the known benign shutdown-path noise (DataLoader worker
   killed at teardown, WandB `teardown_atexit` BrokenPipeError) and are the documented
   final-step WandB drop hazard, not a training failure.
4. **Superseded pre-read claims, stated so the record is clean.** `preread-a1.md` concluded
   at step 76 that the gap was "widening, accelerating" and projected about 14.03 at step 120.
   The full run shows the gap **decelerating** (negative curvature, half-slopes +0.008805 to
   +0.003265) and landing at 13.775, below the projection. The pre-read's reference-KL
   acceleration finding also stops accelerating after about step 77 and levels near
   +0.0037 to +0.0039/step. The pre-read's central falsification (that the 1.9 level is not a
   flat view-noise floor) stands and is reconfirmed here.
5. **Budget and depth.** Ledger `93-long-horizon-stability` shows `max_gpu_hr` 100 (raised
   from 44 by the operator) with about 4.1 h consumed by a1, and `run.json` `iterations` = 7
   with no recorded `revise_depth`, so REVISE is comfortably in budget on both axes.
6. **Verification commands.** Run offline against the run dir rather than by re-pulling
   WandB, since this project has a documented late-step sync-drop hazard and the on-box log
   is the preferred source. Full stdout in `runs/93-long-horizon-stability/analysis-a1.log`.
   No GPU box was touched and no new experiment was run.
