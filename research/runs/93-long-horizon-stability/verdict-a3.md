# Verdict, issue 93 round A, cell a3-srq-parity-k493

VERDICT: REVISE

WandB `k8dvru5l`, project `93-long-horizon-stability`, entity `shamanework-pl`, state
`finished`, 120 of 120 steps, no missing rows. Judged against
`runs/93-long-horizon-stability/AB_AMENDMENT.md`, whose every threshold was committed
blind while a3 was at step 20 to 25. All numbers below are reproduced in
`runs/93-long-horizon-stability/analysis-a3.log` and in the JSON artifacts under
`runs/93-long-horizon-stability/metrics/`.

**Why REVISE and not PASS.** a3 clears every safety veto (V1, V2, V3), and it is the
round-A leader under the committed lexicographic rule because a1 fails V1 and a2 fails
V1 and V2. But the committed objective is failed on both of its clauses: the gap slope
is +0.001300/step against the S-bar pass line of +5.0e-4 (2.60x, and 4.98 HAC standard
errors clear of it, so this is not a noise call), and the gap level is 14.9924 against
the cap of 14.2458 (over by 0.7466 nats). Leading a field on vetoes is not the same as
meeting the objective, and the verdict must not launder one into the other.

**Why not STOP.** a4 is training now and round A continues by design. Nothing here is
falsified in a way that closes the program: a3 is the first *new* arm in round A to
survive the veto set, and its two slope results are statistically real improvements over
the incumbent. Decision-ladder rows 1 through 5 all remain reachable on a4 and a5.

**Why not PASS on the negative either.** The amendment registers a clean negative as a
form of PASS, but only at ladder row 6 ("no arm clears the drift veto, or every arm is
rising"). a3 clears the drift veto, so row 6 is not triggered and the negative is not
yet clean.

---

## 1. Committed criteria, per item

Comparator column is the incumbent `90-prf-exactk-600` (`woqs8zra`), refit on the
identical windows through the identical OLS plus Newey-West(3) code path, so every ratio
is method-matched as well as window-matched.

| # | Committed criterion (AB_AMENDMENT section 4) | Bar | a3 observed | Result | Source |
|---|---|---|---|---|---|
| E1 | reached step 120 | 120 | 120 of 120 rows, state `finished` | ✓ | `metrics/a3_gate.json` |
| E2 | wire budget for round-C promotion | <= 1232 bits/tok/boundary | **1232.5** | **✗ by 0.5 bits (0.041%)** | `actor/comm_eff/logical_pp_bits_sr_quant`, `metrics/a3_local.json` |
| E3 | `actor/ppo_kl` == 0 or explained | 0 | max abs = 0 over n=120, explained (`ppo_mini` == `train_batch`, one tick per step) | ✓ | `analysis-a3.log` S2 |
| E4 | confinement counters non-degenerate | non-degenerate | mask_applications 78176, spectral_corrections 34138, anchor_replay_fires 6 = floor(120/20), rank1_bypass_ticks 19 | ✓ | `analysis-a3.log` S10 |
| V1 | `actor/kl_loss` slope, 100-120, slope only | <= 3.264e-3 | **+0.001616** (hac se 1.32e-4), 0.743x incumbent's +0.002176, 0.495 of the ceiling | **✓** | `analysis-a3.log` S3 |
| V2a | `actor/grad_norm` run max | <= 10.0 | **7.283** at step 14 (72.8% of the bar) | ✓ | S3 |
| V2b | `actor/grad_norm` window mean, 61-120 | <= 3.62 | **2.891962** (n=60, 79.9% of the bar), 1.750x incumbent | ✓ | S3 |
| V3a | `critic/score/mean` level, 100-120 | >= 0.6248 | **0.656855**, 0.9987x incumbent's 0.657692 | ✓ | S3 |
| V3b | significantly positive 61-120 reward slope | t > 0 | +0.001236, hac se 1.84e-4, t = +6.72 | ✓ | S3, S4 |
| O1 | gap slope 61-120 (primary objective) | <= +5.0e-4 | **+0.001300** (hac se 1.61e-4, n=60) = 2.60x bar, +4.98 se above it | **✗** | S3 |
| O2 | gap level 100-120 (secondary objective) | <= 14.2458 | **14.992396** (n=21), over by +0.7466 nats | **✗** | S3 |
| T | E[rho] tie-break (bonus, never a veto) | higher is better | 0.002661 mean / 0.001652 median vs incumbent 0.005028 / 0.002057 = 0.53x | loses the bonus | S3 |

Tally: **E 3 of 4, V1 pass, V2 pass, V3 pass, O 0 of 2, T lost.**

Cross-checked against the plan's own verification command run unmodified,
`scripts/gate93.py --run k8dvru5l --gate-lo 100 --gate-hi 120`, whose literal five-flag
table is `ref_kl_le_baseline False, gap_lt_10 False, gap_lt_3_target False,
reward_slope_parity True, ppo_kl_zero True, e_rho_gt_0p05 False`. Both readings are
published side by side per the amendment's section 5 instruction. The script's own note
records why the absolute `ref_kl` flag is not scoreable on an sr_quant arm.

**Round-A ranking under the committed rule, unchanged by this verdict:** a3 > a1 > a2.
a3 outranks a1 because vetoes are hard filters applied before the objective and a1 fails
V1 at 1.79x plus E2 at 1.87x, even though a1's gap level (13.7511) is better than a3's.
That ordering is the rule working as designed, not a quirk to be apologised for.

---

## 2. A pre-registered prediction failed. Saying so plainly.

`verdict-a1.md` (lines 104 to 112 and 255 to 258) derived an elasticity of **0.494 nats
of gap per e-fold of noise energy** from two anchor points, the incumbent at 19 ||h||^2
mapping to 14.24 nats and a1 at 7 ||h||^2 mapping to 13.751 nats, and predicted publicly,
before a3 produced a step, that a3 at 5 ||h||^2 would land near **13.58 nats**, adding
"if a3 lands at 13.6 the elasticity model is confirmed on a third point".

**a3 landed at 14.9924. The model missed by +1.4124 nats in the pessimistic direction.
The elasticity is falsified as a cross-codec predictor of gap level, and I am recording
that as a failed prediction of mine rather than a surprise about a3.**

Three quantifications of how badly it failed, so the falsification cannot be softened
later:

1. **It missed by 2.86x the entire range it was fitted on.** The two anchors span
   14.2458 minus 13.7511 = 0.4947 nats. The error is 1.4124 nats.
2. **It failed on sign, not only magnitude.** a3's noise energy (5) is 3.8x *lower* than
   the incumbent's (19), so the model requires a3's gap to sit *below* the incumbent's.
   It sits **+0.7466 nats above** it. A one-parameter monotone model that inverts the
   ordering of its own two calibration regimes has no residual predictive content on
   this axis.
3. **It was already dead at step 1, before a single optimizer step.** a3's step-1 gap is
   **14.5284**, which is +0.9484 above the prediction and +1.3462 above a1's own step-1
   gap of 13.1822. No training dynamics could have redeemed it. The prediction failed on
   the intercept, not on the trajectory.

### Why it failed, at the level I can actually support

The mechanism is legible from the source and does not require speculation. From
`verl/workers/comm_eff/activation_quant.py` (subset-mode docstring, lines 38 to 49) and
`verl/workers/comm_eff/activation_mask.py` (line 265, `keep = round((1-p)*H)`), plus the
two launchers:

| arm | mask (coordinate-aligned deletion) | quantization (within-coordinate) | unbiased? |
|---|---|---|---|
| a1 `srq-b1-sr` | **none**, all 1536 channels sent | 1-bit blockwise SR on 100% of channels | yes, `E[SR(h)] = h` |
| a3 `srq-parity-k493` | **exact-k keep 493 of 1536** (67.90% zeroed), rescale H/k = 3.1156 | 2-bit blockwise SR on the kept 32.10% | yes, through both sources |
| incumbent `prf-exactk` | **exact-k keep 77 of 1536** (94.99% zeroed), rescale 1/(1-p) = 20 | **none**, kept values ride at full bf16 | yes, inverted-dropout |

So the answer to "do a3's dropped coordinates behave like a mask" is stronger than
"behave like": **they are literally the same mask primitive**. a3's subset J is drawn by
`prf_token_mask(..., exact_k=True, exact_keep=493)`, the same order statistic on the same
splitmix64 PRF hash that the incumbent's p=0.95 mask uses, keyed identically so J is
bit-identical across the old-logprob, train and ref passes of one step. a3 is not a
lower-noise-energy variant of a1 at all. **a3 is a member of the incumbent's family**, an
exact-k mask with a 6.4x larger keep set, composed with an 8x coarser per-coordinate
quantizer on what it keeps.

That makes the fit's failure structural rather than unlucky. Writing the per-coordinate
variance of each unbiased estimator as a deletion term plus a quantization term:

- deletion term = `(H/k - 1) * h_j^2`: **18.948** for the incumbent, **2.116** for a3,
  **0** for a1.
- quantization term = `(H/k) * Var[SR_b(h_j)]`: **0** for the incumbent (bf16 is exact),
  `1 x Var[SR_1]` for a1, `3.1156 x Var[SR_2]` for a3.

**The two anchor points lie on orthogonal axes.** The incumbent's error variance is 100%
deletion and 0% quantization; a1's is 0% deletion and 100% quantization. A single scalar
"noise energy" collapses both onto one number, and a one-parameter elasticity fitted
through two points on orthogonal axes has no basis for extrapolating to a *mixture* of
them. a3 is the only arm in the program carrying both terms at once, so it was precisely
the point the fit could not reach. The failure was available a priori and I did not see
it.

### What I deliberately do not claim

- **I cannot claim a coherence mechanism here.** Both of a3's error components are
  PRF-keyed on `(base_seed, layer_idx, global_step, sample_id, position_id, channel,
  direction)` and therefore pass-invariant within a step, exactly as a1's and the
  incumbent's are. That is why `actor/ppo_kl` is identically 0 in all of them. So
  within-step coherence does not distinguish these three arms, and the a1/a2 finding it
  came from was about **bias** (round-to-nearest versus stochastic rounding), which a3
  does not have: a3 is unbiased through both randomness sources. Importing "coherence
  gates drift" here would be a category error on my part.
- **I cannot fit level to keep fraction either.** Ordered by fraction of channels
  deleted, 0% (a1) gives 13.751, 67.9% (a3) gives 14.992, 95.0% (incumbent) gives 14.246.
  Non-monotone. With three arms and three axes in play (keep fraction, per-coordinate
  precision, rescale gain) the level ordering is **unidentifiable** from the runs in hand.
  I am not going to fit a second one-parameter model to three points and repeat the error.
- **The honest general lesson is narrower and firmer:** gap **level** carries a large
  per-codec measurement offset with no predictor yet established, so level is not a
  legitimate target for cross-codec extrapolation. The four step-1 gaps, measured before
  any optimizer step and therefore pure instrument, span **3.0319 nats** (a2 11.4964, a1
  13.1822, incumbent 13.8794, a3 14.5284). The amendment had already established this for
  `actor/kl_loss` (a1's fitted view offset 1.8765) and used it to justify judging V1 on
  slope only. The a1 verdict then extrapolated a *level* anyway, on the gap axis, which is
  the same error the amendment had just diagnosed one metric over.

**Cheapest discriminator, if the operator wants the axes separated.** A 1-bit subset arm
at `subset_k=821` costs `821*1 + 821*16/32 = 1231.5` bits, inside the 1232 bar, and moves
along the precision axis at fixed budget. A 2-bit full-width arm would cost 3840 bits
(3.12x budget) and is a diagnostic only, not a candidate. Neither is authorized by any
row of the decision ladder; both are listed as proposals in section 8.

---

## 3. The positive result, at its true strength and no more

**The claim I will defend.** At parity wire budget, a3 beats the incumbent on both slope
axes, and both differences are statistically real, not point-estimate artifacts:

| axis | a3 | incumbent | ratio | z on the difference |
|---|---|---|---|---|
| `actor/kl_loss` slope, 100-120 | +0.001616 (se 1.32e-4) | +0.002176 (se 1.46e-4) | **0.743x** | **-2.85** |
| `rollout_corr/kl` slope, 61-120 | +0.001300 (se 1.61e-4) | +0.001867 (se 1.67e-4) | **0.696x** | **-2.44** |

It does this while passing all three vetoes, at 1232.5 versus 1232 bits, and at reward
parity (section 4). It is the first *new* arm in round A to survive the veto set. That is
a genuine result and the best compressed configuration this program has measured on the
registered "settles" form of the criterion.

**Four things that cap how strongly it can be stated. All four are load-bearing.**

**(a) a3 does not settle. It creeps more slowly.** The registered language is "slope
approaching 0 versus the baseline's +0.0005/step creep". a3 is at +0.001300, which is
2.60x that line and 4.98 HAC standard errors above it. "0.70x the incumbent's creep" and
"settles" are different claims and only the first is supported.

**(b) On totals over the measured window, a3 is worse than the incumbent on both axes.**
Removing each arm's own instrument offset and reading what accumulated over 120 steps:

| arm | gap step 1 | gap 100-120 | gap accumulated | vs incumbent | kl_loss accumulated | vs incumbent |
|---|---|---|---|---|---|---|
| a1 | 13.1822 | 13.7511 | +0.5690 | 1.553x | +0.3423 | 2.328x |
| **a3** | **14.5284** | **14.9924** | **+0.4640** | **1.267x** | **+0.1657** | **1.127x** |
| incumbent | 13.8794 | 14.2458 | +0.3664 | 1.000x | +0.1470 | 1.000x |

So a3 accumulated **1.267x** the incumbent's gap growth and **1.127x** its drift growth
over the same 120 steps. a3's advantage is confined to the **terminal derivative**: its
growth is front-loaded and decelerating (gap quadratic term -4.857e-5, `analysis-a3.log`
section from `metrics/a3_local.json` curvature block), so the last 60 steps look better
than the whole run does. That is a real and possibly deployment-relevant property, since
what matters over a 600-step horizon is the late-time slope, but "0.70x the incumbent"
must never be quoted without "on the 61-120 derivative, against 1.27x on the total".

**(c) The gap level is the worst of any surviving arm, and 86.9% of that is instrument.**
a3's level excess over the incumbent at 100-120 is +0.7466 nats, of which **+0.6489 nats
(86.9%) is already present at step 1**, where the actor weights are still the base
checkpoint and `actor/kl_loss` is 0 in every arm. Only +0.0977 nats of the excess was
accumulated by training. This is a substantive finding about the committed O2 constraint:
it is measuring mostly codec-view offset, not drift. **It is not an excuse, and I am not
applying it.** O2 was committed blind, a3 fails it, and the amendment states that any
threshold changed after the data exists is illegitimate. If the operator judges that the
deployment-relevant quantity is the drift the codec *causes* rather than the offset its
measurement view *imposes*, that is a ratification decision, it must be applied
symmetrically to all five arms, and it belongs to the operator, not to this verdict.

**(d) The horizon test, which is the direct answer to "does this advance the goal".**
Extrapolating each arm on its own measured 61-120 slope from its 100-120 level:

| step | a3 gap | incumbent gap | a3 minus incumbent |
|---|---|---|---|
| 110 (measured) | 14.9924 | 14.2458 | +0.7466 |
| 200 | 15.1094 | 14.4138 | +0.6956 |
| 400 | 15.3694 | 14.7872 | +0.5822 |
| **600** | **15.6294** [95% CI 15.4751, 15.7837] | **15.1606** [95% CI 14.9999, 15.3212] | **+0.4689** |
| crossover | | | **step 1427** |

**So the answer is no, not yet, and not within the horizon the program registered.**
a3's level penalty is +0.7466 nats and its slope credit is +0.000567 nats/step, so it
needs 1317 further steps to break even and first falls below the incumbent's gap at about
**step 1427**, versus a registered horizon of 600. At 600 a3 is still 0.4689 nats worse
than the incumbent. And a3's projected gap(600) of 15.6294 sits 1.1794 nats outside the
about-14.45 bound that the amendment attached to S-bar as its horizon meaning. A gap
that "settles" is the right criterion and a3 moves the right derivative in the right
direction, but a 0.70x slope purchased at a +0.75 nat level does not pay for itself
inside 600 steps. Both the derivative and the level have to move.

---

## 4. The reward tension, resolved: parity, and resolvable

`gate93.py` reports the tension the dispatch names: full-run slope **+0.00322** clears
the 0.00288 parity bar while the gate-window slope is **-0.00010**. Full window table,
HAC(3) standard errors throughout:

| window | n | level | slope | hac se | t vs 0 | t vs 0.00288 | MDS95 |
|---|---|---|---|---|---|---|---|
| 100-120 | 21 | 0.656855 | -0.000099 | 4.37e-4 | -0.23 | **-6.82** | 8.6e-4 |
| 61-120 | 60 | 0.631543 | +0.001236 | 1.84e-4 | +6.72 | -8.94 | 3.6e-4 |
| 21-120 | 100 | 0.567012 | +0.003048 | 3.09e-4 | +9.88 | **+0.54** | 6.1e-4 |
| 1-120 | 120 | 0.531934 | +0.003217 | 1.96e-4 | +16.38 | **+1.71** | 3.9e-4 |

**a3's case is not a1's, and I am not reusing a1's argument.** For a1 the n=21 window was
genuinely underpowered (minimum detectable slope about 0.00281, essentially the bar
itself). a3's gate window is 1.45x tighter (iid se 6.90e-4 versus a1's 1.00e-3), so its
MDS95 is about 8.6e-4 and it **is** powered to reject 0.00288, and it does. The
resolution therefore rests on a different and better-supported fact: **at 100-120 the
reward has saturated in every reference run, including the uncompressed control.**
Incumbent -0.001422, dense -0.000725, a1 -0.001074, a3 -0.000099. A positive-slope bar is
sign-degenerate on that window, which is exactly the measured reason the amendment
committed V3 on level. Reading -0.00010 as a learning failure would condemn the dense
control too.

**Matched against the incumbent on identical windows, a3's learning is at parity, and the
question is resolvable rather than unresolvable:**

| window | a3 | incumbent | difference | z | ratio |
|---|---|---|---|---|---|
| 1-120 | +0.003217 | +0.003212 | +0.000004 | +0.02 | 1.001x |
| 21-120 | +0.003048 | +0.003139 | -0.000091 | -0.22 | 0.971x |
| 61-120 | +0.001236 | +0.001203 | +0.000033 | +0.12 | 1.027x |
| 100-120 | -0.000099 | -0.001422 | +0.001323 | +1.90 | (both near zero) |
| level 100-120 | 0.656855 | 0.657692 | -0.000837 | | **0.9987x** |

Every slope difference is inside 0.25 combined standard errors except the gate window,
where a3 is the *less* negative of the two (z = +1.90, which is not significant at 0.05
and which I will not promote to a win). a3's full-run slope clears the 0.00288 bar by
+1.71 HAC se, statistically the same margin as the incumbent's +1.83. Level story and
slope story therefore agree here rather than diverging: 0.9987x on level, 1.001x on
full-run slope.

**Answer: genuinely at parity. Not sub-parity, not unresolvable.** With one seed the
residual uncertainty is about 0.0004/step on the well-powered windows, so a real learning
deficit larger than roughly 12% of the incumbent's full-run slope would have been visible
and is excluded. Smaller deficits are not excluded, and no capability measurement exists
(val off, no checkpoints, so none can be added retrospectively).

---

## 5. Metrics summary and baseline comparison

Matched windows, matched code path. Dense column from `AB_AMENDMENT.md` section 4.

| quantity | window | a3 | a1 | incumbent | dense |
|---|---|---|---|---|---|
| ref-KL slope (V1) | 100-120 | **+0.001616** (0.743x) | +0.003887 (1.79x) | +0.002176 (1.00x) | 1.64e-5 |
| ref-KL level | 100-120 | 0.911128 | 2.251837 | 0.178583 | |
| gap slope (O1) | 61-120 | **+0.001300** (0.696x) | +0.003265 | +0.001867 | about -1.2e-6 |
| gap level (O2) | 100-120 | **14.992396** | 13.751145 | 14.245787 | 2.42e-4 |
| gap step 1 (pure instrument) | 1 | 14.5284 | 13.1822 | 13.8794 | |
| grad_norm run max / 61+ mean | | 7.283 / 2.892 | 0.898 / 0.649 | 4.645 / 1.652 | 0.054 / 0.066 |
| `critic/score/mean` level | 100-120 | 0.656855 | 0.652948 | 0.657692 | 0.6587 |
| reward slope | 1-120 | +0.003217 | +0.003235 | +0.003212 | +0.001910 |
| entropy level / slope | 100-120 | 7.788831 / +0.000004 | 7.934806 / +0.000309 | 7.816045 / +0.000121 | |
| `response_length/mean` | 100-120 | 661.98 | 687 | 662.74 | |
| E[rho] mean | 100-120 | 0.002661 | 0.005479 | 0.005028 | |
| `actor/ppo_kl` max abs | 1-120 | 0 | 0 | 0 | |
| wire bits/tok/boundary | | **1232.5** | 2304.0 | 1232 | uncompressed |

Health corroboration beyond the vetoes: entropy is flat to +0.000016/step over the full
run with a level 0.027 nats *below* the incumbent's, so there is no uniformization
signature; `response_length/mean` at 100-120 is 0.999x the incumbent's on the same window,
so there is no truncation degeneracy; `actor/grad_norm` trends *down* over the run
(1-120 slope -0.009344) with its maximum at step 14, so the 7.283 is a startup transient
rather than a level, as the dispatch states.

**The least comfortable number in this cell is V2b.** a3's 61-120 grad_norm mean of 2.892
is 79.9% of the 3.62 bar and 1.750x the incumbent's, and its run max of 7.283 is 72.8% of
the 10.0 kill bar. a3 has the narrowest V2 headroom of any surviving arm. That matters for
promotion, not for this verdict: a 200-step b1 or a 600-step round C on a3 has real
exposure to tripping V2, and the favourable within-run trend is the only thing arguing
against it. Flagged in section 8.

---

## 6. Wire budget: parity, with a half-bit literal miss and an instrumentation hazard

**Measured, greppable, constant over all 120 steps:**
`actor/comm_eff/logical_pp_bits_sr_quant = 1232.5`, `logical_pp_bytes_sr_quant = 154.0625`.

The dispatch and the launcher comment both say "1233"; the runtime value is **1232.5**,
and 1233 is its ceiling. The exact figure matters because the bar is exact.

The accounting reconstructs a3's configuration uniquely, which is why this cell is
well-pinned despite having no `set -x` trace (section 7). Per
`activation_quant.py` line 101, bits = `subset_k*bits + subset_k*16/block_size`:

| config | arithmetic | bits | ratio vs 1232 |
|---|---|---|---|
| a1, a2 full-width 1-bit | 1536*1 + 1536*16/32 | 2304.0 (logged) | 1.8701x |
| **a3, subset 2-bit k=493** | **493*2 + 493*16/32** | **1232.5 (logged)** | **1.000406x** |
| a3 with k=492 | 492*2 + 492*16/32 | 1230.0 | 0.9984x |
| incumbent prf exact-k | round((1-0.95)*1536) = 77 channels * 16 (bf16) | 1232 | 1.0000x |

**Does a3 satisfy registered success criterion 4 in a way a1 cannot? Yes, decisively, and
with one caveat that should be fixed rather than argued.**

- At **1.0004x** the incumbent, a3 is at parity for every practical and every scientific
  purpose. a1, at **1.8701x**, is not, and the amendment already states that a
  1.87x-budget arm cannot carry a communication-efficiency headline claim.
- **The literal bar is nonetheless missed.** The committed E2 clause is "wire budget
  <= 1232 for any round-C promotion". 1232.5 > 1232. The overshoot is 0.5 bits, which is
  0.041%, and it is a half-bit artifact of fractional-block scale accounting
  (493/32 = 15.40625 blocks at 16 bits each = 246.5 bits). I score it ✗ because it is ✗,
  and I am not rounding it into a ✓.
- **It is a one-character fix.** `subset_k=492` logs 1230.0 bits, strictly inside the bar,
  at 0.2% less payload, which is far below any effect this program can resolve. Any b1 or
  round-C promotion of the a3 codec should carry that change so the wire clause is met
  literally rather than by appeal to criterion 4's "or an explicit accounting of the
  trade" escape.
- **Which arm could carry a communication-efficiency claim?** On budget alone: a3 (1.0004x,
  1.0000x after the k=492 fix) and a4 (the incumbent's own codec, exactly 1232) can; a1
  and a2 cannot. Combining that with section 1, a3 is currently the only *new* arm that
  is simultaneously at parity budget and veto-clean, which is precisely why it leads
  round A despite failing the objective.

**Instrumentation hazard, verified in source and not inferred.** The two codecs report
their budgets in different units under similar names:

- `actor/comm_eff/logical_pp_bytes_prf = (1.0 - p) * hidden_size` (`state.py` line 481) is
  a **coordinate count**, not bytes. The incumbent's stored value is **76.8**
  (`metrics/incumbent_wire_bits.json`), which is also 0.2 off the 77 channels the exact-k
  mask actually keeps (`activation_mask.py` line 265).
- `actor/comm_eff/logical_pp_bytes_sr_quant = bits / 8` (`state.py` line 501) **is** bytes.

Comparing the two `_bytes_` metrics naively gives 154.0625 / 76.8 = **2.0060x** and would
report a3 as double the incumbent's wire cost. The correct comparison, converting the
coordinate count at the boundary tensor's bf16 width, is **1.0004x**. The program's 1232
figure is right; the metric is misnamed. This matters immediately for a5, whose FRLR
payload lands in the same `logical_pp_bytes_prf` slot via `frlr_payload_per_token`
(32 + 44 + 1 = 77, again a coordinate count), so a5's budget cannot be read off that
metric without an explicit dtype assumption. Flagged in section 8.

---

## 7. Resolved params and plan-versus-ran divergence

Full excerpt in `runs/93-long-horizon-stability/resolved_params-a3.txt`.

**Flag: RESOLVED_CONFIG_MISSING.** `capture_resolved_config.py runs/93-long-horizon-stability`
exits 1 with "no train.log at runs/93-long-horizon-stability/train.log". Pointed at the
synced log it finds no `python3 -m verl.trainer.main_ppo` line (grep count 0) and no
launcher config banner (grep count 0), because `metrics/incoming.log` carries only the
streaming tail of each cell. Same condition as a1 and a2, and the same three-way
reconstruction is used instead:

```
# 1. runtime metric, unique in the launcher's option space
actor/comm_eff/logical_pp_bits_sr_quant = 1232.5  ->  493*2 + 493*16/32
actor_rollout_ref.actor.comm_eff.compression_type=sr_quant
actor_rollout_ref.actor.comm_eff.quant.bits=2
actor_rollout_ref.actor.comm_eff.quant.block_size=32
actor_rollout_ref.actor.comm_eff.quant.rounding=sr
actor_rollout_ref.actor.comm_eff.quant.subset_k=493
# 2. hydra echo present in metrics/incoming.log (pid=139095)
trainer.experiment_name=a3-srq-parity-k493   trainer.total_training_steps=120
trainer.save_freq=-1   trainer.test_freq=-1   trainer.val_before_train=False
# 3. metrics-side corroboration of the protocol
actor/ppo_kl == 0 over 120 steps AND anchor_replay_fires == 6 == floor(120/20)
  => train_batch_size == ppo_mini_batch_size, one optimizer tick per global step,
     anchor cadence 20 in ticks
actor/lr = 1e-06 constant, actor/kl_coef = 0.001 constant
```

**Divergence, plan versus ran: one, and it is itself the finding of section 6.** `run.json`
cell a3 and `run_93_cell.sh` line 76 both describe the budget as "1232.5 -> 1233
bits/token/boundary vs the prf exact-k incumbent's 77*16 = 1232". The plan therefore knew
at authoring time that the arm would sit above 1232, while the amendment later committed
E2 as "<= 1232". The arm ran exactly as planned; the plan and the amendment are 0.5 bits
apart. Nothing else diverges: bits, block size, rounding, subset_k, step count, val-off,
save-off and the anchor and rank-1 defaults all match.

---

## 8. next_actions

Expressed against the committed criteria. Nothing here launches anything: the amendment
authorizes a4 and a5 only, and the decision ladder governs what follows.

```yaml
next_actions:
  - knob: "a4 gap slope, rollout_corr/kl over steps 61-120, HAC(4) one-sided"
    from: "a3 leads at +0.001300 (hac se 1.61e-4), which is 2.60x S-bar"
    to: "to DISPLACE a3 as round-A leader a4 needs <= +0.001066 (one combined hac se
         below a3, assuming a4's own se lands near 1.7e-4), and <= +0.000834 to displace
         it decisively at two se. To actually PASS the objective and fire ladder row 1 it
         needs <= +5.0e-4 with U1 (entropy slope <= +0.0015), U2 (score level >= 0.6248)
         and U4 (response_length >= 60% of its own 21-40 mean) all clean. If a4 lands in
         (5.0e-4, 1.300e-3] it takes the lead but still fails S-bar, which is ladder row 5,
         judgment-fallback REVISE plus needs:human. And U3 governs the trap: slope at or
         below S-bar WHILE U1 or U2 fires is a disqualification, not a crowning."
    rationale: "a4 is the incumbent's own codec plus CVC, so it starts with the two things
                a3 lacks: exactly 1232 bits (E2 literally clean) and a 14.2458 gap level
                (O2 clean at the cap by construction). It therefore only has to win O1 to
                take the lead outright on every clause, whereas a3 leads only on vetoes."
  - knob: "a5 eligibility and tie-break, E[rho] plus wire accounting"
    from: "a3 sets the tie-break floor at E[rho] 0.002661 mean / 0.001652 median, which is
           0.53x the incumbent and 18x below the registered 0.05"
    to: "a5 must clear V1 (<= 3.264e-3 at 100-120), V2 (max <= 10.0, 61-120 mean <= 3.62),
         V3 (level >= 0.6248 plus significant positive 61-120 slope), and reach gap slope
         <= +1.867e-3 with E[rho] in [0.2, 2.0] to fire ladder row 2. Its wire budget must
         NOT be read off actor/comm_eff/logical_pp_bytes_prf: that slot carries FRLR's
         coordinate count (32+44+1 = 77), not bytes, so a5's bits require an explicit
         dtype statement before it can be compared to 1232 at all."
    rationale: "a5 is the only arm that can move E[rho] off the floor, so it is the only
                arm that can win on T rather than on O; and the misnamed metric would
                otherwise report a5's budget wrong by 8x in the flattering direction."
  - knob: "any promotion of the a3 codec: subset_k, plus a capability read"
    from: "subset_k=493 -> 1232.5 bits, 0.5 bits over the committed E2 bar; V2b at 79.9%
           of its bar; zero capability measurement (val off, no checkpoints in round A)"
    to: "subset_k=492 -> 1230.0 bits, literally inside the bar at 0.2% less payload; and
         pre-declare a V2 abort at grad_norm > 10.0 for any 200-step or 600-step extension,
         since a3 has the narrowest V2 headroom of the surviving arms; and treat a3's
         projected gap(600) of 15.6294 (crossover with the incumbent only at step 1427)
         as the thing b1's controller has to beat, not as an acceptable baseline."
    rationale: "the wire fix removes the only E-clause a3 fails, at a change far below
                resolvable effect size; the V2 pre-declaration is cheap insurance on the
                one veto a3 nearly touches; and the horizon arithmetic is what stops a
                slope-only reading of a3 turning into an unjustified round-C promotion."
```

Two items for the operator that are **not** actions I may take:

1. **The O2 level constraint measures mostly instrument.** 86.9% of a3's level excess is
   present at step 1. Re-reading O2 as an offset-corrected or step-1-referenced quantity
   would change a3's standing, but it would be a post-hoc threshold change of the kind
   the amendment declares illegitimate, and it would have to be applied symmetrically to
   all five arms. Operator ratification only.
2. **The axis-separating probes.** A 1-bit subset arm at `subset_k=821` (1231.5 bits,
   inside the bar) would separate "coarser quantizer" from "smaller subset" at fixed
   budget; a 2-bit full-width arm (3840 bits, 3.12x) would do it diagnostically. Neither
   is authorized by any ladder row and neither is proposed as a round-A addition.

---

## 9. Notes

- **RESOLVED_CONFIG_MISSING** for a3, same cause as a1 and a2: no `set -x` trace and no
  launcher banner in the synced tail. Mitigated by an arithmetically unique runtime bit
  count, the hydra echo, and metrics-side corroboration of the protocol. See section 7.
- **Local-log coverage caveat, and why it does not affect the scoring.**
  `metrics/incoming.log` carries 109 of a3's 120 steps; steps 75 to 85 are absent, an rsync
  tail gap, with 0 disagreeing duplicate lines and 0 NaN or Inf entries. The 100-120 window
  is complete locally (n=21) and every 100-120 quantity agrees with the WandB pull to six
  decimals. The 61-120 window is short by 11 rows locally (n=49, gap slope +0.001374)
  versus the WandB full history (n=60, +0.001300). **All scoring above uses the WandB n=60
  fits**, which are authoritative and which the plan's own verification command reads. The
  dispatch's grad mean of 2.892 matches the n=60 value; the local n=49 value is 2.909.
- **Comparator symmetry.** The incumbent was refit here rather than quoted, through the
  same OLS plus Newey-West(3) code as a3, and it reproduces the amendment's frozen values
  exactly (V1 ceiling anchor +0.002176 at 100-120, gap slope +0.001867 at 61-120, gap level
  14.245787, V3 level 0.657692, E[rho] 0.005028). No threshold in this verdict was
  recomputed or moved. The a1 and dense figures are quoted from `analysis-a1.log` and
  `AB_AMENDMENT.md` section 4 respectively.
- **Cross-run caveat on the step-1 offset table.** a1, a2 and a3 share one launcher, one
  dataset and one seed, so their step-1 spread (11.4964 to 14.5284, 3.0319 nats) is exactly
  matched and is sufficient on its own to establish the offset point. The incumbent's
  step-1 value (13.8794) comes from a different run and batch shape and is included for
  context, not as a matched measurement.
- **One of the program's three KLs is structurally silent in round A.** `actor/ppo_kl` is
  identically 0 in a1, a2, a3 and the incumbent because `ppo_mini_batch_size` equals
  `train_batch_size`, so within-step ratio drift cannot be observed at all in this matrix.
  E3 is a genuine pass, not an unpopulated metric, but it also carries no information.
- **Shutdown-path tracebacks are benign, as the dispatch states.** The log tail ends in
  `RuntimeError: DataLoader worker (pid 221758) is killed by signal: Killed` inside
  `signal_handling.py` during teardown, after step 120 was logged and the WandB run reached
  state `finished` with all 120 rows. No divergence signature anywhere: no NaN, no Inf,
  grad_norm trending down, entropy flat, reward at parity.
- **What was not measured, and therefore is not claimed.** No validation, no checkpoints,
  no OOD, no capability number of any kind for a3, and round A saves nothing so none can be
  added retrospectively. Every claim above is about training-time instrumentation. The
  amendment's own section 6 dissent applies in full force to this cell: no experiment in
  this program has yet linked gap level or gap slope to held-out accuracy, so a3's 0.70x
  slope advantage is an improvement in an instrument whose causal relation to capability
  before collapse remains unestablished.
- **Single seed.** Every a3 number is n=1 in seeds. The HAC standard errors quantify
  within-run autocorrelated noise only, not seed-to-seed variation, and the incumbent's own
  val spread of 0.012 over steps 150 to 600 is the relevant scale of what one seed cannot
  resolve.
- Bounded single pass, as dispatched. No re-verification, no second opinion, no box access
  (a4 is training on 45725398). The only network reads were the WandB API for a3, a1, a2
  and the incumbent's histories.
