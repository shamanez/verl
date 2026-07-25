# Verdict, issue #93 round A cell a5 `a5-frlr-r48k28-tis` (FINAL round-A cell)

VERDICT: REVISE

WandB `kfrkehju`, project `93-long-horizon-stability`, entity `shamanework-pl`, state
`finished`, `max_step=120`. Judged against `AB_AMENDMENT.md` sections 3 and 4, committed
BLIND while a3 was at step 20 and a4 and a5 had reported nothing. Every number below is
greppable from `runs/93-long-horizon-stability/`: `analysis-a5.log` (sections 1 to 8),
`metrics/a5_gate.json`, `metrics/a5_windows_wandb.json`, `metrics/roundA_final.json`,
`metrics/a5_onbox_final_parsed.json`, `resolved_params-a5.txt`.

One pass. Nothing was launched; the box is idle by protocol and the round-A to round-B
boundary requires operator sign-off.

## 1. Verdict and why it is REVISE and not PASS or STOP

**Not PASS.** Two of the three committed vetoes fail on the committed windows: V1 drift
slope at 100-120 is 0.004584 against a ceiling of 3.2636e-3 (2.11x the incumbent, ceiling
is 1.5x), and V3 reward level at 100-120 is 0.5908 against a bar of 0.6248 (0.898x the
incumbent's 0.6577). The winner criterion in `AB_AMENDMENT.md` section 4 is lexicographic
with the vetoes ahead of the objective, so an arm that carries the best objective in the
matrix and fails two vetoes is not a winner. Rounding either veto into a pass would be the
exact goalpost move the amendment was posted blind to prevent.

**Not STOP.** None of the STOP triggers is present:

- No falsification. The registered gap criterion in its "settles" form is MET by a5 at
  parity wire: gap slope 61-120 is **-0.002049** against S-bar +5.0e-4, and gap level
  100-120 is **4.4842** against the 14.2458 bar. a5 is the only arm in the matrix whose
  wedge turns over.
- No divergence. `actor/grad_norm` 61+ max 0.19593 and mean 0.14153 (bars 10.0 and 3.62);
  run max 2.913 including the warm-start transient; no NaN anywhere; `actor/ppo_kl` max
  |.| exactly 0.0; `critic/score/max` 1.0 and `critic/score/min` 0.0 at step 120;
  `critic/advantages/max` 2.4748666, which for n=8 binary rewards is exactly the
  standardized advantage of a 1-of-8 group, so prompt groups still split.
- Nothing unmeasurable. All seven committed quantities were measured on 120 of 120 steps,
  and the WandB step-120 row agrees digit for digit with the on-box log
  (`actor/kl_loss` 0.1993504697456956 in both), so the known "final step dropped" hazard
  did not bite here.
- Depth and budget are not exhausted. `run.json` `iterations: 7`; the ledger carries no
  `revise_depth` field, and counting the four prior round-A verdicts (a1, a2, a3, a4, all
  REVISE) gives depth 4 of 7. The ledger `budget_note` records the operator raising the
  cap from 44 to 100 GPU-h; a5 itself cost 4.03 GPU-h (120 steps at 121 s/step).

**Why REVISE is the registered outcome rather than a judgment call:** the pre-registered
decision ladder has no row whose trigger is true on the five measured arms (section 6
below). The registration's own escape hatch then applies, and it is REVISE with
`needs:human`: plan open question 5, "Judgment-fallback authority: RESOLVED. The escape
hatch (analyst REVISE, planner amends only the next step, operator money read on new GPU
spend) is accepted; the fixed matrix stays intact." `AB_AMENDMENT.md` row 5 names the same
disposition in words: "Judgment-fallback REVISE ... REVISE plus needs:human, operator's
call". So REVISE here means: round A is complete, no arm is crowned, and the next GPU spend
is the operator's decision, not mine.

## 2. The committed-criteria table (blind thresholds, matched windows)

Sources: `metrics/roundA_final.json` (committed scorer `scripts/roundA_table.py`),
`metrics/a5_windows_wandb.json` (per-window HAC fits), `metrics/a5_gate.json` (plan
verification command).

| criterion | committed bar | a5 observed | incumbent, same window | result |
|---|---|---|---|---|
| E1 reached step 120 | 120 | 120 of 120 in 4h02m, 121 s/step | 600 | ✓ PASS |
| E2 wire budget | <= 1232 bits/token/boundary | **1232** (`logical_pp_bytes_prf` = 77 coords x 16 bits; `mask_ratio` 0.9498697916666666 = 1 - 77/1536) | 1232 | ✓ PASS, exact parity |
| E3 `actor/ppo_kl` | == 0 or explained | **0.0** max abs | 0.0 | ✓ PASS |
| E4 confinement non-degenerate | counters present and moving | mask_applications 82887, spectral_corrections 34138, anchor_replay_fires 6, rank1_bypass_ticks 19, frlr_q_refreshes 833 | n/a | ✓ PASS |
| V1 drift slope 100-120 | <= 3.2636e-3 (1.5x incumbent) | **0.004583793564233929** = 2.11x | 0.002175763013845237 | ✗ **FAIL** |
| V1 same veto, 61-120 window | <= 3.5162e-3 | 0.002998304976872757 = 1.28x | 0.002344112915375585 | (✓ on this window; see section 3) |
| V2 grad 61+ max / mean | <= 10.0 / <= 3.62 | **0.19593478739261627 / 0.14153027546902497** | 4.1658 / 1.6523 | ✓ PASS, best in matrix |
| V3 score level 100-120 | >= 0.6248 | **0.5908203125** = 0.898x incumbent | 0.6576915922619048 | ✗ **FAIL** |
| V3 second clause, 61-120 reward slope significantly positive | > 0 | +0.003069 +- 0.000255 HAC(4), 2.55x incumbent, z = +5.72 | +0.001203 +- 0.000203 | ✓ (level is what fails) |
| O1 gap slope 61-120 | <= +5.0e-4 (S-bar) | **-0.002049009784720745**, negative | +0.0018669196253652274 | ✓ PASS |
| O2 gap level 100-120 | <= 14.2458 | **4.484150409698486** = 3.18x better | 14.245787030174618 | ✓ PASS |
| T tie-break E[rho] median 100-120 | higher is better; [0.2, 2.0] is the registered row-2 quadrant | **0.39847493171691895**, 193.7x the incumbent | 0.0020570755004882812 | ✓ IN quadrant |

Score: eligibility 4 of 4, vetoes 1 of 3, objective 2 of 2, tie-break in the registered
quadrant. Stretch targets from the machine-readable block: `gap_lt_10` **True** (a5 is the
only arm ever to clear it), `gap_lt_3_target` False, `e_rho_gt_0p05` **True**,
`ppo_kl_zero` True, `ref_kl_le_baseline` True (gate mean 0.1496 sits under the baseline
card's 0.156 to 0.203), `reward_slope_parity` False (+0.00247 full-run against 0.00288).

Metrics summary at the authoritative final step, from the on-box log
(`metrics/a5_onbox_final_parsed.json`, step 120): score 0.599609375, gap 4.312202453613281,
ref-KL 0.1993504697456956, entropy 3.3851022720336914, grad_norm 0.12113713473081589,
response_length 692.3896484375, `rollout_log_ppl` 0.1948087513446808,
`training_log_ppl` 4.484829425811768, IS effective sample size 0.2753135008641815,
`critic/advantages/max` 2.4748666286468506, score max 1.0 min 0.0, E[rho] at that step
3.5895237922668457 - 4.312202453613281 + 1 = 0.2773.

## 3. The V1 reconciliation: per-window fits, HAC(4) standard errors

The apparent contradiction is real and it resolves cleanly. `actor/kl_loss`, OLS with
Newey-West HAC(4) standard errors, identical windows for both runs, n = 20 per window
(`analysis-a5.log` section 2; residual lag-1 autocorrelation reported per fit):

| window | a5 slope | HAC(4) se | acf1 | a5 level | incumbent slope | HAC(4) se | incumbent level | a5/inc slope | z of difference |
|---|---|---|---|---|---|---|---|---|---|
| 2-20 | -0.047626 | 0.031515 | 0.17 | 0.1627 | +0.000121 | 0.000058 | 0.0305 | n/a | -1.52 |
| 21-40 | +0.000225 | 0.000027 | 0.57 | 0.0043 | +0.000185 | 0.000025 | 0.0329 | 1.22x | +1.09 |
| 41-60 | +0.000510 | 0.000026 | 0.10 | 0.0134 | +0.000969 | 0.000109 | 0.0452 | 0.53x | -4.11 |
| 61-80 | +0.001640 | 0.000092 | 0.34 | 0.0322 | +0.002848 | 0.000075 | 0.0860 | 0.58x | -10.15 |
| 81-100 | +0.002982 | 0.000071 | 0.02 | 0.0777 | +0.002065 | 0.000095 | 0.1332 | 1.44x | +7.75 |
| 101-120 | +0.004619 | 0.000150 | -0.23 | 0.1518 | +0.002158 | 0.000153 | 0.1797 | 2.14x | +11.49 |
| 61-74 (the earlier read) | +0.001368 | 0.000071 | 0.04 | 0.0269 | +0.002625 | 0.000029 | 0.0772 | 0.52x | -16.27 |
| 61-120 (committed alt.) | +0.002998 | 0.000182 | 0.84 | 0.0872 | +0.002344 | 0.000038 | 0.1330 | 1.28x | +3.51 |
| 100-120 (committed) | +0.004584 | 0.000140 | -0.20 | 0.1496 | +0.002176 | 0.000137 | 0.1786 | 2.11x | +12.31 |

Note the 2-20 window: a5's step-2 warm-start value is 2.797770 and it collapses to 0.242505
at step 3, 0.011521 at step 4 and 0.002604 at step 5. Any fit that includes step 2 returns a
large negative slope (-0.0476) that is a transient artifact, not a drift measurement. That
is why the reconciliation is fitted from step 21 and the 2-20 row is reported separately and
not used. The committed windows (61-120 and 100-120) are both clean of it.

**Characterisation, in three parts.**

**(a) Yes, a5's drift is accelerating from a near-zero base, monotonically:** 0.000225,
0.000510, 0.001640, 0.002982, 0.004619, an increase of 20.5x across four consecutive
windows, each step-up many HAC standard errors wide. The incumbent does the opposite: it
peaks at 61-80 (+0.002848) and then settles (+0.002065, +0.002158), and its own 600-step log
shows it stays roughly linear thereafter (121-200 slope +0.001455, 201-300 +0.001494,
301-400 +0.001857, 401-500 +0.001589, 501-600 +0.001165, level 0.9071 at 580-600 with no
collapse). So the two arms are on qualitatively different curve shapes inside 1 to 120.

**(b) On LEVEL, the comparison that says whether a5 ends further from the base model, a5 is
CLOSER, at every measured point inside the window:** ref-KL level at 100-120 is
**0.149579 for a5 against 0.178583 for the incumbent (0.838x)**, and dense is 0.003847.
Point values: step 24 a5 0.002871 against 0.031434; step 60 0.018515 against 0.060601;
step 100 0.105939 against 0.155711; step 120 **0.199350 against 0.203385 (0.98x)**. The
69.4x growth in a5's own level from step 24 to step 120 (0.002871 to 0.199350) is therefore
growth out of a 10.9x deficit up to parity, not an overshoot that has happened.

**(c) The window ends exactly at the crossing, and that is the honest limit of the data.**
Fitting both 101-120 lines and evaluating at step 120 gives a5 0.195636 and the incumbent
0.200229; the linear crossing is **step 122**, two steps past the last measurement. Under a
naive constant-slope projection a5 reaches 0.569 at step 200, 1.031 at 300 and 2.416 at 600,
against the incumbent's **measured** 0.303538, 0.460520 and 0.928460. Under saturation it
converges onto the incumbent's trajectory instead. **These data cannot distinguish
saturation from overshoot**, because a5's curve is still accelerating at the moment the run
stops. That is the single most decision-relevant unresolved fact in round A, and it is
resolvable for 80 more steps (section 7, option B).

So: the V1 veto fired on a leading indicator (slope), not on a realized outcome (level).
That is what a slope veto is for, and the amendment committed both windows in advance,
including the note that the 100-120 read "is the read that catches the terminal
acceleration". I apply it as written. But the verdict would be dishonest if it recorded
"a5 drifted further from the base model than the incumbent", because at step 120 it did not.

## 4. Is the V1 failure the same phenomenon as the small-step effect?

Same trajectory, and the small-step effect does not exonerate it. Three claims, in order of
how well the data support them.

**4.1 a5 is phase-lagged, and three independent codec-free markers say so (solid).**
Token-IS at threshold 2.0 shrinks the effective step by 12.4 to 15.7x window on window:
`actor/grad_norm` window means are a5 0.1203 to 0.1435 against the incumbent's 1.7426 to
2.1392, with the incumbent-over-a5 ratio 15.33, 15.68, 12.37, 14.73, **12.55** at 101-120.
At any fixed step the two arms are therefore at different points of their own learning
curves, and all three markers agree on the direction:

- reward: a5's 101-120 slope is still **positive** (+0.000754 +- 0.000530) while the
  incumbent's has gone **negative** (-0.001376 +- 0.000578), z = +2.72;
- sampler perplexity, a codec-free quantity: a5's `rollout_log_ppl` is still falling at
  -0.002571 +- 0.000234 per step while the incumbent's is flat (+0.000005 +- 0.000140);
- drift itself: a5's ref-KL slope is still rising window on window while the incumbent's
  peaked at 61-80.

The consequence for the protocol is real: matched-window discipline matches on **steps**,
not on **learning progress**, and with a 12.5x step-size difference those are not the same
thing.

**4.2 Correcting for the lag does not rescue a5 (solid, and this is the load-bearing
result).** Two independent normalisations:

- *Progress-matched:* a5's 101-120 reward level is 0.5895; the incumbent's nearest matched
  progress window is 61-80 at 0.6068. Compared there, a5's ref-KL slope is
  **1.62x** the incumbent's (0.004619 against 0.002848) and its ref-KL level is **1.76x**
  (0.1518 against 0.0860).
- *Drift per unit of learning:* from step 20 to step 120, a5's ref-KL rose +0.196847 while
  its reward level rose +0.2284, that is **0.8619 nats per unit of score**; the incumbent
  rose +0.169546 on +0.2966, that is **0.5717**. a5 is **1.508x** less
  drift-efficient per unit of learning.

So the committed 2.11x overstates the arm-to-arm difference by roughly 1.3x relative to a
progress-matched read, and a progress-matched read still puts a5 at 1.5 to 1.8x the
incumbent. **The V1 failure is substantive, not an artifact of the window.**

**4.3 Mechanism candidates, none established, listed so they are not confused with the
result above.**

- *Time-varying view offset.* V1 was committed as "slope only, never level, because every
  codec carries a different view offset", which assumes the offset is a constant. a5 breaks
  that assumption: its `training_log_ppl` falls 11.3824 at step 1 to 4.4848 at step 120,
  a 6.90-nat fall, while its sampler `rollout_log_ppl` falls only 0.8878 to 0.1948, a
  0.69-nat fall, so a5's codec view is getting more accurate as training proceeds and it is
  the view, not the sampler, that moves. A view that starts inaccurate and converges can contribute a
  positive component to the measured ref-KL slope that is not policy motion. The direction
  fits (a5's ref-KL at step 5 is 0.002604 against the incumbent's 0.029091, an 11x deficit
  that closes to 0.98x by step 120), but I cannot decompose measured drift into view motion
  and policy motion from these logs. The instrument that would do it is the registered I3
  dense-view probe, which is exactly what round B was scheduled to add. **Flagged as a
  methodological finding about V1's design assumption, not as an excuse for a5.**
- *Warm-start transient contamination.* The step-2 spike (2.797770) is confined to steps 2
  to 4 and cannot reach the committed windows; it does contaminate any all-steps fit, which
  is why gate93's full-run slope column reads +0.00018 per step for a5 and must not be
  compared with a windowed number.
- *Codec-driven rather than optimizer-driven drift.* Plausible, since 12.5x smaller steps
  producing 2.1x the slope is not what optimizer-driven drift looks like, but the
  progress-matched result in 4.2 already shows that whatever the driver is, it is not
  neutralised by shrinking the step. Untested.

## 5. Is the learning deficit tunable? The threshold is the wrong knob

Measured facts at 101-120: mean importance weight **0.1656**, effective sample size
**0.2681**, `rollout_is_ratio_fraction_low` **0.8841**, and from the step-120 summary
`rollout_is_ratio_fraction_high` **0.002929**, `rollout_is_max` **2.0** (the clamp is
active), `logratio_p50` **3.4469** nats, so the median token importance ratio is about
exp(-3.4469) = **0.0318**.

**Argument that raising the threshold buys almost nothing.** The threshold only re-weights
tokens sitting at the upper cap, and only **0.29 percent** of tokens are there. An upper
bound on the mean-weight gain from moving the cap is `frac_high x (thr_new - 2)`:
raising 2.0 to 4.0 gains at most 0.0059 on a base of 0.1656 (**at most 1.04x**), and 2.0 to
8.0 gains at most 0.0176 (**at most 1.11x**). Recovering the missing 12.5x of step size from
the cap is arithmetically impossible: 88.4 percent of the mass is at the LOW end, pushed
there by the arm's own 4.3-nat wedge, not by the clamp. Raising the threshold would mostly
re-admit the variance the threshold exists to suppress, for a percent-level change in mean
weight.

**The knob that would actually move it, and it already exists and was OFF.**
`resolved_params-a5.txt` records
`algorithm.rollout_correction.rollout_is_batch_normalize = False`. Self-normalising the
weights restores unit mean weight while preserving the relative reweighting, that is, it
removes the 6.04x global gradient attenuation (1/0.1656) without removing the token-level
correction that is the arm's hypothesised stabiliser. No new code. Note the honest limit of
that arithmetic: 6.04x of the observed 12.55x grad-norm gap is mean-weight attenuation, so
normalisation is expected to recover part of the step size, not all of it; the remainder sits
in the weighting's variance reduction and direction change, which normalisation preserves by
design.

**Is buying V3 back plausible at all?** Two readings, and I will not pretend they agree:

- Encouraging: a5's reward-level ratio to the incumbent troughs at 0.7474 (41-60) and then
  **recovers**: 0.7719, 0.8513, **0.8975** at 101-120. Its 61-120 reward slope is 2.55x the
  incumbent's at z = +5.72, and its 101-120 slope is still positive while the incumbent's
  has turned negative. If the last window's slope (+0.000754/step) persisted, a5 needs 47
  steps to reach the V3 bar of 0.6248 and 90 steps to reach the incumbent's 0.6577.
- Discouraging: a5's own reward slope is decelerating (+0.005735, +0.003726, +0.000754),
  which is exactly what approaching a plateau near 0.59 to 0.61 looks like. The optimistic
  reading (using the 61-120 slope, 12 steps to the bar) is not credible against that
  deceleration.

**Position:** the deficit is plausibly tunable and the specific plausible knob is
`rollout_is_batch_normalize`, not `rollout_is_threshold`. **This is a hypothesis that
requires one new cell; these data do not establish it.** Falsifiers, stated in advance so
the next cell is decisive either way: (i) if self-normalised token-IS restores reward toward
0.62 to 0.66 **and** the gap slope stays <= +5.0e-4, then the gap result belongs to the FRLR
codec and the V3 deficit was an artifact of the un-normalised weighting, which is the
success case; (ii) if reward recovers **and** the gap slope goes positive with V1 worsening
toward the incumbent, then a5's stability was the small steps all along, the mechanism claim
collapses, and the FRLR-plus-token-IS route is closed; (iii) if reward does not recover, the
deficit is not the mean-weight attenuation and the arm is learning-limited for a reason not
yet identified.

## 6. Round-A standing and the ladder fallthrough

Five-arm final standing, from `metrics/roundA_final.json` (V1 ceilings 3.5161693730633775e-3
at 61-120 and 3.2636445207678555e-3 at 100-120):

| arm | E wire | V1 @100-120 | V2 max/mean | V3 level | gap slope 61-120 | gap level 100-120 | E[rho] |
|---|---|---|---|---|---|---|---|
| dense `90-dense-600` | no compression | 1.64e-5 | 0.068/0.047 | 0.6587 | -4.69e-7 | 0.000242 | 1.0000 |
| incumbent `90-prf-exactk-600` | 1232 | 2.176e-3 (1.00x) | 4.166/1.652 | 0.6577 | +1.867e-3 (3.7x S-bar) | 14.2458 | 0.00206 |
| a1 `a1-srq-b1-sr` | **2304 FAIL** | **3.887e-3 FAIL** | 0.700/0.651 | 0.6529 | +3.265e-3 | 13.7511 | 0.00253 |
| a2 `a2-srq-b1-rn` | died at 62, **ineligible** | **48.5x FAIL** | **9.146/8.199 FAIL** on its tail window; run max 62.238 per AB_AMENDMENT | n/a | +1.7478e-2 | 12.1851 | n/a |
| a3 `a3-srq-parity-k493` | **1232.5 FAIL by 0.5 bits** | 1.616e-3 ✓ | 5.252/2.892 ✓ | 0.6569 ✓ | +1.300e-3 ✗ | 14.9924 ✗ | 0.00165 |
| a4 `a4-prf-exactk-cvc-ce` | 1232 ✓ | **3.955e-3 FAIL (1.82x)** | 3.622/1.838 ✓ | 0.6491 ✓ | +1.685e-3 ✗ | 14.2473 ✓ | 0.00246 |
| **a5 `a5-frlr-r48k28-tis`** | **1232 ✓** | **4.584e-3 FAIL (2.11x)** | **0.196/0.142 ✓ best** | **0.5908 FAIL** | **-2.049e-3 ✓ only pass** | **4.4842 ✓ 3.18x better** | **0.3985 ✓ quadrant** |

**The ladder problem, stated exactly.** No arm clears both the veto set and the objective.
a3 clears all three vetoes and fails both objective clauses (and misses E2 by 0.5 bits);
a5 clears both objective clauses and fails two vetoes. Applying
`AB_AMENDMENT.md` section 3 in order, on all five arms:

| row | trigger | status on the measured matrix |
|---|---|---|
| 0 | a3 gap level <= 10.0 at 1233 bits, vetoes clean | **FALSE**, a3 level 14.9924 |
| 1 | a4 gap slope <= +5.0e-4, vetoes clean, guard clean | **FALSE**, a4 slope +1.685e-3 = 3.37x bar, and a4 V1 fires |
| 2 | a4 not settling; a5 alive, E[rho] in [0.2, 2.0], **vetoes clean**, gap slope <= +1.867e-3 | **FALSE on one limb**: a4 not settling TRUE, E[rho] 0.3985 TRUE, gap slope -2.049e-3 TRUE, **vetoes clean FALSE** (V1 and V3) |
| 3 | a4 trips its uniformization KILL; a5 in the row-2 quadrant | **FALSE**, a4's U1/U2/U4 were all clean (verdict-a4 section on the guard) |
| 4 | a4 KILL; a5 clean but E[rho] < 0.2 | **FALSE** twice: no a4 KILL, and a5's E[rho] is 0.3985 |
| 5 | a4 flat (slope in (5.0e-4, 1.867e-3]) AND veto-clean; a5 dead or low E[rho] | **FALSE**: a4's slope IS in the band, but a4 is not veto-clean, and a5 is neither dead nor low-E[rho] |
| 6 | No arm clears the drift veto, **or** every arm is rising | **FALSE on both limbs**: a3 clears V1 at 1.616e-3 (and a5 clears it on the 61-120 window at 1.28x); and not every arm is rising, a5's gap slope is negative |
| 7 | Any arm grad_norm max > 10, or entropy/reward crash | **FALSE for a3, a4, a5**. a2 tripped it in flight (62.238) and its registered action, kill and no b1, was already executed |

**The ladder falls through. Row 2 required a5 with clean vetoes and row 6 required no arm to
clear V1; a5 fails the first and a3 defeats the second.** The registered judgment fallback
therefore governs: analyst REVISE, planner may amend only the next step, and the operator
makes the money read on any new GPU spend. **Nothing launches on this verdict.** Per
`AB_AMENDMENT.md` section 5 the following still need operator sign-off and are unaffected by
me: ratifying the amendment itself, any b1 launch, round C, the optional val@120 on a4 and
a5 (now past its now-or-never point, since round A saved no checkpoints), and teardown of
box 45725398, for which no standing authorization exists.

One honest note on the window that decided two arms: **a4 and a5 both clear V1 on the
committed 61-120 window (1.07x and 1.28x) and both fail it on the committed 100-120 window
(1.82x and 2.11x).** Both ceilings were fixed in advance, a1 was judged on 100-120 in the
amendment's own ranking table (its "FAIL, 3.887e-3 = 1.79x" is the 100-120 number), and the
a4 verdict already elected 100-120 as the consistent read. I keep that read for a5. The
window choice is load-bearing for two of the three parity-budget arms, it was fixed blind,
and it should be reported that way rather than quietly re-picked.

## 7. next_actions, as costed options for the operator (I am taking none)

Cost basis: measured 121 s/step on box 45725398 (H200 NVL, $3.344/h, team account).

| # | knob | from | to | what it tests | cost | falsifier |
|---|---|---|---|---|---|---|
| A | `algorithm.rollout_correction.rollout_is_batch_normalize` | `False` | `True` (self-normalised token-IS, everything else a5-identical, 120 steps) | Whether the V3 deficit is the ~6x mean-weight attenuation (1/0.1656) and whether a5's gap result survives restoring the step size. The only knob with the arithmetic headroom to move V3 (section 5) | 120 steps = **4.03 GPU-h, about $13.49** | Gap slope turns positive or V1 worsens toward the incumbent while reward recovers, which would show the stability was the small steps, not the codec |
| B | `trainer.total_training_steps` on the a5 config | `120` | `200` (or extend a5 by 80 steps) | The one question the a5 data cannot answer: saturation against overshoot. Naive projection is 0.569 at step 200 against the incumbent's **measured** 0.303538, and the incumbent already has 121-200 in hand for a matched read | 80 steps = **2.69 GPU-h, about $8.99** | a5 kl_loss at 200 <= about 0.31 means the acceleration was a convergence transient and V1's 2.11x was a terminal-window read; >= about 0.5 means the drift compounds and this route is closed |
| C | `algorithm.rollout_correction.rollout_is` | `token` | `off` (FRLR r48 k28 alone, 120 steps) | Attribution: which component owns the -0.002 gap slope (the FRLR codec) and which owns the V3 deficit (token-IS). Currently confounded, since a5 changed both at once | 120 steps = **4.03 GPU-h, about $13.49** | If gap stays near 4.5 with a non-positive slope and reward returns to parity, token-IS is unnecessary and round B has a clean parity-budget candidate; if gap reverts toward 14, the gap result belongs to token-IS, not to FRLR |

Ordering note for the operator, not a decision: B is the cheapest and answers the question
that decides whether a5 is a candidate at all; C is the cleanest attribution; A is the one
that could turn a5 into a winner but also the one that could destroy its only passing axis.
Any two of the three fit inside 7 GPU-h. The registered b1 (200 steps with probe and
controller) costs 6.72 GPU-h at this box speed against the 6.2 quoted in the amendment at
112 s/step.

## 8. What round A established as science

1. **SOLID. The train/inference wedge can be turned over at exact parity wire, and only
   reconstruction accuracy did it.** a5 at 1232 bits/token/boundary is the only arm with a
   negative gap slope (-0.002049 against S-bar +5.0e-4) and the only arm ever to clear the
   literal registered "< 10 nats" gate (level 4.4842 against the incumbent's 14.2458, 3.18x
   better). Direction is unambiguous on the matched 61-120 slope, which is the committed
   statistic: a5 **-0.002049** against incumbent +0.001867, a1 +0.003265, a3 +0.001300, a4
   +0.001685, that is, every other compressed arm's wedge rises and only a5's falls. On
   matched step-1-to-step-120 point values a5 goes 10.7564 to 4.3122 (**-6.4442**) while the
   incumbent goes 13.8794 to 14.2394 (+0.3600), and the incumbent continues to +0.78 over its
   full 600 steps.
2. **SOLID. That fall is codec-view convergence, not a rebased offset, and the decomposition
   is quantitative.** a5's `training_log_ppl` falls 11.3824 to 4.4848 (**-6.90 nats**) while
   its sampler `rollout_log_ppl` falls only 0.8878 to 0.1948 (**-0.69 nats**), so about 91
   percent of the 6.44-nat gap collapse is the training view converging onto the sampler
   rather than the sampler moving. The gap tracks their difference throughout (step 120:
   4.4848 minus 0.1948 = 4.2900 = `log_ppl_diff`, against gap 4.3122), and at 100-120 a5's
   sampler perplexity 0.2221 sits beside the incumbent's 0.1870 while its training view sits
   at 4.6503 against the incumbent's 14.4365. Rank-48 FRLR warm-starts into an accurate
   reconstruction; a random 77-coordinate mask does not.
3. **SOLID, negative, and the round's main result. Reference drift is not bought by fixing
   the gap.** Five arms, two mechanism families (coherence correction via a4's CVC,
   reconstruction accuracy via a5's FRLR), and nothing clears the drift veto and the
   objective together. a5 pays **1.508x** the incumbent's ref-KL per unit of reward gained
   (0.8619 against 0.5717 nats per unit score) and **1.62x** its drift slope even when
   matched on progress rather than on step. Gap and drift are not the same axis, and closing
   the gap did not close the drift.
4. **SOLID. Token-IS at threshold 2.0 is a 12.5x step-size shrink, and the threshold is not
   the knob that controls it.** grad_norm 0.1435 against the incumbent's 1.8019 at 101-120,
   mean weight 0.1656, ESS 0.2681, and only 0.29 percent of tokens at the cap against 88.4
   percent at the low end (median ratio 0.0318). The shrink buys the lowest 61+ grad-norm
   max and mean of any arm in the matrix (0.196/0.142, against a1 0.700/0.651, incumbent
   4.166/1.652, a4 3.622/1.838, a3 5.252/2.892) and costs the V3 level.
   Moving the cap from 2.0 to 8.0 could change the mean weight by at most 1.11x.
5. **HYPOTHESIS, needs one cell each.** (i) a5's V1 failure may be a terminal-window read
   rather than a divergence: its ref-KL level is below the incumbent's at every measured
   step (0.98x at step 120) and the fitted crossing lands at step 122, two steps outside the
   window; saturation against overshoot is unresolved. (ii) V1's "slope only, offsets are
   constant" design assumption is violated by any codec whose fidelity improves during the
   run, which a5 demonstrably is; how much of a5's measured slope is view motion rather than
   policy motion is unmeasured, and the registered I3 dense-view probe is the instrument for
   it. (iii) The V3 deficit is plausibly recoverable via `rollout_is_batch_normalize`, which
   was off.

## 9. Provenance, plan versus ran

The plan resolved from the local cache (`.claude/state/plan-cache/93.md`, `plan_fetch` is
unavailable in this shell, so the GitHub body was not re-read; the run.json success-criteria
snapshot and `AB_AMENDMENT.md` agree with the cached plan on every threshold used here). The
plan's per-cell verification command is "pull history, compute the gate table", which is
`scripts/gate93.py`; it was run verbatim and its stdout is section 1 of
`analysis-a5.log`. The plan's single `analyze.py` pass is registered for round C only and
was not run.

Contract item 3: `python3 scripts/capture_resolved_config.py runs/93-long-horizon-stability`
reports `no train.log at runs/93-long-horizon-stability/train.log`, and
`metrics/incoming.log` carries no expanded `python3 -m verl.trainer.main_ppo` line. **Flag
`RESOLVED_CONFIG_MISSING`** for the shell `set -x` path, as for a1 through a4. The resolved
Hydra tree was instead recovered from the WandB run record (821 flattened keys) plus on-box
runtime echoes and written to `resolved_params-a5.txt`.

Resolved-params excerpt, the arm-defining knobs:

```
actor_rollout_ref.actor.comm_eff.compression_type = prf_mask
actor_rollout_ref.actor.comm_eff.mask.frlr = True
actor_rollout_ref.actor.comm_eff.mask.frlr_rank = 48
actor_rollout_ref.actor.comm_eff.mask.frlr_k = 28
actor_rollout_ref.actor.comm_eff.mask.frlr_unbiased = False
algorithm.rollout_correction.rollout_is = token
algorithm.rollout_correction.rollout_is_threshold = 2
algorithm.rollout_correction.rollout_is_batch_normalize = False
```

Protocol conformance, checked key by key against the a-cell protocol: `data.train_batch_size`
128, `ppo_mini_batch_size` 128, `data.max_prompt_length` 1024,
`data.max_response_length` 2048, `mask.pp_size` 8, `optim.lr` 1e-06 AdamW constant,
`kl_loss_coef` 0.001 `low_var_kl`, `anchor.cadence` 20 / `delay_K` 20 /
`batch_scope` rollout_batch / `snapshot_device` cpu / `warmup_mode` stale_correct,
`lookahead_mode` rank1_relex with `sliding_window` and `window_snapshots` 2 (W2) and
`lookahead_strength` 1, `spectral.beta_anc` 0.25 and `signed_ema_alpha` 0.25 on
`all_floating`, `rollout.n` 8, `save_freq` -1, `test_freq` -1, `val_before_train` False,
`total_training_steps` 120. **No plan-versus-ran divergence found.**

Two provenance points worth recording because they are cross-arm evidence, not just
bookkeeping:

- The arm-exclusive knob is visible in the runtime log, not only in config:
  `metrics/incoming.log:14892` reads
  `[comm_eff][anchor-objective] parity=PASS ... rollout_is_weights=true` in the a5 segment,
  against `rollout_is_weights=false` at `:1625` for every earlier arm. Token-IS really was
  exclusive to a5, so the "token-IS was dead on PRF and sr_quant" prior is not being
  re-tested here, it is being applied to a different codec for the first time.
- Wire parity is confirmed twice, from source and from the run:
  `activation_mask.py:555` gives `kept = r_eff + k + (0 if frlr_unbiased else 1)` =
  48 + 28 + 1 = 77 coordinates, and the run reports
  `actor/comm_eff/logical_pp_bytes_prf` = 77 with `mask_ratio` 0.9498697916666666 =
  1 - 77/1536. At 16 bits per fp16 coordinate that is 1232 bits/token/boundary, exactly the
  incumbent's budget. The field name says bytes and holds coordinates; the 16x hazard flagged
  in `WIRE_BUDGET.md` is live and was handled.

## 10. Notes

- **Established context applied, not re-litigated** (per the brief and the prior verdicts):
  a5's codec-view entropy of 3.4373 at 100-120 is the LEAST corrupted reading in the matrix,
  not a collapse. The dense control reads 0.1815 against its own sampler's 0.1792 (agreement
  to 1 percent, as it must, since an uncompressed run's codec view IS its true view), while
  every compressed arm reads 7.79 to 7.94 against a sampler value of 0.18 to 0.19, so the
  codec inflates the reading about 43x and a HIGH reading is codec mush. The
  Pinsker-plus-Fannes-Audenaert bound (TV <= 0.0354 at ref-KL 0.0025, entropy change capped
  at 0.575 nats over the 151936-token vocabulary against an observed 2.24-nat early drop,
  3.9x the ceiling) independently rules out policy movement as the cause. Degeneracy is
  refuted on codec-free observables: `critic/advantages/max` 2.4748666,
  score max/min 1.0/0.0, `rollout_log_ppl` 0.2221 sitting with the pack (0.179 to 0.188),
  response length 730.5 at 100-120 (longer than the incumbent's 662.7, not collapsing), and
  `response_length/clip_ratio` 0.0566 at step 120.
- **Completion evidence** (contract item 1): no `done.flag` exists in the run dir, but WandB
  state is `finished` at `max_step=120` and the on-box tail
  (`metrics/a5_onbox_tail_steps110-120.txt`) carries `step:118`, `step:119`, `step:120`
  metric lines, with `metrics/a5_onbox_final_parsed.json` matching WandB digit for digit at
  step 120. The trailing tracebacks in `metrics/incoming.log` are the known benign
  shutdown-path DataLoader teardown, after step 120 completed.
- **gate93 slope column is a full-run fit** (`scripts/gate93.py` `full_slope`), which is why
  it prints +0.00018 per step for a5's ref-KL where the committed windowed reads are
  +0.002998 (61-120) and +0.004584 (100-120). The full-run number is contaminated by the
  step-2 warm-start spike and must not be compared with a windowed threshold. Likewise
  gate93's `grad_norm max` 2.913 is a run maximum, while V2's 0.19593 is the 61+ window
  maximum the criterion names.
- **HAC bandwidth.** Fits use Newey-West with lag 4, as the amendment's HAC(4) convention
  specifies. Residual lag-1 autocorrelation on the committed windows is low after windowing
  (-0.23 to +0.57 per window; the 0.84 figure appears on the wider 61-120 fit, which is
  where the brief's 0.67 to 0.85 range comes from), so HAC(4) is conservative here rather
  than generous.
- **Not measured, and therefore not claimed:** capability. Round A saved no checkpoints and
  ran with val off, so no arm in this round has a held-out accuracy number, and the
  amendment's own dissent (section 6, "neither the old bar nor my new one has ever been tied
  to capability") stands unanswered by round A. The optional val@120 on a4 and a5 was a
  now-or-never item at launch time and was not taken, so a5's 0.898x training-reward deficit
  cannot be converted into a statement about model quality.
- **MANUAL_REVIEW_NEEDED is not raised.** The verdict is decisive on the committed criteria.
  The escalation here is the registered one (ladder fallthrough to the operator's money
  read), not analyst doubt about the numbers.
