# Verdict, issue #93 round A, cell `a4-prf-exactk-cvc-ce`

VERDICT: REVISE

| field | value |
|---|---|
| cell | `a4-prf-exactk-cvc-ce`, arm a4, 120/120 steps, 3h52m at 116 s/step |
| WandB | `8rux5ea6`, project `93-long-horizon-stability`, entity `shamanework-pl`, state `finished` |
| comparator | incumbent `90-prf-exactk-600`, WandB `woqs8zra`, project `90-prf-exactk-600`, matched windows only |
| criteria source | `AB_AMENDMENT.md` (S-bar, V1/V2/V3, O, T, ladder) and `A4_GUARD.md` (U1/U2/U3/U4), both committed blind before a4 produced a step |
| plan | issue #93 plan block resolves (`.claude/state/plan-cache/93.md`); per-cell verification command is `gate93.py`, run verbatim in `analysis-a4.log` Section 1 |
| numbers | every figure below is greppable in `runs/93-long-horizon-stability/`: `analysis-a4.log`, `metrics/a4_gate.json`, `metrics/a4_windows_wandb.json`, `resolved_params-a4.txt`, `WIRE_BUDGET.md` |
| ledger | `revise_depth` 0 (no revise field on the row) against `iterations` 7, so REVISE is available |

## 1. Verdict and justification

a4 ran to completion cleanly, cost nothing extra on the wire, kept every safety guard,
did not exhibit the failure mode its blind guard was written to catch, and **did not
achieve its registered success criterion on either of the two committed objective
clauses, while firing the committed drift veto.**

REVISE rather than PASS: two committed objective clauses fail (gap slope 3.37x S-bar at
z=+10.60; gap level over the cap) and one committed veto fires (V1 drift, 1.82x the
matched incumbent at z=+10.20 and 1.21x the committed ceiling at z=+2.90). Nothing here
is a rounding question.

REVISE rather than STOP: the pre-registered ladder does not terminate on this result. It
routes the round-A outcome through a5 (row 2 or row 6), a5 is training now and is the
last arm, no budget or depth is exhausted (ledger `max_gpu_hr` 100, `revise_depth` 0 of
7), and there is no divergence signature (grad_norm run max 3.6216, no NaN, no crash).
a4 is a **clean falsification of one mechanism, not a program stop.**

The falsification is unusually well posed, because a4 is the first arm in this program
whose resolved config differs from the incumbent's by **exactly one science knob**. Every
load-bearing key in `resolved_params-a4.txt` matches the incumbent's pulled config:
`compression_type=prf_mask`, `mask.p=0.95`, `mask.exact_k=True`, `mask.rescale=False`,
`mask.pp_size=8`, `anchor.cadence=20`, `anchor.delay_K=20`,
`anchor.batch_scope=rollout_batch`, `anchor.lookahead_mode=rank1_relex`,
`lookahead_window_snapshots=2`, `spectral.beta_anc=0.25`, `signed_ema_alpha=0.25`,
`kl_loss_coef=0.001`, `optim.lr=1e-06`, `train_batch_size=128`,
`ppo_mini_batch_size=128`, `max_prompt_length=1024`, `max_response_length=2048`,
`rollout.n=8`, `rollout_correction.rollout_is=None`, model `Qwen/Qwen2.5-Math-1.5B`. The
differences are `cvc_lambda` 0 to 0.003, `total_training_steps` 600 to 120 and
`test_freq` 150 to -1 (the incumbent's first validation is at step 150, so no validation
fires inside the compared 1-120 span in either run). So this is as close to a CVC on/off
contrast as this program has produced.

## 2. Committed criteria, per clause

| clause | committed bar | a4 observed | matched incumbent | result |
|---|---|---|---|---|
| **O slope** `rollout_corr/kl` 61-120 | <= +5.0e-4 (S-bar) | **+0.001685 +- 0.000112** HAC(3), n=60, z=+10.60 vs bar | +0.001867 +- 0.000167, ratio **0.9025x**, difference z=**-0.91** | **FAIL, 3.37x the bar** |
| **O level** `rollout_corr/kl` 100-120 | <= 14.2458 | **14.247296** (n=21, sd 0.013538, se of mean 0.002954) | 14.245787 (sd 0.009875); difference +0.001509 nats, se_diff 0.003657, z=**+0.41** | **FAIL by 0.001496 nats = 0.0105 percent** (a statistical tie, see 5) |
| **V1 drift** `actor/kl_loss` slope 100-120 | <= 1.5x incumbent = 3.264e-3 | **+0.003955 +- 0.000096** | +0.002176 +- 0.000146, ratio **1.818x**, z=+10.20; vs ceiling z=**+2.90** | **FAIL** |
| V1 drift, other committed window 61-120 | <= 1.5x incumbent = 3.516e-3 | +0.002514 +- 0.000128 | +0.002344, ratio 1.072x, vs ceiling z=-7.17 | PASS |
| **V2 gradient** | run max <= 10.0, 61-120 mean <= 3.62 | run max **3.6216** at step 78; 61-120 mean **1.837940** | incumbent 61-120 mean 1.870069 | **PASS** |
| **V3 learning** `critic/score/mean` | level 100-120 >= 0.6248, plus significantly positive 61-120 slope | level **0.649089 +- 0.000561** (z=+43 vs bar); 61-120 slope +0.001078 +- 0.000169 (t=+6.4 vs zero) | incumbent level 0.657692; difference z=-1.03, not significantly below | **PASS** |
| **U1 entropy** `actor/entropy` slope 21-120 | <= +0.0015 (kill) | **+0.000093 +- 0.000004**, 16.2x below the bar | incumbent +0.000097 +- 0.000004, difference z=-0.7, indistinguishable | **PASS, guard did NOT fire** |
| **U2 val proxy** = V3 level | >= 0.6248 | 0.649089 | 0.657692 | **PASS, guard did NOT fire** |
| **U3 uniformization signature** | fires only if gap slope reaches S-bar while U1 or U2 fires | gap slope did not reach S-bar | n/a | vacuous |
| **U4 degeneracy** `response_length/mean` | 100-120 >= 0.60x own 21-40 | **667.6875 vs 768.0017 = 0.8694x** | incumbent 662.7369, so a4 is 1.007x the incumbent | **PASS, guard did NOT fire** |
| **E eligibility** | reached 120, wire <= 1232, `ppo_kl` == 0, counters non-degenerate | 120/120 rows, no gaps; **wire 1232 bits/token/boundary = 1.0000x** (`WIRE_BUDGET.md`); `actor/ppo_kl` = 0 at all 120 steps; `mask_applications` 78274, `spectral_corrections` 34138, `anchor_replay_fires` 6, `rank1_fires` 5 | same codec, same budget | **PASS** |
| **T tie-break** E[rho] 100-120 | higher is better, bonus only | mean 0.003167, median 0.002460 | mean 0.005028, median 0.002057 | not decisive (see note) |

Tie-break note, stated because the two statistics disagree: on the **mean** a4 is 0.630x
the incumbent (worse), on the **median** a4 is 1.196x (better). E[rho] over this window is
a heavy-tailed series (a4 max 0.008036, incumbent max 0.039847), so mean and median rank
the two arms in opposite order and the tie-break decides nothing either way. Both are far
below the registered 0.05 bonus, which the amendment already records as never a veto.

Gate-table cross-check, from `gate93.py` verbatim (Section 1 of `analysis-a4.log`):
`ref_kl_le_baseline True`, `gap_lt_10 False`, `gap_lt_3_target False`,
`reward_slope_parity True`, `ppo_kl_zero True`, `e_rho_gt_0p05 False`. Worth recording
that a4's reference-KL **level** at the gate window, 0.1810, sits inside the registered
baseline band 0.156 to 0.203: the V1 failure is a **rate** finding at the terminal window,
not a level blow-up.

## 3. The late drift reversal, characterised

At an interim read over steps 61-75 a4's reference-KL slope was **+0.002221** against the
incumbent's **+0.002675** on the identical window, **0.83x**, the best of any compressed
arm. By the committed 100-120 window a4 is **+0.003955** against **+0.002176**, **1.82x**,
a veto failure. Both reads are reproduced exactly in `metrics/a4_windows_wandb.json`.

### 3a. Sequential 20-step windows, `actor/kl_loss`, identical windows and code path

| window | a4 slope (HAC3) | incumbent slope (HAC3) | ratio |
|---|---|---|---|
| 2-20 | +0.000073 (0.000029) | +0.000121 (0.000062) | 0.60x |
| 21-40 | +0.000241 (0.000098) | +0.000185 (0.000025) | 1.31x |
| 41-60 | +0.001062 (0.000135) | +0.000969 (0.000106) | 1.10x |
| 61-80 | +0.002205 (0.000054) | +0.002848 (0.000074) | 0.77x |
| 81-100 | +0.001860 (0.000202) | +0.002065 (0.000100) | 0.90x |
| **101-120** | **+0.004054 (0.000099)** | **+0.002158 (0.000163)** | **1.88x** |

### 3b. Who moved

Answered directly, not asserted:

- **a4 accelerated.** Its own 81-100 slope +0.001860 against its own 101-120 slope
  +0.004054 is a difference of +0.002194 with combined HAC SE 0.000225, **z=+9.75**.
- **The incumbent did not decelerate.** Its 81-100 +0.002065 against its 101-120
  +0.002158 is +0.000093 with combined SE 0.000191, **z=+0.49**, flat.

So the ratio flip is entirely a4's terminal acceleration, on a series where the
comparator's rate is stationary over the same 40 steps.

### 3c. Where the crossing happens

Trailing 20-step slope of `actor/kl_loss` (a window labelled H covers steps H-19 to H),
full table in `analysis-a4.log` Section 4a:

| window end | a4 | incumbent | ratio | z on difference |
|---|---|---|---|---|
| 80 | +0.002205 | +0.002848 | 0.774x | -6.93 |
| 90 | +0.001580 | +0.002289 | 0.690x | -3.78 |
| 100 | +0.001860 | +0.002065 | 0.901x | -0.94 |
| 105 | +0.002610 | +0.002376 | 1.098x | +1.13 |
| 110 | +0.003520 | +0.002572 | 1.369x | +3.96 |
| 115 | +0.004106 | +0.002494 | 1.647x | +11.59 |
| 120 | +0.004054 | +0.002158 | 1.879x | +10.68 |

a4 is decisively BELOW the incumbent from window end 75 to 95 (z between -6.93 and
-3.07), crosses between window end 100 and 105, and is decisively ABOVE from window end
110 onward. The onset of fresh acceleration is therefore around **step 100 to 105**.

### 3d. Changepoint, and what it does not support

Continuous piecewise-linear (broken-stick) fits over steps 21-120, breakpoint scanned on
the integer grid 35 to 105, with the SSE profile published in `analysis-a4.log` 4c so the
sharpness of each minimum is visible:

| series | tau* | slope left | slope right | SSE reduction vs one line | is tau identified? |
|---|---|---|---|---|---|
| a4 `actor/kl_loss` | 91 | +0.001370 | +0.003732 | 75.3% | **NO.** SSE is 0.00551 to 0.00632 flat across tau 55 to 95; tau=91 beats tau=60 by 1 percent |
| incumbent `actor/kl_loss` | 53 | +0.000472 | +0.002368 | 92.5% | YES, sharp (0.00111 at 55 against 0.00311 at 65) |
| a4 `rollout_corr/kl` | 59 | +0.011100 | +0.001680 | 90.0% | YES |
| incumbent `rollout_corr/kl` | 59 | +0.010723 | +0.001895 | 86.1% | YES |

Two honest conclusions:

1. **a4's reference-KL drift has no single localisable changepoint.** A two-slope model
   beats one line by 75 percent of SSE, but the SSE surface is flat over a 40-step range
   of tau. Read together with 3a, that is the signature of **continuing convexity**
   (a rate that keeps rising) rather than a discrete event. The incumbent is the opposite:
   one sharp break at tau=53, then a stationary +0.00237 for the rest of the window.
2. **The gap axis is structurally identical in the two runs**: both break at tau=59 with
   nearly the same left and right slopes. So whatever distinguishes a4 lives on the
   reference-KL axis, not on the gap's shape.

### 3e. Structural coincidences, checked and not claimed

`resolved_params-a4.txt` gives `ppo_mini_batch_size` = `train_batch_size` = 128, so there
is exactly **one optimizer tick per global step**, and `anchor_replay_fires`=6 over 120
steps confirms the anchor cadence of 20 ticks lands on steps 20, 40, 60, 80, 100, 120.
`rank1_prediction_horizon`=20 and `rank1_window_span`=20 put the RELEX W2 refresh on the
same grid. The onset at step 100 to 105 therefore **coincides in time with the fifth
anchor and RELEX tick.** I do not claim that as a cause, for three reasons that are all in
the data: the four earlier ticks at 20, 40, 60 and 80 produced no such acceleration (a4
was BELOW the incumbent through window ends 75 to 100); the incumbent runs the identical
tick schedule and shows nothing at step 100; and one seed cannot separate a tick effect
from run-to-run variation. **CVC warmup ended at step 20** (`actor/cvc_lambda` reaches its
0.003 ceiling at step 20 and is constant after), which is 80 steps before the onset, so
the warmup boundary is not the coincidence either.

### 3f. The reversal is on both axes, and it flatters the committed window

The same terminal flip appears on the objective series: `rollout_corr/kl` slope at 100-120
is a4 +0.001542 against the incumbent +0.000838, **1.84x** (trailing-window z=+2.32 at
window end 120), even though the committed 61-120 read is 0.90x. So a4's headline "0.90x
the incumbent on the objective" describes a 60-step window that includes 40 steps in which
a4 was genuinely quieter, and excludes the state the run actually ended in. That is a
property of the committed window, not a criticism of it: the window was fixed blind, and I
am scoring it as committed.

## 4. What CVC did, and the trade

### 4a. It entered the loss exactly as designed

From the logged series alone (`analysis-a4.log` Section 5a), `actor/loss` minus
`actor/pg_loss` reconciles against the two added terms to fp64 rounding (residual under
1.5e-8 at every sampled step):

| step | loss | pg_loss | lambda_eff | cvc_ce | CVC term | kl term | loss - pg |
|---|---|---|---|---|---|---|---|
| 1 | +0.137209 | +0.135043 | 0.00015 | 14.4431 | +0.002166 | +0.00000000 | +0.002166 |
| 20 | +0.140656 | +0.097255 | 0.00300 | 14.4567 | +0.043370 | +0.00003058 | +0.043401 |
| 60 | +0.167651 | +0.124327 | 0.00300 | 14.4219 | +0.043266 | +0.00005836 | +0.043324 |
| 120 | +0.140471 | +0.096894 | 0.00300 | 14.4513 | +0.043354 | +0.00022352 | +0.043578 |

So the warmup ramp is verified (0.00015 = 0.003 x 1/20 at step 1, ceiling at step 20), and
at step 120 the CVC term is **31 percent of the total loss** and **194x the reference-KL
regulariser's contribution**. The configured coefficients say the same thing: `cvc_lambda`
0.003 is **3x** `kl_loss_coef` 0.001, so a reward-blind imitation term entered the
objective with three times the coefficient of the term that protects the reference policy.
CE mode adds `lambda_eff * agg_loss(-log_prob)` with no advantage weighting
(`verl/workers/utils/losses.py` on branch `93-mismatch-control-kit`, the `cvc_lambda > 0`
block), i.e. it pulls the codec view toward **every** sampled token regardless of reward.

### 4b. What it was aimed at, and why that matters

`actor/cvc_ce` equals `rollout_corr/training_log_ppl` to mean |diff| **0.0092 nats**
(max 0.048, corr 0.897, n=120), and `rollout_corr/kl` is `training_log_ppl` minus
`rollout_log_ppl` by construction. So CE-mode CVC descends **the gap's own dominant
term**, which is the right target on paper.

The problem is which term of the gap is actually moving. Over steps 21-120
(`analysis-a4.log` Section 9):

| term | a4 change, step 21 to 120 | incumbent change, same steps |
|---|---|---|
| `training_log_ppl` (CVC's target) | -0.035052 nats | -0.018476 nats |
| `rollout_log_ppl` (untouched by CVC) | -0.798395 nats | -0.801428 nats |

The gap's motion over this span is carried by the **rollout** view sharpening by 0.8 nats.
CVC is attached to the term that barely moves.

### 4c. The counterfactual: CVC produced no measurable descent on its own objective

Matched windows, same code path, HAC(3), on `rollout_corr/training_log_ppl`, which both
runs log:

| window | a4 slope (with CVC) | incumbent slope (no CVC) |
|---|---|---|
| 21-40 | -0.001912 (0.000297) | -0.001548 (0.000288) |
| 41-60 | -0.000794 (0.000385) | -0.000866 (0.000218) |
| 61-80 | +0.000341 (0.000151) | +0.000253 (0.000131) |
| 81-100 | +0.000809 (0.000193) | +0.000945 (0.000321) |
| 101-120 | +0.000774 (0.000190) | +0.000736 (0.000208) |
| 21-120 | +0.000033 (0.000113) | +0.000023 (0.000108) |
| level 21-120 | 14.428726 | 14.426413 (a4 is 0.002313 nats HIGHER) |

Every window agrees inside one HAC standard error, and a4's level is marginally worse.
**The series CVC exists to minimise moved the same way, by the same amount, in the run
that has no CVC.** Its own objective was won slightly in 21-60 (slopes negative) and given
back from 61 onward (slopes positive), which is also exactly what the no-CVC control does.

### 4d. The trade, judged against the cardinal rule

The dispatch framing was a 10 percent gap-slope improvement bought with 82 percent more
drift. The measurement is harsher than that, and the honest statement is:

- the **10 percent gap-slope improvement is not statistically present.** a4 0.001685
  against incumbent 0.001867 on the identical window is a difference of -0.000182 with
  combined HAC SE 0.000201, **z=-0.91**. On one seed per arm that is noise on the same
  trajectory, and 4c gives the reason: CVC did not move its own target.
- the **82 percent drift penalty is decisively present.** +0.003955 against +0.002176,
  **z=+10.20** against the incumbent and **z=+2.90** past the committed ceiling.

Against the project's cardinal rule that stability and capability preservation outrank raw
reward, the judgement is not close. a4 spent the only axis the rule protects (drift away
from the reference policy) and bought nothing measurable on the axis it was aiming at.
Even had the 0.90x been real, trading a 10 percent gain on an instrument that has never
been tied to capability for a decisively worse reference-KL rate is the wrong direction
under that rule. **The CVC route is closed at this setting.** No reward damage was
observed (V3 level 0.987x the incumbent, z=-1.03), so this is not a collapse; it is a
mechanism that failed to work while making the protected quantity worse.

### 4e. The predicted failure mode did not happen, and this is what happened instead

`A4_GUARD.md` predicted flattening: a near-uniform policy looks the same through any
codec, so uniformization was the expected cheap route to a better gap. It did not occur.
Entropy slope over 21-120 is +0.000093 +- 0.000004 against the incumbent's +0.000097 +-
0.000004, a difference of z=-0.7, and the entropy **level** differs by 0.0001 nats
(7.812644 against 7.812547). Response length is 0.8694x its own early window and 1.007x
the incumbent. Score level passed. **a4's entropy trajectory is indistinguishable from the
incumbent's**, which is a stronger statement than merely clearing a kill bar 16x.

It is worth being precise about why, because the tempting explanation is wrong. Under
CE mode the flattening route was genuinely **available**: a uniform distribution scores
ln(151936) = **11.93 nats** of NLL on any token, against a4's measured `cvc_ce` of
**14.44 nats**, so flattening toward uniform would have reduced the CVC term by about 2.5
nats. The guard's reasoning was sound and the route was profitable in principle. It simply
was not taken at `cvc_lambda`=0.003 over 100 post-warmup steps at lr 1e-6. Which of the
candidate reasons applies (the policy-gradient and reference-KL directions dominating the
entropy coordinate, or 100 steps being too few to move a 7.8-nat entropy at this learning
rate) **is not resolvable from these series**, and I am not claiming one.

What I can support is the negative and one positive: CVC in CE mode at lambda 0.003 was
**measurably inert on its own objective** (4c) and **inert on entropy, response length and
reward** (this section), while the arm's terminal reference-KL rate is decisively worse
than the matched no-CVC control (3b). The single-knob configuration contrast makes CVC the
leading candidate for that last difference, but with n=1 per arm, different data shuffles
(`data.seed=None`, `shuffle=True` in both) and different physical boxes, a single seed
cannot establish that CVC **caused** the terminal acceleration. That claim needs a
replicate and is not made here.

## 5. The gap-level clause is a tie, and is still scored as committed

a4's 100-120 gap level is **14.247296** against a bar of **14.2458**. That is an excess of
**0.001496 nats, 0.0105 percent**. The bar was set at the incumbent's own measured level,
and the like-for-like test says the two are the same number: difference +0.001509 nats,
se_diff 0.003657 (a4 sd 0.013538 and incumbent sd 0.009875 over n=21 each), **z=+0.41**.

Said plainly: **this clause is a numerical tie with the incumbent, not a defeat.** It is
scored FAIL because the clause was committed blind as an inequality and a tie is not
below the bar, and because reporting a tie as a pass would be exactly the kind of
after-the-fact reading the blind commitment exists to prevent. It carries no independent
weight in the verdict: the objective already fails on the slope clause at z=+10.60, and
the veto already fires at z=+2.90.

## 6. Which ladder row applies

Evaluating `AB_AMENDMENT.md` section 3 in order, on what is measured now:

| row | trigger | status |
|---|---|---|
| 0 | a3 gap level <= 10.0 | FALSE, a3 level 14.9924 |
| 1 | a4 gap slope <= +5.0e-4, vetoes clean, guard clean | **FALSE on two limbs**: slope +0.001685 is 3.37x the bar, and V1 fires |
| 2 | a4 not settling; a5 alive with E[rho] in [0.2, 2.0], vetoes clean, gap slope <= +1.867e-3 | first limb **TRUE**; the rest is unread, a5 is training |
| 3 | a4 trips its uniformization KILL; a5 in the row-2 quadrant | **FALSE**, U1, U2 and U4 all clean |
| 4 | a4 KILL; a5 clean but E[rho] < 0.2 | **FALSE**, no a4 KILL |
| 5 | a4 flat (slope in (5.0e-4, 1.867e-3]) AND veto-clean | slope +0.001685 **IS** in the band; **veto-clean is FALSE** (V1 at 100-120), so the row does not fire |
| 6 | no arm clears the drift veto, or every arm is rising | first limb FALSE (a3 cleared all three vetoes); second limb unread until a5 |
| 7 | grad_norm max > 10, or entropy/reward crash | FALSE, run max 3.6216, entropy flat, reward at parity |

**No row resolves on a4 alone. The outcome falls through to a5: row 2 if a5 lands in the
coherent-and-corrected quadrant, row 6 as the registered clean negative otherwise.** The
ladder handled a4's actual result without needing amendment, which is the whole point of
having committed it blind.

Two observations that make the point concretely rather than rhetorically:

- Row 5 is the row a4 nearly fired. Its slope band was chosen blind and a4's slope landed
  inside it. The **only** thing keeping a4 out is the V1 veto, and V1 was committed with
  **both** window ceilings (3.52e-3 at 61-120 and 3.26e-3 at 100-120) before a4 ran. On
  the 61-120 window alone a4 is veto-clean at 1.07x, and row 5 would have fired and
  authorised an optional b1 bounded-wedge test at parity wire. The 100-120 window is the
  one a1 was judged on, so it is the consistent read, and it is the read that catches the
  terminal acceleration. **The window choice was load-bearing, and it was fixed in
  advance.**
- Rows 3 and 4 both required a4 to trip its uniformization kill, and it did not. The
  guard's job was to prevent a4 being crowned for flattening; instead the arm failed for a
  reason the guard did not anticipate, and the ladder still routed it correctly, because
  the veto set does not depend on the guard.

## 7. Provenance and plan-versus-ran

Contract item 3 was run: `python3 scripts/capture_resolved_config.py
runs/93-long-horizon-stability` exits 1 with `no train.log at
runs/93-long-horizon-stability/train.log`. `metrics/incoming.log` carries no
`python3 -m verl.trainer.main_ppo` line (grep count 0) and no launcher
`=== resolved #93 cell config ===` banner (grep count 0). **Flag
`RESOLVED_CONFIG_MISSING`** for the `set -x` path, as for a1, a2 and a3.

Substitute, and a stronger one: the WandB run config of `8rux5ea6` is the fully resolved
OmegaConf the trainer instantiated. Every flattened key is dumped to
`resolved_params-a4.txt` (821 keys by the writer's count; 815 lines match a strict
`key=value` regex because a few values contain newlines). a4 is the first #93 cell with a
machine-verified full config.

Excerpt of the load-bearing keys:

```
actor_rollout_ref.actor.comm_eff.compression_type=prf_mask
actor_rollout_ref.actor.comm_eff.mask.enabled=True
actor_rollout_ref.actor.comm_eff.mask.p=0.95
actor_rollout_ref.actor.comm_eff.mask.exact_k=True
actor_rollout_ref.actor.comm_eff.mask.rescale=False
actor_rollout_ref.actor.comm_eff.mask.pp_size=8
actor_rollout_ref.actor.cvc_lambda=0.003
actor_rollout_ref.actor.cvc_warmup_steps=20
actor_rollout_ref.actor.kl_loss_coef=0.001
actor_rollout_ref.actor.kl_loss_type=low_var_kl
algorithm.rollout_correction.rollout_is=None
actor_rollout_ref.actor.comm_eff.anchor.cadence=20
actor_rollout_ref.actor.comm_eff.anchor.delay_K=20
actor_rollout_ref.actor.comm_eff.anchor.batch_scope=rollout_batch
actor_rollout_ref.actor.comm_eff.anchor.lookahead_mode=rank1_relex
actor_rollout_ref.actor.comm_eff.anchor.lookahead_window_snapshots=2
actor_rollout_ref.actor.comm_eff.spectral.beta_anc=0.25
actor_rollout_ref.actor.comm_eff.spectral.signed_ema_alpha=0.25
data.train_batch_size=128
actor_rollout_ref.actor.ppo_mini_batch_size=128
trainer.total_training_steps=120
trainer.test_freq=-1
trainer.save_freq=-1
trainer.val_before_train=False
```

Divergences and facts worth recording, none of which changes the verdict:

1. **No `cvc_mode` key exists in the resolved config.** CE mode is established from source
   plus the logged metric name: the branch adds `lambda_eff * agg_loss(-log_prob)` and
   logs it as `cvc_ce`, and the run logs `actor/cvc_ce`. DC mode is off
   (`comm_eff.dc.enabled=False`). So CE is the only active CVC path, but the mode is
   inferred rather than read from a knob. Worth a knob for the next cell that uses CVC.
2. **`comm_eff.powersgd.enabled=True` with `rank=77` while `compression_type=prf_mask`.**
   The PowerSGD block is populated but inert: the run logs only
   `actor/comm_eff/logical_pp_bytes_prf` and no PowerSGD or sr_quant wire field appears in
   its 160 summary keys, and all `anchor_q_*` counters are 0. The incumbent's resolved
   config carries the identical pattern, so this is a launcher default, not an a4 anomaly.
3. **`anchor.owns_q=False`** differs from the documented project default of anchor-owned
   `Q`. It is inert under a mask codec: `anchor_q_updates`, `anchor_q_broadcasts`,
   `anchor_q_activations` and `anchor_q_stage_overwrites` are all 0. The incumbent is
   `owns_q=False` too, so the matched comparison is unaffected.
4. **`mask.rescale=False`**: a4's codec is biased by construction, identical to the
   incumbent. This is the arm's design, recorded so the bias axis is not later attributed
   to CVC.
5. **`ppo_mini_batch_size` == `train_batch_size` == 128**, so anchor cadence in optimizer
   ticks equals cadence in global steps for these cells. This retires the known
   ticks-versus-steps confound for round A.
6. **`test_freq=-1`, `val_before_train=False`, `save_freq=-1`**: a4 took no validation and
   saved no checkpoint. Its only capability read is the train-side proxy
   `critic/score/mean`. There is **no held-out capability measurement for a4 and there
   never can be**, since no checkpoint exists. That limit belongs on any statement about
   a4 and capability preservation, including mine in 4d.

## 8. next_actions

Three, ordered, none of which re-thresholds anything committed.

1. `{knob: actor_rollout_ref.actor.cvc_lambda, from: 0.003, to: 0.0, rationale: "CE-mode
   CVC produced no measurable descent on its own objective (matched-window
   training_log_ppl slopes agree with the no-CVC incumbent inside 1 HAC SE across all five
   windows, and a4's level is 0.0023 nats higher) while the arm's terminal reference-KL
   rate is 1.82x the incumbent at z=+10.20. Do not carry CVC into b1 or round C. If CVC is
   ever revisited it should be in DC mode against the term that actually moves, the
   rollout view, not the static training view."}`
2. `{knob: "b1 arm selection, run.json cells[5] b1-<winner>-ctrl", from: "<winner>
   unresolved", to: "a5 if it meets the row-2 quadrant (E[rho] in [0.2, 2.0], vetoes
   clean, gap slope 61-120 <= +1.867e-3), else the registered round-A clean negative under
   row 6 with a3 named as the only veto-clean arm", rationale: "ladder rows 0, 1, 3, 4, 5
   and 7 are all resolved FALSE by a4's measured result, so the branch is entirely a5's.
   Both remaining rows escalate to the operator for money and for ratification of the
   amendment, so nothing launches on this verdict."}`
3. `{knob: "veto reporting protocol for the remaining arm", from: "single committed window
   per clause", to: "report V1 on BOTH committed windows (61-120 and 100-120) plus the
   trailing-20-step slope at the final step, for a5 and for any b1 cell", rationale: "a4
   would have read veto-clean on 61-120 alone at 1.07x and fired ladder row 5; it fails at
   1.82x on 100-120. Both ceilings were committed in AB_AMENDMENT.md section 4, so this
   changes no threshold, it only stops a terminal acceleration hiding inside a 60-step
   average. a4's trailing-window series crossed from 0.90x to 1.88x in 20 steps."}`

## 9. What a5 must show for a positive round-A result, under the committed criteria

a5 (`a5-frlr-r48k28-tis`, FRLR rank 48 k 28 plus decoupled token-IS 2.0) is the last arm
and the only remaining path. Under the already-committed bars it must clear **all** of:

1. **E, eligibility**: reach step 120; wire <= 1232 bits/token/boundary. Per
   `WIRE_BUDGET.md` a5's payload is 48 + 28 + 1 = 77 coordinates = 1232 bits, exact
   parity, **but the field `logical_pp_bytes_prf` will read 77, not 1232**. Multiply by 16
   before any comparison, or the arm will look 16x cheaper than it is. Also
   `actor/ppo_kl` == 0 or explained: a5 is the one arm where a non-zero value is expected
   to need explaining, since token-IS reweights the update.
2. **V1 drift**: `actor/kl_loss` slope <= 1.5x the incumbent on **both** committed
   windows, i.e. <= 3.516e-3 at 61-120 **and** <= 3.264e-3 at 100-120. This is the clause
   that killed a1 (1.79x), a2 (48.5x) and now a4 (1.82x). It is the binding constraint of
   round A. Token-IS is the mechanism with a prior claim on it (roughly 40 to 78 percent
   KL reduction in earlier work), so a5 is the arm designed to clear it.
3. **V2 gradient**: run max <= 10.0 and 61-120 mean <= 3.62.
4. **V3 learning**: `critic/score/mean` level at 100-120 >= 0.6248, with a significantly
   positive 61-120 slope.
5. **O objective, for row 1**: `rollout_corr/kl` slope over 61-120 <= **+5.0e-4** with
   level at 100-120 <= 14.2458. Row 1 is the only row that produces an unqualified
   positive.
6. **O objective, for row 2** (the realistic positive): gap slope 61-120 <= **+1.867e-3**,
   the incumbent's own creep, **plus** `E[rho]` at 100-120 in **[0.2, 2.0]**. Note how far
   that E[rho] requirement is from anything measured so far: the incumbent sits at 0.00503
   mean and a4 at 0.003167, roughly **40 to 60 times below** the bottom of the row-2 band.
   Row 2 is a claim that a5's IS correction restores mean importance weight to order 1,
   which no arm in this program has yet demonstrated. If a5's E[rho] lands under 0.2 with
   vetoes otherwise clean, the ladder sends it to **row 4** (I4 in DC mode for round B),
   not to row 2.
7. And, from a4's result specifically: a5 must clear V1 **at the terminal window**, not
   only on the 60-step average, and its trailing-20 slope at step 120 should not be
   accelerating away from the incumbent's. a4 passed 61-120 and failed 100-120. Check both
   before any b1 recommendation.

If a5 fails V1, round A closes as the registered **clean negative** (row 6), which the
issue body already labels a form of PASS: at 1232 to 2304 bits the wedge grows without
bound in every mechanism family tested (1-bit SR, 1-bit RN, 2-bit subset parity, CVC on
the incumbent codec, FRLR plus token-IS), which closes the activation-precision axis
quantitatively. In that case a3 is the arm to name as the only veto-clean configuration,
and the OOD read comes off the incumbent's R2 checkpoints.

## 10. Round-A standing after a4

| arm | E, wire | V1 drift (100-120) | V2 grad | V3 learn | O slope 61-120 | O level 100-120 | standing |
|---|---|---|---|---|---|---|---|
| incumbent `90-prf-exactk-600` | 1232, 1.0000x | +0.002176 (1.00x) | 1.870 mean | 0.657692 | +0.001867 (3.73x bar) | 14.245787 | reference |
| a1 `a1-srq-b1-sr` | 2304, **1.8701x FAIL** | +0.003887, **1.79x FAIL** | pass | pass | +0.003265 (6.5x bar) | 13.7511 | out on E and V1 |
| a2 `a2-srq-b1-rn` | died at step 62, **ineligible** | **48.5x FAIL** | **62.238 FAIL** | n/a | +0.017478 (35x bar) | 12.1851 | killed under the pre-authorised rule |
| a3 `a3-srq-parity-k493` | 1232.5, **0.5 bits over** | +0.001616, 0.743x **PASS** | 7.283 / 2.892 **PASS** | 0.656855 **PASS** | +0.001300 (2.60x bar) **FAIL** | 14.9924 **FAIL** | **the only arm to clear all three vetoes; current leader on the lexicographic rule** |
| a4 `a4-prf-exactk-cvc-ce` | **1232, 1.0000x parity** | **+0.003955, 1.818x FAIL** | 3.6216 / 1.838 **PASS** | 0.649089 **PASS** | +0.001685 (3.37x bar) **FAIL** | 14.247296 **FAIL by 0.0105 percent, a tie** | out on V1; objective fails both clauses at exact wire parity |
| a5 `a5-frlr-r48k28-tis` | 1232 by design | running | running | running | running | running | last arm, the only remaining path to a positive round A |

Note for whoever writes the round-A close: a4 and a3 fail the objective in different ways
and the difference is informative. a3 has the better gap slope (0.696x the incumbent,
significant at z=-2.44) but a level 0.75 nats too high, most of which is a step-1
instrument offset. a4 has the level (tied to 0.01 percent, being the incumbent's own
codec) but no significant slope improvement. **Nothing tested so far has both**, and the
two failures are on opposite sides of the two-sided objective, which is itself the
strongest available support for the amendment's decision to make the objective two-sided.

## 11. Notes

- **Completion.** No `done.flag` in the run dir, and the `tmux` limb of the completion
  test cannot be applied per-cell because session `run-93` is alive with a5. Completion is
  established from the cell's own record: WandB `8rux5ea6` state `finished`, max_step 120,
  120/120 history rows with **no missing steps**, `total_training_steps=120` in the
  resolved config, `actor/comm_eff/spectral_step=120` at the last row. Metrics are
  non-empty (`metrics/a4_gate.json`, `metrics/a4_windows_wandb.json`).
- **No box access.** Nothing in this verdict touched the GPU box; a5 is training. All
  series were pulled read-only through the WandB API.
- **`RESOLVED_CONFIG_MISSING`** is flagged for the `set -x` path (section 7), mitigated by
  the full resolved WandB config.
- **Uncertainty discipline.** Residual lag-1 autocorrelation on these series runs from
  -0.29 to +0.98 depending on window (0.87 on a4's `actor/kl_loss` at 61-120, 0.03 on
  `rollout_corr/kl` at 61-120), so every standard error quoted is Newey-West HAC, never
  iid. Bandwidth is L=3 for all headline figures, which is what reproduces the committed
  figures; L=4 is also in `metrics/a4_windows_wandb.json` for every window and moves no
  conclusion (for example the O slope SE is 0.000112 at L=3 and 0.000122 at L=4).
- **Horizon caution, not a claim.** The incumbent's 600 steps are measured:
  `actor/kl_loss` 0.203385 at 120, 0.460520 at 300, 0.928460 at 600, with slope +0.001571
  over 100-600 and +0.000344 over 580-600, i.e. **the incumbent's drift decelerates over
  the long horizon**. a4's terminal rate of +0.004054 is 2.58x the incumbent's 100-600
  average and 11.8x its 580-600 rate; if it persisted, `kl_loss(600)` would be about 2.17
  nats against the incumbent's measured 0.928. The incumbent's own trajectory shows such
  rates do not persist, so this is a caution about the terminal state, not a projection.
  Both arms' linear gap(600) extrapolations from 61-120 (a4 15.077, incumbent 15.164)
  overshoot the incumbent's measured 14.658 by about 0.5 nats, so 60-step gap
  extrapolations should not be quoted as predictions.
- **Gap accumulation deliberately not used.** Anchored on the 2-20 window mean, a4
  accumulates 0.472602 nats against the incumbent's 0.496287 (0.952x, a4 better); anchored
  on the step-1 value, 0.4660 against 0.3600 (1.294x, a4 worse). The two anchorings
  disagree in the sign of the conclusion, so no accumulation claim is made in either
  direction. Recorded so that a later session does not discover one of the two and read it
  as a finding.
- **Single seed.** One run per arm. The a4-versus-incumbent contrast is single-knob on
  every load-bearing key, which is the cleanest contrast in this program, but different
  data shuffles (`data.seed=None`), different physical boxes and different dates remain,
  and causal attribution of the terminal acceleration to CVC is therefore stated as
  leading candidate, not as established.
- **No re-verification.** One pass, as contracted. `gate93.py` was run verbatim as the
  plan's per-cell verification command; the plan's `analyze.py --emit verdict.md` pass is
  registered for round C only and was not run.
- **Doubt register.** None material enough for `MANUAL_REVIEW_NEEDED`. The two judgement
  calls a human may want to look at are both stated explicitly and scored conservatively:
  the gap-level clause is a statistical tie scored as FAIL (section 5), and V1 passes on
  one committed window and fails on the other with the failing one taken as decisive
  (section 6, first bullet).
