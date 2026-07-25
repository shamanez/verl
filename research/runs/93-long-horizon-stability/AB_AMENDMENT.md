# A-to-B amendment, committed BLIND (a3 at step 20, a4 and a5 have reported nothing)

Posted deliberately now rather than at the boundary. Every threshold below is fixed before the data that would be used to fit it exists. Once a4 and a5 land, no row can be added, moved or re-thresholded. If any number here changes after a4 reports, that change is illegitimate and this comment is the evidence.

Nothing launches on the back of this. It requires operator ratification.

## 1. Two corrections to my own earlier framing in this thread

**Correction 1: I said the sub-10-nats gate "may be mis-specified because the incumbent itself fails it". The fact is true; the inference was wrong, and in the operator's favour.**

The issue body registers the gap criterion as a **disjunction, in two separate places**:

- Section 0 (the end goal): "a train-inference gap that either **settles** on ..."
- Section 7, success criterion 1: "train-inference gap **settles** (slope -> ~0 vs the baseline's **+0.0005/step** creep) **or** sits < 3 nats with E[rho] > 0.05"

So reading the gap on its **slope against +0.0005/step** is the registration's own primary form of the criterion. The absolute "< 10 nats" appears only in the machine-readable stage-1 gate block. Electing the slope form is therefore **choosing between two registered readings, not moving a goalpost**. That is a materially different situation from the one I described twice earlier, and it is the single most important finding of this analysis.

For the record on provenance, since a sub-analysis got this wrong and I repeated part of it: the string "14.24" appears **zero** times in the issue body. The body reports the incumbent's gap as "13.88 -> 14.66". The 14.24 figure is a harness constant at `research/scripts/gate93.py:36`, written mid-a1. The number is right, the "same sentence as the threshold" provenance claim was not.

**Correction 2: my claim that "gap and drift can move in opposite directions" was an artifact of reading the gap LEVEL rather than its slope.**

Read as slopes, gap and drift are **rank-concordant across all four measured configurations (Spearman 1.0). There is no dissociation.** a2's apparently attractive level was an intercept effect: round-to-nearest is the MSE-minimising rounding at fixed bits, so it shrinks the *measured* level by construction (intercept 11.298 versus a1's 13.011), while a2's gap **slope** is **+1.7478e-2**, which is 1.98x a1's +8.805e-3 and 2.26x the incumbent's +7.741e-3.

So the winner rule does **not** need the gap demoted. It needs a **derivative term on the gap axis**, which the registration's own "settles" language already supplies. That is a much less invasive repair than the one I floated.

## 2. The committed replacement threshold

> **S-bar = +5.0e-4 nats/step.** An arm PASSES the gap criterion if the OLS slope of `rollout_corr/kl` over **steps 61-120** (n=60, HAC(4) standard error, one-sided) is **<= +5.0e-4**, with mean level at steps 100-120 **<= 14.2458** and wire budget **<= 1232** bits/token/boundary.
>
> "< 10 nats" and "< 3 nats" are retained as **reported stretch targets**, with the measured 6.5x to 8.5x wire-budget price of reaching them attached.

Four independent anchors, none of which depends on a3, a4 or a5:

1. **It is already written in the pre-registration** ("versus the baseline's +0.0005/step creep"). Electing a number the registration itself wrote down is the least data-conditioned choice available.
2. **It is a tightening, not a relaxation.** It is 0.27x the incumbent's measured 61-120 creep of +1.867e-3, so **the incumbent fails it by 3.7x**. A rebaseline to "beats 14.24" would have been a free pass to the status quo; this is not.
3. **It is resolvable.** It sits 2.9 HAC standard errors from the incumbent's own slope estimate (se 1.751e-4 at n=60), so an arm at S-bar is statistically distinguishable from both zero and from the incumbent.
4. **It has horizon meaning.** It bounds gap(600) at about 14.45 by linear extrapolation, against the incumbent's measured 14.658 and a1's projected 18.3.

**Passing means** compression-induced train/inference mismatch is bounded and does not compound over the 600-step horizon at the deployment wire budget. That is the claim the deployment premise actually needs. **Failing means** that at 1232 to 2304 bits the wedge grows without bound in every mechanism family tested, which together with the elasticity result closes the activation-precision axis quantitatively. The issue body already registers that negative as itself a form of PASS.

## 3. Pre-registered decision ladder (apply the first row whose trigger is true)

Costs are incremental GPU-h from this boundary; add 9 to 22 percent at the box's observed 112 to 137 s/step.

| # | Trigger | Meaning | Action | GPU-h | Escalation |
|---|---|---|---|---|---|
| 0 | a3 gap level <= 10.0 at 1233 bits, drift and grad vetoes clean | The literal registered gate is satisfiable at parity budget; the elasticity extrapolation was wrong | b1 on a3, no amendment needed | 6.2 | money read only |
| 1 | a4 gap slope (61-120) <= +5.0e-4, vetoes clean, uniformization guard clean | CVC turns the wedge over; the registered "settles" disjunct is MET | b1 on a4 plus control plane | 6.2 | ratify amendment plus money |
| 2 | a4 not settling; a5 alive with E[rho] in [0.2, 2.0], vetoes clean, gap slope <= +1.867e-3 | The never-tried coherent-and-corrected quadrant; IS safety net restored | b1 on a5 plus token-IS | 6.2 | ratify plus money |
| 3 | a4 trips its uniformization KILL; a5 hits the row-2 quadrant | CVC route closed, correction route open | b1 on a5 | 6.2 | ratify plus money |
| 4 | a4 KILL; a5 clean but E[rho] < 0.2 | The registered contingency fires verbatim ("switch I4 to DC mode (b) for round B") | b1 with I4 in DC mode on the best veto-clean codec | 6.2 | REVISE plus needs:human |
| 5 | a4 flat (slope in (5.0e-4, 1.867e-3]), veto-clean; a5 dead or low E[rho] | Wedge bounded at incumbent creep but not turning; a partial mechanism result | Judgment-fallback REVISE. Optional b1 as a bounded-wedge test, ONLY at wire <= 1232 | 6.2 or 0 | REVISE plus needs:human, operator's call |
| 6 | No arm clears the drift veto, or every arm is rising | **Registered CLEAN NEGATIVE**, which the issue body explicitly labels a form of PASS | STOP at round A, publish the negative, OOD read off the incumbent's R2 checkpoints | 1.0 | STOP plus teardown authorization |
| 7 | Any arm: grad_norm max > 10, or entropy/reward crash | Collapse signature | Kill in flight, no b1 | 0 | flag only |

**Round C is authorized by no row.** It is a separate sign-off after b1 reports.

## 4. Revised winner criterion, lexicographic

All reads matched-window against the incumbent's own log, never cross-window. (That discipline is not theoretical: a cross-window comparison is exactly what produced my a2 error earlier in this thread.)

- **E, eligibility:** reached step 120; wire budget <= 1232 for any round-C promotion; `actor/ppo_kl` == 0 or explained; confinement counters non-degenerate.
- **V1, drift veto:** `actor/kl_loss` slope <= 1.5x the incumbent's on the same window (ceilings **3.52e-3** at 61-120, **3.26e-3** at 100-120). **Slope only, never level**, because a1's fitted intercept is 1.8765 nats of stochastic-rounding view offset, corroborated by a2's step-2 value of 0.639 against a1's 1.909, and every sr_quant arm carries a different offset.
- **V2, gradient veto:** run max <= **10.0** and window mean <= **3.62**. Calibration note: matched-window grad_norm is 0.0543 for dense and 1.808 for the incumbent, a 33x spread between two healthy runs, so this is a tripwire for a2-class blowups (62.238), not a fine-grained health score.
- **V3, learning veto:** `critic/score/mean` **LEVEL** at 100-120 >= 0.95 x 0.65769 = **0.6248**, plus a significantly positive 61-120 slope. **Level, not slope**, for two measured reasons: at 100-120 all three reference runs have *negative* reward slope (incumbent -1.422e-3, dense -7.25e-4, a1 -1.074e-3), which makes a slope bar sign-degenerate there; and **the dense control itself fails the registered 0.00288 full-run parity bar** (2-120 slope +1.910e-3, HAC se 3.075e-4). A parity bar that the uncompressed control fails is not a usable bar.
- **O, objective (two-sided gap):** minimise gap slope (61-120) against the S-bar pass line, then minimise gap level (100-120) subject to <= 14.2458.
- **T, tie-break:** higher E[rho]. Note the incumbent sits at 0.00503 at 100-120, so the registered "E[rho] > 0.05" is failed by the incumbent too and is a bonus, never a veto.

Ranking of what is already measured:

| config | E | V1 drift | V2 grad | V3 learn | gap slope | gap level |
|---|---|---|---|---|---|---|
| dense 90-dense-600 | anchor, no compression | 1.64e-5 (0.008x) | 0.054 / 0.066 | 0.6587 | **-1.23e-6** | 2.42e-4 |
| incumbent PRF exact-k | pass, 1232 bits | 2.176e-3 (1.00x) | 1.808 / 2.486 | 0.6577 | +1.867e-3 (3.7x bar) | 14.2458 |
| a1 (1-bit SR) | **FAIL**, 2304 bits = 1.87x | **FAIL**, 3.887e-3 = 1.79x | pass, 0.649 / 0.898 | pass, 0.6529 | +3.265e-3 (6.5x bar) | 13.7511 |
| a2 (1-bit RN) | **FAIL**, died at 62 | **FAIL**, 48.5x | **FAIL**, 7.574 / 62.238 | pass on level | +1.7478e-2 (35x bar) | 12.1851 |

**The a2 counterexample is answered with the gap kept primary:** the two-sided criterion rejects a2 **by 35x on the gap axis itself**, no reordering required. The vetoes are belt-and-braces, rejecting it independently by 48.5x on drift and firing on gradient at step 1 rather than step 60.

On a1's eligibility: the registration's criterion 4 permits "wire budget <= incumbent **or an explicit accounting of the trade**", so 2304 bits is not an absolute bar. But a 1.87x-budget arm cannot carry a communication-efficiency headline claim, and that should be stated rather than finessed.

## 5. What proceeds without the operator, and what does not

**Autonomous** (no new science decision, inside the approved fixed matrix): let a3 finish; launch a4 then a5 as registered; run `gate93.py` **unmodified** and publish the literal five-arm flag table including every FALSE alongside the revised matched-window table, both clearly labelled; launch nothing past a5.

**Needs operator sign-off:**

1. **Ratifying this amendment**: demoting "gap < 10 nats" in the machine-readable gate to a reported stretch target, and electing S-bar as the STOP-triggering clause.
2. **Any b1 launch** (6.2 h, about $21), which the registration escalates at this boundary regardless of verdict.
3. **Round C** (21.5 h including OOD, about $72).
4. **Optional val@120 on a4 and a5** (0.2 h each). Round A saves no checkpoints, so this is now-or-never at launch time, and it is the only direct capability measurement round A can still produce. The launcher knob for it is now in place (`8e83adcc`, default unchanged). If taken it must be pre-declared **diagnostic-only and not a ranking input**, since a1 to a3 cannot have it and applying an unregistered criterion to a subset of arms would be worse than not measuring at all.
5. **Teardown of box 45725398.** No standing authorization exists for this box; the 2026-07-19 grant covers 44955290 and 45287179 only.

Ledger if items 2 and 3 are taken: 6.6 burned, plus about 12.5 to finish round A, plus 6.2 for b1, plus 21.5 for round C, landing near **47 of the 100 h cap**.

## 6. Dissent, stated fairly

The strongest surviving argument against all of the above is not procedural: **neither the old bar nor my new one has ever been tied to capability.** No experiment in this program links gap level or gap slope to held-out accuracy. This program's own OOD dose-response found a compressed arm at step 100 matching dense at step 100 on all 10 benchmarks despite roughly 1000x the reference KL, with damage appearing only at collapse. If the wedge is causally inert before collapse, the old bar was measuring an instrument and **so is mine**, and the honest move is to stop, publish, and re-register with a capability-linked gate.

The cost half of that case is also sound. The incumbent's val spread over steps 150 to 600 is 0.012, so one seed cannot resolve a val gain below roughly 0.012 to 0.02. Round C is therefore a **non-inferiority certificate plus drift and gap curves, not a superiority test**, and that is a thin deliverable against a config that already has 600/600 clean.

I recommend against that position only because the registered hypothesis and success criteria both state the gap criterion in the settling form, so the absolute reading was never the sole registered claim. But item 4 above (val@120 on a4 and a5) is the cheapest available partial answer to it, which is why it is offered rather than quietly skipped.
