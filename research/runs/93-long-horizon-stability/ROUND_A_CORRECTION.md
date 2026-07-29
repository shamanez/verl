# CORRECTION to the round-A boundary escalation above: my recommendation used the wrong knob, and I overstated the collapse case

The a5 verdict landed after I posted the escalation and it corrects me on two material points. Both change the recommendation. The five-arm table, the "no winner" conclusion, and the bias mechanism all stand.

## Correction 1: the token-IS threshold is the WRONG KNOB, so my recommended probe would have done almost nothing

I recommended a 120-step probe at IS threshold 4.0, reasoning that widening the cap would admit more gradient signal. That reasoning is wrong, and the run's own distribution shows why.

At steps 100-120:

| quantity | value |
|---|---|
| tokens AT the cap (`ratio_fraction_high`) | **0.33 percent** |
| tokens at the LOW end (`ratio_fraction_low`) | **88.38 percent** |
| mean IS weight | **0.166** |
| median ratio | 34.4 (log 3.539), i.e. the bulk sits far from the cap |
| ESS | 0.268 |

**Only 0.33 percent of tokens are at the cap, so raising it 2.0 to 8.0 buys at most about 1.11x mean weight.** The 12.55x gradient attenuation is not caused by clipping at the top; it is the **low tail** dragging the mean weight to 0.166.

**The right knob exists and was switched off.** `algorithm.rollout_correction.rollout_is_batch_normalize`, which `verl/trainer/ppo/core_algos.py:2409` documents as "whether to normalize IS weights to mean=1.0", **defaults to `False`** (confirmed in `algorithm.py:177` and all four generated trainer YAMLs). Self-normalising divides by the mean weight and therefore restores about **6.03x** of the 12.55x attenuation directly.

**Revised recommendation: the probe cell should set `rollout_is_batch_normalize=True`, not raise the threshold.** Same cost, about 4.0 GPU-h, same three riders (checkpoints at 0/60/120; actor-vs-ref KL logged through the anchor's existing paired dense replay; step-120 val plus 2 OOD). The registered promote bar and falsifiers from the escalation above carry over unchanged.

## Correction 2: a5's drift LEVEL is BELOW the incumbent's, and my collapse extrapolation was stated with more confidence than the data support

My escalation leaned on the drift SLOPE and did not report the LEVEL. The level is the realized outcome, and on it a5 is **better than the incumbent at every measured window**:

| window | a5 ref-KL | incumbent | ratio |
|---|---|---|---|
| 61-80 | 0.032237 | 0.086025 | **0.375x** |
| 81-100 | 0.077708 | 0.133153 | **0.584x** |
| 100-120 | 0.149579 | 0.178583 | **0.838x** |
| step 120 exact | 0.199350 | 0.203385 | **0.980x** |

So a5 sits **closer to the base model than the incumbent** throughout, though the ratio is converging on 1 and the fitted crossing is about step 122, just outside the window. **The V1 veto fired on a leading indicator, not on a realized harm.** That belongs in the record and I omitted it.

And the saturation question is genuinely open, which undercuts my extrapolation:

| window | incumbent slope | a5 slope |
|---|---|---|
| 21-40 | +0.000185 | +0.000225 |
| 41-60 | +0.000969 | +0.000510 |
| 61-80 | **+0.002848 (peak)** | +0.001640 |
| 81-100 | +0.002065 | +0.002982 |
| 101-120 | +0.002158 | +0.004619 |

**The incumbent's slope peaks at 61-80 and then settles**, and its actual 600-step trajectory is roughly linear and benign: 0.19 at 120, 0.31 at 200, 0.45 at 300, 0.62 at 400, **0.91 at 600**, never approaching the 3-8 nat collapse band. a5's slope is still accelerating monotonically with no sign of saturation through step 120.

So the honest statement is conditional, not the flat claim I made: **if a5's acceleration persists it crosses 3 nats near step 334 (naive t^2.65 gives 0.77 at step 200 and 2.26 at step 300); if it saturates the way the incumbent's did after step 80, it may not.** Distinguishing saturation from overshoot is not resolvable from 120 steps, and extrapolating a two-point power law 2.8x beyond the data is fragile. I should have said that the first time.

## What survives against a5, and it is not nothing

Correcting the level and the extrapolation does not clear a5, because the small-step story does not exonerate its drift once you match on progress rather than step. a5 is behind the incumbent on learning, so a matched-step comparison flatters it. Progress-matched (a5's steps 101-120 at score 0.5895 against the incumbent's 61-80 at 0.6068):

- a5 shows **1.62x the drift slope** and **1.76x the drift level**
- drift per unit learning is **1.508x** (0.8619 versus 0.5717 nats per unit score)

So per unit of capability actually acquired, a5 moves further from the base model than the incumbent does. That is a real count against it, it is independent of the extrapolation, and it is the version of the objection I would defend.

One further caveat the verdict raises and I endorse: a **time-varying** view offset would break V1's design assumption that codec offsets are constant, which would mean the V1 reading is partly contaminated after all. That is unresolvable retrospectively because round A saved no checkpoints, and it is precisely what the dense-replay rider fixes going forward.

## Net effect on the decision

Unchanged: round A has no winner, the ladder falls through, the bias mechanism explains a5's drift, and the recommended next step is one 4-hour pre-registered probe rather than committing 28 hours to rounds B and C.

Changed: the probe should flip `rollout_is_batch_normalize` to True rather than raise the IS threshold, and the case against a5 rests on **drift per unit learning (1.508x)** rather than on a collapse extrapolation I cannot support from 120 steps.
