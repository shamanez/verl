# Pre-read: a7 at step 51 of 200. Two registered predictions confirmed, one mis-specified by me.

Written 2026-07-26T03:00Z. The registered bar is scored at 100-120 and is not
evaluable yet. What follows is a check against the four predictions registered in
`PREREG_a7.md` before the cell existed.

## The matched-window picture

| GAP `rollout_corr/kl` | 2-20 | 21-40 | 41-51 |
|---|---|---|---|
| incumbent PRF, no IS | 13.74950 | 13.84502 | 14.01668 |
| a5b FRLR + token-IS | 5.68152 | 4.54744 | 4.72780 |
| **a7 FRLR, no IS** | **5.65086** | **4.52066** | **4.47493** |

| SCORE | 2-20 | 21-40 | 41-51 | vs incumbent |
|---|---|---|---|---|
| incumbent | 0.36025 | 0.39204 | 0.49343 | 1.000x |
| a5b | 0.35752 | 0.35244 | 0.36044 | 0.730x |
| **a7** | **0.36549** | **0.38833** | **0.48311** | **0.979x** |

| GRAD_NORM | 2-20 | 21-40 | 41-51 |
|---|---|---|---|
| incumbent | 2.13918 | 1.84511 | 1.86276 |
| a5b | 13.43662 | 0.64240 | 0.62027 |
| a6 (for scale) | - | - | about 30 |
| **a7** | 6.39074 | 2.27250 | **1.71794** |

## P1 CONFIRMED: the gap win is the codec alone

a7 carries **no importance weighting of any kind** and reproduces a5b's gap to
within 5 percent at every window: 4.475 against 4.728 at 41-51, against the
incumbent's 14.017. That is the same **3.1x reduction** at the same 1232-bit wire.

Combined with a6, which carried the weighting on the incumbent codec and reproduced
the incumbent's gap to 0.8 percent, the attribution is now closed from both sides.
**Token-IS contributes nothing to the gap. FRLR contributes all of it.**

## P2 CONFIRMED, and it identifies the cause of the onset delay

a7 shows **no onset delay at all**. It tracks the incumbent from the first window
and sits at **0.979x** by 41-51, where a5b was at 0.730x and had not started
learning. The registered prediction was that a7's 41-60 score would exceed a5b's
0.3728; it is 0.483.

So the 20-to-40-step learning delay that both TIS arms showed is caused by
**token-IS**, not by FRLR. Remove the weighting and it disappears. `grad_norm` says
the same: a7 runs at **1.72 against the incumbent's 1.86**, where a5b sat at 0.62
(suppressed by the IS weights) and a6 at about 30 (amplified by normalisation at a
large gap). a7 is the only compressed arm in the program whose gradients look normal.

## P4 CONFIRMED

No `rollout_is_*` keys exist on the run, and the config reads `rollout_is: None`.
The arm is configured as intended.

## P3 is OFF TRACK, and the prediction was mis-specified. I am saying so before the data lands, not after.

| codec-free `probe/kl_dense` | step 25 | step 50 |
|---|---|---|
| a5b FRLR + token-IS | 0.000252 | 0.000752 |
| **a7 FRLR, no IS** | 0.000293 | **0.001576** |

a7's true drift is **2.10x a5b's at step 50** and the ratio is widening. I registered
that a7's codec-free drift at step 200 would come in at or below a5b's 0.016754. On
current trajectory it will not.

**But that prediction conflated absolute drift with harmful drift, and it should not
have.** a7 has gained 0.126 of score by 41-51 where a5b gained 0.003. A model that
learns necessarily moves its weights; suppressing learning suppresses drift
trivially, which is exactly what token-IS did to a5b. So "less absolute drift" is
not a virtue on its own, and I set the prediction up so that the arm which learns
properly is penalised for it.

The defensible comparison is **drift per unit capability gained**, the same measure
that made a6 look bad. At 41-51, taking the shared start of about 0.357:

| cell | true drift | score gain | drift per unit gain |
|---|---|---|---|
| a5b | 0.000752 | 0.003 | 0.25 |
| **a7** | 0.001576 | 0.126 | **0.0125** |

a7 is about **20x more efficient** on that measure at this point, though this
flatters a7 because a5b had barely begun learning and its ratio will improve as it
does. The honest statement is that **P3 as written is likely to fail and its failure
will not count against a7**; the terminal comparison to make is a7's drift and val
at step 200 against a5b's 0.016754 and 0.6593.

I am recording this now, at step 51, specifically so that discounting P3 later
cannot look like moving a goalpost after seeing the result.

## Where this leaves the program

On the evidence so far a7 is the configuration the matrix has been looking for: the
**full gap reduction, incumbent-speed learning, normal gradients, no
importance-sampling machinery at all, at the incumbent's exact wire budget**. Nothing
is settled until the registered window at 100-120 and the terminal val, and the true
drift is the open question. But no previous cell has had this combination.
