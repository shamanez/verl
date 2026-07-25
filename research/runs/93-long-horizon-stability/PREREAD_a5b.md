# Pre-read: a5b at step 51 of 200. NOT a verdict.

Written 2026-07-25T15:10Z while the cell is still running, from matched-window
pulls made to prove the scoring pipeline works before termination. The registered
bar is scored at 100-120 and cannot be evaluated yet. Everything below is a
partial-window observation and none of it may be substituted for the verdict.

## The matched-window picture at steps 2-51

Every column is the same step window for all three runs. Cross-window comparison
is the error this program has already made twice, so nothing here compares
different windows.

| | score 2-20 | 21-40 | 41-51 | gain | grad_norm med (>=20) | ESS 21-51 |
|---|---|---|---|---|---|---|
| incumbent PRF | 0.3614 | 0.4019 | **0.4964** | **+0.1350** | 1.5831 | n/a (IS off) |
| a5 FRLR+TIS | 0.3611 | 0.3652 | 0.3722 | +0.0111 | 0.1312 | 0.2385 |
| a5b, +bnorm | 0.3575 | 0.3524 | **0.3604** | **+0.0029** | **0.6239** | 0.2354 |

| | gap 2-20 | 21-40 | 41-51 | drift 2-20 | 21-40 | 41-51 |
|---|---|---|---|---|---|---|
| incumbent PRF | 13.7495 | 13.8450 | 14.0167 | 0.03052 | 0.03290 | 0.04084 |
| a5 FRLR+TIS | 5.6377 | 4.5093 | 4.5864 | 0.16268 | 0.00431 | 0.01116 |
| a5b, +bnorm | 5.6815 | 4.5474 | 4.7278 | 0.16312 | 0.00564 | 0.01882 |

All three begin at the same score, about 0.36, as they must: same base model.

## The headline: the knob worked and it changed nothing that mattered

`batch_normalize` did exactly what it was built to do. The logged
`rollout_is_batch_norm_factor` is 0.186, so it divides by 0.186 and scales the
update back up 5.38x nominal; the realised effect on gradient norm is **4.76x**
(0.1312 to 0.6239 median), which is the mechanism confirmed quantitatively.

And learning did not move. a5b gained **+0.0029** over the same 50 steps in which
the incumbent gained **+0.1350**, and it is if anything marginally flatter than
a5's +0.0111. Gap and drift are also essentially unchanged from a5 (gap 4.73
versus 4.59, a 3.1 percent difference).

**This refutes my own diagnosis.** The round-A correction memo argued that a5
failed to learn because the mean IS weight of 0.166 scaled every gradient down
about 6x, and that removing the blanket shrinkage was therefore the fix. The
shrinkage has now been removed under controlled conditions, gradients are back to
0.394x the incumbent's rather than 0.083x, and the learning deficit is
unchanged. Gradient **magnitude** was not the binding constraint.

That points the deficit at the gradient's **direction**: either FRLR produces a
badly aimed gradient, or token-IS reweighting damages credit assignment. Those
are exactly the two things cell a6 separates, which is a stroke of luck given a6
was queued for a different reason.

It is also the program's own central law appearing in a second place. Round A
established that for capability **damage**, coherent direction gates the outcome
and magnitude does not (a1 versus a2: 2.7x noise energy at zero bias moved drift
not at all; flipping to biased rounding at identical wire moved it 6.9x). The
same distinction now appears for **learning**: restoring magnitude 4.76x did not
restore learning.

## A registered falsifier is vacuous and must be withdrawn

The bar carried three pre-registered falsifiers, the first being: *if ESS reaches
>= 0.5 and score is still below 0.6248, the deficit is FRLR-caused not IS-caused.*

**That falsifier can never fire, and I should have seen it before registering
it.** Kish effective sample size is

```
ESS = (sum w)^2 / (n * sum w^2)
```

which is invariant under `w -> w/c` for any constant `c`. Batch normalisation is
exactly such a rescale, so it cannot move ESS by construction. Measured: ESS went
0.2385 to 0.2354, a 1.3 percent difference attributable to noise. On top of that,
verl computes every `rollout_corr/rollout_is_*` metric at
`rollout_corr_helper.py:604`, which is before the normalisation block, so the
logged ESS is the raw distribution and is doubly unable to respond.

The falsifier is therefore withdrawn as unevaluable rather than quietly ignored.
The question it was meant to settle, FRLR's fault or the weighting's, is settled
instead by **a6**, which holds the weighting fixed and changes the codec. That is
a better instrument than ESS would have been.

The other two falsifiers stand and remain evaluable at 100-120.

## What is still genuinely open

- The registered window is 100-120. a5b is at 51. Score could still turn up; the
  incumbent's own curve was 0.4019 at 21-40 before reaching 0.6577 later, so a
  slow start is not by itself disqualifying. What is unusual is that a5b is
  **flat**, not slow.
- a5b's drift level at 41-51 is 1.69x a5's (0.01882 versus 0.01116), consistent
  with more gradient producing more movement, but still only **0.46x the
  incumbent's** 0.04084. The drift veto fired on a5 for slope, not level, and the
  level story continues to favour the FRLR arms.
- The windowed gap slope over 20-45 is +0.005927/step, which is 11.9x the
  registered +5.0e-4 bar, but that bar is scored at 100-120 and these slopes
  decelerate. Nothing follows from the early window.

## Health

Zero error markers. `grad_norm` peaked at 204.386 but that is a step-2 transient
(116.9, 204.4, 30.4 at steps 1-3, then under 1.2 from step 8 onward, median 0.64
from step 20). Max at step >= 20 is 0.878. Not a stability concern.
`actor/ppo_kl` is exactly 0, correct by construction.
