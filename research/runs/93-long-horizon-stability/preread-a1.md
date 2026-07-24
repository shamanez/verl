# a1 pre-read at step 76 of 120 (not the verdict)

Cell `a1-srq-b1-sr` (WandB `h0n67q3a`), sr_quant bits=1 block=32 rounding=sr,
GRPO on Qwen2.5-Math-1.5B / MATH, 128/128, 1024/2048, pp 8, LR 1e-6, val off.

Measured three ways: WandB `scan_history` (max step 77), the on-box
`/workspace/runs/a1-srq-b1-sr/train.log` (max step 76), and an adversarial
statistics pass whose mandate was to refute the trend. All three agree to four
significant figures on every shared quantity (ref-KL slope 0.003402 vs
0.0033993 vs 0.0033993, spread 0.08 percent). No parse error, no WandB
truncation: the on-box log has 76 unique `global_step` values, zero gaps, zero
duplicates, zero NaN.

## Headline: the "flat 1.9 noise floor" premise is falsified

The program brief records a1's `actor/kl_loss` as sitting on a flat ~1.9
stochastic-rounding view-noise floor from step 2, to be judged on slope above
that floor. Only the first stretch is flat. Over steps 2 to 18 the slope is
+0.0004/step (p = 0.32, not distinguishable from zero); from there the level
climbs monotonically and the climb accelerates.

- level 1.9096 (step 2) to 2.1391 (step 76)
- OLS slope +0.003399/step, r2 0.93, t = 30.9; Theil-Sen +0.003393 agrees
- residual SD about the fitted trend 0.0207 nats, which is only 1 percent of a
  level near 2.0, and raw spread about the mean is 3.7x that, so most of the
  observed variation IS the trend and not view noise
- residuals are autocorrelated (lag-1 ACF 0.83, Durbin-Watson 0.32), so the
  textbook SE of 0.00011 understates uncertainty about 2x; Newey-West HAC SE is
  0.00022 and a 20k moving-block bootstrap gives 0.00024. Significance survives
  every variant (HAC t = 15.6), as do Spearman and Mann-Kendall.
- the slope is not stable: steps 2-39 give +0.001743, steps 40-77 give
  +0.004592 (slope-change z = 10.6, positive quadratic term)
- halves differ in level (1.9271 vs 2.0591, Welch t = 14.2) while the
  DETRENDED within-half noise amplitude is constant (0.0126 vs 0.0122). So the
  codec's measurement noise is stable and the LEVEL is drifting. The constant
  part, the fitted intercept near 1.86, is the plausible SR view offset; the
  +0.0034/step part is not view noise.

Corroboration that this is a real wedge rather than a growing measurement
artifact: `rollout_corr/kl` climbs at +0.008622/step with r2 0.932, tracking
`actor/kl_loss` (r2 0.93). Two independently logged mismatch channels rising
together is the #90 wedge-growth signature, not a static offset.

## Gate flags against the #90 baseline card

| gate item | reading at step 76 | flag |
|---|---|---|
| reference KL, ABSOLUTE vs 0.156-0.203 | 1.9096 to 2.1391 nats, of which about 1.86 is the 1-bit SR view offset | not judgeable by design; a1 must NOT be scored as failing on the absolute number |
| reference KL, SLOPE vs baseline 0.0015/step | 0.002707/step (steps 2-60), 0.003399/step (2-76), accelerating | FAIL, 1.8x to 2.3x baseline |
| train-inference gap vs gate < 10, target < 3, incumbent 14.24 | 13.650 nats, widening +0.008622/step, accelerating (0.00287 over 1-30, 0.00918 over 30-76), projects about 14.03 at step 120 | FAIL |
| E[rho] vs baseline 0.0014 | 0.0066 (median of last 10; the single last value is unreliable, step 75 spikes to 1.0984) | 4.7x baseline, but about 150x below the 1.0 the identity needs, and declining |
| reward slope vs parity bar 0.00288 | +0.004434/step, 1.54x the bar and 1.39x baseline 0.0032; reward 0.3662 to 0.6514 | PASS |
| `actor/ppo_kl` about 0 | exactly 0.0 at all 76 steps, `pg_clipfrac` likewise | PASS, and exactly as expected: `train_batch_size=128` equals `ppo_mini_batch_size=128`, so there is ONE inner update per step, the ratio is identically 1, and both metrics are exactly zero by construction rather than unpopulated |
| collapse / capability damage | entropy 7.8996 to 7.9254 (flat to slightly up), grad_norm 0.8634 to 0.6628 (falling), response_length 748 to 687 tok, `kl_coef` pinned at 0.001 throughout | PASS, no signature |

## What this means

a1 is learning well and is not collapsing, but on the one quantity this program
exists to improve, the train-inference gap, it buys almost nothing: 13.65 nats
against the incumbent's 14.24, a 4 percent reduction, nowhere near the sub-10
gate or the sub-3 target, and widening. Meanwhile its reference KL drifts 1.8x
to 2.3x faster than baseline and accelerating. The provisional reading is that
1-bit stochastic rounding does not deliver the hoped gap-at-source reduction.

This is one arm's result, not a program STOP. The stage-1 gate is "at least one
arm clears", judged at the end of round A, so the matrix continues. It does
raise the stakes on a4 (PRF exact-k plus CVC, the arm designed to bend the gap
slope negative) and a5 (FRLR plus token-IS, the small-gap-and-corrected-drift
quadrant).

Watch item for the final verdict: KL is accelerating while reward is linear, so
the "productive movement" reading has a limited shelf life. Steps 100 to 120 are
where a1 is most likely to turn over. Extrapolated view KL at 120 is 2.29 to
2.34 nats.

## Consequence for the a2 kill gate

a2's pre-authorized early stop is "kill at step 60 if a2's reference-KL slope is
at least 2x a1's". Because a1's slope accelerates, that rule is only well posed
on a fixed window:

- **Fit a2 over steps 2 to 60 and compare against a1's steps 2 to 60 slope of
  0.002707/step. Threshold = 0.005414/step.**
- Using a1's full-run slope instead would give 0.006799/step, a gate 26 percent
  more permissive, which would let a genuinely worse a2 survive.
- Per-arm HAC SE is about 0.00023, so the band 0.0045 to 0.0063 is statistically
  inconclusive. A confident kill needs a2 above 0.0063; a confident acquittal
  needs a2 below 0.0045; inside the band, decide on the reward slope and gap
  corroboration rather than the point estimate alone.

a1's slope is decisively nonzero under OLS, Theil-Sen, HAC, moving-block
bootstrap, Spearman and Mann-Kendall, so the 2x rule needs no artificial noise
floor to be meaningful.
