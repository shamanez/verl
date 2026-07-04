# Math-only projector ablation (EXP-61 / issue #61)

## Headline

On the Big-Math (EXP-58) fp32 weight trajectory (50 snapshots at 20 global steps
each = 1000 global steps), an exhaustive six-axis projector ablation was scored
against the do-nothing hold_stale baseline. The single best-fitting Math
projector is adaptive_linear (rolling_ls_k, K=3) at the freshest anchor Delta=1,
horizon h=1 - the only regime where projection materially beats do-nothing:
op-cell weight_proj_ratio_median = 0.98879 (bootstrap CI [0.98763, 0.99043],
does not straddle 1.0), pred_evr_pooled = +0.02203, positive skill CI
[+0.019, +0.025]. Under the >=0.01 prefer-simplicity rule the fixed-global
damped_linear (lam*=0.2, ratio 0.98995) is statistically equivalent, so the
deployable recommendation is the SIMPLER fixed rule at Delta=1,h=1.

## Absolute prediction accuracy

Absolute accuracy (pred_evr_pooled, per-scalar pred_r2_scalar_*) is reported
first for every arm including hold_stale (evr = 0 by definition). Every
projector's pred_evr_pooled is at or below 0 at the primary op-point
(per-tick Delta=5, h=10 = 200 global steps); it turns positive ONLY at the very
short-Delta / short-h corner. This is the signature of a near-incoherent
trajectory: there is little linear structure to fit beyond one 20-step tick.

## Is projection usable on Math?

Marginally, and only at the freshest anchor. At the primary op-point (Delta=5,h=10)
every method collapses to do-nothing (damped_linear lam*=0, ratio 1.00000;
adaptive arms ratio ~ 1.0002). The projection benefit exists only for
Delta <= 3 and h <= 3 (~ <=60 global steps of staleness AND horizon), decays
monotonically with both Delta and h, and never exceeds ~1.1%. The mechanistic
reason is in the coherence map: consec_delta_cos ~ 0.15 uniformly across all
Delta, all tensor families, and all training phases - near-orthogonal consecutive
updates (vs ~0.86 on GSM8K). Recommendation for the comm-eff ANCHOR on
Math-like regimes: short-refresh / hold-stale; a fixed damped-linear projection
buys ~1% only when the anchor is one tick (20 steps) stale, and bias should be
EXCLUDED from projection (it is actively harmed).
