# Curve match — critic/score/mean (steps 1..50)

- candidate: `curvematch_cleangrad_blend_e09_c5_d5.jsonl` (50 steps)
- dense ref: `curvematch_dense_ref_50step.jsonl` (50 steps)
- spectral floor: `curvematch_spectral_baseline_c5_d5.jsonl` (50 steps)
- common steps compared: 50 (first=1, last=50)

- **mean |Δ| over 1..50: 0.0703**  (tol 0.05) → FAIL
- **final-step |Δ| @ step 50: 0.0273**  (tol 0.05) → OK
- slope sign: candidate Δ=+0.6680, dense Δ=+0.7061 → MATCH
- candidate level: first=0.1455 final=0.8135  | dense final=0.8408
- floor mean |Δ| vs dense: 0.5963 (candidate must BEAT this AND reach dense tol)
- floor final level: 0.1445 (pure-masked ~0.13 ⇒ collapse)

## CURVE_MATCH: NO-MATCH
(headline reward criterion only — analyst must also confirm pg_loss tracking, grad_norm finite, and the constraint greps before PASS)
