# Curve match — critic/score/mean (steps 1..50)

- candidate: `curvematch_anchorblend_b0_c5_d5.jsonl` (18 steps)
- dense ref: `curvematch_dense_ref_50step.jsonl` (50 steps)
- common steps compared: 18 (first=1, last=18)

- **mean |Δ| over 1..50: 0.4992**  (tol 0.05) → FAIL
- **final-step |Δ| @ step 18: 0.7969**  (tol 0.05) → FAIL
- slope sign: candidate Δ=-0.0986, dense Δ=+0.6816 → MISMATCH
- candidate level: first=0.1182 final=0.0195  | dense final=0.8164

## CURVE_MATCH: NO-MATCH
(headline reward criterion only — analyst must also confirm pg_loss tracking, grad_norm finite, and the constraint greps before PASS)
