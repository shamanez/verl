# Curve match — critic/score/mean (steps 1..50)

- candidate: `curvematch_anchorblend_c5_d5.jsonl` (23 steps)
- dense ref: `curvematch_dense_ref_50step.jsonl` (50 steps)
- common steps compared: 23 (first=1, last=23)

- **mean |Δ| over 1..50: 0.5315**  (tol 0.05) → FAIL
- **final-step |Δ| @ step 23: 0.7441**  (tol 0.05) → FAIL
- slope sign: candidate Δ=-0.1016, dense Δ=+0.6406 → MISMATCH
- candidate level: first=0.1328 final=0.0312  | dense final=0.7754

## CURVE_MATCH: NO-MATCH
(headline reward criterion only — analyst must also confirm pg_loss tracking, grad_norm finite, and the constraint greps before PASS)
