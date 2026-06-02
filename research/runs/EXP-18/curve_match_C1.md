# Curve match — critic/score/mean (steps 1..50)

- candidate: `curvematch_anchorinject_c5_d5.jsonl` (34 steps)
- dense ref: `curvematch_dense_ref_50step.jsonl` (50 steps)
- spectral floor: `curvematch_spectral_baseline_c5_d5.jsonl` (50 steps)
- common steps compared: 34 (first=1, last=34)

- **mean |Δ| over 1..50: 0.6107**  (tol 0.05) → FAIL
- **final-step |Δ| @ step 34: 0.8047**  (tol 0.05) → FAIL
- slope sign: candidate Δ=-0.1338, dense Δ=+0.6699 → MISMATCH
- candidate level: first=0.1338 final=0.0000  | dense final=0.8047
- floor mean |Δ| vs dense: 0.5963 (candidate must BEAT this AND reach dense tol)
- floor final level: 0.1445 (pure-masked ~0.13 ⇒ collapse)

## CURVE_MATCH: NO-MATCH
(headline reward criterion only — analyst must also confirm pg_loss tracking, grad_norm finite, and the constraint greps before PASS)
