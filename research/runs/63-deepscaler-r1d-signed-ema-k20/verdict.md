# Verdict — #63 comm-eff signed_ema vs dense (DeepScaleR RLVR, R1-Distill-1.5B, AIME val)

VERDICT: PASS (directional — TRUNCATED run, operator-cut 2026-07-10)

On this truncated read, comm-eff `signed_ema` (β_anc=0.50) holds the dense line: within
noise on the AIME headline and tightly on the low-variance training-reward curve. Not a
full-length verdict — the sweep was cut short (dense 102, b50 ~100, b50-la 50, b00 dropped).

## Criteria (AIME-2024 avg@8, `val-core/math-ai/aime24/acc/mean@8`; 30 problems, std ~0.09)
| criterion | observed | target | result |
|---|---|---|---|
| dense-control reward-health floor | AIME@100 = 0.254 | ≥ 0.10 | ✅ |
| all attempted cells no NaN / non-finite grad | dense 102, b50 99, b50-la 50 clean | no NaN | ✅ |
| headline: max{b50, b50-la} step-100 AIME ≥ dense − 0.05 | b50@100 = 0.2125 vs dense 0.254 (Δ 0.041) | within 0.05 | ✅ (inside band + noise) |
| corroborating train-reward parity | b50 tracked dense's 0.44–0.53 band through step 99 | overlap | ✅ |
| step-100 checkpoint → R2 | b50 global_step_100 mirrored (20 obj, 26.5 GiB) | present | ✅ |
| b00 ablation | NOT RUN (dropped in the operator cut) | report-only | ⚠️ dropped |

## Notes
- b50-la (b50 + fixed_linear lookahead): cut at step 50; 2 warmup fallbacks then 3 real
  projections; reward tracked dense over the short window.
- Notable: comm-eff arms held reward/AIME parity at ~8× higher policy entropy than dense
  (~6.6 vs ~0.8) — anchor + spectral correction keeps capability on the dense line.
- b50 val@100 recovered offline (verl `val_only` resume of its step-100 checkpoint); the
  live run crashed at the step-100 save on an R2 config bug (since fixed).
- Caveat: fork's Hendrycks `math_reward` scorer ≠ the RLVR-Linearity paper's, so absolute
  AIME numbers are not paper-comparable (internal cross-arm comparison is valid).

## Verification
python scripts/analyze.py runs/63-deepscaler-r1d-signed-ema-k20 --emit verdict.md
(analysis done from WandB + the preserved train logs under runs/<id>/logs/; the run was
operator-truncated so this verdict.md was authored to match the close comment SSOT.)
