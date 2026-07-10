# Quick Report — Issue #63: comm-eff `signed_ema` vs dense on DeepScaleR RLVR

> **TRUNCATED RUN — directional early-signal read only.** The operator cut this
> run short on 2026-07-10 before the planned full sweep completed. Treat the
> numbers below as an early directional signal, **not** a full-length verdict.

**Bottom line:** On this truncated read, comm-eff `signed_ema` (β_anc=0.50) **holds
the dense line** — AIME@100 0.2125 vs dense 0.254 (~0.04 apart, inside both the
0.05 parity band and the ~0.09 AIME noise floor), with train-reward curves
overlapping tightly throughout.

## Setup (one line)

DeepSeek-R1-Distill-Qwen-1.5B on DeepScaleR RLVR (`DigitalLearningGmbH/MATH-lighteval`,
prompt 2048 / response 16384 / rollout n=16), RLVR-Linearity reproduction surface.
Validation = **AIME-2024 only** (`math-ai/aime24`, avg@8, 30 problems, std ~0.09).
Comm-eff regime: PowerSGD rank-77 act basis, anchor owns Q cadence/delay_K=20/20,
spectral `signed_ema` α=0.25.

## Results

| Arm | Steps reached | AIME@100 (or last) | Reward tracks dense? |
|---|---|---|---|
| **dense-control** (comm-eff OFF, reference) | 102 (full) | **0.254** | — (is the reference) |
| **signed-ema-b50** (β_anc=0.50, primary) | ~100 | **0.2125** (val@100, offline) | Yes (0.44–0.53 band) |
| **signed-ema-b50-la** (b50 + fixed_linear lookahead, str 1.0) | 50 (cut) | 0.20 (last @ step 25) | Yes |
| **signed-ema-b00** (β_anc=0.00 ablation) | not run (dropped) | — | — |

- Dense AIME avg@8 curve (0/25/50/75/100): 0.208 / 0.192 / 0.213 / 0.188 / 0.254.
- b50 AIME curve (0/25/50/75): 0.183 / 0.20 / 0.229 / 0.217; **val@100 = 0.2125**
  recovered offline from the step-100 checkpoint via verl `val_only`.
- b50-la lookahead fired as designed: 2 warmup fallbacks (steps 10, 20) + real
  projections from step 30 (3 by step 50).

## Notable finding — entropy decoupling

The comm-eff arms ran at **~8× higher policy entropy than dense** (~6.6 vs ~0.8)
yet held both reward and AIME parity. Compression visibly perturbs the output
distribution, but the anchor + spectral correction keeps task capability pinned
to the dense line — an interesting result worth following up.

## Caveats (read prominently)

- **Truncated:** dense 102, b50 ~100, b50-la 50; b00 **dropped**. This is not the
  planned full sweep.
- **AIME is high-variance:** 30 problems, std ~0.09 — single-point gaps of ~0.04
  are within noise.
- **Not paper-comparable:** the fork's AIME scorer (Hendrycks `math_reward`)
  differs from the RLVR-Linearity paper's scorer, so absolute numbers are not
  comparable to the paper. Internal (arm-vs-arm) comparison remains valid.

## Artifacts

- Preserved logs: `runs/63-deepscaler-r1d-signed-ema-k20/logs/`
  (`train_dense-control.log`, `train_signed-ema-b50.log`,
  `train_signed-ema-b50-la.log`, `val100_train.log`).
- b50 step-100 checkpoint mirrored to R2:
  `shamane-pluralis/autonomous-harness-rlvr-compression/63-deepscaler-r1d-signed-ema-k20/signed-ema-b50/checkpoints/global_step_100/`.

*The harness stress-test evaluation for this run is tracked separately as a
comment on issue #63.*
