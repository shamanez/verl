# Post-Experiment Summary Plan

Compact handoff for future planning (execution plans are run artifacts and get
deleted after each issue; this file persists). Results live in
`research/runs/SUMMARY.md` + each run's `verdict.md` + W&B.

## Current best config (leading SOTA — provisional)

| item | value |
|---|---|
| method | **`signed_ema`, α=0.5, `beta_anc=0.50`** |
| val@50 | **0.7635** (highest measured; > B2 0.7528; dense band 0.75–0.78) |
| status | **provisional — EXP-34 verdict REVISE** (single draw + best-of-3; ties B2 within ±0.024 noise; clears the bar only over the signed_ema β=0 ref). Confirm with a β=0.50 replicate before promoting. |
| substrate | PowerSGD `r=77` + anchor (owns Q, `cadence=delay_K=5`, paired replay, CPU snapshot), `clean_cadence=0`, full coverage, `disable_custom_all_reduce=true` |
| comm | bytes ratio ≈ `0.0505` |

**Established baseline (replicated):** B2 = same substrate, merger `delayed_ef`
(`λ=1`, `beta_anc=0`) → val@50 **0.7528**. signed_ema β=0.50 differs ONLY in the merger.

## Tested knobs

| knob family | tested values | takeaway |
|---|---|---|
| merger | `delayed_ef` (B2) vs `signed_ema` | signed_ema β=0 = 0.7271 (< B2); **+β_anc lifts it** |
| `beta_anc` on `signed_ema` | `0.25`, `0.50`, `0.75` | **non-flat, peaks at 0.50** (0.7612 / 0.7635 / 0.7225) — EXP-34 |
| `beta_anc` on `delayed_ef` | `0`–`1` | flat 0–0.75; β=1 cold-M collapse — EXP-33 |
| δ-momentum / adaptive-λ / perturbation / control-variate / sub-basis | various | all null vs B2 — EXP-31 |

## Planning rule

The immediate next experiment is the **β=0.50 replicate** (2–3 draws on `signed_ema`
α=0.5 β=0.50, take the mean) to confirm or retract the SOTA claim. Until that lands,
**B2 remains the safe replicated base**; if confirmed, promote `signed_ema` α=0.5
β=0.50 to the locked base. Start from one of these unchanged and vary a single knob;
do not rebuild deleted plan files or import invalid (pre-paired-replay) anchor claims.
