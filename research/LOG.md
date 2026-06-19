# Research Log

The detailed historical log has been folded into `runs/SUMMARY.md`, W&B, git
history, and merged code. The next phase should use method names and settings,
not old run labels.

## Current State

- **Current leading method (provisional SOTA):** `signed_ema` α=0.5, **`beta_anc=0.50`** → greedy GSM8K val@50 **0.7635** — highest measured, in the dense band. Status **REVISE** (single draw + best-of-3; ties the compatibility reference within ±0.024 noise; pending a β=0.50 replicate to confirm).
- **Normal research path:** EMA-family mergers, with `signed_ema` as the default. Keep the working reference parameters (`lambda=1`, `beta_anc=0`) only for compatibility checks and floors, not as a future planning target.
- **Shared substrate:** PowerSGD `r=77`, anchor owns `Q`, paired replay, CPU snapshot, cadence/delay `5`, `clean_cadence=0`, `disable_custom_all_reduce=true`; bytes ratio ≈ `0.0505`.
- **Dense reference:** about `0.75-0.78` greedy GSM8K val@50.
- **`beta_anc`:** NON-flat on `signed_ema` (peaks at 0.50: 0.7612 / 0.7635 / 0.7225, EXP-34); keep EMA behavior as the active research signal.

## Next-Phase Rule

Next experiment = the **β=0.50 replicate** (2–3 draws on `signed_ema` α=0.5 β=0.50) to
confirm or retract the SOTA. Until it lands, start from the current EMA baseline; if
confirmed, promote `signed_ema` α=0.5 β=0.50. Vary one knob at a time; do not import old
anchor-gradient claims or run labels into new plans.
