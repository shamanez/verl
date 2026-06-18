# Research Log

The detailed historical log has been folded into `runs/SUMMARY.md`, W&B, git
history, and merged code. The next phase should use method names and settings,
not old run labels.

## Current State

- **Current leading method (provisional SOTA):** `signed_ema` α=0.5, **`beta_anc=0.50`** → greedy GSM8K val@50 **0.7635** — highest measured, above B2 (0.7528), in the dense band. Status **REVISE** (single draw + best-of-3; ties B2 within ±0.024 noise; pending a β=0.50 replicate to confirm).
- **Established baseline (replicated):** B2 `delayed_ef` (`lambda=1`, `beta_anc=0`) → val@50 **0.7528**. signed_ema β=0.50 differs ONLY in the merger.
- **Shared substrate:** PowerSGD `r=77`, anchor owns `Q`, paired replay, CPU snapshot, cadence/delay `5`, `clean_cadence=0`, `disable_custom_all_reduce=true`; bytes ratio ≈ `0.0505`.
- **Dense reference:** about `0.75-0.78` greedy GSM8K val@50.
- **`beta_anc`:** NON-flat on `signed_ema` (peaks at 0.50: 0.7612 / 0.7635 / 0.7225, EXP-34); flat on `delayed_ef` (EXP-33).

## Next-Phase Rule

Next experiment = the **β=0.50 replicate** (2–3 draws on `signed_ema` α=0.5 β=0.50) to
confirm or retract the SOTA. Until it lands, start from **B2 unchanged**; if confirmed,
promote `signed_ema` α=0.5 β=0.50. Vary one knob at a time; do not import old
anchor-gradient claims or run labels into new plans.
