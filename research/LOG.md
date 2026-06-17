# Research Log

The detailed historical log has been folded into `runs/SUMMARY.md`, W&B, git
history, and merged code. The next phase should use method names and settings,
not old run labels.

## Current State

- **Best confirmed method:** B2 `delayed_ef` error feedback.
- **Default settings:** PowerSGD `r=77`, anchor owns `Q`, paired replay, CPU
  snapshot, cadence/delay `5`, `lambda=1`, `beta_anc=0`, `clean_cadence=0`.
- **Dense reference:** about `0.75-0.78` greedy GSM8K val@50.
- **Comm-efficient reference:** about `0.735-0.754` greedy GSM8K val@50 with
  bytes ratio about `0.0505`.
- **Beta sweep:** beta `0.5` was the nominal high draw, but within noise; beta
  `0` remains the default.
- **Signed EMA:** `alpha=0.5` is the signed-EMA point worth remembering, but it
  is not promoted over B2.

## Next-Phase Rule

Start from B2 unchanged and vary one knob at a time. Do not import old
anchor-gradient claims or old run labels into new plans.
