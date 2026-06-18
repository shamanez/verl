# Research Status — 2026-06-18 (accelerated comm-eff base promoted)

## Current base (promoted, the default loop)

`examples/grpo_trainer/vast_comm_eff_accel_base_qwen25_1p5b_grpo_gsm8k.sh` —
`signed_ema(α=0.25, β_anc=0.50)` on the accel surface @ gpu_mem 0.55, `diagnostics=false`,
PowerSGD r=77 anchor substrate. **~25 min train / ~28 min wall** per 50-step run.

Baseline reference val@50 (n=1, this surface):

| arm | run | val@50 |
|---|---|---|
| dense control (comm-eff OFF) | EXP-36C | **0.7657** |
| comm-eff `signed_ema(0.25, 0.50)` | EXP-36B | **0.7362** |

## Notes

- The vLLM speed knobs **gpu_mem 0.75 / chunked_prefill / forward_prefetch** were tried
  (the old EXP-36) and **dropped** — they gave no speedup and added rollout noise. Not part
  of the base.
- `diagnostics=false` is in the base (math-neutral; `EXP-36B/NEUTRALITY_REVIEW.md`).
- The only axis that may vary is the **merger**; all other knobs locked
  (`runs/FIXED_CONTROL_SURFACE.md`).

## Next experiment rule

Start from the accel base launcher unchanged, vary a single merger knob, keep every run
under ~25 min train. Box lifecycle is operator-owned.
