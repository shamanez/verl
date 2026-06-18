# Post-Experiment Summary Plan

Compact handoff for future planning. Execution plan files are deleted after each
issue; this file persists. Results live in `research/runs/SUMMARY.md` + each run's
`verdict.md` + W&B.

## Current base (the default loop)

| item | value |
|---|---|
| launcher | `examples/grpo_trainer/vast_comm_eff_accel_base_qwen25_1p5b_grpo_gsm8k.sh` |
| method | `signed_ema`, α=0.25, β_anc=0.50 |
| surface | accel: resp 2048, dyn-bsz, rollout TP=1, gpu_mem 0.55, ppo_max_token 24576, 50 steps, val@25/50, diagnostics off |
| substrate | PowerSGD r=77 + anchor (owns Q, cadence/delay_K=5, clean=0, paired replay, `disable_custom_all_reduce`) |
| speed | ~25 min train / ~28 min wall per 50-step run |
| val@50 (n=1) | comm-eff **0.7362** (EXP-36B) vs dense **0.7657** (EXP-36C); bytes ratio ≈0.0505 |

## Tested knobs (closed)

| knob family | takeaway |
|---|---|
| merger | `signed_ema` is the core; `delayed_ef` (β=0) is the legacy replicated ref |
| `beta_anc` on signed_ema | non-flat, peaks 0.50 (EXP-34, old surface) |
| `signed_ema_alpha` | peaks 0.25 = 0.7528 (EXP-35 α-sweep, old surface); α=0.0 does NOT ignite |
| δ-momentum / adaptive-λ / perturbation / control-variate / sub-basis | all null vs baseline (EXP-31) |

## Planning rule

Start from the accel base launcher **unchanged** and vary a SINGLE knob — the merger
(`spectral.correction_mode` and its α / β_anc). Everything else is locked
(`runs/FIXED_CONTROL_SURFACE.md`). Every run must finish under ~25 min train.

Do **not** reintroduce the dropped vLLM speed knobs (gpu_mem 0.75 / `chunked_prefill` /
`forward_prefetch`): they gave no speedup and added rollout noise. Do not rebuild deleted
plan files or import invalid (pre-paired-replay) anchor claims.
