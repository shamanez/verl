# Research Runs Summary

Durable record (full run dirs de-bloated; provenance = this file + each run's
`verdict.md` + W&B + git history + merged code).

## Current base — accelerated comm-eff loop (EXP-36B, 2026-06-18)

The canonical base for every future test is
`examples/grpo_trainer/vast_comm_eff_accel_base_qwen25_1p5b_grpo_gsm8k.sh`:

- **accel surface** — response 2048, dynamic-bsz, rollout TP=1, gpu_mem_util **0.55**,
  ppo_max_token 24576, 50 steps, val@25/50, no val-before-train
- **core merger** — `signed_ema` (α=0.25, β_anc=0.50)
- **speed knob** — `diagnostics=false` (math-neutral; static review + `EXP-36B/NEUTRALITY_REVIEW.md`)
- **substrate** — locked PowerSGD r=77 anchor circuit (anchor owns `Q`, cadence/delay_K=5,
  clean=0, paired replay, `disable_custom_all_reduce`)

**Speed:** ~25 min train / ~28 min wall per 50-step comm-eff run (~12 min dense) — vs
~2 h on the old 16384 surface.

**Reference val@50 on this surface (n=1, noisy — rollout nondeterminism ≈ ±0.024/draw):**

| arm | run | val@25 | val@50 |
|---|---|---|---|
| dense control (comm-eff OFF) | EXP-36C | 0.7627 | **0.7657** |
| comm-eff `signed_ema(0.25, 0.50)` | EXP-36B | 0.7263 | **0.7362** |

Dense leads the single comm-eff draw by ~0.030 on the identical @0.55 surface
(bytes ratio ≈0.0505). Both n=1 — the working baseline, not a verdict.

## Lineage (established)

- **Substrate** — PowerSGD r=77 on the mandatory anchor circuit reaches dense parity at
  ~5% gradient comm. Locked.
- **Merger** — `delayed_ef` (β_anc=0) is the legacy replicated reference; `signed_ema` is
  the current core. The (α=0.25, β_anc=0.50) config comes from two old-surface sweeps:
  β_anc peaked at 0.50 (EXP-34); α peaked at 0.25 = 0.7528 (EXP-35), and α=0.0 does NOT
  ignite.
- **Anchor-usage levers** (perturbation, δ-momentum, adaptive-λ, control-variate,
  sub-basis) — all null vs baseline (EXP-31).
- **Reference floors** — no-merger PowerSGD = 0.6300; dense full-gradient band = 0.75–0.78.

## Bottom line

The accelerated comm-eff base (`signed_ema` α=0.25, β_anc=0.50, @0.55, diagnostics off)
is the default loop: val@50 ≈ 0.736 (n=1) vs dense ≈ 0.766, at ~5% gradient comm and
~25 min/run. Vary only the merger; every other knob is locked
(`FIXED_CONTROL_SURFACE.md`).
