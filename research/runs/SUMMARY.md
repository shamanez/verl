# Research Runs Summary

Durable record (full run dirs de-bloated; provenance = this file + each run's
`verdict.md` + W&B + git history + merged code).

## Accelerated base — the default loop (EXP-36, 2026-06-18)

`examples/grpo_trainer/vast_comm_eff_accel_base_*.sh` is now the canonical base for
future tests: accel surface (resp 2048, dynamic-bsz, rollout TP=1, gpu_mem_util 0.55,
ppo_max_token 24576, 50 steps, val@25/50) + `signed_ema(α=0.25, β_anc=0.50)` + the
proven math-neutral `diagnostics=false` speed knob, on the locked PowerSGD r=77 anchor
substrate. **Faster:** ~25 min/50 steps (comm-eff), ~11.5 min (dense) — vs ~2 h on the
old 16384 surface; 0.75 mem-util gave no speedup (dropped).

Accel-surface val@50 (n=1, NOISY): **dense 0.7657** > **comm-eff signed_ema(0.25,0.50)
0.7362** (gap −0.030 on the identical @0.55 surface; comm-eff spans 0.70–0.75 across 3
draws). dense@0.75=0.7695 ≈ dense@0.55 (the 0.55 surface costs dense ~nothing).
`diagnostics=false` verified math-neutral (static review + EXP-36B; see
`runs/EXP-36B/NEUTRALITY_REVIEW.md`). Issue #36. The "Current SOTA" below is the
ORIGINAL 16384 surface (α=0.5) — a different, slower surface.

## Current SOTA — `signed_ema` (α=0.5, **β_anc=0.50**)

**GSM8K greedy val@50 = 0.7635** — the highest we have measured. Above the prior
B2 `delayed_ef` baseline (0.7528) and inside the dense band (0.75–0.78), at ~5%
fast-path gradient communication (PowerSGD r=77).

> **Status: provisional (EXP-34 verdict = REVISE).** It is a single draw + best-of-3
> selection. The margin over the `signed_ema` β=0 reference (0.7271) is **+0.036** and
> clears the +0.024 noise bar; the margin over B2 (+0.011) is *within* ±0.024 noise (a
> tie at the top, not a clean surpass of B2). Confirm with a **β=0.50 replicate (2–3
> draws, take the mean)** before treating it as a hard surpass.

### Best hyperparameters (the SOTA config)

| knob | value |
|---|---|
| codec | PowerSGD `rank=77`, `q_basis=act` |
| anchor | enabled, owns `Q`, `cadence=5`, `delay_K=5`, `replay_paired_batch=true`, `snapshot_device=cpu` |
| **merger** | **`correction_mode=signed_ema`, `signed_ema_alpha=0.5`, `beta_anc=0.50`** |
| coverage | `spectral.max_targets=-1` (all 196 matrices), `ema_device=cpu`, `clean_cadence=0` |
| vLLM | `disable_custom_all_reduce=true` (NCCL all-reduce; box-compat, greedy-val-neutral) |
| training | Qwen2.5-1.5B-Instruct + GSM8K, GRPO (no KL, no entropy), batch 128 / mini 64, lr `1e-6`, rollout `n=8`, response 16384 |
| measurement | `total_training_steps=50`, `test_freq=25`, `val_before_train=False` (val @ 25, 50) |
| comm | bytes ratio ≈ `0.0504` |

It differs from the prior B2 baseline ONLY in the merger: B2 = same substrate with
`correction_mode=delayed_ef`, `λ=1`, `beta_anc=0` → val@50 **0.7528** (established,
replicated lineage). Adding `beta_anc=0.50` under `signed_ema` is what lifts it.

## What we tested (short)

- **Substrate** — PowerSGD r=77 on the mandatory anchor circuit (owns Q, delay_K=5): reaches dense parity at ~5% gradient comm. Locked.
- **Merger family** — `delayed_ef` (B2, 0.7528) vs `signed_ema`. `signed_ema` at β_anc=0 = 0.7271 (below B2); adding anchor-EMA `β_anc` lifts it.
- **β_anc sweep on `signed_ema`** (EXP-34): 0.25 → 0.7612, **0.50 → 0.7635 (peak)**, 0.75 → 0.7225. Non-flat, peaks at 0.50 — unlike `delayed_ef`, where β_anc was flat (EXP-33: 0.738–0.753 across 0–0.75, β=1 cold-M collapse).
- **Other anchor-usage levers** — perturbation, δ-momentum, adaptive-λ, control-variate, sub-basis: all null vs B2 (EXP-31).
- **Reference floors** — no-merger PowerSGD = 0.6300; dense full-gradient = 0.75–0.78.

## Bottom line

`signed_ema` (α=0.5, **β_anc=0.50**) is the current best config — val@50 **0.7635**,
provisional pending a β=0.50 replicate. B2 (`delayed_ef`, β=0) is the established
replicated baseline at parity (0.7528). Both hit dense parity at ~5% gradient comm.
