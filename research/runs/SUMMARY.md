# Research Runs Summary

Durable record (full run dirs de-bloated; provenance = this file + each run's
`verdict.md` + W&B + git history + merged code).

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
