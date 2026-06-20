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
- **Merger** — `signed_ema` is the normal method for all forward research. The
  active baseline is (α=0.25, β_anc=0.50), selected from two old-surface sweeps:
  β_anc peaked at 0.50 (EXP-34); α peaked at 0.25 = 0.7528 (EXP-35), and α=0.0 does NOT
  ignite. Keep the proven compatibility parameters (`λ=1`, `β_anc=0`) available only
  as a quiet reference floor, not as a planning target.
- **Anchor-usage levers** (perturbation, δ-momentum, adaptive-λ, control-variate,
  sub-basis) — all null vs baseline (EXP-31).
- **Reference floors** — no-merger PowerSGD = 0.6300; dense full-gradient band = 0.75–0.78.

## EXP-37 — latency × merger instability study (2026-06-20)

How anchor latency (cadence/delay_K) interacts with the merger, at **100 steps** (crossing the
GSM8K epoch-2 boundary ~step 58). Accel surface, Vast team H200×4 (all boxes torn down). Full
verdict: [`reports/comm-eff-grpo/why-grpo-fails-sft-works.html`](../reports/comm-eff-grpo/why-grpo-fails-sft-works.html).

| run | merger | latency | β_anc | val@25/50/75/100 | outcome |
|---|---|---|---|---|---|
| EXP-37D | dense (comm-eff OFF) | — | — | .752/.766/.777/**.783** | STABLE, monotonic |
| EXP-37B | signed_ema | 5/5 | 0.50 | .738/.681/.698/**.735** | STABLE (PASS) |
| EXP-37 | signed_ema | 20/20 | 0.50 | — | TERMINAL collapse ~step 61 |
| EXP-37C | signed_ema | 20/20 | 0 | .681/.537/.701/**.346** | OSCILLATING collapse (STOP) |
| EXP-37E | delayed_ef | 20/20 | 0.50 | .649/.676/.581/**.608** | DEGRADATION, no ignition |

1. **Latency is the failure knob.** 5/5 stable/near-dense; **20/20 breaks BOTH merger families**.
   (delayed_ef = B2 was near-dense 0.7528 at 5/5; degrades to 0.608 at 20/20.)
2. **Two symptoms, one cause.** Cause = `K>τ` off-policy staleness; symptom set by merger geometry:
   **signed_ema (sign-replace) IGNITES** (entropy collapse + length explosion); **delayed_ef
   (additive) STALLS** (entropy high/oscillating, length flat ~200, grad_norm 2→76 as the held
   stale δ tracks ‖θ_t−θ_{t−K}‖; sub-baseline plateau ~0.61 — still above 37C's 0.35).
3. **Compression-specific, not epoch.** Dense (37D) sails through the epoch-2 boundary to 0.783;
   the back-half instability is the stale-anchor path, not the second data pass.

**Why it works for SFT, fails for on-policy GRPO:** the anchor imports the SFT assumption *stale
full-grad ≈ current full-grad* (= gradient-field stationarity over K). SFT optimizes a FIXED
dataset, so it holds (bias ≤ curvature·‖Δθ‖, error-feedback-recoverable). GRPO's gradient is over
the CURRENT policy's samples, so the anchor is the EXACT on-policy gradient of the OLD policy
π_{θ_{t−K}} (paired-replay self-consistent) transplanted with NO importance-sampling reweighting —
a low-variance, **BIASED** "valid gradient for the wrong policy" (worse than noise because
persistent). Paired-replay irony: a "valid" M lowered the variance of a biased signal ⇒ a
*more*-persistent carrier. Full argument + falsifiers in the report above.

## Bottom line

The accelerated comm-eff base (`signed_ema` α=0.25, β_anc=0.50, @0.55, diagnostics off)
is the default loop: val@50 ≈ 0.736 (n=1) vs dense ≈ 0.766, at ~5% gradient comm and
~25 min/run. Future research should stay on EMA-family mergers; every other knob is locked
(`FIXED_CONTROL_SURFACE.md`). **Realistic-latency caveat (EXP-37):** the method holds only at
LOW anchor latency (5/5) — at 20/20 every merger fails (ignite or stall), because the stale
anchor is an off-policy gradient for a defunct policy. Latency, not the merger, is the binding
constraint for the decentralized-PP target.
