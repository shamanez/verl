# Base-model capability + Big-Math trajectories (empirical inputs for the report)

## Base-model capability eval (no RL) — the "is GSM8K easy?" test
Untrained **Qwen2.5-1.5B-Instruct**, identical `\boxed{}` prompt + `math_reward` verifier,
200 random test problems each, greedy decode (on-box, 2026-06-01):

| dataset | base \boxed accuracy | emitted \boxed |
|---|---|---|
| **GSM8K** | **0.715** (143/200) | 192/200 |
| **Big-Math-RL-Verified-filtered** | **0.480** (96/200) | 183/200 |

**Reading:** GSM8K is *already* ~72% solved with zero RL → it is **easy** for this model. Big-Math
is substantially harder (48% base). This is the clean, format-controlled capability gap.

## RL trajectories (val accuracy / reward)
| run | dataset | method | val start→end | reward start→end |
|---|---|---|---|---|
| EXP-16 cell6 | GSM8K | dense | — | — (final val 0.741) |
| EXP-17 | GSM8K | masked p=0.9 + clean@20 | 0.085→**0.735** | 0.108→0.749 |
| EXP-19 | Big-Math | masked p=0.9 + clean@20 | 0.56→**0.55 (flat)** | ~0.40 flat |
| EXP-20 | Big-Math | **dense** (comm-eff OFF) | 0.558→**~0.59–0.61** | 0.41→~0.55–0.59 |

EXP-20 dense val by step: {10:0.558, 20:0.584, 30:0.568, 40:0.574, 50:0.566, 60:0.594, 70:0.594,
80:0.608, 90:0.602, 100:0.586} (peaked ~0.608, noisy plateau ~0.59) — modest but real climb above
the ~0.55 base, **unlike masked which never moved**. (EXP-20 stopped at step ~102 by operator.)

## The thesis these numbers support
- **GSM8K easy (base 0.715):** RL only needs to *elicit/sharpen* a latent capability (small headroom
  0.715→0.735). A coarse, 95%-communication-compressed gradient (masked + true-grad-only-every-20)
  carries enough directional signal to do that → **masked-clean@20 ≈ dense (0.735 vs 0.741).**
- **Big-Math hard (base 0.48):** genuine learning is required. Dense GRPO extracts a modest real gain
  (→~0.61); the lossy masked gradient cannot, so it **stalls flat (~0.55).** Headroom exists (dense
  finds it) → the masked stall is a *method/gradient-fidelity limitation exposed by task difficulty*,
  not a lack of headroom.
- I.e. **the masked+clean@20 gradient's information loss is tolerable for elicitation, fatal for
  learning.** Easy tasks hide the degradation; hard tasks expose it.

## rollout_corr context (EXP-19, run zejoupvf) — the train-inference gap
Masked steps: kl≈16.8, pearson(actor,rollout)≈0.004, ppl_ratio≈2e7, training_log_ppl≈17 vs rollout≈0.36.
Clean steps (20/40/60/80): kl≈0.0003, pearson≈0.9996, ppl_ratio≈1.0. → clean-resettable sawtooth.
The masked forward is ≈decorrelated from the sampler, yet the run still ascends on GSM8K — the theory
deliverable (theory.md) must explain why (the update is advantage-weighted ∇log π_masked with a
self-consistent masked PPO ratio; clean@20 re-anchors before bias accumulates).
