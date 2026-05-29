# Communication-efficient baseline — findings

Permanent reference. PASS verdict for the full communication-efficient GRPO
pipeline at smoke scale on Qwen2.5-1.5B-Instruct + GSM8K.

## Configuration

The verified-PASS configuration of the communication-efficient method:

- **Activation mask** (PRF Bernoulli at pipeline-boundary decoder blocks):
  `p=0.9`, `mask_recompute=true` — fires on BOTH gradient-feeding forwards
  (the actor-train forward AND the `compute_log_prob` recompute).
- **Anchor circuit** (hookless clone of the live FSDP module): refresh
  every 5 PPO substeps from a 5-substep-stale weight snapshot, unmasked
  GRPO-actor-loss backward harvesting `G_anchor` into the EMA `M_anchor`.
- **Spectral correction** (anchor-EMA → full thin SVD → Tikhonov →
  two-sided projection → α-blend): `alpha=0.5`, `tau=0.01`,
  `beta_anc=0.9`, `seed_anchor_cache=false`, full SVD with GPU-resident
  EMA and cached basis. Up to 4 targeted matrices per step.
- **Objective**: pg_loss only — no KL, no entropy
  (`use_kl_loss=False`, `use_kl_in_reward=False`, `entropy_coeff=0`).
- **FSDP**: `use_orig_params=true` so the spectral correction hook sees a
  full 2D Tensor post-FSDP-reduce.

## Headline result

| Metric | Value |
|---|---|
| Trainer steps reached | 20 / 20 |
| Reward — mean(steps 1-10) | 0.069 |
| Reward — mean(steps 11-20) | 0.125 |
| Second-half improvement | **+82% over first half** |
| Peak step reward | 0.25 (steps 12, 17, 18 — 4× the step-1 value) |
| Final three steps | (0.25, 0.25, 0.1875) — sustained high-reward window |

## Comm-eff counters at step 20

| Counter | Value |
|---|---|
| `mask_applications/train` | 280 |
| `mask_applications/old_logprob` | 140 |
| `mask_applications/{rollout, ref_logprob, val, infer, ckpt}` | 0 |
| `mask_ratio` | 0.8998 (configured `p=0.9` ±0.0001) |
| `anchor_backwards` | 10 (cadence=5, 50 substeps / 5) |
| `anchor_mask_applications` | 0 (GUARD 5) |
| `anchor_grad_corrected` | 0 (GUARD 6) |
| `anchor_rollouts_generated` | 0 |
| `anchor_rewards_recomputed` | 0 |
| `anchor_optimizer_steps` | 0 |
| `spectral_corrections` | 160 |
| `||dM_anchor||_mean` trajectory | 0 → 0 → 0 → 0.311 → 1.119 → 0.272 → 0.631 → 0.086 → 0.459 → 0.071 |
| `actor/grad_norm` | finite at every substep; no NaN/Inf |

## Loss decomposition (confirms no-KL no-entropy)

At step 20: `actor/loss = actor/pg_loss` exactly. The objective is purely
the policy gradient.

## What this run is FOR

Use as the comm-eff side of any apples-to-apples comparison against the
dense baseline (`runs/baseline/`):

- "Does the comm-eff method's no-op path equal dense?" → compare against
  `runs/baseline/` with `comm_eff.enabled=false`.
- "Does the comm-eff method's full path learn?" → this run is the answer
  at smoke scale.
- "Does it scale to paper-scale rollouts?" → that is the active
  investigation; see `notes/investigation-prompt-grad-norm.md`.

## Source files

- Launcher: `examples/grpo_trainer/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh`
- Verdict: `runs/communication-baseline/verdict.md`
- Training log: `runs/communication-baseline/train.log`
- Reproducibility manifest: `runs/communication-baseline/REPRODUCIBILITY.md`
- Reference plan: `.claude/plans/communication-baseline.md`
