# Communication-efficient baseline — reference plan

> Permanent reference run for the communication-efficient GRPO method.
> Held alongside `baseline` (the dense control) as the second pole of any
> apples-to-apples comparison. This is not an experiment plan; it is the
> static specification of what the comm-eff baseline IS.

## What it is

Single-cell GRPO smoke that exercises the full communication-efficient
pipeline (activation mask + asynchronous anchor circuit + spectral
correction) on Qwen2.5-1.5B-Instruct + GSM8K, with the no-KL no-entropy
objective the method is designed for. PASS verdict + headline result in
`runs/communication-baseline/verdict.md`.

## Source-of-truth files

| Artifact | Path |
|---|---|
| Launcher | `examples/grpo_trainer/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh` |
| Verdict | `runs/communication-baseline/verdict.md` |
| Training log | `runs/communication-baseline/train.log` |
| Reproducibility manifest | `runs/communication-baseline/REPRODUCIBILITY.md` |

## Method configuration

The launcher's defaults are this configuration; override individual knobs
via env vars only with intent. The configuration is what passed all 13
success criteria in the recorded smoke run.

| Knob | Value | Notes |
|---|---|---|
| `comm_eff.enabled` | `true` | Master switch |
| `comm_eff.mask.enabled` | `true` | |
| `comm_eff.mask.p` | `0.9` | PRF Bernoulli, fraction masked |
| `comm_eff.mask.mask_recompute` | `true` | Mask fires on BOTH gradient-feeding forwards (actor-train AND `compute_log_prob`) |
| `comm_eff.anchor.enabled` | `true` | |
| `comm_eff.anchor.cadence` | `5` | Anchor refresh every 5 PPO substeps |
| `comm_eff.anchor.delay_K` | `5` | Weight snapshot staleness |
| `comm_eff.spectral.enabled` | `true` | |
| `comm_eff.spectral.alpha` | `0.5` | Blend `α·G_mask + (1−α)·G_filt` |
| `comm_eff.spectral.tau` | `0.01` | Tikhonov damping |
| `comm_eff.spectral.beta_anc` | `0.9` | EMA decay for `M_anchor` |
| `comm_eff.spectral.seed_anchor_cache` | `false` | Live anchor populates `M_anchor` from zero |
| `comm_eff.spectral.ema_device` | `gpu` | `M_anchor` lives on HBM |
| `comm_eff.spectral.svd_mode` | `full` | Full thin SVD |
| `comm_eff.spectral.basis_cache` | `cache` | Reuse U/S/V across PPO mini-batches |
| `comm_eff.spectral.max_targets` | `4` | Smoke cap |
| `actor.use_kl_loss` | `False` | No KL term |
| `algorithm.use_kl_in_reward` | `False` | No KL in reward |
| `actor.entropy_coeff` | `0` | No entropy bonus |
| `actor.fsdp_config.use_orig_params` | `true` | Required so the spectral hook sees full 2D Tensor post-FSDP-reduce |

## Run shape (the recorded smoke)

| Knob | Value |
|---|---|
| GPUs | 4×H200 |
| `TRAIN_BATCH_SIZE` | 8 |
| `PPO_MINI_BATCH_SIZE` | 4 |
| `ROLLOUT_N` | 2 |
| `MAX_PROMPT_LENGTH` | 256 |
| `MAX_RESPONSE_LENGTH` | 256 |
| Trainer steps | 20 |
| Anchor fires expected | 10 |

The launcher in `examples/grpo_trainer/` defaults to baseline-scale
rollouts (`TRAIN_BATCH=128, ROLLOUT_N=8, MAX_RESPONSE=16384`) — overrideable
via env vars; the smoke configuration is captured in the
`REPRODUCIBILITY.md` recipe at `runs/communication-baseline/`.

## What this PASS verifies

1. **Implementation works end-to-end** — all comm-eff hooks fire,
   gradients flow, optimizer step happens.
2. **All six structural guards hold** — anchor never masked
   (`anchor_mask_applications=0`); anchor never spectrally corrected
   (`anchor_grad_corrected=0`); anchor never generates rollouts / recomputes
   rewards / takes optimizer steps (all 0).
3. **Mask is confined to the gradient-feeding forwards** — counters for
   rollout, ref_logprob, val, infer, ckpt are all 0; only train and
   old_logprob paths see masks.
4. **EMA learns** — `||dM_anchor||` evolves multi-order across the 10
   anchor fires (not stuck at zero).
5. **Visible learning under full compression** — reward second-half is
   +82% above first-half; three high-reward peaks late in the 20 steps.

## What this PASS does NOT verify

- Paper-scale rollout shape behavior (smoke uses tiny batch + 256 context).
  The investigation in `notes/investigation-prompt-grad-norm.md` addresses
  the symptoms observed at paper scale.
- Memory budget at paper scale (the anchor clone's ~3 GB park cost is
  documented in `notes/anchor-memory-cost.md`).
- Importance-sampling stability under independent PRF masks on the two
  gradient-feeding forwards (documented as a candidate root cause in the
  investigation prompt).

## How to re-run this exact configuration

See `runs/communication-baseline/REPRODUCIBILITY.md`.
