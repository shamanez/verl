# Communication-efficient baseline — reproducibility manifest

Permanent reference run for the communication-efficient GRPO method:
Qwen2.5-1.5B-Instruct on GSM8K, 4×H200, single-cell, 20 trainer steps.

## Source-of-truth: the launcher

The launcher lives at
`examples/grpo_trainer/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh` on
the `vast-ai-workload` branch. **Do not** copy or fork the launcher into
`runs/communication-baseline/`; the one in `examples/` is the single source
of truth.

## Method configuration (the load-bearing knobs)

- **Objective**: pg_loss only (`actor.use_kl_loss=False`,
  `algorithm.use_kl_in_reward=False`, `actor.entropy_coeff=0`).
- **Activation mask**: PRF Bernoulli at pipeline-boundary decoder blocks,
  `p=0.9`, `mask_recompute=true` (fires on BOTH the actor-train forward and
  the `compute_log_prob` recompute — the two gradient-feeding forwards).
- **Anchor circuit**: hookless clone of the live FSDP module, refresh every
  5 PPO substeps from a 5-substep-stale weight snapshot.
- **Spectral correction**: `alpha=0.5`, `tau=0.01`, `beta_anc=0.9`,
  `seed_anchor_cache=false`, `ema_device=gpu`, `svd_mode=full`,
  `basis_cache=cache`, `max_targets=4`.
- **FSDP**: `use_orig_params=true` so the post-reduce gradient surfaces as a
  full 2D Tensor (the spectral correction needs the unsharded matrix).

## Run shape (smoke)

- 4×H200, 1 instance, ~14 min wall (~$3-4).
- TRAIN_BATCH=8, ROLLOUT_N=2, MAX_PROMPT=256, MAX_RESPONSE=256.
- 20 trainer steps; cadence=5 ⇒ 8 anchor fires; mask_recompute=true gives
  per-step ratio mask_applications/{train,old_logprob} ≈ 2:1.

## Headline result

- All 13 success criteria PASS (mask infrastructure + anchor isolation
  guards + spectral correction firing + visible learning).
- Reward curve over the 20 steps: mean(steps 11-20)=0.125 vs mean(steps
  1-10)=0.069 → +82% second-half improvement.
- Three 0.25 reward peaks at steps 12 / 17 / 18 (4× the step-1 value).
- All anchor guards held: `anchor_mask_applications=0`,
  `anchor_grad_corrected=0`, `anchor_rollouts_generated=0`,
  `anchor_rewards_recomputed=0`, `anchor_optimizer_steps=0`.
- `||dM_anchor||_mean` evolves multi-order across the 10 anchor fires
  (0.013 → 0.49), confirming the anchor EMA is actually learning.

## Re-running this exact configuration

From any laptop with `~/.config/verl-research/secrets.env` (HF + WandB +
VAST keys) and the SSH key registered on Vast:

```bash
cd /path/to/verl
git fetch origin
git checkout vast-ai-workload
git pull --ff-only

# Provision a Vast.ai box
source ~/.config/verl-research/secrets.env
bash research/.claude/skills/vast-provision/run.sh \
  --query 'num_gpus=4 gpu_name=H200 gpu_ram>=140 cuda_max_good>=13.0 reliability>=0.95 rentable=true verified=true' \
  --max-price 24.0 --count 1 --disk-gb 200

# Push a stripped HF + WandB secrets file (NO VAST_API_KEY) to the box, then:
ssh -i ~/.ssh/vast_ai -p <port> root@<host> '
  cd /workspace/verl && git pull && \
  TRAIN_BATCH_SIZE=8 PPO_MINI_BATCH_SIZE=4 ROLLOUT_N=2 \
  MAX_PROMPT_LENGTH=256 MAX_RESPONSE_LENGTH=256 \
  PPO_MAX_TOKEN_LEN_PER_GPU=4096 LOG_PROB_MAX_TOKEN_LEN_PER_GPU=4096 \
  TOTAL_EPOCHS=1 TOTAL_TRAINING_STEPS=20 TEST_FREQ=-1 SAVE_FREQ=-1 \
  bash examples/grpo_trainer/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh
'
```

## Artifacts

- `verdict.md` — the PASS verdict + 13-criterion checklist
- `train.log` — full training log (Ray + driver, all 20 step metric lines)
- `done.flag` — clean-exit marker
- `verify/20260528T160000.md` — operator-override pre-skip of `codex-verify`
