#!/usr/bin/env bash
# EXP-9 iter2 hot-fix per analyst REVISE — knob relaxation:
#   spectral.alpha 0.3 -> 0.5  (more raw masked grad, less spectral mix)
#   spectral.tau   0.001 -> 0.01  (broader Tikhonov damping)
#   mask.p         0.95 -> 0.9   (double retained activation surface 5% -> 10%)
# Goal: criterion 13 "visible learning" passes via reduced compression aggressiveness
set -euo pipefail
cd /workspace/verl
git status --short | head -5

mkdir -p /workspace/runs/EXP-9/iterations

# Source secrets (HF + WandB), already on box
if [ -f /root/.config/verl-research/secrets.env ]; then
  set -a
  . /root/.config/verl-research/secrets.env
  set +a
fi

echo "=== EXP-9 iter2 launch $(date -Iseconds) ==="
echo "  spectral.alpha=0.5 (iter1=0.3)"
echo "  spectral.tau=0.01 (iter1=0.001)"
echo "  mask.p=0.9 (iter1=0.95, mask_ratio target shifts to 0.9 +- 0.02)"

PROJECT_NAME=verl_compression_research \
EXPERIMENT_NAME=m2-final-noKL-maskrecompute-aps-iter2 \
TRAIN_BATCH_SIZE=8 PPO_MINI_BATCH_SIZE=4 ROLLOUT_N=2 \
MAX_PROMPT_LENGTH=256 MAX_RESPONSE_LENGTH=256 \
PPO_MAX_TOKEN_LEN_PER_GPU=4096 LOG_PROB_MAX_TOKEN_LEN_PER_GPU=4096 \
SAVE_FREQ=-1 TEST_FREQ=-1 TOTAL_EPOCHS=1 \
bash examples/grpo_trainer/vast_baseline_qwen25_1p5b_grpo_gsm8k.sh \
  trainer.total_training_steps=20 \
  trainer.val_before_train=False \
  actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.fsdp_config.use_orig_params=true \
  actor_rollout_ref.actor.use_kl_loss=False \
  algorithm.use_kl_in_reward=False \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.comm_eff.enabled=true \
  actor_rollout_ref.actor.comm_eff.mask.enabled=true \
  actor_rollout_ref.actor.comm_eff.mask.p=0.9 \
  actor_rollout_ref.actor.comm_eff.mask.mask_recompute=true \
  actor_rollout_ref.actor.comm_eff.spectral.enabled=true \
  actor_rollout_ref.actor.comm_eff.spectral.alpha=0.5 \
  actor_rollout_ref.actor.comm_eff.spectral.tau=0.01 \
  actor_rollout_ref.actor.comm_eff.spectral.beta_anc=0.9 \
  actor_rollout_ref.actor.comm_eff.spectral.seed_anchor_cache=false \
  actor_rollout_ref.actor.comm_eff.spectral.ema_device=gpu \
  actor_rollout_ref.actor.comm_eff.spectral.svd_mode=full \
  actor_rollout_ref.actor.comm_eff.spectral.basis_cache=cache \
  actor_rollout_ref.actor.comm_eff.spectral.max_targets=4 \
  actor_rollout_ref.actor.comm_eff.anchor.enabled=true \
  actor_rollout_ref.actor.comm_eff.anchor.cadence=4 \
  actor_rollout_ref.actor.comm_eff.anchor.delay_K=4 \
  2>&1 | tee /workspace/runs/EXP-9/train_iter2.log

echo "$(date -Iseconds) [launch_iter2.sh] EXP-9 iter2 finished" > /workspace/runs/EXP-9/done_iter2.flag
