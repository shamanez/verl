#!/usr/bin/env bash
# EXP-13 iter2 — OOM mitigation: halve PPO_MAX_TOKEN_LEN_PER_GPU + drop vLLM mem-util
# Keep user-mandated knobs: TRAIN_BATCH=128, ROLLOUT_N=8, MAX_PROMPT=1024, MAX_RESPONSE=16384
set -euo pipefail
cd /workspace/verl
mkdir -p /workspace/runs/EXP-13
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if [ -f /root/.config/verl-research/secrets.env ]; then
  set -a; . /root/.config/verl-research/secrets.env; set +a
fi
echo "=== EXP-13 iter2 launch $(date -Iseconds) ==="
echo "  fix: PPO_MAX_TOKEN_LEN_PER_GPU=18432 (was 36864)"
echo "  fix: vllm gpu_memory_utilization=0.3 (was 0.4)"
echo "  fix: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
PROJECT_NAME=verl_compression_research \
EXPERIMENT_NAME=exp13-100step-m90ap-c5-val25-iter2 \
TRAIN_BATCH_SIZE=128 PPO_MINI_BATCH_SIZE=32 ROLLOUT_N=8 \
MAX_PROMPT_LENGTH=1024 MAX_RESPONSE_LENGTH=16384 \
PPO_MAX_TOKEN_LEN_PER_GPU=18432 LOG_PROB_MAX_TOKEN_LEN_PER_GPU=18432 \
SAVE_FREQ=-1 TEST_FREQ=25 TOTAL_EPOCHS=1 \
bash examples/grpo_trainer/vast_baseline_qwen25_1p5b_grpo_gsm8k.sh \
  trainer.total_training_steps=100 \
  trainer.val_before_train=True \
  trainer.test_freq=25 \
  actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.fsdp_config.use_orig_params=true \
  actor_rollout_ref.actor.use_kl_loss=False \
  algorithm.use_kl_in_reward=False \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
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
  actor_rollout_ref.actor.comm_eff.anchor.cadence=5 \
  actor_rollout_ref.actor.comm_eff.anchor.delay_K=5 \
  2>&1 | tee /workspace/runs/EXP-13/train_iter2.log
echo "$(date -Iseconds) [launch_iter2.sh] EXP-13 iter2 finished" > /workspace/runs/EXP-13/done_iter2.flag
