#!/usr/bin/env bash
# EXP-13 — 100-step comm-eff PP-RL run on Qwen2.5-1.5B + GSM8K, baseline-scale rollouts
# Lineage:  EXP-9 iter2 PASS (knobs verified to produce visible learning at 20 steps).
# Goal:     extend +82% improvement signal across 100 trainer steps with periodic
#           validation every 25 steps for a clean GRPO learning curve.
#
# Knob choices (user-confirmed 2026-05-28):
#   - Compression family: M90+AP (iter2 PASS)
#   - mask.p              = 0.9     (retain 10% activation surface)
#   - mask.mask_recompute = true    (EXP-9 extension — both fast-circuit forwards)
#   - spectral.alpha      = 0.5     (mid blend, more raw masked grad than EXP-9 iter1)
#   - spectral.tau        = 0.01    (broader Tikhonov tail-damping)
#   - spectral.beta_anc   = 0.9     (EMA decay)
#   - anchor.cadence      = 5       (anchor every 5 PPO substeps — user request)
#   - anchor.delay_K      = 5       (5-substep weight staleness)
#   - svd_mode, ema_device, basis_cache: full/gpu/cache (faithful)
#   - max_targets         = 4       (smoke cap; same as EXP-9)
#
# Rollouts/batch/lengths: SAME AS EXP-3 BASELINE
#   - TRAIN_BATCH_SIZE     = 128
#   - ROLLOUT_N            = 8
#   - MAX_PROMPT_LENGTH    = 1024
#   - MAX_RESPONSE_LENGTH  = 16384  (16K — full context like baseline)
#   - PPO/log_prob max tokens per GPU = 36864
#
# Validation schedule:
#   - val_before_train = true        (baseline reading at step 0)
#   - test_freq        = 25          (val at steps 25, 50, 75, 100)
#   ⇒ 5 validation runs total (steps 0, 25, 50, 75, 100)
#
# No KL, no entropy bonus (EXP-9 inheritance).
set -euo pipefail
cd /workspace/verl
git status --short | head -5

mkdir -p /workspace/runs/EXP-13/{iterations,curve-snapshots}

# Source secrets (HF + WandB), already on box
if [ -f /root/.config/verl-research/secrets.env ]; then
  set -a
  . /root/.config/verl-research/secrets.env
  set +a
fi

echo "=== EXP-13 launch $(date -Iseconds) ==="
echo "  100-step M90+AP comm-eff GRPO on Qwen2.5-1.5B / GSM8K"
echo "  baseline-scale: TRAIN_BATCH=128 ROLLOUT_N=8 MAX_RESPONSE=16384"
echo "  val_before_train=true test_freq=25 ⇒ val at 0, 25, 50, 75, 100"

PROJECT_NAME=verl_compression_research \
EXPERIMENT_NAME=exp13-100step-m90ap-c5-val25 \
TRAIN_BATCH_SIZE=128 PPO_MINI_BATCH_SIZE=32 ROLLOUT_N=8 \
MAX_PROMPT_LENGTH=1024 MAX_RESPONSE_LENGTH=16384 \
PPO_MAX_TOKEN_LEN_PER_GPU=36864 LOG_PROB_MAX_TOKEN_LEN_PER_GPU=36864 \
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
  2>&1 | tee /workspace/runs/EXP-13/train.log

echo "$(date -Iseconds) [launch.sh] EXP-13 finished" > /workspace/runs/EXP-13/done.flag
