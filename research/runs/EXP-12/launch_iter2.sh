#!/usr/bin/env bash
# EXP-12 iter2 — 2 anchor-enabled cells, 10 steps each, post-hot-fix relaunch.
# Cells: m2-anchor-faithful-iter2 (faithful storage), m2-anchor-lean-iter2 (lean storage).
# Cell 2 (m2-anchor-off) already PASSED at 5 steps (iter1) — no need to re-run.
set -uo pipefail
RUN_DIR=/workspace/runs/EXP-12
cd /workspace/verl

export PROJECT_NAME=verl_compression_research
export TRAIN_BATCH_SIZE=8
export PPO_MINI_BATCH_SIZE=4
export ROLLOUT_N=2
export MAX_PROMPT_LENGTH=256
export MAX_RESPONSE_LENGTH=256
export PPO_MAX_TOKEN_LEN_PER_GPU=4096
export LOG_PROB_MAX_TOKEN_LEN_PER_GPU=4096
export REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU=4096
export ENTROPY_COEFF=0.001
export SAVE_FREQ=-1
export TEST_FREQ=-1
export TOTAL_EPOCHS=1

run_cell () {
  local NAME=$1
  shift
  local LOG=$RUN_DIR/train_${NAME}.log
  local FLAG=$RUN_DIR/done_${NAME}.flag
  rm -f $FLAG
  echo "=== EXP-12 iter2 START cell=$NAME at $(date -Iseconds) ===" | tee -a $LOG
  EXPERIMENT_NAME=$NAME \
  bash examples/grpo_trainer/vast_baseline_qwen25_1p5b_grpo_gsm8k.sh \
    trainer.total_training_steps=10 \
    trainer.val_before_train=False \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.fsdp_config.use_orig_params=true \
    actor_rollout_ref.actor.comm_eff.enabled=true \
    actor_rollout_ref.actor.comm_eff.mask.enabled=true actor_rollout_ref.actor.comm_eff.mask.p=0.95 \
    actor_rollout_ref.actor.comm_eff.spectral.enabled=true actor_rollout_ref.actor.comm_eff.spectral.alpha=0.3 actor_rollout_ref.actor.comm_eff.spectral.tau=0.001 actor_rollout_ref.actor.comm_eff.spectral.beta_anc=0.95 \
    actor_rollout_ref.actor.comm_eff.spectral.seed_anchor_cache=false \
    actor_rollout_ref.actor.comm_eff.anchor.enabled=true actor_rollout_ref.actor.comm_eff.anchor.cadence=1 actor_rollout_ref.actor.comm_eff.anchor.delay_K=1 \
    "$@" > $LOG 2>&1
  local EC=$?
  touch $FLAG
  echo "=== EXP-12 iter2 END cell=$NAME exit=$EC at $(date -Iseconds) ===" | tee -a $LOG
  return $EC
}

# Cell A — faithful storage, max_targets=4, 10 steps.
run_cell m2-anchor-faithful-iter2 \
  actor_rollout_ref.actor.comm_eff.spectral.ema_device=gpu \
  actor_rollout_ref.actor.comm_eff.spectral.svd_mode=full \
  actor_rollout_ref.actor.comm_eff.spectral.basis_cache=cache \
  actor_rollout_ref.actor.comm_eff.spectral.max_targets=4 \
  || echo "[iter2] cell faithful exit $?"

# Cell B — lean storage, max_targets=-1, 10 steps.
run_cell m2-anchor-lean-iter2 \
  actor_rollout_ref.actor.comm_eff.spectral.ema_device=cpu \
  actor_rollout_ref.actor.comm_eff.spectral.svd_mode=lowrank \
  actor_rollout_ref.actor.comm_eff.spectral.rank=8 \
  actor_rollout_ref.actor.comm_eff.spectral.basis_cache=cache \
  actor_rollout_ref.actor.comm_eff.spectral.max_targets=-1 \
  || echo "[iter2] cell lean exit $?"

touch $RUN_DIR/done_iter2.flag
echo "=== EXP-12 iter2 chain done at $(date -Iseconds) ==="
