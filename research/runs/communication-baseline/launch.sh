#!/usr/bin/env bash
# EXP-9 launch — M2 final smoke (no KL, mask the full fast circuit including
# the old-logprob recompute, end-to-end 20-step GRPO). Single cell.
#
# Runs inside the verl-research-vllm020 Vast.ai container. The template's
# onstart has already:
#   * pulled the verlai/verl:vllm020.dev1 image,
#   * cloned shamanez/verl @ vast-ai-workload into /workspace/verl,
#   * pip-installed verl --no-deps,
#   * exported HF_TOKEN + WANDB_API_KEY into the container env.
# This script replaces the shipped vast-ai-workload checkout with the
# experimental exp/9-m2-final-noKL-maskrecompute-aps branch, re-installs verl
# in-place, pre-creates the per-experiment dirs, and launches the canonical
# smoke command from plan §"Smoke launch command".

set -euo pipefail

EXP_ID="EXP-9"
EXP_BRANCH="exp/9-m2-final-noKL-maskrecompute-aps"
RUN_DIR="/workspace/runs/${EXP_ID}"

# Identity for any in-container commits (commit-hotfix.sh uses these).
git config --global user.email "harness@verl-research.local"
git config --global user.name  "verl-research-harness"

# --- Apply the experimental bundle, replacing the vast-ai-workload checkout.
# Idempotent: if /workspace/verl is already on the exp branch (re-launch after
# a transient failure like missing secrets), skip the bundle clone + pip-install
# steps. We DO insist that the branch matches; a mismatch is a hard fail.
mkdir -p "$RUN_DIR/iterations" "$RUN_DIR/curve-snapshots" "$RUN_DIR/hotfix-patches"

if [[ -f "$RUN_DIR/exp.bundle" ]]; then
  cd /workspace
  CURRENT_BRANCH=""
  if [[ -d verl/.git ]]; then
    CURRENT_BRANCH="$(cd verl && git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '')"
  fi
  if [[ "$CURRENT_BRANCH" != "$EXP_BRANCH" ]]; then
    if [[ -d verl ]]; then
      # Preserve template-installed tree (only on the FIRST swap).
      if [[ ! -d verl.upstream-vast-ai-workload ]]; then
        mv verl verl.upstream-vast-ai-workload
      else
        rm -rf verl  # already preserved on a prior attempt; drop the duplicate
      fi
    fi
    git clone -b "$EXP_BRANCH" "$RUN_DIR/exp.bundle" verl
    cd /workspace/verl
    git remote set-url origin https://github.com/shamanez/verl.git || true
    pip install --no-deps -e . > /workspace/pip-exp-install.log 2>&1
  else
    echo "[launch.sh] /workspace/verl already on $EXP_BRANCH — skipping bundle clone + pip install"
  fi
fi

cd /workspace/verl

# --- Sanity guard: confirm we're on the EXP-9 branch with the mask_recompute
# field present in the schema. A loud assert here turns a mis-shipped bundle
# into an immediate failure instead of a silent EXP-3 baseline run.
git rev-parse --abbrev-ref HEAD | grep -qx "$EXP_BRANCH" || {
  echo "[launch.sh] FATAL: not on $EXP_BRANCH (HEAD=$(git rev-parse --abbrev-ref HEAD))" >&2
  exit 2
}
grep -q "mask_recompute" verl/workers/config/comm_eff.py || {
  echo "[launch.sh] FATAL: mask_recompute field missing in verl/workers/config/comm_eff.py" >&2
  exit 2
}

# Surface the discovery line the analyst greps for.
echo "[launch.sh] cwd=$(pwd) branch=$(git rev-parse --abbrev-ref HEAD) head=$(git rev-parse --short HEAD)"

# --- Canonical smoke command (plan §"Smoke launch command", verbatim).
PROJECT_NAME=verl_compression_research \
EXPERIMENT_NAME=m2-final-noKL-maskrecompute-aps \
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
  actor_rollout_ref.actor.comm_eff.mask.p=0.95 \
  actor_rollout_ref.actor.comm_eff.mask.mask_recompute=true \
  actor_rollout_ref.actor.comm_eff.spectral.enabled=true \
  actor_rollout_ref.actor.comm_eff.spectral.alpha=0.3 \
  actor_rollout_ref.actor.comm_eff.spectral.tau=0.001 \
  actor_rollout_ref.actor.comm_eff.spectral.beta_anc=0.9 \
  actor_rollout_ref.actor.comm_eff.spectral.seed_anchor_cache=false \
  actor_rollout_ref.actor.comm_eff.spectral.ema_device=gpu \
  actor_rollout_ref.actor.comm_eff.spectral.svd_mode=full \
  actor_rollout_ref.actor.comm_eff.spectral.basis_cache=cache \
  actor_rollout_ref.actor.comm_eff.spectral.max_targets=4 \
  actor_rollout_ref.actor.comm_eff.anchor.enabled=true \
  actor_rollout_ref.actor.comm_eff.anchor.cadence=4 \
  actor_rollout_ref.actor.comm_eff.anchor.delay_K=4 \
  2>&1 | tee "$RUN_DIR/train.log"

touch "$RUN_DIR/done.flag"
echo "$(date -Iseconds) [launch.sh] EXP-9 finished" >> "$RUN_DIR/done.flag"
