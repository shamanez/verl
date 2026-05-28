#!/usr/bin/env bash
# EXP-12 — 3-cell sequential smoke (REVISE child of EXP-8).
#
# Cells (per plan §Experiment design, non-negotiable #2 "STRICTLY SEQUENTIAL"):
#   1. m2-anchor-faithful  — anchor enabled, faithful (gpu/full/cache), max_targets=4
#   2. m2-anchor-off       — anchor DISABLED (EXP-7 spectral reproduction)
#   3. m2-anchor-lean      — anchor enabled, lean (cpu/lowrank/cache), max_targets=-1
#
# Each cell:
#   - runs trainer.total_training_steps=5
#   - has a unique EXPERIMENT_NAME (EXP-6 caveat: shared name auto-resumes)
#   - stdout redirected to /workspace/runs/EXP-12/train_<cell>.log
#   - writes /workspace/runs/EXP-12/done_<cell>.flag on exit (success OR fail)
#   - the next cell starts ONLY after the previous one exits — no `&`, no parallel
#
# `|| echo` after each cell keeps `set -e` from aborting the chain when one cell
# fails — the analyst grades partial chains (EXP-7 / EXP-8 precedent).
#
# Bundle handling (per experiment-runner contract): if exp.bundle is shipped,
# unpack the EXP-12 branch on top of the template-installed verl tree.
set -euo pipefail

RUN_DIR="/workspace/runs/EXP-12"
mkdir -p "$RUN_DIR/iterations" "$RUN_DIR/hotfix-patches" "$RUN_DIR/metrics"

# Configure git identity for in-container commits (commit-hotfix.sh uses these).
git config --global user.email "harness@verl-research.local" || true
git config --global user.name  "verl-research-harness" || true

# --- Apply the EXP-12 branch from the shipped bundle -----------------------
if [[ -f "$RUN_DIR/exp.bundle" ]]; then
  cd /workspace
  if [[ -d verl && ! -d verl.upstream-vast-ai-workload ]]; then
    mv verl verl.upstream-vast-ai-workload
  elif [[ -d verl ]]; then
    rm -rf verl
  fi
  git clone -b exp/12-anchor-detach "$RUN_DIR/exp.bundle" verl
  cd /workspace/verl
  # Point origin at the fork so any in-container push lands on shamanez/verl.
  git remote set-url origin https://github.com/shamanez/verl.git || true
  uv pip install --no-deps -e . > /workspace/pip.log 2>&1 || pip install --no-deps -e . > /workspace/pip.log 2>&1
fi

cd /workspace/verl

# --- Pre-flight: GPU count + per-tier vLLM mem utilization decision --------
# The orchestrator's training-log-monitor reads `gpu_memory_utilization` from the
# tmux log, so emit it loudly.
DETECTED_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
if (( DETECTED_GPUS < 4 || DETECTED_GPUS > 8 )); then
  echo "FATAL: this recipe requires 4..8 GPUs; detected $DETECTED_GPUS" >&2
  exit 1
fi
GPU_NAME=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | head -1 | tr -d ' ')
GPU_MEM_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')
GPU_MEM_GB=$(( GPU_MEM_MIB / 1024 ))
echo "=== EXP-12 launch: detected $DETECTED_GPUS x $GPU_NAME (${GPU_MEM_GB} GB each) ==="

# Consumer-card vLLM tuning (plan §Notes for runner): 32 GB cards -> 0.6,
# 24 GB cards (4090) -> 0.5, H100/H200 -> launcher default.
VLLM_GPU_MEM_UTIL_OVERRIDE=""
if (( GPU_MEM_GB <= 24 )); then
  VLLM_GPU_MEM_UTIL_OVERRIDE="actor_rollout_ref.rollout.gpu_memory_utilization=0.5"
  echo "=== EXP-12 launch: 24 GB-class consumer GPU detected -> gpu_memory_utilization=0.5 ==="
elif (( GPU_MEM_GB <= 32 )); then
  VLLM_GPU_MEM_UTIL_OVERRIDE="actor_rollout_ref.rollout.gpu_memory_utilization=0.6"
  echo "=== EXP-12 launch: 32 GB-class consumer GPU detected -> gpu_memory_utilization=0.6 ==="
else
  echo "=== EXP-12 launch: >32 GB GPU (H100/H200 class) -> launcher default gpu_memory_utilization ==="
fi

# --- Common smoke-shape env (mirrors plan §Smoke launch commands) ----------
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

# --- run_cell() — one tmux-style cell wrapper. -----------------------------
# Pattern modelled on EXP-7's vast_exp7_spectral_smoke.sh: per-cell log, per-cell
# done.flag, returns the cell's exit code so the chain wrapper can `|| echo`.
run_cell () {
  local cell_name="$1"; shift
  local cell_log="$RUN_DIR/train_${cell_name}.log"
  local cell_flag="$RUN_DIR/done_${cell_name}.flag"
  echo "=== [$(date -u +%FT%TZ)] EXP-12 cell '$cell_name' starting ==="
  EXPERIMENT_NAME="$cell_name" \
    bash examples/grpo_trainer/vast_baseline_qwen25_1p5b_grpo_gsm8k.sh \
      trainer.total_training_steps=5 \
      trainer.val_before_train=False \
      actor_rollout_ref.actor.ppo_epochs=1 \
      actor_rollout_ref.actor.fsdp_config.use_orig_params=true \
      ${VLLM_GPU_MEM_UTIL_OVERRIDE:+$VLLM_GPU_MEM_UTIL_OVERRIDE} \
      "$@" \
      > "$cell_log" 2>&1
  local rc=$?
  touch "$cell_flag"
  echo "=== [$(date -u +%FT%TZ)] EXP-12 cell '$cell_name' exited rc=$rc (log: $cell_log) ==="
  return $rc
}

# --- CELL 1: m2-anchor-faithful ============================================
run_cell m2-anchor-faithful \
  actor_rollout_ref.actor.comm_eff.enabled=true \
  actor_rollout_ref.actor.comm_eff.mask.enabled=true actor_rollout_ref.actor.comm_eff.mask.p=0.95 \
  actor_rollout_ref.actor.comm_eff.spectral.enabled=true \
  actor_rollout_ref.actor.comm_eff.spectral.alpha=0.3 \
  actor_rollout_ref.actor.comm_eff.spectral.tau=0.001 \
  actor_rollout_ref.actor.comm_eff.spectral.beta_anc=0.95 \
  actor_rollout_ref.actor.comm_eff.spectral.seed_anchor_cache=false \
  actor_rollout_ref.actor.comm_eff.spectral.ema_device=gpu \
  actor_rollout_ref.actor.comm_eff.spectral.svd_mode=full \
  actor_rollout_ref.actor.comm_eff.spectral.basis_cache=cache \
  actor_rollout_ref.actor.comm_eff.spectral.max_targets=4 \
  actor_rollout_ref.actor.comm_eff.anchor.enabled=true \
  actor_rollout_ref.actor.comm_eff.anchor.cadence=1 \
  actor_rollout_ref.actor.comm_eff.anchor.delay_K=1 \
  || echo "[EXP-12] cell m2-anchor-faithful exited $?"

# --- CELL 2: m2-anchor-off (EXP-7 reproduction) ============================
run_cell m2-anchor-off \
  actor_rollout_ref.actor.comm_eff.enabled=true \
  actor_rollout_ref.actor.comm_eff.mask.enabled=true actor_rollout_ref.actor.comm_eff.mask.p=0.95 \
  actor_rollout_ref.actor.comm_eff.spectral.enabled=true \
  actor_rollout_ref.actor.comm_eff.spectral.alpha=0.3 \
  actor_rollout_ref.actor.comm_eff.spectral.tau=0.001 \
  actor_rollout_ref.actor.comm_eff.spectral.beta_anc=0.95 \
  actor_rollout_ref.actor.comm_eff.spectral.seed_anchor_cache=true \
  actor_rollout_ref.actor.comm_eff.spectral.ema_device=gpu \
  actor_rollout_ref.actor.comm_eff.spectral.svd_mode=full \
  actor_rollout_ref.actor.comm_eff.spectral.basis_cache=cache \
  actor_rollout_ref.actor.comm_eff.spectral.max_targets=4 \
  actor_rollout_ref.actor.comm_eff.anchor.enabled=false \
  || echo "[EXP-12] cell m2-anchor-off exited $?"

# --- CELL 3: m2-anchor-lean ================================================
run_cell m2-anchor-lean \
  actor_rollout_ref.actor.comm_eff.enabled=true \
  actor_rollout_ref.actor.comm_eff.mask.enabled=true actor_rollout_ref.actor.comm_eff.mask.p=0.95 \
  actor_rollout_ref.actor.comm_eff.spectral.enabled=true \
  actor_rollout_ref.actor.comm_eff.spectral.alpha=0.3 \
  actor_rollout_ref.actor.comm_eff.spectral.tau=0.001 \
  actor_rollout_ref.actor.comm_eff.spectral.beta_anc=0.95 \
  actor_rollout_ref.actor.comm_eff.spectral.seed_anchor_cache=false \
  actor_rollout_ref.actor.comm_eff.spectral.ema_device=cpu \
  actor_rollout_ref.actor.comm_eff.spectral.svd_mode=lowrank \
  actor_rollout_ref.actor.comm_eff.spectral.rank=8 \
  actor_rollout_ref.actor.comm_eff.spectral.basis_cache=cache \
  actor_rollout_ref.actor.comm_eff.spectral.max_targets=-1 \
  actor_rollout_ref.actor.comm_eff.anchor.enabled=true \
  actor_rollout_ref.actor.comm_eff.anchor.cadence=1 \
  actor_rollout_ref.actor.comm_eff.anchor.delay_K=1 \
  || echo "[EXP-12] cell m2-anchor-lean exited $?"

# Aggregate done.flag — the orchestrator's done-flag predicate reads this AND
# the monitor cross-checks the per-cell logs (chain-doesn't-abort wrapper writes
# this through silent Ray errors per EXP-8 precedent).
touch "$RUN_DIR/done.flag"
echo "=== EXP-12 chain done at $(date -u +%FT%TZ) ==="
