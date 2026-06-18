#!/usr/bin/env bash
# EXP-36 — accelerated-surface re-baseline: efficient comm_eff C2 (signed_ema a=0.25) + dense control.
# 2 cells back-to-back, one reused 4xH200 box. CELL 1 (efficient comm_eff peak) FIRST, then CELL 2 (dense).
# Code: exp/spectral-diagnostics-knob @ 3300cc61 (adds comm_eff.spectral.diagnostics knob).
# set -uo pipefail (NOT -e): a crashed/ignited cell must NOT abort the remaining cell — a kill IS data.
set -uo pipefail
cd /workspace/verl
git config --global user.email "harness@verl-research.local" 2>/dev/null || true
git config --global user.name  "verl-research-harness" 2>/dev/null || true

RUNROOT=/workspace/runs/EXP-36
mkdir -p "$RUNROOT"

# GSM8K is pre-staged at ~/data/gsm8k; the launcher re-preps idempotently if missing. Wait briefly.
for _ in $(seq 1 60); do [[ -f ~/data/gsm8k/train.parquet && -f ~/data/gsm8k/test.parquet ]] && break; sleep 5; done

# run_cell NAME LAUNCHER_REL_PATH [extra hydra args...]
# COMM_EFF_ENABLED is read from the environment (default unset -> launcher default true);
# CELL 2 exports COMM_EFF_ENABLED=false before calling this.
run_cell () {
  local NAME="$1"; shift
  local LAUNCHER="$1"; shift
  local SRC DST RC
  echo "=== [$(date -u +%FT%TZ)] CELL $NAME START (launcher=$LAUNCHER comm_eff_enabled=${COMM_EFF_ENABLED:-<launcher-default>}) ==="
  # Common acceleration env + fixed run-control, shared by BOTH cells.
  USE_DYNAMIC_BSZ=True \
  PPO_MAX_TOKEN_LEN_PER_GPU=24576 \
  LOG_PROB_MAX_TOKEN_LEN_PER_GPU=32768 \
  REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU=32768 \
  MAX_RESPONSE_LENGTH=2048 \
  ROLLOUT_TP=1 \
  ROLLOUT_GPU_MEM_UTIL=0.75 \
  TRAIN_BATCH_SIZE=128 \
  PPO_MINI_BATCH_SIZE=64 \
  TOTAL_TRAINING_STEPS=50 \
  TEST_FREQ=25 \
  PROJECT_NAME=verl_compression_research_accel_rebaseline \
  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  EXPERIMENT_NAME="$NAME" \
    bash "examples/grpo_trainer/$LAUNCHER" \
    actor_rollout_ref.rollout.enable_chunked_prefill=true \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096 \
    actor_rollout_ref.actor.fsdp_config.forward_prefetch=true \
    actor_rollout_ref.ref.fsdp_config.forward_prefetch=true \
    trainer.save_freq=0 \
    trainer.val_before_train=False \
    "$@"
  RC=$?
  echo "=== [$(date -u +%FT%TZ)] CELL $NAME END rc=$RC ==="
  SRC=/workspace/verl/runs/$NAME
  DST=$RUNROOT/$NAME
  mkdir -p "$DST"
  cp -f "$SRC/train.log" "$DST/train.log" 2>/dev/null || true
  cp -f "$SRC/done.flag" "$DST/done.flag" 2>/dev/null || true
  cp -f "$SRC/EARLY_STOP_SIGNAL" "$DST/EARLY_STOP_SIGNAL" 2>/dev/null || true
  echo "$RC" > "$DST/rc.txt"
  # Best-effort ground-truth resolved-config capture from the set -x trace (never abort the run).
  if [[ -f research/scripts/capture_resolved_config.py ]]; then
    python research/scripts/capture_resolved_config.py "$DST" >/dev/null 2>&1 || true
  fi
}

# ---- CELL 1: efficient comm_eff peak (signed_ema alpha=0.25, beta_anc=0.50, diagnostics=false) ----
# delayed_ef base launcher + signed_ema Hydra override (last-wins). ema_device/snapshot_device kept
# at launcher cpu defaults (operator rejected gpu as OOM-risky).
run_cell exp-36-c2eff-a025 vast_comm_eff_b2_sota_qwen25_1p5b_grpo_gsm8k.sh \
  actor_rollout_ref.actor.comm_eff.spectral.correction_mode=signed_ema \
  actor_rollout_ref.actor.comm_eff.spectral.signed_ema_alpha=0.25 \
  actor_rollout_ref.actor.comm_eff.spectral.beta_anc=0.50 \
  actor_rollout_ref.actor.comm_eff.spectral.diagnostics=false

# ---- CELL 2: dense control = comm_eff OFF ----
# COMM_EFF_ENABLED=false drives the dense path; the trailing comm_eff.enabled=false is belt-and-
# suspenders (last-wins). NO diagnostics override (comm_eff off => irrelevant key for this cell).
COMM_EFF_ENABLED=false \
  run_cell exp-36-dense vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh \
  actor_rollout_ref.actor.comm_eff.enabled=false

touch "$RUNROOT/done.flag"
echo "=== EXP-36 ALL CELLS COMPLETE $(date -u +%FT%TZ) ==="
