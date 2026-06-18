#!/usr/bin/env bash
# EXP-36B — BIT-NEUTRAL confirmation of EXP-35 C2 (c2eff @ signed_ema alpha=0.25, beta_anc=0.50).
# ONE cell, reused 4xH200 box 41420622 (team account). Verifies that turning the comm_eff
# spectral DIAGNOSTICS OFF (diagnostics=false) reproduces EXP-35 C2's val@50 = 0.7528.
#
# The ONLY result-affecting delta vs the proven EXP-35 C2 config is:
#     actor_rollout_ref.actor.comm_eff.spectral.diagnostics=false
# Everything else is byte-for-byte EXP-35 C2: gpu_mem_util=0.55, NO chunked_prefill,
# NO forward_prefetch (default off), signed_ema alpha=0.25 / beta_anc=0.50, ema/snapshot
# device cpu (launcher defaults). save_freq=0 + update_weights_bucket_megabytes=4096 are
# non-result-affecting (checkpointing/weight-sync plumbing; greedy val is independent).
# Code: exp/spectral-diagnostics-knob @ 3300cc61 (adds comm_eff.spectral.diagnostics knob).
# set -uo pipefail (NOT -e): a crashed/ignited cell must NOT silently abort post-processing.
set -uo pipefail
cd /workspace/verl
git config --global user.email "harness@verl-research.local" 2>/dev/null || true
git config --global user.name  "verl-research-harness" 2>/dev/null || true

RUNROOT=/workspace/runs/EXP-36B
mkdir -p "$RUNROOT"

# GSM8K is pre-staged at ~/data/gsm8k; the launcher re-preps idempotently if missing. Wait briefly.
for _ in $(seq 1 60); do [[ -f ~/data/gsm8k/train.parquet && -f ~/data/gsm8k/test.parquet ]] && break; sleep 5; done

# run_cell NAME LAUNCHER_REL_PATH [extra hydra args...]
run_cell () {
  local NAME="$1"; shift
  local LAUNCHER="$1"; shift
  local SRC DST RC
  echo "=== [$(date -u +%FT%TZ)] CELL $NAME START (launcher=$LAUNCHER) ==="
  # Acceleration env + fixed run-control. gpu_mem_util=0.55 (EXP-35 C2), NO chunked_prefill override.
  USE_DYNAMIC_BSZ=True \
  PPO_MAX_TOKEN_LEN_PER_GPU=24576 \
  LOG_PROB_MAX_TOKEN_LEN_PER_GPU=32768 \
  REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU=32768 \
  MAX_RESPONSE_LENGTH=2048 \
  ROLLOUT_TP=1 \
  ROLLOUT_GPU_MEM_UTIL=0.55 \
  TRAIN_BATCH_SIZE=128 \
  PPO_MINI_BATCH_SIZE=64 \
  TOTAL_TRAINING_STEPS=50 \
  TEST_FREQ=25 \
  PROJECT_NAME=verl_compression_research_accel_rebaseline \
  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  EXPERIMENT_NAME="$NAME" \
    bash "examples/grpo_trainer/$LAUNCHER" \
    actor_rollout_ref.actor.comm_eff.spectral.correction_mode=signed_ema \
    actor_rollout_ref.actor.comm_eff.spectral.signed_ema_alpha=0.25 \
    actor_rollout_ref.actor.comm_eff.spectral.beta_anc=0.50 \
    actor_rollout_ref.actor.comm_eff.spectral.diagnostics=false \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096 \
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

# ---- SINGLE cell: c2eff @ signed_ema alpha=0.25, beta_anc=0.50, DIAGNOSTICS OFF, gpu_mem_util 0.55 ----
run_cell exp-36-c2eff-055-diag vast_comm_eff_b2_sota_qwen25_1p5b_grpo_gsm8k.sh

touch "$RUNROOT/done.flag"
echo "=== EXP-36B CELL COMPLETE $(date -u +%FT%TZ) ==="
