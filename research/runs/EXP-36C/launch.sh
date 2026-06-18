#!/usr/bin/env bash
# EXP-36C — DENSE CONTROL at the CLEAN @0.55 surface. Apples-to-apples dense
# baseline matching EXP-36B (the clean comm_eff confirmation cell, val@50 target
# ~0.7528) EXCEPT comm_eff is OFF. This is the dense bar for the accelerated
# clean-@0.55 re-baseline.
#
# Surface = EXP-36B's EXACTLY, with comm_eff disabled:
#   gpu_mem_util=0.55, NO enable_chunked_prefill override (inner script default False),
#   NO forward_prefetch override (config default False), MAX_RESPONSE_LENGTH=2048,
#   USE_DYNAMIC_BSZ, PPO_MAX_TOKEN_LEN_PER_GPU=24576, log/ref token len 32768,
#   batch128/mini64, 50 steps, test_freq 25, val_before_train=False, save_freq=0.
#
# comm_eff disabled robustly: COMM_EFF_ENABLED=false (drives the dense path in the
# launcher, line 219->440) PLUS the trailing actor...comm_eff.enabled=false Hydra
# override (last-wins, belt-and-suspenders). NO diagnostics override (comm_eff off
# => spectral.diagnostics is an irrelevant key for this cell).
# Code: vast-ai-workload @ 3300cc61 (comm_eff OFF => spectral-diagnostics knob irrelevant for dense).
# set -uo pipefail (NOT -e): a crashed cell must NOT silently abort post-processing.
set -uo pipefail
cd /workspace/verl
git config --global user.email "harness@verl-research.local" 2>/dev/null || true
git config --global user.name  "verl-research-harness" 2>/dev/null || true

RUNROOT=/workspace/runs/EXP-36C
mkdir -p "$RUNROOT"

# GSM8K is pre-staged at ~/data/gsm8k; the launcher re-preps idempotently if missing. Wait briefly.
for _ in $(seq 1 60); do [[ -f ~/data/gsm8k/train.parquet && -f ~/data/gsm8k/test.parquet ]] && break; sleep 5; done

# run_cell NAME LAUNCHER_REL_PATH [extra hydra args...]
run_cell () {
  local NAME="$1"; shift
  local LAUNCHER="$1"; shift
  local SRC DST RC
  echo "=== [$(date -u +%FT%TZ)] CELL $NAME START (launcher=$LAUNCHER comm_eff_enabled=${COMM_EFF_ENABLED:-<launcher-default>}) ==="
  # Clean @0.55 acceleration env + fixed run-control. gpu_mem_util=0.55 (EXP-36B surface),
  # NO chunked_prefill override (inner run_qwen3_4b_fsdp.sh hardcodes enable_chunked_prefill=False),
  # NO forward_prefetch override (ppo_trainer config default False).
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
    actor_rollout_ref.actor.comm_eff.enabled=false \
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

# ---- SINGLE cell: DENSE control (comm_eff OFF) at the clean @0.55 surface ----
# COMM_EFF_ENABLED=false drives the dense path; trailing comm_eff.enabled=false is
# belt-and-suspenders (last-wins). DENSE launcher = vast_comm_eff_baseline (run it
# with comm_eff off => byte-identical-to-upstream dense training).
COMM_EFF_ENABLED=false \
  run_cell exp-36-dense-055 vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh

touch "$RUNROOT/done.flag"
echo "=== EXP-36C CELL COMPLETE $(date -u +%FT%TZ) ==="
