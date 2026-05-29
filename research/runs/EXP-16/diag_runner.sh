#!/usr/bin/env bash
# EXP-16 grad-norm comparison on the REAL 4xB200/FSDP setup.
# Two 3-step runs to compare against cell 1 (mask, NO rescale; already done):
#   A) mask + rescale
#   B) fully dense (COMM_EFF_ENABLED=false — all comm-eff off)
# No pre-train validation, no mid validation, no checkpoint saving (fast).
set -uo pipefail
cd /workspace/verl
export HF_HOME=/workspace/.hf_home
DIAG=/workspace/runs/EXP-16/diag
mkdir -p "$DIAG"
LAUNCHER=examples/grpo_trainer/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh

cleanup(){
  pkill -9 -f "verl.trainer.main_ppo" 2>/dev/null || true
  pkill -9 -f "vLLMHttpServer" 2>/dev/null || true
  ray stop --force >/dev/null 2>&1 || true
  sleep 6
}

run(){
  local name="$1"; shift
  echo "==== DIAG RUN $name START $(date -u +%FT%TZ) ===="
  # launcher default LOG = /workspace/verl/runs/<NAME>/train.log (dir created by it,
  # and done.flag lands there too — no path mismatch since we don't override LOG).
  env "$@" \
      VAL_BEFORE_TRAIN=False TOTAL_TRAINING_STEPS=3 TEST_FREQ=999 SAVE_FREQ=999 \
      EXPERIMENT_NAME="$name" \
      bash "$LAUNCHER" > "$DIAG/$name.console" 2>&1
  echo "==== DIAG RUN $name rc=$? END $(date -u +%FT%TZ) ===="
}

# A) mask + rescale (3 steps)
run diag_mask_rescale_3step \
  COMM_EFF_ENABLED=true COMM_EFF_MASK_ENABLED=true COMM_EFF_MASK_P=0.9 \
  COMM_EFF_MASK_RESCALE=true COMM_EFF_MASK_RECOMPUTE=true COMM_EFF_CLEAN_CADENCE=0 \
  COMM_EFF_ANCHOR_ENABLED=false COMM_EFF_SPECTRAL_ENABLED=false
cleanup

# B) fully dense — all communication-efficiency switched off (3 steps)
run diag_dense_off_3step COMM_EFF_ENABLED=false
cleanup

echo "==== ALL DIAG RUNS DONE $(date -u +%FT%TZ) ===="
touch "$DIAG/DIAG_DONE"
