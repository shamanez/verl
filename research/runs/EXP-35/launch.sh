#!/usr/bin/env bash
# EXP-35 — signed_ema alpha sweep on accelerated 4xH200 surface (dynamic bsz + TP=1 + max_response 2048).
# 5 cells back-to-back, one box. C3 (alpha=0.5 control / surface-validation gate) runs FIRST.
# One-knob: every cell differs ONLY in signed_ema_alpha. set -uo pipefail (NOT -e): an ignited/crashed
# cell must NOT abort the remaining cells — a kill IS data.
set -uo pipefail
cd /workspace/verl
git config --global user.email "harness@verl-research.local" 2>/dev/null || true
git config --global user.name  "verl-research-harness" 2>/dev/null || true

RUNROOT=/workspace/runs/EXP-35
mkdir -p "$RUNROOT"

# GSM8K is pre-staged at ~/data/gsm8k; the launcher re-preps idempotently if missing. Wait briefly.
for _ in $(seq 1 60); do [[ -f ~/data/gsm8k/train.parquet && -f ~/data/gsm8k/test.parquet ]] && break; sleep 5; done

run_cell () {
  local NAME="$1"
  local ALPHA="$2"
  local SRC DST RC
  echo "=== [$(date -u +%FT%TZ)] CELL $NAME alpha=$ALPHA START ==="
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
  PROJECT_NAME=verl_compression_research_alpha_sweep_signed_ema \
  EXPERIMENT_NAME="$NAME" \
    bash examples/grpo_trainer/vast_comm_eff_b2_sota_qwen25_1p5b_grpo_gsm8k.sh \
    actor_rollout_ref.actor.comm_eff.spectral.correction_mode=signed_ema \
    actor_rollout_ref.actor.comm_eff.spectral.signed_ema_alpha="$ALPHA" \
    actor_rollout_ref.actor.comm_eff.spectral.beta_anc=0.50 \
    trainer.val_before_train=False
  RC=$?
  echo "=== [$(date -u +%FT%TZ)] CELL $NAME alpha=$ALPHA END rc=$RC ==="
  SRC=/workspace/verl/runs/$NAME
  DST=$RUNROOT/$NAME
  mkdir -p "$DST"
  cp -f "$SRC/train.log" "$DST/train.log" 2>/dev/null || true
  cp -f "$SRC/done.flag" "$DST/done.flag" 2>/dev/null || true
  cp -f "$SRC/EARLY_STOP_SIGNAL" "$DST/EARLY_STOP_SIGNAL" 2>/dev/null || true
  echo "$RC" > "$DST/rc.txt"
}

# C3 control FIRST (surface-validation gate known early), then endpoints + interior.
run_cell exp-35-c3-a050 0.5
run_cell exp-35-c1-a000 0.0
run_cell exp-35-c2-a025 0.25
run_cell exp-35-c4-a075 0.75
run_cell exp-35-c5-a100 1.0

touch "$RUNROOT/done.flag"
echo "=== EXP-35 ALL CELLS COMPLETE $(date -u +%FT%TZ) ==="
