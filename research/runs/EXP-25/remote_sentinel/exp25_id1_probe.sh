#!/usr/bin/env bash
# EXP-25 id-1 probe (R2 anchor-owns-Q + R3 signed_ema merger). Runs inside the Vast box.
# Anchor ON, R2(owns_q) OFF, spectral ON (REQUIRED for M to build + ||dM||>0),
# signed_ema merger present (mode=signed_ema, α=0), cadence=1 delay_K=1 so the
# anchor FIRES every step in a 3-step probe. TEST_FREQ=0. PowerSGD r=77, mask off.
set -uo pipefail
set -a; source ~/.config/verl-research/secrets.env; set +a

cd /workspace/verl
git config --global user.email "harness@verl-research.local" 2>/dev/null || true
git config --global user.name  "verl-research-harness"        2>/dev/null || true

EXPERIMENT_NAME="exp25_id1_R2R3"
PERSIST_LOG="/workspace/runs/EXP-25/logs/${EXPERIMENT_NAME}.log"
mkdir -p "$(dirname "$PERSIST_LOG")"

echo "=== EXP-25 id-0 probe START $(date -u +%FT%TZ) commit=$(git rev-parse --short HEAD) ===" | tee "$PERSIST_LOG"

COMM_EFF_ENABLED=true \
COMM_EFF_COMPRESSION_TYPE=powersgd \
COMM_EFF_MASK_ENABLED=false \
COMM_EFF_POWERSGD_RANK=77 \
COMM_EFF_POWERSGD_SYNC_BASIS=true \
COMM_EFF_ANCHOR_ENABLED=true \
COMM_EFF_ANCHOR_OWNS_Q=true \
COMM_EFF_ANCHOR_CADENCE=1 \
COMM_EFF_ANCHOR_DELAY_K=1 \
COMM_EFF_SPECTRAL_ENABLED=true \
COMM_EFF_SPECTRAL_CORRECTION_MODE=signed_ema \
COMM_EFF_SPECTRAL_SIGNED_EMA_ALPHA=0.0 \
COMM_EFF_SPECTRAL_BETA_ANC=0.95 \
COMM_EFF_SPECTRAL_EMA_DEVICE=cpu \
COMM_EFF_SPECTRAL_MAX_TARGETS=-1 \
COMM_EFF_SPECTRAL_CADENCE=1 \
COMM_EFF_CLEAN_CADENCE=0 \
PPO_MAX_TOKEN_LEN_PER_GPU=18432 \
TOTAL_TRAINING_STEPS=3 \
TEST_FREQ=0 \
VAL_BEFORE_TRAIN=False \
EXPERIMENT_NAME="$EXPERIMENT_NAME" \
  bash examples/grpo_trainer/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh 2>&1 | tee -a "$PERSIST_LOG"

RC=${PIPESTATUS[0]}
echo "=== EXP-25 id-0 probe END $(date -u +%FT%TZ) rc=$RC ===" | tee -a "$PERSIST_LOG"
# Mirror the launcher's own train.log into the persisted dir too (belt-and-braces).
cp -f "/workspace/verl/runs/${EXPERIMENT_NAME}/train.log" "/workspace/runs/EXP-25/logs/${EXPERIMENT_NAME}.trainlog" 2>/dev/null || true
exit $RC
