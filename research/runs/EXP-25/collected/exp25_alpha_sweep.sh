#!/usr/bin/env bash
# EXP-25 id-2 alpha sweep — 3 arms BACK-TO-BACK on the same warm box.
# Vary ONLY signed_ema_alpha {0.0, 0.3, 0.5}. Production config:
#   anchor owns_q=true, signed_ema merger, cadence=5 delay_K=5 (PINNED — launcher
#   defaults delay_K=20), PowerSGD r=77, mask off, ema cpu, max_targets=-1,
#   beta_anc=0.95, clean_cadence=0, 50 steps, val@25.
# COLD-M NOTE: at cadence=5 the anchor first fires at step 5, so M is COLD for
# steps 1-4 -> the alpha=0.0 arm is where the COLD-M fallback is genuinely tested
# (steps 1-4 must show merger_coldM_fallbacks=196 + grads=G_noisy, NOT zeroed).
set -uo pipefail
set -a; source ~/.config/verl-research/secrets.env; set +a

cd /workspace/verl
git config --global user.email "harness@verl-research.local" 2>/dev/null || true
git config --global user.name  "verl-research-harness"        2>/dev/null || true

for A in 0p0 0p3 0p5; do
  case "$A" in
    0p0) AV=0.0 ;;
    0p3) AV=0.3 ;;
    0p5) AV=0.5 ;;
  esac
  EXPERIMENT_NAME="exp25_alpha_${A}"
  PERSIST_LOG="/workspace/runs/EXP-25/logs/${EXPERIMENT_NAME}.log"
  mkdir -p "$(dirname "$PERSIST_LOG")"

  echo "=== EXP-25 alpha=${AV} (${EXPERIMENT_NAME}) START $(date -u +%FT%TZ) commit=$(git rev-parse --short HEAD) ===" | tee "$PERSIST_LOG"

  COMM_EFF_ENABLED=true \
  COMM_EFF_COMPRESSION_TYPE=powersgd \
  COMM_EFF_MASK_ENABLED=false \
  COMM_EFF_POWERSGD_RANK=77 \
  COMM_EFF_POWERSGD_SYNC_BASIS=true \
  COMM_EFF_ANCHOR_ENABLED=true \
  COMM_EFF_ANCHOR_OWNS_Q=true \
  COMM_EFF_ANCHOR_CADENCE=5 \
  COMM_EFF_ANCHOR_DELAY_K=5 \
  COMM_EFF_SPECTRAL_ENABLED=true \
  COMM_EFF_SPECTRAL_CORRECTION_MODE=signed_ema \
  COMM_EFF_SPECTRAL_SIGNED_EMA_ALPHA="$AV" \
  COMM_EFF_SPECTRAL_BETA_ANC=0.95 \
  COMM_EFF_SPECTRAL_EMA_DEVICE=cpu \
  COMM_EFF_SPECTRAL_MAX_TARGETS=-1 \
  COMM_EFF_SPECTRAL_CADENCE=1 \
  COMM_EFF_CLEAN_CADENCE=0 \
  PPO_MAX_TOKEN_LEN_PER_GPU=18432 \
  TOTAL_TRAINING_STEPS=50 \
  TEST_FREQ=25 \
  VAL_BEFORE_TRAIN=False \
  EXPERIMENT_NAME="$EXPERIMENT_NAME" \
    bash examples/grpo_trainer/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh 2>&1 | tee -a "$PERSIST_LOG"

  RC=${PIPESTATUS[0]}
  echo "=== EXP-25 alpha=${AV} END $(date -u +%FT%TZ) rc=${RC} ===" | tee -a "$PERSIST_LOG"
  cp -f "/workspace/verl/runs/${EXPERIMENT_NAME}/train.log" "/workspace/runs/EXP-25/logs/${EXPERIMENT_NAME}.trainlog" 2>/dev/null || true
  # Per-arm sentinel for the orchestrator (launcher also writes runs/<EXP>/done.flag).
  echo "$(date -Iseconds) rc=${RC}" > "/workspace/runs/EXP-25/${EXPERIMENT_NAME}.arm-done"
done
echo "$(date -Iseconds) all 3 arms done" > "/workspace/runs/EXP-25/alpha_sweep_ALL.done"
