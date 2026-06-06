#!/usr/bin/env bash
# EXP-25 KL DIAGNOSTIC — α=0 signed_ema (the CATASTROPHIC-collapse arm) but with the KL
# divergence loss ENABLED. Question: does a KL brake to the base policy prevent the
# sign-reversal length-explosion collapse (val@50 α=0 was 0.354)?
#
# IDENTICAL to exp25_alpha_0p0 in every comm-eff knob (powersgd r77, anchor cadence=5/
# delay_K=5/owns_q, signed_ema α=0, max_targets=-1, ema cpu, clean_cadence=0) EXCEPT:
#   USE_KL_LOSS=True  -> actor.use_kl_loss=true; verl auto-spins the frozen ref policy
#   (ray_trainer need_reference_policy). kl_loss_coef defaults to 0.001 + kl_loss_type
#   low_var_kl (actor/actor.yaml:109,112) — exactly the requested 0.001 gentle brake.
#   (KL_LOSS_COEF env is set for the banner; the effective value is the 0.001 config default.)
set -uo pipefail
set -a; source ~/.config/verl-research/secrets.env; set +a

cd /workspace/verl
git config --global user.email "harness@verl-research.local" 2>/dev/null || true
git config --global user.name  "verl-research-harness"        2>/dev/null || true

EXPERIMENT_NAME="exp25_a0_kl001"
PERSIST_LOG="/workspace/runs/EXP-25/logs/${EXPERIMENT_NAME}.log"
mkdir -p "$(dirname "$PERSIST_LOG")"
echo "=== EXP-25 KL-diagnostic α=0 + KL(0.001) (${EXPERIMENT_NAME}) START $(date -u +%FT%TZ) commit=$(git rev-parse --short HEAD) ===" | tee "$PERSIST_LOG"

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
COMM_EFF_SPECTRAL_SIGNED_EMA_ALPHA=0.0 \
COMM_EFF_SPECTRAL_BETA_ANC=0.95 \
COMM_EFF_SPECTRAL_EMA_DEVICE=cpu \
COMM_EFF_SPECTRAL_MAX_TARGETS=-1 \
COMM_EFF_SPECTRAL_CADENCE=1 \
COMM_EFF_CLEAN_CADENCE=0 \
USE_KL_LOSS=True \
USE_KL_IN_REWARD=False \
KL_LOSS_COEF=0.001 \
PPO_MAX_TOKEN_LEN_PER_GPU=18432 \
TOTAL_TRAINING_STEPS=50 \
TEST_FREQ=25 \
VAL_BEFORE_TRAIN=False \
EXPERIMENT_NAME="$EXPERIMENT_NAME" \
  bash examples/grpo_trainer/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh 2>&1 | tee -a "$PERSIST_LOG"

RC=${PIPESTATUS[0]}
echo "=== EXP-25 KL-diagnostic END $(date -u +%FT%TZ) rc=${RC} ===" | tee -a "$PERSIST_LOG"
cp -f "/workspace/verl/runs/${EXPERIMENT_NAME}/train.log" "/workspace/runs/EXP-25/logs/${EXPERIMENT_NAME}.trainlog" 2>/dev/null || true
echo "$(date -Iseconds) rc=${RC}" > "/workspace/runs/EXP-25/${EXPERIMENT_NAME}.kltest-done"
