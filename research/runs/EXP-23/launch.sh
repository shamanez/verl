#!/usr/bin/env bash
# EXP-23 ARM launcher — runs INSIDE the Vast.ai container. The ORCHESTRATOR
# invokes this (one tmux session per arm) ONLY on smoke PASS. NOT used by the
# runner's smoke dispatch (that is smoke.sh).
#
#   Usage:  bash /workspace/runs/EXP-23/launch.sh {A1|A2|A3}
#
# All three arms hold the EXP-20 PowerSGD r=77 codec block constant and vary ONLY
# the refresh axis:
#   A1 = no refresh        (clean_cadence=0, anchor OFF, spectral OFF, 36864)
#   A2 = stale inject      (anchor delay_K=5 cadence=5 + spectral inject gamma=1 cadence=5, 18432 + ema_device=cpu)
#   A3 = stale blend       (as A2 but spectral blend eta=0.5)
# 50 steps, test_freq=10 (val at 0/10/20/30/40/50). delay_K trap: pass 5 explicitly (launcher default 20).
set -uo pipefail
ARM="${1:-}"
if [[ -z "$ARM" ]]; then echo "usage: launch.sh A1|A2|A3" >&2; exit 2; fi
RUN_DIR=/workspace/runs/EXP-23

git config --global user.email "harness@verl-research.local"
git config --global user.name  "verl-research-harness"

# Apply the exp/23 bundle once (idempotent — shared with smoke.sh).
if [[ -f "$RUN_DIR/exp.bundle" && ! -f "$RUN_DIR/.bundle_applied" ]]; then
  cd /workspace
  [[ -d verl ]] && mv verl verl.upstream-vast-ai-workload
  git clone -b "exp/23-stale-reanchor" "$RUN_DIR/exp.bundle" verl
  cd /workspace/verl
  git remote set-url origin https://github.com/shamanez/verl.git || true
  uv pip install --no-deps -e . > "$RUN_DIR/pip.log" 2>&1 || pip install --no-deps -e . > "$RUN_DIR/pip.log" 2>&1
  touch "$RUN_DIR/.bundle_applied"
fi
cd /workspace/verl

# Shared EXP-20 PowerSGD r=77 codec block + fixed surface (50-step arms).
export COMM_EFF_ENABLED=true
export COMM_EFF_COMPRESSION_TYPE=powersgd
export COMM_EFF_POWERSGD_RANK=77
export COMM_EFF_POWERSGD_SYNC_BASIS=true
export COMM_EFF_POWERSGD_UPDATE_CADENCE=1
export COMM_EFF_POWERSGD_WARM_START=true
export COMM_EFF_POWERSGD_COMPRESS_RECOMPUTE=true
export COMM_EFF_POWERSGD_QR_DTYPE=fp32
export COMM_EFF_POWERSGD_SEED=0
export COMM_EFF_POWERSGD_PP_SIZE=8
export COMM_EFF_POWERSGD_REORTHO_EPS=1e-6
export COMM_EFF_CLEAN_CADENCE=0
export TOTAL_TRAINING_STEPS=50
export TEST_FREQ=10
export VAL_BEFORE_TRAIN=True

case "$ARM" in
  A1)
    export EXPERIMENT_NAME=exp-23-A1-no-refresh
    export COMM_EFF_ANCHOR_ENABLED=false
    export COMM_EFF_SPECTRAL_ENABLED=false
    export PPO_MAX_TOKEN_LEN_PER_GPU=36864 LOG_PROB_MAX_TOKEN_LEN_PER_GPU=36864 REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU=36864
    ;;
  A2)
    export EXPERIMENT_NAME=exp-23-A2-stale-inject
    export COMM_EFF_ANCHOR_ENABLED=true COMM_EFF_ANCHOR_CADENCE=5 COMM_EFF_ANCHOR_DELAY_K=5
    export COMM_EFF_SPECTRAL_ENABLED=true COMM_EFF_SPECTRAL_CORRECTION_MODE=inject COMM_EFF_SPECTRAL_INJECT_GAMMA=1.0 COMM_EFF_SPECTRAL_CADENCE=5 COMM_EFF_SPECTRAL_EMA_DEVICE=cpu
    export PPO_MAX_TOKEN_LEN_PER_GPU=18432 LOG_PROB_MAX_TOKEN_LEN_PER_GPU=18432 REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU=18432
    ;;
  A3)
    export EXPERIMENT_NAME=exp-23-A3-stale-blend
    export COMM_EFF_ANCHOR_ENABLED=true COMM_EFF_ANCHOR_CADENCE=5 COMM_EFF_ANCHOR_DELAY_K=5
    export COMM_EFF_SPECTRAL_ENABLED=true COMM_EFF_SPECTRAL_CORRECTION_MODE=blend COMM_EFF_SPECTRAL_BLEND_ETA=0.5 COMM_EFF_SPECTRAL_CADENCE=5 COMM_EFF_SPECTRAL_EMA_DEVICE=cpu
    export PPO_MAX_TOKEN_LEN_PER_GPU=18432 LOG_PROB_MAX_TOKEN_LEN_PER_GPU=18432 REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU=18432
    ;;
  *) echo "FATAL: unknown ARM '$ARM' (expect A1|A2|A3)" >&2; exit 2 ;;
esac

LOG="$RUN_DIR/${EXPERIMENT_NAME}.train.log"
echo "=== EXP-23 $ARM launching: $EXPERIMENT_NAME -> $LOG ==="
LOG="$LOG" bash examples/grpo_trainer/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh
RC=$?
echo "$(date -Iseconds) $ARM done rc=$RC" > "$RUN_DIR/${EXPERIMENT_NAME}.done.flag"
exit "$RC"
