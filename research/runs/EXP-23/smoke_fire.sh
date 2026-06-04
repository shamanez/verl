#!/usr/bin/env bash
# EXP-23 PROBE_FIRE — the anchor-FIRES proof. Runs INSIDE the container.
# Identical to PROBE_ON EXCEPT anchor.cadence=1 + spectral.cadence=1 so the
# anchor + spectral correction FIRE on step 1 within the <=2-step budget
# (PROBE_ON used cadence=5, which never hits step%5==0 in 2 steps — so it could
# not exercise invariants #3/#4). The A2/A3 50-step arms keep cadence=5; the
# smoke uses cadence=1 purely to exercise the circuit cheaply (the documented
# smoke pattern — anchor.py: "Smoke uses 1 (fire every step)"). This is ALSO the
# real test of the anchor's SECOND full forward/backward memory at 18432 + ema_device=cpu.
set -uo pipefail
RUN_DIR=/workspace/runs/EXP-23
LOG_FIRE="$RUN_DIR/smoke_fire.log"
cd /workspace/verl
echo "=== verl HEAD: $(git rev-parse --short HEAD) on $(git rev-parse --abbrev-ref HEAD) ==="

# Shared EXP-20 PowerSGD r=77 codec block.
export COMM_EFF_ENABLED=true COMM_EFF_COMPRESSION_TYPE=powersgd
export COMM_EFF_POWERSGD_RANK=77 COMM_EFF_POWERSGD_SYNC_BASIS=true COMM_EFF_POWERSGD_UPDATE_CADENCE=1
export COMM_EFF_POWERSGD_WARM_START=true COMM_EFF_POWERSGD_COMPRESS_RECOMPUTE=true
export COMM_EFF_POWERSGD_QR_DTYPE=fp32 COMM_EFF_POWERSGD_SEED=0 COMM_EFF_POWERSGD_PP_SIZE=8 COMM_EFF_POWERSGD_REORTHO_EPS=1e-6
export COMM_EFF_CLEAN_CADENCE=0
export TOTAL_TRAINING_STEPS=2 TEST_FREQ=0 VAL_BEFORE_TRAIN=False
# Anchor + spectral inject, CADENCE=1 so they fire on step 1. delay_K=5 (serves oldest-available during warmup).
export EXPERIMENT_NAME=exp-23-smoke-fire
export COMM_EFF_ANCHOR_ENABLED=true COMM_EFF_ANCHOR_CADENCE=1 COMM_EFF_ANCHOR_DELAY_K=5
export COMM_EFF_SPECTRAL_ENABLED=true COMM_EFF_SPECTRAL_CORRECTION_MODE=inject COMM_EFF_SPECTRAL_INJECT_GAMMA=1.0 COMM_EFF_SPECTRAL_CADENCE=1 COMM_EFF_SPECTRAL_EMA_DEVICE=cpu
export PPO_MAX_TOKEN_LEN_PER_GPU=18432 LOG_PROB_MAX_TOKEN_LEN_PER_GPU=18432 REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU=18432

echo "### EXP-23 PROBE_FIRE (anchor+spectral cadence=1) $(date -u +%FT%TZ)"
LOG="$LOG_FIRE" bash examples/grpo_trainer/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh
RC=$?
echo "=== PROBE_FIRE exit rc=$RC $(date -u +%FT%TZ) ==="
echo "{\"probe_fire_rc\": $RC, \"ts\": \"$(date -u +%FT%TZ)\"}" > "$RUN_DIR/smoke_fire.done.flag"
