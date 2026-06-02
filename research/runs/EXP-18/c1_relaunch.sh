#!/usr/bin/env bash
# EXP-18 / M4 — C1 (anchorinject) RELAUNCH after the FSDP name-key fix
# (canonicalize ._fsdp_wrapped_module infix; commit e65e2c98a on
# exp/18-anchorinject-c5d5). Reuses box 39132674 / 208.64.254.75 (4xH200).
# Env VERBATIM. ema_device=cpu is the OOM fix (M_anchor ~5GB off HBM).
# MANDATORY pins: ANCHOR_DELAY_K=5, CLEAN_CADENCE=0, ANCHOR_CADENCE=5, MAX_RESPONSE 16384.
set -uo pipefail
cd /workspace/verl
rm -f /workspace/runs/EXP-18/done_curvematch_anchorinject_c5_d5.flag

PROJECT_NAME=comm_eff_curve_match_m4 EXPERIMENT_NAME=curvematch_anchorinject_c5_d5 \
COMM_EFF_ENABLED=true COMM_EFF_MASK_ENABLED=true COMM_EFF_MASK_P=0.9 COMM_EFF_MASK_RESCALE=true COMM_EFF_MASK_RECOMPUTE=true \
COMM_EFF_CLEAN_CADENCE=0 COMM_EFF_ANCHOR_ENABLED=true COMM_EFF_ANCHOR_CADENCE=5 COMM_EFF_ANCHOR_DELAY_K=5 \
COMM_EFF_SPECTRAL_ENABLED=true COMM_EFF_SPECTRAL_MAX_TARGETS=-1 COMM_EFF_SPECTRAL_SEED_ANCHOR_CACHE=false COMM_EFF_SPECTRAL_EMA_DEVICE=cpu \
PPO_MAX_TOKEN_LEN_PER_GPU=18432 LOG_PROB_MAX_TOKEN_LEN_PER_GPU=18432 REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU=18432 \
TOTAL_TRAINING_STEPS=50 VAL_BEFORE_TRAIN=False TEST_FREQ=100000 USE_DYNAMIC_BSZ=True NGPUS_PER_NODE=4 \
bash examples/grpo_trainer/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh \
  actor_rollout_ref.actor.comm_eff.spectral.correction_mode=inject actor_rollout_ref.actor.comm_eff.spectral.inject_gamma=1.0 \
  > /workspace/runs/EXP-18/train_curvematch_anchorinject_c5_d5.log 2>&1
echo "$(date -Iseconds) rc=$?" > /workspace/runs/EXP-18/done_curvematch_anchorinject_c5_d5.flag
