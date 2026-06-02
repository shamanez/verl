#!/usr/bin/env bash
# EXP-18 M4 candidate C2 (convex blend) — runs inside the Vast.ai box (instance
# 39132674 / 208.64.254.75:23828, 4xH200). REUSES the box; EXP-18 ledger row
# stays RUNNING. The blend branch exp/18-anchorblend-c5d5 was applied by cloning
# /workspace/runs/EXP-18/exp-c2.bundle over /workspace/verl (editable install
# points there; blend code verified live + 40/40 spectral tests pass on box).
#
# C2 vs C1: C1 (correction_mode=inject, gamma=1) ADDED a scale-matched orthogonal
# force -> reward collapsed 0.13->0.0. C2 REPLACES via convex blend
# G_corr=(1-eta)*G_mask + eta*scale*M_anchor (eta=0.7) at a stable magnitude.
#
# Inherits ALL C1 fixes: ema_device=cpu, PPO_MAX_TOKEN_LEN_PER_GPU=18432,
# seed_anchor_cache=false, spectral.max_targets=-1, anchor c5/d5, clean=0.
# MANDATORY pins (INVALID if violated): ANCHOR_DELAY_K=5, CLEAN_CADENCE=0,
# ANCHOR_CADENCE=5, MAX_RESPONSE 16384 untouched.
set -euo pipefail

cd /workspace/verl
rm -f /workspace/runs/EXP-18/done_curvematch_anchorblend_c5_d5.flag

PROJECT_NAME=comm_eff_curve_match_m4 EXPERIMENT_NAME=curvematch_anchorblend_c5_d5 \
COMM_EFF_ENABLED=true COMM_EFF_MASK_ENABLED=true COMM_EFF_MASK_P=0.9 COMM_EFF_MASK_RESCALE=true COMM_EFF_MASK_RECOMPUTE=true \
COMM_EFF_CLEAN_CADENCE=0 COMM_EFF_ANCHOR_ENABLED=true COMM_EFF_ANCHOR_CADENCE=5 COMM_EFF_ANCHOR_DELAY_K=5 \
COMM_EFF_SPECTRAL_ENABLED=true COMM_EFF_SPECTRAL_MAX_TARGETS=-1 COMM_EFF_SPECTRAL_SEED_ANCHOR_CACHE=false COMM_EFF_SPECTRAL_EMA_DEVICE=cpu \
PPO_MAX_TOKEN_LEN_PER_GPU=18432 LOG_PROB_MAX_TOKEN_LEN_PER_GPU=18432 REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU=18432 \
TOTAL_TRAINING_STEPS=50 VAL_BEFORE_TRAIN=False TEST_FREQ=100000 USE_DYNAMIC_BSZ=True NGPUS_PER_NODE=4 \
bash examples/grpo_trainer/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh \
  actor_rollout_ref.actor.comm_eff.spectral.correction_mode=blend actor_rollout_ref.actor.comm_eff.spectral.blend_eta=0.7 \
  > /workspace/runs/EXP-18/train_curvematch_anchorblend_c5_d5.log 2>&1
echo "$(date -Iseconds) rc=$?" > /workspace/runs/EXP-18/done_curvematch_anchorblend_c5_d5.flag
