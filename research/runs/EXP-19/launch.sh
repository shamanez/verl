#!/usr/bin/env bash
# EXP-19 — masked p=0.9 + clean_cadence=20 on Big-Math, REWARD BUG FIXED.
#
# Re-run of EXP-18 with the corrected reward function:
#   data_source="math_bigmath" -> math_reward.compute_score
#   (extracts last \boxed{} from full solution, is_equiv normalised comparison)
# EXP-18 used data_source="math_dapo" which routes to is_correct_minerva (looks
# for "Answer:" regex, never fires on \boxed{} output -> all rollouts scored -1.0).
#
# Config identical to EXP-17/EXP-18 (mask p=0.9, rescale ON, mask_recompute ON,
# clean_cadence=20, anchor+spectral OFF, no-KL no-entropy).
# Dataset: gshasiri/Big-Math-RL-Verified-filtered, 20000 train / 500 val.
# Total: 120 steps / 1 epoch. Clean steps at 20/40/60/80/100/120.
set -euo pipefail

git config --global user.email "harness@verl-research.local"
git config --global user.name  "verl-research-harness"

cd /workspace/verl

mkdir -p /workspace/runs/EXP-19

PPO_MAX_TOKEN_LEN_PER_GPU=98304 \
LOG_PROB_MAX_TOKEN_LEN_PER_GPU=98304 \
REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU=98304 \
DATA_DIR=/root/data/bigmath \
PROJECT_NAME=verl_compression_research \
EXPERIMENT_NAME=grpo_mask_p0p9_clean20_bigmath_fixed \
COMM_EFF_ENABLED=true COMM_EFF_MASK_ENABLED=true COMM_EFF_MASK_P=0.9 \
COMM_EFF_MASK_RESCALE=true COMM_EFF_MASK_RECOMPUTE=true \
COMM_EFF_CLEAN_CADENCE=20 COMM_EFF_ANCHOR_ENABLED=false COMM_EFF_SPECTRAL_ENABLED=false \
TOTAL_EPOCHS=1 TOTAL_TRAINING_STEPS=120 TEST_FREQ=10 VAL_BEFORE_TRAIN=True USE_DYNAMIC_BSZ=True \
  bash examples/grpo_trainer/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh \
  > /workspace/runs/EXP-19/train.log 2>&1

echo "$(date -Iseconds) done" > /workspace/runs/EXP-19/done.flag
