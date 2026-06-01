#!/usr/bin/env bash
# EXP-18 — masked p=0.9 + clean_cadence=20 on the HARDER Big-Math dataset.
# Reuses instance 38877541 (4x H200) right after EXP-17 finished (GPU-idle min).
#
# Purpose: stress-test the EXP-17 comm-eff config (per-(token,channel) mask p=0.9,
# rescale ON, clean_cadence=20, anchor+spectral OFF) on gshasiri/Big-Math-RL-Verified-filtered
# (verl parquet at /root/data/bigmath, data_source=DigitalLearningGmbH/MATH-lighteval =>
# math_reward \boxed{} verifier with is_equiv normalization) to see whether COLLAPSE/divergence
# appears on a much harder math distribution than GSM8K.
# NOTE: v2 — the first attempt used data_source=math_dapo whose default verifier scrapes for
# an "Answer:" token (not \boxed{}), producing a confounded/biased reward; fixed to math_reward.
#
# Pure-config + new dataset; NO method patch (same launcher knobs as EXP-17).
set -euo pipefail

git config --global user.email "harness@verl-research.local"
git config --global user.name  "verl-research-harness"

cd /workspace/verl

# Same EXP-16/EXP-17-proven perf token budget (max_token_len=98304; anchor+spectral OFF).
# DATA_DIR points the launcher at the pre-built Big-Math parquets (it skips gsm8k prep when
# train.parquet+test.parquet already exist there). Horizon: 1 epoch cap at 120 steps =>
# 6 clean cycles at 20/40/60/80/100/120 (comparable to EXP-17's 5 cycles / 116 steps).
PPO_MAX_TOKEN_LEN_PER_GPU=98304 \
LOG_PROB_MAX_TOKEN_LEN_PER_GPU=98304 \
REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU=98304 \
DATA_DIR=/root/data/bigmath \
PROJECT_NAME=verl_compression_research \
EXPERIMENT_NAME=grpo_mask_p0p9_clean20_bigmath_collapse_v2 \
COMM_EFF_ENABLED=true COMM_EFF_MASK_ENABLED=true COMM_EFF_MASK_P=0.9 \
COMM_EFF_MASK_RESCALE=true COMM_EFF_MASK_RECOMPUTE=true \
COMM_EFF_CLEAN_CADENCE=20 COMM_EFF_ANCHOR_ENABLED=false COMM_EFF_SPECTRAL_ENABLED=false \
TOTAL_EPOCHS=1 TOTAL_TRAINING_STEPS=120 TEST_FREQ=10 VAL_BEFORE_TRAIN=True USE_DYNAMIC_BSZ=True \
  bash examples/grpo_trainer/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh \
  > /workspace/runs/EXP-18/train.log 2>&1

echo "$(date -Iseconds) done" > /workspace/runs/EXP-18/done.flag
