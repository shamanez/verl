#!/usr/bin/env bash
# EXP-20 — DENSE BASELINE on Big-Math (comm-eff OFF = pure verl GRPO).
#
# Dense control for EXP-19: identical dataset (Big-Math, math_bigmath reward),
# identical model (Qwen2.5-1.5B-Instruct), identical training shape (120 steps,
# batch 128, lr 1e-6, n=8 rollouts), but with COMM_EFF_ENABLED=false so no mask,
# no clean step overhead, and rollout policy == training policy (ppl_ratio ~1).
#
# Purpose: establish what reward trajectory / val accuracy the UNMASKED model
# achieves on Big-Math under the same compute budget. This is the reference that
# EXP-19 (masked) is compared against to assess whether the mask hurts, helps,
# or is neutral on this harder distribution.
set -euo pipefail

git config --global user.email "harness@verl-research.local"
git config --global user.name  "verl-research-harness"

cd /workspace/verl

mkdir -p /workspace/runs/EXP-20

# Token budget 32768 (not 98304): the ALL-DENSE path has higher peak activation
# memory than EXP-19's masked path (mask_recompute lowers peak); on 143GB H200 the
# dense update left too little for vLLM KV-cache wake_up -> CUDA OOM at step 18.
# 32768 (> 17408 = max single seq, so dynamic-bsz still valid) lowers peak. This is a
# packing/perf knob, NOT a learning hyperparam (effective batch 128/64/n8 unchanged),
# so the masked-vs-dense comparison is unaffected.
PPO_MAX_TOKEN_LEN_PER_GPU=32768 \
LOG_PROB_MAX_TOKEN_LEN_PER_GPU=32768 \
REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU=32768 \
DATA_DIR=/root/data/bigmath \
PROJECT_NAME=verl_compression_research \
EXPERIMENT_NAME=grpo_dense_bigmath_baseline \
COMM_EFF_ENABLED=false COMM_EFF_MASK_ENABLED=false \
TOTAL_EPOCHS=1 TOTAL_TRAINING_STEPS=120 TEST_FREQ=10 VAL_BEFORE_TRAIN=True USE_DYNAMIC_BSZ=True \
  bash examples/grpo_trainer/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh \
  > /workspace/runs/EXP-20/train.log 2>&1

echo "$(date -Iseconds) done" > /workspace/runs/EXP-20/done.flag
