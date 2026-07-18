#!/usr/bin/env bash
# Current communication-efficient GRPO run on Qwen2.5-Math-1.5B and MATH.
# The 1024-token prompt plus 3072-token response is the 4096-token protocol.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export VERL_ROOT="$(cd "$HERE/../.." && pwd)"

# Bare invocation uses the current project default.
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-Math-1.5B}"
export DATA_DIR="${DATA_DIR:-$HOME/data/math}"
if [[ ! -f "$DATA_DIR/train.parquet" || ! -f "$DATA_DIR/test.parquet" ]]; then
  echo "FATAL: prepared MATH train/test parquet files are required in $DATA_DIR" >&2
  exit 1
fi
export MAX_PROMPT_LENGTH=1024
export MAX_RESPONSE_LENGTH=3072
export ROLLOUT_N=8
export ROLLOUT_TP=1
export TRAIN_BATCH_SIZE=512
export PPO_MINI_BATCH_SIZE=256
export ACTOR_LR=1e-6
export USE_KL_LOSS=True
export USE_KL_IN_REWARD=False
export KL_LOSS_COEF=0.001

# Match RELEX/scripts/eval.py's Qwen prompt byte-for-byte for BOTH rollout
# training and validation. The prepared parquet intentionally stays unchanged:
# this template replaces the prepared parquet's user suffix and pins RELEX's explicit
# system message, avoiding the Math base tokenizer's duplicate boxed prompt.
RELEX_QWEN_CHAT_TEMPLATE_FILE="$HERE/relex_qwen_chat_template.jinja"
if [[ ! -f "$RELEX_QWEN_CHAT_TEMPLATE_FILE" ]]; then
  echo "FATAL: missing RELEX Qwen chat template: $RELEX_QWEN_CHAT_TEMPLATE_FILE" >&2
  exit 1
fi
export RELEX_QWEN_CHAT_TEMPLATE
RELEX_QWEN_CHAT_TEMPLATE="$(<"$RELEX_QWEN_CHAT_TEMPLATE_FILE")"

# 512 prompts / 256-prompt mini-batch = exactly two optimizer ticks per
# GRPO step, preserving the documented generator ticks 1,19,39,59 -> 79.
export USE_DYNAMIC_BSZ=True
export ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.55}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Canonical pure rank-1 arm. Every default remains pinned, while explicit env
# overrides let the comparison launcher print the same values Hydra receives.
export COMM_EFF_ENABLED="${COMM_EFF_ENABLED:-true}"
export COMM_EFF_COMPRESSION_TYPE="${COMM_EFF_COMPRESSION_TYPE:-powersgd}"
# prf_mask codec (active iff COMM_EFF_COMPRESSION_TYPE=prf_mask). Anchor-independent
# PRF activation mask; mutually exclusive with PowerSGD and cannot anchor-own-Q, so a
# prf_mask arm also sets COMM_EFF_ANCHOR_OWNS_Q=false COMM_EFF_POWERSGD_FAST_Q_BOOTSTRAP=false.
# Defaults keep the powersgd path byte-for-byte unchanged (mask disabled).
export COMM_EFF_MASK_ENABLED="${COMM_EFF_MASK_ENABLED:-false}"
export COMM_EFF_MASK_P="${COMM_EFF_MASK_P:-0.95}"
export COMM_EFF_MASK_RESCALE="${COMM_EFF_MASK_RESCALE:-false}"
export COMM_EFF_MASK_RECOMPUTE="${COMM_EFF_MASK_RECOMPUTE:-false}"
export COMM_EFF_MASK_SEED="${COMM_EFF_MASK_SEED:-0}"
export COMM_EFF_MASK_PP_SIZE="${COMM_EFF_MASK_PP_SIZE:-8}"
export COMM_EFF_POWERSGD_RANK="${COMM_EFF_POWERSGD_RANK:-77}"
export COMM_EFF_POWERSGD_FAST_Q_BOOTSTRAP="${COMM_EFF_POWERSGD_FAST_Q_BOOTSTRAP:-true}"
export COMM_EFF_ANCHOR_ENABLED="${COMM_EFF_ANCHOR_ENABLED:-true}"
export COMM_EFF_ANCHOR_CADENCE="${COMM_EFF_ANCHOR_CADENCE:-20}"
export COMM_EFF_ANCHOR_DELAY_K="${COMM_EFF_ANCHOR_DELAY_K:-20}"
export COMM_EFF_ANCHOR_OWNS_Q="${COMM_EFF_ANCHOR_OWNS_Q:-true}"
export COMM_EFF_ANCHOR_REPLAY_PAIRED_BATCH="${COMM_EFF_ANCHOR_REPLAY_PAIRED_BATCH:-true}"
export COMM_EFF_ANCHOR_BATCH_SCOPE="${COMM_EFF_ANCHOR_BATCH_SCOPE:-rollout_batch}"
export COMM_EFF_ANCHOR_SNAPSHOT_DEVICE="${COMM_EFF_ANCHOR_SNAPSHOT_DEVICE:-cpu}"
export COMM_EFF_ANCHOR_LOOKAHEAD_ANCHOR="${COMM_EFF_ANCHOR_LOOKAHEAD_ANCHOR:-true}"
export COMM_EFF_ANCHOR_LOOKAHEAD_MODE="${COMM_EFF_ANCHOR_LOOKAHEAD_MODE:-rank1_relex}"
export COMM_EFF_ANCHOR_LOOKAHEAD_STRENGTH="${COMM_EFF_ANCHOR_LOOKAHEAD_STRENGTH:-1.0}"
export COMM_EFF_ANCHOR_LOOKAHEAD_ROLLOUT_SOURCE="${COMM_EFF_ANCHOR_LOOKAHEAD_ROLLOUT_SOURCE:-auto}"
export COMM_EFF_ANCHOR_LOOKAHEAD_WINDOW_SNAPSHOTS="${COMM_EFF_ANCHOR_LOOKAHEAD_WINDOW_SNAPSHOTS:-2}"
export COMM_EFF_ANCHOR_WARMUP_MODE="${COMM_EFF_ANCHOR_WARMUP_MODE:-stale_correct}"
export COMM_EFF_ANCHOR_LOOKAHEAD_MIN_SNAPSHOTS="${COMM_EFF_ANCHOR_LOOKAHEAD_MIN_SNAPSHOTS:-2}"
# rank1_relex delta-base history mode. sliding_window (default) keeps the last
# `window` checkpoints (base advances); growing_fixed_base pins the seeded base
# and grows the base-relative delta history. max_snapshots caps growing_fixed_base
# retention (-1 unbounded; must stay -1 with sliding_window). The sliding_window /
# -1 defaults reproduce prior behavior byte-for-byte.
export COMM_EFF_ANCHOR_LOOKAHEAD_HISTORY_MODE="${COMM_EFF_ANCHOR_LOOKAHEAD_HISTORY_MODE:-sliding_window}"
export COMM_EFF_ANCHOR_LOOKAHEAD_MAX_SNAPSHOTS="${COMM_EFF_ANCHOR_LOOKAHEAD_MAX_SNAPSHOTS:--1}"
export COMM_EFF_SPECTRAL_ENABLED="${COMM_EFF_SPECTRAL_ENABLED:-true}"
export COMM_EFF_SPECTRAL_TARGET_SCOPE="${COMM_EFF_SPECTRAL_TARGET_SCOPE:-all_floating}"
export COMM_EFF_SPECTRAL_DIAGNOSTICS="${COMM_EFF_SPECTRAL_DIAGNOSTICS:-false}"
export COMM_EFF_SPECTRAL_CADENCE="${COMM_EFF_SPECTRAL_CADENCE:-1}"
export COMM_EFF_SPECTRAL_BETA_ANC="${COMM_EFF_SPECTRAL_BETA_ANC:-0.25}"
export COMM_EFF_SPECTRAL_SIGNED_EMA_ALPHA="${COMM_EFF_SPECTRAL_SIGNED_EMA_ALPHA:-0.25}"
export COMM_EFF_SPECTRAL_EMA_DEVICE="${COMM_EFF_SPECTRAL_EMA_DEVICE:-cpu}"
export COMM_EFF_SPECTRAL_MAX_TARGETS="${COMM_EFF_SPECTRAL_MAX_TARGETS:--1}"

# Run controls may change duration/logging, but not the fixed scientific surface.
# MATH has 14 full 512-prompt batches per epoch after prompt filtering. Eight
# epochs ensure trainer.total_training_steps=100 is the stopping condition.
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-8}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-100}"
export TEST_FREQ="${TEST_FREQ:-25}"
export SAVE_FREQ="${SAVE_FREQ:--1}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-True}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-comm_eff_grpo_math_qwen25_math_1p5b}"
export LOG="${LOG:-$VERL_ROOT/runs/$EXPERIMENT_NAME/train.log}"

exec bash "$HERE/vast_comm_eff_engine_grpo.sh" \
  'actor_rollout_ref.model.custom_chat_template=${oc.env:RELEX_QWEN_CHAT_TEMPLATE}' \
  "$@"
