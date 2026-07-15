#!/usr/bin/env bash
# Pure sliding rank1_relex + Q-only warmup on the fixed Qwen2.5/MATH control.
# The 1024-token prompt plus 3072-token response is the 4096-token protocol.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export VERL_ROOT="$(cd "$HERE/../.." && pwd)"

# The proof run is complete; this launcher now defaults to the required final
# model while preserving every other fixed MATH/GRPO knob.
# Method selection is still provisional: W=4 below is the RELEX-style sliding
# adaptation being measured, not a declared champion. After the corrected
# matrix completes, the
# winner recorded in research/.claude/project.yaml will back a neutral MATH
# launcher whose bare invocation selects that method. Until then, use explicit
# comparison-arm names for scientific claims.
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
# this template replaces its legacy user suffix and pins RELEX's explicit
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
export COMM_EFF_POWERSGD_RANK="${COMM_EFF_POWERSGD_RANK:-77}"
export COMM_EFF_POWERSGD_Q_BASIS="${COMM_EFF_POWERSGD_Q_BASIS:-act}"
export COMM_EFF_POWERSGD_Q_BASIS_PASSIVE="${COMM_EFF_POWERSGD_Q_BASIS_PASSIVE:-[]}"
export COMM_EFF_POWERSGD_FAST_Q_BOOTSTRAP="${COMM_EFF_POWERSGD_FAST_Q_BOOTSTRAP:-false}"
export COMM_EFF_ANCHOR_ENABLED="${COMM_EFF_ANCHOR_ENABLED:-true}"
export COMM_EFF_ANCHOR_CADENCE="${COMM_EFF_ANCHOR_CADENCE:-20}"
export COMM_EFF_ANCHOR_DELAY_K="${COMM_EFF_ANCHOR_DELAY_K:-20}"
export COMM_EFF_ANCHOR_OWNS_Q="${COMM_EFF_ANCHOR_OWNS_Q:-true}"
export COMM_EFF_ANCHOR_REPLAY_PAIRED_BATCH="${COMM_EFF_ANCHOR_REPLAY_PAIRED_BATCH:-true}"
export COMM_EFF_ANCHOR_BATCH_SCOPE="${COMM_EFF_ANCHOR_BATCH_SCOPE:-ppo_minibatch}"
export COMM_EFF_ANCHOR_SNAPSHOT_DEVICE="${COMM_EFF_ANCHOR_SNAPSHOT_DEVICE:-cpu}"
export COMM_EFF_ANCHOR_LOOKAHEAD_ANCHOR="${COMM_EFF_ANCHOR_LOOKAHEAD_ANCHOR:-true}"
export COMM_EFF_ANCHOR_LOOKAHEAD_MODE="${COMM_EFF_ANCHOR_LOOKAHEAD_MODE:-rank1_relex}"
export COMM_EFF_ANCHOR_LOOKAHEAD_STRENGTH="${COMM_EFF_ANCHOR_LOOKAHEAD_STRENGTH:-1.0}"
export COMM_EFF_ANCHOR_LOOKAHEAD_ROLLOUT_SOURCE="${COMM_EFF_ANCHOR_LOOKAHEAD_ROLLOUT_SOURCE:-auto}"
export COMM_EFF_ANCHOR_LOOKAHEAD_WINDOW_SNAPSHOTS="${COMM_EFF_ANCHOR_LOOKAHEAD_WINDOW_SNAPSHOTS:-4}"
export COMM_EFF_ANCHOR_WARMUP_MODE="${COMM_EFF_ANCHOR_WARMUP_MODE:-q_only}"
export COMM_EFF_ANCHOR_LOOKAHEAD_MIN_SNAPSHOTS="${COMM_EFF_ANCHOR_LOOKAHEAD_MIN_SNAPSHOTS:--1}"
export COMM_EFF_CLEAN_CADENCE="${COMM_EFF_CLEAN_CADENCE:-0}"
export COMM_EFF_SPECTRAL_ENABLED="${COMM_EFF_SPECTRAL_ENABLED:-true}"
export COMM_EFF_SPECTRAL_TARGET_SCOPE="${COMM_EFF_SPECTRAL_TARGET_SCOPE:-decoder_matrices}"
export COMM_EFF_SPECTRAL_DIAGNOSTICS="${COMM_EFF_SPECTRAL_DIAGNOSTICS:-true}"
export COMM_EFF_SPECTRAL_CADENCE="${COMM_EFF_SPECTRAL_CADENCE:-1}"
export COMM_EFF_SPECTRAL_CORRECTION_MODE="${COMM_EFF_SPECTRAL_CORRECTION_MODE:-signed_ema}"
export COMM_EFF_SPECTRAL_BETA_ANC="${COMM_EFF_SPECTRAL_BETA_ANC:-0.50}"
export COMM_EFF_SPECTRAL_SIGNED_EMA_ALPHA="${COMM_EFF_SPECTRAL_SIGNED_EMA_ALPHA:-0.25}"
export COMM_EFF_SPECTRAL_EMA_DEVICE="${COMM_EFF_SPECTRAL_EMA_DEVICE:-cpu}"
export COMM_EFF_SPECTRAL_MAX_TARGETS="${COMM_EFF_SPECTRAL_MAX_TARGETS:--1}"
export COMM_EFF_CAPTURE_ENABLED="${COMM_EFF_CAPTURE_ENABLED:-false}"
export COMM_EFF_CAPTURE_G_DENSE="${COMM_EFF_CAPTURE_G_DENSE:-false}"
export COMM_EFF_CAPTURE_FRESH_ANCHOR="${COMM_EFF_CAPTURE_FRESH_ANCHOR:-false}"

# Run controls may change duration/logging, but not the fixed scientific surface.
# MATH has 14 full 512-prompt batches per epoch after prompt filtering. Eight
# epochs are therefore required for trainer.total_training_steps=100 to be the
# actual stopping condition (the inherited two-epoch default stops at step 28).
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-8}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-100}"
export TEST_FREQ="${TEST_FREQ:-25}"
export SAVE_FREQ="${SAVE_FREQ:--1}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-True}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-rank1_relex_qonly_math_qwen25_1p5b}"
export LOG="${LOG:-$VERL_ROOT/runs/$EXPERIMENT_NAME/train.log}"

exec bash "$HERE/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh" \
  actor_rollout_ref.actor.comm_eff.probe.geometry_enabled=false \
  'actor_rollout_ref.model.custom_chat_template=${oc.env:RELEX_QWEN_CHAT_TEMPLATE}' \
  "$@"
