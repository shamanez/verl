#!/usr/bin/env bash
# launch_recover.sh — recover the two disk-full casualties on the now-clean disk:
#   1. base-gsm8k re-val (the first S_base ran during full-disk => garbage 0.0758)
#   2. dense-bigmath-s7 re-run (crashed on step-75 checkpoint save => lost step-75 val)
# Both write train.log (heartbeat). SAVE_FREQ disabled on the training cell (no
# checkpoint needed => no disk refill). done_recover.flag at end.
set -uo pipefail
RUN_ID=64-middle-block-freeze-grpo
RUN_DIR=/workspace/runs/$RUN_ID
VERL_DIR=/workspace/verl
BIGMATH_DIR=/root/data/bigmath
SECRETS="$HOME/.config/verl-research/secrets.env"
ACCEL_LAUNCHER=examples/grpo_trainer/vast_comm_eff_accel_base_qwen25_1p5b_grpo_gsm8k.sh
BASE_LAUNCHER=examples/grpo_trainer/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh
mkdir -p "$RUN_DIR"
echo "=== [$(date -Iseconds)] launch_recover.sh START ==="
[[ -r "$SECRETS" ]] && { source "$SECRETS"; export HF_TOKEN="${HF_TOKEN:-}" HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}" HUGGINGFACE_HUB_TOKEN="${HF_TOKEN:-}" WANDB_API_KEY="${WANDB_API_KEY:-}"; } || { echo FATAL secrets; exit 1; }
cd "$VERL_DIR" || { echo FATAL no verl; exit 1; }
echo "=== verl @ $(git rev-parse --short HEAD 2>/dev/null) ($(git rev-parse --abbrev-ref HEAD 2>/dev/null)) ==="
python3 -c "import verl,torch;print('verl OK')" || exit 1
export PROJECT_NAME="$RUN_ID"; export WANDB_RUN_GROUP="$RUN_ID"
df -h /workspace | tail -1

# 1. base-gsm8k re-val (val_only; clean disk) -> S_base gsm8k
echo "=== [$(date -Iseconds)] RECOVER base-gsm8k-v2 (val_only) ==="
env EXPERIMENT_NAME="64-base-gsm8k-v2" LOG="$RUN_DIR/train.log" \
  TOTAL_TRAINING_STEPS=1 TEST_FREQ=1 VAL_BEFORE_TRAIN=True \
  bash "$ACCEL_LAUNCHER" trainer.val_only=True actor_rollout_ref.actor.comm_eff.enabled=false || true
cp -f "$RUN_DIR/train.log" "$RUN_DIR/train_base-gsm8k-v2.log" 2>/dev/null || true
echo "$(date -Iseconds)" > "$RUN_DIR/done_base-gsm8k-v2.flag"

# 2. dense-bigmath-s7 re-run (training, no checkpoint save) -> step-75 val
echo "=== [$(date -Iseconds)] RECOVER dense-bigmath-s7 (75-step train) ==="
env EXPERIMENT_NAME="64-dense-bigmath-s7" LOG="$RUN_DIR/train.log" \
  COMM_EFF_ENABLED=false TOTAL_TRAINING_STEPS=75 TEST_FREQ=25 VAL_BEFORE_TRAIN=False \
  USE_DYNAMIC_BSZ=True ROLLOUT_TP=1 ROLLOUT_GPU_MEM_UTIL=0.55 PPO_MAX_TOKEN_LEN_PER_GPU=24576 \
  TRAIN_BATCH_SIZE=128 PPO_MINI_BATCH_SIZE=64 ROLLOUT_N=8 ACTOR_LR=1e-6 COMM_EFF_CAPTURE_ENABLED=false \
  MAX_RESPONSE_LENGTH=4096 MAX_PROMPT_LENGTH=1024 DATA_DIR="$BIGMATH_DIR" SAVE_FREQ=1000 \
  bash "$BASE_LAUNCHER" actor_rollout_ref.actor.comm_eff.enabled=false data.seed=7 trainer.save_freq=1000 || true
cp -f "$RUN_DIR/train.log" "$RUN_DIR/train_dense-bigmath-s7.log" 2>/dev/null || true
echo "$(date -Iseconds)" > "$RUN_DIR/done_rerun-s7.flag"

echo "$(date -Iseconds) recover done" > "$RUN_DIR/done_recover.flag"
echo "=== [$(date -Iseconds)] launch_recover.sh END ==="
