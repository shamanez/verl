#!/usr/bin/env bash
# launch_rerun_s7.sh — re-run ONLY dense-bigmath-s7. Its step-75 val was lost when
# the original cell crashed on a disk-full checkpoint save (torch serialization
# error). Checkpoints since deleted (186G free), and SAVE_FREQ disabled here so no
# checkpoint is written (not a deliverable) => no disk risk. Same knobs as the
# matrix cell. Writes to train.log (heartbeat). done_rerun-s7.flag at end.
set -uo pipefail
RUN_ID=64-middle-block-freeze-grpo
RUN_DIR=/workspace/runs/$RUN_ID
VERL_DIR=/workspace/verl
BIGMATH_DIR=/root/data/bigmath
SECRETS="$HOME/.config/verl-research/secrets.env"
BASE_LAUNCHER=examples/grpo_trainer/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh
mkdir -p "$RUN_DIR"
echo "=== [$(date -Iseconds)] launch_rerun_s7.sh START (recover dense-bigmath-s7 step-75 val) ==="
[[ -r "$SECRETS" ]] && { source "$SECRETS"; export HF_TOKEN="${HF_TOKEN:-}" HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}" HUGGINGFACE_HUB_TOKEN="${HF_TOKEN:-}" WANDB_API_KEY="${WANDB_API_KEY:-}"; } || { echo FATAL secrets; exit 1; }
cd "$VERL_DIR" || { echo FATAL no verl; exit 1; }
echo "=== verl @ $(git rev-parse --short HEAD 2>/dev/null) ($(git rev-parse --abbrev-ref HEAD 2>/dev/null)) ==="
python3 -c "import verl,torch;print('verl OK')" || exit 1
export PROJECT_NAME="$RUN_ID"; export WANDB_RUN_GROUP="$RUN_ID"
cell=dense-bigmath-s7
env EXPERIMENT_NAME="64-$cell" LOG="$RUN_DIR/train.log" \
  COMM_EFF_ENABLED=false TOTAL_TRAINING_STEPS=75 TEST_FREQ=25 VAL_BEFORE_TRAIN=False \
  USE_DYNAMIC_BSZ=True ROLLOUT_TP=1 ROLLOUT_GPU_MEM_UTIL=0.55 PPO_MAX_TOKEN_LEN_PER_GPU=24576 \
  TRAIN_BATCH_SIZE=128 PPO_MINI_BATCH_SIZE=64 ROLLOUT_N=8 ACTOR_LR=1e-6 COMM_EFF_CAPTURE_ENABLED=false \
  MAX_RESPONSE_LENGTH=4096 MAX_PROMPT_LENGTH=1024 DATA_DIR="$BIGMATH_DIR" SAVE_FREQ=1000 \
  bash "$BASE_LAUNCHER" actor_rollout_ref.actor.comm_eff.enabled=false data.seed=7 trainer.save_freq=1000
rc=$?
cp -f "$RUN_DIR/train.log" "$RUN_DIR/train_${cell}.log" 2>/dev/null || true
echo "$(date -Iseconds) rc=$rc" > "$RUN_DIR/done_rerun-s7.flag"
echo "=== [$(date -Iseconds)] launch_rerun_s7.sh END (rc=$rc) ==="
