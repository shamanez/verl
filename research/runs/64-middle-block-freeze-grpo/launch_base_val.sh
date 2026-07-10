#!/usr/bin/env bash
# launch_base_val.sh — measure S_base (base Qwen2.5-1.5B-Instruct val, NO training)
# on GSM8K + Big-Math, so /analyze can compute C(block)=(S_frozen-S_base)/(S_dense-S_base)
# without waiting on the operator. val_only=True => initial validation then exit.
# Same val config as the matrix cells (same launcher, same eval keys) => comparable.
# Runs AFTER the matrix, on the same box, before teardown. Writes to train.log (heartbeat).
set -uo pipefail
RUN_ID=64-middle-block-freeze-grpo
RUN_DIR=/workspace/runs/$RUN_ID
VERL_DIR=/workspace/verl
BIGMATH_DIR=/root/data/bigmath
SECRETS="$HOME/.config/verl-research/secrets.env"
EXP_BRANCH=exp/64-dense-wandbfix
ACCEL_LAUNCHER=examples/grpo_trainer/vast_comm_eff_accel_base_qwen25_1p5b_grpo_gsm8k.sh
BASE_LAUNCHER=examples/grpo_trainer/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh
mkdir -p "$RUN_DIR"
echo "=== [$(date -Iseconds)] launch_base_val.sh START (measure S_base) ==="
[[ -r "$SECRETS" ]] && { source "$SECRETS"; export HF_TOKEN="${HF_TOKEN:-}" HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}" HUGGINGFACE_HUB_TOKEN="${HF_TOKEN:-}" WANDB_API_KEY="${WANDB_API_KEY:-}"; } || { echo "FATAL: secrets missing"; exit 1; }
cd "$VERL_DIR" 2>/dev/null || { echo "FATAL: no verl"; exit 1; }
echo "=== verl @ $(git rev-parse --short HEAD 2>/dev/null) ($(git rev-parse --abbrev-ref HEAD 2>/dev/null)) ==="
python3 -c "import verl,torch;print('verl OK',torch.__version__)" || { echo FATAL; exit 1; }
export PROJECT_NAME="$RUN_ID"; export WANDB_RUN_GROUP="$RUN_ID"

run_val() {
  local cell="$1" launcher="$2"; shift 2
  echo "=== [$(date -Iseconds)] BASEVAL $cell START (wandb 64-$cell) ==="
  env EXPERIMENT_NAME="64-$cell" LOG="$RUN_DIR/train.log" "$@" \
    bash "$launcher" trainer.val_only=True actor_rollout_ref.actor.comm_eff.enabled=false
  local rc=$?
  cp -f "$RUN_DIR/train.log" "$RUN_DIR/train_${cell}.log" 2>/dev/null || true
  echo "$(date -Iseconds) rc=$rc" > "$RUN_DIR/done_${cell}.flag"
  echo "=== [$(date -Iseconds)] BASEVAL $cell END (rc=$rc) ==="
}

# base GSM8K val (accel surface, resp=1024 — matches the gsm8k matrix cells)
run_val base-gsm8k "$ACCEL_LAUNCHER" TOTAL_TRAINING_STEPS=1 TEST_FREQ=1 VAL_BEFORE_TRAIN=True

# base Big-Math val (resp=4096 dyn-bsz — matches the bigmath matrix cells)
if [[ -f "$BIGMATH_DIR/train.parquet" ]]; then
  run_val base-bigmath "$BASE_LAUNCHER" COMM_EFF_ENABLED=false TOTAL_TRAINING_STEPS=1 TEST_FREQ=1 VAL_BEFORE_TRAIN=True \
    USE_DYNAMIC_BSZ=True ROLLOUT_TP=1 ROLLOUT_GPU_MEM_UTIL=0.55 PPO_MAX_TOKEN_LEN_PER_GPU=24576 \
    TRAIN_BATCH_SIZE=128 PPO_MINI_BATCH_SIZE=64 ROLLOUT_N=8 COMM_EFF_CAPTURE_ENABLED=false \
    MAX_RESPONSE_LENGTH=4096 MAX_PROMPT_LENGTH=1024 DATA_DIR="$BIGMATH_DIR"
else
  echo "WARN: bigmath data absent — skipping base-bigmath val" >&2
fi
echo "$(date -Iseconds) base-val done" > "$RUN_DIR/done_baseval.flag"
echo "=== [$(date -Iseconds)] launch_base_val.sh END ==="
