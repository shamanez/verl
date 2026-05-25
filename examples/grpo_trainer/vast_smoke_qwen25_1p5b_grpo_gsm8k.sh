#!/usr/bin/env bash
# vast_smoke_qwen25_1p5b_grpo_gsm8k.sh
#
# Single-GPU verl GRPO smoke for vast.ai targeting a 1.5B model. Goal: reach
# `global_step: 1` with Qwen2.5-1.5B-Instruct on one ≥40 GB GPU (A100-80GB,
# H100, L40S, etc.). This is the acceptance test for the vast-provision /
# vast-teardown skills — if a single optimizer step lands with non-NaN loss
# in WandB, the provisioned box is "good enough for 1.5B GRPO."
#
# Iteration loop is identical to the 0.5B sibling: edit locally on
# vast-ai-workload, push, git pull on the box, re-run. See
# examples/grpo_trainer/VAST_README.md.
set -euo pipefail

SECRETS_FILE="${SECRETS_FILE:-$HOME/.config/verl-research/secrets.env}"
if [[ ! -r "$SECRETS_FILE" ]]; then
  cat >&2 <<EOF
FATAL: $SECRETS_FILE not found on this box.

  From the laptop, push a stripped copy:
    grep -E '^(HF_TOKEN|WANDB_API_KEY)=' ~/.config/verl-research/secrets.env > /tmp/secrets-box.env
    chmod 600 /tmp/secrets-box.env
    scp -P <port> /tmp/secrets-box.env root@<host>:~/.config/verl-research/secrets.env
    shred -u /tmp/secrets-box.env
EOF
  exit 1
fi
# shellcheck disable=SC1090
source "$SECRETS_FILE"
: "${HF_TOKEN:?HF_TOKEN missing from $SECRETS_FILE}"
: "${WANDB_API_KEY:?WANDB_API_KEY missing from $SECRETS_FILE}"
if [[ -n "${VAST_API_KEY:-}" ]]; then
  echo "FATAL: VAST_API_KEY leaked into the instance. Re-strip the file on the laptop." >&2
  exit 1
fi

# Vast.ai containers default to `ulimit -n 1024`, which is too low for verl's
# Ray + vLLM + FSDP spawn pattern: ZMQ allocates FDs per worker socket, hits
# the cap mid-init, and pthread_create then returns EAGAIN ("Resource
# temporarily unavailable") — the failure surface is the EngineCore actor
# dying before the first rollout. Bump FDs and (defensively) user processes
# to the cgroup ceiling before any python3 invocation.
ulimit -n 65535 || echo "WARN: could not bump open-files ulimit" >&2
ulimit -u 65535 2>/dev/null || true

export HF_TOKEN HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-$HF_TOKEN}" \
       HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-$HF_TOKEN}" WANDB_API_KEY

cd "${VERL_ROOT:-/workspace/verl}"

# Qwen2.5-1.5B-Instruct — Apache-2.0, no HF gating, ~3 GB bf16.
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-1.5B-Instruct}"

DATA_DIR="${DATA_DIR:-$HOME/data/gsm8k}"
if [[ ! -f "$DATA_DIR/train.parquet" ]]; then
  echo "=== preprocess GSM8K -> $DATA_DIR ==="
  python3 examples/data_preprocess/gsm8k.py --local_save_dir "$DATA_DIR"
fi
export TRAIN_FILE="$DATA_DIR/train.parquet"
export TEST_FILE="$DATA_DIR/test.parquet"
ls -la "$DATA_DIR"

# Single GPU, 1 step, no checkpoint, vLLM eager.
# Batch sizes are bumped vs the 0.5B sibling because 1.5B with grad-ckpt on
# a single ≥40 GB GPU still has plenty of room — but kept small enough that
# a 24 GB fallback also stands a chance.
export NGPUS_PER_NODE=1
export ROLLOUT_TP=1
export ROLLOUT_N=2
export ROLLOUT_GPU_MEM_UTIL=0.5
export TRAIN_BATCH_SIZE=8
export PPO_MINI_BATCH_SIZE=4
export PPO_MICRO_BATCH_SIZE_PER_GPU=1
export LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=1
export MAX_PROMPT_LENGTH=512
export MAX_RESPONSE_LENGTH=512
export PROJECT_NAME="${PROJECT_NAME:-verl_compression_research}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen25_1p5b_grpo_gsm8k_vast_smoke}"

LOG="${LOG:-/workspace/verl/runs/vast_smoke_1p5b/train.log}"
mkdir -p "$(dirname "$LOG")"

echo "=== launching GRPO with $MODEL_PATH on 1 GPU (log: $LOG) ==="
bash examples/grpo_trainer/run_qwen3_4b_fsdp.sh \
  +trainer.total_training_steps=1 \
  trainer.save_freq=-1 \
  trainer.test_freq=-1 \
  actor_rollout_ref.rollout.enforce_eager=True \
  2>&1 | tee "$LOG"
