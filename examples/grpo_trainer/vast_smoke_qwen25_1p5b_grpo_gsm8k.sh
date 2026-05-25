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

# Single GPU, 1 step, no checkpoint, vLLM eager. Knobs tuned tight to fit
# under the Vast.ai container's cgroup pids.max (typically 1792); rollout
# pressure shrinks the per-step worker thread spawn burst and KV-cache
# shm pressure while still exercising the full FSDP + vLLM weight-transfer
# path.
export NGPUS_PER_NODE=1
export ROLLOUT_TP=1
export ROLLOUT_N=1
export ROLLOUT_GPU_MEM_UTIL=0.7
export TRAIN_BATCH_SIZE=4
export PPO_MINI_BATCH_SIZE=4
export PPO_MICRO_BATCH_SIZE_PER_GPU=1
export LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=1
export MAX_PROMPT_LENGTH=512
export MAX_RESPONSE_LENGTH=256
export PROJECT_NAME="${PROJECT_NAME:-verl_compression_research}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen25_1p5b_grpo_gsm8k_vast_smoke}"

LOG="${LOG:-/workspace/verl/runs/vast_smoke_1p5b/train.log}"
mkdir -p "$(dirname "$LOG")"

# Vast.ai hosts cap the container cgroup at `/sys/fs/cgroup/pids/pids.max`
# (typically 1792, read-only from inside the container). verl's vLLM
# rollout stack — Ray raylet + Ray dashboard sub-modules + vLLM multiproc
# executor + vLLM EngineCore + ZMQ I/O threads in
# verl/workers/rollout/vllm_rollout/bucketed_weight_transfer.py — easily
# spawns 1700+ pthreads at the FSDP->vLLM weight-transfer boundary,
# tripping the cap. The failure surface is
# `Resource temporarily unavailable (src/thread.cpp:241)` followed by
# zmq_socket SIGABRT and EngineCore death.
#
# Verl only registers `vllm/sglang/trtllm` for `mode=async` in
# `verl.workers.rollout.base._ROLLOUT_REGISTRY`; `hf` is a public class
# but not selectable via the trainer (asserts in `get_rollout_class`).
# So we keep vllm and instead trim Ray's idle thread footprint to leave
# headroom for vLLM's spawn burst:
#   - RAY_DISABLE_DASHBOARD=1 drops the dashboard's ~30 helper actors.
#   - OMP/MKL/TOKENIZERS=1 drops OpenMP/MKL/HF threadpools that each
#     Ray worker would otherwise import (each saves ~10-30 threads).
#   - RAY_DISABLE_USAGE_STATS=1 drops the usage telemetry actor.
export RAY_DISABLE_DASHBOARD=1
export RAY_DISABLE_USAGE_STATS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

ROLLOUT_NAME="${ROLLOUT_NAME:-vllm}"

echo "=== launching GRPO with $MODEL_PATH on 1 GPU rollout=$ROLLOUT_NAME (log: $LOG) ==="
echo "=== cgroup pids: $(cat /sys/fs/cgroup/pids/pids.current 2>/dev/null)/$(cat /sys/fs/cgroup/pids/pids.max 2>/dev/null) ==="
bash examples/grpo_trainer/run_qwen3_4b_fsdp.sh \
  +trainer.total_training_steps=1 \
  trainer.save_freq=-1 \
  trainer.test_freq=-1 \
  actor_rollout_ref.rollout.name="$ROLLOUT_NAME" \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=2048 \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=2048 \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=4096 \
  2>&1 | tee "$LOG"
