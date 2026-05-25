#!/usr/bin/env bash
# vast_smoke_qwen25_0p5b_grpo_gsm8k.sh
#
# Single-GPU verl GRPO smoke for vast.ai, designed to be runnable directly
# from the box after the verl-research-vllm020 template's onstart has
# cloned this fork into /workspace/verl. NO scp'd files, NO /tmp scripts —
# this script IS the smoke, lives in the fork at examples/grpo_trainer/, and
# you iterate on it by editing locally, committing+pushing to
# `vast-ai-workload`, then `git pull && bash <thisfile>` on the box.
#
# Goal: reach `global_step: 1` with a 0.5B model on a 1×24GB GPU. OOM after
# step 1 is acceptable — the point is to exercise the full pipeline (model
# download, vLLM rollout engine, FSDP wrap, reward scoring, optimizer step).
#
# Prereqs on the box (template onstart handles them):
#   - /workspace/verl checked out from shamanez/verl @ vast-ai-workload
#   - verl pip-installed editable with --no-deps (preserves bundled vllm/torch)
#   - ~/.config/verl-research/secrets.env present, containing HF_TOKEN +
#     WANDB_API_KEY (push from the laptop via scp after provisioning;
#     VAST_API_KEY MUST be stripped — see the harness's hardening discipline).
#
# Iteration loop (e.g. when fitting GPUs / hitting OOM):
#   1. On laptop:  edit this file, tweak batch / dtype / TP / etc.
#   2. On laptop:  git commit -am "smoke: tweak X" && git push origin vast-ai-workload
#   3. On box:     cd /workspace/verl && git pull && bash examples/grpo_trainer/vast_smoke_qwen25_0p5b_grpo_gsm8k.sh
#
# See examples/grpo_trainer/VAST_README.md for the broader pattern.
set -euo pipefail

# ---------------------------------------------------------------------------
# Secrets (HF + WandB only on the box; VAST_API_KEY MUST NOT live here).
# ---------------------------------------------------------------------------
SECRETS_FILE="${SECRETS_FILE:-$HOME/.config/verl-research/secrets.env}"
if [[ ! -r "$SECRETS_FILE" ]]; then
  cat >&2 <<EOF
FATAL: $SECRETS_FILE not found on this box.

  From the laptop, push a stripped copy:
    grep -E '^(HF_TOKEN|WANDB_API_KEY)=' ~/.config/verl-research/secrets.env > /tmp/secrets-box.env
    chmod 600 /tmp/secrets-box.env
    scp -P <port> /tmp/secrets-box.env root@<host>:~/.config/verl-research/secrets.env
    shred -u /tmp/secrets-box.env  # remove the laptop copy
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

# Make HF + WandB clients see the creds explicitly.
export HF_TOKEN HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-$HF_TOKEN}" \
       HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-$HF_TOKEN}" WANDB_API_KEY

cd "${VERL_ROOT:-/workspace/verl}"

# ---------------------------------------------------------------------------
# Knobs (override at invocation with `MODEL_PATH=... bash <this>`).
# ---------------------------------------------------------------------------
# Open-weights Qwen 0.5B — Apache-2.0, no HF gating, ~1 GB. Sized for 24GB GPU.
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-0.5B-Instruct}"

# GSM8K parquet (preprocessed if absent).
DATA_DIR="${DATA_DIR:-$HOME/data/gsm8k}"
if [[ ! -f "$DATA_DIR/train.parquet" ]]; then
  echo "=== preprocess GSM8K → $DATA_DIR ==="
  python3 examples/data_preprocess/gsm8k.py --local_save_dir "$DATA_DIR"
fi
export TRAIN_FILE="$DATA_DIR/train.parquet"
export TEST_FILE="$DATA_DIR/test.parquet"
ls -la "$DATA_DIR"

# Single GPU, 1 step, no checkpoint, vLLM eager (skip CUDA graph capture on
# consumer GPUs where graph capture can stutter), wandb logging on.
export NGPUS_PER_NODE=1
export ROLLOUT_TP=1
export ROLLOUT_N=2
export TRAIN_BATCH_SIZE=8
export PPO_MINI_BATCH_SIZE=4
export PROJECT_NAME="${PROJECT_NAME:-verl_compression_research}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen25_0p5b_grpo_gsm8k_vast_smoke}"

# ---------------------------------------------------------------------------
# Launch — reuse upstream's run_qwen3_4b_fsdp.sh as the canonical launcher,
# overriding the knobs above + a few Hydra flags via CLI. Keeping the
# upstream launcher path intact means future verl-project mainline updates
# merge cleanly into this fork.
# ---------------------------------------------------------------------------
LOG="${LOG:-/workspace/verl/runs/vast_smoke/train.log}"
mkdir -p "$(dirname "$LOG")"

echo "=== launching GRPO with $MODEL_PATH on 1 GPU (log: $LOG) ==="
bash examples/grpo_trainer/run_qwen3_4b_fsdp.sh \
  +trainer.total_training_steps=1 \
  trainer.save_freq=-1 \
  trainer.test_freq=-1 \
  actor_rollout_ref.rollout.enforce_eager=True \
  2>&1 | tee "$LOG"
