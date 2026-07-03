#!/usr/bin/env bash
# vast_baseline_qwen25_1p5b_grpo_gsm8k.sh
#
# REAL GRPO BASELINE (the DENSE CONTROL) — Qwen2.5-1.5B-Instruct on GSM8K,
# 1..8 GPUs, FSDP + vLLM rollout, 2 epochs over the train split, eval on the
# test split. Not a smoke test. Acceptance = pass@1 improvement on test.
# DEFAULT for analysis runs is a SINGLE 1xH200 (auto resp=1024 / TP=1, proven on
# 1xH200 at val ~0.77); 4..8 GPUs run the full 16K-context control.
#
# Objective: NO KL, no entropy (pg_loss only). This dense control learns cleanly
# on GSM8K and matches the comm-eff method's no-KL objective, so dense-vs-
# comm-eff is apples-to-apples. This is THE reference dense run.
#
# Runs on a Vast.ai instance provisioned from the verl-research-vllm020
# template (which clones shamanez/verl @ vast-ai-workload into /workspace/verl
# and pip-installs verl editable, preserving the verlai image's bundled
# torch / vllm / megatron / TE / deepep). No scp'd scripts; this file IS
# the launcher, lives in the fork at examples/grpo_trainer/, and you iterate
# on it by editing locally, committing+pushing to vast-ai-workload, then
# `git pull && bash <thisfile>` on the box.
#
# Prereqs on the box (template handles 1-2; you handle 3):
#   1. /workspace/verl checked out from shamanez/verl @ vast-ai-workload
#   2. verl pip-installed --no-deps -e .
#   3. ~/.config/verl-research/secrets.env present, containing ONLY HF_TOKEN
#      and WANDB_API_KEY (push a stripped copy from the laptop after
#      provisioning; VAST_API_KEY MUST NOT live on the box).
#
# Hardware: 1..8 GPUs (HARD-FAILS outside that range). Default box: 1×H200
# (project ladder 1×H200 → 1×B200 → 2×H200, reliability >0.99 — see
# research/.claude/project.yaml `default_compute`). On 1 GPU the launcher
# auto-selects the proven resp=1024 analysis surface; the full 16K-context
# control still wants the legacy 4..8 shapes (explicit operator request only).
#
# Iteration loop (e.g. tuning batch sizes when fitting GPUs):
#   laptop: edit this file
#   laptop: git commit -am "<note>" && git push origin vast-ai-workload
#   box:    cd /workspace/verl && git pull && bash examples/grpo_trainer/vast_baseline_qwen25_1p5b_grpo_gsm8k.sh
#
# See examples/grpo_trainer/VAST_README.md for the broader pattern.
set -euo pipefail

# ---------------------------------------------------------------------------
# 1. Secrets (HF + WandB only on the box; VAST_API_KEY MUST NOT live here).
# ---------------------------------------------------------------------------
SECRETS_FILE="${SECRETS_FILE:-$HOME/.config/verl-research/secrets.env}"
if [[ ! -r "$SECRETS_FILE" ]]; then
  cat >&2 <<EOF
FATAL: $SECRETS_FILE not found on this box.

  From the laptop, push a stripped copy:
    grep -E '^export (HF_TOKEN|WANDB_API_KEY)=' ~/.config/verl-research/secrets.env > /tmp/secrets-box.env
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

# Expose HF token under the names every HF client variant looks for.
export HF_TOKEN \
       HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-$HF_TOKEN}" \
       HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-$HF_TOKEN}" \
       WANDB_API_KEY

# ---------------------------------------------------------------------------
# 2. GPU count — normal/dense GRPO run accepts 1..8 GPUs.
# ---------------------------------------------------------------------------
# Single GPU (1xH200) is a SUPPORTED DEFAULT for this dense control / analysis
# runs (operator-authorized 2026-06-30): the normal GRPO ran cleanly on 1xH200
# at resp=1024 (val ~0.77). The comm-eff research loop's 4..8
# mandate (16K-context headroom) does NOT bind the dense control. The full 16K
# production surface still needs 4..8 GPUs; on 1 GPU the launcher auto-defaults
# to the analysis-friendly surface below.
DETECTED_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
if (( DETECTED_GPUS < 1 || DETECTED_GPUS > 8 )); then
  echo "FATAL: this recipe requires 1..8 GPUs; detected $DETECTED_GPUS" >&2
  exit 1
fi
export NGPUS_PER_NODE="$DETECTED_GPUS"
echo "=== detected $NGPUS_PER_NODE GPUs ($(nvidia-smi -L | head -1)) ==="

# Single-GPU (1xH200) analysis default. The 16K-context production surface does
# not fit one card with n=8 rollouts, so on 1 GPU default to the surface proven
# on 1xH200: resp=1024, TP=1, ppo token budget 16384, vLLM mem
# 0.5. 4..8 GPUs keep the 16K control unchanged. Every value stays env-overridable
# (these set defaults BEFORE the §5 ${VAR:-...} lines, so an explicit env wins).
if (( DETECTED_GPUS == 1 )); then
  echo "=== single-GPU normal run: defaulting to the 1xH200 analysis surface (resp=1024, TP=1) ===" >&2
  export ROLLOUT_TP="${ROLLOUT_TP:-1}"
  export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-1024}"
  export PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-16384}"
  export ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.5}"
fi

# ---------------------------------------------------------------------------
# 3. ulimit + cgroup probe (see VAST_README.md for the pids.max gotcha).
# ---------------------------------------------------------------------------
ulimit -n 65535 || echo "WARN: could not bump open-files ulimit" >&2
ulimit -u 65535 2>/dev/null || true

PIDS_MAX=$(cat /sys/fs/cgroup/pids/pids.max 2>/dev/null || echo unknown)
echo "=== cgroup pids.max=$PIDS_MAX (need >= 4096 for full vLLM + Ray stack on multi-GPU) ==="
if [[ "$PIDS_MAX" =~ ^[0-9]+$ ]] && (( PIDS_MAX <= 2048 )); then
  echo "FATAL: this Vast host's cgroup pids.max ($PIDS_MAX) is too tight for verl's" >&2
  echo "       FSDP + vLLM + Ray stack on >=4 GPUs (peak ~1700 threads per GPU)." >&2
  echo "       Tear down this instance and reprovision on a different machine_id." >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# 4. Dataset — preprocess GSM8K to parquet if not already there.
# ---------------------------------------------------------------------------
cd "${VERL_ROOT:-/workspace/verl}"

DATA_DIR="${DATA_DIR:-$HOME/data/gsm8k}"
if [[ ! -f "$DATA_DIR/train.parquet" || ! -f "$DATA_DIR/test.parquet" ]]; then
  echo "=== preprocess GSM8K -> $DATA_DIR ==="
  mkdir -p "$DATA_DIR"
  python3 examples/data_preprocess/gsm8k.py --local_save_dir "$DATA_DIR"
fi
export TRAIN_FILE="$DATA_DIR/train.parquet"
export TEST_FILE="$DATA_DIR/test.parquet"
echo "=== train: $(python3 -c "import pyarrow.parquet as p; print(p.read_table('$TRAIN_FILE').num_rows)") rows ==="
echo "=== test:  $(python3 -c "import pyarrow.parquet as p; print(p.read_table('$TEST_FILE').num_rows)") rows ==="

# ---------------------------------------------------------------------------
# 5. Model + training config — REAL run, not a toy.
# ---------------------------------------------------------------------------
# Qwen2.5-1.5B-Instruct — Apache-2.0, no HF gating, ~3 GB bf16.
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-1.5B-Instruct}"

# Rollout shape. n=8 rollouts/prompt is the canonical GRPO group size.
export ROLLOUT_TP="${ROLLOUT_TP:-2}"
export ROLLOUT_N="${ROLLOUT_N:-8}"
# 0.4 leaves ~50% of each GPU for FSDP weights + Adam state + activations.
# vLLM 0.20 KV cache is paged so this is the available pool, not the peak.
export ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.4}"

# Batch sizes. Global train_batch * ROLLOUT_N = sequences per GRPO step.
# 128 prompts * 8 rollouts = 1024 sequences/step; PPO mini_batch=64 means
# 2 PPO iterations per step.
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-64}"
# Micro-batch is a fallback; use_dynamic_bsz=True (passed below) packs by
# token budget instead. Keep micro at 1 so a fallback never explodes memory.
export PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
export LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}"

# Context windows — REQUIRED to be large; the user mandate is max 16K
# response. With paged KV cache, longer max doesn't cost upfront memory.
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-16384}"

# GRPO objective — NO KL, no entropy. No-KL dense GRPO learns cleanly on GSM8K;
# the comm-eff method is evaluated no-KL, so the dense control is no-KL too for
# an apples-to-apples comparison. This is THE reference dense run for the
# comm-eff project.
export ACTOR_LR="${ACTOR_LR:-1e-6}"
export USE_KL_LOSS="${USE_KL_LOSS:-False}"
export USE_KL_IN_REWARD="${USE_KL_IN_REWARD:-False}"
export KL_LOSS_COEF="${KL_LOSS_COEF:-0.001}"   # unused when USE_KL_LOSS=False
export ENTROPY_COEFF="${ENTROPY_COEFF:-0}"

# Run schedule.
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-2}"
export SAVE_FREQ="${SAVE_FREQ:-50}"
export TEST_FREQ="${TEST_FREQ:-25}"

# WandB project + experiment (MANDATORY logger; the canonical recipe sets
# trainer.logger='["console","wandb"]' unconditionally).
export PROJECT_NAME="${PROJECT_NAME:-verl_compression_research}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen25_1p5b_grpo_gsm8k_baseline}"

# Token budget per micro-batch for dynamic batching. Sized for ~2 sequences
# of (prompt+response) ≈ 17K tokens each, with ~10% slack. This is the
# single most important OOM-avoidance knob with 16K context.
PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-36864}"
LOG_PROB_MAX_TOKEN_LEN_PER_GPU="${LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-36864}"
REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU="${REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-36864}"

LOG="${LOG:-/workspace/verl/runs/qwen25_1p5b_grpo_gsm8k_baseline/train.log}"
mkdir -p "$(dirname "$LOG")"

cat <<EOF
=== launching GRPO baseline ===
  model:            $MODEL_PATH
  GPUs:             $NGPUS_PER_NODE
  rollout TP × N:   ${ROLLOUT_TP} × ${ROLLOUT_N}
  vLLM mem util:    $ROLLOUT_GPU_MEM_UTIL
  train batch:      $TRAIN_BATCH_SIZE prompts (× $ROLLOUT_N rollouts = $(( TRAIN_BATCH_SIZE * ROLLOUT_N )) sequences/step)
  ppo mini batch:   $PPO_MINI_BATCH_SIZE
  ppo max tokens/GPU/micro: $PPO_MAX_TOKEN_LEN_PER_GPU (dynamic_bsz=True)
  prompt / response: $MAX_PROMPT_LENGTH / $MAX_RESPONSE_LENGTH
  epochs:           $TOTAL_EPOCHS  (save every $SAVE_FREQ, validate every $TEST_FREQ)
  objective:        pg_loss only (use_kl_loss=$USE_KL_LOSS, use_kl_in_reward=$USE_KL_IN_REWARD, entropy_coeff=$ENTROPY_COEFF)
  wandb:            $PROJECT_NAME / $EXPERIMENT_NAME
  log:              $LOG
=== launching ===
EOF

# ---------------------------------------------------------------------------
# 6. Launch — reuse upstream's per-recipe script for the verbatim main_ppo
#    invocation, overriding the OOM-relevant Hydra knobs via positional args.
# ---------------------------------------------------------------------------
bash examples/grpo_trainer/run_qwen3_4b_fsdp.sh \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$PPO_MAX_TOKEN_LEN_PER_GPU" \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="$LOG_PROB_MAX_TOKEN_LEN_PER_GPU" \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="$REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU" \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.use_kl_loss="$USE_KL_LOSS" \
  algorithm.use_kl_in_reward="$USE_KL_IN_REWARD" \
  actor_rollout_ref.actor.entropy_coeff="$ENTROPY_COEFF" \
  "$@" \
  2>&1 | tee "$LOG"

touch /workspace/verl/runs/qwen25_1p5b_grpo_gsm8k_baseline/done.flag
echo "=== done at $(date -u +%FT%TZ) ==="
