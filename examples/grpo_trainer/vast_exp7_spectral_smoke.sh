#!/usr/bin/env bash
# vast_exp7_spectral_smoke.sh
#
# EXP-7 — M2 spectral-correction filter + FSDP gradient-application-point
# DISCOVERY smoke. Seeded two-step GRPO on Qwen2.5-1.5B-Instruct / GSM8K.
#
# This is NOT a convergence run. It is the smallest run that exercises
#   masked(off here) fwd/bwd -> FSDP grad reduction -> spectral correction ->
#   AdamW
# across >1 optimizer update, with a SEEDED anchor cache (no live anchor
# circuit yet), so the FSDP engine can instrument and log the gradient
# representation at the correction point (the headline deliverable) and the
# per-target ||G_proj - G_mask||/||G_mask|| ratio.
#
# CELL is selected by the env var CELL:
#   CELL=spectral_on   (PRIMARY) comm_eff.enabled=true, spectral.enabled=true,
#                      alpha=0.3, tau=1e-3, beta_anc=0.95, seed_anchor_cache=true,
#                      masking OFF.
#   CELL=disabled      (REGRESSION) comm_eff.spectral.enabled=false ⇒ must be a
#                      strict no-op identical to EXP-5 / dense GRPO.
#
# Reuses the baseline launcher's secrets/GPU/cgroup/dataset bring-up by sourcing
# its preamble indirectly: we set the tiny-smoke env knobs, then call the same
# run_qwen3_4b_fsdp.sh recipe with the smoke + comm_eff Hydra overrides.
set -euo pipefail

CELL="${CELL:-spectral_on}"

# ---------------------------------------------------------------------------
# 1. Secrets (HF + WandB only on the box; VAST_API_KEY MUST NOT live here).
# ---------------------------------------------------------------------------
SECRETS_FILE="${SECRETS_FILE:-$HOME/.config/verl-research/secrets.env}"
if [[ ! -r "$SECRETS_FILE" ]]; then
  echo "FATAL: $SECRETS_FILE not found on this box." >&2
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
export HF_TOKEN \
       HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-$HF_TOKEN}" \
       HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-$HF_TOKEN}" \
       WANDB_API_KEY

# ---------------------------------------------------------------------------
# 2. GPU count — multi-GPU MANDATE (4..8).
# ---------------------------------------------------------------------------
DETECTED_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
if (( DETECTED_GPUS < 4 || DETECTED_GPUS > 8 )); then
  echo "FATAL: this recipe requires 4..8 GPUs; detected $DETECTED_GPUS" >&2
  exit 1
fi
export NGPUS_PER_NODE="$DETECTED_GPUS"
echo "=== detected $NGPUS_PER_NODE GPUs ($(nvidia-smi -L | head -1)) ==="

ulimit -n 65535 || echo "WARN: could not bump open-files ulimit" >&2
ulimit -u 65535 2>/dev/null || true
PIDS_MAX=$(cat /sys/fs/cgroup/pids/pids.max 2>/dev/null || echo unknown)
echo "=== cgroup pids.max=$PIDS_MAX ==="
if [[ "$PIDS_MAX" =~ ^[0-9]+$ ]] && (( PIDS_MAX <= 2048 )); then
  echo "FATAL: cgroup pids.max ($PIDS_MAX) too tight; reprovision on another machine_id." >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# 3. Dataset.
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

# ---------------------------------------------------------------------------
# 4. Tiny-smoke shape (per EXP-7 plan Notes for runner).
# ---------------------------------------------------------------------------
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-1.5B-Instruct}"
export TRAIN_BATCH_SIZE=8
export PPO_MINI_BATCH_SIZE=4
export PPO_MICRO_BATCH_SIZE_PER_GPU=1
export LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=1
export MAX_PROMPT_LENGTH=256
export MAX_RESPONSE_LENGTH=256
export ROLLOUT_N=2
export ROLLOUT_TP="${ROLLOUT_TP:-2}"
export ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.4}"
export ACTOR_LR="${ACTOR_LR:-1e-6}"
export KL_LOSS_COEF="${KL_LOSS_COEF:-0.001}"
export SAVE_FREQ="${SAVE_FREQ:--1}"     # no checkpoint for a 2-step smoke
export TEST_FREQ="${TEST_FREQ:--1}"     # no mid-run eval
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
export PROJECT_NAME="${PROJECT_NAME:-verl_compression_research}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-exp7_spectral_${CELL}}"

# ---------------------------------------------------------------------------
# 5. comm_eff overrides per cell. Masking is OFF in both (combined mask+spectral
#    is EXP-9). Spectral is the variable.
# ---------------------------------------------------------------------------
COMM_EFF_ARGS=()
if [[ "$CELL" == "spectral_on" ]]; then
  COMM_EFF_ARGS=(
    actor_rollout_ref.actor.comm_eff.enabled=true
    actor_rollout_ref.actor.comm_eff.mask.enabled=false
    actor_rollout_ref.actor.comm_eff.mask.p=0.0
    actor_rollout_ref.actor.comm_eff.spectral.enabled=true
    actor_rollout_ref.actor.comm_eff.spectral.alpha=0.3
    actor_rollout_ref.actor.comm_eff.spectral.tau=0.001
    actor_rollout_ref.actor.comm_eff.spectral.beta_anc=0.95
    actor_rollout_ref.actor.comm_eff.spectral.seed_anchor_cache=true
  )
elif [[ "$CELL" == "disabled" ]]; then
  COMM_EFF_ARGS=(
    actor_rollout_ref.actor.comm_eff.enabled=false
    actor_rollout_ref.actor.comm_eff.spectral.enabled=false
  )
else
  echo "FATAL: unknown CELL=$CELL (expected spectral_on | disabled)" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# 6. Smoke schedule overrides: exactly 2 optimizer trainer steps, no pre-train
#    validation, single PPO epoch.
# ---------------------------------------------------------------------------
SMOKE_ARGS=(
  trainer.total_training_steps=2
  trainer.val_before_train=False
  actor_rollout_ref.actor.ppo_epochs=1
)

RUN_DIR="/workspace/runs/EXP-7"
LOG="$RUN_DIR/train_${CELL}.log"
mkdir -p "$RUN_DIR/metrics"
export LOG

cat <<EOF
=== EXP-7 spectral smoke (CELL=$CELL) ===
  model:   $MODEL_PATH   GPUs: $NGPUS_PER_NODE
  shape:   batch=$TRAIN_BATCH_SIZE mini=$PPO_MINI_BATCH_SIZE rollout_n=$ROLLOUT_N prompt/resp=$MAX_PROMPT_LENGTH/$MAX_RESPONSE_LENGTH
  steps:   2  (val_before_train=False, ppo_epochs=1)
  comm_eff:${COMM_EFF_ARGS[*]}
  log:     $LOG
=== launching ===
EOF

bash examples/grpo_trainer/run_qwen3_4b_fsdp.sh \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.use_remove_padding=True \
  "${COMM_EFF_ARGS[@]}" \
  "${SMOKE_ARGS[@]}" \
  "$@" \
  2>&1 | tee "$LOG"

# done.flag: pre-created dir avoids the carried-over baseline-launcher bug where
# a hardcoded path under a differently-named run dir trips `set -e`.
mkdir -p "$RUN_DIR"
touch "$RUN_DIR/done_${CELL}.flag"
echo "=== EXP-7 CELL=$CELL done at $(date -u +%FT%TZ) ==="
