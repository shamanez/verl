#!/usr/bin/env bash
# vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh
#
# COMMUNICATION-EFFICIENT GRPO — Qwen2.5-1.5B-Instruct on GSM8K, multi-GPU
# (4..8), FSDP + vLLM rollout. Mirrors the dense baseline launcher
# (vast_baseline_qwen25_1p5b_grpo_gsm8k.sh) one-for-one in training shape so
# the two compare apples-to-apples; the ONLY differences are the
# communication-efficient method's hydra knobs and the no-KL / no-entropy
# objective.
#
# ===========================================================================
# THE "moment of truth" launcher for the next runs. EVERY circuit is an
# independent env toggle so the full ablation grid is one launcher:
#
#   comm-eff master ........ COMM_EFF_ENABLED          (true)   off => byte-identical dense
#   masking ................ COMM_EFF_MASK_ENABLED     (true)
#   rescale ................ COMM_EFF_MASK_RESCALE     (true)   inverted-dropout h*mask/(1-p)
#   granularity ............ COMM_EFF_MASK_GRANULARITY (channel) channel | element
#   naive clean cadence .... COMM_EFF_CLEAN_CADENCE    (0=OFF)  full (unmasked) grad every N steps
#   anchor ................. COMM_EFF_ANCHOR_ENABLED   (false)
#   spectral correction .... COMM_EFF_SPECTRAL_ENABLED (false)
#
# Defaults encode EXP-14's findings (GitHub #14, verdict runs/EXP-14/verdict.md):
#   * granularity=channel (per-channel mask) — packing-invariant ⇒ EXACT
#     cross-pass IS consistency, the only no-plumbing route (per-element would
#     need per-token keying). DEFAULT.
#   * rescale=true — inverted-dropout h*mask/(1-p) preserves E[h] and tames the
#     mask's magnitude-collapse grad_norm explosion (paper-scale 771 -> ~1.5).
#     DEFAULT ON.  ⚠ rescale fixes grad_norm but EXP-14 showed it does NOT, by
#     itself, recover LEARNING at p=0.9 (val stayed flat). The mask-rate sweep
#     (p=0.9->0.5->0.1) is the open question — see GitHub #15.
#   * clean_cadence=0 (OFF). The periodic full-(unmasked)-gradient step is the
#     NAIVE cadence method; EXP-14 proved it is NOT sustainable — masked steps
#     still explode and PPO pg_clipfrac climbs toward saturation (0.26->0.44),
#     which kills learning. Opt-in knob only, do not ship it as the method.
#   * anchor + spectral OFF — start from the mask-only path; layer these on
#     only after a masked config is shown to actually LEARN (val/score, not
#     just a bounded grad_norm).
# ===========================================================================
#
# Runs on a Vast.ai instance provisioned from the verl-research-vllm020
# template (clones shamanez/verl @ vast-ai-workload into /workspace/verl and
# pip-installs verl editable). This file IS the launcher; iterate by editing
# locally, committing+pushing to vast-ai-workload, then
# `git pull && bash <thisfile>` on the box.
#
# Prereqs on the box:
#   1. /workspace/verl checked out from shamanez/verl @ vast-ai-workload
#   2. verl pip-installed --no-deps -e .
#   3. ~/.config/verl-research/secrets.env present (ONLY HF_TOKEN + WANDB_API_KEY).
#
# Hardware: multi-GPU only (4..8). With anchor OFF (default) the ~3 GB anchor
# clone is NOT allocated, so 4×H200 fits the restored baseline knobs
# (mini=64, wedge=36864, util=0.4) comfortably. Only re-enabling the anchor
# (COMM_EFF_ANCHOR_ENABLED=true) brings the clone back — then prefer 8×GPU or
# halve PPO_MAX_TOKEN_LEN_PER_GPU to 18432.
#
# Ablation examples:
#   # mask-only, no rescale (reproduce the explosion):
#   COMM_EFF_MASK_RESCALE=false EXPERIMENT_NAME=ce_mask_only bash <thisfile>
#   # dense control via the same launcher:
#   COMM_EFF_ENABLED=false EXPERIMENT_NAME=ce_off_dense bash <thisfile>
#   # mask-rate sweep point:
#   COMM_EFF_MASK_P=0.5 EXPERIMENT_NAME=ce_p0p5 bash <thisfile>
#
# See examples/grpo_trainer/VAST_README.md for the broader Vast.ai pattern.
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

# Expose HF token under every name HF clients look for.
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
  echo "       (1.5B GRPO with 16K response + n=8 rollouts needs the headroom)" >&2
  exit 1
fi
export NGPUS_PER_NODE="$DETECTED_GPUS"
echo "=== detected $NGPUS_PER_NODE GPUs ($(nvidia-smi -L | head -1)) ==="

# ---------------------------------------------------------------------------
# 3. ulimit + cgroup probe.
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
# 5. Model + training config — matches the dense baseline launcher 1:1
#    EXCEPT the objective is no-KL no-entropy (the method's design) and
#    `actor.fsdp_config.use_orig_params=true` (so the optional spectral hook,
#    when enabled, sees full 2D Tensor gradients post-FSDP-reduce).
# ---------------------------------------------------------------------------
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-1.5B-Instruct}"

# Rollout shape — n=8 rollouts/prompt, paged KV.
export ROLLOUT_TP="${ROLLOUT_TP:-2}"
export ROLLOUT_N="${ROLLOUT_N:-8}"
export ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.4}"

# Batch sizes — match baseline.
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-64}"
export PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
export LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}"

# Context windows — match baseline (16K response).
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-16384}"

# GRPO objective — no KL, no entropy (communication-efficient method design;
# this matches the dense baseline, which is also no-KL, for apples-to-apples).
export ACTOR_LR="${ACTOR_LR:-1e-6}"
export USE_KL_LOSS="${USE_KL_LOSS:-False}"
export USE_KL_IN_REWARD="${USE_KL_IN_REWARD:-False}"
export KL_LOSS_COEF="${KL_LOSS_COEF:-0.001}"   # unused when USE_KL_LOSS=False
export ENTROPY_COEFF="${ENTROPY_COEFF:-0}"

# Run schedule — match baseline.
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-2}"
export SAVE_FREQ="${SAVE_FREQ:-50}"
export TEST_FREQ="${TEST_FREQ:-25}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-True}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-100}"

# WandB project + experiment.
export PROJECT_NAME="${PROJECT_NAME:-verl_compression_research}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen25_1p5b_grpo_gsm8k_comm_eff_baseline}"

# Token budget per micro-batch for dynamic batching.
PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-36864}"
LOG_PROB_MAX_TOKEN_LEN_PER_GPU="${LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-36864}"
REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU="${REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-36864}"

# ---------------------------------------------------------------------------
# 6. Communication-efficient method — hydra knob surface (see header).
#    Every circuit is an independent env toggle. Defaults = the mask-only
#    "comm-eff baseline" (mask + rescale + per-channel; cadence/anchor/spectral
#    OFF). Field names mirror verl/trainer/config/actor/actor.yaml exactly —
#    do NOT reference a knob absent from that schema (Hydra struct-mode rejects
#    unknown keys regardless of enabled flags; that bit us on clean_cadence).
# ---------------------------------------------------------------------------
COMM_EFF_ENABLED="${COMM_EFF_ENABLED:-true}"                          # master switch (false => dense)
# --- activation mask ---
COMM_EFF_MASK_ENABLED="${COMM_EFF_MASK_ENABLED:-true}"
COMM_EFF_MASK_P="${COMM_EFF_MASK_P:-0.9}"                             # masked fraction (sweep 0.9->0.5->0.1, #15)
COMM_EFF_MASK_GRANULARITY="${COMM_EFF_MASK_GRANULARITY:-channel}"     # channel (default) | element (legacy)
COMM_EFF_MASK_RESCALE="${COMM_EFF_MASK_RESCALE:-true}"               # inverted-dropout h*mask/(1-p)
COMM_EFF_MASK_RECOMPUTE="${COMM_EFF_MASK_RECOMPUTE:-true}"            # mask the old_logprob forward too
COMM_EFF_MASK_CONSISTENT="${COMM_EFF_MASK_CONSISTENT:-true}"          # consistent_across_forwards (no-op under channel)
COMM_EFF_MASK_SEED="${COMM_EFF_MASK_SEED:-0}"                         # PRF base seed
COMM_EFF_MASK_PP_SIZE="${COMM_EFF_MASK_PP_SIZE:-8}"                   # simulated pipeline depth (boundary blocks)
# --- naive periodic clean (unmasked) step: 0=OFF. NOT sustainable (PPO clip saturation, EXP-14). ---
COMM_EFF_CLEAN_CADENCE="${COMM_EFF_CLEAN_CADENCE:-0}"
# --- anchor circuit (OFF by default) ---
COMM_EFF_ANCHOR_ENABLED="${COMM_EFF_ANCHOR_ENABLED:-false}"
COMM_EFF_ANCHOR_CADENCE="${COMM_EFF_ANCHOR_CADENCE:-5}"
# --- spectral correction (OFF by default) ---
COMM_EFF_SPECTRAL_ENABLED="${COMM_EFF_SPECTRAL_ENABLED:-false}"
COMM_EFF_SPECTRAL_ALPHA="${COMM_EFF_SPECTRAL_ALPHA:-0.5}"
COMM_EFF_SPECTRAL_TAU="${COMM_EFF_SPECTRAL_TAU:-0.01}"
COMM_EFF_SPECTRAL_BETA_ANC="${COMM_EFF_SPECTRAL_BETA_ANC:-0.9}"
COMM_EFF_SPECTRAL_EMA_DEVICE="${COMM_EFF_SPECTRAL_EMA_DEVICE:-gpu}"
COMM_EFF_SPECTRAL_SVD_MODE="${COMM_EFF_SPECTRAL_SVD_MODE:-full}"
COMM_EFF_SPECTRAL_BASIS_CACHE="${COMM_EFF_SPECTRAL_BASIS_CACHE:-cache}"
COMM_EFF_SPECTRAL_MAX_TARGETS="${COMM_EFF_SPECTRAL_MAX_TARGETS:-4}"
COMM_EFF_SPECTRAL_SEED_ANCHOR_CACHE="${COMM_EFF_SPECTRAL_SEED_ANCHOR_CACHE:-true}"

if [[ "${COMM_EFF_ANCHOR_ENABLED}" == "true" ]]; then
  echo "WARN: anchor enabled -> ~3 GB clone/rank is back; prefer 8×GPU or halve PPO_MAX_TOKEN_LEN_PER_GPU to 18432." >&2
fi

LOG="${LOG:-/workspace/verl/runs/${EXPERIMENT_NAME}/train.log}"
mkdir -p "$(dirname "$LOG")"

cat <<EOF
=== launching communication-efficient GRPO ===
  model:               $MODEL_PATH
  GPUs:                $NGPUS_PER_NODE
  rollout TP × N:      ${ROLLOUT_TP} × ${ROLLOUT_N}
  vLLM mem util:       $ROLLOUT_GPU_MEM_UTIL
  train batch:         $TRAIN_BATCH_SIZE prompts (× $ROLLOUT_N = $(( TRAIN_BATCH_SIZE * ROLLOUT_N )) seqs/step)
  ppo mini batch:      $PPO_MINI_BATCH_SIZE
  ppo max tokens/GPU:  $PPO_MAX_TOKEN_LEN_PER_GPU (dynamic_bsz=True)
  prompt / response:   $MAX_PROMPT_LENGTH / $MAX_RESPONSE_LENGTH
  epochs:              $TOTAL_EPOCHS  (save $SAVE_FREQ, validate $TEST_FREQ, total steps $TOTAL_TRAINING_STEPS)
  val_before_train:    $VAL_BEFORE_TRAIN
  objective:           pg_loss only (use_kl_loss=$USE_KL_LOSS, use_kl_in_reward=$USE_KL_IN_REWARD, entropy_coeff=$ENTROPY_COEFF)
  comm_eff master:     $COMM_EFF_ENABLED
  mask:                enabled=$COMM_EFF_MASK_ENABLED p=$COMM_EFF_MASK_P granularity=$COMM_EFF_MASK_GRANULARITY rescale=$COMM_EFF_MASK_RESCALE recompute=$COMM_EFF_MASK_RECOMPUTE consistent=$COMM_EFF_MASK_CONSISTENT seed=$COMM_EFF_MASK_SEED pp_size=$COMM_EFF_MASK_PP_SIZE
  clean_cadence:       $COMM_EFF_CLEAN_CADENCE  (0=off; naive periodic full-grad step — NOT sustainable, EXP-14)
  anchor:              enabled=$COMM_EFF_ANCHOR_ENABLED cadence=$COMM_EFF_ANCHOR_CADENCE
  spectral:            enabled=$COMM_EFF_SPECTRAL_ENABLED alpha=$COMM_EFF_SPECTRAL_ALPHA tau=$COMM_EFF_SPECTRAL_TAU beta_anc=$COMM_EFF_SPECTRAL_BETA_ANC max_targets=$COMM_EFF_SPECTRAL_MAX_TARGETS
  wandb:               $PROJECT_NAME / $EXPERIMENT_NAME
  log:                 $LOG
=== launching ===
EOF

# ---------------------------------------------------------------------------
# 7. Launch — reuse upstream's per-recipe script for the verbatim main_ppo
#    invocation, overriding the OOM-relevant + comm-eff Hydra knobs. Every
#    enabled flag comes from env so the full ablation grid is a one-liner.
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
  actor_rollout_ref.actor.fsdp_config.use_orig_params=true \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.use_kl_loss="$USE_KL_LOSS" \
  algorithm.use_kl_in_reward="$USE_KL_IN_REWARD" \
  actor_rollout_ref.actor.entropy_coeff="$ENTROPY_COEFF" \
  trainer.total_training_steps="$TOTAL_TRAINING_STEPS" \
  trainer.val_before_train="$VAL_BEFORE_TRAIN" \
  actor_rollout_ref.actor.comm_eff.enabled="$COMM_EFF_ENABLED" \
  actor_rollout_ref.actor.comm_eff.clean_cadence="$COMM_EFF_CLEAN_CADENCE" \
  actor_rollout_ref.actor.comm_eff.mask.enabled="$COMM_EFF_MASK_ENABLED" \
  actor_rollout_ref.actor.comm_eff.mask.p="$COMM_EFF_MASK_P" \
  actor_rollout_ref.actor.comm_eff.mask.granularity="$COMM_EFF_MASK_GRANULARITY" \
  actor_rollout_ref.actor.comm_eff.mask.rescale="$COMM_EFF_MASK_RESCALE" \
  actor_rollout_ref.actor.comm_eff.mask.mask_recompute="$COMM_EFF_MASK_RECOMPUTE" \
  actor_rollout_ref.actor.comm_eff.mask.consistent_across_forwards="$COMM_EFF_MASK_CONSISTENT" \
  actor_rollout_ref.actor.comm_eff.mask.seed="$COMM_EFF_MASK_SEED" \
  actor_rollout_ref.actor.comm_eff.mask.pp_size="$COMM_EFF_MASK_PP_SIZE" \
  actor_rollout_ref.actor.comm_eff.anchor.enabled="$COMM_EFF_ANCHOR_ENABLED" \
  actor_rollout_ref.actor.comm_eff.anchor.cadence="$COMM_EFF_ANCHOR_CADENCE" \
  actor_rollout_ref.actor.comm_eff.spectral.enabled="$COMM_EFF_SPECTRAL_ENABLED" \
  actor_rollout_ref.actor.comm_eff.spectral.alpha="$COMM_EFF_SPECTRAL_ALPHA" \
  actor_rollout_ref.actor.comm_eff.spectral.tau="$COMM_EFF_SPECTRAL_TAU" \
  actor_rollout_ref.actor.comm_eff.spectral.beta_anc="$COMM_EFF_SPECTRAL_BETA_ANC" \
  actor_rollout_ref.actor.comm_eff.spectral.seed_anchor_cache="$COMM_EFF_SPECTRAL_SEED_ANCHOR_CACHE" \
  actor_rollout_ref.actor.comm_eff.spectral.ema_device="$COMM_EFF_SPECTRAL_EMA_DEVICE" \
  actor_rollout_ref.actor.comm_eff.spectral.svd_mode="$COMM_EFF_SPECTRAL_SVD_MODE" \
  actor_rollout_ref.actor.comm_eff.spectral.basis_cache="$COMM_EFF_SPECTRAL_BASIS_CACHE" \
  actor_rollout_ref.actor.comm_eff.spectral.max_targets="$COMM_EFF_SPECTRAL_MAX_TARGETS" \
  "$@" \
  2>&1 | tee "$LOG"

touch "/workspace/verl/runs/${EXPERIMENT_NAME}/done.flag"
echo "=== done at $(date -u +%FT%TZ) ==="
