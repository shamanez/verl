#!/usr/bin/env bash
# vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh
#
# COMMUNICATION-EFFICIENT GRPO — Qwen2.5-1.5B-Instruct on GSM8K, 1..8 GPUs
# (default box: 1×H200 per project.yaml `default_compute`; legacy 4..8 shapes
# on explicit operator request), FSDP + vLLM rollout. Mirrors the dense baseline launcher
# (vast_baseline_qwen25_1p5b_grpo_gsm8k.sh) one-for-one in training shape so
# the two compare apples-to-apples; the ONLY differences are the
# communication-efficient method's hydra knobs and the no-KL / no-entropy
# objective.
#
# ===========================================================================
# Canonical communication-efficient base launcher. The defaults below configure
# the anchor circuit on a PowerSGD codec. Every circuit is still an
# independent env toggle so ablations stay one-liners:
#
#   comm-eff master ........ COMM_EFF_ENABLED          (true)     off => dense path
#   codec .................. COMM_EFF_COMPRESSION_TYPE (powersgd) compression codec
#   PowerSGD rank .......... COMM_EFF_POWERSGD_RANK    (77)       byte-matched to mask p=0.95 (H=1536)
#   anchor ................. COMM_EFF_ANCHOR_ENABLED   (true)     stale full-grad reference M
#   anchor owns Q .......... COMM_EFF_ANCHOR_OWNS_Q    (true)     the ONLY thing that updates Q
#   anchor staleness ....... COMM_EFF_ANCHOR_DELAY_K   (5)        forward from theta_{t-5}
#   anchor refresh ......... COMM_EFF_ANCHOR_CADENCE   (5)        recompute M+Q every 5 ticks
#   merger ................. COMM_EFF_SPECTRAL_CORRECTION_MODE (signed_ema)  folds anchor M into G via a signed EMA
#   merger EMA decay ....... COMM_EFF_SPECTRAL_BETA_ANC         (0.50)        anchor-gradient EMA decay
#   clean cadence .......... COMM_EFF_CLEAN_CADENCE    (0=OFF)    periodic uncompressed step
#   PRF mask ............... COMM_EFF_MASK_ENABLED     (false)    prf_mask codec
#
# Base in one line: PowerSGD r=77 plus a continuously maintained, stale,
# full-coverage (196 matrices, DP-reduced) anchor gradient M, refreshed every
# `cadence` ticks from a no-hook isolated clone; the anchor OWNS the PowerSGD
# basis Q (computes Q<-orth(V) from its stale-forward activations and broadcasts
# it — the fast circuit is a read-only consumer, fail-closed from ever writing
# Q); the signed_ema merger folds M into the fast gradient via a signed EMA
# (alpha, beta_anc).
#
# Keep the model, dataset, rollout shape, and optimizer surface fixed unless the
# run is explicitly testing one of those axes. Validation metrics drive comparisons.
# ===========================================================================
#
# Runs on a Vast.ai instance provisioned from the verl-research-vllm020
# template (clones shamanez/verl @ autonomous-harness-v1 into /workspace/verl and
# pip-installs verl editable). This file IS the launcher; iterate by editing
# locally, committing+pushing to autonomous-harness-v1, then
# `git pull && bash <thisfile>` on the box.
#
# Prereqs on the box:
#   1. /workspace/verl checked out from shamanez/verl @ autonomous-harness-v1
#   2. verl pip-installed --no-deps -e .
#   3. ~/.config/verl-research/secrets.env present (ONLY HF_TOKEN + WANDB_API_KEY).
#
# Hardware: 1..8 GPUs; default box 1×H200 (project ladder — project.yaml
# `default_compute`). The anchor allocates a ~3 GB
# no-hook clone/rank for its stale forward-backward, so the default actor token
# budget is already halved (PPO_MAX_TOKEN_LEN_PER_GPU=18432, sized to fit the
# legacy 4×H200 shape and comfortable on 1×H200 at resp=1024); this
# is the default memory posture. Disabling the anchor (COMM_EFF_ANCHOR_ENABLED=false,
# a reference-only ablation) frees the clone and you can raise it back to 36864.
#
# Examples:
#   # signed_ema is the default merger. Override run length / name as needed:
#   TOTAL_TRAINING_STEPS=100 EXPERIMENT_NAME=ce_100 bash <thisfile>
#   # dense control via the same launcher:
#   COMM_EFF_ENABLED=false EXPERIMENT_NAME=ce_off_dense bash <thisfile>
#   # prf_mask codec comparison run; cannot anchor-own Q:
#   COMM_EFF_COMPRESSION_TYPE=prf_mask COMM_EFF_MASK_ENABLED=true \
#     COMM_EFF_ANCHOR_OWNS_Q=false EXPERIMENT_NAME=ce_mask_ref bash <thisfile>
#
# See examples/grpo_trainer/COMM_EFF_CONFIG.md for the full knob reference and
# examples/grpo_trainer/VAST_README.md for the broader Vast.ai pattern.
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
# 2. GPU count — 1..8 (default 1, matching the 1×H200 default box).
#    Since 2026-07-03 single-GPU is the DEFAULT (proven on 1×H200 for both the
#    GSM8K fast surface and Big-Math @ resp 4096; DP degree changes only the
#    reduction order at fixed global batch, and the 16K-context rationale
#    behind the old 4..8 mandate is defused by resp<=4096 surfaces). The legacy
#    4×H200 / 8×H100 shapes remain supported for explicit operator request.
#    ALLOW_SINGLE_GPU is accepted for back-compat but no longer required;
#    REQUIRE_MULTI_GPU=1 restores the legacy 4..8 hard gate.
# ---------------------------------------------------------------------------
DETECTED_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
GPU_MIN=1
if [[ "${REQUIRE_MULTI_GPU:-0}" == "1" ]]; then
  GPU_MIN=4
fi
if (( DETECTED_GPUS < GPU_MIN || DETECTED_GPUS > 8 )); then
  echo "FATAL: this recipe requires ${GPU_MIN}..8 GPUs; detected $DETECTED_GPUS" >&2
  if (( GPU_MIN > 1 )); then
    echo "       (1.5B GRPO with 16K response + n=8 rollouts needs the headroom;" >&2
    echo "        set ALLOW_SINGLE_GPU=1 for the single-GPU weight-trajectory collection path)" >&2
  fi
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
VERL_ROOT="${VERL_ROOT:-/workspace/verl}"
cd "$VERL_ROOT"

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
#    `actor.fsdp_config.use_orig_params=true` (REQUIRED so the anchor + merger
#    hooks see full 2D Tensor gradients post-FSDP-reduce — the base needs it).
# ---------------------------------------------------------------------------
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-1.5B-Instruct}"

# Rollout shape — n=8 rollouts/prompt, paged KV.
export ROLLOUT_TP="${ROLLOUT_TP:-2}"
# Clamp TP to the detected GPU count: the single-GPU collection path forces TP=1
# (rollout tensor-parallel can't exceed the device count). No effect on >=TP GPUs.
if (( ROLLOUT_TP > DETECTED_GPUS )); then
  echo "=== clamping ROLLOUT_TP $ROLLOUT_TP -> $DETECTED_GPUS (single-GPU path) ==="
  export ROLLOUT_TP="$DETECTED_GPUS"
fi
export ROLLOUT_N="${ROLLOUT_N:-8}"
export ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.4}"

# Batch sizes — match baseline.
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-64}"
export PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
export LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}"

# Static batching for trackability: dynamic batching OFF by default => each
# micro-batch is exactly ppo_micro_batch_size_per_gpu=1 sequence with
# deterministic packing (one sequence per forward, easy to follow). Flip
# USE_DYNAMIC_BSZ=True to restore token-balanced dynamic batching (the
# per-element mask is packing-invariant, so both modes are correct).
export USE_DYNAMIC_BSZ="${USE_DYNAMIC_BSZ:-False}"

# Train-inference mismatch DIAGNOSTIC (read-only; does NOT change training).
# calculate_log_probs=True makes vLLM return its rollout log-probs so the trainer
# logs training/rollout_probs_diff_* and rollout_corr/* (vLLM rollout vs the
# train-engine-recomputed old_log_prob). Rollout CORRECTION stays STRICTLY OFF
# (rollout_is/rollout_rs=null, bypass_mode=false): old_log_prob is always
# recomputed by the train engine and vLLM log-probs are never used in the loss.
# NB with comm-eff masking + mask_recompute=true the recompute is masked, so for
# comm-eff runs the diff also reflects masking — read it on the dense control
# (COMM_EFF_ENABLED=false) for the pure train-inference mismatch.
export ROLLOUT_CALC_LOGPROBS="${ROLLOUT_CALC_LOGPROBS:-True}"

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

# ---------------------------------------------------------------------------
# Checkpoint -> Cloudflare R2 on-the-go mirror (EXP-58). Mirrors the naming of the
# WEIGHT_TRAJ_R2_* knobs. When CKPT_R2_ENABLED=false (DEFAULT) the trainer's
# _save_checkpoint is byte-identical to upstream verl (no R2 threads, no r2_sink
# import on the save path) — the whole feature is a strict no-op. Set true to
# mirror every global_step_<N>/ tree (all rank shards + data.pt + huggingface +
# fsdp_config + the root latest_checkpointed_iteration.txt) to R2 and delete each
# local file after a verified upload, so peak local disk stays ~1 in-flight ckpt +
# staging instead of the keep-all total. The CKPT_R2_* async knobs are consumed by
# _save_checkpoint via the env; only CKPT_R2_ENABLED is threaded through Hydra
# (trainer.checkpoint_r2_enabled) since that is the gate. R2 creds/prefix come from
# the SAME env as the weight-traj stream (R2_BUCKET/R2_ENDPOINT|R2_ACCOUNT_ID/
# R2_ACCESS_KEY_ID/R2_SECRET_ACCESS_KEY/R2_EXPERIMENT/R2_REGIME) — the checkpoints
# land under autonomous-harness-rlvr-compression/$R2_EXPERIMENT/$R2_REGIME/checkpoints (a DISTINCT prefix
# from .../weights, so no key collision with the weight-traj snapshots).
export CKPT_R2_ENABLED="${CKPT_R2_ENABLED:-false}"
export CKPT_R2_ASYNC="${CKPT_R2_ASYNC:-true}"
export CKPT_R2_DELETE_LOCAL="${CKPT_R2_DELETE_LOCAL:-true}"
export CKPT_R2_MAX_STAGED_GB="${CKPT_R2_MAX_STAGED_GB:-50}"
export CKPT_R2_WORKERS="${CKPT_R2_WORKERS:-4}"
export CKPT_R2_FLUSH_TIMEOUT="${CKPT_R2_FLUSH_TIMEOUT:-1800}"

# WandB project + experiment.
export PROJECT_NAME="${PROJECT_NAME:-verl_compression_research}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen25_1p5b_grpo_gsm8k_comm_eff_baseline}"

# Token budget per micro-batch for dynamic batching. Actor budget halved to
# 18432 (from 36864) to fit the anchor's ~3 GB clone on 4×H200; log_prob/ref
# keep 36864 because those paths do not allocate the clone.
PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-18432}"
LOG_PROB_MAX_TOKEN_LEN_PER_GPU="${LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-36864}"
REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU="${REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-36864}"

# ---------------------------------------------------------------------------
# 6. Communication-efficient method.
#    Defaults configure the PowerSGD + anchor delayed-EF path while exposing
#    comm-eff knobs as env overrides. Field names mirror the actor config schema;
#    Hydra struct mode rejects unknown keys.
# ---------------------------------------------------------------------------
COMM_EFF_ENABLED="${COMM_EFF_ENABLED:-true}"                          # master switch (false => dense)
# --- codec selector: dense | prf_mask | powersgd ---
# powersgd is the default codec and is the only codec compatible with anchor-owned
# Q. prf_mask is available for comparison runs; dense disables communication
# compression.
COMM_EFF_COMPRESSION_TYPE="${COMM_EFF_COMPRESSION_TYPE:-powersgd}"
# --- activation mask (reference-only codec; OFF in the PowerSGD anchor base) ---
COMM_EFF_MASK_ENABLED="${COMM_EFF_MASK_ENABLED:-false}"
COMM_EFF_MASK_P="${COMM_EFF_MASK_P:-0.9}"                             # masked fraction for prf_mask
COMM_EFF_MASK_RESCALE="${COMM_EFF_MASK_RESCALE:-true}"               # inverted-dropout h*mask/(1-p)
COMM_EFF_MASK_RECOMPUTE="${COMM_EFF_MASK_RECOMPUTE:-true}"            # mask the old_logprob forward too
COMM_EFF_MASK_SEED="${COMM_EFF_MASK_SEED:-0}"                         # PRF base seed
COMM_EFF_MASK_PP_SIZE="${COMM_EFF_MASK_PP_SIZE:-8}"                   # simulated pipeline depth (boundary blocks)
# --- periodic dense step: 0=OFF. Kept only as a diagnostic control. ---
COMM_EFF_CLEAN_CADENCE="${COMM_EFF_CLEAN_CADENCE:-0}"
# --- anchor circuit ---
COMM_EFF_ANCHOR_ENABLED="${COMM_EFF_ANCHOR_ENABLED:-true}"
# Anchor cadence is measured in optimizer ticks, not trainer global steps.
COMM_EFF_ANCHOR_CADENCE="${COMM_EFF_ANCHOR_CADENCE:-5}"
# The anchor forwards from a delay_K-tick-stale weight snapshot.
COMM_EFF_ANCHOR_DELAY_K="${COMM_EFF_ANCHOR_DELAY_K:-5}"
# When true, only the anchor refresh updates the PowerSGD basis Q.
COMM_EFF_ANCHOR_OWNS_Q="${COMM_EFF_ANCHOR_OWNS_Q:-true}"
# Paired replay uses the same batch/weights the fast circuit saw so the
# correction tracks codec error rather than batch mismatch.
COMM_EFF_ANCHOR_REPLAY_PAIRED_BATCH="${COMM_EFF_ANCHOR_REPLAY_PAIRED_BATCH:-true}"
COMM_EFF_ANCHOR_SNAPSHOT_DEVICE="${COMM_EFF_ANCHOR_SNAPSHOT_DEVICE:-cpu}"
# Opt-in weight projection. Defaults keep the existing path strictly disabled.
COMM_EFF_ANCHOR_LOOKAHEAD_ANCHOR="${COMM_EFF_ANCHOR_LOOKAHEAD_ANCHOR:-false}"
COMM_EFF_ANCHOR_LOOKAHEAD_MODE="${COMM_EFF_ANCHOR_LOOKAHEAD_MODE:-disabled}"
COMM_EFF_ANCHOR_LOOKAHEAD_STRENGTH="${COMM_EFF_ANCHOR_LOOKAHEAD_STRENGTH:-1.0}"
COMM_EFF_ANCHOR_LOOKAHEAD_ROLLOUT_SOURCE="${COMM_EFF_ANCHOR_LOOKAHEAD_ROLLOUT_SOURCE:-auto}"
COMM_EFF_ANCHOR_LOOKAHEAD_WINDOW_SNAPSHOTS="${COMM_EFF_ANCHOR_LOOKAHEAD_WINDOW_SNAPSHOTS:-4}"
# Warmup behavior before the look-ahead projector is ready. "stale_correct"
# (default) = today's byte-identical behavior; "no_correct" = skip the anchor
# pass + M update while warming (requires the projector on AND owns_q=false);
# "q_only" = rank1-only forward/no-backward Q refresh with M/correction off.
COMM_EFF_ANCHOR_WARMUP_MODE="${COMM_EFF_ANCHOR_WARMUP_MODE:-stale_correct}"
# Min ring snapshots before the projector engages. -1 (default) = mode source
# count; 2 = project from the earliest legal fire (fire 2).
COMM_EFF_ANCHOR_LOOKAHEAD_MIN_SNAPSHOTS="${COMM_EFF_ANCHOR_LOOKAHEAD_MIN_SNAPSHOTS:--1}"
# --- anchor-guided gradient correction / merger ---
COMM_EFF_SPECTRAL_ENABLED="${COMM_EFF_SPECTRAL_ENABLED:-true}"
COMM_EFF_SPECTRAL_TARGET_SCOPE="${COMM_EFF_SPECTRAL_TARGET_SCOPE:-decoder_matrices}"
COMM_EFF_SPECTRAL_DIAGNOSTICS="${COMM_EFF_SPECTRAL_DIAGNOSTICS:-true}"
COMM_EFF_SPECTRAL_BETA_ANC="${COMM_EFF_SPECTRAL_BETA_ANC:-0.50}"       # anchor-gradient EMA decay (signed_ema baseline)
COMM_EFF_SPECTRAL_SIGNED_EMA_ALPHA="${COMM_EFF_SPECTRAL_SIGNED_EMA_ALPHA:-0.25}"   # signed_ema mixing weight
# Correction cadence in optimizer ticks.
COMM_EFF_SPECTRAL_CADENCE="${COMM_EFF_SPECTRAL_CADENCE:-1}"
COMM_EFF_SPECTRAL_EMA_DEVICE="${COMM_EFF_SPECTRAL_EMA_DEVICE:-cpu}"    # offload full-coverage M (OOM guard)
# Cap target matrices per correction. -1 means no cap.
COMM_EFF_SPECTRAL_MAX_TARGETS="${COMM_EFF_SPECTRAL_MAX_TARGETS:--1}"
# signed_ema folds the anchor M into the fast gradient via a signed EMA.
COMM_EFF_SPECTRAL_CORRECTION_MODE="${COMM_EFF_SPECTRAL_CORRECTION_MODE:-signed_ema}"
# Legacy dose knob, retained for the plain-PowerSGD limiting case (lambda=0.0); unused by signed_ema.
COMM_EFF_SPECTRAL_DELAYED_EF_LAMBDA="${COMM_EFF_SPECTRAL_DELAYED_EF_LAMBDA:-1.0}"
COMM_EFF_SPECTRAL_INJECT_GAMMA="${COMM_EFF_SPECTRAL_INJECT_GAMMA:-1.0}"             # force when correction_mode=inject
COMM_EFF_SPECTRAL_BLEND_ETA="${COMM_EFF_SPECTRAL_BLEND_ETA:-0.5}"                   # weight when correction_mode=blend
# --- ef_powersgd merger ---
COMM_EFF_SPECTRAL_EF_DECAY="${COMM_EFF_SPECTRAL_EF_DECAY:-0.0}"
COMM_EFF_SPECTRAL_EF_CLIP="${COMM_EFF_SPECTRAL_EF_CLIP:-0.0}"
# --- optional additive stale-anchor sub-basis ---
COMM_EFF_SPECTRAL_DELTA_SUBBASIS_RANK="${COMM_EFF_SPECTRAL_DELTA_SUBBASIS_RANK:-0}"
COMM_EFF_SPECTRAL_DELTA_SUBBASIS_FAMILY="${COMM_EFF_SPECTRAL_DELTA_SUBBASIS_FAMILY:-tail}"  # tail | grad
# Sub-basis weight schedule. With decay_steps=0 the weight is constant.
COMM_EFF_SPECTRAL_DELTA_SUBBASIS_WEIGHT="${COMM_EFF_SPECTRAL_DELTA_SUBBASIS_WEIGHT:-1.0}"
COMM_EFF_SPECTRAL_DELTA_SUBBASIS_DECAY_STEPS="${COMM_EFF_SPECTRAL_DELTA_SUBBASIS_DECAY_STEPS:-0}"
COMM_EFF_SPECTRAL_DELTA_SUBBASIS_HOLD_STEPS="${COMM_EFF_SPECTRAL_DELTA_SUBBASIS_HOLD_STEPS:-0}"
# --- optional zero-mean perturbation after correction ---
COMM_EFF_SPECTRAL_PERTURB_SIGMA="${COMM_EFF_SPECTRAL_PERTURB_SIGMA:-0.0}"
COMM_EFF_SPECTRAL_PERTURB_SEED="${COMM_EFF_SPECTRAL_PERTURB_SEED:-0}"
# --- optional correction momentum ---
COMM_EFF_SPECTRAL_DELTA_MOMENTUM_MU="${COMM_EFF_SPECTRAL_DELTA_MOMENTUM_MU:-0.0}"
COMM_EFF_SPECTRAL_DELTA_MOMENTUM_AGE_DECAY="${COMM_EFF_SPECTRAL_DELTA_MOMENTUM_AGE_DECAY:-false}"  # true|false
# --- optional adaptive delayed-EF dose ---
COMM_EFF_SPECTRAL_ADAPTIVE_LAMBDA_MODE="${COMM_EFF_SPECTRAL_ADAPTIVE_LAMBDA_MODE:-off}"  # off|cos|ratio
COMM_EFF_SPECTRAL_ADAPTIVE_LAMBDA_KAPPA="${COMM_EFF_SPECTRAL_ADAPTIVE_LAMBDA_KAPPA:-0.0}"
COMM_EFF_SPECTRAL_LAMBDA_CAP="${COMM_EFF_SPECTRAL_LAMBDA_CAP:-2.0}"
# --- optional correction-delta compression ---
COMM_EFF_SPECTRAL_R_DELTA="${COMM_EFF_SPECTRAL_R_DELTA:-0}"
# --- Q-basis family ---
COMM_EFF_POWERSGD_Q_BASIS="${COMM_EFF_POWERSGD_Q_BASIS:-act}"
# Passive families are accumulated inside the anchor pass without affecting
# the live fast path or optimizer.
COMM_EFF_POWERSGD_Q_BASIS_PASSIVE="${COMM_EFF_POWERSGD_Q_BASIS_PASSIVE:-[]}"
# Hybrid split at fixed rank r; -1/-1 lets the implementation choose.
COMM_EFF_POWERSGD_HYBRID_ACT_COLS="${COMM_EFF_POWERSGD_HYBRID_ACT_COLS:--1}"
COMM_EFF_POWERSGD_HYBRID_GRAD_COLS="${COMM_EFF_POWERSGD_HYBRID_GRAD_COLS:--1}"
# --- optional tensor capture probes ---
COMM_EFF_CAPTURE_ENABLED="${COMM_EFF_CAPTURE_ENABLED:-false}"
COMM_EFF_CAPTURE_DIR="${COMM_EFF_CAPTURE_DIR:-/workspace/captures}"   # rsynced to runs//captures/
COMM_EFF_CAPTURE_MAX_TICKS="${COMM_EFF_CAPTURE_MAX_TICKS:-10}"        # audit needs ~5-10 ticks
COMM_EFF_CAPTURE_STRATIFIED="${COMM_EFF_CAPTURE_STRATIFIED:-0}"       # >0 => N targets/matrix-type (volume guard)
COMM_EFF_CAPTURE_G_DENSE="${COMM_EFF_CAPTURE_G_DENSE:-false}"         # parallel uncompressed G_dense backward (highest-OOM-risk probe)
COMM_EFF_CAPTURE_FRESH_ANCHOR="${COMM_EFF_CAPTURE_FRESH_ANCHOR:-false}"  # delay_K=0 fresh-anchor measurement probe (the Option-A dense reference)
# Loss for the delay_K=0 fresh-anchor probe. Dump-only; never optimizer input.
COMM_EFF_CAPTURE_FRESH_ANCHOR_LOSS="${COMM_EFF_CAPTURE_FRESH_ANCHOR_LOSS:-clean_pg}"
COMM_EFF_CAPTURE_DUMP_DTYPE="${COMM_EFF_CAPTURE_DUMP_DTYPE:-fp32}"    # fp32 REQUIRED for the fidelity invariant
# min_tick skips cold-Q ticks while preserving the max_ticks capture budget.
COMM_EFF_CAPTURE_MIN_TICK="${COMM_EFF_CAPTURE_MIN_TICK:-0}"
COMM_EFF_CAPTURE_RANK0_ONLY="${COMM_EFF_CAPTURE_RANK0_ONLY:-true}"    # capture rank0 only (disk guard); default true
# --- causal sampled-weight verification for rank1_relex ---
COMM_EFF_RANK1_PROJECTION_PROBE_ENABLED="${COMM_EFF_RANK1_PROJECTION_PROBE_ENABLED:-false}"
COMM_EFF_RANK1_PROJECTION_PROBE_SAMPLES="${COMM_EFF_RANK1_PROJECTION_PROBE_SAMPLES:-16}"
COMM_EFF_PROBE_OUT_DIR="${COMM_EFF_PROBE_OUT_DIR:-}"
COMM_EFF_PROBE_RANK0_ONLY="${COMM_EFF_PROBE_RANK0_ONLY:-true}"
# --- PowerSGD activation compression ---
COMM_EFF_POWERSGD_RANK="${COMM_EFF_POWERSGD_RANK:-77}"               # r=77 ≡ p=0.95 (0.05·H, H=1536)
COMM_EFF_POWERSGD_SEED="${COMM_EFF_POWERSGD_SEED:-0}"                 # per-layer basis seed base
COMM_EFF_POWERSGD_PP_SIZE="${COMM_EFF_POWERSGD_PP_SIZE:-8}"           # boundary blocks (same as mask)
COMM_EFF_POWERSGD_UPDATE_CADENCE="${COMM_EFF_POWERSGD_UPDATE_CADENCE:-1}"  # orth(V) every N steps
COMM_EFF_POWERSGD_WARM_START="${COMM_EFF_POWERSGD_WARM_START:-true}"  # carry Q across steps
COMM_EFF_POWERSGD_COMPRESS_RECOMPUTE="${COMM_EFF_POWERSGD_COMPRESS_RECOMPUTE:-true}"  # project old-logprob too
COMM_EFF_POWERSGD_SYNC_BASIS="${COMM_EFF_POWERSGD_SYNC_BASIS:-true}"  # all-reduce V across DP => single shared consensus Q (REQUIRED under DP)
COMM_EFF_POWERSGD_FAST_Q_BOOTSTRAP="${COMM_EFF_POWERSGD_FAST_Q_BOOTSTRAP:-false}"  # one discarded dense observation before the first compressed old-logprob
COMM_EFF_POWERSGD_QR_DTYPE="${COMM_EFF_POWERSGD_QR_DTYPE:-fp32}"      # fp32 required for stable orthogonalization
COMM_EFF_POWERSGD_REORTHO_EPS="${COMM_EFF_POWERSGD_REORTHO_EPS:-1e-6}"

# ---- vLLM all-reduce: default to NCCL (disable_custom_all_reduce=true) ---------
# WHY true by default: (1) some Vast H100/H200 boxes crash in vLLM's custom
# all-reduce (CUDA-IPC under the mp executor) at KV-cache init; NCCL avoids it.
# (2) It is greedy-val-neutral, so we hold it TRUE as a controlled var across ALL
# arms — every run has held it TRUE — keeping val@50 comparisons apples-to-
# apples. Override DISABLE_CUSTOM_ALL_REDUCE=false only on a box that does not crash.
DISABLE_CUSTOM_ALL_REDUCE="${DISABLE_CUSTOM_ALL_REDUCE:-true}"
VLLM_ALLREDUCE_OVERRIDE=()
if [[ "${DISABLE_CUSTOM_ALL_REDUCE}" == "true" ]]; then
  VLLM_ALLREDUCE_OVERRIDE+=("+actor_rollout_ref.rollout.engine_kwargs.vllm.disable_custom_all_reduce=true")
fi

if [[ "${COMM_EFF_ANCHOR_ENABLED}" == "true" ]]; then
  echo "INFO: anchor ON (the base) -> ~3 GB no-hook clone/rank; the default PPO_MAX_TOKEN_LEN_PER_GPU=18432 fits 4×H200. If you raise it, prefer 8×GPU." >&2
fi

LOG="${LOG:-$VERL_ROOT/runs/${EXPERIMENT_NAME}/train.log}"
mkdir -p "$(dirname "$LOG")"
COMM_EFF_PROBE_OUT_DIR="${COMM_EFF_PROBE_OUT_DIR:-$(dirname "$LOG")/rank1_projection_probe}"

cat <<EOF
=== launching communication-efficient GRPO ===
  model:               $MODEL_PATH
  GPUs:                $NGPUS_PER_NODE
  rollout TP × N:      ${ROLLOUT_TP} × ${ROLLOUT_N}
  vLLM mem util:       $ROLLOUT_GPU_MEM_UTIL
  train batch:         $TRAIN_BATCH_SIZE prompts (× $ROLLOUT_N = $(( TRAIN_BATCH_SIZE * ROLLOUT_N )) seqs/step)
  ppo mini batch:      $PPO_MINI_BATCH_SIZE
  batching:            dynamic_bsz=$USE_DYNAMIC_BSZ  (when False: micro_batch_per_gpu=$PPO_MICRO_BATCH_SIZE_PER_GPU, max_tokens/GPU=$PPO_MAX_TOKEN_LEN_PER_GPU ignored)
  prompt / response:   $MAX_PROMPT_LENGTH / $MAX_RESPONSE_LENGTH
  epochs:              $TOTAL_EPOCHS  (save $SAVE_FREQ, validate $TEST_FREQ, total steps $TOTAL_TRAINING_STEPS)
  val_before_train:    $VAL_BEFORE_TRAIN
  objective:           pg_loss only (use_kl_loss=$USE_KL_LOSS, use_kl_in_reward=$USE_KL_IN_REWARD, entropy_coeff=$ENTROPY_COEFF)
  mismatch diag:       calculate_log_probs=$ROLLOUT_CALC_LOGPROBS (logs training/rollout_probs_diff_*); rollout correction STRICTLY OFF (recompute old_log_prob)
  comm_eff master:     $COMM_EFF_ENABLED
  compression_type:    $COMM_EFF_COMPRESSION_TYPE  (dense|prf_mask|powersgd; dense can fall back to mask.enabled)
  mask:                enabled=$COMM_EFF_MASK_ENABLED p=$COMM_EFF_MASK_P rescale=$COMM_EFF_MASK_RESCALE recompute=$COMM_EFF_MASK_RECOMPUTE seed=$COMM_EFF_MASK_SEED pp_size=$COMM_EFF_MASK_PP_SIZE
  powersgd:            rank=$COMM_EFF_POWERSGD_RANK update_cadence=$COMM_EFF_POWERSGD_UPDATE_CADENCE warm_start=$COMM_EFF_POWERSGD_WARM_START compress_recompute=$COMM_EFF_POWERSGD_COMPRESS_RECOMPUTE sync_basis=$COMM_EFF_POWERSGD_SYNC_BASIS fast_q_bootstrap=$COMM_EFF_POWERSGD_FAST_Q_BOOTSTRAP qr_dtype=$COMM_EFF_POWERSGD_QR_DTYPE  (active iff compression_type=powersgd)
  clean_cadence:       $COMM_EFF_CLEAN_CADENCE  (0=off)
  anchor:              enabled=$COMM_EFF_ANCHOR_ENABLED cadence=$COMM_EFF_ANCHOR_CADENCE delay_K=$COMM_EFF_ANCHOR_DELAY_K owns_q=$COMM_EFF_ANCHOR_OWNS_Q replay_paired_batch=$COMM_EFF_ANCHOR_REPLAY_PAIRED_BATCH snapshot_device=$COMM_EFF_ANCHOR_SNAPSHOT_DEVICE
  lookahead:           enabled=$COMM_EFF_ANCHOR_LOOKAHEAD_ANCHOR mode=$COMM_EFF_ANCHOR_LOOKAHEAD_MODE strength=$COMM_EFF_ANCHOR_LOOKAHEAD_STRENGTH rollout_source=$COMM_EFF_ANCHOR_LOOKAHEAD_ROLLOUT_SOURCE window=$COMM_EFF_ANCHOR_LOOKAHEAD_WINDOW_SNAPSHOTS warmup=$COMM_EFF_ANCHOR_WARMUP_MODE min_snapshots=$COMM_EFF_ANCHOR_LOOKAHEAD_MIN_SNAPSHOTS
  spectral:            enabled=$COMM_EFF_SPECTRAL_ENABLED target_scope=$COMM_EFF_SPECTRAL_TARGET_SCOPE diagnostics=$COMM_EFF_SPECTRAL_DIAGNOSTICS beta_anc=$COMM_EFF_SPECTRAL_BETA_ANC cadence=$COMM_EFF_SPECTRAL_CADENCE max_targets=$COMM_EFF_SPECTRAL_MAX_TARGETS ema_device=$COMM_EFF_SPECTRAL_EMA_DEVICE
  spectral correction: mode=$COMM_EFF_SPECTRAL_CORRECTION_MODE signed_ema_alpha=$COMM_EFF_SPECTRAL_SIGNED_EMA_ALPHA beta_anc=$COMM_EFF_SPECTRAL_BETA_ANC (legacy: delayed_ef_lambda=$COMM_EFF_SPECTRAL_DELAYED_EF_LAMBDA inject_gamma=$COMM_EFF_SPECTRAL_INJECT_GAMMA blend_eta=$COMM_EFF_SPECTRAL_BLEND_ETA)
  ef_powersgd:         ef_decay=$COMM_EFF_SPECTRAL_EF_DECAY ef_clip=$COMM_EFF_SPECTRAL_EF_CLIP
  subbasis:            delta_subbasis_rank=$COMM_EFF_SPECTRAL_DELTA_SUBBASIS_RANK family=$COMM_EFF_SPECTRAL_DELTA_SUBBASIS_FAMILY r_delta=$COMM_EFF_SPECTRAL_R_DELTA
  subbasis schedule:   delta_subbasis_weight=$COMM_EFF_SPECTRAL_DELTA_SUBBASIS_WEIGHT decay_steps=$COMM_EFF_SPECTRAL_DELTA_SUBBASIS_DECAY_STEPS hold_steps=$COMM_EFF_SPECTRAL_DELTA_SUBBASIS_HOLD_STEPS
  perturb:             perturb_sigma=$COMM_EFF_SPECTRAL_PERTURB_SIGMA perturb_seed=$COMM_EFF_SPECTRAL_PERTURB_SEED
  delta_momentum:      delta_momentum_mu=$COMM_EFF_SPECTRAL_DELTA_MOMENTUM_MU age_decay=$COMM_EFF_SPECTRAL_DELTA_MOMENTUM_AGE_DECAY
  adaptive_lambda:     adaptive_lambda_mode=$COMM_EFF_SPECTRAL_ADAPTIVE_LAMBDA_MODE kappa=$COMM_EFF_SPECTRAL_ADAPTIVE_LAMBDA_KAPPA lambda_cap=$COMM_EFF_SPECTRAL_LAMBDA_CAP
  q_basis:             live=$COMM_EFF_POWERSGD_Q_BASIS passive=$COMM_EFF_POWERSGD_Q_BASIS_PASSIVE hybrid=($COMM_EFF_POWERSGD_HYBRID_ACT_COLS+$COMM_EFF_POWERSGD_HYBRID_GRAD_COLS)
  capture:             enabled=$COMM_EFF_CAPTURE_ENABLED dir=$COMM_EFF_CAPTURE_DIR max_ticks=$COMM_EFF_CAPTURE_MAX_TICKS min_tick=$COMM_EFF_CAPTURE_MIN_TICK stratified=$COMM_EFF_CAPTURE_STRATIFIED rank0_only=$COMM_EFF_CAPTURE_RANK0_ONLY g_dense=$COMM_EFF_CAPTURE_G_DENSE fresh_anchor=$COMM_EFF_CAPTURE_FRESH_ANCHOR fresh_anchor_loss=$COMM_EFF_CAPTURE_FRESH_ANCHOR_LOSS dump_dtype=$COMM_EFF_CAPTURE_DUMP_DTYPE
  wandb:               $PROJECT_NAME / $EXPERIMENT_NAME
  log:                 $LOG
=== launching ===
EOF

# ---------------------------------------------------------------------------
# 6b.  early-stop instrumentation (greppable). A lightweight background
#     watcher tails the LIVE training log for the corrupting-failure patterns
#     the training-log-monitor kills a cell on (non-finite grad_norm/loss, FSDP
#     backward/hook errors, DTensor/aten::copy_ writeback errors, the
#     string-metric reduce crash, use_orig_params guard, spectral crashes). On
#     the FIRST match it writes a one-line `EARLY_STOP_SIGNAL: <pattern> @ <line>`
#     to the log AND a `runs/<experiment>/EARLY_STOP_SIGNAL` sentinel file, then exits.
#     This is a SIGNAL ONLY — it does NOT kill training (the monitor/runner owns
#     teardown). It just gives the monitor a single high-signal grep target so it
#     does not have to re-derive the regex from the rescue-trigger list. Strict
#     opt-in side effect: writes only into this run's own dir.
#
#      fix (CRITICAL — a clean run used to hang here forever): the old
#     watcher ran `tail -F | grep -m1` in a backgrounded subshell and the EXIT
#     trap killed only the SUBSHELL pid, orphaning the child `tail -F`. On a
#     CLEAN run grep -m1 never matches, `tail -F` follows the (now-idle) log
#     forever, the subshell never exits, and THIS SCRIPT blocks in its implicit
#     `wait` at end-of-script — so the back-to-back sequence could never finish
#     autonomously (GPUs idle at $/hr). Two independent guards now prevent that:
#       (1) `tail --pid="$TRAIN_PID" -F` — the follower DIES when the training
#           process exits (clean or crash), closing the pipe so grep hits EOF and
#           the subshell returns; nothing is left to wait on.
#       (2) the watcher runs in its OWN process group (setsid); one cleanup
#           function verifies that private PGID before signalling it, then reaps
#           the launcher and removes the EXIT trap before the parent can return.
#     Net: the watcher never leaves a dangling follower and never blocks this
#     script on clean completion — for EVERY cell in the sequence.
# ---------------------------------------------------------------------------
RUN_DIR="$(dirname "$LOG")"
EARLY_STOP_SENTINEL="$RUN_DIR/EARLY_STOP_SIGNAL"
rm -f "$EARLY_STOP_SENTINEL"
# Early-stop patterns catch numeric instability and FSDP/spectral safety errors.
# \bnan/inf word-boundary guards avoid matching "infer"/"information". The
# watcher is a no-op until $LOG starts filling.
EARLY_STOP_RE='([Nn]a[Nn] detected|RuntimeError: .*use_orig_params|summon_full_params.*(error|Error|assert)|could not convert string to float|aten::copy_.*(mismatch|size)|torch\.distributed\.fsdp.*(error|Error)|(loss|grad_norm|pg_loss|policy_loss|reward)[^A-Za-z].{0,80}\b([Nn]a[Nn]|[Ii]nf)\b|\b([Nn]a[Nn]|[Ii]nf)\b.{0,40}(loss|grad_norm))'

# ---------------------------------------------------------------------------
# 7. Launch — reuse upstream's per-recipe script for the verbatim main_ppo
#    invocation, overriding the OOM-relevant + comm-eff Hydra knobs. Every
#    enabled flag comes from env so the full ablation grid is a one-liner.
#
#    : launch the training in the BACKGROUND, capture its PID, start the
#    early-stop watcher bound to that PID, then `wait` on training explicitly.
#    The watcher self-terminates when training exits (guard 1), and the verified
#    cleanup path reaps its private process group exactly once (guard 2).
# ---------------------------------------------------------------------------
bash examples/grpo_trainer/run_qwen3_4b_fsdp.sh \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$PPO_MAX_TOKEN_LEN_PER_GPU" \
  actor_rollout_ref.actor.use_dynamic_bsz="$USE_DYNAMIC_BSZ" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$PPO_MICRO_BATCH_SIZE_PER_GPU" \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="$LOG_PROB_MAX_TOKEN_LEN_PER_GPU" \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz="$USE_DYNAMIC_BSZ" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$LOG_PROB_MICRO_BATCH_SIZE_PER_GPU" \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="$REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU" \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz="$USE_DYNAMIC_BSZ" \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$LOG_PROB_MICRO_BATCH_SIZE_PER_GPU" \
  actor_rollout_ref.rollout.calculate_log_probs="$ROLLOUT_CALC_LOGPROBS" \
  algorithm.rollout_correction.rollout_is=null \
  algorithm.rollout_correction.rollout_rs=null \
  algorithm.rollout_correction.bypass_mode=false \
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
  trainer.checkpoint_r2_enabled="$CKPT_R2_ENABLED" \
  actor_rollout_ref.actor.comm_eff.enabled="$COMM_EFF_ENABLED" \
  actor_rollout_ref.actor.comm_eff.compression_type="$COMM_EFF_COMPRESSION_TYPE" \
  actor_rollout_ref.actor.comm_eff.clean_cadence="$COMM_EFF_CLEAN_CADENCE" \
  actor_rollout_ref.actor.comm_eff.mask.enabled="$COMM_EFF_MASK_ENABLED" \
  actor_rollout_ref.actor.comm_eff.mask.p="$COMM_EFF_MASK_P" \
  actor_rollout_ref.actor.comm_eff.mask.rescale="$COMM_EFF_MASK_RESCALE" \
  actor_rollout_ref.actor.comm_eff.mask.mask_recompute="$COMM_EFF_MASK_RECOMPUTE" \
  actor_rollout_ref.actor.comm_eff.mask.seed="$COMM_EFF_MASK_SEED" \
  actor_rollout_ref.actor.comm_eff.mask.pp_size="$COMM_EFF_MASK_PP_SIZE" \
  actor_rollout_ref.actor.comm_eff.anchor.enabled="$COMM_EFF_ANCHOR_ENABLED" \
  actor_rollout_ref.actor.comm_eff.anchor.cadence="$COMM_EFF_ANCHOR_CADENCE" \
  actor_rollout_ref.actor.comm_eff.anchor.delay_K="$COMM_EFF_ANCHOR_DELAY_K" \
  actor_rollout_ref.actor.comm_eff.anchor.owns_q="$COMM_EFF_ANCHOR_OWNS_Q" \
  actor_rollout_ref.actor.comm_eff.anchor.replay_paired_batch="$COMM_EFF_ANCHOR_REPLAY_PAIRED_BATCH" \
  actor_rollout_ref.actor.comm_eff.anchor.snapshot_device="$COMM_EFF_ANCHOR_SNAPSHOT_DEVICE" \
  actor_rollout_ref.actor.comm_eff.anchor.lookahead_anchor="$COMM_EFF_ANCHOR_LOOKAHEAD_ANCHOR" \
  actor_rollout_ref.actor.comm_eff.anchor.lookahead_mode="$COMM_EFF_ANCHOR_LOOKAHEAD_MODE" \
  actor_rollout_ref.actor.comm_eff.anchor.lookahead_strength="$COMM_EFF_ANCHOR_LOOKAHEAD_STRENGTH" \
  actor_rollout_ref.actor.comm_eff.anchor.lookahead_rollout_source="$COMM_EFF_ANCHOR_LOOKAHEAD_ROLLOUT_SOURCE" \
  actor_rollout_ref.actor.comm_eff.anchor.lookahead_window_snapshots="$COMM_EFF_ANCHOR_LOOKAHEAD_WINDOW_SNAPSHOTS" \
  actor_rollout_ref.actor.comm_eff.anchor.warmup_mode="$COMM_EFF_ANCHOR_WARMUP_MODE" \
  actor_rollout_ref.actor.comm_eff.anchor.lookahead_min_snapshots="$COMM_EFF_ANCHOR_LOOKAHEAD_MIN_SNAPSHOTS" \
  actor_rollout_ref.actor.comm_eff.spectral.enabled="$COMM_EFF_SPECTRAL_ENABLED" \
  actor_rollout_ref.actor.comm_eff.spectral.target_scope="$COMM_EFF_SPECTRAL_TARGET_SCOPE" \
  actor_rollout_ref.actor.comm_eff.spectral.diagnostics="$COMM_EFF_SPECTRAL_DIAGNOSTICS" \
  actor_rollout_ref.actor.comm_eff.spectral.beta_anc="$COMM_EFF_SPECTRAL_BETA_ANC" \
  actor_rollout_ref.actor.comm_eff.spectral.ema_device="$COMM_EFF_SPECTRAL_EMA_DEVICE" \
  actor_rollout_ref.actor.comm_eff.spectral.max_targets="$COMM_EFF_SPECTRAL_MAX_TARGETS" \
  actor_rollout_ref.actor.comm_eff.spectral.cadence="$COMM_EFF_SPECTRAL_CADENCE" \
  actor_rollout_ref.actor.comm_eff.spectral.correction_mode="$COMM_EFF_SPECTRAL_CORRECTION_MODE" \
  actor_rollout_ref.actor.comm_eff.spectral.signed_ema_alpha="$COMM_EFF_SPECTRAL_SIGNED_EMA_ALPHA" \
  actor_rollout_ref.actor.comm_eff.spectral.delayed_ef_lambda="$COMM_EFF_SPECTRAL_DELAYED_EF_LAMBDA" \
  actor_rollout_ref.actor.comm_eff.spectral.inject_gamma="$COMM_EFF_SPECTRAL_INJECT_GAMMA" \
  actor_rollout_ref.actor.comm_eff.spectral.blend_eta="$COMM_EFF_SPECTRAL_BLEND_ETA" \
  actor_rollout_ref.actor.comm_eff.powersgd.rank="$COMM_EFF_POWERSGD_RANK" \
  actor_rollout_ref.actor.comm_eff.powersgd.seed="$COMM_EFF_POWERSGD_SEED" \
  actor_rollout_ref.actor.comm_eff.powersgd.pp_size="$COMM_EFF_POWERSGD_PP_SIZE" \
  actor_rollout_ref.actor.comm_eff.powersgd.update_cadence="$COMM_EFF_POWERSGD_UPDATE_CADENCE" \
  actor_rollout_ref.actor.comm_eff.powersgd.warm_start="$COMM_EFF_POWERSGD_WARM_START" \
  actor_rollout_ref.actor.comm_eff.powersgd.compress_recompute="$COMM_EFF_POWERSGD_COMPRESS_RECOMPUTE" \
  actor_rollout_ref.actor.comm_eff.powersgd.sync_basis="$COMM_EFF_POWERSGD_SYNC_BASIS" \
  actor_rollout_ref.actor.comm_eff.powersgd.fast_q_bootstrap="$COMM_EFF_POWERSGD_FAST_Q_BOOTSTRAP" \
  actor_rollout_ref.actor.comm_eff.powersgd.qr_dtype="$COMM_EFF_POWERSGD_QR_DTYPE" \
  actor_rollout_ref.actor.comm_eff.powersgd.reortho_eps="$COMM_EFF_POWERSGD_REORTHO_EPS" \
  actor_rollout_ref.actor.comm_eff.powersgd.q_basis="$COMM_EFF_POWERSGD_Q_BASIS" \
  actor_rollout_ref.actor.comm_eff.powersgd.q_basis_passive="$COMM_EFF_POWERSGD_Q_BASIS_PASSIVE" \
  actor_rollout_ref.actor.comm_eff.powersgd.hybrid_act_cols="$COMM_EFF_POWERSGD_HYBRID_ACT_COLS" \
  actor_rollout_ref.actor.comm_eff.powersgd.hybrid_grad_cols="$COMM_EFF_POWERSGD_HYBRID_GRAD_COLS" \
  actor_rollout_ref.actor.comm_eff.spectral.ef_decay="$COMM_EFF_SPECTRAL_EF_DECAY" \
  actor_rollout_ref.actor.comm_eff.spectral.ef_clip="$COMM_EFF_SPECTRAL_EF_CLIP" \
  actor_rollout_ref.actor.comm_eff.spectral.delta_subbasis_rank="$COMM_EFF_SPECTRAL_DELTA_SUBBASIS_RANK" \
  actor_rollout_ref.actor.comm_eff.spectral.delta_subbasis_family="$COMM_EFF_SPECTRAL_DELTA_SUBBASIS_FAMILY" \
  actor_rollout_ref.actor.comm_eff.spectral.delta_subbasis_weight="$COMM_EFF_SPECTRAL_DELTA_SUBBASIS_WEIGHT" \
  actor_rollout_ref.actor.comm_eff.spectral.delta_subbasis_decay_steps="$COMM_EFF_SPECTRAL_DELTA_SUBBASIS_DECAY_STEPS" \
  actor_rollout_ref.actor.comm_eff.spectral.delta_subbasis_hold_steps="$COMM_EFF_SPECTRAL_DELTA_SUBBASIS_HOLD_STEPS" \
  actor_rollout_ref.actor.comm_eff.spectral.r_delta="$COMM_EFF_SPECTRAL_R_DELTA" \
  actor_rollout_ref.actor.comm_eff.spectral.perturb_sigma="$COMM_EFF_SPECTRAL_PERTURB_SIGMA" \
  actor_rollout_ref.actor.comm_eff.spectral.perturb_seed="$COMM_EFF_SPECTRAL_PERTURB_SEED" \
  actor_rollout_ref.actor.comm_eff.spectral.delta_momentum_mu="$COMM_EFF_SPECTRAL_DELTA_MOMENTUM_MU" \
  actor_rollout_ref.actor.comm_eff.spectral.delta_momentum_age_decay="$COMM_EFF_SPECTRAL_DELTA_MOMENTUM_AGE_DECAY" \
  actor_rollout_ref.actor.comm_eff.spectral.adaptive_lambda_mode="$COMM_EFF_SPECTRAL_ADAPTIVE_LAMBDA_MODE" \
  actor_rollout_ref.actor.comm_eff.spectral.adaptive_lambda_kappa="$COMM_EFF_SPECTRAL_ADAPTIVE_LAMBDA_KAPPA" \
  actor_rollout_ref.actor.comm_eff.spectral.lambda_cap="$COMM_EFF_SPECTRAL_LAMBDA_CAP" \
  actor_rollout_ref.actor.comm_eff.capture.enabled="$COMM_EFF_CAPTURE_ENABLED" \
  actor_rollout_ref.actor.comm_eff.capture.capture_dir="$COMM_EFF_CAPTURE_DIR" \
  actor_rollout_ref.actor.comm_eff.capture.max_ticks="$COMM_EFF_CAPTURE_MAX_TICKS" \
  actor_rollout_ref.actor.comm_eff.capture.stratified_targets="$COMM_EFF_CAPTURE_STRATIFIED" \
  actor_rollout_ref.actor.comm_eff.capture.capture_g_dense="$COMM_EFF_CAPTURE_G_DENSE" \
  actor_rollout_ref.actor.comm_eff.capture.capture_fresh_anchor="$COMM_EFF_CAPTURE_FRESH_ANCHOR" \
  actor_rollout_ref.actor.comm_eff.capture.fresh_anchor_loss="$COMM_EFF_CAPTURE_FRESH_ANCHOR_LOSS" \
  actor_rollout_ref.actor.comm_eff.capture.dump_dtype="$COMM_EFF_CAPTURE_DUMP_DTYPE" \
  actor_rollout_ref.actor.comm_eff.capture.min_tick="$COMM_EFF_CAPTURE_MIN_TICK" \
  actor_rollout_ref.actor.comm_eff.capture.rank0_only="$COMM_EFF_CAPTURE_RANK0_ONLY" \
  actor_rollout_ref.actor.comm_eff.probe.rank1_projection_enabled="$COMM_EFF_RANK1_PROJECTION_PROBE_ENABLED" \
  actor_rollout_ref.actor.comm_eff.probe.rank1_projection_samples="$COMM_EFF_RANK1_PROJECTION_PROBE_SAMPLES" \
  actor_rollout_ref.actor.comm_eff.probe.out_dir="$COMM_EFF_PROBE_OUT_DIR" \
  actor_rollout_ref.actor.comm_eff.probe.rank0_only="$COMM_EFF_PROBE_RANK0_ONLY" \
  "${VLLM_ALLREDUCE_OVERRIDE[@]+"${VLLM_ALLREDUCE_OVERRIDE[@]}"}" \
  "$@" \
  > "$LOG" 2>&1 &
TRAIN_PID=$!

#  early-stop watcher — bound to TRAIN_PID + its own process group.
# Guard 1: `tail --pid="$TRAIN_PID" -F` dies when training exits (clean or
# crash), so grep hits EOF and the watcher subshell returns. Guard 2: setsid
# puts the watcher in its own pgroup. Cleanup verifies that the recorded group
# is private before signalling it, reaps the launcher exactly once, and clears
# the EXIT trap before final W&B sync. The watcher only SIGNALS (sentinel + log
# line); the monitor/runner owns teardown.
setsid bash -c '
  LOG="$1"; RE="$2"; EXP="$3"; SENT="$4"; TPID="$5"
  for _ in $(seq 1 120); do [[ -f "$LOG" ]] && break; sleep 1; done
  if MATCH=$(stdbuf -oL tail --pid="$TPID" -n +1 -F "$LOG" 2>/dev/null | grep -m1 -nE "$RE"); then
    {
      echo "EARLY_STOP_SIGNAL: matched corrupting-failure pattern in $EXP"
      echo "EARLY_STOP_SIGNAL: $MATCH"
      echo "EARLY_STOP_SIGNAL: training-log-monitor should classify + recommend kill-switch for this cell."
    } | tee -a "$LOG"
    printf "%s\t%s\n" "$EXP" "$MATCH" > "$SENT"
  fi
' _ "$LOG" "$EARLY_STOP_RE" "$EXPERIMENT_NAME" "$EARLY_STOP_SENTINEL" "$TRAIN_PID" &
EARLY_STOP_WATCHER_PID=$!
EARLY_STOP_WATCHER_PGID="$(ps -o pgid= -p "$EARLY_STOP_WATCHER_PID" 2>/dev/null | tr -d '[:space:]')"
EARLY_STOP_OWNER_PGID="$(ps -o pgid= -p "$$" 2>/dev/null | tr -d '[:space:]')"

cleanup_early_stop_watcher() {
  local pid="${EARLY_STOP_WATCHER_PID:-}"
  local pgid="${EARLY_STOP_WATCHER_PGID:-}"
  local owner_pgid="${EARLY_STOP_OWNER_PGID:-}"
  [[ -n "$pid" ]] || return 0

  # A negative PID signals a whole process group. Do that only when `setsid`
  # demonstrably created the private group we requested. Falling back to the
  # launcher PID is safer than ever signalling the training/sweep group; the
  # watcher's `tail --pid=$TRAIN_PID` remains the independent self-exit guard.
  if [[ "$pid" =~ ^[0-9]+$ && "$pgid" =~ ^[0-9]+$ && "$pgid" == "$pid" && "$pgid" != "$owner_pgid" ]]; then
    kill -- -"$pgid" 2>/dev/null || true
  elif [[ "$pid" =~ ^[0-9]+$ ]]; then
    kill "$pid" 2>/dev/null || true
  fi
  wait "$pid" 2>/dev/null || true
  EARLY_STOP_WATCHER_PID=""
  EARLY_STOP_WATCHER_PGID=""
}

trap cleanup_early_stop_watcher EXIT

# Block on TRAINING ONLY (not the watcher) — when main_ppo finishes, we proceed
# to done.flag immediately; watcher cleanup is explicit before final sync. `wait $PID`
# returns the child's exit status; capture it WITHOUT tripping `set -e` (the
# `|| TRAIN_RC=$?` keeps a non-zero training exit from aborting before we can
# clean up the watcher + write done.flag + propagate the status).
TRAIN_RC=0
wait "$TRAIN_PID" || TRAIN_RC=$?

# Stop and reap the watcher once, then remove the EXIT trap. The old path sent
# an unchecked group kill here and repeated it from EXIT after final sync; in a
# nested multi-arm launcher that could terminate the sweep shell between arms.
cleanup_early_stop_watcher
trap - EXIT

# ---------------------------------------------------------------------------
# WandB final-flush. Ray teardown can race WandB's async uploader, so resync the
# local run directory after training and before done.flag. Best-effort only; the
# local train.log remains authoritative.
if command -v wandb >/dev/null 2>&1; then
  WANDB_RUN_DIR=$(ls -dt "$VERL_ROOT"/wandb/run-* "$VERL_ROOT"/wandb/offline-run-* 2>/dev/null | head -1 || true)
  if [[ -n "${WANDB_RUN_DIR:-}" ]]; then
    echo "=== wandb sync $WANDB_RUN_DIR (flush final history before teardown) ==="
    timeout 240 wandb sync "$WANDB_RUN_DIR" 2>&1 | tail -8 \
      || echo "WARN: wandb sync failed/timed out — final point may be missing online; local train.log is authoritative" >&2
  fi
fi

touch "$VERL_ROOT/runs/${EXPERIMENT_NAME}/done.flag"
echo "=== done at $(date -u +%FT%TZ) (train_rc=$TRAIN_RC) ==="
# Propagate the training exit status so the  launch.sh `run_step` sees a
# real failure (set -e / `|| true` semantics in the driver still apply).
exit "$TRAIN_RC"
