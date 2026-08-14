#!/usr/bin/env bash
# vast_comm_eff_engine_grpo.sh
#
# COMMUNICATION-EFFICIENT GRPO engine: model/dataset-agnostic, driven by env
# (default surface: Qwen2.5-Math-1.5B on MATH, 1..8 GPUs), FSDP + vLLM rollout.
# This is the shared engine every MATH launcher
# `exec`s (MODEL_PATH / DATA_DIR / EXPERIMENT_NAME are overridden by the caller);
# the dense control is the same surface with `COMM_EFF_ENABLED=false`. The ONLY
# differences from dense are the communication-efficient method's Hydra knobs.
#
# ===========================================================================
# Canonical communication-efficient base launcher. The defaults below configure
# the anchor circuit on a PowerSGD codec. The retained settings remain explicit
# environment variables:
#
#   comm-eff master ........ COMM_EFF_ENABLED          (true)     off => dense path
#   codec .................. COMM_EFF_COMPRESSION_TYPE (powersgd) compression codec
#   PowerSGD rank .......... COMM_EFF_POWERSGD_RANK    (77)
#   anchor ................. COMM_EFF_ANCHOR_ENABLED   (true)     paired dense-gradient reference M
#   anchor owns Q .......... COMM_EFF_ANCHOR_OWNS_Q    (true)     the ONLY thing that updates Q
#   anchor staleness ....... COMM_EFF_ANCHOR_DELAY_K   (20)       forward from theta_{t-20}
#   anchor refresh ......... COMM_EFF_ANCHOR_CADENCE   (20)       recompute M+Q every 20 ticks
#   anchor batch scope ..... COMM_EFF_ANCHOR_BATCH_SCOPE (ppo_minibatch) shared Q+M data scope
#   weight projection ...... rank1_relex W4/min2/strength1         progressive W2->W3->W4
#   merger EMA decay ....... COMM_EFF_SPECTRAL_BETA_ANC         (0.50)        anchor-gradient EMA decay
#
# Base in one line: PowerSGD r=77 plus a continuously maintained, stale,
# full-coverage (all floating parameters, DP-reduced) anchor gradient M, refreshed every
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
# Hardware: 1..8 GPUs. The project provisioning ladder selects a compatible
# shape. The anchor keeps a no-hook clone per rank for its stale
# forward-backward, so the default actor token
# budget is already bounded (PPO_MAX_TOKEN_LEN_PER_GPU=18432) for the 1024/3072
# prompt/response surface.
#
# Examples:
#   # signed_ema is the default merger. Override run length / name as needed:
#   TOTAL_TRAINING_STEPS=100 EXPERIMENT_NAME=ce_100 bash <thisfile>
#   # dense control via the same launcher:
#   COMM_EFF_ENABLED=false EXPERIMENT_NAME=ce_off_dense bash <thisfile>
#
# See examples/grpo_trainer/COMM_EFF_CONFIG.md for the compact current defaults and
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
# 2. GPU count — 1..8. The fixed model/data/method surface can run on any
#    compatible shape.
#    REQUIRE_MULTI_GPU=1 optionally enforces the existing 4-GPU gate.
# ---------------------------------------------------------------------------
DETECTED_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
GPU_MIN=1
if [[ "${REQUIRE_MULTI_GPU:-0}" == "1" ]]; then
  GPU_MIN=4
fi
if (( DETECTED_GPUS < GPU_MIN || DETECTED_GPUS > 8 )); then
  echo "FATAL: this recipe requires ${GPU_MIN}..8 GPUs; detected $DETECTED_GPUS" >&2
  if (( GPU_MIN > 1 )); then
    echo "       (REQUIRE_MULTI_GPU=1 requested the multi-GPU gate.)" >&2
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
# 4. Dataset — MATH parquet (auto-prep fallback if not already there).
#    The canonical MATH prep is research/scripts/prepare_rlvr_math.py (last-\boxed{}
#    + is_equiv reward); MATH launchers pre-prepare and pass DATA_DIR, so this
#    upstream fallback only fires for a bare engine run with no prepared data.
# ---------------------------------------------------------------------------
VERL_ROOT="${VERL_ROOT:-/workspace/verl}"
cd "$VERL_ROOT"

DATA_DIR="${DATA_DIR:-$HOME/data/math}"
if [[ ! -f "$DATA_DIR/train.parquet" || ! -f "$DATA_DIR/test.parquet" ]]; then
  echo "=== preprocess MATH -> $DATA_DIR ==="
  mkdir -p "$DATA_DIR"
  python3 examples/data_preprocess/math_dataset.py --local_save_dir "$DATA_DIR"
fi
export TRAIN_FILE="$DATA_DIR/train.parquet"
export TEST_FILE="$DATA_DIR/test.parquet"
echo "=== train: $(python3 -c "import pyarrow.parquet as p; print(p.read_table('$TRAIN_FILE').num_rows)") rows ==="
echo "=== test:  $(python3 -c "import pyarrow.parquet as p; print(p.read_table('$TEST_FILE').num_rows)") rows ==="

# ---------------------------------------------------------------------------
# 5. Model + training config. `actor.fsdp_config.use_orig_params=true` is
#    required so the anchor and signed-EMA hooks see full 2D Tensor gradients
#    after FSDP reduction.
# ---------------------------------------------------------------------------
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-Math-1.5B}"

# Rollout shape — n=8 rollouts/prompt, paged KV.
export ROLLOUT_TP="${ROLLOUT_TP:-1}"
# Clamp TP to the detected GPU count: the single-GPU collection path forces TP=1
# (rollout tensor-parallel can't exceed the device count). No effect on >=TP GPUs.
if (( ROLLOUT_TP > DETECTED_GPUS )); then
  echo "=== clamping ROLLOUT_TP $ROLLOUT_TP -> $DETECTED_GPUS (single-GPU path) ==="
  export ROLLOUT_TP="$DETECTED_GPUS"
fi
export ROLLOUT_N="${ROLLOUT_N:-8}"
export ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.55}"

# Batch sizes for the current default surface.
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-512}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-256}"
export PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
export LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}"

# The reference uses dynamic token-balanced micro-batching.
export USE_DYNAMIC_BSZ="${USE_DYNAMIC_BSZ:-True}"

# Train-inference mismatch DIAGNOSTIC (read-only; does NOT change training).
# calculate_log_probs=True makes vLLM return its rollout log-probs so the trainer
# logs training/rollout_probs_diff_* and rollout_corr/* (vLLM rollout vs the
# train-engine-recomputed old_log_prob). Rollout CORRECTION stays STRICTLY OFF
# by default (rollout_is=null, rollout_rs=null, bypass_mode=false): old_log_prob
# is always recomputed by the train engine and vLLM log-probs are never used in
# the loss. ROLLOUT_IS=token|sequence turns on decoupled importance weighting
# (issue #93 arm A5, FRLR only: token-IS is measured dead on PRF/sr_quant views).
export ROLLOUT_CALC_LOGPROBS="${ROLLOUT_CALC_LOGPROBS:-True}"
export ROLLOUT_IS="${ROLLOUT_IS:-null}"
export ROLLOUT_IS_THRESHOLD="${ROLLOUT_IS_THRESHOLD:-2.0}"
# Self-normalized IS: divide the weights by their batch mean so they average 1.0.
# Default false preserves existing behaviour for every other issue. Turning it on
# keeps the RELATIVE token reweighting (the correction you want) while removing the
# blanket shrinkage: #93 cell a5 measured a mean IS weight of 0.166, which scaled
# every gradient down about 6x and starved learning.
export ROLLOUT_IS_BATCH_NORMALIZE="${ROLLOUT_IS_BATCH_NORMALIZE:-false}"
case "${ROLLOUT_IS_BATCH_NORMALIZE}" in true|false) ;; *) echo "FATAL: bad ROLLOUT_IS_BATCH_NORMALIZE='${ROLLOUT_IS_BATCH_NORMALIZE}' (true|false)." >&2; exit 1;; esac
case "${ROLLOUT_IS}" in null|token|sequence) ;; *) echo "FATAL: bad ROLLOUT_IS='${ROLLOUT_IS}' (null|token|sequence)." >&2; exit 1;; esac

# Context window — 4096 total tokens in the reference protocol.
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-3072}"

# GRPO objective. The canonical MATH wrapper pins these values for comparisons.
export ACTOR_LR="${ACTOR_LR:-1e-6}"
export USE_KL_LOSS="${USE_KL_LOSS:-True}"
export USE_KL_IN_REWARD="${USE_KL_IN_REWARD:-False}"
export KL_LOSS_COEF="${KL_LOSS_COEF:-0.001}"
export ENTROPY_COEFF="${ENTROPY_COEFF:-0}"

# Run schedule — eight dataset epochs allow the explicit 100-step cap to win.
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-8}"
export SAVE_FREQ="${SAVE_FREQ:--1}"
export TEST_FREQ="${TEST_FREQ:-25}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-True}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-100}"

# ---------------------------------------------------------------------------
# Checkpoint -> Cloudflare R2 on-the-go mirror. When
# CKPT_R2_ENABLED=false (DEFAULT) the trainer's
# _save_checkpoint is byte-identical to upstream verl (no R2 threads, no r2_sink
# import on the save path) — the whole feature is a strict no-op. Set true to
# mirror every global_step_<N>/ tree (all rank shards + data.pt + huggingface +
# fsdp_config + the root latest_checkpointed_iteration.txt) to R2 and delete each
# local file after a verified upload, so peak local disk stays ~1 in-flight ckpt +
# staging instead of the keep-all total. The CKPT_R2_* async knobs are consumed by
# _save_checkpoint via the env; only CKPT_R2_ENABLED is threaded through Hydra
# (trainer.checkpoint_r2_enabled) since that is the gate. R2 creds/prefix come from
# the shared R2 env (R2_BUCKET/R2_ENDPOINT|R2_ACCOUNT_ID/
# R2_ACCESS_KEY_ID/R2_SECRET_ACCESS_KEY/R2_EXPERIMENT/R2_REGIME) — the checkpoints
# land under autonomous-harness-rlvr-compression/$R2_EXPERIMENT/$R2_REGIME/checkpoints.
export CKPT_R2_ENABLED="${CKPT_R2_ENABLED:-false}"
export CKPT_R2_ASYNC="${CKPT_R2_ASYNC:-true}"
export CKPT_R2_DELETE_LOCAL="${CKPT_R2_DELETE_LOCAL:-true}"
export CKPT_R2_MAX_STAGED_GB="${CKPT_R2_MAX_STAGED_GB:-50}"
export CKPT_R2_WORKERS="${CKPT_R2_WORKERS:-4}"

# WandB project + experiment. Harness runs set WANDB_RUN_GROUP to the per-issue
# run_id (e.g. 83-growing-fixed-base-anchor); default the project to it so each
# issue gets its own issue-prefixed WandB project automatically. An explicit
# PROJECT_NAME still wins; standalone engine runs (no group) keep the shared
# verl_compression_research fallback.
export PROJECT_NAME="${PROJECT_NAME:-${WANDB_RUN_GROUP:-verl_compression_research}}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen25_math_1p5b_grpo_math_comm_eff}"

# Token budget per micro-batch for dynamic batching. The actor budget is 18432
# to leave room for the anchor's isolated clone; log-probability paths can use a
# larger budget because they do not allocate that clone.
PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-18432}"
LOG_PROB_MAX_TOKEN_LEN_PER_GPU="${LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-36864}"
REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU="${REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-36864}"

# ---------------------------------------------------------------------------
# 6. Communication-efficient method.
#    Defaults configure the PowerSGD + delayed dense-anchor path while exposing
#    comm-eff knobs as env overrides. Field names mirror the actor config schema;
#    Hydra struct mode rejects unknown keys.
# ---------------------------------------------------------------------------
COMM_EFF_ENABLED="${COMM_EFF_ENABLED:-true}"                          # master switch (false => dense)
# --- codec selector: dense | prf_mask | powersgd ---
# PowerSGD is the default communication-efficient path; prf_mask is the
# per-(token, dim) PRF Bernoulli activation mask; dense is the control.
COMM_EFF_COMPRESSION_TYPE="${COMM_EFF_COMPRESSION_TYPE:-powersgd}"
# --- prf_mask codec (active iff COMM_EFF_COMPRESSION_TYPE=prf_mask) ---
# Anchor-independent boundary activation mask. Mutually exclusive with PowerSGD;
# it cannot anchor-own-Q, so a prf_mask arm must set COMM_EFF_ANCHOR_OWNS_Q=false
# (and COMM_EFF_POWERSGD_FAST_Q_BOOTSTRAP is inert under this codec).
COMM_EFF_MASK_ENABLED="${COMM_EFF_MASK_ENABLED:-false}"              # per-codec enable (redundant once compression_type=prf_mask)
COMM_EFF_MASK_P="${COMM_EFF_MASK_P:-0.95}"                           # masked (zeroed) fraction in [0,1]
COMM_EFF_MASK_RESCALE="${COMM_EFF_MASK_RESCALE:-false}"             # inverted-dropout 1/(1-p) rescale (needs p<1)
COMM_EFF_MASK_RECOMPUTE="${COMM_EFF_MASK_RECOMPUTE:-false}"         # also mask the old-logprob recompute
COMM_EFF_MASK_REFERENCE="${COMM_EFF_MASK_REFERENCE:-false}"         # also mask the reference-KL forward (codec-vs-codec KL)
COMM_EFF_MASK_SEED="${COMM_EFF_MASK_SEED:-0}"                        # base seed folded into the mask PRF key
COMM_EFF_MASK_PP_SIZE="${COMM_EFF_MASK_PP_SIZE:-8}"                  # logical pipeline-shard count (boundary blocks)
# issue #89 codec levers — default-off so the baseline PRF stays bit-identical.
COMM_EFF_MASK_RESCALE_MODE="${COMM_EFF_MASK_RESCALE_MODE:-auto}"     # none|constant|rms_match|auto magnitude restoration (lever 1)
COMM_EFF_MASK_EXACT_K="${COMM_EFF_MASK_EXACT_K:-false}"              # keep EXACTLY round((1-p)*H)/token via hash order stat (lever 2)
COMM_EFF_MASK_DENSE_EVERY="${COMM_EFF_MASK_DENSE_EVERY:-0}"          # issue #93: 0=off; N>0 bypasses the codec entirely on every step where global_step%N==0 (dense fwd+bwd, anchor suppressed)
COMM_EFF_MASK_ANTITHETIC="${COMM_EFF_MASK_ANTITHETIC:-false}"        # step t+1 keeps the antithetic complement of step t (lever 5)
COMM_EFF_MASK_P_BY_BOUNDARY="${COMM_EFF_MASK_P_BY_BOUNDARY:-}"       # optional per-boundary p vector, e.g. [0.92,..] mean ~p (lever 4); empty=off
COMM_EFF_MASK_FRLR="${COMM_EFF_MASK_FRLR:-false}"                    # FRLR "32+44+1" fresh-residual low-rank codec (issue #89); off=baseline PRF
COMM_EFF_MASK_FRLR_RANK="${COMM_EFF_MASK_FRLR_RANK:-32}"             # FRLR core rank r (step-frozen activation-derived Q, H x r)
COMM_EFF_MASK_FRLR_K="${COMM_EFF_MASK_FRLR_K:-44}"                   # FRLR per-token PRF-fresh exact-k residual subset size
COMM_EFF_MASK_FRLR_UNBIASED="${COMM_EFF_MASK_FRLR_UNBIASED:-false}"  # H/k constant gain (E[h_hat|h,Q]=h) instead of capped norm matching
COMM_EFF_MASK_FRLR_Q_CADENCE="${COMM_EFF_MASK_FRLR_Q_CADENCE:-1}"    # refresh FRLR Q every N global steps (1=every step); frozen between refreshes, sketch accumulates
# --- sr_quant codec (active iff COMM_EFF_COMPRESSION_TYPE=sr_quant) ---
# Dense low-bit stochastic-rounding quantization of the boundary activations
# (and of the boundary backward gradient). Reuses COMM_EFF_MASK_RECOMPUTE /
# COMM_EFF_MASK_REFERENCE / COMM_EFF_MASK_SEED / COMM_EFF_MASK_PP_SIZE for
# eligibility and keying (MASK_P / RESCALE / EXACT_K etc. are ignored). Like
# prf_mask it cannot anchor-own-Q: an sr_quant arm must set
# COMM_EFF_ANCHOR_OWNS_Q=false (COMM_EFF_POWERSGD_FAST_Q_BOOTSTRAP is inert).
COMM_EFF_QUANT_BITS="${COMM_EFF_QUANT_BITS:-1}"                      # bits/channel; 2**bits uniform levels in [-s,+s]
COMM_EFF_QUANT_BLOCK_SIZE="${COMM_EFF_QUANT_BLOCK_SIZE:-32}"         # channels per fp16 absmax-scale block (0 = whole-token scale)
COMM_EFF_QUANT_ROUNDING="${COMM_EFF_QUANT_ROUNDING:-sr}"             # sr = unbiased PRF stochastic rounding | rn = round-to-nearest (biased ablation control)
COMM_EFF_QUANT_SUBSET_K="${COMM_EFF_QUANT_SUBSET_K:-0}"              # issue #93 I5 byte-parity hybrid: quantize only a PRF-fresh exact-k channel subset J/token, rescale H/k; 0 = full width
# --- dense-view probe + adaptive KL coefficient (issue #93 I3) ---
# Every PROBE_EVERY trainer steps the trainer reruns the step's actor +
# reference logprob passes once with the codec silent (measurement only, no
# backward) and logs probe/kl_dense, probe/kl_gain, probe/gap_dense. The
# controller (CTRL_ENABLED) retunes kl_loss_coef by projected dual ascent in
# log space toward max(FLOOR, GAIN x dense_table(step)); the LR brake only
# DETECTS and logs (probe/lr_brake_triggered), it never mutates the LR.
# Defaults off: every existing config is bit-identical unchanged.
COMM_EFF_PROBE_EVERY="${COMM_EFF_PROBE_EVERY:-0}"                    # probe cadence in trainer steps (0 = off)
COMM_EFF_PROBE_CTRL_ENABLED="${COMM_EFF_PROBE_CTRL_ENABLED:-false}"  # adaptive KL coefficient controller (requires PROBE_EVERY >= 1)
COMM_EFF_PROBE_KL_TARGET_TABLE="${COMM_EFF_PROBE_KL_TARGET_TABLE:-}" # "step:value,step:value" dense-control reference-KL curve; empty = floor only
COMM_EFF_PROBE_KL_TARGET_FLOOR="${COMM_EFF_PROBE_KL_TARGET_FLOOR:-0.005}"  # setpoint floor c_floor (nats)
COMM_EFF_PROBE_KL_TARGET_GAIN="${COMM_EFF_PROBE_KL_TARGET_GAIN:-2.0}"      # setpoint multiplier on the interpolated dense KL
COMM_EFF_PROBE_CTRL_KI="${COMM_EFF_PROBE_CTRL_KI:-0.3}"              # integral (dual-ascent) gain
COMM_EFF_PROBE_CTRL_KP="${COMM_EFF_PROBE_CTRL_KP:-0.1}"              # proportional damping gain
COMM_EFF_PROBE_CTRL_BETA_MIN="${COMM_EFF_PROBE_CTRL_BETA_MIN:-2e-4}" # beta projection lower bound
COMM_EFF_PROBE_CTRL_BETA_MAX="${COMM_EFF_PROBE_CTRL_BETA_MAX:-0.05}" # beta projection upper bound
# --- CVC: train the train-inference disagreement down (issue #93 I4) ---
# CE mode: loss += lambda_eff * (-mean log pi_theta(a_t)) on response tokens
# (the codec view under an active codec); lambda_eff ramps linearly over
# CVC_WARMUP_STEPS trainer steps. DC mode (DC-GRPO, arXiv 2606.08779):
# advantage shaping A_t -= lambda * |p_train - p_inf| with a once-per-step
# projected dual update of lambda toward DC_TARGET (the measured step-1 static
# per-token discrepancy floor plus slack; REQUIRED when DC is enabled, no
# default magic). Both default off (bit-identical), zero extra forward passes.
COMM_EFF_CVC_LAMBDA="${COMM_EFF_CVC_LAMBDA:-0.0}"                # CE-mode weight (0 = off)
COMM_EFF_CVC_WARMUP_STEPS="${COMM_EFF_CVC_WARMUP_STEPS:-20}"     # linear lambda ramp in trainer steps
COMM_EFF_DC_ENABLED="${COMM_EFF_DC_ENABLED:-false}"              # DC-GRPO advantage shaping
COMM_EFF_DC_ETA="${COMM_EFF_DC_ETA:-1.0}"                        # dual ascent step size
COMM_EFF_DC_TARGET="${COMM_EFF_DC_TARGET:--1.0}"                 # per-token discrepancy setpoint; -1.0 = unset sentinel
COMM_EFF_DC_LAMBDA0="${COMM_EFF_DC_LAMBDA0:-0.05}"               # initial shaping strength
COMM_EFF_DC_LAMBDA_MAX="${COMM_EFF_DC_LAMBDA_MAX:-1.0}"          # lambda projection upper bound
# --- anchor circuit ---
COMM_EFF_ANCHOR_ENABLED="${COMM_EFF_ANCHOR_ENABLED:-true}"
# Anchor cadence is measured in optimizer ticks, not trainer global steps.
COMM_EFF_ANCHOR_CADENCE="${COMM_EFF_ANCHOR_CADENCE:-20}"
# The anchor forwards from a delay_K-tick-stale weight snapshot.
COMM_EFF_ANCHOR_DELAY_K="${COMM_EFF_ANCHOR_DELAY_K:-20}"
# When true, only the anchor refresh updates the PowerSGD basis Q.
COMM_EFF_ANCHOR_OWNS_Q="${COMM_EFF_ANCHOR_OWNS_Q:-true}"
# Paired replay uses the same batch/weights the fast circuit saw so the
# correction tracks codec error rather than batch mismatch.
COMM_EFF_ANCHOR_REPLAY_PAIRED_BATCH="${COMM_EFF_ANCHOR_REPLAY_PAIRED_BATCH:-true}"
# Shared data scope for the anchor-owned Q observation and dense M backward.
# ppo_minibatch consumes one PPO mini-batch; rollout_batch consumes the complete
# pre-split actor update while dynamic microbatching still bounds peak activation
# memory.
COMM_EFF_ANCHOR_BATCH_SCOPE="${COMM_EFF_ANCHOR_BATCH_SCOPE:-ppo_minibatch}"
COMM_EFF_ANCHOR_SNAPSHOT_DEVICE="${COMM_EFF_ANCHOR_SNAPSHOT_DEVICE:-cpu}"
# Rank1-RELEX projects each floating weight tensor independently.
COMM_EFF_ANCHOR_LOOKAHEAD_ANCHOR="${COMM_EFF_ANCHOR_LOOKAHEAD_ANCHOR:-true}"
COMM_EFF_ANCHOR_LOOKAHEAD_MODE="${COMM_EFF_ANCHOR_LOOKAHEAD_MODE:-rank1_relex}"
COMM_EFF_ANCHOR_LOOKAHEAD_STRENGTH="${COMM_EFF_ANCHOR_LOOKAHEAD_STRENGTH:-1.0}"
COMM_EFF_ANCHOR_LOOKAHEAD_ROLLOUT_SOURCE="${COMM_EFF_ANCHOR_LOOKAHEAD_ROLLOUT_SOURCE:-auto}"
COMM_EFF_ANCHOR_LOOKAHEAD_WINDOW_SNAPSHOTS="${COMM_EFF_ANCHOR_LOOKAHEAD_WINDOW_SNAPSHOTS:-4}"
# Warmup behavior before the rank1-RELEX projector is ready. "stale_correct"
# runs the delayed dense-anchor path; "q_only" performs a rank1-only
# forward/no-backward Q refresh with M/correction off.
COMM_EFF_ANCHOR_WARMUP_MODE="${COMM_EFF_ANCHOR_WARMUP_MODE:-stale_correct}"
# Min ring snapshots before the projector engages. Two starts from the earliest
# legal fire; -1 waits for the full retained window.
COMM_EFF_ANCHOR_LOOKAHEAD_MIN_SNAPSHOTS="${COMM_EFF_ANCHOR_LOOKAHEAD_MIN_SNAPSHOTS:-2}"
# rank1_relex delta-base history mode: sliding_window (default) keeps the last
# `window` checkpoints (base advances); growing_fixed_base pins the seeded base
# and grows the base-relative delta history. max_snapshots caps growing_fixed_base
# retention (-1 unbounded; must stay -1 with sliding_window). Defaults reproduce
# prior behavior byte-for-byte.
COMM_EFF_ANCHOR_LOOKAHEAD_HISTORY_MODE="${COMM_EFF_ANCHOR_LOOKAHEAD_HISTORY_MODE:-sliding_window}"
COMM_EFF_ANCHOR_LOOKAHEAD_MAX_SNAPSHOTS="${COMM_EFF_ANCHOR_LOOKAHEAD_MAX_SNAPSHOTS:--1}"
# Anchor-sourced optimizer-state reset (anchor.opt_reset). Every CADENCE
# optimizer ticks (the same tick units as the anchor cadence) the fast AdamW
# moments are overwritten with moments the anchor maintains from its clean
# dense replay gradients (mode=anchor_moments, norm-matched when SCALE_MATCH)
# or zeroed (mode=zero). Defaults off (bit-identical trainer path).
COMM_EFF_OPT_RESET_ENABLED="${COMM_EFF_OPT_RESET_ENABLED:-false}"
COMM_EFF_OPT_RESET_CADENCE="${COMM_EFF_OPT_RESET_CADENCE:-50}"
COMM_EFF_OPT_RESET_MODE="${COMM_EFF_OPT_RESET_MODE:-anchor_moments}"
COMM_EFF_OPT_RESET_B1="${COMM_EFF_OPT_RESET_B1:-0.8}"
COMM_EFF_OPT_RESET_B2="${COMM_EFF_OPT_RESET_B2:-0.95}"
COMM_EFF_OPT_RESET_SCALE_MATCH="${COMM_EFF_OPT_RESET_SCALE_MATCH:-true}"
# --- anchor-guided gradient correction / merger ---
COMM_EFF_SPECTRAL_ENABLED="${COMM_EFF_SPECTRAL_ENABLED:-true}"
COMM_EFF_SPECTRAL_TARGET_SCOPE="${COMM_EFF_SPECTRAL_TARGET_SCOPE:-all_floating}"
COMM_EFF_SPECTRAL_DIAGNOSTICS="${COMM_EFF_SPECTRAL_DIAGNOSTICS:-false}"
COMM_EFF_SPECTRAL_BETA_ANC="${COMM_EFF_SPECTRAL_BETA_ANC:-0.50}"       # anchor-gradient EMA decay (signed_ema baseline)
COMM_EFF_SPECTRAL_SIGNED_EMA_ALPHA="${COMM_EFF_SPECTRAL_SIGNED_EMA_ALPHA:-0.25}"   # signed_ema mixing weight
# Merger dispatch: signed_ema (default) or delayed_ef (additive anchor residual,
# delta = M - G_comp refreshed once per anchor fire, G_corr = G_comp + lambda*delta).
COMM_EFF_SPECTRAL_CORRECTION_MODE="${COMM_EFF_SPECTRAL_CORRECTION_MODE:-signed_ema}"
COMM_EFF_SPECTRAL_DELAYED_EF_LAMBDA="${COMM_EFF_SPECTRAL_DELAYED_EF_LAMBDA:-1.0}"   # delayed_ef residual weight (0 = identity)
# blend: G_corr = (1-eta)*G_comp + eta*(||G_comp||/||M||)*M (convex value merger, 0 = identity)
COMM_EFF_SPECTRAL_BLEND_ETA="${COMM_EFF_SPECTRAL_BLEND_ETA:-0.5}"
# Correction cadence in optimizer ticks.
COMM_EFF_SPECTRAL_CADENCE="${COMM_EFF_SPECTRAL_CADENCE:-1}"
COMM_EFF_SPECTRAL_EMA_DEVICE="${COMM_EFF_SPECTRAL_EMA_DEVICE:-cpu}"    # offload full-coverage M (OOM guard)
# Cap target matrices per correction. -1 means no cap.
COMM_EFF_SPECTRAL_MAX_TARGETS="${COMM_EFF_SPECTRAL_MAX_TARGETS:--1}"
# --- PowerSGD activation compression ---
COMM_EFF_POWERSGD_RANK="${COMM_EFF_POWERSGD_RANK:-77}"
COMM_EFF_POWERSGD_SEED="${COMM_EFF_POWERSGD_SEED:-0}"                 # per-layer basis seed base
COMM_EFF_POWERSGD_PP_SIZE="${COMM_EFF_POWERSGD_PP_SIZE:-8}"           # boundary blocks
COMM_EFF_POWERSGD_UPDATE_CADENCE="${COMM_EFF_POWERSGD_UPDATE_CADENCE:-1}"  # orth(V) every N steps
COMM_EFF_POWERSGD_WARM_START="${COMM_EFF_POWERSGD_WARM_START:-true}"  # carry Q across steps
COMM_EFF_POWERSGD_COMPRESS_RECOMPUTE="${COMM_EFF_POWERSGD_COMPRESS_RECOMPUTE:-true}"  # project old-logprob too
COMM_EFF_POWERSGD_COMPRESS_REFERENCE="${COMM_EFF_POWERSGD_COMPRESS_REFERENCE:-true}"  # project the frozen reference-KL forward too (shares this step's anchor-owned Q; dense fallback until Q1 exists). false => dense reference control
COMM_EFF_POWERSGD_SYNC_BASIS="${COMM_EFF_POWERSGD_SYNC_BASIS:-true}"  # all-reduce V across DP => single shared consensus Q (REQUIRED under DP)
COMM_EFF_POWERSGD_FAST_Q_BOOTSTRAP="${COMM_EFF_POWERSGD_FAST_Q_BOOTSTRAP:-true}"  # one discarded dense observation before the first compressed old-logprob
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
  mismatch diag:       calculate_log_probs=$ROLLOUT_CALC_LOGPROBS (logs training/rollout_probs_diff_*); rollout_is=$ROLLOUT_IS threshold=$ROLLOUT_IS_THRESHOLD (null = correction OFF, recompute old_log_prob)
  comm_eff master:     $COMM_EFF_ENABLED
  compression_type:    $COMM_EFF_COMPRESSION_TYPE  (dense|prf_mask|powersgd|sr_quant)
  prf_mask:            enabled=$COMM_EFF_MASK_ENABLED p=$COMM_EFF_MASK_P rescale=$COMM_EFF_MASK_RESCALE rescale_mode=$COMM_EFF_MASK_RESCALE_MODE mask_recompute=$COMM_EFF_MASK_RECOMPUTE mask_reference=$COMM_EFF_MASK_REFERENCE seed=$COMM_EFF_MASK_SEED pp_size=$COMM_EFF_MASK_PP_SIZE  (active iff compression_type=prf_mask)
  sr_quant:            bits=$COMM_EFF_QUANT_BITS block_size=$COMM_EFF_QUANT_BLOCK_SIZE rounding=$COMM_EFF_QUANT_ROUNDING subset_k=$COMM_EFF_QUANT_SUBSET_K  (active iff compression_type=sr_quant; reuses mask recompute/reference/seed/pp_size)
  probe:               every=$COMM_EFF_PROBE_EVERY ctrl=$COMM_EFF_PROBE_CTRL_ENABLED table=[${COMM_EFF_PROBE_KL_TARGET_TABLE:-<unset>}] floor=$COMM_EFF_PROBE_KL_TARGET_FLOOR gain=$COMM_EFF_PROBE_KL_TARGET_GAIN ki=$COMM_EFF_PROBE_CTRL_KI kp=$COMM_EFF_PROBE_CTRL_KP beta=[$COMM_EFF_PROBE_CTRL_BETA_MIN,$COMM_EFF_PROBE_CTRL_BETA_MAX]  (issue #93 I3; every=0 => off)
  cvc:                 ce_lambda=$COMM_EFF_CVC_LAMBDA warmup=$COMM_EFF_CVC_WARMUP_STEPS dc=$COMM_EFF_DC_ENABLED dc_eta=$COMM_EFF_DC_ETA dc_target=$COMM_EFF_DC_TARGET dc_lambda0=$COMM_EFF_DC_LAMBDA0 dc_lambda_max=$COMM_EFF_DC_LAMBDA_MAX  (issue #93 I4; lambda=0 + dc=false => off)
  dense_every:         $COMM_EFF_MASK_DENSE_EVERY  (0=off; N>0 = full-fidelity uncompressed fwd+bwd on every step where global_step%N==0, anchor suppressed there)
  prf_mask levers:     exact_k=$COMM_EFF_MASK_EXACT_K antithetic=$COMM_EFF_MASK_ANTITHETIC p_by_boundary=[${COMM_EFF_MASK_P_BY_BOUNDARY:-<unset>}] frlr=$COMM_EFF_MASK_FRLR frlr_rank=$COMM_EFF_MASK_FRLR_RANK frlr_k=$COMM_EFF_MASK_FRLR_K frlr_unbiased=$COMM_EFF_MASK_FRLR_UNBIASED frlr_q_cadence=$COMM_EFF_MASK_FRLR_Q_CADENCE  (issue #89; all off => baseline PRF)
  powersgd:            rank=$COMM_EFF_POWERSGD_RANK seed=$COMM_EFF_POWERSGD_SEED pp_size=$COMM_EFF_POWERSGD_PP_SIZE update_cadence=$COMM_EFF_POWERSGD_UPDATE_CADENCE warm_start=$COMM_EFF_POWERSGD_WARM_START compress_recompute=$COMM_EFF_POWERSGD_COMPRESS_RECOMPUTE compress_reference=$COMM_EFF_POWERSGD_COMPRESS_REFERENCE sync_basis=$COMM_EFF_POWERSGD_SYNC_BASIS fast_q_bootstrap=$COMM_EFF_POWERSGD_FAST_Q_BOOTSTRAP qr_dtype=$COMM_EFF_POWERSGD_QR_DTYPE reortho_eps=$COMM_EFF_POWERSGD_REORTHO_EPS  (active iff compression_type=powersgd)
  anchor:              enabled=$COMM_EFF_ANCHOR_ENABLED cadence=$COMM_EFF_ANCHOR_CADENCE delay_K=$COMM_EFF_ANCHOR_DELAY_K owns_q=$COMM_EFF_ANCHOR_OWNS_Q replay_paired_batch=$COMM_EFF_ANCHOR_REPLAY_PAIRED_BATCH batch_scope=$COMM_EFF_ANCHOR_BATCH_SCOPE snapshot_device=$COMM_EFF_ANCHOR_SNAPSHOT_DEVICE
  opt_reset:           enabled=$COMM_EFF_OPT_RESET_ENABLED cadence=$COMM_EFF_OPT_RESET_CADENCE mode=$COMM_EFF_OPT_RESET_MODE b1=$COMM_EFF_OPT_RESET_B1 b2=$COMM_EFF_OPT_RESET_B2 scale_match=$COMM_EFF_OPT_RESET_SCALE_MATCH  (anchor-sourced AdamW-moment overwrite; enabled=false => off)
  lookahead:           enabled=$COMM_EFF_ANCHOR_LOOKAHEAD_ANCHOR mode=$COMM_EFF_ANCHOR_LOOKAHEAD_MODE strength=$COMM_EFF_ANCHOR_LOOKAHEAD_STRENGTH rollout_source=$COMM_EFF_ANCHOR_LOOKAHEAD_ROLLOUT_SOURCE window=$COMM_EFF_ANCHOR_LOOKAHEAD_WINDOW_SNAPSHOTS warmup=$COMM_EFF_ANCHOR_WARMUP_MODE min_snapshots=$COMM_EFF_ANCHOR_LOOKAHEAD_MIN_SNAPSHOTS history_mode=$COMM_EFF_ANCHOR_LOOKAHEAD_HISTORY_MODE max_snapshots=$COMM_EFF_ANCHOR_LOOKAHEAD_MAX_SNAPSHOTS
  spectral:            enabled=$COMM_EFF_SPECTRAL_ENABLED target_scope=$COMM_EFF_SPECTRAL_TARGET_SCOPE diagnostics=$COMM_EFF_SPECTRAL_DIAGNOSTICS beta_anc=$COMM_EFF_SPECTRAL_BETA_ANC cadence=$COMM_EFF_SPECTRAL_CADENCE max_targets=$COMM_EFF_SPECTRAL_MAX_TARGETS ema_device=$COMM_EFF_SPECTRAL_EMA_DEVICE
  merger:               mode=$COMM_EFF_SPECTRAL_CORRECTION_MODE alpha=$COMM_EFF_SPECTRAL_SIGNED_EMA_ALPHA lambda=$COMM_EFF_SPECTRAL_DELAYED_EF_LAMBDA eta=$COMM_EFF_SPECTRAL_BLEND_ETA beta_anc=$COMM_EFF_SPECTRAL_BETA_ANC
  wandb:               $PROJECT_NAME / $EXPERIMENT_NAME
  log:                 $LOG
=== launching ===
EOF

# ---------------------------------------------------------------------------
# 6a. Resolved-codec boot gate (issue #89) — runs BEFORE any GPU spend. Fails
#     fast on a mis-resolved prf_mask codec (the money gate: never pay to train
#     a config that silently fell through to dense or would crash validation).
# ---------------------------------------------------------------------------
MASK_PBB_OVERRIDE=()
if [[ "${COMM_EFF_ENABLED}" == "true" && "${COMM_EFF_COMPRESSION_TYPE}" == "prf_mask" ]]; then
  [[ "${COMM_EFF_MASK_ENABLED}" == "true" ]] || { echo "FATAL: compression_type=prf_mask but COMM_EFF_MASK_ENABLED != true — codec would resolve to dense." >&2; exit 1; }
  # Only FRLR carries a basis Q the anchor can own (issue #93). The PLAIN PRF
  # mask is a PRF of seed/step/layer with nothing to own, so owns_q stays false
  # there. Under FRLR + owns_q the fast path is gated off as a Q writer and the
  # anchor refreshes the basis when it fires, so the anchor must be enabled.
  if [[ "${COMM_EFF_ANCHOR_OWNS_Q}" == "true" ]]; then
    [[ "${COMM_EFF_MASK_FRLR}" == "true" ]] || { echo "FATAL: prf_mask requires COMM_EFF_ANCHOR_OWNS_Q=false unless COMM_EFF_MASK_FRLR=true (the plain mask has no basis Q)." >&2; exit 1; }
    [[ "${COMM_EFF_ANCHOR_ENABLED}" == "true" ]] || { echo "FATAL: COMM_EFF_MASK_FRLR with COMM_EFF_ANCHOR_OWNS_Q=true requires COMM_EFF_ANCHOR_ENABLED=true (the anchor is the only Q updater)." >&2; exit 1; }
  fi
  case "${COMM_EFF_MASK_RESCALE_MODE}" in none|constant|rms_match|auto) ;; *) echo "FATAL: bad COMM_EFF_MASK_RESCALE_MODE='${COMM_EFF_MASK_RESCALE_MODE}' (none|constant|rms_match|auto)." >&2; exit 1;; esac
  if [[ "${COMM_EFF_MASK_FRLR}" == "true" ]]; then
    # FRLR is mutually exclusive with the plain-mask levers and the plain
    # rescale path (it draws its own PRF-fresh exact-k subset J and applies
    # its own detached norm matching). Fail before GPU spend, matching the
    # CommEffConfig validation that would reject it on the box.
    [[ "${COMM_EFF_MASK_EXACT_K}" == "false" && "${COMM_EFF_MASK_ANTITHETIC}" == "false" && -z "${COMM_EFF_MASK_P_BY_BOUNDARY}" ]] || { echo "FATAL: frlr=true requires exact_k=false, antithetic=false, p_by_boundary unset." >&2; exit 1; }
    [[ "${COMM_EFF_MASK_RESCALE}" == "false" ]] || { echo "FATAL: frlr=true requires COMM_EFF_MASK_RESCALE=false (FRLR does its own norm matching)." >&2; exit 1; }
    case "${COMM_EFF_MASK_RESCALE_MODE}" in none|auto) ;; *) echo "FATAL: frlr=true requires COMM_EFF_MASK_RESCALE_MODE=none|auto." >&2; exit 1;; esac
    [[ "${COMM_EFF_MASK_FRLR_Q_CADENCE}" =~ ^[1-9][0-9]*$ ]] || { echo "FATAL: COMM_EFF_MASK_FRLR_Q_CADENCE='${COMM_EFF_MASK_FRLR_Q_CADENCE}' must be an integer >= 1 (1 = every-step Q refresh)." >&2; exit 1; }
  fi
  echo "=== resolved codec OK (before GPU): prf_mask p=$COMM_EFF_MASK_P rescale_mode=$COMM_EFF_MASK_RESCALE_MODE exact_k=$COMM_EFF_MASK_EXACT_K antithetic=$COMM_EFF_MASK_ANTITHETIC p_by_boundary=[${COMM_EFF_MASK_P_BY_BOUNDARY}] frlr=$COMM_EFF_MASK_FRLR rank=$COMM_EFF_MASK_FRLR_RANK k=$COMM_EFF_MASK_FRLR_K unbiased=$COMM_EFF_MASK_FRLR_UNBIASED q_cadence=$COMM_EFF_MASK_FRLR_Q_CADENCE ==="
fi
if [[ "${COMM_EFF_ENABLED}" == "true" && "${COMM_EFF_COMPRESSION_TYPE}" == "sr_quant" ]]; then
  # Mirror the prf_mask gate: sr_quant carries no PowerSGD basis Q, so the
  # anchor cannot own it; fail before any GPU spend, matching CommEffConfig.
  [[ "${COMM_EFF_ANCHOR_OWNS_Q}" == "true" ]] && { echo "FATAL: sr_quant requires COMM_EFF_ANCHOR_OWNS_Q=false (quantizer has no PowerSGD basis Q)." >&2; exit 1; }
  [[ "${COMM_EFF_QUANT_BITS}" =~ ^[1-9][0-9]*$ ]] || { echo "FATAL: COMM_EFF_QUANT_BITS='${COMM_EFF_QUANT_BITS}' must be an integer >= 1." >&2; exit 1; }
  [[ "${COMM_EFF_QUANT_BLOCK_SIZE}" =~ ^[0-9]+$ ]] || { echo "FATAL: COMM_EFF_QUANT_BLOCK_SIZE='${COMM_EFF_QUANT_BLOCK_SIZE}' must be an integer >= 0 (0 = whole-token scale)." >&2; exit 1; }
  case "${COMM_EFF_QUANT_ROUNDING}" in sr|rn) ;; *) echo "FATAL: bad COMM_EFF_QUANT_ROUNDING='${COMM_EFF_QUANT_ROUNDING}' (sr|rn)." >&2; exit 1;; esac
  [[ "${COMM_EFF_QUANT_SUBSET_K}" =~ ^[0-9]+$ ]] || { echo "FATAL: COMM_EFF_QUANT_SUBSET_K='${COMM_EFF_QUANT_SUBSET_K}' must be an integer >= 0 (0 = full width)." >&2; exit 1; }
  if (( COMM_EFF_QUANT_SUBSET_K > 0 )); then
    # Exact wire accounting for the subset arm (issue #93 4.3): payload
    # k*bits + one fp16 scale per block of KEPT channels, ragged tail
    # pro-rata; J costs no index bits (PRF-derivable at the receiver).
    # Target parity arm k=493 bits=2 block=32 -> 1233 bits vs incumbent 1232.
    SUBSET_BITS_LINE="$(python3 -c "
import math
k = ${COMM_EFF_QUANT_SUBSET_K}; b = ${COMM_EFF_QUANT_BITS}; blk = ${COMM_EFF_QUANT_BLOCK_SIZE}
eff = k if (blk <= 0 or blk >= k) else blk
payload = k * b
scales = k * 16 / eff
total = payload + scales
print(f'payload {payload} + fp16 scales {scales:g} = {total:g} bits/token/boundary'
      f' (ceil {math.ceil(total)}; incumbent prf exact-k 77x16 = 1232)')
")" || { echo "FATAL: subset bit-accounting computation failed." >&2; exit 1; }
    echo "=== sr_quant subset accounting (before GPU): k=$COMM_EFF_QUANT_SUBSET_K bits=$COMM_EFF_QUANT_BITS block=$COMM_EFF_QUANT_BLOCK_SIZE -> $SUBSET_BITS_LINE ==="
  fi
  echo "=== resolved codec OK (before GPU): sr_quant bits=$COMM_EFF_QUANT_BITS block_size=$COMM_EFF_QUANT_BLOCK_SIZE rounding=$COMM_EFF_QUANT_ROUNDING subset_k=$COMM_EFF_QUANT_SUBSET_K mask_recompute=$COMM_EFF_MASK_RECOMPUTE mask_reference=$COMM_EFF_MASK_REFERENCE seed=$COMM_EFF_MASK_SEED pp_size=$COMM_EFF_MASK_PP_SIZE ==="
fi
# p_by_boundary is a Hydra list literal; only override it when set (empty = default []).
if [[ -n "${COMM_EFF_MASK_P_BY_BOUNDARY}" ]]; then
  MASK_PBB_OVERRIDE+=("actor_rollout_ref.actor.comm_eff.mask.p_by_boundary=${COMM_EFF_MASK_P_BY_BOUNDARY}")
fi
# Probe/controller boot gate (issue #93 I3): fail before any GPU spend,
# matching the CommEffProbeConfig validation that would reject it on the box.
[[ "${COMM_EFF_PROBE_EVERY}" =~ ^[0-9]+$ ]] || { echo "FATAL: COMM_EFF_PROBE_EVERY='${COMM_EFF_PROBE_EVERY}' must be an integer >= 0 (0 = off)." >&2; exit 1; }
case "${COMM_EFF_PROBE_CTRL_ENABLED}" in true|false) ;; *) echo "FATAL: bad COMM_EFF_PROBE_CTRL_ENABLED='${COMM_EFF_PROBE_CTRL_ENABLED}' (true|false)." >&2; exit 1;; esac
if [[ "${COMM_EFF_PROBE_CTRL_ENABLED}" == "true" ]]; then
  [[ "${COMM_EFF_PROBE_EVERY}" -ge 1 ]] || { echo "FATAL: COMM_EFF_PROBE_CTRL_ENABLED=true requires COMM_EFF_PROBE_EVERY >= 1 (the controller only updates at probes)." >&2; exit 1; }
fi
if [[ -n "${COMM_EFF_PROBE_KL_TARGET_TABLE}" ]]; then
  # "step:value,step:value": integer steps, plain/scientific float values.
  TABLE_ENTRY_RE='[0-9]+:[0-9]*\.?[0-9]+([eE][+-]?[0-9]+)?'
  [[ "${COMM_EFF_PROBE_KL_TARGET_TABLE}" =~ ^${TABLE_ENTRY_RE}(,${TABLE_ENTRY_RE})*$ ]] || { echo "FATAL: COMM_EFF_PROBE_KL_TARGET_TABLE='${COMM_EFF_PROBE_KL_TARGET_TABLE}' must be 'step:value,step:value'." >&2; exit 1; }
fi
if [[ "${COMM_EFF_PROBE_EVERY}" -ge 1 ]]; then
  echo "=== resolved probe OK (before GPU): every=$COMM_EFF_PROBE_EVERY ctrl=$COMM_EFF_PROBE_CTRL_ENABLED table=[${COMM_EFF_PROBE_KL_TARGET_TABLE}] floor=$COMM_EFF_PROBE_KL_TARGET_FLOOR gain=$COMM_EFF_PROBE_KL_TARGET_GAIN ki=$COMM_EFF_PROBE_CTRL_KI kp=$COMM_EFF_PROBE_CTRL_KP beta=[$COMM_EFF_PROBE_CTRL_BETA_MIN,$COMM_EFF_PROBE_CTRL_BETA_MAX] ==="
fi
# The table rides Hydra as a quoted string (commas would otherwise parse as a
# choice sweep); only override when set (empty = default "").
PROBE_TABLE_OVERRIDE=()
if [[ -n "${COMM_EFF_PROBE_KL_TARGET_TABLE}" ]]; then
  PROBE_TABLE_OVERRIDE+=("actor_rollout_ref.actor.comm_eff.probe.kl_target_table='${COMM_EFF_PROBE_KL_TARGET_TABLE}'")
fi
# CVC/DC boot gate (issue #93 I4): fail before any GPU spend, matching the
# ActorConfig / CommEffDCConfig validation that would reject it on the box.
FLOAT_RE='-?[0-9]*\.?[0-9]+([eE][+-]?[0-9]+)?'
[[ "${COMM_EFF_CVC_LAMBDA}" =~ ^${FLOAT_RE}$ && ! "${COMM_EFF_CVC_LAMBDA}" =~ ^- ]] || { echo "FATAL: COMM_EFF_CVC_LAMBDA='${COMM_EFF_CVC_LAMBDA}' must be a float >= 0 (0 = off)." >&2; exit 1; }
[[ "${COMM_EFF_CVC_WARMUP_STEPS}" =~ ^[0-9]+$ ]] || { echo "FATAL: COMM_EFF_CVC_WARMUP_STEPS='${COMM_EFF_CVC_WARMUP_STEPS}' must be an integer >= 0." >&2; exit 1; }
case "${COMM_EFF_DC_ENABLED}" in true|false) ;; *) echo "FATAL: bad COMM_EFF_DC_ENABLED='${COMM_EFF_DC_ENABLED}' (true|false)." >&2; exit 1;; esac
if [[ "${COMM_EFF_DC_ENABLED}" == "true" ]]; then
  # target has no default magic: it is the measured step-1 static per-token
  # discrepancy floor plus slack; the -1.0 sentinel must be replaced.
  [[ "${COMM_EFF_DC_TARGET}" =~ ^${FLOAT_RE}$ && ! "${COMM_EFF_DC_TARGET}" =~ ^- ]] || { echo "FATAL: COMM_EFF_DC_ENABLED=true requires an explicit COMM_EFF_DC_TARGET >= 0 (measured step-1 static floor + slack); got '${COMM_EFF_DC_TARGET}'." >&2; exit 1; }
  echo "=== resolved cvc/dc OK (before GPU): ce_lambda=$COMM_EFF_CVC_LAMBDA warmup=$COMM_EFF_CVC_WARMUP_STEPS dc=true eta=$COMM_EFF_DC_ETA target=$COMM_EFF_DC_TARGET lambda0=$COMM_EFF_DC_LAMBDA0 lambda_max=$COMM_EFF_DC_LAMBDA_MAX ==="
fi

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
# 7. Launch the retained GRPO surface directly. Every method flag comes from
#    env so the rank1-RELEX comparisons and dense control share one engine.
#
#    : launch the training in the BACKGROUND, capture its PID, start the
#    early-stop watcher bound to that PID, then `wait` on training explicitly.
#    The watcher self-terminates when training exits (guard 1), and the verified
#    cleanup path reaps its private process group exactly once (guard 2).
# ---------------------------------------------------------------------------
# set -x so train.log carries the fully-resolved main_ppo command line for
# capture_resolved_config.py. Secrets were sourced+exported far above, so they
# are never traced here (only the safe Hydra args expand under xtrace).
set -x
python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files="$TRAIN_FILE" \
  data.val_files="$TEST_FILE" \
  data.train_batch_size="$TRAIN_BATCH_SIZE" \
  data.max_prompt_length="$MAX_PROMPT_LENGTH" \
  data.max_response_length="$MAX_RESPONSE_LENGTH" \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr="$ACTOR_LR" \
  actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$PPO_MICRO_BATCH_SIZE_PER_GPU" \
  actor_rollout_ref.actor.use_kl_loss="$USE_KL_LOSS" \
  actor_rollout_ref.actor.kl_loss_coef="$KL_LOSS_COEF" \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff="$ENTROPY_COEFF" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$LOG_PROB_MICRO_BATCH_SIZE_PER_GPU" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="$ROLLOUT_TP" \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization="$ROLLOUT_GPU_MEM_UTIL" \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096 \
  actor_rollout_ref.rollout.n="$ROLLOUT_N" \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$LOG_PROB_MICRO_BATCH_SIZE_PER_GPU" \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  trainer.critic_warmup=0 \
  trainer.logger='["console","wandb"]' \
  trainer.project_name="$PROJECT_NAME" \
  trainer.experiment_name="$EXPERIMENT_NAME" \
  trainer.n_gpus_per_node="$NGPUS_PER_NODE" \
  trainer.nnodes="${NNODES:-1}" \
  trainer.save_freq="$SAVE_FREQ" \
  trainer.test_freq="$TEST_FREQ" \
  trainer.total_epochs="$TOTAL_EPOCHS" \
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
  algorithm.rollout_correction.rollout_is="$ROLLOUT_IS" \
  algorithm.rollout_correction.rollout_is_threshold="$ROLLOUT_IS_THRESHOLD" \
  algorithm.rollout_correction.rollout_is_batch_normalize="$ROLLOUT_IS_BATCH_NORMALIZE" \
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
  actor_rollout_ref.actor.comm_eff.mask.enabled="$COMM_EFF_MASK_ENABLED" \
  actor_rollout_ref.actor.comm_eff.mask.p="$COMM_EFF_MASK_P" \
  actor_rollout_ref.actor.comm_eff.mask.rescale="$COMM_EFF_MASK_RESCALE" \
  actor_rollout_ref.actor.comm_eff.mask.mask_recompute="$COMM_EFF_MASK_RECOMPUTE" \
  actor_rollout_ref.actor.comm_eff.mask.mask_reference="$COMM_EFF_MASK_REFERENCE" \
  actor_rollout_ref.actor.comm_eff.mask.seed="$COMM_EFF_MASK_SEED" \
  actor_rollout_ref.actor.comm_eff.mask.pp_size="$COMM_EFF_MASK_PP_SIZE" \
  actor_rollout_ref.actor.comm_eff.mask.rescale_mode="$COMM_EFF_MASK_RESCALE_MODE" \
  actor_rollout_ref.actor.comm_eff.mask.exact_k="$COMM_EFF_MASK_EXACT_K" \
  actor_rollout_ref.actor.comm_eff.mask.dense_every="$COMM_EFF_MASK_DENSE_EVERY" \
  actor_rollout_ref.actor.comm_eff.mask.antithetic="$COMM_EFF_MASK_ANTITHETIC" \
  actor_rollout_ref.actor.comm_eff.mask.frlr="$COMM_EFF_MASK_FRLR" \
  actor_rollout_ref.actor.comm_eff.mask.frlr_rank="$COMM_EFF_MASK_FRLR_RANK" \
  actor_rollout_ref.actor.comm_eff.mask.frlr_k="$COMM_EFF_MASK_FRLR_K" \
  actor_rollout_ref.actor.comm_eff.mask.frlr_unbiased="$COMM_EFF_MASK_FRLR_UNBIASED" \
  actor_rollout_ref.actor.comm_eff.mask.frlr_q_cadence="$COMM_EFF_MASK_FRLR_Q_CADENCE" \
  actor_rollout_ref.actor.comm_eff.quant.bits="$COMM_EFF_QUANT_BITS" \
  actor_rollout_ref.actor.comm_eff.quant.block_size="$COMM_EFF_QUANT_BLOCK_SIZE" \
  actor_rollout_ref.actor.comm_eff.quant.rounding="$COMM_EFF_QUANT_ROUNDING" \
  actor_rollout_ref.actor.comm_eff.quant.subset_k="$COMM_EFF_QUANT_SUBSET_K" \
  actor_rollout_ref.actor.comm_eff.probe.probe_every="$COMM_EFF_PROBE_EVERY" \
  actor_rollout_ref.actor.comm_eff.probe.ctrl_enabled="$COMM_EFF_PROBE_CTRL_ENABLED" \
  actor_rollout_ref.actor.comm_eff.probe.kl_target_floor="$COMM_EFF_PROBE_KL_TARGET_FLOOR" \
  actor_rollout_ref.actor.comm_eff.probe.kl_target_gain="$COMM_EFF_PROBE_KL_TARGET_GAIN" \
  actor_rollout_ref.actor.comm_eff.probe.ctrl_ki="$COMM_EFF_PROBE_CTRL_KI" \
  actor_rollout_ref.actor.comm_eff.probe.ctrl_kp="$COMM_EFF_PROBE_CTRL_KP" \
  actor_rollout_ref.actor.comm_eff.probe.ctrl_beta_min="$COMM_EFF_PROBE_CTRL_BETA_MIN" \
  actor_rollout_ref.actor.comm_eff.probe.ctrl_beta_max="$COMM_EFF_PROBE_CTRL_BETA_MAX" \
  actor_rollout_ref.actor.cvc_lambda="$COMM_EFF_CVC_LAMBDA" \
  actor_rollout_ref.actor.cvc_warmup_steps="$COMM_EFF_CVC_WARMUP_STEPS" \
  actor_rollout_ref.actor.comm_eff.dc.enabled="$COMM_EFF_DC_ENABLED" \
  actor_rollout_ref.actor.comm_eff.dc.eta="$COMM_EFF_DC_ETA" \
  actor_rollout_ref.actor.comm_eff.dc.target="$COMM_EFF_DC_TARGET" \
  actor_rollout_ref.actor.comm_eff.dc.lambda0="$COMM_EFF_DC_LAMBDA0" \
  actor_rollout_ref.actor.comm_eff.dc.lambda_max="$COMM_EFF_DC_LAMBDA_MAX" \
  actor_rollout_ref.actor.comm_eff.anchor.enabled="$COMM_EFF_ANCHOR_ENABLED" \
  actor_rollout_ref.actor.comm_eff.anchor.cadence="$COMM_EFF_ANCHOR_CADENCE" \
  actor_rollout_ref.actor.comm_eff.anchor.delay_K="$COMM_EFF_ANCHOR_DELAY_K" \
  actor_rollout_ref.actor.comm_eff.anchor.owns_q="$COMM_EFF_ANCHOR_OWNS_Q" \
  actor_rollout_ref.actor.comm_eff.anchor.replay_paired_batch="$COMM_EFF_ANCHOR_REPLAY_PAIRED_BATCH" \
  actor_rollout_ref.actor.comm_eff.anchor.batch_scope="$COMM_EFF_ANCHOR_BATCH_SCOPE" \
  actor_rollout_ref.actor.comm_eff.anchor.snapshot_device="$COMM_EFF_ANCHOR_SNAPSHOT_DEVICE" \
  actor_rollout_ref.actor.comm_eff.anchor.lookahead_anchor="$COMM_EFF_ANCHOR_LOOKAHEAD_ANCHOR" \
  actor_rollout_ref.actor.comm_eff.anchor.lookahead_mode="$COMM_EFF_ANCHOR_LOOKAHEAD_MODE" \
  actor_rollout_ref.actor.comm_eff.anchor.lookahead_strength="$COMM_EFF_ANCHOR_LOOKAHEAD_STRENGTH" \
  actor_rollout_ref.actor.comm_eff.anchor.lookahead_rollout_source="$COMM_EFF_ANCHOR_LOOKAHEAD_ROLLOUT_SOURCE" \
  actor_rollout_ref.actor.comm_eff.anchor.lookahead_window_snapshots="$COMM_EFF_ANCHOR_LOOKAHEAD_WINDOW_SNAPSHOTS" \
  actor_rollout_ref.actor.comm_eff.anchor.warmup_mode="$COMM_EFF_ANCHOR_WARMUP_MODE" \
  actor_rollout_ref.actor.comm_eff.anchor.lookahead_min_snapshots="$COMM_EFF_ANCHOR_LOOKAHEAD_MIN_SNAPSHOTS" \
  actor_rollout_ref.actor.comm_eff.anchor.lookahead_history_mode="$COMM_EFF_ANCHOR_LOOKAHEAD_HISTORY_MODE" \
  actor_rollout_ref.actor.comm_eff.anchor.lookahead_max_snapshots="$COMM_EFF_ANCHOR_LOOKAHEAD_MAX_SNAPSHOTS" \
  actor_rollout_ref.actor.comm_eff.anchor.opt_reset.enabled="$COMM_EFF_OPT_RESET_ENABLED" \
  actor_rollout_ref.actor.comm_eff.anchor.opt_reset.cadence="$COMM_EFF_OPT_RESET_CADENCE" \
  actor_rollout_ref.actor.comm_eff.anchor.opt_reset.mode="$COMM_EFF_OPT_RESET_MODE" \
  actor_rollout_ref.actor.comm_eff.anchor.opt_reset.beta1="$COMM_EFF_OPT_RESET_B1" \
  actor_rollout_ref.actor.comm_eff.anchor.opt_reset.beta2="$COMM_EFF_OPT_RESET_B2" \
  actor_rollout_ref.actor.comm_eff.anchor.opt_reset.scale_match="$COMM_EFF_OPT_RESET_SCALE_MATCH" \
  actor_rollout_ref.actor.comm_eff.spectral.enabled="$COMM_EFF_SPECTRAL_ENABLED" \
  actor_rollout_ref.actor.comm_eff.spectral.target_scope="$COMM_EFF_SPECTRAL_TARGET_SCOPE" \
  actor_rollout_ref.actor.comm_eff.spectral.diagnostics="$COMM_EFF_SPECTRAL_DIAGNOSTICS" \
  actor_rollout_ref.actor.comm_eff.spectral.beta_anc="$COMM_EFF_SPECTRAL_BETA_ANC" \
  actor_rollout_ref.actor.comm_eff.spectral.ema_device="$COMM_EFF_SPECTRAL_EMA_DEVICE" \
  actor_rollout_ref.actor.comm_eff.spectral.max_targets="$COMM_EFF_SPECTRAL_MAX_TARGETS" \
  actor_rollout_ref.actor.comm_eff.spectral.cadence="$COMM_EFF_SPECTRAL_CADENCE" \
  actor_rollout_ref.actor.comm_eff.spectral.signed_ema_alpha="$COMM_EFF_SPECTRAL_SIGNED_EMA_ALPHA" \
  actor_rollout_ref.actor.comm_eff.spectral.correction_mode="$COMM_EFF_SPECTRAL_CORRECTION_MODE" \
  actor_rollout_ref.actor.comm_eff.spectral.delayed_ef_lambda="$COMM_EFF_SPECTRAL_DELAYED_EF_LAMBDA" \
  actor_rollout_ref.actor.comm_eff.spectral.blend_eta="$COMM_EFF_SPECTRAL_BLEND_ETA" \
  actor_rollout_ref.actor.comm_eff.powersgd.rank="$COMM_EFF_POWERSGD_RANK" \
  actor_rollout_ref.actor.comm_eff.powersgd.seed="$COMM_EFF_POWERSGD_SEED" \
  actor_rollout_ref.actor.comm_eff.powersgd.pp_size="$COMM_EFF_POWERSGD_PP_SIZE" \
  actor_rollout_ref.actor.comm_eff.powersgd.update_cadence="$COMM_EFF_POWERSGD_UPDATE_CADENCE" \
  actor_rollout_ref.actor.comm_eff.powersgd.warm_start="$COMM_EFF_POWERSGD_WARM_START" \
  actor_rollout_ref.actor.comm_eff.powersgd.compress_recompute="$COMM_EFF_POWERSGD_COMPRESS_RECOMPUTE" \
  actor_rollout_ref.actor.comm_eff.powersgd.compress_reference="$COMM_EFF_POWERSGD_COMPRESS_REFERENCE" \
  actor_rollout_ref.actor.comm_eff.powersgd.sync_basis="$COMM_EFF_POWERSGD_SYNC_BASIS" \
  actor_rollout_ref.actor.comm_eff.powersgd.fast_q_bootstrap="$COMM_EFF_POWERSGD_FAST_Q_BOOTSTRAP" \
  actor_rollout_ref.actor.comm_eff.powersgd.qr_dtype="$COMM_EFF_POWERSGD_QR_DTYPE" \
  actor_rollout_ref.actor.comm_eff.powersgd.reortho_eps="$COMM_EFF_POWERSGD_REORTHO_EPS" \
  "${VLLM_ALLREDUCE_OVERRIDE[@]+"${VLLM_ALLREDUCE_OVERRIDE[@]}"}" \
  "${MASK_PBB_OVERRIDE[@]+"${MASK_PBB_OVERRIDE[@]}"}" \
  "${PROBE_TABLE_OVERRIDE[@]+"${PROBE_TABLE_OVERRIDE[@]}"}" \
  "$@" \
  > "$LOG" 2>&1 &
TRAIN_PID=$!
{ set +x; } 2>/dev/null   # stop tracing once the resolved command is in the log

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
