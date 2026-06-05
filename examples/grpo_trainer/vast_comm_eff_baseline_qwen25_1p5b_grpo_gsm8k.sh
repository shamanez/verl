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
#   naive clean cadence .... COMM_EFF_CLEAN_CADENCE    (0=OFF)  full (unmasked) grad every N steps
#   anchor ................. COMM_EFF_ANCHOR_ENABLED   (false)
#   spectral correction .... COMM_EFF_SPECTRAL_ENABLED (false)
#
# Defaults encode the project's findings on masked GRPO. NB a large grad_norm is
# a SYMPTOM, not the disease: Adam's update is scale-invariant (mhat/sqrt(vhat)
# cancels any constant gradient scaling) and bounded to ~order(lr), and verl
# grad-clips on top — so a big raw norm cannot itself "explode" the update. The
# two real failure modes are BIAS and VARIANCE.
#   * the mask is per-element, keyed on each token's stable (sample_id,
#     position_id) so it is packing-invariant across the differently-packed
#     old_logprob and train forwards (exact cross-pass consistency).
#   * rescale=true — inverted-dropout h*mask/(1-p) restores E[h*mask/(1-p)]=h,
#     i.e. an UNBIASED mask. This is the load-bearing correctness property, not
#     a "grad_norm tamer". WITHOUT it the mask is biased (E[h*mask]=(1-p)*h: the
#     forward sits off-distribution, the GRPO importance ratio is corrupted),
#     and Adam cannot fix a biased direction. DEFAULT ON.  ⚠ rescale is
#     NECESSARY but NOT sufficient — it trades bias for VARIANCE (~p/(1-p)), so
#     plain masked GRPO at high p is unbiased-but-noisy and still does not learn
#     in a short run. That variance is what the anchor + spectral + grad-clip
#     machinery (and a lower mask rate) exist to tame. Open: the mask-rate sweep
#     p=0.9 -> 0.5 -> 0.1.
#   * clean_cadence=0 (OFF). The periodic full-(unmasked)-gradient step is the
#     NAIVE cadence method; it is NOT sustainable — the masked steps stay
#     corrupted and the PPO clip fraction climbs toward saturation, so clipped
#     tokens stop contributing gradient. Opt-in knob only, do not ship it.
#   * anchor + spectral OFF — start from the mask-only path; layer these on only
#     after a masked config is shown to actually LEARN (val/score), and to
#     control the rescale's variance. Judge on learning, not on the grad_norm.
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
#   # mask-only, no rescale (the biased-mask A/B point):
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
#    "comm-eff baseline" (mask + rescale; cadence/anchor/spectral
#    OFF). Field names mirror verl/trainer/config/actor/actor.yaml exactly —
#    do NOT reference a knob absent from that schema (Hydra struct-mode rejects
#    unknown keys regardless of enabled flags; that bit us on clean_cadence).
# ---------------------------------------------------------------------------
COMM_EFF_ENABLED="${COMM_EFF_ENABLED:-true}"                          # master switch (false => dense)
# --- EXP-20/M6 codec selector: dense | prf_mask | powersgd ---
# "dense" (default) keeps the LEGACY behavior: the codec is selected by
# COMM_EFF_MASK_ENABLED below (mask on => prf_mask), so every prior comm-eff run
# is byte-unchanged. Set COMM_EFF_COMPRESSION_TYPE=powersgd to select the EXP-20
# PowerSGD activation projector instead. The mask arm and the powersgd arm thus
# call THIS SAME launcher with only the codec knobs differing (stability
# contract — never re-type the baseline).
COMM_EFF_COMPRESSION_TYPE="${COMM_EFF_COMPRESSION_TYPE:-dense}"
# --- activation mask ---
COMM_EFF_MASK_ENABLED="${COMM_EFF_MASK_ENABLED:-true}"
COMM_EFF_MASK_P="${COMM_EFF_MASK_P:-0.9}"                             # masked fraction (sweep 0.9->0.5->0.1, #15)
COMM_EFF_MASK_RESCALE="${COMM_EFF_MASK_RESCALE:-true}"               # inverted-dropout h*mask/(1-p)
COMM_EFF_MASK_RECOMPUTE="${COMM_EFF_MASK_RECOMPUTE:-true}"            # mask the old_logprob forward too
COMM_EFF_MASK_SEED="${COMM_EFF_MASK_SEED:-0}"                         # PRF base seed
COMM_EFF_MASK_PP_SIZE="${COMM_EFF_MASK_PP_SIZE:-8}"                   # simulated pipeline depth (boundary blocks)
# Fallback if training is unstable: try N unmasked warmup steps (not yet
# implemented) and/or COMM_EFF_MASK_RESCALE=true (theory's 1/(1-p), bf16-risky).
# --- naive periodic clean (unmasked) step: 0=OFF. NOT sustainable (PPO clip saturation). ---
COMM_EFF_CLEAN_CADENCE="${COMM_EFF_CLEAN_CADENCE:-0}"
# --- anchor circuit (OFF by default) ---
COMM_EFF_ANCHOR_ENABLED="${COMM_EFF_ANCHOR_ENABLED:-false}"
COMM_EFF_ANCHOR_CADENCE="${COMM_EFF_ANCHOR_CADENCE:-5}"
# EXP-16: staleness (in optimizer steps) of the weight snapshot the anchor
# forwards from. Config field already exists (comm_eff.anchor.delay_K, default
# 20, validated >=0); EXP-16 plumbs it through env. Default keeps the schema
# default so non-EXP-16 runs are unchanged.
COMM_EFF_ANCHOR_DELAY_K="${COMM_EFF_ANCHOR_DELAY_K:-20}"
# EXP-25 (R2): anchor-owns-Q. When true the ANCHOR owns the PowerSGD basis Q
# (fast maybe_update_basis + fast sketch gated OFF; anchor computes Q ← orth(V)
# from its slow-net stale-forward activations and broadcasts Q + M every refresh).
# false (default) = EXP-20 fast-owns-Q (byte-identical). Active iff
# COMM_EFF_COMPRESSION_TYPE=powersgd AND COMM_EFF_ANCHOR_ENABLED=true.
COMM_EFF_ANCHOR_OWNS_Q="${COMM_EFF_ANCHOR_OWNS_Q:-false}"
# --- spectral correction (OFF by default) ---
COMM_EFF_SPECTRAL_ENABLED="${COMM_EFF_SPECTRAL_ENABLED:-false}"
COMM_EFF_SPECTRAL_ALPHA="${COMM_EFF_SPECTRAL_ALPHA:-0.5}"
COMM_EFF_SPECTRAL_TAU="${COMM_EFF_SPECTRAL_TAU:-0.01}"
COMM_EFF_SPECTRAL_BETA_ANC="${COMM_EFF_SPECTRAL_BETA_ANC:-0.9}"
# EXP-16: spectral-correction cadence in optimizer steps. NEW config field
# (comm_eff.spectral.cadence). Default 1 = fire every step = the pre-EXP-16
# behavior (strict no-op for every prior config + the disabled path). Set to 2
# (aligned with COMM_EFF_ANCHOR_CADENCE=2) so the correction fires only on the
# steps the anchor EMA was just refreshed (a fresh basis, never a stale one).
COMM_EFF_SPECTRAL_CADENCE="${COMM_EFF_SPECTRAL_CADENCE:-1}"
COMM_EFF_SPECTRAL_EMA_DEVICE="${COMM_EFF_SPECTRAL_EMA_DEVICE:-gpu}"
COMM_EFF_SPECTRAL_SVD_MODE="${COMM_EFF_SPECTRAL_SVD_MODE:-full}"
COMM_EFF_SPECTRAL_BASIS_CACHE="${COMM_EFF_SPECTRAL_BASIS_CACHE:-cache}"
COMM_EFF_SPECTRAL_MAX_TARGETS="${COMM_EFF_SPECTRAL_MAX_TARGETS:-4}"
COMM_EFF_SPECTRAL_SEED_ANCHOR_CACHE="${COMM_EFF_SPECTRAL_SEED_ANCHOR_CACHE:-true}"
# EXP-18/M4 + EXP-23: spectral correction MODE and its two additive-mode knobs.
# These three config fields already exist + are validated in
# verl/workers/config/comm_eff.py (correction_mode L493-496, inject_gamma
# L498-499, blend_eta L502-503) and are read into the SpectralFilter in
# verl/workers/comm_eff/state.py (L385-387) — but the launcher did NOT pass them
# to Hydra, so an un-overridden run silently fell to the dataclass default
# correction_mode=reweight, which project evidence (EXP-21) proved INERT on this
# circuit (cos~0.5, G_filt≈0). Defaults here MIRROR the dataclass defaults
# EXACTLY (reweight / 1.0 / 0.5) so an un-overridden run is byte-unchanged; the
# EXP-23 arms set COMM_EFF_SPECTRAL_CORRECTION_MODE=inject (A2) / blend (A3).
COMM_EFF_SPECTRAL_CORRECTION_MODE="${COMM_EFF_SPECTRAL_CORRECTION_MODE:-reweight}"  # reweight (default, dataclass) | inject | blend | signed_ema (EXP-25/R3)
COMM_EFF_SPECTRAL_INJECT_GAMMA="${COMM_EFF_SPECTRAL_INJECT_GAMMA:-1.0}"             # inject force (correction_mode=inject); dataclass default 1.0
COMM_EFF_SPECTRAL_BLEND_ETA="${COMM_EFF_SPECTRAL_BLEND_ETA:-0.5}"                   # convex-blend weight (correction_mode=blend); dataclass default 0.5
# EXP-25 (R3): signed_ema merger weight alpha in G=alpha*G_noisy+(1-alpha)*|G_noisy|*sign(M).
# alpha=0 = pure sign-merger (SFT default); alpha=1 = G_noisy unchanged. THE swept axis.
# Active iff correction_mode=signed_ema. dataclass default 0.0.
COMM_EFF_SPECTRAL_SIGNED_EMA_ALPHA="${COMM_EFF_SPECTRAL_SIGNED_EMA_ALPHA:-0.0}"
# --- EXP-20/M6 PowerSGD activation compression (active iff
#     COMM_EFF_COMPRESSION_TYPE=powersgd). Defaults = the issue VII.1 candidate:
#     rank=102 (byte-matched to the PRF mask at p=0.95), warm block power
#     iteration every step, compress the old-logprob recompute (=> ρ≈1),
#     sync_basis=true (single shared consensus Q across DP ranks — REQUIRED
#     under DP), fp32 QR (REQUIRED — bf16-QR loses orthogonality). ---
COMM_EFF_POWERSGD_RANK="${COMM_EFF_POWERSGD_RANK:-102}"               # r; 102 ≡ p=0.95 (q·H=102.4)
COMM_EFF_POWERSGD_SEED="${COMM_EFF_POWERSGD_SEED:-0}"                 # per-layer basis seed base
COMM_EFF_POWERSGD_PP_SIZE="${COMM_EFF_POWERSGD_PP_SIZE:-8}"           # boundary blocks (same as mask)
COMM_EFF_POWERSGD_UPDATE_CADENCE="${COMM_EFF_POWERSGD_UPDATE_CADENCE:-1}"  # orth(V) every N steps
COMM_EFF_POWERSGD_WARM_START="${COMM_EFF_POWERSGD_WARM_START:-true}"  # carry Q across steps
COMM_EFF_POWERSGD_COMPRESS_RECOMPUTE="${COMM_EFF_POWERSGD_COMPRESS_RECOMPUTE:-true}"  # project old-logprob too
COMM_EFF_POWERSGD_SYNC_BASIS="${COMM_EFF_POWERSGD_SYNC_BASIS:-true}"  # all-reduce V across DP => single shared consensus Q (REQUIRED under DP)
COMM_EFF_POWERSGD_QR_DTYPE="${COMM_EFF_POWERSGD_QR_DTYPE:-fp32}"      # fp32 REQUIRED (INF-14); bf16 diagnostic
COMM_EFF_POWERSGD_REORTHO_EPS="${COMM_EFF_POWERSGD_REORTHO_EPS:-1e-6}"

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
  batching:            dynamic_bsz=$USE_DYNAMIC_BSZ  (when False: micro_batch_per_gpu=$PPO_MICRO_BATCH_SIZE_PER_GPU, max_tokens/GPU=$PPO_MAX_TOKEN_LEN_PER_GPU ignored)
  prompt / response:   $MAX_PROMPT_LENGTH / $MAX_RESPONSE_LENGTH
  epochs:              $TOTAL_EPOCHS  (save $SAVE_FREQ, validate $TEST_FREQ, total steps $TOTAL_TRAINING_STEPS)
  val_before_train:    $VAL_BEFORE_TRAIN
  objective:           pg_loss only (use_kl_loss=$USE_KL_LOSS, use_kl_in_reward=$USE_KL_IN_REWARD, entropy_coeff=$ENTROPY_COEFF)
  mismatch diag:       calculate_log_probs=$ROLLOUT_CALC_LOGPROBS (logs training/rollout_probs_diff_*); rollout correction STRICTLY OFF (recompute old_log_prob)
  comm_eff master:     $COMM_EFF_ENABLED
  compression_type:    $COMM_EFF_COMPRESSION_TYPE  (dense|prf_mask|powersgd; dense => legacy mask-by-flag)
  mask:                enabled=$COMM_EFF_MASK_ENABLED p=$COMM_EFF_MASK_P rescale=$COMM_EFF_MASK_RESCALE recompute=$COMM_EFF_MASK_RECOMPUTE seed=$COMM_EFF_MASK_SEED pp_size=$COMM_EFF_MASK_PP_SIZE
  powersgd:            rank=$COMM_EFF_POWERSGD_RANK update_cadence=$COMM_EFF_POWERSGD_UPDATE_CADENCE warm_start=$COMM_EFF_POWERSGD_WARM_START compress_recompute=$COMM_EFF_POWERSGD_COMPRESS_RECOMPUTE sync_basis=$COMM_EFF_POWERSGD_SYNC_BASIS qr_dtype=$COMM_EFF_POWERSGD_QR_DTYPE  (active iff compression_type=powersgd)
  clean_cadence:       $COMM_EFF_CLEAN_CADENCE  (0=off; naive periodic full-grad step — NOT sustainable)
  anchor:              enabled=$COMM_EFF_ANCHOR_ENABLED cadence=$COMM_EFF_ANCHOR_CADENCE delay_K=$COMM_EFF_ANCHOR_DELAY_K owns_q=$COMM_EFF_ANCHOR_OWNS_Q
  spectral:            enabled=$COMM_EFF_SPECTRAL_ENABLED alpha=$COMM_EFF_SPECTRAL_ALPHA tau=$COMM_EFF_SPECTRAL_TAU beta_anc=$COMM_EFF_SPECTRAL_BETA_ANC cadence=$COMM_EFF_SPECTRAL_CADENCE max_targets=$COMM_EFF_SPECTRAL_MAX_TARGETS seed_anchor_cache=$COMM_EFF_SPECTRAL_SEED_ANCHOR_CACHE ema_device=$COMM_EFF_SPECTRAL_EMA_DEVICE
  spectral correction: mode=$COMM_EFF_SPECTRAL_CORRECTION_MODE inject_gamma=$COMM_EFF_SPECTRAL_INJECT_GAMMA blend_eta=$COMM_EFF_SPECTRAL_BLEND_ETA signed_ema_alpha=$COMM_EFF_SPECTRAL_SIGNED_EMA_ALPHA
  wandb:               $PROJECT_NAME / $EXPERIMENT_NAME
  log:                 $LOG
=== launching ===
EOF

# ---------------------------------------------------------------------------
# 6b. EXP-16 early-stop instrumentation (greppable). A lightweight background
#     watcher tails the LIVE training log for the corrupting-failure patterns
#     the training-log-monitor kills a cell on (non-finite grad_norm/loss, FSDP
#     backward/hook errors, DTensor/aten::copy_ writeback errors, the
#     string-metric reduce crash, use_orig_params guard, spectral crashes). On
#     the FIRST match it writes a one-line `EARLY_STOP_SIGNAL: <pattern> @ <line>`
#     to the log AND a `runs/<EXP>/EARLY_STOP_SIGNAL` sentinel file, then exits.
#     This is a SIGNAL ONLY — it does NOT kill training (the monitor/runner owns
#     teardown). It just gives the monitor a single high-signal grep target so it
#     does not have to re-derive the regex from the rescue-trigger list. Strict
#     opt-in side effect: writes only into this run's own dir.
#
#     EXP-20 fix (CRITICAL — a clean run used to hang here forever): the old
#     watcher ran `tail -F | grep -m1` in a backgrounded subshell and the EXIT
#     trap killed only the SUBSHELL pid, orphaning the child `tail -F`. On a
#     CLEAN run grep -m1 never matches, `tail -F` follows the (now-idle) log
#     forever, the subshell never exits, and THIS SCRIPT blocks in its implicit
#     `wait` at end-of-script — so the back-to-back sequence could never finish
#     autonomously (GPUs idle at $/hr). Two independent guards now prevent that:
#       (1) `tail --pid="$TRAIN_PID" -F` — the follower DIES when the training
#           process exits (clean or crash), closing the pipe so grep hits EOF and
#           the subshell returns; nothing is left to wait on.
#       (2) the watcher runs in its OWN process group (setsid) and the EXIT trap
#           kills the WHOLE group (`kill -- -$PGID`), reaping tail+grep+subshell
#           even if (1) somehow doesn't fire.
#     Net: the watcher never leaves a dangling follower and never blocks this
#     script on clean completion — for EVERY cell in the sequence.
# ---------------------------------------------------------------------------
RUN_DIR="$(dirname "$LOG")"
EARLY_STOP_SENTINEL="$RUN_DIR/EARLY_STOP_SIGNAL"
rm -f "$EARLY_STOP_SENTINEL"
# Patterns mirror the plan's ## Rescue triggers (numeric-only stability + FSDP/
# spectral safety). \bnan/inf word-boundary guards avoid matching "infer"/
# "information". The watcher is a no-op until $LOG starts filling.
EARLY_STOP_RE='([Nn]a[Nn] detected|RuntimeError: .*use_orig_params|summon_full_params.*(error|Error|assert)|could not convert string to float|aten::copy_.*(mismatch|size)|torch\.distributed\.fsdp.*(error|Error)|(loss|grad_norm|pg_loss|policy_loss|reward)[^A-Za-z].{0,80}\b([Nn]a[Nn]|[Ii]nf)\b|\b([Nn]a[Nn]|[Ii]nf)\b.{0,40}(loss|grad_norm))'

# ---------------------------------------------------------------------------
# 7. Launch — reuse upstream's per-recipe script for the verbatim main_ppo
#    invocation, overriding the OOM-relevant + comm-eff Hydra knobs. Every
#    enabled flag comes from env so the full ablation grid is a one-liner.
#
#    EXP-20: launch the training in the BACKGROUND, capture its PID, start the
#    early-stop watcher bound to that PID, then `wait` on training explicitly.
#    The watcher self-terminates when training exits (guard 1), and the EXIT
#    trap reaps the watcher's whole process group (guard 2).
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
  actor_rollout_ref.actor.comm_eff.spectral.enabled="$COMM_EFF_SPECTRAL_ENABLED" \
  actor_rollout_ref.actor.comm_eff.spectral.alpha="$COMM_EFF_SPECTRAL_ALPHA" \
  actor_rollout_ref.actor.comm_eff.spectral.tau="$COMM_EFF_SPECTRAL_TAU" \
  actor_rollout_ref.actor.comm_eff.spectral.beta_anc="$COMM_EFF_SPECTRAL_BETA_ANC" \
  actor_rollout_ref.actor.comm_eff.spectral.seed_anchor_cache="$COMM_EFF_SPECTRAL_SEED_ANCHOR_CACHE" \
  actor_rollout_ref.actor.comm_eff.spectral.ema_device="$COMM_EFF_SPECTRAL_EMA_DEVICE" \
  actor_rollout_ref.actor.comm_eff.spectral.svd_mode="$COMM_EFF_SPECTRAL_SVD_MODE" \
  actor_rollout_ref.actor.comm_eff.spectral.basis_cache="$COMM_EFF_SPECTRAL_BASIS_CACHE" \
  actor_rollout_ref.actor.comm_eff.spectral.max_targets="$COMM_EFF_SPECTRAL_MAX_TARGETS" \
  actor_rollout_ref.actor.comm_eff.spectral.cadence="$COMM_EFF_SPECTRAL_CADENCE" \
  actor_rollout_ref.actor.comm_eff.spectral.correction_mode="$COMM_EFF_SPECTRAL_CORRECTION_MODE" \
  actor_rollout_ref.actor.comm_eff.spectral.inject_gamma="$COMM_EFF_SPECTRAL_INJECT_GAMMA" \
  actor_rollout_ref.actor.comm_eff.spectral.blend_eta="$COMM_EFF_SPECTRAL_BLEND_ETA" \
  actor_rollout_ref.actor.comm_eff.spectral.signed_ema_alpha="$COMM_EFF_SPECTRAL_SIGNED_EMA_ALPHA" \
  actor_rollout_ref.actor.comm_eff.powersgd.rank="$COMM_EFF_POWERSGD_RANK" \
  actor_rollout_ref.actor.comm_eff.powersgd.seed="$COMM_EFF_POWERSGD_SEED" \
  actor_rollout_ref.actor.comm_eff.powersgd.pp_size="$COMM_EFF_POWERSGD_PP_SIZE" \
  actor_rollout_ref.actor.comm_eff.powersgd.update_cadence="$COMM_EFF_POWERSGD_UPDATE_CADENCE" \
  actor_rollout_ref.actor.comm_eff.powersgd.warm_start="$COMM_EFF_POWERSGD_WARM_START" \
  actor_rollout_ref.actor.comm_eff.powersgd.compress_recompute="$COMM_EFF_POWERSGD_COMPRESS_RECOMPUTE" \
  actor_rollout_ref.actor.comm_eff.powersgd.sync_basis="$COMM_EFF_POWERSGD_SYNC_BASIS" \
  actor_rollout_ref.actor.comm_eff.powersgd.qr_dtype="$COMM_EFF_POWERSGD_QR_DTYPE" \
  actor_rollout_ref.actor.comm_eff.powersgd.reortho_eps="$COMM_EFF_POWERSGD_REORTHO_EPS" \
  "$@" \
  > "$LOG" 2>&1 &
TRAIN_PID=$!

# EXP-20 early-stop watcher — bound to TRAIN_PID + its own process group.
# Guard 1: `tail --pid="$TRAIN_PID" -F` dies when training exits (clean or
# crash), so grep hits EOF and the watcher subshell returns. Guard 2: setsid
# puts the watcher in its own pgroup; the EXIT trap kills the whole group so
# nothing dangles. The watcher only SIGNALS (sentinel + log line); the
# monitor/runner owns teardown.
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
# Reap the watcher's WHOLE process group on exit (guard 2). setsid makes the
# watcher a group leader, so its PGID == its PID; kill -- -PGID reaps tail+grep
# too. `|| true` so a self-terminated watcher (guard 1 already fired) is fine.
trap '[[ -n "${EARLY_STOP_WATCHER_PID:-}" ]] && kill -- -"$EARLY_STOP_WATCHER_PID" 2>/dev/null; true' EXIT

# Block on TRAINING ONLY (not the watcher) — when main_ppo finishes, we proceed
# to done.flag immediately; the EXIT trap then reaps the watcher. `wait $PID`
# returns the child's exit status; capture it WITHOUT tripping `set -e` (the
# `|| TRAIN_RC=$?` keeps a non-zero training exit from aborting before we can
# clean up the watcher + write done.flag + propagate the status).
TRAIN_RC=0
wait "$TRAIN_PID" || TRAIN_RC=$?

# Proactively stop the watcher now that training is done (belt-and-braces with
# the EXIT trap + guard 1), so back-to-back cells never accumulate watchers.
kill -- -"$EARLY_STOP_WATCHER_PID" 2>/dev/null || true

# ---------------------------------------------------------------------------
# WandB final-flush (fix for the truncated-dashboard bug). verl/Ray SIGKILLs
# its workers at teardown before WandB's async uploader flushes the LAST batch,
# so the online run is cut ~2 steps short — the final (step-N) validation point
# never lands and the run is marked "crashed" (observed across EXP-20 + the
# dense control: lastHistoryStep=48 of 50, summary stuck at the step-40/25 val).
# The COMPLETE history is always in the local .wandb file, so re-sync it here —
# AFTER training, BEFORE done.flag (which gates the orchestrator's teardown) —
# to upload the missing tail and finish the run cleanly. Best-effort + bounded:
# never blocks done.flag (the local train.log stays the authoritative record).
if command -v wandb >/dev/null 2>&1; then
  WANDB_RUN_DIR=$(ls -dt /workspace/verl/wandb/run-* /workspace/verl/wandb/offline-run-* 2>/dev/null | head -1 || true)
  if [[ -n "${WANDB_RUN_DIR:-}" ]]; then
    echo "=== wandb sync $WANDB_RUN_DIR (flush final history before teardown) ==="
    timeout 240 wandb sync "$WANDB_RUN_DIR" 2>&1 | tail -8 \
      || echo "WARN: wandb sync failed/timed out — final point may be missing online; local train.log is authoritative" >&2
  fi
fi

touch "/workspace/verl/runs/${EXPERIMENT_NAME}/done.flag"
echo "=== done at $(date -u +%FT%TZ) (train_rc=$TRAIN_RC) ==="
# Propagate the training exit status so the EXP-20 launch.sh `run_step` sees a
# real failure (set -e / `|| true` semantics in the driver still apply).
exit "$TRAIN_RC"
