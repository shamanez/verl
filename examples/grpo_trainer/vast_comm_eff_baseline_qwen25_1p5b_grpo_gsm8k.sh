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
# THE canonical communication-efficient base launcher. As of issue #25
# (EXP-25, 2026-06-09) the comm-eff base is the ANCHOR CIRCUIT on a PowerSGD
# codec, and the defaults below ENCODE it. Every circuit is still an
# independent env toggle so ablations stay one-liners:
#
#   comm-eff master ........ COMM_EFF_ENABLED          (true)     off => byte-identical dense
#   codec .................. COMM_EFF_COMPRESSION_TYPE (powersgd) the locked compressor
#   PowerSGD rank .......... COMM_EFF_POWERSGD_RANK    (77)       byte-matched to mask p=0.95 (H=1536)
#   anchor (MANDATORY) ..... COMM_EFF_ANCHOR_ENABLED   (true)     stale full-grad reference M
#   anchor owns Q .......... COMM_EFF_ANCHOR_OWNS_Q    (true)     the ONLY thing that updates Q
#   anchor staleness ....... COMM_EFF_ANCHOR_DELAY_K   (5)        forward from theta_{t-5}
#   anchor refresh ......... COMM_EFF_ANCHOR_CADENCE   (5)        recompute M+Q every 5 ticks
#   merger ................. COMM_EFF_SPECTRAL_ENABLED (true)     signed_ema fold of M into G
#   merger weight alpha .... COMM_EFF_SPECTRAL_SIGNED_EMA_ALPHA (0.5)
#   naive clean cadence .... COMM_EFF_CLEAN_CADENCE    (0=OFF)    DEAD — the anchor replaced it
#   legacy mask ............ COMM_EFF_MASK_ENABLED     (false)    prf_mask codec, reference only
#
# THE BASE in one line: PowerSGD r=77 + a continuously-maintained, delay_K=5
# stale, full-coverage (196 matrices, DP-reduced) anchor gradient EMA M,
# refreshed every 5 ticks from a no-hook isolated clone; the anchor OWNS the
# PowerSGD basis Q (computes Q<-orth(V) from its stale-forward activations and
# broadcasts it — the fast circuit is a read-only consumer, fail-closed from
# ever writing Q); the signed_ema merger folds M into the fast gradient as
# G = alpha*G_noisy + (1-alpha)*|G_noisy|*sign(M). These exact values are the
# EXP-25 ground truth (runs/EXP-25/resolved_params.txt, alpha=0.5 arm).
#
# WHY this base + the honest result (do NOT restate numbers here — they live in
# research/runs/SUMMARY.md): the anchor is the REALISTIC decentralized-PP
# setting — a continuously-maintained stale anchor replaces the old clean_cadence
# periodic-dense-step crutch, which was unrealizable (full-H transfer + itself
# stale on a slow link). The circuit is mechanically PROVEN (R1 full-coverage
# DP-reduced M + R2 anchor-owns-Q probe gates green). The signed_ema MERGER,
# however, is FALSIFIED — net-harmful vs plain PowerSGD (#25 verdict STOP). So
# the substrate is the settled base and the MERGER (correction_mode + its
# weights) is the open research axis you sweep from here; the substrate is held
# fixed. Grad_norm is a SYMPTOM not the disease (Adam is scale-invariant + verl
# grad-clips); judge on val/critic-score, not grad_norm.
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
# Hardware: multi-GPU only (4..8). The mandatory anchor allocates a ~3 GB
# no-hook clone/rank for its stale forward-backward, so the default actor token
# budget is already halved (PPO_MAX_TOKEN_LEN_PER_GPU=18432) to fit 4×H200; this
# is the EXP-25 footprint. Disabling the anchor (COMM_EFF_ANCHOR_ENABLED=false,
# a reference-only ablation) frees the clone and you can raise it back to 36864.
#
# Ablation examples (the substrate is held fixed; the MERGER is the axis):
#   # sweep the merger weight (the research axis):
#   COMM_EFF_SPECTRAL_SIGNED_EMA_ALPHA=0.7 EXPERIMENT_NAME=ce_a0p7 bash <thisfile>
#   # dense control via the same launcher (master switch off => byte-identical):
#   COMM_EFF_ENABLED=false EXPERIMENT_NAME=ce_off_dense bash <thisfile>
#   # legacy prf_mask codec (reference only — NOT the base; cannot anchor-own-Q):
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
#    `actor.fsdp_config.use_orig_params=true` (REQUIRED so the anchor + merger
#    hooks see full 2D Tensor gradients post-FSDP-reduce — the base needs it).
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

# Token budget per micro-batch for dynamic batching. Actor budget halved to
# 18432 (from 36864) to fit the mandatory anchor's ~3 GB clone on 4×H200 (the
# EXP-25 footprint); log_prob/ref keep 36864 (no clone on those paths).
PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-18432}"
LOG_PROB_MAX_TOKEN_LEN_PER_GPU="${LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-36864}"
REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU="${REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-36864}"

# ---------------------------------------------------------------------------
# 6. Communication-efficient method — hydra knob surface (see header).
#    Every circuit is an independent env toggle. Defaults = the ANCHOR-CIRCUIT
#    base (PowerSGD r=77 + anchor on + anchor-owns-Q + signed_ema merger; the
#    EXP-25 ground truth, runs/EXP-25/resolved_params.txt). Field names mirror
#    verl/trainer/config/actor/actor.yaml exactly — do NOT reference a knob
#    absent from that schema (Hydra struct-mode rejects unknown keys regardless
#    of enabled flags; that bit us on clean_cadence).
# ---------------------------------------------------------------------------
COMM_EFF_ENABLED="${COMM_EFF_ENABLED:-true}"                          # master switch (false => dense)
# --- codec selector: dense | prf_mask | powersgd ---
# powersgd (default) = the locked compressor for the anchor base; it is the ONLY
# codec compatible with anchor-owns-Q. prf_mask is retained as a reference-only
# codec (COMM_EFF_COMPRESSION_TYPE=prf_mask + COMM_EFF_MASK_ENABLED=true +
# COMM_EFF_ANCHOR_OWNS_Q=false); "dense" selects the codec by COMM_EFF_MASK_ENABLED
# (legacy). All arms call THIS SAME launcher with only the codec/merger knobs
# differing (stability contract — never re-type the baseline).
COMM_EFF_COMPRESSION_TYPE="${COMM_EFF_COMPRESSION_TYPE:-powersgd}"
# --- activation mask (reference-only codec; OFF in the PowerSGD anchor base) ---
COMM_EFF_MASK_ENABLED="${COMM_EFF_MASK_ENABLED:-false}"
COMM_EFF_MASK_P="${COMM_EFF_MASK_P:-0.9}"                             # masked fraction (sweep 0.9->0.5->0.1, #15)
COMM_EFF_MASK_RESCALE="${COMM_EFF_MASK_RESCALE:-true}"               # inverted-dropout h*mask/(1-p)
COMM_EFF_MASK_RECOMPUTE="${COMM_EFF_MASK_RECOMPUTE:-true}"            # mask the old_logprob forward too
COMM_EFF_MASK_SEED="${COMM_EFF_MASK_SEED:-0}"                         # PRF base seed
COMM_EFF_MASK_PP_SIZE="${COMM_EFF_MASK_PP_SIZE:-8}"                   # simulated pipeline depth (boundary blocks)
# Fallback if training is unstable: try N unmasked warmup steps (not yet
# implemented) and/or COMM_EFF_MASK_RESCALE=true (theory's 1/(1-p), bf16-risky).
# --- naive periodic clean (unmasked) step: 0=OFF. DEAD — the anchor replaced it
#     (a full-rank clean@K is not realizable on a slow link + is itself stale).
#     Leave at 0; do not re-enable in the base. ---
COMM_EFF_CLEAN_CADENCE="${COMM_EFF_CLEAN_CADENCE:-0}"
# --- anchor circuit (MANDATORY — the comm-eff base; EXP-25) ---
COMM_EFF_ANCHOR_ENABLED="${COMM_EFF_ANCHOR_ENABLED:-true}"
# how OFTEN the anchor recomputes M+Q, in optimizer/mini-batch TICKS (not global
# steps: train_batch 128 / ppo_mini 64 = 2 ticks/step, so cadence 5 ≈ every 2.5 steps).
COMM_EFF_ANCHOR_CADENCE="${COMM_EFF_ANCHOR_CADENCE:-5}"
# staleness: the anchor forwards from a delay_K-tick-stale weight snapshot.
# delay_K=5 is the canonical staleness (matches cadence). NB the schema default
# is 20 — the base PINS it to 5 here; do not rely on the schema default.
COMM_EFF_ANCHOR_DELAY_K="${COMM_EFF_ANCHOR_DELAY_K:-5}"
# EXP-25 (R2): anchor-owns-Q — THE structural inversion + a mandatory base
# property. true (default): the ANCHOR is the ONLY thing that updates the
# PowerSGD basis Q (fast maybe_update_basis + fast sketch gated OFF, fail-closed;
# anchor computes Q ← orth(V) from its stale-forward activations and broadcasts
# Q + M every refresh). Set false only for the reference-only fast-owns-Q
# ablation. Active iff COMM_EFF_COMPRESSION_TYPE=powersgd AND anchor enabled.
COMM_EFF_ANCHOR_OWNS_Q="${COMM_EFF_ANCHOR_OWNS_Q:-true}"
# --- anchor-guided gradient correction = the MERGER (ON in the base; EXP-25/R3).
#     The merger is the open RESEARCH AXIS — signed_ema is falsified (why + the
#     numbers: research/runs/SUMMARY.md); sweep correction_mode + its weights from
#     here. The combiners consult only the anchor-gradient EMA M_anchor. ---
COMM_EFF_SPECTRAL_ENABLED="${COMM_EFF_SPECTRAL_ENABLED:-true}"
COMM_EFF_SPECTRAL_BETA_ANC="${COMM_EFF_SPECTRAL_BETA_ANC:-0.95}"       # M EMA decay (matches the SL reference)
# Correction cadence in optimizer ticks. 1 = fire every tick (the merger must
# fire every fast step).
COMM_EFF_SPECTRAL_CADENCE="${COMM_EFF_SPECTRAL_CADENCE:-1}"
COMM_EFF_SPECTRAL_EMA_DEVICE="${COMM_EFF_SPECTRAL_EMA_DEVICE:-cpu}"    # offload full-coverage M (OOM guard)
# Cap on target matrices corrected per tick. -1 = no cap = full coverage of all
# 196 projection matrices the merger corrects (a >=0 cap silently drops merger
# targets — diagnostic throttle only).
COMM_EFF_SPECTRAL_MAX_TARGETS="${COMM_EFF_SPECTRAL_MAX_TARGETS:--1}"
# EXP-25 (R3): merger MODE. signed_ema (default) = the live merger; inject/blend
# remain as alternate combiners. Validated in verl/workers/config/comm_eff.py,
# read into the SpectralFilter in verl/workers/comm_eff/state.py.
COMM_EFF_SPECTRAL_CORRECTION_MODE="${COMM_EFF_SPECTRAL_CORRECTION_MODE:-signed_ema}"  # signed_ema | inject | blend
COMM_EFF_SPECTRAL_INJECT_GAMMA="${COMM_EFF_SPECTRAL_INJECT_GAMMA:-1.0}"             # force when correction_mode=inject
COMM_EFF_SPECTRAL_BLEND_ETA="${COMM_EFF_SPECTRAL_BLEND_ETA:-0.5}"                   # weight when correction_mode=blend
# EXP-25 (R3): signed_ema merger weight alpha in G=alpha*G_noisy+(1-alpha)*|G_noisy|*sign(M).
# alpha=0 = pure sign-merger (collapses); alpha=1 = G_noisy unchanged (= no merge).
# 0.5 (default) = the EXP-25 best / resolved-config arm. THE swept research axis.
COMM_EFF_SPECTRAL_SIGNED_EMA_ALPHA="${COMM_EFF_SPECTRAL_SIGNED_EMA_ALPHA:-0.5}"
# --- EXP-26 Step B: ef_powersgd merger (direction-PRESERVING error-feedback) ---
# Select with COMM_EFF_SPECTRAL_CORRECTION_MODE=ef_powersgd. Re-adds the dropped
# off-subspace residual to G_comp with NO sign term (keeps G_comp's direction).
# ef_decay=ef_clip=0 (defaults) ⇒ G_corr==G_comp = the plain-PowerSGD limiting
# case (the EF-residual-disabled ablation). A live arm sets ef_clip>0 (residual
# norm cap as a fraction of ||G_comp||) and optionally ef_decay in [0,1).
COMM_EFF_SPECTRAL_EF_DECAY="${COMM_EFF_SPECTRAL_EF_DECAY:-0.0}"
COMM_EFF_SPECTRAL_EF_CLIP="${COMM_EFF_SPECTRAL_EF_CLIP:-0.0}"
# --- EXP-31 Cell D: additive stale-anchor rank-r_sb sub-basis (delayed_ef) ---
# delta_subbasis_rank > 0 ADDS rank_{r_sb}(S) into the delayed_ef correction term
# (S = the act-deflated stale weight gradient δ when family=tail, the default; or
# the raw stale anchor gradient M_rep when family=grad). The forward codec Q is
# untouched (Step-C avoided by construction). 0 (default, OFF) ⇒ the merger is the
# EXACT B2 path (off-path parity — preserves every existing run incl. Cell A).
# Cell D production: COMM_EFF_SPECTRAL_DELTA_SUBBASIS_RANK=2 (family=tail).
COMM_EFF_SPECTRAL_DELTA_SUBBASIS_RANK="${COMM_EFF_SPECTRAL_DELTA_SUBBASIS_RANK:-0}"
COMM_EFF_SPECTRAL_DELTA_SUBBASIS_FAMILY="${COMM_EFF_SPECTRAL_DELTA_SUBBASIS_FAMILY:-tail}"  # tail | grad
# EXP-31 Cell D γ-knob (over-amplification fix): the sub-basis WEIGHT γ + a HOLD-
# then-DECAY schedule. γ holds at full WEIGHT for HOLD_STEPS, THEN decays linearly
# over DECAY_STEPS: γ_t = WEIGHT*(1 if step<HOLD_STEPS else max(0, 1 -
# (step-HOLD_STEPS)/DECAY_STEPS)) (constant WEIGHT when DECAY_STEPS=0). WEIGHT=1.0 +
# DECAY_STEPS=0 (defaults) = the EXACT current Cell D behaviour (γ_t≡1); WEIGHT=0 ⇒
# B2; HOLD_STEPS=0 (default) = the existing linear-from-step-0 decay (bitwise).
# γ-decay-over-full-run: WEIGHT=1.0 DECAY_STEPS=50 HOLD_STEPS=0. Hold-then-decay
# (preserve r2's early lead AND finish clean): WEIGHT=1.0 HOLD_STEPS=25 DECAY_STEPS=25.
# Constant half-dose: WEIGHT=0.5 DECAY_STEPS=0. (active iff mode=delayed_ef + rank>0).
COMM_EFF_SPECTRAL_DELTA_SUBBASIS_WEIGHT="${COMM_EFF_SPECTRAL_DELTA_SUBBASIS_WEIGHT:-1.0}"
COMM_EFF_SPECTRAL_DELTA_SUBBASIS_DECAY_STEPS="${COMM_EFF_SPECTRAL_DELTA_SUBBASIS_DECAY_STEPS:-0}"
COMM_EFF_SPECTRAL_DELTA_SUBBASIS_HOLD_STEPS="${COMM_EFF_SPECTRAL_DELTA_SUBBASIS_HOLD_STEPS:-0}"
# --- EXP-31 surpass lever: zero-mean tunable cross-rank-identical perturbation ---
# perturb_sigma > 0 ADDS σ·‖G_corr‖·ξ (ξ a unit Gaussian seeded by (perturb_seed,
# target, step) — identical on every DP rank, fresh per step ⇒ zero-mean) AFTER the
# delayed_ef correction term: SGLD/SAM-style beneficial noise → flatter minima →
# potentially beats dense on greedy val. 0.0 (default, OFF) ⇒ G_corr is the EXACT
# delayed_ef / Cell-D path (composes with DELTA_SUBBASIS_RANK=0 ⇒ bitwise-B2). Local,
# ZERO added communication. The surpass sweep: COMM_EFF_SPECTRAL_PERTURB_SIGMA in
# {0.05, 0.10, 0.20} on the B2 substrate (active iff mode=delayed_ef).
COMM_EFF_SPECTRAL_PERTURB_SIGMA="${COMM_EFF_SPECTRAL_PERTURB_SIGMA:-0.0}"
COMM_EFF_SPECTRAL_PERTURB_SEED="${COMM_EFF_SPECTRAL_PERTURB_SEED:-0}"
# --- EXP-31 Cell C: correction-δ compression rank (SECONDARY savings) ---
# r_delta > 0 compresses the correction δ to r_delta columns before injection
# (the Cell C residual-codec savings cell; a SEPARATE later code change). 0
# (default, OFF) ⇒ δ injected uncompressed (the B2 / Cell D path).
COMM_EFF_SPECTRAL_R_DELTA="${COMM_EFF_SPECTRAL_R_DELTA:-0}"
# --- EXP-26 Step C: Q-basis FAMILY (content of orth(V) at FIXED rank) ---
# "act" (default) = the EXP-25 activation-energy basis (byte-identical substrate).
# RLVR-native families {grad,adv,tail,hybrid,ticket} are IMPLEMENTED (EXP-26 Step
# C1); the compressor fails loud on an UN-implemented family only. This is the LIVE
# basis the fast/training path consumes (the C2/Step-B LIVE-family arm). The C1
# screen keeps this "act" LIVE while families accumulate PASSIVELY (below). Steps
# A/B (with q_basis=act) are byte-identical to the substrate.
COMM_EFF_POWERSGD_Q_BASIS="${COMM_EFF_POWERSGD_Q_BASIS:-act}"
# --- EXP-26 Step C1 PASSIVE screen: families to passively accumulate inside the
#     anchor pass (off the live Q / fast path / optimizer) so ONE short run builds
#     candidate bases for ALL families at once. Hydra list literal, e.g.
#     COMM_EFF_POWERSGD_Q_BASIS_PASSIVE='[act,grad,adv,tail,hybrid,ticket]'.
#     Empty default => no passive accumulation (byte-identical). ---
COMM_EFF_POWERSGD_Q_BASIS_PASSIVE="${COMM_EFF_POWERSGD_Q_BASIS_PASSIVE:-[]}"
# --- EXP-26 Step C1 hybrid family column split at FIXED rank r (act + grad == r).
#     -1/-1 (default) = AUTO: act=ceil(r/2), grad=r-act (39 + 38 = 77 at r=77, the
#     STEP_C_SPEC.md split). Set BOTH explicitly (>=0, summing to r) to override. ---
COMM_EFF_POWERSGD_HYBRID_ACT_COLS="${COMM_EFF_POWERSGD_HYBRID_ACT_COLS:--1}"
COMM_EFF_POWERSGD_HYBRID_GRAD_COLS="${COMM_EFF_POWERSGD_HYBRID_GRAD_COLS:--1}"
# --- EXP-26 Step A: real-gradient geometry-audit tensor capture (OFF by default;
#     a strict no-op ⇒ byte-identical to the EXP-25 substrate). When enabled the
#     comm-eff hooks dump fp32 tensors (A/Â/Q, G_comp, G_corr, M/G_anchor, the
#     parallel G_dense, the delay_K=0 fresh-anchor probe) keyed by (global_step,
#     optimizer_tick, target_name, shape, dtype, norm) under CAPTURE_DIR. Every
#     dump is detached/dump-only — NO numerical side effect on the optimizer. ---
COMM_EFF_CAPTURE_ENABLED="${COMM_EFF_CAPTURE_ENABLED:-false}"
COMM_EFF_CAPTURE_DIR="${COMM_EFF_CAPTURE_DIR:-/workspace/captures}"   # rsynced to runs/EXP-26/captures/
COMM_EFF_CAPTURE_MAX_TICKS="${COMM_EFF_CAPTURE_MAX_TICKS:-10}"        # audit needs ~5-10 ticks
COMM_EFF_CAPTURE_STRATIFIED="${COMM_EFF_CAPTURE_STRATIFIED:-0}"       # >0 => N targets/matrix-type (volume guard)
COMM_EFF_CAPTURE_G_DENSE="${COMM_EFF_CAPTURE_G_DENSE:-false}"         # parallel uncompressed G_dense backward (highest-OOM-risk probe)
COMM_EFF_CAPTURE_FRESH_ANCHOR="${COMM_EFF_CAPTURE_FRESH_ANCHOR:-false}"  # delay_K=0 fresh-anchor measurement probe (the Option-A dense reference)
# EXP-26 Step C/B should-have: loss for the delay_K=0 fresh-anchor probe.
# clean_pg (default, ratio≡1 like the anchor refresh) | ppo_clip (the fast path's
# PPO ratio/clip loss — removes the loss-mismatch confound; gives a clean
# cos(G_fresh_ppo, G_corr) direction test). Dump-only probe; never the optimizer.
COMM_EFF_CAPTURE_FRESH_ANCHOR_LOSS="${COMM_EFF_CAPTURE_FRESH_ANCHOR_LOSS:-clean_pg}"
COMM_EFF_CAPTURE_DUMP_DTYPE="${COMM_EFF_CAPTURE_DUMP_DTYPE:-fp32}"    # fp32 REQUIRED for the fidelity invariant
# EXP-26 (bug #7 fix): min_tick skips cold-Q ticks so the max_ticks budget holds
# the POST-warm anchor fires (where G_fresh_anchor pairs with warm-Q G_comp/G_corr
# = the H1 inputs). Was silently dropped before (declared in config/yaml/capture.py
# but NEVER wired here). 0 = capture from start.
COMM_EFF_CAPTURE_MIN_TICK="${COMM_EFF_CAPTURE_MIN_TICK:-0}"
COMM_EFF_CAPTURE_RANK0_ONLY="${COMM_EFF_CAPTURE_RANK0_ONLY:-true}"    # capture rank0 only (disk guard); default true
# --- PowerSGD activation compression (the base codec). Defaults: rank=77
#     (byte-matched to the prf_mask at p=0.95 for H=1536: 0.05·1536≈77), block
#     power iteration, compress the old-logprob recompute (=> ρ≈1), sync_basis=true
#     (single shared consensus Q across DP — REQUIRED under DP), fp32 QR (REQUIRED
#     — bf16-QR loses orthogonality). NB Q is updated by the ANCHOR (owns_q=true),
#     NOT by the fast update_cadence path (which is gated off in the base). ---
COMM_EFF_POWERSGD_RANK="${COMM_EFF_POWERSGD_RANK:-77}"               # r=77 ≡ p=0.95 (0.05·H, H=1536)
COMM_EFF_POWERSGD_SEED="${COMM_EFF_POWERSGD_SEED:-0}"                 # per-layer basis seed base
COMM_EFF_POWERSGD_PP_SIZE="${COMM_EFF_POWERSGD_PP_SIZE:-8}"           # boundary blocks (same as mask)
COMM_EFF_POWERSGD_UPDATE_CADENCE="${COMM_EFF_POWERSGD_UPDATE_CADENCE:-1}"  # orth(V) every N steps
COMM_EFF_POWERSGD_WARM_START="${COMM_EFF_POWERSGD_WARM_START:-true}"  # carry Q across steps
COMM_EFF_POWERSGD_COMPRESS_RECOMPUTE="${COMM_EFF_POWERSGD_COMPRESS_RECOMPUTE:-true}"  # project old-logprob too
COMM_EFF_POWERSGD_SYNC_BASIS="${COMM_EFF_POWERSGD_SYNC_BASIS:-true}"  # all-reduce V across DP => single shared consensus Q (REQUIRED under DP)
COMM_EFF_POWERSGD_QR_DTYPE="${COMM_EFF_POWERSGD_QR_DTYPE:-fp32}"      # fp32 REQUIRED (INF-14); bf16 diagnostic
COMM_EFF_POWERSGD_REORTHO_EPS="${COMM_EFF_POWERSGD_REORTHO_EPS:-1e-6}"

if [[ "${COMM_EFF_ANCHOR_ENABLED}" == "true" ]]; then
  echo "INFO: anchor ON (the base) -> ~3 GB no-hook clone/rank; the default PPO_MAX_TOKEN_LEN_PER_GPU=18432 fits 4×H200. If you raise it, prefer 8×GPU." >&2
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
  spectral:            enabled=$COMM_EFF_SPECTRAL_ENABLED beta_anc=$COMM_EFF_SPECTRAL_BETA_ANC cadence=$COMM_EFF_SPECTRAL_CADENCE max_targets=$COMM_EFF_SPECTRAL_MAX_TARGETS ema_device=$COMM_EFF_SPECTRAL_EMA_DEVICE
  spectral correction: mode=$COMM_EFF_SPECTRAL_CORRECTION_MODE inject_gamma=$COMM_EFF_SPECTRAL_INJECT_GAMMA blend_eta=$COMM_EFF_SPECTRAL_BLEND_ETA signed_ema_alpha=$COMM_EFF_SPECTRAL_SIGNED_EMA_ALPHA
  ef_powersgd (EXP-26): ef_decay=$COMM_EFF_SPECTRAL_EF_DECAY ef_clip=$COMM_EFF_SPECTRAL_EF_CLIP  (active iff mode=ef_powersgd; 0/0 => G_corr==G_comp)
  subbasis (EXP-31 D):  delta_subbasis_rank=$COMM_EFF_SPECTRAL_DELTA_SUBBASIS_RANK family=$COMM_EFF_SPECTRAL_DELTA_SUBBASIS_FAMILY r_delta=$COMM_EFF_SPECTRAL_R_DELTA  (active iff mode=delayed_ef; rank=0 => correction==delta = B2)
  subbasis γ-knob:      delta_subbasis_weight=$COMM_EFF_SPECTRAL_DELTA_SUBBASIS_WEIGHT decay_steps=$COMM_EFF_SPECTRAL_DELTA_SUBBASIS_DECAY_STEPS hold_steps=$COMM_EFF_SPECTRAL_DELTA_SUBBASIS_HOLD_STEPS  (γ_t=weight*(1 if step<hold_steps else max(0,1-(step-hold_steps)/decay_steps)); weight=1+decay_steps=0 => γ≡1 = current Cell D; weight=0 => B2; hold_steps=0 => linear-from-0 decay)
  perturb (EXP-31):     perturb_sigma=$COMM_EFF_SPECTRAL_PERTURB_SIGMA perturb_seed=$COMM_EFF_SPECTRAL_PERTURB_SEED  (active iff mode=delayed_ef; σ=0 => g_corr unperturbed = B2/Cell-D; σ>0 adds σ·‖g_corr‖·unit-ξ, ξ seeded by (seed,target,step) => cross-rank identical, zero-mean over steps)
  q_basis (EXP-26):    live=$COMM_EFF_POWERSGD_Q_BASIS  passive=$COMM_EFF_POWERSGD_Q_BASIS_PASSIVE  hybrid=($COMM_EFF_POWERSGD_HYBRID_ACT_COLS+$COMM_EFF_POWERSGD_HYBRID_GRAD_COLS)  (act=byte-identical; C1 screen: live act + passive families)
  capture (EXP-26-A):  enabled=$COMM_EFF_CAPTURE_ENABLED dir=$COMM_EFF_CAPTURE_DIR max_ticks=$COMM_EFF_CAPTURE_MAX_TICKS min_tick=$COMM_EFF_CAPTURE_MIN_TICK stratified=$COMM_EFF_CAPTURE_STRATIFIED rank0_only=$COMM_EFF_CAPTURE_RANK0_ONLY g_dense=$COMM_EFF_CAPTURE_G_DENSE fresh_anchor=$COMM_EFF_CAPTURE_FRESH_ANCHOR fresh_anchor_loss=$COMM_EFF_CAPTURE_FRESH_ANCHOR_LOSS dump_dtype=$COMM_EFF_CAPTURE_DUMP_DTYPE
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
  actor_rollout_ref.actor.comm_eff.spectral.beta_anc="$COMM_EFF_SPECTRAL_BETA_ANC" \
  actor_rollout_ref.actor.comm_eff.spectral.ema_device="$COMM_EFF_SPECTRAL_EMA_DEVICE" \
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
