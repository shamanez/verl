#!/usr/bin/env bash
# Accelerated communication-efficient base. ***START HERE for future tests.***
#
# Faster surface (resp 2048 / dynamic-bsz / rollout TP=1 + diagnostics OFF,
# ~25 min / 50 steps) on the LOCKED PowerSGD r=77 anchor substrate, with the
# signed_ema core merger (alpha=0.25, beta_anc=0.50). Everything else is pinned;
# only run length / name are overridable. signed_ema_alpha + diagnostics have no
# env mapping in the generic launcher, so they ride as trailing Hydra args.
# See research/runs/FIXED_CONTROL_SURFACE.md (and EXP-36B/NEUTRALITY_REVIEW.md
# for the proof diagnostics=false is math-neutral).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- comm-eff substrate (UNCHANGED from b2_sota) ----------------------------
export COMM_EFF_ENABLED=true
export COMM_EFF_COMPRESSION_TYPE=powersgd
export COMM_EFF_POWERSGD_RANK=77            # byte-matched to mask p=0.95 (H=1536)
export COMM_EFF_POWERSGD_Q_BASIS=act        # forward/recon basis
export COMM_EFF_ANCHOR_ENABLED=true         # MANDATORY stale full-grad reference M
export COMM_EFF_ANCHOR_OWNS_Q=true          # the anchor is the ONLY thing that updates Q
export COMM_EFF_ANCHOR_CADENCE=5
export COMM_EFF_ANCHOR_DELAY_K=5
export COMM_EFF_ANCHOR_REPLAY_PAIRED_BATCH=true   # paired replay for generator-consistent M
export COMM_EFF_ANCHOR_SNAPSHOT_DEVICE=cpu        # OOM guard
export COMM_EFF_CLEAN_CADENCE=0             # anchor replaces periodic clean steps
export COMM_EFF_SPECTRAL_ENABLED=true
export COMM_EFF_SPECTRAL_EMA_DEVICE=cpu               # OOM guard (keep full-coverage M off-GPU)
export COMM_EFF_SPECTRAL_MAX_TARGETS=-1               # full coverage (all 196 matrices)

# ---- core method: signed_ema (alpha=0.25, beta_anc=0.50) --------------------
export COMM_EFF_SPECTRAL_CORRECTION_MODE=signed_ema
export COMM_EFF_SPECTRAL_BETA_ANC=0.50

# ---- accel surface (faster than the 16K/static/TP=2 b2_sota surface) --------
export USE_DYNAMIC_BSZ=True
export MAX_RESPONSE_LENGTH=2048
export MAX_PROMPT_LENGTH=1024
export ROLLOUT_TP=1
export ROLLOUT_GPU_MEM_UTIL=0.55
export PPO_MAX_TOKEN_LEN_PER_GPU=24576
export TRAIN_BATCH_SIZE=128
export PPO_MINI_BATCH_SIZE=64
export ROLLOUT_N=8
export ACTOR_LR=1e-6

# ---- standing OOM guards ----------------------------------------------------
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ---- diagnostics OFF on this production baseline (FIXED_CONTROL_SURFACE) -----
export COMM_EFF_CAPTURE_ENABLED=false

# ---- measurement / naming (overridable; everything above is pinned) ---------
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-50}"
export TEST_FREQ="${TEST_FREQ:-25}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-comm_eff_accel_base}"

# ---- vLLM all-reduce: controlled var (default true, the Vast IPC-crash guard);
#      the generic launcher converts the selector to the Hydra override. ----------
export DISABLE_CUSTOM_ALL_REDUCE="${DISABLE_CUSTOM_ALL_REDUCE:-true}"

echo "=== accelerated comm-eff base (signed_ema a=0.25 beta_anc=0.50, r=77 anchor circuit) ===" >&2
echo "    surface: resp=$MAX_RESPONSE_LENGTH dyn_bsz=$USE_DYNAMIC_BSZ rollout_tp=$ROLLOUT_TP mem_util=$ROLLOUT_GPU_MEM_UTIL diagnostics=off" >&2
echo "    steps=$TOTAL_TRAINING_STEPS test_freq=$TEST_FREQ name=$EXPERIMENT_NAME disable_custom_all_reduce=$DISABLE_CUSTOM_ALL_REDUCE" >&2

# signed_ema_alpha + diagnostics have no env mapping in the generic launcher;
# pass them as TRAILING Hydra args. "$@" comes LAST so callers can override.
exec bash "$HERE/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh" \
  actor_rollout_ref.actor.comm_eff.spectral.signed_ema_alpha=0.25 \
  actor_rollout_ref.actor.comm_eff.spectral.diagnostics=false \
  "$@"
