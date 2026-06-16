#!/usr/bin/env bash
# Explicit delayed_ef comm-eff wrapper.
#
# ============================================================================
# COMMUNICATION-EFFICIENT DELAYED_EF BASELINE.  ***START HERE.***
# ============================================================================
# This wrapper pins the delayed_ef anchor/PowerSGD baseline explicitly:
#
#   delayed_ef merger on the PowerSGD r=77 anchor circuit
#        G_corr = G_comp + lambda*(M_rep - G_comp_ring),  lambda=1, beta_anc=0
# See research/runs/SUMMARY.md for durable run summaries.
#
# This script is self-contained: it sets the baseline substrate explicitly and
# then calls the canonical comm-eff launcher.
#
# Change only intentional env overrides on top of this script. Keep the codec,
# compression rate, projection basis Q, batch, and generation side fixed unless
# you are deliberately testing those axes.
#
#   Perturbation:   COMM_EFF_SPECTRAL_PERTURB_SIGMA=0.03 \
#                     EXPERIMENT_NAME=perturb bash <thisfile>
#
# Run length / cadence / name are overridable; everything else is pinned.
# ============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- delayed_ef baseline substrate -----------------------------------------
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
export COMM_EFF_SPECTRAL_CORRECTION_MODE=delayed_ef
export COMM_EFF_SPECTRAL_DELAYED_EF_LAMBDA=1.0
export COMM_EFF_SPECTRAL_BETA_ANC=0.0                 # use the latest fire's M
export COMM_EFF_SPECTRAL_EMA_DEVICE=cpu               # OOM guard (keep full-coverage M off-GPU)
export COMM_EFF_SPECTRAL_MAX_TARGETS=-1               # full coverage (all 196 matrices)

# ---- standing OOM guards (anchor allocates a ~3 GB no-hook clone/rank) -------
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-18432}"

# ---- diagnostics OFF on this production baseline (FIXED_CONTROL_SURFACE) -----
export COMM_EFF_CAPTURE_ENABLED=false

# ---- measurement / naming (overridable; everything above is pinned) ---------
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-55}"
export TEST_FREQ="${TEST_FREQ:-25}"
export VAL_BEFORE_TRAIN=True
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-delayed_ef_comm_eff}"

# ---- box compatibility: some Vast hosts crash in vLLM custom all-reduce
#      (CUDA-IPC under the mp executor) during KV-cache init. Opt in if that
#      happens; it is validation-neutral (NCCL all-reduce instead). -----------
DISABLE_CUSTOM_ALL_REDUCE="${DISABLE_CUSTOM_ALL_REDUCE:-false}"
EXTRA_OVERRIDES=()
if [[ "${DISABLE_CUSTOM_ALL_REDUCE}" == "true" ]]; then
  EXTRA_OVERRIDES+=("+actor_rollout_ref.rollout.engine_kwargs.vllm.disable_custom_all_reduce=true")
fi

echo "=== delayed_ef comm-eff baseline (lambda=1, r=77 anchor circuit) ===" >&2
echo "    steps=$TOTAL_TRAINING_STEPS test_freq=$TEST_FREQ name=$EXPERIMENT_NAME disable_custom_all_reduce=$DISABLE_CUSTOM_ALL_REDUCE" >&2
echo "    (optional anchor-usage levers default OFF)" >&2

# Portable empty-array expansion under `set -u` (bash 3.2+ safe).
exec bash "$HERE/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh" \
  "${EXTRA_OVERRIDES[@]+"${EXTRA_OVERRIDES[@]}"}" "$@"
