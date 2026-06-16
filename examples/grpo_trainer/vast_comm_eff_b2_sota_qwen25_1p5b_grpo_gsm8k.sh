#!/usr/bin/env bash
# vast_comm_eff_b2_sota_qwen25_1p5b_grpo_gsm8k.sh
#
# ============================================================================
# THE COMMUNICATION-EFFICIENT SOTA BASELINE — "B2".  ***START HERE.***
# ============================================================================
# This is the FROZEN FLOOR every improvement experiment builds on. It pins the
# best comm-efficient method to date and reproduces it EXACTLY:
#
#   B2 = delayed_ef merger on the PowerSGD r=77 anchor circuit
#        G_corr = G_comp + lambda*(M_rep - G_comp_ring),  lambda=1, beta_anc=0
#   Result: greedy GSM8K val@50 ~= 0.74-0.75  ==  PARITY with the dense control
#           (band 0.75-0.78) at ~5% of the inter-stage gradient-communication cost.
#
# Ground truth (the canonical knob set): research/runs/EXP-31/B2_baseline/
#   resolved_params_B2.txt  +  verdict.md  +  README.md.  Results + why:
#   research/runs/SUMMARY.md.  The locked surface: research/runs/FIXED_CONTROL_SURFACE.md.
#
# This script is SELF-CONTAINED: it sets the full B2 substrate explicitly and
# then calls the canonical comm-eff launcher. Run it bare to reproduce B2.
#
# ---- HOW TO IMPROVE IT (issue #31): the ONE allowed axis -------------------
# Hold this substrate FIXED and change ONLY how the stale anchor gradient is
# USED, by setting a lever's env var ON TOP of this script (every lever defaults
# OFF here => bare = bitwise B2). Do NOT change the codec / compression rate /
# projection basis Q / batch / generation side — those are locked.
#
#   L4 perturbation:   COMM_EFF_SPECTRAL_PERTURB_SIGMA=0.03 \
#                        EXPERIMENT_NAME=L4_perturb bash <thisfile>
#   (L2 delta-momentum / L3 adaptive-dose / L1 control-variate are added on the
#    exp/31 branch; once built they expose their own env knobs the same way.)
#
# Run length / cadence / name are overridable; everything else is pinned:
#   TOTAL_TRAINING_STEPS=100 bash <thisfile>     # extend the winner past 50
# ============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- B2 LOCKED SUBSTRATE (from resolved_params_B2.txt) ----------------------
export COMM_EFF_ENABLED=true
export COMM_EFF_COMPRESSION_TYPE=powersgd
export COMM_EFF_POWERSGD_RANK=77            # byte-matched to mask p=0.95 (H=1536)
export COMM_EFF_POWERSGD_Q_BASIS=act        # forward/recon basis — NEVER change (Step-C dead)
export COMM_EFF_ANCHOR_ENABLED=true         # MANDATORY stale full-grad reference M
export COMM_EFF_ANCHOR_OWNS_Q=true          # the anchor is the ONLY thing that updates Q
export COMM_EFF_ANCHOR_CADENCE=5
export COMM_EFF_ANCHOR_DELAY_K=5
export COMM_EFF_ANCHOR_REPLAY_PAIRED_BATCH=true   # valid on-policy M (EXP-29)
export COMM_EFF_ANCHOR_SNAPSHOT_DEVICE=cpu        # OOM guard
export COMM_EFF_CLEAN_CADENCE=0             # DEAD — the anchor replaced the clean step
export COMM_EFF_SPECTRAL_ENABLED=true
export COMM_EFF_SPECTRAL_CORRECTION_MODE=delayed_ef   # the SOTA merger
export COMM_EFF_SPECTRAL_DELAYED_EF_LAMBDA=1.0        # B2 dose
export COMM_EFF_SPECTRAL_BETA_ANC=0.0                 # use the latest fire's M
export COMM_EFF_SPECTRAL_EMA_DEVICE=cpu               # OOM guard (keep full-coverage M off-GPU)
export COMM_EFF_SPECTRAL_MAX_TARGETS=-1               # full coverage (all 196 matrices)

# ---- standing OOM guards (anchor allocates a ~3 GB no-hook clone/rank) -------
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-18432}"

# ---- diagnostics OFF on this production baseline (FIXED_CONTROL_SURFACE) -----
export COMM_EFF_CAPTURE_ENABLED=false

# ---- measurement / naming (overridable; everything above is pinned) ---------
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-55}"   # 55 = val@50 checkpoint + 5-step buffer so val@50 syncs to WandB before exit (val still read @50)
export TEST_FREQ="${TEST_FREQ:-25}"
export VAL_BEFORE_TRAIN=True
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-b2_sota_comm_eff}"

# ---- box compatibility: some Vast hosts crash in vLLM custom all-reduce
#      (CUDA-IPC under the mp executor) during KV-cache init. Opt in if that
#      happens; it is greedy-val-neutral (NCCL all-reduce instead). -----------
DISABLE_CUSTOM_ALL_REDUCE="${DISABLE_CUSTOM_ALL_REDUCE:-false}"
EXTRA_OVERRIDES=()
if [[ "${DISABLE_CUSTOM_ALL_REDUCE}" == "true" ]]; then
  EXTRA_OVERRIDES+=("+actor_rollout_ref.rollout.engine_kwargs.vllm.disable_custom_all_reduce=true")
fi

echo "=== B2 SOTA comm-eff baseline (delayed_ef lambda=1, r=77 anchor circuit) ===" >&2
echo "    steps=$TOTAL_TRAINING_STEPS test_freq=$TEST_FREQ name=$EXPERIMENT_NAME disable_custom_all_reduce=$DISABLE_CUSTOM_ALL_REDUCE" >&2
echo "    (all issue-#31 anchor-usage levers default OFF => this is bitwise B2)" >&2

# Portable empty-array expansion under `set -u` (bash 3.2+ safe).
exec bash "$HERE/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh" \
  "${EXTRA_OVERRIDES[@]+"${EXTRA_OVERRIDES[@]}"}" "$@"
