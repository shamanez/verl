#!/usr/bin/env bash
# CURRENT COMM-EFF BASELINE — and it is the PROBLEM STATE. ***START HERE.***
#
# This is the baseline every future test compares against. It runs the
# signed_ema merger on the LOCKED PowerSGD r=77 anchor substrate, on the fast
# 1K surface, at HIGH anchor latency (cadence/delay_K = 20/20).
#
# WHY HIGH LATENCY: at 20/20 the stale anchor gradient has rotated ~orthogonal
# to the live gradient (the "k-collapse"). This is the failure we are trying to
# fix — see Priority 1 (reports/priority-1-anchor-staleness-k-collapse.html).
# The baseline deliberately sits in the regime where the problem appears.
#
# CAVEAT (be honest about this): in the earlier 100-step study the collapse
# became visible around step ~61. This launcher runs 50 steps for speed, so a
# bare 50-step run CONFIGURES the collapse regime but may not fully manifest it.
# Push TOTAL_TRAINING_STEPS=100 to actually watch it ignite.
#
# Everything is pinned; only run length / name are overridable. signed_ema_alpha
# has no env mapping in the generic launcher, so it rides as a trailing Hydra arg.
# The locked surface + substrate ARE the pinned exports in THIS file (below) —
# it is the authoritative value sheet (the old runs/FIXED_CONTROL_SURFACE.md was pruned).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- comm-eff substrate (LOCKED) --------------------------------------------
export COMM_EFF_ENABLED=true
export COMM_EFF_COMPRESSION_TYPE=powersgd
export COMM_EFF_POWERSGD_RANK=77            # byte-matched to mask p=0.95 (H=1536)
export COMM_EFF_POWERSGD_Q_BASIS=act        # forward/recon basis
export COMM_EFF_ANCHOR_ENABLED=true         # MANDATORY stale full-grad reference M
export COMM_EFF_ANCHOR_OWNS_Q=true          # the anchor is the ONLY thing that updates Q
export COMM_EFF_ANCHOR_CADENCE=20           # HIGH latency — the k-collapse regime (Priority 1)
export COMM_EFF_ANCHOR_DELAY_K=20           # HIGH latency — the k-collapse regime (Priority 1)
export COMM_EFF_ANCHOR_REPLAY_PAIRED_BATCH=true   # paired replay for generator-consistent M
export COMM_EFF_ANCHOR_BATCH_SCOPE=ppo_minibatch  # historical half-update Q+M scope
export COMM_EFF_ANCHOR_SNAPSHOT_DEVICE=cpu        # OOM guard
# Weight projection is not part of this locked control, even if the caller's
# environment exported rank1/fixed lookahead knobs for another experiment.
export COMM_EFF_ANCHOR_LOOKAHEAD_ANCHOR=false
export COMM_EFF_ANCHOR_LOOKAHEAD_MODE=disabled
export COMM_EFF_ANCHOR_LOOKAHEAD_STRENGTH=1.0
export COMM_EFF_ANCHOR_LOOKAHEAD_ROLLOUT_SOURCE=auto
export COMM_EFF_ANCHOR_LOOKAHEAD_WINDOW_SNAPSHOTS=4
export COMM_EFF_ANCHOR_WARMUP_MODE=stale_correct
export COMM_EFF_ANCHOR_LOOKAHEAD_MIN_SNAPSHOTS=-1
export COMM_EFF_CLEAN_CADENCE=0             # anchor replaces periodic clean steps
export COMM_EFF_SPECTRAL_ENABLED=true
export COMM_EFF_SPECTRAL_EMA_DEVICE=cpu               # OOM guard (keep full-coverage M off-GPU)
export COMM_EFF_SPECTRAL_MAX_TARGETS=-1               # full coverage (all 196 matrices)

# ---- core method: signed_ema (alpha=0.25, beta_anc=0.50) --------------------
export COMM_EFF_SPECTRAL_CORRECTION_MODE=signed_ema
export COMM_EFF_SPECTRAL_BETA_ANC=0.50

# ---- fast 1K surface (resp 1024 for quick turnaround) -----------------------
export USE_DYNAMIC_BSZ=True
export MAX_RESPONSE_LENGTH=1024
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
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-comm_eff_baseline_collapse}"

# ---- vLLM all-reduce: controlled var (default true, the Vast IPC-crash guard);
#      the generic launcher converts the selector to the Hydra override. ----------
export DISABLE_CUSTOM_ALL_REDUCE="${DISABLE_CUSTOM_ALL_REDUCE:-true}"

echo "=== comm-eff baseline (signed_ema a=0.25 beta_anc=0.50, r=77 anchor, HIGH-latency 20/20 = k-collapse regime) ===" >&2
echo "    surface: resp=$MAX_RESPONSE_LENGTH dyn_bsz=$USE_DYNAMIC_BSZ rollout_tp=$ROLLOUT_TP mem_util=$ROLLOUT_GPU_MEM_UTIL diagnostics=off" >&2
echo "    anchor: cadence=$COMM_EFF_ANCHOR_CADENCE delay_K=$COMM_EFF_ANCHOR_DELAY_K (collapse visible ~step 61 → use TOTAL_TRAINING_STEPS=100 to manifest)" >&2
echo "    steps=$TOTAL_TRAINING_STEPS test_freq=$TEST_FREQ name=$EXPERIMENT_NAME disable_custom_all_reduce=$DISABLE_CUSTOM_ALL_REDUCE" >&2

# signed_ema_alpha has no env mapping in the generic launcher; pass it as a
# TRAILING Hydra arg. "$@" comes LAST so callers can override.
exec bash "$HERE/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh" \
  actor_rollout_ref.actor.comm_eff.spectral.signed_ema_alpha=0.25 \
  actor_rollout_ref.actor.comm_eff.spectral.diagnostics=false \
  "$@"
