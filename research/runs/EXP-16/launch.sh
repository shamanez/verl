#!/usr/bin/env bash
# EXP-16 per-cell launcher. Runs ONE training cell of the stability matrix on the
# Vast.ai box, applying the exp/16-short-run-stability-matrix branch first.
#
#   usage:  bash /workspace/runs/EXP-16/launch.sh <cell_index>   # cell_index in 1..6
#
# Cell 0 (the GPU pre-flight gate) is NOT a training cell — run it via
# cell0_preflight.sh. Cells 1..6 each call the canonical comm-eff launcher with
# ONLY the env knobs this cell varies (the launcher carries the dense baseline).
# Each cell writes /workspace/runs/EXP-16/metrics/<EXPERIMENT_NAME>/train.log and,
# on clean exit, /workspace/verl/runs/<EXPERIMENT_NAME>/done.flag.
set -euo pipefail

CELL="${1:?usage: launch.sh <cell_index 1..6>}"
RUN_ROOT=/workspace/runs/EXP-16
LAUNCHER=examples/grpo_trainer/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh

# Git identity for any in-container commits (commit-hotfix.sh uses these).
git config --global user.email "harness@verl-research.local"
git config --global user.name  "verl-research-harness"

# ---------------------------------------------------------------------------
# Apply the experimental branch from the shipped bundle (code_change=true).
# Done ONCE (idempotent): if /workspace/verl is already on the exp branch with
# the right HEAD, skip. The template's onstart cloned shamanez/verl@vast-ai-workload
# into /workspace/verl; we replace it with the exp branch from the bundle so the
# T5 spectral.cadence gate + anchor.delay_K plumbing are present for cell 5.
# ---------------------------------------------------------------------------
EXP_BRANCH=exp/16-short-run-stability-matrix
BUNDLE="$RUN_ROOT/exp.bundle"
BUNDLE_HEAD=$(git bundle list-heads "$BUNDLE" 2>/dev/null | awk '{print $1}' | head -1)
cd /workspace
NEED_APPLY=1
if [[ -d /workspace/verl/.git ]]; then
  CUR=$(git -C /workspace/verl rev-parse HEAD 2>/dev/null || echo none)
  CURBR=$(git -C /workspace/verl rev-parse --abbrev-ref HEAD 2>/dev/null || echo none)
  if [[ "$CUR" == "$BUNDLE_HEAD" && "$CURBR" == "$EXP_BRANCH" ]]; then
    NEED_APPLY=0
    echo "=== /workspace/verl already on $EXP_BRANCH @ $CUR — skip bundle apply ==="
  fi
fi
if [[ "$NEED_APPLY" == "1" ]]; then
  echo "=== applying $EXP_BRANCH from bundle (HEAD=$BUNDLE_HEAD) ==="
  [[ -d /workspace/verl && ! -d /workspace/verl.upstream-vast-ai-workload ]] && \
    mv /workspace/verl /workspace/verl.upstream-vast-ai-workload
  rm -rf /workspace/verl
  git clone -b "$EXP_BRANCH" "$BUNDLE" /workspace/verl
  cd /workspace/verl
  # Point origin at the fork so any in-container push goes to the right repo.
  git remote set-url origin https://github.com/shamanez/verl.git || true
  uv pip install --no-deps -e . > /workspace/pip.log 2>&1 || \
    pip install --no-deps -e . > /workspace/pip.log 2>&1
  echo "=== verl reinstalled from exp branch (see /workspace/pip.log) ==="
fi

cd /workspace/verl

# Sanity: the T5 change must be present (cell 5 depends on it; cell 6 proves no-op).
grep -q 'comm_eff.spectral.cadence=' "$LAUNCHER" || {
  echo "FATAL: launcher on the box lacks spectral.cadence plumbing — wrong branch?" >&2; exit 3; }
grep -q 'use_orig_params=true' "$LAUNCHER" || {
  echo "FATAL: launcher on the box lacks use_orig_params=true (cell 5 spectral REQUIRES it)" >&2; exit 3; }

# ---------------------------------------------------------------------------
# Per-cell env-override blocks (mirror runs/EXP-16/config.yaml exactly).
# EVERY cell sets TOTAL_TRAINING_STEPS + TEST_FREQ explicitly (launcher default
# is 100/25 — a forgotten override blows the budget / skips validation).
# ---------------------------------------------------------------------------
case "$CELL" in
  1)
    export EXPERIMENT_NAME=grpo_mask_channel_p0p9_no_rescale_10steps
    ENV=( COMM_EFF_ENABLED=true COMM_EFF_MASK_ENABLED=true COMM_EFF_MASK_P=0.9
          COMM_EFF_MASK_RESCALE=false COMM_EFF_MASK_RECOMPUTE=true
          COMM_EFF_CLEAN_CADENCE=0 COMM_EFF_ANCHOR_ENABLED=false
          COMM_EFF_SPECTRAL_ENABLED=false TOTAL_TRAINING_STEPS=10 TEST_FREQ=10 )
    ;;
  2)
    export EXPERIMENT_NAME=grpo_mask_channel_p0p9_rescale_10steps
    ENV=( COMM_EFF_ENABLED=true COMM_EFF_MASK_ENABLED=true COMM_EFF_MASK_P=0.9
          COMM_EFF_MASK_RESCALE=true COMM_EFF_MASK_RECOMPUTE=true
          COMM_EFF_CLEAN_CADENCE=0 COMM_EFF_ANCHOR_ENABLED=false
          COMM_EFF_SPECTRAL_ENABLED=false TOTAL_TRAINING_STEPS=10 TEST_FREQ=10 )
    ;;
  3)
    export EXPERIMENT_NAME=grpo_mask_channel_p0p9_no_rescale_clean_every4_20steps
    ENV=( COMM_EFF_ENABLED=true COMM_EFF_MASK_ENABLED=true COMM_EFF_MASK_P=0.9
          COMM_EFF_MASK_RESCALE=false COMM_EFF_MASK_RECOMPUTE=true
          COMM_EFF_CLEAN_CADENCE=4 COMM_EFF_ANCHOR_ENABLED=false
          COMM_EFF_SPECTRAL_ENABLED=false TOTAL_TRAINING_STEPS=20 TEST_FREQ=20 )
    ;;
  4)
    export EXPERIMENT_NAME=grpo_mask_channel_p0p9_rescale_clean_every4_20steps
    ENV=( COMM_EFF_ENABLED=true COMM_EFF_MASK_ENABLED=true COMM_EFF_MASK_P=0.9
          COMM_EFF_MASK_RESCALE=true COMM_EFF_MASK_RECOMPUTE=true
          COMM_EFF_CLEAN_CADENCE=4 COMM_EFF_ANCHOR_ENABLED=false
          COMM_EFF_SPECTRAL_ENABLED=false TOTAL_TRAINING_STEPS=20 TEST_FREQ=20 )
    ;;
  5)
    # Spectral switch-on — REQUIRES the T5 change (spectral.cadence + anchor.delay_K).
    export EXPERIMENT_NAME=grpo_mask_channel_p0p9_rescale_anchor2_spectral2_20steps
    ENV=( COMM_EFF_ENABLED=true COMM_EFF_MASK_ENABLED=true COMM_EFF_MASK_P=0.9
          COMM_EFF_MASK_RESCALE=true COMM_EFF_MASK_RECOMPUTE=true
          COMM_EFF_CLEAN_CADENCE=0
          COMM_EFF_ANCHOR_ENABLED=true COMM_EFF_ANCHOR_CADENCE=2 COMM_EFF_ANCHOR_DELAY_K=2
          COMM_EFF_SPECTRAL_ENABLED=true COMM_EFF_SPECTRAL_CADENCE=2
          COMM_EFF_SPECTRAL_ALPHA=0.5 COMM_EFF_SPECTRAL_TAU=0.01 COMM_EFF_SPECTRAL_BETA_ANC=0.9
          COMM_EFF_SPECTRAL_EMA_DEVICE=gpu COMM_EFF_SPECTRAL_SVD_MODE=full
          COMM_EFF_SPECTRAL_BASIS_CACHE=cache TOTAL_TRAINING_STEPS=20 TEST_FREQ=20 )
    ;;
  6)
    export EXPERIMENT_NAME=dense_grpo_comm_eff_off_25step_reference
    ENV=( COMM_EFF_ENABLED=false TOTAL_TRAINING_STEPS=25 TEST_FREQ=25 )
    ;;
  *)
    echo "FATAL: unknown cell index '$CELL' (expect 1..6)" >&2; exit 2 ;;
esac

# Per-cell metrics dir on the box (rsynced back to laptop runs/EXP-16/metrics/<name>/).
CELL_LOG_DIR="$RUN_ROOT/metrics/$EXPERIMENT_NAME"
mkdir -p "$CELL_LOG_DIR"
export LOG="$CELL_LOG_DIR/train.log"

echo "=== EXP-16 cell $CELL -> $EXPERIMENT_NAME ==="
echo "=== env overrides: ${ENV[*]} ==="

# Run the canonical comm-eff launcher under `set -x` tracing (the launcher itself
# enables it via run_qwen3_4b_fsdp.sh) with the per-cell env in front. The fully
# expanded main_ppo command lands in $LOG for resolved_params.txt extraction.
env "${ENV[@]}" bash "$LAUNCHER"

echo "=== EXP-16 cell $CELL done at $(date -u +%FT%TZ) ==="
