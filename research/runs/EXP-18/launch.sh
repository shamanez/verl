#!/usr/bin/env bash
# EXP-18 (M4 curve-match) — FIRST dispatch: provisions the ONE box the whole
# recursive search reuses, and runs the TWO no-code reference cells back-to-back
# in a single tmux (no idle gap between them):
#   CELL 1 — dense reference (the TARGET curve)       COMM_EFF_ENABLED=false
#   CELL 2 — spectral floor (as-implemented spectral) COMM_EFF_ENABLED=true, anchor c5/delay_K5, clean OFF
#
# code_change=FALSE for THIS dispatch: both cells run the ALREADY-COMMITTED code
# on vast-ai-workload (the template onstart cloned shamanez/verl @ vast-ai-workload
# into /workspace/verl). NO exp/* branch, NO bundle is applied here.
#
# set -u + pipefail but NOT -e: CELL 2 MUST run even if CELL 1 errors — the
# analyst sorts health from WandB / the per-cell logs.
set -uo pipefail

RUN_DIR=/workspace/runs/EXP-18
mkdir -p "$RUN_DIR"
NGPUS=4   # provisioned count (4xH200, tier0); the launcher auto-detects and lands on the same value.

# git identity (harmless here; kept for parity with the contract template).
git config --global user.email "harness@verl-research.local" || true
git config --global user.name  "verl-research-harness"       || true

run_cell () {
  # $1 = EXPERIMENT_NAME ; remaining args = VAR=VAL env exports for the launcher.
  local name="$1"; shift
  local log="$RUN_DIR/train_${name}.log"
  echo "=== [EXP-18] launching cell ${name} at $(date -Iseconds) ===" | tee -a "$log"
  cd /workspace/verl
  env "$@" \
    bash examples/grpo_trainer/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh \
    >> "$log" 2>&1
  local rc=$?
  echo "$(date -Iseconds) cell=${name} exit_code=${rc}" > "$RUN_DIR/done_${name}.flag"
  echo "=== [EXP-18] cell ${name} finished rc=${rc} at $(date -Iseconds) ===" | tee -a "$log"
  return 0   # never abort the chain on a cell error
}

# ---------------------------------------------------------------------------
# CELL 1 — dense reference (TARGET curve). Byte-identical dense (comm-eff OFF).
# ---------------------------------------------------------------------------
run_cell curvematch_dense_ref_50step \
  PROJECT_NAME=comm_eff_curve_match_m4 \
  EXPERIMENT_NAME=curvematch_dense_ref_50step \
  COMM_EFF_ENABLED=false \
  TOTAL_TRAINING_STEPS=50 \
  VAL_BEFORE_TRAIN=False \
  TEST_FREQ=100000 \
  USE_DYNAMIC_BSZ=True \
  NGPUS_PER_NODE="$NGPUS"

# ---------------------------------------------------------------------------
# CELL 2 — spectral floor (current spectral, AS-IMPLEMENTED, no patch).
#   NON-NEGOTIABLE constraint pins (plan §HARD CONSTRAINTS):
#     COMM_EFF_ANCHOR_DELAY_K=5   (launcher DEFAULTS this to 20 — MUST be 5; never 0/20)
#     COMM_EFF_CLEAN_CADENCE=0    (clean step OFF — the correction must stand alone)
#     COMM_EFF_ANCHOR_CADENCE=5
# ---------------------------------------------------------------------------
run_cell curvematch_spectral_baseline_c5_d5 \
  PROJECT_NAME=comm_eff_curve_match_m4 \
  EXPERIMENT_NAME=curvematch_spectral_baseline_c5_d5 \
  COMM_EFF_ENABLED=true \
  COMM_EFF_MASK_ENABLED=true \
  COMM_EFF_MASK_P=0.9 \
  COMM_EFF_MASK_RESCALE=true \
  COMM_EFF_MASK_RECOMPUTE=true \
  COMM_EFF_CLEAN_CADENCE=0 \
  COMM_EFF_ANCHOR_ENABLED=true \
  COMM_EFF_ANCHOR_CADENCE=5 \
  COMM_EFF_ANCHOR_DELAY_K=5 \
  COMM_EFF_SPECTRAL_ENABLED=true \
  TOTAL_TRAINING_STEPS=50 \
  VAL_BEFORE_TRAIN=False \
  TEST_FREQ=100000 \
  USE_DYNAMIC_BSZ=True \
  NGPUS_PER_NODE="$NGPUS"

# Aggregate done flag — written only after BOTH cells have returned.
echo "$(date -Iseconds) both cells done" > "$RUN_DIR/done.flag"
echo "=== [EXP-18] all cells complete at $(date -Iseconds) ===" | tee -a "$RUN_DIR/train_curvematch_dense_ref_50step.log"
