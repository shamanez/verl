#!/usr/bin/env bash
# EXP-26 A0_dense RE-RUN — runs INSIDE the Vast.ai container, AFTER A1+A2 finish.
# Reuses the warm /workspace/verl checkout (already cloned from the exp/26 bundle);
# first fast-forwards it to the pushed hotfix commit so the unified-capture-tick
# fix is live, then runs ONLY the fixed A0 arm (comm_eff ENABLED + true-identity
# codec/merger so the capture hooks + the parallel G_dense backward fire while
# applying ZERO compression). max_parallel=1: the caller launches this ONLY after
# confirming A1 and A2 are done.
set -euo pipefail
cd /workspace/verl

# Fast-forward the warm checkout to the hotfix commit (origin already has it).
git config --global user.email "harness@verl-research.local"
git config --global user.name  "verl-research-harness"
echo "=== A0 re-run: fetching + resetting /workspace/verl to origin/exp/26 hotfix ===" \
  | tee -a /workspace/runs/EXP-26/a0_rerun.log
git fetch origin exp/26-geometry-audit-ef-powersgd 2>&1 | tail -3 | tee -a /workspace/runs/EXP-26/a0_rerun.log
git reset --hard origin/exp/26-geometry-audit-ef-powersgd 2>&1 | tail -2 | tee -a /workspace/runs/EXP-26/a0_rerun.log
HEAD_SHA=$(git rev-parse --short HEAD)
echo "=== /workspace/verl now at $HEAD_SHA ===" | tee -a /workspace/runs/EXP-26/a0_rerun.log
# Re-install editable in case any new module needs registering (no-deps, fast).
uv pip install --no-deps -e . > /workspace/a0_pip.log 2>&1 || pip install --no-deps -e . > /workspace/a0_pip.log 2>&1 || true

# Pre-run probe: the unified-tick + ef invariants must be green on the box.
echo "=== A0 re-run pre-run probe (hard gate) ===" | tee -a /workspace/runs/EXP-26/a0_rerun.log
python -m pytest tests/workers/comm_eff/test_ef_powersgd_exp26.py -q \
  >> /workspace/runs/EXP-26/a0_rerun.log 2>&1 || {
    echo "PROBE_FAILED: A0 re-run invariants did not pass" | tee -a /workspace/runs/EXP-26/a0_rerun.log
    exit 7
  }
echo "=== A0 re-run pre-run invariants GREEN ===" | tee -a /workspace/runs/EXP-26/a0_rerun.log

# Fresh capture dir for the A0 re-run (don't collide with the failed first A0).
CAPDIR=/workspace/captures/A0_dense
rm -rf "$CAPDIR"; mkdir -p "$CAPDIR"

# Same Step-A capture env as the other arms.
export COMM_EFF_CAPTURE_ENABLED=true
export COMM_EFF_CAPTURE_DIR="$CAPDIR"
export COMM_EFF_CAPTURE_MAX_TICKS=8
export COMM_EFF_CAPTURE_STRATIFIED=4
export COMM_EFF_CAPTURE_G_DENSE=true
export COMM_EFF_CAPTURE_FRESH_ANCHOR=true
export COMM_EFF_CAPTURE_DUMP_DTYPE=fp32
export TOTAL_TRAINING_STEPS=6
export TEST_FREQ=1000
export VAL_BEFORE_TRAIN=False
export SAVE_FREQ=1000

# A0 dense reference: comm_eff ENABLED, TRUE-IDENTITY codec + merger.
echo "=== EXP-26 A0_dense RE-RUN START $(date -Iseconds) ===" | tee -a /workspace/runs/EXP-26/a0_rerun.log
COMM_EFF_ENABLED=true \
COMM_EFF_COMPRESSION_TYPE=dense \
COMM_EFF_MASK_ENABLED=false \
COMM_EFF_SPECTRAL_ENABLED=true \
COMM_EFF_SPECTRAL_CORRECTION_MODE=ef_powersgd \
COMM_EFF_SPECTRAL_EF_DECAY=0.0 \
COMM_EFF_SPECTRAL_EF_CLIP=0.0 \
EXPERIMENT_NAME=exp26_A0_dense_rerun \
  bash examples/grpo_trainer/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh \
  > /workspace/runs/EXP-26/train_A0_dense_rerun.log 2>&1 || \
  echo "A0_RERUN_ARM_FAILED (see train_A0_dense_rerun.log)" | tee -a /workspace/runs/EXP-26/a0_rerun.log

# Mirror dumps under the run dir for rsync-back.
mkdir -p /workspace/runs/EXP-26/captures/A0_dense
cp -r "$CAPDIR"/. /workspace/runs/EXP-26/captures/A0_dense/ 2>/dev/null || true
echo "=== EXP-26 A0_dense RE-RUN DONE $(date -Iseconds) ===" | tee -a /workspace/runs/EXP-26/a0_rerun.log
touch /workspace/runs/EXP-26/a0_rerun.done.flag
