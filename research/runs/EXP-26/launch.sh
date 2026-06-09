#!/usr/bin/env bash
# EXP-26 Step A launch.sh — runs INSIDE the Vast.ai container.
# The template's onstart cloned shamanez/verl@vast-ai-workload into /workspace/verl
# and pip-installed it. This script REPLACES that tree with the exp/26 branch from
# the shipped bundle, then runs the 3 Step-A audit arms SEQUENTIALLY (max_parallel=1)
# with capture ON. Step A is the GATE — B/C launch only after the analyst's DECISION,
# from a separate re-materialized launch (the runner re-invokes on the warm box).
set -euo pipefail
cd /workspace/runs/EXP-26

git config --global user.email "harness@verl-research.local"
git config --global user.name  "verl-research-harness"

# ---- Apply the experimental bundle (code_change=true) ----
BUNDLE=/workspace/runs/EXP-26/exp.bundle
if [[ -f "$BUNDLE" ]]; then
  cd /workspace
  [[ -d verl ]] && mv verl verl.upstream-vast-ai-workload   # preserve template tree
  git clone -b "exp/26-geometry-audit-ef-powersgd" "$BUNDLE" verl
  cd /workspace/verl
  git remote set-url origin https://github.com/shamanez/verl.git || true
  uv pip install --no-deps -e . > /workspace/pip.log 2>&1 || \
    pip install --no-deps -e . > /workspace/pip.log 2>&1
fi

cd /workspace/verl

# ---- Pre-run probe: import the patched modules + run the CPU invariant tests on
#      the box so a hard-gate regression aborts BEFORE the expensive capture run.
echo "=== EXP-26 pre-run probe: CPU invariant tests (hard gate) ===" | tee -a /workspace/runs/EXP-26/probe.log
python -m pytest tests/workers/comm_eff/test_ef_powersgd_exp26.py -q \
  >> /workspace/runs/EXP-26/probe.log 2>&1 || {
    echo "PROBE_FAILED: EXP-26 ef_powersgd/capture CPU invariants did not pass on the box" \
      | tee -a /workspace/runs/EXP-26/probe.log
    echo "=== aborting before the GPU capture run (hard-gate) ===" | tee -a /workspace/runs/EXP-26/probe.log
    exit 7
  }
echo "=== pre-run CPU invariants GREEN ===" | tee -a /workspace/runs/EXP-26/probe.log

# ---- Step A: the 3 audit arms, SEQUENTIAL (max_parallel=1) ----
# Common Step-A capture env (the audit needs G_dense + the delay_K=0 fresh probe).
export COMM_EFF_CAPTURE_ENABLED=true
export COMM_EFF_CAPTURE_DIR=/workspace/captures
export COMM_EFF_CAPTURE_MAX_TICKS=8
export COMM_EFF_CAPTURE_STRATIFIED=4
export COMM_EFF_CAPTURE_G_DENSE=true
export COMM_EFF_CAPTURE_FRESH_ANCHOR=true
export COMM_EFF_CAPTURE_DUMP_DTYPE=fp32
export TOTAL_TRAINING_STEPS=6
export TEST_FREQ=1000
export VAL_BEFORE_TRAIN=False
export SAVE_FREQ=1000

LAUNCHER=examples/grpo_trainer/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh

run_arm () {
  local arm="$1"; shift
  local capdir="/workspace/captures/$arm"
  echo "=== EXP-26 Step A arm=$arm START $(date -Iseconds) ===" | tee -a /workspace/runs/EXP-26/stepA.log
  # Per-arm capture subdir so the 3 arms' dumps don't collide; rsynced back per-arm.
  COMM_EFF_CAPTURE_DIR="$capdir" \
    "$@" bash "$LAUNCHER" \
    > "/workspace/runs/EXP-26/train_${arm}.log" 2>&1 || {
      echo "ARM_FAILED: $arm (see train_${arm}.log)" | tee -a /workspace/runs/EXP-26/stepA.log
    }
  # Mirror dumps + log under the run dir so sync-metrics rsyncs them to the laptop.
  mkdir -p "/workspace/runs/EXP-26/captures/$arm"
  cp -r "$capdir"/. "/workspace/runs/EXP-26/captures/$arm/" 2>/dev/null || true
  echo "=== EXP-26 Step A arm=$arm DONE $(date -Iseconds) ===" | tee -a /workspace/runs/EXP-26/stepA.log
}

# A0 dense control. EXP-26 hotfix: capture.enabled has NO effect when
# comm_eff.enabled=false (the master flag gates the whole comm_eff path incl.
# the capture writer + the G_dense clone backward). So run A0 with comm_eff
# ENABLED but a TRUE-IDENTITY codec/merger: compression_type=dense (no projector
# => fast grad IS the dense grad, G_comp==G_dense) + mask OFF + ef_powersgd with
# ef_decay=ef_clip=0 (the limiting-case identity => G_corr==G_comp). The anchor
# stays on for parity with A1/A2. This makes the capture hooks + the parallel
# uncompressed G_dense backward fire while applying ZERO compression/correction
# (verified true-dense: powersgd compressor is None, merger is a no-op).
run_arm A0_dense \
  env COMM_EFF_ENABLED=true \
      COMM_EFF_COMPRESSION_TYPE=dense \
      COMM_EFF_MASK_ENABLED=false \
      COMM_EFF_SPECTRAL_ENABLED=true \
      COMM_EFF_SPECTRAL_CORRECTION_MODE=ef_powersgd \
      COMM_EFF_SPECTRAL_EF_DECAY=0.0 \
      COMM_EFF_SPECTRAL_EF_CLIP=0.0 \
      EXPERIMENT_NAME=exp26_A0_dense

# A1 plain PowerSGD r77 (anchor on + owns Q, NO merger) — H1 discriminator.
run_arm A1_powersgd_r77 \
  env COMM_EFF_SPECTRAL_ENABLED=false EXPERIMENT_NAME=exp26_A1_powersgd_r77

# A2 EXP-25 anchor + signed_ema alpha=0.5 — the falsified merger; confirms H1.
run_arm A2_signed_ema_a0p5 \
  env COMM_EFF_SPECTRAL_CORRECTION_MODE=signed_ema \
      COMM_EFF_SPECTRAL_SIGNED_EMA_ALPHA=0.5 \
      EXPERIMENT_NAME=exp26_A2_signed_ema_a0p5

echo "$(date -Iseconds) stepA_all_done" > /workspace/runs/EXP-26/stepA.done.flag
echo "=== EXP-26 Step A COMPLETE — captures under /workspace/runs/EXP-26/captures/<arm>/ ===" \
  | tee -a /workspace/runs/EXP-26/stepA.log
echo "=== analyst runs: python research/scripts/geometry_audit.py runs/EXP-26 (per arm) ===" \
  | tee -a /workspace/runs/EXP-26/stepA.log
# Step A is the GATE: do NOT auto-launch B/C. The runner re-materializes + launches
# the decided stage on the warm box after the analyst returns the A-DECISION.
touch /workspace/runs/EXP-26/done.flag
