#!/usr/bin/env bash
# EXP-25 anchor-circuit-default — runs inside the Vast.ai container.
# The template onstart already cloned shamanez/verl @ vast-ai-workload into
# /workspace/verl and pip-installed it. We REPLACE that with the
# exp/25-anchor-default branch from the shipped bundle (code_change=true).
#
# Per-stage functions; the detached tmux runs the stage named by $1. The
# experiment-runner SSHes in to verify each probe's invariants from the log,
# then triggers the next stage on the SAME warm box (commit-hotfix loop between
# probes; back-to-back alpha arms).
set -uo pipefail

RUN_DIR=/workspace/runs/EXP-25
LOG_DIR="$RUN_DIR/logs"
mkdir -p "$LOG_DIR"
BRANCH="exp/25-anchor-default"
LAUNCHER=examples/grpo_trainer/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh

git config --global user.email "harness@verl-research.local" 2>/dev/null || true
git config --global user.name  "verl-research-harness"      2>/dev/null || true

apply_bundle() {
  cd /workspace
  if [[ -f "$RUN_DIR/exp.bundle" ]]; then
    if [[ -d verl && ! -d verl.upstream-vast-ai-workload ]]; then
      mv verl verl.upstream-vast-ai-workload
    fi
    rm -rf verl
    git clone -b "$BRANCH" "$RUN_DIR/exp.bundle" verl 2>&1 | tail -3
    cd /workspace/verl
    git remote set-url origin https://github.com/shamanez/verl.git || true
    echo "=== applied bundle: $(git log --oneline -1) ==="
    (command -v uv >/dev/null 2>&1 && uv pip install --no-deps -e . || pip install --no-deps -e .) \
      > /workspace/pip.log 2>&1
    echo "=== pip --no-deps reinstall done (tail) ==="; tail -3 /workspace/pip.log
  else
    echo "ERROR: no exp.bundle at $RUN_DIR" >&2; return 1
  fi
}

reapply_bundle() {
  cd /workspace/verl || return 1
  git fetch origin "$BRANCH" 2>&1 | tail -2
  # single-branch template clones lack the origin/<branch> ref; FETCH_HEAD always
  # points at the just-fetched tip.
  git reset --hard FETCH_HEAD 2>&1 | tail -2
  echo "=== reapplied: $(git log --oneline -1) ==="
  (command -v uv >/dev/null 2>&1 && uv pip install --no-deps -e . || pip install --no-deps -e .) \
    > /workspace/pip.log 2>&1
  echo "=== pip reinstall done ==="
}

common_env() {
  export COMM_EFF_ENABLED=true
  export COMM_EFF_COMPRESSION_TYPE=powersgd
  export COMM_EFF_MASK_ENABLED=false
  export COMM_EFF_POWERSGD_RANK=77
  export COMM_EFF_POWERSGD_SYNC_BASIS=true
  export COMM_EFF_POWERSGD_QR_DTYPE=fp32
  export COMM_EFF_CLEAN_CADENCE=0
  export COMM_EFF_ANCHOR_ENABLED=true
  export COMM_EFF_SPECTRAL_ENABLED=true
  export COMM_EFF_SPECTRAL_BETA_ANC=0.95
  export COMM_EFF_SPECTRAL_CADENCE=1
  export COMM_EFF_SPECTRAL_MAX_TARGETS=-1
  export COMM_EFF_SPECTRAL_EMA_DEVICE=cpu
  export PPO_MAX_TOKEN_LEN_PER_GPU=18432
  # Skip the ~15-min pre-train validation rollout. The id-0/id-1 probes only need
  # the 2 training steps to fire the anchor; the arms get val@25 + val@50 from
  # TEST_FREQ=25 (the plan's metric), so a val@0 baseline is pure GPU cost.
  export VAL_BEFORE_TRAIN=False
}

run_launcher() {
  local logfile="$1"
  cd /workspace/verl
  LOG="$logfile" bash "$LAUNCHER" 2>&1 | tee -a "$logfile"
}

probe0() {
  common_env
  export COMM_EFF_ANCHOR_OWNS_Q=false
  # R3 OFF for the anchor-M-isolation probe: signed_ema @ alpha=1.0 is the IDENTITY
  # merger (G_corr = 1*G_noisy + 0*|G|*sign(M) = G_noisy), so M is computed + verified
  # with NO correction applied. (The old 'reweight' mode was removed in e9931b23e;
  # signed_ema/inject/blend are the only valid modes now.)
  export COMM_EFF_SPECTRAL_CORRECTION_MODE=signed_ema
  export COMM_EFF_SPECTRAL_SIGNED_EMA_ALPHA=1.0
  export COMM_EFF_ANCHOR_CADENCE=1
  export COMM_EFF_ANCHOR_DELAY_K=1
  export TOTAL_TRAINING_STEPS=2
  export TEST_FREQ=0
  export EXPERIMENT_NAME=exp-25-probe0-anchorM
  local lf="$LOG_DIR/probe0.log"
  echo "=== EXP-25 id-0 probe (anchor M, R2/R3 OFF, cadence=1 delay_K=1, 2 steps) ===" | tee "$lf"
  run_launcher "$lf"
  echo "=== probe0 EXIT rc=$? ===" | tee -a "$lf"
  touch "$RUN_DIR/probe0.done.flag"
}

probe1() {
  common_env
  export COMM_EFF_ANCHOR_OWNS_Q=true
  export COMM_EFF_SPECTRAL_CORRECTION_MODE=signed_ema
  export COMM_EFF_SPECTRAL_SIGNED_EMA_ALPHA=0.0
  export COMM_EFF_ANCHOR_CADENCE=1
  export COMM_EFF_ANCHOR_DELAY_K=1
  export TOTAL_TRAINING_STEPS=2
  export TEST_FREQ=0
  export EXPERIMENT_NAME=exp-25-probe1-allflags
  local lf="$LOG_DIR/probe1.log"
  echo "=== EXP-25 id-1 probe (ALL flags ON, alpha=0, cadence=1 delay_K=1, 2 steps) ===" | tee "$lf"
  run_launcher "$lf"
  echo "=== probe1 EXIT rc=$? ===" | tee -a "$lf"
  touch "$RUN_DIR/probe1.done.flag"
}

arm() {
  local alpha="${1:?usage: arm <alpha>}"
  local tag="${alpha//./p}"
  common_env
  export COMM_EFF_ANCHOR_OWNS_Q=true
  export COMM_EFF_SPECTRAL_CORRECTION_MODE=signed_ema
  export COMM_EFF_SPECTRAL_SIGNED_EMA_ALPHA="$alpha"
  export COMM_EFF_ANCHOR_CADENCE=5
  export COMM_EFF_ANCHOR_DELAY_K=5
  export TOTAL_TRAINING_STEPS=50
  export TEST_FREQ=25
  export EXPERIMENT_NAME="exp-25-alpha${tag}"
  local lf="$LOG_DIR/arm-alpha${tag}.log"
  echo "=== EXP-25 id-2 arm alpha=$alpha (cadence=5 delay_K=5, 50 steps, val@25) ===" | tee "$lf"
  run_launcher "$lf"
  local rc=$?
  echo "=== arm alpha=$alpha EXIT rc=$rc ===" | tee -a "$lf"
  touch "$RUN_DIR/arm-alpha${tag}.done.flag"
  return $rc
}

setup_and_probe0() {
  apply_bundle || { echo "BUNDLE APPLY FAILED" >&2; exit 1; }
  probe0
}

all_arms() {
  arm 0.0
  arm 0.3
  arm 0.5
  touch "$RUN_DIR/done.flag"
  echo "=== EXP-25 alpha sweep COMPLETE (all 3 arms) ==="
}

STAGE="${1:-setup_and_probe0}"
shift || true
case "$STAGE" in
  setup_and_probe0) setup_and_probe0 ;;
  apply_bundle)     apply_bundle ;;
  reapply_bundle)   reapply_bundle ;;
  probe0)           probe0 ;;
  probe1)           probe1 ;;
  arm)              arm "$@" ;;
  all_arms)         all_arms ;;
  *) echo "unknown stage: $STAGE" >&2; exit 2 ;;
esac
