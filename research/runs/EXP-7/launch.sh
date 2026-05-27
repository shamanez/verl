#!/usr/bin/env bash
# EXP-7 launch.sh — runs inside the Vast.ai container.
#
# The template onstart has already cloned shamanez/verl @ vast-ai-workload into
# /workspace/verl and pip-installed it. For this code_change=true experiment we
# replace that tree with the exp/7-spectral-fsdp-discovery branch from the
# shipped bundle, then run the seeded two-step GRPO spectral smoke:
#   PRIMARY:    CELL=spectral_on  (the FSDP grad-repr discovery + correction)
#   REGRESSION: CELL=disabled     (must be a strict no-op vs dense / EXP-5)
set -euo pipefail
cd /workspace/runs/EXP-7

git config --global user.email "harness@verl-research.local"
git config --global user.name  "verl-research-harness"

# ---- apply the experimental branch from the shipped bundle ----
BUNDLE=/workspace/runs/EXP-7/exp.bundle
if [[ -f "$BUNDLE" ]]; then
  cd /workspace
  [[ -d verl ]] && mv verl verl.upstream-vast-ai-workload   # preserve template tree
  git clone -b "exp/7-spectral-fsdp-discovery" "$BUNDLE" verl
  cd /workspace/verl
  git remote set-url origin https://github.com/shamanez/verl.git || true
  uv pip install --no-deps -e . > /workspace/pip.log 2>&1 || \
    pip install --no-deps -e . > /workspace/pip.log 2>&1
fi

export VERL_ROOT=/workspace/verl
cd "$VERL_ROOT"

# ---- PRIMARY cell: spectral on (headline FSDP discovery + correction) ----
echo "===== EXP-7 cell: spectral_on ====="
CELL=spectral_on bash examples/grpo_trainer/vast_exp7_spectral_smoke.sh || {
  echo "EXP-7 spectral_on cell FAILED (exit $?)" >&2
  # do not abort the whole launch before the regression cell records evidence
}

# ---- REGRESSION cell: disabled (no-op parity vs dense / EXP-5) ----
echo "===== EXP-7 cell: disabled (regression) ====="
CELL=disabled bash examples/grpo_trainer/vast_exp7_spectral_smoke.sh || {
  echo "EXP-7 disabled cell FAILED (exit $?)" >&2
}

# Aggregate done flag for the orchestrator / sync-metrics.
mkdir -p /workspace/runs/EXP-7
echo "$(date -Iseconds) done" > /workspace/runs/EXP-7/done.flag
echo "=== EXP-7 launch.sh complete at $(date -u +%FT%TZ) ==="
