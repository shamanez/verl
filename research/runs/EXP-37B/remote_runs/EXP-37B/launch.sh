#!/usr/bin/env bash
# EXP-37B launch — single cell: signed_ema(alpha=0.25, beta_anc=0.50) accel base at
# the known-good 5/5 anchor latency, extended to 100 steps. This is the cheap
# one-variable control isolating whether EXP-37's post-step-50 collapse was
# latency-driven (20/20) or GSM8K epoch-boundary-driven (~step 58).
#
# code_change=false. Config-only run on the LOCKED accel base launcher.
# NO anchor overrides are passed: 5/5 is the accel-base DEFAULT (bare-exported
# COMM_EFF_ANCHOR_CADENCE=5 / COMM_EFF_ANCHOR_DELAY_K=5). Passing trailing args
# would risk leaking a stray override -> REVISE. So we pass ONLY the four
# measurement/identity env vars the plan specifies and NO trailing Hydra args.
#   plan: .claude/plans/37.md  §Notes for runner
set -euo pipefail
cd /workspace/verl

# Git identity for any in-container commit-hotfix (harmless no-op for this run).
git config --global user.email "harness@verl-research.local" 2>/dev/null || true
git config --global user.name  "verl-research-harness"        2>/dev/null || true

# Canonical training log. sync-metrics.sh tails /workspace/train.log for the
# heartbeat, and training-log-monitor greps it for Traceback/OOM/NaN. Exporting
# LOG makes the generic launcher write main_ppo's full output (resolved cmd +
# steps + validation + errors) directly there. The launcher BANNER goes to this
# script's stdout, which the tmux wrapper redirects to launch-banner.log (NOT
# /workspace/train.log) so the two never clobber each other.
export LOG=/workspace/train.log

PROJECT_NAME=verl_compression_research_accel_rebaseline \
EXPERIMENT_NAME=exp-37b-cad5-delay5-100step \
TOTAL_TRAINING_STEPS=100 \
TEST_FREQ=25 \
LOG=/workspace/train.log \
  bash examples/grpo_trainer/vast_comm_eff_accel_base_qwen25_1p5b_grpo_gsm8k.sh

echo "$(date -Iseconds) done" > /workspace/runs/EXP-37B/done.flag
