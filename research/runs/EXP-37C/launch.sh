#!/usr/bin/env bash
# EXP-37C launch — single cell: signed_ema(alpha=0.25, BETA_ANC=0.0) accel base at the
# known-good 5/5 anchor latency, 100 steps. SIBLING of EXP-37B; the ONLY delta vs EXP-37B
# is beta_anc 0.50 -> 0.0 (no anchor-gradient EMA: M = the latest fire's anchor gradient
# only, zero moving-average history). Same WandB project for direct overlay.
#
# beta_anc=0 CANNOT be set via env: the accel base BARE-exports COMM_EFF_SPECTRAL_BETA_ANC=0.50,
# which clobbers caller env. So we override it as a TRAILING Hydra arg (Hydra last-wins) —
# the same mechanism EXP-37 used for cadence/delay 20/20.
#   plan: .claude/plans/37.md  §Notes for runner  +  memory [[anchor-gradient-ema-beta0-grpo]]
#
# *** BANNER FOOTGUN ***: the launcher banner will print `spectral: ... beta_anc=0.50`
# (it echoes the BARE export, not the trailing override). DO NOT trust the banner for
# beta_anc on this run — verify beta_anc=0.0 in the resolved main_ppo command in train.log
# (grep "spectral.beta_anc=" -> last wins) and in the WandB run config.
set -euo pipefail
cd /workspace/verl

git config --global user.email "harness@verl-research.local" 2>/dev/null || true
git config --global user.name  "verl-research-harness"        2>/dev/null || true

# Canonical training log (monitor greps it; sync-metrics tails it for the heartbeat).
export LOG=/workspace/train.log
mkdir -p /workspace/runs/EXP-37C

PROJECT_NAME=verl_compression_research_accel_rebaseline \
EXPERIMENT_NAME=exp-37c-cad5-delay5-beta0-100step \
TOTAL_TRAINING_STEPS=100 \
TEST_FREQ=25 \
LOG=/workspace/train.log \
  bash examples/grpo_trainer/vast_comm_eff_accel_base_qwen25_1p5b_grpo_gsm8k.sh \
  actor_rollout_ref.actor.comm_eff.spectral.beta_anc=0.0

echo "$(date -Iseconds) done" > /workspace/runs/EXP-37C/done.flag
