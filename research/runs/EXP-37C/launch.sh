#!/usr/bin/env bash
# EXP-37C launch — single cell: signed_ema(alpha=0.25, BETA_ANC=0.0) accel base at
# anchor latency 20/20 (cadence=20, delay_K=20), 100 steps. Two deltas vs EXP-37B:
#   (1) beta_anc 0.50 -> 0.0  (no anchor-gradient EMA: M = latest fire only, no history)
#   (2) latency  5/5  -> 20/20 (high staleness — the SAME latency that collapsed EXP-37)
# Purpose: at the high latency that collapsed EXP-37 (20/20, beta_anc=0.50), does dropping
# the EMA history (beta_anc=0) change the back-half collapse? Same WandB project for overlay
# vs EXP-37 (20/20, beta0.50, collapsed), EXP-37B (5/5, beta0.50, stable), EXP-37D (dense).
#
# NEITHER beta_anc NOR cadence/delay can be set via env: the accel base BARE-exports
# COMM_EFF_SPECTRAL_BETA_ANC=0.50, COMM_EFF_ANCHOR_CADENCE=5, COMM_EFF_ANCHOR_DELAY_K=5,
# which clobber caller env. So all three ride as TRAILING Hydra args (Hydra last-wins) —
# the same mechanism EXP-37 used for 20/20.
#   memory: [[anchor-gradient-ema-beta0-grpo]] [[accel-base-cadence-banner-misleading]]
#
# *** BANNER FOOTGUN ***: the banner will print `beta_anc=0.50 cadence=5 delay_K=5` (it
# echoes the BARE exports, NOT the trailing overrides). DO NOT trust the banner. Verify in
# the resolved main_ppo command in train.log (grep -> LAST value wins) AND the WandB config:
#   spectral.beta_anc=0.0 , anchor.cadence=20 , anchor.delay_K=20
# Latency realized at 20/20: anchor_backwards == 10 (200 ticks / cadence 20), NOT 40.
set -euo pipefail
cd /workspace/verl

git config --global user.email "harness@verl-research.local" 2>/dev/null || true
git config --global user.name  "verl-research-harness"        2>/dev/null || true

export LOG=/workspace/train.log
mkdir -p /workspace/runs/EXP-37C

PROJECT_NAME=verl_compression_research_accel_rebaseline \
EXPERIMENT_NAME=exp-37c-cad20-delay20-beta0-100step \
TOTAL_TRAINING_STEPS=100 \
TEST_FREQ=25 \
LOG=/workspace/train.log \
  bash examples/grpo_trainer/vast_comm_eff_accel_base_qwen25_1p5b_grpo_gsm8k.sh \
  actor_rollout_ref.actor.comm_eff.spectral.beta_anc=0.0 \
  actor_rollout_ref.actor.comm_eff.anchor.cadence=20 \
  actor_rollout_ref.actor.comm_eff.anchor.delay_K=20

echo "$(date -Iseconds) done" > /workspace/runs/EXP-37C/done.flag
