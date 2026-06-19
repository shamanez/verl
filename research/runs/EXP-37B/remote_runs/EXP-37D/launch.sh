#!/usr/bin/env bash
# EXP-37D launch — DENSE control (comm-eff OFF) on the accel surface, 100 steps.
# Same accel surface as EXP-37B/37C (resp 2048, dynamic bsz, rollout TP=1, gpu_mem 0.55,
# ppo_max_token 24576, batch 128 / mini 64, n=8, lr 1e-6, total_epochs=2) but the comm-eff
# MASTER SWITCH is OFF => byte-identical to upstream dense verl.
#
# PURPOSE: does crossing the GSM8K epoch-2 boundary (~step 58) out to 100 steps cause
# back-half (50-100) instability even WITHOUT compression? Isolates dataset-revisit / GRPO
# epoch effects from the compressed-merger instability. Dense@50 reference = EXP-36C (~0.7657).
# Overlays EXP-37B (signed_ema beta0.50), EXP-37C (beta0), EXP-37 (20/20), EXP-36B/36C in the
# same WandB project.
#
# comm-eff OFF CANNOT be set via env: the accel base BARE-exports COMM_EFF_ENABLED=true,
# clobbering caller env. So we override it as a TRAILING Hydra arg (Hydra last-wins). The
# comm-eff substrate args (powersgd/anchor/spectral) + signed_ema_alpha/diagnostics are still
# passed by the launcher but are INERT once enabled=false (dense backward path).
#
# *** BANNER FOOTGUN ***: the banner prints `comm_eff master: true` (it echoes the bare
# export, not the override). DO NOT trust it — verify comm_eff.enabled=false in the resolved
# main_ppo command in train.log (grep "comm_eff.enabled=" -> last wins) and the WandB config.
set -euo pipefail
cd /workspace/verl

git config --global user.email "harness@verl-research.local" 2>/dev/null || true
git config --global user.name  "verl-research-harness"        2>/dev/null || true

export LOG=/workspace/train.log
mkdir -p /workspace/runs/EXP-37D

PROJECT_NAME=verl_compression_research_accel_rebaseline \
EXPERIMENT_NAME=exp-37d-dense-100step \
TOTAL_TRAINING_STEPS=100 \
TEST_FREQ=25 \
LOG=/workspace/train.log \
  bash examples/grpo_trainer/vast_comm_eff_accel_base_qwen25_1p5b_grpo_gsm8k.sh \
  actor_rollout_ref.actor.comm_eff.enabled=false

echo "$(date -Iseconds) done" > /workspace/runs/EXP-37D/done.flag
