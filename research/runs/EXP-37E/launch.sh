#!/usr/bin/env bash
# EXP-37E launch — ERROR-FEEDBACK merger (ef_powersgd) at its BEST-PROVEN params,
# run at anchor latency 20/20, 100 steps. "Just to check what happens" at high latency
# with the EF method (vs signed_ema, which oscillated/collapsed at 20/20 in EXP-37/37C).
#
# Best-proven EF = EXP-26 arm exp26_B_ef_r2 (W&B tilwe80t, val@50 0.7210 = M6 record):
#   correction_mode=ef_powersgd, ef_clip=1.0, ef_decay=0.9, beta_anc=0.95
#   (signed_ema_alpha is DEAD in ef_powersgd mode). NOTE EXP-26's own alarm: the SIBLING
#   realization with the same clip=1.0 IGNITED — EF is borderline even at 5/5, so 20/20 is
#   a genuine stress test.
#
# Deltas vs the accel base (signed_ema 0.25 / beta_anc 0.50 / 5/5): ALL ride as TRAILING
# Hydra overrides (Hydra last-wins; the accel base bare-exports clobber caller env):
#   correction_mode signed_ema -> ef_powersgd ; ef_clip 0->1.0 ; ef_decay 0->0.9 ;
#   beta_anc 0.50 -> 0.95 ; anchor cadence/delay 5/5 -> 20/20.
#
# *** BANNER FOOTGUN ***: the banner echoes the BARE exports (mode=signed_ema, beta_anc=0.50,
# cadence=5, delay_K=5) — ALL WRONG for this run. Verify the resolved main_ppo command in
# train.log (grep -> LAST value wins) + the WandB config:
#   correction_mode=ef_powersgd, ef_clip=1.0, ef_decay=0.9, beta_anc=0.95, cadence=20, delay_K=20.
# Latency realized at 20/20: anchor_backwards == 10 (200 ticks / cadence 20). First Q + first M
# computed at the first anchor fire = optimizer tick 20 = global step 10 (delay_K=20 ticks).
set -euo pipefail
cd /workspace/verl

git config --global user.email "harness@verl-research.local" 2>/dev/null || true
git config --global user.name  "verl-research-harness"        2>/dev/null || true

export LOG=/workspace/train.log
mkdir -p /workspace/runs/EXP-37E

PROJECT_NAME=verl_compression_research_accel_rebaseline \
EXPERIMENT_NAME=exp-37e-ef-powersgd-cad20-delay20-100step \
TOTAL_TRAINING_STEPS=100 \
TEST_FREQ=25 \
LOG=/workspace/train.log \
  bash examples/grpo_trainer/vast_comm_eff_accel_base_qwen25_1p5b_grpo_gsm8k.sh \
  actor_rollout_ref.actor.comm_eff.spectral.correction_mode=ef_powersgd \
  actor_rollout_ref.actor.comm_eff.spectral.ef_clip=1.0 \
  actor_rollout_ref.actor.comm_eff.spectral.ef_decay=0.9 \
  actor_rollout_ref.actor.comm_eff.spectral.beta_anc=0.95 \
  actor_rollout_ref.actor.comm_eff.anchor.cadence=20 \
  actor_rollout_ref.actor.comm_eff.anchor.delay_K=20

echo "$(date -Iseconds) done" > /workspace/runs/EXP-37E/done.flag
