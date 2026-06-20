#!/usr/bin/env bash
# EXP-37E launch — delayed_ef merger (B2 family) at anchor latency 20/20, beta_anc=0.5,
# 100 steps. Tests the best non-signed_ema method (delayed_ef = EXP-30 cell B2, val@50
# 0.7528 = near-dense at 5/5) at high latency.
#
# delayed_ef merger (direction-preserving, NO sign term):
#     M_rep    = beta_anc-EMA of the stale paired anchor gradient G_anchor
#     delta(t) = M_rep - G_comp_ring(t-K)        # K-delayed codec residual on the same (batch,theta) pair
#     G_corr   = G_comp + lambda * delta(t)      # add back the dropped full-vs-compressed gap
#   refreshed at each anchor fire, held between fires.
#
# CONFIG (operator-chosen):
#   correction_mode = delayed_ef ; delayed_ef_lambda = 1.0 ; beta_anc = 0.5 ;
#   anchor cadence/delay_K = 20/20.
# NOTE on beta_anc=0.5: B2 SOTA proved delayed_ef at beta_anc=0 (short memory). Operator's
# choice of beta_anc=0.5 here is motivated by signed_ema collapsing SHARPLY at beta_anc=0
# (EXP-37C, 20/20) -> a half-life-smoothed M_rep is the variable under test at 20/20. So
# M_rep is a 2-fire EMA of G_anchor, NOT the latest-only used by B2 SOTA.
#
# Deltas vs the accel base (signed_ema 0.25 / beta_anc 0.50 / 5/5) — ALL trailing Hydra
# overrides (Hydra last-wins; the accel base bare-exports clobber caller env):
#   correction_mode signed_ema -> delayed_ef ; delayed_ef_lambda -> 1.0 ;
#   beta_anc 0.50 -> 0.5 (same value, set explicitly) ; anchor cadence/delay 5/5 -> 20/20.
#
# *** BANNER FOOTGUN ***: the banner echoes the BARE exports (mode=signed_ema, cadence=5,
# delay_K=5) — WRONG for this run. Verify the resolved main_ppo command in train.log
# (grep -> LAST value wins) + the WandB config:
#   correction_mode=delayed_ef, delayed_ef_lambda=1.0, beta_anc=0.5, cadence=20, delay_K=20.
# Latency realized at 20/20: anchor_backwards == 10. First M_rep + first Q + first delta at the
# first anchor fire = optimizer tick 20 = global step 10 (delay_K=20 ticks); delta inert before.
set -euo pipefail
cd /workspace/verl

git config --global user.email "harness@verl-research.local" 2>/dev/null || true
git config --global user.name  "verl-research-harness"        2>/dev/null || true

export LOG=/workspace/train.log
mkdir -p /workspace/runs/EXP-37E

PROJECT_NAME=verl_compression_research_accel_rebaseline \
EXPERIMENT_NAME=exp-37e-delayed-ef-cad20-delay20-beta05-100step \
TOTAL_TRAINING_STEPS=100 \
TEST_FREQ=25 \
LOG=/workspace/train.log \
  bash examples/grpo_trainer/vast_comm_eff_accel_base_qwen25_1p5b_grpo_gsm8k.sh \
  actor_rollout_ref.actor.comm_eff.spectral.correction_mode=delayed_ef \
  actor_rollout_ref.actor.comm_eff.spectral.delayed_ef_lambda=1.0 \
  actor_rollout_ref.actor.comm_eff.spectral.beta_anc=0.5 \
  actor_rollout_ref.actor.comm_eff.anchor.cadence=20 \
  actor_rollout_ref.actor.comm_eff.anchor.delay_K=20

echo "$(date -Iseconds) done" > /workspace/runs/EXP-37E/done.flag
