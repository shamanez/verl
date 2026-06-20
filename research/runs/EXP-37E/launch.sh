#!/usr/bin/env bash
# EXP-37E launch — delayed_ef merger (B2) at its BEST-PROVEN config, run at anchor
# latency 20/20, 100 steps. "Just to check what happens" to the best non-signed_ema
# method at high latency (signed_ema oscillated/collapsed at 20/20 in EXP-37/37C).
#
# Best-proven delayed_ef = EXP-30 cell B2 (issue #30, verdict PASS, val@50 0.7528 =
# near-dense, +0.0318 over the ef_powersgd 0.7210 floor, 0.0008 UNDER dense 0.7536):
#   correction_mode=delayed_ef, delayed_ef_lambda=1.0, beta_anc=0.0 (short-memory:
#   M_rep = latest paired G_anchor only, NO EMA). Merger:
#     delta(t) = M_rep - G_comp_ring(t-K) ;  G_corr = G_comp + lambda*delta   (NO sign)
#   i.e. add back the K-delayed exact codec residual (full-vs-compressed gap measured
#   K ticks ago on the SAME (batch, theta) pair).
#
# WHY 20/20 is the test (issue #30 "carrier law"): ignition needs the residual carrier's
# autocorrelation time >> cadence. B2's stability came from beta_anc=0 (delta held <= cadence
# ticks). At 20/20 the held delta stretches to ~20 ticks and the residual is measured against a
# 20-tick-stale pair -> does the near-dense B2 survive 4x its proven 5/5 latency, or ignite?
#
# Deltas vs the accel base (signed_ema 0.25 / beta_anc 0.50 / 5/5) — ALL trailing Hydra
# overrides (Hydra last-wins; the accel base bare-exports clobber caller env):
#   correction_mode signed_ema -> delayed_ef ; delayed_ef_lambda ->1.0 ; beta_anc 0.50 -> 0.0 ;
#   anchor cadence/delay 5/5 -> 20/20.  (ef_clip/ef_decay are ef_powersgd-only -> NOT set.)
#
# *** BANNER FOOTGUN ***: the banner echoes the BARE exports (mode=signed_ema, beta_anc=0.50,
# cadence=5, delay_K=5) — ALL WRONG for this run. Verify the resolved main_ppo command in
# train.log (grep -> LAST value wins) + the WandB config:
#   correction_mode=delayed_ef, delayed_ef_lambda=1.0, beta_anc=0.0, cadence=20, delay_K=20.
# Latency realized at 20/20: anchor_backwards == 10. First M_rep + first Q + first delta at the
# first anchor fire = optimizer tick 20 = global step 10 (delay_K=20 ticks); delta inert (cold) before.
set -euo pipefail
cd /workspace/verl

git config --global user.email "harness@verl-research.local" 2>/dev/null || true
git config --global user.name  "verl-research-harness"        2>/dev/null || true

export LOG=/workspace/train.log
mkdir -p /workspace/runs/EXP-37E

PROJECT_NAME=verl_compression_research_accel_rebaseline \
EXPERIMENT_NAME=exp-37e-delayed-ef-cad20-delay20-100step \
TOTAL_TRAINING_STEPS=100 \
TEST_FREQ=25 \
LOG=/workspace/train.log \
  bash examples/grpo_trainer/vast_comm_eff_accel_base_qwen25_1p5b_grpo_gsm8k.sh \
  actor_rollout_ref.actor.comm_eff.spectral.correction_mode=delayed_ef \
  actor_rollout_ref.actor.comm_eff.spectral.delayed_ef_lambda=1.0 \
  actor_rollout_ref.actor.comm_eff.spectral.beta_anc=0.0 \
  actor_rollout_ref.actor.comm_eff.anchor.cadence=20 \
  actor_rollout_ref.actor.comm_eff.anchor.delay_K=20

echo "$(date -Iseconds) done" > /workspace/runs/EXP-37E/done.flag
