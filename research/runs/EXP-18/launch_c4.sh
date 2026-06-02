#!/usr/bin/env bash
# EXP-18 / M4 candidate C4 — CLEAN anchor policy-gradient (ratio==1) + blend eta=0.7.
# Runs inside the Vast.ai box (instance 39132674 / 208.64.254.75:23828, 4xH200).
# REUSES the box; the EXP-18 ledger row stays RUNNING. No provisioning.
#
# Code delta vs C3 (the ONLY change): the anchor pass now uses a plain
# policy-gradient loss (anchor_pg_loss: ratio==1, no clip, no old_log_probs)
# instead of the fast-path PPO loss. C1/C2/C3 all degraded the policy because
# the anchor reused the MASKED-path old_log_probs against its UNMASKED forward
# -> importance ratio != 1 -> the PPO clip mangled G_anchor, so M_anchor was
# NEVER the clean true gradient the M4 method assumes. C4 fixes that (gradient
# = -(A*grad logpi_unmasked) at the delay_K=5 stale weights). First VALID test
# of the blend correction.
#
# The exp/18-anchorcleangrad-c5d5 branch (forked from the C2 blend tip 60e616ce,
# inherits the canon-naming fix + blend mode + OOM fixes) is applied by cloning
# /workspace/runs/EXP-18/exp-c4.bundle over /workspace/verl and reinstalling
# editable. 119/119 comm_eff tests pass; the anchor-PG smoke runs before launch.
#
# MANDATORY pins (INVALID if violated): ANCHOR_DELAY_K=5, CLEAN_CADENCE=0,
# ANCHOR_CADENCE=5, MAX_RESPONSE 16384 untouched. beta_anc=0.0 keeps the fresh
# raw anchor; with the clean PG it is now the clean fresh stale gradient.
set -uo pipefail

cd /workspace/verl
rm -f /workspace/runs/EXP-18/done_curvematch_cleangrad_blend_c5_d5.flag

PROJECT_NAME=comm_eff_curve_match_m4 EXPERIMENT_NAME=curvematch_cleangrad_blend_c5_d5 \
COMM_EFF_ENABLED=true COMM_EFF_MASK_ENABLED=true COMM_EFF_MASK_P=0.9 COMM_EFF_MASK_RESCALE=true COMM_EFF_MASK_RECOMPUTE=true \
COMM_EFF_CLEAN_CADENCE=0 COMM_EFF_ANCHOR_ENABLED=true COMM_EFF_ANCHOR_CADENCE=5 COMM_EFF_ANCHOR_DELAY_K=5 \
COMM_EFF_SPECTRAL_ENABLED=true COMM_EFF_SPECTRAL_MAX_TARGETS=-1 COMM_EFF_SPECTRAL_SEED_ANCHOR_CACHE=false COMM_EFF_SPECTRAL_EMA_DEVICE=cpu COMM_EFF_SPECTRAL_BETA_ANC=0.0 \
PPO_MAX_TOKEN_LEN_PER_GPU=18432 LOG_PROB_MAX_TOKEN_LEN_PER_GPU=18432 REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU=18432 \
TOTAL_TRAINING_STEPS=50 VAL_BEFORE_TRAIN=False TEST_FREQ=100000 USE_DYNAMIC_BSZ=True NGPUS_PER_NODE=4 \
bash examples/grpo_trainer/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh \
  actor_rollout_ref.actor.comm_eff.spectral.correction_mode=blend actor_rollout_ref.actor.comm_eff.spectral.blend_eta=0.7 \
  > /workspace/runs/EXP-18/train_curvematch_cleangrad_blend_c5_d5.log 2>&1
RC=$?
echo "$(date -Iseconds) rc=$RC" > /workspace/runs/EXP-18/done_curvematch_cleangrad_blend_c5_d5.flag
echo "=== C4 (clean-PG anchor + blend) finished rc=$RC at $(date -u +%FT%TZ) ==="
