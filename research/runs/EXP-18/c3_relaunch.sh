#!/usr/bin/env bash
# EXP-18 / M4 — C3 (final iteration 3/3): blend eta=0.7 + beta_anc=0.0.
# Single-variable change from C2 (which used beta_anc=0.9 and DECLINED 0.13->0.03):
# beta_anc=0.0 => M_anchor = the RAW last delay_K=5 gradient (no ~50-step EMA smear).
# Decides: was C2's degradation the EMA averaging (=> C3 lifts, tunable) or the
# delay_K=5 staleness itself (=> C3 also degrades => STOP: realistic-staleness
# target unreachable by direct stale-gradient forcing). Reuses the exp/18-anchorblend-c5d5
# code already installed at /workspace/verl (blend mode). Config-only relaunch.
set -uo pipefail
cd /workspace/verl
rm -f /workspace/runs/EXP-18/done_curvematch_anchorblend_b0_c5_d5.flag

PROJECT_NAME=comm_eff_curve_match_m4 EXPERIMENT_NAME=curvematch_anchorblend_b0_c5_d5 \
COMM_EFF_ENABLED=true COMM_EFF_MASK_ENABLED=true COMM_EFF_MASK_P=0.9 COMM_EFF_MASK_RESCALE=true COMM_EFF_MASK_RECOMPUTE=true \
COMM_EFF_CLEAN_CADENCE=0 COMM_EFF_ANCHOR_ENABLED=true COMM_EFF_ANCHOR_CADENCE=5 COMM_EFF_ANCHOR_DELAY_K=5 \
COMM_EFF_SPECTRAL_ENABLED=true COMM_EFF_SPECTRAL_MAX_TARGETS=-1 COMM_EFF_SPECTRAL_SEED_ANCHOR_CACHE=false COMM_EFF_SPECTRAL_EMA_DEVICE=cpu \
COMM_EFF_SPECTRAL_BETA_ANC=0.0 \
PPO_MAX_TOKEN_LEN_PER_GPU=18432 LOG_PROB_MAX_TOKEN_LEN_PER_GPU=18432 REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU=18432 \
TOTAL_TRAINING_STEPS=50 VAL_BEFORE_TRAIN=False TEST_FREQ=100000 USE_DYNAMIC_BSZ=True NGPUS_PER_NODE=4 \
bash examples/grpo_trainer/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh \
  actor_rollout_ref.actor.comm_eff.spectral.correction_mode=blend actor_rollout_ref.actor.comm_eff.spectral.blend_eta=0.7 \
  > /workspace/runs/EXP-18/train_curvematch_anchorblend_b0_c5_d5.log 2>&1
RC=$?
echo "$(date -Iseconds) rc=$RC" > /workspace/runs/EXP-18/done_curvematch_anchorblend_b0_c5_d5.flag
echo "=== C3 (blend beta0) finished rc=$RC at $(date -u +%FT%TZ) ==="
