#!/usr/bin/env bash
# EXP-18 / M4 — C5: clean-PG anchor (exp/18-anchorcleangrad-c5d5, already on box) + blend eta=0.9.
# C4 (eta=0.7) PROVED the method: final|Δ|=0.005, slope MATCH, reward 0.13->0.836≈dense 0.841,
# but mean|Δ|=0.077 (>0.05) due to the cadence-5 anchor warmup lag (steps 5-15). C5 raises eta
# 0.7->0.9 (more weight on the CLEAN stale gradient) for faster post-anchor-fire catch-up to
# shrink that warmup-window gap and clinch mean|Δ|<=0.05. Config-only; code unchanged from C4.
# cadence=5 held (plan pin). MANDATORY pins: ANCHOR_DELAY_K=5, CLEAN_CADENCE=0, ANCHOR_CADENCE=5, MAX_RESPONSE 16384.
set -uo pipefail
cd /workspace/verl
rm -f /workspace/runs/EXP-18/done_curvematch_cleangrad_blend_e09_c5_d5.flag

PROJECT_NAME=comm_eff_curve_match_m4 EXPERIMENT_NAME=curvematch_cleangrad_blend_e09_c5_d5 \
COMM_EFF_ENABLED=true COMM_EFF_MASK_ENABLED=true COMM_EFF_MASK_P=0.9 COMM_EFF_MASK_RESCALE=true COMM_EFF_MASK_RECOMPUTE=true \
COMM_EFF_CLEAN_CADENCE=0 COMM_EFF_ANCHOR_ENABLED=true COMM_EFF_ANCHOR_CADENCE=5 COMM_EFF_ANCHOR_DELAY_K=5 \
COMM_EFF_SPECTRAL_ENABLED=true COMM_EFF_SPECTRAL_MAX_TARGETS=-1 COMM_EFF_SPECTRAL_SEED_ANCHOR_CACHE=false COMM_EFF_SPECTRAL_EMA_DEVICE=cpu COMM_EFF_SPECTRAL_BETA_ANC=0.0 \
PPO_MAX_TOKEN_LEN_PER_GPU=18432 LOG_PROB_MAX_TOKEN_LEN_PER_GPU=18432 REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU=18432 \
TOTAL_TRAINING_STEPS=50 VAL_BEFORE_TRAIN=False TEST_FREQ=100000 SAVE_FREQ=100000 USE_DYNAMIC_BSZ=True NGPUS_PER_NODE=4 \
bash examples/grpo_trainer/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh \
  actor_rollout_ref.actor.comm_eff.spectral.correction_mode=blend actor_rollout_ref.actor.comm_eff.spectral.blend_eta=0.9 \
  > /workspace/runs/EXP-18/train_curvematch_cleangrad_blend_e09_c5_d5.log 2>&1
RC=$?
echo "$(date -Iseconds) rc=$RC" > /workspace/runs/EXP-18/done_curvematch_cleangrad_blend_e09_c5_d5.flag
echo "=== C5 (clean-PG + blend eta0.9) finished rc=$RC at $(date -u +%FT%TZ) ==="
