#!/usr/bin/env bash
# EXP-18 cell-2 RE-RUN — spectral floor (as-implemented) with the anchor-OOM fix.
# First attempt OOM'd in the anchor's unsharded full backward (_maybe_comm_eff_anchor_refresh
# -> _forward_backward_batch_inner -> loss.backward()) at the first anchor fire (~step 3).
# Fix = the launcher's own documented anchor-ON mitigation (line ~262): halve
# PPO_MAX_TOKEN_LEN_PER_GPU 36864 -> 18432 (shrinks the per-micro-batch activation
# footprint of BOTH the fast path AND the anchor backward). NO code change, NO response-length
# reduction (16384 fixed by mandate), NO method-math change. Same constraint pins as before.
set -uo pipefail
cd /workspace/verl
# Clear the spurious done flags the chain wrapper wrote through the OOM crash.
rm -f /workspace/runs/EXP-18/done.flag /workspace/runs/EXP-18/done_curvematch_spectral_baseline_c5_d5.flag
mkdir -p /workspace/runs/EXP-18

PROJECT_NAME=comm_eff_curve_match_m4 EXPERIMENT_NAME=curvematch_spectral_baseline_c5_d5 \
COMM_EFF_ENABLED=true \
COMM_EFF_MASK_ENABLED=true COMM_EFF_MASK_P=0.9 COMM_EFF_MASK_RESCALE=true COMM_EFF_MASK_RECOMPUTE=true \
COMM_EFF_CLEAN_CADENCE=0 \
COMM_EFF_ANCHOR_ENABLED=true COMM_EFF_ANCHOR_CADENCE=5 COMM_EFF_ANCHOR_DELAY_K=5 \
COMM_EFF_SPECTRAL_ENABLED=true \
PPO_MAX_TOKEN_LEN_PER_GPU=18432 LOG_PROB_MAX_TOKEN_LEN_PER_GPU=18432 REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU=18432 \
TOTAL_TRAINING_STEPS=50 VAL_BEFORE_TRAIN=False TEST_FREQ=100000 USE_DYNAMIC_BSZ=True \
NGPUS_PER_NODE=4 \
bash examples/grpo_trainer/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh \
  > /workspace/runs/EXP-18/train_curvematch_spectral_baseline_c5_d5.log 2>&1
RC=$?
echo "$(date -Iseconds) rc=$RC" > /workspace/runs/EXP-18/done_curvematch_spectral_baseline_c5_d5.flag
echo "=== floor rerun finished rc=$RC at $(date -u +%FT%TZ) ==="
