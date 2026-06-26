#!/usr/bin/env bash
# EXP-42 final cell: the LEARNED (adaptive-coeff) projector. The 3 fixed-alpha
# cells all collapsed via an OSCILLATING extrapolated-anchor alignment whose
# amplitude scales with alpha (A75 inverted at step 20, A50 at step 40, A25 at
# step ~34). L tests whether online per-block residual coeffs DAMPEN that
# oscillation. delay_K=cadence=20, cold-start alpha=1.0, 100 steps.
set -uo pipefail
RUN=/workspace/runs/EXP-42
cd /workspace/verl

CELL=L
LOG="$RUN/train_${CELL}.log"
echo "=== [EXP-42] cell ${CELL}: cadence=20 delay_K=20 mode=learned_linear_with_fixed_linear_cold_start strength=1.0 steps=100 ===" | tee "$LOG"
TOTAL_TRAINING_STEPS=100 TEST_FREQ=25 VAL_BEFORE_TRAIN=False \
EXPERIMENT_NAME="exp42-cell${CELL}" \
  bash examples/grpo_trainer/vast_comm_eff_accel_base_qwen25_1p5b_grpo_gsm8k.sh \
    actor_rollout_ref.actor.comm_eff.anchor.cadence=20 \
    actor_rollout_ref.actor.comm_eff.anchor.delay_K=20 \
    actor_rollout_ref.actor.comm_eff.anchor.lookahead_anchor=true \
    actor_rollout_ref.actor.comm_eff.anchor.lookahead_mode=learned_linear_with_fixed_linear_cold_start \
    actor_rollout_ref.actor.comm_eff.anchor.lookahead_strength=1.0 \
    >> "$LOG" 2>&1
echo "$(date -Iseconds) cell ${CELL} exit_rc=$?" | tee -a "$LOG"
grep -E "comm_eff|max_response_length|total_training_steps|lookahead" "$LOG" > "$RUN/resolved_params_${CELL}.txt" 2>/dev/null || true
echo "$(date -Iseconds) done" > "$RUN/done_L.flag"
echo "$(date -Iseconds) L complete (A25@38,A75@~27,A50@83 collapsed+skipped)" > "$RUN/done.flag"
echo "=== [EXP-42] cell L complete ==="
