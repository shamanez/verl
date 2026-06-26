#!/usr/bin/env bash
# EXP-42 remaining cells after A50 (alpha=0.50) collapsed at step 83 and was killed.
# Bundle already applied. Runs A75 (fixed_linear strength=0.75) + L (learned),
# delay_K=cadence=20, 100 steps. Same run_cell contract.
set -uo pipefail
RUN=/workspace/runs/EXP-42
cd /workspace/verl

run_cell () {
  local CELL="$1" MODE="$2" STRENGTH="$3" STEPS="$4" TFREQ="$5"
  local LOG="$RUN/train_${CELL}.log"
  cd /workspace/verl
  echo "=== [EXP-42] cell ${CELL}: cadence=20 delay_K=20 mode=${MODE} strength=${STRENGTH} steps=${STEPS} ===" | tee "$LOG"
  TOTAL_TRAINING_STEPS="${STEPS}" TEST_FREQ="${TFREQ}" VAL_BEFORE_TRAIN=False \
  EXPERIMENT_NAME="exp42-cell${CELL}" \
    bash examples/grpo_trainer/vast_comm_eff_accel_base_qwen25_1p5b_grpo_gsm8k.sh \
      actor_rollout_ref.actor.comm_eff.anchor.cadence=20 \
      actor_rollout_ref.actor.comm_eff.anchor.delay_K=20 \
      actor_rollout_ref.actor.comm_eff.anchor.lookahead_anchor=true \
      actor_rollout_ref.actor.comm_eff.anchor.lookahead_mode="${MODE}" \
      actor_rollout_ref.actor.comm_eff.anchor.lookahead_strength="${STRENGTH}" \
      >> "$LOG" 2>&1
  local RC=$?
  echo "$(date -Iseconds) cell ${CELL} exit_rc=${RC}" | tee -a "$LOG"
  grep -E "comm_eff|max_response_length|total_training_steps|lookahead" "$LOG" > "$RUN/resolved_params_${CELL}.txt" 2>/dev/null || true
  return $RC
}

run_cell A75 fixed_linear 0.75 100 25   # predict 15 ticks ahead
echo "$(date -Iseconds) done" > "$RUN/done_A75.flag"
run_cell L learned_linear_with_fixed_linear_cold_start 1.0 100 25
echo "$(date -Iseconds) done" > "$RUN/done_L.flag"

echo "$(date -Iseconds) A75+L complete (A25@38, A50@83 collapsed+skipped)" > "$RUN/done.flag"
echo "=== [EXP-42] A75+L complete ==="
