#!/usr/bin/env bash
# EXP-42 scored cells — the look-ahead HORIZON sweep at FIXED staleness.
# delay_K = cadence = 20 held constant (the realistic async lag, same as EXP-41).
# The only axis is the projection strength alpha (lookahead_strength), i.e. HOW
# FAR AHEAD theta_hat predicts: alpha=0.25/0.5/0.75 = 5/10/15 ticks ahead, plus a
# learned (adaptive-coeff) cell cold-started at alpha=1.0. EXP-41 already ran the
# alpha=1.0 (20-ticks-ahead, full catch-up) point = collapse/val 0.0478; the 5/5
# reference band (val@100 0.7066) is reused from EXP-41 cell A (same 1K surface).
# All chained back-to-back in ONE tmux on ONE box.
set -uo pipefail
RUN=/workspace/runs/EXP-42
cd "$RUN"
git config --global user.email "harness@verl-research.local"
git config --global user.name  "verl-research-harness"

if [[ -f "$RUN/exp.bundle" && ! -f "$RUN/.bundle_applied" ]]; then
  cd /workspace
  [[ -d verl ]] && mv verl verl.upstream-vast-ai-workload
  git clone -b exp/42-lookahead-horizon "$RUN/exp.bundle" verl
  cd /workspace/verl
  git remote set-url origin https://github.com/shamanez/verl.git || true
  uv pip install --no-deps -e . > /workspace/pip.log 2>&1 || pip install --no-deps -e . >> /workspace/pip.log 2>&1
  touch "$RUN/.bundle_applied"
fi

LAUNCHER=examples/grpo_trainer/vast_comm_eff_accel_base_qwen25_1p5b_grpo_gsm8k.sh

run_cell () {
  local CELL="$1" MODE="$2" STRENGTH="$3" STEPS="$4" TFREQ="$5"
  local LOG="$RUN/train_${CELL}.log"
  cd /workspace/verl
  echo "=== [EXP-42] cell ${CELL}: cadence=20 delay_K=20 mode=${MODE} strength=${STRENGTH} steps=${STEPS} ===" | tee "$LOG"
  TOTAL_TRAINING_STEPS="${STEPS}" TEST_FREQ="${TFREQ}" VAL_BEFORE_TRAIN=False \
  EXPERIMENT_NAME="exp42-cell${CELL}" \
    bash "$LAUNCHER" \
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

# Horizon sweep (fixed_linear), delay_K=20:
run_cell A25 fixed_linear 0.25 100 25   # predict  5 ticks ahead
echo "$(date -Iseconds) done" > "$RUN/done_A25.flag"
run_cell A50 fixed_linear 0.50 100 25   # predict 10 ticks ahead
echo "$(date -Iseconds) done" > "$RUN/done_A50.flag"
run_cell A75 fixed_linear 0.75 100 25   # predict 15 ticks ahead
echo "$(date -Iseconds) done" > "$RUN/done_A75.flag"
# Adaptive (learned) cell — cold-start alpha=1.0, coeffs adapt online:
run_cell L learned_linear_with_fixed_linear_cold_start 1.0 100 25
echo "$(date -Iseconds) done" > "$RUN/done_L.flag"

echo "$(date -Iseconds) A25+A50+A75+L complete" > "$RUN/done.flag"
echo "=== [EXP-42] all cells complete ==="
