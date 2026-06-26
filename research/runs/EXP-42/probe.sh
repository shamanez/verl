#!/usr/bin/env bash
# EXP-42 STEP 1 — fire-forcing invariant probes for the NEW code paths, run
# BEFORE the scored cells. Forces the anchor to fire every tick (cadence=delay_K=1)
# so the look-ahead executes within a few global steps. Two probes back-to-back:
#   P1 fixed-alpha (lookahead_strength=0.5): verify the NEW horizon knob plumbs
#      (diagnostic prints strength=0.5000, coeffs (1.5,-0.5,0)) and all EXP-41
#      fixed_linear invariants still hold (identity, no-leak, isolation, canary).
#   P2 learned: the never-run learned path — verify cold-start identity (first
#      learned fire == fixed prediction), the cross-rank learned-coeff max-rel-dev
#      scalar emits & ~0, no-leak of the retrospective residual, bounded 3-pt ring.
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

probe_run () {
  local TAG="$1" MODE="$2" STRENGTH="$3" STEPS="$4"
  local LOG="$RUN/probe_${TAG}.log"
  cd /workspace/verl
  echo "=== [EXP-42] PROBE ${TAG}: cadence=1 delay_K=1 mode=${MODE} strength=${STRENGTH} steps=${STEPS} diagnostics=on ===" | tee "$LOG"
  TOTAL_TRAINING_STEPS="${STEPS}" TEST_FREQ=100 VAL_BEFORE_TRAIN=False \
  EXPERIMENT_NAME="exp42-probe-${TAG}" \
    bash "$LAUNCHER" \
      actor_rollout_ref.actor.comm_eff.anchor.cadence=1 \
      actor_rollout_ref.actor.comm_eff.anchor.delay_K=1 \
      actor_rollout_ref.actor.comm_eff.anchor.lookahead_anchor=true \
      actor_rollout_ref.actor.comm_eff.anchor.lookahead_mode="${MODE}" \
      actor_rollout_ref.actor.comm_eff.anchor.lookahead_strength="${STRENGTH}" \
      actor_rollout_ref.actor.comm_eff.spectral.diagnostics=true \
      >> "$LOG" 2>&1
  echo "$(date -Iseconds) probe ${TAG} exit_rc=$?" | tee -a "$LOG"
}

# P1 — fixed-alpha horizon knob (alpha=0.5 ⇒ predict 1 tick ahead at delay_K=1; the
# point is to exercise coeffs=(1.5,-0.5,0) and confirm the knob plumbs end-to-end).
probe_run fixed_alpha fixed_linear 0.5 4
# P2 — learned path (3-pt ring + retrospective residual + cross-rank coeff dev).
probe_run learned learned_linear_with_fixed_linear_cold_start 1.0 6

echo "$(date -Iseconds) probes complete" > "$RUN/done_probe.flag"
