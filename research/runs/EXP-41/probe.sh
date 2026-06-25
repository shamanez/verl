#!/usr/bin/env bash
# EXP-41 STEP 1 — the fire-forcing CORRECTNESS-INVARIANT probe. Runs INSIDE the
# container in the FOREGROUND, BEFORE the scored cells. Forces the anchor to
# fire every tick (cadence=delay_K=1) and runs ~4 global steps so the look-ahead
# history a fire consumes (theta[t-1], theta[t-2]) is retained and the
# fixed-linear identity (theta_hat==2*theta[t-1]-theta[t-2]) executes. lookahead
# ENABLED fixed_linear; diagnostics ON so the source-snapshot canary + per-fire
# lines print. The full cells stay at 5/5 (A) and 20/20 (B) — this latency
# override is probe-only and never reaches a scored run.
set -uo pipefail
RUN=/workspace/runs/EXP-41
cd "$RUN"

git config --global user.email "harness@verl-research.local"
git config --global user.name  "verl-research-harness"

# Apply the exp/41 bundle if not already (launch.sh also guards on this flag).
if [[ -f "$RUN/exp.bundle" && ! -f "$RUN/.bundle_applied" ]]; then
  cd /workspace
  [[ -d verl ]] && mv verl verl.upstream-vast-ai-workload
  git clone -b exp/41-lookahead-anchor "$RUN/exp.bundle" verl
  cd /workspace/verl
  git remote set-url origin https://github.com/shamanez/verl.git || true
  uv pip install --no-deps -e . > /workspace/pip.log 2>&1 || pip install --no-deps -e . >> /workspace/pip.log 2>&1
  touch "$RUN/.bundle_applied"
fi

LAUNCHER=examples/grpo_trainer/vast_comm_eff_accel_base_qwen25_1p5b_grpo_gsm8k.sh
LOG="$RUN/probe.log"
cd /workspace/verl
echo "=== [EXP-41] FIRE-FORCING PROBE: cadence=1 delay_K=1 lookahead=fixed_linear steps=4 diagnostics=on ===" | tee "$LOG"

TOTAL_TRAINING_STEPS=4 TEST_FREQ=100 VAL_BEFORE_TRAIN=False \
EXPERIMENT_NAME="exp41-probe" \
  bash "$LAUNCHER" \
    actor_rollout_ref.actor.comm_eff.anchor.cadence=1 \
    actor_rollout_ref.actor.comm_eff.anchor.delay_K=1 \
    actor_rollout_ref.actor.comm_eff.anchor.lookahead_anchor=true \
    actor_rollout_ref.actor.comm_eff.anchor.lookahead_mode=fixed_linear \
    actor_rollout_ref.actor.comm_eff.spectral.diagnostics=true \
    >> "$LOG" 2>&1
RC=$?
echo "$(date -Iseconds) probe exit_rc=${RC}" | tee -a "$LOG"
echo "$(date -Iseconds) rc=${RC}" > "$RUN/done_probe.flag"

# Capture the resolved config for the runner's invariant grep.
if [[ -f /workspace/verl/research/scripts/capture_resolved_config.py ]]; then
  python /workspace/verl/research/scripts/capture_resolved_config.py "$LOG" \
    > "$RUN/resolved_params_probe.txt" 2>/dev/null || \
    grep -E "comm_eff|max_response_length|total_training_steps" "$LOG" > "$RUN/resolved_params_probe.txt" 2>/dev/null || true
fi
exit $RC
