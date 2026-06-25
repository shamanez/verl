#!/usr/bin/env bash
# EXP-41 scored cells A->B, chained back-to-back in ONE process (this script),
# run in a detached tmux on the SAME box. Runs INSIDE the Vast.ai container.
# The template onstart cloned shamanez/verl @ vast-ai-workload + pip-installed
# verl; we replace that tree with the exp/41-lookahead-anchor branch (the
# look-ahead patch) from the shipped bundle.
#
# Canonical launcher: vast_comm_eff_accel_base_qwen25_1p5b_grpo_gsm8k.sh. It
# HARDCODES export COMM_EFF_ANCHOR_CADENCE=20 / DELAY_K=20 and bakes signed_ema /
# beta_anc=0.50 / resp=1024. Cadence/delay_K + the lookahead knobs are overridden
# via the launcher's trailing Hydra passthrough ("$@" -> main_ppo, LAST-WINS).
# Re-exporting the env var is clobbered by the launcher's own export, so we ONLY
# use Hydra args. Banner echoes the bare-exported 20/20 — DO NOT trust it; verify
# from resolved_params.txt + lookahead_source_ticks in metrics.
set -uo pipefail
RUN=/workspace/runs/EXP-41
cd "$RUN"

git config --global user.email "harness@verl-research.local"
git config --global user.name  "verl-research-harness"

# ---- apply the exp/41 bundle (code_change=true) -----------------------------
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

run_cell () {
  local CELL="$1" CADENCE="$2" DELAY_K="$3" LA_ANCHOR="$4" LA_MODE="$5" STEPS="$6" TFREQ="$7"
  local LOG="$RUN/train_${CELL}.log"
  cd /workspace/verl
  echo "=== [EXP-41] cell ${CELL}: cadence=${CADENCE} delay_K=${DELAY_K} lookahead_anchor=${LA_ANCHOR} mode=${LA_MODE} steps=${STEPS} ===" | tee "$LOG"
  # TOTAL_TRAINING_STEPS / TEST_FREQ are ${VAR:-} overridable in the launcher.
  # The lookahead knobs + cadence/delay_K overrides ride as TRAILING Hydra args
  # (forwarded through the launcher's "$@" -> main_ppo, last-wins). The lookahead
  # knobs have NO env mapping, so the Hydra arg is the ONLY way to set them.
  TOTAL_TRAINING_STEPS="${STEPS}" TEST_FREQ="${TFREQ}" VAL_BEFORE_TRAIN=False \
  EXPERIMENT_NAME="exp41-cell${CELL}" \
    bash "$LAUNCHER" \
      actor_rollout_ref.actor.comm_eff.anchor.cadence="${CADENCE}" \
      actor_rollout_ref.actor.comm_eff.anchor.delay_K="${DELAY_K}" \
      actor_rollout_ref.actor.comm_eff.anchor.lookahead_anchor="${LA_ANCHOR}" \
      actor_rollout_ref.actor.comm_eff.anchor.lookahead_mode="${LA_MODE}" \
      >> "$LOG" 2>&1
  local RC=$?
  echo "$(date -Iseconds) cell ${CELL} exit_rc=${RC}" | tee -a "$LOG"
  # Capture the resolved Hydra config (ground truth) from the set -x trace.
  if [[ -f /workspace/verl/research/scripts/capture_resolved_config.py ]]; then
    python /workspace/verl/research/scripts/capture_resolved_config.py "$LOG" \
      > "$RUN/resolved_params_${CELL}.txt" 2>/dev/null || \
      grep -E "comm_eff|max_response_length|total_training_steps" "$LOG" > "$RUN/resolved_params_${CELL}.txt" 2>/dev/null || true
  else
    grep -E "comm_eff|max_response_length|total_training_steps" "$LOG" > "$RUN/resolved_params_${CELL}.txt" 2>/dev/null || true
  fi
  return $RC
}

# ---- cell A: 5/5 reference (lookahead DISABLED) ------------------------------
run_cell A 5 5 false disabled 100 25
echo "$(date -Iseconds) done" > "$RUN/done_A.flag"

# ---- cell B: fixed-linear look-ahead at 20/20 -------------------------------
run_cell B 20 20 true fixed_linear 100 25
echo "$(date -Iseconds) done" > "$RUN/done_B.flag"

# ---- aggregate done flag after B --------------------------------------------
echo "$(date -Iseconds) A+B complete" > "$RUN/done.flag"
echo "=== [EXP-41] cells A+B complete ==="
