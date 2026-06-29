#!/usr/bin/env bash
# EXP-42 per-REGIME launcher (WEIGHT-projection accuracy study) — runs ON the box.
# Two regimes, STRICTLY sequential: regimeA (plain GRPO) -> regimeB (PowerSGD r=77,
# codec ONLY). Re-materialised 2026-06-29 for the 2-regime single-GPU weight-traj
# design (the old 3-cell grad-proj scaffold is SUPERSEDED — prior gradient study).
#
# The two regimes differ ONLY in the compression keys (comm_eff.enabled,
# compression_type, powersgd.rank, anchor.enabled, spectral.enabled). The
# weight-trajectory instrument (probe.weight_traj.*), the RL surface, model, and
# data are IDENTICAL -> the resolved_params diff across regimes is a SUBSET of the
# allowed compression keys (success criterion: controlled variables identical).
#
# Single-GPU is operator-AUTHORISED for EXP-42 (2026-06-29): ALLOW_SINGLE_GPU=1
# relaxes the 4..8-GPU mandate, ROLLOUT_TP=1. Regime B codec is an IN-GRAPH
# activation projection (M_hat=(M@Q)@Qᵀ) so it fires on 1 GPU (no PP/DP needed) —
# verified post-hoc by theta_A != theta_B + nonzero reconstruction error.
set -uo pipefail
REGIME="${1:?usage: run_cell.sh regimeA|regimeB}"
cd /workspace/verl
[[ -f "$HOME/.verl_auth.env" ]] && source "$HOME/.verl_auth.env" || true
export DATA_DIR="${DATA_DIR:-$HOME/data/gsm8k}"

case "$REGIME" in
  regimeA)  # plain GRPO — byte-identical dense path (the clean predictability ceiling)
    export COMM_EFF_ENABLED=false
    ;;
  regimeB)  # GRPO + activation compression, CODEC ONLY (anchor + spectral OFF)
    export COMM_EFF_ENABLED=true
    export COMM_EFF_COMPRESSION_TYPE=powersgd
    export COMM_EFF_POWERSGD_RANK=77
    export COMM_EFF_ANCHOR_ENABLED=false
    export COMM_EFF_SPECTRAL_ENABLED=false
    ;;
  *) echo "usage: run_cell.sh regimeA|regimeB"; exit 2 ;;
esac

# RUN_DIR + WEIGHT_TRAJ_SELECT_ALL let the SAME cell serve both passes:
#   narrow (the 196 projector set):  defaults -> RUN_DIR=/workspace/runs/EXP-42, select_all=false
#   widened (completeness, ALL matrices incl. excluded embed/norm/bias):
#       RUN_DIR=/workspace/runs/EXP-42-all WEIGHT_TRAJ_SELECT_ALL=true
# Separate RUN_DIR keeps the widened sketches from clobbering the narrow study.
RUN_DIR="${RUN_DIR:-/workspace/runs/EXP-42}"
SELECT_ALL="${WEIGHT_TRAJ_SELECT_ALL:-false}"
EXPN="exp42-${REGIME}${EXPN_SUFFIX:-}"
OUT="${RUN_DIR}/${REGIME}"
WEIGHTS="$OUT/weights"
mkdir -p "$WEIGHTS"
# main_ppo (incl. [comm_eff][weight_traj], val, response_length) -> *_internal.log
# (matches the plan's grep). Launcher banner + `set -x` resolved command (ground
# truth for resolved_params) -> driver.log.
INTERNAL_LOG="$OUT/train_${REGIME}_internal.log"

echo "=== EXP-42 $REGIME: comm_eff.enabled=$COMM_EFF_ENABLED weight_traj=ON select_all=$SELECT_ALL exp=$EXPN out_dir=$WEIGHTS ==="
LOG="$INTERNAL_LOG" \
ALLOW_SINGLE_GPU=1 \
ROLLOUT_TP=1 \
MAX_RESPONSE_LENGTH=1024 \
TOTAL_TRAINING_STEPS=80 \
TEST_FREQ=40 \
USE_DYNAMIC_BSZ=True \
VAL_BEFORE_TRAIN=False \
PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-16384}" \
ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.5}" \
EXPERIMENT_NAME="$EXPN" \
bash examples/grpo_trainer/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh \
  actor_rollout_ref.actor.comm_eff.probe.weight_traj.enabled=true \
  actor_rollout_ref.actor.comm_eff.probe.weight_traj.k=4096 \
  actor_rollout_ref.actor.comm_eff.probe.weight_traj.select_all=$SELECT_ALL \
  actor_rollout_ref.actor.comm_eff.probe.weight_traj.out_dir="$WEIGHTS" \
  'actor_rollout_ref.actor.comm_eff.probe.weight_traj.calib_deltas=[10]' \
  'actor_rollout_ref.actor.comm_eff.probe.weight_traj.calib_horizons=[5,10,20]' \
  actor_rollout_ref.actor.comm_eff.probe.weight_traj.calib_stride=0 \
  actor_rollout_ref.actor.comm_eff.probe.weight_traj.calib_max_snapshots=8 \
  > "$OUT/driver.log" 2>&1
RC=$?
echo "$(date -u +%FT%TZ) done $REGIME rc=$RC" > "$OUT/done.flag"
echo "=== $REGIME finished rc=$RC ; internal_log=$INTERNAL_LOG ==="
exit $RC
