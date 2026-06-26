#!/usr/bin/env bash
# EXP-42 per-cell launcher — runs ON the Vast box. Sequential: run1 -> run2 -> run3.
#
# The 3 cells differ ONLY in lookahead_anchor / lookahead_mode / lookahead_strength.
# The grad-projection-accuracy instrument (probe.grad_proj_enabled) + signed_ema
# merger + M substrate + PowerSGD r=77 + anchor 20/20 + everything else is IDENTICAL
# (the EXP-42 controlled surface) — so the resolved_params diff across cells contains
# ONLY the 3 lookahead keys (success criterion).
#
# Surface (overrides the launcher's defaults to the EXP-42 surface):
#   anchor cadence=10 delay_K=10 (shorter staleness => less k-collapse, cleaner read
#   on projection vs raw-stale vs no-projection), max_response=1024, total_steps=100,
#   test_freq=25 (val@25/50/75/100).
# Launcher defaults already match the rest: signed_ema (correction_mode), beta_anc=0.50,
# signed_ema_alpha=0.25, powersgd.rank=77, owns_q=true, clean_cadence=0,
# replay_paired_batch=true, snapshot_device=cpu, ema_device=cpu, max_targets=-1 (196),
# lr=1e-6, train_batch=128, mini=64, rollout.n=8, Qwen2.5-1.5B-Instruct, GSM8K.
set -uo pipefail
CELL="${1:?usage: run_cell.sh run1|run2|run3}"
cd /workspace/verl
[[ -f "$HOME/.verl_auth.env" ]] && source "$HOME/.verl_auth.env" || true
export DATA_DIR="${DATA_DIR:-$HOME/data/gsm8k}"

case "$CELL" in
  run1) LA_ANCHOR=true;  LA_MODE=fixed_linear;                                LA_STR=0.50 ;;
  run2) LA_ANCHOR=true;  LA_MODE=learned_linear_with_fixed_linear_cold_start; LA_STR=0.50 ;;
  run3) LA_ANCHOR=false; LA_MODE=disabled;                                    LA_STR=1.0  ;;  # strength inert when disabled
  *) echo "usage: run_cell.sh run1|run2|run3"; exit 2 ;;
esac

EXPN="exp42-${CELL}"
OUT="/workspace/runs/EXP-42/${CELL}"
mkdir -p "$OUT/gradproj"
# main_ppo (incl. [grad-proj-probe], val, response_length) -> *_internal.log (matches
# the plan's grep train_*_internal.log). Launcher banner + `set -x` resolved command
# (ground truth for resolved_params) -> driver.log.
INTERNAL_LOG="$OUT/train_${CELL}_internal.log"

echo "=== EXP-42 $CELL: lookahead_anchor=$LA_ANCHOR mode=$LA_MODE strength=$LA_STR grad_proj=ON ==="
LOG="$INTERNAL_LOG" \
COMM_EFF_ANCHOR_CADENCE=10 \
COMM_EFF_ANCHOR_DELAY_K=10 \
MAX_RESPONSE_LENGTH=1024 \
TOTAL_TRAINING_STEPS=100 \
TEST_FREQ=25 \
EXPERIMENT_NAME="$EXPN" \
bash examples/grpo_trainer/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh \
  actor_rollout_ref.actor.comm_eff.anchor.lookahead_anchor=$LA_ANCHOR \
  actor_rollout_ref.actor.comm_eff.anchor.lookahead_mode=$LA_MODE \
  actor_rollout_ref.actor.comm_eff.anchor.lookahead_strength=$LA_STR \
  actor_rollout_ref.actor.comm_eff.probe.grad_proj_enabled=true \
  actor_rollout_ref.actor.comm_eff.probe.grad_proj_out_dir="$OUT/gradproj" \
  > "$OUT/driver.log" 2>&1
RC=$?
echo "$(date -u +%FT%TZ) done $CELL rc=$RC" > "$OUT/done.flag"
echo "=== $CELL finished rc=$RC ; internal_log=$INTERNAL_LOG ==="
exit $RC
