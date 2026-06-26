#!/usr/bin/env bash
# EXP-42 GPU smoke PROBE — validates the FSDP/backend hard-gate invariants the CPU
# probe cannot (1 off-path/dump-only at runtime, 4 same-batch, 5 no-leak in the engine,
# 7 backend/FSDP/HBM): the +2 detached backwards compose with FSDP + grad-ckpt, no
# NaN/OOM, a PROJECTING fire logs a clean [grad-proj-probe] line, cross_rank_max_rel_dev
# ~0 across ranks, and the telemetry-only asserts (no optimizer step / no mask hook) hold.
#
# Short + cheap: cadence=2 delay_K=2 so the look-ahead ring warms (n_points=2) and PROJECTS
# within a few global steps; 4 steps; reduced batch for speed but real max_response=1024.
# Arg 1 = mode: fixed_linear (default, the +2-backward path) | learned | off (run3 +1 path).
set -uo pipefail
MODE="${1:-fixed_linear}"
cd /workspace/verl
[[ -f "$HOME/.verl_auth.env" ]] && source "$HOME/.verl_auth.env" || true
export DATA_DIR="${DATA_DIR:-$HOME/data/gsm8k}"
export WANDB_MODE="${WANDB_MODE:-offline}"   # smoke: no WandB run

case "$MODE" in
  fixed_linear) LA_ANCHOR=true;  LA_MODE=fixed_linear;                                LA_STR=0.50 ;;
  learned)      LA_ANCHOR=true;  LA_MODE=learned_linear_with_fixed_linear_cold_start; LA_STR=0.50 ;;
  off)          LA_ANCHOR=false; LA_MODE=disabled;                                    LA_STR=1.0  ;;
  *) echo "usage: smoke.sh [fixed_linear|learned|off]"; exit 2 ;;
esac

OUT="/workspace/runs/EXP-42/smoke_${MODE}"
mkdir -p "$OUT/gradproj"
echo "=== EXP-42 SMOKE mode=$MODE (anchor=$LA_ANCHOR la_mode=$LA_MODE str=$LA_STR) ==="
LOG="$OUT/train_smoke_internal.log" \
COMM_EFF_ANCHOR_CADENCE=2 \
COMM_EFF_ANCHOR_DELAY_K=2 \
MAX_RESPONSE_LENGTH=1024 \
TOTAL_TRAINING_STEPS=4 \
TEST_FREQ=100 \
VAL_BEFORE_TRAIN=False \
TRAIN_BATCH_SIZE=16 \
PPO_MINI_BATCH_SIZE=8 \
ROLLOUT_N=4 \
EXPERIMENT_NAME="exp42-smoke-${MODE}" \
bash examples/grpo_trainer/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh \
  actor_rollout_ref.actor.comm_eff.anchor.lookahead_anchor=$LA_ANCHOR \
  actor_rollout_ref.actor.comm_eff.anchor.lookahead_mode=$LA_MODE \
  actor_rollout_ref.actor.comm_eff.anchor.lookahead_strength=$LA_STR \
  actor_rollout_ref.actor.comm_eff.probe.grad_proj_enabled=true \
  actor_rollout_ref.actor.comm_eff.probe.grad_proj_out_dir="$OUT/gradproj" \
  > "$OUT/driver.log" 2>&1
RC=$?
echo "rc=$RC done" > "$OUT/done.flag"

echo "=========================== SMOKE VERDICT (mode=$MODE rc=$RC) ==========================="
L="$OUT/train_smoke_internal.log"
echo "-- [grad-proj-probe] lines --"; grep -E "\[grad-proj-probe\]" "$L" 2>/dev/null | tail -8 || echo "NONE (FAIL: no probe fire logged)"
echo "-- crash signatures (expect NONE) --"; grep -nE "Traceback|CUDA out of memory|OutOfMemory|NaN|assert .*GUARD|AssertionError|Error" "$L" "$OUT/driver.log" 2>/dev/null | grep -ivE "error_if|no error|max_position" | tail -10 || echo "none"
echo "-- val/loss progressed? --"; grep -oE "actor/(pg_loss|grad_norm):[-0-9.e]+|val-core[^ ]*:[0-9.]+" "$L" 2>/dev/null | tail -6 || true
echo "PROBE_SMOKE_DONE rc=$RC"
