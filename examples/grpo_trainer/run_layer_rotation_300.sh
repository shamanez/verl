#!/usr/bin/env bash
# run_layer_rotation_300.sh - issue #95 layer-rotation GRPO.
#
# Cloned from run_dense_600.sh so surface identity with the reused 90-dense-600
# control is enforced by the LAUNCHER, not by hand: the same three sed patches
# (TRAIN_BATCH_SIZE=128, PPO_MINI_BATCH_SIZE=128, MAX_RESPONSE_LENGTH=2048) with the
# same fail-hard grep assertions, the same COMM_EFF_ENABLED=false dense path, the
# same GPU_MEM=0.72, the same ACTOR_LR=1e-6.
#
# Deltas vs run_dense_600.sh, all deliberate:
#   * BRANCH defaults to the issue's exp branch (the code change lives there; the
#     dense launcher's autonomous-harness-v1 reset would WIPE it).
#   * TOTAL_STEPS 300 (on the control's val grid), TEST_FREQ 150, SAVE_FREQ 300.
#   * LAYER_SCHEDULE / ROTATE_EVERY / ROTATE_ADAM / ROTATE_STATE_DEVICE / LAYER_OTHER
#     threaded through to the ray workers by plain env inheritance (proven on this
#     launcher family by issue #64's TRAIN_LAYERS).
#
# Usage (one cell per invocation):
#   LAYER_SCHEDULE=static:14   VAL_BEFORE_TRAIN=True  EXPERIMENT_NAME=95-static-layer14 bash <thisfile>
#   LAYER_SCHEDULE=rotate:11-15 VAL_BEFORE_TRAIN=False EXPERIMENT_NAME=95-rotate-band5 bash <thisfile>
set -uo pipefail
BRANCH="${BRANCH:-exp/95-layer-rotation-grpo}"
REPO="https://github.com/shamanez/verl.git"
WORK="${WORK:-/workspace}"
TOTAL_STEPS="${TOTAL_STEPS:-300}"
TEST_FREQ="${TEST_FREQ:-150}"
GPU_MEM="${GPU_MEM:-0.72}"
RUN_ID="${RUN_ID:-95-layer-rotation-grpo}"
RUN_DIR="$WORK/runs/$RUN_ID"; mkdir -p "$RUN_DIR"; cd "$WORK"
# 1. obtain verl @ $BRANCH (fast path reuse else shallow clone). Reset is to the
#    EXP branch, so a relaunch always lands on the committed layer-rotation code.
if [[ -d verl/.git ]] && (cd verl && git remote set-url origin "$REPO" \
     && git fetch --depth 1 origin "$BRANCH" && git checkout -B "$BRANCH" FETCH_HEAD \
     && git reset --hard FETCH_HEAD); then echo "=== reused checkout ==="; else
  { [[ -e verl ]] && mv verl "verl.stale.$(date +%s)"; true; }
  git clone --depth 1 --single-branch -b "$BRANCH" "$REPO" verl; fi
cd "$WORK/verl"; echo "=== verl HEAD: $(git rev-parse HEAD) ==="
# 2. editable install if needed
python3 -c "import verl" 2>/dev/null || { command -v uv >/dev/null 2>&1 && uv pip install --no-deps -e . || python3 -m pip install --no-deps -e . ; }
python3 -c "import verl" || { echo "FATAL: verl import failed" >&2; exit 1; }
# 2b. money gate: the layer-rotation hook must be importable from THIS checkout.
#     A stale checkout mislabeled as the code change is the one failure that
#     silently spends the whole budget on a dense run.
python3 -c "from verl.workers.layer_rotation import build_controller, parse_layer_schedule" \
  || { echo "FATAL: verl.workers.layer_rotation missing -- wrong branch/checkout" >&2; exit 1; }
# 3. MATH parquet (byte-identical to the control: defaults are train-cap 20000,
#    val-size 500, seed 42, so the 499-row val set is the control's own).
export DATA_DIR="${DATA_DIR:-$HOME/data/math}"
[[ -f "$DATA_DIR/train.parquet" && -f "$DATA_DIR/test.parquet" ]] || \
  python3 research/scripts/prepare_rlvr_math.py --dataset math --local_save_dir "$DATA_DIR" 2>&1 | tail -6
[[ -f "$DATA_DIR/train.parquet" && -f "$DATA_DIR/test.parquet" ]] || { echo "FATAL: MATH parquet missing" >&2; exit 1; }
# 4. patch base launcher scalars (the SAME three edits as the dense control)
BASE="examples/grpo_trainer/run_qwen25_math_1p5b_rank1_relex_fsdp.sh"
PATCHED="examples/grpo_trainer/run_layer_rotation_300.gen.sh"
sed -e 's/^export MAX_RESPONSE_LENGTH=3072$/export MAX_RESPONSE_LENGTH=2048/' \
    -e 's/^export TRAIN_BATCH_SIZE=512$/export TRAIN_BATCH_SIZE=128/' \
    -e 's/^export PPO_MINI_BATCH_SIZE=256$/export PPO_MINI_BATCH_SIZE=128/' \
    "$BASE" > "$PATCHED"; chmod +x "$PATCHED"
grep -q '^export TRAIN_BATCH_SIZE=128$'     "$PATCHED" || { echo "FATAL batch patch"  >&2; exit 1; }
grep -q '^export PPO_MINI_BATCH_SIZE=128$'  "$PATCHED" || { echo "FATAL mini patch"   >&2; exit 1; }
grep -q '^export MAX_RESPONSE_LENGTH=2048$' "$PATCHED" || { echo "FATAL resp patch"   >&2; exit 1; }
# The base launcher pins ACTOR_LR unconditionally, so our export below cannot move
# it. Assert the pinned value IS the control's 1e-6: C is only interpretable if
# every arm shares arm A's optimiser surface.
grep -q '^export ACTOR_LR=1e-6$'            "$PATCHED" || { echo "FATAL: base ACTOR_LR is not 1e-6" >&2; exit 1; }
# 5. DENSE everywhere: no PowerSGD, no PRF mask, no anchor, no pipeline compression.
#    Issue #95 measures the TRAINABLE SURFACE, not a codec.
export COMM_EFF_ENABLED=false
[[ "$COMM_EFF_ENABLED" == "false" ]] || { echo "FATAL: issue #95 requires COMM_EFF_ENABLED=false" >&2; exit 1; }
export ROLLOUT_GPU_MEM_UTIL="$GPU_MEM"
export ACTOR_LR="${ACTOR_LR:-1e-6}"
export TOTAL_TRAINING_STEPS="$TOTAL_STEPS"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-20}"
export TEST_FREQ="$TEST_FREQ"
export SAVE_FREQ="${SAVE_FREQ:-300}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"
# 6. LAYER_SCHEDULE and its support knobs (0-indexed, inclusive ranges, 28 decoder
#    layers on Qwen2.5-Math-1.5B). Unset/empty means dense (the reused control's
#    surface); the engine prints a LOUD warning in that case so a schedule that
#    failed to reach the worker cannot be mistaken for an arm.
export LAYER_SCHEDULE="${LAYER_SCHEDULE:-}"
export ROTATE_EVERY="${ROTATE_EVERY:-1}"
export ROTATE_ADAM="${ROTATE_ADAM:-persist_park}"
export ROTATE_STATE_DEVICE="${ROTATE_STATE_DEVICE:-cpu}"
export LAYER_OTHER="${LAYER_OTHER:-freeze}"
case "$LAYER_SCHEDULE" in
  ""|static:*|rotate:*) : ;;
  *) echo "FATAL: LAYER_SCHEDULE='$LAYER_SCHEDULE' must be empty, 'static:<idx|lo-hi>' or 'rotate:<idx|lo-hi>'" >&2; exit 1 ;;
esac
case "$LAYER_OTHER" in freeze|train) : ;; *) echo "FATAL: LAYER_OTHER='$LAYER_OTHER' must be freeze|train" >&2; exit 1 ;; esac
case "$ROTATE_ADAM" in persist_park|reset) : ;; *) echo "FATAL: ROTATE_ADAM='$ROTATE_ADAM' must be persist_park|reset" >&2; exit 1 ;; esac
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-$RUN_ID}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-$RUN_ID}"
export PROJECT_NAME="${PROJECT_NAME:-$WANDB_RUN_GROUP}"
export LOG="${LOG:-$RUN_DIR/train.log}"
[[ -n "${WANDB_API_KEY:-}" ]] || export WANDB_MODE="${WANDB_MODE:-offline}"
echo "=== launching $EXPERIMENT_NAME (group $WANDB_RUN_GROUP): DENSE codec, batch 128 / mini 128, 1024/2048, gpu_mem $GPU_MEM, $TOTAL_STEPS steps, test_freq $TEST_FREQ, save_freq $SAVE_FREQ ==="
echo "=== LAYER_SCHEDULE='${LAYER_SCHEDULE:-<unset: DENSE>}' ROTATE_EVERY=$ROTATE_EVERY ROTATE_ADAM=$ROTATE_ADAM ROTATE_STATE_DEVICE=$ROTATE_STATE_DEVICE LAYER_OTHER=$LAYER_OTHER ==="
exec bash "$PATCHED"
