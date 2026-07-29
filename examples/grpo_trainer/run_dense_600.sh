#!/usr/bin/env bash
# run_dense_600.sh - DENSE control for the #90 comparison. Byte-for-byte the same
# surface as run_prf_exactk_600.sh EXCEPT compression is OFF (COMM_EFF_ENABLED=false,
# the engine's documented dense path). Qwen2.5-Math-1.5B / MATH, batch 128 / ppo-mini
# 128, 1024/2048, gpu_mem 0.72, 600 steps, val 0/150/300/450/600, save_freq 100 -> R2.
set -uo pipefail
BRANCH="autonomous-harness-v1"
REPO="https://github.com/shamanez/verl.git"
WORK="${WORK:-/workspace}"
TOTAL_STEPS="${TOTAL_STEPS:-600}"
TEST_FREQ="${TEST_FREQ:-150}"
GPU_MEM="${GPU_MEM:-0.72}"
RUN_ID="${RUN_ID:-90-dense-600}"
RUN_DIR="$WORK/runs/$RUN_ID"; mkdir -p "$RUN_DIR"; cd "$WORK"
# 1. obtain verl @ autonomous-harness-v1 (fast path reuse else shallow clone)
if [[ -d verl/.git ]] && (cd verl && git remote set-url origin "$REPO" \
     && git fetch --depth 1 origin "$BRANCH" && git checkout -B "$BRANCH" FETCH_HEAD \
     && git reset --hard FETCH_HEAD); then echo "=== reused checkout ==="; else
  { [[ -e verl ]] && mv verl "verl.stale.$(date +%s)"; true; }
  git clone --depth 1 --single-branch -b "$BRANCH" "$REPO" verl; fi
cd "$WORK/verl"; echo "=== verl HEAD: $(git rev-parse HEAD) ==="
# 2. editable install if needed
python3 -c "import verl" 2>/dev/null || { command -v uv >/dev/null 2>&1 && uv pip install --no-deps -e . || python3 -m pip install --no-deps -e . ; }
python3 -c "import verl" || { echo "FATAL: verl import failed" >&2; exit 1; }
# 3. MATH parquet
export DATA_DIR="${DATA_DIR:-$HOME/data/math}"
[[ -f "$DATA_DIR/train.parquet" && -f "$DATA_DIR/test.parquet" ]] || \
  python3 research/scripts/prepare_rlvr_math.py --dataset math --local_save_dir "$DATA_DIR" 2>&1 | tail -6
[[ -f "$DATA_DIR/train.parquet" && -f "$DATA_DIR/test.parquet" ]] || { echo "FATAL: MATH parquet missing" >&2; exit 1; }
# 4. patch base launcher scalars (same three edits as the PRF run)
BASE="examples/grpo_trainer/run_qwen25_math_1p5b_rank1_relex_fsdp.sh"
PATCHED="examples/grpo_trainer/run_dense_600.gen.sh"
sed -e 's/^export MAX_RESPONSE_LENGTH=3072$/export MAX_RESPONSE_LENGTH=2048/' \
    -e 's/^export TRAIN_BATCH_SIZE=512$/export TRAIN_BATCH_SIZE=128/' \
    -e 's/^export PPO_MINI_BATCH_SIZE=256$/export PPO_MINI_BATCH_SIZE=128/' \
    "$BASE" > "$PATCHED"; chmod +x "$PATCHED"
grep -q '^export TRAIN_BATCH_SIZE=128$'     "$PATCHED" || { echo "FATAL batch patch"  >&2; exit 1; }
grep -q '^export PPO_MINI_BATCH_SIZE=128$'  "$PATCHED" || { echo "FATAL mini patch"   >&2; exit 1; }
grep -q '^export MAX_RESPONSE_LENGTH=2048$' "$PATCHED" || { echo "FATAL resp patch"   >&2; exit 1; }
# 5. DENSE: master compression switch OFF (the ONLY science delta vs #90)
export COMM_EFF_ENABLED=false
export ROLLOUT_GPU_MEM_UTIL="$GPU_MEM"
export TOTAL_TRAINING_STEPS="$TOTAL_STEPS"
export TOTAL_EPOCHS=20
export TEST_FREQ="$TEST_FREQ"
export SAVE_FREQ="${SAVE_FREQ:-100}"
export VAL_BEFORE_TRAIN=True
export EXPERIMENT_NAME="$RUN_ID"; export PROJECT_NAME="$RUN_ID"; export WANDB_RUN_GROUP="$RUN_ID"
export LOG="$RUN_DIR/train.log"
[[ -n "${WANDB_API_KEY:-}" ]] || export WANDB_MODE="${WANDB_MODE:-offline}"
echo "=== launching $RUN_ID: DENSE (comm_eff OFF), batch 128 / mini 128, 1024/2048, gpu_mem $GPU_MEM, $TOTAL_STEPS steps, test_freq $TEST_FREQ ==="
exec bash "$PATCHED"
