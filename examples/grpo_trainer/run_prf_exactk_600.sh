#!/usr/bin/env bash
# run_prf_exactk_600.sh
# Self-contained bring-up and launch of a 600-step communication-efficient GRPO
# run on ONE H200 (>=140GB), Qwen2.5-Math-1.5B on MATH, using the best PRF
# activation codec found in issue #89: prf_mask, p=0.95, constant rescale,
# exact-k (fixed 77 of 1536 values per token, 95 percent compression).
#
# Deltas from the project default, per the operator:
#   train_batch_size 512 -> 128, ppo_mini_batch_size 256 -> 128
#     (128/128 = ONE on-policy optimizer update per generation, ratio == 1)
#   max_response_length 3072 -> 2048  (max_prompt_length stays 1024)
#   rollout gpu_memory_utilization 0.55 -> 0.72  (bigger KV cache = faster rollout)
#   total_training_steps 600, val at 0/150/300/450/600
#
# Run inside tmux. The engine tees training to $LOG (the heartbeat log).
set -uo pipefail

# -------- knobs you may edit --------
BRANCH="autonomous-harness-v1"          # has the merged #89 codec + engine
REPO="https://github.com/shamanez/verl.git"
WORK="${WORK:-/workspace}"
TOTAL_STEPS="${TOTAL_STEPS:-600}"
TEST_FREQ="${TEST_FREQ:-150}"           # set -1 to disable val for max throughput
GPU_MEM="${GPU_MEM:-0.72}"              # dial to 0.55 if you hit OOM, up to ~0.80 if not
RUN_ID="${RUN_ID:-90-prf-exactk-600}"
# ------------------------------------

RUN_DIR="$WORK/runs/$RUN_ID"
mkdir -p "$RUN_DIR"
cd "$WORK"

# 1. Obtain verl @ autonomous-harness-v1 (fast path reuse, else shallow clone).
if [[ -d verl/.git ]] \
   && (cd verl && git remote set-url origin "$REPO" \
       && git fetch --depth 1 origin "$BRANCH" && git checkout -B "$BRANCH" FETCH_HEAD \
       && git reset --hard FETCH_HEAD); then
  echo "=== reused checkout, reset to origin/$BRANCH ==="
else
  { [[ -e verl ]] && mv verl "verl.stale.$(date +%s)"; true; }
  git clone --depth 1 --single-branch -b "$BRANCH" "$REPO" verl
fi
cd "$WORK/verl"
echo "=== verl HEAD: $(git rev-parse HEAD) ==="

# 2. Editable install (no deps) if verl is not importable yet.
python3 -c "import verl" 2>/dev/null || {
  if command -v uv >/dev/null 2>&1; then uv pip install --no-deps -e . ; else python3 -m pip install --no-deps -e . ; fi
}
python3 -c "import verl" || { echo "FATAL: verl import failed after install" >&2; exit 1; }

# 2b. Money gate: prove the exact-k codec lever is really in this checkout.
python3 - <<'PY' || { echo "FATAL: prf_mask exact_k lever absent from checkout" >&2; exit 1; }
import inspect
from verl.workers.comm_eff.activation_mask import prf_token_mask
assert "exact_k" in inspect.signature(prf_token_mask).parameters, "prf_token_mask missing exact_k"
print("OK: prf_mask exact_k present")
PY

# 3. MATH parquet ($HOME/data/math), prepared with the canonical research prep.
export DATA_DIR="${DATA_DIR:-$HOME/data/math}"
if [[ ! -f "$DATA_DIR/train.parquet" || ! -f "$DATA_DIR/test.parquet" ]]; then
  echo "=== preparing MATH parquet in $DATA_DIR ==="
  python3 research/scripts/prepare_rlvr_math.py --dataset math --local_save_dir "$DATA_DIR" 2>&1 | tail -6
fi
[[ -f "$DATA_DIR/train.parquet" && -f "$DATA_DIR/test.parquet" ]] \
  || { echo "FATAL: MATH parquet unavailable in $DATA_DIR" >&2; exit 1; }

# 4. Patched launcher copy: only the three hardcoded scalars change.
BASE="examples/grpo_trainer/run_qwen25_math_1p5b_rank1_relex_fsdp.sh"
PATCHED="examples/grpo_trainer/run_prf_exactk_600.gen.sh"
sed -e 's/^export MAX_RESPONSE_LENGTH=3072$/export MAX_RESPONSE_LENGTH=2048/' \
    -e 's/^export TRAIN_BATCH_SIZE=512$/export TRAIN_BATCH_SIZE=128/' \
    -e 's/^export PPO_MINI_BATCH_SIZE=256$/export PPO_MINI_BATCH_SIZE=128/' \
    "$BASE" > "$PATCHED"
chmod +x "$PATCHED"
# fail loud if the base launcher shape drifted and a substitution missed
grep -q '^export TRAIN_BATCH_SIZE=128$'    "$PATCHED" || { echo "FATAL: batch patch missed"    >&2; exit 1; }
grep -q '^export PPO_MINI_BATCH_SIZE=128$' "$PATCHED" || { echo "FATAL: mini-batch patch missed" >&2; exit 1; }
grep -q '^export MAX_RESPONSE_LENGTH=2048$' "$PATCHED" || { echo "FATAL: response patch missed"  >&2; exit 1; }

# 5. Best-PRF codec (exact env combo proven by the #89 prf-exact-k cell) + run controls.
export COMM_EFF_ENABLED=true
export COMM_EFF_COMPRESSION_TYPE=prf_mask
export COMM_EFF_MASK_ENABLED=true
export COMM_EFF_MASK_P=0.95
export COMM_EFF_MASK_RESCALE_MODE=constant
export COMM_EFF_MASK_EXACT_K=true
export COMM_EFF_MASK_RECOMPUTE=true
export COMM_EFF_MASK_REFERENCE=true
export COMM_EFF_MASK_PP_SIZE=8
export COMM_EFF_ANCHOR_OWNS_Q=false                 # prf_mask cannot anchor-own-Q
export COMM_EFF_POWERSGD_FAST_Q_BOOTSTRAP=false
export ROLLOUT_GPU_MEM_UTIL="$GPU_MEM"
export TOTAL_TRAINING_STEPS="$TOTAL_STEPS"
export TOTAL_EPOCHS=20                               # >=11 needed so 600 steps is the stop at batch 128
export TEST_FREQ="$TEST_FREQ"
export SAVE_FREQ=-1
export VAL_BEFORE_TRAIN=True
export EXPERIMENT_NAME="$RUN_ID"
export PROJECT_NAME="$RUN_ID"
export WANDB_RUN_GROUP="$RUN_ID"
export LOG="$RUN_DIR/train.log"
# WandB: use it if a key is present, else fall back to offline (no crash).
[[ -n "${WANDB_API_KEY:-}" ]] || export WANDB_MODE="${WANDB_MODE:-offline}"

echo "=== launching $RUN_ID: prf_mask+exact_k (constant rescale), batch 128 / mini 128 (1 on-policy tick/step), 1024/2048, gpu_mem $GPU_MEM, $TOTAL_STEPS steps, test_freq $TEST_FREQ ==="
exec bash "$PATCHED"
