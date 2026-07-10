#!/usr/bin/env bash
# launch_dense.sh — issue #64 DENSE CONTROL arm (operator-added 2026-07-10 for an
# apples-to-apples comparison vs the block-freeze cells). Full-parameter GRPO
# (NO TRAIN_LAYERS ⇒ the freeze hook no-ops ⇒ all 1.54B params train), comm-eff
# OFF, on the SAME box/code(exp/64)/data/surface as the freeze cells — the ONLY
# delta vs freeze-block-l11-15-* is that TRAIN_LAYERS is unset. TWO cells,
# SEQUENTIAL: dense-gsm8k (accel, resp=1024 — fresh S_full on the freeze surface,
# validates the 0.7657 ref) then dense-bigmath (resp=4096 — supplies the MISSING
# Big-Math S_full so C(block) is computable there, not just the fallback).
#
# Runs AFTER launch.sh's freeze cells finish, on the warm box: reuses the already
# cloned /workspace/verl (exp/64) and the already-prepped /root/data/{gsm8k,bigmath}
# — no re-clone, no bundle. Own tmux (run-64-dense), own logs (train_dense.log,
# train_dense-<cell>.log), own flags (done_dense.flag) so it never clobbers the
# freeze run's artifacts.
set -uo pipefail

RUN_ID=64-middle-block-freeze-grpo
RUN_DIR=/workspace/runs/$RUN_ID
VERL_DIR=/workspace/verl
BIGMATH_DIR=/root/data/bigmath
SECRETS="$HOME/.config/verl-research/secrets.env"
ACCEL_LAUNCHER=examples/grpo_trainer/vast_comm_eff_accel_base_qwen25_1p5b_grpo_gsm8k.sh
BASE_LAUNCHER=examples/grpo_trainer/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh
mkdir -p "$RUN_DIR"

echo "=== [$(date -Iseconds)] launch_dense.sh START $RUN_ID (dense control arm) ==="

# --- secrets (HF for any data re-check, WandB for logging) ---
if [[ -r "$SECRETS" ]]; then
  # shellcheck disable=SC1090
  source "$SECRETS"
  export HF_TOKEN="${HF_TOKEN:-}" HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}" \
         HUGGINGFACE_HUB_TOKEN="${HF_TOKEN:-}" WANDB_API_KEY="${WANDB_API_KEY:-}"
else
  echo "FATAL: $SECRETS missing on box." >&2; exit 1
fi

# --- ensure /workspace/verl is exp/64 (same code as the freeze cells: dense =
#     exp/64 with TRAIN_LAYERS unset). Clone from GitHub if the box isn't already
#     on it (fresh box carries the template's pinned branch). GitHub-first, no bundle. ---
EXP_BRANCH=exp/64-dense-wandbfix   # = exp/64 (same freeze-hook code) + the #65 wandb final-step flush, so dense step-75 vals land in wandb natively
cd /workspace
CUR=$(git -C "$VERL_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || echo none)
if [[ "$CUR" != "$EXP_BRANCH" ]]; then
  echo "=== verl on '$CUR' — cloning $EXP_BRANCH from GitHub ==="
  [[ -e "$VERL_DIR" ]] && mv "$VERL_DIR" "${VERL_DIR}.upstream.$(date +%s)"
  git clone -b "$EXP_BRANCH" https://github.com/shamanez/verl.git "$VERL_DIR" \
    || { echo "FATAL: clone $EXP_BRANCH failed." >&2; exit 1; }
  cd "$VERL_DIR" && git remote set-url origin https://github.com/shamanez/verl.git 2>/dev/null || true
  if ! python3 -c "import verl" 2>/dev/null; then
    echo "=== editable install (--no-deps; image provides torch/vLLM) ==="
    (uv pip install --no-deps -e . || pip install --no-deps -e .) > /workspace/pip.log 2>&1 \
      || { echo "FATAL: verl editable install failed." >&2; tail -20 /workspace/pip.log >&2; exit 1; }
  fi
fi
cd "$VERL_DIR"
echo "=== verl @ $(git rev-parse --short HEAD 2>/dev/null) (branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null)) ==="
python3 -c "import verl, torch; print('=== verl OK; torch', torch.__version__, 'cuda', torch.cuda.is_available(), '===')" \
  || { echo "FATAL: verl/torch import failed." >&2; exit 1; }

export PROJECT_NAME="$RUN_ID"
export WANDB_RUN_GROUP="$RUN_ID"

CFG_CRASH_RE='OutOfMemoryError|CUDA out of memory|torch\.OutOfMemoryError|ModuleNotFoundError|ImportError:|hydra\.errors|omegaconf\.errors|ConfigAttributeError|MissingMandatoryValue'

run_cell() {
  local cell="$1" launcher="$2"; shift 2
  local envs=() hydra=() rc exp
  while [[ $# -gt 0 && "$1" != "--" ]]; do envs+=("$1"); shift; done
  [[ "${1:-}" == "--" ]] && shift
  hydra=("$@")
  exp="64-$cell"
  mkdir -p "$VERL_DIR/runs/$exp"
  echo "=== [$(date -Iseconds)] CELL $cell START (wandb run $exp | launcher $(basename "$launcher")) ==="
  # Write to train.log (NOT train_dense.log): the reaper's heartbeat = mtime of
  # the synced train.log. On the first dense run the arm wrote to train_dense.log,
  # so train.log went silent when the freeze cells ended and the box was reaped
  # (no-heartbeat-30min) mid dense-bigmath. Writing here keeps the heartbeat alive.
  env EXPERIMENT_NAME="$exp" LOG="$RUN_DIR/train.log" ${envs[@]+"${envs[@]}"} \
    bash "$launcher" ${hydra[@]+"${hydra[@]}"}
  rc=$?
  cp -f "$RUN_DIR/train.log" "$RUN_DIR/train_${cell}.log" 2>/dev/null || true
  if [[ $rc -eq 0 ]]; then
    echo "$(date -Iseconds)" > "$RUN_DIR/done_${cell}.flag"
    echo "=== [$(date -Iseconds)] CELL $cell DONE (rc=0) ==="
    return 0
  fi
  echo "$(date -Iseconds) rc=$rc" > "$RUN_DIR/fail_${cell}.flag"
  echo "=== [$(date -Iseconds)] CELL $cell FAILED (rc=$rc) ===" >&2
  if grep -qE "$CFG_CRASH_RE" "$RUN_DIR/train_${cell}.log" 2>/dev/null; then
    echo "$(date -Iseconds) cell=$cell rc=$rc — config-level crash; HALTING dense arm" > "$RUN_DIR/halt_dense.flag"
    echo "=== HALT: config-level crash in $cell — NOT advancing ===" >&2
    echo "$(date -Iseconds) halted after $cell" > "$RUN_DIR/done_dense.flag"
    exit 1
  fi
  echo "=== $cell failed at SCIENCE level (no config-crash signature) — continuing ===" >&2
  return "$rc"
}

# ---------------------------------------------------------------------------
# CELL 3 — dense-gsm8k  (accel surface, resp=1024; full-parameter dense =
#   comm-eff OFF + NO TRAIN_LAYERS. Fresh S_full on the exact freeze surface.)
# ---------------------------------------------------------------------------
run_cell dense-gsm8k "$ACCEL_LAUNCHER" \
  TOTAL_TRAINING_STEPS=75 TEST_FREQ=25 VAL_BEFORE_TRAIN=False \
  -- \
  actor_rollout_ref.actor.comm_eff.enabled=false

# ---------------------------------------------------------------------------
# 3b. Big-Math data (should already be prepped by launch.sh's cell 2; guard idempotently).
# ---------------------------------------------------------------------------
BIGMATH_READY=1
if [[ ! -f "$BIGMATH_DIR/train.parquet" || ! -f "$BIGMATH_DIR/test.parquet" ]]; then
  echo "=== [$(date -Iseconds)] Big-Math parquets absent — preparing -> $BIGMATH_DIR ==="
  if ! python3 research/scripts/bigmath_dapo.py --local_save_dir "$BIGMATH_DIR" \
        --train-cap 0 --val-size 500 --seed 42; then
    echo "$(date -Iseconds) big-math data prep failed" > "$RUN_DIR/fail_dense-bigmath.flag"
    echo "=== Big-Math prep FAILED — skipping dense-bigmath (dense-gsm8k stands) ===" >&2
    BIGMATH_READY=0
  fi
fi

# ---------------------------------------------------------------------------
# CELL 4 — dense-bigmath  (resp=4096; full-parameter dense. Supplies the missing
#   Big-Math S_full. Same speed knobs as freeze-block-l11-15-bigmath, minus TRAIN_LAYERS.)
# ---------------------------------------------------------------------------
if [[ "$BIGMATH_READY" -eq 1 ]]; then
  run_cell dense-bigmath "$BASE_LAUNCHER" \
    COMM_EFF_ENABLED=false TOTAL_TRAINING_STEPS=75 TEST_FREQ=25 VAL_BEFORE_TRAIN=False \
    USE_DYNAMIC_BSZ=True ROLLOUT_TP=1 ROLLOUT_GPU_MEM_UTIL=0.55 PPO_MAX_TOKEN_LEN_PER_GPU=24576 \
    TRAIN_BATCH_SIZE=128 PPO_MINI_BATCH_SIZE=64 ROLLOUT_N=8 ACTOR_LR=1e-6 COMM_EFF_CAPTURE_ENABLED=false \
    MAX_RESPONSE_LENGTH=4096 MAX_PROMPT_LENGTH=1024 DATA_DIR="$BIGMATH_DIR" \
    -- \
    actor_rollout_ref.actor.comm_eff.enabled=false
fi

echo "$(date -Iseconds) dense cells attempted" > "$RUN_DIR/done_dense.flag"
echo "=== [$(date -Iseconds)] launch_dense.sh END $RUN_ID ==="
