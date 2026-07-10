#!/usr/bin/env bash
# launch_matrix.sh — issue #64 full 2x-seed replication matrix (operator directive
# 2026-07-10). 8 cells = {frozen (train ONLY block L11-15) , dense (full-param)}
# x {gsm8k , bigmath} x {seed data.seed=42 , seed data.seed=7}. All on ONE box,
# on exp/64-dense-wandbfix (= exp/64 freeze-hook code + the #65 wandb final-step
# flush), so EVERY cell's step-75 val lands in wandb natively. comm-eff OFF on all.
# frozen ⇔ TRAIN_LAYERS=11-15 ; dense ⇔ TRAIN_LAYERS unset (hook no-ops). Only the
# seed (data.seed, the data-shuffle/batch-composition knob) and the freeze differ.
#
# Writes training to train.log (the reaper's heartbeat = run.json remote_log);
# each cell archived to train_<cell>.log. Order: gsm8k block (fast, ~35m each)
# then bigmath block (~60m each). ~6.5h total.
set -uo pipefail

RUN_ID=64-middle-block-freeze-grpo
RUN_DIR=/workspace/runs/$RUN_ID
VERL_DIR=/workspace/verl
BIGMATH_DIR=/root/data/bigmath
SECRETS="$HOME/.config/verl-research/secrets.env"
EXP_BRANCH=exp/64-dense-wandbfix
ACCEL_LAUNCHER=examples/grpo_trainer/vast_comm_eff_accel_base_qwen25_1p5b_grpo_gsm8k.sh
BASE_LAUNCHER=examples/grpo_trainer/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh
mkdir -p "$RUN_DIR"
echo "=== [$(date -Iseconds)] launch_matrix.sh START $RUN_ID (2x-seed replication) ==="

# --- secrets ---
if [[ -r "$SECRETS" ]]; then
  # shellcheck disable=SC1090
  source "$SECRETS"
  export HF_TOKEN="${HF_TOKEN:-}" HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}" \
         HUGGINGFACE_HUB_TOKEN="${HF_TOKEN:-}" WANDB_API_KEY="${WANDB_API_KEY:-}"
else echo "FATAL: $SECRETS missing on box." >&2; exit 1; fi

# --- verl bootstrap: clone exp/64-dense-wandbfix (GitHub-first) if not already on it ---
cd /workspace
CUR=$(git -C "$VERL_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || echo none)
if [[ "$CUR" != "$EXP_BRANCH" ]]; then
  echo "=== verl on '$CUR' — cloning $EXP_BRANCH from GitHub ==="
  [[ -e "$VERL_DIR" ]] && mv "$VERL_DIR" "${VERL_DIR}.upstream.$(date +%s)"
  git clone -b "$EXP_BRANCH" https://github.com/shamanez/verl.git "$VERL_DIR" \
    || { echo "FATAL: clone $EXP_BRANCH failed." >&2; exit 1; }
  cd "$VERL_DIR" && git remote set-url origin https://github.com/shamanez/verl.git 2>/dev/null || true
  if ! python3 -c "import verl" 2>/dev/null; then
    (uv pip install --no-deps -e . || pip install --no-deps -e .) > /workspace/pip.log 2>&1 \
      || { echo "FATAL: verl editable install failed." >&2; tail -20 /workspace/pip.log >&2; exit 1; }
  fi
fi
cd "$VERL_DIR"
echo "=== verl @ $(git rev-parse --short HEAD 2>/dev/null) (branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null)) ==="
python3 -c "import verl, torch; print('=== verl OK; torch', torch.__version__, 'cuda', torch.cuda.is_available(), '===')" \
  || { echo "FATAL: verl/torch import failed." >&2; exit 1; }
python3 -c "import verl.utils.tracking as t; assert hasattr(t.Tracking,'finish'), 'wandb fix missing'; print('=== wandb final-step flush present ===')" \
  || { echo "FATAL: Tracking.finish absent — wrong branch." >&2; exit 1; }
python3 -c "from verl.workers.comm_eff.activation_mask import parse_train_layers; print('=== freeze hook present ===')" \
  || { echo "FATAL: freeze hook absent — wrong branch." >&2; exit 1; }

export PROJECT_NAME="$RUN_ID"
export WANDB_RUN_GROUP="$RUN_ID"

CFG_CRASH_RE='OutOfMemoryError|CUDA out of memory|torch\.OutOfMemoryError|ModuleNotFoundError|ImportError:|hydra\.errors|omegaconf\.errors|ConfigAttributeError|MissingMandatoryValue|ConfigKeyError|\[TRAIN_LAYERS\][^A-Za-z]{0,6}(freeze sanity FAILED|grad-flow FAILED|immutability FAILED)'

run_cell() {
  local cell="$1" launcher="$2"; shift 2
  local envs=() hydra=() rc exp
  while [[ $# -gt 0 && "$1" != "--" ]]; do envs+=("$1"); shift; done
  [[ "${1:-}" == "--" ]] && shift
  hydra=("$@")
  exp="64-$cell"
  mkdir -p "$VERL_DIR/runs/$exp"
  echo "=== [$(date -Iseconds)] CELL $cell START (wandb $exp | $(basename "$launcher")) ==="
  env EXPERIMENT_NAME="$exp" LOG="$RUN_DIR/train.log" ${envs[@]+"${envs[@]}"} \
    bash "$launcher" ${hydra[@]+"${hydra[@]}"}
  rc=$?
  cp -f "$RUN_DIR/train.log" "$RUN_DIR/train_${cell}.log" 2>/dev/null || true
  if [[ $rc -eq 0 ]]; then
    echo "$(date -Iseconds)" > "$RUN_DIR/done_${cell}.flag"
    echo "=== [$(date -Iseconds)] CELL $cell DONE (rc=0) ==="; return 0
  fi
  echo "$(date -Iseconds) rc=$rc" > "$RUN_DIR/fail_${cell}.flag"
  echo "=== [$(date -Iseconds)] CELL $cell FAILED (rc=$rc) ===" >&2
  if grep -qE "$CFG_CRASH_RE" "$RUN_DIR/train_${cell}.log" 2>/dev/null; then
    echo "$(date -Iseconds) cell=$cell rc=$rc config-crash; HALTING matrix" > "$RUN_DIR/halt_matrix.flag"
    echo "$(date -Iseconds) halted after $cell" > "$RUN_DIR/done_matrix.flag"
    echo "=== HALT: config-level crash in $cell — NOT advancing ===" >&2; exit 1
  fi
  echo "=== $cell science-level failure — continuing ===" >&2; return "$rc"
}

# ============================ GSM8K BLOCK (accel, resp=1024) ============================
# comm-eff OFF via trailing Hydra (accel wrapper hard-exports COMM_EFF_ENABLED=true).
run_cell frozen-gsm8k-s42 "$ACCEL_LAUNCHER" \
  TRAIN_LAYERS=11-15 TOTAL_TRAINING_STEPS=75 TEST_FREQ=25 VAL_BEFORE_TRAIN=False \
  -- actor_rollout_ref.actor.comm_eff.enabled=false data.seed=42
run_cell dense-gsm8k-s42 "$ACCEL_LAUNCHER" \
  TOTAL_TRAINING_STEPS=75 TEST_FREQ=25 VAL_BEFORE_TRAIN=False \
  -- actor_rollout_ref.actor.comm_eff.enabled=false data.seed=42
run_cell frozen-gsm8k-s7 "$ACCEL_LAUNCHER" \
  TRAIN_LAYERS=11-15 TOTAL_TRAINING_STEPS=75 TEST_FREQ=25 VAL_BEFORE_TRAIN=False \
  -- actor_rollout_ref.actor.comm_eff.enabled=false data.seed=7
run_cell dense-gsm8k-s7 "$ACCEL_LAUNCHER" \
  TOTAL_TRAINING_STEPS=75 TEST_FREQ=25 VAL_BEFORE_TRAIN=False \
  -- actor_rollout_ref.actor.comm_eff.enabled=false data.seed=7

# ---- Big-Math data prep (once, before the bigmath block) ----
BIGMATH_READY=1
if [[ ! -f "$BIGMATH_DIR/train.parquet" || ! -f "$BIGMATH_DIR/test.parquet" ]]; then
  echo "=== [$(date -Iseconds)] prepare Big-Math -> $BIGMATH_DIR ==="
  if ! python3 research/scripts/bigmath_dapo.py --local_save_dir "$BIGMATH_DIR" --train-cap 0 --val-size 500 --seed 42; then
    echo "$(date -Iseconds) big-math prep failed" > "$RUN_DIR/fail_bigmath-prep.flag"
    echo "=== Big-Math prep FAILED — skipping bigmath block ===" >&2; BIGMATH_READY=0
  fi
fi

# ============================ BIG-MATH BLOCK (baseline, resp=4096, dyn-bsz) ============================
BIG_KNOBS=(COMM_EFF_ENABLED=false TOTAL_TRAINING_STEPS=75 TEST_FREQ=25 VAL_BEFORE_TRAIN=False
  USE_DYNAMIC_BSZ=True ROLLOUT_TP=1 ROLLOUT_GPU_MEM_UTIL=0.55 PPO_MAX_TOKEN_LEN_PER_GPU=24576
  TRAIN_BATCH_SIZE=128 PPO_MINI_BATCH_SIZE=64 ROLLOUT_N=8 ACTOR_LR=1e-6 COMM_EFF_CAPTURE_ENABLED=false
  MAX_RESPONSE_LENGTH=4096 MAX_PROMPT_LENGTH=1024 DATA_DIR="$BIGMATH_DIR")
if [[ "$BIGMATH_READY" -eq 1 ]]; then
  run_cell frozen-bigmath-s42 "$BASE_LAUNCHER" TRAIN_LAYERS=11-15 "${BIG_KNOBS[@]}" \
    -- actor_rollout_ref.actor.comm_eff.enabled=false data.seed=42
  run_cell dense-bigmath-s42 "$BASE_LAUNCHER" "${BIG_KNOBS[@]}" \
    -- actor_rollout_ref.actor.comm_eff.enabled=false data.seed=42
  run_cell frozen-bigmath-s7 "$BASE_LAUNCHER" TRAIN_LAYERS=11-15 "${BIG_KNOBS[@]}" \
    -- actor_rollout_ref.actor.comm_eff.enabled=false data.seed=7
  run_cell dense-bigmath-s7 "$BASE_LAUNCHER" "${BIG_KNOBS[@]}" \
    -- actor_rollout_ref.actor.comm_eff.enabled=false data.seed=7
fi

echo "$(date -Iseconds) matrix attempted" > "$RUN_DIR/done_matrix.flag"
echo "=== [$(date -Iseconds)] launch_matrix.sh END $RUN_ID ==="
