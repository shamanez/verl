#!/usr/bin/env bash
# launch.sh — issue #64: blockwise-freeze GRPO. Train ONLY decoder block L11-15
# (0-indexed, center L14) for 75 steps; comm-eff OFF (dense). TWO cells, SEQUENTIAL,
# one box: GSM8K (accel surface, resp=1024) then Big-Math (resp=4096).
#
# The freeze is driven by the NEW env knob TRAIN_LAYERS=11-15 (inclusive 0-indexed
# range; unset/empty ⇒ dense parity), implemented on exp/64 in
#   verl/workers/engine/fsdp/transformer_impl.py  (freeze hook + invariant-6 asserts)
#   verl/workers/comm_eff/activation_mask.py       (parse_train_layers/apply_block_freeze)
# The engine logs "[TRAIN_LAYERS] freeze ACTIVE ... trainable=X/Y" + grad-flow +
# immutability at step 1 (the on-box freeze-smoke) and HARD-RAISES on a bad freeze.
#
# Run under tmux; stdout -> /workspace/runs/<id>/launch.log (orchestration + bootstrap
# + data-prep). Per-cell TRAINING output -> train.log (the monitor's remote_log, live
# global_step of the ACTIVE cell); each finished cell preserved to train_<cell>.log.
#
# NOT set -e: a cell's non-zero exit is recorded (fail_<cell>.flag). A CONFIG-LEVEL
# crash (OOM / ModuleNotFound / hydra / freeze money-gate FAIL) HALTS the sweep — it
# would recur in the next cell (both cells share the freeze hook + memory config), so
# auto-advancing just pre-pays boot+crash (#63 cell-failure policy). Only a science
# NaN/divergence in THIS cell's numbers falls through to the next cell.
set -uo pipefail

RUN_ID=64-middle-block-freeze-grpo
RUN_DIR=/workspace/runs/$RUN_ID
BASE_BRANCH=autonomous-harness-v1
EXP_BRANCH=exp/64-middle-block-freeze-grpo
VERL_DIR=/workspace/verl
BUNDLE="$RUN_DIR/exp.bundle"
BIGMATH_DIR=/root/data/bigmath
SECRETS="$HOME/.config/verl-research/secrets.env"
ACCEL_LAUNCHER=examples/grpo_trainer/vast_comm_eff_accel_base_qwen25_1p5b_grpo_gsm8k.sh
BASE_LAUNCHER=examples/grpo_trainer/vast_comm_eff_baseline_qwen25_1p5b_grpo_gsm8k.sh
mkdir -p "$RUN_DIR"

echo "=== [$(date -Iseconds)] launch.sh START $RUN_ID ==="

# ---------------------------------------------------------------------------
# 0. Secrets (Big-Math prep needs HF; each cell launcher re-sources + validates).
# ---------------------------------------------------------------------------
if [[ -r "$SECRETS" ]]; then
  # shellcheck disable=SC1090
  source "$SECRETS"
  export HF_TOKEN="${HF_TOKEN:-}" \
         HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}" \
         HUGGINGFACE_HUB_TOKEN="${HF_TOKEN:-}" \
         WANDB_API_KEY="${WANDB_API_KEY:-}"
else
  echo "FATAL: $SECRETS missing on box — push a stripped HF_TOKEN+WANDB_API_KEY copy before launch." >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# 1. verl bootstrap — code_change:true ⇒ the exp/64 CODE travels in exp.bundle.
#    Replace the template's pinned clone with the bundle so the freeze hook is the
#    verl SOURCE. Editable install (image provides torch/vLLM/ray). A missing
#    bundle falls back to the base branch (freeze would then be absent → the
#    engine's WARNING + trainable-count log flag it before spend).
# ---------------------------------------------------------------------------
cd /workspace
if [[ -f "$BUNDLE" ]]; then
  echo "=== code_change: cloning exp/64 from $BUNDLE ==="
  [[ -e "$VERL_DIR" ]] && mv "$VERL_DIR" "${VERL_DIR}.upstream.$(date +%s)"
  git clone -b "$EXP_BRANCH" "$BUNDLE" "$VERL_DIR"
  cd "$VERL_DIR"
  git remote set-url origin https://github.com/shamanez/verl.git 2>/dev/null || true
else
  echo "WARN: $BUNDLE absent — falling back to $BASE_BRANCH (freeze hook may be MISSING)." >&2
  if [[ ! -d "$VERL_DIR/.git" ]]; then
    [[ -e "$VERL_DIR" ]] && mv "$VERL_DIR" "${VERL_DIR}.stale.$(date +%s)"
    git clone https://github.com/shamanez/verl.git "$VERL_DIR"
  fi
  cd "$VERL_DIR"
  git remote set-url origin https://github.com/shamanez/verl.git 2>/dev/null || true
  git fetch origin "$BASE_BRANCH"
  git checkout -B "$BASE_BRANCH" FETCH_HEAD
fi
CUR="$(git rev-parse --abbrev-ref HEAD 2>/dev/null)"
echo "=== verl @ $(git rev-parse --short HEAD) (branch=$CUR) ==="
# Editable install so the exp/64 source (freeze hook) is what runs.
if ! python3 -c "import verl" 2>/dev/null; then
  echo "=== verl not importable — editable install (--no-deps; image provides torch/vLLM) ==="
  (uv pip install --no-deps -e . || pip install --no-deps -e .) > /workspace/pip.log 2>&1 \
    || { echo "FATAL: verl editable install failed — see /workspace/pip.log" >&2; tail -25 /workspace/pip.log >&2; exit 1; }
fi
python3 -c "import verl, torch; print('=== verl OK; torch', torch.__version__, 'cuda', torch.cuda.is_available(), '===')" \
  || { echo "FATAL: verl/torch import failed after bootstrap." >&2; exit 1; }
# Prove the freeze hook is actually present in the installed source (money gate:
# a stale/wrong checkout must NOT silently run dense and mislabel it as a freeze).
python3 -c "from verl.workers.comm_eff.activation_mask import parse_train_layers, apply_block_freeze; print('=== TRAIN_LAYERS freeze hook present (exp/64) ===')" \
  || { echo "FATAL: exp/64 freeze hook (parse_train_layers/apply_block_freeze) NOT importable — refusing to spend on a dense run mislabeled as a freeze." >&2; exit 1; }

# ---------------------------------------------------------------------------
# 2. Shared WandB identity (per-issue project = run_id; group both cells).
#    entity resolves from the box's WANDB login (shamanework-pl) as in every
#    prior run — not forced here.
# ---------------------------------------------------------------------------
export PROJECT_NAME="$RUN_ID"
export WANDB_RUN_GROUP="$RUN_ID"

# ---------------------------------------------------------------------------
# 3. Cell runner. LOG=$RUN_DIR/train.log for every cell (single live log the
#    monitor tails); preserved to train_<cell>.log after. Env deltas BEFORE `--`,
#    per-cell Hydra AFTER. Pre-create the launcher's hardcoded done-flag dir so its
#    final touch can't false-fail a good cell. Systematic-failure tripwire halts on
#    a config-level/freeze crash (recurs in every remaining cell).
# ---------------------------------------------------------------------------
CFG_CRASH_RE='OutOfMemoryError|CUDA out of memory|torch\.OutOfMemoryError|ModuleNotFoundError|ImportError:|hydra\.errors|omegaconf\.errors|ConfigAttributeError|MissingMandatoryValue|\[TRAIN_LAYERS\][^A-Za-z]{0,6}(freeze sanity FAILED|grad-flow FAILED|immutability FAILED)|RuntimeError: \[TRAIN_LAYERS\]'

run_cell() {
  local cell="$1" launcher="$2"; shift 2
  local envs=() hydra=() rc exp
  while [[ $# -gt 0 && "$1" != "--" ]]; do envs+=("$1"); shift; done
  [[ "${1:-}" == "--" ]] && shift
  hydra=("$@")
  exp="64-$cell"
  mkdir -p "$VERL_DIR/runs/$exp"
  echo "=== [$(date -Iseconds)] CELL $cell START (wandb run $exp | launcher $(basename "$launcher")) ==="
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
    echo "$(date -Iseconds) cell=$cell rc=$rc — config-level/freeze crash signature; HALTING (remaining cells share the freeze hook + memory config)" \
      > "$RUN_DIR/halt.flag"
    echo "=== HALT: config-level/freeze crash in $cell — NOT advancing to remaining cells ===" >&2
    echo "$(date -Iseconds) halted after $cell" > "$RUN_DIR/done.flag"
    exit 1
  fi
  echo "=== $cell failed at SCIENCE level (no config-crash signature) — continuing to next cell ===" >&2
  return "$rc"
}

# ---------------------------------------------------------------------------
# CELL 1 — freeze-block-l11-15-gsm8k  (accel surface, hard-pins resp=1024;
#   S_full=0.7657 dense ref was measured on THIS surface). Dense = comm-eff OFF:
#   the accel wrapper hard-`export`s COMM_EFF_ENABLED=true, so the ENV switch is
#   overridden — the trailing Hydra override actor..comm_eff.enabled=false is the
#   real dense switch ("$@" is passed LAST by both launchers ⇒ last-wins). GSM8K
#   data auto-preps (DATA_DIR unset ⇒ launcher default /root/data/gsm8k).
# ---------------------------------------------------------------------------
run_cell freeze-block-l11-15-gsm8k "$ACCEL_LAUNCHER" \
  TRAIN_LAYERS=11-15 TOTAL_TRAINING_STEPS=75 TEST_FREQ=25 VAL_BEFORE_TRAIN=False \
  -- \
  actor_rollout_ref.actor.comm_eff.enabled=false

# ---------------------------------------------------------------------------
# 2b. Big-Math data prep (right before CELL 2; per project.yaml datasets.hard).
#     Independent of CELL 1 — a prep failure skips CELL 2 only, never retro-fails
#     CELL 1. Outputs train.parquet + test.parquet (the launcher's expected names).
# ---------------------------------------------------------------------------
BIGMATH_READY=1
if [[ ! -f "$BIGMATH_DIR/train.parquet" || ! -f "$BIGMATH_DIR/test.parquet" ]]; then
  echo "=== [$(date -Iseconds)] prepare Big-Math -> $BIGMATH_DIR ==="
  if ! python3 research/scripts/bigmath_dapo.py --local_save_dir "$BIGMATH_DIR" \
        --train-cap 0 --val-size 500 --seed 42; then
    echo "$(date -Iseconds) big-math data prep failed" > "$RUN_DIR/fail_freeze-block-l11-15-bigmath.flag"
    echo "=== Big-Math prep FAILED — skipping CELL 2 (CELL 1 result stands) ===" >&2
    BIGMATH_READY=0
  fi
fi

# ---------------------------------------------------------------------------
# CELL 2 — freeze-block-l11-15-bigmath  (resp=4096; the accel WRAPPER cannot host
#   Big-Math because it hard-`export`s resp=1024, so run the underlying baseline
#   launcher directly + the accel wrapper's pinned speed knobs at resp 4096.
#   COMM_EFF_ENABLED=false env works here (baseline reads ${VAR:-true}); the Hydra
#   override is belt-and-suspenders. PPO_MAX_TOKEN_LEN_PER_GPU=24576 >= prompt+resp
#   = 5120 (one-sequence floor). USE_DYNAMIC_BSZ defaults False in this launcher —
#   set True explicitly (plan runner-gotcha).
# ---------------------------------------------------------------------------
if [[ "$BIGMATH_READY" -eq 1 ]]; then
  run_cell freeze-block-l11-15-bigmath "$BASE_LAUNCHER" \
    COMM_EFF_ENABLED=false TRAIN_LAYERS=11-15 TOTAL_TRAINING_STEPS=75 TEST_FREQ=25 VAL_BEFORE_TRAIN=False \
    USE_DYNAMIC_BSZ=True ROLLOUT_TP=1 ROLLOUT_GPU_MEM_UTIL=0.55 PPO_MAX_TOKEN_LEN_PER_GPU=24576 \
    TRAIN_BATCH_SIZE=128 PPO_MINI_BATCH_SIZE=64 ROLLOUT_N=8 ACTOR_LR=1e-6 COMM_EFF_CAPTURE_ENABLED=false \
    MAX_RESPONSE_LENGTH=4096 MAX_PROMPT_LENGTH=1024 DATA_DIR="$BIGMATH_DIR" \
    -- \
    actor_rollout_ref.actor.comm_eff.enabled=false
fi

echo "$(date -Iseconds) all cells attempted" > "$RUN_DIR/done.flag"
echo "=== [$(date -Iseconds)] launch.sh END $RUN_ID ==="
