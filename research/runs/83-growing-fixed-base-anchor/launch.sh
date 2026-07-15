#!/usr/bin/env bash
# launch.sh v2 — issue #83: best weight-projection method by end-to-end accuracy
# (fast ablation; operator redirection 2026-07-15 supersedes the v1 100-step arms).
# code_change: true (exp/83-growing-fixed-base-anchor @ 91ecbdd4).
#
# Cells (sequential, ONE box, unattended): mem-probe [v1 gate, done-flag skip] ->
# w2-fire-smoke (gate) -> w4-fire-smoke (gate) -> proj-current-60 -> proj-w2-60 ->
# proj-w4-60. Gates are stop-the-world; accuracy arms continue on science-level
# failure (the ranking survives a dead arm) and halt on config-level signatures.
#
# Logs: each cell writes $RUN_DIR/<cell>/train.log (analyst greps
# runs/<id>/*/train.log). A background `tail -F` mirrors the ACTIVE cell's log
# into $RUN_DIR/train.log, which is the run.json remote_log the sync-metrics hook
# + teardown reaper probe (tail -n3 | cksum) use as the liveness heartbeat — so
# the heartbeat advances with real training progress and correctly goes stale if
# training freezes (the mirror only forwards content that was actually written).
#
# NOT set -e globally: cell exit codes are managed explicitly for the failure
# policy (gate-halt / config-level-halt / science-fail-continue).
set -uo pipefail

ID="83-growing-fixed-base-anchor"
RUN_DIR="/workspace/runs/$ID"
RUN_LOG="$RUN_DIR/train.log"          # heartbeat aggregate + tmux redirect target
mkdir -p "$RUN_DIR"
echo "=== [83] launch.sh start $(date -Iseconds) ==="

# ---------------------------------------------------------------------------
# 1. Bootstrap exp/83 (GitHub-first, bundle-fallback), editable install, then
#    PROVE this change's env passthrough is present (money gate: never spend on
#    a stale checkout that would run the growing arm silently as sliding_window).
# ---------------------------------------------------------------------------
cd /workspace
[[ -e verl ]] && mv verl "verl.upstream.$(date +%s)"
if git clone -b "exp/$ID" https://github.com/shamanez/verl.git verl; then
  echo "=== code_change: cloned exp/$ID from GitHub ==="
elif [[ -f "$RUN_DIR/exp.bundle" ]]; then
  echo "=== GitHub unreachable — falling back to exp.bundle ==="
  git clone -b "exp/$ID" "$RUN_DIR/exp.bundle" verl
else
  echo "FATAL: cannot obtain exp/$ID (GitHub unreachable AND no bundle on box)." >&2
  echo "  recovery: rsync ONLY runs/$ID/exp.bundle to the box, then relaunch." >&2
  exit 1
fi
cd /workspace/verl
git remote set-url origin https://github.com/shamanez/verl.git 2>/dev/null || true
echo "=== verl HEAD: $(git rev-parse HEAD) (want 91ecbdd4) ==="
# uv is absent on some operator-attach boxes (and refuses system-python without a
# venv); prefer uv when usable, else fall back to system pip. All heavy deps are
# already installed, so this --no-deps editable install only re-points the `verl`
# package at the fresh exp/83 checkout -- pip and uv are equivalent for that.
if command -v uv >/dev/null 2>&1; then UVPIP="uv pip"; else UVPIP="python3 -m pip"; fi
echo "=== editable install via: $UVPIP ==="
$UVPIP install --no-deps -e . > /workspace/pip.log 2>&1 \
  || { echo "FATAL: pip install failed (see /workspace/pip.log)" >&2; tail -20 /workspace/pip.log >&2; exit 1; }
python3 -c "import verl" || { echo "FATAL: verl import failed after bootstrap" >&2; exit 1; }

ENGINE="examples/grpo_trainer/vast_comm_eff_engine_grpo.sh"
LAUNCHER="examples/grpo_trainer/run_qwen25_math_1p5b_rank1_relex_fsdp.sh"
if grep -qF 'lookahead_history_mode="$COMM_EFF_ANCHOR_LOOKAHEAD_HISTORY_MODE"' "$ENGINE" \
   && grep -qF 'lookahead_max_snapshots="$COMM_EFF_ANCHOR_LOOKAHEAD_MAX_SNAPSHOTS"' "$ENGINE" \
   && grep -qF 'COMM_EFF_ANCHOR_LOOKAHEAD_HISTORY_MODE:-sliding_window' "$LAUNCHER"; then
  echo "=== money gate OK: env passthrough present in both launchers ==="
else
  echo "FATAL: exp/$ID env passthrough NOT in launchers — stale checkout, refusing to spend." >&2
  exit 1
fi
python3 -c "from verl.workers.config.comm_eff import CommEffAnchorConfig as A; a=A(); assert hasattr(a,'lookahead_history_mode') and hasattr(a,'lookahead_max_snapshots'), 'schema missing new anchor fields'" \
  || { echo "FATAL: comm_eff schema missing lookahead_history_mode/max_snapshots" >&2; exit 1; }

# ---------------------------------------------------------------------------
# 2. MATH data — canonical honest data_source (DigitalLearningGmbH/MATH-lighteval
#    -> math_reward). The launcher hard-FATALs if the parquet is absent; the
#    canonical prep is REQUIRED (the engine's upstream fallback uses a different
#    data_source/routing than the plan's verified val route).
# ---------------------------------------------------------------------------
export DATA_DIR="${DATA_DIR:-$HOME/data/math}"
if [[ ! -f "$DATA_DIR/train.parquet" || ! -f "$DATA_DIR/test.parquet" ]]; then
  echo "=== preparing MATH -> $DATA_DIR (canonical prepare_rlvr_math.py) ==="
  python3 research/scripts/prepare_rlvr_math.py --dataset math --local_save_dir "$DATA_DIR" \
    || { echo "FATAL: MATH data prep failed" >&2; exit 1; }
fi

# ---------------------------------------------------------------------------
# 3. Cell runner + failure policy.
# ---------------------------------------------------------------------------
# Config-level (systematic) signatures: recur in EVERY remaining cell (all cells
# share the memory/config surface), so a hit HALTS the sweep instead of paying
# boot+crash per arm. Anything else on a full arm is treated as science-level
# (NaN/divergence in THIS arm) and the sweep continues so the other arm runs.
CONFIG_LEVEL_RE='OutOfMemoryError|CUDA out of memory|ModuleNotFoundError|No module named|No such file or directory|hydra\.errors|is not in struct|ConfigAttributeError|ConfigKeyError|Error executing job|cgroup pids.max|requires .*GPUs; detected'
HALTED=0

run_cell() {   # run_cell <cell> <gate:0|1> <ENVVAR=val ...>
  local cell="$1" gate="$2"; shift 2
  local cdir="$RUN_DIR/$cell" clog
  # Skip-if-done: idempotent relaunch — a cell that already completed in a prior
  # launch of this run (done flag present) is carried forward, not re-run. This is
  # how the passed v1 mem-probe carries into the v2 chain.
  if [[ -f "$RUN_DIR/done_$cell.flag" ]]; then
    echo "=== CELL $cell SKIP (done flag from a previous launch: $(cat "$RUN_DIR/done_$cell.flag")) ==="
    return 0
  fi
  mkdir -p "$cdir"; clog="$cdir/train.log"; : > "$clog"
  echo ""
  echo "=== CELL $cell START $(date -Iseconds) (gate=$gate) ==="
  echo "=== resolved cell env: EXPERIMENT_NAME=83-$cell WANDB_RUN_GROUP=$ID LOG=$clog $* ==="
  # heartbeat mirror: forward this cell's log to our stdout (-> $RUN_LOG).
  tail -n +1 -F "$clog" 2>/dev/null &
  local tpid=$!
  local rc=0
  env EXPERIMENT_NAME="83-$cell" WANDB_RUN_GROUP="$ID" LOG="$clog" "$@" \
    bash "$LAUNCHER" || rc=$?
  kill "$tpid" 2>/dev/null || true; wait "$tpid" 2>/dev/null || true
  tail -n 80 "$clog" 2>/dev/null || true     # flush the cell's tail into $RUN_LOG
  echo "=== CELL $cell END rc=$rc $(date -Iseconds) ==="
  if (( rc == 0 )); then
    echo "$(date -Iseconds)" > "$RUN_DIR/done_$cell.flag"
    return 0
  fi
  if (( gate == 1 )); then
    echo "HALT: gate cell '$cell' failed (rc=$rc) — feasibility NOT validated; not spending the matrix." | tee -a "$clog"
    printf '%s gate-cell %s rc=%s\n' "$(date -Iseconds)" "$cell" "$rc" > "$RUN_DIR/halt.flag"
    HALTED=1; return 1
  fi
  if grep -qiE "$CONFIG_LEVEL_RE" "$clog"; then
    echo "HALT: cell '$cell' config-level failure (rc=$rc) — would recur in every remaining cell." | tee -a "$clog"
    printf '%s config-level %s rc=%s\n' "$(date -Iseconds)" "$cell" "$rc" > "$RUN_DIR/halt.flag"
    HALTED=1; return 1
  fi
  echo "SCIENCE-FAIL: cell '$cell' (rc=$rc) — likely NaN/divergence in THIS arm; other arm still runs." | tee -a "$clog"
  printf '%s rc=%s\n' "$(date -Iseconds)" "$rc" > "$RUN_DIR/fail_$cell.flag"
  return 0
}

# smoke_signature_gate <cell> <grep-ERE> <label>: after a gate smoke completes,
# prove its config actually took effect (the silent-failure class this issue hit
# twice: struct-mode reject + PR #27's YAML gap). Missing signature -> HALT.
smoke_signature_gate() {
  local cell="$1" re="$2" label="$3"
  (( HALTED == 1 )) && return 0
  if grep -qE "$re" "$RUN_DIR/$cell/train.log"; then
    echo "=== signature gate OK: $cell shows $label ==="
  else
    echo "FATAL: $cell finished but signature '$label' ($re) NOT in its log — config silently ineffective; halting before the arms spend." \
      | tee -a "$RUN_DIR/$cell/train.log"
    printf '%s signature-gate %s\n' "$(date -Iseconds)" "$cell" > "$RUN_DIR/halt.flag"
    HALTED=1
  fi
}

# --- Cell 1: v1 mem-probe / GO-NO-GO gate (24 steps, growing config) ---------
# Done-flag skip carries the v1 pass forward; if absent (fresh box) it re-runs.
run_cell rollout-batch-mem-probe 1 \
  COMM_EFF_ANCHOR_BATCH_SCOPE=rollout_batch \
  COMM_EFF_ANCHOR_LOOKAHEAD_HISTORY_MODE=growing_fixed_base \
  COMM_EFF_ANCHOR_LOOKAHEAD_MAX_SNAPSHOTS=-1 \
  VAL_BEFORE_TRAIN=False TEST_FREQ=1000 TOTAL_TRAINING_STEPS=24
smoke_signature_gate rollout-batch-mem-probe 'history_mode=growing_fixed_base' 'growing_fixed_base fire (v1 invariant + growing-mechanism deliverable)'

# --- Cell 2: W2 fire-smoke (gate, 12 steps, one anchor fire at step 10) ------
if (( HALTED == 0 )); then
  run_cell w2-fire-smoke 1 \
    COMM_EFF_ANCHOR_BATCH_SCOPE=rollout_batch \
    COMM_EFF_ANCHOR_LOOKAHEAD_WINDOW_SNAPSHOTS=2 \
    VAL_BEFORE_TRAIN=False TEST_FREQ=1000 TOTAL_TRAINING_STEPS=12
  smoke_signature_gate w2-fire-smoke 'checkpoints=[0-9]+/2|window=2 ' 'window=2 config (fire denominator /2 or engine echo)'
fi

# --- Cell 3: W4 fire-smoke (gate, 12 steps, one warmup fire at step 10) ------
if (( HALTED == 0 )); then
  run_cell w4-fire-smoke 1 \
    COMM_EFF_ANCHOR_BATCH_SCOPE=rollout_batch \
    COMM_EFF_ANCHOR_LOOKAHEAD_MIN_SNAPSHOTS=4 \
    COMM_EFF_ANCHOR_LOOKAHEAD_WINDOW_SNAPSHOTS=4 \
    VAL_BEFORE_TRAIN=False TEST_FREQ=1000 TOTAL_TRAINING_STEPS=12
  smoke_signature_gate w4-fire-smoke 'min_snapshots=4' 'min_snapshots=4 config (engine echo)'
fi

# --- Cell 4: accuracy arm, CURRENT progressive baseline (60 steps) -----------
# VAL_BEFORE_TRAIN=True gives the single arm-invariant step-0 val + val route smoke.
if (( HALTED == 0 )); then
  run_cell proj-current-60 0 \
    COMM_EFF_ANCHOR_BATCH_SCOPE=rollout_batch \
    VAL_BEFORE_TRAIN=True TEST_FREQ=20 TOTAL_TRAINING_STEPS=60
fi

# --- Cell 5: accuracy arm, fixed W2 secant (60 steps) ------------------------
if (( HALTED == 0 )); then
  run_cell proj-w2-60 0 \
    COMM_EFF_ANCHOR_BATCH_SCOPE=rollout_batch \
    COMM_EFF_ANCHOR_LOOKAHEAD_WINDOW_SNAPSHOTS=2 \
    VAL_BEFORE_TRAIN=False TEST_FREQ=20 TOTAL_TRAINING_STEPS=60
fi

# --- Cell 6: accuracy arm, fixed W4 full fit (60 steps) ----------------------
if (( HALTED == 0 )); then
  run_cell proj-w4-60 0 \
    COMM_EFF_ANCHOR_BATCH_SCOPE=rollout_batch \
    COMM_EFF_ANCHOR_LOOKAHEAD_MIN_SNAPSHOTS=4 \
    COMM_EFF_ANCHOR_LOOKAHEAD_WINDOW_SNAPSHOTS=4 \
    VAL_BEFORE_TRAIN=False TEST_FREQ=20 TOTAL_TRAINING_STEPS=60
fi

if (( HALTED == 1 )); then
  echo "=== [83] launch.sh HALTED $(date -Iseconds) — see halt.flag ==="
  exit 1
fi
echo "$(date -Iseconds) done" > "$RUN_DIR/done.flag"
echo "=== [83] launch.sh DONE $(date -Iseconds) ==="
