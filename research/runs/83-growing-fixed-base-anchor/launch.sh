#!/usr/bin/env bash
# launch.sh — issue #83: test growing_fixed_base rank1_relex anchor mode.
# code_change: true (env passthrough for lookahead_history_mode / _max_snapshots
# on exp/83-growing-fixed-base-anchor @ 40a30270).
#
# Cells (sequential, ONE box): probe (GO/NO-GO gate) -> growing arm -> sliding
# control. The probe is stop-the-world: if it fails OR does not prove the growing
# mode actually took effect, we HALT before spending the two full arms.
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
echo "=== verl HEAD: $(git rev-parse HEAD) (want 40a30270) ==="
uv pip install --no-deps -e . > /workspace/pip.log 2>&1 \
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

# --- Cell 1: mem-probe / GO-NO-GO gate (24 steps, no val) -------------------
run_cell rollout-batch-mem-probe 1 \
  COMM_EFF_ANCHOR_BATCH_SCOPE=rollout_batch \
  COMM_EFF_ANCHOR_LOOKAHEAD_HISTORY_MODE=growing_fixed_base \
  COMM_EFF_ANCHOR_LOOKAHEAD_MAX_SNAPSHOTS=-1 \
  VAL_BEFORE_TRAIN=False TEST_FREQ=1000 TOTAL_TRAINING_STEPS=24

# Silent-failure gate at the probe boundary (plan correctness invariant): the
# probe MUST prove the growing mode took effect before the matrix spends. A probe
# that reached the end but never fired history_mode=growing_fixed_base means the
# env passthrough was silently ineffective -> a null masquerading as a result.
if (( HALTED == 0 )); then
  if grep -qE 'history_mode=growing_fixed_base' "$RUN_DIR/rollout-batch-mem-probe/train.log"; then
    echo "=== silent-failure gate OK: probe fired history_mode=growing_fixed_base ==="
  else
    echo "FATAL: probe finished but NO 'history_mode=growing_fixed_base' per-fire line — env passthrough silently ineffective; refusing to spend the matrix." \
      | tee -a "$RUN_DIR/rollout-batch-mem-probe/train.log"
    printf '%s probe-no-growing-fire\n' "$(date -Iseconds)" > "$RUN_DIR/halt.flag"
    HALTED=1
  fi
fi

# --- Cell 2: growing arm (100 steps, val@0/25/50/75/100) --------------------
if (( HALTED == 0 )); then
  run_cell growing-fixed-base-rollout-batch 0 \
    COMM_EFF_ANCHOR_BATCH_SCOPE=rollout_batch \
    COMM_EFF_ANCHOR_LOOKAHEAD_HISTORY_MODE=growing_fixed_base \
    COMM_EFF_ANCHOR_LOOKAHEAD_MAX_SNAPSHOTS=-1 \
    VAL_BEFORE_TRAIN=True
fi

# --- Cell 3: sliding-window paired control (100 steps, no step-0 val) -------
if (( HALTED == 0 )); then
  run_cell sliding-window-rollout-batch 0 \
    COMM_EFF_ANCHOR_BATCH_SCOPE=rollout_batch \
    COMM_EFF_ANCHOR_LOOKAHEAD_HISTORY_MODE=sliding_window \
    VAL_BEFORE_TRAIN=False
fi

if (( HALTED == 1 )); then
  echo "=== [83] launch.sh HALTED $(date -Iseconds) — see halt.flag ==="
  exit 1
fi
echo "$(date -Iseconds) done" > "$RUN_DIR/done.flag"
echo "=== [83] launch.sh DONE $(date -Iseconds) ==="
