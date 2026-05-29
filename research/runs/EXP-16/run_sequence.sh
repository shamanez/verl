#!/usr/bin/env bash
# EXP-16 warm-box sequencer. Runs cell 0 (GPU pre-flight GATE) then cells 1..6
# back-to-back on the SAME warm box (shared docker / verl checkout / dataset
# cache / model weights). STRICTLY sequential (max_parallel=1) — one cell at a
# time, by mandate. This is the single tmux session the orchestrator launches;
# the training-log-monitor watches the active cell's metrics/<name>/train.log.
#
# Cell 0 is an on_fail:stop GATE: if it fails, NO training cell runs and the box
# is left for teardown (the PROVISIONED/RUNNING ledger row drives the Stop hook).
# Each training cell is launched via launch.sh <n>. The launcher touches a
# per-cell done.flag on clean exit; on a corrupting failure the EARLY_STOP_SIGNAL
# watcher (baked into the launcher) drops runs/EXP-16/metrics/<name>/EARLY_STOP_SIGNAL.
#
#   usage:  bash /workspace/runs/EXP-16/run_sequence.sh [start_cell]
#           start_cell defaults to 0 (the gate). Pass 1 to skip the gate (only if
#           cell0.PASS already exists from a prior run on this warm box).
set -uo pipefail

RUN_ROOT=/workspace/runs/EXP-16
START="${1:-0}"
SEQ_LOG="$RUN_ROOT/run_sequence.log"
mkdir -p "$RUN_ROOT"

log(){ echo "[$(date -u +%FT%TZ)] [run_sequence] $*" | tee -a "$SEQ_LOG"; }

# --- Cell 0 gate (unless skipped + already passed) ---
if (( START <= 0 )); then
  log "starting cell 0 (GPU pre-flight mask-consistency GATE; on_fail:stop)"
  if bash "$RUN_ROOT/cell0_preflight.sh"; then
    log "cell 0 PASS (sentinel: $RUN_ROOT/cell0.PASS)"
  else
    log "cell 0 FAIL — STOP. NO training cell will run. Box left for teardown."
    exit 1
  fi
  START=1
fi

if [[ ! -f "$RUN_ROOT/cell0.PASS" ]]; then
  log "refusing to run training cells: cell0.PASS sentinel absent (gate not passed)."
  exit 1
fi

# --- Cells 1..6, strictly sequential ---
declare -A CELL_NAME=(
  [1]=grpo_mask_channel_p0p9_no_rescale_10steps
  [2]=grpo_mask_channel_p0p9_rescale_10steps
  [3]=grpo_mask_channel_p0p9_no_rescale_clean_every4_20steps
  [4]=grpo_mask_channel_p0p9_rescale_clean_every4_20steps
  [5]=grpo_mask_channel_p0p9_rescale_anchor2_spectral2_20steps
  [6]=dense_grpo_comm_eff_off_25step_reference
)

for n in 1 2 3 4 5 6; do
  (( n < START )) && { log "skipping cell $n (start_cell=$START)"; continue; }
  name="${CELL_NAME[$n]}"
  cell_dir="$RUN_ROOT/metrics/$name"
  if [[ -f "$cell_dir/done.flag" ]]; then
    log "cell $n ($name) already has done.flag — skip (warm-box resume)"
    continue
  fi
  log "=== launching cell $n -> $name ==="
  # launch.sh runs the cell in the FOREGROUND of this sequencer; the monitor
  # tails metrics/<name>/train.log live. set +e so a cell failure is recorded,
  # the sequence STOPS (sequential mandate: inspect before next), and the box is
  # left up for the monitor/analyst + Stop-hook teardown.
  if bash "$RUN_ROOT/launch.sh" "$n"; then
    if [[ -f "$cell_dir/done.flag" ]]; then
      log "cell $n ($name) completed (done.flag present)"
    else
      log "cell $n ($name) exited 0 but NO done.flag — treating as incomplete; STOP."
      exit 2
    fi
  else
    rc=$?
    log "cell $n ($name) FAILED rc=$rc."
    [[ -f "$cell_dir/EARLY_STOP_SIGNAL" ]] && log "EARLY_STOP_SIGNAL present: $(cat "$cell_dir/EARLY_STOP_SIGNAL")"
    # Per plan on_fail semantics: cells 1-5 record the failure signature and the
    # human/monitor decides; cell 6 failing is a whole-experiment STOP. Either
    # way, STOP the automated sequence here (sequential mandate: do not blindly
    # roll into the next cell on an unclassified failure). The orchestrator can
    # resume the next cell explicitly after inspection:
    #   tmux new -d -s exp-16-cell$((n+1))-<host> 'bash /workspace/runs/EXP-16/run_sequence.sh '"$((n+1))"
    log "STOP sequence at cell $n for inspection. Resume next cell with: run_sequence.sh $((n+1))"
    exit "$rc"
  fi
done

log "=== ALL CELLS 1..6 COMPLETE ==="
: > "$RUN_ROOT/SEQUENCE_DONE"
