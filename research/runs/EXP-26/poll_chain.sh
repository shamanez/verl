#!/bin/bash
# EXP-26 C2+B chain — lightweight laptop-side poller (orchestrator helper).
# Sits between full training-log-monitor dispatches: every POLL_S seconds it
# SSH-probes the box, keeps the teardown-hook heartbeat fresh (same rule as
# sync-metrics.sh: append ONLY on SSH success), and EXITS with a status line on
# any event the orchestrator must act on. Read-only on the box; never tears down.
set -uo pipefail

HOST=145.241.108.98; PORT=40280; KEY=~/.ssh/vast_ai_name
RUNDIR=/Users/shamane/Documents/verl/research/runs/EXP-26
CELLS=(exp26_C2_hybrid exp26_B_plain exp26_B_dense exp26_B_ef)
POLL_S=180; MAX_POLLS=80          # ~4 h window
STALL_POLLS=8                      # 24 min unchanged log = stall candidate
mkdir -p "$RUNDIR/metrics"

prev_flags=-1; prev_size=0; stall_n=0; ssh_fail=0
for ((i=1; i<=MAX_POLLS; i++)); do
  # one SSH round-trip gathers everything
  OUT=$(ssh -i "$KEY" -o ConnectTimeout=15 -o BatchMode=yes \
        -o StrictHostKeyChecking=accept-new -p "$PORT" "root@$HOST" '
    flags=""; for c in exp26_C2_hybrid exp26_B_plain exp26_B_dense exp26_B_ef; do
      [[ -f /workspace/runs/EXP-26/$c.done.flag ]] && flags="$flags $c"; done
    echo "FLAGS:$flags"
    [[ -f /workspace/runs/EXP-26/bef_chain.done.flag ]] && echo "CHAIN_DONE:yes"
    tmux has-session -t exp-26-145_241_108_98 2>/dev/null && echo "TMUX:alive" || echo "TMUX:dead"
    for c in exp26_C2_hybrid exp26_B_plain exp26_B_dense exp26_B_ef; do
      L=/workspace/verl/runs/$c/train.log
      if [[ -f $L && ! -f /workspace/runs/EXP-26/$c.done.flag ]]; then
        echo "ACTIVE:$c"
        echo "SIZE:$(stat -c %s "$L")"
        echo "ERRS:$(grep -c -E "Traceback|CUDA out of memory|NaN detected|Ray.*unhandled" "$L" 2>/dev/null)"
        echo "LAST:$(grep -E "step:[0-9]+|Training Progress" "$L" | tail -1 | cut -c1-200)"
        echo "TAILBLOCK_BEGIN"; tail -n 40 "$L"; echo "TAILBLOCK_END"
        break
      fi
    done
    echo "UTIL:$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr "\n" "," )"
  ' 2>/dev/null)
  rc=$?

  if [[ $rc -ne 0 || -z "$OUT" ]]; then
    ssh_fail=$((ssh_fail+1))
    echo "[$(date -Iseconds)] poll=$i SSH_FAIL n=$ssh_fail" >> "$RUNDIR/monitor-detail.log"
    if (( ssh_fail >= 8 )); then echo "STATUS:SSH_UNREACHABLE polls=$i"; exit 0; fi
    sleep "$POLL_S"; continue
  fi
  ssh_fail=0

  flags_line=$(grep '^FLAGS:' <<<"$OUT" | head -1)
  nflags=$(wc -w <<<"${flags_line#FLAGS:}" | tr -d ' ')
  tmux_state=$(grep '^TMUX:' <<<"$OUT" | head -1 | cut -d: -f2)
  active=$(grep '^ACTIVE:' <<<"$OUT" | head -1 | cut -d: -f2)
  size=$(grep '^SIZE:' <<<"$OUT" | head -1 | cut -d: -f2)
  errs=$(grep '^ERRS:' <<<"$OUT" | head -1 | cut -d: -f2)
  last=$(grep '^LAST:' <<<"$OUT" | head -1 | cut -c6-)
  util=$(grep '^UTIL:' <<<"$OUT" | head -1 | cut -d: -f2)

  # heartbeat: only on SSH success with content (sync-metrics rule)
  { echo "--- $(date -Iseconds) host=$HOST port=$PORT (poll_chain) active=$active ---"
    sed -n '/TAILBLOCK_BEGIN/,/TAILBLOCK_END/p' <<<"$OUT" | head -45
  } >> "$RUNDIR/metrics/incoming.log"

  echo "[$(date -Iseconds)] poll=$i flags=$nflags tmux=$tmux_state active=$active errs=${errs:-?} util=$util last=${last:0:140}" >> "$RUNDIR/monitor-detail.log"

  if grep -q '^CHAIN_DONE:yes' <<<"$OUT"; then echo "STATUS:CHAIN_DONE flags=$nflags"; exit 0; fi
  if [[ $prev_flags -ge 0 && $nflags -gt $prev_flags ]]; then
    echo "STATUS:CELL_DONE flags_now=$nflags done_cells=${flags_line#FLAGS:}"; exit 0; fi
  prev_flags=$nflags
  if [[ -n "${errs:-}" && "${errs:-0}" -gt 0 ]]; then
    echo "STATUS:ERROR cell=$active errs=$errs last=$last"; exit 0; fi
  if [[ "$tmux_state" == "dead" ]]; then echo "STATUS:TMUX_DEAD flags=$nflags"; exit 0; fi
  if [[ -n "$active" && "$size" == "$prev_size" ]]; then
    stall_n=$((stall_n+1))
    if (( stall_n >= STALL_POLLS )); then echo "STATUS:STALL cell=$active size=$size util=$util"; exit 0; fi
  else stall_n=0; fi
  prev_size=$size

  sleep "$POLL_S"
done
echo "STATUS:TIMEOUT_4H flags=$prev_flags"
