#!/usr/bin/env bash
# Issue #93 GPU-occupancy chain: a8 -> a9 -> a10, with a no-new-code fallback.
#
# Deliberately ONE sequential process rather than two armed watchers. Four
# separate watcher bugs in this program came from a watcher deciding "am I done"
# off the SHARED tmux session name: two watchers armed on "run-93 is gone" both
# fire when the FIRST cell ends. Here the loop knows which cell it just launched.
#
# a9 and a10 both depend on NEW runtime code (anchor-owned FRLR, 1ff5e775 +
# f0f4a167). If a9 dies very early that code is broken, and a10 would die the same
# way, ending the chain and idling the GPU overnight. So an early a9 death
# diverts to the 600-step durability run on a8's config, which needs no new code
# and is the program's third planned run anyway.
set -uo pipefail

SESSION=run-93
CHAINLOG=/workspace/chain-93c.log
EARLY_DEATH_STEP=30   # a9 dying before this means the new code is broken

stamp() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
say() { echo "[$(stamp)] $*" >> "$CHAINLOG"; }

last_step() {
  grep -aoE 'global_step:[0-9]+' "/workspace/runs/$1/train.log" 2>/dev/null \
    | tail -1 | grep -oE '[0-9]+' || true
}

snapshot_terminal() {
  # WandB drops the final step to an atexit race, so the on-box log is the only
  # record of it. Captured for every cell the chain retires.
  local cell="$1" log="/workspace/runs/$1/train.log" out="/workspace/runs/$1/final"
  mkdir -p "$out"
  tail -800 "$log" > "$out/tail800.txt" 2>/dev/null || true
  grep -aE 'global_step:(19[0-9]|200|59[0-9]|600) ' "$log" > "$out/final_steps.txt" 2>/dev/null || true
  grep -acE 'Traceback|CUDA out of memory|FATAL' "$log" > "$out/error_count.txt" 2>/dev/null || true
  grep -aoE 'val-core[^ ]*|val/[^ ]*' "$log" | tail -40 > "$out/val_lines.txt" 2>/dev/null || true
  grep -aE 'frlr-anchor-q' "$log" | tail -20 > "$out/anchor_q_refreshes.txt" 2>/dev/null || true
  say "$cell: terminal snapshot written to $out"
}

launch() {
  local next="$1" script="$2"
  mkdir -p "/workspace/runs/$next"
  : > "/workspace/runs/$next/train.log"
  # Re-point the heartbeat BEFORE launching: the reaper kills the box at
  # no-heartbeat-30min and reads /workspace/train.log, so it must follow the
  # cell that is actually running.
  ln -sfn "/workspace/runs/$next/train.log" /workspace/train.log
  say "heartbeat -> /workspace/runs/$next/train.log; launching $next via $script"
  tmux new-session -d -s "$SESSION" "bash $script 2>&1 | tee -a /workspace/runs/$next/train.log"
  for _ in $(seq 1 20); do
    tmux has-session -t "$SESSION" 2>/dev/null && { say "$next is up"; return 0; }
    sleep 15
  done
  say "FATAL: $next session never came up. GPU IS IDLE."
  return 1
}

wait_for_end() {
  say "waiting for $1 to end (tmux $SESSION)"
  while tmux has-session -t "$SESSION" 2>/dev/null; do sleep 60; done
  say "$SESSION gone; $1 terminated at step ${2:-}$(last_step "$1")"
}

# --- stage 1: a8 ends -> a9 -------------------------------------------------
wait_for_end a8-frlr-qcad20-200
snapshot_terminal a8-frlr-qcad20-200
launch a9-frlr-anchorq-200 /workspace/launch_a9.sh || exit 1
sleep 600

# --- stage 2: a9 ends -> a10, OR the fallback if a9 died early --------------
wait_for_end a9-frlr-anchorq-200
snapshot_terminal a9-frlr-anchorq-200
A9_STEP="$(last_step a9-frlr-anchorq-200)"
A9_STEP="${A9_STEP:-0}"
if (( A9_STEP < EARLY_DEATH_STEP )); then
  say "a9 died at step $A9_STEP (< $EARLY_DEATH_STEP): the anchor-owned code is"
  say "broken, so a10 would fail identically. Diverting to the 600-step"
  say "durability run on a8's config, which needs no new code."
  launch c600-frlr-qcad20-fallback /workspace/launch_c600.sh || exit 1
else
  say "a9 reached step $A9_STEP; the anchor-owned code works. Proceeding to a10."
  launch a10-frlr-anchorq-unbiased-200 /workspace/launch_a10.sh || exit 1
fi
sleep 600

# --- stage 3: whatever ran second ends -> stop, and say so loudly -----------
while tmux has-session -t "$SESSION" 2>/dev/null; do sleep 60; done
say "chain complete. NOTHING IS CHAINED BEHIND THIS CELL: the GPU is now idle"
say "and burning. The 600-step run 3 is NOT authorized; raise needs:human."
