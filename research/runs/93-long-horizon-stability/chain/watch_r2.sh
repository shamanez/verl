#!/usr/bin/env bash
# Emit new lines from the R2 backfill log, once each.
#
# TERMINAL CONDITION IS THE LOG'S OWN FINAL LINE, not a tmux session name. The
# first version keyed on `tmux has-session -t r2-backfill2`; the session was later
# renamed to r2-backfill3 and the monitor declared the job finished while it was
# still running. That is the fifth instance in this program of a watcher deciding
# "am I done" from shared or mutable state instead of evidence of its own
# completion. The script's closing line is evidence; a session name is not.
set -uo pipefail
SSH=(ssh -i "$HOME/.ssh/vast_ai" -p 8602 -o ConnectTimeout=20 -o StrictHostKeyChecking=no root@50.46.253.92)
ST=/private/tmp/claude-501/-Users-shamane-Documents-new-harness-verl-research/9a1943ac-7738-4cea-8cb8-1e5038d59b86/scratchpad/monstate
mkdir -p "$ST"; touch "$ST/r2b2.seen"
while true; do
  out=$("${SSH[@]}" 'sed -e "s/^/R2|/" /workspace/r2-backfill2.log 2>/dev/null
    # Any r2-backfill* session, so a rename cannot look like completion.
    echo "SESS|$(tmux ls 2>/dev/null | cut -d: -f1 | grep -c "^r2-backfill" || true)"' 2>/dev/null) || { sleep 180; continue; }
  printf '%s\n' "$out" | sed -n 's/^R2|//p' | while IFS= read -r ln; do
    grep -qxF "$ln" "$ST/r2b2.seen" 2>/dev/null || { echo "[R2] $ln"; echo "$ln" >> "$ST/r2b2.seen"; }
  done
  # Terminal: the script's own closing line.
  if printf '%s\n' "$out" | sed -n 's/^R2|//p' | grep -q "backfill2 done"; then
    echo "[R2] TERMINAL: backfill wrote its done line"; exit 0
  fi
  # Not terminal, but worth surfacing: no worker session and no done line.
  nsess=$(printf '%s\n' "$out" | sed -n 's/^SESS|//p' | tail -1)
  if [[ "${nsess:-0}" == "0" ]]; then
    echo "[R2] WARNING: no r2-backfill* session and no done line; the backfill may have died"
  fi
  sleep 180
done
