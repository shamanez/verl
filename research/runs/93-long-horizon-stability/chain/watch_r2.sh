#!/usr/bin/env bash
# Emit new lines from the R2 backfill2 log, once each. Selective: the script only
# logs per-cell milestones, files over 1GB, and failures, so every line is
# actionable. Exits when the backfill's tmux session is gone AND its final line
# has been emitted.
set -uo pipefail
SSH=(ssh -i "$HOME/.ssh/vast_ai" -p 8602 -o ConnectTimeout=20 -o StrictHostKeyChecking=no root@50.46.253.92)
ST=/private/tmp/claude-501/-Users-shamane-Documents-new-harness-verl-research/9a1943ac-7738-4cea-8cb8-1e5038d59b86/scratchpad/monstate
mkdir -p "$ST"; touch "$ST/r2b2.seen"
while true; do
  out=$("${SSH[@]}" 'sed -e "s/^/R2|/" /workspace/r2-backfill2.log 2>/dev/null; echo "ALIVE|$(tmux has-session -t r2-backfill2 2>/dev/null && echo yes || echo no)"' 2>/dev/null) || { sleep 180; continue; }
  printf '%s\n' "$out" | sed -n 's/^R2|//p' | while IFS= read -r ln; do
    grep -qxF "$ln" "$ST/r2b2.seen" 2>/dev/null || { echo "[R2] $ln"; echo "$ln" >> "$ST/r2b2.seen"; }
  done
  alive=$(printf '%s\n' "$out" | sed -n 's/^ALIVE|//p' | tail -1)
  if [[ "$alive" == "no" ]]; then echo "[R2] backfill2 session gone"; exit 0; fi
  sleep 180
done
