#!/usr/bin/env bash
# Poll the #93 box and emit ONLY new, actionable events. State kept locally so a
# restart does not replay history. Prefix-tagged remote output; never
# line-number-parsed (a `grep -c || echo 0` two-line emission broke a watcher
# into a false TERMINAL earlier in this program).
set -uo pipefail
SSH=(ssh -i "$HOME/.ssh/vast_ai" -p 8602 -o ConnectTimeout=20 -o StrictHostKeyChecking=no root@50.46.253.92)
ST=/private/tmp/claude-501/-Users-shamane-Documents-new-harness-verl-research/9a1943ac-7738-4cea-8cb8-1e5038d59b86/scratchpad/monstate
mkdir -p "$ST"
touch "$ST/chain.seen" "$ST/backfill.seen" "$ST/anchorq.seen" "$ST/fail.seen" "$ST/steps.seen"

while true; do
  raw=$("${SSH[@]}" '
    L=$(readlink -f /workspace/train.log)
    echo "CELL|$(basename $(dirname $L))"
    echo "STEP|$(grep -aoE "global_step:[0-9]+" "$L" 2>/dev/null | tail -1 | grep -oE "[0-9]+")"
    echo "SESS|$(tmux ls 2>/dev/null | cut -d: -f1 | tr "\n" ",")"
    sed -e "s/^/CHAIN|/" /workspace/chain-93c.log 2>/dev/null
    sed -e "s/^/BACK|/" /workspace/r2-backfill.log 2>/dev/null
    grep -aE "frlr-anchor-q" "$L" 2>/dev/null | head -3 | sed -e "s/^/ANCHORQ|/"
    grep -aE "Traceback|CUDA out of memory|FATAL|AssertionError|Killed|torch.OutOfMemory" "$L" 2>/dev/null | head -6 | sed -e "s/^/FAIL|/"
  ' 2>/dev/null) || { sleep 120; continue; }

  cell=$(printf '%s\n' "$raw" | sed -n 's/^CELL|//p' | tail -1)
  step=$(printf '%s\n' "$raw" | sed -n 's/^STEP|//p' | tail -1)
  sess=$(printf '%s\n' "$raw" | sed -n 's/^SESS|//p' | tail -1)

  # chain + backfill: emit each line once
  for tag in CHAIN BACK; do
    f=$(printf '%s' "$tag" | tr 'A-Z' 'a-z'); f="$ST/${f/chain/chain}.seen"
    [[ "$tag" == BACK ]] && f="$ST/backfill.seen"
    [[ "$tag" == CHAIN ]] && f="$ST/chain.seen"
    printf '%s\n' "$raw" | sed -n "s/^$tag|//p" | while IFS= read -r ln; do
      grep -qxF "$ln" "$f" 2>/dev/null || { echo "[$tag] $ln"; echo "$ln" >> "$f"; }
    done
  done

  # anchor-Q refresh: the a9 validation. Emit the first few, then stay quiet.
  printf '%s\n' "$raw" | sed -n 's/^ANCHORQ|//p' | while IFS= read -r ln; do
    grep -qxF "$ln" "$ST/anchorq.seen" 2>/dev/null || { echo "[ANCHOR-Q OK] $ln"; echo "$ln" >> "$ST/anchorq.seen"; }
  done

  # failures: anything I would act on
  printf '%s\n' "$raw" | sed -n 's/^FAIL|//p' | while IFS= read -r ln; do
    grep -qxF "$ln" "$ST/fail.seen" 2>/dev/null || { echo "[FAIL $cell] $ln"; echo "$ln" >> "$ST/fail.seen"; }
  done

  # step milestones: the early-kill windows and the finish
  if [[ -n "$step" ]]; then
    for m in 20 60 80 100 120 150 200 300 600; do
      if (( step >= m )); then
        key="$cell:$m"
        grep -qxF "$key" "$ST/steps.seen" 2>/dev/null || { echo "[STEP $cell] reached $m (now $step); sessions=$sess"; echo "$key" >> "$ST/steps.seen"; }
      fi
    done
  fi

  # GPU idle is the top interrupt
  if [[ "$sess" != *run-93* ]]; then
    key="idle:$cell"
    grep -qxF "$key" "$ST/steps.seen" 2>/dev/null || { echo "[IDLE] no run-93 session; last cell $cell at step ${step:-?}; sessions=$sess"; echo "$key" >> "$ST/steps.seen"; }
  fi
  sleep 120
done
