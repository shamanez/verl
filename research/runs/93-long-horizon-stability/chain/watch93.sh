#!/usr/bin/env bash
# Laptop-side watcher for issue #93: emits one line per state transition on the
# a5b -> a6 handoff, plus milestone and stall lines, and exits once a6 is
# confirmed training or the chain reports a failure.
SSHC=(ssh -o StrictHostKeyChecking=no -o ConnectTimeout=25 -o BatchMode=yes \
      -i "$HOME/.ssh/vast_ai" -p 8602 root@50.46.253.92)

fails=0; last_chain=""; last_a5b=""; stall=0; milestone=0; gone=0

probe() {
  "${SSHC[@]}" 'bash -s' 2>/dev/null <<'EOF'
sess=$(tmux has-session -t run-93 2>/dev/null && echo live || echo gone)
a5b=$(grep -aoE "global_step:[0-9]+" /workspace/runs/a5b-frlr-bnorm-200/train.log 2>/dev/null | tail -1)
a6=$(grep -aoE "global_step:[0-9]+" /workspace/runs/a6-prf-exactk-tis-bnorm-200/train.log 2>/dev/null | tail -1)
err=$(grep -acE "Traceback|CUDA out of memory|FATAL:" /workspace/runs/a6-prf-exactk-tis-bnorm-200/train.log 2>/dev/null || echo 0)
gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
echo "sess=$sess a5b=${a5b:-none} a6=${a6:-none} gpu=${gpu:-NA}MiB a6err=$err"
tail -1 /workspace/chain-93.log 2>/dev/null
EOF
}

while true; do
  out="$(probe)"
  if [ -z "$out" ]; then
    fails=$((fails + 1))
    if [ "$fails" -ge 3 ]; then
      echo "BOX UNREACHABLE 3 polls in a row at $(date -u +%H:%MZ); GPU state unknown"
      fails=0
    fi
    sleep 300; continue
  fi
  fails=0
  state="$(printf '%s\n' "$out" | sed -n 1p)"
  chain="$(printf '%s\n' "$out" | sed -n 2p)"

  if [ "$chain" != "$last_chain" ] && [ -n "$chain" ]; then
    echo "CHAIN: $chain"
    last_chain="$chain"
  fi

  case "$state" in *"a6err=0"*) ;; *) echo "A6 ERROR MARKERS PRESENT: $state" ;; esac

  a5b="$(printf '%s\n' "$state" | grep -oE 'a5b=global_step:[0-9]+' | grep -oE '[0-9]+$')"
  if [ -n "$a5b" ]; then
    if [ "$a5b" = "$last_a5b" ]; then
      stall=$((stall + 1))
      if [ "$stall" -eq 3 ] && printf '%s' "$state" | grep -q "sess=live"; then
        echo "STALL: a5b stuck at step $a5b for 15 min (nominal is 117 s/step) -- $state"
      fi
    else
      stall=0
      m=$((a5b / 50))
      if [ "$m" -gt "$milestone" ]; then
        milestone="$m"
        echo "PROGRESS: a5b at step $a5b/200 -- $state"
      fi
    fi
    last_a5b="$a5b"
  fi

  # Independent idle detector. Does NOT trust the chain to report its own death:
  # if nothing is running in tmux and a6 has not reached a step, the GPU is idle
  # regardless of what chain-93.log last said. The chain needs at most about 13
  # min from a5b's exit to `tmux new-session` (up to 12 min waiting for the GPU
  # to free), so 5 polls of 5 min gives it a wide margin before alarming.
  if printf '%s' "$state" | grep -q "sess=gone" \
     && ! printf '%s' "$state" | grep -qE 'a6=global_step:[0-9]+'; then
    gone=$((gone + 1))
    if [ "$gone" -eq 5 ]; then
      echo "GPU IDLE: nothing running in tmux for 25 min and a6 has no step. Chain may be dead. $state | chain=$chain"
      exit 1
    fi
  else
    gone=0
  fi

  if printf '%s' "$chain" | grep -q "LAUNCHED" \
     && printf '%s' "$state" | grep -qE 'a6=global_step:[0-9]+'; then
    echo "HANDOFF COMPLETE: a5b finished and a6 is training -- $state"
    exit 0
  fi
  if printf '%s' "$chain" | grep -qE "ALERT|LAUNCH FAILED"; then
    echo "HANDOFF FAILED, GPU MAY BE IDLE: $chain -- $state"
    exit 1
  fi
  sleep 300
done
