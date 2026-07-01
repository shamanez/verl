#!/usr/bin/env bash
# Hold-open keepalive for EXP-58 box 43387501 (operator directive: another agent
# is taking over; do NOT let the no-heartbeat-30min teardown reaper kill it).
# Touches the heartbeat file every 10 min for up to 24h, then exits (the 96 GPU-hr
# budget backstop remains the ultimate cap). Safe to kill once the other agent's
# run keeps incoming.log fresh on its own.
HB="/Users/shamane/Documents/verl/research/runs/EXP-58/metrics/incoming.log"
LOG="/Users/shamane/Documents/verl/research/runs/EXP-58/heartbeat_keepalive.log"
for i in $(seq 1 144); do
  [ -f "$HB" ] && touch "$HB"
  echo "[$(date -Iseconds)] keepalive touch #$i -> $HB" >> "$LOG"
  sleep 600
done
echo "[$(date -Iseconds)] keepalive EXITED after 24h window" >> "$LOG"
