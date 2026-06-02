#!/usr/bin/env bash
# Reusable background watcher for one EXP-18 training cell on the Vast box.
# Emits a HEARTBEAT line (~every 9th poll / ~4.5 min: GPU util, step, reward,
# grad_norm, anchor/inject counts, OOM count) and a TERMINAL line on
# OOM / done-flag / traceback / 40-min timeout, then exits. Each stdout line is a Monitor event.
#
# Usage:  bash scripts/cell_watch.sh <EXPERIMENT_NAME> [HOST] [PORT] [once]
#   e.g.  bash scripts/cell_watch.sh curvematch_anchorinject_c5_d5
EXP="${1:?usage: cell_watch.sh <EXPERIMENT_NAME> [HOST] [PORT] [once]}"
HOST="${2:-208.64.254.75}"; PORT="${3:-23828}"; MODE="${4:-loop}"
KEY="$HOME/.ssh/vast_ai_name"
LOG="/workspace/runs/EXP-18/train_${EXP}.log"
FLAG="/workspace/runs/EXP-18/done_${EXP}.flag"
RCMD="nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader | tr '\n' '/'; \
echo -n ' | step='; grep -aoE 'global_step:[0-9]+' $LOG 2>/dev/null | tail -1 | tr -d '\n'; \
echo -n ' | '; grep -aoE 'critic/score/mean:[0-9.]+' $LOG 2>/dev/null | tail -1 | tr -d '\n'; \
echo -n ' | gnorm='; grep -aoE 'actor/grad_norm:[0-9.eE+-]+' $LOG 2>/dev/null | tail -1 | tr -d '\n'; \
echo -n ' | anchor='; grep -acE 'anchor refresh step=' $LOG 2>/dev/null | tr -d '\n'; \
echo -n ' | inject='; grep -acE '\[comm_eff\]\[EXP-18\]\[inject\]' $LOG 2>/dev/null | tr -d '\n'; \
echo -n ' | OOM='; grep -acE 'OutOfMemoryError|CUDA out of memory' $LOG 2>/dev/null | tr -d '\n'; \
echo -n ' | TB='; grep -acE 'Traceback \(most recent' $LOG 2>/dev/null | tr -d '\n'; \
echo -n ' | flag='; cat $FLAG 2>/dev/null | tr -d '\n'; echo"
poll() {
  ssh -i "$KEY" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 -o BatchMode=yes \
      -o ServerAliveInterval=5 -o ServerAliveCountMax=3 -p "$PORT" root@"$HOST" "$RCMD" 2>/dev/null || echo "SSH_FAIL"
}
if [ "$MODE" = "once" ]; then poll; exit 0; fi
for i in $(seq 1 80); do
  OUT=$(poll)
  if echo "$OUT" | grep -qE 'OOM=[1-9]'; then echo "${EXP}_TERMINAL_OOM :: $OUT"; exit 0; fi
  if echo "$OUT" | grep -qE 'flag=.*rc='; then echo "${EXP}_TERMINAL_DONE :: $OUT"; exit 0; fi
  if echo "$OUT" | grep -qE 'TB=[1-9]'; then echo "${EXP}_TERMINAL_TRACEBACK :: $OUT"; exit 0; fi
  if [ $(( (i-1) % 9 )) -eq 0 ]; then echo "${EXP}_HEARTBEAT poll=$i :: $OUT"; fi
  sleep 30
done
echo "${EXP}_WATCH_TIMEOUT_40MIN :: last=$OUT"; exit 0
