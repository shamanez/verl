#!/usr/bin/env bash
# Background watcher for the EXP-18 spectral-floor re-run. Emits a HEARTBEAT line
# (GPU util + step + score + anchor count) every ~9th poll (~4.5 min) and a
# TERMINAL line on OOM / done-flag / 40-min timeout, then exits. Each stdout line
# is a Monitor event. Pass "once" to do a single poll (for testing).
KEY="$HOME/.ssh/vast_ai_name"; HOST=208.64.254.75; PORT=23828
LOG=/workspace/runs/EXP-18/train_curvematch_spectral_baseline_c5_d5.log
FLAG=/workspace/runs/EXP-18/done_curvematch_spectral_baseline_c5_d5.flag
RCMD="nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader | tr '\n' '/'; \
echo -n ' | step='; grep -aoE 'global_step:[0-9]+' $LOG 2>/dev/null | tail -1 | tr -d '\n'; \
echo -n ' | '; grep -aoE 'critic/score/mean:[0-9.]+' $LOG 2>/dev/null | tail -1 | tr -d '\n'; \
echo -n ' | gnorm='; grep -aoE 'actor/grad_norm:[0-9.eE+-]+' $LOG 2>/dev/null | tail -1 | tr -d '\n'; \
echo -n ' | anchor='; grep -acE 'anchor refresh step=' $LOG 2>/dev/null | tr -d '\n'; \
echo -n ' | OOM='; grep -acE 'OutOfMemoryError|CUDA out of memory' $LOG 2>/dev/null | tr -d '\n'; \
echo -n ' | flag='; cat $FLAG 2>/dev/null | tr -d '\n'; echo"
poll() {
  ssh -i "$KEY" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 -o BatchMode=yes -o ServerAliveInterval=5 -o ServerAliveCountMax=3 -p "$PORT" root@"$HOST" "$RCMD" 2>/dev/null || echo "SSH_FAIL"
}
if [ "${1:-}" = "once" ]; then poll; exit 0; fi
for i in $(seq 1 80); do
  OUT=$(poll)
  if echo "$OUT" | grep -qE 'OOM=[1-9]'; then echo "FLOOR_TERMINAL_OOM :: $OUT"; exit 0; fi
  if echo "$OUT" | grep -qE 'flag=.*rc='; then echo "FLOOR_TERMINAL_DONE :: $OUT"; exit 0; fi
  if [ $(( (i-1) % 9 )) -eq 0 ]; then echo "FLOOR_HEARTBEAT poll=$i :: $OUT"; fi
  sleep 30
done
echo "FLOOR_WATCH_TIMEOUT_40MIN :: last=$OUT"; exit 0
