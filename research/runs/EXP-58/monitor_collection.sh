#!/usr/bin/env bash
# EXP-58 collection monitor — runs on the LAPTOP, SSHes the box each cycle.
# Two jobs:
#   1. HEARTBEAT: refresh runs/EXP-58/metrics/incoming.log (MTIME) each cycle IFF
#      ssh ok AND (tmux alive OR done.flag) — this defeats the >30-min auto-teardown
#      while the multi-hour collection runs (and until we tear down manually).
#   2. ALERTS: emit ONE stdout line (=> a Claude notification) ONLY on a milestone
#      (new checkpoint step landed in R2), a real error, a GPU stall, or done.
# Silent cycles (no stdout) = healthy. Designed for Monitor(persistent:true).
set -uo pipefail
ROOT=/Users/shamane/Documents/verl/research
INC="$ROOT/runs/EXP-58/metrics/incoming.log"
STATE="$ROOT/runs/EXP-58/metrics/.monitor-state"
SSH="ssh -i $HOME/.ssh/vast_ai -p 40381 -o StrictHostKeyChecking=no -o ConnectTimeout=15 -o BatchMode=yes root@145.241.107.153"
set -a; . "$HOME/.config/verl-research/secrets.env"; set +a
export AWS_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID" AWS_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY" AWS_DEFAULT_REGION=auto
EP="${R2_ENDPOINT:-https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com}"
CKPT="s3://$R2_BUCKET/verl-research/EXP-58/regimeA/checkpoints/"
WTS="s3://$R2_BUCKET/verl-research/EXP-58/regimeA/weights/"

prev_ck=-1; prev_wt=-1; stall=0; sshfail=0; last_logmt=0
CYCLE="${CYCLE:-150}"
while true; do
  SNAP=$($SSH '
    LOG=/workspace/runs/EXP-58/collection/train_collection_internal.log
    STEP=$(grep -oE "training/global_step:[0-9]+" "$LOG" 2>/dev/null | tail -1 | cut -d: -f2)
    UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d " ")
    LOGMT=$(stat -c %Y "$LOG" 2>/dev/null || echo 0)
    ALIVE=$(tmux has-session -t exp58-collection 2>/dev/null && echo yes || echo no)
    DONE=$([ -f /workspace/runs/EXP-58/collection/done.flag ] && tr -d "\n" < /workspace/runs/EXP-58/collection/done.flag || echo no)
    LASTERR=$(grep -iE "CUDA out of memory|RuntimeError|AssertionError|Ray.*nhandled|non-finite|Error executing method|torch.*OutOfMemory" "$LOG" 2>/dev/null | grep -viE "atexit|wandb|DataLoader worker|FutureWarning|signal_handling|deprecat" | tail -1 | cut -c1-160)
    echo "STEP=${STEP:-0}|UTIL=${UTIL:-NA}|LOGMT=${LOGMT}|ALIVE=${ALIVE}|DONE=${DONE}|LASTERR=${LASTERR}"
  ' 2>/dev/null) || SNAP="SSHFAIL"

  if [[ "$SNAP" == "SSHFAIL" || -z "$SNAP" ]]; then
    sshfail=$((sshfail+1))
    [[ $sshfail -eq 3 ]] && echo "[$(date -u +%H:%M:%S)] ALERT ssh-fail x3 — box unreachable, check instance 43387501"
    sleep "$CYCLE"; continue
  fi
  sshfail=0
  STEP=$(sed -n 's/.*STEP=\([0-9]*\).*/\1/p' <<<"$SNAP")
  UTIL=$(sed -n 's/.*UTIL=\([^|]*\).*/\1/p' <<<"$SNAP")
  LOGMT=$(sed -n 's/.*LOGMT=\([0-9]*\).*/\1/p' <<<"$SNAP")
  ALIVE=$(sed -n 's/.*ALIVE=\([a-z]*\).*/\1/p' <<<"$SNAP")
  DONE=$(sed -n 's/.*DONE=\([^|]*\).*/\1/p' <<<"$SNAP")
  LASTERR=$(sed -n 's/.*LASTERR=//p' <<<"$SNAP")

  # R2 accrual (authoritative progress): count landed heavy shards (=steps saved) + weight snapshots.
  ck=$(aws s3 ls "$CKPT" --endpoint-url "$EP" --recursive 2>/dev/null | grep -c "optim_world_size" || true)
  wt=$(aws s3 ls "$WTS" --endpoint-url "$EP" --recursive 2>/dev/null | grep -cE "/step_[0-9]+\.pt" || true)

  # HEARTBEAT — refresh incoming.log only when box reachable AND (training alive OR done).
  if [[ "$ALIVE" == "yes" || "$DONE" != "no" ]]; then
    echo "[$(date -Iseconds)] EXP-58 collection heartbeat step=$STEP util=$UTIL% ckptsR2=$ck/50 wtR2=$wt/50 alive=$ALIVE done=$DONE" >> "$INC"
  fi
  echo "step=$STEP util=$UTIL ck=$ck wt=$wt alive=$ALIVE done=$DONE" > "$STATE"

  # ALERT: new checkpoint milestone
  if [[ "$ck" =~ ^[0-9]+$ && "$ck" -gt "$prev_ck" && "$prev_ck" -ge 0 ]]; then
    echo "[$(date -u +%H:%M:%S)] MILESTONE checkpoint(s) in R2: $ck/50  (weights $wt/50, train step $STEP)"
  fi
  [[ "$ck" =~ ^[0-9]+$ ]] && prev_ck=$ck
  [[ "$wt" =~ ^[0-9]+$ ]] && prev_wt=$wt

  # ALERT: real error
  if [[ -n "$LASTERR" ]]; then
    echo "[$(date -u +%H:%M:%S)] ALERT error in collection log: $LASTERR"
  fi

  # ALERT: GPU stall (util 0/low, log not advancing, tmux alive, not done) sustained
  if [[ "$ALIVE" == "yes" && "$DONE" == "no" && "$UTIL" =~ ^[0-9]+$ && "$UTIL" -le 3 && "$LOGMT" == "$last_logmt" ]]; then
    stall=$((stall+1))
    [[ $stall -eq 4 ]] && echo "[$(date -u +%H:%M:%S)] ALERT GPU stall: util<=3% and log frozen for ~$((4*CYCLE))s at step $STEP"
  else
    stall=0
  fi
  last_logmt=$LOGMT

  # DONE / dead
  if [[ "$DONE" != "no" ]]; then
    echo "[$(date -u +%H:%M:%S)] DONE collection finished: $DONE  (ckptsR2=$ck/50 wtR2=$wt/50 laststep=$STEP)"
    break
  fi
  if [[ "$ALIVE" == "no" ]]; then
    echo "[$(date -u +%H:%M:%S)] ALERT tmux exp58-collection GONE and no done.flag (premature death) at step $STEP ck=$ck wt=$wt"
    break
  fi
  sleep "$CYCLE"
done
echo "[$(date -u +%H:%M:%S)] monitor exiting"
