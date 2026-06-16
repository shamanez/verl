#!/usr/bin/env bash
# EXP-33 Window 3 monitoring loop
# Polls every 30s for up to 40 min (80 polls)
# Writes per-poll to monitor-detail.log
# Writes final state to monitor-loop-result.json

set -euo pipefail

SSH="ssh -i ~/.ssh/vast_ai -p 40154 -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=20"
HOST="root@84.8.116.228"
RUNDIR="/Users/shamane/Documents/verl/research/runs/EXP-33"
LOG="$RUNDIR/monitor-detail.log"
RESULT_FILE="$RUNDIR/monitor-loop-result.json"
WANDB_POLLED_AT=0  # epoch seconds of last wandb poll
POLL_NUM=0
MAX_POLLS=80
START_TIME=$(date +%s)
MAX_WALL=2400  # 40 min

# GPU stall tracking
GPU_STALL_COUNT=0
GPU_STALL_THRESHOLD=4

# Cell rsync tracking
RSYNCED_CELLS=""

log() {
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$LOG"
}

# Source secrets for WandB
if [ -f ~/.config/verl-research/secrets.env ]; then
    set +u
    source ~/.config/verl-research/secrets.env
    set -u
fi

do_rsync() {
    local cell="$1"
    log "RSYNC: pulling $cell logs"
    mkdir -p "$RUNDIR/$cell"
    rsync -avz --timeout=30 -e "ssh -i ~/.ssh/vast_ai -o StrictHostKeyChecking=no -p 40154" \
        "$HOST:/workspace/verl/runs/$cell/train.log" \
        "$RUNDIR/$cell/train.log" 2>&1 | tail -3 | tee -a "$LOG" || true
    # Also pull done flag if it exists
    rsync -avz --timeout=30 -e "ssh -i ~/.ssh/vast_ai -o StrictHostKeyChecking=no -p 40154" \
        "$HOST:/workspace/runs/EXP-33/done_${cell}.flag" \
        "$RUNDIR/done_${cell}.flag" 2>/dev/null || true
}

poll_wandb() {
    local cell="$1"
    local now=$(date +%s)
    # Only poll every ~90s
    if (( now - WANDB_POLLED_AT < 90 )); then return; fi
    WANDB_POLLED_AT=$now

    if [ -z "${WANDB_API_KEY:-}" ]; then
        log "WANDB: API key not set, skipping"
        return
    fi

    # Query WandB for latest run state and scalars
    local query='{"query":"{ project(entityName:\"shamanework-pl\", name:\"verl_compression_research_beta_sweep\") { runs(filters:{\"display_name\":{\"$in\":[\"'"$cell"'\"]}} first:1) { edges { node { name state historyLineCount summaryMetrics } } } } }"}'
    local resp
    resp=$(curl -s --max-time 15 -H "Authorization: Bearer $WANDB_API_KEY" \
        -H "Content-Type: application/json" \
        -d "$query" \
        "https://api.wandb.ai/graphql" 2>&1) || true

    if echo "$resp" | python3 -c "import sys,json; d=json.load(sys.stdin); runs=d['data']['project']['runs']['edges']; print('WANDB_OK' if runs else 'WANDB_NO_RUN')" 2>/dev/null | grep -q "WANDB_OK"; then
        local summary
        summary=$(echo "$resp" | python3 -c "
import sys, json
d = json.load(sys.stdin)
edges = d['data']['project']['runs']['edges']
if not edges:
    print('NO_RUN')
    sys.exit(0)
node = edges[0]['node']
state = node.get('state','?')
hlc = node.get('historyLineCount', '?')
sm_raw = node.get('summaryMetrics','{}')
sm = json.loads(sm_raw) if sm_raw else {}
acc = sm.get('val-core/openai/gsm8k/acc/mean@1', sm.get('train/val-core/openai/gsm8k/acc/mean@1','?'))
step = sm.get('_step','?')
br = sm.get('actor/comm/bytes_ratio','?')
grad = sm.get('actor/grad_norm','?')
print(f'state={state} hlc={hlc} step={step} val_acc={acc} bytes_ratio={br} grad_norm={grad}')
" 2>/dev/null || echo "PARSE_ERR")
        log "WANDB [$cell]: $summary"
    else
        log "WANDB [$cell]: UNREACHABLE or error ($(echo "$resp" | head -c 200))"
    fi
}

write_result() {
    local exit_state="$1"
    local rec="$2"
    local elapsed=$(( $(date +%s) - START_TIME ))
    cat > "$RESULT_FILE" << EOF
{
  "exit_state": "$exit_state",
  "recommendation": "$rec",
  "elapsed_s": $elapsed,
  "poll_count": $POLL_NUM,
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "note": "See monitor-detail.log for full trace"
}
EOF
    log "EXIT: state=$exit_state rec=$rec elapsed=${elapsed}s"
}

log "=== MONITOR LOOP START: window-3, EXP-33, polls every 30s, max 40min ==="
log "SSH: $HOST, tmux=exp-33-84_8_116_228"
log "Expected: C0(b0p00) ~step25 RUNNING, key event = val@50 ~step50 + done_b0p00.flag + C1 launch"

while true; do
    POLL_NUM=$(( POLL_NUM + 1 ))
    NOW=$(date +%s)
    ELAPSED=$(( NOW - START_TIME ))
    POLL_TIME=$(date -u +%Y-%m-%dT%H:%M:%SZ)

    if (( ELAPSED >= MAX_WALL )); then
        log "TIMEOUT: 40 min elapsed at poll $POLL_NUM"
        write_result "TIMEOUT" "continue_monitoring"
        break
    fi

    # Run the remote poll
    POLL_RAW=$($SSH "$HOST" 'python3 - ' <<'PYEOF'
import re, sys, os, subprocess, glob, json

def read_log_tail(path, nbytes=131072):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2); sz = f.tell()
            f.seek(max(0, sz - nbytes))
            return f.read().decode("utf-8", errors="replace"), sz
    except Exception as e:
        return None, 0

out = {}

# TMUX alive
out["tmux"] = "ALIVE" if os.system("tmux has-session -t exp-33-84_8_116_228 2>/dev/null") == 0 else "DEAD"

# Done flags
flags = glob.glob("/workspace/runs/EXP-33/done*.flag")
out["done_flags"] = [os.path.basename(f) for f in flags]
out["agg_done"] = os.path.exists("/workspace/runs/EXP-33/done.flag")
out["control_fail"] = os.path.exists("/workspace/runs/EXP-33/CONTROL_FAIL.flag")

# Active symlink
try:
    out["active_symlink"] = os.readlink("/workspace/train.log")
except:
    out["active_symlink"] = "NONE"

# Per cell scan
active_cells = ["b0p00","b0p25","b0p50","b0p75","b1p00"]
per_cell = {}
for cell in active_cells:
    log_path = f"/workspace/verl/runs/{cell}/train.log"
    if not os.path.exists(log_path):
        per_cell[cell] = {"exists": False}
        continue
    tail, sz = read_log_tail(log_path, 131072)
    if not tail:
        per_cell[cell] = {"exists": True, "sz": sz, "error": "CANNOT_READ"}
        continue
    lines = tail.split("\n")
    c = {"exists": True, "sz": sz}

    step_lines = [(i,l) for i,l in enumerate(lines) if "global_step:" in l and "critic/score" in l]
    if step_lines:
        ln, last_step = step_lines[-1]
        ms = re.search(r"global_step:(\d+)", last_step); c["step"] = int(ms.group(1)) if ms else -1
        ms2 = re.search(r"critic/score/mean:([\d.]+)", last_step); c["score"] = float(ms2.group(1)) if ms2 else None
        ms3 = re.search(r"response_length/mean:([\d.]+)", last_step); c["resp_len"] = float(ms3.group(1)) if ms3 else None
        ms4 = re.search(r"response_length/clip_ratio:([\d.]+)", last_step); c["clip_ratio"] = float(ms4.group(1)) if ms4 else None
        ms5 = re.search(r"anchor_backwards:(\d+)", last_step); c["anchor_bwd"] = int(ms5.group(1)) if ms5 else None
        ms6 = re.search(r"bytes_ratio:([\d.]+)", last_step); c["bytes_ratio"] = float(ms6.group(1)) if ms6 else None
    else:
        c["step"] = -1

    c["vals"] = {}
    for i, l in enumerate(lines):
        if "val-core/openai/gsm8k/acc/mean@1" in l and "step:" in l:
            ms = re.search(r"step:(\d+)", l)
            ma = re.search(r"val-core/openai/gsm8k/acc/mean@1:([\d.]+)", l)
            if ms and ma:
                c["vals"][int(ms.group(1))] = float(ma.group(1))

    c["errors"] = sum(1 for l in lines if any(p in l for p in ["Traceback (most recent", "RuntimeError:", "CUDA out of memory", "NaN detected", "FATAL", "EngineCore", "custom_all_reduce"]))

    # Error details (first 3)
    err_details = []
    for l in lines:
        if any(p in l for p in ["Traceback (most recent", "RuntimeError:", "CUDA out of memory", "NaN detected", "FATAL", "EngineCore", "custom_all_reduce"]):
            err_details.append(l[:200])
            if len(err_details) >= 3: break
    c["err_details"] = err_details

    # Ignition tripwires
    clip_ratios = []
    resp_lens = []
    for l in lines:
        if "global_step:" in l and "response_length/clip_ratio" in l:
            m_s = re.search(r"global_step:(\d+)", l)
            m_cr = re.search(r"response_length/clip_ratio:([\d.]+)", l)
            m_rl = re.search(r"response_length/mean:([\d.]+)", l)
            if m_s and m_cr:
                step_n = int(m_s.group(1))
                cr = float(m_cr.group(1))
                rl = float(m_rl.group(1)) if m_rl else 0.0
                clip_ratios.append((step_n, cr))
                resp_lens.append((step_n, rl))

    consec_caps = 0; max_consec = 0; prev_capped = False
    for st, cr in clip_ratios:
        if cr > 0:
            if prev_capped: consec_caps += 1
            else: consec_caps = 1
            max_consec = max(max_consec, consec_caps)
            prev_capped = True
        else:
            consec_caps = 0; prev_capped = False
    c["P1_consec_caps"] = max_consec
    c["P1_FIRE"] = max_consec >= 2

    early_lens = [rl for st, rl in resp_lens if 10 <= st <= 30]
    c["early_resp_len_mean"] = sum(early_lens)/len(early_lens) if early_lens else 0.0
    c["E1_FIRE"] = c["early_resp_len_mean"] > 4000

    per_cell[cell] = c

out["per_cell"] = per_cell

r = subprocess.run(["nvidia-smi","--query-gpu=index,utilization.gpu,memory.used,memory.total","--format=csv,noheader"], capture_output=True, text=True)
out["gpu"] = r.stdout.strip()

try:
    with open("/workspace/runs/EXP-33/driver.log","rb") as f:
        f.seek(0,2); sz=f.tell(); f.seek(max(0,sz-1000))
        drv = f.read().decode("utf-8", errors="replace")
    driver_lines = [l for l in drv.split("\n") if l.strip()]
    out["driver_tail"] = driver_lines[-3:]
except:
    out["driver_tail"] = []

print(json.dumps(out))
PYEOF
2>&1) || POLL_RAW="SSH_FAILED"

    # Extract JSON from output (skip SSH banner)
    POLL_JSON=$(echo "$POLL_RAW" | grep '^{' | head -1 || echo "")

    if [ -z "$POLL_JSON" ]; then
        log "POLL $POLL_NUM [$POLL_TIME]: SSH_FAILED or no JSON (raw=$(echo "$POLL_RAW" | head -c 200))"
        # Check if SSH has been failing for >2 min
        if [ $POLL_NUM -gt 4 ]; then
            log "ENV_FAILURE: SSH unreachable, recommend env_failure"
            write_result "ENV_FAILURE" "env_failure"
            break
        fi
        sleep 30
        continue
    fi

    # Parse JSON with Python
    PARSE_RESULT=$(python3 - "$POLL_JSON" "$POLL_TIME" "$POLL_NUM" "$RUNDIR" << 'PYEOF2'
import sys, json, re

poll_json = sys.argv[1]
ts = sys.argv[2]
poll_num = int(sys.argv[3])
rundir = sys.argv[4]

try:
    d = json.loads(poll_json)
except Exception as e:
    print(f"JSON_ERR:{e}")
    sys.exit(1)

tmux = d.get("tmux","?")
done_flags = d.get("done_flags",[])
agg_done = d.get("agg_done", False)
control_fail = d.get("control_fail", False)
active_sym = d.get("active_symlink","?")
per_cell = d.get("per_cell",{})
gpu_str = d.get("gpu","")

# Parse GPU utils
gpu_utils = []
for line in gpu_str.split("\n"):
    parts = [p.strip() for p in line.split(",")]
    if len(parts) == 4:
        try:
            util = int(parts[1].replace("%","").strip())
            gpu_utils.append(util)
        except:
            pass

all_gpus_idle = all(u <= 5 for u in gpu_utils) if gpu_utils else False
any_gpu_active = any(u > 5 for u in gpu_utils) if gpu_utils else True

# Determine active cell
active_cell = "NONE"
for cell in ["b0p00","b0p25","b0p50","b0p75","b1p00"]:
    if cell in active_sym or (per_cell.get(cell,{}).get("exists") and per_cell.get(cell,{}).get("step",-1) >= 0):
        # Most recent cell with data
        active_cell = cell

# Write to log
with open(f"{rundir}/monitor-detail.log","a") as f:
    f.write(f"\nPOLL {poll_num} [{ts}]\n")
    f.write(f"  TMUX={tmux} AGG_DONE={agg_done} CONTROL_FAIL={control_fail}\n")
    f.write(f"  DONE_FLAGS={done_flags}\n")
    f.write(f"  ACTIVE_SYMLINK={active_sym}\n")

    for cell in ["b0p00","b0p25","b0p50","b0p75","b1p00"]:
        c = per_cell.get(cell,{})
        if not c.get("exists"):
            continue
        step = c.get("step",-1)
        if step < 0 and not c.get("vals"):
            continue
        vals = c.get("vals",{})
        val_str = " ".join([f"val@{k}={v:.5f}" for k,v in sorted(vals.items())])
        p1 = "IGNITION_P1!" if c.get("P1_FIRE") else ""
        e1 = "IGNITION_E1!" if c.get("E1_FIRE") else ""
        ignition = p1 or e1 or "OK"
        errs = c.get("errors",0)
        err_d = " | ERR: " + str(c.get("err_details",[])) if errs > 0 else ""
        f.write(f"  {cell}: step={step} score={c.get('score')} resp_len={c.get('resp_len')} clip={c.get('clip_ratio')} bytes_ratio={c.get('bytes_ratio')} anc_bwd={c.get('anchor_bwd')} errors={errs} | {val_str} | IGN={ignition}{err_d}\n")

    gpu_lines = gpu_str.replace("\n"," | ")
    f.write(f"  GPU: {gpu_lines}\n")
    f.write(f"  GPU_UTILS_VALS: {gpu_utils}\n")
    f.write(f"  ALL_GPUS_IDLE: {all_gpus_idle}\n")

    drv = d.get("driver_tail",[])
    if drv:
        f.write(f"  DRIVER_TAIL: {drv[-1][:200]}\n")

# Compute exit conditions
b0 = per_cell.get("b0p00",{})
b0_step = b0.get("step",-1)
b0_vals = b0.get("vals",{})
b0_val50 = b0_vals.get(50, None)

# Output structured summary
summary = {
    "tmux": tmux,
    "agg_done": agg_done,
    "control_fail": control_fail,
    "done_flags": done_flags,
    "active_sym": active_sym,
    "b0_step": b0_step,
    "b0_vals": b0_vals,
    "b0_val50": b0_val50,
    "b0_errors": b0.get("errors",0),
    "b0_P1": b0.get("P1_FIRE",False),
    "b0_E1": b0.get("E1_FIRE",False),
    "gpu_utils": gpu_utils,
    "all_gpus_idle": all_gpus_idle,
    "done_b0p00": "done_b0p00.flag" in done_flags,
    "done_b0p25": "done_b0p25.flag" in done_flags,
}
# Include all cell steps
for cell in ["b0p25","b0p50","b0p75","b1p00"]:
    c = per_cell.get(cell,{})
    summary[f"{cell}_step"] = c.get("step",-1)
    summary[f"{cell}_vals"] = c.get("vals",{})
    summary[f"{cell}_errors"] = c.get("errors",0)

import json
print("SUMMARY:" + json.dumps(summary))
PYEOF2
)

    # Extract summary JSON
    SUMMARY_JSON=$(echo "$PARSE_RESULT" | grep '^SUMMARY:' | sed 's/^SUMMARY://' || echo "")

    if [ -z "$SUMMARY_JSON" ]; then
        log "POLL $POLL_NUM [$POLL_TIME]: PARSE_FAILED"
        sleep 30
        continue
    fi

    # Check exit conditions via Python
    EXIT_CHECK=$(python3 - "$SUMMARY_JSON" "$POLL_NUM" "$GPU_STALL_COUNT" "$RSYNCED_CELLS" << 'PYEOF3'
import sys, json

summary_json = sys.argv[1]
poll_num = int(sys.argv[2])
gpu_stall_count = int(sys.argv[3])
rsynced = sys.argv[4]

d = json.loads(summary_json)

tmux = d["tmux"]
agg_done = d["agg_done"]
control_fail = d["control_fail"]
done_flags = d["done_flags"]
b0_step = d["b0_step"]
b0_vals = {int(k):v for k,v in d["b0_vals"].items()}
b0_val50 = d.get("b0_val50")
gpu_utils = d.get("gpu_utils",[])
all_idle = d.get("all_gpus_idle", False)

# Cell done tracking
done_b0p00 = "done_b0p00.flag" in done_flags
done_count = len(done_flags)

# Summary print for logging
print(f"STEP: b0p00={b0_step} b0_vals={sorted(b0_vals.items())} val@50={b0_val50}")
print(f"TMUX={tmux} DONE={done_flags} AGG_DONE={agg_done} CTRL_FAIL={control_fail}")
print(f"GPU_UTILS={gpu_utils} ALL_IDLE={all_idle}")

# EXIT: aggregate done
if agg_done:
    print("EXIT:DONE_AGGREGATE:dispatch_analyst")
    sys.exit(0)

# EXIT: CONTROL_FAIL
if control_fail:
    print("EXIT:CONTROL_FAIL:control_fail_reprovision")
    sys.exit(0)

# EXIT: 3+ cells done and tmux dead
if done_count >= 3 and tmux == "DEAD":
    print("EXIT:DONE_3FLAGS:dispatch_analyst")
    sys.exit(0)

# EXIT: TMUX dead and fewer than expected done flags
if tmux == "DEAD" and done_count < 5:
    print(f"EXIT:TMUX_DEAD_PREMATURE:env_failure  (done={done_count}, expected 5)")
    sys.exit(0)

# GPU stall check (all <=5% for 4 consecutive polls)
if all_idle and tmux == "ALIVE" and not agg_done:
    new_stall = gpu_stall_count + 1
    print(f"GPU_STALL_COUNT:{new_stall}")
    if new_stall >= 4:
        print("EXIT:GPU_STALL:investigate_stall")
        sys.exit(0)
else:
    print("GPU_STALL_COUNT:0")

# Done flag tracking for rsync
for cell, flag in [("b0p00","done_b0p00.flag"),("b0p25","done_b0p25.flag"),("b0p50","done_b0p50.flag"),("b0p75","done_b0p75.flag"),("b1p00","done_b1p00.flag")]:
    if flag in done_flags and cell not in rsynced:
        print(f"RSYNC_NEEDED:{cell}")

print("CONTINUE:polling")
PYEOF3
)

    log "POLL $POLL_NUM [$POLL_TIME]: $(echo "$EXIT_CHECK" | grep '^STEP:' | head -1)"
    log "  $(echo "$EXIT_CHECK" | grep '^TMUX=' | head -1)"
    log "  $(echo "$EXIT_CHECK" | grep '^GPU_UTILS=' | head -1)"

    # Handle GPU stall count
    NEW_STALL=$(echo "$EXIT_CHECK" | grep '^GPU_STALL_COUNT:' | sed 's/GPU_STALL_COUNT://' || echo "")
    if [ -n "$NEW_STALL" ]; then
        GPU_STALL_COUNT=$NEW_STALL
        if [ "$GPU_STALL_COUNT" -gt 0 ]; then
            log "  GPU_STALL_COUNT=$GPU_STALL_COUNT (threshold=4)"
        fi
    fi

    # Handle rsync needed
    for RSYNC_CELL in $(echo "$EXIT_CHECK" | grep '^RSYNC_NEEDED:' | sed 's/RSYNC_NEEDED://'); do
        log "  RSYNC: pulling $RSYNC_CELL (done flag appeared)"
        do_rsync "$RSYNC_CELL"
        RSYNCED_CELLS="$RSYNCED_CELLS $RSYNC_CELL"
    done

    # Check for exit conditions
    EXIT_LINE=$(echo "$EXIT_CHECK" | grep '^EXIT:' | head -1 || echo "")
    if [ -n "$EXIT_LINE" ]; then
        EXIT_STATE=$(echo "$EXIT_LINE" | cut -d: -f2)
        EXIT_REC=$(echo "$EXIT_LINE" | cut -d: -f3)
        log "EXIT CONDITION: $EXIT_LINE"
        # Do final rsync for all cells with done flags
        for cell in b0p00 b0p25 b0p50 b0p75 b1p00; do
            if echo "$RSYNCED_CELLS" | grep -q "$cell" || true; then
                : # already done
            fi
            do_rsync "$cell" || true
        done
        write_result "$EXIT_STATE" "$EXIT_REC"
        break
    fi

    # WandB poll (every ~3 polls)
    if (( POLL_NUM % 3 == 0 )); then
        # Determine active cell from summary
        ACTIVE_CELL=$(python3 -c "
import json, sys
d = json.loads(sys.argv[1])
sym = d.get('active_sym','')
for cell in ['b0p00','b0p25','b0p50','b0p75','b1p00']:
    if cell in sym:
        print(cell); sys.exit(0)
print('b0p00')
" "$SUMMARY_JSON" 2>/dev/null || echo "b0p00")
        poll_wandb "$ACTIVE_CELL" || true
    fi

    sleep 30
done

log "=== MONITOR LOOP ENDED ==="
