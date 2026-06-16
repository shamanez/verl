#!/usr/bin/env bash
# EXP-33 monitor poll script - runs in background
# Called per poll by the monitoring loop

SSH="ssh -i ~/.ssh/vast_ai -p 40154 -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=15 root@84.8.116.228"
LOG="/Users/shamane/Documents/verl/research/runs/EXP-33/monitor-detail.log"
RUNDIR="/Users/shamane/Documents/verl/research/runs/EXP-33"

POLL_TIME=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

RESULT=$($SSH 'python3 - ' <<'PYEOF'
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

# Find active cell log
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

    # Last step
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

    # Val lines
    c["vals"] = {}
    for i, l in enumerate(lines):
        if "val-core/openai/gsm8k/acc/mean@1" in l and "step:" in l:
            ms = re.search(r"step:(\d+)", l)
            ma = re.search(r"val-core/openai/gsm8k/acc/mean@1:([\d.]+)", l)
            if ms and ma:
                c["vals"][int(ms.group(1))] = float(ma.group(1))

    # Error count
    c["errors"] = sum(1 for l in lines if any(p in l for p in ["Traceback (most recent", "RuntimeError:", "CUDA out of memory", "NaN detected", "FATAL", "EngineCore", "custom_all_reduce"]))

    # Ignition tripwires P1/P2/P3/E1
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

    # P1: >= 2 consecutive cap-pins (clip_ratio > 0 on consecutive steps)
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

    # P2/P3/E1: early resp_len stats
    early_lens = [rl for st, rl in resp_lens if 10 <= st <= 30]
    if early_lens:
        c["early_resp_len_mean"] = sum(early_lens)/len(early_lens)
        c["E1_FIRE"] = c["early_resp_len_mean"] > 4000  # E1: len/max > 4k at steps 10-30

    per_cell[cell] = c

out["per_cell"] = per_cell

# GPU util
r = subprocess.run(["nvidia-smi","--query-gpu=index,utilization.gpu,memory.used,memory.total","--format=csv,noheader"], capture_output=True, text=True)
out["gpu"] = r.stdout.strip()

# Driver log last 5 lines
try:
    with open("/workspace/runs/EXP-33/driver.log","rb") as f:
        f.seek(0,2); sz=f.tell(); f.seek(max(0,sz-2000))
        drv = f.read().decode("utf-8", errors="replace")
    driver_lines = [l for l in drv.split("\n") if l.strip()]
    out["driver_tail"] = driver_lines[-5:]
except:
    out["driver_tail"] = []

print(json.dumps(out))
PYEOF
2>&1)

echo "$POLL_TIME|$RESULT" >> "$LOG.raw"

# Parse and format
python3 - "$POLL_TIME" "$RESULT" "$RUNDIR" << 'PYEOF2'
import sys, json, re

ts = sys.argv[1]
raw = sys.argv[2]
rundir = sys.argv[3]

# Strip SSH banner
lines = raw.split("\n")
json_lines = [l for l in lines if l.startswith("{")]
if not json_lines:
    with open(f"{rundir}/monitor-detail.log", "a") as f:
        f.write(f"\nPOLL [{ts}] ERROR: no JSON in output\n")
        f.write(raw[:2000] + "\n")
    print(f"[{ts}] POLL_ERROR: no JSON")
    sys.exit(1)

try:
    d = json.loads(json_lines[0])
except Exception as e:
    with open(f"{rundir}/monitor-detail.log", "a") as f:
        f.write(f"\nPOLL [{ts}] JSON_PARSE_ERROR: {e}\n")
    print(f"[{ts}] JSON_PARSE_ERROR: {e}")
    sys.exit(1)

with open(f"{rundir}/monitor-detail.log", "a") as f:
    f.write(f"\nPOLL [{ts}]\n")
    f.write(f"  TMUX={d.get('tmux')} DONE_FLAGS={d.get('done_flags')} AGG_DONE={d.get('agg_done')} CONTROL_FAIL={d.get('control_fail')}\n")
    f.write(f"  ACTIVE_SYMLINK={d.get('active_symlink')}\n")

    for cell in ["b0p00","b0p25","b0p50","b0p75","b1p00"]:
        c = d.get("per_cell", {}).get(cell, {})
        if not c.get("exists"):
            continue
        step = c.get("step",-1)
        if step < 0:
            continue
        vals = c.get("vals",{})
        val_str = " ".join([f"val@{k}={v:.5f}" for k,v in sorted(vals.items())])
        ignition = "IGNITION!" if c.get("P1_FIRE") or c.get("E1_FIRE") else "OK"
        f.write(f"  {cell.upper()}: step={step} score={c.get('score')} resp_len={c.get('resp_len')} clip={c.get('clip_ratio')} bytes_ratio={c.get('bytes_ratio')} anc_bwd={c.get('anchor_bwd')} errors={c.get('errors')} | {val_str} | IGNITION={ignition}\n")

    gpu = d.get("gpu","")
    f.write(f"  GPU: {gpu.replace(chr(10),' | ')}\n")

    drv = d.get("driver_tail", [])
    if drv:
        f.write(f"  DRIVER_TAIL: {drv[-1][:200]}\n")

# Print summary line
pc = d.get("per_cell",{})
b0 = pc.get("b0p00",{})
step = b0.get("step",-1)
vals = b0.get("vals",{})
val_strs = [f"val@{k}={v:.5f}" for k,v in sorted(vals.items())]
print(f"[{ts}] tmux={d.get('tmux')} done_flags={d.get('done_flags')} b0p00_step={step} {' '.join(val_strs)} errors={b0.get('errors',0)}")
PYEOF2
