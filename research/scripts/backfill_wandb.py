import os
import re
import sys
from pathlib import Path

import wandb

# args: <train_log> <run_id> <target_step>
# Relog the final training step (WandB's async uploader drops the last 1-2 steps,
# esp. val@<final>). resume="must" appends; parses the `step:N - k:v - ...` line.
log_path, run_id, target = sys.argv[1], sys.argv[2], int(sys.argv[3])

line = None
with open(log_path, errors="ignore") as f:
    for ln in f:
        if f"training/global_step:{target}" in ln:
            line = ln
            break
if line is None:
    print(f"NO LINE for global_step:{target}")
    sys.exit(2)

line = re.sub(r"\x1b\[[0-9;]*m", "", line)
i = line.find("global_seqlen")
if i > 0:
    line = line[i:]
metrics = {}
for tok in line.split(" - "):
    tok = tok.strip()
    if ":" not in tok:
        continue
    k, v = tok.split(":", 1)
    k = k.strip()
    v = v.strip()
    if " " in k or "(" in k or k == "":
        continue
    try:
        metrics[k] = float(v)
    except ValueError:
        pass
print(f"parsed {len(metrics)} keys for step {target}")
preview_keys = ["training/global_step"]
preview_keys.extend(sorted(key for key in metrics if key.startswith("val-core/")))
preview_keys.extend(("critic/score/mean", "actor/comm_eff/anchor_align_cos", "response_length/mean"))
for key in dict.fromkeys(preview_keys):
    print(f"  {key} = {metrics.get(key)}")

# Keep wandb's local staging dir UNDER the run dir so close_cleanup (which
# removes only runs/<id>/) sweeps it — otherwise wandb.init() drops a ./wandb/
# in cwd (research/), a shared dir no cleanup step ever removes.
_wdir = os.environ.get("WANDB_DIR")
if not _wdir:
    _p = Path(log_path).resolve()           # runs/<id>/metrics/train.log -> runs/<id>
    _wdir = str(_p.parents[1]) if len(_p.parents) >= 2 else str(_p.parent)
os.makedirs(_wdir, exist_ok=True)
run = wandb.init(project="verl_compression_research", entity="shamanework-pl",
                 id=run_id, resume="must", dir=_wdir)
wandb.log(metrics, step=target)
wandb.finish()
print("RELOG DONE for", run_id, "step", target)
