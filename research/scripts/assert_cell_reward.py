"""Judge one smoke cell's train.log: are rewards non-null and NaN/Inf-free?

Reuses parse_train_log.py's per-step JSONL. A smoke cell PASSES (for issue #62
stage 3) when:
  - at least one TRAIN reward value exists and is finite (non-null, no NaN/Inf);
  - at least one VALIDATION reward value exists and is finite;
  - no reward/score metric anywhere is NaN or Inf.

Train reward key = critic/score/mean (verl prints the batch reward mean here even
for critic-free GRPO). Validation reward = any key beginning with "val" that
contains "score" or "reward". Prints a JSON verdict and exits non-zero on FAIL so
the matrix driver can record per-cell status.

Usage:
  python research/scripts/assert_cell_reward.py runs/<exp>/train.log
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

ANSI = re.compile(r"\x1b\[[0-9;]*m")
PREFIX = re.compile(r"^.*?(?=\bstep:\d)")
PAIR = re.compile(r"([A-Za-z0-9_./]+):(-?[0-9][0-9.eE+\-]*|nan|inf|-inf)")

TRAIN_REWARD_KEYS = ("critic/score/mean", "critic/rewards/mean")


def _to_num(v: str):
    vl = v.lower()
    if vl in ("nan", "inf", "-inf"):
        return float(vl)
    try:
        return float(v)
    except ValueError:
        return None


def parse_steps(log: Path):
    rows = []
    for raw in log.read_text(errors="replace").splitlines():
        line = ANSI.sub("", raw)
        if "step:" not in line or " - " not in line:
            continue
        line = PREFIX.sub("", line)
        if not line.startswith("step:"):
            continue
        row = {}
        for k, v in PAIR.findall(line):
            num = _to_num(v)
            if num is not None:
                row[k] = num
        if row:
            rows.append(row)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("log", type=Path)
    args = ap.parse_args()

    verdict = {"log": str(args.log), "pass": False, "reason": "",
               "train_reward": None, "val_reward": None, "nan_keys": []}
    if not args.log.exists():
        verdict["reason"] = "train.log missing"
        print(json.dumps(verdict))
        return 2

    rows = parse_steps(args.log)
    if not rows:
        verdict["reason"] = "no metric lines parsed"
        print(json.dumps(verdict))
        return 1

    train_vals, val_vals, nan_keys = [], [], []
    for row in rows:
        for k, v in row.items():
            low = k.lower()
            is_reward = ("score" in low) or ("reward" in low)
            if is_reward and (math.isnan(v) or math.isinf(v)):
                nan_keys.append(k)
            if k in TRAIN_REWARD_KEYS and math.isfinite(v):
                train_vals.append(v)
            if low.startswith("val") and is_reward and math.isfinite(v):
                val_vals.append(v)

    verdict["train_reward"] = train_vals[-1] if train_vals else None
    verdict["val_reward"] = val_vals[-1] if val_vals else None
    verdict["nan_keys"] = sorted(set(nan_keys))

    if nan_keys:
        verdict["reason"] = f"NaN/Inf in reward keys: {sorted(set(nan_keys))[:5]}"
    elif not train_vals:
        verdict["reason"] = "no finite train reward (critic/score/mean) found"
    elif not val_vals:
        verdict["reason"] = "no finite validation reward found"
    else:
        verdict["pass"] = True
        verdict["reason"] = "non-null train+val rewards, no NaN/Inf"

    print(json.dumps(verdict))
    return 0 if verdict["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
