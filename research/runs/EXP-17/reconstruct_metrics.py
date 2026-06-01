#!/usr/bin/env python3
"""Reconstruct metrics/train.jsonl from train.log for EXP-17.

The per-step scalar jsonl was never synced (sync-metrics errored on a missing
hotfix-patches dir). The trainer prints the full verl metrics dict each step in
the form:

    (TaskRunner pid=NNNN) step:N - key:value - key:value - ...

This parser strips the ray prefix and any trailing ray-progress noise, splits
on ' - ', and emits one JSON object per training step (step >= 1; step:0 is the
val-before-train line which is folded into step0 val fields). Output is sorted
by global_step.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

RUN = Path(__file__).resolve().parent
LOG = RUN / "train.log"
OUT = RUN / "metrics" / "train.jsonl"

# ray prefix like "\x1b[36m(TaskRunner pid=9831)\x1b[0m "
PREFIX = re.compile(r"^\x1b\[[0-9;]*m?\(TaskRunner pid=\d+\)\x1b\[[0-9;]*m?\s*")
# a trailing ray progress bar can be glued onto the line; cut it off
PROGRESS = re.compile(r"\x1b\[[0-9;]*m?\(TaskRunner pid=\d+\)")


def parse_value(s: str):
    s = s.strip()
    # try int, then float, else keep string
    try:
        if re.fullmatch(r"-?\d+", s):
            return int(s)
        return float(s)
    except ValueError:
        return s


def main() -> int:
    rows: dict[int, dict] = {}
    for raw in LOG.read_text(errors="replace").splitlines():
        line = PREFIX.sub("", raw)
        if not line.startswith("step:"):
            continue
        # strip any glued-on progress bar / second ray prefix
        line = PROGRESS.split(line)[0].strip()
        # also strip a bare "Training Progress" suffix if present
        line = re.split(r"\s*Training Progress", line)[0].strip()
        parts = line.split(" - ")
        # first token is "step:N"
        head = parts[0]
        m = re.match(r"step:(\d+)", head)
        if not m:
            continue
        step = int(m.group(1))
        rec: dict = {}
        # include the leading step token's value too (== global_step usually)
        for tok in parts:
            if ":" not in tok:
                continue
            k, _, v = tok.partition(":")
            k = k.strip()
            if k == "step":
                continue
            rec[k] = parse_value(v)
        rec["step"] = step
        # prefer training/global_step if present
        gs = rec.get("training/global_step", step)
        # merge (step:0 val-only line keeps step 0)
        if step in rows:
            rows[step].update(rec)
        else:
            rows[step] = rec

    with OUT.open("w") as fh:
        for step in sorted(rows):
            fh.write(json.dumps(rows[step]) + "\n")
    print(f"reconstruct: wrote {OUT} with {len(rows)} step rows "
          f"(steps {min(rows)}..{max(rows)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
