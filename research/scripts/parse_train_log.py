"""Parse a verl console train.log into a per-step metrics JSONL.

The WandB-independent fallback for the curve-match. verl's console logger prints
one ` - `-separated `key:value` line per optimizer step (e.g.
`step:31 - actor/pg_loss:0.0053 - actor/grad_norm:0.35 - critic/score/mean:0.818 - training/global_step:31 - ...`).
When a run's WandB history is incomplete (e.g. the chained dense+floor launch left
the dense run `state=crashed` with 48/50 steps flushed), this recovers the FULL
per-step curve from the local log the monitor rsynced.

Usage:
    python research/scripts/parse_train_log.py \
        runs/EXP-18/train_curvematch_dense_ref_50step.log \
        --out runs/EXP-18/metrics/curvematch_dense_ref_50step.jsonl

Emits one JSON object per step with `step` (= training/global_step, else the
leading step:N) and every numeric `key:value` pair on that line. Idempotent;
overwrites --out. De-dups by step (last line for a step wins).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ANSI = re.compile(r"\x1b\[[0-9;]*m")
PREFIX = re.compile(r"^.*?(?=\bstep:\d)")  # strip "(TaskRunner pid=N) " etc. up to "step:N"
# A token is key:value where key may contain / and _ and . ; value is the rest up to the next " - ".
PAIR = re.compile(r"([A-Za-z0-9_./]+):(-?[0-9][0-9.eE+\-]*|nan|inf|-inf)")


def _to_num(v: str):
    vl = v.lower()
    if vl in ("nan", "inf", "-inf"):
        return float(vl)
    try:
        return float(v)
    except ValueError:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("log", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    if not args.log.exists():
        print(f"parse_train_log: log not found: {args.log}")
        return 2

    by_step: dict[int, dict] = {}
    text = args.log.read_text(errors="replace")
    for raw in text.splitlines():
        line = ANSI.sub("", raw)
        # Only metric lines carry the leading "step:N - " token AND a global_step.
        if "step:" not in line or " - " not in line:
            continue
        line = PREFIX.sub("", line)
        if not line.startswith("step:"):
            continue
        row: dict = {}
        for k, v in PAIR.findall(line):
            num = _to_num(v)
            if num is not None:
                row[k] = num
        # Resolve the training step.
        step = row.get("training/global_step", row.get("step"))
        if step is None:
            continue
        try:
            step = int(step)
        except (TypeError, ValueError):
            continue
        row["step"] = step
        by_step[step] = row  # last line for a step wins

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        for step in sorted(by_step):
            fh.write(json.dumps(by_step[step]) + "\n")

    steps = sorted(by_step)
    print(f"parse_train_log: wrote {len(steps)} steps → {args.out}"
          + (f" (first={steps[0]}, last={steps[-1]})" if steps else " (no steps parsed)"))
    return 0 if steps else 1


if __name__ == "__main__":
    raise SystemExit(main())
