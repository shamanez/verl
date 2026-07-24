# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Analyse a finished experiment and emit a verdict.md skeleton.

Called by the analyst subagent as:
    python research/scripts/analyze.py runs/<run-id> --emit verdict.md

Default behaviour (sufficient for M0 smoke):
- If `done.flag` exists in the run dir, treat the run as complete.
- Read every `metrics/*.jsonl` file in the run dir and compute a small summary.
- Emit a verdict.md skeleton with VERDICT: PASS (for M0) or VERDICT: PENDING
  (for real experiments — the analyst fills the success-criteria checkboxes
  by inspecting the metrics summary and the plan).

This script is intentionally a scaffold. Per-experiment plans wire their own
predicates by passing extra flags or by the analyst editing the emitted file.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def _load_jsonl(p: Path) -> list[dict]:
    rows: list[dict] = []
    if not p.exists():
        return rows
    with p.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _summarise(rows: list[dict]) -> dict:
    if not rows:
        return {"n_rows": 0}
    keys = sorted({k for r in rows for k in r.keys()})  # noqa: F841  (kept: documents the row schema)
    out: dict = {"n_rows": len(rows), "first": rows[0], "last": rows[-1]}
    nums = defaultdict(list)
    for r in rows:
        for k, v in r.items():
            if isinstance(v, int | float) and v == v:  # filter NaN
                nums[k].append(v)
    out["min"] = {k: min(v) for k, v in nums.items()}
    out["max"] = {k: max(v) for k, v in nums.items()}
    out["mean"] = {k: sum(v) / len(v) for k, v in nums.items()}
    return out


def _is_m0_smoke(run_dir: Path) -> bool:
    # A run on real project hardware (H100/H200/B200) is NEVER auto-PASSed as a
    # smoke test — the analyst must judge it against the plan's criteria. The old
    # logic keyed on gpu_name == "h100" and returned True (=> auto-PASS) for
    # anything else, so every real H200/B200 run was silently marked PASS. Invert:
    # positively recognise the fixed hardware and treat those runs as NON-smoke.
    handles_dir = run_dir / "handles"
    if not handles_dir.is_dir():
        return False
    known = ("h100", "h200", "b200")
    for hf in handles_dir.glob("*.json"):
        try:
            h = json.loads(hf.read_text())
            if any(g in str(h.get("gpu_name", "")).lower() for g in known):
                return False  # real GPU -> real run -> analyst judges (PENDING)
        except Exception:
            continue
    return True  # no recognisable GPU handle -> a scaffold/smoke run


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--emit", default="verdict.md", help="filename (inside run_dir) to write the verdict to")
    args = ap.parse_args()

    run_dir: Path = args.run_dir.resolve()
    if not run_dir.is_dir():
        print(f"analyze: not a directory: {run_dir}", file=sys.stderr)
        return 2

    exp_id = run_dir.name
    done = (run_dir / "done.flag").exists()
    metrics_dir = run_dir / "metrics"
    metric_files = sorted(metrics_dir.glob("*.jsonl")) if metrics_dir.is_dir() else []
    summaries = {p.stem: _summarise(_load_jsonl(p)) for p in metric_files}

    has_nan = False
    nan_step = None
    train_summary = summaries.get("train", {})
    if train_summary:
        last = train_summary.get("last", {})
        for k, v in last.items():
            if isinstance(v, float) and v != v:
                has_nan = True
                nan_step = last.get("step")
                break

    is_smoke = _is_m0_smoke(run_dir)
    if has_nan:
        verdict = "STOP"
        note = f"non-finite metric detected at step {nan_step}"
    elif is_smoke and done:
        verdict = "PASS"
        note = "M0 smoke: done.flag present, no NaN — harness validated"
    elif not done:
        verdict = "PENDING"
        note = "done.flag not yet present; analyst should re-run after completion"
    else:
        verdict = "PENDING"
        note = (
            "real experiment: analyst must fill success-criteria checkboxes by "
            "applying the plan's predicate to the metrics summary below"
        )

    out = run_dir / args.emit
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")

    lines: list[str] = []
    lines.append(f"# Verdict {exp_id} — {ts}")
    lines.append("")
    lines.append("## Result")
    lines.append(f"VERDICT: {verdict}")
    lines.append(f"note: {note}")
    lines.append("")
    lines.append("## Success criteria")
    lines.append("- [ ] (paste from plan; analyst marks observed values here)")
    lines.append("")
    lines.append("## Metrics summary")
    if not summaries:
        lines.append("(no metrics/*.jsonl files found)")
    else:
        for name, s in summaries.items():
            lines.append(f"### {name}.jsonl ({s.get('n_rows', 0)} rows)")
            if s.get("last"):
                lines.append("last row keys: " + ", ".join(sorted(s["last"].keys())))
            for k, v in (s.get("mean") or {}).items():
                lo = s["min"].get(k)
                hi = s["max"].get(k)
                lines.append(f"- {k}: mean={v:.6g} min={lo:.6g} max={hi:.6g}")
            lines.append("")
    lines.append("## Comparisons to baseline_run")
    lines.append("(diff_against_baseline.py output — paste if a baseline was specified)")
    lines.append("")
    lines.append("## next_actions (REVISE only)")
    lines.append("(omit unless VERDICT is REVISE)")
    lines.append("")
    lines.append("## Notes")
    lines.append(note)
    lines.append("")

    out.write_text("\n".join(lines))
    print(f"analyze: wrote {out} verdict={verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
