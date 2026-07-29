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

"""Sum spend across the runs.jsonl ledger.

Default: prints a JSON summary of {currently_running_dph, today_spent, month_spent, lifetime_spent}.

Usage:
    python research/scripts/check_budget.py                       # all-time summary
    python research/scripts/check_budget.py --month               # restrict to current month
    python research/scripts/check_budget.py runs/<run-id>         # one run only
    python research/scripts/check_budget.py --cap-check           # exit 2 if monthly cap exceeded

Reads `.claude/state/runs.jsonl` (one JSON object per line). Each row carries:
    id, handles, started_at_epoch, dph (sum across handles), max_gpu_hr,
    per_node_gpus, status, [torn_down_at, teardown_reason]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def _parse_iso(s: str) -> float:
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
    except Exception:
        return 0.0


def _load_ledger(p: Path) -> list[dict]:
    if not p.exists():
        return []
    rows = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _load_budget(p: Path) -> dict:
    """Caps live in project.yaml's `budget:` block (budget.json was retired).

    Parsed with a dumb line scanner so we don't need a yaml dependency; falls
    back to a legacy budget.json path if one is explicitly passed and exists.
    """
    if p.suffix == ".json" and p.exists():
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError:
            return {"monthly_cap_usd": None}
    if not p.exists():
        return {"monthly_cap_usd": None}
    caps, in_budget = {}, False
    for line in p.read_text().splitlines():
        if line.startswith("budget:"):
            in_budget = True
            continue
        if in_budget:
            if line and not line.startswith((" ", "\t", "#")):
                break
            stripped = line.strip()
            for key in ("monthly_cap_usd", "weekly_cap_usd"):
                if stripped.startswith(f"{key}:"):
                    val = stripped.split(":", 1)[1].split("#")[0].strip()
                    try:
                        caps[key] = float(val)
                    except ValueError:
                        pass
    return caps or {"monthly_cap_usd": None}


def _row_spent_usd(row: dict, now_epoch: float) -> float:
    dph = float(row.get("dph", 0) or 0)
    started = float(row.get("started_at_epoch") or _parse_iso(row.get("started_at", "")))
    if started <= 0 or dph <= 0:
        return 0.0
    if row.get("status") == "TORN_DOWN":
        ended = float(_parse_iso(row.get("torn_down_at", ""))) or now_epoch
    else:
        ended = now_epoch
    hours = max(0.0, (ended - started) / 3600.0)
    return dph * hours


def main() -> int:
    here = Path(__file__).resolve().parent.parent
    default_ledger = here / ".claude" / "state" / "runs.jsonl"
    default_budget = here / ".claude" / "project.yaml"

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("scope", nargs="?", default=None, help="optional single run dir to inspect")
    ap.add_argument("--ledger", type=Path, default=default_ledger)
    ap.add_argument("--budget", type=Path, default=default_budget)
    ap.add_argument("--month", action="store_true", help="restrict accounting to the current calendar month")
    ap.add_argument(
        "--cap-check", action="store_true", help="exit 2 if monthly_cap_usd in project.yaml budget: is exceeded"
    )
    args = ap.parse_args()

    rows = _load_ledger(args.ledger)
    if args.scope:
        target_id = "".join(ch for ch in Path(args.scope).name if ch.isdigit())
        rows = [r for r in rows if "".join(ch for ch in r.get("id", "") if ch.isdigit()) == target_id]

    now = time.time()
    month_start = datetime.now(timezone.utc).replace(day=1, hour=0, minute=0, second=0, microsecond=0).timestamp()

    running_dph = sum(float(r.get("dph", 0) or 0) for r in rows if r.get("status") == "RUNNING")
    lifetime = sum(_row_spent_usd(r, now) for r in rows)
    this_month = sum(
        _row_spent_usd(r, now)
        for r in rows
        if float(r.get("started_at_epoch") or _parse_iso(r.get("started_at", ""))) >= month_start
    )

    summary = {
        "running_count": sum(1 for r in rows if r.get("status") == "RUNNING"),
        "running_dph": round(running_dph, 4),
        "lifetime_spent_usd": round(lifetime, 4),
        "month_spent_usd": round(this_month, 4),
        "monthly_cap_usd": _load_budget(args.budget).get("monthly_cap_usd"),
    }

    if args.month:
        summary["scope"] = "month"
    elif args.scope:
        summary["scope"] = args.scope

    print(json.dumps(summary, indent=2))

    if args.cap_check:
        cap = summary["monthly_cap_usd"]
        if cap is not None and summary["month_spent_usd"] > cap:
            print(
                f"check_budget: month spend ${summary['month_spent_usd']:.2f} exceeds cap ${cap:.2f}", file=sys.stderr
            )
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
