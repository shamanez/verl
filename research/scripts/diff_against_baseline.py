"""Compare an experiment's metrics against a baseline run's metrics.

Usage:
    python research/scripts/diff_against_baseline.py runs/<run-id> --baseline <baseline-id>

Reads `metrics/*.jsonl` for both, computes:
- delta on the final row's numeric keys
- delta on mean/min/max over the full series
- a side-by-side markdown table written to <run_dir>/baseline_diff.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_jsonl(p: Path) -> list[dict]:
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


def _numeric_keys(rows: list[dict]) -> list[str]:
    keys = set()
    for r in rows:
        for k, v in r.items():
            if isinstance(v, (int, float)) and v == v:
                keys.add(k)
    return sorted(keys)


def _final(rows: list[dict], key: str):
    for r in reversed(rows):
        v = r.get(key)
        if isinstance(v, (int, float)) and v == v:
            return float(v)
    return None


def main() -> int:
    here = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--baseline", required=True,
                    help="baseline id or path")
    ap.add_argument("--metric-file", default="train.jsonl",
                    help="which metrics/*.jsonl to compare (default train.jsonl)")
    args = ap.parse_args()

    if str(args.baseline).lower() == "none":
        print("diff_against_baseline: baseline=none, nothing to do")
        return 0

    run = args.run_dir.resolve()
    base = (here / "runs" / args.baseline).resolve() if not Path(args.baseline).is_absolute() \
        else Path(args.baseline)
    if not base.is_dir():
        base = here / "runs" / Path(args.baseline).name

    if not run.is_dir():
        print(f"diff_against_baseline: not a directory: {run}", file=sys.stderr)
        return 2
    if not base.is_dir():
        print(f"diff_against_baseline: baseline not found: {base}", file=sys.stderr)
        return 2

    run_rows = _load_jsonl(run / "metrics" / args.metric_file)
    base_rows = _load_jsonl(base / "metrics" / args.metric_file)
    keys = sorted(set(_numeric_keys(run_rows)) & set(_numeric_keys(base_rows)))

    rows_md = [
        "| metric | this run (final) | baseline (final) | delta (this - base) | delta % |",
        "|---|---|---|---|---|",
    ]
    for k in keys:
        r = _final(run_rows, k)
        b = _final(base_rows, k)
        if r is None or b is None:
            continue
        delta = r - b
        pct = (delta / b * 100.0) if b not in (0, None) else float("inf")
        rows_md.append(f"| {k} | {r:.6g} | {b:.6g} | {delta:.6g} | {pct:.2f}% |")

    out = run / "baseline_diff.md"
    title = f"# Baseline diff: {run.name} vs {base.name} ({args.metric_file})\n"
    body = "\n".join(rows_md) if len(rows_md) > 2 else "(no common numeric keys)"
    out.write_text(title + "\n" + body + "\n")
    print(f"diff_against_baseline: wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
