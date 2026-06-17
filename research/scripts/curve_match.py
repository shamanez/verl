"""Per-step training-signal CURVE MATCH between a candidate run and the dense reference.

The curve-match headline metric is
NOT an endpoint comparison — it is the per-step distance between two curves over
steps 1..N:

  - mean over steps 1..N of |metric_candidate(t) - metric_dense(t)|   (default tol 0.05)
  - final-step |metric_candidate(N) - metric_dense(N)|                 (default tol 0.05)
  - within-window slope SIGN of each (must match — candidate moves WITH dense)
  - candidate must not collapse to the pure-masked floor (reported as candidate final level)

`diff_against_baseline.py` only diffs final rows; this aligns the two curves by
training step and computes the trajectory-tracking metric the verdict needs.

Usage:
    python research/scripts/curve_match.py \
        --candidate runs/current/metrics/curvematch_anchorinject_c5_d5.jsonl \
        --dense     runs/current/metrics/curvematch_dense_ref_50step.jsonl \
        [--floor    runs/current/metrics/curvematch_spectral_baseline_c5_d5.jsonl] \
        [--metric critic/score/mean] [--max-step 50] [--tol 0.05]

Reads JSONL produced by fetch_wandb_history.py (rows carry `step` + scalar metrics).
Prints a markdown report to stdout and (with --out) writes it. Exit 0 = MATCH on the
chosen metric, 3 = NO-MATCH, 2 = error (missing data). The analyst still applies the
FULL predicate (pg_loss tracking, grad_norm finite, constraint greps) on top of this.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_series(path: Path, metric: str, max_step: int) -> dict[int, float]:
    """step -> metric value, for steps in 1..max_step where the metric is finite."""
    series: dict[int, float] = {}
    if not path.exists():
        return series
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        step = row.get("step", row.get("training/global_step", row.get("_step")))
        val = row.get(metric)
        try:
            step = int(step)
        except (TypeError, ValueError):
            continue
        if not isinstance(val, (int, float)) or val != val:  # finite only
            continue
        if 1 <= step <= max_step:
            series[step] = float(val)  # last write wins for a given step
    return series


def _slope_sign(series: dict[int, float]) -> float:
    if len(series) < 2:
        return 0.0
    steps = sorted(series)
    return series[steps[-1]] - series[steps[0]]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidate", required=True, type=Path)
    ap.add_argument("--dense", required=True, type=Path)
    ap.add_argument("--floor", type=Path, default=None, help="optional spectral-floor JSONL")
    ap.add_argument("--metric", default="critic/score/mean")
    ap.add_argument("--max-step", type=int, default=50)
    ap.add_argument("--tol", type=float, default=0.05)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    cand = _load_series(args.candidate, args.metric, args.max_step)
    dense = _load_series(args.dense, args.metric, args.max_step)
    floor = _load_series(args.floor, args.metric, args.max_step) if args.floor else {}

    if not dense:
        print(f"curve_match: no dense reference series for {args.metric} in {args.dense}", file=sys.stderr)
        return 2
    if not cand:
        print(f"curve_match: no candidate series for {args.metric} in {args.candidate}", file=sys.stderr)
        return 2

    common = sorted(set(cand) & set(dense))
    if not common:
        print("curve_match: candidate and dense share no common steps", file=sys.stderr)
        return 2

    diffs = [abs(cand[s] - dense[s]) for s in common]
    mean_abs = sum(diffs) / len(diffs)
    last = common[-1]
    final_abs = abs(cand[last] - dense[last])

    cand_slope = _slope_sign({s: cand[s] for s in common})
    dense_slope = _slope_sign({s: dense[s] for s in common})
    slope_match = (cand_slope >= 0) == (dense_slope >= 0)

    mean_ok = mean_abs <= args.tol
    final_ok = final_abs <= args.tol
    match = mean_ok and final_ok and slope_match

    L = []
    L.append(f"# Curve match — {args.metric} (steps 1..{args.max_step})")
    L.append("")
    L.append(f"- candidate: `{args.candidate.name}` ({len(cand)} steps)")
    L.append(f"- dense ref: `{args.dense.name}` ({len(dense)} steps)")
    if floor:
        L.append(f"- spectral floor: `{args.floor.name}` ({len(floor)} steps)")
    L.append(f"- common steps compared: {len(common)} (first={common[0]}, last={last})")
    L.append("")
    L.append(f"- **mean |Δ| over 1..{args.max_step}: {mean_abs:.4f}**  (tol {args.tol}) → {'OK' if mean_ok else 'FAIL'}")
    L.append(f"- **final-step |Δ| @ step {last}: {final_abs:.4f}**  (tol {args.tol}) → {'OK' if final_ok else 'FAIL'}")
    L.append(f"- slope sign: candidate Δ={cand_slope:+.4f}, dense Δ={dense_slope:+.4f} → {'MATCH' if slope_match else 'MISMATCH'}")
    L.append(f"- candidate level: first={cand[common[0]]:.4f} final={cand[last]:.4f}  | dense final={dense[last]:.4f}")
    if floor:
        fl_common = sorted(set(floor) & set(dense))
        if fl_common:
            fl_mean = sum(abs(floor[s] - dense[s]) for s in fl_common) / len(fl_common)
            L.append(f"- floor mean |Δ| vs dense: {fl_mean:.4f} (candidate must BEAT this AND reach dense tol)")
            L.append(f"- floor final level: {floor[max(fl_common)]:.4f} (pure-masked ~0.13 ⇒ collapse)")
    L.append("")
    L.append(f"## CURVE_MATCH: {'MATCH' if match else 'NO-MATCH'}")
    L.append("(headline reward criterion only — analyst must also confirm pg_loss tracking, "
             "grad_norm finite, and the constraint greps before PASS)")
    report = "\n".join(L) + "\n"

    print(report)
    if args.out:
        args.out.write_text(report)
        print(f"curve_match: wrote {args.out}")
    return 0 if match else 3


if __name__ == "__main__":
    raise SystemExit(main())
