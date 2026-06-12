#!/usr/bin/env python3
"""EXP-30 sequence step 3: GATE-B1 / GATE-B2 evaluation (laptop-side).

Applies .claude/plans/30.md §Pre-registered gates VERBATIM — thresholds
0.10 / 2x / >=80%-of-fires / [0.1, 1.5] / <=~0.02 nats are NOT touchable here.

Statistic operationalization (from the plan, statistic only):
  - "matrix-median" = median over the 196 per-matrix cosines at one fire
    (already computed on-box into *_matrix_median fields).
  - GATE-B1 OPEN <=> median-over-post-warmup-fires of m1_matrix_median >= 0.10
    AND median(m1) >= 2 * median(m2)
    AND per-fire (m1 >= 2*m2) holds in >= 80% of post-warmup fires.
  - GATE-B2 OPEN <=> median-over-fires of m5_ratio_matrix_median in [0.1, 1.5]
    AND max loss_mismatch_nats <= 0.02.

Writes runs/EXP-30/stepA_gate.md with OPEN/CLOSED per gate + per-fire table.
"""
import json
import statistics
import sys
from pathlib import Path

RUN_DIR = Path(__file__).resolve().parent
FIRES = RUN_DIR / "metrics" / "stepA_fires.jsonl"
OUT = RUN_DIR / "stepA_gate.md"

REQUIRED = [
    "step", "tick", "warmup_fallback",
    "m1_matrix_median", "m2_matrix_median", "m3_matrix_median",
    "m4_j1", "m4_j2", "m4_j3", "m4_j4", "m4_j5",
    "m5_ratio_matrix_median", "m5_cos_matrix_median",
    "m6_matrix_median", "m7_stable_rank_median", "m7_top1pct_mass_median",
    "loss_mismatch_nats",
]


def main() -> int:
    if not FIRES.exists():
        print(f"STEPA_GATE_UNCOMPUTABLE: {FIRES} missing", file=sys.stderr)
        return 2
    fires = [json.loads(l) for l in FIRES.read_text().splitlines() if l.strip()]
    missing = [(i, k) for i, f in enumerate(fires) for k in REQUIRED if k not in f]
    if missing:
        print(f"STEPA_GATE_UNCOMPUTABLE: missing fields {missing[:10]}", file=sys.stderr)
        return 2
    post = [f for f in fires if not f.get("warmup_fallback")]
    if len(post) < 7:
        print(f"STEPA_GATE_UNCOMPUTABLE: only {len(post)} post-warmup fires (<7); "
              f"plan says extend Step A to <=25 steps", file=sys.stderr)
        return 2

    m1 = [f["m1_matrix_median"] for f in post]
    m2 = [f["m2_matrix_median"] for f in post]
    m5 = [f["m5_ratio_matrix_median"] for f in post]
    lm = [f["loss_mismatch_nats"] for f in post]
    m4j = {j: statistics.median(f[f"m4_j{j}"] for f in post) for j in range(1, 6)}

    med_m1, med_m2, med_m5 = (statistics.median(x) for x in (m1, m2, m5))
    paired_frac = sum(a >= 2 * b for a, b in zip(m1, m2)) / len(post)

    b1_open = (med_m1 >= 0.10) and (med_m1 >= 2 * med_m2) and (paired_frac >= 0.80)
    b2_open = (0.1 <= med_m5 <= 1.5) and (max(lm) <= 0.02)

    rows = "\n".join(
        f"| {f['step']} | {f['tick']} | {f['m1_matrix_median']:.4f} | {f['m2_matrix_median']:.4f} "
        f"| {f['m3_matrix_median']:.4f} | {f['m4_j4']:.4f} | {f['m4_j5']:.4f} "
        f"| {f['m5_ratio_matrix_median']:.4f} | {f['m5_cos_matrix_median']:.4f} "
        f"| {f['m6_matrix_median']:.4f} | {f['m7_stable_rank_median']:.2f} "
        f"| {f['m7_top1pct_mass_median']:.4f} | {f['loss_mismatch_nats']:.4f} |"
        for f in post
    )
    md = f"""# EXP-30 Step-A gate evaluation

source: metrics/stepA_fires.jsonl · fires total={len(fires)} post-warmup={len(post)}
rules: .claude/plans/30.md §Pre-registered gates (VERBATIM; thresholds untouched)

## GATE-B1 (blend, cell B1): **{"OPEN" if b1_open else "CLOSED"}**

- median-over-fires m1_matrix_median = {med_m1:.4f}  (>= 0.10? {med_m1 >= 0.10})
- median m2 (old-M null) = {med_m2:.4f}; median(m1) >= 2*median(m2)? {med_m1 >= 2 * med_m2}
- paired per-fire m1 >= 2*m2 fraction = {paired_frac:.2f}  (>= 0.80? {paired_frac >= 0.80})

## GATE-B2 (delayed_ef, cell B2): **{"OPEN" if b2_open else "CLOSED"}**

- median-over-fires m5_ratio_matrix_median = {med_m5:.4f}  (in [0.1, 1.5]? {0.1 <= med_m5 <= 1.5})
- max loss_mismatch_nats = {max(lm):.4f}  (<= 0.02? {max(lm) <= 0.02})

## H_decorr context (m4 lag-autocorrelation medians, post-warmup)

j=1 {m4j[1]:.4f} · j=2 {m4j[2]:.4f} · j=3 {m4j[3]:.4f} · j=4 {m4j[4]:.4f} · j=5 {m4j[5]:.4f}

## Per-fire table (post-warmup)

| step | tick | m1 | m2 | m3 | m4_j4 | m4_j5 | m5_ratio | m5_cos | m6 | m7_srank | m7_top1% | loss_mismatch |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
{rows}

(m6 at fire 2 shares the tick-5 replay pair with fire 1 — structural artifact, real cross-pair values start fire 3.)
"""
    OUT.write_text(md)
    print(md)
    print(f"GATE-B1={'OPEN' if b1_open else 'CLOSED'} GATE-B2={'OPEN' if b2_open else 'CLOSED'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
