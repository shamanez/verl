"""Gap-sensitivity sweep: forecast skill and update-direction persistence vs gap G.

This is the crux experiment. The gap=10 grid found that the dense RELEX trajectory's
10-step weight increments are nearly orthogonal (secant has no directional signal),
unlike the live compressed run (direction cosine 0.906). Here we vary the source
spacing G and ask: does directional persistence - the secant's core assumption -
recover at finer gaps? And does forecast skill follow?

For each (G, W) over every valid anchor on disk (h=1, rank1_relex, alpha=1):
  * pooled forecast skill  = 1 - sum(proj_SSE) / sum(stale_SSE)   (energy-weighted)
  * macro forecast skill   = mean over tensors of per-tensor skill
  * mean direction cosine  = mean over tensors of cos(pred_update, actual_update)
  * delta persistence      = GLOBAL cosine between the two most recent source deltas
                             D1 = theta(t-G) - theta(t-2G), D2 = theta(t) - theta(t-G),
                             computed exactly by streaming dot/norm accumulation in fp64
                             across all tensors (this is what the W=2 secant assumes
                             stays ~1). W=2 uses the single available consecutive pair.

Emits gap_sweep.csv + gap_sweep.json for the report's skill-vs-gap / cosine-vs-gap plots.

Usage:
  python gap_sweep.py --ckpt_dir <ckpts> --out_dir <outputs> \
      --gaps 1,2,3,5,10,20 --windows 2,4
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re

import harness_projector as hp
import torch
from run_forecast_ablation import CkptStore, tensor_type, whole_tensor_metrics

SAMPLE_COORDS = 250000  # match run_forecast_ablation_fast; 0 = full tensor
_IDX_CACHE = {}


def _vec(store, step, name):
    """Return (possibly coordinate-sampled) 1-D vector for a tensor, plus (numel_full,
    n_sampled). Sampling coords are fixed per tensor (deterministic) so every step of a
    tensor uses the SAME coordinates - required for meaningful deltas/cosines."""
    t = store.get(step, name).reshape(-1)
    d = t.numel()
    if SAMPLE_COORDS and d > SAMPLE_COORDS:
        if name not in _IDX_CACHE:
            seed = int(hashlib.md5(name.encode()).hexdigest()[:8], 16)
            g = torch.Generator().manual_seed(seed)
            _IDX_CACHE[name] = torch.randint(0, d, (SAMPLE_COORDS,), generator=g)
        return t[_IDX_CACHE[name]], d, SAMPLE_COORDS
    return t, d, d


def global_cos(store, names, sa, sb, sc):
    """Global cosine between D1=theta(sb)-theta(sa) and D2=theta(sc)-theta(sb),
    accumulated in fp64 over all tensors (coordinate-sampled for the large ones)."""
    dot = n1 = n2 = 0.0
    for name in names:
        a = _vec(store, sa, name)[0].to(torch.float64)
        b = _vec(store, sb, name)[0].to(torch.float64)
        c = _vec(store, sc, name)[0].to(torch.float64)
        d1 = b - a
        d2 = c - b
        dot += float(torch.sum(d1 * d2))
        n1 += float(torch.sum(d1 * d1))
        n2 += float(torch.sum(d2 * d2))
    if n1 <= 0 or n2 <= 0:
        return float("nan")
    return dot / math.sqrt(n1 * n2)


def run(store, names, *, anchor, W, gap, h=1, strength=1.0):
    src = [anchor - (W - 1 - i) * gap for i in range(W)]
    target = anchor + h * gap
    proj_sse = stale_sse = 0.0
    skills, coss = [], []
    for name in names:
        vecs = [_vec(store, s, name) for s in src]
        snaps = [v[0] for v in vecs]
        numel_full, n_sampled = vecs[0][1], vecs[0][2]
        pool_factor = numel_full / n_sampled
        latest = snaps[-1]
        if not torch.is_floating_point(latest):
            continue
        actual = _vec(store, target, name)[0]
        proj, _ = hp.project_rank1_tensor(snaps, list(src), target, strength=strength, rank=1)
        m = whole_tensor_metrics(proj, latest, actual)
        proj_sse += m["proj_sse"] * pool_factor
        stale_sse += m["stale_sse"] * pool_factor
        skills.append(m["skill"])
        coss.append(m["direction_cos"])
    pooled = (1.0 - proj_sse / stale_sse) if stale_sse > 0 else 0.0
    macro = sum(skills) / len(skills) if skills else 0.0
    macro_cos = sum(coss) / len(coss) if coss else 0.0
    # persistence: cosine between the two most recent source deltas (needs W>=2 + a point before)
    persist = float("nan")
    if W >= 2:
        sa, sb, sc = src[-2] - gap, src[-2], src[-1]  # (t-2G, t-G, t) using the two newest + one back
        if store.has_step(sa) and store.has_step(sb) and store.has_step(sc):
            persist = global_cos(store, names, sa, sb, sc)
    return dict(
        anchor=anchor,
        W=W,
        gap=gap,
        horizon=h,
        target=target,
        pooled_skill=pooled,
        macro_skill=macro,
        macro_cos=macro_cos,
        persist_cos=persist,
        stale_sse=stale_sse,
        proj_sse=proj_sse,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--gaps", default="1,2,3,5,10,20")
    ap.add_argument("--windows", default="2,4")
    ap.add_argument("--skip_embedding", action="store_true")
    ap.add_argument(
        "--only_steps", default="", help="Restrict usable steps (drop the frozen/anomalous tail, e.g. exclude step_80)."
    )
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    store = CkptStore(args.ckpt_dir)
    avail = sorted(
        int(m.group(1)) for m in (re.match(r"global_step_(\d+)$", d) for d in os.listdir(args.ckpt_dir)) if m
    )
    if args.only_steps:
        keep = {int(x) for x in args.only_steps.split(",")}
        avail = [s for s in avail if s in keep]
    avail_set = set(avail)
    print(f"available steps: {avail}")
    names = None
    for s in avail:
        names = store.names(s) if names is None else (names & store.names(s))
    if args.skip_embedding:
        names = {n for n in names if tensor_type(n) != "embed"}
    names = sorted(names)
    print(f"tracking {len(names)} tensors")

    gaps = [int(x) for x in args.gaps.split(",")]
    windows = [int(x) for x in args.windows.split(",")]
    rows = []
    for gap in gaps:
        for W in windows:
            for anchor in avail:
                src = [anchor - (W - 1 - i) * gap for i in range(W)]
                target = anchor + gap
                need = set(src) | {target}
                if not need <= avail_set:
                    continue
                r = run(store, names, anchor=anchor, W=W, gap=gap)
                rows.append(r)
                print(
                    f"gap={gap:>2} W={W} anchor={anchor:>3} tgt={target:>3} -> "
                    f"pooled={r['pooled_skill']:+.3f} macro_cos={r['macro_cos']:+.3f} "
                    f"persist={r['persist_cos']:+.3f}"
                )
    # aggregate per (gap,W) over anchors
    import statistics as stt
    from collections import defaultdict

    agg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        k = (r["gap"], r["W"])
        for f in ("pooled_skill", "macro_skill", "macro_cos", "persist_cos"):
            v = r[f]
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                agg[k][f].append(v)
    summary = []
    for (gap, W), d in sorted(agg.items()):
        row = dict(gap=gap, W=W, n_anchors=len(d["pooled_skill"]))
        for f in ("pooled_skill", "macro_skill", "macro_cos", "persist_cos"):
            vals = d[f]
            row[f] = stt.mean(vals) if vals else None
            row[f + "_std"] = stt.pstdev(vals) if len(vals) > 1 else 0.0
        summary.append(row)
        print(
            f"  [agg] gap={gap:>2} W={W} n={row['n_anchors']} pooled={row['pooled_skill']} persist={row['persist_cos']}"
        )

    with open(os.path.join(args.out_dir, "gap_sweep.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    json.dump(
        {"rows": rows, "summary": summary, "available_steps": avail},
        open(os.path.join(args.out_dir, "gap_sweep.json"), "w"),
        indent=2,
    )
    print(f"\nwrote {len(rows)} rows + {len(summary)} (gap,W) aggregates -> {args.out_dir}")


if __name__ == "__main__":
    main()
