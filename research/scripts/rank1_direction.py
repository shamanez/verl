#!/usr/bin/env python3
"""rank1_direction.py — measure how STABLE the rank-1 direction v1 is over
training, reusing the cached delta-Grams from a rank1_scorecard run.

The question this answers (for the report): "the updates are rank-1 — i.e. one
direction in weight space — but does that direction stay put, or does it drift?"
RELEX (Wei et al. 2026) claims the basis does NOT need to rotate: one shared v1
per tensor for the whole run. We test that on our own trace.

METHOD (all in the Gram domain, no d-dim vectors, no trace re-read).
For a matrix, from the cached consecutive-delta Gram D we form the base-delta
Gram B[i,j] = <theta_i - theta_0, theta_j - theta_0>. A window of W ticks ending
at anchor t_a has top-1 right singular vector
    v1(t_a) = sum_i (u1[i]/sigma1) * (theta_{w_i} - theta_0),   ||v1|| = 1,
so the alignment of two windows' directions is a pure quadratic form in B:
    cos( v1(a), v1(b) ) = sum_ij (u1^a[i]/s^a)(u1^b[j]/s^b) B[a_i, b_j].
Signs of singular vectors are arbitrary, so we report |cos| in [0,1] (1 = same
direction / perfectly stable, 0 = orthogonal / fully rotated).

We slide a fixed-width window across training and report, per matrix and pooled
per layer / block:
  - consecutive |cos| between windows 10 ticks apart  (local stability),
  - |cos| of every window's v1 against the EARLIEST window's v1  (cumulative
    drift from the initial direction),
  - the per-10-tick rotation angle in degrees.

Output: <out>/direction_stability.json. Fast (seconds) when the Gram cache for
the matching tick plan already exists.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from weight_proj import rank1_traj as R1        # noqa: E402
from weight_proj import structure as ST         # noqa: E402
import moat_scorecard as MS                      # noqa: E402


def rebuild_plan(base, anchors, wspecs, h_list, stride, n_ticks):
    pairs = [(a, R1_resolve(w, a, base, stride)) for a in anchors for w in wspecs]
    return R1.build_tick_plan(base, pairs, h_list, stride, n_ticks)


def R1_resolve(wspec, anchor, base, stride):
    if wspec == "prefix":
        return (anchor - base - 1) // stride + 1
    return int(wspec)


def v1_combo(fit) -> dict[int, float]:
    """{position: coeff on (theta_pos - theta_0)} for the unit vector v1."""
    if not fit.valid[0] or fit.sigma[0] <= 0:
        return {}
    return {p: float(fit.U[i, 0] / fit.sigma[0])
            for i, p in enumerate(fit.window_pos)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scorecard", default="runs/RANK1-ANALYSIS/scorecard",
                    help="dir holding gram_cache.npz from a rank1_scorecard run")
    ap.add_argument("--manifest",
                    default="runs/EXP-57/regimeA/weights/full_manifest.jsonl")
    ap.add_argument("--out", default="runs/RANK1-ANALYSIS")
    # the tick plan that produced the cache (defaults match the late run)
    ap.add_argument("--anchors", default="79,119")
    ap.add_argument("--window-grid", default="8,16,32,prefix")
    ap.add_argument("--h", default="1,2,5,10,20,40")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--base-tick", type=int, default=0)
    # the sliding-window probe
    ap.add_argument("--probe-window", type=int, default=16,
                    help="fixed W for the v1 stability probe")
    ap.add_argument("--probe-anchors", default="",
                    help="comma anchors to slide over; default auto every 10")
    args = ap.parse_args()

    names = MS._manifest_names(args.manifest)
    dims = MS._manifest_dims(args.manifest)
    n_ticks = MS._manifest_nticks(args.manifest)
    anchors = [int(x) for x in args.anchors.split(",")]
    wspecs = [w.strip() for w in args.window_grid.split(",")]
    h_list = [int(x) for x in args.h.split(",")]
    plan = rebuild_plan(args.base_tick, anchors, wspecs, h_list, args.stride,
                        n_ticks)
    sel = set(plan.selected)

    cache = os.path.join(args.scorecard, "gram_cache.npz")
    assert os.path.exists(cache), f"no gram cache at {cache} — run rank1_scorecard first"
    z = np.load(cache, allow_pickle=False)
    scoped = [n for n in names if n in z.files]

    # sliding anchors: every 10 ticks whose full probe window fits in the cache
    W = args.probe_window
    if args.probe_anchors:
        probe = [int(x) for x in args.probe_anchors.split(",")]
    else:
        probe = [a for a in range(args.base_tick + W, n_ticks, 10)
                 if all((a - i) in sel for i in range(W))]
    print(f"[dir] {len(scoped)} matrices; probe W={W}; "
          f"{len(probe)} sliding anchors {probe[0]}..{probe[-1]}", flush=True)
    assert len(probe) >= 3, "need >= 3 sliding anchors for a stability curve"

    base_pos = plan.pos(args.base_tick)
    # per-matrix: consecutive cosines and cos-vs-first
    consec = {}          # matrix -> list of |cos| between adjacent anchors
    vs_first = {}        # matrix -> list of |cos| vs earliest window
    unit_err = []        # sanity: ||v1||^2 deviations from 1
    for name in scoped:
        D = np.asarray(z[name], dtype=np.float64)
        B = R1.TrajGram(plan, D).base_gram(base_pos)
        combos = []
        for a in probe:
            fit = R1.fit_rank_r(B, plan, a, W, args.stride, 1)
            c = v1_combo(fit)
            combos.append(c)
            if c:
                unit_err.append(abs(R1.quad(B, c, c) - 1.0))
        cs, vf = [], []
        for k in range(len(combos)):
            if combos[k] and combos[0]:
                vf.append(abs(R1.quad(B, combos[k], combos[0])))
            if k and combos[k] and combos[k - 1]:
                cs.append(abs(R1.quad(B, combos[k], combos[k - 1])))
        consec[name] = cs
        vs_first[name] = vf

    def agg(dic, sel_names):
        # median over the selected matrices at each anchor index
        cols = None
        for n in sel_names:
            v = dic.get(n, [])
            if cols is None:
                cols = [[] for _ in v]
            for i, x in enumerate(v):
                if i < len(cols) and np.isfinite(x):
                    cols[i].append(x)
        return [float(np.median(c)) if c else float("nan") for c in (cols or [])]

    layers = sorted({ST.layer_index(n) for n in scoped
                     if ST.layer_index(n) is not None})
    result = {
        "probe_window": W, "stride": args.stride, "base_tick": args.base_tick,
        "probe_anchors": probe,
        "consecutive_gap_ticks": (probe[1] - probe[0]) if len(probe) > 1 else None,
        "unit_norm_max_err": float(max(unit_err)) if unit_err else None,
        "global": {
            "consecutive_abscos_median": agg(consec, scoped),
            "vs_first_abscos_median": agg(vs_first, scoped),
        },
        "per_layer": {}, "per_block_type": {},
    }
    for L in layers:
        ns = [n for n in scoped if ST.layer_index(n) == L]
        result["per_layer"][str(L)] = {
            "consecutive_abscos_median": agg(consec, ns),
            "vs_first_abscos_median": agg(vs_first, ns),
            "n_matrices": len(ns),
        }
    for bt in sorted({ST.classify(n)["block_type"] for n in scoped}):
        ns = [n for n in scoped if ST.classify(n)["block_type"] == bt]
        result["per_block_type"][bt] = {
            "consecutive_abscos_median": agg(consec, ns),
            "vs_first_abscos_median": agg(vs_first, ns),
            "n_matrices": len(ns),
        }
    # headline scalars
    gc = result["global"]["consecutive_abscos_median"]
    gf = result["global"]["vs_first_abscos_median"]
    gc_med = float(np.nanmedian(gc)) if gc else float("nan")
    result["headline"] = {
        "consecutive_abscos_median_over_anchors": gc_med,
        "consecutive_rotation_deg_per_gap":
            float(math.degrees(math.acos(min(1.0, gc_med)))) if np.isfinite(gc_med)
            else None,
        "vs_first_abscos_last": float(gf[-1]) if gf else None,
        "total_span_ticks": (probe[-1] - probe[0]) if len(probe) > 1 else None,
    }
    os.makedirs(args.out, exist_ok=True)
    outp = os.path.join(args.out, "direction_stability.json")
    with open(outp, "w") as f:
        json.dump(MS._clean(result), f, indent=2)
    h = result["headline"]
    print(f"[dir] unit-norm max err {result['unit_norm_max_err']:.2e}", flush=True)
    print(f"[dir] consecutive |cos| median = {h['consecutive_abscos_median_over_anchors']:.4f} "
          f"(≈ {h['consecutive_rotation_deg_per_gap']:.1f}° per "
          f"{result['consecutive_gap_ticks']} ticks)", flush=True)
    print(f"[dir] v1 vs earliest window, last anchor: |cos| = "
          f"{h['vs_first_abscos_last']:.4f} over {h['total_span_ticks']} ticks", flush=True)
    print(f"[dir] wrote {outp}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
