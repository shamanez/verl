#!/usr/bin/env python3
"""rank1_scorecard.py — OFFLINE scorecard for the RELEX-style rank-1
trajectory-SVD predictor family (Wei et al. 2026) on a full-weight trace.

A NEW, SEPARATE lane beside moat_scorecard.py: this file IMPORTS the existing
harness (metric contract, mmap trace reader, chunk planner, taxonomy) and
modifies none of it. The online comm-eff path (verl/workers/comm_eff/*) is not
involved at all. Family math lives in weight_proj/rank1_traj.py.

WHAT ONE RUN PRODUCES. For every scoped matrix and every
(window_spec × anchor × rank × h) cell:
  - rank{r}_traj rows      θ̂ = θ_base + Σ_j ĉ_j(T)·v_j   (the new family)
  - two_point_window rows  Paper-A Weight Extrapolation over the SAME window span
  - naive_last2 rows       CORE naive_linear at Δ = stride (2 most recent ticks)
  - hold_stale rows        do-nothing reference (ratio == 1 gate)
plus block/super-block/global aggregate rows (exact concatenated-vector
semantics via additive moments) and a summary.json + stdout table answering the
two design questions: how many delta checkpoints does the SVD need (window
axis), and how far ahead does it stay skillful (h axis).

METRIC-CONTRACT PIN. Every ratio/skill/dir_cos/radial/tangential is produced by
metrics.full_metric_row via moat_scorecard.surrogate_metric_row on the window's
EXACT (‖e‖², ‖b‖², ⟨e,b⟩) — the same one-true-code-path discipline as the MOAT
scorecard, with the same baseline convention (b = θ_anchor − θ_truth, so
hold-stale scores exactly 1 and rows are directly comparable to CORE-4 rows at
equal h).

STREAMING. One pass over the union of needed ticks per chunk of matrices
(row-sharded to --cap-elems): per unit, float64 consecutive-tick delta rows are
held for the selected ticks only and reduced to a full (S−1)×(S−1) delta Gram
by one BLAS syrk per unit; per-matrix Grams are cached (gram_cache.npz) so
re-sweeps with the same tick plan skip the trace entirely. Everything after
the Gram is small-matrix math (see rank1_traj.py docstring for the identity).

REAL-DATA AUDIT (--audit N, on by default). After the sweep, N sampled
(matrix, window, anchor, h) cells are recomputed the DIRECT way — raw snapshot
vectors loaded from the trace, explicit numpy SVD + linear fit + prediction,
metrics straight from metrics.full_metric_row on real d-dim vectors — and
asserted to match the Gram-path rows within 1e-6 relative. The same battery
(plus synthetic exactness/noise/curvature/reconstruction/guard tests) runs
trace-free under --self-test.

Usage (on the analysis box):
  python3 rank1_scorecard.py --self-test
  python3 rank1_scorecard.py \
      --trace-root /workspace/trace/EXP-57 \
      --manifest runs/EXP-57/regimeA/weights/full_manifest.jsonl \
      --out runs/RANK1-ANALYSIS/scorecard \
      --scope panel --anchors 79,119 --window-grid 8,16,32,prefix \
      --h 1,2,5,10,20,40 --rank-grid 1,2
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from weight_proj import metrics as M                  # noqa: E402
from weight_proj import rank1_traj as R1              # noqa: E402
from weight_proj import structure as ST               # noqa: E402
import moat_scorecard as MS                           # noqa: E402  (reused, unmodified)

ROW_SCHEMA = "rank1-scorecard-rows-v1"
METRIC_CONTRACT_EXPECTED = "weight-proj-metrics-v1"
RANK1_CONTRACT_EXPECTED = "weight-proj-rank1-traj-v1"
PARITY_RTOL = 1e-6                # Gram-path vs direct-tensor-path agreement
PANEL_LAYERS = (0, 7, 13, 20, 27)  # representative depth slice for --scope panel


def log(msg: str) -> None:
    print(f"[rank1] {time.strftime('%H:%M:%S')} {msg}", flush=True)


# =============================================================================
# Scope resolution (which matrices participate)
# =============================================================================
def resolve_scope(names: list[str], scope: str) -> list[str]:
    """panel: PANEL_LAYERS' full per-layer sets + final norm (embed excluded);
    all: every manifest matrix; no-embed: all minus embed; regex:<pat>: filter."""
    if scope == "all":
        return list(names)
    if scope == "no-embed":
        return [n for n in names if n != "model.embed_tokens.weight"]
    if scope == "panel":
        keep = []
        for n in names:
            li = ST.layer_index(n)
            if li is not None and li in PANEL_LAYERS:
                keep.append(n)
            elif n == "model.norm.weight":
                keep.append(n)
        return keep
    if scope.startswith("regex:"):
        pat = re.compile(scope[len("regex:"):])
        return [n for n in names if pat.search(n)]
    raise SystemExit(f"unknown --scope {scope!r}")


def resolve_window(wspec: str, anchor: int, base_tick: int, stride: int) -> int:
    """'prefix' -> every stride-th tick back from the anchor down to base+1
    (the paper-faithful full-observed-prefix window); else the integer W."""
    if wspec == "prefix":
        w = (anchor - base_tick - 1) // stride + 1
    else:
        w = int(wspec)
    assert w >= 2, f"window {wspec} at anchor {anchor} resolves to {w} < 2"
    return w


# =============================================================================
# Streaming Gram accumulation (the only part that touches the trace)
# =============================================================================
def _fingerprint(trace_root: str, selected: list[int], name_dims: list[tuple[str, int]],
                 dtype: str) -> str:
    blob = json.dumps({"root": os.path.realpath(trace_root), "ticks": selected,
                       "nd": sorted(name_dims), "dtype": dtype,
                       "contract": R1.RANK1_CONTRACT}, sort_keys=True)
    return hashlib.md5(blob.encode()).hexdigest()


def accumulate_grams(reader, name_dims: list[tuple[str, int]], selected: list[int],
                     cap_elems: int) -> dict[str, np.ndarray]:
    """One streaming pass; returns {matrix_name: (S−1)×(S−1) float64 delta Gram}.

    Chunks matrices via moat_scorecard.plan_chunks (row-sharding oversized ones);
    Grams are additive over shards, so shard syrk results sum into the matrix
    Gram exactly. Deltas are differenced in float64 from the raw fp32 snapshots.
    """
    S = len(selected)
    grams = {name: np.zeros((S - 1, S - 1), dtype=np.float64)
             for name, _ in name_dims}
    chunks = MS.plan_chunks(name_dims, cap_elems)
    total_units = sum(len(c) for c in chunks)
    log(f"streaming {len(name_dims)} matrices as {total_units} units in "
        f"{len(chunks)} chunks over {S} ticks (cap {cap_elems:,} elems)")
    t0 = time.time()
    for ci, chunk in enumerate(chunks):
        rows = {id(u): np.empty((S - 1, u.n), dtype=np.float64) for u in chunk}
        prev: dict[int, np.ndarray] = {}
        for k, tick in enumerate(selected):
            sd = reader.load_raw(tick)
            for u in chunk:
                cur = reader.slice_f64(sd, u.name, u.a, u.b)
                if k:
                    np.subtract(cur, prev[id(u)], out=rows[id(u)][k - 1])
                prev[id(u)] = cur
            del sd
        for u in chunk:
            Dm = rows.pop(id(u))
            grams[u.name] += Dm @ Dm.T
        el = time.time() - t0
        log(f"  chunk {ci + 1}/{len(chunks)} done ({el:.0f}s elapsed, "
            f"ETA {el / (ci + 1) * (len(chunks) - ci - 1):.0f}s)")
    return grams


def load_or_build_grams(args, reader, name_dims, selected, dtype):
    cache_path = os.path.join(args.out, "gram_cache.npz")
    fp = _fingerprint(args.trace_root, selected, name_dims, dtype)
    if os.path.exists(cache_path) and not args.force_recompute:
        z = np.load(cache_path, allow_pickle=False)
        if str(z.get("__fp__")) == fp:
            log(f"gram cache HIT ({cache_path})")
            return {name: z[name] for name, _ in name_dims}
        log("gram cache STALE (fingerprint mismatch) — recomputing")
    grams = accumulate_grams(reader, name_dims, selected, args.cap_elems)
    os.makedirs(args.out, exist_ok=True)
    np.savez_compressed(cache_path, __fp__=np.str_(fp), **grams)
    log(f"gram cache written ({cache_path})")
    return grams


# =============================================================================
# Per-matrix sweep — one code path shared by emit, self-test and audit
# =============================================================================
def matrix_cells(D: np.ndarray, plan: R1.TickPlan, base_tick: int,
                 anchors: list[int], window_specs: list[str], h_grid: list[int],
                 stride: int, rank_grid: list[int]):
    """Yield raw scoring cells for ONE matrix: dicts with identity + moments +
    fit diagnostics. Metrics are attached later (surrogate_metric_row) so group
    aggregation can sum the SAME moments the per-matrix rows are scored from."""
    tg = R1.TrajGram(plan, D)
    B = tg.base_gram(plan.pos(base_tick))
    for anchor in anchors:
        a_pos = plan.pos(anchor)
        # ---- window-free baselines: hold_stale + naive_last2 (Δ = stride) ----
        prev_tick = anchor - stride
        p_pos = plan.pos(prev_tick)
        for h in h_grid:
            t_pos = plan.pos(anchor + h)
            for method, pred in (
                    ("hold_stale", R1.hold_stale_pred(a_pos)),
                    ("naive_last2", R1.two_anchor_pred(a_pos, anchor, p_pos,
                                                       prev_tick, anchor + h))):
                e2, b2, eb = R1.family_moments(B, pred, a_pos, t_pos)
                yield {"method": method, "rank": None, "window_spec": None,
                       "window": None, "window_span": (stride if method ==
                                                       "naive_last2" else None),
                       "anchor_tick": anchor, "h_ticks": h,
                       "target_tick": anchor + h,
                       "e2": e2, "b2": b2, "eb": eb}
        # ---- windowed families ----
        for wspec in window_specs:
            w = resolve_window(wspec, anchor, base_tick, stride)
            span = (w - 1) * stride
            first_tick = anchor - span
            f_pos = plan.pos(first_tick)
            fits = {r: R1.fit_rank_r(B, plan, anchor, w, stride, r)
                    for r in rank_grid}
            for h in h_grid:
                target = anchor + h
                t_pos = plan.pos(target)
                pred2 = R1.two_anchor_pred(a_pos, anchor, f_pos, first_tick, target)
                e2, b2, eb = R1.family_moments(B, pred2, a_pos, t_pos)
                yield {"method": "two_point_window", "rank": None,
                       "window_spec": wspec, "window": w, "window_span": span,
                       "anchor_tick": anchor, "h_ticks": h, "target_tick": target,
                       "e2": e2, "b2": b2, "eb": eb}
                for r in rank_grid:
                    fit = fits[r]
                    pred = R1.rank1_pred(fit, target)
                    e2, b2, eb = R1.family_moments(B, pred, a_pos, t_pos)
                    used = int(fit.valid.sum())
                    yield {"method": f"rank{r}_traj", "rank": r,
                           "window_spec": wspec, "window": w, "window_span": span,
                           "anchor_tick": anchor, "h_ticks": h,
                           "target_tick": target,
                           "e2": e2, "b2": b2, "eb": eb,
                           "sigma1": float(fit.sigma[0]),
                           "evr1": float(fit.evr[0]),
                           "evr_sum_r": float(np.nansum(fit.evr[:used])) if used
                           else float("nan"),
                           "coef_r2_1": float(fit.coef_r2[0]),
                           "coef_r2_min": (float(np.nanmin(fit.coef_r2[:used]))
                                           if used else float("nan")),
                           "slope1": float(fit.slope[0]),
                           "intercept1": float(fit.intercept[0]),
                           "n_comp_valid": used}


def cell_metrics(cell: dict) -> dict:
    """Attach the contract metrics to a cell's moments (one true code path)."""
    met = MS.surrogate_metric_row(cell["e2"], cell["b2"], cell["eb"])
    span = cell.get("window_span")
    h = cell["h_ticks"]
    out = dict(cell)
    out.update({k: met[k] for k in ("err_norm", "base_norm", "weight_proj_ratio",
                                    "dir_cos", "radial", "tangential", "skill")})
    out["extrap_factor"] = (float(h) / span) if span else None
    return out


# =============================================================================
# Emit (real-trace run)
# =============================================================================
def run_emit(args) -> int:
    names = MS._manifest_names(args.manifest)
    dims = MS._manifest_dims(args.manifest)
    n_ticks = MS._manifest_nticks(args.manifest)
    dtype = MS._manifest_dtype(args.manifest)
    scoped = resolve_scope(names, args.scope)
    assert scoped, "scope matched no matrices"
    name_dims = [(n, dims[n]) for n in scoped]
    n_elems_total = sum(d for _, d in name_dims)
    log(f"scope={args.scope}: {len(scoped)} matrices, {n_elems_total:,} elems; "
        f"trace n_ticks={n_ticks} dtype={dtype}")

    plan = R1.build_tick_plan(args.base_tick, args.anchor_list, args.window_ints,
                              args.h_list, args.stride, n_ticks)
    log(f"tick plan: {plan.n} selected ticks "
        f"({plan.selected[0]}..{plan.selected[-1]}); windows={args.window_specs} "
        f"anchors={args.anchor_list} h={args.h_list} ranks={args.rank_grid} "
        f"stride={args.stride} base={args.base_tick}")

    reader = MS.MmapTraceReader(args.trace_root)
    missing = [t for t in plan.selected
               if not os.path.exists(reader.path(t))]
    assert not missing, f"trace missing ticks: {missing[:8]}"

    grams = load_or_build_grams(args, reader, name_dims, plan.selected, dtype)

    # ---- per-matrix rows + additive group moments -----------------------------
    rows: list[dict] = []
    agg: dict[tuple, dict] = {}       # (method,rank,wspec,anchor,h,kind,key) -> moments
    diag_pool: dict[tuple, dict] = {} # same key -> lists of per-matrix diagnostics

    def _agg(cell: dict, kind: str, key: str, n_elems: int):
        k = (cell["method"], cell["rank"], cell["window_spec"],
             cell["anchor_tick"], cell["h_ticks"], kind, key)
        a = agg.setdefault(k, {"e2": 0.0, "b2": 0.0, "eb": 0.0, "n_members": 0,
                               "n_elems": 0, "window": cell["window"],
                               "window_span": cell["window_span"],
                               "target_tick": cell["target_tick"]})
        a["e2"] += cell["e2"]; a["b2"] += cell["b2"]; a["eb"] += cell["eb"]
        a["n_members"] += 1; a["n_elems"] += n_elems
        if cell["method"].endswith("_traj"):
            d = diag_pool.setdefault(k, {"coef_r2_1": [], "evr1": []})
            d["coef_r2_1"].append(cell.get("coef_r2_1"))
            d["evr1"].append(cell.get("evr1"))

    t0 = time.time()
    for i, (name, d) in enumerate(name_dims):
        cls = ST.classify(name)
        for cell in matrix_cells(grams[name], plan, args.base_tick,
                                 args.anchor_list, args.window_specs,
                                 args.h_list, args.stride, args.rank_grid):
            row = cell_metrics(cell)
            row.update(cls)
            row.update({"schema": ROW_SCHEMA, "group_kind": "matrix",
                        "group_key": name, "n_elems": d,
                        "stride": args.stride, "base_tick": args.base_tick})
            rows.append(row)
            _agg(cell, "global", "all", d)
            _agg(cell, "super_block", cls["super_block"], d)
            _agg(cell, "block_type", cls["block_type"], d)
        if (i + 1) % 20 == 0 or i + 1 == len(name_dims):
            log(f"  swept {i + 1}/{len(name_dims)} matrices "
                f"({time.time() - t0:.0f}s)")

    for k, a in agg.items():
        method, rank, wspec, anchor, h, kind, key = k
        met = MS.surrogate_metric_row(a["e2"], a["b2"], a["eb"])
        row = {"schema": ROW_SCHEMA, "method": method, "rank": rank,
               "window_spec": wspec, "window": a["window"],
               "window_span": a["window_span"], "anchor_tick": anchor,
               "h_ticks": h, "target_tick": a["target_tick"],
               "group_kind": kind, "group_key": key,
               "matrix_name": None, "layer_idx": None, "block_type": None,
               "super_block": None, "special": None,
               "n_elems": a["n_elems"], "n_members": a["n_members"],
               "stride": args.stride, "base_tick": args.base_tick,
               "e2": a["e2"], "b2": a["b2"], "eb": a["eb"],
               "extrap_factor": (float(h) / a["window_span"]
                                 if a["window_span"] else None)}
        row.update({kk: met[kk] for kk in ("err_norm", "base_norm",
                                           "weight_proj_ratio", "dir_cos",
                                           "radial", "tangential", "skill")})
        dp = diag_pool.get(k)
        if dp:
            r2v = np.array([v for v in dp["coef_r2_1"] if v is not None],
                           dtype=np.float64)
            evrv = np.array([v for v in dp["evr1"] if v is not None],
                            dtype=np.float64)
            row["coef_r2_1_median"] = (float(np.nanmedian(r2v)) if r2v.size
                                       else None)
            row["evr1_median"] = float(np.nanmedian(evrv)) if evrv.size else None
        rows.append(row)

    # ---- hold_stale identity gate --------------------------------------------
    hs = [r for r in rows if r["method"] == "hold_stale"
          and np.isfinite(r.get("weight_proj_ratio") or float("nan"))]
    worst = max((abs(r["weight_proj_ratio"] - 1.0) for r in hs), default=0.0)
    assert worst <= 1e-9, f"hold_stale identity violated: |ratio-1| max {worst}"
    log(f"hold_stale identity gate PASS (worst |ratio-1| = {worst:.2e} over "
        f"{len(hs)} rows)")

    # ---- write artifacts ------------------------------------------------------
    os.makedirs(args.out, exist_ok=True)
    rows_path = os.path.join(args.out, "rows.jsonl")
    with open(rows_path, "w") as f:
        for r in rows:
            f.write(json.dumps(MS._clean(r)) + "\n")
    summary = build_summary(args, rows, plan)
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(MS._clean(summary), f, indent=2)
    log(f"wrote {len(rows)} rows -> {rows_path}")
    print_summary(summary)

    # ---- real-data audit (direct-tensor parity on sampled cells) -------------
    if args.audit > 0:
        n_bad = run_audit(args, reader, plan, grams, name_dims)
        if n_bad:
            log(f"AUDIT FAIL: {n_bad} cells exceeded parity tolerance")
            return 1
    return 0


# =============================================================================
# Summary
# =============================================================================
def _median(vals) -> float:
    v = [x for x in vals if x is not None and np.isfinite(x)]
    return float(np.median(v)) if v else float("nan")


def build_summary(args, rows: list[dict], plan) -> dict:
    glob = [r for r in rows if r["group_kind"] == "global"]
    table = {}
    for r in glob:
        key = (r["method"], r["window_spec"] or "-", r["h_ticks"])
        table.setdefault(key, []).append(r["weight_proj_ratio"])
    cells = [{"method": m, "window_spec": w, "h_ticks": h,
              "ratio_median_over_anchors": _median(v)}
             for (m, w, h), v in sorted(table.items(),
                                        key=lambda kv: (kv[0][0], str(kv[0][1]),
                                                        kv[0][2]))]
    mat = [r for r in rows if r["group_kind"] == "matrix"
           and r["method"].endswith("_traj") and r["rank"] == 1]
    return {
        "schema": ROW_SCHEMA,
        "config": {"trace_root": args.trace_root, "scope": args.scope,
                   "anchors": args.anchor_list, "window_grid": args.window_specs,
                   "h_grid": args.h_list, "rank_grid": args.rank_grid,
                   "stride": args.stride, "base_tick": args.base_tick,
                   "n_selected_ticks": plan.n},
        "paper_replication": {
            "coef_r2_1_median": _median(r.get("coef_r2_1") for r in mat),
            "coef_r2_1_frac_gt_0.98": (float(np.mean([r["coef_r2_1"] > 0.98
                                       for r in mat if r.get("coef_r2_1") is not None
                                       and np.isfinite(r["coef_r2_1"])]))
                                       if mat else float("nan")),
            "evr1_median": _median(r.get("evr1") for r in mat),
        },
        "global_ratio_table": cells,
    }


def print_summary(summary: dict) -> None:
    pr = summary["paper_replication"]
    print("\n=== rank-1 trajectory family — paper-replication diagnostics ===")
    print(f"  coef-linearity R2 (rank-1, per-matrix median): "
          f"{pr['coef_r2_1_median']:.4f}   (paper claims > 0.98)")
    print(f"  frac matrices with coef R2 > 0.98:              "
          f"{pr['coef_r2_1_frac_gt_0.98']:.2%}")
    print(f"  rank-1 energy share EVR1 (median):              "
          f"{pr['evr1_median']:.2%}   (paper: ~81% of a rank-5 window)")
    print("\n=== global pooled weight_proj_ratio (median over anchors; <1 = beats hold-stale) ===")
    by_mw: dict[tuple, dict[int, float]] = {}
    hs_all = sorted({c["h_ticks"] for c in summary["global_ratio_table"]})
    for c in summary["global_ratio_table"]:
        by_mw.setdefault((c["method"], c["window_spec"]), {})[c["h_ticks"]] = \
            c["ratio_median_over_anchors"]
    head = "method/window".ljust(28) + "".join(f"h={h}".rjust(9) for h in hs_all)
    print(head)
    print("-" * len(head))
    for (m, w), hv in sorted(by_mw.items()):
        lab = f"{m}[{w}]".ljust(28)
        line = "".join(
            (f"{hv[h]:9.4f}" if h in hv and hv[h] is not None
             and np.isfinite(hv[h]) else "        -") for h in hs_all)
        print(lab + line)
    print()


# =============================================================================
# Real-data audit — direct-tensor recompute of sampled cells
# =============================================================================
def run_audit(args, reader, plan, grams, name_dims) -> int:
    """Recompute up to --audit cells the DIRECT way (raw vectors, numpy SVD,
    metrics.full_metric_row on real tensors) and compare with the Gram path."""
    small = sorted(name_dims, key=lambda nd: nd[1])
    picks = []
    finite_w = [w for w in args.window_specs if w != "prefix"] or ["prefix"]
    for name, d in small:
        if d < 4:                      # skip degenerate slivers
            continue
        picks.append(name)
        if len(picks) >= args.audit:
            break
    n_bad = 0
    wspec = finite_w[0]
    anchor = args.anchor_list[0]
    h = args.h_list[len(args.h_list) // 2]
    rank = args.rank_grid[0]
    w = resolve_window(wspec, anchor, args.base_tick, args.stride)
    window_ticks = [anchor - i * args.stride for i in range(w)][::-1]
    need = sorted(set(window_ticks) | {args.base_tick, anchor, anchor + h})
    log(f"audit: {len(picks)} matrices x cell (W={wspec}->{w}, anchor={anchor}, "
        f"h={h}, rank={rank}) recomputed from raw snapshots")
    for name in picks:
        thetas = {t: reader.load_matrix_f64(t, name) for t in need}
        pred, diag = R1.direct_fit_predict(thetas, args.base_tick, window_ticks,
                                           anchor + h, rank)
        met_direct = M.full_metric_row(pred, thetas[anchor + h], thetas[anchor],
                                       None)
        # Gram-path cell for the same coordinates
        cell = next(c for c in matrix_cells(
            grams[name], plan, args.base_tick, [anchor], [wspec], [h],
            args.stride, [rank]) if c["method"] == f"rank{rank}_traj"
            and c["h_ticks"] == h)
        met_gram = cell_metrics(cell)
        ok = True
        for kk in ("weight_proj_ratio", "dir_cos", "skill"):
            a, b = met_direct[kk], met_gram[kk]
            if np.isfinite(a) and np.isfinite(b):
                rel = abs(a - b) / max(abs(a), abs(b), 1e-30)
                if rel > PARITY_RTOL:
                    ok = False
                    log(f"  AUDIT MISMATCH {name} {kk}: direct={a:.9g} "
                        f"gram={b:.9g} rel={rel:.2e}")
        # reconstruction invariant: snapshot-coeff replay == direct prediction
        B = R1.TrajGram(plan, grams[name]).base_gram(plan.pos(args.base_tick))
        fit = R1.fit_rank_r(B, plan, anchor, w, args.stride, rank)
        pos_list, gam, base_coef = fit.snapshot_coeffs(anchor + h)
        tick_of = {plan.pos(t): t for t in need}
        recon = base_coef * thetas[args.base_tick]
        for p, g in zip(pos_list, gam):
            recon = recon + g * thetas[tick_of[p]]
        rel = (np.linalg.norm(recon - pred)
               / max(np.linalg.norm(pred - thetas[args.base_tick]), 1e-30))
        if rel > 1e-6:
            ok = False
            log(f"  AUDIT RECONSTRUCTION MISMATCH {name}: rel={rel:.2e}")
        if ok:
            log(f"  audit PASS {name} (ratio direct={met_direct['weight_proj_ratio']:.6f} "
                f"gram={met_gram['weight_proj_ratio']:.6f}, evr1={diag['evr'][0]:.3f}, "
                f"coefR2={diag['coef_r2'][0]:.4f})")
        else:
            n_bad += 1
    return n_bad


# =============================================================================
# Self-test battery (trace-free, synthetic)
# =============================================================================
def _mk_plan(n_ticks, base, anchors, wspecs, hs, stride):
    wints = sorted({resolve_window(w, a, base, stride)
                    for w in wspecs for a in anchors})
    return R1.build_tick_plan(base, anchors, wints, hs, stride, n_ticks)


def _cells_for(thetas, n_ticks, base, anchors, wspecs, hs, stride, ranks):
    plan = _mk_plan(n_ticks, base, anchors, wspecs, hs, stride)
    D = R1.gram_from_snapshots(thetas, plan.selected)
    cells = [cell_metrics(c) for c in matrix_cells(
        D, plan, base, anchors, wspecs, hs, stride, ranks)]
    return plan, cells


def _get(cells, method, h, wspec="__any__"):
    for c in cells:
        if c["method"] == method and c["h_ticks"] == h and \
                (wspec == "__any__" or c["window_spec"] == wspec):
            return c
    raise KeyError((method, h, wspec))


def run_selftest() -> int:
    failures: list[str] = []

    def check(name: str, ok: bool, detail: str = ""):
        status = "PASS" if ok else "FAIL"
        log(f"  [{status}] {name}{(' — ' + detail) if detail else ''}")
        if not ok:
            failures.append(name)

    rng = np.random.default_rng(45)
    d, n_ticks = 512, 60
    v1 = rng.standard_normal(d); v1 /= np.linalg.norm(v1)
    v2 = rng.standard_normal(d); v2 -= v2 @ v1 * v1; v2 /= np.linalg.norm(v2)
    theta0 = rng.standard_normal(d)

    # ---- T1: exact rank-1 linear trajectory -> everything is exact -----------
    thetas = {t: theta0 + (0.03 * t) * v1 for t in range(n_ticks)}
    _, cells = _cells_for(thetas, n_ticks, 0, [40], ["8", "prefix"],
                          [5, 19], 1, [1])
    c = _get(cells, "rank1_traj", 19, "prefix")
    check("T1 exact-linear: rank1 ratio ~ 0", c["weight_proj_ratio"] <= 1e-6,
          f"ratio={c['weight_proj_ratio']:.2e}")
    check("T1 exact-linear: coef R2 == 1", c["coef_r2_1"] >= 1 - 1e-9,
          f"r2={c['coef_r2_1']:.12f}")
    check("T1 exact-linear: EVR1 == 1", c["evr1"] >= 1 - 1e-9,
          f"evr1={c['evr1']:.12f}")
    hs_row = _get(cells, "hold_stale", 19)
    check("T1 hold_stale identity ratio == 1",
          abs(hs_row["weight_proj_ratio"] - 1.0) <= 1e-9,
          f"|ratio-1|={abs(hs_row['weight_proj_ratio'] - 1):.2e}")
    nl = _get(cells, "naive_last2", 19)
    check("T1 exact-linear: naive_last2 also ~ 0 (both nail a line)",
          nl["weight_proj_ratio"] <= 1e-5, f"ratio={nl['weight_proj_ratio']:.2e}")

    # ---- T2: rank-1 + iid per-tick noise -> SVD family beats raw 2-point ----
    noise = {t: 0.003 * rng.standard_normal(d) for t in range(n_ticks)}
    thetas2 = {t: theta0 + (0.02 * t) * v1 + noise[t] for t in range(n_ticks)}
    _, cells2 = _cells_for(thetas2, n_ticks, 0, [40], ["24"], [20], 1, [1])
    r1 = _get(cells2, "rank1_traj", 20)
    nl2 = _get(cells2, "naive_last2", 20)
    tp = _get(cells2, "two_point_window", 20)
    check("T2 noisy: rank1 beats naive_last2",
          r1["weight_proj_ratio"] < nl2["weight_proj_ratio"],
          f"rank1={r1['weight_proj_ratio']:.4f} naive={nl2['weight_proj_ratio']:.4f}")
    check("T2 noisy: rank1 beats two_point over same window",
          r1["weight_proj_ratio"] < tp["weight_proj_ratio"],
          f"rank1={r1['weight_proj_ratio']:.4f} two_point={tp['weight_proj_ratio']:.4f}")
    check("T2 noisy: rank1 strongly skillful (ratio < 0.5)",
          r1["weight_proj_ratio"] < 0.5, f"ratio={r1['weight_proj_ratio']:.4f}")
    check("T2 noisy: coef R2 still high", r1["coef_r2_1"] > 0.9,
          f"r2={r1['coef_r2_1']:.4f}")

    # ---- T3: curved (rotating-direction) trajectory -> graceful degradation --
    thetas3 = {t: theta0 + (0.02 * t) * (math.cos(0.03 * t) * v1
                                         + math.sin(0.03 * t) * v2)
               for t in range(n_ticks)}
    _, cells3 = _cells_for(thetas3, n_ticks, 0, [40], ["16"], [5, 19], 1, [1, 2])
    r5 = _get(cells3, "rank1_traj", 5)
    r19 = _get(cells3, "rank1_traj", 19)
    check("T3 curved: EVR1 < 1 (rotation leaks energy to comp-2)",
          r5["evr1"] < 0.9999, f"evr1={r5['evr1']:.6f}")
    check("T3 curved: error grows with horizon",
          r19["err_norm"] > r5["err_norm"],
          f"err(h=19)={r19['err_norm']:.4g} err(h=5)={r5['err_norm']:.4g}")
    check("T3 curved: rows finite", np.isfinite(r19["weight_proj_ratio"]))

    # ---- T4: Gram-path vs direct-tensor parity + reconstruction --------------
    steps = 0.01 * rng.standard_normal((24, 2048))
    walk = np.cumsum(steps, axis=0)
    thetas4 = {t: (walk[t] if t else np.zeros(2048)) + 5.0 for t in range(24)}
    plan4, cells4 = _cells_for(thetas4, 24, 0, [16], ["8"], [3, 7], 1, [2])
    D4 = R1.gram_from_snapshots(thetas4, plan4.selected)
    B4 = R1.TrajGram(plan4, D4).base_gram(plan4.pos(0))
    window_ticks = list(range(9, 17))
    for h in (3, 7):
        gcell = _get(cells4, "rank2_traj", h)
        pred, diag = R1.direct_fit_predict(thetas4, 0, window_ticks, 16 + h, 2)
        met = M.full_metric_row(pred, thetas4[16 + h], thetas4[16], None)
        ok = True
        for kk in ("weight_proj_ratio", "dir_cos", "skill"):
            a, b = met[kk], gcell[kk]
            rel = abs(a - b) / max(abs(a), abs(b), 1e-30)
            if rel > PARITY_RTOL:
                ok = False
        check(f"T4 parity gram-vs-direct (h={h})", ok,
              f"ratio direct={met['weight_proj_ratio']:.8f} "
              f"gram={gcell['weight_proj_ratio']:.8f}")
        fit4 = R1.fit_rank_r(B4, plan4, 16, 8, 1, 2)
        pos_list, gam, base_coef = fit4.snapshot_coeffs(16 + h)
        tick_at = {plan4.pos(t): t for t in plan4.selected}
        recon = base_coef * thetas4[0]
        for p, g in zip(pos_list, gam):
            recon = recon + g * thetas4[tick_at[p]]
        rel = (np.linalg.norm(recon - pred)
               / max(np.linalg.norm(pred - thetas4[0]), 1e-30))
        check(f"T4 reconstruction-invariant (h={h})", rel <= 1e-9,
              f"rel={rel:.2e}")
        r2g = R1.fit_rank_r(B4, plan4, 16, 8, 1, 2)
        check(f"T4 fit diagnostics match (h={h})",
              np.allclose(r2g.sigma, diag["sigma"], rtol=1e-8) and
              np.allclose(np.nan_to_num(r2g.coef_r2),
                          np.nan_to_num(np.array(diag["coef_r2"])), rtol=1e-6),
              f"sigma gram={r2g.sigma} direct={diag['sigma']}")

    # ---- T5: base-tick shift invariance on the exact-linear trace ------------
    plan5a = _mk_plan(n_ticks, 0, [40], ["8"], [10], 1)
    plan5b = _mk_plan(n_ticks, 20, [40], ["8"], [10], 1)
    preds = []
    for base, plan in ((0, plan5a), (20, plan5b)):
        D = R1.gram_from_snapshots(thetas, plan.selected)
        B = R1.TrajGram(plan, D).base_gram(plan.pos(base))
        fit = R1.fit_rank_r(B, plan, 40, 8, 1, 1)
        pos_list, gam, base_coef = fit.snapshot_coeffs(50)
        tick_at = {plan.pos(t): t for t in plan.selected}
        recon = base_coef * thetas[base]
        for p, g in zip(pos_list, gam):
            recon = recon + g * thetas[tick_at[p]]
        preds.append(recon)
    rel = (np.linalg.norm(preds[0] - preds[1])
           / max(np.linalg.norm(preds[0] - theta0), 1e-30))
    check("T5 base-shift invariance (exact-linear trace)", rel <= 1e-8,
          f"rel={rel:.2e}")

    # ---- T6: plan guards ------------------------------------------------------
    for name, fn in (
            ("h=0 rejected", lambda: R1.build_tick_plan(0, [40], [8], [0], 1, 60)),
            ("window-through-base rejected",
             lambda: R1.build_tick_plan(10, [12], [8], [5], 1, 60)),
            ("target-past-trace rejected",
             lambda: R1.build_tick_plan(0, [40], [8], [30], 1, 60))):
        try:
            fn()
            check(f"T6 guard: {name}", False, "no assertion raised")
        except AssertionError:
            check(f"T6 guard: {name}", True)

    n = len(failures)
    log(f"self-test: {'ALL PASS' if not n else f'{n} FAILURES: {failures}'}")
    return 0 if not n else 1


# =============================================================================
# main
# =============================================================================
def main() -> int:
    ap = argparse.ArgumentParser(
        description="RELEX rank-1 trajectory family — offline scorecard lane")
    ap.add_argument("--trace-root", default="",
                    help="trace root (full/tick_<N>/tick_<N>.pt)")
    ap.add_argument("--manifest",
                    default="runs/EXP-57/regimeA/weights/full_manifest.jsonl")
    ap.add_argument("--out", default="runs/RANK1-ANALYSIS/scorecard")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--scope", default="panel",
                    help="panel | all | no-embed | regex:<pat>")
    ap.add_argument("--anchors", default="79,119",
                    help="scoring anchor ticks (predictions launch from here)")
    ap.add_argument("--window-grid", default="8,16,32,prefix",
                    help="delta-checkpoint counts feeding the SVD; 'prefix' = "
                         "all ticks from base+1 to the anchor (paper-faithful)")
    ap.add_argument("--h", dest="h_grid", default="1,2,5,10,20,40",
                    help="horizons in ticks ahead of the anchor")
    ap.add_argument("--rank-grid", default="1,2",
                    help="subspace ranks (paper: 1 wins; >1 = ablation)")
    ap.add_argument("--stride", type=int, default=1,
                    help="tick spacing inside the window (2 ~ per-step cadence)")
    ap.add_argument("--base-tick", type=int, default=0,
                    help="the theta_0 deltas are measured from")
    ap.add_argument("--cap-elems", type=int, default=6_000_000,
                    help="max elements co-resident per streamed chunk")
    ap.add_argument("--audit", type=int, default=3,
                    help="N smallest matrices spot-audited the direct way (0=off)")
    ap.add_argument("--force-recompute", action="store_true")
    args = ap.parse_args()

    assert M.METRIC_CONTRACT == METRIC_CONTRACT_EXPECTED
    assert R1.RANK1_CONTRACT == RANK1_CONTRACT_EXPECTED

    if args.self_test:
        return run_selftest()

    assert args.trace_root, "--trace-root required for emit"
    args.anchor_list = [int(x) for x in args.anchors.split(",")]
    args.window_specs = [w.strip() for w in args.window_grid.split(",")]
    args.h_list = [int(x) for x in args.h_grid.split(",")]
    args.rank_grid = [int(x) for x in args.rank_grid.split(",")]
    args.window_ints = sorted({
        resolve_window(w, a, args.base_tick, args.stride)
        for w in args.window_specs for a in args.anchor_list})
    return run_emit(args)


if __name__ == "__main__":
    sys.exit(main())
