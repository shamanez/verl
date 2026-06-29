#!/usr/bin/env python3
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

"""EXP-42 weight-projection-accuracy sweep (MacBook, GPU-free).

Reads the per-tick weight-trajectory SKETCH produced on the GPU box by
``verl.workers.comm_eff.capture.WeightTrajObserver`` (one ``sketch_tick_*.npz``
+ ``manifest.jsonl`` + ``calib.jsonl`` per regime under ``runs/EXP-42/<regime>/
weights/``) and replays the look-ahead weight predictor across the full
horizon × method × spacing grid for BOTH regimes — entirely offline, so the
H200 is torn down as soon as collection ends.

The PRIMARY metric is::

    weight_proj_ratio = ||θ̂ − target|| / ||θ_stale − target||   (HEADLINE; <1 helps)
    dir_cos           = cos(θ_stale − θ_old, target − θ_stale)    (overshoot sign)

Both are functions of weight-DIFFERENCE vectors, and the count-sketch is LINEAR
(``sketch(θ_t − θ_s) == sketch(θ_t) − sketch(θ_s)``), so every norm/cosine is
reconstructable from the saved per-tick sketches (rel. std ≈ 1/√k). The learned
residual is a per-matrix scalar mean-shift, replayed offline from the saved
per-matrix means.

This module ALSO holds the NumPy re-implementations of the on-device predictor
(``compute_theta_hat_ref`` / ``learned_update_ref``) and count-sketch
(``count_sketch_tables`` / ``count_sketch``). The hard-gate predictor-parity
probe (``research/runs/EXP-42/probe_cpu.py``) asserts these reproduce the
on-device ``verl.workers.comm_eff.lookahead`` / ``...capture.CountSketch``
outputs bit-for-bit on a CPU fixture.

Usage::

    python research/scripts/weight_proj_sweep.py runs/EXP-42 --emit report.html --calib-tol 0.05
"""

from __future__ import annotations

import argparse
import glob
import html
import json
import os
from typing import Optional

import numpy as np

# Default selector — mirrors capture.WEIGHT_TRAJ_DEFAULT_SUBSTRS / the projector.
DEFAULT_SUBSTRS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
# The full offline horizon grid (ticks ahead of the newest snapshot).
HORIZON_GRID = (1, 2, 3, 5, 8, 10, 13, 20, 30)
SPACING_GRID = (5, 10)  # Δ between the two source snapshots (= anchor cadence)
METHODS = ("fixed_linear", "learned_linear")


# ====================================================================== #
# Predictor re-implementation (parity target for lookahead.py)
# ====================================================================== #
def is_lookahead_target(name: str, substrs) -> bool:
    """Mirror ``lookahead.is_lookahead_target`` (substr match; 2-D enforced by caller)."""
    return bool(substrs) and any(s in name for s in substrs)


def compute_theta_hat_ref(sources: list, coeffs, *, target_substrs, residual: Optional[dict] = None):
    """NumPy mirror of ``verl.workers.comm_eff.lookahead.compute_theta_hat``.

    ``sources`` is ``[S0, S1, (S2)]`` newest-first, each ``{name -> np.ndarray
    (fp32)}``. Targets (2-D + substr match) get the affine combination ``a1*S0 +
    a2*S1 (+ a3*S2) (+ residual)``; every other key takes ``S0`` unchanged (the
    LayerNorm/embedding exclusion). Returns ``(theta_hat, sorted(excluded))``.
    """
    assert len(sources) >= 2, f"compute_theta_hat_ref needs >= 2 sources, got {len(sources)}"
    s0, s1 = sources[0], sources[1]
    s2 = sources[2] if len(sources) >= 3 else None
    a1, a2, a3 = float(coeffs[0]), float(coeffs[1]), float(coeffs[2] if len(coeffs) >= 3 else 0.0)
    theta_hat, excluded = {}, []
    for name, p0 in s0.items():
        p0 = np.asarray(p0, dtype=np.float32)
        is_target = is_lookahead_target(name, target_substrs) and p0.ndim == 2
        if not is_target:
            theta_hat[name] = p0
            excluded.append(name)
            continue
        p1 = s1.get(name)
        if p1 is None or np.asarray(p1).shape != p0.shape:
            theta_hat[name] = p0
            excluded.append(name)
            continue
        acc = a1 * p0.astype(np.float32) + a2 * np.asarray(p1, dtype=np.float32)
        if s2 is not None and a3 != 0.0:
            p2 = s2.get(name)
            if p2 is not None and np.asarray(p2).shape == p0.shape:
                acc = acc + a3 * np.asarray(p2, dtype=np.float32)
        if residual is not None:
            r = residual.get(name)
            if r is not None:
                acc = acc + np.float32(r)
        theta_hat[name] = acc.astype(p0.dtype)
    return theta_hat, sorted(excluded)


def coeffs_for_alpha(alpha: float):
    """Look-ahead coeffs ``(1+α, −α, 0)`` — the AsyncPP seed at α=1 (lookahead.py)."""
    return (1.0 + float(alpha), -float(alpha), 0.0)


def learned_update_ref(residual: dict, theta_true_prev: dict, theta_hat_prev: dict, *, target_substrs,
                       lr: float = 0.1, clip: float = 1.0e-3) -> dict:
    """NumPy mirror of ``LookaheadProjector.update_from_retrospective``.

    ``r ← clip(r + lr·mean(θ_true_prev − θ̂_prev), ±clip)`` per target matrix.
    """
    residual = dict(residual)
    if theta_true_prev is None or theta_hat_prev is None:
        return residual
    for name, t_true in theta_true_prev.items():
        if not is_lookahead_target(name, target_substrs):
            continue
        t_hat = theta_hat_prev.get(name)
        if t_hat is None or np.asarray(t_hat).shape != np.asarray(t_true).shape:
            continue
        err = float((np.asarray(t_true, np.float64) - np.asarray(t_hat, np.float64)).mean())
        new = float(residual.get(name, 0.0)) + lr * err
        residual[name] = max(-clip, min(clip, new))
    return residual


# ====================================================================== #
# Count-sketch re-implementation (parity target for capture.CountSketch)
# ====================================================================== #
def count_sketch_tables(d: int, k: int):
    """Re-draw the (bucket, sign) tables — identical to ``capture.CountSketch``."""
    rng = np.random.default_rng([int(d), int(k)])
    buckets = rng.integers(0, int(k), size=int(d), dtype=np.int64)
    signs = (rng.integers(0, 2, size=int(d), dtype=np.int8).astype(np.float32) * 2.0) - 1.0
    return buckets, signs


def count_sketch(x_flat: np.ndarray, k: int, *, tables=None) -> np.ndarray:
    """Count-sketch of a flat vector ``x`` — matches ``capture.CountSketch.sketch``."""
    x = np.asarray(x_flat, dtype=np.float32).reshape(-1)
    if tables is None:
        tables = count_sketch_tables(x.size, k)
    buckets, signs = tables
    return np.bincount(buckets, weights=(signs * x).astype(np.float64), minlength=k).astype(np.float32)


def ones_sketch(d: int, k: int, *, tables=None) -> np.ndarray:
    """Sketch of the all-ones vector (the learned residual's per-matrix carrier)."""
    if tables is None:
        tables = count_sketch_tables(d, k)
    buckets, signs = tables
    return np.bincount(buckets, weights=signs.astype(np.float64), minlength=k).astype(np.float32)


# ====================================================================== #
# Sketch-trace loading
# ====================================================================== #
def load_trace(weights_dir: str):
    """Load a regime's sketch trace → ``(ticks, names, sketches, means, dims, k)``.

    ``sketches[name]`` is an ``(n_ticks, k)`` fp32 array (one row per tick, tick
    order); ``means[name]`` is an ``(n_ticks,)`` fp32 array; ``dims[name]`` is the
    matrix element count ``d`` (for the ones-sketch).
    """
    manifest_path = os.path.join(weights_dir, "manifest.jsonl")
    rows = []
    with open(manifest_path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rows.sort(key=lambda r: int(r["tick"]))
    ticks = [int(r["tick"]) for r in rows]
    k = int(rows[0]["k"])
    names = [m["name"] for m in rows[0]["matrices"]]
    san = {m["name"]: m["sanitized"] for m in rows[0]["matrices"]}
    dims = {m["name"]: int(m["d"]) for m in rows[0]["matrices"]}
    sketches = {n: np.zeros((len(rows), k), dtype=np.float32) for n in names}
    means = {n: np.zeros(len(rows), dtype=np.float32) for n in names}
    for i, r in enumerate(rows):
        npz = np.load(os.path.join(weights_dir, r["sketch_path"]))
        per_mean = {m["name"]: float(m["mean"]) for m in r["matrices"]}
        for n in names:
            sketches[n][i] = npz[san[n]].astype(np.float32)
            means[n][i] = per_mean[n]
    return ticks, names, sketches, means, dims, k


# ====================================================================== #
# The sweep
# ====================================================================== #
def _pct(a, q):
    return float(np.percentile(np.asarray(a, np.float64), q)) if len(a) else float("nan")


def sweep_regime(weights_dir: str, *, deltas=SPACING_GRID, horizons=HORIZON_GRID,
                 methods=METHODS) -> dict:
    """Compute weight_proj_ratio / dir_cos vs horizon for one regime, all methods."""
    ticks, names, sketches, means, dims, k = load_trace(weights_dir)
    n = len(ticks)
    idx = {t: i for i, t in enumerate(ticks)}
    contiguous = ticks == list(range(ticks[0], ticks[0] + n))
    ones = {nm: ones_sketch(dims[nm], k) for nm in names} if "learned_linear" in methods else {}

    results = {}
    for delta in deltas:
        # learned residual replayed forward over fire ticks spaced by `delta`
        # (the anchor cadence): r updated from the PRIOR fire's retrospective err.
        learned_resid = {}  # name -> scalar, evolves with the trajectory
        learned_resid_at = {}  # anchor_tick -> snapshot of residual dict
        if "learned_linear" in methods:
            fires = [t for t in ticks if (t - ticks[0]) % delta == 0]
            prev_hat_mean = None  # name -> mean(θ̂) at the prior fire
            prev_fire = None
            for f in fires:
                if prev_fire is not None and prev_hat_mean is not None:
                    # retrospective: θ_true at THIS fire vs θ̂ predicted at prior fire
                    for nm in names:
                        if nm not in dims:
                            continue
                        err = float(means[nm][idx[f]]) - float(prev_hat_mean.get(nm, means[nm][idx[f]]))
                        new = float(learned_resid.get(nm, 0.0)) + 0.1 * err
                        learned_resid[nm] = max(-1e-3, min(1e-3, new))
                learned_resid_at[f] = dict(learned_resid)
                # the fixed extrapolation's mean at this fire (α=1 horizon=delta)
                if (f - delta) in idx:
                    a = 1.0
                    prev_hat_mean = {
                        nm: (1.0 + a) * float(means[nm][idx[f]]) - a * float(means[nm][idx[f - delta]])
                        for nm in names
                    }
                    prev_fire = f

        for method in methods:
            for h in horizons:
                alpha = float(h) / float(delta)
                w1_all, dcos_all = [], []
                # valid anchors s: need θ[s-Δ], θ[s], θ[s+h] all present
                for s in ticks:
                    if (s - delta) not in idx or (s + h) not in idx:
                        continue
                    i_s, i_old, i_tgt = idx[s], idx[s - delta], idx[s + h]
                    resid = learned_resid_at.get(s, {}) if method == "learned_linear" else None
                    for nm in names:
                        ss = sketches[nm]
                        d_old = ss[i_s] - ss[i_old]   # sketch(θ_stale − θ_old)
                        d_tgt = ss[i_s] - ss[i_tgt]   # sketch(θ_stale − target)
                        proj = d_tgt + alpha * d_old  # sketch(θ̂_fix − target)
                        if resid is not None and nm in resid and resid[nm] != 0.0:
                            proj = proj + np.float32(resid[nm]) * ones[nm]
                        den = float(np.linalg.norm(d_tgt))
                        if den <= 0.0:
                            continue
                        w1_all.append(float(np.linalg.norm(proj)) / den)
                        no = float(np.linalg.norm(d_old))
                        if no > 0.0:
                            ip = float(np.dot(d_old, d_tgt))
                            dcos_all.append(-ip / (no * den))
                results[(method, delta, h)] = {
                    "alpha": alpha,
                    "n": len(w1_all),
                    "w1_p10": _pct(w1_all, 10),
                    "w1_p50": _pct(w1_all, 50),
                    "w1_p90": _pct(w1_all, 90),
                    "dir_cos_p50": _pct(dcos_all, 50),
                }
    return {"results": results, "n_ticks": n, "ticks_contiguous": bool(contiguous),
            "n_matrices": len(names), "k": k}


def crossover_horizon(results: dict, method: str, delta: int, horizons=HORIZON_GRID):
    """Largest horizon h with median weight_proj_ratio < 1 (the key number)."""
    best = None
    for h in horizons:
        r = results.get((method, delta, h))
        if r and r["w1_p50"] < 1.0:
            best = h
    return best


# ====================================================================== #
# Calib validation
# ====================================================================== #
def validate_against_calib(sweep: dict, weights_dir: str, tol: float = 0.05) -> dict:
    """Compare the sketch-derived headline scalars to the on-box EXACT calib.

    The calib rows are per-ANCHOR exact medians; the sweep is the cross-anchor
    median. We compare the median calib value to the matching sweep cell within
    ``tol`` relative — the hard sketch-fidelity gate.
    """
    calib_path = os.path.join(weights_dir, "calib.jsonl")
    if not os.path.exists(calib_path):
        return {"available": False}
    rows = [json.loads(l) for l in open(calib_path) if l.strip()]
    if not rows:
        return {"available": False}
    by_cfg = {}
    for r in rows:
        by_cfg.setdefault((int(r["delta"]), int(r["h"])), []).append(r)
    checks = []
    res = sweep["results"]
    for (delta, h), grp in by_cfg.items():
        calib_w1 = float(np.median([g["weight_proj_ratio_p50"] for g in grp if g.get("weight_proj_ratio_p50") is not None]))
        cell = res.get(("fixed_linear", delta, h))
        if cell is None:
            continue
        rel = abs(cell["w1_p50"] - calib_w1) / (abs(calib_w1) + 1e-12)
        checks.append({"delta": delta, "h": h, "calib_w1_p50": calib_w1,
                       "sketch_w1_p50": cell["w1_p50"], "rel_err": rel, "pass": rel <= tol})
    return {"available": True, "tol": tol, "checks": checks,
            "all_pass": all(c["pass"] for c in checks) if checks else False}


# ====================================================================== #
# HTML report
# ====================================================================== #
def render_html(regimes: dict, calib: dict, deltas=SPACING_GRID, horizons=HORIZON_GRID) -> str:
    def esc(x):
        return html.escape(str(x))

    parts = ["<h1>EXP-42 — weight-projection accuracy vs horizon</h1>"]
    parts.append("<p>weight_proj_ratio = ||θ̂−target|| / ||θ_stale−target||. "
                 "<b>&lt;1 ⇒ projection lands closer than raw-stale</b> (helps); "
                 "≥1 ⇒ no help / overshoot. dir_cos &lt;0 ⇒ the past update points away "
                 "from the future update (the sign-flip that ignited prior collapses).</p>")
    # operating-point answer
    parts.append("<h2>Operating point (Δ=10, α=1 ⇒ h=10; under/over-shoot h∈{5,20})</h2><ul>")
    for rname, sweep in regimes.items():
        res = sweep["results"]
        for method in METHODS:
            for h in (5, 10, 20):
                cell = res.get((method, 10, h))
                if cell is None:
                    continue
                verdict = "HELPS" if cell["w1_p50"] < 1.0 else "no help"
                parts.append(
                    f"<li><b>{esc(rname)}</b> / {esc(method)} / h={h}: "
                    f"w1_p50={cell['w1_p50']:.4f} ({verdict}), dir_cos={cell['dir_cos_p50']:.4f}, "
                    f"n={cell['n']}</li>"
                )
    parts.append("</ul>")
    # crossover table
    parts.append("<h2>Crossover horizon h* (largest h with median ratio &lt;1)</h2><table border=1 cellpadding=4><tr><th>regime</th><th>method</th><th>Δ</th><th>h*</th></tr>")
    for rname, sweep in regimes.items():
        for method in METHODS:
            for delta in deltas:
                hstar = crossover_horizon(sweep["results"], method, delta, horizons)
                parts.append(f"<tr><td>{esc(rname)}</td><td>{esc(method)}</td><td>{delta}</td><td>{esc(hstar)}</td></tr>")
    parts.append("</table>")
    # full curves
    for rname, sweep in regimes.items():
        parts.append(f"<h2>{esc(rname)} — full curves (w1_p50 [p10,p90] · dir_cos)</h2>")
        for method in METHODS:
            for delta in deltas:
                parts.append(f"<h4>{esc(method)} Δ={delta}</h4><table border=1 cellpadding=3><tr><th>h</th><th>α</th><th>w1_p50</th><th>w1_p10</th><th>w1_p90</th><th>dir_cos</th><th>n</th></tr>")
                for h in horizons:
                    c = sweep["results"].get((method, delta, h))
                    if c is None:
                        continue
                    parts.append(
                        f"<tr><td>{h}</td><td>{c['alpha']:.2f}</td><td>{c['w1_p50']:.4f}</td>"
                        f"<td>{c['w1_p10']:.4f}</td><td>{c['w1_p90']:.4f}</td>"
                        f"<td>{c['dir_cos_p50']:.4f}</td><td>{c['n']}</td></tr>"
                    )
                parts.append("</table>")
    # calib
    parts.append("<h2>Sketch fidelity vs on-box EXACT calib</h2>")
    if calib.get("available"):
        parts.append(f"<p>tol={calib['tol']:.2%} · all_pass={calib['all_pass']}</p><table border=1 cellpadding=3><tr><th>Δ</th><th>h</th><th>calib w1_p50</th><th>sketch w1_p50</th><th>rel_err</th><th>pass</th></tr>")
        for c in calib["checks"]:
            parts.append(f"<tr><td>{c['delta']}</td><td>{c['h']}</td><td>{c['calib_w1_p50']:.4f}</td><td>{c['sketch_w1_p50']:.4f}</td><td>{c['rel_err']:.2%}</td><td>{c['pass']}</td></tr>")
        parts.append("</table>")
    else:
        parts.append("<p>No calib.jsonl found.</p>")
    return "<html><head><meta charset='utf-8'><title>EXP-42 weight-projection accuracy</title></head><body>" + "".join(parts) + "</body></html>"


def find_regimes(run_dir: str) -> dict:
    """Locate ``<regime>/weights/`` dirs under the run dir."""
    out = {}
    for wd in sorted(glob.glob(os.path.join(run_dir, "*", "weights"))):
        if os.path.exists(os.path.join(wd, "manifest.jsonl")):
            out[os.path.basename(os.path.dirname(wd))] = wd
    # also accept a bare weights/ directly under run_dir
    bare = os.path.join(run_dir, "weights")
    if os.path.exists(os.path.join(bare, "manifest.jsonl")):
        out.setdefault("default", bare)
    return out


def main():
    ap = argparse.ArgumentParser(description="EXP-42 weight-projection-accuracy sweep")
    ap.add_argument("run_dir", help="runs/EXP-42 (expects <regime>/weights/ inside)")
    ap.add_argument("--emit", default=None, help="write the HTML report here")
    ap.add_argument("--calib-tol", type=float, default=0.05, help="sketch-fidelity tolerance (rel)")
    ap.add_argument("--json", default=None, help="also dump the raw sweep results as JSON here")
    args = ap.parse_args()

    regime_dirs = find_regimes(args.run_dir)
    if not regime_dirs:
        raise SystemExit(f"no <regime>/weights/manifest.jsonl found under {args.run_dir}")

    regimes, calib_any = {}, {"available": False}
    out_json = {}
    for rname, wd in regime_dirs.items():
        print(f"[sweep] {rname}: {wd}")
        sweep = sweep_regime(wd)
        regimes[rname] = sweep
        calib = validate_against_calib(sweep, wd, tol=args.calib_tol)
        if calib.get("available"):
            calib_any = calib
        out_json[rname] = {
            "n_ticks": sweep["n_ticks"], "n_matrices": sweep["n_matrices"], "k": sweep["k"],
            "results": {f"{m}|{d}|{h}": v for (m, d, h), v in sweep["results"].items()},
            "calib": calib,
        }
        for method in METHODS:
            for delta in SPACING_GRID:
                hstar = crossover_horizon(sweep["results"], method, delta)
                print(f"  {method} Δ={delta}: crossover h*={hstar}")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out_json, fh, indent=2)
        print(f"[sweep] wrote {args.json}")
    if args.emit:
        with open(args.emit, "w") as fh:
            fh.write(render_html(regimes, calib_any))
        print(f"[sweep] wrote {args.emit}")


if __name__ == "__main__":
    main()
