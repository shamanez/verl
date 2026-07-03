#!/usr/bin/env python3
"""moat_scorecard.py — #45 shared scorecard CONTRACT harness for the EXP-57 fp32 trace.

The thin standalone replay harness the MOAT lanes (#47/#48/#49) plug their methods
into, and whose emitted tables #56 renders. GPU-free; runs on the big-disk analysis
box over a pre-downloaded trace (--trace-root), or on any machine for --verify-schema.

CLI contract (plan 45 `## Verification commands`):
    python scripts/moat_scorecard.py --trace-root $TRACE --self-test
    python scripts/moat_scorecard.py --trace-root $TRACE \
        --method hold_stale,naive_linear --delta 5,10,20 --h 1,2,5,10,20,30,40 \
        --operating-point 20,20 --also 10,10 --out runs/MOAT-45-ANALYSIS/scorecard/
    python scripts/moat_scorecard.py --verify-schema runs/MOAT-45-ANALYSIS/scorecard/

(Delta_ticks, h_ticks) semantics — the crux of the contract (plan `## Notes for runner`):
raw optimizer-tick indices 0..n_ticks-1, NO subsampling. For a window with latest
anchor at tick t: Delta = spacing of the anchors the method uses (linear uses
theta_{t-Delta} and theta_t); h = horizon from t to the scoring point t+h. Then
    stale_error       = ||theta_t - theta_{t+h}||
    proj_error        = ||theta_hat_{t+h} - theta_{t+h}||
    weight_proj_ratio = proj_error / stale_error          (metrics.weight_proj_ratio)
hold_stale sets theta_hat = theta_t  => ratio == 1.0 exactly, for every Delta.
naive_linear: v = (theta_t - theta_{t-Delta})/Delta; theta_hat = theta_t + h*v.
Window validity: t-Delta >= 0 and t+h <= n_ticks-1; n_windows = n_ticks - h - Delta.

DESIGN — one bounded streaming pass + exact delta-Gram sufficient statistics.
The ~987 GB trace is never held in RAM (bounded-footprint single-streaming-pass R2 access discipline). Matrices are
packed into chunks (a whole matrix, or a row-shard of an oversized one); each chunk
streams the ticks ONCE, mmap-loading only its own tensors from each tick_<N>.pt,
maintaining a rolling float64 ring of the last BAND-1 consecutive deltas
d_i = theta_{i+1} - theta_i (BAND = max Delta + max h, so every window's pairs are
in-band). Per delta it records the banded Gram D[i, l] = <d_i, d_{i-l}> plus
per-matrix trajectory accumulators. Every window quantity is then an EXACT block sum
over D (float64, no big-number cancellation):
    A = [t-Delta, t)  B = [t, t+h)      (delta indices)
    b = theta_t - theta_{t+h} = -sum_B d;   e = kappa*sum_A d - sum_B d,  kappa = h/Delta
    ||e||^2 = k^2*S_AA + S_BB - 2k*S_AB;  ||b||^2 = S_BB;  <e,b> = S_BB - k*S_AB
Group vectors are concatenations of member matrices (sweep.concat_group semantics),
so group sums are the SUMS of member sums — the true joint ratio, never an average.

METRIC-CONTRACT PIN (metric math REUSED, never re-derived). Every reported
ratio/skill/dir_cos/radial/tangential is produced by metrics.full_metric_row on a
pair of surrogate vectors (e_hat, b_hat) in R^2 constructed to have EXACTLY the
window's (||e||, ||b||, <e,b>) geometry — the division/sqrt/dot all happen inside
weight_proj.metrics. An off-path parity battery (--self-test) recomputes sampled
windows the DIRECT way — predictors.Order1.predict on the actual loaded snapshot
tensors, then metrics.full_metric_row — and asserts agreement <= 1e-6 relative.

Method plugin interface for the lanes (#47/#48/#49): subclass MoatMethod and
register_method() it. `predict(history, delta, h) -> theta_hat` mirrors
predictors._BasePredictor.predict (history = ordered [(tick, theta)] anchors ending
at theta_t). Fixed linear two-anchor methods ride the fast Gram path via kappa();
fit-methods (needs_fit=True) MUST derive their fit windows through
predictors.fit_score_split — the leakage guard is wired in fit_window_positions().
Out-of-scope families (OrderPoly/EMA/Learnable*/GeneralRegression) are NOT wired.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from weight_proj import metrics as M            # noqa: E402
from weight_proj import predictors as P         # noqa: E402
from weight_proj import structure as ST         # noqa: E402
from weight_proj import tick_select as TS        # noqa: E402  (#47 --cadence)

METRIC_CONTRACT_EXPECTED = "weight-proj-metrics-v1"
# #47 cache-schema version — folded into _fingerprint so pre-#47 caches (no R2
# summaries, no cadence/tick-set) rebuild cleanly instead of loading stale.
SCHEMA_VERSION = "moat-scorecard-v47.1"
DEFAULT_MANIFEST = "runs/EXP-57/regimeA/weights/full_manifest.jsonl"
DEFAULT_OUT = "runs/MOAT-45-ANALYSIS/scorecard"
PARITY_RTOL = 1e-6          # off-path parity: surrogate-path vs direct-tensor-path
IDENTITY_TOL = 1e-6         # hold-stale gate: |ratio-1| and |skill| per plan
R2_STRONG = 0.7             # per-scalar linearity R² "strong" threshold (Wang et al.)
R2_HIST_BINS = 100          # 100 bins over [0,1] for histogram-merged group R²
R2_CONST_EPS = 1e-300       # SS_tot <= this ⇒ constant scalar (excluded + counted)
PAPER_SENTINEL_DELTA = 0    # paper_linear rows key delta_ticks=0 (delta is derived)

REQUIRED_ROW_KEYS = [
    # identity of the row
    "method", "delta_ticks", "h_ticks", "group_kind", "group_key",
    # structure axes
    "matrix_name", "layer_idx", "special", "block_type", "super_block",
    # minimal metrics (median + p10 + p90)
    "stale_error_median", "proj_error_median",
    "weight_proj_ratio_median", "weight_proj_ratio_p10", "weight_proj_ratio_p90",
    "skill_median", "skill_p10", "skill_p90",
    # diagnostics (radial/tangential intentionally optional: only-on-failure)
    "traj_r2", "consec_delta_cos", "delta_norm", "coverage", "dir_cos_median",
    # bookkeeping
    "n_windows", "in_bounds", "n_nan_windows",
    # derived
    "h_star", "best_delta",
    # tied-lm_head bookkeeping
    "tied",
    # ---- #47 REQUIRED superset (existing 29 keys above unchanged) ----
    "cadence", "unit",                             # regime tag
    "anchor_mode", "delta_resolved", "beta",       # two-anchor descriptor (paper-equiv)
    "r2_median", "r2_frac_gt_0.7", "n_excluded_const",  # per-scalar linearity R²
    "lam_star",                                    # OOS-selected lambda (None off damped)
]

# base visuals (#45) present in every regime; #47 adds the lambda / R² / paper visuals.
VISUAL_KEYS = ["a_accuracy_vs_horizon", "b_delta_sensitivity",
               "c_target_horizon_sweep", "d_traj_r2",
               "e_ratio_heatmap", "f_hstar_heatmap", "g_special_groups",
               "h_lambda_selection", "i_r2_histogram", "j_r2_depth_block_heatmap",
               "k_r2_ratio_coupling"]
# regime-S-only visual (paper_linear present) — declared per-emit in meta.visual_keys
VISUAL_KEY_PAPER = "m_paper_equivalence"


def log(msg: str) -> None:
    print(f"[moat] {time.strftime('%H:%M:%S')} {msg}", flush=True)


# =============================================================================
# Method plugins
# =============================================================================
class MoatMethod:
    """Plugin base for the lanes. Contract: predict(history, delta, h) -> theta_hat.

    history = ordered [(tick, theta)] anchors OLDEST first, LAST = theta_t (the
    latest anchor; causality: every history tick <= t < t+h — asserted by the
    engine). Fit-methods set needs_fit=True and obtain their retrospective fit
    window via fit_window_positions() (predictors.fit_score_split under the hood)
    so they can never see the scoring point.
    """
    name = "base"
    needs_fit = False

    def anchor_offsets(self, delta: int) -> list[int]:
        """Anchor tick offsets relative to t (e.g. [-delta, 0]) this method loads."""
        raise NotImplementedError

    def predict(self, history, delta: int, h: int):
        raise NotImplementedError


class TwoAnchorLinear(MoatMethod):
    """theta_hat = theta_t + kappa(delta,h) * (theta_t - theta_{t-delta}).

    Both #45 reference methods are this family. predict() REUSES
    predictors.Order1 — with anchors spaced delta ticks apart, Order1's step
    argument is kappa = damp*h_arg, so predict(history, kappa) yields exactly
    theta_t + kappa*(theta_t - theta_{t-delta}).
    """
    def __init__(self, name: str):
        self.name = name
        self._o1 = P.Order1("fixed", damp=1.0)

    def kappa(self, delta: int, h: int) -> float:
        raise NotImplementedError

    def anchor_offsets(self, delta: int) -> list[int]:
        return [-delta, 0]

    def predict(self, history, delta: int, h: int):
        assert len(history) == 2, "two-anchor method needs exactly [theta_{t-D}, theta_t]"
        return self._o1.predict(history, self.kappa(delta, h))

    def window_stats(self, s_aa, s_ab, s_bb, delta: int, h: int):
        """(||e||^2, ||b||^2, <e,b>) from the delta-Gram block sums (see module doc)."""
        k = self.kappa(delta, h)
        e2 = k * k * s_aa + s_bb - 2.0 * k * s_ab
        b2 = s_bb
        eb = s_bb - k * s_ab
        return e2, b2, eb


class HoldStale(TwoAnchorLinear):
    """Identity / zero-skill reference: theta_hat = theta_t (kappa == 0), ratio == 1."""
    def __init__(self):
        super().__init__("hold_stale")

    def kappa(self, delta: int, h: int) -> float:
        return 0.0


class NaiveLinear(TwoAnchorLinear):
    """v = (theta_t - theta_{t-delta})/delta; theta_hat = theta_t + h*v (kappa = h/delta)."""
    def __init__(self):
        super().__init__("naive_linear")

    def kappa(self, delta: int, h: int) -> float:
        return float(h) / float(delta)


class DampedLinear(TwoAnchorLinear):
    """Damped two-anchor linear: theta_hat = theta_t + lambda*(h/delta)*(theta_t-theta_{t-delta}).

    kappa = lambda * h / delta, so the family NESTS both #45 references:
      lambda = 1.0  ->  kappa = h/delta  == naive_linear   (bit-for-bit identity)
      lambda = 0.0  ->  kappa = 0        == hold_stale      (ratio == 1, skill == 0)
    `self.lam` is settable; the emit's OOS selector picks lambda per SCORED window on
    strictly-earlier windows (leakage-guarded), so a damped row's ratio is the OOS
    ratio (never an in-sample oracle). `predict()`/`window_stats()` use the current
    self.lam — used by the off-path parity self-test at a fixed lambda.
    """
    def __init__(self, lam: float = 1.0):
        super().__init__("damped_linear")
        self.lam = float(lam)

    def kappa(self, delta: int, h: int) -> float:
        return self.lam * float(h) / float(delta)


def _damped_e2(k: float, s_aa, s_ab, s_bb):
    """||e||^2 for kappa=k over the delta-Gram block sums (vectorized over windows)."""
    return k * k * s_aa + s_bb - 2.0 * k * s_ab


_REGISTRY: dict[str, MoatMethod] = {}


def register_method(m: MoatMethod) -> None:
    _REGISTRY[m.name] = m


register_method(HoldStale())
register_method(NaiveLinear())
register_method(DampedLinear())


def fit_window_positions(n_ticks: int, t: int, fit_len: int) -> list[int]:
    """Leakage-guarded retrospective fit window for needs_fit methods (#49).

    Wires predictors.fit_score_split: the fit window ends strictly BEFORE the
    anchor t (itself strictly before the scoring point t+h), so a fit method can
    never see the scoring point. #45 ships no fit method; the guard is wired here
    so #49 is safe by construction.
    """
    fit_idx, score_idx = P.fit_score_split(list(range(n_ticks)), t, fit_len)
    assert max(fit_idx) < score_idx <= t, "leakage: fit window reaches the anchor"
    return fit_idx


# =============================================================================
# Trace access (mmap-selective; never deletes — the trace is operator-owned)
# =============================================================================
class MmapTraceReader:
    """Selective reader over <trace_root>/full/tick_<N>/tick_<N>.pt (flat fallback).

    torch.load(mmap=True) maps the zip storages lazily; slicing one tensor pages in
    only its bytes, so a chunk pass reads ~its own share of the trace, not 987 GB.
    Mirrors r2_stream.LocalSnapshotSource's layout contract (REUSED conventions);
    kept separate because LocalSnapshotSource materializes full fp32 copies of
    whole state dicts, which the chunked engine must not do.
    """
    def __init__(self, trace_root: str, tickset: list[int] | None = None):
        # tickset maps the STEP index s (0..n-1, the cadence-reindexed axis every
        # downstream stage works in) to the REAL on-disk tick. None => identity
        # (step == tick). For --cadence per-step, tickset = [0,2,4,…] so step s
        # loads tick_{2s} — the cadence-reindex invariant.
        self.trace_root = trace_root
        self.tickset = tickset

    def real_tick(self, step: int) -> int:
        return step if self.tickset is None else int(self.tickset[step])

    def path(self, step: int) -> str:
        tick = self.real_tick(step)
        nested = os.path.join(self.trace_root, "full", f"tick_{tick}", f"tick_{tick}.pt")
        if os.path.exists(nested):
            return nested
        return os.path.join(self.trace_root, "full", f"tick_{tick}.pt")

    def present_ticks(self, n_ticks: int) -> list[int]:
        # validate the SELECTED set (the n_ticks step indices), not raw ticks 0..n-1
        return [s for s in range(n_ticks) if os.path.exists(self.path(s))]

    def load_raw(self, tick: int):
        import torch
        return torch.load(self.path(tick), map_location="cpu",
                          mmap=True, weights_only=True)

    def slice_f64(self, sd, name: str, a: int, b: int) -> np.ndarray:
        ten = sd[name]
        assert str(ten.dtype) == "torch.float32", f"{name}: dtype {ten.dtype} != fp32"
        return ten.reshape(-1)[a:b].numpy().astype(np.float64)

    def load_matrix_f64(self, tick: int, name: str) -> np.ndarray:
        sd = self.load_raw(tick)
        out = self.slice_f64(sd, name, 0, sd[name].numel())
        del sd
        return out


class InMemoryReader:
    """Synthetic-trace reader for the self-test battery: {tick: {name: 1-D f32}}.

    Cadence-aware exactly like MmapTraceReader: an optional `tickset` maps the step
    index to the real synthetic tick, so the cadence-reindex invariant is testable
    entirely in memory.
    """
    def __init__(self, ticks: dict, tickset: list[int] | None = None):
        self.ticks = ticks
        self.tickset = tickset

    def real_tick(self, step: int) -> int:
        return step if self.tickset is None else int(self.tickset[step])

    def present_ticks(self, n_ticks: int) -> list[int]:
        return [s for s in range(n_ticks) if self.real_tick(s) in self.ticks]

    def load_raw(self, step: int):
        return self.ticks[self.real_tick(step)]

    def slice_f64(self, sd, name: str, a: int, b: int) -> np.ndarray:
        return np.asarray(sd[name]).reshape(-1)[a:b].astype(np.float64)

    def load_matrix_f64(self, step: int, name: str) -> np.ndarray:
        arr = np.asarray(self.ticks[self.real_tick(step)][name]).reshape(-1)
        return arr.astype(np.float64)


# =============================================================================
# Streaming sufficient-stats engine (bounded footprint; float64 throughout)
# =============================================================================
class _Unit:
    """One streamed unit: a whole matrix, or a contiguous element shard of one."""
    __slots__ = ("name", "a", "b", "n", "ring", "prev", "c", "V", "W",
                 "sum_phi2", "D", "nnz", "P")

    def __init__(self, name: str, a: int, b: int):
        self.name, self.a, self.b, self.n = name, a, b, b - a


def plan_chunks(name_dims: list[tuple[str, int]], cap_elems: int) -> list[list[_Unit]]:
    """Pack matrices (row-sharding any matrix > cap_elems) into chunks <= cap_elems."""
    units: list[_Unit] = []
    for name, d in name_dims:
        if d <= cap_elems:
            units.append(_Unit(name, 0, d))
        else:
            n_shards = math.ceil(d / cap_elems)
            step = math.ceil(d / n_shards)
            for a in range(0, d, step):
                units.append(_Unit(name, a, min(a + step, d)))
    units.sort(key=lambda u: -u.n)
    chunks: list[list[_Unit]] = []
    sizes: list[int] = []
    for u in units:
        placed = False
        for i, s in enumerate(sizes):
            if s + u.n <= cap_elems:
                chunks[i].append(u)
                sizes[i] += u.n
                placed = True
                break
        if not placed:
            chunks.append([u])
            sizes.append(u.n)
    return chunks


# ---- per-scalar linearity R² (Wang et al. 2026; the MUST metric) -------------
def per_element_r2(V: np.ndarray, W: np.ndarray, P: np.ndarray, N: int):
    """EXACT per-element simple-OLS R² of phi_t vs the step index t = 0..N-1.

    Sufficient stats (per element): V = sum_t phi_t, W = sum_t (t-tbar) phi_t,
    P = sum_t phi_t^2. Then
        SS_tot = P - V^2/N ,  SS_reg = W^2 / S_tt ,  S_tt = N(N^2-1)/12
        R² = SS_reg / SS_tot   (== R²(theta vs t): constant-shift invariant)
    Returns (r2, const_mask). const_mask = SS_tot <= R2_CONST_EPS (constant scalar
    under fp32 — the paper's exclusion). Clipped to [0,1] (simple-OLS bound
    SS_reg <= SS_tot; only fp rounding can escape it)."""
    N = int(N)
    S_tt = N * (N * N - 1) / 12.0
    ss_tot = P - V * V / N
    ss_reg = W * W / S_tt
    const_mask = ss_tot <= R2_CONST_EPS
    with np.errstate(invalid="ignore", divide="ignore"):
        r2 = np.where(const_mask, np.nan, ss_reg / ss_tot)
    r2 = np.clip(r2, 0.0, 1.0)
    return r2, const_mask


def r2_summary_from_elements(r2: np.ndarray, const_mask: np.ndarray):
    """(100-bin [0,1] histogram, n_excluded_const, n_valid) for the valid scalars."""
    valid = r2[~const_mask]
    valid = valid[np.isfinite(valid)]
    hist, _ = np.histogram(valid, bins=R2_HIST_BINS, range=(0.0, 1.0))
    return hist.astype(np.int64), int(np.sum(const_mask)), int(valid.size)


def r2_from_hist(hist: np.ndarray):
    """(median, frac>0.7) read off a merged 100-bin histogram (<=1% median error).

    Group aggregates sum member histograms then read here — additive, no trace access.
    frac>R2_STRONG counts bins whose center exceeds R2_STRONG."""
    hist = np.asarray(hist, dtype=np.float64)
    n = float(hist.sum())
    if n <= 0:
        return float("nan"), float("nan")
    centers = (np.arange(R2_HIST_BINS) + 0.5) / R2_HIST_BINS
    cum = np.cumsum(hist)
    mi = int(np.searchsorted(cum, (n + 1.0) / 2.0))
    med = float(centers[min(mi, R2_HIST_BINS - 1)])
    thresh_bin = int(np.ceil(R2_STRONG * R2_HIST_BINS))   # bins [70:] -> center > 0.7
    frac = float(hist[thresh_bin:].sum()) / n
    return med, frac


def stream_stats(reader, name_dims: list[tuple[str, int]], n_ticks: int, band: int,
                 ram_gb: float = 40.0, tag: str = "",
                 retain_r2: set | None = None) -> dict:
    """ONE bounded streaming pass per chunk -> per-matrix sufficient statistics.

    Returns {name: {"D": (n_ticks-1, band) f64 banded delta-Gram, "nnz": (n_ticks-1,)
    int64 changed-element counts, "d": int, "sum_phi2": float, "v2": float,
    "wc2": float, "r2_hist": (100,) int64, "n_excluded_const": int, "r2_n_valid": int}}
    where phi_t = theta_t - theta_0, V = sum_t phi_t, W_c = sum_t (t - tbar) phi_t;
    sum_phi2 = sum_t ||phi_t||^2, v2 = ||V||^2, wc2 = ||W_c||^2 (the trajectory
    linear-fit sufficient stats — additive over concatenation, so group traj_r2 is
    exact).

    #47: a PER-ELEMENT accumulator P = sum_t phi_t^2 rides the same pass beside V/W
    (one extra f64 vector per unit). At chunk end, while V/W/P are still per-element,
    each matrix reduces to the per-scalar linearity R² summary {r2_hist (100-bin),
    n_excluded_const, r2_n_valid} — R²(theta vs t) == R²(phi vs t) (shift-invariant).
    `retain_r2` names small matrices whose raw per-element (V,W,P) are kept (in
    "_r2_ve") for the off-path parity self-test; NEVER used in production (huge).
    """
    band = min(band, n_ticks - 1)
    assert band >= 2, "band must cover at least lag 1 (consec_delta_cos)"
    nd = n_ticks - 1
    cap_elems = max(int(ram_gb * 1e9 / ((band + 9) * 8)), 1_000_000)  # +9: ring+V/W/c/P/prev/cur
    chunks = plan_chunks(name_dims, cap_elems)
    tbar = (n_ticks - 1) / 2.0
    retain_r2 = retain_r2 or set()
    stats: dict = {}
    log(f"stream{tag}: {len(name_dims)} matrices, {sum(len(c) for c in chunks)} units, "
        f"{len(chunks)} chunks (cap {cap_elems:,} elems), n_ticks={n_ticks}, band={band}")
    for ci, chunk in enumerate(chunks):
        t0 = time.time()
        for u in chunk:
            u.ring = np.zeros((band, u.n), dtype=np.float64)
            u.prev = None
            u.c = np.zeros(u.n, dtype=np.float64)
            u.V = np.zeros(u.n, dtype=np.float64)
            u.W = np.zeros(u.n, dtype=np.float64)
            u.P = np.zeros(u.n, dtype=np.float64)      # #47: per-element sum_t phi_t^2
            u.sum_phi2 = 0.0
            u.D = np.zeros((nd, band), dtype=np.float64)
            u.nnz = np.zeros(nd, dtype=np.int64)
        for t in range(n_ticks):
            sd = reader.load_raw(t)
            for u in chunk:
                cur = reader.slice_f64(sd, u.name, u.a, u.b)
                if t == 0:
                    u.prev = cur
                    continue
                i = t - 1                      # delta index: d_i = theta_{i+1} - theta_i
                w = i % band
                np.subtract(cur, u.prev, out=u.ring[w])
                d = u.ring[w]
                u.prev = cur
                dots = u.ring @ d              # dgemv over the full ring (stale rows skipped below)
                max_lag = min(i, band - 1)
                for lag in range(max_lag + 1):
                    u.D[i, lag] = dots[(i - lag) % band]
                u.nnz[i] = int(np.count_nonzero(d))
                # trajectory accumulators: phi_t = c after adding d_{t-1}
                u.c += d
                u.sum_phi2 += float(u.c @ u.c)
                u.V += u.c
                u.W += (t - tbar) * u.c
                u.P += u.c * u.c               # #47: per-element phi_t^2
            del sd
        for u in chunk:
            key = u.name
            if key not in stats:
                stats[key] = {"D": np.zeros((nd, band), dtype=np.float64),
                              "nnz": np.zeros(nd, dtype=np.int64), "d": 0,
                              "sum_phi2": 0.0, "v2": 0.0, "wc2": 0.0,
                              "r2_hist": np.zeros(R2_HIST_BINS, dtype=np.int64),
                              "n_excluded_const": 0, "r2_n_valid": 0}
            s = stats[key]
            s["D"] += u.D
            s["nnz"] += u.nnz
            s["d"] += u.n
            s["sum_phi2"] += u.sum_phi2
            s["v2"] += float(u.V @ u.V)        # additive over element shards (concat)
            s["wc2"] += float(u.W @ u.W)
            # #47 per-scalar R²: reduce this shard's per-element (V,W,P) to a hist
            r2e, cmask = per_element_r2(u.V, u.W, u.P, n_ticks)
            hist, n_excl, n_val = r2_summary_from_elements(r2e, cmask)
            s["r2_hist"] += hist
            s["n_excluded_const"] += n_excl
            s["r2_n_valid"] += n_val
            if u.name in retain_r2:
                if "_r2_ve" in s:
                    raise AssertionError(f"retain_r2 matrix {u.name} is sharded — "
                                         "retain only tiny single-shard matrices")
                s["_r2_ve"] = (u.V.copy(), u.W.copy(), u.P.copy())
            u.ring = u.prev = u.c = u.V = u.W = u.P = u.D = None
        log(f"stream{tag}: chunk {ci + 1}/{len(chunks)} done in {time.time() - t0:.1f}s")
    return stats


# ---- stats cache -------------------------------------------------------------
def _fingerprint(name_dims, n_ticks, band, trace_root,
                 cadence: str = "per-tick", tickset: list | None = None) -> str:
    """Cache identity. #47 folds in the cadence, the SELECTED tick-set, and
    SCHEMA_VERSION so regimes S/T never collide and pre-#47 caches (no R² block)
    rebuild cleanly instead of silently loading a stale schema."""
    h = hashlib.sha256()
    h.update(json.dumps({"nd": name_dims, "t": n_ticks, "b": band,
                         "root": os.path.realpath(trace_root),
                         "cadence": cadence,
                         "tickset": list(tickset) if tickset is not None else None,
                         "schema": SCHEMA_VERSION},
                        sort_keys=True).encode())
    return h.hexdigest()[:16]


def save_stats_cache(path: str, stats: dict, fingerprint: str) -> None:
    arrays, meta = {}, {"fingerprint": fingerprint, "scalars": {}}
    for i, (name, s) in enumerate(sorted(stats.items())):
        arrays[f"D_{i}"] = s["D"]
        arrays[f"nnz_{i}"] = s["nnz"]
        arrays[f"r2hist_{i}"] = s["r2_hist"]
        meta["scalars"][name] = {"idx": i, "d": s["d"], "sum_phi2": s["sum_phi2"],
                                 "v2": s["v2"], "wc2": s["wc2"],
                                 "n_excluded_const": s["n_excluded_const"],
                                 "r2_n_valid": s["r2_n_valid"]}
    arrays["___meta___"] = np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)
    np.savez_compressed(path, **arrays)


def load_stats_cache(path: str, fingerprint: str) -> dict | None:
    if not os.path.exists(path):
        return None
    try:
        z = np.load(path)
        meta = json.loads(bytes(z["___meta___"]).decode())
        if meta["fingerprint"] != fingerprint:
            log(f"stats cache fingerprint mismatch — recomputing ({path})")
            return None
        out = {}
        for name, sc in meta["scalars"].items():
            i = sc["idx"]
            out[name] = {"D": z[f"D_{i}"], "nnz": z[f"nnz_{i}"], "d": sc["d"],
                         "sum_phi2": sc["sum_phi2"], "v2": sc["v2"], "wc2": sc["wc2"],
                         "r2_hist": z[f"r2hist_{i}"],
                         "n_excluded_const": int(sc["n_excluded_const"]),
                         "r2_n_valid": int(sc["r2_n_valid"])}
        return out
    except Exception as e:                      # corrupt cache -> recompute
        log(f"stats cache unreadable ({e}) — recomputing")
        return None


# =============================================================================
# Post-processing: block sums, groups, rows (all metric math via weight_proj.metrics)
# =============================================================================
def prefix_from_banded(Db: np.ndarray, band: int) -> np.ndarray:
    """Inclusive 2-D prefix sums P of the symmetric delta-Gram (zeros off-band).

    Every window block lies entirely within the band, and inclusion-exclusion of
    prefix rectangles cancels all off-block cells exactly, so the zero fill is safe.
    """
    nd = Db.shape[0]
    Dfull = np.zeros((nd, nd), dtype=np.float64)
    for lag in range(min(band, nd)):
        idx = np.arange(lag, nd)
        Dfull[idx, idx - lag] = Db[idx, lag]
        if lag:
            Dfull[idx - lag, idx] = Db[idx, lag]
    Ppre = np.zeros((nd + 1, nd + 1), dtype=np.float64)
    np.cumsum(np.cumsum(Dfull, axis=0), axis=1, out=Ppre[1:, 1:])
    return Ppre


def cell_window_sums(Ppre: np.ndarray, delta: int, h: int, n_ticks: int):
    """(S_AA, S_AB, S_BB) arrays over windows t = delta .. n_ticks-1-h (vectorized)."""
    t = np.arange(delta, n_ticks - h)
    if t.size == 0:
        z = np.zeros(0)
        return z, z, z
    a0, a1, b0, b1 = t - delta, t, t, t + h    # A = [t-delta, t), B = [t, t+h)
    S = lambda r0, r1, c0, c1: (Ppre[r1, c1] - Ppre[r0, c1]
                                - Ppre[r1, c0] + Ppre[r0, c0])
    return S(a0, a1, a0, a1), S(a0, a1, b0, b1), S(b0, b1, b0, b1)


def surrogate_metric_row(e2: float, b2: float, eb: float) -> dict:
    """metrics.full_metric_row on R^2 surrogates with EXACT (||e||,||b||,<e,b>) geometry.

    theta_now' = 0, theta_stale' = b_hat, theta_hat' = e_hat, so inside metrics.py:
    e = e_hat, b = b_hat — the ratio/skill/dir_cos/radial/tangential arithmetic is
    the metrics.py code verbatim, including its DENOM_EPS NaN guard (the only
    permitted NaN source).
    """
    b2 = max(b2, 0.0)
    e2 = max(e2, 0.0)
    nb = math.sqrt(b2)
    if nb > 0.0:
        r = eb / nb
        e_hat = np.array([r, math.sqrt(max(e2 - r * r, 0.0))])
    else:
        e_hat = np.array([math.sqrt(e2), 0.0])
    b_hat = np.array([nb, 0.0])
    return M.full_metric_row(e_hat, np.zeros(2), b_hat, None)


def _pcts(vals: np.ndarray) -> tuple:
    fin = vals[np.isfinite(vals)]
    if fin.size == 0:
        return (float("nan"),) * 3
    p = np.percentile(fin, [50.0, 10.0, 90.0])
    return float(p[0]), float(p[1]), float(p[2])


def build_groups(names: list[str]) -> list[dict]:
    """Enumerate every reporting group with its structure axes (plan aggregates)."""
    cls = {n: ST.classify(n) for n in names}
    axes0 = {"matrix_name": None, "layer_idx": None, "special": None,
             "block_type": None, "super_block": None}
    groups: list[dict] = [
        {"kind": "global", "key": "all", "members": list(names), **axes0}]
    for n in names:                                   # per-matrix rows
        c = cls[n]
        groups.append({"kind": "matrix", "key": n, "members": [n],
                       "matrix_name": n, "layer_idx": c["layer_idx"],
                       "special": c["special"], "block_type": c["block_type"],
                       "super_block": c["super_block"]})
    by_bt: dict = {}
    by_sb: dict = {}
    by_layer: dict = {}
    by_lb: dict = {}
    for n in names:
        c = cls[n]
        by_bt.setdefault(c["block_type"], []).append(n)
        by_sb.setdefault(c["super_block"], []).append(n)
        if c["layer_idx"] is not None:
            by_layer.setdefault(c["layer_idx"], []).append(n)
            by_lb.setdefault((c["layer_idx"], c["block_type"]), []).append(n)
    for bt, mem in sorted(by_bt.items()):
        groups.append({"kind": "block_type", "key": bt, "members": mem, **axes0,
                       "block_type": bt, "super_block": ST._SB_OF_BT[bt]})
    for sb, mem in sorted(by_sb.items()):
        groups.append({"kind": "super_block", "key": sb, "members": mem, **axes0,
                       "super_block": sb})
    for li, mem in sorted(by_layer.items()):
        groups.append({"kind": "layer", "key": str(li), "members": mem, **axes0,
                       "layer_idx": li})
    for (li, bt), mem in sorted(by_lb.items()):
        groups.append({"kind": "layer_block", "key": f"L{li}.{bt}", "members": mem,
                       **axes0, "layer_idx": li, "block_type": bt,
                       "super_block": ST._SB_OF_BT[bt]})
    for sp in ("embed", "norm", "bias"):              # explicit special-group rows
        mem = by_bt.get(sp, [])
        if mem:
            groups.append({"kind": "special", "key": sp, "members": mem, **axes0,
                           "special": sp, "block_type": sp, "super_block": sp})
    return groups


def group_diagnostics(members: list[str], stats: dict, n_ticks: int) -> dict:
    """Method-agnostic trajectory diagnostics, exact over the concatenated group."""
    nd = n_ticks - 1
    d0 = np.zeros(nd)
    d1 = np.zeros(nd)
    nnz = np.zeros(nd)
    d_tot = 0
    sum_phi2 = v2 = wc2 = 0.0
    for m in members:
        s = stats[m]
        d0 += s["D"][:, 0]
        d1 += s["D"][:, 1]
        nnz += s["nnz"]
        d_tot += s["d"]
        sum_phi2 += s["sum_phi2"]
        v2 += s["v2"]
        wc2 += s["wc2"]
    ss_tot = sum_phi2 - v2 / n_ticks
    s_tt = n_ticks * (n_ticks ** 2 - 1) / 12.0
    ss_reg = wc2 / s_tt
    traj_r2 = float(ss_reg / ss_tot) if ss_tot > 1e-300 else float("nan")
    den = np.sqrt(d0[:-1] * d0[1:])
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = np.where(den > 0, d1[1:] / den, np.nan)
    delta_norms = np.sqrt(np.maximum(d0, 0.0))
    return {"traj_r2": traj_r2,
            "consec_delta_cos": _pcts(cos)[0],
            "delta_norm": _pcts(delta_norms)[0],
            "coverage": _pcts(nnz / max(d_tot, 1))[0]}


def group_r2(members: list[str], stats: dict):
    """Group per-scalar linearity R²: merge member 100-bin histograms (additive,
    no trace access) -> (r2_median, r2_frac_gt_0.7, n_excluded_const, merged_hist)."""
    hist = np.zeros(R2_HIST_BINS, dtype=np.int64)
    n_excl = 0
    for m in members:
        s = stats[m]
        hist = hist + s["r2_hist"]
        n_excl += int(s["n_excluded_const"])
    med, frac = r2_from_hist(hist)
    return med, frac, n_excl, hist


def _oos_fit_end(j: int, h: int) -> int:
    """Last causal fit-window index for SCORED window j (anchor position j): every fit
    window j' has scoring point (j'+h) STRICTLY < the anchor j. Returns -1 (warm-up) if
    the causal set is empty. The assert IS the OOS leakage guard — a fit end reaching
    the anchor trips it (mirrors predictors.fit_score_split's max(fit)<score contract)."""
    fit_end = j - h - 1
    if fit_end >= 0:
        assert fit_end + h < j, (
            f"OOS LEAKAGE: fit scoring point {fit_end + h} not strictly < anchor {j}")
    return fit_end


def damped_cell(saa, sab, sbb, delta: int, h: int, lam_grid) -> dict:
    """OOS walk-forward damped scoring for one (group, delta, h) cell.

    For each SCORED window j (anchor position j, 0..nw-1): select lambda* =
    argmin over the grid of the summed ||e(lambda)||^2 on the causal fit set
    {j' : j'+h < j} (leakage-guarded via _oos_fit_end), then score window j with
    kappa = lambda*·h/delta through the surrogate metric path (contract-faithful).
    Warm-up windows (empty fit set) get NaN ratio (dropped from the median) and are
    counted. Also returns the in-sample (no-split) median-ratio-vs-lambda curve for
    the lambda-selection visual + the in-sample oracle lambda (the honesty gap)."""
    nw = int(saa.size)
    lams = np.asarray(lam_grid, dtype=np.float64)
    L = lams.size
    kk = lams * (float(h) / float(delta))                    # (L,) kappa per lambda
    # per-window per-lambda ||e||^2 == DampedLinear.window_stats block-sum path
    e2 = (kk[:, None] ** 2) * saa[None, :] + sbb[None, :] - 2.0 * kk[:, None] * sab[None, :]
    e2 = np.maximum(e2, 0.0)                                  # (L, nw)
    nb = np.sqrt(np.maximum(sbb, 0.0))                        # (nw,)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio_lw = np.where(nb[None, :] > M.DENOM_EPS, np.sqrt(e2) / nb[None, :], np.nan)
    insample_med = np.full(L, np.nan)
    for l in range(L):
        fin = ratio_lw[l][np.isfinite(ratio_lw[l])]
        if fin.size:
            insample_med[l] = float(np.median(fin))
    if np.all(np.isnan(insample_med)):
        lam_oracle, oracle_med = float("nan"), float("nan")
    else:
        oi = int(np.nanargmin(insample_med))
        lam_oracle, oracle_med = float(lams[oi]), float(insample_med[oi])
    cumE = np.cumsum(e2, axis=1)                             # (L, nw) cumulative fit err
    ratios = np.full(nw, np.nan); skills = np.full(nw, np.nan)
    dcoss = np.full(nw, np.nan); stales = np.full(nw, np.nan); projs = np.full(nw, np.nan)
    lam_star = np.full(nw, np.nan); n_warmup = 0
    for j in range(nw):
        fit_end = _oos_fit_end(j, h)
        if fit_end < 0:
            n_warmup += 1
            continue
        li = int(np.argmin(cumE[:, fit_end]))
        lam_star[j] = lams[li]
        r = surrogate_metric_row(e2[li, j], sbb[j], sbb[j] - kk[li] * sab[j])
        ratios[j] = r["weight_proj_ratio"]; skills[j] = r["skill"]
        dcoss[j] = r["dir_cos"]; stales[j] = r["base_norm"]; projs[j] = r["err_norm"]
    lam_star_med = float(np.nanmedian(lam_star)) if np.any(np.isfinite(lam_star)) else float("nan")
    return {"ratios": ratios, "skills": skills, "dcoss": dcoss, "stales": stales,
            "projs": projs, "lam_star": lam_star, "lam_star_med": lam_star_med,
            "n_warmup": n_warmup, "n_oos_scored": nw - n_warmup,
            "lam_oracle": lam_oracle, "oracle_med": oracle_med,
            "insample": (lams.tolist(), [None if not np.isfinite(v) else float(v)
                                         for v in insample_med])}


def compute_rows(stats: dict, names: list[str], n_ticks: int, band: int,
                 methods: list[str], deltas: list[int], hs: list[int],
                 op_point: tuple[int, int], also_points: list[tuple[int, int]],
                 cadence: str = "per-tick", unit: str = "tick", lam_grid=None):
    """The full scorecard: atomic rows + per-window ratio store (for h*/visuals).

    #47 additions: `damped_linear` rows use the OOS walk-forward selector (damped_cell);
    every group carries its per-scalar linearity R² (r2_median / r2_frac_gt_0.7 /
    n_excluded_const, method-independent); every row is tagged with cadence/unit and the
    two-anchor descriptor {anchor_mode='fixed', delta_resolved=delta, beta=1+h/delta}.
    Returns (rows, ratio_store, lam_select) where lam_select[(delta,h)] = (lams, med)
    is the GLOBAL-group in-sample lambda-selection curve for the visual."""
    if "damped_linear" in methods:
        assert lam_grid is not None and len(lam_grid) > 0, "damped_linear needs --lam-grid"
    groups = build_groups(names)
    # per-matrix prefix sums once; per-matrix per-cell block sums once
    cells = [(d, h) for d in deltas for h in hs]
    per_matrix: dict = {}
    for n in names:
        Ppre = prefix_from_banded(stats[n]["D"], band)
        per_matrix[n] = {c: cell_window_sums(Ppre, c[0], c[1], n_ticks) for c in cells}
        del Ppre
    rows: list[dict] = []
    ratio_store: dict = {}                       # (method, delta, h, kind, key) -> ratios
    lam_select: dict = {}                        # (delta,h) -> (lams, insample_med) [GLOBAL]
    for g in groups:
        diag = group_diagnostics(g["members"], stats, n_ticks)
        gr2_med, gr2_frac, gr2_nexcl, _ = group_r2(g["members"], stats)   # per-scalar R²
        is_global = (g["kind"] == "global")
        gsums = {}
        for c in cells:
            saa = sab = sbb = None
            for m in g["members"]:
                a, x, b = per_matrix[m][c]
                if saa is None:
                    saa, sab, sbb = a.copy(), x.copy(), b.copy()
                else:
                    saa += a
                    sab += x
                    sbb += b
            gsums[c] = (saa, sab, sbb)
        for mname in methods:
            meth = _REGISTRY[mname]
            med_by_dh: dict = {}
            cell_row_map: dict = {}
            for (delta, h) in cells:
                saa, sab, sbb = gsums[(delta, h)]
                nw_expected = n_ticks - h - delta
                in_bounds = nw_expected > 0
                nw = int(saa.size)
                assert nw == max(nw_expected, 0), \
                    f"n_windows {nw} != {nw_expected} for cell ({delta},{h})"
                dmp = None
                if mname == "damped_linear":
                    dmp = damped_cell(saa, sab, sbb, delta, h, lam_grid)
                    ratios, skills = dmp["ratios"], dmp["skills"]
                    dcoss, stales, projs = dmp["dcoss"], dmp["stales"], dmp["projs"]
                    if is_global:
                        lam_select[(delta, h)] = dmp["insample"]
                else:
                    ratios = np.full(nw, np.nan)
                    skills = np.full(nw, np.nan)
                    dcoss = np.full(nw, np.nan)
                    stales = np.full(nw, np.nan)
                    projs = np.full(nw, np.nan)
                    for j in range(nw):
                        e2, b2, eb = meth.window_stats(saa[j], sab[j], sbb[j], delta, h)
                        r = surrogate_metric_row(e2, b2, eb)
                        ratios[j] = r["weight_proj_ratio"]
                        skills[j] = r["skill"]
                        dcoss[j] = r["dir_cos"]
                        stales[j] = r["base_norm"]
                        projs[j] = r["err_norm"]
                rm, r10, r90 = _pcts(ratios)
                sm, s10, s90 = _pcts(skills)
                row = {
                    "method": mname, "delta_ticks": delta, "h_ticks": h,
                    "group_kind": g["kind"], "group_key": str(g["key"]),
                    "matrix_name": g["matrix_name"], "layer_idx": g["layer_idx"],
                    "special": g["special"], "block_type": g["block_type"],
                    "super_block": g["super_block"],
                    "stale_error_median": _pcts(stales)[0],
                    "proj_error_median": _pcts(projs)[0],
                    "weight_proj_ratio_median": rm,
                    "weight_proj_ratio_p10": r10, "weight_proj_ratio_p90": r90,
                    "skill_median": sm, "skill_p10": s10, "skill_p90": s90,
                    "dir_cos_median": _pcts(dcoss)[0],
                    "traj_r2": diag["traj_r2"],
                    "consec_delta_cos": diag["consec_delta_cos"],
                    "delta_norm": diag["delta_norm"], "coverage": diag["coverage"],
                    "n_windows": nw, "in_bounds": bool(in_bounds),
                    "n_nan_windows": int(np.sum(~np.isfinite(ratios))),
                    "h_star": None, "best_delta": None,
                    "tied": False,
                    # ---- #47 superset fields (uniform schema across methods) ----
                    "cadence": cadence, "unit": unit,
                    "anchor_mode": "fixed", "delta_resolved": delta,
                    "beta": 1.0 + float(h) / float(delta),
                    "r2_median": gr2_med, "r2_frac_gt_0.7": gr2_frac,
                    "n_excluded_const": gr2_nexcl,
                    "lam_star": (dmp["lam_star_med"] if dmp else None),
                    "lam_oracle": (dmp["lam_oracle"] if dmp else None),
                    "ratio_oracle_median": (dmp["oracle_med"] if dmp else None),
                    "n_warmup": (dmp["n_warmup"] if dmp else 0),
                    "n_oos_scored": (dmp["n_oos_scored"] if dmp else nw),
                }
                rows.append(row)
                cell_row_map[(delta, h)] = row
                ratio_store[(mname, delta, h, g["kind"], str(g["key"]))] = ratios
                med_by_dh[(delta, h)] = rm
            # derived: h* per (method, delta, group) — REUSES metrics.crossover_hstar;
            # best_delta = argmin over Delta of the median ratio at the operating h
            for delta in deltas:
                h_to = {h: [v for v in
                            ratio_store[(mname, delta, h, g["kind"], str(g["key"]))]
                            if np.isfinite(v)] for h in hs}
                hstar = int(M.crossover_hstar(h_to))
                for h in hs:
                    cell_row_map[(delta, h)]["h_star"] = hstar
            op_h = op_point[1] if op_point[1] in hs else max(hs)
            cand = [(med_by_dh.get((d, op_h), float("nan")), d) for d in deltas]
            fin = [(v, d) for v, d in cand if np.isfinite(v)]
            best_delta = min(fin, key=lambda x: (x[0], x[1]))[1] if fin else None
            for row in cell_row_map.values():
                row["best_delta"] = best_delta
    # synthesized TIED lm_head rows from the embed special rows (rule 6)
    lm_rows = [ST.synthesize_tied_lm_head(r) for r in rows
               if r["group_kind"] == "special" and r["group_key"] == "embed"]
    rows.extend(lm_rows)
    for (mname, delta, h, kind, key), ratios in list(ratio_store.items()):
        if kind == "special" and key == "embed":
            ratio_store[(mname, delta, h, "special", "lm_head")] = ratios
    return rows, ratio_store, lam_select


def _spearman(x: list, y: list) -> float:
    """Spearman rank correlation over paired finite (x,y); NaN if < 3 pairs."""
    xy = [(a, b) for a, b in zip(x, y)
          if a is not None and b is not None and np.isfinite(a) and np.isfinite(b)]
    if len(xy) < 3:
        return float("nan")
    xa = np.array([p[0] for p in xy]); ya = np.array([p[1] for p in xy])
    rx = np.argsort(np.argsort(xa)).astype(float)
    ry = np.argsort(np.argsort(ya)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    den = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / den) if den > 0 else float("nan")


def build_visuals(rows: list[dict], methods: list[str],
                  deltas: list[int], hs: list[int], op: tuple[int, int],
                  lam_select: dict | None = None, stats: dict | None = None,
                  paper_panel: dict | None = None) -> dict:
    idx = {(r["method"], r["delta_ticks"], r["h_ticks"],
            r["group_kind"], r["group_key"]): r for r in rows}
    op_d, op_h = op
    gget = lambda m, d, h: idx.get((m, d, h, "global", "all"), {})
    # paper_linear is off the (delta x h) grid (delta derived per window) -> excluded
    # from the grid visuals a-g; it gets its own m_paper_equivalence panel instead.
    gm = [m for m in methods if m != "paper_linear"]
    vis = {
        "a_accuracy_vs_horizon": {
            m: {"delta": op_d, "h": hs,
                "ratio_median": [gget(m, op_d, h).get("weight_proj_ratio_median") for h in hs],
                "ratio_p10": [gget(m, op_d, h).get("weight_proj_ratio_p10") for h in hs],
                "ratio_p90": [gget(m, op_d, h).get("weight_proj_ratio_p90") for h in hs]}
            for m in gm},
        "b_delta_sensitivity": {
            m: {"h": op_h, "delta": deltas,
                "ratio_median": [gget(m, d, op_h).get("weight_proj_ratio_median") for d in deltas]}
            for m in gm},
        "c_target_horizon_sweep": {
            m: {str(d): {"h": hs,
                         "ratio_median": [gget(m, d, h).get("weight_proj_ratio_median") for h in hs]}
                for d in deltas}
            for m in gm},
    }
    mat_rows = [r for r in rows if r["group_kind"] == "matrix"
                and r["method"] == gm[0] and r["delta_ticks"] == deltas[0]
                and r["h_ticks"] == hs[0]]
    lb_keys = sorted({r["block_type"] for r in rows if r["group_kind"] == "layer_block"})
    layers = sorted({r["layer_idx"] for r in rows if r["group_kind"] == "layer_block"})
    lb = lambda m, d, h, li, bt, field: idx.get(
        (m, d, h, "layer_block", f"L{li}.{bt}"), {}).get(field)
    vis["d_traj_r2"] = {
        "per_matrix": [{"matrix_name": r["matrix_name"], "layer_idx": r["layer_idx"],
                        "block_type": r["block_type"], "traj_r2": r["traj_r2"]}
                       for r in mat_rows],
        "depth_block_heatmap": {
            "layers": layers, "block_types": lb_keys,
            "traj_r2": [[lb(gm[0], deltas[0], hs[0], li, bt, "traj_r2")
                         for bt in lb_keys] for li in layers]},
    }
    vis["e_ratio_heatmap"] = {
        m: {"operating_point": [op_d, op_h], "layers": layers, "block_types": lb_keys,
            "ratio_median": [[lb(m, op_d, op_h, li, bt, "weight_proj_ratio_median")
                              for bt in lb_keys] for li in layers]}
        for m in gm}
    vis["f_hstar_heatmap"] = {
        m: {"delta": op_d, "layers": layers, "block_types": lb_keys,
            "h_star": [[lb(m, op_d, op_h, li, bt, "h_star")
                        for bt in lb_keys] for li in layers]}
        for m in gm}
    specials = ["embed", "norm", "bias", "lm_head"]
    vis["g_special_groups"] = {
        m: [dict(idx.get((m, op_d, op_h, "special", sp), {}), group=sp)
            for sp in specials]
        for m in gm}

    # ---- #47 h: lambda-selection (in-sample median ratio vs lambda per (delta,h)) ----
    lam_select = lam_select or {}
    vis["h_lambda_selection"] = {
        "operating_point": [op_d, op_h],
        "cells": {f"{d},{h}": {"lambda": lam_select[(d, h)][0],
                               "ratio_median": lam_select[(d, h)][1]}
                  for (d, h) in sorted(lam_select)}}

    # ---- #47 i: per-scalar R² histogram (global + per super_block) ----
    r2_hist_data = {}
    if stats is not None:
        allnames = sorted({r["matrix_name"] for r in rows if r["group_kind"] == "matrix"})
        _, _, _, ghist = group_r2(allnames, stats)
        r2_hist_data["global"] = {"bins": R2_HIST_BINS,
                                  "counts": [int(x) for x in ghist]}
        by_sb: dict = {}
        for r in rows:
            if r["group_kind"] == "matrix":
                by_sb.setdefault(r["super_block"], []).append(r["matrix_name"])
        r2_hist_data["by_super_block"] = {}
        for sb, mem in sorted(by_sb.items()):
            _, _, _, hh = group_r2(mem, stats)
            r2_hist_data["by_super_block"][sb] = [int(x) for x in hh]
    vis["i_r2_histogram"] = r2_hist_data

    # ---- #47 j: depth x block per-scalar R² heatmap (DISTINCT from d_traj_r2) ----
    vis["j_r2_depth_block_heatmap"] = {
        "layers": layers, "block_types": lb_keys,
        "r2_median": [[lb(gm[0], deltas[0], hs[0], li, bt, "r2_median")
                       for bt in lb_keys] for li in layers]}

    # ---- #47 k: R²-vs-ratio coupling (per-group median R² vs OOS-damped op-point ratio) ----
    coup_method = "damped_linear" if "damped_linear" in methods else gm[0]
    pts = []
    for r in rows:
        if (r["method"] == coup_method and r["delta_ticks"] == op_d
                and r["h_ticks"] == op_h
                and r["group_kind"] in ("block_type", "super_block", "layer")):
            pts.append({"group_kind": r["group_kind"], "group_key": r["group_key"],
                        "r2_median": r["r2_median"],
                        "ratio_median": r["weight_proj_ratio_median"]})
    vis["k_r2_ratio_coupling"] = {
        "method": coup_method, "operating_point": [op_d, op_h], "points": pts,
        "spearman": _spearman([p["r2_median"] for p in pts],
                              [p["ratio_median"] for p in pts])}

    # ---- #47 m: paper-equivalence panel (regime S only; passed in by run_emit) ----
    if paper_panel is not None:
        vis[VISUAL_KEY_PAPER] = paper_panel
    return vis


# =============================================================================
# paper_linear — the Wang et al. 2026 weight-space extrapolation protocol arm
# (regime S ONLY; direct-scored OUTSIDE the banded cache; delta grows with t)
# =============================================================================
def _paper_windows(n_ticks: int, hs: list[int], anchor_frac: float, stride: int):
    """(h, t, t0, delta_resolved) windows: t0=floor(frac*t), t>=20, strided anchors,
    t+h<=n_ticks-1. Asserts the App. E.1 anchor rule 0.20 <= t0/t <= 0.30."""
    windows = []
    needed: set[int] = set()
    for h in hs:
        for t in range(20, n_ticks - h, stride):
            t0 = int(math.floor(anchor_frac * t))
            if t0 < 1:
                continue
            frac_res = t0 / t
            assert 0.20 <= frac_res <= 0.30, (
                f"paper anchor t0/t={frac_res:.3f} outside [0.20,0.30] at t={t} "
                f"(t>=20 required; frac={anchor_frac})")
            windows.append((h, t, t0, t - t0))
            needed.update((t0, t, t + h))
    return windows, sorted(needed)


def compute_paper_rows(reader, names, dims, n_ticks, hs, groups, anchor_frac,
                       stride, stats, cadence, unit, ram_gb=40.0):
    """Direct-score paper_linear over mmap slice reads. theta_hat = theta_t +
    (h/delta_resolved)*(theta_t - theta_{t0}) — the SAME Order1 secant as naive_linear
    with delta=delta_resolved (t0=floor(frac*t)). Its cells NEVER enter the banded
    stats_cache and MUST NOT change `band`. Returns (paper_rows, paper_panel)."""
    windows, needed = _paper_windows(n_ticks, hs, anchor_frac, stride)
    log(f"paper_linear: {len(windows)} windows over h={hs}, {len(needed)} unique steps, "
        f"frac={anchor_frac}, stride={stride}")
    name_dims = [(n, dims[n]) for n in names]
    cap = max(int(ram_gb * 1e9 / (max(len(needed), 1) * 8)), 1_000_000)
    chunks = plan_chunks(name_dims, cap)
    acc: dict = {}                       # (matrix, h, t) -> np.array([e2, b2, eb])
    for ci, chunk in enumerate(chunks):
        tc = time.time()
        buf: dict = {}
        for step in needed:
            sd = reader.load_raw(step)
            buf[step] = {i: reader.slice_f64(sd, u.name, u.a, u.b)
                         for i, u in enumerate(chunk)}
            del sd
        for i, u in enumerate(chunk):
            for (h, t, t0, dres) in windows:
                a = buf[t0][i]; tt = buf[t][i]; s = buf[t + h][i]
                k = float(h) / float(dres)
                e = (tt + k * (tt - a)) - s        # Order1 secant residual
                b = tt - s                          # stale baseline displacement
                trip = np.array([float(e @ e), float(b @ b), float(e @ b)])
                key = (u.name, h, t)
                acc[key] = acc[key] + trip if key in acc else trip
        buf = None
        log(f"paper_linear: chunk {ci + 1}/{len(chunks)} done in {time.time() - tc:.1f}s")
    win_by_h: dict = {}
    for (h, t, t0, dres) in windows:
        win_by_h.setdefault(h, []).append((t, t0, dres))
    paper_rows: list[dict] = []
    panel = {"operating_h": None, "h": list(hs), "betas_by_h": {}, "ratio_by_h": {}}
    for g in groups:
        diag = group_diagnostics(g["members"], stats, n_ticks)
        gr2_med, gr2_frac, gr2_nexcl, _ = group_r2(g["members"], stats)
        for h in hs:
            wl = win_by_h.get(h, [])
            ratios = np.full(len(wl), np.nan); skills = np.full(len(wl), np.nan)
            dcoss = np.full(len(wl), np.nan); stales = np.full(len(wl), np.nan)
            projs = np.full(len(wl), np.nan)
            betas = []; dress = []
            for j, (t, t0, dres) in enumerate(wl):
                e2 = b2 = eb = 0.0
                for m in g["members"]:
                    v = acc.get((m, h, t))
                    if v is not None:
                        e2 += v[0]; b2 += v[1]; eb += v[2]
                r = surrogate_metric_row(e2, b2, eb)
                ratios[j] = r["weight_proj_ratio"]; skills[j] = r["skill"]
                dcoss[j] = r["dir_cos"]; stales[j] = r["base_norm"]; projs[j] = r["err_norm"]
                betas.append(1.0 + float(h) / float(dres)); dress.append(dres)
            rm, r10, r90 = _pcts(ratios)
            sm, s10, s90 = _pcts(skills)
            row = {
                "method": "paper_linear", "delta_ticks": PAPER_SENTINEL_DELTA,
                "h_ticks": h, "group_kind": g["kind"], "group_key": str(g["key"]),
                "matrix_name": g["matrix_name"], "layer_idx": g["layer_idx"],
                "special": g["special"], "block_type": g["block_type"],
                "super_block": g["super_block"],
                "stale_error_median": _pcts(stales)[0], "proj_error_median": _pcts(projs)[0],
                "weight_proj_ratio_median": rm,
                "weight_proj_ratio_p10": r10, "weight_proj_ratio_p90": r90,
                "skill_median": sm, "skill_p10": s10, "skill_p90": s90,
                "dir_cos_median": _pcts(dcoss)[0],
                "traj_r2": diag["traj_r2"], "consec_delta_cos": diag["consec_delta_cos"],
                "delta_norm": diag["delta_norm"], "coverage": diag["coverage"],
                "n_windows": len(wl), "in_bounds": bool(len(wl) > 0),
                "n_nan_windows": int(np.sum(~np.isfinite(ratios))),
                "h_star": None, "best_delta": None, "tied": False,
                "cadence": cadence, "unit": unit,
                "anchor_mode": "frac25",
                "delta_resolved": (float(np.median(dress)) if dress else None),
                "delta_resolved_min": (int(np.min(dress)) if dress else None),
                "delta_resolved_max": (int(np.max(dress)) if dress else None),
                "beta": (float(np.median(betas)) if betas else None),
                "beta_min": (float(np.min(betas)) if betas else None),
                "beta_max": (float(np.max(betas)) if betas else None),
                "r2_median": gr2_med, "r2_frac_gt_0.7": gr2_frac,
                "n_excluded_const": gr2_nexcl, "lam_star": None,
                "lam_oracle": None, "ratio_oracle_median": None,
                "n_warmup": 0, "n_oos_scored": len(wl),
            }
            paper_rows.append(row)
            if g["kind"] == "global":
                panel["betas_by_h"][str(h)] = [float(x) for x in betas]
                panel["ratio_by_h"][str(h)] = rm
    return paper_rows, panel


# =============================================================================
# Gates (printed as `GATE <name>: PASS|FAIL` — the analyst greps these)
# =============================================================================
def run_gates(rows: list[dict], part: dict, n_ticks: int, methods: list[str],
              deltas: list[int], hs: list[int],
              op: tuple[int, int], also: list[tuple[int, int]]) -> dict:
    gates: dict[str, tuple[bool, str]] = {}
    hold = [r for r in rows if r["method"] == "hold_stale" and r["in_bounds"]
            and not r["tied"]]
    if hold:
        bad = [r for r in hold
               if not (abs(r["weight_proj_ratio_median"] - 1.0) <= IDENTITY_TOL
                       and abs(r["skill_median"]) <= IDENTITY_TOL)]
        worst = max((abs(r["weight_proj_ratio_median"] - 1.0) for r in hold
                     if np.isfinite(r["weight_proj_ratio_median"])), default=float("nan"))
        gates["hold_stale_identity"] = (
            not bad, f"{len(hold)} rows, worst |ratio-1| = {worst:.3e}, "
                     f"violations = {len(bad)}")
    else:
        gates["hold_stale_identity"] = (False, "no hold_stale rows emitted")
    gates["structure_partition"] = (part["ok"],
                                    "; ".join(part["failures"]) or
                                    f"338-partition exact "
                                    f"(other={part['block_type_counts'].get('other', 0)})")
    nl = [r for r in rows if r["method"] == "naive_linear" and r["in_bounds"]]
    n_nan = sum(r["n_nan_windows"] for r in nl)
    gates["naive_linear_finite"] = (
        bool(nl) and n_nan == 0,
        f"{len(nl)} rows, denom-guard NaN windows = {n_nan} (only permitted NaN)")
    # bounds_honesty: paper_linear is OFF the (delta x h) banded grid (delta derived
    # per window, sentinel delta_ticks=0) — its n_windows is the strided-anchor count,
    # so it is exempt from the n_ticks-h-delta formula (damped stays IN, n_windows==nw).
    grid_rows = [r for r in rows if r["method"] != "paper_linear"]
    bad_nw = [r for r in grid_rows if r["in_bounds"] and not r["tied"]
              and r["n_windows"] != n_ticks - r["h_ticks"] - r["delta_ticks"]]
    zero_nw = [r for r in grid_rows if r["in_bounds"] and r["n_windows"] <= 0]
    gates["bounds_honesty"] = (
        not bad_nw and not zero_nw,
        f"n_windows == n_ticks-h-delta on all in-bounds grid rows "
        f"(violations={len(bad_nw)}, zero={len(zero_nw)})")
    have = {(r["group_kind"], r["group_key"]) for r in rows}
    need_specials = {("special", s) for s in ("embed", "norm", "bias", "lm_head")}
    miss = [k for k in ({("global", "all")} | need_specials) if k not in have]
    lm = [r for r in rows if r["group_kind"] == "special" and r["group_key"] == "lm_head"]
    gates["aggregates_present"] = (
        not miss and bool(lm) and all(r["tied"] for r in lm),
        f"missing={miss}; lm_head rows={len(lm)} (tied)")
    no_cov = [r for r in rows if not (np.isfinite(r["delta_norm"])
                                      and np.isfinite(r["coverage"]))]
    gates["coverage_safety"] = (
        not no_cov, f"rows missing delta_norm/coverage = {len(no_cov)}")
    ops = [op] + list(also)
    grid_methods = [m for m in methods if m != "paper_linear"]
    row_keys = {(r["method"], r["delta_ticks"], r["h_ticks"], r["group_kind"],
                 r["group_key"]) for r in rows}
    miss_op = [(m, p) for m in grid_methods for p in ops
               if (m, p[0], p[1], "global", "all") not in row_keys]
    gates["operating_points"] = (not miss_op, f"missing={miss_op}")
    # #47: per-scalar R² well-defined on every group row (in [0,1] or None; never NaN leak)
    r2bad = [r for r in rows if r["r2_median"] is not None
             and not (0.0 <= r["r2_median"] <= 1.0)]
    gates["r2_well_defined"] = (
        not r2bad, f"per-scalar r2_median in [0,1] on all group rows "
                   f"(out-of-range={len(r2bad)})")
    # #47: paper_linear present for every h at global (regime S only)
    if "paper_linear" in methods:
        miss_paper = [h for h in hs if ("paper_linear", PAPER_SENTINEL_DELTA, h,
                                        "global", "all") not in row_keys]
        pw = [r for r in rows if r["method"] == "paper_linear"
              and r["group_kind"] == "global"]
        gates["paper_linear_present"] = (
            not miss_paper and bool(pw),
            f"global paper rows for h={hs} (missing h={miss_paper}); "
            f"{len(pw)} global paper rows, anchor_mode="
            f"{sorted({r['anchor_mode'] for r in pw})}")
    for name, (ok, detail) in gates.items():
        print(f"GATE {name}: {'PASS' if ok else 'FAIL'} — {detail}", flush=True)
    return {k: {"pass": bool(v[0]), "detail": v[1]} for k, v in gates.items()}


# =============================================================================
# Off-path parity: surrogate-path rows vs DIRECT predictors+metrics on real tensors
# =============================================================================
def parity_check(reader, stats: dict, names: list[str], n_ticks: int, band: int,
                 cells: list[tuple[int, int]], methods: list[str]) -> tuple[bool, list]:
    """For sampled (matrix, cell, window): recompute via predictors.Order1.predict on
    the ACTUAL loaded snapshot vectors + metrics.full_metric_row, and compare with the
    engine's surrogate-path row. Asserts the metric-contract pin quantitatively."""
    results = []
    ok = True
    for name in names:
        Ppre = prefix_from_banded(stats[name]["D"], band)
        for (delta, h) in cells:
            saa, sab, sbb = cell_window_sums(Ppre, delta, h, n_ticks)
            if saa.size == 0:
                continue
            for j in sorted({0, saa.size // 2, saa.size - 1}):
                t = delta + j
                assert t - delta >= 0 and t + h <= n_ticks - 1, "window out of causal bounds"
                th_a = reader.load_matrix_f64(t - delta, name)
                th_t = reader.load_matrix_f64(t, name)
                th_s = reader.load_matrix_f64(t + h, name)
                history = [(t - delta, _to_torch(th_a)), (t, _to_torch(th_t))]
                assert max(tick for tick, _ in history) <= t < t + h, "causality violated"
                for mname in methods:
                    meth = _REGISTRY[mname]
                    theta_hat = meth.predict(history, delta, h)
                    # numpy f64 in -> metrics._f32 keeps f64 (no fp32 re-quantization),
                    # so direct-vs-surrogate agreement is float-assoc-level exact
                    direct = M.full_metric_row(theta_hat.numpy(), th_s, th_t, None)
                    e2, b2, eb = meth.window_stats(saa[j], sab[j], sbb[j], delta, h)
                    surro = surrogate_metric_row(e2, b2, eb)
                    worst = 0.0
                    for fld in ("err_norm", "base_norm", "weight_proj_ratio",
                                "skill", "dir_cos"):
                        dv, sv = direct[fld], surro[fld]
                        if math.isnan(dv) and math.isnan(sv):
                            continue
                        if abs(dv - sv) <= 1e-9:   # exact-zero fields (e.g. skill at ratio 1)
                            continue
                        rel = abs(dv - sv) / max(abs(dv), abs(sv), 1e-12)
                        worst = max(worst, rel)
                    good = worst <= PARITY_RTOL
                    ok = ok and good
                    results.append({"matrix": name, "cell": [delta, h], "t": t,
                                    "method": mname, "worst_rel": worst, "ok": good})
    return ok, results


def _to_torch(arr64: np.ndarray):
    import torch
    return torch.from_numpy(arr64)


# =============================================================================
# Self-test battery (plan `## Correctness invariants`)
# =============================================================================
def _synthetic_reader(seed: int = 45):
    """Tiny in-memory trace: noisy-linear + EXACTLY-linear + near-static matrices."""
    rng = np.random.default_rng(seed)
    n_ticks, dims = 30, {"syn.a": 96, "syn.b": 64, "syn.linear_exact": 80,
                         "syn.static": 48, "syn.c": 128, "syn.d": 72}
    base = {k: rng.normal(0, 0.02, size=d).astype(np.float32)
            for k, d in dims.items()}
    vel = {k: rng.normal(0, 1e-4, size=d) for k, d in dims.items()}
    vel["syn.static"] *= 1e-6
    ticks = {}
    for t in range(n_ticks):
        sd = {}
        for k, d in dims.items():
            noise = 0.0 if k == "syn.linear_exact" else rng.normal(0, 2e-5, size=d)
            sd[k] = (base[k].astype(np.float64) + t * vel[k] + noise).astype(np.float32)
        ticks[t] = sd
    return InMemoryReader(ticks), list(dims.items()), n_ticks


def _synthetic_reader_r2(seed: int = 47):
    """Self-test trace for the per-scalar R² invariants: an EXACTLY-constant matrix
    (must be excluded + counted), an exactly-linear one (R²≈1), and a noisy one."""
    rng = np.random.default_rng(seed)
    n_ticks = 40
    dims = {"r2.const": 50, "r2.linear": 60, "r2.noisy": 70, "r2.mixed": 40}
    base = {k: rng.normal(0, 0.02, size=d).astype(np.float32) for k, d in dims.items()}
    vel = {k: rng.normal(0, 1e-3, size=d) for k, d in dims.items()}
    vel["r2.const"][:] = 0.0
    ticks = {}
    for t in range(n_ticks):
        sd = {}
        for k, d in dims.items():
            noise = (0.0 if k in ("r2.const", "r2.linear")
                     else rng.normal(0, (2e-3 if k == "r2.noisy" else 5e-4), size=d))
            sd[k] = (base[k].astype(np.float64) + t * vel[k] + noise).astype(np.float32)
        ticks[t] = sd
    return InMemoryReader(ticks), list(dims.items()), n_ticks


def _synthetic_linear_reader(n_real: int = 24):
    """EXACTLY-linear-in-tick trace for the cadence-reindex invariant: theta_tau =
    base + tau*vel, so per-step deltas (tickset [0,2,…]) are EXACTLY 2x per-tick deltas."""
    rng = np.random.default_rng(4747)
    dims = {"lin.a": 40, "lin.b": 32}
    base = {k: rng.normal(0, 0.02, size=d).astype(np.float64) for k, d in dims.items()}
    vel = {k: rng.normal(0, 1e-3, size=d) for k, d in dims.items()}
    ticks = {t: {k: (base[k] + t * vel[k]).astype(np.float32) for k in dims}
             for t in range(n_real)}
    return InMemoryReader(ticks), list(dims.items()), vel


def _block_sums_all(stats, names, n_ticks, band, deltas, hs):
    """Per (group, cell) block sums (saa,sab,sbb) over all groups — for identity tests."""
    groups = build_groups(names)
    per = {}
    for nm in names:
        Pp = prefix_from_banded(stats[nm]["D"], band)
        per[nm] = {(d, h): cell_window_sums(Pp, d, h, n_ticks) for d in deltas for h in hs}
    out = []
    for g in groups:
        for d in deltas:
            for h in hs:
                saa = sab = sbb = None
                for mm in g["members"]:
                    a, x, b = per[mm][(d, h)]
                    if saa is None:
                        saa, sab, sbb = a.copy(), x.copy(), b.copy()
                    else:
                        saa += a; sab += x; sbb += b
                out.append((d, h, saa, sab, sbb))
    return out


def _damped_identity_worst(stats, names, n_ticks, band, deltas, hs):
    """worst |damped(lam=1).e2 - naive.e2| and |damped(lam=0).e2 - hold.e2| over all cells."""
    d1, d0, nai, hld = DampedLinear(1.0), DampedLinear(0.0), NaiveLinear(), HoldStale()
    w1 = w0 = 0.0
    for (d, h, saa, sab, sbb) in _block_sums_all(stats, names, n_ticks, band, deltas, hs):
        if saa.size == 0:
            continue
        e2_d1 = _damped_e2(d1.kappa(d, h), saa, sab, sbb)
        e2_n = _damped_e2(nai.kappa(d, h), saa, sab, sbb)
        e2_d0 = _damped_e2(d0.kappa(d, h), saa, sab, sbb)
        e2_h = _damped_e2(hld.kappa(d, h), saa, sab, sbb)
        w1 = max(w1, float(np.max(np.abs(e2_d1 - e2_n))))
        w0 = max(w0, float(np.max(np.abs(e2_d0 - e2_h))))
    return w1, w0


def _r2_direct(reader, name, n_ticks):
    """Independent per-element R² via numpy polyfit over the loaded trajectory (t=0..N-1)."""
    traj = np.stack([reader.load_matrix_f64(t, name) for t in range(n_ticks)], axis=0)
    tbar = (n_ticks - 1) / 2.0
    ti = np.arange(n_ticks) - tbar
    mu = traj.mean(0)
    ss_tot = ((traj - mu) ** 2).sum(0)
    S_tt = n_ticks * (n_ticks ** 2 - 1) / 12.0
    ss_reg = ((ti[:, None] * (traj - mu)).sum(0)) ** 2 / S_tt
    const = ss_tot <= R2_CONST_EPS
    with np.errstate(invalid="ignore", divide="ignore"):
        r2 = np.where(const, np.nan, np.clip(ss_reg / ss_tot, 0.0, 1.0))
    return r2, const


def run_selftest(args) -> int:
    out_dir = args.out or DEFAULT_OUT
    os.makedirs(out_dir, exist_ok=True)
    inv: dict[str, tuple[bool, str]] = {}

    # -- invariant: metric-contract pin ---------------------------------------
    pin_ok = M.METRIC_CONTRACT == METRIC_CONTRACT_EXPECTED
    inv["metric_contract_pin"] = (
        pin_ok, f"weight_proj.metrics.METRIC_CONTRACT == {M.METRIC_CONTRACT!r}")

    # -- invariant: structure partition on the real manifest ------------------
    if os.path.exists(args.manifest):
        man_names = _manifest_names(args.manifest)
        part = ST.partition(man_names)
        inv["structure_partition"] = (part["ok"],
                                      "; ".join(part["failures"]) or
                                      f"exact 338-partition, other=0 "
                                      f"(bt={part['block_type_counts']})")
    else:
        part = None
        inv["structure_partition"] = (False, f"manifest missing: {args.manifest}")

    # -- invariant: causality / leakage guard ---------------------------------
    lg_ok, lg_detail = P.leakage_guard_selftest()
    fw = fit_window_positions(160, 100, 4)
    wired = max(fw) < 100
    inv["causality_leakage_guard"] = (
        lg_ok and wired,
        f"{lg_detail}; fit_window_positions wired (max fit idx {max(fw)} < anchor 100)")

    # -- synthetic engine battery (identity / exact-linear / parity) ----------
    reader, name_dims, n_ticks_s = _synthetic_reader()
    sdeltas, shs = [2, 3], [1, 2, 4]
    sband = max(sdeltas) + max(shs)
    sstats = stream_stats(reader, name_dims, n_ticks_s, sband, ram_gb=0.02, tag="-syn")
    srows, _, _ = compute_rows(sstats, [n for n, _ in name_dims], n_ticks_s, sband,
                            ["hold_stale", "naive_linear"], sdeltas, shs,
                            (3, 2), [(2, 1)])
    hold = [r for r in srows if r["method"] == "hold_stale" and not r["tied"]]
    worst_id = max(abs(r["weight_proj_ratio_median"] - 1.0) for r in hold)
    worst_sk = max(abs(r["skill_median"]) for r in hold)
    inv["hold_stale_identity_synthetic"] = (
        worst_id <= 1e-9 and worst_sk <= 1e-9,
        f"{len(hold)} rows: worst |ratio-1|={worst_id:.2e}, worst |skill|={worst_sk:.2e}")
    lin = [r for r in srows if r["method"] == "naive_linear"
           and r["group_kind"] == "matrix" and r["group_key"] == "syn.linear_exact"]
    worst_lin = max(r["weight_proj_ratio_median"] for r in lin)
    # trajectory is exactly linear UP TO fp32 storage rounding of theta (~2e-9/elem),
    # so the ratio floors near rounding/||b|| ~ 1e-5, not 0
    inv["naive_linear_exact_trajectory"] = (
        worst_lin <= 1e-3,
        f"exactly-linear matrix: worst naive_linear ratio {worst_lin:.2e} "
        f"(expect ~fp32-rounding floor << 1)")
    p_ok, p_res = parity_check(reader, sstats, ["syn.a", "syn.c"],
                               n_ticks_s, sband, [(2, 1), (3, 4)],
                               ["hold_stale", "naive_linear"])
    worst_p = max((r["worst_rel"] for r in p_res), default=float("nan"))
    inv["offpath_parity_synthetic"] = (
        p_ok, f"{len(p_res)} samples, worst rel diff {worst_p:.2e} "
              f"(tol {PARITY_RTOL:g}) — surrogate path == direct "
              f"predictors.Order1+metrics.full_metric_row")

    # ======================= #47 additive invariants =========================
    # -- damped lambda=1==naive_linear, lambda=0==hold_stale (BOTH regimes) ------
    id_ok, id_det = True, []
    for cad, nt in (("per-tick", 14), ("per-step", 12)):
        ts = TS.select_ticks(cad, nt)
        rc = InMemoryReader(reader.ticks, tickset=ts)
        stc = stream_stats(rc, name_dims, nt, sband, ram_gb=0.05, tag=f"-id-{cad}")
        w1, w0 = _damped_identity_worst(stc, [n for n, _ in name_dims], nt, sband,
                                        sdeltas, shs)
        id_ok = id_ok and w1 <= 1e-9 and w0 <= 1e-9
        id_det.append(f"{cad}: |lam1-naive|={w1:.1e} |lam0-hold|={w0:.1e}")
    inv["damped_lambda_identities"] = (id_ok, "; ".join(id_det))

    # -- cadence reindex: per-step maps step s -> tick 2s; recovers step-spaced deltas
    lreader, lname_dims, lvel = _synthetic_linear_reader(24)
    ps = TS.select_ticks("per-step", 10)          # [0,2,…,18]
    pt = TS.select_ticks("per-tick", 10)          # [0,1,…,9]
    ps_reader = InMemoryReader(lreader.ticks, tickset=ps)
    reindex_ok = all(ps_reader.real_tick(s) == 2 * s for s in range(10))
    present_ok = len(ps_reader.present_ticks(10)) == 10   # all SELECTED ticks present
    # exactly-linear -> per-step consecutive delta == 2x per-tick delta (element-wise)
    d_ps = ps_reader.load_matrix_f64(1, "lin.a") - ps_reader.load_matrix_f64(0, "lin.a")
    d_pt = InMemoryReader(lreader.ticks, tickset=pt).load_matrix_f64(1, "lin.a") - \
        InMemoryReader(lreader.ticks, tickset=pt).load_matrix_f64(0, "lin.a")
    step_delta_ok = np.allclose(d_ps, 2.0 * d_pt, atol=1e-6)
    ndl = [(n, d) for n, d in lname_dims]
    fp_ps = _fingerprint(ndl, 10, 5, "/x", cadence="per-step", tickset=ps)
    fp_pt = _fingerprint(ndl, 10, 5, "/x", cadence="per-tick", tickset=pt)
    fp_distinct = fp_ps != fp_pt
    inv["cadence_reindex"] = (
        reindex_ok and present_ok and step_delta_ok and fp_distinct,
        f"per-step step->tick 2s={reindex_ok}; present-set validated={present_ok}; "
        f"per-step delta==2x per-tick={step_delta_ok}; fingerprint distinct={fp_distinct}")

    # -- OOS leakage guard: _oos_fit_end never leaks; an intentional leak trips assert
    guard_ok = True
    for j in range(2, 30):
        for h in (1, 3, 5):
            fe = _oos_fit_end(j, h)
            if fe >= 0 and fe + h >= j:            # a returned fit end must NOT leak
                guard_ok = False
    # a fit end that reaches the anchor (scoring point >= anchor) MUST trip the guard
    leak_trips = False
    try:
        bad_fit_end, anchor, h = 5, 5, 1           # scoring point 5+1=6 >= anchor 5 (leak)
        assert bad_fit_end + h < anchor, "LEAK"    # the SAME predicate _oos_fit_end asserts
    except AssertionError:
        leak_trips = True
    # fit_score_split (the wired guard for #49 fit-methods) still refuses an overlap
    lg2_ok, _ = P.leakage_guard_selftest()
    inv["oos_leakage_guard"] = (
        guard_ok and leak_trips and lg2_ok,
        f"_oos_fit_end causal on 2<=j<30 x h in (1,3,5)={guard_ok}; "
        f"intentional-leak trips assert={leak_trips}; fit_score_split guard={lg2_ok}")

    # -- damped off-path parity: window_stats block-sum == direct DampedLinear.predict
    dmp_lam = 0.4
    dmeth = DampedLinear(dmp_lam)
    _saved = _REGISTRY.get("damped_linear")
    register_method(dmeth)                        # temp: fixed-lambda for parity
    dp_ok, dp_res = parity_check(reader, sstats, ["syn.a", "syn.d"], n_ticks_s, sband,
                                 [(2, 1), (3, 4)], ["damped_linear"])
    register_method(_saved)                        # restore the OOS damped placeholder
    worst_dp = max((r["worst_rel"] for r in dp_res), default=float("nan"))
    inv["damped_offpath_parity"] = (
        dp_ok, f"lam={dmp_lam}: {len(dp_res)} samples, worst rel diff {worst_dp:.2e} "
               f"(tol {PARITY_RTOL:g}) — block-sum window_stats == Order1(kappa)+metrics")

    # -- per-scalar R² off-path parity + [0,1] bounds + constant exclusion --------
    r2reader, r2nd, r2n = _synthetic_reader_r2()
    r2names = [n for n, _ in r2nd]
    r2stats = stream_stats(r2reader, r2nd, r2n, band=6, ram_gb=0.05, tag="-r2",
                           retain_r2=set(r2names))
    r2_ok, r2_bounds_ok, excl_ok = True, True, True
    r2_det = []
    for nm in r2names:
        V, W, Pv = r2stats[nm]["_r2_ve"]
        r2_stream, cmask_s = per_element_r2(V, W, Pv, r2n)
        r2_dir, cmask_d = _r2_direct(r2reader, nm, r2n)
        med_s = float(np.nanmedian(r2_stream[~cmask_s])) if np.any(~cmask_s) else float("nan")
        med_d = float(np.nanmedian(r2_dir[~cmask_d])) if np.any(~cmask_d) else float("nan")
        fr_s = float(np.mean(r2_stream[~cmask_s] > R2_STRONG)) if np.any(~cmask_s) else float("nan")
        fr_d = float(np.mean(r2_dir[~cmask_d] > R2_STRONG)) if np.any(~cmask_d) else float("nan")
        dmed = 0.0 if (math.isnan(med_s) and math.isnan(med_d)) else abs(med_s - med_d)
        dfr = 0.0 if (math.isnan(fr_s) and math.isnan(fr_d)) else abs(fr_s - fr_d)
        excl_match = int(cmask_s.sum()) == int(cmask_d.sum())
        valid = r2_stream[~cmask_s]
        valid = valid[np.isfinite(valid)]
        in_bounds = valid.size == 0 or (valid.min() >= 0.0 and valid.max() <= 1.0)
        r2_ok = r2_ok and dmed <= 1e-6 and dfr <= 1e-6
        excl_ok = excl_ok and excl_match
        r2_bounds_ok = r2_bounds_ok and in_bounds
        r2_det.append(f"{nm}: |dmed|={dmed:.1e} |dfr|={dfr:.1e} nexcl={int(cmask_s.sum())}")
    const_excluded = int(r2stats["r2.const"]["n_excluded_const"]) == 50
    inv["r2_offpath_parity"] = (
        r2_ok and excl_ok, "; ".join(r2_det))
    inv["r2_bounds_and_exclusion"] = (
        r2_bounds_ok and const_excluded,
        f"all valid R² in [0,1]={r2_bounds_ok}; r2.const fully excluded "
        f"(n_excluded_const={r2stats['r2.const']['n_excluded_const']}/50)={const_excluded}")

    # -- paper_linear anchor rule + cross-path parity (direct == naive(delta_resolved)) --
    pp_band = 30
    ppstats = stream_stats(r2reader, r2nd, r2n, band=pp_band, ram_gb=0.05, tag="-pp")
    windows, needed = _paper_windows(r2n, [2, 4], anchor_frac=0.25, stride=2)
    anchor_rule_ok = all((0.20 <= t0 / t <= 0.30) and t >= 20 and dres == t - t0
                         for (h, t, t0, dres) in windows)
    pp_worst = 0.0
    nai = NaiveLinear()
    nm = "r2.noisy"                                # non-linear -> e2 not ~0 (meaningful parity)
    Pp = prefix_from_banded(ppstats[nm]["D"], pp_band)
    for (h, t, t0, dres) in windows:
        if dres + h > pp_band:                    # only band-fitting windows are comparable
            continue
        a = r2reader.load_matrix_f64(t0, nm); tt = r2reader.load_matrix_f64(t, nm)
        s = r2reader.load_matrix_f64(t + h, nm)
        k = h / dres
        e = (tt + k * (tt - a)) - s; b = tt - s
        de2, db2, deb = float(e @ e), float(b @ b), float(e @ b)   # direct triple
        saa, sab, sbb = cell_window_sums(Pp, dres, h, r2n)
        jj = t - dres                              # window index for anchor t at delta=dres
        be2, bb2, beb = nai.window_stats(saa[jj], sab[jj], sbb[jj], dres, h)  # band triple
        for dv, sv in ((de2, be2), (db2, bb2), (deb, beb)):
            pp_worst = max(pp_worst, abs(dv - sv) / max(abs(dv), abs(sv), 1e-12))
    inv["paper_anchor_and_parity"] = (
        anchor_rule_ok and pp_worst <= PARITY_RTOL,
        f"anchor rule t0=floor(0.25t),t>=20,0.20<=t0/t<=0.30,dres=t-t0={anchor_rule_ok}; "
        f"direct (e2,b2,eb)==naive(delta_resolved) worst rel {pp_worst:.2e} "
        f"(tol {PARITY_RTOL:g})")

    # -- real-trace subset battery ---------------------------------------------
    det_soft = (True, "not run (no trace)")
    if args.trace_root and part is not None:
        reader_r = MmapTraceReader(args.trace_root)
        deltas, hs = args.deltas, args.hs
        band = max(deltas) + max(hs)
        n_ticks = args.n_ticks
        present = reader_r.present_ticks(n_ticks)
        if len(present) < n_ticks:
            inv["trace_ready"] = (
                False, f"trace incomplete: {len(present)}/{n_ticks} ticks at "
                       f"{args.trace_root}")
        else:
            inv["trace_ready"] = (True, f"{n_ticks} ticks present")
            cls = {n: ST.classify(n) for n in man_names}
            subset = [n for n in man_names
                      if cls[n]["block_type"] in ("norm", "bias")
                      or (cls[n]["layer_idx"] == 0 and
                          cls[n]["super_block"] in ("attention", "mlp"))]
            dims = _manifest_dims(args.manifest)
            sub_dims = [(n, dims[n]) for n in subset]
            log(f"real-trace subset: {len(subset)} matrices, "
                f"{sum(d for _, d in sub_dims):,} elems/tick")
            rstats = stream_stats(reader_r, sub_dims, n_ticks, band,
                                  ram_gb=args.ram_gb, tag="-real")
            rrows, _, _ = compute_rows(rstats, subset, n_ticks, band,
                                    ["hold_stale", "naive_linear"], deltas, hs,
                                    args.op_point, args.also_points)
            rhold = [r for r in rrows if r["method"] == "hold_stale"
                     and r["in_bounds"] and not r["tied"]]
            worst_r = max(abs(r["weight_proj_ratio_median"] - 1.0) for r in rhold)
            inv["hold_stale_identity_real"] = (
                worst_r <= IDENTITY_TOL,
                f"{len(rhold)} rows over all {len(deltas) * len(hs)} cells, "
                f"worst |ratio-1| = {worst_r:.2e} (tol {IDENTITY_TOL:g})")
            rnl = [r for r in rrows if r["method"] == "naive_linear" and r["in_bounds"]]
            n_nan = sum(r["n_nan_windows"] for r in rnl)
            bad_nw = [r for r in rrows if r["in_bounds"] and not r["tied"] and
                      r["n_windows"] != n_ticks - r["h_ticks"] - r["delta_ticks"]]
            inv["finite_and_bounds_real"] = (
                n_nan == 0 and not bad_nw,
                f"naive_linear NaN windows = {n_nan}; n_windows formula "
                f"violations = {len(bad_nw)}")
            par_names = [n for n in subset
                         if cls[n]["block_type"] in ("norm", "bias")][:2] + \
                        [n for n in subset if cls[n]["block_type"] == "q_proj"][:1] + \
                        [n for n in subset if cls[n]["block_type"] == "down_proj"][:1]
            pc_cells = [(min(deltas), min(hs)), (max(deltas), max(hs)),
                        args.op_point if args.op_point[0] in deltas else
                        (deltas[0], hs[0])]
            rp_ok, rp_res = parity_check(reader_r, rstats, par_names, n_ticks, band,
                                         pc_cells, ["hold_stale", "naive_linear"])
            worst_rp = max((r["worst_rel"] for r in rp_res), default=float("nan"))
            inv["offpath_parity_real"] = (
                rp_ok, f"{len(rp_res)} samples on {par_names}, worst rel diff "
                       f"{worst_rp:.2e} (tol {PARITY_RTOL:g})")
            # determinism (soft): re-stream 2 tiny matrices; recompute a cell twice
            tiny = [(n, dims[n]) for n in subset
                    if cls[n]["block_type"] == "norm"][:2]
            s1 = stream_stats(reader_r, tiny, n_ticks, band, ram_gb=1, tag="-det1")
            s2 = stream_stats(reader_r, tiny, n_ticks, band, ram_gb=1, tag="-det2")
            det_stream = all(np.array_equal(s1[n]["D"], s2[n]["D"]) for n, _ in tiny)
            r1, _, _ = compute_rows(rstats, [tiny[0][0]], n_ticks, band, ["naive_linear"],
                                 [deltas[0]], [hs[0]], args.op_point, [])
            r2, _, _ = compute_rows(rstats, [tiny[0][0]], n_ticks, band, ["naive_linear"],
                                 [deltas[0]], [hs[0]], args.op_point, [])
            det_rows = json.dumps(_clean(r1)) == json.dumps(_clean(r2))
            det_soft = (det_stream and det_rows,
                        f"re-streamed D byte-identical={det_stream}, "
                        f"recomputed rows byte-identical={det_rows}")
    elif args.trace_root:
        inv["trace_ready"] = (False, "manifest missing — cannot stream subset")

    inv["determinism_soft"] = det_soft

    hard = {k: v for k, v in inv.items() if k != "determinism_soft"}
    go = all(ok for ok, _ in hard.values())
    for k, (ok, detail) in inv.items():
        soft = " (soft)" if k == "determinism_soft" else ""
        print(f"INVARIANT {k}{soft}: {'PASS' if ok else 'FAIL'} — {detail}", flush=True)
    with open(os.path.join(out_dir, "selftest.json"), "w") as f:
        json.dump({k: {"pass": bool(ok), "detail": d, "hard": k != "determinism_soft"}
                   for k, (ok, d) in inv.items()}, f, indent=2)
    print(f"SELFTEST: {'GO' if go else 'NO-GO'}", flush=True)
    return 0 if go else 1


# =============================================================================
# Emit (step 4) and schema verification (step 5)
# =============================================================================
def _manifest_names(path: str) -> list[str]:
    row0 = json.loads(open(path).readline())
    return [m["name"] for m in row0["matrices"]]


def _manifest_dims(path: str) -> dict:
    row0 = json.loads(open(path).readline())
    return {m["name"]: int(m["d"]) for m in row0["matrices"]}


def _manifest_nticks(path: str) -> int:
    return sum(1 for l in open(path) if l.strip())


def _clean(x):
    """NaN/Inf -> null so the emitted JSON is strictly parseable (round-trip gate)."""
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return float(x) if math.isfinite(float(x)) else None
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def run_emit(args) -> int:
    assert M.METRIC_CONTRACT == METRIC_CONTRACT_EXPECTED, (
        f"metric-contract drift: {M.METRIC_CONTRACT!r} != {METRIC_CONTRACT_EXPECTED!r}")
    out_dir = args.out or DEFAULT_OUT
    os.makedirs(out_dir, exist_ok=True)
    names = _manifest_names(args.manifest)
    dims = _manifest_dims(args.manifest)
    part = ST.partition(names)
    if not part["ok"]:
        print(f"GATE structure_partition: FAIL — {part['failures']}", flush=True)
        print("EMIT: NO-GO", flush=True)
        return 1
    n_ticks = args.n_ticks
    deltas, hs = args.deltas, args.hs
    band = max(deltas) + max(hs)
    # #47 cadence: resolve the SELECTED tick set; step index s -> real tick tickset[s].
    tickset = TS.select_ticks(args.cadence, n_ticks)
    unit = "global_step" if args.cadence == "per-step" else "tick"
    assert len(tickset) == n_ticks, (
        f"cadence {args.cadence} yields {len(tickset)} ticks != n_ticks {n_ticks} "
        f"(per-step needs n_ticks=80, per-tick needs 160 for the EXP-57 trace)")
    reader = MmapTraceReader(args.trace_root, tickset=tickset)
    present = reader.present_ticks(n_ticks)
    if len(present) < n_ticks:
        print(f"TRACE-INCOMPLETE: {len(present)}/{n_ticks} selected {args.cadence} "
              f"ticks at {args.trace_root}", flush=True)
        print("EMIT: NO-GO", flush=True)
        return 2
    log(f"cadence={args.cadence} unit={unit} n_ticks={n_ticks} band={band} "
        f"tickset[0:4]={tickset[:4]}..{tickset[-1]}")
    name_dims = [(n, dims[n]) for n in names]
    fp = _fingerprint(name_dims, n_ticks, band, args.trace_root,
                      cadence=args.cadence, tickset=tickset)
    cache_path = os.path.join(out_dir, "stats_cache.npz")
    stats = None if args.force_recompute else load_stats_cache(cache_path, fp)
    if stats is None:
        t0 = time.time()
        stats = stream_stats(reader, name_dims, n_ticks, band, ram_gb=args.ram_gb)
        log(f"full streaming pass done in {(time.time() - t0) / 60:.1f} min")
        save_stats_cache(cache_path, stats, fp)
        log(f"stats cache saved: {cache_path}")
    else:
        log(f"stats cache reused: {cache_path}")
    methods = args.methods
    cached_methods = [m for m in methods if m != "paper_linear"]
    rows, ratio_store, lam_select = compute_rows(
        stats, names, n_ticks, band, cached_methods, deltas, hs,
        args.op_point, args.also_points,
        cadence=args.cadence, unit=unit, lam_grid=args.lam_gridv)
    log(f"{len(rows)} banded-cache atomic rows computed")
    paper_panel = None
    if "paper_linear" in methods:
        groups = build_groups(names)
        t0 = time.time()
        paper_rows, paper_panel = compute_paper_rows(
            reader, names, dims, n_ticks, hs, groups,
            args.paper_anchor_frac, args.paper_stride, stats,
            args.cadence, unit, ram_gb=args.ram_gb)
        paper_panel["operating_h"] = args.op_point[1]
        rows.extend(paper_rows)
        log(f"paper_linear direct pass done in {(time.time() - t0) / 60:.1f} min "
            f"({len(paper_rows)} rows)")
    vis = build_visuals(rows, methods, deltas, hs, args.op_point,
                        lam_select=lam_select, stats=stats, paper_panel=paper_panel)
    gates = run_gates(rows, part, n_ticks, methods, deltas, hs,
                      args.op_point, args.also_points)
    gall = next((r for r in rows if r["method"] == cached_methods[0]
                 and r["group_kind"] == "global"), {})
    linearity_r2 = {"r2_median": gall.get("r2_median"),
                    "r2_frac_gt_0.7": gall.get("r2_frac_gt_0.7"),
                    "n_excluded_const": gall.get("n_excluded_const")}
    visual_keys = list(vis.keys())
    meta = {
        "experiment": "EXP-47 (MOAT ANCHOR linear/damped-linear lane)",
        "metric_contract": M.METRIC_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "trace_root": os.path.realpath(args.trace_root),
        "manifest": args.manifest,
        "n_ticks": n_ticks, "band": band,
        "cadence": args.cadence, "unit": unit,
        "tickset_first": tickset[0], "tickset_last": tickset[-1],
        "tickset_stride": (tickset[1] - tickset[0]) if len(tickset) > 1 else 1,
        "methods": methods, "delta_ticks": deltas, "h_ticks": hs,
        "lam_grid": args.lam_gridv,
        "paper_anchor_frac": args.paper_anchor_frac, "paper_stride": args.paper_stride,
        "paper_sentinel_delta": PAPER_SENTINEL_DELTA,
        "operating_point": list(args.op_point),
        "also_points": [list(p) for p in args.also_points],
        "n_matrices": len(names),
        "structure_block_type_counts": part["block_type_counts"],
        "structure_super_block_counts": part["super_block_counts"],
        "required_row_keys": REQUIRED_ROW_KEYS,
        "visual_keys": visual_keys,
        "linearity_r2": linearity_r2,
        "gates": gates,
        "stats_cache_fingerprint": fp,
        "n_rows": len(rows),
    }
    with open(os.path.join(out_dir, "scorecard.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(_clean(r), sort_keys=True) + "\n")
    with open(os.path.join(out_dir, "visuals.json"), "w") as f:
        json.dump(_clean(vis), f, indent=1)
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(_clean(meta), f, indent=2)
    log(f"emitted {out_dir}/scorecard.jsonl (+ visuals.json, meta.json)")
    go = all(g["pass"] for g in gates.values())
    print(f"EMIT: {'GO' if go else 'NO-GO'}", flush=True)
    return 0 if go else 1


def run_verify_schema(scorecard_dir: str) -> int:
    """Round-trip verifier: a consumer (a lane, or #56) can load the tables and
    recover per-(method, Delta, h, group) rows. Pure json/numpy — runs anywhere."""
    problems: list[str] = []
    sc_path = os.path.join(scorecard_dir, "scorecard.jsonl")
    vis_path = os.path.join(scorecard_dir, "visuals.json")
    meta_path = os.path.join(scorecard_dir, "meta.json")
    for p in (sc_path, vis_path, meta_path):
        if not os.path.exists(p):
            problems.append(f"missing file: {p}")
    rows, meta, vis = [], {}, {}
    if not problems:
        try:
            rows = [json.loads(l) for l in open(sc_path) if l.strip()]
            meta = json.load(open(meta_path))
            vis = json.load(open(vis_path))
        except Exception as e:
            problems.append(f"round-trip parse failure: {e}")
    if rows and not problems:
        for i, r in enumerate(rows):
            missing = [k for k in REQUIRED_ROW_KEYS if k not in r]
            if missing:
                problems.append(f"row {i} missing keys {missing}")
                break
        index: dict = {}
        for r in rows:
            k = (r["method"], r["delta_ticks"], r["h_ticks"],
                 r["group_kind"], r["group_key"])
            if k in index:
                problems.append(f"duplicate row key {k}")
                break
            index[k] = r
        methods = meta.get("methods", [])
        deltas = meta.get("delta_ticks", [])
        hs = meta.get("h_ticks", [])
        n_ticks = meta.get("n_ticks", 0)
        # paper_linear is OFF the (delta x h) grid (derived delta, sentinel delta_ticks)
        grid_methods = [m for m in methods if m != "paper_linear"]
        paper_delta = meta.get("paper_sentinel_delta", 0)
        for m in grid_methods:
            for d in deltas:
                for h in hs:
                    if (m, d, h, "global", "all") not in index:
                        problems.append(f"missing global row for ({m},{d},{h})")
        if "paper_linear" in methods:
            for h in hs:
                if ("paper_linear", paper_delta, h, "global", "all") not in index:
                    problems.append(f"missing global paper_linear row for h={h}")
            pw = [r for r in rows if r["method"] == "paper_linear"]
            if pw and not all(r.get("anchor_mode") == "frac25"
                              and r.get("delta_resolved") is not None
                              and r.get("beta") is not None for r in pw
                              if r.get("in_bounds")):
                problems.append("paper_linear rows missing anchor_mode/delta_resolved/beta")
        kinds = {}
        for r in rows:
            kinds.setdefault(r["group_kind"], set()).add(r["group_key"])
        exp_bt, exp_sb = ST.expected_counts()
        need_bt = {k for k, v in exp_bt.items() if v > 0}
        need_sb = {k for k, v in exp_sb.items() if v > 0}
        if not need_bt <= kinds.get("block_type", set()):
            problems.append(f"block_type aggregates missing: "
                            f"{sorted(need_bt - kinds.get('block_type', set()))}")
        if not need_sb <= kinds.get("super_block", set()):
            problems.append(f"super_block aggregates missing: "
                            f"{sorted(need_sb - kinds.get('super_block', set()))}")
        n_layers = ST.N_LAYERS if meta.get("n_matrices") == 338 else None
        if n_layers:
            miss_layers = {str(i) for i in range(n_layers)} - kinds.get("layer", set())
            if miss_layers:
                problems.append(f"layer aggregates missing: {sorted(miss_layers)[:5]}")
            if len(kinds.get("layer_block", set())) < 7 * n_layers:
                problems.append(f"layer_block aggregates: "
                                f"{len(kinds.get('layer_block', set()))} < {7 * n_layers}")
            if len(kinds.get("matrix", set())) != meta.get("n_matrices"):
                problems.append(f"matrix rows {len(kinds.get('matrix', set()))} != "
                                f"{meta.get('n_matrices')}")
        need_special = {"embed", "norm", "bias", "lm_head"}
        if not need_special <= kinds.get("special", set()):
            problems.append(f"special rows missing: "
                            f"{sorted(need_special - kinds.get('special', set()))}")
        lm = [r for r in rows if r["group_kind"] == "special"
              and r["group_key"] == "lm_head"]
        if lm and not all(r.get("tied") and r.get("tied_to") == "embed" for r in lm):
            problems.append("lm_head rows not marked tied/tied_to=embed")
        for m in grid_methods:
            ops = [meta.get("operating_point", [])] + meta.get("also_points", [])
            for p in ops:
                if len(p) == 2 and (m, p[0], p[1], "global", "all") not in index:
                    problems.append(f"operating-point row missing: ({m},{p})")
        for r in rows:
            if (r["in_bounds"] and not r.get("tied") and n_ticks
                    and r["method"] != "paper_linear"):     # paper is off the banded grid
                if r["n_windows"] != n_ticks - r["h_ticks"] - r["delta_ticks"]:
                    problems.append(f"n_windows mismatch on "
                                    f"({r['method']},{r['delta_ticks']},{r['h_ticks']},"
                                    f"{r['group_kind']},{r['group_key']})")
                    break
        # visuals: check the ACTUAL emitted set (meta.visual_keys), so a regime-T table
        # (no paper panel) verifies without demanding the regime-S-only m_paper visual.
        for key in meta.get("visual_keys", VISUAL_KEYS):
            v = vis.get(key)
            if v is None or (isinstance(v, (dict, list)) and len(v) == 0):
                problems.append(f"visual array {key} missing/empty")
        # round-trip: re-serialize a sample of rows and compare parsed equality
        for r in rows[:: max(len(rows) // 50, 1)]:
            if json.loads(json.dumps(r, sort_keys=True)) != r:
                problems.append("round-trip re-serialization mismatch")
                break
        n_nan = sum(r.get("n_nan_windows", 0) or 0 for r in rows
                    if r.get("method") == "naive_linear")
        print(f"[schema] rows={len(rows)} kinds="
              f"{ {k: len(v) for k, v in kinds.items()} } "
              f"naive_linear denom-guard NaN windows={n_nan}", flush=True)
    for p in problems:
        print(f"SCHEMA-PROBLEM: {p}", flush=True)
    go = not problems
    print(f"SCHEMA: {'GO' if go else 'NO-GO'}", flush=True)
    return 0 if go else 1


# =============================================================================
# CLI
# =============================================================================
def _parse_pair(s: str) -> tuple[int, int]:
    a, b = s.split(",")
    return int(a), int(b)


def main() -> int:
    ap = argparse.ArgumentParser(description="#45 MOAT scorecard contract harness")
    ap.add_argument("--trace-root", default="",
                    help="local trace root (full/tick_<N>/tick_<N>.pt)")
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--method", default="hold_stale,naive_linear")
    ap.add_argument("--delta", default="5,10,20")
    ap.add_argument("--h", dest="h_grid", default="1,2,5,10,20,30,40")
    ap.add_argument("--operating-point", default="20,20",
                    help="primary cadence-faithful (Delta,h)")
    ap.add_argument("--also", default="10,10",
                    help="secondary fast-example (Delta,h); '' to skip")
    ap.add_argument("--out", default="")
    ap.add_argument("--verify-schema", default="",
                    help="scorecard dir to round-trip verify (no trace needed)")
    ap.add_argument("--n-ticks", type=int, default=0,
                    help="default: rows in --manifest")
    ap.add_argument("--ram-gb", type=float, default=40.0)
    ap.add_argument("--force-recompute", action="store_true")
    # ---- #47 flags ----
    ap.add_argument("--cadence", default="per-tick", choices=["per-tick", "per-step"],
                    help="per-tick=all ticks 0..n-1; per-step=even ticks [0,2,…] "
                         "reindexed (GLOBAL STEPS, paper-comparable)")
    ap.add_argument("--lam-grid", default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
                    help="damped_linear lambda grid (nests hold_stale@0, naive@1)")
    ap.add_argument("--paper-anchor-frac", type=float, default=0.25,
                    help="paper_linear t0 = floor(frac*t) (Wang et al. App. E.1, 0.20-0.30)")
    ap.add_argument("--paper-stride", type=int, default=2,
                    help="paper_linear anchor stride over t (regime S only)")
    args = ap.parse_args()

    if args.verify_schema:
        return run_verify_schema(args.verify_schema)

    assert M.METRIC_CONTRACT == METRIC_CONTRACT_EXPECTED, (
        f"metric-contract drift: {M.METRIC_CONTRACT!r} != {METRIC_CONTRACT_EXPECTED!r}")
    args.methods = [m.strip() for m in args.method.split(",") if m.strip()]
    for m in args.methods:
        # paper_linear is a direct-scored arm (not a fixed-kappa registry method)
        assert m in _REGISTRY or m == "paper_linear", \
            f"unknown method {m!r}; registered: {sorted(_REGISTRY)} + paper_linear"
    args.deltas = [int(x) for x in args.delta.split(",")]
    args.hs = [int(x) for x in args.h_grid.split(",")]
    args.op_point = _parse_pair(args.operating_point)
    args.also_points = [_parse_pair(args.also)] if args.also else []
    args.lam_gridv = [float(x) for x in args.lam_grid.split(",")]
    if args.cadence == "per-step" and "paper_linear" not in args.methods:
        pass  # paper is regime-S-only but optional; regime T never includes it
    if "paper_linear" in args.methods and args.cadence != "per-step":
        raise SystemExit("paper_linear is regime-S (per-step) ONLY — the paper protocol "
                         "is checkpoint/per-step-like; per-tick is out of scope")
    if not args.n_ticks:
        args.n_ticks = _manifest_nticks(args.manifest) if os.path.exists(args.manifest) else 160

    if args.self_test:
        return run_selftest(args)
    assert args.trace_root, "--trace-root required for emit"
    return run_emit(args)


if __name__ == "__main__":
    sys.exit(main())
