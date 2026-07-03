#!/usr/bin/env python3
"""weight_proj/rank1_traj.py — RELEX-style rank-1 trajectory-SVD predictor family
(Wei et al. 2026, arXiv:2605.21468) for the OFFLINE weight-projection analysis.

This is a NEW family alongside the raw-space CORE arms (naive_linear /
second-order / adaptive in moat_scorecard.py). It never touches the online
comm-eff path. The method, per weight tensor:

  Step 1 (subspace)   Δθ_t = θ_t − θ_base for the W window checkpoints ending at
                      the anchor t_a; stack into M ∈ R^{W×d}; top-r right singular
                      vectors v_1..v_r (r=1 is the paper's default).
  Step 2 (dynamics)   c_j(t) = ⟨Δθ_t, v_j⟩ is fit as c_j(t) = a_j·t + b_j by least
                      squares over the window (paper: R² > 0.98 for r=1).
  Step 3 (predict)    θ̂_T = θ_base + Σ_j ĉ_j(T)·v_j,  ĉ_j(T) = a_j·T + b_j.

GRAM-DOMAIN IDENTITY (why no d-dim vector is ever needed for scoring).
With G = M Mᵀ (W×W) and eigh(G) = Σ_j λ_j u_j u_jᵀ (λ descending, u orthonormal):
  σ_j = √λ_j,   v_j = Mᵀ u_j / σ_j,   c_j = M v_j = σ_j u_j          (exact SVD)
so the coefficient trajectory is σ_j·u_j — read directly off the eigenvectors —
and every scoring quantity is a quadratic form in base-delta inner products:
  ⟨v_j, Δθ_x⟩      = (1/σ_j) Σ_i u_j[i]·B[w_i, x]
  ‖e‖², ‖b‖², ⟨e,b⟩ via `quad()` below, where e = θ̂_T − θ_T, b = θ_anchor − θ_T
  (the SAME baseline-displacement denominator convention as the MOAT scorecard:
  hold-stale-at-anchor scores ratio == 1 exactly).
B itself comes from the full Gram D of CONSECUTIVE selected-tick deltas via 2-D
prefix sums — deltas are differenced upstream in float64 from the raw snapshots,
so no big-number cancellation ever enters the accumulation (the same discipline
as moat_scorecard's banded-Gram engine; here the Gram is FULL, not banded,
because the trajectory matrix couples every window pair).

RECONSTRUCTION INVARIANT (the family stays inside the linear-combination class):
  θ̂_T = θ_base + Σ_i γ_i·Δθ_{w_i},   γ_i = Σ_j ĉ_j(T)·u_j[i]/σ_j
i.e. an explicit linear combination of the RAW window snapshots with coefficient
γ_i on θ_{w_i} and 1 − Σ_i γ_i on θ_base. `snapshot_coeffs()` exposes it; the
self-test asserts predict-via-coeffs == direct prediction.

SIGN CONVENTION: eigenvector signs are arbitrary; predictions are invariant under
(u_j, v_j) → (−u_j, −v_j) since ĉ_j flips with u_j. Nothing here depends on sign.

Everything in this module is pure numpy/float64 and I/O-free. The streaming
accumulation over the real trace lives in research/scripts/rank1_scorecard.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

RANK1_CONTRACT = "weight-proj-rank1-traj-v1"

# eigenvalue floor: component j is usable iff λ_j > EIG_REL_FLOOR * trace(G)
EIG_REL_FLOOR = 1e-12


# =============================================================================
# Tick plan — the union of ticks a sweep needs (window ∪ targets ∪ base)
# =============================================================================
@dataclass
class TickPlan:
    """Sorted unique REAL tick numbers a sweep touches, with position lookup."""
    selected: list[int]
    _pos: dict[int, int] = field(default_factory=dict)

    def __post_init__(self):
        assert self.selected == sorted(set(self.selected)), "ticks must be sorted unique"
        assert len(self.selected) >= 3, "need >= 3 ticks (base + window + target)"
        self._pos = {t: i for i, t in enumerate(self.selected)}

    def pos(self, tick: int) -> int:
        return self._pos[tick]

    @property
    def n(self) -> int:
        return len(self.selected)


def build_tick_plan(base_tick: int, anchor_windows: list[tuple[int, int]],
                    h_grid: list[int], stride: int, n_ticks: int) -> TickPlan:
    """Union of {base} ∪ window ticks ∪ target ticks; validates every bound.

    `anchor_windows` is a list of (anchor tick t_a, window size W) PAIRS —
    windows are per-anchor because a 'prefix' window resolves to a different W
    at each anchor. Window ticks are t_a − i*stride, i = 0..W−1 — all must be
    STRICTLY after base_tick (deltas from base must be non-degenerate) and >= 0.
    Targets: t_a + h for every anchor — must be < n_ticks and strictly AFTER the
    anchor (h >= 1: the leakage direction is one-way by construction).
    """
    assert stride >= 1
    sel = {base_tick}
    anchors = sorted({a for a, _ in anchor_windows})
    for t_a, w in anchor_windows:
        assert 0 <= t_a < n_ticks, f"anchor {t_a} outside trace [0,{n_ticks})"
        assert w >= 2, f"window {w} too small: need >= 2 checkpoints for a slope"
        lo = t_a - (w - 1) * stride
        assert lo > base_tick, (
            f"window (anchor={t_a}, W={w}, stride={stride}) reaches tick {lo} "
            f"<= base_tick {base_tick}")
        sel.update(t_a - i * stride for i in range(w))
    for t_a in anchors:
        for h in h_grid:
            assert h >= 1, f"h must be >= 1 (got {h}): target must be after anchor"
            assert t_a + h < n_ticks, f"target {t_a}+{h} outside trace [0,{n_ticks})"
            sel.add(t_a + h)
    return TickPlan(sorted(sel))


# =============================================================================
# Base-delta Gram from the consecutive-delta Gram (prefix-sum identity)
# =============================================================================
class TrajGram:
    """Wraps D[k,l] = ⟨d_k, d_l⟩ (full Gram of consecutive selected-tick deltas
    d_k = θ_{s_k} − θ_{s_{k−1}}, k = 1..S−1) and serves base-delta Grams.

    A[i,j] = ⟨θ_{s_i}−θ_{s_0}, θ_{s_j}−θ_{s_0}⟩ = Σ_{k≤i} Σ_{l≤j} D[k,l] (prefix sum);
    re-basing to position b is bilinear inclusion-exclusion:
      B_b[i,j] = A[i,j] − A[b,j] − A[i,b] + A[b,b].
    D is ADDITIVE over element shards of a matrix and over member matrices of a
    group, so per-matrix Grams sum to exact group Grams (concatenated-vector
    semantics, same as the MOAT engine's group rows).
    """

    def __init__(self, plan: TickPlan, D: np.ndarray):
        S = plan.n
        assert D.shape == (S - 1, S - 1), f"D shape {D.shape} != {(S-1, S-1)}"
        self.plan = plan
        self.D = np.asarray(D, dtype=np.float64)
        A = np.zeros((S, S), dtype=np.float64)
        np.cumsum(np.cumsum(self.D, axis=0), axis=1, out=A[1:, 1:])
        self._A = 0.5 * (A + A.T)          # symmetrize away cumsum rounding skew
        self._base_cache: dict[int, np.ndarray] = {}

    def base_gram(self, base_pos: int) -> np.ndarray:
        """B[i,j] = ⟨θ_i − θ_base, θ_j − θ_base⟩ over ALL selected positions."""
        if base_pos not in self._base_cache:
            A = self._A
            B = (A - A[base_pos:base_pos + 1, :] - A[:, base_pos:base_pos + 1]
                 + A[base_pos, base_pos])
            self._base_cache[base_pos] = B
        return self._base_cache[base_pos]


def quad(B: np.ndarray, u: dict[int, float], v: dict[int, float]) -> float:
    """⟨Σ_i u_i Δθ_i, Σ_j v_j Δθ_j⟩ = Σ_ij u_i v_j B[i,j] (positions are B indices)."""
    return float(sum(cu * cv * B[i, j] for i, cu in u.items() for j, cv in v.items()))


# =============================================================================
# The rank-r trajectory fit (Steps 1+2 in the Gram domain)
# =============================================================================
@dataclass
class RankRFit:
    """Per-tensor trajectory SVD + per-component linear coefficient fit."""
    window_pos: list[int]        # positions (into the plan) of the window ticks, ascending
    window_ticks: list[int]      # real tick numbers, ascending; last = anchor
    rank: int
    sigma: np.ndarray            # (r,) singular values (0 where invalid)
    U: np.ndarray                # (W, r) trajectory-side eigenvectors
    coef: np.ndarray             # (W, r) coefficient trajectories c_j = σ_j·u_j
    slope: np.ndarray            # (r,) a_j of c_j(t) = a_j t + b_j
    intercept: np.ndarray        # (r,) b_j
    coef_r2: np.ndarray          # (r,) least-squares fit R² per component (NaN if degenerate)
    evr: np.ndarray              # (r,) λ_j / trace(G) — energy share per component
    trace_g: float               # Σ_i ‖Δθ_{w_i}‖²
    valid: np.ndarray            # (r,) bool — component has λ above the relative floor

    @property
    def anchor_pos(self) -> int:
        return self.window_pos[-1]

    @property
    def anchor_tick(self) -> int:
        return self.window_ticks[-1]

    def chat(self, target_tick: int) -> np.ndarray:
        """(r,) predicted coefficients ĉ_j(T) = a_j·T + b_j (0 for invalid comps)."""
        c = self.slope * float(target_tick) + self.intercept
        return np.where(self.valid, c, 0.0)

    def delta_coeffs(self, target_tick: int) -> dict[int, float]:
        """γ over window positions: θ̂ = θ_base + Σ_i γ_i Δθ_{w_i} (reconstruction)."""
        ch = self.chat(target_tick)
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = np.where(self.valid & (self.sigma > 0.0), ch / self.sigma, 0.0)
        gamma = self.U @ scale               # (W,)
        return {p: float(g) for p, g in zip(self.window_pos, gamma)}

    def snapshot_coeffs(self, target_tick: int) -> tuple[list[int], list[float], float]:
        """(window positions, coefficients on θ_{w_i}, coefficient on θ_base).

        θ̂_T = (1 − Σγ)·θ_base + Σ_i γ_i·θ_{w_i} — the family's explicit membership
        in the linear-combination-of-raw-snapshots class.
        """
        d = self.delta_coeffs(target_tick)
        gam = [d[p] for p in self.window_pos]
        return list(self.window_pos), gam, 1.0 - float(sum(gam))


def _linfit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Least-squares y = a·x + b; returns (a, b, R²). R² NaN if Var(y) ~ 0."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    xbar, ybar = float(x.mean()), float(y.mean())
    vx = float(((x - xbar) ** 2).sum())
    a = float(((x - xbar) * (y - ybar)).sum()) / vx if vx > 0 else 0.0
    b = ybar - a * xbar
    ss_res = float(((y - (a * x + b)) ** 2).sum())
    ss_tot = float(((y - ybar) ** 2).sum())
    r2 = float("nan") if ss_tot <= 0.0 else 1.0 - ss_res / ss_tot
    return a, b, r2


def fit_rank_r(B: np.ndarray, plan: TickPlan, anchor_tick: int, window: int,
               stride: int, rank: int) -> RankRFit:
    """Steps 1+2: eigh of the window Gram + per-component linear coefficient fit.

    B must be the base-delta Gram (TrajGram.base_gram of the chosen base). The
    window is the `window` ticks ending at the anchor, spaced `stride` apart.
    """
    window_ticks = [anchor_tick - i * stride for i in range(window)][::-1]
    window_pos = [plan.pos(t) for t in window_ticks]
    G = B[np.ix_(window_pos, window_pos)]
    G = 0.5 * (G + G.T)
    lam, u = np.linalg.eigh(G)               # ascending
    lam, u = lam[::-1], u[:, ::-1]           # descending
    trace_g = float(np.trace(G))
    r = min(rank, len(window_ticks))
    lam_r = np.clip(lam[:r], 0.0, None)
    valid = lam_r > max(trace_g, 0.0) * EIG_REL_FLOOR
    sigma = np.sqrt(lam_r)
    U = u[:, :r]
    coef = U * sigma[None, :]                # (W, r): c_j = σ_j u_j
    slope = np.zeros(r)
    intercept = np.zeros(r)
    coef_r2 = np.full(r, np.nan)
    x = np.asarray(window_ticks, dtype=np.float64)
    for j in range(r):
        if not valid[j]:
            continue
        slope[j], intercept[j], coef_r2[j] = _linfit(x, coef[:, j])
    evr = np.divide(lam_r, trace_g, out=np.full(r, np.nan),
                    where=trace_g > 0.0)
    return RankRFit(window_pos=window_pos, window_ticks=window_ticks, rank=r,
                    sigma=sigma, U=U, coef=coef, slope=slope, intercept=intercept,
                    coef_r2=coef_r2, evr=evr, trace_g=trace_g, valid=valid)


# =============================================================================
# Scoring moments (Step 3): ‖e‖², ‖b‖², ⟨e,b⟩ for every family, one code path
# =============================================================================
def family_moments(B: np.ndarray, pred: dict[int, float], anchor_pos: int,
                   target_pos: int) -> tuple[float, float, float]:
    """(‖e‖², ‖b‖², ⟨e,b⟩) with e = θ̂ − θ_T and b = θ_anchor − θ_T.

    `pred` maps position → coefficient of Δθ_position in θ̂ − θ_base (delta space).
    The truth enters as −1·Δθ_target; the base θ_base cancels in every difference,
    so predictions anchored at the base and the raw θ_anchor/θ_T compare exactly.
    """
    e = dict(pred)
    e[target_pos] = e.get(target_pos, 0.0) - 1.0
    b = {anchor_pos: 1.0, target_pos: -1.0}
    return quad(B, e, e), quad(B, b, b), quad(B, e, b)


def rank1_pred(fit: RankRFit, target_tick: int) -> dict[int, float]:
    """Prediction coefficients (delta space) for the rank-r trajectory family."""
    return fit.delta_coeffs(target_tick)


def hold_stale_pred(anchor_pos: int) -> dict[int, float]:
    """θ̂ = θ_anchor — the do-nothing reference; scores ratio == 1 exactly."""
    return {anchor_pos: 1.0}


def two_anchor_pred(anchor_pos: int, anchor_tick: int, prev_pos: int,
                    prev_tick: int, target_tick: int) -> dict[int, float]:
    """Raw-space two-point linear extrapolation (Paper A's Weight Extrapolation):
    θ̂ = θ_a + κ·(θ_a − θ_prev), κ = (T − t_a)/(t_a − t_prev).

    With (prev = anchor − stride) this is the CORE naive_linear at Δ = stride;
    with (prev = window start) it is the fair two-point-over-the-same-window
    baseline the rank-1 family must beat to justify the SVD.
    """
    assert anchor_tick > prev_tick
    kappa = (float(target_tick) - anchor_tick) / (anchor_tick - prev_tick)
    return {anchor_pos: 1.0 + kappa, prev_pos: -kappa}


def gram_from_snapshots(thetas: dict[int, np.ndarray], selected: list[int]
                        ) -> np.ndarray:
    """D[k,l] = ⟨d_k, d_l⟩ over consecutive selected-tick deltas, from raw arrays.

    In-memory mirror of the streaming accumulation in rank1_scorecard.py (same
    float64 differencing discipline) — used by the self-test battery and the
    independent spot-audit so tests and production share ONE math definition
    but not one code path.
    """
    rows = []
    prev = None
    for t in selected:
        cur = np.asarray(thetas[t], dtype=np.float64).reshape(-1)
        if prev is not None:
            rows.append(cur - prev)
        prev = cur
    Dm = np.stack(rows)                      # (S-1, d)
    return Dm @ Dm.T


# =============================================================================
# Direct-tensor reference path (parity target; used by self-tests and spot audit)
# =============================================================================
def direct_fit_predict(thetas: dict[int, np.ndarray], base_tick: int,
                       window_ticks: list[int], target_tick: int, rank: int
                       ) -> tuple[np.ndarray, dict]:
    """Reference implementation on ACTUAL tensors (numpy SVD, d-dim algebra).

    Returns (θ̂_T, diag) where diag carries sigma/coef_r2/evr for cross-checks.
    Independent of the Gram path: builds M explicitly, calls np.linalg.svd.
    """
    base = np.asarray(thetas[base_tick], dtype=np.float64).reshape(-1)
    M = np.stack([np.asarray(thetas[t], dtype=np.float64).reshape(-1) - base
                  for t in window_ticks])                       # (W, d)
    Uf, sf, Vt = np.linalg.svd(M, full_matrices=False)
    r = min(rank, len(window_ticks))
    tot = float((sf ** 2).sum())
    x = np.asarray(window_ticks, dtype=np.float64)
    pred = base.copy()
    diag = {"sigma": sf[:r].copy(), "coef_r2": [], "evr": [],
            "slope": [], "intercept": []}
    for j in range(r):
        if tot <= 0.0 or sf[j] ** 2 <= tot * EIG_REL_FLOOR:
            diag["coef_r2"].append(float("nan"))
            diag["evr"].append(float("nan"))
            diag["slope"].append(0.0)
            diag["intercept"].append(0.0)
            continue
        c = M @ Vt[j]                                           # (W,) coefficients
        a, b, r2 = _linfit(x, c)
        diag["coef_r2"].append(r2)
        diag["evr"].append(float(sf[j] ** 2) / tot)
        diag["slope"].append(a)
        diag["intercept"].append(b)
        pred = pred + (a * float(target_tick) + b) * Vt[j]
    return pred, diag
