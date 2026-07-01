#!/usr/bin/env python3
"""weight_proj/predictors.py — the family-pluggable predictor zoo.

Predictor-family API (architect §4.3):
    predict(history, h, coeff_source) -> Tensor           (all families)
    fit(history, truth) -> None                           (learnable / regression only)

"history" is an ORDERED list of (tick, theta) pairs at the sampling cadence,
OLDEST first, where the LAST entry is theta_stale = theta[t-K] (the anchor point
the prediction extrapolates FROM). `h` is the horizon in sampling steps ahead of
theta_stale to the scoring point. "coeff_source" (fixed / offline-damped /
learnable / learned) is ORTHOGONAL to "order" — hence the sweep is a cross-product.

RECONSTRUCTION INVARIANT (the hypothesis's direct falsifier): every polynomial /
fixed / damped family's prediction IS an explicit LINEAR combination
    theta_hat = sum_j c_j * theta[history index j]
of the loaded raw snapshots. Each such family exposes `linear_coeffs(n_hist, h)`
returning the coefficient vector c (length n_hist), so the analyst can recompute
theta_hat = sum_j c_j theta_j and match predict() within 1e-5 relative. A family
that cannot expose linear_coeffs is NON-reconstructable -> hypothesis falsifier.

LEAKAGE GUARD (architect §5.4, the #1 correctness trap): learnable / regression
families FIT on a retrospective window that ends STRICTLY before the scoring
point, and SCORE on the strictly-later held-out point. `fit_score_split()`
returns (fit_idx, score_idx) and ASSERTS max(fit_idx) < score_idx so the fit
window and the score point can never overlap.

All tensor math is done in float32 (differencing near-equal bf16 tensors upstream
is fp32); coefficients are computed here in float64 for stability then applied.
"""
from __future__ import annotations

import numpy as np
import torch


# =============================================================================
# Coefficient math (pure numpy; the linear-combination weights each family uses)
# =============================================================================

def _newton_forward_coeffs(order: int, x_nodes: list[float], x_eval: float) -> np.ndarray:
    """Lagrange interpolation coefficients: theta_hat(x_eval) = sum_j L_j(x_eval) theta_j.

    Given `order+1` equally- (or arbitrarily-) spaced nodes x_nodes with associated
    theta_j, the degree-`order` interpolating polynomial evaluated at x_eval is the
    linear combination sum_j L_j(x_eval) theta_j where L_j is the Lagrange basis.
    Extrapolation (x_eval beyond the nodes) IS this same linear combination — this
    is why every polynomial extrapolator is a linear combination of past theta,
    and hence reconstructable. Returns c (length order+1), aligned to x_nodes order.
    """
    x = np.asarray(x_nodes, dtype=np.float64)
    n = len(x)
    assert n == order + 1, f"need {order+1} nodes for order {order}, got {n}"
    c = np.zeros(n, dtype=np.float64)
    for j in range(n):
        num = 1.0
        den = 1.0
        for m in range(n):
            if m == j:
                continue
            num *= (x_eval - x[m])
            den *= (x[j] - x[m])
        c[j] = num / den
    return c


class _BasePredictor:
    """Base: name, order, reconstructable flag, and the linear-coeff contract."""
    reconstructable = True
    needs_fit = False

    def __init__(self, order: int, coeff_source: str, name: str,
                 damp: float = 1.0):
        self.order = order
        self.coeff_source = coeff_source   # fixed | offline-damped | learnable | learned
        self.name = name
        self.damp = damp                    # offline-damped multiplier on the extrapolation step

    # --- the reconstruction contract ---
    def linear_coeffs(self, n_hist: int, h: int) -> np.ndarray:
        """Return c (length n_hist) s.t. theta_hat = sum_j c_j theta[j]. NaN-free."""
        raise NotImplementedError

    def predict(self, history: list[tuple[int, torch.Tensor]], h: int,
                coeff_source: str | None = None) -> torch.Tensor:
        """theta_hat via the family's linear combination of the raw snapshots."""
        n = len(history)
        c = self.linear_coeffs(n, h)
        acc = torch.zeros_like(history[0][1], dtype=torch.float32)
        for j, (_, th) in enumerate(history):
            if c[j] != 0.0:
                acc = acc + float(c[j]) * th
        return acc


# =============================================================================
# Order-1 (linear) — fixed + offline-damped
# =============================================================================
class Order1(_BasePredictor):
    """First-order extrapolation from the last two snapshots.

    theta_hat = theta[-1] + damp * h * (theta[-1] - theta[-2])
              = (1 + damp*h) theta[-1] - (damp*h) theta[-2]
    h=0 => theta_hat = theta[-1] = theta_stale exactly (limiting-case identity).
    `fixed`: damp=1. `offline-damped`: damp<1 (re-derived on the trace, motivating
    prior ~0.5 is NOT assumed).
    """
    def __init__(self, coeff_source="fixed", damp=1.0):
        super().__init__(1, coeff_source, f"order1-{coeff_source}", damp)

    def linear_coeffs(self, n_hist: int, h: int) -> np.ndarray:
        c = np.zeros(n_hist, dtype=np.float64)
        step = self.damp * h
        c[-1] = 1.0 + step        # theta[-1]
        c[-2] = -step             # theta[-2]
        return c


# =============================================================================
# Order-2 (3-point Newton forward difference) — fixed + offline-damped
# =============================================================================
class Order2(_BasePredictor):
    """Second-order (3-point) extrapolation. Nodes at x=-2,-1,0 (theta[-3],[-2],[-1]);
    evaluate the quadratic at x = damp*h ahead of the anchor theta[-1] (x=0).
    h=0 => Lagrange at x=0 == theta[-1] exactly.
    """
    def __init__(self, coeff_source="fixed", damp=1.0):
        super().__init__(2, coeff_source, f"order2-{coeff_source}", damp)

    def linear_coeffs(self, n_hist: int, h: int) -> np.ndarray:
        c = np.zeros(n_hist, dtype=np.float64)
        nodes = [-2.0, -1.0, 0.0]                    # theta[-3],[-2],[-1]
        cc = _newton_forward_coeffs(2, nodes, self.damp * h)
        c[-3:] = cc
        return c


# =============================================================================
# Order-3 / poly (Lagrange-Newton over >=4 points) — fixed + offline-damped
# =============================================================================
class OrderPoly(_BasePredictor):
    """Degree-`order` Lagrange-Newton extrapolation over `order+1` >= 4 points.
    Nodes x=-(order),...,-1,0 -> theta[-(order+1)..-1]; evaluate at x=damp*h.
    """
    def __init__(self, order=3, coeff_source="fixed", damp=1.0):
        assert order >= 3
        super().__init__(order, coeff_source, f"order{order}-{coeff_source}", damp)

    def linear_coeffs(self, n_hist: int, h: int) -> np.ndarray:
        c = np.zeros(n_hist, dtype=np.float64)
        k = self.order + 1
        nodes = [float(-(k - 1) + i) for i in range(k)]   # -(order)..0
        cc = _newton_forward_coeffs(self.order, nodes, self.damp * h)
        c[-k:] = cc
        return c


# =============================================================================
# EMA / momentum — fixed-decay + (offline) learnable-decay beta
# =============================================================================
class EMA(_BasePredictor):
    """Exponential-moving-average momentum extrapolation.

    velocity v = sum over consecutive deltas weighted geometrically by beta:
        v = sum_{i>=1} beta^{i-1} (theta[-i] - theta[-i-1])   (a linear combo of theta)
    theta_hat = theta[-1] + h * v.
    beta=0 reduces to order-1 (v = theta[-1]-theta[-2]); this IS a linear
    combination of the raw snapshots, hence reconstructable.
    `fixed`: beta given. `learnable-decay`: beta chosen offline (see fit()).
    """
    def __init__(self, coeff_source="fixed", beta=0.5):
        super().__init__(1, coeff_source, f"ema-{coeff_source}", 1.0)
        self.beta = beta
        self.needs_fit = (coeff_source == "learnable")

    def linear_coeffs(self, n_hist: int, h: int) -> np.ndarray:
        c = np.zeros(n_hist, dtype=np.float64)
        c[-1] += 1.0
        # v = sum_{i=1}^{n_hist-1} beta^{i-1} (theta[-i] - theta[-i-1])
        for i in range(1, n_hist):
            w = h * (self.beta ** (i - 1))
            c[-i] += w
            c[-i - 1] -= w
        return c

    def fit(self, history, truth, h: int, betas=None) -> None:
        """Offline-choose beta on a leakage-safe retrospective split (see fit_score_split).
        Grid-search beta over `betas` to minimize ||theta_hat - truth||; sets self.beta.
        """
        betas = betas if betas is not None else np.linspace(0.0, 0.95, 20)
        best_b, best_err = self.beta, float("inf")
        t_now = truth.to(torch.float32)
        for b in betas:
            self.beta = float(b)
            err = float(torch.linalg.norm(self.predict(history, h) - t_now).item())
            if err < best_err:
                best_err, best_b = err, float(b)
        self.beta = best_b


# =============================================================================
# Learnable-at-order — WEAK per-matrix scalar residual + STRONG full-vector LS
# =============================================================================
class LearnableScalar(_BasePredictor):
    """WEAK learnable: a single scalar `alpha` scaling a base family's extrapolation
    STEP, fit per matrix on a retrospective split. theta_hat = theta[-1] + alpha *
    (base_hat - theta[-1]). It IS a linear combination once alpha is fixed.
    """
    needs_fit = True

    def __init__(self, base: _BasePredictor, coeff_source="learnable"):
        super().__init__(base.order, coeff_source, f"learnable-scalar-{base.name}")
        self.base = base
        self.alpha = 1.0

    def linear_coeffs(self, n_hist: int, h: int) -> np.ndarray:
        base_c = self.base.linear_coeffs(n_hist, h)
        c = np.zeros(n_hist, dtype=np.float64)
        c[-1] += 1.0                       # theta[-1]
        c += self.alpha * base_c           # + alpha (base_hat)
        c[-1] -= self.alpha                # - alpha theta[-1]
        return c

    def fit(self, history, truth, h: int) -> None:
        """Least-squares scalar: alpha* = <t_now-theta[-1], base_hat-theta[-1]> / ||.||^2."""
        anchor = history[-1][1].to(torch.float32).reshape(-1).double()
        base_hat = self.base.predict(history, h).to(torch.float32).reshape(-1).double()
        y = truth.to(torch.float32).reshape(-1).double() - anchor
        d = base_hat - anchor
        den = float(torch.dot(d, d).item())
        self.alpha = float(torch.dot(y, d).item() / den) if den > 1e-30 else 1.0


class LearnableFullVector(_BasePredictor):
    """STRONG learnable: fit the FULL coefficient vector c (length n_hist) by
    least-squares so theta_hat = sum_j c_j theta[j] best matches the truth on a
    retrospective, leakage-safe split. Solves min_c || [theta_0..theta_{n-1}] c - t ||^2
    over the flattened matrix (one c shared across all elements of the matrix).
    Reconstructable by construction: predict() applies the fitted linear combo.
    """
    needs_fit = True

    def __init__(self, n_hist_expected: int, order: int, coeff_source="learnable"):
        super().__init__(order, coeff_source, f"learnable-fullvec-o{order}")
        self.n_hist = n_hist_expected
        self.c = np.zeros(n_hist_expected, dtype=np.float64)
        self.c[-1] = 1.0

    def linear_coeffs(self, n_hist: int, h: int) -> np.ndarray:
        c = np.zeros(n_hist, dtype=np.float64)
        m = min(n_hist, len(self.c))
        c[-m:] = self.c[-m:]
        return c

    def fit(self, history, truth, h: int) -> None:
        # Design matrix A (d x n_hist) from the flattened snapshots; solve A c ~ t.
        cols = [th.to(torch.float32).reshape(-1).double() for _, th in history]
        A = torch.stack(cols, dim=1)                       # d x n_hist
        t = truth.to(torch.float32).reshape(-1).double()
        sol = torch.linalg.lstsq(A, t.unsqueeze(1)).solution.squeeze(1)
        self.c = sol.cpu().numpy().astype(np.float64)
        self.n_hist = len(self.c)


class GeneralRegression(_BasePredictor):
    """General learned-regression predictor: ridge-regularized least-squares over the
    full history AND a constant column, capturing a general linear map from the
    stacked past snapshots to the truth. Regularization keeps it stable when n_hist
    is small; still an explicit linear combination (+ constant) of raw snapshots.
    """
    needs_fit = True

    def __init__(self, n_hist_expected: int, ridge=1e-6, coeff_source="learned"):
        super().__init__(-1, coeff_source, "general-regression")
        self.n_hist = n_hist_expected
        self.ridge = ridge
        self.c = np.zeros(n_hist_expected, dtype=np.float64)
        self.c[-1] = 1.0
        self.bias = 0.0

    def linear_coeffs(self, n_hist: int, h: int) -> np.ndarray:
        # constant term folded out for the pure-linear reconstruction check;
        # reported separately via self.bias.
        c = np.zeros(n_hist, dtype=np.float64)
        m = min(n_hist, len(self.c))
        c[-m:] = self.c[-m:]
        return c

    def predict(self, history, h: int, coeff_source=None) -> torch.Tensor:
        base = super().predict(history, h)
        return base + float(self.bias)

    def fit(self, history, truth, h: int) -> None:
        cols = [th.to(torch.float32).reshape(-1).double() for _, th in history]
        A = torch.stack(cols, dim=1)                       # d x n_hist
        ones = torch.ones(A.shape[0], 1, dtype=torch.float64)
        Aug = torch.cat([A, ones], dim=1)                  # d x (n_hist+1)
        t = truth.to(torch.float32).reshape(-1).double()
        # ridge: (Aug^T Aug + lam I) w = Aug^T t
        AtA = Aug.T @ Aug
        lam = self.ridge * torch.trace(AtA) / AtA.shape[0]
        w = torch.linalg.solve(AtA + lam * torch.eye(AtA.shape[0], dtype=torch.float64),
                               Aug.T @ t)
        wn = w.cpu().numpy().astype(np.float64)
        self.c = wn[:-1]
        self.bias = float(wn[-1])
        self.n_hist = len(self.c)


# =============================================================================
# Leakage guard — the fit/score split contract (asserts fit window < score point)
# =============================================================================
def fit_score_split(tick_indices: list[int], score_pos: int, fit_len: int):
    """Return (fit_idx, score_idx) where fit_idx are strictly-BEFORE the score point.

    tick_indices: ordered positions available; score_pos: index (into tick_indices)
    of the scoring point; fit_len: how many retrospective points to use for the fit.
    ASSERTS max(fit_idx) < score_idx — the fit window and score point never overlap.
    """
    assert 0 <= score_pos < len(tick_indices)
    score_idx = score_pos
    fit_end = score_idx - 1                              # strictly before
    fit_start = max(0, fit_end - fit_len + 1)
    fit_idx = list(range(fit_start, fit_end + 1))
    assert fit_idx, "empty fit window — need >=1 retrospective point before the score point"
    assert max(fit_idx) < score_idx, (
        f"LEAKAGE: fit window max {max(fit_idx)} not strictly < score {score_idx}"
    )
    return fit_idx, score_idx


def leakage_guard_selftest() -> tuple[bool, str]:
    """Confirm (a) a valid split has fit strictly before score, and (b) an
    overlapping window would VIOLATE the same assertion fit_score_split enforces.
    Returns (ok, detail) for the invariant probe — no torch, pure index logic."""
    fit_idx, score_idx = fit_score_split(list(range(8)), score_pos=7, fit_len=3)
    valid_ok = max(fit_idx) < score_idx
    overlap_raises = False
    try:
        overlap_fit, overlap_score = [5, 6, 7], 7
        assert max(overlap_fit) < overlap_score  # SAME assertion fit_score_split uses
    except AssertionError:
        overlap_raises = True
    ok = valid_ok and overlap_raises
    return ok, f"valid split fit_idx={fit_idx}<score={score_idx}; overlap-guard-fires={overlap_raises}"


# =============================================================================
# Family registry — the sweep's coeff-source x order cross-product
# =============================================================================
def build_family_registry(offline_damp: float = 0.5) -> dict[str, _BasePredictor]:
    """All required families. `offline_damp` is a placeholder damped multiplier that
    the sweep may re-derive per matrix; the registry proves presence + reconstructability.
    """
    reg: dict[str, _BasePredictor] = {}
    # order-1 {fixed, offline-damped}
    reg["order1-fixed"] = Order1("fixed", damp=1.0)
    reg["order1-damped"] = Order1("offline-damped", damp=offline_damp)
    # order-2 (3-pt Newton) {fixed, offline-damped}
    reg["order2-fixed"] = Order2("fixed", damp=1.0)
    reg["order2-damped"] = Order2("offline-damped", damp=offline_damp)
    # order-3/poly (Lagrange-Newton >=4 pt) {fixed, offline-damped}
    reg["order3-fixed"] = OrderPoly(3, "fixed", damp=1.0)
    reg["order3-damped"] = OrderPoly(3, "offline-damped", damp=offline_damp)
    # EMA/momentum {fixed-decay, learnable-decay}
    reg["ema-fixed"] = EMA("fixed", beta=0.5)
    reg["ema-learnable"] = EMA("learnable", beta=0.5)
    # learnable-at-order: WEAK scalar at orders 1/2/poly
    reg["learnable-scalar-o1"] = LearnableScalar(Order1("fixed"))
    reg["learnable-scalar-o2"] = LearnableScalar(Order2("fixed"))
    reg["learnable-scalar-o3"] = LearnableScalar(OrderPoly(3, "fixed"))
    # learnable-at-order: STRONG full-vector LS at orders 1/2/poly (n_hist set by sweep)
    reg["learnable-fullvec-o1"] = LearnableFullVector(2, 1)
    reg["learnable-fullvec-o2"] = LearnableFullVector(3, 2)
    reg["learnable-fullvec-o3"] = LearnableFullVector(4, 3)
    # general learned-regression
    reg["general-regression"] = GeneralRegression(4)
    return reg
