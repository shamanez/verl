#!/usr/bin/env python3
"""weight_proj/metrics.py — SOLE OWNER of the GPU-free weight-projection metric math.

Boundary B1 (plan EXP-44 / issue #45): this module owns the engine metric CODE;
issue #45 owns ONLY the proxy-ranking layer + its own report builder. The two
never edit the same file. Any consumer (#45/#52/#53/#54) that needs a metric
imports it from HERE and asserts VERBATIM equality against the definitions below.

================================================================================
CANONICAL METRIC DEFINITIONS (the single source of truth; #45 asserts these)
================================================================================

Setup. At scoring point (tick) `t` we compare, for a matrix / block / layer:
  theta_now   = theta[t]                 the current (truth) weights at t
  theta_stale = theta[t - K]             the stale reference (K = Delta, in ticks)
  theta_hat   = predictor.predict(...)   the predicted weights extrapolated to t

Everything is done in float32. All tensors are flattened to 1-D vectors before
the vector algebra; a "group" (block/layer) concatenates its member matrices'
flattened vectors into ONE vector so the group ratio is the true joint ratio, not
an average of per-matrix ratios.

Error and baseline displacement vectors:
  e = theta_hat - theta_now              the predictor's residual (what's left)
  b = theta_stale - theta_now            the raw-stale baseline displacement
                                         (== -(theta_now - theta_stale))

  NOTE the denominator is the RAW-STALE displacement ||theta_stale - theta_now||,
  i.e. "how far the weights moved over the Delta window". A predictor that does
  nothing (theta_hat = theta_stale) scores weight_proj_ratio == 1 exactly.

--------------------------------------------------------------------------------
weight_proj_ratio  =  ||theta_hat - theta_now||  /  ||theta_stale - theta_now||
                   =  ||e|| / ||b||
--------------------------------------------------------------------------------
  Ratio < 1  => the predictor beats doing-nothing (closed part of the gap).
  Ratio == 1 => no skill (h=0 identity / pure stale). Ratio > 1 => harmful.
  Denominator guard: if ||b|| <= eps the window carries no signal -> NaN (caller
  treats NaN as "undefined", never as skill).

--------------------------------------------------------------------------------
dir_cos  =  <e, b> / (||e|| * ||b||)      cosine between residual and baseline
--------------------------------------------------------------------------------
  Direction alignment of the leftover error with the displacement it had to cover.

--------------------------------------------------------------------------------
Radial / tangential split of e along the UNIT baseline-displacement direction
  u = b / ||b||                          unit displacement direction
  radial     = <e, u>                    signed scalar component of e along u
  tangential = || e - <e,u> u ||         Euclidean norm of the orthogonal part
--------------------------------------------------------------------------------
  Identity check: radial^2 + tangential^2 == ||e||^2 (up to fp32 rounding).
  Radial = "how much of the leftover error is still along the move direction"
  (under/over-shoot); tangential = "how much points sideways" (wrong direction).

--------------------------------------------------------------------------------
skill  =  1 - weight_proj_ratio^2
--------------------------------------------------------------------------------
  Fraction of the baseline displacement ENERGY removed by the predictor.
  skill in (-inf, 1]; skill == 0 at ratio 1; skill -> 1 as ||e|| -> 0.

--------------------------------------------------------------------------------
SNR  =  ||e||  /  noise_floor
--------------------------------------------------------------------------------
  Residual magnitude in units of the bf16 noise floor for the SAME group (see
  noise_floor.py). NOTE (EXP-44 correction): `noise_floor` is the bf16 DIFFERENCED
  floor (correlated-difference quantization noise on the CHANGED support), NOT the
  ||theta||-scaled STORAGE floor (a category error over-estimating the true floor by
  ~600-2200x). The true correlated floor of an unchanging value is the EMPIRICAL
  ZERO-MOTION NULL == 0.0; `differenced_floor` is reported as an honest 0.5-ULP
  UPPER-BOUND. The bf16-reliability GATE is the DIRECTEDNESS discriminator
  (noise_floor.directedness_exponent): cumulative displacement ~ h^p with p >= 0.8
  => DIRECTED drift (real signal), which rounding noise (random walk p~0.5) cannot
  produce. This SNR (formula unchanged; #45 asserts it verbatim) is kept as a
  secondary numeric context; the GATE decision uses directedness + zero-motion null.

--------------------------------------------------------------------------------
crossover  h*  =  largest h such that median_over_scoring_points(ratio(h)) < 1.0
--------------------------------------------------------------------------------
  The furthest horizon (in ticks/steps ahead) at which the family still, on
  median, beats doing-nothing. h* = None if the median ratio is >= 1 at every h
  (never skillful) — reported as h* = 0 sentinel by convention when no h qualifies.

================================================================================
All functions take float32 torch tensors (already flattened by the caller for
group metrics) OR numpy arrays; they cast to float32 numpy internally so the math
is dtype-stable and identical regardless of caller. NO bf16 arithmetic happens
here — differencing near-equal bf16 tensors is done upstream in fp32.
"""
from __future__ import annotations

import numpy as np

# ---- canonical constants (consumers assert against these) --------------------
DENOM_EPS = 1e-12          # ||b|| guard for weight_proj_ratio / dir_cos / radial
SNR_FLOOR_THRESH = 3.0     # SNR <= this => (group,h) flagged `bf16-unreliable`
CROSSOVER_SENTINEL = 0     # h* when NO horizon has median ratio < 1

# machine-checkable string tag of the metric contract version; #45 can assert it
METRIC_CONTRACT = "weight-proj-metrics-v1"


def _f32(x) -> np.ndarray:
    """Flatten to a 1-D float32 numpy array (dtype-stable; no bf16 arithmetic)."""
    if hasattr(x, "detach"):  # torch tensor
        x = x.detach().to("cpu").to(dtype=__import__("torch").float32).numpy()
    a = np.asarray(x, dtype=np.float64)   # accumulate in f64, report f32-safe
    return a.reshape(-1)


def l2(x) -> float:
    """Euclidean norm (float)."""
    return float(np.linalg.norm(_f32(x)))


def error_vector(theta_hat, theta_now) -> np.ndarray:
    """e = theta_hat - theta_now (fp32)."""
    return _f32(theta_hat) - _f32(theta_now)


def baseline_displacement(theta_stale, theta_now) -> np.ndarray:
    """b = theta_stale - theta_now  (the raw-stale displacement; ratio denominator)."""
    return _f32(theta_stale) - _f32(theta_now)


def weight_proj_ratio(theta_hat, theta_now, theta_stale) -> float:
    """||theta_hat - theta_now|| / ||theta_stale - theta_now||. NaN if denom ~ 0."""
    e = error_vector(theta_hat, theta_now)
    b = baseline_displacement(theta_stale, theta_now)
    den = float(np.linalg.norm(b))
    if den <= DENOM_EPS:
        return float("nan")
    return float(np.linalg.norm(e)) / den


def dir_cos(theta_hat, theta_now, theta_stale) -> float:
    """<e,b> / (||e|| ||b||). NaN if either norm ~ 0."""
    e = error_vector(theta_hat, theta_now)
    b = baseline_displacement(theta_stale, theta_now)
    ne = float(np.linalg.norm(e))
    nb = float(np.linalg.norm(b))
    if ne <= DENOM_EPS or nb <= DENOM_EPS:
        return float("nan")
    return float(np.dot(e, b)) / (ne * nb)


def radial_tangential(theta_hat, theta_now, theta_stale) -> tuple[float, float]:
    """Split e along the UNIT baseline-displacement direction u = b/||b||.

    Returns (radial, tangential) with radial = <e,u> (signed) and
    tangential = ||e - <e,u> u||. Guarantees radial^2 + tangential^2 == ||e||^2.
    """
    e = error_vector(theta_hat, theta_now)
    b = baseline_displacement(theta_stale, theta_now)
    nb = float(np.linalg.norm(b))
    if nb <= DENOM_EPS:
        return float("nan"), float("nan")
    u = b / nb
    radial = float(np.dot(e, u))
    tang_vec = e - radial * u
    tangential = float(np.linalg.norm(tang_vec))
    return radial, tangential


def skill(ratio: float) -> float:
    """1 - weight_proj_ratio^2 (energy fraction of the displacement removed)."""
    if ratio is None or (isinstance(ratio, float) and np.isnan(ratio)):
        return float("nan")
    return 1.0 - float(ratio) ** 2


def snr(theta_hat, theta_now, noise_floor: float) -> float:
    """||theta_hat - theta_now|| / noise_floor. NaN if floor ~ 0."""
    e = error_vector(theta_hat, theta_now)
    if noise_floor is None or noise_floor <= DENOM_EPS:
        return float("nan")
    return float(np.linalg.norm(e)) / float(noise_floor)


def is_bf16_unreliable(snr_value: float, thresh: float = SNR_FLOOR_THRESH) -> bool:
    """True => residual is at/below the bf16 floor; flag, do NOT report as a ratio."""
    if snr_value is None or np.isnan(snr_value):
        return True
    return float(snr_value) <= float(thresh)


def crossover_hstar(h_to_ratios: dict[int, list[float]]) -> int:
    """h* = largest h whose MEDIAN ratio over scoring points is < 1.0.

    `h_to_ratios` maps horizon h -> list of weight_proj_ratio values (one per
    scoring point). NaNs are dropped before the median. Returns CROSSOVER_SENTINEL
    (0) if no horizon qualifies.
    """
    best = CROSSOVER_SENTINEL
    for h in sorted(h_to_ratios):
        vals = [v for v in h_to_ratios[h] if v is not None and not np.isnan(v)]
        if not vals:
            continue
        med = float(np.median(vals))
        if med < 1.0:
            best = max(best, int(h))
    return best


def full_metric_row(theta_hat, theta_now, theta_stale, noise_floor: float | None) -> dict:
    """Compute the FULL per-(group,h) metric row in one pass (fp32).

    Returns a dict with every canonical metric; the caller aggregates rows into
    per-(family x order x coeff x Delta x h x grouping) records and computes h*
    across scoring points afterwards.
    """
    e = error_vector(theta_hat, theta_now)
    b = baseline_displacement(theta_stale, theta_now)
    ne = float(np.linalg.norm(e))
    nb = float(np.linalg.norm(b))
    ratio = float("nan") if nb <= DENOM_EPS else ne / nb
    if ne <= DENOM_EPS or nb <= DENOM_EPS:
        dcos = float("nan")
    else:
        dcos = float(np.dot(e, b)) / (ne * nb)
    if nb <= DENOM_EPS:
        radial = tangential = float("nan")
    else:
        u = b / nb
        radial = float(np.dot(e, u))
        tangential = float(np.linalg.norm(e - radial * u))
    snr_v = (float("nan") if (noise_floor is None or noise_floor <= DENOM_EPS)
             else ne / float(noise_floor))
    return {
        "err_norm": ne,
        "base_norm": nb,
        "weight_proj_ratio": ratio,
        "dir_cos": dcos,
        "radial": radial,
        "tangential": tangential,
        "skill": skill(ratio),
        "snr": snr_v,
        "bf16_unreliable": is_bf16_unreliable(snr_v),
    }


# =============================================================================
# Regression view (v1 REGRESSION contract — ADDITIVE; every function above and
# METRIC_CONTRACT itself are byte-identical to weight-proj-metrics-v1)
# =============================================================================
# machine-checkable string tag of the regression-metric contract version
REGRESSION_CONTRACT = "weight-proj-regression-v1"
PRED_HIST_BINS = 200            # per-scalar prediction-R² histogram bins
PRED_HIST_RANGE = (-1.0, 1.0)   # values clipped into this range for the histogram
PRED_R2_STRONG = 0.7            # paper's strong-linearity line reused for prediction


def pooled_evr(sum_e2: float, sum_b2: float) -> float:
    """Pooled explained-variance ratio vs the stale baseline over SCORED windows.

    pred_evr_pooled = 1 - sum_w ||e_w||^2 / sum_w ||b_w||^2.
    NaN iff sum_b2 <= DENOM_EPS**2 (the only permitted NaN). A predictor that
    holds the stale weights (e == b per window) scores EXACTLY 0.
    """
    if sum_b2 <= DENOM_EPS ** 2:
        return float("nan")
    return 1.0 - float(sum_e2) / float(sum_b2)


def per_scalar_pred_r2(Yhat: np.ndarray, Ytrue: np.ndarray,
                       const_eps: float = 1e-300):
    """Classical per-coordinate R² of predicted vs ACTUAL FUTURE weights.

    Yhat, Ytrue: f64 [W, k] (W scored windows). For coordinate i:
      ybar_i   = mean_w Ytrue[w, i]
      SS_res_i = sum_w (Yhat[w,i] - Ytrue[w,i])^2
      SS_tot_i = sum_w (Ytrue[w,i] - ybar_i)^2
      R2_i     = 1 - SS_res_i / SS_tot_i    # NOT clamped; can be negative
    const_mask_i = SS_tot_i <= const_eps (excluded + counted; mirrors the
    linearity-R² R2_CONST_EPS semantics). Returns (r2[k], const_mask[k]).
    """
    Yhat = np.asarray(Yhat, dtype=np.float64)
    Ytrue = np.asarray(Ytrue, dtype=np.float64)
    ybar = Ytrue.mean(axis=0)
    ss_res = ((Yhat - Ytrue) ** 2).sum(axis=0)
    ss_tot = ((Ytrue - ybar) ** 2).sum(axis=0)
    const_mask = ss_tot <= const_eps
    with np.errstate(invalid="ignore", divide="ignore"):
        r2 = np.where(const_mask, np.nan, 1.0 - ss_res / ss_tot)
    return r2, const_mask


def pred_r2_summary(r2: np.ndarray, const_mask: np.ndarray):
    """(hist[PRED_HIST_BINS] over PRED_HIST_RANGE with values clipped into [-1,1],
    n_excluded_const, n_valid, frac_lt_0). Mergeable by summation across matrices."""
    valid = r2[~const_mask]
    valid = valid[np.isfinite(valid)]
    clipped = np.clip(valid, PRED_HIST_RANGE[0], PRED_HIST_RANGE[1])
    hist, _ = np.histogram(clipped, bins=PRED_HIST_BINS, range=PRED_HIST_RANGE)
    frac_lt_0 = float(np.mean(valid < 0.0)) if valid.size else float("nan")
    return hist.astype(np.int64), int(np.sum(const_mask)), int(valid.size), frac_lt_0


def pred_r2_from_hist(hist: np.ndarray):
    """(median, frac_gt_0.7, frac_lt_0) read off a merged prediction-R² histogram
    (cumsum median; bins with center > PRED_R2_STRONG; bins with center < 0)."""
    hist = np.asarray(hist, dtype=np.float64)
    n = float(hist.sum())
    if n <= 0:
        return float("nan"), float("nan"), float("nan")
    lo, hi = PRED_HIST_RANGE
    centers = lo + (np.arange(PRED_HIST_BINS) + 0.5) * (hi - lo) / PRED_HIST_BINS
    cum = np.cumsum(hist)
    mi = int(np.searchsorted(cum, (n + 1.0) / 2.0))
    med = float(centers[min(mi, PRED_HIST_BINS - 1)])
    frac_gt = float(hist[centers > PRED_R2_STRONG].sum()) / n
    frac_lt0 = float(hist[centers < 0.0].sum()) / n
    return med, frac_gt, frac_lt0
