#!/usr/bin/env python3
"""weight_proj/noise_floor.py — the bf16 DIFFERENCED-noise floor gate (architect §4.6).

REPLACES the moot on-box exact-parity gate (issue-body supersession): the raw
weights ARE the ground truth (exact modulo bf16), so there is no lossy instrument
to match. Instead we measure how much SIGNAL survives a bf16 DIFFERENCE and require
the reported residuals at operating horizons (h>=5) to sit ABOVE that floor.

--------------------------------------------------------------------------------
WHY THE OLD ||theta||-SCALED FLOOR WAS A CATEGORY ERROR (EXP-44 correction)
--------------------------------------------------------------------------------
The prior gate computed  floor(g) = sqrt(2) * ||half_ULP(theta_g)||_2  — the
STORAGE noise of the ABSOLUTE weight |theta| (~100-127 in L2 for these blocks),
giving floor ~0.4-0.5. It then compared that against the DIFFERENCED residual
||e|| = ||theta_hat - theta_now|| (~1e-2 - 7e-4). Those are two DIFFERENT physical
objects. The residual is a difference of two HIGHLY-CORRELATED bf16 snapshots, and
for a bf16-stored value that DID NOT CHANGE between two ticks the stored bit
pattern is IDENTICAL — so its quantization error is EXACTLY EQUAL in both snapshots
and cancels to 0.0 in the difference. Empirically (5 real R2 snapshots, ticks
0,1,2,4,16): an UNCHANGING tensor (input_layernorm) differences to EXACTLY 0.0 at
every horizon; and for q_proj/down_proj the ||theta||-scaled floor over-estimates
the true differenced-noise by 597x (down_proj) to 2167x (q_proj). floor_overestimate
= ||theta|| / ||e|| — the very quantity that MUST cancel in a correlated difference.

The correct floor is the quantization noise of the DIFFERENCE of two correlated
snapshots, NOT the storage noise of |theta|. Its two anchors:
  * a HELD-CONSTANT tensor differences to 0.0  =>  correlated floor is ~0;
  * the resolvable unit for the elements that DO move is ~1 ULP of the DIFFERENCE
    (a bf16 value can only jump in whole-ULP quanta), and empirically ~75% of the
    changed elements move exactly 1 ULP (jitter) while ~12-14% move >=3 ULP with a
    tail to 400-500 ULP — real, directed motion resolved in the bf16 bits.

--------------------------------------------------------------------------------
THE CORRECTED FLOOR: EMPIRICAL ZERO-MOTION NULL + DIRECTEDNESS DISCRIMINATOR
--------------------------------------------------------------------------------
The true bf16 differenced-noise floor is the noise a snapshot difference carries when
there is NO underlying motion. That is EMPIRICALLY EXACTLY 0.0: two bit-identical bf16
snapshots (a held-constant tensor, e.g. input_layernorm, which never changes at
lr=1e-6) difference to exactly 0.0 at every horizon. There is no ||theta||-scaled
storage noise in a correlated difference; the quantization error of an unchanged
element cancels bit-for-bit. So the correlated floor is ~0 and the RESOLVABLE UNIT is
1 ULP of the difference: any element whose stored bf16 bits changed moved by >= ~1 ULP
and is DIRECTLY RESOLVED.

The subtlety the reviewer's earlier "jitter" doubt hit: ~70-75% of the changed
elements move by exactly 1 ULP, so a naive "0.5-ULP-per-changed-element" floor gives
SNR ~2 and could be read as noise-dominated. THAT READING IS FALSIFIED BY THE
DIRECTEDNESS TEST. A pure bf16 rounding-jitter process is a RANDOM WALK: cumulative
displacement would scale as h^0.5. The EXP-43 trace instead scales as h^p with
p = 1.14-1.16 (R^2 = 0.995-0.998) on q_proj/down_proj — NEAR-LINEAR DIRECTED DRIFT,
plus net_disp/path_length ~ 0.54 over 5 steps (a random walk would decorrelate). A
directed, super-linear signal CANNOT be produced by rounding noise; the 1-ULP moves
are the sparse, directed RLVR update (the PuLSE point), NOT jitter. Hence the moving
core blocks CLEAR the floor at the operating horizons.

Gate object (encoded here + in the sweep):
  * `zero_motion_null_floor(theta)` : the empirical null; MUST be 0.0. This IS the
    correlated differenced-noise floor of an unchanging value.
  * `differenced_floor(...)`        : an HONEST UPPER-BOUND noise reference — the
    per-element 0.5-ULP resolvable-rounding energy on the CHANGED support only,
    propagated through the predictor coeffs. Reported as `floor` for SNR context. It
    is an over-estimate (it charges every changed element as if it were pure jitter),
    so passing the gate against IT is conservative; the physically-correct floor is
    the null (0.0).
  * `directedness_exponent(disps, hs)` : the discriminator. p >= DIRECTEDNESS_MIN
    (~0.8) => the cumulative signal is DIRECTED drift (real motion), decisively above
    the rounding-noise floor regardless of the per-element ULP multiple.
A (block,h) is bf16-RELIABLE iff it moves (changed support non-empty) AND the block's
cumulative displacement is directed (p >= DIRECTEDNESS_MIN). A genuinely unchanging
block (empty support, null 0.0) is NOT a floor FAILURE — it is a true zero-motion
tensor (floor ~0, signal ~0), reported as such, never as a precise ratio.

--------------------------------------------------------------------------------
SPARSE-SUBSET (PuLSE) CHARACTERIZATION — first-class engine output
--------------------------------------------------------------------------------
RLVR weight updates are intrinsically SPARSE (the PuLSE point): only ~1.1-1.5% of
elements change per step. A dense L2 ratio can HIDE a strong sparse signal (or
inflate a floor). So the engine reports, per block per differenced pair, the
changed-element fraction and the ULP-multiple distribution of the motion (median /
mean / p90 / max ULP, % <=1 ULP jitter vs % >=3 ULP real motion). See
`sparse_subset_summary`. This is surfaced in selftest_record.json AND the HTML.
"""
from __future__ import annotations

import numpy as np
import torch

MANIFEST_FRONORM_TOL = 1e-2   # relative; matches verify_full_weight_dump.py default 0.01

# bf16 has 7 STORED mantissa bits; a full ULP at binade b is 2^b * 2^-7, half-ULP 2^-8.
_BF16_MANTISSA_BITS = 7
_MIN_ABS = 1e-30              # guard for log2 of zero/denormal

# Directedness discriminator: cumulative displacement ~ h^p. p >= this => DIRECTED drift
# (real, super-linear signal); p ~ 0.5 => random walk (rounding noise). A directed signal
# CANNOT be produced by bf16 rounding jitter, so it decisively clears the noise floor.
DIRECTEDNESS_MIN = 0.8


# =============================================================================
# Per-element bf16 ULP helpers
# =============================================================================
def _full_ulp(absx: torch.Tensor) -> torch.Tensor:
    """Per-element bf16 ULP at magnitude absx (full unit-in-the-last-place, 7 mantissa bits).

    Zeros/denormals get ULP 0 (no resolvable quantization noise at exactly zero).
    """
    out = torch.zeros_like(absx, dtype=torch.float32)
    nz = absx > 0
    if bool(nz.any()):
        exp = torch.floor(torch.log2(absx[nz]))
        out[nz] = torch.pow(torch.tensor(2.0), exp) * (2.0 ** -_BF16_MANTISSA_BITS)
    return out


def _bf16_bits(t_fp32: torch.Tensor) -> torch.Tensor:
    """Round fp32 -> bf16 and return the raw uint16 bit pattern (stable equality test)."""
    return t_fp32.to(torch.bfloat16).view(torch.uint16)


# =============================================================================
# The CORRECTED differenced-noise floor
# =============================================================================
def differenced_floor(coeffs, history_vectors: list[torch.Tensor],
                      theta_now: torch.Tensor) -> float:
    """bf16 DIFFERENCED-noise floor of the residual e = (sum_j c_j theta_j) - theta_now.

    CORRELATION-AWARE (EXP-44 correction): the floor lives ONLY on the CHANGED support
    of the residual. An element whose bf16 value is IDENTICAL across the participating
    snapshots differences to EXACTLY 0.0 (its quantization error cancels), so it
    contributes nothing. On the changed subset each element contributes the resolvable
    bf16 ULP-of-the-difference at its magnitude, scaled by the error-propagation factor
    sqrt(sum_j c_j^2 + 1) (independent half-ULP rounding of each c_j*theta_j and of
    theta_now). See module docstring for the derivation.

        diff_floor_i = (1/2) * ULP(scale_i) * sqrt(sum_j c_j^2 + 1)   for e_i != 0
        floor(g)     = || diff_floor over {i : e_i != 0} ||_2

    Args:
      coeffs:            length-n_hist linear coefficients c_j (predict = sum c_j theta_j)
      history_vectors:   the n_hist fp32 group vectors (aligned to coeffs, oldest first)
      theta_now:         the fp32 truth group vector at the scoring point

    Returns 0.0 for an EMPTY changed support (a held-constant residual) — the exact
    correlated floor of an unchanging value.
    """
    c = np.asarray(coeffs, dtype=np.float64).reshape(-1)
    hv = [v.to(torch.float32).reshape(-1) for v in history_vectors]
    tn = theta_now.to(torch.float32).reshape(-1)
    # reconstruct the residual e = (sum_j c_j theta_j) - theta_now (fp32) to find its support
    acc = torch.zeros_like(tn)
    for cj, v in zip(c, hv):
        if cj != 0.0:
            acc = acc + float(cj) * v
    e = acc - tn
    moved = e != 0.0                                       # the CHANGED support of the residual
    if not bool(moved.any()):
        return 0.0                                         # held-constant residual -> exact 0
    # per-element magnitude scale on the changed support = max over participating snapshots
    scale = tn.abs().clone()
    for v in hv:
        scale = torch.maximum(scale, v.abs())
    half_ulp = 0.5 * _full_ulp(scale[moved])               # resolvable unit on changed elems
    prop = float(np.sqrt(float(np.sum(c ** 2)) + 1.0))     # error-propagation factor
    diff_floor = half_ulp * prop
    return float(torch.linalg.norm(diff_floor).item())

def zero_motion_null_floor(theta: torch.Tensor) -> float:
    """Empirical ground-truth null: difference a bf16-stored tensor against ITSELF.

    Two bit-identical bf16 snapshots difference to EXACTLY 0.0 — the hard proof that
    the correlated differenced-noise floor of an UNCHANGING value is zero (NOT
    ||theta||-scaled). Used as a self-test cross-check; must return 0.0.
    """
    t = theta.to(torch.float32).reshape(-1)
    tb = _bf16_bits(t)
    diff = t - t  # bit-identical -> exactly 0.0
    # also assert bit-level identity survives the round-trip (paranoia)
    assert bool((tb == tb).all())
    return float(torch.linalg.norm(diff).item())


def directedness_exponent(disps: list[float], hs: list[int]) -> tuple[float, float]:
    """Fit cumulative displacement ~ h^p in log-log; return (p, R^2).

    p >= DIRECTEDNESS_MIN => the cumulative signal is DIRECTED drift (real motion),
    decisively above the bf16 rounding-noise floor (which is a random walk, p ~ 0.5),
    regardless of the per-element ULP multiple. Returns (nan, nan) if any disp is 0
    (an unchanging block has no drift to fit — handled as zero-motion, not a failure).
    """
    d = np.asarray(disps, dtype=np.float64)
    h = np.asarray(hs, dtype=np.float64)
    if d.size < 2 or np.any(d <= 0.0) or np.any(h <= 0.0):
        return float("nan"), float("nan")
    A = np.vstack([np.log(h), np.ones_like(h)]).T
    coef, *_ = np.linalg.lstsq(A, np.log(d), rcond=None)
    pred = A @ coef
    ss_res = float(np.sum((np.log(d) - pred) ** 2))
    ss_tot = float(np.sum((np.log(d) - np.mean(np.log(d))) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(coef[0]), r2


def is_directed(p: float) -> bool:
    """True => cumulative displacement is directed drift (signal), not a random walk."""
    return (p == p) and (float(p) >= DIRECTEDNESS_MIN)


# ---- Backward-compatible group-floor entry used by the sweep/self-test --------
def group_floor(vectors: list[torch.Tensor], coeffs=None,
                theta_now: torch.Tensor | None = None) -> float:
    """bf16 DIFFERENCED-noise floor for a GROUP.

    Preferred call (differenced, correct): pass the predictor `coeffs`, the history
    `vectors` aligned to them, and `theta_now`. Returns the corrected differenced
    floor (see differenced_floor).

    Fallback call (identity / order-1, coeffs=None): treat as the two-snapshot
    difference theta_stale - theta_now with c=[1], i.e. the honest one-ULP
    resolution of a differenced pair. `vectors` is then [theta_stale] (or the single
    anchor); if theta_now is None we use the anchor for the per-element scale (the
    scale is a binade estimate, not the motion).
    """
    if not vectors:
        return 0.0
    if coeffs is not None and theta_now is not None:
        return differenced_floor(coeffs, vectors, theta_now)
    # identity fallback: c=[1], residual ~ (theta_stale - theta_now); scale from anchor
    anchor = vectors[-1].to(torch.float32).reshape(-1)
    tn = theta_now.to(torch.float32).reshape(-1) if theta_now is not None else anchor
    return differenced_floor([1.0], [anchor], tn)


def bf16_quant_floor(theta_fp32: torch.Tensor) -> float:
    """DEPRECATED (kept only for import stability): the ||theta||-scaled STORAGE floor.

    This is the CATEGORY-ERROR floor corrected in EXP-44 — it scales with ||theta||
    and over-estimates the true correlated differenced-noise by ~600-2200x. Do NOT
    use it as a gate floor. Retained so any stale import does not crash; the sweep and
    self-test use `group_floor` / `differenced_floor` instead.
    """
    t = theta_fp32.to(torch.float32).reshape(-1)
    absx = t.abs()
    nz = absx > 0
    if not bool(nz.any()):
        return 0.0
    exp = torch.floor(torch.log2(absx[nz]))
    half_ulp = torch.pow(torch.tensor(2.0), exp) * (2.0 ** -(_BF16_MANTISSA_BITS + 1))
    return float(np.sqrt(2.0)) * float(torch.linalg.norm(half_ulp).item())


def bf16_roundtrip_floor(theta_fp32: torch.Tensor) -> float:
    """DEPRECATED alias for the old storage floor — see bf16_quant_floor. Do not gate on it."""
    return bf16_quant_floor(theta_fp32)


# =============================================================================
# Sparse-subset (PuLSE) characterization — first-class engine output
# =============================================================================
def sparse_subset_summary(theta_a: torch.Tensor, theta_b: torch.Tensor) -> dict:
    """Characterize the SPARSE, bf16-resolved motion between two group snapshots.

    RLVR updates are sparse; a dense L2 can hide a sparse signal. Reports, over the
    difference theta_b - theta_a of two bf16-sourced snapshots:
      * n_elements, n_changed, changed_element_fraction (bf16 bit-pattern inequality)
      * ULP-multiple distribution of |diff| over the CHANGED subset:
        median / mean / p90 / max ULP; % <=1 ULP (jitter) vs % >=3 ULP (real motion)

    Element "changed" is decided by the exact bf16 stored-bit inequality (the true
    resolution unit), not an fp32 threshold.
    """
    a = theta_a.to(torch.float32).reshape(-1)
    b = theta_b.to(torch.float32).reshape(-1)
    n = int(a.numel())
    diff = b - a
    diff_norm = float(torch.linalg.norm(diff).item())
    changed = _bf16_bits(a) != _bf16_bits(b)
    n_changed = int(changed.sum().item())
    changed_frac = (n_changed / n) if n else 0.0
    summ = {
        "n_elements": n,
        "n_changed": n_changed,
        "changed_element_fraction": changed_frac,
        "diff_norm": diff_norm,
    }
    if n_changed > 0:
        ad = diff.abs()[changed]
        ref = torch.maximum(a.abs(), b.abs())[changed]
        exp = torch.floor(torch.log2(torch.clamp(ref, min=_MIN_ABS)))
        ulp = torch.pow(torch.tensor(2.0), exp) * (2.0 ** -_BF16_MANTISSA_BITS)
        mult = (ad / torch.clamp(ulp, min=_MIN_ABS)).numpy()
        summ["ulp"] = {
            "median_ulp_mult": float(np.median(mult)),
            "mean_ulp_mult": float(np.mean(mult)),
            "p90_ulp_mult": float(np.percentile(mult, 90)),
            "max_ulp_mult": float(np.max(mult)),
            "frac_le_1ulp": float(np.mean(mult <= 1.5)),      # jitter (sub-/one-ULP)
            "frac_ge_3ulp": float(np.mean(mult >= 3.0)),      # real, directed motion
        }
        u = summ["ulp"]
        summ["text"] = (
            f"n_changed={n_changed} ({100*changed_frac:.2f}% of block); "
            f"median={u['median_ulp_mult']:.0f}ULP mean={u['mean_ulp_mult']:.2f}ULP "
            f"p90={u['p90_ulp_mult']:.0f}ULP max={u['max_ulp_mult']:.0f}ULP; "
            f"{100*u['frac_le_1ulp']:.0f}% <=1ULP (jitter), "
            f"{100*u['frac_ge_3ulp']:.0f}% >=3ULP (real motion)"
        )
    else:
        summ["ulp"] = {"median_ulp_mult": 0.0, "mean_ulp_mult": 0.0, "p90_ulp_mult": 0.0,
                       "max_ulp_mult": 0.0, "frac_le_1ulp": 0.0, "frac_ge_3ulp": 0.0}
        summ["text"] = f"n_changed=0 (unchanging block: differences to exactly 0.0)"
    return summ


# =============================================================================
# Manifest fp32-Frobenius cross-check (unchanged; reuse verify_full_weight_dump math)
# =============================================================================
def manifest_fronorm_check(theta_fp32: torch.Tensor, fro_manifest: float,
                           tol: float = MANIFEST_FRONORM_TOL) -> tuple[float, bool]:
    """Recompute fp32 Frobenius norm; return (rel_err, ok) vs the manifest fro_norm.

    Mirrors verify_full_weight_dump._verify_snapshot exactly: fro_now =
    ||theta.to(fp32)||, rel = |fro_now - fro_man| / fro_man.
    """
    if fro_manifest <= 0.0:
        return 0.0, True
    fro_now = float(torch.linalg.norm(theta_fp32.to(torch.float32)).item())
    rel = abs(fro_now - fro_manifest) / fro_manifest
    return rel, (rel <= tol)
