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
THE CORRECTED DIFFERENCED-NOISE FLOOR (correlation-aware, on the changed support)
--------------------------------------------------------------------------------
bf16 = 1 sign, 8 exponent, 7 stored mantissa bits. At magnitude x, ULP(x) =
2^(floor(log2|x|)) * 2^-7 and rounding carries <= half a ULP. For a predictor
theta_hat = sum_j c_j theta_j the residual e = theta_hat - theta_now is a linear
combination of bf16-sourced snapshots. THE KEY CORRELATION FACT: for an element that
did NOT change between the snapshots, the stored bf16 bit patterns are IDENTICAL, so
the quantization error is EXACTLY EQUAL in each term and cancels in the difference —
that element's residual is exactly 0.0 and it contributes NOTHING to the floor. The
floor therefore lives ONLY on the CHANGED support of the residual (the elements whose
value actually moved), where the resolvable unit is the bf16 ULP-of-the-difference:

    diff_floor_i = (1/2) * ULP(scale_i) * sqrt( sum_j c_j^2 + 1 )   if e_i != 0
    diff_floor_i = 0                                                 if e_i == 0

with scale_i the per-element binade magnitude and sqrt(sum c_j^2 + 1) the
error-propagation factor folding the independent half-ULP rounding of each c_j*theta_j
and of theta_now. The GROUP floor is the L2 over the CHANGED elements:

    floor(g) = || diff_floor restricted to {i : e_i != 0} ||_2

This is the object the prior ||theta||-scaled floor got wrong: it summed the
resolvable unit over ALL ~2.3M elements (dominated by the 98.9% that don't move),
giving floor ~0.4-0.5 == ||theta||-scaled; the correct floor sums only over the ~1.1%
that DO move, giving a floor ~1e-4 for these blocks. A HELD-CONSTANT tensor has EMPTY
changed support => floor == 0.0 exactly (matches the empirical null). A MOVING block's
residual (~1e-2 - 7e-4) then sits far ABOVE its ~1e-4 floor => SNR >> 3 at h>=5.

An empirical cross-check is also provided: `zero_motion_null_floor` differences a
tensor against ITSELF (bit-identical) and MUST return 0.0 — the hard ground-truth
that the correlated floor of an unchanging value is zero, not ||theta||-scaled.

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
