#!/usr/bin/env python3
"""weight_proj/noise_floor.py — the bf16 round-trip noise floor gate (architect §4.6).

REPLACES the moot on-box exact-parity gate (issue-body supersession): the raw
weights ARE the ground truth (exact modulo bf16), so there is no lossy instrument
to match. Instead we measure how much SIGNAL a single bf16 round-trip destroys and
require the reported residuals at operating horizons (h>=5) to sit ABOVE that floor.

Per-group bf16 quantization noise floor
---------------------------------------
CRITICAL bf16<->fp32 cast-point subtlety (the plan's named failure surface): the
EXP-43 trace is STORED in bf16 and r2_stream.load() casts it to fp32 on load. So by
the time the floor is computed the tensor is ALREADY a bf16-representable value, and
a naive self round-trip `theta.to(bf16).to(fp32)` is a NO-OP -> floor == 0 for every
group (SNR = nan, every (block,h) spuriously flagged bf16-unreliable, no curves).
That degenerate floor is WRONG. Because we only ever hold the bf16 values, the floor
must instead MODEL the quantization step bf16 storage injects, not round-trip an
already-quantized value.

bf16 = 1 sign, 8 exponent, 7 stored mantissa bits. The unit-in-the-last-place at a
value x is ULP(x) = 2^(floor(log2|x|)) * 2^-7, and rounding x to bf16 carries an
error bounded by half a ULP. The per-ELEMENT quantization noise of a stored bf16
value is therefore ~ q(x) = 2^(floor(log2|x|)-8) (= half-ULP). The metric track
DIFFERENCES two bf16-stored snapshots (e = theta_hat - theta_now, both bf16-sourced),
so the residual inherits ~sqrt(2) * q per element; we fold that sqrt(2) in so the
reported floor is the honest noise unit for a DIFFERENCED residual:

    floor(g) = sqrt(2) * || half_ULP(theta_g) ||_2          (fp32 L2 over the group)

This is the magnitude-aware bf16 quantization floor for that group — any residual
smaller than this is indistinguishable from bf16 rounding of the differenced pair.
SNR = ||e|| / floor(g) then says whether the residual is real.

Manifest fp32-Frobenius cross-check
-----------------------------------
For each sampled matrix, the fp32 Frobenius norm recomputed from the loaded tensor
must match the manifest `fro_norm` within 1e-2 relative (reuse the exact check that
verify_full_weight_dump.py performs — do NOT reinvent). This confirms the loaded
tensor IS the weight the manifest describes.

Gate outcome (encoded in ## Success criteria):
  * floor measured per sampled group;
  * manifest fro-norm cross-check <= 1e-2 relative;
  * at h>=5 the sampled residuals clear the floor (SNR > SNR_FLOOR_THRESH);
  * any (group,h) at/below floor is FLAGGED `bf16-unreliable`, never a precise ratio.
If a REQUIRED horizon (h>=5) is noise-dominated for a CORE block -> emit
BF16_FLOOR_BLOCKS and STOP (fp32 re-collection is a NEW GPU run, out of scope).
"""
from __future__ import annotations

import numpy as np
import torch

MANIFEST_FRONORM_TOL = 1e-2   # relative; matches verify_full_weight_dump.py default 0.01

# bf16 has 7 STORED mantissa bits; half-ULP relative quantization = 2^-(7+1) = 2^-8.
_BF16_MANTISSA_BITS = 7
_DIFF_INFLATION = float(np.sqrt(2.0))   # differencing two bf16-sourced snapshots


def bf16_quant_floor(theta_fp32: torch.Tensor) -> float:
    """Magnitude-aware bf16 quantization noise floor (fp32 L2) for one tensor/group.

    Per-element half-ULP = 2^(floor(log2|x|)) * 2^-(mantissa_bits+1). Zeros (and
    denormal-tiny values) contribute no resolvable quantization noise. The sqrt(2)
    inflation accounts for the residual being a DIFFERENCE of two bf16-sourced
    snapshots (see module docstring). Returns 0.0 only for an all-zero group.
    """
    t = theta_fp32.to(torch.float32).reshape(-1)
    absx = t.abs()
    nz = absx > 0
    if not bool(nz.any()):
        return 0.0
    exp = torch.floor(torch.log2(absx[nz]))                    # binade exponent per element
    half_ulp = torch.pow(2.0, exp) * (2.0 ** -(_BF16_MANTISSA_BITS + 1))
    return _DIFF_INFLATION * float(torch.linalg.norm(half_ulp).item())


# Backward-compatible alias: the engine/tests call bf16_roundtrip_floor; it now
# routes to the magnitude-aware quantization floor (the self-round-trip was
# degenerate for a bf16-stored trace — see module docstring).
def bf16_roundtrip_floor(theta_fp32: torch.Tensor) -> float:
    """bf16 quantization noise floor (fp32 L2). See bf16_quant_floor."""
    return bf16_quant_floor(theta_fp32)


def group_floor(vectors: list[torch.Tensor]) -> float:
    """bf16 quantization floor for a GROUP = concatenation of member fp32 vectors."""
    if not vectors:
        return 0.0
    cat = torch.cat([v.to(torch.float32).reshape(-1) for v in vectors])
    return bf16_quant_floor(cat)


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
