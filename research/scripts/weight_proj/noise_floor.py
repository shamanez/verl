#!/usr/bin/env python3
"""weight_proj/noise_floor.py — the bf16 round-trip noise floor gate (architect §4.6).

REPLACES the moot on-box exact-parity gate (issue-body supersession): the raw
weights ARE the ground truth (exact modulo bf16), so there is no lossy instrument
to match. Instead we measure how much SIGNAL a single bf16 round-trip destroys and
require the reported residuals at operating horizons (h>=5) to sit ABOVE that floor.

Per-group bf16 round-trip noise floor
--------------------------------------
For a group g (matrix / block / layer) at a reference snapshot theta:
    theta_bf16   = theta_fp32.to(bf16).to(fp32)      one bf16 round-trip
    floor(g)     = || theta_fp32 - theta_bf16 ||     (fp32 L2 of the rounding error)
This is the magnitude of pure bf16 quantization noise for that group — any residual
smaller than this is indistinguishable from rounding. Because the trace is stored
bf16, differencing two snapshots inherits ~sqrt(2)*floor of quantization noise; we
report the single-round-trip floor as the conservative per-group unit and let SNR =
||e|| / floor say whether the residual is real.

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


def bf16_roundtrip_floor(theta_fp32: torch.Tensor) -> float:
    """|| theta - roundtrip_bf16(theta) ||  in fp32 L2 for one tensor/group vector."""
    t = theta_fp32.to(torch.float32)
    rt = t.to(torch.bfloat16).to(torch.float32)
    return float(torch.linalg.norm((t - rt).reshape(-1)).item())


def group_floor(vectors: list[torch.Tensor]) -> float:
    """bf16 round-trip floor for a GROUP = concatenation of member fp32 vectors."""
    if not vectors:
        return 0.0
    cat = torch.cat([v.to(torch.float32).reshape(-1) for v in vectors])
    return bf16_roundtrip_floor(cat)


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
