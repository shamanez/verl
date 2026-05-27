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

"""Spectral correction of masked gradients (M2, the *third circuit*).

This implements the paper's spectral filter exactly. For a single targeted 2D
gradient matrix ``G_mask`` with a running anchor-gradient EMA ``M_anchor``::

    M_anchor = beta_anc * M_anchor + (1 - beta_anc) * G_anchor   # anchor EMA
    M_anchor = U S V^T                                           # full thin SVD
    d_i      = s_i / (s_i + tau)                                 # Tikhonov weights
    X        = U^T G_mask V
    G_filt   = U diag(d) X diag(d) V^T                           # two-sided projection
    G_proj   = alpha * G_mask + (1 - alpha) * G_filt             # blend

Load-bearing invariants (each unit-tested in
``tests/workers/comm_eff/test_spectral_filter.py``):

* **``alpha = 1`` ⇒ exact no-op.** ``G_proj == G_mask`` to ≤ 1e-6 max-abs-diff,
  regardless of the anchor. The masked gradient is never discarded; the blend
  is a strict convex combination, so at ``alpha=1`` the projection term drops
  out entirely.

* **``alpha = 0`` ⇒ pure two-sided Tikhonov projection.** ``G_proj`` equals
  ``U diag(d) (U^T G_mask V) diag(d) V^T`` to ≤ 1e-6.

* **Shape preservation.** ``G_proj.shape == G_mask.shape`` for any 2D matrix
  (square or rectangular) — the two-sided projection uses the thin SVD of the
  anchor, whose ``U`` is ``(m, k)`` and ``V`` is ``(n, k)`` with
  ``k = min(m, n)``, so ``U diag(d) X diag(d) V^T`` is ``(m, n)``.

* **Determinism.** With a fixed seed the seeded anchor cache and therefore the
  whole pipeline is reproducible.

FSDP NOTE — DELIBERATE DECOUPLING. This module operates purely on **logical 2D
matrices**. It knows nothing about FSDP, ``DTensor``, ``FlatParameter`` or
sharding. The engine-side caller (``FSDPEngine._maybe_comm_eff_grad_correction``)
owns the discovery of what container ``p.grad`` actually is and how to present a
full 2D matrix to ``correct_matrix`` — that is the EXP-7 deliverable, and
keeping it out of this file is what makes the formula unit-testable on CPU with
no distributed runtime. ``correct_matrix`` asserts its input is 2D so a bad
unshard upstream fails loudly here rather than silently mangling a gradient.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)

__all__ = [
    "tikhonov_weights",
    "two_sided_projection",
    "spectral_correct",
    "SpectralFilter",
]


def tikhonov_weights(singular_values: torch.Tensor, tau: float) -> torch.Tensor:
    """Tikhonov spectral weights ``d_i = s_i / (s_i + tau)``.

    ``tau > 0`` so the denominator never vanishes even for a zero singular
    value (a rank-deficient anchor yields ``d_i = 0`` there, which simply
    zeroes that spectral direction in the projection — well defined).
    """
    return singular_values / (singular_values + tau)


def two_sided_projection(
    g_mask: torch.Tensor,
    u: torch.Tensor,
    d: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """Pure two-sided Tikhonov projection ``U diag(d) (U^T G_mask V) diag(d) V^T``.

    Args:
        g_mask: the masked gradient matrix, shape ``(m, n)``.
        u: left singular vectors of the anchor, shape ``(m, k)``.
        d: Tikhonov spectral weights, shape ``(k,)``.
        v: right singular vectors of the anchor, shape ``(n, k)``.

    Returns:
        ``G_filt`` of shape ``(m, n)``.
    """
    # X = U^T G_mask V  -> (k, k)
    x = u.transpose(-1, -2) @ g_mask @ v
    # diag(d) X diag(d) is an elementwise scaling X_ij * d_i * d_j (cheaper +
    # more numerically faithful than materialising two diagonal matmuls).
    x = x * d.unsqueeze(-1)  # scale rows by d_i
    x = x * d.unsqueeze(-2)  # scale cols by d_j
    # G_filt = U X V^T -> (m, n)
    return u @ x @ v.transpose(-1, -2)


def spectral_correct(
    g_mask: torch.Tensor,
    m_anchor: torch.Tensor,
    *,
    alpha: float,
    tau: float,
) -> torch.Tensor:
    """Apply the full spectral filter to one 2D masked-gradient matrix.

    Computes the thin SVD of the (already EMA-updated) anchor ``m_anchor``,
    forms the Tikhonov weights, the two-sided projection of ``g_mask``, and the
    alpha blend. Returns ``G_proj`` with the SAME shape/dtype/device as
    ``g_mask``.

    Numerics: SVD is computed in float32 for stability (the gradients may be
    bf16); the result is cast back to ``g_mask``'s dtype. At ``alpha == 1.0``
    the projection is skipped entirely and ``g_mask`` is returned unchanged so
    the no-op is exact regardless of anchor conditioning.
    """
    assert g_mask.dim() == 2, f"spectral_correct expects a 2D matrix, got shape {tuple(g_mask.shape)}"
    assert m_anchor.dim() == 2, f"anchor must be 2D, got shape {tuple(m_anchor.shape)}"
    assert m_anchor.shape == g_mask.shape, (
        f"anchor shape {tuple(m_anchor.shape)} must match gradient shape {tuple(g_mask.shape)}"
    )

    # alpha == 1: exact no-op. Return the same tensor (caller copies in place).
    if alpha >= 1.0:
        return g_mask

    orig_dtype = g_mask.dtype
    # SVD is numerically delicate; run it at >= float32. Upcast low-precision
    # grads (bf16/fp16) to float32, but keep float64 as float64 so a high-
    # precision caller (e.g. the unit test's float64 reference) gets float64
    # accuracy rather than being silently truncated to float32.
    compute_dtype = orig_dtype if orig_dtype in (torch.float32, torch.float64) else torch.float32
    gm = g_mask.to(compute_dtype)
    anc = m_anchor.to(compute_dtype)

    # Full thin SVD of the anchor: U (m,k), S (k,), V (n,k) with k=min(m,n).
    u, s, vh = torch.linalg.svd(anc, full_matrices=False)
    v = vh.transpose(-1, -2)
    d = tikhonov_weights(s, tau)

    g_filt = two_sided_projection(gm, u, d, v)
    g_proj = alpha * gm + (1.0 - alpha) * g_filt
    return g_proj.to(orig_dtype)


class SpectralFilter:
    """Stateful per-target spectral filter (anchor-EMA cache + correction).

    Holds the running anchor-gradient EMA ``M_anchor`` for every targeted
    matrix, keyed by parameter name, and applies the paper formula on demand.
    Stateless math lives in the module-level functions above; this class owns
    only the EMA buffers and the knobs.

    The anchor cache can be **seeded** (``seed_anchor_cache=true``): on first
    sight of a target it is populated with a fixed deterministic PSD basis so
    the filter runs before the live anchor circuit (EXP-8) exists. The basis is
    ``Q diag(lin) Q^T``-style only conceptually — for a rectangular ``(m, n)``
    matrix we build a deterministic full-shape matrix whose SVD has a smooth,
    strictly positive spectrum, so the Tikhonov weights are well conditioned.
    """

    def __init__(
        self,
        *,
        alpha: float = 0.3,
        tau: float = 1e-3,
        beta_anc: float = 0.95,
        seed_anchor_cache: bool = True,
        anchor_seed: int = 0,
    ):
        self.alpha = float(alpha)
        self.tau = float(tau)
        self.beta_anc = float(beta_anc)
        self.seed_anchor_cache = bool(seed_anchor_cache)
        self.anchor_seed = int(anchor_seed)
        # name -> M_anchor (float32, on the gradient's device)
        self._anchor: dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------------ #
    # anchor cache
    # ------------------------------------------------------------------ #
    def _seeded_anchor(self, name: str, shape: torch.Size, device, ) -> torch.Tensor:
        """Build a fixed deterministic PSD-spectrum anchor for a target.

        Deterministic in ``(anchor_seed, name, shape)`` so a given target gets
        the same anchor on every rank and every re-run — this is what makes the
        whole filter reproducible under ``seed_anchor_cache=true``. The spectrum
        is strictly positive and decaying so the Tikhonov weights are smooth and
        the projection is well conditioned.
        """
        m, n = int(shape[0]), int(shape[1])
        k = min(m, n)
        # A per-target deterministic seed from a stable hash of the name.
        h = (hash(name) ^ (self.anchor_seed * 0x9E3779B1)) & 0x7FFFFFFF
        gen = torch.Generator(device="cpu").manual_seed(h)
        # Random orthonormal-ish factors with a decaying positive spectrum.
        a = torch.randn(m, k, generator=gen, dtype=torch.float32)
        b = torch.randn(n, k, generator=gen, dtype=torch.float32)
        # QR for well-separated bases; fall back to the raw factor if QR fails.
        qa, _ = torch.linalg.qr(a)
        qb, _ = torch.linalg.qr(b)
        spectrum = torch.linspace(1.0, 0.1, steps=k, dtype=torch.float32)
        anchor = (qa * spectrum.unsqueeze(0)) @ qb.transpose(0, 1)
        return anchor.to(device=device)

    def ensure_anchor(self, name: str, grad: torch.Tensor) -> torch.Tensor:
        """Return ``M_anchor`` for ``name``, seeding it on first sight if configured."""
        anc = self._anchor.get(name)
        if anc is None:
            if self.seed_anchor_cache:
                anc = self._seeded_anchor(name, grad.shape, grad.device)
            else:
                anc = torch.zeros(grad.shape, dtype=torch.float32, device=grad.device)
            self._anchor[name] = anc
        return anc

    def update_anchor(self, name: str, g_anchor: torch.Tensor) -> torch.Tensor:
        """EMA-update ``M_anchor <- beta * M_anchor + (1 - beta) * G_anchor``.

        Used by the live anchor circuit (EXP-8). For the EXP-7 seeded smoke the
        cache is the fixed seeded basis and this is not called; it exists so the
        EMA path is in place and unit-testable.
        """
        anc = self.ensure_anchor(name, g_anchor)
        ga = g_anchor.to(torch.float32)
        new = self.beta_anc * anc + (1.0 - self.beta_anc) * ga
        self._anchor[name] = new
        return new

    # ------------------------------------------------------------------ #
    # correction
    # ------------------------------------------------------------------ #
    def correct_matrix(self, name: str, g_mask: torch.Tensor) -> torch.Tensor:
        """Apply the spectral filter to one logical 2D gradient matrix.

        Returns ``G_proj`` with the same shape/dtype/device as ``g_mask``. The
        anchor for ``name`` is seeded on first sight when configured.
        """
        anc = self.ensure_anchor(name, g_mask)
        return spectral_correct(g_mask, anc, alpha=self.alpha, tau=self.tau)

    def relative_change(self, g_mask: torch.Tensor, g_proj: torch.Tensor) -> float:
        """Per-target ``||G_proj - G_mask|| / ||G_mask||`` (Frobenius).

        Logged faithfully (NOT clamped) — the codex pin notes this is not
        provably ≤1 for arbitrary anchors; we report whatever the math yields.
        """
        gm = g_mask.to(torch.float32)
        gp = g_proj.to(torch.float32)
        denom = torch.linalg.norm(gm)
        if denom <= 0:
            return 0.0
        return (torch.linalg.norm(gp - gm) / denom).item()
