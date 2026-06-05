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

"""Spectral correction of masked gradients.

For a single targeted 2D gradient matrix ``G_mask`` with a running
anchor-gradient EMA ``M_anchor``::

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

FSDP note: this module operates purely on **logical 2D matrices**. It knows
nothing about FSDP, ``DTensor``, ``FlatParameter`` or
sharding. The engine-side caller (``FSDPEngine._maybe_comm_eff_grad_correction``)
owns the discovery of what container ``p.grad`` actually is and how to present a
full 2D matrix to ``correct_matrix``. Keeping that out of this file makes the
formula unit-testable on CPU with no distributed runtime. ``correct_matrix``
asserts its input is 2D so a bad
unshard upstream fails loudly here rather than silently mangling a gradient.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)

__all__ = [
    "tikhonov_weights",
    "two_sided_projection",
    "spectral_correct",
    "compute_basis",
    "SpectralFilter",
    "apply_spectral_correction_to_params",
]


def _canon(name: str) -> str:
    """Canonicalize a parameter name by stripping the FSDP per-layer-wrap infix.

    The anchor EMA is fed from the anchor clone's ``named_parameters()`` and read
    back via the live FSDP module's summoned ``named_parameters()``. The clone
    can have plain names while the live module's names carry the
    ``._fsdp_wrapped_module.`` infix. Without canonicalization the feed-side key
    (``model.layers.0.self_attn.q_proj.weight``) and the read-side key
    (``model.layers.0._fsdp_wrapped_module.self_attn.q_proj.weight``) never
    match, so ``M_anchor`` reads as zero at injection/correction time and the
    correction silently no-ops.

    Canonicalizing at every ``self._anchor`` / ``self._basis`` key boundary
    makes the two keys IDENTICAL. Safe in BOTH build paths: a successful
    deepcopy yields infixed names that canon to the same as the live-canon
    names; the fallback's non-infixed names are already canonical (no-op).
    A leading ``_fsdp_wrapped_module.`` (root-wrap, no dot prefix) is also
    stripped.
    """
    name = name.replace("._fsdp_wrapped_module", "")
    if name.startswith("_fsdp_wrapped_module."):
        name = name[len("_fsdp_wrapped_module."):]
    return name


def compute_basis(m_anchor: torch.Tensor, *, svd_mode: str = "full", rank: int = 8) -> tuple:
    """Compute the (U, S, V) basis of the anchor used by the two-sided projection.

    ``svd_mode="full"`` uses ``torch.linalg.svd(full_matrices=False)`` — the
    exact thin SVD; ``U`` is ``(m, k)``, ``S`` is ``(k,)``,
    ``V`` is ``(n, k)`` with ``k = min(m, n)``.

    ``svd_mode="lowrank"`` uses ``torch.svd_lowrank(m_anchor, q=rank)``, returning
    a rank-``min(rank, k)`` basis: ``U (m, r)``, ``S (r,)``, ``V (n, r)``. The
    two-sided projection contracts ``U^T G V`` so any ``r <= k`` is shape-valid
    and yields a rank-``r`` reconstruction whose error is non-increasing in
    ``rank`` (covered by ``test_spectral_filter.py``).

    The SVD is computed at the anchor's dtype (the caller upcasts to >= float32).
    Returns ``(u, s, v)`` all on ``m_anchor.device``.
    """
    assert m_anchor.dim() == 2, f"compute_basis expects a 2D matrix, got {tuple(m_anchor.shape)}"
    if svd_mode == "lowrank":
        m, n = m_anchor.shape
        k = min(m, n)
        q = max(1, min(int(rank), k))
        # torch.svd_lowrank returns U (m,q), S (q,), V (n,q) already in the
        # V (not V^T) convention the two-sided projection expects. It is a
        # RANDOMIZED algorithm; niter power-iterations sharpen the approximation
        # of the top-q subspace (default niter=2 can be noisy for ill-separated
        # spectra). When q == k we are asking for the full rank, where the random
        # projection adds nothing useful and can perturb tiny singular
        # directions — fall back to the exact thin SVD for the q==k case so the
        # full-rank reconstruction is exact (and reconstruction error is
        # monotone in rank in the limit).
        if q >= k:
            u, s, vh = torch.linalg.svd(m_anchor, full_matrices=False)
            return u, s, vh.transpose(-1, -2)
        u, s, v = torch.svd_lowrank(m_anchor, q=q, niter=4)
        return u, s, v
    # full thin SVD
    u, s, vh = torch.linalg.svd(m_anchor, full_matrices=False)
    v = vh.transpose(-1, -2)
    return u, s, v


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
    svd_mode: str = "full",
    rank: int = 8,
    basis: Optional[tuple] = None,
) -> torch.Tensor:
    """Apply the full spectral filter to one 2D masked-gradient matrix.

    Computes (or reuses) the SVD basis of the (already EMA-updated) anchor
    ``m_anchor``, forms the Tikhonov weights, the two-sided projection of
    ``g_mask``, and the alpha blend. Returns ``G_proj`` with the SAME
    shape/dtype/device as ``g_mask``.

    ``svd_mode`` selects ``full`` thin SVD vs ``lowrank``
    (``torch.svd_lowrank(q=rank)``). When ``basis`` (a precomputed ``(u, s, v)``)
    is supplied it is reused verbatim and ``m_anchor`` is consulted only for its
    shape/dtype — this is the ``basis_cache=cache`` path (compute U/S/V once at
    refresh, reuse across fast mini-batches). When ``basis is None`` the SVD is
    computed here (``basis_cache=recompute``).

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

    if basis is None:
        anc = m_anchor.to(compute_dtype)
        u, s, v = compute_basis(anc, svd_mode=svd_mode, rank=rank)
    else:
        u, s, v = basis
        u = u.to(compute_dtype)
        s = s.to(compute_dtype)
        v = v.to(compute_dtype)
    d = tikhonov_weights(s, tau)

    g_filt = two_sided_projection(gm, u, d, v)
    g_proj = alpha * gm + (1.0 - alpha) * g_filt
    return g_proj.to(orig_dtype)


class SpectralFilter:
    """Stateful per-target spectral filter (anchor-EMA cache + correction).

    Holds the running anchor-gradient EMA ``M_anchor`` for every targeted
    matrix, keyed by parameter name, and applies the correction on demand.
    Stateless math lives in the module-level functions above; this class owns
    only the EMA buffers and the knobs.

    The anchor cache can be **seeded** (``seed_anchor_cache=true``): on first
    sight of a target it is populated with a fixed deterministic PSD basis so
    the filter runs before a live anchor refresh exists. The basis is
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
        ema_device: str = "gpu",
        svd_mode: str = "full",
        basis_cache: str = "cache",
        rank: int = 8,
        correction_mode: str = "reweight",
        inject_gamma: float = 1.0,
        blend_eta: float = 0.5,
    ):
        self.alpha = float(alpha)
        self.tau = float(tau)
        self.beta_anc = float(beta_anc)
        self.seed_anchor_cache = bool(seed_anchor_cache)
        self.anchor_seed = int(anchor_seed)
        # Storage layer defaults: gpu/full/cache. Validation happens in
        # CommEffConfig.__post_init__ so by the time the
        # filter is built the values are known-good — assert defensively anyway.
        assert ema_device in ("gpu", "cpu"), ema_device
        assert svd_mode in ("full", "lowrank"), svd_mode
        assert basis_cache in ("cache", "recompute"), basis_cache
        self.ema_device = str(ema_device)
        self.svd_mode = str(svd_mode)
        self.basis_cache = str(basis_cache)
        self.rank = int(rank)
        # Correction mode. "reweight" = two-sided Tikhonov reweighting;
        # "inject" = additive injection of the scale-matched anchor complement;
        # "blend" = convex blend toward the scale-matched anchor.
        # Validated in CommEffConfig.__post_init__; assert defensively here too.
        assert correction_mode in ("reweight", "inject", "blend"), correction_mode
        self.correction_mode = str(correction_mode)
        self.inject_gamma = float(inject_gamma)
        self.blend_eta = float(blend_eta)
        # name -> M_anchor (float32). Lives on the gradient's device when
        # ema_device=gpu; on (pinned) CPU when ema_device=cpu (moved to the
        # gradient's device only inside update_anchor / correct_matrix).
        self._anchor: dict[str, torch.Tensor] = {}
        # name -> cached SVD basis (u, s, v) on the gradient's device, computed
        # once per refresh under basis_cache=cache and reused by every fast
        # mini-batch's correct_matrix until the next refresh. Empty under
        # basis_cache=recompute (the SVD is recomputed per correct_matrix).
        self._basis: dict[str, tuple] = {}

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
        # A per-target deterministic seed from a STABLE hash of the name.
        #
        # Python's builtin hash() is salted per-process via
        # PYTHONHASHSEED, so two FSDP ranks (separate processes) would seed
        # DIFFERENT anchors for the same parameter name -> each rank builds a
        # different M_anchor -> a different G_proj -> cross-rank replica
        # divergence / corrupted correction once the sharded grads recombine.
        # sha256 is stable across processes and Python versions, so every rank
        # derives the IDENTICAL anchor for a given (name, anchor_seed).
        name_hash = int.from_bytes(hashlib.sha256(name.encode("utf-8")).digest()[:8], "big")
        h = (name_hash ^ (self.anchor_seed * 0x9E3779B1)) & 0x7FFFFFFF
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

    def _ema_storage_device(self, grad_device):
        """Device the EMA tensor is STORED on between refreshes.

        ``ema_device=cpu`` keeps ``M_anchor`` on CPU (pinned when the grad lives
        on CUDA so the per-refresh H2D/D2H is fast); ``ema_device=gpu`` keeps it
        on the gradient's device (HBM, faithful).
        """
        return torch.device("cpu") if self.ema_device == "cpu" else grad_device

    def ensure_anchor(self, name: str, grad: torch.Tensor) -> torch.Tensor:
        """Return ``M_anchor`` for ``name``, seeding it on first sight if configured.

        The returned tensor lives on the EMA storage device (CPU when
        ``ema_device=cpu``, else the gradient's device). Seeding builds the
        deterministic basis on the grad's device, then moves it to storage.
        """
        name = _canon(name)  # match feed-side and read-side keys
        anc = self._anchor.get(name)
        if anc is None:
            store_dev = self._ema_storage_device(grad.device)
            if self.seed_anchor_cache:
                anc = self._seeded_anchor(name, grad.shape, grad.device).to(store_dev)
            else:
                anc = torch.zeros(grad.shape, dtype=torch.float32, device=store_dev)
            if store_dev.type == "cpu" and grad.device.type == "cuda":
                anc = anc.pin_memory()
            self._anchor[name] = anc
        return anc

    def anchor_on(self, name: str, device) -> torch.Tensor:
        """Return ``M_anchor`` for ``name`` moved to ``device`` (no-op if already there).

        Used at refresh/correction time to bring a CPU-offloaded EMA onto the
        compute device. The stored copy is left on its storage device.
        """
        name = _canon(name)  # match feed-side and read-side keys
        anc = self._anchor[name]
        return anc.to(device) if anc.device != torch.device(device) else anc

    def update_anchor(self, name: str, g_anchor: torch.Tensor) -> torch.Tensor:
        """EMA-update ``M_anchor <- beta * M_anchor + (1 - beta) * G_anchor`` (RAW).

        This is the live-anchor entry point: ``g_anchor`` is the
        RAW per-target gradient read BEFORE any ``correct_matrix`` call, so the
        anchor gradient never passes through the spectral projection. The EMA is
        computed on the gradient's device (bringing a CPU-offloaded ``M_anchor``
        up first), then the result is stored back on the EMA storage device.

        Under ``basis_cache=cache`` the cached ``(u, s, v)`` basis for ``name``
        is refreshed here (once per anchor refresh) so every subsequent fast
        mini-batch reuses it; under ``recompute`` no basis is cached.
        """
        name = _canon(name)  # store EMA + basis under the canonical key
        self.ensure_anchor(name, g_anchor)
        compute_dev = g_anchor.device
        anc = self.anchor_on(name, compute_dev).to(torch.float32)
        ga = g_anchor.to(torch.float32)
        new = self.beta_anc * anc + (1.0 - self.beta_anc) * ga
        # Store back on the EMA storage device (CPU offload re-pins).
        store_dev = self._ema_storage_device(compute_dev)
        stored = new.to(store_dev)
        if store_dev.type == "cpu" and compute_dev.type == "cuda":
            stored = stored.pin_memory()
        self._anchor[name] = stored
        # Cache the basis ON THE COMPUTE DEVICE for reuse by fast mini-batches.
        # Inject and blend modes need no SVD basis (they combine the
        # scale-matched anchor with G_mask directly), so skip the cache —
        # computing the full SVD of every targeted matrix per refresh would
        # stall the run.
        if self.basis_cache == "cache" and self.correction_mode not in ("inject", "blend"):
            self._basis[name] = compute_basis(new, svd_mode=self.svd_mode, rank=self.rank)
        return new

    def refresh_basis(self, name: str, device=None) -> tuple:
        """Force-(re)compute and cache the ``(u, s, v)`` basis for ``name``.

        Used when the cache must be primed without an EMA update (e.g. the
        seeded-cache path). The basis is
        cached on ``device`` (defaults to the EMA's current device).
        """
        name = _canon(name)  # match feed-side and read-side keys
        anc = self._anchor[name]
        dev = device if device is not None else anc.device
        basis = compute_basis(anc.to(dev).to(torch.float32), svd_mode=self.svd_mode, rank=self.rank)
        self._basis[name] = basis
        return basis

    # ------------------------------------------------------------------ #
    # correction
    # ------------------------------------------------------------------ #
    def correct_matrix(self, name: str, g_mask: torch.Tensor) -> torch.Tensor:
        """Apply the spectral filter to one logical 2D gradient matrix.

        Returns ``G_proj`` with the same shape/dtype/device as ``g_mask``. The
        anchor for ``name`` is seeded on first sight when configured.

        ``basis_cache=cache`` reuses the cached ``(u, s, v)`` from the most
        recent refresh (computing it once on first sight if absent — e.g. the
        seeded-cache case); ``basis_cache=recompute`` recomputes the SVD here.
        A CPU-offloaded EMA is
        brought onto the gradient's device for the (recompute) SVD.
        """
        name = _canon(name)  # match feed-side and read-side keys
        self.ensure_anchor(name, g_mask)
        basis = None
        if self.basis_cache == "cache":
            basis = self._basis.get(name)
            if basis is None:
                # Prime from the current EMA (seeded or freshly-zeroed) on the
                # gradient's device so the cache exists for subsequent batches.
                basis = self.refresh_basis(name, device=g_mask.device)
            else:
                # Cached basis may live on a different device after offload moves;
                # spectral_correct re-casts/moves dtype, but ensure device match.
                basis = tuple(t.to(g_mask.device) for t in basis)
        anc = self.anchor_on(name, g_mask.device)
        return spectral_correct(
            g_mask,
            anc,
            alpha=self.alpha,
            tau=self.tau,
            svd_mode=self.svd_mode,
            rank=self.rank,
            basis=basis,
        )

    def inject_matrix(self, name: str, g_mask: torch.Tensor) -> torch.Tensor:
        """Additive injection: G_corr = G_mask + gamma*scale*(M_anchor - P_Gmask(M_anchor)).

        Supplies the component of the stale true-gradient EMA M_anchor that G_mask
        does NOT already span (the part masking rotated away), scale-matched to
        ||G_mask|| (rescale inflates ||G_mask|| ~9x; Adam+grad-clip make the
        *direction* the load-bearing quantity). Under orthogonality (cos≈0) the
        projection ~0 and this is scale-matched direct injection of M_anchor.
        Returns G_corr with g_mask's shape/dtype/device.
        """
        name = _canon(name)  # read M_anchor under the same key the feed wrote
        self.ensure_anchor(name, g_mask)
        anc = self.anchor_on(name, g_mask.device).to(torch.float32)
        gm = g_mask.to(torch.float32)
        eps = 1e-12
        gm_norm = torch.linalg.norm(gm)
        anc_norm = torch.linalg.norm(anc)
        if anc_norm <= eps or gm_norm <= eps:
            return g_mask  # anchor not warmed / zero grad → no-op
        coeff = (gm * anc).sum() / (gm_norm * gm_norm + eps)   # <G_mask,M_anchor>/||G_mask||^2
        complement = anc - coeff * gm
        scale = gm_norm / (anc_norm + eps)
        g_corr = gm + self.inject_gamma * scale * complement
        # Diagnostic: cosine(G_mask, M_anchor) — measures orthogonality on the LIVE anchor.
        cos = (coeff * gm_norm / (anc_norm + eps)).item()
        print(f"[comm_eff][EXP-18][inject] {name} cos(G_mask,M_anchor)={cos:.4f} "
              f"gamma={self.inject_gamma} scale={scale.item():.4f} "
              f"||inj||/||G_mask||={(torch.linalg.norm(self.inject_gamma*scale*complement)/(gm_norm+eps)).item():.4f}",
              flush=True)
        return g_corr.to(g_mask.dtype)

    def blend_matrix(self, name: str, g_mask: torch.Tensor) -> torch.Tensor:
        """Convex blend: G_corr = (1-eta)*G_mask + eta*scale*M_anchor.

        REPLACES (downweights) the biased G_mask with the scale-matched stale
        true-gradient EMA M_anchor, scale=||G_mask||/||M_anchor||. Unlike inject
        (which ADDS an orthogonal force and inflates magnitude to sqrt(2)*||G_mask||
        at eta=1), the convex blend keeps a stable magnitude:
        for orthogonal terms ||G_corr|| = ||G_mask||*sqrt((1-eta)^2 + eta^2) <= ||G_mask||.
        eta->0 returns G_mask exactly; eta->1 returns the scale-matched M_anchor.
        Returns G_corr with g_mask's shape/dtype/device.
        """
        name = _canon(name)  # read M_anchor under the same key the feed wrote
        self.ensure_anchor(name, g_mask)
        anc = self.anchor_on(name, g_mask.device).to(torch.float32)
        gm = g_mask.to(torch.float32)
        eps = 1e-12
        gm_norm = torch.linalg.norm(gm)
        anc_norm = torch.linalg.norm(anc)
        if anc_norm <= eps or gm_norm <= eps:
            return g_mask  # anchor not warmed / zero grad → no-op (returns G_mask)
        eta = self.blend_eta
        scale = gm_norm / (anc_norm + eps)
        g_corr = (1.0 - eta) * gm + eta * scale * anc
        # Diagnostic: cosine(G_mask, M_anchor) on the LIVE anchor + magnitude ratio.
        cos = ((gm * anc).sum() / (gm_norm * anc_norm + eps)).item()
        print(f"[comm_eff][EXP-18][blend] {name} eta={eta} cos(G_mask,M_anchor)={cos:.4f} "
              f"||G_corr||/||G_mask||={(torch.linalg.norm(g_corr) / (gm_norm + eps)).item():.4f}",
              flush=True)
        return g_corr.to(g_mask.dtype)

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


def apply_spectral_correction_to_params(
    named_params,
    *,
    spectral: "SpectralFilter",
    target_substrs,
    max_targets: int,
    state,
    discovery_meta: dict,
    full_grad_of,
    writeback,
) -> int:
    """FSDP-agnostic core of the spectral grad-correction hook (CPU-testable).

    This is the load-bearing iteration/discovery/correction loop, pulled out of
    ``FSDPEngine._maybe_comm_eff_grad_correction`` so it runs on CPU with no
    ``torch.distributed`` / FSDP runtime (see
    ``tests/workers/comm_eff/test_grad_correction_hook.py``). The FSDP-specific
    bits are injected as callables:

    * ``full_grad_of(grad) -> (full_2d_tensor, meta)`` resolves the **full
      logical 2D matrix** for a raw ``.grad`` (identity for a plain CPU/FSDP1-
      summoned tensor; ``grad.full_tensor()`` for an FSDP2 ``DTensor``) and
      returns a small ``meta`` dict (container type, is_dtensor, placements,
      mesh) describing the container — recorded once in the discovery log.
    * ``writeback(grad, g_proj)`` copies the corrected full matrix back into the
      (possibly sharded) ``.grad`` in place.

    Discovery/correction contract:
    * the discovery log is recorded **once**, on the first target with a
      non-``None`` grad, **regardless of gradient magnitude** (a near-zero grad
      still proves the hook ran — only ``grad is None`` is skipped); and
    * ``state.spectral_corrections`` increments per corrected 2D matrix.

    Returns the number of matrices corrected.
    """
    instrumented = bool(state.fsdp_grad_repr)  # log discovery only once
    corrected = 0

    for name, p in named_params:
        grad = getattr(p, "grad", None)
        if grad is None:
            continue
        if not any(s in name for s in target_substrs):
            continue
        if max_targets >= 0 and corrected >= max_targets:
            break

        full, container_meta = full_grad_of(grad)
        logical_shape = tuple(full.shape)
        if full.dim() != 2:
            # Skip non-2D targets (norms/biases excluded by substr anyway).
            continue

        if not instrumented:
            repr_log = {"target_name": name, "logical_2d_shape": str(logical_shape)}
            repr_log.update(container_meta)
            repr_log.update(discovery_meta)
            state.fsdp_grad_repr = repr_log
            logger.warning("comm_eff FSDP grad-repr discovery: %s", repr_log)
            print(f"[comm_eff][EXP-7][FSDP-DISCOVERY] {repr_log}", flush=True)
            instrumented = True

        _mode = getattr(spectral, "correction_mode", "reweight")
        if _mode == "inject":
            g_proj = spectral.inject_matrix(name, full)
        elif _mode == "blend":
            g_proj = spectral.blend_matrix(name, full)
        else:
            g_proj = spectral.correct_matrix(name, full)
        rel = spectral.relative_change(full, g_proj)
        state.spectral_rel_change[name] = rel
        print(
            f"[comm_eff][EXP-7][spectral] {name} alpha={spectral.alpha} "
            f"rel_change=||G_proj-G_mask||/||G_mask||={rel:.6f} "
            f"shape={logical_shape} grad_type={container_meta.get('grad_container_type')}",
            flush=True,
        )
        with torch.no_grad():
            writeback(grad, g_proj)

        corrected += 1
        state.spectral_corrections += 1

    if corrected:
        logger.info("comm_eff: spectral correction applied to %d target matrices", corrected)
    return corrected
