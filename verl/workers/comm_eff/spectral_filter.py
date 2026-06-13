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

"""Anchor-guided correction of compressed (masked) gradients.

The live correction is the **signed-EMA merger** (EXP-25/R3). For a single
targeted 2D gradient matrix ``G_noisy`` (the fast, activation-compressed
gradient) with a running anchor-gradient EMA ``M_anchor``::

    M_anchor = beta_anc * M_anchor + (1 - beta_anc) * G_anchor    # anchor EMA
    G_corr   = alpha * G_noisy + (1 - alpha) * |G_noisy| * sign(M_anchor)

The MAGNITUDE comes from the fast compressed gradient ``G_noisy``; the SIGN
comes from the β-EMA of the K-stale unmasked anchor gradient ``M_anchor``.
``alpha`` (``signed_ema_alpha``) is the swept axis: ``alpha=0`` ⇒ pure
``|G_noisy|·sign(M)``; ``alpha=1`` ⇒ ``G_noisy`` unchanged. A matrix-level
cold-M guard returns ``G_noisy`` unchanged whenever ``M`` is unwarmed/zero, so
the merger never silently zeroes a gradient (see :meth:`SpectralFilter.signed_ema_matrix`).

Two older anchor combiners remain available via ``correction_mode``:

* ``inject`` — additive injection of the scale-matched anchor complement
  (:meth:`SpectralFilter.inject_matrix`).
* ``blend`` — convex blend toward the scale-matched anchor
  (:meth:`SpectralFilter.blend_matrix`).

All three combiners consult ONLY the anchor-gradient EMA ``M_anchor`` (no SVD,
no basis cache). The initial SVD/Tikhonov/two-sided-projection "reweight"
method and its seeded-anchor cache were removed (EXP-25) — they were the dead
spectral-correction path the project no longer uses.

FSDP note: this module operates purely on **logical 2D matrices**. It knows
nothing about FSDP, ``DTensor``, ``FlatParameter`` or sharding. The engine-side
caller (``FSDPEngine._maybe_comm_eff_grad_correction``) owns the discovery of
what container ``p.grad`` actually is and how to present a full 2D matrix to the
merger. Keeping that out of this file makes the formula unit-testable on CPU
with no distributed runtime.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)

__all__ = [
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

    Canonicalizing at every ``self._anchor`` key boundary
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


class SpectralFilter:
    """Stateful per-target anchor-guided gradient corrector (anchor-EMA + merger).

    Holds the running anchor-gradient EMA ``M_anchor`` for every targeted
    matrix, keyed by parameter name, and applies the correction on demand. The
    live correction is the signed-EMA merger (``correction_mode="signed_ema"``);
    ``inject`` and ``blend`` are alternate anchor combiners. All of them consult
    only ``M_anchor`` — there is no SVD/basis state.

    The anchor EMA cold-starts at zeros (``ensure_anchor``) and is populated by
    the live anchor circuit via :meth:`update_anchor`. Until ``M_anchor`` is
    warmed, the merger / combiners return ``G_noisy`` unchanged (the cold-M
    guard), so an unwarmed anchor is a no-op, never a silent grad-zeroing.
    """

    def __init__(
        self,
        *,
        beta_anc: float = 0.95,
        ema_device: str = "gpu",
        correction_mode: str = "signed_ema",
        inject_gamma: float = 1.0,
        blend_eta: float = 0.5,
        signed_ema_alpha: float = 0.0,
        ef_decay: float = 0.0,
        ef_clip: float = 0.0,
        delayed_ef_lambda: float = 0.0,
        delta_subbasis_rank: int = 0,
        delta_subbasis_family: str = "tail",
        base_seed: int = 0,
    ):
        self.beta_anc = float(beta_anc)
        # Storage layer default: gpu. Validation happens in
        # CommEffConfig.__post_init__ so by the time the
        # filter is built the values are known-good — assert defensively anyway.
        assert ema_device in ("gpu", "cpu"), ema_device
        # Correction mode (the anchor combiner the fast-path grad uses):
        # "none" (EXP-30 Step A) = INERT — the M EMA is still maintained
        # (β_anc=0 ⇒ M_rep = latest paired G_anc_rep) but NO correction is ever
        # applied or written back; the optimizer consumes the raw G_comp;
        # "inject" = additive injection of the scale-matched anchor complement;
        # "blend" = convex blend toward the scale-matched anchor;
        # "signed_ema" (EXP-25/R3) = alpha*G_noisy + (1-alpha)*|G_noisy|*sign(M);
        # "ef_powersgd" (EXP-26 Step B) = direction-preserving error-feedback,
        # G_corr = G_comp + clipped-residual, NO sign term;
        # "delayed_ef" (EXP-30 B2) = K-delayed exact codec residual,
        # G_corr = G_comp + lambda*(M_rep - G_comp_ring(t-K)), delta refreshed at
        # anchor fires and HELD between them.
        # Validated in CommEffConfig.__post_init__; assert defensively here too.
        self.ema_device = str(ema_device)
        assert correction_mode in ("none", "inject", "blend", "signed_ema", "ef_powersgd", "delayed_ef"), (
            correction_mode
        )
        self.correction_mode = str(correction_mode)
        self.inject_gamma = float(inject_gamma)
        self.blend_eta = float(blend_eta)
        # EXP-25 (R3): the signed_ema merger weight alpha.
        self.signed_ema_alpha = float(signed_ema_alpha)
        # EXP-26 Step B: error-feedback residual knobs. decay=clip=0 ⇒ the merger
        # reduces to plain PowerSGD (G_corr == G_comp) — the limiting-case identity.
        self.ef_decay = float(ef_decay)
        self.ef_clip = float(ef_clip)
        # EXP-30 B2: the delayed_ef residual weight λ. 0.0 (default, OFF/legacy
        # posture) ⇒ delayed_ef_matrix returns G_comp EXACTLY (the limiting-case
        # identity invariant); the B2 cell sets λ=1.0 explicitly.
        self.delayed_ef_lambda = float(delayed_ef_lambda)
        # EXP-31 Cell D: additive stale-anchor rank-r sub-basis folded into the
        # delayed_ef correction term. ``delta_subbasis_rank`` r_sb > 0 enables the
        # NEW additive term ``δ_subbasis = rank_{r_sb}(S)`` where the source S is
        # the act-deflated stale weight-gradient (``family="tail"``, S = δ_B2) or
        # the raw stale anchor gradient (``family="grad"``, S = M_rep). r_sb = 0
        # (default) SKIPS the sub-basis branch entirely (the rank-0 path is the
        # EXACT B2 path, bitwise — Correctness invariant "off-path parity"). The
        # sub-basis enters ONLY the correction δ (the forward codec Q is never
        # read/written here ⇒ Step-C avoidance by construction). Validated in
        # CommEffConfig.__post_init__; assert defensively here too.
        self.delta_subbasis_rank = int(delta_subbasis_rank)
        assert self.delta_subbasis_rank >= 0, self.delta_subbasis_rank
        assert delta_subbasis_family in ("tail", "grad"), delta_subbasis_family
        self.delta_subbasis_family = str(delta_subbasis_family)
        # EXP-31 Cell D: base seed for the per-target randomized SVD generator.
        # The low-rank sub-basis is built with ``torch.svd_lowrank`` (randomized),
        # whose result depends on a random projection. δ_B2 / M_rep are already
        # DP-MEAN-identical across ranks, so to keep δ_subbasis BIT-IDENTICAL
        # across DP ranks the random projection MUST be seeded deterministically
        # (same on every rank). We mix this base_seed with a per-target salt
        # derived from the target name (see ``_subbasis_seed``) so each target
        # gets its own reproducible generator while staying cross-rank identical.
        self.base_seed = int(base_seed)
        # EXP-30 B2: per-target HELD residual δ (detached fp32, EMA-storage
        # device). Refreshed when a fire-aligned ring entry exists (the anchor
        # just refreshed M_rep AND G_comp_ring(t−K) is the exact pair), HELD on
        # the in-between ticks, shape-keyed reset. β_anc=0 keeps zero EMA memory
        # in M itself; the hold is the cadence-window transport, not a carrier.
        self._delayed_ef_delta: dict[str, torch.Tensor] = {}
        # Per-step counters (reset by the engine loop): how many targets
        # REFRESHED δ this step vs reused the held one vs fell back cold.
        self.delayed_ef_refreshed = 0
        self.delayed_ef_held = 0
        # EXP-31 Cell D: per-step counters (reset by the engine loop): how many
        # targets had the additive sub-basis APPLIED vs SKIPPED because the
        # randomized SVD was degenerate (zero/NaN source, r_sb > min-dim, etc.).
        self.delayed_ef_subbasis_applied = 0
        self.delayed_ef_subbasis_skipped = 0
        # EXP-31 Cell D: per-fire ‖δ_subbasis‖/‖δ_B2‖ ratios collected this step
        # (so the engine loop can log the median on the existing delayed_ef line).
        self._subbasis_energy_ratios: list[float] = []
        # EXP-25 (R3): per-step count of matrices whose M was cold (||M||<=eps) so
        # the merger no-op'd to G_noisy (the silent grad-zeroing guard). Reset by
        # the engine each grad-correction step before the loop.
        self.merger_coldM_fallbacks = 0
        # EXP-26: per-step count of ef_powersgd targets whose accumulated residual
        # e_t was RESET because the target's logical 2D shape changed (no stale
        # carry across a shape change). Reset by the engine each grad-correction
        # step before the loop.
        self.residual_reset_on_shape_mismatch = 0
        # name -> M_anchor (float32). Lives on the gradient's device when
        # ema_device=gpu; on (pinned) CPU when ema_device=cpu (moved to the
        # gradient's device only inside update_anchor / the combiner).
        self._anchor: dict[str, torch.Tensor] = {}
        # EXP-26 Step B: per-target accumulated error-feedback residual e_t
        # (detached fp32). Lives on the EMA storage device; shape-keyed reset.
        # Used ONLY by ef_powersgd; never read by the optimizer directly.
        self._ef_residual: dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------------ #
    # anchor cache
    # ------------------------------------------------------------------ #
    def _ema_storage_device(self, grad_device):
        """Device the EMA tensor is STORED on between refreshes.

        ``ema_device=cpu`` keeps ``M_anchor`` on CPU (pinned when the grad lives
        on CUDA so the per-refresh H2D/D2H is fast); ``ema_device=gpu`` keeps it
        on the gradient's device (HBM, faithful).
        """
        return torch.device("cpu") if self.ema_device == "cpu" else grad_device

    def ensure_anchor(self, name: str, grad: torch.Tensor) -> torch.Tensor:
        """Return ``M_anchor`` for ``name``, cold-starting it at zeros on first sight.

        The returned tensor lives on the EMA storage device (CPU when
        ``ema_device=cpu``, else the gradient's device). The cold start is
        ``torch.zeros`` — until the live anchor circuit warms it via
        :meth:`update_anchor`, the combiners' cold-M guard treats it as a no-op.
        """
        name = _canon(name)  # match feed-side and read-side keys
        anc = self._anchor.get(name)
        if anc is None:
            store_dev = self._ema_storage_device(grad.device)
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

        This is the live-anchor entry point: ``g_anchor`` is the RAW per-target
        gradient read BEFORE any fast-path combiner runs, so the anchor gradient
        never passes through the correction. The EMA is computed on the
        gradient's device (bringing a CPU-offloaded ``M_anchor`` up first), then
        the result is stored back on the EMA storage device.
        """
        name = _canon(name)  # store EMA under the canonical key
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
        return new

    # ------------------------------------------------------------------ #
    # correction
    # ------------------------------------------------------------------ #
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

    def signed_ema_matrix(self, name: str, g_mask: torch.Tensor) -> torch.Tensor:
        """EXP-25 (R3) signed-EMA merger: ``G_corr = α·G_noisy + (1−α)·|G_noisy|·sign(M)``.

        The SL-validated merger. The MAGNITUDE comes from the fast compressed
        gradient ``G_noisy`` (= ``g_mask``), the SIGN from the β-EMA of the
        K-stale anchor gradient ``M_anchor`` (NOT the fresh full gradient).
        ``α`` (``signed_ema_alpha``) is the swept axis; ``α=0`` ⇒ pure
        ``|G_noisy|·sign(M)`` (the SFT default), ``α=1`` ⇒ ``G_noisy`` unchanged.

        **COLD-M FALLBACK (MANDATORY — silent grad-zeroing guard).** Mirrors the
        cold-anchor guard in :meth:`blend_matrix` (``if anc_norm <= eps: return
        g_mask``). When ``M[name]`` is unwarmed/zero (the first ``delay_K`` steps
        before the first anchor refresh, and any matrix ``M`` does not cover),
        ``sign(0)=0`` and at ``α=0`` the term ``(1−α)·|G_noisy|·sign(M)=0`` ⇒
        ``G_corr = α·G_noisy = 0`` — the gradient is SILENTLY ZEROED and the run
        keeps going while quietly not learning that matrix. To prevent this we
        return ``g_mask`` UNCHANGED (behave as ``α=1`` for that matrix) and bump
        ``self.merger_coldM_fallbacks`` so the probe can prove the fallback fired
        on step 1 (M cold) and then stopped after M warms.

        Returns ``G_corr`` with ``g_mask``'s shape/dtype/device.
        """
        name = _canon(name)  # read M_anchor under the same key the feed wrote
        self.ensure_anchor(name, g_mask)
        anc = self.anchor_on(name, g_mask.device).to(torch.float32)
        gm = g_mask.to(torch.float32)
        eps = 1e-12
        anc_norm = torch.linalg.norm(anc)
        if anc_norm <= eps:
            # COLD M → return G_noisy UNCHANGED (NOT zeroed). Count the fallback.
            self.merger_coldM_fallbacks += 1
            return g_mask
        alpha = self.signed_ema_alpha
        # |G_noisy| * sign(M): magnitude from the fast compressed grad, sign from
        # the stale-anchor EMA. sign(M) is ±1 on warmed entries (anc_norm>eps
        # guarantees a non-trivial M, though individual entries can still be 0 →
        # sign 0, which correctly zeroes only those single coordinates, not the
        # whole matrix — the matrix-level cold guard above is what prevents the
        # catastrophic all-zero case).
        g_corr = alpha * gm + (1.0 - alpha) * gm.abs() * torch.sign(anc)
        return g_corr.to(g_mask.dtype)

    def ef_powersgd_matrix(self, name: str, g_mask: torch.Tensor) -> torch.Tensor:
        """EXP-26 Step B: direction-PRESERVING error-feedback PowerSGD merger.

        ``G_corr = G_comp + e_t`` where ``e_t`` is the accumulated, decayed,
        norm-clipped OFF-SUBSPACE residual — the component of the stale anchor EMA
        ``M_anchor`` that ``G_comp`` (= ``g_mask``) does NOT already span (exactly
        the low-rank-compression bias the audit measures). There is **NO sign
        term**: the correction only ADDS the dropped off-principal energy, so
        ``G_corr`` keeps ``G_comp``'s direction/sign (direction-preserving, not
        sign-replacing — the EXP-25 ``signed_ema`` failure mode is structurally
        excluded).

        Update (per targeted matrix, all detached fp32)::

            comp_t  = M_anchor - <G_comp,M_anchor>/||G_comp||² · G_comp   # off-subspace
            e_t     = ef_decay · e_{t-1} + comp_t                          # EMA residual
            e_t     = clip(e_t,  ef_clip · ||G_comp||)                     # shape-aware norm cap
            G_corr  = G_comp + e_t                                         # NO sign

        **Limiting-case identity (Correctness invariant).** With ``ef_decay=0``
        AND ``ef_clip=0`` the clip floors ``e_t`` to the zero vector, so
        ``G_corr == G_comp`` bit-for-bit ⇒ ef_powersgd reduces to plain PowerSGD.

        **Shape-aware / clipped / detached (Correctness invariant).** The residual
        is keyed by the target's logical 2D shape; on a shape change the stale
        residual is RESET to 0 (no cross-shape carry, counted in
        ``residual_reset_on_shape_mismatch``). It is norm-clipped relative to
        ``||G_comp||`` and is detached from autograd (it is built from already-
        detached ``M_anchor`` + ``g_mask``). The cold-M guard returns ``g_mask``
        unchanged (and clears any stale residual) so an unwarmed anchor is a no-op,
        never a silent grad change.

        Returns ``G_corr`` with ``g_mask``'s shape/dtype/device.
        """
        name = _canon(name)  # read M_anchor / residual under the same key the feed wrote
        self.ensure_anchor(name, g_mask)
        anc = self.anchor_on(name, g_mask.device).to(torch.float32)
        gm = g_mask.to(torch.float32)
        eps = 1e-12
        gm_norm = torch.linalg.norm(gm)
        anc_norm = torch.linalg.norm(anc)
        # COLD M (anchor unwarmed) → return G_comp UNCHANGED; drop any stale
        # residual so a later warm step starts clean.
        if anc_norm <= eps or gm_norm <= eps:
            self.merger_coldM_fallbacks += 1
            self._ef_residual.pop(name, None)
            return g_mask

        # Off-subspace component of the anchor EMA: the part of M_anchor that
        # G_comp does NOT span (== the inject "complement", but here it is the
        # residual we error-feedback). coeff = <G_comp,M_anchor>/||G_comp||².
        coeff = (gm * anc).sum() / (gm_norm * gm_norm + eps)
        comp_t = anc - coeff * gm

        # Shape-aware residual carry: reset on a logical-shape change so a stale
        # residual from a differently-shaped target never leaks in.
        prev = self._ef_residual.get(name)
        if prev is not None and tuple(prev.shape) != tuple(gm.shape):
            prev = None
            self.residual_reset_on_shape_mismatch += 1
        if prev is not None:
            prev = prev.to(comp_t.device, torch.float32)
            e_t = self.ef_decay * prev + comp_t
        else:
            e_t = comp_t

        # Norm-clip the residual relative to ||G_comp||: ||e_t|| <= ef_clip·||G_comp||.
        # ef_clip=0 ⇒ the cap is 0 ⇒ e_t is scaled to the zero vector ⇒ G_corr==G_comp.
        cap = self.ef_clip * gm_norm
        e_norm = torch.linalg.norm(e_t)
        if float(cap.item()) <= 0.0:
            e_t = torch.zeros_like(e_t)
        elif float(e_norm.item()) > float(cap.item()):
            e_t = e_t * (cap / (e_norm + eps))

        # Persist the (clipped) residual on the EMA storage device, DETACHED.
        store_dev = self._ema_storage_device(g_mask.device)
        stored = e_t.detach().to(store_dev)
        if store_dev.type == "cpu" and g_mask.device.type == "cuda":
            stored = stored.pin_memory()
        self._ef_residual[name] = stored

        g_corr = gm + e_t
        return g_corr.to(g_mask.dtype)

    # ------------------------------------------------------------------ #
    # EXP-31 Cell D: additive stale-anchor sub-basis (weight-gradient tail)
    # ------------------------------------------------------------------ #
    def _subbasis_seed(self, name: str) -> int:
        """Deterministic per-target seed for the randomized SVD generator.

        Mixes ``self.base_seed`` with a stable per-target salt derived from the
        canonical target name so (a) every target gets its OWN reproducible
        generator and (b) the seed is a pure function of (base_seed, name) — it
        contains NO rank-local / device-local state — so it is IDENTICAL on every
        DP rank. Because the source ``S`` (δ_B2 or M_rep) is already DP-MEAN
        identical across ranks, a rank-invariant seed makes ``torch.svd_lowrank``
        return bit-identical columns on every rank (the multi-rank-agreement
        invariant). The salt is a stable non-cryptographic hash of the canonical
        name (Python's ``hash`` is salted per-process, so we roll our own FNV-1a
        over the UTF-8 bytes to stay reproducible across processes/ranks).
        """
        salt = 0x811C9DC5  # FNV-1a 32-bit offset basis
        for b in _canon(name).encode("utf-8"):
            salt = ((salt ^ b) * 0x01000193) & 0xFFFFFFFF
        return (int(self.base_seed) * 1_000_003 + salt) & 0x7FFFFFFF

    def _subbasis_delta(self, name: str, source: torch.Tensor, r: int) -> Optional[torch.Tensor]:
        """Rank-``r`` reconstruction of ``source`` via a seeded randomized SVD.

        Returns ``U[:, :r] diag(s[:r]) V[:, :r]ᵀ`` (fp32, detached, on
        ``source``'s device, ``source``'s shape) — the dominant rank-``r``
        direction of the source, which for ``family="tail"`` (source = δ_B2 =
        the act-deflated stale weight gradient) is exactly the off-act-principal
        direction the activation codec structurally drops, and for
        ``family="grad"`` (source = M_rep) is the raw stale-anchor top-``r``.

        Determinism: the randomized projection uses a per-target
        :func:`torch.Generator` seeded by :meth:`_subbasis_seed`, so the columns
        are bit-identical across DP ranks (the source is already DP-mean
        identical). ``niter=2`` matches the act-basis block-power-iteration depth.

        Shape-guarded: returns ``None`` (the caller counts a SKIP and folds in
        the plain B2 δ unchanged) when the source is degenerate — non-finite,
        ~zero-norm, fewer than ``r`` usable directions (``r > min(shape)``), or
        not 2D — so a pathological target never injects garbage or raises.
        """
        if r <= 0:
            return None
        S = source.detach().to(torch.float32)
        if S.dim() != 2:
            return None
        m, n = S.shape
        q = min(int(r), m, n)
        if q <= 0:
            return None
        s_norm = torch.linalg.norm(S)
        if not torch.isfinite(s_norm) or float(s_norm.item()) <= 1e-12:
            return None
        # Seeded randomized SVD. ``torch.svd_lowrank`` draws an internal random
        # (n × q) projection from the GLOBAL RNG (it has no ``generator`` arg), so
        # determinism + cross-rank agreement is achieved by seeding the global RNG
        # with a rank-INVARIANT per-target seed. To leave NO global side effect on
        # the surrounding training RNG stream we save/restore the RNG state around
        # the call (``torch.random.fork_rng`` covers CPU + the input device). The
        # seed is a pure function of (base_seed, canonical-name) ⇒ identical on
        # every DP rank; the source S is already DP-mean identical ⇒ the columns
        # are bit-identical across ranks (the multi-rank-agreement invariant).
        seed = self._subbasis_seed(name)
        try:
            _devices = [S.device] if S.device.type == "cuda" else []
            with torch.random.fork_rng(devices=_devices, enabled=True):
                torch.manual_seed(seed)
                if S.device.type == "cuda":
                    torch.cuda.manual_seed_all(seed)
                U, s, V = torch.svd_lowrank(S, q=q, niter=2)
        except Exception:  # numerically degenerate sketch → skip, never raise
            return None
        if not (torch.isfinite(U).all() and torch.isfinite(s).all() and torch.isfinite(V).all()):
            return None
        # U[:, :q] diag(s[:q]) V[:, :q]ᵀ — the rank-q reconstruction of S.
        recon = (U[:, :q] * s[:q].unsqueeze(0)) @ V[:, :q].transpose(-2, -1)
        return recon.detach().to(source.device, torch.float32)

    def delayed_ef_matrix(self, name: str, g_comp: torch.Tensor, ring_grad: Optional[torch.Tensor] = None):
        """EXP-30 B2: K-delayed EXACT codec residual (the anchor-feasible EF analogue).

        ::

            δ(t)      = M_rep(t) − G_comp_ring(t−K)     # codec error on IDENTICAL (batch, θ)
            G_corr(t) = G_comp(t) + λ·δ                  # δ refreshed at fires, HELD between

        **EXP-31 Cell D: additive stale-anchor sub-basis.** When
        ``delta_subbasis_rank`` (r_sb) > 0, a rank-r_sb low-rank reconstruction of
        the source S is ADDED to the correction term::

            δ_subbasis = rank_{r_sb}(S)                   # seeded randomized SVD
            G_corr(t)  = G_comp(t) + λ·(δ + δ_subbasis)   # forward Q UNCHANGED

        ``family="tail"`` (default) takes ``S = δ`` (the act-deflated stale weight
        gradient = the off-act-principal direction the codec drops);
        ``family="grad"`` takes ``S = M_rep`` (the raw stale anchor gradient). The
        sub-basis enters ONLY this correction term — the forward/recon codec Q is
        never read or written here, so Step-C is avoided by construction. r_sb = 0
        (default) SKIPS the sub-basis branch ENTIRELY ⇒ ``G_corr = G_comp + λ·δ``
        bitwise (off-path parity). δ_subbasis is built from δ / M_rep (both
        DP-mean) via a per-target SEEDED randomized SVD, so it is bit-identical
        across DP ranks (determinism / multi-rank-agreement invariant).

        ``M_rep`` is the anchor EMA at ``β_anc=0`` — exactly the latest fire's
        generator-consistent ``G_anc_rep`` (the EXP-29 paired replay gradient).
        ``ring_grad`` is the fast COMPRESSED gradient stored at tick ``t−K`` by
        the fire-aware :class:`~verl.workers.comm_eff.state.FastGradRing` — the
        SAME (batch, θ) pair the anchor just replayed, so δ is the codec's
        weight-gradient error, not a batch effect. When ``ring_grad`` is given
        (a fire-aligned tick) δ is REFRESHED and persisted; on the in-between
        ticks the HELD δ is re-applied (the telescoping
        ``Σ_t G_corr ≈ Σ_t G_full(t−K) + drift`` needs the per-tick injection).

        **Limiting-case identity (Correctness invariant).** ``λ == 0`` returns
        ``g_comp`` EXACTLY (the same tensor object — bitwise; no fp32 round
        trip), so ``delayed_ef`` at λ=0 is plain PowerSGD.

        **Scale contract (the #25 mean-vs-sum trap).** ``M_rep`` is fed from the
        DP-MEAN-reduced anchor gradient and ``ring_grad`` from the FSDP-mean
        fast gradient under the same ``agg_loss`` normalization; this method
        applies no rescaling, so δ is well-scaled iff both feeds honor that —
        pinned by the scale-consistency unit test.

        Cold guards (never a silent grad change): unwarmed M, a missing/
        mismatched ring entry with no held δ, or a shape change ⇒ return
        ``g_comp`` unchanged and count ``merger_coldM_fallbacks``; a shape
        change also drops the stale held δ.
        """
        lam = float(self.delayed_ef_lambda)
        if lam == 0.0:
            return g_comp  # EXACT identity — the λ=0 limiting case (bitwise).
        name = _canon(name)
        self.ensure_anchor(name, g_comp)
        anc = self.anchor_on(name, g_comp.device).to(torch.float32)
        gm = g_comp.to(torch.float32)
        eps = 1e-12

        # Shape-aware held-δ carry: a logical-shape change drops the stale δ
        # (counted) BEFORE any other guard — no cross-shape leak, ever.
        held = self._delayed_ef_delta.get(name)
        if held is not None and tuple(held.shape) != tuple(gm.shape):
            held = None
            self._delayed_ef_delta.pop(name, None)
            self.residual_reset_on_shape_mismatch += 1

        anc_norm = torch.linalg.norm(anc)
        if anc_norm <= eps or tuple(anc.shape) != tuple(gm.shape):
            # COLD M (or a stored M whose logical shape no longer matches the
            # target) → G_comp unchanged; drop any held δ so a later warm step
            # starts clean.
            self.merger_coldM_fallbacks += 1
            self._delayed_ef_delta.pop(name, None)
            return g_comp

        if ring_grad is not None and tuple(ring_grad.shape) == tuple(gm.shape):
            # Fire-aligned tick: REFRESH δ from the exact (batch, θ) pair.
            rg = ring_grad.detach().to(g_comp.device, torch.float32)
            delta = (anc - rg).detach()
            store_dev = self._ema_storage_device(g_comp.device)
            stored = delta.to(store_dev)
            if store_dev.type == "cpu" and g_comp.device.type == "cuda":
                stored = stored.pin_memory()
            self._delayed_ef_delta[name] = stored
            self.delayed_ef_refreshed += 1
        elif held is not None:
            delta = held.to(g_comp.device, torch.float32)
            self.delayed_ef_held += 1
        else:
            # No exact pair yet (pre-first-fire warmup) → no-op, never invent δ.
            self.merger_coldM_fallbacks += 1
            return g_comp

        # EXP-31 Cell D: additive stale-anchor rank-r_sb sub-basis. When OFF
        # (delta_subbasis_rank == 0) this branch is SKIPPED ENTIRELY (not
        # computed-then-zeroed) so ``correction == delta`` is the EXACT B2 path
        # (off-path-parity, bitwise). When ON, the source S is the act-deflated
        # stale weight gradient δ (family="tail") — the off-act-principal
        # direction the codec misses — or the raw stale anchor gradient M_rep
        # (family="grad"); δ_subbasis = rank_{r_sb}(S) is added to δ. The forward
        # codec Q is never touched ⇒ Step-C avoidance by construction.
        if self.delta_subbasis_rank > 0:
            source = delta if self.delta_subbasis_family == "tail" else anc
            delta_sb = self._subbasis_delta(name, source, self.delta_subbasis_rank)
            if delta_sb is not None:
                correction = delta + delta_sb
                self.delayed_ef_subbasis_applied += 1
                _dn = float(torch.linalg.norm(delta).item())
                if _dn > 1e-12:
                    self._subbasis_energy_ratios.append(
                        float(torch.linalg.norm(delta_sb).item()) / _dn
                    )
            else:
                # Degenerate source → fall back to the plain B2 δ (never garbage).
                correction = delta
                self.delayed_ef_subbasis_skipped += 1
        else:
            correction = delta

        g_corr = gm + lam * correction
        return g_corr.to(g_comp.dtype)

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
    # EXP-30 Step A: correction_mode="none" is INERT by contract — no per-target
    # walk, no writeback, no counter bump; the optimizer consumes the raw
    # gradients untouched. (The engine hook also early-returns before the FSDP
    # summon for this mode; handling it here keeps the CPU-testable core safe
    # for any direct caller.)
    if getattr(spectral, "correction_mode", "signed_ema") == "none":
        return 0

    instrumented = bool(state.fsdp_grad_repr)  # log discovery only once
    corrected = 0
    # EXP-25 (R3): reset the per-step cold-M fallback counter before the loop so
    # the [comm_eff][merger] line below reports THIS step's fallbacks (N==target
    # count on step 1 when M is cold, → 0 after M warms). Mirror it onto the
    # state so comm_eff metrics can surface it.
    spectral.merger_coldM_fallbacks = 0
    # EXP-26 Step B: reset the per-step ef_powersgd residual-reset counter so the
    # [comm_eff][merger] line + metrics report THIS step's shape-mismatch resets.
    spectral.residual_reset_on_shape_mismatch = 0
    # EXP-30 B2: per-step delayed_ef refresh/hold counters + the fire-aware ring
    # context, resolved ONCE before the loop. The ring lives on the state (built
    # by CommEffState.build when correction_mode=delayed_ef); the current tick is
    # the anchor's per-train_batch counter (the cadence/staleness clock). The
    # exact ``t − delay_K`` entry exists only on fire-aligned ticks — those are
    # the δ-refresh ticks; in between, delayed_ef_matrix re-applies the held δ.
    # The loop also COLLECTS this tick's RAW pre-correction G_comp for the ring
    # when the tick is retained, and pushes AFTER the walk (a same-tick get can
    # never see its own push).
    spectral.delayed_ef_refreshed = 0
    spectral.delayed_ef_held = 0
    # EXP-31 Cell D: reset the per-step additive-sub-basis counters + the per-fire
    # energy-ratio accumulator so the [comm_eff][EXP-30][delayed_ef] line reports
    # THIS step's sub-basis activity. All zero on the OFF path (rank 0).
    spectral.delayed_ef_subbasis_applied = 0
    spectral.delayed_ef_subbasis_skipped = 0
    spectral._subbasis_energy_ratios = []
    _ring = None
    _ring_entry_grads = None
    _ring_push: dict = {}
    _ring_push_norms: dict = {}
    _tick = 0
    _delay_K = 0
    if getattr(spectral, "correction_mode", "signed_ema") == "delayed_ef":
        _ring = getattr(state, "fast_grad_ring", None)
        _tick = int(getattr(state, "anchor_step", 0) or 0)
        if _ring is not None:
            _anc_cfg = getattr(getattr(state, "config", None), "anchor", None)
            _delay_K = int(getattr(_anc_cfg, "delay_K", _ring.delay_K)) if _anc_cfg is not None else _ring.delay_K
            _entry = _ring.get(_tick - _delay_K)
            _ring_entry_grads = _entry[0] if _entry is not None else None
    # EXP-26 Step A: optional capture writer + the UNIFIED (gs, tick) key, threaded
    # from the engine. None ⇒ no dump (the byte-identical path). The optimizer tick
    # is state.capture_tick() — the SINGLE per-train_batch tick stamped at the start
    # of the fast-path forward — so G_comp/G_corr co-locate with the powersgd-hook
    # A/Â/Q, the anchor M/G_anchor, and the parallel G_dense under ONE key.
    _cap = getattr(state, "_capture_writer", None)
    _cap_gs = int(getattr(state, "global_step", -1) or -1)
    _cap_tick = int(state.capture_tick()) if hasattr(state, "capture_tick") else int(getattr(state, "spectral_step", 0) or 0)

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

        # EXP-26 Step A: dump G_comp (the merger INPUT — the fast compressed
        # gradient) BEFORE any correction, detached/fp32. No-op when _cap is None.
        if _cap is not None:
            _cap.dump(
                role="G_comp", target_name=name, tensor=full,
                global_step=_cap_gs, optimizer_tick=_cap_tick,
            )

        _mode = getattr(spectral, "correction_mode", "signed_ema")
        if _mode == "inject":
            g_proj = spectral.inject_matrix(name, full)
        elif _mode == "blend":
            g_proj = spectral.blend_matrix(name, full)
        elif _mode == "signed_ema":
            g_proj = spectral.signed_ema_matrix(name, full)
        elif _mode == "ef_powersgd":
            g_proj = spectral.ef_powersgd_matrix(name, full)
        elif _mode == "delayed_ef":
            # EXP-30 B2: collect this tick's RAW pre-correction G_comp for the
            # fire-aware ring BEFORE correcting (the ring must hold the codec's
            # output, never the merged gradient), then apply the K-delayed
            # residual. CPU fp32 storage — the zero-GPU-growth invariant.
            if _ring is not None and _ring.tick_retained(_tick):
                # copy=True is LOAD-BEARING: on an already-CPU/fp32 grad,
                # .to("cpu", fp32) is a no-op alias and the in-place writeback
                # below would silently mutate the stored ring entry.
                _raw = full.detach().to(device="cpu", dtype=torch.float32, copy=True)
                _ring_push[_canon(name)] = _raw
                _ring_push_norms[_canon(name)] = float(torch.linalg.norm(_raw).item())
            _rg = _ring_entry_grads.get(_canon(name)) if _ring_entry_grads is not None else None
            g_proj = spectral.delayed_ef_matrix(name, full, ring_grad=_rg)
        else:
            raise ValueError(
                f"comm_eff spectral correction_mode={_mode!r} is not supported; "
                "expected one of (none, inject, blend, signed_ema, ef_powersgd, delayed_ef)"
            )
        # EXP-26 Step A: dump G_corr (post-merger, pre-Adam — what the optimizer
        # will consume after writeback), detached/fp32.
        if _cap is not None:
            _cap.dump(
                role="G_corr", target_name=name, tensor=g_proj,
                global_step=_cap_gs, optimizer_tick=_cap_tick,
            )
        rel = spectral.relative_change(full, g_proj)
        state.spectral_rel_change[name] = rel
        print(
            f"[comm_eff][EXP-7][spectral] {name} correction_mode={_mode} "
            f"rel_change=||G_proj-G_mask||/||G_mask||={rel:.6f} "
            f"shape={logical_shape} grad_type={container_meta.get('grad_container_type')}",
            flush=True,
        )
        with torch.no_grad():
            writeback(grad, g_proj)

        corrected += 1
        state.spectral_corrections += 1

    # EXP-25 (R3): surface the merger's per-step cold-M fallback count + the
    # corrected-matrix count so the probe can grep them. On step 1 (M cold) the
    # fallback count == corrected (the merger no-op'd every matrix to G_noisy, NOT
    # zeroed); after M warms it drops to ~0. A signed_ema run with
    # merger_coldM_fallbacks==corrected on a LATE step would mean M never warmed
    # (coverage / broadcast broken).
    _mode = getattr(spectral, "correction_mode", "signed_ema")
    if _mode == "signed_ema":
        cold = int(getattr(spectral, "merger_coldM_fallbacks", 0))
        if hasattr(state, "merger_coldM_fallbacks"):
            state.merger_coldM_fallbacks = cold
        print(
            f"[comm_eff][merger] correction_mode=signed_ema alpha={spectral.signed_ema_alpha} "
            f"corrected={corrected} merger_coldM_fallbacks={cold} "
            f"(cold==corrected ⇒ M still cold this step; cold==0 ⇒ M fully warm)",
            flush=True,
        )
    elif _mode == "ef_powersgd":
        # EXP-26 Step B: surface the merger's per-step cold-M fallback + the
        # shape-mismatch residual-reset count so the probe can grep them. With
        # ef_decay=ef_clip=0 (the limiting case) G_corr==G_comp on every target.
        cold = int(getattr(spectral, "merger_coldM_fallbacks", 0))
        resets = int(getattr(spectral, "residual_reset_on_shape_mismatch", 0))
        if hasattr(state, "merger_coldM_fallbacks"):
            state.merger_coldM_fallbacks = cold
        if hasattr(state, "residual_reset_on_shape_mismatch"):
            state.residual_reset_on_shape_mismatch = resets
        print(
            f"[comm_eff][merger] correction_mode=ef_powersgd ef_decay={spectral.ef_decay} "
            f"ef_clip={spectral.ef_clip} corrected={corrected} merger_coldM_fallbacks={cold} "
            f"residual_reset_on_shape_mismatch={resets} "
            f"(ef_decay==ef_clip==0 ⇒ G_corr==G_comp, the plain-PowerSGD limiting case)",
            flush=True,
        )
    elif _mode == "delayed_ef":
        # EXP-30 B2: push this tick's collected RAW G_comp into the fire-aware
        # ring (post-walk, so the same-tick get never saw it), then surface the
        # per-step refresh/hold/fallback counts + the per-fire B2 tier-1 scalar
        # ||δ||/||G_comp_ring|| (median over refreshed targets) so the analyst
        # can grep "bounded, batch-refreshed, no monotone climb".
        if _ring is not None and _ring_push:
            _ring.push(_tick, _ring_push, _ring_push_norms)
        if _ring is not None and _ring_entry_grads is not None:
            _ring.pop(_tick - _delay_K)  # consumed entry — fires advance, never re-requested
        cold = int(getattr(spectral, "merger_coldM_fallbacks", 0))
        refreshed = int(getattr(spectral, "delayed_ef_refreshed", 0))
        held = int(getattr(spectral, "delayed_ef_held", 0))
        if hasattr(state, "merger_coldM_fallbacks"):
            state.merger_coldM_fallbacks = cold
        _ratio_line = ""
        if refreshed and _ring_entry_grads is not None:
            import statistics as _st

            _ratios = []
            for _n, _d in spectral._delayed_ef_delta.items():
                _g = _ring_entry_grads.get(_n)
                if _g is None:
                    continue
                _gn = float(torch.linalg.norm(_g.to(torch.float32)).item())
                if _gn > 1e-12:
                    _ratios.append(float(torch.linalg.norm(_d.to(torch.float32)).item()) / _gn)
            if _ratios:
                _ratio_line = f" delta_ratio_median={_st.median(_ratios):.6f}"
        # EXP-31 Cell D: surface the additive-sub-basis activity. applied/skipped
        # count per-target sub-basis folds; subbasis_energy_ratio (median
        # ‖δ_subbasis‖/‖δ_B2‖ over applied targets) is the headline geometry
        # scalar. All zero / absent on the OFF path (delta_subbasis_rank=0), so the
        # B2 line is unchanged in spirit (the extra fields read 0 / nan).
        import statistics as _st2

        _sb_applied = int(getattr(spectral, "delayed_ef_subbasis_applied", 0))
        _sb_skipped = int(getattr(spectral, "delayed_ef_subbasis_skipped", 0))
        _sb_ratios = list(getattr(spectral, "_subbasis_energy_ratios", []) or [])
        _sb_ratio_med = _st2.median(_sb_ratios) if _sb_ratios else float("nan")
        print(
            f"[comm_eff][EXP-30][delayed_ef] tick={_tick} lambda={spectral.delayed_ef_lambda} "
            f"corrected={corrected} refreshed={refreshed} held={held} "
            f"merger_coldM_fallbacks={cold} ring_entries={len(_ring) if _ring is not None else 0}"
            f"{_ratio_line} "
            f"subbasis_rank={getattr(spectral, 'delta_subbasis_rank', 0)} "
            f"subbasis_family={getattr(spectral, 'delta_subbasis_family', 'tail')} "
            f"subbasis_applied={_sb_applied} subbasis_skipped={_sb_skipped} "
            f"subbasis_energy_ratio={_sb_ratio_med:.6f} "
            f"(lambda==0 ⇒ G_corr==G_comp exactly; delta refreshes at fires, held between; "
            f"subbasis_rank==0 ⇒ correction==delta exactly = B2)",
            flush=True,
        )

    if corrected:
        logger.info("comm_eff: spectral correction applied to %d target matrices", corrected)
    return corrected
