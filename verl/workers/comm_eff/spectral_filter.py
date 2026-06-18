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

"""Anchor-guided correction of compressed gradients.

The filter stores a per-target anchor-gradient EMA ``M_anchor`` and applies the
selected ``correction_mode`` to the fast compressed gradient before the optimizer
step. Supported modes are ``delayed_ef``, ``ef_powersgd``, ``signed_ema``,
``inject``, ``blend`` and ``none``.

``delayed_ef`` is the paired-replay codec-residual path::

    delta(t)  = M_rep - G_comp_ring(t - K)
    G_corr(t) = G_comp(t) + lambda * delta(t)

The residual refreshes at anchor fires and is held between fires. A matrix-level
cold-M guard returns the fast gradient unchanged whenever the anchor value is
unwarmed or shape-mismatched, so the correction path never silently zeroes or
invents a gradient.

FSDP note: this module operates purely on **logical 2D matrices**. It knows
nothing about FSDP, ``DTensor``, ``FlatParameter`` or sharding. The engine-side
caller (``FSDPEngine._maybe_comm_eff_grad_correction``) owns the discovery of
what container ``p.grad`` actually is and how to present a full 2D matrix to the
merger. Keeping that out of this file makes the formula unit-testable on CPU
with no distributed runtime.
"""

from __future__ import annotations

import logging
from collections import deque
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
        name = name[len("_fsdp_wrapped_module.") :]
    return name


class SpectralFilter:
    """Stateful per-target anchor-guided gradient corrector (anchor-EMA + merger).

    Holds the running anchor-gradient EMA ``M_anchor`` for every targeted
    matrix, keyed by parameter name, and applies the selected correction on
    demand. All correction modes consult only logical 2D matrices; FSDP-specific
    extraction and writeback stay in the engine.

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
        delta_subbasis_weight: float = 1.0,
        delta_subbasis_decay_steps: int = 0,
        delta_subbasis_hold_steps: int = 0,
        base_seed: int = 0,
        perturb_sigma: float = 0.0,
        perturb_seed: int = 0,
        delta_momentum_mu: float = 0.0,
        delta_momentum_age_decay: bool = False,
        adaptive_lambda_mode: str = "off",
        adaptive_lambda_kappa: float = 0.0,
        lambda_cap: float = 2.0,
        diagnostics: bool = True,
    ):
        self.beta_anc = float(beta_anc)
        # When False, skip the per-step DIAGNOSTIC overhead (per-matrix
        # relative_change() compute+GPU->CPU sync + diagnostic prints; the
        # anchor relevance probe is gated in the engine). Default True =
        # byte-identical to prior behavior. Nothing the optimizer sees changes:
        # the g_corr writeback, the canary assert, and the aggregate counters
        # are preserved in both states.
        self.diagnostics = bool(diagnostics)
        # Storage layer default: gpu. Validation happens in
        # CommEffConfig.__post_init__ so by the time the
        # filter is built the values are known-good — assert defensively anyway.
        assert ema_device in ("gpu", "cpu"), ema_device
        # Correction mode (the anchor combiner the fast-path grad uses):
        # "none" = inert; the M EMA is still maintained but no correction is
        # applied or written back and the optimizer consumes the raw G_comp;
        # "inject" = additive injection of the scale-matched anchor complement;
        # "blend" = convex blend toward the scale-matched anchor;
        # "signed_ema" = alpha*G_noisy + (1-alpha)*|G_noisy|*sign(M);
        # "ef_powersgd" = direction-preserving error feedback;
        # "delayed_ef" = K-delayed exact codec residual,
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
        # signed_ema merger weight alpha.
        self.signed_ema_alpha = float(signed_ema_alpha)
        # Error-feedback residual knobs. decay=clip=0 reduces to plain PowerSGD.
        self.ef_decay = float(ef_decay)
        self.ef_clip = float(ef_clip)
        # delayed_ef residual weight. 0.0 returns G_comp exactly.
        self.delayed_ef_lambda = float(delayed_ef_lambda)
        # Optional additive stale-anchor rank-r sub-basis folded into the
        # delayed_ef correction term. r_sb=0 skips the branch entirely. The
        # sub-basis enters only the correction; the forward codec Q is untouched.
        self.delta_subbasis_rank = int(delta_subbasis_rank)
        assert self.delta_subbasis_rank >= 0, self.delta_subbasis_rank
        assert delta_subbasis_family in ("tail", "grad"), delta_subbasis_family
        self.delta_subbasis_family = str(delta_subbasis_family)
        # Sub-basis weight and optional hold-then-decay schedule. The scalar
        # gamma_t scales the deterministic DP-mean sub-basis and does not affect
        # the seeded SVD. current_step is stamped by the engine before correction.
        self.delta_subbasis_weight = float(delta_subbasis_weight)
        assert self.delta_subbasis_weight >= 0.0, self.delta_subbasis_weight
        self.delta_subbasis_decay_steps = int(delta_subbasis_decay_steps)
        assert self.delta_subbasis_decay_steps >= 0, self.delta_subbasis_decay_steps
        # Number of steps gamma holds at full weight before decaying.
        self.delta_subbasis_hold_steps = int(delta_subbasis_hold_steps)
        assert self.delta_subbasis_hold_steps >= 0, self.delta_subbasis_hold_steps
        # The current TRAINING step (set by the engine each grad-correction step;
        # read by delayed_ef_matrix to compute the decay factor). 0 until set.
        self.current_step = 0
        # Base seed for the per-target randomized SVD generator.
        # The low-rank sub-basis is built with ``torch.svd_lowrank`` (randomized),
        # whose result depends on a random projection. The source matrices are
        # DP-mean-identical across ranks, so to keep the sub-basis bit-identical
        # across DP ranks the random projection MUST be seeded deterministically
        # (same on every rank). We mix this base_seed with a per-target salt
        # derived from the target name (see ``_subbasis_seed``) so each target
        # gets its own reproducible generator while staying cross-rank identical.
        self.base_seed = int(base_seed)
        # Optional zero-mean, sigma-scaled, cross-rank-identical perturbation added
        # after the delayed_ef correction. sigma=0 skips the branch. The noise
        # direction is drawn from a per-(perturb_seed, target, current_step) seed that is a
        # pure function of those three — NO rank/device-local state — so every DP
        # rank draws the SAME direction (the multi-rank-agreement invariant; else the ranks
        # would descend in DIFFERENT directions and diverge). Fresh per step ⇒
        # zero-mean over training.
        self.perturb_sigma = float(perturb_sigma)
        assert self.perturb_sigma >= 0.0, self.perturb_sigma
        self.perturb_seed = int(perturb_seed)
        # Per-step count of targets the perturbation was applied to.
        self.delayed_ef_perturb_applied = 0
        # Normalized delta-momentum. mu=0 skips the branch. The recurrence
        # m <- mu*m + (1-mu)*delta has stationary gain 1 for a constant delta.
        # age_decay fades the applied held correction by mu**age.
        self.delta_momentum_mu = float(delta_momentum_mu)
        assert 0.0 <= self.delta_momentum_mu < 1.0, self.delta_momentum_mu
        assert isinstance(delta_momentum_age_decay, bool), delta_momentum_age_decay
        self.delta_momentum_age_decay = bool(delta_momentum_age_decay)
        # Per-target delta-momentum EMA buffer m, detached fp32 and shape-keyed.
        self._delta_momentum: dict[str, torch.Tensor] = {}
        # Per-target last REFRESH step (the optimizer/training step at which m was
        # last accumulated), used by age_decay to fade the held correction by the
        # number of ticks since the last fire. Pure scalar bookkeeping, cross-rank
        # identical (current_step is DP-identical).
        self._delta_momentum_last_step: dict[str, int] = {}
        # Per-step count of targets the momentum buffer was applied to.
        self.delayed_ef_momentum_applied = 0
        # Adaptive dose. mode="off" or kappa=0 keeps lambda_t constant. Otherwise
        # lambda_t = clamp(lambda + kappa*(median(c)-c_t), 0, lambda_cap), using
        # DP-mean inputs so all ranks agree.
        assert adaptive_lambda_mode in ("off", "cos", "ratio"), adaptive_lambda_mode
        self.adaptive_lambda_mode = str(adaptive_lambda_mode)
        self.adaptive_lambda_kappa = float(adaptive_lambda_kappa)
        assert self.adaptive_lambda_kappa >= 0.0, self.adaptive_lambda_kappa
        self.lambda_cap = float(lambda_cap)
        assert self.lambda_cap >= 0.0, self.lambda_cap
        # Per-target bounded agreement history for the running median. A deque of
        # plain Python floats (no tensor, no rank-local state) keyed by
        # canonical target name. Empty (and untouched) on the OFF path.
        self._adaptive_lambda_hist: dict[str, deque[float]] = {}
        # The fixed history bound (last N c_t feeding the median c̄).
        self._adaptive_lambda_hist_len = 64
        # Per-step sum + count of the applied λ_t (reset by the engine loop) so the
        # engine can log the mean λ_t. 0 on the OFF path (λ_t≡λ, branch not taken).
        self.delayed_ef_adaptive_lambda_applied = 0
        self._adaptive_lambda_sum = 0.0
        # Per-target held delayed_ef residual, detached fp32 on the EMA-storage
        # device). Refreshed when a fire-aligned ring entry exists (the anchor
        # just refreshed M_rep AND G_comp_ring(t−K) is the exact pair), HELD on
        # the in-between ticks, shape-keyed reset. β_anc=0 keeps zero EMA memory
        # in M itself; the hold is the cadence-window transport, not a carrier.
        self._delayed_ef_delta: dict[str, torch.Tensor] = {}
        # Per-step counters (reset by the engine loop): how many targets
        # REFRESHED δ this step vs reused the held one vs fell back cold.
        self.delayed_ef_refreshed = 0
        self.delayed_ef_held = 0
        # Per-step counters: how many
        # targets had the additive sub-basis APPLIED vs SKIPPED because the
        # randomized SVD was degenerate (zero/NaN source, r_sb > min-dim, etc.).
        self.delayed_ef_subbasis_applied = 0
        self.delayed_ef_subbasis_skipped = 0
        # Per-fire ||delta_subbasis||/||delta|| ratios collected this step.
        self._subbasis_energy_ratios: list[float] = []
        # Per-step count of matrices whose M was cold (||M||<=eps) so
        # the merger no-op'd to G_noisy (the silent grad-zeroing guard). Reset by
        # the engine each grad-correction step before the loop.
        self.merger_coldM_fallbacks = 0
        # Per-step count of ef_powersgd targets whose accumulated residual
        # e_t was RESET because the target's logical 2D shape changed (no stale
        # carry across a shape change). Reset by the engine each grad-correction
        # step before the loop.
        self.residual_reset_on_shape_mismatch = 0
        # name -> M_anchor (float32). Lives on the gradient's device when
        # ema_device=gpu; on (pinned) CPU when ema_device=cpu (moved to the
        # gradient's device only inside update_anchor / the combiner).
        self._anchor: dict[str, torch.Tensor] = {}
        # Per-target accumulated error-feedback residual e_t
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
        coeff = (gm * anc).sum() / (gm_norm * gm_norm + eps)  # <G_mask,M_anchor>/||G_mask||^2
        complement = anc - coeff * gm
        scale = gm_norm / (anc_norm + eps)
        g_corr = gm + self.inject_gamma * scale * complement
        if self.diagnostics:
            # Diagnostic: cosine(G_mask, M_anchor) — measures orthogonality on the LIVE anchor.
            cos = (coeff * gm_norm / (anc_norm + eps)).item()
            inj_ratio = (torch.linalg.norm(self.inject_gamma * scale * complement) / (gm_norm + eps)).item()
            print(
                f"[comm_eff][inject] {name} cos(G_mask,M_anchor)={cos:.4f} "
                f"gamma={self.inject_gamma} scale={scale.item():.4f} ||inj||/||G_mask||={inj_ratio:.4f}",
                flush=True,
            )
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
        if self.diagnostics:
            # Diagnostic: cosine(G_mask, M_anchor) on the LIVE anchor + magnitude ratio.
            cos = ((gm * anc).sum() / (gm_norm * anc_norm + eps)).item()
            print(
                f"[comm_eff][blend] {name} eta={eta} cos(G_mask,M_anchor)={cos:.4f} "
                f"||G_corr||/||G_mask||={(torch.linalg.norm(g_corr) / (gm_norm + eps)).item():.4f}",
                flush=True,
            )
        return g_corr.to(g_mask.dtype)

    def signed_ema_matrix(self, name: str, g_mask: torch.Tensor) -> torch.Tensor:
        """Signed-EMA merger: ``G_corr = alpha*G_noisy + (1-alpha)*|G_noisy|*sign(M)``.

        The magnitude comes from the fast compressed gradient ``G_noisy``
        (``g_mask``), while the sign comes from the anchor EMA ``M_anchor``.
        ``alpha=0`` gives the pure sign-merger; ``alpha=1`` returns ``G_noisy``.

        **COLD-M FALLBACK.** Mirrors the
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
        # whole matrix; the matrix-level cold guard above prevents all-zero
        # replacement before the anchor is warm.
        g_corr = alpha * gm + (1.0 - alpha) * gm.abs() * torch.sign(anc)
        return g_corr.to(g_mask.dtype)

    def ef_powersgd_matrix(self, name: str, g_mask: torch.Tensor) -> torch.Tensor:
        """Direction-preserving error-feedback PowerSGD merger.

        ``G_corr = G_comp + e_t`` where ``e_t`` is the accumulated, decayed,
        norm-clipped OFF-SUBSPACE residual — the component of the stale anchor EMA
        ``M_anchor`` that ``G_comp`` (= ``g_mask``) does NOT already span. There is **NO sign
        term**: the correction only ADDS the dropped off-principal energy, so
        ``G_corr`` keeps ``G_comp``'s direction/sign (direction-preserving, not
        sign-replacing).

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
    # Additive stale-anchor sub-basis (weight-gradient tail)
    # ------------------------------------------------------------------ #
    def _subbasis_seed(self, name: str) -> int:
        """Deterministic per-target seed for the randomized SVD generator.

        Mixes ``self.base_seed`` with a stable per-target salt derived from the
        canonical target name so (a) every target gets its OWN reproducible
        generator and (b) the seed is a pure function of (base_seed, name) — it
        contains NO rank-local / device-local state — so it is IDENTICAL on every
        DP rank. Because the source ``S`` (delta or M_rep) is already DP-MEAN
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
        direction of the source, which for ``family="tail"`` (source = delta =
        the act-deflated stale weight gradient) is exactly the off-act-principal
        direction the activation codec structurally drops, and for
        ``family="grad"`` (source = M_rep) is the raw stale-anchor top-``r``.

        Determinism: the randomized projection uses a per-target
        :func:`torch.Generator` seeded by :meth:`_subbasis_seed`, so the columns
        are bit-identical across DP ranks (the source is already DP-mean
        identical). ``niter=2`` matches the act-basis block-power-iteration depth.

        Shape-guarded: returns ``None`` (the caller counts a skip and folds in
        the plain delta unchanged) when the source is degenerate — non-finite,
        ~zero-norm, fewer than ``r`` usable directions (``r > min(shape)``), or
        not 2D — so a degenerate target never injects invalid values or raises.
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

    def _perturb_seed(self, name: str) -> int:
        """Deterministic per-(target, step) seed for the perturbation generator.

        Mixes ``self.perturb_seed`` with a stable per-target salt (the SAME FNV-1a
        hash of the canonical target name that :meth:`_subbasis_seed` uses) and the
        current TRAINING step ``self.current_step``. The result is a pure function
        of ``(perturb_seed, canonical-name, current_step)`` — it contains NO
        rank-local / device-local state — so it is IDENTICAL on every DP rank, which
        makes the drawn ξ bit-identical across ranks (the multi-rank-agreement
        invariant: a per-rank ξ would push the ranks in different directions and
        diverge). It is FRESH every step (``current_step`` advances) so the
        perturbation is zero-mean over training, not a fixed bias. We roll our own
        FNV-1a salt (Python's ``hash`` is per-process salted) for cross-process /
        cross-rank reproducibility.
        """
        salt = 0x811C9DC5  # FNV-1a 32-bit offset basis
        for b in _canon(name).encode("utf-8"):
            salt = ((salt ^ b) * 0x01000193) & 0xFFFFFFFF
        # Mix perturb_seed, the per-target salt, and the step into one 31-bit seed.
        # The step term uses a large odd multiplier so consecutive steps land in
        # well-separated regions of the generator's state space.
        mixed = (int(self.perturb_seed) * 1_000_003 + salt + int(self.current_step) * 2_654_435_761) & 0x7FFFFFFF
        return mixed

    def _apply_perturbation(self, name: str, g: torch.Tensor) -> torch.Tensor:
        """Add zero-mean, sigma-scaled, cross-rank-identical noise.

        ``g_corr ← g_corr + σ·‖g_corr‖·ξ`` where ξ is a UNIT-normalized isotropic
        Gaussian drawn from a per-(perturb_seed, target, step) seed (so the
        perturbation magnitude is exactly ``σ·‖g_corr‖`` and the DIRECTION is
        cross-rank identical). ``g`` is already detached (it is built from detached
        ``G_comp`` + ``M_rep`` + ring), so the perturbation adds NO autograd history.

        Guards (return ``g`` unchanged): a non-finite
        ‖g‖, a ~zero-norm ‖g‖ (≤ 1e-12 ⇒ nothing to scale against), or a degenerate
        ξ with ‖ξ‖ == 0. ξ is generated on CPU (a ``torch.Generator('cpu')`` is the
        portable, rank-deterministic source — CUDA generators are device-local and
        would NOT agree across ranks), then moved to ``g``'s device/dtype. Counts
        ``delayed_ef_perturb_applied`` on a successful application.
        """
        sigma = self.perturb_sigma
        if sigma <= 0.0:
            return g
        gnorm = torch.linalg.norm(g.to(torch.float32))
        if not torch.isfinite(gnorm) or float(gnorm.item()) <= 1e-12:
            return g  # nothing to scale against
        seed = self._perturb_seed(name)
        # CPU generator: device-INDEPENDENT + rank-deterministic. A CUDA generator
        # is device-local, so two ranks on different GPUs would draw DIFFERENT ξ and
        # diverge — the CPU draw guarantees the cross-rank-identical direction.
        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        xi = torch.randn(g.shape, generator=gen, dtype=torch.float32, device="cpu")
        xi_norm = torch.linalg.norm(xi)
        if not torch.isfinite(xi_norm) or float(xi_norm.item()) <= 0.0:
            return g  # degenerate draw → no-op
        xi = xi / xi_norm  # unit direction
        # Move ξ to g's device/dtype; scale by σ·‖g‖. The perturbation magnitude is
        # ‖σ·‖g‖·ξ‖ = σ·‖g‖ (ξ is unit) — exactly the spec's scale contract.
        perturbation = (sigma * gnorm.to(g.device)) * xi.to(g.device, g.dtype)
        self.delayed_ef_perturb_applied += 1
        return g + perturbation

    def _subbasis_gamma(self) -> float:
        """Sub-basis weight gamma_t at ``self.current_step``.

        ``γ_t = delta_subbasis_weight · decay_factor`` — a HOLD-then-decay schedule
        with ``h = delta_subbasis_hold_steps``, ``d = delta_subbasis_decay_steps``,
        ``s = current_step``::

            decay_factor = 1.0                              if d <= 0   (constant γ = weight)
                         = 1.0                              if s < h    (the HOLD shelf)
                         = max(0.0, 1.0 − (s − h) / d)       else        (linear ramp 1→0)

        i.e. γ holds at the full ``weight`` for steps ``0..h-1``, then decays
        linearly to 0 over the next ``d`` steps (reaching 0 at ``s == h + d``,
        clamped at 0 past the horizon). With ``decay_steps=0`` this is a constant
        ``weight``. With ``hold_steps=0`` the shelf is empty and the schedule starts
        decaying immediately. ``weight=0`` skips the sub-basis term. This scalar
        has no effect on the seeded sub-basis SVD.
        """
        w = self.delta_subbasis_weight
        d = self.delta_subbasis_decay_steps
        h = self.delta_subbasis_hold_steps
        if d <= 0:
            return float(w)
        if self.current_step < h:
            # The HOLD shelf: full weight. With h=0 this branch is never taken,
            # so the formula below reduces to linear-from-0 decay.
            return float(w)
        decay_factor = 1.0 - (float(self.current_step - h) / float(d))
        if decay_factor < 0.0:
            decay_factor = 0.0
        return float(w) * decay_factor

    # ------------------------------------------------------------------ #
    # Delta momentum (normalized EMA, stationary gain 1)
    # ------------------------------------------------------------------ #
    def _apply_delta_momentum(self, name: str, delta: torch.Tensor, refreshed: bool) -> torch.Tensor:
        """Return the correction after the normalized-EMA δ-momentum transform.

        OFF-GUARD: ``delta_momentum_mu == 0.0`` skips the branch, returns
        ``delta`` unchanged, and touches no buffer. ``name`` is assumed already
        canonicalized by the caller.

        NORMALIZED EMA (stationary gain EXACTLY 1):

        * REFRESH tick (``refreshed=True`` — a fire-aligned tick where δ was just
          recomputed from the exact (batch, θ) pair): accumulate
          ``m ← μ·m + (1−μ)·δ`` (first fire for this target: ``m = δ.clone()``), then
          the correction is ``m``. The stationary gain of this recurrence is EXACTLY
          1 (a constant δ stream drives ``m -> δ``), a re-weighting rather than a
          gain increase.
          The accumulation happens ONLY at refresh ticks (δ is HELD between fires in
          :meth:`delayed_ef_matrix`; accumulating every tick would re-add the same
          δ across the hold window).
        * HELD tick (``refreshed=False``): the correction is the HELD buffer ``m``
          (NOT re-accumulated). With ``delta_momentum_age_decay`` the APPLIED
          correction is scaled by ``μ ** (current_step − last_refresh_step)`` so a
          long hold fades it → 0 (the async staleness-degrade requirement); the
          STORED buffer ``m`` is left unchanged.

        The buffer ``m`` is built from the DP-mean ``delta`` (cross-rank identical),
        detached fp32, on the EMA storage device, shape-keyed reset (mirrors the
        held-δ shape guard: a logical-shape change drops the stale buffer + the
        last-step bookkeeping). If, on a HELD tick, no buffer exists yet (the first
        fire has not happened for this target) the plain ``delta`` is returned so the
        correction is never silently invented.

        Returns the correction tensor (fp32, on ``delta``'s device).
        """
        mu = self.delta_momentum_mu
        if mu == 0.0:
            return delta  # OFF: no buffer touched.

        # Shape-keyed reset (mirror the held-δ guard): a logical-shape change drops
        # the stale buffer + its last-refresh bookkeeping so nothing cross-shape leaks.
        m = self._delta_momentum.get(name)
        if m is not None and tuple(m.shape) != tuple(delta.shape):
            m = None
            self._delta_momentum.pop(name, None)
            self._delta_momentum_last_step.pop(name, None)

        store_dev = self._ema_storage_device(delta.device)

        if refreshed:
            # Accumulate ONLY at refresh ticks (δ is HELD between fires). Normalized
            # EMA ⇒ stationary gain EXACTLY 1.
            if m is None:
                m_new = delta.detach().to(torch.float32).clone()
            else:
                m_dev = m.to(delta.device, torch.float32)
                m_new = mu * m_dev + (1.0 - mu) * delta.detach().to(torch.float32)
            stored = m_new.detach().to(store_dev)
            if store_dev.type == "cpu" and delta.device.type == "cuda":
                stored = stored.pin_memory()
            self._delta_momentum[name] = stored
            self._delta_momentum_last_step[name] = int(self.current_step)
            self.delayed_ef_momentum_applied += 1
            return m_new

        # HELD tick: use the held buffer (do NOT re-accumulate). No buffer yet ⇒
        # fall back to the plain δ (never invent a correction).
        if m is None:
            return delta
        correction = m.to(delta.device, torch.float32)
        if self.delta_momentum_age_decay:
            last = self._delta_momentum_last_step.get(name, int(self.current_step))
            age = int(self.current_step) - int(last)
            if age < 0:
                age = 0
            # APPLIED-only scaling: the stored buffer is unchanged; only this tick's
            # injected correction fades by μ**age so a long hold → 0 (async degrade).
            correction = correction * (mu**age)
        self.delayed_ef_momentum_applied += 1
        return correction

    # ------------------------------------------------------------------ #
    # Adaptive dose (centered gate)
    # ------------------------------------------------------------------ #
    def _adaptive_lambda(self, name: str, gm: torch.Tensor, anc: torch.Tensor, delta_raw: torch.Tensor) -> float:
        """Return the per-target, per-tick dose ``λ_t`` (MEAN-1 CENTERED gate).

        OFF-GUARD: ``adaptive_lambda_mode == "off"`` OR
        ``adaptive_lambda_kappa == 0.0`` ⇒ ``λ_t = self.delayed_ef_lambda`` (the
        constant dose) and no history is touched. ``name`` is assumed already
        canonicalized.

        When ON::

            c_t = cos(gm, anc)        [mode=cos]   OR   ‖delta_raw‖/‖gm‖  [mode=ratio]
            c̄   = running MEDIAN of c_t over the last ``_adaptive_lambda_hist_len`` ticks
            λ_t = clamp(self.delayed_ef_lambda + κ·(c̄ − c_t), 0.0, lambda_cap)

        Centered: because ``c̄`` is the median of the very ``c_t`` stream, the
        deviation ``(c̄ − c_t)`` is centered at ~0 ⇒ ``E[λ_t] ≈ delayed_ef_lambda``
        for a stationary agreement distribution. Only the step-to-step deviation
        changes the dose.

        ``c_t`` uses the RAW δ (captured BEFORE the L2 momentum transform). The
        history deque carries plain Python floats (no tensor / no rank-local state);
        ``gm`` (FSDP DP-mean) and ``anc`` (all-reduced-mean M_rep) are DP-identical ⇒
        ``c_t`` / ``c̄`` / ``λ_t`` are bit-identical across ranks. The current tick's
        ``c_t`` is appended to the history AND included in the median (so the very
        first tick has ``c̄ = c_t`` ⇒ deviation 0 ⇒ ``λ_t = delayed_ef_lambda``).
        """
        lam = float(self.delayed_ef_lambda)
        if self.adaptive_lambda_mode == "off" or self.adaptive_lambda_kappa == 0.0:
            return lam  # OFF: constant dose, no history touched.

        eps = 1e-12
        if self.adaptive_lambda_mode == "cos":
            gm_norm = torch.linalg.norm(gm)
            anc_norm = torch.linalg.norm(anc)
            c_t = float(((gm * anc).sum() / (gm_norm * anc_norm + eps)).item())
        else:  # "ratio": ‖delta‖/‖gm‖
            gm_norm = torch.linalg.norm(gm)
            d_norm = torch.linalg.norm(delta_raw)
            c_t = float((d_norm / (gm_norm + eps)).item())
        if not (c_t == c_t):  # NaN guard: fall back to the constant dose.
            return lam

        hist = self._adaptive_lambda_hist.get(name)
        if hist is None:
            hist = deque(maxlen=self._adaptive_lambda_hist_len)
            self._adaptive_lambda_hist[name] = hist
        hist.append(c_t)
        # Running median c̄ over the bounded history (includes the current c_t).
        vals = sorted(hist)
        nlen = len(vals)
        mid = nlen // 2
        c_bar = vals[mid] if (nlen % 2 == 1) else 0.5 * (vals[mid - 1] + vals[mid])

        lam_t = lam + self.adaptive_lambda_kappa * (c_bar - c_t)
        # Bounded raw dose (variable-staleness safety): a stale or invalid M cannot spike
        # λ_t beyond lambda_cap; the floor 0 forbids a sign-flipping negative dose.
        if lam_t < 0.0:
            lam_t = 0.0
        elif lam_t > self.lambda_cap:
            lam_t = self.lambda_cap
        self.delayed_ef_adaptive_lambda_applied += 1
        self._adaptive_lambda_sum += float(lam_t)
        return float(lam_t)

    def delayed_ef_matrix(self, name: str, g_comp: torch.Tensor, ring_grad: Optional[torch.Tensor] = None):
        """K-delayed exact codec residual.

        ::

            δ(t)      = M_rep(t) − G_comp_ring(t−K)     # codec error on IDENTICAL (batch, θ)
            G_corr(t) = G_comp(t) + λ·δ                  # δ refreshed at fires, HELD between

        **Additive stale-anchor sub-basis.** When
        ``delta_subbasis_rank`` (r_sb) > 0 AND the weight γ_t > 0, a rank-r_sb
        low-rank reconstruction of the source S is ADDED to the correction term,
        scaled by the (optionally decaying) weight γ_t::

            delta_subbasis = rank_{r_sb}(S)                   # seeded randomized SVD
            γ_t = weight · schedule(step, hold_steps, decay_steps)
            G_corr(t)  = G_comp(t) + lambda*(delta + gamma_t*delta_subbasis)

        ``weight=1.0, decay_steps=0`` keeps gamma_t at 1.0; ``weight=0`` skips the
        sub-basis branch, leaving ``correction == delta``. gamma_t is a scalar; see
        :meth:`_subbasis_gamma`.

        ``family="tail"`` (default) takes ``S = δ`` (the act-deflated stale weight
        gradient = the off-act-principal direction the codec drops);
        ``family="grad"`` takes ``S = M_rep`` (the raw stale anchor gradient). The
        sub-basis enters ONLY this correction term — the forward/recon codec Q is
        never read or written here, so forward-basis updates are avoided. r_sb = 0
        (default) SKIPS the sub-basis branch ENTIRELY ⇒ ``G_corr = G_comp + λ·δ``
        bitwise (off-path parity). δ_subbasis is built from δ / M_rep (both
        DP-mean) via a per-target SEEDED randomized SVD, so it is bit-identical
        across DP ranks (determinism / multi-rank-agreement invariant).

        ``M_rep`` is the anchor EMA at ``β_anc=0`` — exactly the latest fire's
        generator-consistent ``G_anc_rep`` from paired replay.
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

        **Scale contract.** ``M_rep`` is fed from the
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
            refreshed = True
        elif held is not None:
            delta = held.to(g_comp.device, torch.float32)
            self.delayed_ef_held += 1
            refreshed = False
        else:
            # No exact pair yet (pre-first-fire warmup) → no-op, never invent δ.
            self.merger_coldM_fallbacks += 1
            return g_comp

        # Adaptive dose reads the raw delta before the momentum transform
        # for its agreement metric c_t = ‖δ‖/‖gm‖, so capture it here; cos-mode reads
        # gm/anc directly. λ_t is computed AFTER the correction is finalized, but the
        # raw δ it depends on is this fire's exact codec residual, not the momentum
        # buffer. When adaptive dose is off, the helper returns the constant dose.
        delta_raw = delta

        # Delta momentum. When off
        # (delta_momentum_mu == 0.0) the helper returns ``delta`` UNCHANGED (the same
        # object) and touches no buffer. When on, the per-target buffer m ← μ·m + (1−μ)·δ is accumulated
        # ONLY at refresh ticks (δ is HELD between fires) and the held buffer (faded
        # by age when age_decay) is the correction on the in-between ticks. The buffer
        # is the DP-mean δ ⇒ cross-rank identical. The rest of the function (sub-basis
        # branch + g_corr) then uses this transformed ``delta``.
        delta = self._apply_delta_momentum(name, delta, refreshed)

        # Additive stale-anchor rank-r_sb sub-basis (gamma-weighted).
        # When OFF (delta_subbasis_rank == 0) OR the weight γ_t == 0 this branch is
        # skipped entirely (not computed-then-zeroed), so ``correction == delta``.
        # When on, the source S is the act-deflated stale weight gradient δ
        # (family="tail") — the off-act-principal direction the codec misses — or
        # the raw stale anchor gradient M_rep (family="grad"); δ_subbasis =
        # rank_{r_sb}(S) is added to δ, scaled by gamma_t. The forward codec Q is
        # never touched.
        gamma_t = self._subbasis_gamma() if self.delta_subbasis_rank > 0 else 0.0
        if self.delta_subbasis_rank > 0 and gamma_t != 0.0:
            source = delta if self.delta_subbasis_family == "tail" else anc
            delta_sb = self._subbasis_delta(name, source, self.delta_subbasis_rank)
            if delta_sb is not None:
                correction = delta + gamma_t * delta_sb
                self.delayed_ef_subbasis_applied += 1
                _dn = float(torch.linalg.norm(delta).item())
                if _dn > 1e-12:
                    # ||gamma_t*delta_subbasis|| / ||delta||: the effective injected energy ratio
                    # (so the logged median reflects the decayed weight, not the raw
                    # sub-basis).
                    self._subbasis_energy_ratios.append(abs(gamma_t) * float(torch.linalg.norm(delta_sb).item()) / _dn)
            else:
                # Degenerate source: fall back to the plain delta.
                correction = delta
                self.delayed_ef_subbasis_skipped += 1
        else:
            # rank-0 OR gamma_t==0: use the plain delta and skip the SVD.
            correction = delta

        # Adaptive dose. The constant ``lam`` is
        # replaced by a per-target, per-tick λ_t = clamp(λ + κ·(c̄ − c_t), 0, cap)
        # built from the agreement c_t (cos(gm,anc) or ‖δ_raw‖/‖gm‖) and its running
        # median c̄. When off (mode="off" OR κ=0), the helper returns the constant
        # ``self.delayed_ef_lambda`` and touches no history. c_t/c̄/λ_t are built from the
        # DP-mean gm + anc ⇒ cross-rank identical.
        lam_t = self._adaptive_lambda(name, gm, anc, delta_raw)
        g_corr = gm + lam_t * correction
        # Optional zero-mean sigma-scaled cross-rank-identical perturbation.
        if self.perturb_sigma > 0.0:
            g_corr = self._apply_perturbation(name, g_corr)
        return g_corr.to(g_comp.dtype)

    def relative_change(self, g_mask: torch.Tensor, g_proj: torch.Tensor) -> float:
        """Per-target ``||G_proj - G_mask|| / ||G_mask||`` (Frobenius).

        Logged faithfully (not clamped): this is not provably ≤1 for arbitrary
        anchors, so report whatever the math yields.
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
    spectral: SpectralFilter,
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
    # correction_mode="none" is inert by contract: no per-target
    # walk, no writeback, no counter bump; the optimizer consumes the raw
    # gradients untouched. (The engine hook also early-returns before the FSDP
    # summon for this mode; handling it here keeps the CPU-testable core safe
    # for any direct caller.)
    if getattr(spectral, "correction_mode", "signed_ema") == "none":
        return 0

    instrumented = bool(state.fsdp_grad_repr)  # log discovery only once
    corrected = 0
    # Reset the per-step cold-M fallback counter before the loop so
    # the [comm_eff][merger] line below reports THIS step's fallbacks (N==target
    # count on step 1 when M is cold, → 0 after M warms). Mirror it onto the
    # state so comm_eff metrics can surface it.
    spectral.merger_coldM_fallbacks = 0
    # Reset the per-step ef_powersgd residual-reset counter so the
    # [comm_eff][merger] line + metrics report THIS step's shape-mismatch resets.
    spectral.residual_reset_on_shape_mismatch = 0
    # Per-step delayed_ef refresh/hold counters plus fire-aware ring
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
    # Reset the per-step additive-sub-basis counters plus the per-fire
    # energy-ratio accumulator so the [comm_eff][delayed_ef] line reports
    # THIS step's sub-basis activity. All zero on the OFF path (rank 0).
    spectral.delayed_ef_subbasis_applied = 0
    spectral.delayed_ef_subbasis_skipped = 0
    spectral._subbasis_energy_ratios = []
    # Reset the per-step perturbation-applied counter so the
    # [comm_eff][delayed_ef] line reports THIS step's perturbation activity.
    # 0 on the OFF path (perturb_sigma=0) and on cold/skip ticks.
    spectral.delayed_ef_perturb_applied = 0
    # Reset per-step delta-momentum and adaptive-lambda telemetry so the
    # [comm_eff][delayed_ef] line reports THIS step's lever activity. All
    # zero on the OFF paths (delta_momentum_mu=0 / adaptive_lambda_mode=off|κ=0).
    spectral.delayed_ef_momentum_applied = 0
    spectral.delayed_ef_adaptive_lambda_applied = 0
    spectral._adaptive_lambda_sum = 0.0
    # Stamp this step onto the filter so the sub-basis decay
    # schedule (``_subbasis_gamma``) reads the current training step. Cleanly wired
    # from state.global_step (the same counter the capture key uses below). Defaults
    # to 0 when the state lacks it (CPU-test ducks) ⇒ γ_t = weight at step 0. With
    # the OFF defaults (weight=1, decay_steps=0) γ_t is a constant 1.0 regardless,
    # so this assignment is a no-op when the feature is disabled.
    spectral.current_step = int(getattr(state, "global_step", 0) or 0)
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
    # Optional capture writer and the unified (gs, tick) key, threaded
    # from the engine. None means no dump. The optimizer tick
    # is state.capture_tick() — the SINGLE per-train_batch tick stamped at the start
    # of the fast-path forward — so G_comp/G_corr co-locate with the powersgd-hook
    # A/Â/Q, the anchor M/G_anchor, and the parallel G_dense under ONE key.
    _cap = getattr(state, "_capture_writer", None)
    _cap_gs = int(getattr(state, "global_step", -1) or -1)
    _cap_tick = (
        int(state.capture_tick()) if hasattr(state, "capture_tick") else int(getattr(state, "spectral_step", 0) or 0)
    )

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
            # The stdout DISCOVERY line is a pure diagnostic echo of state.fsdp_grad_repr
            # (which is preserved + surfaced into metrics unconditionally above);
            # gate only the print. Fires once per run regardless.
            if getattr(spectral, "diagnostics", True):
                print(f"[comm_eff][FSDP-DISCOVERY] {repr_log}", flush=True)
            instrumented = True

        # Dump G_comp, the merger input: the fast compressed
        # gradient) BEFORE any correction, detached/fp32. No-op when _cap is None.
        if _cap is not None:
            _cap.dump(
                role="G_comp",
                target_name=name,
                tensor=full,
                global_step=_cap_gs,
                optimizer_tick=_cap_tick,
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
            # Collect this tick's raw pre-correction G_comp for the
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
        # Dump G_corr, post-merger and pre-Adam: what the optimizer
        # will consume after writeback), detached/fp32.
        if _cap is not None:
            _cap.dump(
                role="G_corr",
                target_name=name,
                tensor=g_proj,
                global_step=_cap_gs,
                optimizer_tick=_cap_tick,
            )
        # DIAGNOSTIC ONLY: the per-matrix rel_change compute is a GPU->CPU
        # .item() sync (~196/step) that nothing the optimizer sees consumes.
        # Gated by spectral.diagnostics (default True = byte-identical). The
        # g_corr writeback below is UNCONDITIONAL.
        if getattr(spectral, "diagnostics", True):
            rel = spectral.relative_change(full, g_proj)
            state.spectral_rel_change[name] = rel
            print(
                f"[comm_eff][spectral] {name} correction_mode={_mode} "
                f"rel_change=||G_proj-G_mask||/||G_mask||={rel:.6f} "
                f"shape={logical_shape} grad_type={container_meta.get('grad_container_type')}",
                flush=True,
            )
        with torch.no_grad():
            writeback(grad, g_proj)

        corrected += 1
        state.spectral_corrections += 1

    # Surface the merger's per-step cold-M fallback count plus the
    # corrected-matrix count so the probe can grep them. On step 1 (M cold) the
    # fallback count == corrected (the merger no-op'd every matrix to G_noisy, NOT
    # zeroed); after M warms it drops to ~0. A signed_ema run with
    # merger_coldM_fallbacks==corrected on a LATE step would mean M never warmed
    # (coverage / broadcast broken).
    _mode = getattr(spectral, "correction_mode", "signed_ema")
    if _mode == "signed_ema":
        cold = int(getattr(spectral, "merger_coldM_fallbacks", 0))
        # Aggregate counter: preserved in both states (NOT diagnostic).
        if hasattr(state, "merger_coldM_fallbacks"):
            state.merger_coldM_fallbacks = cold
        if getattr(spectral, "diagnostics", True):
            print(
                f"[comm_eff][merger] correction_mode=signed_ema alpha={spectral.signed_ema_alpha} "
                f"corrected={corrected} merger_coldM_fallbacks={cold} "
                f"(cold==corrected ⇒ M still cold this step; cold==0 ⇒ M fully warm)",
                flush=True,
            )
    elif _mode == "ef_powersgd":
        # Surface the merger's per-step cold-M fallback plus the
        # shape-mismatch residual-reset count so the probe can grep them. With
        # ef_decay=ef_clip=0 (the limiting case) G_corr==G_comp on every target.
        cold = int(getattr(spectral, "merger_coldM_fallbacks", 0))
        resets = int(getattr(spectral, "residual_reset_on_shape_mismatch", 0))
        # Aggregate counters: preserved in both states (NOT diagnostic).
        if hasattr(state, "merger_coldM_fallbacks"):
            state.merger_coldM_fallbacks = cold
        if hasattr(state, "residual_reset_on_shape_mismatch"):
            state.residual_reset_on_shape_mismatch = resets
        if getattr(spectral, "diagnostics", True):
            print(
                f"[comm_eff][merger] correction_mode=ef_powersgd ef_decay={spectral.ef_decay} "
                f"ef_clip={spectral.ef_clip} corrected={corrected} merger_coldM_fallbacks={cold} "
                f"residual_reset_on_shape_mismatch={resets} "
                f"(ef_decay==ef_clip==0 ⇒ G_corr==G_comp, the plain-PowerSGD limiting case)",
                flush=True,
            )
    elif _mode == "delayed_ef":
        # Push this tick's collected raw G_comp into the fire-aware
        # ring (post-walk, so the same-tick get never saw it), then surface the
        # per-step refresh/hold/fallback counts plus the per-fire scalar
        # ||δ||/||G_comp_ring|| (median over refreshed targets) so the analyst
        # can grep "bounded, batch-refreshed, no monotone climb".
        if _ring is not None and _ring_push:
            _ring.push(_tick, _ring_push, _ring_push_norms)
        if _ring is not None and _ring_entry_grads is not None:
            _ring.pop(_tick - _delay_K)  # consumed entry — fires advance, never re-requested
        cold = int(getattr(spectral, "merger_coldM_fallbacks", 0))
        refreshed = int(getattr(spectral, "delayed_ef_refreshed", 0))
        held = int(getattr(spectral, "delayed_ef_held", 0))
        # Aggregate counter: preserved in both states (NOT diagnostic).
        if hasattr(state, "merger_coldM_fallbacks"):
            state.merger_coldM_fallbacks = cold
        # Everything below (the per-fire ||δ|| ratio scan + the sub-basis/perturb
        # readouts + the summary line) is DIAGNOSTIC ONLY: it feeds only the
        # print and includes per-ring-grad GPU->CPU .item() syncs. The ring
        # push/pop and the cold-M counter above are UNCONDITIONAL. Gated by
        # spectral.diagnostics (default True = byte-identical).
        if getattr(spectral, "diagnostics", True):
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
            # Surface the additive-sub-basis activity. applied/skipped
            # count per-target sub-basis folds; subbasis_energy_ratio (median
            # ||delta_subbasis||/||delta|| over applied targets) is the geometry
            # scalar. All zero or absent when delta_subbasis_rank=0.
            import statistics as _st2

            _sb_applied = int(getattr(spectral, "delayed_ef_subbasis_applied", 0))
            _sb_skipped = int(getattr(spectral, "delayed_ef_subbasis_skipped", 0))
            _sb_ratios = list(getattr(spectral, "_subbasis_energy_ratios", []) or [])
            _sb_ratio_med = _st2.median(_sb_ratios) if _sb_ratios else float("nan")
            # Sub-basis weight gamma_t applied this step after
            # the linear decay over delta_subbasis_decay_steps). 1.0 on the OFF default
            # (weight=1, decay_steps=0); 0.0 means the sub-basis branch was skipped.
            _sb_gamma = spectral._subbasis_gamma() if getattr(spectral, "delta_subbasis_rank", 0) > 0 else 0.0
            # Zero-mean perturbation sigma plus how many targets it
            # was applied to THIS step. σ=0 (OFF) ⇒ perturb_applied==0 (the line reads
            # the perturbation never fired).
            _pt_sigma = float(getattr(spectral, "perturb_sigma", 0.0))
            _pt_applied = int(getattr(spectral, "delayed_ef_perturb_applied", 0))
            print(
                f"[comm_eff][delayed_ef] tick={_tick} lambda={spectral.delayed_ef_lambda} "
                f"corrected={corrected} refreshed={refreshed} held={held} "
                f"merger_coldM_fallbacks={cold} ring_entries={len(_ring) if _ring is not None else 0}"
                f"{_ratio_line} "
                f"subbasis_rank={getattr(spectral, 'delta_subbasis_rank', 0)} "
                f"subbasis_family={getattr(spectral, 'delta_subbasis_family', 'tail')} "
                f"subbasis_weight={getattr(spectral, 'delta_subbasis_weight', 1.0)} "
                f"subbasis_decay_steps={getattr(spectral, 'delta_subbasis_decay_steps', 0)} "
                f"subbasis_hold_steps={getattr(spectral, 'delta_subbasis_hold_steps', 0)} "
                f"subbasis_step={getattr(spectral, 'current_step', 0)} "
                f"subbasis_gamma={_sb_gamma:.6f} "
                f"subbasis_applied={_sb_applied} subbasis_skipped={_sb_skipped} "
                f"subbasis_energy_ratio={_sb_ratio_med:.6f} "
                f"perturb_sigma={_pt_sigma} perturb_seed={getattr(spectral, 'perturb_seed', 0)} "
                f"perturb_applied={_pt_applied} "
                f"(lambda==0 ⇒ G_corr==G_comp exactly; delta refreshes at fires, held between; "
                f"subbasis_rank==0 OR gamma==0 ⇒ correction==delta exactly; "
                f"perturb_sigma==0 ⇒ g_corr unperturbed)",
                flush=True,
            )

    if corrected:
        logger.info("comm_eff: spectral correction applied to %d target matrices", corrected)
    return corrected
