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

"""Anchor-guided correction of compressed actor gradients.

The dense anchor supplies a per-parameter gradient EMA ``M``. Before the
optimizer step, the selected ``correction_mode`` merges it into the fast
compressed gradient.

``signed_ema`` keeps the compressed gradient's magnitude and blends its sign
with ``sign(M)``::

    G_corr = alpha * G_comp + (1 - alpha) * abs(G_comp) * sign(M)

An optional learned gate (``signed_gate="learned"``) doses that correction
between anchor fires. The fire tick applies it at full strength with the fresh
``M`` (the proven use-once dose). Each held tick applies::

    G_corr = (1 - w) * G_comp + w * G_signed,   w = rho * decay**age

where ``age`` counts correction ticks since the fire and ``rho`` is a
per-target EMA of the measured sign agreement between the direction that was
being held (``sign(M_old)``) and the next fire's fresh dense anchor gradient
(``sign(G_anchor)``), clamped to [0, 1]. Each pipeline stage measures its own
tensors locally, so the gate costs no link traffic. ``rho`` starts at 0: the
schedule begins as use-once and earns between-fire dose only from agreement
evidence. ``decay < 1`` is enforced so the learned schedule stays inside the
geometric envelope and can never re-create standing full-strength reuse.

``delayed_ef`` is the additive anchor-residual path. On the tick whose anchor
fire refreshed ``M`` (at ``beta_anc=0``, ``M`` IS that fire's raw dense anchor
gradient, computed at the RELEX-projected weights on the SAME batch as this
tick's fast gradient), the residual is rebuilt and applied; between fires the
HELD residual is re-applied to each new fast gradient::

    delta      = M - G_comp(fire tick)          # refreshed once per anchor fire
    G_corr(t)  = G_comp(t) + lambda * delta     # held delta between fires

At ``lambda=1`` and ``beta_anc=0`` the fire tick reduces exactly to
``G_corr = G_anchor`` (the compressed gradient cancels), and ``lambda=0`` is a
bitwise identity. Whether stale ``delta`` reuse between fires happens at all is
the caller's ``spectral.cadence``: 1 re-applies the held residual every tick,
``cadence == anchor.cadence`` applies it on fire ticks only.

An unwarmed or shape-mismatched anchor entry is a strict no-op for that tensor
in both modes. FSDP extraction and writeback remain engine responsibilities;
this module only operates on logical full tensors.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

__all__ = ["SpectralFilter", "apply_spectral_correction_to_params", "is_spectral_target"]

SPECTRAL_TARGET_SCOPES = ("decoder_matrices", "all_floating")
SPECTRAL_CORRECTION_MODES = ("signed_ema", "delayed_ef", "blend")
SIGNED_GATE_MODES = ("off", "learned")
# EMA coefficient for the per-target sign-agreement estimate rho behind the
# learned signed gate: rho <- (1-beta)*rho + beta*rho_raw at each anchor fire.
# A constant, not a knob: it only sets how fast the gate trusts new agreement
# evidence (0.5 covers 94% of a step change within 4 fires).
GATE_RHO_EMA_BETA = 0.5


def is_spectral_target(
    name: str,
    tensor: torch.Tensor,
    *,
    target_substrs,
    target_scope: str = "decoder_matrices",
) -> bool:
    """Return whether ``tensor`` belongs to the configured M/correction set."""

    if target_scope not in SPECTRAL_TARGET_SCOPES:
        raise ValueError(
            f"unknown comm_eff spectral target_scope={target_scope!r}; expected one of {SPECTRAL_TARGET_SCOPES}"
        )
    if target_scope == "all_floating":
        return bool(tensor.is_floating_point())
    return tensor.dim() == 2 and any(part in name for part in target_substrs)


def _canon(name: str) -> str:
    """Strip FSDP wrapping infixes so live and anchor-clone names share one key."""

    name = name.replace("._fsdp_wrapped_module", "")
    if name.startswith("_fsdp_wrapped_module."):
        name = name[len("_fsdp_wrapped_module.") :]
    return name


class SpectralFilter:
    """Per-target dense-anchor EMA and signed-gradient merger."""

    def __init__(
        self,
        *,
        beta_anc: float = 0.50,
        ema_device: str = "cpu",
        signed_ema_alpha: float = 0.25,
        correction_mode: str = "signed_ema",
        delayed_ef_lambda: float = 1.0,
        delayed_ef_decay: float = 1.0,
        blend_eta: float = 0.5,
        signed_gate: str = "off",
        signed_gate_decay: float = 0.75,
        diagnostics: bool = False,
    ):
        if not 0.0 <= float(beta_anc) <= 1.0:
            raise ValueError(f"beta_anc must be in [0, 1]; got {beta_anc}")
        if ema_device not in ("gpu", "cpu"):
            raise ValueError(f"ema_device must be one of (gpu, cpu); got {ema_device!r}")
        if not 0.0 <= float(signed_ema_alpha) <= 1.0:
            raise ValueError(f"signed_ema_alpha must be in [0, 1]; got {signed_ema_alpha}")
        if correction_mode not in SPECTRAL_CORRECTION_MODES:
            raise ValueError(f"correction_mode must be one of {SPECTRAL_CORRECTION_MODES}; got {correction_mode!r}")
        if not float(delayed_ef_lambda) >= 0.0:
            raise ValueError(f"delayed_ef_lambda must be >= 0; got {delayed_ef_lambda}")
        if not 0.0 <= float(delayed_ef_decay) <= 1.0:
            raise ValueError(f"delayed_ef_decay must be in [0, 1]; got {delayed_ef_decay}")
        if not 0.0 <= float(blend_eta) <= 1.0:
            raise ValueError(f"blend_eta must be in [0, 1]; got {blend_eta}")
        if signed_gate not in SIGNED_GATE_MODES:
            raise ValueError(f"signed_gate must be one of {SIGNED_GATE_MODES}; got {signed_gate!r}")
        if not 0.0 <= float(signed_gate_decay) < 1.0:
            raise ValueError(
                "signed_gate_decay must be in [0, 1): decay=1 would let the gated correction "
                f"stand between fires indefinitely, the reuse pattern the gate exists to prevent; "
                f"got {signed_gate_decay}"
            )

        self.beta_anc = float(beta_anc)
        self.ema_device = str(ema_device)
        self.signed_ema_alpha = float(signed_ema_alpha)
        self.correction_mode = str(correction_mode)
        self.delayed_ef_lambda = float(delayed_ef_lambda)
        self.delayed_ef_decay = float(delayed_ef_decay)
        self.blend_eta = float(blend_eta)
        self.diagnostics = bool(diagnostics)
        self.merger_coldM_fallbacks = 0
        self._anchor: dict[str, torch.Tensor] = {}
        # delayed_ef state. The residual delta is held per target between anchor
        # fires; _m_version counts update_anchor() calls per target and
        # _delta_m_version stamps the M version each held delta was built from,
        # so the residual refreshes exactly once per fire, on the first
        # correction after it (which, given the engine ordering refresh ->
        # backward -> correction, is the fire tick itself).
        self._delayed_ef_delta: dict[str, torch.Tensor] = {}
        self._m_version: dict[str, int] = {}
        self._delta_m_version: dict[str, int] = {}
        # Age of the held delta in correction ticks since its refresh (0 on the
        # fire tick). Drives the annealed weight lambda * decay**age; at
        # decay=1.0 the weight is constant and the arm-G behavior is bitwise
        # unchanged.
        self._delta_age: dict[str, int] = {}
        self.delayed_ef_refreshed = 0
        self.delayed_ef_held = 0
        self.signed_gate = str(signed_gate)
        self.signed_gate_decay = float(signed_gate_decay)
        # Learned signed-gate state (empty and never touched while
        # signed_gate="off"). _gate_rho is the per-target sign-agreement EMA
        # (absent = 0.0, so the schedule starts as use-once);
        # _gate_m_version/_gate_age mirror the delayed_ef fire-detection
        # machinery for the signed path; _gate_last_w is the most recent
        # applied weight (telemetry).
        self._gate_rho: dict[str, float] = {}
        self._gate_m_version: dict[str, int] = {}
        self._gate_age: dict[str, int] = {}
        self._gate_last_w = 0.0
        self.signed_gate_refreshed = 0
        self.signed_gate_held = 0

    def _ema_storage_device(self, grad_device):
        return torch.device("cpu") if self.ema_device == "cpu" else torch.device(grad_device)

    def ensure_anchor(self, name: str, grad: torch.Tensor) -> torch.Tensor:
        """Return M for ``name``, cold-starting or shape-resetting it to zero."""

        name = _canon(name)
        anchor = self._anchor.get(name)
        store_device = self._ema_storage_device(grad.device)
        if anchor is None or tuple(anchor.shape) != tuple(grad.shape):
            anchor = torch.zeros(grad.shape, dtype=torch.float32, device=store_device)
            if store_device.type == "cpu" and grad.device.type != "cpu":
                anchor = anchor.pin_memory()
            self._anchor[name] = anchor
        return anchor

    def anchor_on(self, name: str, device) -> torch.Tensor:
        anchor = self._anchor[_canon(name)]
        device = torch.device(device)
        return anchor.to(device) if anchor.device != device else anchor

    def update_anchor(self, name: str, g_anchor: torch.Tensor) -> torch.Tensor:
        """Update ``M <- beta*M + (1-beta)*G_anchor`` from the raw anchor grad."""

        name = _canon(name)
        self.ensure_anchor(name, g_anchor)
        anchor = self.anchor_on(name, g_anchor.device).to(torch.float32)
        g_fresh = g_anchor.to(torch.float32)
        if self.signed_gate == "learned":
            if torch.linalg.norm(anchor) > 1e-12 and tuple(anchor.shape) == tuple(g_fresh.shape):
                # Grade the direction that was being held over the closing
                # interval against this fire's fresh dense evidence. Zero
                # entries contribute 0 to the product mean, deflating rho
                # (conservative: less between-fire dose, never more).
                rho_raw = float((torch.sign(anchor) * torch.sign(g_fresh)).mean())
                rho_raw = min(max(rho_raw, 0.0), 1.0)
                previous = self._gate_rho.get(name, 0.0)
                self._gate_rho[name] = (1.0 - GATE_RHO_EMA_BETA) * previous + GATE_RHO_EMA_BETA * rho_raw
            else:
                # First-ever fire (or a shape reset zeroed M): no held
                # direction existed for this fire to grade. Restart at 0.
                self._gate_rho.pop(name, None)
        updated = self.beta_anc * anchor + (1.0 - self.beta_anc) * g_fresh
        store_device = self._ema_storage_device(g_anchor.device)
        stored = updated.to(store_device)
        if store_device.type == "cpu" and g_anchor.device.type != "cpu":
            stored = stored.pin_memory()
        self._anchor[name] = stored
        # Version stamp for delayed_ef: this is the only writer of M, and it only
        # runs on anchor fires, so "M version advanced" IS "a fire refreshed M".
        self._m_version[name] = self._m_version.get(name, 0) + 1
        return updated

    def signed_ema_matrix(self, name: str, g_comp: torch.Tensor) -> torch.Tensor:
        """Apply signed EMA (gated between fires when ``signed_gate="learned"``).

        Returns ``g_comp`` unchanged while M is cold. With the gate off the
        arithmetic is bitwise the historical signed_ema. With the gate learned,
        the fire tick (first correction after ``update_anchor``, the same
        detection the delayed_ef path uses) applies the correction at full
        strength; each held tick applies ``(1-w)*G_comp + w*G_signed`` with
        ``w = rho * decay**age``, and ``w == 0`` (rho unmeasured or graded to
        zero) skips the fp32 round trip entirely.
        """

        name = _canon(name)
        self.ensure_anchor(name, g_comp)
        anchor = self.anchor_on(name, g_comp.device).to(torch.float32)
        if torch.linalg.norm(anchor) <= 1e-12:
            self.merger_coldM_fallbacks += 1
            if self.signed_gate != "off":
                # A cold or shape-reset M invalidates the gate schedule for
                # this target; the next fire restarts it (and rho, via
                # update_anchor).
                self._gate_m_version.pop(name, None)
                self._gate_age.pop(name, None)
            return g_comp
        grad = g_comp.to(torch.float32)
        alpha = self.signed_ema_alpha
        weight = 1.0
        if self.signed_gate == "learned":
            m_version = self._m_version.get(name, 0)
            if m_version > self._gate_m_version.get(name, 0):
                # Fire tick: full strength with the fresh M, the proven
                # use-once dose.
                self._gate_m_version[name] = m_version
                self._gate_age[name] = 0
                self.signed_gate_refreshed += 1
                self._gate_last_w = 1.0
            else:
                age = self._gate_age.get(name, 0) + 1
                self._gate_age[name] = age
                self.signed_gate_held += 1
                weight = self._gate_rho.get(name, 0.0) * self.signed_gate_decay**age
                self._gate_last_w = float(weight)
                if weight == 0.0:
                    return g_comp
        corrected = alpha * grad + (1.0 - alpha) * grad.abs() * torch.sign(anchor)
        if weight != 1.0:
            corrected = (1.0 - weight) * grad + weight * corrected
        return corrected.to(g_comp.dtype)

    def gate_rho_mean(self) -> float:
        """Mean per-target sign-agreement EMA (0.0 before any measurement)."""

        if not self._gate_rho:
            return 0.0
        return float(sum(self._gate_rho.values()) / len(self._gate_rho))

    def delayed_ef_matrix(self, name: str, g_comp: torch.Tensor) -> torch.Tensor:
        """Additive anchor residual: ``G_corr = G_comp + lambda * decay**age * delta``.

        On the first correction after an anchor fire (the fire tick itself,
        given the engine ordering), ``delta = M - G_comp`` is rebuilt from the
        freshly refreshed ``M`` at ``age=0`` and persisted on the EMA-storage
        device; between fires the held ``delta`` is re-applied with its weight
        ANNEALED by ``decay**age``, where ``age`` counts correction ticks since
        the refresh. At ``lambda=1`` and ``beta_anc=0`` the fire tick returns
        exactly the anchor gradient regardless of ``decay``. ``decay=1.0`` is
        the constant-weight arm-G behavior bitwise; ``decay=0.0`` zeroes every
        held re-application, i.e. the arm-H fire-only dose at any cadence.

        Guards (never a silent grad change): ``lambda=0`` returns ``g_comp``
        bitwise; an unwarmed or shape-mismatched ``M`` returns ``g_comp`` and
        drops any held ``delta``; no held ``delta`` before the first fire
        returns ``g_comp``. All three count ``merger_coldM_fallbacks``.
        """
        lam = self.delayed_ef_lambda
        if lam == 0.0:
            return g_comp
        name = _canon(name)
        self.ensure_anchor(name, g_comp)
        anchor = self.anchor_on(name, g_comp.device).to(torch.float32)
        grad = g_comp.to(torch.float32)

        held = self._delayed_ef_delta.get(name)
        if held is not None and tuple(held.shape) != tuple(grad.shape):
            held = None
            self._delayed_ef_delta.pop(name, None)
            self._delta_m_version.pop(name, None)
            self._delta_age.pop(name, None)
        if torch.linalg.norm(anchor) <= 1e-12 or tuple(anchor.shape) != tuple(grad.shape):
            self.merger_coldM_fallbacks += 1
            self._delayed_ef_delta.pop(name, None)
            self._delta_m_version.pop(name, None)
            self._delta_age.pop(name, None)
            return g_comp

        m_version = self._m_version.get(name, 0)
        if m_version > self._delta_m_version.get(name, 0):
            delta = (anchor - grad).detach()
            store_device = self._ema_storage_device(g_comp.device)
            stored = delta.to(store_device)
            if store_device.type == "cpu" and g_comp.device.type != "cpu":
                stored = stored.pin_memory()
            self._delayed_ef_delta[name] = stored
            self._delta_m_version[name] = m_version
            self._delta_age[name] = 0
            self.delayed_ef_refreshed += 1
            weight = lam
        elif held is not None:
            age = self._delta_age.get(name, 0) + 1
            self._delta_age[name] = age
            self.delayed_ef_held += 1
            weight = lam * self.delayed_ef_decay**age
            if weight == 0.0:
                # decay=0: the held application is annealed to nothing. Skip
                # the fp32 round trip rather than add an exact zero.
                return g_comp
            delta = held.to(g_comp.device, torch.float32)
        else:
            # M is warm but no fire has been seen since the last reset: never
            # invent a residual.
            self.merger_coldM_fallbacks += 1
            return g_comp

        corrected = grad + weight * delta
        return corrected.to(g_comp.dtype)

    def blend_matrix(self, name: str, g_comp: torch.Tensor) -> torch.Tensor:
        """Norm-matched convex blend: ``G_corr = (1-eta)*G_comp + eta*(||G_comp||/||M||)*M``.

        The VALUE merger from the pre-fork menu (EXP-30 B1 scored val@50 0.7422
        at eta=0.3 against delayed_ef 0.7528 and dense 0.7839), ported verbatim:
        ``M`` is rescaled to the compressed gradient's Frobenius norm and the
        two are convexly combined, so real heterogeneous per-coordinate
        magnitudes from the dense anchor flow into Adam's moments (no sign
        transplant, no uniform-magnitude field) and the update energy is
        bounded: for orthogonal terms
        ``||G_corr|| = ||G_comp|| * sqrt((1-eta)^2 + eta^2) <= ||G_comp||``.
        Unlike ``delayed_ef`` there is NO held residual: the blend is re-formed
        against each tick's fresh ``G_comp``, and ``M`` alone carries the
        staleness between fires.

        ``eta=0`` returns ``g_comp`` bitwise (the identity limiting case).
        Cold guards: an unwarmed/shape-mismatched ``M`` or a zero-norm
        ``g_comp`` returns ``g_comp`` unchanged and counts
        ``merger_coldM_fallbacks`` (the historical version did not count it;
        this port does, for telemetry parity with the other modes).
        """
        eta = self.blend_eta
        if eta == 0.0:
            return g_comp
        name = _canon(name)
        self.ensure_anchor(name, g_comp)
        anchor = self.anchor_on(name, g_comp.device).to(torch.float32)
        grad = g_comp.to(torch.float32)
        eps = 1e-12
        anchor_norm = torch.linalg.norm(anchor)
        grad_norm = torch.linalg.norm(grad)
        if anchor_norm <= eps or grad_norm <= eps or tuple(anchor.shape) != tuple(grad.shape):
            self.merger_coldM_fallbacks += 1
            return g_comp
        scale = grad_norm / (anchor_norm + eps)
        corrected = (1.0 - eta) * grad + eta * scale * anchor
        return corrected.to(g_comp.dtype)

    def correct_matrix(self, name: str, g_comp: torch.Tensor) -> torch.Tensor:
        """Dispatch ``g_comp`` through the configured ``correction_mode``."""

        if self.correction_mode == "delayed_ef":
            return self.delayed_ef_matrix(name, g_comp)
        if self.correction_mode == "blend":
            return self.blend_matrix(name, g_comp)
        return self.signed_ema_matrix(name, g_comp)

    def relative_change(self, g_comp: torch.Tensor, g_corr: torch.Tensor) -> float:
        """Return ``||G_corr-G_comp|| / ||G_comp||`` for diagnostics."""

        base = g_comp.to(torch.float32)
        corrected = g_corr.to(torch.float32)
        denominator = torch.linalg.norm(base)
        if denominator <= 0:
            return 0.0
        return (torch.linalg.norm(corrected - base) / denominator).item()


def apply_spectral_correction_to_params(
    named_params,
    *,
    spectral: SpectralFilter,
    target_substrs,
    target_scope: str = "decoder_matrices",
    max_targets: int,
    state,
    discovery_meta: dict,
    full_grad_of,
    writeback,
) -> int:
    """Apply the configured correction mode to selected logical gradients and write them back."""

    instrumented = bool(state.fsdp_grad_repr)
    corrected_count = 0
    spectral.merger_coldM_fallbacks = 0

    for name, parameter in named_params:
        grad = getattr(parameter, "grad", None)
        if grad is None:
            continue
        if not is_spectral_target(
            name,
            parameter,
            target_substrs=target_substrs,
            target_scope=target_scope,
        ):
            continue
        if max_targets >= 0 and corrected_count >= max_targets:
            break

        full_grad, container_meta = full_grad_of(grad)
        if not full_grad.is_floating_point():
            continue
        logical_shape = tuple(full_grad.shape)

        if not instrumented:
            representation = {"target_name": name, "logical_2d_shape": str(logical_shape)}
            representation.update(container_meta)
            representation.update(discovery_meta)
            state.fsdp_grad_repr = representation
            logger.warning("comm_eff FSDP grad-repr discovery: %s", representation)
            if spectral.diagnostics:
                print(f"[comm_eff][FSDP-DISCOVERY] {representation}", flush=True)
            instrumented = True

        corrected = spectral.correct_matrix(name, full_grad)
        if spectral.diagnostics:
            relative_change = spectral.relative_change(full_grad, corrected)
            state.spectral_rel_change[name] = relative_change
            print(
                f"[comm_eff][{spectral.correction_mode}] {name} rel_change={relative_change:.6f} "
                f"shape={logical_shape} grad_type={container_meta.get('grad_container_type')}",
                flush=True,
            )
        with torch.no_grad():
            writeback(grad, corrected)
        corrected_count += 1
        state.spectral_corrections += 1

    state.merger_coldM_fallbacks = int(spectral.merger_coldM_fallbacks)
    # Cumulative delayed_ef counters (0 forever under signed_ema): refreshed
    # should factor as n_targets x anchor_fires, held as n_targets x (correction
    # ticks between fires). Their sum plus coldM fallbacks accounts for every
    # correction call.
    state.delayed_ef_refreshed = int(spectral.delayed_ef_refreshed)
    state.delayed_ef_held = int(spectral.delayed_ef_held)
    # Cumulative signed-gate counters (0 forever unless signed_gate="learned"):
    # refreshed factors as n_targets x anchor_fires, held as n_targets x held
    # correction ticks. rho_mean and w_last are the learned schedule itself,
    # surfaced so WandB shows what the gate learned over the run.
    state.signed_gate_refreshed = int(spectral.signed_gate_refreshed)
    state.signed_gate_held = int(spectral.signed_gate_held)
    state.signed_gate_rho_mean = spectral.gate_rho_mean()
    state.signed_gate_w_last = float(spectral._gate_last_w)
    if spectral.diagnostics and corrected_count:
        print(
            f"[comm_eff][{spectral.correction_mode}] alpha={spectral.signed_ema_alpha} "
            f"lambda={spectral.delayed_ef_lambda} "
            f"corrected={corrected_count} cold_M={spectral.merger_coldM_fallbacks} "
            f"delta_refreshed={spectral.delayed_ef_refreshed} delta_held={spectral.delayed_ef_held}",
            flush=True,
        )
    if corrected_count:
        logger.info("comm_eff: %s correction applied to %d tensors", spectral.correction_mode, corrected_count)
    return corrected_count
