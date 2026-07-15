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

"""Signed-EMA correction for PowerSGD-compressed actor gradients.

The dense anchor supplies a per-parameter gradient EMA ``M``. Before the
optimizer step, the merger keeps the compressed gradient's magnitude
and blends its sign with ``sign(M)``::

    G_corr = alpha * G_comp + (1 - alpha) * abs(G_comp) * sign(M)

An unwarmed or shape-mismatched anchor entry is a strict no-op for that tensor.
FSDP extraction and writeback remain engine responsibilities; this module only
operates on logical full tensors.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

__all__ = ["SpectralFilter", "apply_spectral_correction_to_params", "is_spectral_target"]

SPECTRAL_TARGET_SCOPES = ("decoder_matrices", "all_floating")


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
        diagnostics: bool = False,
    ):
        if not 0.0 <= float(beta_anc) <= 1.0:
            raise ValueError(f"beta_anc must be in [0, 1]; got {beta_anc}")
        if ema_device not in ("gpu", "cpu"):
            raise ValueError(f"ema_device must be one of (gpu, cpu); got {ema_device!r}")
        if not 0.0 <= float(signed_ema_alpha) <= 1.0:
            raise ValueError(f"signed_ema_alpha must be in [0, 1]; got {signed_ema_alpha}")

        self.beta_anc = float(beta_anc)
        self.ema_device = str(ema_device)
        self.signed_ema_alpha = float(signed_ema_alpha)
        self.diagnostics = bool(diagnostics)
        self.merger_coldM_fallbacks = 0
        self._anchor: dict[str, torch.Tensor] = {}

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
        updated = self.beta_anc * anchor + (1.0 - self.beta_anc) * g_anchor.to(torch.float32)
        store_device = self._ema_storage_device(g_anchor.device)
        stored = updated.to(store_device)
        if store_device.type == "cpu" and g_anchor.device.type != "cpu":
            stored = stored.pin_memory()
        self._anchor[name] = stored
        return updated

    def signed_ema_matrix(self, name: str, g_comp: torch.Tensor) -> torch.Tensor:
        """Apply signed EMA, or return ``g_comp`` unchanged while M is cold."""

        name = _canon(name)
        self.ensure_anchor(name, g_comp)
        anchor = self.anchor_on(name, g_comp.device).to(torch.float32)
        if torch.linalg.norm(anchor) <= 1e-12:
            self.merger_coldM_fallbacks += 1
            return g_comp
        grad = g_comp.to(torch.float32)
        alpha = self.signed_ema_alpha
        corrected = alpha * grad + (1.0 - alpha) * grad.abs() * torch.sign(anchor)
        return corrected.to(g_comp.dtype)

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
    """Apply signed EMA to selected logical gradients and write them back."""

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

        corrected = spectral.signed_ema_matrix(name, full_grad)
        if spectral.diagnostics:
            relative_change = spectral.relative_change(full_grad, corrected)
            state.spectral_rel_change[name] = relative_change
            print(
                f"[comm_eff][signed_ema] {name} rel_change={relative_change:.6f} "
                f"shape={logical_shape} grad_type={container_meta.get('grad_container_type')}",
                flush=True,
            )
        with torch.no_grad():
            writeback(grad, corrected)
        corrected_count += 1
        state.spectral_corrections += 1

    state.merger_coldM_fallbacks = int(spectral.merger_coldM_fallbacks)
    if spectral.diagnostics and corrected_count:
        print(
            f"[comm_eff][signed_ema] alpha={spectral.signed_ema_alpha} "
            f"corrected={corrected_count} cold_M={spectral.merger_coldM_fallbacks}",
            flush=True,
        )
    if corrected_count:
        logger.info("comm_eff: signed-EMA correction applied to %d tensors", corrected_count)
    return corrected_count
