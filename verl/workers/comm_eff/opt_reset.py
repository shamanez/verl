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

"""Anchor-sourced optimizer-moment reset for the fast circuit.

Under a lossy activation codec the fast AdamW moments are built from
codec-noised gradients, so compression bias can accumulate in the optimizer
state itself. :class:`AnchorOptMoments` maintains parallel fp32 CPU moment
EMAs from the anchor circuit's clean, DP-averaged dense replay gradients
(fed the same tensors that feed the signed EMA ``M``, keyed the same way),
and :func:`reset_optimizer_moments` overwrites the fast ``exp_avg`` /
``exp_avg_sq`` with them (``mode='anchor_moments'``, optionally norm-matched
via ``rho``) or zeroes them (``mode='zero'``). ``state['step']`` is never
touched. FSDP shard extraction and writeback remain engine responsibilities;
this module only operates on logical full tensors plus engine-supplied
callables.
"""

from __future__ import annotations

import math
from typing import Optional

import torch

from verl.workers.comm_eff.spectral_filter import _canon

__all__ = ["AnchorOptMoments", "reset_optimizer_moments"]


class AnchorOptMoments:
    """Per-target fp32 CPU AdamW-style moment EMAs over the anchor gradients.

    Storage mirrors ``SpectralFilter._anchor`` exactly: keys are canonical
    parameter names, tensors are FULL logical shapes, fp32, on CPU
    (pin-memory'd when fed from a CUDA gradient).
    """

    def __init__(self, *, beta1: float = 0.8, beta2: float = 0.95):
        if not 0.0 < float(beta1) < 1.0:
            raise ValueError(f"beta1 must be in (0, 1); got {beta1}")
        if not 0.0 < float(beta2) < 1.0:
            raise ValueError(f"beta2 must be in (0, 1); got {beta2}")
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        # Number of anchor fires folded in; 0 means the reset must skip.
        self.fires = 0
        self._m: dict[str, torch.Tensor] = {}
        self._v: dict[str, torch.Tensor] = {}

    def _store(self, tensor: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        stored = tensor.to(torch.device("cpu"))
        if source.device.type != "cpu":
            stored = stored.pin_memory()
        return stored

    def update(self, grads: dict) -> None:
        """Fold one anchor fire: ``m <- b1*m + (1-b1)*G``, ``v <- b2*v + (1-b2)*G*G``."""

        for name, g_anchor in grads.items():
            name = _canon(name)
            g32 = g_anchor.detach().to(torch.float32)
            m_prev = self._m.get(name)
            if m_prev is None or tuple(m_prev.shape) != tuple(g32.shape):
                m_prev = torch.zeros(g32.shape, dtype=torch.float32, device="cpu")
                v_prev = torch.zeros(g32.shape, dtype=torch.float32, device="cpu")
            else:
                v_prev = self._v[name]
            m_new = self.beta1 * m_prev.to(g32.device) + (1.0 - self.beta1) * g32
            v_new = self.beta2 * v_prev.to(g32.device) + (1.0 - self.beta2) * g32 * g32
            self._m[name] = self._store(m_new, g32)
            self._v[name] = self._store(v_new, g32)
        self.fires += 1

    def get(self, name: str):
        """Return ``(m, v)`` full fp32 CPU tensors for ``name``, or ``None``."""

        name = _canon(name)
        m = self._m.get(name)
        if m is None:
            return None
        return m, self._v[name]

    def m_sq_sum(self) -> float:
        """Global ``sum(m^2)`` over every anchor exp_avg tensor.

        The moments are built from DP-reduced gradients, so they are already
        global and bit-identical across ranks; no collective is needed here.
        """

        total = 0.0
        for m in self._m.values():
            m32 = m.to(torch.float32)
            total += float(torch.sum(m32 * m32).item())
        return total


def _default_writeback(state_tensor: torch.Tensor, param: torch.Tensor, full: torch.Tensor) -> None:
    del param
    state_tensor.copy_(full.reshape(state_tensor.shape).to(device=state_tensor.device, dtype=state_tensor.dtype))


def _default_sq_sum_of(state_tensor: torch.Tensor) -> float:
    t32 = state_tensor.detach().to(torch.float32)
    return float(torch.sum(t32 * t32).item())


def reset_optimizer_moments(
    optimizer: torch.optim.Optimizer,
    named_params,
    *,
    moments: AnchorOptMoments,
    mode: str,
    scale_match: bool,
    writeback=None,
    sq_sum_of=None,
    reduce_sq_sum=None,
) -> Optional[float]:
    """Overwrite the fast AdamW moments from the anchor-maintained moments.

    ``named_params`` is an iterable of ``(name, param)`` over the live module
    (FSDP wrap infixes are stripped via ``_canon`` when looking up moments).
    The three callables are the engine's sharding adapters; the defaults are
    correct for a plain (non-FSDP, single-rank) module:

    * ``writeback(state_tensor, param, full_fp32)`` writes a full logical fp32
      tensor into a (possibly shard-shaped) optimizer state tensor in place;
    * ``sq_sum_of(state_tensor)`` returns the local ``sum(t^2)`` as a float;
    * ``reduce_sq_sum(local)`` sums the local value across the sharding group
      (identity when single-rank), the same all-reduce(SUM)-of-local-sq-sums
      pattern the grad-norm path uses.

    ``mode='anchor_moments'`` writes ``rho * m_anc`` into ``exp_avg`` and
    ``rho^2 * v_anc`` (clamped >= 0) into ``exp_avg_sq`` and returns ``rho``;
    params with no anchor entry are skipped. ``mode='zero'`` zeroes both
    moments of every param and returns ``None``. ``state['step']`` is never
    touched in either mode.
    """

    if writeback is None:
        writeback = _default_writeback
    if sq_sum_of is None:
        sq_sum_of = _default_sq_sum_of
    if reduce_sq_sum is None:

        def reduce_sq_sum(local: float) -> float:
            return local

    if mode == "zero":
        with torch.no_grad():
            for param_state in optimizer.state.values():
                for key in ("exp_avg", "exp_avg_sq"):
                    tensor = param_state.get(key)
                    if isinstance(tensor, torch.Tensor):
                        tensor.zero_()
        return None
    if mode != "anchor_moments":
        raise ValueError(f"opt_reset mode must be one of ('anchor_moments', 'zero'); got {mode!r}")

    rho = 1.0
    if scale_match:
        # Shards are disjoint slices of the logical tensors, so the local
        # sum of squares all-reduced(SUM) over the sharding group is the exact
        # global L2^2 (the same geometry the grad-norm/clip path relies on).
        local_sq = 0.0
        for param_state in optimizer.state.values():
            tensor = param_state.get("exp_avg")
            if isinstance(tensor, torch.Tensor):
                local_sq += sq_sum_of(tensor)
        fast_norm = math.sqrt(max(reduce_sq_sum(local_sq), 0.0))
        anchor_norm = math.sqrt(max(moments.m_sq_sum(), 0.0))
        rho = fast_norm / (anchor_norm + 1e-12)

    with torch.no_grad():
        for name, param in named_params:
            param_state = optimizer.state.get(param)
            if not param_state:
                continue
            exp_avg = param_state.get("exp_avg")
            exp_avg_sq = param_state.get("exp_avg_sq")
            if not isinstance(exp_avg, torch.Tensor) or not isinstance(exp_avg_sq, torch.Tensor):
                continue
            pair = moments.get(name)
            if pair is None:
                continue
            m_full, v_full = pair
            writeback(exp_avg, param, rho * m_full.to(torch.float32))
            writeback(exp_avg_sq, param, (rho * rho * v_full.to(torch.float32)).clamp_min(0.0))
    return rho
