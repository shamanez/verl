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

"""Pipeline-boundary activation masking (Algorithm A, actor-train only).

This implements the *first circuit* of the two-circuit compression method: a
deterministic pseudo-random-function (PRF) Bernoulli mask applied **in-graph**
to the hidden-state output of selected pipeline-boundary decoder blocks during
the actor train forward/backward pass.

Key properties (each is load-bearing and unit-tested in
``tests/workers/comm_eff/test_activation_mask.py``):

* **Form:** ``h_tilde = h * mask`` element-wise, in-graph. There is **no**
  ``1 / (1 - p)`` forward rescale — the direct product is written. Rescaling at
  ``p=0.95`` destabilises bf16; we do not claim unbiasedness for the no-rescale
  port.

* **``p`` is the masked fraction.** ``mask`` is ``0`` (zeroed) with probability
  ``p`` and ``1`` (kept) with probability ``1 - p``. So the measured
  *mask ratio* (fraction of zeroed elements) tracks the configured ``p``. This
  matches the EXP-5 success criterion ``comm_eff/mask_ratio ≈ p ± 0.02``.

* **PRF key, not activation values.** The mask is drawn from a seeded PRF whose
  key is ``(boundary id, global optimizer step, optimizer-substep / microbatch
  identity, sequence-shard identity, hidden size, base run seed)``. The mask
  **never** depends on the activation tensor values — only on its shape and the
  key. So the same key reproduces the same mask across ranks and re-runs, and
  two different activation tensors with the same key/shape get the same mask.

* **Boundary layers from ``model.config``.** ``num_layers`` and ``hidden_size``
  are read from the live module (or its config), never hardcoded. The block
  indices are partitioned into ``pp_size`` contiguous shards and the **last
  block of each shard except the final shard** is masked. For ``L=16,
  pp_size=8`` this is ``[1, 3, 5, 7, 9, 11, 13]``.

* **Top-k masking is forbidden** — only random PRF masking. Top-k introduces
  structured bias the later spectral filter cannot remove.

* **Hooks are train-only and ephemeral.** They are registered on entry to the
  actor train forward/backward and removed on exit, so a later log-prob / ref /
  ``infer_batch`` / validation / checkpoint forward on the same module sees no
  masking.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch
import torch.nn as nn

# The set of execution-path tags on which masking is permitted to fire.
# Imported from the state module (cheap, no torch import there) so the assert
# below and the state stay in lockstep. EXP-5 → EXP-12 the only eligible tag is
# ``train``; EXP-9 widens eligibility to ``{train, old_logprob}`` *iff*
# ``comm_eff.mask.mask_recompute=true`` via ``mask_eligible_tags(state)``,
# which is evaluated at hook-fire time (per call) so flipping the knob does
# not require restarting the worker. ``None`` (the anchor pass / GUARD 5) is
# never eligible — the anchor runs unmasked unconditionally.
from verl.workers.comm_eff.state import (
    MASK_ELIGIBLE_TAGS,
    TRAIN_TAG,
    mask_eligible_tags,
)

logger = logging.getLogger(__name__)

__all__ = [
    "decoder_boundary_indices",
    "find_decoder_layers",
    "prf_mask",
    "ActivationMasker",
]

# A large odd 64-bit constant mixed into the PRF seed so the mask stream is well
# separated from any other RNG stream in the run. Splitmix64-style finaliser.
_PRF_GOLDEN = 0x9E3779B97F4A7C15
_U64 = (1 << 64) - 1


def decoder_boundary_indices(num_layers: int, pp_size: int) -> list[int]:
    """Return pipeline-boundary block indices for masking.

    The ``num_layers`` decoder blocks are partitioned into ``pp_size``
    contiguous shards (the same near-even split a real pipeline-parallel layout
    would use). The boundary of shard ``i`` is its **last** block index; we mask
    every shard boundary **except the final shard's** (the final shard's last
    block is the model's last decoder block, which feeds the LM head — masking
    it is masking the output, not a pipeline boundary).

    For ``L=16, pp_size=8`` the shards are
    ``[0,1] [2,3] [4,5] [6,7] [8,9] [10,11] [12,13] [14,15]`` whose last indices
    are ``[1,3,5,7,9,11,13,15]``; dropping the final shard's ``15`` gives
    ``[1,3,5,7,9,11,13]``.

    Args:
        num_layers: Number of decoder blocks ``L`` (from ``model.config``).
        pp_size: Logical pipeline-shard count (a config knob, not a real split).

    Returns:
        Sorted list of boundary block indices, length ``min(pp_size, L) - 1``.
    """
    if num_layers < 1:
        raise ValueError(f"num_layers must be >= 1; got {num_layers}")
    if pp_size < 1:
        raise ValueError(f"pp_size must be >= 1; got {pp_size}")
    # Cap pp_size at num_layers so we never produce an empty shard.
    eff_pp = min(pp_size, num_layers)
    if eff_pp == 1:
        return []

    # Contiguous near-even partition: shard i covers [starts[i], starts[i+1]).
    # last index of shard i is starts[i+1] - 1. We take shards 0..eff_pp-2
    # (every shard except the final one).
    base = num_layers // eff_pp
    rem = num_layers % eff_pp
    boundaries: list[int] = []
    cursor = 0
    for i in range(eff_pp):
        shard_len = base + (1 if i < rem else 0)
        cursor += shard_len
        last_idx = cursor - 1
        if i < eff_pp - 1:  # skip the final shard's boundary
            boundaries.append(last_idx)
    return boundaries


def find_decoder_layers(module: nn.Module) -> Optional[nn.ModuleList]:
    """Locate the decoder-block ``nn.ModuleList`` inside a (possibly wrapped) model.

    Walks the module tree (FSDP / DDP / PEFT wrappers are transparent because we
    use ``named_modules``) and returns the first ``nn.ModuleList`` whose entries
    look like transformer decoder blocks (heuristic: the attribute is named
    ``layers`` / ``h`` / ``blocks`` and contains >1 modules). This avoids
    hardcoding a model class so the masker works for any HF causal-LM verl
    trains.

    Returns ``None`` if no decoder-block list is found (the masker then no-ops
    rather than crashing — surfaced via a warning so the analyst sees it).
    """
    candidates: list[tuple[str, nn.ModuleList]] = []
    for name, sub in module.named_modules():
        if isinstance(sub, nn.ModuleList) and len(sub) > 1:
            leaf = name.rsplit(".", 1)[-1]
            if leaf in ("layers", "h", "blocks", "decoder"):
                candidates.append((name, sub))
    if not candidates:
        # fall back to the longest ModuleList of >1 modules
        all_lists = [
            (name, sub)
            for name, sub in module.named_modules()
            if isinstance(sub, nn.ModuleList) and len(sub) > 1
        ]
        if not all_lists:
            return None
        all_lists.sort(key=lambda kv: len(kv[1]), reverse=True)
        return all_lists[0][1]
    # Prefer the longest matching list (the decoder stack dominates layer count).
    candidates.sort(key=lambda kv: len(kv[1]), reverse=True)
    return candidates[0][1]


def _splitmix64(x: int) -> int:
    """One round of the splitmix64 finaliser. Deterministic, well-distributed."""
    x = (x + _PRF_GOLDEN) & _U64
    z = x
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _U64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _U64
    z = z ^ (z >> 31)
    return z & _U64


def _derive_seed(key: tuple[int, ...]) -> int:
    """Fold a PRF key tuple into a single 64-bit seed.

    Pure function of the key only — never of activation values. Order-sensitive
    so distinct key components do not alias.
    """
    acc = 0
    for component in key:
        # Mix each component in turn; cast to a non-negative 64-bit int first.
        acc = _splitmix64(acc ^ (int(component) & _U64))
    return acc & _U64


def prf_mask(
    shape: tuple[int, ...],
    key: tuple[int, ...],
    p: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build a deterministic Bernoulli keep/zero mask from a PRF key.

    The mask is ``0`` (zeroed) with probability ``p`` and ``1`` (kept) with
    probability ``1 - p``, drawn from a ``torch.Generator`` seeded purely by
    ``key`` (folded via splitmix64). Identical ``key`` + ``shape`` => identical
    mask, on any rank, in any process, independent of the activation values.

    Args:
        shape: Shape of the activation tensor to mask.
        key: PRF key tuple (boundary id, global step, substep id, seq-shard id,
            hidden size, base seed). Must NOT include activation values.
        p: Masked fraction in ``[0, 1]`` (probability an element is zeroed).
        device: Device for the returned mask.
        dtype: Dtype for the returned mask (matches the activation).

    Returns:
        A ``mask`` tensor of ``shape`` with entries in ``{0, 1}``.
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"mask p must be in [0, 1]; got {p}")
    seed = _derive_seed(key)
    # torch.Generator takes a signed int64; fold the high bit in.
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed & 0x7FFFFFFFFFFFFFFF)
    # Draw uniform on CPU for cross-device reproducibility, then move. keep iff
    # u >= p  =>  P(zero) = P(u < p) = p, so the masked fraction is exactly p in
    # expectation, independent of device RNG implementation.
    u = torch.rand(shape, generator=gen, dtype=torch.float32)
    keep = (u >= p).to(dtype=dtype)
    return keep.to(device=device, non_blocking=True)


class ActivationMasker:
    """Registers/clears in-graph activation-mask forward hooks on boundary blocks.

    One instance is owned by the engine. ``register(module)`` installs a forward
    hook on each boundary decoder block; ``unregister()`` removes them. The hooks
    must be live **only** during the actor train forward/backward — the engine
    registers on entry to ``forward_backward_batch`` (train) and unregisters on
    exit, so log-prob / ref / infer / validation / checkpoint forwards never see
    a mask.

    The PRF key per hook fire is composed from:
      * the boundary block index (stable per hook),
      * ``global_step`` (trainer optimizer step),
      * ``substep`` (optimizer-substep / microbatch identity within the step),
      * a sequence-shard id (0 when no SP; set by the engine when present),
      * ``hidden_size`` (last dim of the activation),
      * ``base_seed`` (``comm_eff.mask.seed``).

    ``global_step`` / ``substep`` / ``seq_shard`` are set by the engine via
    ``set_context(...)`` before each forward so the same rollout batch reused
    over multiple PPO mini-batches gets distinct masks per substep.
    """

    def __init__(self, *, p: float, base_seed: int, pp_size: int, state: Any = None):
        self.p = float(p)
        self.base_seed = int(base_seed)
        self.pp_size = int(pp_size)
        self._state = state  # CommEffState, for the mask_applications counter
        self._handles: list[Any] = []
        self._boundary_set: set[int] = set()
        self.boundary_indices: list[int] = []
        # Per-forward context, set by the engine before forward_backward.
        self._global_step = 0
        self._substep = 0
        self._seq_shard = 0
        # Last-measured masked fraction per boundary, surfaced as comm_eff/mask_ratio.
        self.last_mask_ratio: dict[int, float] = {}

    def set_context(self, *, global_step: int, substep: int, seq_shard: int = 0) -> None:
        """Set the PRF-key context for the next forward pass."""
        self._global_step = int(global_step)
        self._substep = int(substep)
        self._seq_shard = int(seq_shard)

    def _make_hook(self, layer_idx: int):
        masker = self

        def _hook(_mod: nn.Module, _inputs: tuple, output: Any):
            # HF decoder blocks return either a Tensor or a tuple whose first
            # element is the hidden state. Mask the hidden state in-graph.
            if isinstance(output, tuple):
                h = output[0]
            else:
                h = output
            if not torch.is_tensor(h):
                return output
            hidden_size = h.shape[-1]
            key = (
                layer_idx,
                masker._global_step,
                masker._substep,
                masker._seq_shard,
                hidden_size,
                masker.base_seed,
            )
            # EXP-6 contamination guard (EXP-9 extension). The mask must NEVER
            # fire outside the set of eligible paths. The default eligibility is
            # ``{train}`` (EXP-5 → EXP-12); EXP-9 widens it to
            # ``{train, old_logprob}`` when ``state.mask.mask_recompute=true``
            # so the fast (masked) circuit covers BOTH gradient-feeding forwards
            # in pipeline-parallel RL. ``None`` (anchor pass) is NEVER eligible —
            # the anchor stays unmasked (GUARD 5). The engine only registers
            # these hooks when ``state.mask_active`` is set (around
            # ``update_actor`` and, with mask_recompute, around
            # ``compute_log_prob``), but a future caller that mis-sets the flag
            # or a path-tag mismatch would silently corrupt
            # rollout / ref / val / infer / checkpoint forwards. We turn that
            # into a LOUD failure: assert the tag is in the eligible set.
            # Disabled under ``python -O``, but the per-path counter below
            # still records any leak as a falsifier.
            state = masker._state
            if state is not None and hasattr(state, "path_tag"):
                tag = state.path_tag
                eligible = mask_eligible_tags(state)
                assert tag in eligible, (
                    "comm_eff activation mask fired on an ineligible path "
                    f"(path_tag={tag!r}, eligible={sorted(eligible)}); masking "
                    "is confined to the actor-train forward/backward (and, with "
                    "comm_eff.mask.mask_recompute=true, the old-logprob recompute). "
                    "Any other path (rollout / ref_logprob / val / infer / ckpt) "
                    "or the anchor pass (path_tag=None) is contamination of the "
                    "RL measurement machinery and is a hard falsifier."
                )
            mask = prf_mask(tuple(h.shape), key, masker.p, device=h.device, dtype=h.dtype)
            # h_tilde = h * mask, in-graph (no 1/(1-p) rescale). The multiply is
            # tracked by autograd so the masked gradient flows to the optimizer.
            h_tilde = h * mask
            # Instrumentation (does not affect the graph): measured masked fraction.
            with torch.no_grad():
                masker.last_mask_ratio[layer_idx] = float(1.0 - mask.mean().item())
            if state is not None:
                # Records both the aggregate and the per-path counter; the
                # latter is what the analyst greps by KEY PREFIX. Falls back to
                # the EXP-5 aggregate-only shape if note_mask_application is
                # absent (e.g. a _FakeState in a unit test).
                if hasattr(state, "note_mask_application"):
                    state.note_mask_application()
                else:
                    state.mask_applications += 1
            if isinstance(output, tuple):
                return (h_tilde,) + tuple(output[1:])
            return h_tilde

        return _hook

    def register(self, module: nn.Module) -> None:
        """Install forward hooks on the boundary decoder blocks of ``module``.

        Idempotent guard: if hooks are already registered this is a no-op (the
        engine pairs register/unregister, but a defensive guard avoids double
        registration leaking a mask onto a later pass).
        """
        if self._handles:
            return
        layers = find_decoder_layers(module)
        if layers is None:
            logger.warning(
                "comm_eff.activation_mask: could not locate decoder layers on %s; "
                "no mask hooks registered (masking is a no-op this pass)",
                type(module).__name__,
            )
            return
        num_layers = len(layers)
        self.boundary_indices = decoder_boundary_indices(num_layers, self.pp_size)
        self._boundary_set = set(self.boundary_indices)
        for idx in self.boundary_indices:
            handle = layers[idx].register_forward_hook(self._make_hook(idx))
            self._handles.append(handle)
        logger.info(
            "comm_eff.activation_mask: registered mask hooks on boundaries %s "
            "(L=%d, pp_size=%d, p=%.4f)",
            self.boundary_indices,
            num_layers,
            self.pp_size,
            self.p,
        )

    def unregister(self) -> None:
        """Remove all mask hooks. Must be called on exit of the train forward."""
        for handle in self._handles:
            handle.remove()
        self._handles = []

    @property
    def is_registered(self) -> bool:
        return bool(self._handles)
