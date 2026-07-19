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

"""Pipeline-boundary activation masking (actor-train only).

A deterministic per-(token, dimension) PRF Bernoulli mask applied in-graph
(``h_tilde = h * mask``) to the hidden-state output of selected pipeline-boundary
decoder blocks. Each token independently keeps ``round((1-p)*H)`` of its ``H``
dims.

The mask is keyed on each token's stable identity ``(sample_id, position_id)``,
not its position inside a packed micro-batch, so it is packing-invariant: a token
gets the same mask in the old-logprob recompute and the actor-train forward and
across every PPO mini-batch / epoch of one ``global_step``. ``p`` is the masked
(zeroed) fraction. The draw uses a counter-based splitmix64 PRF (no Generator
state), so it is identical on CPU/GPU, across ranks, and under
forward-recomputation (gradient checkpointing). Hooks are registered only for the
masked forward and removed on exit. Top-k masking is forbidden (random only).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch
import torch.nn as nn

from verl.workers.comm_eff.state import (
    mask_eligible_tags,
)

logger = logging.getLogger(__name__)

__all__ = [
    "decoder_boundary_indices",
    "find_decoder_layers",
    "prf_token_mask",
    "ActivationMasker",
]

# Splitmix64 finaliser constants.
_PRF_GOLDEN = 0x9E3779B97F4A7C15
_PRF_MIX1 = 0xBF58476D1CE4E5B9
_PRF_MIX2 = 0x94D049BB133111EB
_U64 = (1 << 64) - 1
# Uniform is the top 53 bits of the hash (exact in a float64 mantissa / int64).
_PRF_2POW53 = 1 << 53


def decoder_boundary_indices(num_layers: int, pp_size: int) -> list[int]:
    """Return the pipeline-boundary block indices to mask.

    The ``num_layers`` decoder blocks are split into ``pp_size`` contiguous
    shards; the last block of every shard *except the final shard* is a boundary.
    ``L=16, pp_size=8`` -> ``[1, 3, 5, 7, 9, 11, 13]``.
    """
    if num_layers < 1:
        raise ValueError(f"num_layers must be >= 1; got {num_layers}")
    if pp_size < 1:
        raise ValueError(f"pp_size must be >= 1; got {pp_size}")
    # Cap pp_size at num_layers so we never produce an empty shard.
    eff_pp = min(pp_size, num_layers)
    if eff_pp == 1:
        return []

    base = num_layers // eff_pp
    rem = num_layers % eff_pp
    boundaries: list[int] = []
    cursor = 0
    for i in range(eff_pp):
        shard_len = base + (1 if i < rem else 0)
        cursor += shard_len
        if i < eff_pp - 1:  # skip the final shard's boundary
            boundaries.append(cursor - 1)
    return boundaries


def find_decoder_layers(module: nn.Module) -> Optional[nn.ModuleList]:
    """Locate the decoder-block ``nn.ModuleList`` inside a (possibly wrapped) model.

    Returns the longest ``nn.ModuleList`` named ``layers``/``h``/``blocks``/
    ``decoder`` (falling back to the longest list of >1 modules), so the masker
    works for any HF causal-LM without hardcoding a model class. ``None`` if none
    is found (the masker then no-ops with a warning).
    """
    candidates: list[tuple[str, nn.ModuleList]] = []
    for name, sub in module.named_modules():
        if isinstance(sub, nn.ModuleList) and len(sub) > 1:
            leaf = name.rsplit(".", 1)[-1]
            if leaf in ("layers", "h", "blocks", "decoder"):
                candidates.append((name, sub))
    if not candidates:
        all_lists = [
            (name, sub) for name, sub in module.named_modules() if isinstance(sub, nn.ModuleList) and len(sub) > 1
        ]
        if not all_lists:
            return None
        all_lists.sort(key=lambda kv: len(kv[1]), reverse=True)
        return all_lists[0][1]
    candidates.sort(key=lambda kv: len(kv[1]), reverse=True)
    return candidates[0][1]


def _splitmix64(x: int) -> int:
    """One scalar round of the splitmix64 finaliser."""
    x = (x + _PRF_GOLDEN) & _U64
    z = x
    z = ((z ^ (z >> 30)) * _PRF_MIX1) & _U64
    z = ((z ^ (z >> 27)) * _PRF_MIX2) & _U64
    z = z ^ (z >> 31)
    return z & _U64


def _derive_seed(key: tuple[int, ...]) -> int:
    """Fold a PRF key tuple into a 64-bit seed (order-sensitive, value-free)."""
    acc = 0
    for component in key:
        acc = _splitmix64(acc ^ (int(component) & _U64))
    return acc & _U64


def _u64_to_i64(value: int) -> int:
    """Reinterpret a uint64 value as the signed int64 with the same bits.

    torch has no uint64; the tensor PRF runs on int64, whose two's-complement
    arithmetic matches uint64 mod 2**64.
    """
    return value - (1 << 64) if value >= (1 << 63) else value


_GOLDEN_I64 = _u64_to_i64(_PRF_GOLDEN)
_MIX1_I64 = _u64_to_i64(_PRF_MIX1)
_MIX2_I64 = _u64_to_i64(_PRF_MIX2)


def _logical_rshift(x: torch.Tensor, n: int) -> torch.Tensor:
    """Unsigned right shift of a signed int64 tensor (clear the sign-extended bits)."""
    return (x >> n) & ((1 << (64 - n)) - 1)


def _splitmix64_tensor(x: torch.Tensor) -> torch.Tensor:
    """Vectorized splitmix64 finaliser, bit-identical to :func:`_splitmix64`."""
    x = x + _GOLDEN_I64
    x = x ^ _logical_rshift(x, 30)
    x = x * _MIX1_I64
    x = x ^ _logical_rshift(x, 27)
    x = x * _MIX2_I64
    x = x ^ _logical_rshift(x, 31)
    return x


def prf_token_mask(
    sample_ids: torch.Tensor,
    position_ids: torch.Tensor,
    *,
    layer_idx: int,
    global_step: int,
    base_seed: int,
    hidden_size: int,
    p: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Deterministic per-(token, dim) Bernoulli keep/zero mask.

    Entry ``(t, j)`` is keyed on ``(base_seed, layer_idx, global_step,
    sample_ids[t], position_ids[t], j)`` and kept iff its PRF draw ``>= p``, so
    the zeroed fraction tracks ``p``.

    Args:
        sample_ids: ``(N,)`` per-token stable sample identity.
        position_ids: ``(N,)`` per-token position within its sequence.
        hidden_size: ``H`` channels the mask is drawn over.
        p: Masked fraction in ``[0, 1]``.

    Returns:
        ``(N, hidden_size)`` mask of ``{0, 1}`` in ``dtype`` on ``device``.
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"mask p must be in [0, 1]; got {p}")
    sid = sample_ids.reshape(-1).to(device=device, dtype=torch.int64)
    pos = position_ids.reshape(-1).to(device=device, dtype=torch.int64)
    if sid.numel() != pos.numel():
        raise ValueError(f"sample_ids and position_ids length mismatch: {sid.numel()} vs {pos.numel()}")

    # Fold the scalar prefix, then the per-token ids, then the channel index.
    # Left-fold equivalence makes this bit-identical to _derive_seed over the
    # full key tuple per (token, channel).
    prefix = _u64_to_i64(_derive_seed((base_seed, layer_idx, global_step)))
    acc = _splitmix64_tensor(sid ^ prefix)  # fold sample_id   -> (N,)
    acc = _splitmix64_tensor(acc ^ pos)  # fold position_id -> (N,)
    channel = torch.arange(hidden_size, device=device, dtype=torch.int64)
    h = _splitmix64_tensor(acc.unsqueeze(1) ^ channel.unsqueeze(0))  # (N, H)

    # keep iff (top-53-bit uniform) >= p, done in integer space (no float tile).
    hash53 = _logical_rshift(h, 11)
    threshold = int(p * _PRF_2POW53)
    return (hash53 >= threshold).to(dtype=dtype)


class ActivationMasker:
    """Installs/clears in-graph activation-mask forward hooks on boundary blocks.

    ``register(module)`` installs a forward hook on each boundary decoder block;
    ``unregister()`` removes them. The engine pairs register/unregister around the
    masked forward only. Before each micro-batch forward the engine calls
    ``set_context(...)`` with ``global_step`` and the token-aligned
    ``sample_ids`` / ``position_ids`` that key the per-element mask.
    """

    def __init__(
        self,
        *,
        p: float,
        base_seed: int,
        pp_size: int,
        rescale: bool = False,
        rescale_mode: str = "auto",
        state: Any = None,
    ):
        self.p = float(p)
        self.base_seed = int(base_seed)
        self.pp_size = int(pp_size)
        # Magnitude-restoration scheme applied to h*mask. `rescale_mode` selects
        # it; the `rescale` bool is honored when rescale_mode == "auto".
        #   "none"      -> h*mask                             (raw product)
        #   "constant"  -> h*mask/(1-p)                        (inverted dropout; E[h_tilde]=h)
        #   "rms_match" -> h*mask*detach(rms_true/rms_masked)  (per-token EXACT RMS match: the
        #                  downstream pre-norm RMSNorm divides by the TRUE pre-mask RMS)
        #   "auto"      -> "constant" if rescale else "none"
        self.rescale = bool(rescale)
        mode = str(rescale_mode).lower()
        if mode == "auto":
            mode = "constant" if self.rescale else "none"
        if mode not in ("none", "constant", "rms_match"):
            raise ValueError(f"mask rescale_mode must be one of none|constant|rms_match|auto; got {rescale_mode!r}")
        self.rescale_mode = mode
        self._rescale_gain = (1.0 / (1.0 - self.p)) if (mode == "constant" and self.p < 1.0) else 1.0
        self._state = state  # CommEffState, for the mask_applications counter
        self._handles: list[Any] = []
        self._boundary_set: set[int] = set()
        self.boundary_indices: list[int] = []
        # Per-forward context, set by the engine before each micro-batch forward.
        self._global_step = 0
        self._sample_ids: Optional[torch.Tensor] = None
        self._position_ids: Optional[torch.Tensor] = None
        # Last-measured masked fraction per boundary (comm_eff/mask_ratio).
        self.last_mask_ratio: dict[int, float] = {}
        # Hidden size H, recorded on first fire. Used to surface the PRF logical
        # PP byte budget (kept coords/token = (1-p)*H) for comparison against
        # PowerSGD's n*r. None until a mask fires.
        self.hidden_size: Optional[int] = None

    def set_context(
        self,
        *,
        global_step: int,
        sample_ids: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> None:
        """Set the PRF-key context (step + per-token stable ids) for the next forward."""
        self._global_step = int(global_step)
        self._sample_ids = None if sample_ids is None else sample_ids.reshape(-1)
        self._position_ids = None if position_ids is None else position_ids.reshape(-1)

    def _make_hook(self, layer_idx: int):
        masker = self

        def _hook(_mod: nn.Module, _inputs: tuple, output: Any):
            h = output[0] if isinstance(output, tuple) else output
            if not torch.is_tensor(h):
                return output

            # Confinement guard: the mask must fire only on eligible paths
            # ({train}, plus old_logprob when mask_recompute). Any other tag (or
            # None, the anchor pass) is contamination. The per-path counter below
            # still records a leak if asserts are disabled under -O.
            state = masker._state
            if state is not None and hasattr(state, "path_tag"):
                tag = state.path_tag
                eligible = mask_eligible_tags(state)
                assert tag in eligible, (
                    f"comm_eff mask fired on an ineligible path (path_tag={tag!r}, "
                    f"eligible={sorted(eligible)}); masking is confined to the "
                    "actor-train forward (and old-logprob recompute when "
                    "mask_recompute=true)."
                )

            hidden_size = h.shape[-1]
            if masker.hidden_size is None:
                masker.hidden_size = int(hidden_size)
            n_tokens = h.numel() // hidden_size
            sample_ids = masker._sample_ids
            position_ids = masker._position_ids
            if sample_ids is None or position_ids is None:
                raise RuntimeError(
                    "comm_eff mask fired without per-token identity: call "
                    "set_context(sample_ids=..., position_ids=...) before each masked "
                    "forward (the per-element mask has no positional fallback)."
                )
            if sample_ids.numel() != n_tokens or position_ids.numel() != n_tokens:
                raise RuntimeError(
                    f"comm_eff mask token-axis mismatch: activation has {n_tokens} "
                    f"tokens but got {sample_ids.numel()} sample_ids / "
                    f"{position_ids.numel()} position_ids (SP>1 / non-rmpad is out of scope)."
                )

            mask = prf_token_mask(
                sample_ids,
                position_ids,
                layer_idx=layer_idx,
                global_step=masker._global_step,
                base_seed=masker.base_seed,
                hidden_size=hidden_size,
                p=masker.p,
                device=h.device,
                dtype=h.dtype,
            ).view(h.shape)
            if masker.rescale_mode == "rms_match":
                # Idea 2b, realized self-contained and comms-valid: rescale the
                # masked activation by a DETACHED per-token gain so its RMS equals
                # the TRUE (pre-mask) RMS. The downstream pre-norm RMSNorm then
                # divides by the true RMS -> benign 1/RMS backward (no collapse
                # blow-up). Comms: rms_true is a 1-float/token side channel
                # (~1/((1-p)*H) overhead); rms_masked is recoverable on the
                # receiver from the kept (communicated) entries. The gain is
                # detached -> backward is mask*const (benign), like the constant
                # path but per-token exact. fp32 for bf16 safety; an all-masked
                # token yields h_tilde=0 (0 * finite gain), never NaN.
                hm = h * mask
                rms_true = h.float().pow(2).mean(dim=-1, keepdim=True).add(1e-8).sqrt()
                rms_masked = hm.float().pow(2).mean(dim=-1, keepdim=True).add(1e-8).sqrt()
                gain = (rms_true / rms_masked).detach().to(h.dtype)
                h_tilde = hm * gain
            else:
                h_tilde = h * mask * masker._rescale_gain if masker._rescale_gain != 1.0 else h * mask
            with torch.no_grad():
                masker.last_mask_ratio[layer_idx] = float(1.0 - mask.mean().item())
            if state is not None:
                if hasattr(state, "note_mask_application"):
                    state.note_mask_application()
                else:
                    state.mask_applications += 1
            if isinstance(output, tuple):
                return (h_tilde,) + tuple(output[1:])
            return h_tilde

        return _hook

    def register(self, module: nn.Module) -> None:
        """Install forward hooks on the boundary decoder blocks (idempotent).

        Clears any stale per-token context so a fire before ``set_context`` fails
        explicit rather than reusing the previous forward's identities.
        """
        if self._handles:
            return
        self._sample_ids = None
        self._position_ids = None
        layers = find_decoder_layers(module)
        if layers is None:
            logger.warning(
                "comm_eff.activation_mask: could not locate decoder layers on %s; "
                "no mask hooks registered (no-op this pass)",
                type(module).__name__,
            )
            return
        self.boundary_indices = decoder_boundary_indices(len(layers), self.pp_size)
        self._boundary_set = set(self.boundary_indices)
        for idx in self.boundary_indices:
            self._handles.append(layers[idx].register_forward_hook(self._make_hook(idx)))
        logger.info(
            "comm_eff.activation_mask: registered hooks on boundaries %s (L=%d, pp_size=%d, p=%.4f)",
            self.boundary_indices,
            len(layers),
            self.pp_size,
            self.p,
        )

    def unregister(self) -> None:
        """Remove all mask hooks. Called on exit of the masked forward."""
        for handle in self._handles:
            handle.remove()
        self._handles = []

    @property
    def is_registered(self) -> bool:
        return bool(self._handles)
