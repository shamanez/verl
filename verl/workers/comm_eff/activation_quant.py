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

"""Pipeline-boundary stochastic-rounding activation quantization (sr_quant codec).

Dense low-bit STOCHASTIC-ROUNDING quantization of the hidden-state output of the
pipeline-boundary decoder blocks, plus the same quantization of the upstream
gradient at the boundary on backward (modeling the compressed backward wire).
This is the dense-but-low-precision counterfactual to the PRF mask (sparse but
full-precision): every channel crosses the wire, at ``bits`` bits each, with one
fp16 scale per (token, block).

Forward, per token ``h`` (fp32 arithmetic): with blockwise scales
``s = absmax`` over each contiguous ``block_size``-dim block of the hidden dim
(detached, ``clamp_min 1e-8``; ``block_size=0`` means one whole-token scale),
``L = 2**bits`` uniform levels span ``[-s, +s]`` with spacing
``D = 2s/(L-1)``. Stochastic rounding draws ``u`` from a counter-based
splitmix64 PRF per (token, channel) and rounds up iff ``u < frac`` where
``frac`` is the fractional position between the two bracketing levels, so
``E[q] = h`` exactly (``|h| <= s`` within each block). For ``bits=1`` this
reduces to ``q in {-s, +s}`` with ``P(+s) = (h/s + 1)/2``.
``rounding="rn"`` instead rounds deterministically to the nearest level (no PRF
draw; biased, an ablation control). Backward quantizes the upstream gradient
onto its own grid (own blockwise absmax scales, fresh PRF ``direction`` subkey)
and returns it, so ``E[g_hat] = g``.

Subset mode (``subset_k > 0``, issue #93 I5, the byte-parity hybrid): per token
draw a PRF-fresh EXACT-``subset_k`` channel subset ``J`` with the mask codec's
order-statistic machinery (:func:`~verl.workers.comm_eff.activation_mask.\
prf_token_mask` with ``exact_k=True``/``exact_keep``, keyed identically to the
mask, so ``J`` is bit-identical across the old/train/ref passes of one step),
apply the blockwise SR quantization to the ``J`` values only (blocks of
``block_size`` CONSECUTIVE KEPT channels in ascending channel order), zero
elsewhere, and rescale by ``H/|J|``. Unbiased through BOTH randomness sources:
``P(j in J) = |J|/H`` by hash symmetry and ``E[SR(h_j) | J] = h_j``, so
``E[q] = h``. The backward wire applies the SAME ``J`` (the exact adjoint of
the subset+rescale map) with the backward SR subkey, so ``E[g_hat] = g``.
``subset_k = 0`` (default) is the full-width codec, byte-identical to before.

The PRF draw is keyed on ``(base_seed, layer_idx, global_step, sample_id,
position_id, channel, direction)``: the mask PRF key plus a trailing
``direction`` component (0 forward / 1 backward). There is NO path-dependent
component, so the old-logprob / train / reference forwards of one
``global_step`` (and any gradient-checkpoint recompute) produce bit-identical
outputs, preserving the PPO ratio identity (ratio == 1, ppo_kl == 0) at the
first inner step; across steps the draw is fresh. Hooks are registered only for
the eligible forwards and removed on exit, exactly like the PRF mask.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from verl.workers.comm_eff.activation_mask import (
    _PRF_2POW53,
    _derive_seed,
    _logical_rshift,
    _splitmix64_tensor,
    _u64_to_i64,
    decoder_boundary_indices,
    find_decoder_layers,
    prf_token_mask,
)
from verl.workers.comm_eff.state import mask_eligible_tags

logger = logging.getLogger(__name__)

__all__ = [
    "FORWARD_DIRECTION",
    "BACKWARD_DIRECTION",
    "prf_token_uniform",
    "sr_quantize",
    "BoundarySRQuant",
    "ActivationQuantizer",
]

# PRF key direction component: forward activation vs backward gradient wire.
FORWARD_DIRECTION = 0
BACKWARD_DIRECTION = 1

# Per-(token, block) scale floor: an all-zero token still gets a strictly
# positive scale, so q is level-exact with |q| <= 1e-8 and never NaN.
_SCALE_EPS = 1e-8
# One fp16 scale scalar per (token, block, boundary) rides the wire beside the
# b-bit payload: logical bits/token/boundary = H*bits + n_blocks*16.
_SCALE_BITS = 16

ROUNDING_MODES = ("sr", "rn")


def prf_token_uniform(
    sample_ids: torch.Tensor,
    position_ids: torch.Tensor,
    *,
    layer_idx: int,
    global_step: int,
    base_seed: int,
    hidden_size: int,
    direction: int,
    device: torch.device,
) -> torch.Tensor:
    """Deterministic per-(token, channel) uniform draw in ``[0, 1)``.

    Entry ``(t, j)`` is keyed on ``(base_seed, layer_idx, global_step,
    sample_ids[t], position_ids[t], j, direction)``: the mask PRF key plus a
    trailing ``direction`` component so the backward wire draws a fresh,
    reproducible subkey. Left-fold equivalence makes every entry bit-identical
    to :func:`_derive_seed` over the full key tuple. The uniform is the top 53
    bits of the final hash over ``2**53``, exactly like ``prf_token_mask``'s
    draw (exact in a float64 mantissa).

    Returns:
        ``(N, hidden_size)`` float64 uniforms in ``[0, 1)`` on ``device``.
    """
    if direction not in (FORWARD_DIRECTION, BACKWARD_DIRECTION):
        raise ValueError(
            f"direction must be {FORWARD_DIRECTION} (forward) or {BACKWARD_DIRECTION} (backward); got {direction}"
        )
    sid = sample_ids.reshape(-1).to(device=device, dtype=torch.int64)
    pos = position_ids.reshape(-1).to(device=device, dtype=torch.int64)
    if sid.numel() != pos.numel():
        raise ValueError(f"sample_ids and position_ids length mismatch: {sid.numel()} vs {pos.numel()}")

    # Fold the scalar prefix, the per-token ids, the channel index, then the
    # direction: bit-identical to _derive_seed over the full key tuple.
    prefix = _u64_to_i64(_derive_seed((base_seed, layer_idx, global_step)))
    acc = _splitmix64_tensor(sid ^ prefix)  # fold sample_id   -> (N,)
    acc = _splitmix64_tensor(acc ^ pos)  # fold position_id -> (N,)
    channel = torch.arange(hidden_size, device=device, dtype=torch.int64)
    h = _splitmix64_tensor(acc.unsqueeze(1) ^ channel.unsqueeze(0))  # (N, H)
    h = _splitmix64_tensor(h ^ int(direction))  # fold direction -> (N, H)

    # Top-53-bit uniform; 53 bits are exact in a float64 mantissa.
    hash53 = _logical_rshift(h, 11)
    return hash53.to(torch.float64) / float(_PRF_2POW53)


def _block_scales(m: torch.Tensor, *, hidden_size: int, block_size: int) -> tuple[torch.Tensor, int]:
    """Per-element blockwise absmax scale for a ``(N, H)`` fp32 matrix.

    The hidden dim is split into contiguous blocks of ``block_size`` channels
    (``block_size <= 0`` or ``>= H`` means one whole-token block); each block's
    scale is its absmax, ``clamp_min 1e-8``, broadcast back over the block's
    channels. A non-divisible tail forms its own (shorter) final block.

    Returns:
        ``(s, n_blocks)`` where ``s`` is the ``(N, H)`` per-element scale.
    """
    eff_block = hidden_size if (block_size <= 0 or block_size >= hidden_size) else int(block_size)
    n_blocks = (hidden_size + eff_block - 1) // eff_block
    pad = n_blocks * eff_block - hidden_size
    mb = m if pad == 0 else F.pad(m, (0, pad))  # zero-pad never raises an absmax
    s_block = mb.reshape(-1, n_blocks, eff_block).abs().amax(dim=-1, keepdim=True).clamp_min(_SCALE_EPS)
    s = s_block.expand(-1, n_blocks, eff_block).reshape(-1, n_blocks * eff_block)[:, :hidden_size]
    return s.contiguous(), n_blocks


def _grid_round(
    m: torch.Tensor, *, bits: int, block_size: int, rounding: str, u: Optional[torch.Tensor]
) -> torch.Tensor:
    """Round a ``(N, W)`` fp32 matrix onto its blockwise ``L = 2**bits`` grid.

    Shared kernel for the full-width and gathered-subset paths. Blockwise scale
    ``s`` = absmax over each contiguous ``block_size``-dim block of the width
    axis (detached, fp32, ``clamp_min 1e-8``; ``block_size=0`` means one
    whole-row scale); levels span ``[-s, +s]`` with spacing ``D = 2s/(L-1)``.
    ``rounding="sr"``: ``k = floor((m+s)/D)`` clamped to ``[0, L-2]``,
    ``lo = -s + k*D``, ``frac = (m - lo)/D`` in ``[0, 1]``,
    ``q = lo + D * 1{u < frac}`` with the caller-supplied ``(N, W)`` float64
    PRF uniform ``u``: ``E[q] = m`` exactly since ``|m| <= s`` within each
    block. ``rounding="rn"``: deterministic round-to-nearest to the same grid
    (``u`` ignored; biased, idempotent: the ablation control).
    """
    width = int(m.shape[-1])
    n_levels = 2 ** int(bits)
    s, _ = _block_scales(m, hidden_size=width, block_size=int(block_size))  # (N, W) fp32
    spacing = (2.0 * s) / float(n_levels - 1)

    if rounding == "rn":
        # Deterministic round-to-nearest onto the same grid: no PRF draw, so
        # cross-pass identity is trivial. Biased in general (E[q] != m).
        idx = torch.round((m + s) / spacing).clamp_(0.0, float(n_levels - 1))
        return -s + idx * spacing

    k = torch.floor((m + s) / spacing).clamp_(0.0, float(n_levels - 2))
    lo = -s + k * spacing
    frac = ((m - lo) / spacing).clamp_(0.0, 1.0)
    return lo + spacing * (u < frac.to(torch.float64)).to(torch.float32)


def sr_quantize(
    x: torch.Tensor,
    sample_ids: Optional[torch.Tensor],
    position_ids: Optional[torch.Tensor],
    *,
    layer_idx: int,
    global_step: int,
    base_seed: int,
    bits: int = 1,
    direction: int = FORWARD_DIRECTION,
    block_size: int = 0,
    rounding: str = "sr",
    subset_k: int = 0,
) -> torch.Tensor:
    """Quantize ``x`` onto ``L = 2**bits`` uniform levels per (token, block).

    Full-width mode (``subset_k=0``, default): every channel is rounded onto
    its blockwise grid (see :func:`_grid_round`). Arithmetic runs in fp32; the
    result is cast back to ``x.dtype``. An all-zero token (``s`` clamped)
    yields ``|q| <= 1e-8`` and no NaN.

    Subset mode (``subset_k > 0``, issue #93 I5): draw the per-token PRF-fresh
    EXACT-``subset_k`` channel subset ``J`` via ``prf_token_mask(exact_k=True,
    exact_keep=subset_k)`` (keyed exactly like the mask codec: no ``direction``
    component, so forward and backward share ``J`` and every pass of one step
    sees the same ``J``), gather the kept values in ascending channel order,
    round them on THEIR OWN blockwise grid (blocks of ``block_size``
    consecutive kept channels; the SR uniform is still keyed by the ORIGINAL
    channel index), scatter back, zero elsewhere, and rescale by
    ``H/subset_k``. Unbiased through both the subset draw and the stochastic
    rounding: ``E[q] = (H/k) * P(j in J) * E[SR(h_j) | J] = h``.
    """
    if bits < 1:
        raise ValueError(f"quant bits must be >= 1; got {bits}")
    if block_size < 0:
        raise ValueError(f"quant block_size must be >= 0; got {block_size}")
    if rounding not in ROUNDING_MODES:
        raise ValueError(f"quant rounding must be one of {ROUNDING_MODES}; got {rounding!r}")
    if subset_k < 0:
        raise ValueError(f"quant subset_k must be >= 0 (0 = full-width); got {subset_k}")
    hidden_size = int(x.shape[-1])
    orig_shape = x.shape
    orig_dtype = x.dtype
    m = x.detach().reshape(-1, hidden_size).to(torch.float32)

    # The subset draw needs per-token identity even under rn (J is PRF-keyed).
    if (rounding == "sr" or subset_k > 0) and (sample_ids is None or position_ids is None):
        raise RuntimeError(
            "sr_quantize requires per-token identity for the PRF draw: pass "
            "sample_ids/position_ids (rounding='sr' and/or subset_k > 0; the "
            "PRF has no positional fallback)."
        )

    def _uniform() -> torch.Tensor:
        return prf_token_uniform(
            sample_ids,
            position_ids,
            layer_idx=layer_idx,
            global_step=global_step,
            base_seed=base_seed,
            hidden_size=hidden_size,
            direction=direction,
            device=m.device,
        )

    if subset_k == 0:
        u = _uniform() if rounding == "sr" else None
        q = _grid_round(m, bits=bits, block_size=int(block_size), rounding=rounding, u=u)
        return q.reshape(orig_shape).to(orig_dtype)

    if subset_k > hidden_size:
        raise ValueError(f"quant subset_k={subset_k} exceeds the hidden size H={hidden_size}")
    keep = prf_token_mask(
        sample_ids,
        position_ids,
        layer_idx=layer_idx,
        global_step=global_step,
        base_seed=base_seed,
        hidden_size=hidden_size,
        p=0.0,  # exact_keep overrides the keep count; p never enters the key
        device=m.device,
        dtype=torch.float32,
        exact_k=True,
        exact_keep=int(subset_k),
    ).bool()
    n_tokens = m.shape[0]
    # Row-major boolean gather => exactly subset_k kept channels per token, in
    # ascending channel order (the wire's packing order).
    channel = torch.arange(hidden_size, device=m.device).expand(n_tokens, hidden_size)
    kept_idx = channel[keep].reshape(n_tokens, int(subset_k))
    m_kept = m[keep].reshape(n_tokens, int(subset_k))
    u_kept = None
    if rounding == "sr":
        # Keyed by the ORIGINAL channel index: gather-order-independent, and
        # bit-identical to the full-width draw at the same (token, channel).
        u_kept = _uniform()[keep].reshape(n_tokens, int(subset_k))
    q_kept = _grid_round(m_kept, bits=bits, block_size=int(block_size), rounding=rounding, u=u_kept)
    q = torch.zeros_like(m)
    q.scatter_(1, kept_idx, q_kept * (float(hidden_size) / float(subset_k)))
    return q.reshape(orig_shape).to(orig_dtype)


class BoundarySRQuant(torch.autograd.Function):
    """Quantize a boundary activation forward AND its upstream gradient backward.

    ``forward(h)`` returns ``SR_b(h)`` (``direction=0``); ``backward(g)``
    returns ``SR_b(g)`` (``direction=1``, its own blockwise absmax scales,
    fresh PRF subkey), modeling the compressed backward wire: ``E[q] = h`` and
    ``E[g_hat] = g``. Both draws share the ``(base_seed, layer_idx,
    global_step, sample_id, position_id, channel)`` key prefix, so within one
    ``global_step`` the old-logprob / train / reference forwards and any
    gradient-checkpoint recompute are bit-identical (the PPO ratio identity is
    preserved), while every new ``global_step`` draws fresh. ``rounding="rn"``
    applies deterministic round-to-nearest on both wires instead. Subset mode
    (``subset_k > 0``) shares one PRF subset ``J`` between the wires (its key
    has no direction component), so the backward's subset+rescale is the exact
    adjoint of the forward's and only the SR draw uses the backward subkey.
    """

    @staticmethod
    def forward(
        ctx,
        h: torch.Tensor,
        sample_ids: torch.Tensor,
        position_ids: torch.Tensor,
        layer_idx: int,
        global_step: int,
        base_seed: int,
        bits: int,
        block_size: int = 0,
        rounding: str = "sr",
        subset_k: int = 0,
    ) -> torch.Tensor:
        ctx.save_for_backward(sample_ids, position_ids)
        ctx.quant_key = (
            int(layer_idx),
            int(global_step),
            int(base_seed),
            int(bits),
            int(block_size),
            str(rounding),
            int(subset_k),
        )
        return sr_quantize(
            h,
            sample_ids,
            position_ids,
            layer_idx=layer_idx,
            global_step=global_step,
            base_seed=base_seed,
            bits=bits,
            direction=FORWARD_DIRECTION,
            block_size=block_size,
            rounding=rounding,
            subset_k=subset_k,
        )

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        sample_ids, position_ids = ctx.saved_tensors
        layer_idx, global_step, base_seed, bits, block_size, rounding, subset_k = ctx.quant_key
        g_hat = sr_quantize(
            grad_output,
            sample_ids,
            position_ids,
            layer_idx=layer_idx,
            global_step=global_step,
            base_seed=base_seed,
            bits=bits,
            direction=BACKWARD_DIRECTION,
            block_size=block_size,
            rounding=rounding,
            subset_k=subset_k,
        )
        return g_hat, None, None, None, None, None, None, None, None, None


class ActivationQuantizer:
    """Installs/clears in-graph SR-quantization forward hooks on boundary blocks.

    Same ``register(module)`` / ``unregister()`` / ``set_context(...)``
    lifecycle, ``decoder_boundary_indices`` boundaries and ``pp_size`` semantics
    as :class:`~verl.workers.comm_eff.activation_mask.ActivationMasker`.
    Selected by ``comm_eff.compression_type='sr_quant'``.

    Knob reuse: sr_quant reuses the mask sub-config for its eligibility and
    keying: ``mask.mask_recompute`` / ``mask.mask_reference`` widen the
    eligible path tags exactly as for prf_mask (via ``mask_eligible_tags``),
    and ``mask.seed`` / ``mask.pp_size`` provide the PRF base seed and the
    boundary placement. ``mask.p`` / ``rescale*`` / ``exact_k`` /
    ``antithetic`` / ``frlr*`` are IGNORED by sr_quant (no Bernoulli mask is
    drawn). Like prf_mask, sr_quant carries no PowerSGD basis, so the anchor
    cannot own ``Q`` under this codec (``anchor.owns_q=false`` required).
    """

    def __init__(
        self,
        *,
        bits: int = 1,
        base_seed: int = 0,
        pp_size: int = 8,
        block_size: int = 32,
        rounding: str = "sr",
        subset_k: int = 0,
        state: Any = None,
    ):
        if isinstance(bits, bool) or int(bits) < 1:
            raise ValueError(f"quant bits must be an integer >= 1; got {bits!r}")
        if isinstance(block_size, bool) or int(block_size) < 0:
            raise ValueError(f"quant block_size must be an integer >= 0; got {block_size!r}")
        if str(rounding) not in ROUNDING_MODES:
            raise ValueError(f"quant rounding must be one of {ROUNDING_MODES}; got {rounding!r}")
        if isinstance(subset_k, bool) or int(subset_k) < 0:
            raise ValueError(f"quant subset_k must be an integer >= 0 (0 = full-width); got {subset_k!r}")
        self.bits = int(bits)
        self.base_seed = int(base_seed)
        self.pp_size = int(pp_size)
        self.block_size = int(block_size)
        self.rounding = str(rounding)
        self.subset_k = int(subset_k)
        self._state = state  # CommEffState, for the applications counter
        self._handles: list[Any] = []
        self._boundary_set: set[int] = set()
        self.boundary_indices: list[int] = []
        # Per-forward context, set by the engine before each micro-batch forward.
        self._global_step = 0
        self._sample_ids: Optional[torch.Tensor] = None
        self._position_ids: Optional[torch.Tensor] = None
        # Hidden size H, recorded on first fire. Used to surface the logical PP
        # bit budget per token per boundary, the sr_quant analogue of
        # logical_pp_bytes_prf: H*bits + n_blocks*16 full-width (payload + fp16
        # blockwise scales), k*bits + k*16/block in subset mode (kept payload +
        # pro-rata scales; J costs no index bits, it is PRF-derivable).
        self.hidden_size: Optional[int] = None
        self.logical_pp_bits_sr_quant: Optional[float] = None

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
        quantizer = self

        def _hook(_mod: nn.Module, _inputs: tuple, output: Any):
            h = output[0] if isinstance(output, tuple) else output
            if not torch.is_tensor(h):
                return output

            # Confinement guard: the quantizer must fire only on eligible paths
            # ({train}, plus old_logprob / ref_logprob when mask_recompute /
            # mask_reference). Any other tag (or None, the anchor pass) is
            # contamination. Mirrors the ActivationMasker guard exactly.
            state = quantizer._state
            if state is not None and hasattr(state, "path_tag"):
                tag = state.path_tag
                eligible = mask_eligible_tags(state)
                assert tag in eligible, (
                    f"comm_eff sr_quant fired on an ineligible path (path_tag={tag!r}, "
                    f"eligible={sorted(eligible)}); quantization is confined to the "
                    "actor-train forward (and old-logprob / reference forwards when "
                    "mask_recompute / mask_reference are true)."
                )

            hidden_size = h.shape[-1]
            if quantizer.hidden_size is None:
                quantizer.hidden_size = int(hidden_size)
            if quantizer.subset_k > 0:
                # Subset wire: only the subset_k kept values cross, packed
                # contiguously with one fp16 scale per block_size KEPT channels.
                # The ragged tail block is counted pro-rata (16/block bits per
                # kept channel), the #93 4.3 parity accounting: k=493, bits=2,
                # block=32 -> 493*2 + 493*16/32 = 1232.5 (vs incumbent 1232).
                # J itself is PRF-derivable at the receiver: zero index bits.
                k = quantizer.subset_k
                eff_block = k if (quantizer.block_size <= 0 or quantizer.block_size >= k) else quantizer.block_size
                quantizer.logical_pp_bits_sr_quant = float(k * quantizer.bits + k * _SCALE_BITS / eff_block)
            else:
                eff_block = (
                    int(hidden_size)
                    if (quantizer.block_size <= 0 or quantizer.block_size >= int(hidden_size))
                    else quantizer.block_size
                )
                n_blocks = (int(hidden_size) + eff_block - 1) // eff_block
                quantizer.logical_pp_bits_sr_quant = float(int(hidden_size) * quantizer.bits + n_blocks * _SCALE_BITS)
            n_tokens = h.numel() // hidden_size
            sample_ids = quantizer._sample_ids
            position_ids = quantizer._position_ids
            if sample_ids is None or position_ids is None:
                raise RuntimeError(
                    "comm_eff sr_quant fired without per-token identity: call "
                    "set_context(sample_ids=..., position_ids=...) before each quantized "
                    "forward (the PRF draw has no positional fallback)."
                )
            if sample_ids.numel() != n_tokens or position_ids.numel() != n_tokens:
                raise RuntimeError(
                    f"comm_eff sr_quant token-axis mismatch: activation has {n_tokens} "
                    f"tokens but got {sample_ids.numel()} sample_ids / "
                    f"{position_ids.numel()} position_ids (SP>1 / non-rmpad is out of scope)."
                )

            h_tilde = BoundarySRQuant.apply(
                h,
                sample_ids,
                position_ids,
                layer_idx,
                quantizer._global_step,
                quantizer.base_seed,
                quantizer.bits,
                quantizer.block_size,
                quantizer.rounding,
                quantizer.subset_k,
            )
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
                "comm_eff.activation_quant: could not locate decoder layers on %s; "
                "no quant hooks registered (no-op this pass)",
                type(module).__name__,
            )
            return
        self.boundary_indices = decoder_boundary_indices(len(layers), self.pp_size)
        self._boundary_set = set(self.boundary_indices)
        for idx in self.boundary_indices:
            self._handles.append(layers[idx].register_forward_hook(self._make_hook(idx)))
        logger.info(
            "comm_eff.activation_quant: registered hooks on boundaries %s "
            "(L=%d, pp_size=%d, bits=%d, block_size=%d, rounding=%s, subset_k=%d)",
            self.boundary_indices,
            len(layers),
            self.pp_size,
            self.bits,
            self.block_size,
            self.rounding,
            self.subset_k,
        )

    def unregister(self) -> None:
        """Remove all quant hooks. Called on exit of the quantized forward."""
        for handle in self._handles:
            handle.remove()
        self._handles = []

    @property
    def is_registered(self) -> bool:
        return bool(self._handles)
