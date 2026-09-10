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

"""AQ-SGD boundary activation compression (``aq_sgd`` codec).

Wang et al., "Fine-tuning Language Models over Slow Networks using Activation
Quantization with Guarantees" (NeurIPS 2022). AQ-SGD does NOT quantize the
activation; it quantizes the CHANGE of the activation for the same training
example between visits. Each side of a boundary keeps a local buffer
``m(xi)``; the sending stage transmits ``Q(a(xi) - m(xi))`` and both sides
update ``m(xi) <- m(xi) + Q(a(xi) - m(xi))``. The receiving stage uses that
running reconstruction as the forward activation. The backward gradient is
quantized DIRECTLY (the paper's own choice: gradients carry no cross-visit
history to difference against).

Why this codec exists here: it is the strongest named baseline from the
pipeline-parallel activation-compression literature, and the one the RLVR
codec-selection table is missing. It is also the codec whose central premise
this setting stresses, which is the point of running it.

THE PREMISE, AND WHAT RLVR DOES TO IT
-------------------------------------
AQ-SGD's guarantee rests on a self-enforcing loop: training stabilizes, so the
activation for a FIXED example changes less between visits, so the same bit
budget spent on the delta gives a smaller error, so training stabilizes
further. It needs a training example to RECUR with its activation intact.

On-policy RLVR breaks that at the response. Every optimizer step draws fresh
prompts and samples fresh responses from the current policy, so:

* **Prompt-prefix tokens recur.** A MATH problem is revisited each epoch with
  byte-identical token ids at identical positions, so its boundary activation
  differs only by the weight drift accumulated since the last visit. This is
  exactly the regime AQ-SGD was designed for, and the delta really is small.
* **Response tokens do not recur.** The token occupying a given position is
  resampled, so a buffered value from a previous visit is the activation of a
  DIFFERENT token. Differencing against it is not compression: for roughly
  independent activations ``E||a - m||^2 = ||a||^2 + ||m||^2``, so the delta
  carries about ``sqrt(2)`` times the norm of the value, a coarser grid at
  fixed bits, and a LARGER error than quantizing the value directly.

``scope`` selects which positions are delta-coded. ``prompt`` (the default)
confines the buffer to the prefix, giving AQ-SGD its best case in this
regime and leaving response positions on the memoryless path. ``all`` is the
literal port and buffers every position; it is the arm that measures the
norm-inflation penalty above rather than assuming it.

PATH IDENTITY, AND WHY THE UPDATE IS DEFERRED
---------------------------------------------
AQ-SGD is inherently path-dependent: ``m`` moves as it is used. This project's
codecs must instead be bit-identical across the several passes of ONE
optimizer step (old-logprob recompute, policy forward, gradient-checkpoint
recompute, reference forward), because that identity is what makes the PPO
ratio start each step at exactly one. The two are reconciled by treating one
optimizer step as one VISIT:

* every eligible pass READS the buffer as it stood at the end of the previous
  step;
* the ``train`` pass STAGES its refreshed buffer, and the stage is published
  only when the step counter advances.

A plain "skip the second write" guard would not be enough, because the second
pass would then read what the first pass published and the two would disagree.
Deferring publication to the step boundary is what makes the
gradient-checkpoint recompute re-derive the identical reconstruction. The
reference forward (different weights, so a different ``a``) never stages at
all.

STORAGE, WHICH IS PART OF THE RESULT
------------------------------------
The buffer is indexed by ``(example_id, position, boundary)`` and must span the
REUSE DISTANCE to score a hit. In RLVR that distance is a full epoch of
prompts. At Pair 1 (7.5k MATH problems, 7 boundaries, H=1536, bf16) covering
prompt prefixes alone costs O(10 GB) of host memory, and covering full
2048-token sequences costs O(100 TB). The keyed mask it is compared against
stores nothing at all. ``capacity_bytes`` caps the store with LRU eviction on
``(example_id, boundary)``, so a box with less memory than the reuse distance
lowers the measured hit rate instead of silently corrupting the codec.

Buffers are per process, which is faithful: the paper notes that under data
parallelism the store divides by the parallel degree but then pays for data
shuffling. A prompt that lands on a different DP rank than last epoch is a
miss, and ``aq_sgd/hit_rate`` reports the consequence rather than hiding it.

WIRE ACCOUNTING
---------------
``subset_k > 0`` sends only a PRF-fresh exact-``subset_k`` channel subset ``J``
per token, keyed exactly like the mask codec so ``J`` is shared by the forward
and backward wires and by every pass of one step. Unlike the ``sr_quant``
codec, the un-sent channels are NOT zeroed and there is no ``H/k`` gain: the
receiver already holds ``m`` there, so it keeps it. Reconstruction is
``m + scatter_J(Q(a_J - m_J))``, a coordinate-descent refinement of the
buffer. Bits per token per boundary are then
``subset_k*bits + subset_k*16/block_size``, the same ledger ``sr_quant``
reports, so an arm can be byte-matched against PRF exact-k.

``first_visit`` decides what a COLD entry sends, and the two settings are two
different experiments:

* ``rescaled`` (default) keeps the byte budget. A cold visit has no buffer to
  refine, so it falls back to the unbiased sparse estimate
  ``(H/k) * scatter_J(Q(a_J))``, which is precisely the ``sr_quant`` subset
  codec. The arm therefore interpolates: cold tokens are ``sr_quant``, warm
  tokens are AQ-SGD, and the single variable against an ``sr_quant`` arm at
  the same ``(bits, subset_k, block_size)`` is delta coding itself.
* ``dense`` is faithful to the paper, which sends the first message
  uncompressed. It is off-budget by construction and exists to MEASURE the
  resulting payload, since in this regime most tokens are first visits. Use it
  for a payload accounting run, not for a byte-matched accuracy comparison.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Any, Optional

import torch
import torch.nn as nn

from verl.workers.comm_eff.activation_mask import (
    _splitmix64_tensor,
    decoder_boundary_indices,
    find_decoder_layers,
    prf_token_mask,
)
from verl.workers.comm_eff.activation_quant import (
    _SCALE_BITS,
    BACKWARD_DIRECTION,
    FORWARD_DIRECTION,
    ROUNDING_MODES,
    _grid_round,
    prf_token_uniform,
)
from verl.workers.comm_eff.state import mask_eligible_tags

logger = logging.getLogger(__file__)

__all__ = [
    "AQSGD_SCOPES",
    "AQSGD_FIRST_VISIT_MODES",
    "AQSGDBuffer",
    "BoundaryAQSGD",
    "ActivationAQSGD",
    "aqsgd_example_ids",
]

# Which token positions carry a buffer. ``prompt`` = the recurring prefix only
# (AQ-SGD's premise holds); ``all`` = every position (the literal port).
AQSGD_SCOPES = ("prompt", "all")

# What a cold buffer entry sends. See the module docstring.
AQSGD_FIRST_VISIT_MODES = ("rescaled", "dense")

# The buffer is stored in bf16: two bytes per channel per (example, position,
# boundary), and halves the store against fp32.
#
# bf16 NOT fp16, and the range is the whole reason. What gets buffered is the
# reconstruction ``x_hat``, and on a COLD row under ``rescaled`` that carries
# the ``H/k`` gain (3.1156 at Pair 1), so the stored slab sits ~3x above the
# true activation scale. Qwen2.5's massive-activation channels then cross
# fp16's 65504 ceiling at a true magnitude of only ~21k, the entry saturates to
# ``inf``, and the first WARM read differences against it: the block absmax
# goes inf, the grid spacing with it, and ``floor((m + s) / spacing)`` is
# ``floor(nan)``. One saturated channel returns nan for all 32 channels of its
# block on every row that shares it, the loss goes nan, and the optimizer step
# is skipped. That is exactly what killed the first three aq_sgd arms at step
# 59, their first warm step. bf16 spends 3 mantissa bits to buy 2^127 of range,
# which is free here: the delta is quantized to ``bits`` (2) per channel
# afterwards, so a 0.4% reference error is far below the wire's own resolution.
_BUFFER_DTYPE = torch.bfloat16
_BUFFER_ITEMSIZE = 2

# Default LRU cap on the host-side store. 16 GiB is about one epoch of Pair 1's
# prompt prefixes at n_examples * 7 boundaries * 1536 dims * 2 bytes, which is
# roughly 0.16 GB per buffered token position. Kept identical to the config
# dataclass default, the Hydra default and the launcher default: the fanout's
# host-RAM gate reserves a fixed budget per codec arm, so a layer that
# defaulted higher on its own would silently overrun it.
_DEFAULT_CAPACITY_BYTES = 16 * (1024**3)


def aqsgd_example_ids(
    prompt_token_ids: torch.Tensor,
    prompt_mask: torch.Tensor,
) -> torch.Tensor:
    """Content hash of each row's prompt, the persistent example identity.

    AQ-SGD needs an example id that survives across epochs. The batch does not
    carry one: ``comm_eff_sample_id`` is a row index within a rank's shard and
    ``uid`` is a fresh uuid4 per step, both of which identify a row inside ONE
    step and nothing beyond it. The prompt's own token ids are the identity
    that does persist, and hashing them needs no plumbing through the trainer,
    agrees across ranks and processes, and is stable across epochs by
    construction.

    The hash folds position into each token's key and sums, which is one
    vectorized ``splitmix64`` over the ``(B, P)`` id matrix rather than a
    sequential fold over ``P``:

        ``id_i = sum_j 1{mask_ij} * splitmix64(tok_ij ^ splitmix64(j + 1))``

    accumulated mod ``2**64``, finalised, and masked to 63 bits. Position
    enters the key, so the hash is order sensitive; distinct MATH prompts
    colliding is a ``~2**-63`` event per pair.

    Args:
        prompt_token_ids: ``(B, P)`` integer prompt ids (padding included).
        prompt_mask: ``(B, P)`` mask, non-zero on real prompt tokens. Padding
            must be masked or padded rows would hash together.

    Returns:
        ``(B,)`` non-negative int64 example ids.
    """
    if prompt_token_ids.dim() != 2:
        raise ValueError(f"prompt_token_ids must be (B, P); got {tuple(prompt_token_ids.shape)}")
    if prompt_mask.shape != prompt_token_ids.shape:
        raise ValueError(
            f"prompt_mask {tuple(prompt_mask.shape)} must match prompt_token_ids {tuple(prompt_token_ids.shape)}"
        )
    device = prompt_token_ids.device
    n_pos = int(prompt_token_ids.shape[1])
    toks = prompt_token_ids.to(torch.int64)
    # splitmix64 of the 1-based position, broadcast over rows.
    pos_key = _splitmix64_tensor(torch.arange(1, n_pos + 1, dtype=torch.int64, device=device))
    mixed = _splitmix64_tensor(torch.bitwise_xor(toks, pos_key.unsqueeze(0)))
    keep = prompt_mask.to(torch.bool)
    # int64 addition wraps mod 2**64 in two's complement, which is the intended
    # accumulator; the sign is stripped afterwards.
    acc = torch.where(keep, mixed, torch.zeros_like(mixed)).sum(dim=1)
    # Clear the sign bit rather than calling abs(), which has no fixed point at
    # int64 min. The remaining 63 bits are the id.
    return torch.bitwise_and(_splitmix64_tensor(acc), 0x7FFFFFFFFFFFFFFF)


class AQSGDBuffer:
    """The ``m(xi)`` store: buffered boundary activations by example and boundary.

    One entry per ``(example_id, boundary)``, holding an ``(n_positions, H)``
    bf16 tensor on ``device`` (``cpu`` by default; the store is far too large
    for HBM) plus the optimizer step it was last written at. Eviction is LRU on
    whole entries, capped by ``capacity_bytes``.

    Entries are keyed by example rather than by row so that the ``G`` rollouts
    sharing a prompt within a step share one prefix entry, which is correct:
    the prefix activation does not depend on the response sampled after it.
    """

    def __init__(
        self,
        *,
        capacity_bytes: int = _DEFAULT_CAPACITY_BYTES,
        device: str = "cpu",
    ):
        if isinstance(capacity_bytes, bool) or int(capacity_bytes) <= 0:
            raise ValueError(f"aq_sgd capacity_bytes must be a positive integer; got {capacity_bytes!r}")
        self.capacity_bytes = int(capacity_bytes)
        self.device = torch.device(device)
        # (example_id, boundary) -> [values (n_pos, H) bf16, step_last_updated]
        self._store: OrderedDict[tuple[int, int], list] = OrderedDict()
        self._bytes = 0
        # Telemetry. Counted in TOKEN-POSITIONS, not entries, so the hit rate is
        # the fraction of buffered traffic that actually had a history.
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.writes = 0
        self.nonfinite_rows = 0

    def __len__(self) -> int:
        return len(self._store)

    @property
    def bytes_used(self) -> int:
        return self._bytes

    @property
    def hit_rate(self) -> float:
        seen = self.hits + self.misses
        return (self.hits / seen) if seen else 0.0

    def get(self, example_id: int, boundary: int) -> Optional[list]:
        """Return the live ``[values, step]`` entry, marking it most-recently-used."""
        key = (int(example_id), int(boundary))
        entry = self._store.get(key)
        if entry is not None:
            self._store.move_to_end(key)
        return entry

    def put(self, example_id: int, boundary: int, values: torch.Tensor, step: int) -> None:
        """Insert or replace an entry, evicting LRU until the cap is respected."""
        key = (int(example_id), int(boundary))
        vals = values.detach().to(device=self.device, dtype=_BUFFER_DTYPE).contiguous()
        size = vals.numel() * _BUFFER_ITEMSIZE
        old = self._store.pop(key, None)
        if old is not None:
            self._bytes -= old[0].numel() * _BUFFER_ITEMSIZE
        self._store[key] = [vals, int(step)]
        self._bytes += size
        self.writes += 1
        while self._bytes > self.capacity_bytes and len(self._store) > 1:
            _, evicted = self._store.popitem(last=False)
            self._bytes -= evicted[0].numel() * _BUFFER_ITEMSIZE
            self.evictions += 1

    def reset(self) -> None:
        self._store.clear()
        self._bytes = 0

    def telemetry(self) -> dict:
        return {
            "aq_sgd/buffer_entries": float(len(self._store)),
            "aq_sgd/buffer_gib": self._bytes / float(1024**3),
            "aq_sgd/hit_rate": self.hit_rate,
            "aq_sgd/hits": float(self.hits),
            "aq_sgd/misses": float(self.misses),
            "aq_sgd/evictions": float(self.evictions),
            "aq_sgd/nonfinite_rows": float(self.nonfinite_rows),
        }


def aqsgd_reconstruct(
    x: torch.Tensor,
    m: Optional[torch.Tensor],
    warm: Optional[torch.Tensor],
    sample_ids: Optional[torch.Tensor],
    position_ids: Optional[torch.Tensor],
    *,
    layer_idx: int,
    global_step: int,
    base_seed: int,
    bits: int,
    direction: int = FORWARD_DIRECTION,
    block_size: int = 32,
    rounding: str = "sr",
    subset_k: int = 0,
    first_visit: str = "rescaled",
) -> torch.Tensor:
    """The AQ-SGD boundary message, reconstructed at the receiver.

    Returns what the downstream stage actually consumes, and (as a side effect
    of being a pure function of its inputs) what both sides then hold as the
    refreshed buffer.

    WARM rows (``warm`` true, ``m`` finite): the wire carries
    ``Q(x_J - m_J)`` and the receiver reconstructs

        ``x_hat = m + scatter_J(Q(x_J - m_J))``

    so channels outside ``J`` KEEP their buffered value. There is no ``H/k``
    gain: unlike a sparsifier, nothing was zeroed, so nothing needs
    rescaling. Over successive visits this is coordinate-descent refinement of
    ``m`` toward ``x``.

    COLD rows have no history to difference against, and what they send is the
    experiment ``first_visit`` selects:

    * ``rescaled`` holds the byte budget by falling back to the unbiased
      sparse estimate ``(H/k) * scatter_J(Q(x_J))``, identical to the
      ``sr_quant`` subset codec at the same knobs. The gain is required here
      precisely BECAUSE the un-sent channels are zero on a cold row.
    * ``dense`` sends ``x`` at full precision, faithful to the paper's first
      message and off-budget by construction.

    ``subset_k = 0`` sends every channel, so ``J`` is everything, the cold
    fallback needs no gain, and warm rows carry a full-width quantized delta.

    Args:
        x: ``(..., H)`` activation (or upstream gradient) at the boundary.
        m: ``(N, H)`` buffered values aligned to ``x``'s flattened token axis,
            or ``None`` when no row is warm.
        warm: ``(N,)`` bool, true where ``m`` holds a real previous visit.
        first_visit: see above.

    Returns:
        ``x_hat``, shaped and typed like ``x``.
    """
    if bits < 1:
        raise ValueError(f"aq_sgd bits must be >= 1; got {bits}")
    if rounding not in ROUNDING_MODES:
        raise ValueError(f"aq_sgd rounding must be one of {ROUNDING_MODES}; got {rounding!r}")
    if first_visit not in AQSGD_FIRST_VISIT_MODES:
        raise ValueError(f"aq_sgd first_visit must be one of {AQSGD_FIRST_VISIT_MODES}; got {first_visit!r}")
    if subset_k < 0:
        raise ValueError(f"aq_sgd subset_k must be >= 0 (0 = full-width); got {subset_k}")

    hidden_size = int(x.shape[-1])
    if subset_k > hidden_size:
        raise ValueError(f"aq_sgd subset_k={subset_k} exceeds the hidden size H={hidden_size}")
    orig_shape = x.shape
    orig_dtype = x.dtype
    v = x.detach().reshape(-1, hidden_size).to(torch.float32)
    n_tokens = int(v.shape[0])

    if (rounding == "sr" or subset_k > 0) and (sample_ids is None or position_ids is None):
        raise RuntimeError(
            "aq_sgd requires per-token identity for the PRF draw: pass "
            "sample_ids/position_ids (rounding='sr' and/or subset_k > 0)."
        )

    # Buffered values, zero where cold. A cold row differences against zero,
    # which is the same arithmetic as quantizing the value itself.
    if m is None or warm is None:
        m_v = torch.zeros_like(v)
        warm_rows = torch.zeros(n_tokens, dtype=torch.bool, device=v.device)
    else:
        m_v = m.reshape(-1, hidden_size).to(device=v.device, dtype=torch.float32)
        warm_rows = warm.reshape(-1).to(device=v.device, dtype=torch.bool)
        m_v = torch.where(warm_rows.unsqueeze(1), m_v, torch.zeros_like(m_v))

    # The faithful cold path sends x uncompressed, so those rows bypass the
    # codec entirely and the buffer they seed is exact.
    dense_cold = first_visit == "dense"

    delta = v - m_v

    if subset_k == 0:
        u = None
        if rounding == "sr":
            u = prf_token_uniform(
                sample_ids,
                position_ids,
                layer_idx=layer_idx,
                global_step=global_step,
                base_seed=base_seed,
                hidden_size=hidden_size,
                direction=direction,
                device=v.device,
            )
        q = _grid_round(delta, bits=bits, block_size=int(block_size), rounding=rounding, u=u)
        x_hat = m_v + q
        if dense_cold:
            x_hat = torch.where(warm_rows.unsqueeze(1), x_hat, v)
        return x_hat.reshape(orig_shape).to(orig_dtype)

    # Subset wire. J is keyed exactly like the mask codec (no direction
    # component), so the forward and backward wires and every pass of one step
    # share it, and the receiver derives it without spending index bits.
    keep = prf_token_mask(
        sample_ids,
        position_ids,
        layer_idx=layer_idx,
        global_step=global_step,
        base_seed=base_seed,
        hidden_size=hidden_size,
        p=0.0,  # exact_keep sets the count; p never enters the key
        device=v.device,
        dtype=torch.float32,
        exact_k=True,
        exact_keep=int(subset_k),
    ).bool()
    channel = torch.arange(hidden_size, device=v.device).expand(n_tokens, hidden_size)
    kept_idx = channel[keep].reshape(n_tokens, int(subset_k))
    delta_kept = delta[keep].reshape(n_tokens, int(subset_k))
    u_kept = None
    if rounding == "sr":
        # Keyed on the ORIGINAL channel index, so the draw is independent of
        # gather order and matches the full-width draw at the same (token,
        # channel).
        u_kept = prf_token_uniform(
            sample_ids,
            position_ids,
            layer_idx=layer_idx,
            global_step=global_step,
            base_seed=base_seed,
            hidden_size=hidden_size,
            direction=direction,
            device=v.device,
        )[keep].reshape(n_tokens, int(subset_k))
    q_kept = _grid_round(delta_kept, bits=bits, block_size=int(block_size), rounding=rounding, u=u_kept)

    # Warm rows refine the buffer in place on J and keep m elsewhere. Cold rows
    # under ``rescaled`` have m == 0, so the scatter leaves zeros off J and the
    # H/k gain is what restores the estimate's scale.
    gain = float(hidden_size) / float(subset_k)
    scatter_warm = torch.zeros_like(v)
    scatter_warm.scatter_(1, kept_idx, q_kept)
    x_hat_warm = m_v + scatter_warm
    scatter_cold = torch.zeros_like(v)
    scatter_cold.scatter_(1, kept_idx, q_kept * gain)
    x_hat_cold = v if dense_cold else scatter_cold
    x_hat = torch.where(warm_rows.unsqueeze(1), x_hat_warm, x_hat_cold)
    return x_hat.reshape(orig_shape).to(orig_dtype)


class BoundaryAQSGD(torch.autograd.Function):
    """AQ-SGD forward reconstruction with a directly quantized backward wire.

    ``forward(h)`` returns the AQ-SGD reconstruction of ``h`` against the
    buffer (:func:`aqsgd_reconstruct`, ``direction=0``). ``backward(g)``
    quantizes the upstream gradient DIRECTLY, with no buffer and its own
    blockwise scales and PRF subkey (``direction=1``), which is the paper's own
    treatment of the backward pass: a gradient has no cross-visit history to
    difference against. Passing ``m=None`` on backward is what makes it direct.

    The subset ``J`` is shared between the wires, so the backward's
    subset-and-gain map is the exact adjoint of the forward's cold path.
    """

    @staticmethod
    def forward(
        ctx,
        h: torch.Tensor,
        m: Optional[torch.Tensor],
        warm: Optional[torch.Tensor],
        sample_ids: torch.Tensor,
        position_ids: torch.Tensor,
        layer_idx: int,
        global_step: int,
        base_seed: int,
        bits: int,
        block_size: int,
        rounding: str,
        subset_k: int,
        first_visit: str,
    ) -> torch.Tensor:
        ctx.save_for_backward(sample_ids, position_ids)
        ctx.aqsgd_key = (
            int(layer_idx),
            int(global_step),
            int(base_seed),
            int(bits),
            int(block_size),
            str(rounding),
            int(subset_k),
        )
        return aqsgd_reconstruct(
            h,
            m,
            warm,
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
            first_visit=first_visit,
        )

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        sample_ids, position_ids = ctx.saved_tensors
        layer_idx, global_step, base_seed, bits, block_size, rounding, subset_k = ctx.aqsgd_key
        g_hat = aqsgd_reconstruct(
            grad_output,
            None,  # no buffer on the backward wire: direct quantization
            None,
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
            first_visit="rescaled",  # every backward row is cold by construction
        )
        return (g_hat,) + (None,) * 12


class ActivationAQSGD:
    """Installs/clears in-graph AQ-SGD forward hooks on the boundary blocks.

    Exposes the same ``register(module)`` / ``set_context(...)`` /
    ``unregister()`` lifecycle, ``decoder_boundary_indices`` boundaries and
    ``pp_size`` semantics as
    :class:`~verl.workers.comm_eff.activation_mask.ActivationMasker` and
    :class:`~verl.workers.comm_eff.activation_quant.ActivationQuantizer`, so
    the one engine call site serves every per-token codec. Selected by
    ``comm_eff.compression_type='aq_sgd'``.

    ``set_context`` takes two fields the other codecs do not need:
    ``example_ids`` (the persistent per-token example identity from
    :func:`aqsgd_example_ids`) and ``prompt_lens`` (per-token prompt length,
    which bounds ``scope='prompt'``). Both are required: without a persistent
    identity there is no buffer to look up and the codec would silently
    degenerate into memoryless quantization.

    Knob reuse matches ``sr_quant``: ``mask.mask_recompute`` /
    ``mask.mask_reference`` widen the eligible path tags, and ``mask.seed`` /
    ``mask.pp_size`` supply the PRF base seed and boundary placement. Like
    ``prf_mask`` and ``sr_quant`` it carries no PowerSGD basis, so
    ``anchor.owns_q`` must be false.
    """

    def __init__(
        self,
        *,
        bits: int = 2,
        base_seed: int = 0,
        pp_size: int = 8,
        block_size: int = 32,
        rounding: str = "sr",
        subset_k: int = 0,
        scope: str = "prompt",
        first_visit: str = "rescaled",
        capacity_bytes: int = _DEFAULT_CAPACITY_BYTES,
        buffer_device: str = "cpu",
        max_positions: int = 0,
        state: Any = None,
    ):
        if isinstance(bits, bool) or int(bits) < 1:
            raise ValueError(f"aq_sgd bits must be an integer >= 1; got {bits!r}")
        if isinstance(block_size, bool) or int(block_size) < 0:
            raise ValueError(f"aq_sgd block_size must be an integer >= 0; got {block_size!r}")
        if str(rounding) not in ROUNDING_MODES:
            raise ValueError(f"aq_sgd rounding must be one of {ROUNDING_MODES}; got {rounding!r}")
        if isinstance(subset_k, bool) or int(subset_k) < 0:
            raise ValueError(f"aq_sgd subset_k must be an integer >= 0 (0 = full-width); got {subset_k!r}")
        if str(scope) not in AQSGD_SCOPES:
            raise ValueError(f"aq_sgd scope must be one of {AQSGD_SCOPES}; got {scope!r}")
        if str(first_visit) not in AQSGD_FIRST_VISIT_MODES:
            raise ValueError(f"aq_sgd first_visit must be one of {AQSGD_FIRST_VISIT_MODES}; got {first_visit!r}")
        if isinstance(max_positions, bool) or int(max_positions) < 0:
            raise ValueError(f"aq_sgd max_positions must be an integer >= 0 (0 = unbounded); got {max_positions!r}")
        self.bits = int(bits)
        self.base_seed = int(base_seed)
        self.pp_size = int(pp_size)
        self.block_size = int(block_size)
        self.rounding = str(rounding)
        self.subset_k = int(subset_k)
        self.scope = str(scope)
        self.first_visit = str(first_visit)
        self.max_positions = int(max_positions)
        self.buffer = AQSGDBuffer(capacity_bytes=capacity_bytes, device=buffer_device)
        self._state = state
        self._handles: list[Any] = []
        self._boundary_set: set[int] = set()
        self.boundary_indices: list[int] = []
        # Per-forward context, set by the engine before each micro-batch.
        self._global_step = 0
        self._sample_ids: Optional[torch.Tensor] = None
        self._position_ids: Optional[torch.Tensor] = None
        self._example_ids: Optional[torch.Tensor] = None
        self._prompt_lens: Optional[torch.Tensor] = None
        self.hidden_size: Optional[int] = None
        self.logical_pp_bits_aq_sgd: Optional[float] = None
        # Staged, not yet published, buffer slabs for the CURRENT step, keyed
        # (example_id, boundary). Published by _flush_pending at the next step
        # boundary. See _stage_buffer for why publication is deferred.
        self._pending: dict = {}
        self._staged_step = 0
        # Diagnostic: mean ||x - m|| / ||x|| over WARM rows. This is the number
        # AQ-SGD's premise is about. Below 1 the buffer is predictive and the
        # delta really is cheaper to send than the value; at about sqrt(2) the
        # buffered row is unrelated to the current one and delta coding is
        # actively worse than quantizing the value.
        self.delta_ratio_sum = 0.0
        self.delta_ratio_count = 0

    def set_context(
        self,
        *,
        global_step: int,
        sample_ids: torch.Tensor,
        position_ids: torch.Tensor,
        example_ids: Optional[torch.Tensor] = None,
        prompt_lens: Optional[torch.Tensor] = None,
    ) -> None:
        """Set the PRF key and the buffer key for the next forward.

        A change of ``global_step`` publishes the previous step's staged
        buffer. Every pass of one step therefore reads the same ``m``, which is
        what keeps the reconstruction pass-identical and the PPO ratio at one.
        """
        if int(global_step) != int(self._global_step):
            self._flush_pending()
        self._global_step = int(global_step)
        self._sample_ids = None if sample_ids is None else sample_ids.reshape(-1)
        self._position_ids = None if position_ids is None else position_ids.reshape(-1)
        self._example_ids = None if example_ids is None else example_ids.reshape(-1)
        self._prompt_lens = None if prompt_lens is None else prompt_lens.reshape(-1)

    @property
    def mean_delta_ratio(self) -> float:
        return (self.delta_ratio_sum / self.delta_ratio_count) if self.delta_ratio_count else 0.0

    def telemetry(self) -> dict:
        out = self.buffer.telemetry()
        out["aq_sgd/delta_ratio"] = self.mean_delta_ratio
        if self.logical_pp_bits_aq_sgd is not None:
            out["aq_sgd/logical_pp_bits"] = float(self.logical_pp_bits_aq_sgd)
        return out

    def _in_scope(self, position_ids: torch.Tensor) -> torch.Tensor:
        """Which token positions carry a buffer under the configured scope."""
        if self.scope == "prompt":
            if self._prompt_lens is None:
                raise RuntimeError(
                    "comm_eff aq_sgd scope='prompt' needs per-token prompt_lens: "
                    "call set_context(prompt_lens=...) before each forward."
                )
            in_scope = position_ids < self._prompt_lens.to(position_ids.device)
        else:
            in_scope = torch.ones_like(position_ids, dtype=torch.bool)
        if self.max_positions > 0:
            in_scope = in_scope & (position_ids < self.max_positions)
        return in_scope

    def _record_bits(self, hidden_size: int) -> None:
        """Log the per-token per-boundary wire cost, the byte-parity ledger."""
        if self.subset_k > 0:
            k = self.subset_k
            eff_block = k if (self.block_size <= 0 or self.block_size >= k) else self.block_size
            self.logical_pp_bits_aq_sgd = float(k * self.bits + k * _SCALE_BITS / eff_block)
        else:
            eff_block = (
                int(hidden_size) if (self.block_size <= 0 or self.block_size >= int(hidden_size)) else self.block_size
            )
            n_blocks = (int(hidden_size) + eff_block - 1) // eff_block
            self.logical_pp_bits_aq_sgd = float(int(hidden_size) * self.bits + n_blocks * _SCALE_BITS)

    def _gather_buffer(
        self,
        layer_idx: int,
        example_ids: torch.Tensor,
        position_ids: torch.Tensor,
        in_scope: torch.Tensor,
        hidden_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Assemble ``(m, warm)`` for this micro-batch's token axis.

        Loops over the DISTINCT examples present, which is at most the number of
        prompts in the micro-batch, and pulls each one's ``(n_pos, H)`` slab
        across once. Counts hits and misses in token-positions so the reported
        hit rate is the fraction of buffered TRAFFIC that had a history.
        """
        n_tokens = int(example_ids.numel())
        if not bool(in_scope.any()):
            self.buffer.misses += 0
            return None, None
        m = torch.zeros((n_tokens, hidden_size), device=device, dtype=torch.float32)
        warm = torch.zeros(n_tokens, dtype=torch.bool, device=device)
        scoped_examples = torch.unique(example_ids[in_scope])
        for ex in scoped_examples.tolist():
            entry = self.buffer.get(int(ex), layer_idx)
            if entry is None:
                continue
            values = entry[0]
            n_pos = int(values.shape[0])
            sel = in_scope & (example_ids == int(ex)) & (position_ids < n_pos)
            n_sel = int(sel.sum().item())
            if n_sel == 0:
                continue
            rows = position_ids[sel].to(device=values.device)
            m[sel] = values.index_select(0, rows).to(device=device, dtype=torch.float32)
            warm[sel] = True
        # A non-finite buffered row cannot be differenced against: the block
        # absmax becomes inf, the grid spacing with it, and the rounding returns
        # nan for every channel of the block. Demote such a row to COLD so it
        # takes the sr_quant fallback instead of poisoning the forward, and
        # count it, because a silent demotion would read as a low hit rate.
        # bf16 storage should make this unreachable; it is kept as a guard so
        # the failure mode is a slightly worse codec and never a nan loss.
        if bool(warm.any()):
            bad = warm & ~torch.isfinite(m).all(dim=1)
            n_bad = int(bad.sum().item())
            if n_bad:
                m = torch.where(bad.unsqueeze(1), torch.zeros_like(m), m)
                warm = warm & ~bad
                self.buffer.nonfinite_rows += n_bad

        n_scope = int(in_scope.sum().item())
        n_warm = int(warm.sum().item())
        self.buffer.hits += n_warm
        self.buffer.misses += n_scope - n_warm
        return m, warm

    def _stage_buffer(
        self,
        layer_idx: int,
        x_hat: torch.Tensor,
        example_ids: torch.Tensor,
        position_ids: torch.Tensor,
        sample_ids: torch.Tensor,
        in_scope: torch.Tensor,
        hidden_size: int,
    ) -> None:
        """STAGE the refreshed buffer for this visit. Does not publish it.

        Both sides of a boundary end a visit holding ``m + Q(a - m)``, which is
        exactly the reconstruction the receiver consumed, so the new buffer is
        taken from ``x_hat`` and needs no second quantization.

        Staging rather than writing is what preserves pass identity, and a
        plain "skip if already written this step" guard is NOT enough: it stops
        the second write but the second pass would then READ what the first
        pass published, so the two passes of one step would disagree. Instead
        nothing is published until :meth:`_flush_pending` runs at the next step
        boundary, so every pass of step ``t`` reads the buffer as it stood at
        the end of step ``t-1``.

        The ``G`` rollouts of one prompt disagree about the refreshed prefix,
        because the subset ``J`` is keyed on the row's ``sample_id``. The stage
        is taken from the LOWEST ``sample_id`` present for that example, which
        is deterministic and leaves one ``m`` per example as the paper
        specifies.
        """
        vals = x_hat.detach().reshape(-1, hidden_size)
        for ex in torch.unique(example_ids[in_scope]).tolist():
            ex_int = int(ex)
            key = (ex_int, int(layer_idx))
            here = in_scope & (example_ids == ex_int)
            if not bool(here.any()):
                continue
            # One representative row, so the stored prefix is self-consistent.
            row = int(sample_ids[here].min().item())
            sel = here & (sample_ids == row)
            if not bool(sel.any()):
                continue
            pos = position_ids[sel]
            n_pos = int(pos.max().item()) + 1
            slab = torch.zeros((n_pos, hidden_size), device=vals.device, dtype=vals.dtype)
            entry = self.buffer.get(ex_int, layer_idx)
            if entry is not None:
                prev = entry[0].to(device=vals.device, dtype=vals.dtype)
                keep = min(n_pos, int(prev.shape[0]))
                slab[:keep] = prev[:keep]
            slab.index_copy_(0, pos, vals[sel])
            self._pending[key] = slab.detach().to("cpu", torch.float32)

    def _flush_pending(self) -> None:
        """Publish the staged buffer. Called when the step counter advances.

        Deferring publication to a step boundary is what makes every pass of a
        step read one fixed ``m``. The final step's stage is never published,
        which costs nothing: no later pass would read it.
        """
        if not self._pending:
            return
        for (ex_int, layer_idx), slab in self._pending.items():
            self.buffer.put(ex_int, layer_idx, slab, self._staged_step)
        self._pending = {}

    def _make_hook(self, layer_idx: int):
        codec = self

        def _hook(_mod: nn.Module, _inputs: tuple, output: Any):
            h = output[0] if isinstance(output, tuple) else output
            if not torch.is_tensor(h):
                return output

            # Confinement guard, identical to the mask and quant codecs: firing
            # anywhere but an eligible path is contamination. The anchor pass
            # (tag None) must stay dense, and it must NOT touch the buffer,
            # since its weights are the forecast rather than the current ones.
            state = codec._state
            tag = None
            if state is not None and hasattr(state, "path_tag"):
                tag = state.path_tag
                eligible = mask_eligible_tags(state)
                assert tag in eligible, (
                    f"comm_eff aq_sgd fired on an ineligible path (path_tag={tag!r}, "
                    f"eligible={sorted(eligible)}); the codec is confined to the "
                    "actor-train forward (and old-logprob / reference forwards "
                    "when mask_recompute / mask_reference are true)."
                )

            hidden_size = int(h.shape[-1])
            if codec.hidden_size is None:
                codec.hidden_size = hidden_size
            codec._record_bits(hidden_size)

            n_tokens = h.numel() // hidden_size
            sample_ids = codec._sample_ids
            position_ids = codec._position_ids
            example_ids = codec._example_ids
            if sample_ids is None or position_ids is None:
                raise RuntimeError(
                    "comm_eff aq_sgd fired without per-token identity: call "
                    "set_context(sample_ids=..., position_ids=...) before each forward."
                )
            if example_ids is None:
                raise RuntimeError(
                    "comm_eff aq_sgd fired without per-token example_ids: the codec "
                    "keys its activation buffer on a PERSISTENT example identity, and "
                    "without one it would degenerate into memoryless quantization. "
                    "Call set_context(example_ids=...) (see aqsgd_example_ids)."
                )
            for name, ids in (("sample_ids", sample_ids), ("position_ids", position_ids), ("example_ids", example_ids)):
                if ids.numel() != n_tokens:
                    raise RuntimeError(
                        f"comm_eff aq_sgd token-axis mismatch: activation has {n_tokens} "
                        f"tokens but got {ids.numel()} {name} (SP>1 / non-rmpad is out of scope)."
                    )

            device = h.device
            position_ids = position_ids.to(device)
            example_ids = example_ids.to(device)
            sample_ids_dev = sample_ids.to(device)
            in_scope = codec._in_scope(position_ids)

            m, warm = codec._gather_buffer(layer_idx, example_ids, position_ids, in_scope, hidden_size, device, h.dtype)
            if m is not None and warm is not None and bool(warm.any()):
                with torch.no_grad():
                    hv = h.detach().reshape(-1, hidden_size).to(torch.float32)[warm]
                    mv = m[warm]
                    num = torch.linalg.vector_norm(hv - mv, dim=1)
                    den = torch.linalg.vector_norm(hv, dim=1).clamp_min(1e-8)
                    codec.delta_ratio_sum += float((num / den).mean().item())
                    codec.delta_ratio_count += 1

            h_tilde = BoundaryAQSGD.apply(
                h,
                m,
                warm,
                sample_ids_dev,
                position_ids,
                layer_idx,
                codec._global_step,
                codec.base_seed,
                codec.bits,
                codec.block_size,
                codec.rounding,
                codec.subset_k,
                codec.first_visit,
            )

            # Only the train pass advances the buffer. The reference forward
            # runs different weights, and the old-logprob pass would otherwise
            # race the train pass for the same visit.
            if tag == "train" and bool(in_scope.any()):
                with torch.no_grad():
                    codec._staged_step = codec._global_step
                    codec._stage_buffer(
                        layer_idx, h_tilde, example_ids, position_ids, sample_ids_dev, in_scope, hidden_size
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

        Clears the stale per-token context (but NOT the buffer, which is the
        codec's cross-step state) so a fire before ``set_context`` fails
        explicitly instead of reusing the previous forward's identities.
        """
        if self._handles:
            return
        # NOT cleared here. The engine registers and unregisters the codec once
        # per eligible PASS (forward_backward_batch registers on entry and
        # unregisters in its finally), not once per run, so clearing the stage
        # here destroyed the train pass's staged buffer before the next step's
        # set_context could publish it. The buffer and its pending stage are
        # cross-step state and outlive registration, exactly like self.buffer.
        self._sample_ids = None
        self._position_ids = None
        self._example_ids = None
        self._prompt_lens = None
        layers = find_decoder_layers(module)
        if layers is None:
            logger.warning(
                "comm_eff.activation_aqsgd: could not locate decoder layers on %s; "
                "no aq_sgd hooks registered (no-op this pass)",
                type(module).__name__,
            )
            return
        self.boundary_indices = decoder_boundary_indices(len(layers), self.pp_size)
        self._boundary_set = set(self.boundary_indices)
        for idx in self.boundary_indices:
            self._handles.append(layers[idx].register_forward_hook(self._make_hook(idx)))
        logger.info(
            "comm_eff.activation_aqsgd: registered hooks on boundaries %s "
            "(L=%d, pp_size=%d, bits=%d, block_size=%d, rounding=%s, subset_k=%d, "
            "scope=%s, first_visit=%s, capacity=%.1f GiB, buffer_device=%s)",
            self.boundary_indices,
            len(layers),
            self.pp_size,
            self.bits,
            self.block_size,
            self.rounding,
            self.subset_k,
            self.scope,
            self.first_visit,
            self.buffer.capacity_bytes / float(1024**3),
            self.buffer.device,
        )

    def unregister(self) -> None:
        """Remove all aq_sgd hooks. The buffer survives, by design."""
        for handle in self._handles:
            handle.remove()
        self._handles = []

    @property
    def is_registered(self) -> bool:
        return bool(self._handles)
