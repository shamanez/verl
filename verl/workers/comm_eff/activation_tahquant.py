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

"""TAH-Quant boundary codec: Tile-wise Adaptive Hadamard Quantization.

Port of TAH-Quant (He et al., arXiv:2506.01352v2, 8 May 2026), the activation
codec for pipeline parallelism over slow networks that supersedes AQ-SGD. It
lands here as the third member of the QUANTIZATION family, next to ``sr_quant``
(memoryless stochastic-rounding block quantization) and ``aq_sgd`` (buffered
delta coding), and it is the byte-matched arm that answers a question neither of
those can: whether spending the same budget on OUTLIER SUPPRESSION beats
spending it on more channels or on a history.

WHY IT FITS RLVR BETTER THAN AQ-SGD
-----------------------------------
It is FULLY STATELESS. Every decision (which tile is the pivot, whether to
rotate, how many bits) is computed from the current tile at send time. There is
no per-example buffer, no error feedback, no residual.

That matters here specifically. AQ-SGD's guarantee rests on an example
RECURRING so its activations can be differenced against a stored visit, and
on-policy RLVR resamples every response each step: at Pair 1 the recurring
prompt prefix is only 16.3% of boundary traffic, which is why the ``aq_sgd`` arm
has to scope to the prompt and why ``aqsgd-all`` sits near a 0.7% hit rate.
TAH-Quant removes exactly the assumption RLVR violates, and it removes the
20 GiB host buffer (and that buffer's failure modes) with it.

Two consequences fall out for free. PASS IDENTITY needs no machinery: the
reconstruction is a deterministic function of the current activation, so the
train, old-logprob and reference passes of one step agree by construction
rather than by keying, which is what keeps the PPO ratio starting at one.
(The optional stochastic-rounding variant is the one exception, and it restores
identity the same way ``sr_quant`` does, with a PRF draw keyed on
sample/position/layer/step/direction.) And there is no shared basis, so
``owns_q`` must be false: the anchor has nothing to own.

THE ALGORITHM, AND THE THREE PLACES WE DEVIATE
----------------------------------------------
Per token, the width is partitioned into contiguous tiles of ``tile`` channels
(paper Section 3.1; tiles never span tokens). For each tile:

1. ENTROPY-GUIDED BIT ALLOCATION (Section 3.2). ``p_k = |a_k| / (||a||_1 + eps)``
   and the tile's entropy ranks it against every other tile OF THE SAME SAMPLE.
   The top ``int4_frac`` by entropy get INT4, the rest INT3. High entropy earns
   more bits because a flat tile has no single channel the rotation can isolate.

   DEVIATION 1, and it is a defect in the paper. Its Eq (2) is printed as
   ``H = sum_k p_k log(p_k + zeta)``, with NO leading minus sign, which makes it
   the NEGATIVE of Shannon entropy: as printed it is maximal for a one-hot tile
   and minimal for a uniform one, the exact opposite of the paper's own prose
   ("H is high when energy is spread evenly ... and low when the tile contains a
   sharp outlier") and of the top-p% rule built on it. Verified glyph by glyph
   against the PDF at 400 dpi. We implement TRUE Shannon entropy,
   ``-sum p log(p + zeta)``, which is what the paper means everywhere it reasons
   about H.

2. CONDITIONAL HADAMARD WITH PIVOT SWAP (Section 3.3). With ``alpha^(1)`` and
   ``alpha^(2)`` the two largest magnitudes, ``r = |alpha^(1)| / (|alpha^(2)| + rho)``.
   If ``r > tau`` the tile holds an outlier: swap coordinate 0 with the argmax
   coordinate ``d`` so the pivot sits at position 0, then rotate,
   ``alpha_dot = alpha P_d H_G / sqrt(G)``. Otherwise the tile is quantized
   as-is. The rotation spreads the outlier's energy over the whole tile so the
   affine range is no longer wasted on it.

3. ASYMMETRIC AFFINE QUANTIZATION per tile, with its own scale and zero-point
   over the tile's own ``[min, max]``.

   DEVIATION 2. The paper never states a rounding mode anywhere (the terms
   "stochastic rounding", "round to nearest" and "nearest" do not appear in the
   document) and cites a QAT parameterization for the quantizer form only. Its
   Assumption 4.4 does NOT require an unbiased quantizer: it bounds relative
   bias, ``||E[g_hat] - grad f||^2 <= (1 - delta)||grad f||^2``, calibrated
   empirically at ``delta = 0.9``. Our own evidence pulls the other way, since
   issue #93 killed a round-to-nearest arm at step 60 and PRF exact-k is
   unbiased. So ``rounding`` is a knob: ``rn`` is the paper's implied
   deterministic default, ``sr`` restores unbiasedness within the tile, and the
   pair is a one-variable ablation the paper cannot answer.

DEVIATION 3, THE BUDGET, WHICH IS THE WHOLE REASON THIS IS A SUBSET CODEC
-------------------------------------------------------------------------
TAH-Quant's published operating point is 4.409 bits/element (80% INT4 + 20%
INT3 at ``G = 64``, plus 39 bits/tile of metadata). At ``H = 1536`` that is
6772.8 bits/token/boundary against PRF exact-k's 1232, so the published setting
is **5.50x our entire budget**, and the paper states INT4/INT3 is "the
lowest-bit configuration that remains consistently stable" with INT2 often
failing to converge. We already operate a factor of five below its stability
floor, and we get there by spending bits on FEW CHANNELS rather than few bits on
all of them.

So the codec runs on a PRF-keyed subset of ``subset_k`` channels per token, the
same device ``sr_quant`` and ``aq_sgd`` use, and costs nothing to signal because
both sides derive the subset from the shared key. Tiles then tile the GATHERED
width, and the final tile may be partial. Un-sent channels are zero and the
survivors carry the ``H/k`` gain, so the estimate keeps the activation's scale.

WHY ``tile = 32`` AND NOT THE PAPER'S 64
----------------------------------------
The metadata width is what decides this, and it is forced. Reproducing the
paper's 4.41 leaves a UNIQUE metadata width of 39 bits/tile, of which
scale and zero-point take 16 each. At ``G = 64`` the pivot index needs
``ceil(log2 64) = 6`` bits, and 16 + 16 + 1 + 6 = 39 exactly, which leaves NO
room for the one bit the receiver needs to know whether the rotation was
applied. The paper's own accounting therefore cannot pay for the conditional
gate its algorithm requires: it enumerates only "scale, zero-point, and bit-map
metadata" and mentions neither the pivot index nor a transform flag, though a
receiver cannot invert without both.

At ``tile = 32`` the pivot index needs only 5 bits, which frees exactly the bit
the flag costs: 16 + 16 + 1 + 5 + 1 = 39, the same width, with the conditional
gate intact and honestly paid for. Since the rotation's benefit is FLAT in tile
size (measured 3.60x / 3.72x / 3.69x error reduction at 2 bits for G = 16 / 32 /
64 on one-outlier tiles), nothing is given up, and 32 is a tile size the paper
itself ablates. Hence the default, and hence the ledger:

    242 * 3.8  +  ceil(242 / 32) * 39  =  919.6 + 312  =  1231.6 bits
    against PRF exact-k's 77 * 16 = 1232    ->  0.99968x

which is tighter parity than the ``sr_quant`` arm's 1.0004x. The three
byte-matched arms then differ only in HOW they spend one budget:

    PRF exact-k    77 channels  x 16.000 bits   ( 5.0% coverage)
    sr_quant      493 channels  x  2.500 bits   (32.1% coverage)
    tah_quant     242 channels  x  5.089 bits   (15.8% coverage)

WHAT IS NOT PORTED
------------------
The paper compresses the BACKWARD gradient with a naive fixed-point quantizer
at 6-8 bits rather than with TAH-Quant (Table 3's "fw~4 bw6"). We apply the same
codec in both directions, as ``prf_mask``, ``sr_quant`` and ``aq_sgd`` all do
here, because that is what makes the arm readable against them and against the
ledger. The paper's Appendix H shows TAH-Quant works backward too and beats
naive 4-bit there decisively, so this is the better-supported direction to
deviate in.
"""

import logging
import math
from typing import Any, Optional

import torch
import torch.nn as nn

from verl.workers.comm_eff.activation_mask import (
    decoder_boundary_indices,
    find_decoder_layers,
    prf_token_mask,
)
from verl.workers.comm_eff.activation_quant import (
    BACKWARD_DIRECTION,
    FORWARD_DIRECTION,
    ROUNDING_MODES,
    prf_token_uniform,
)
from verl.workers.comm_eff.state import mask_eligible_tags

logger = logging.getLogger(__file__)

__all__ = [
    "ActivationTAHQuant",
    "BoundaryTAHQuant",
    "TAHQUANT_INT4_BITS",
    "TAHQUANT_INT3_BITS",
    "tahquant_hadamard",
    "tahquant_metadata_bits",
    "tahquant_reconstruct",
    "tahquant_tile_entropy",
]

# The two precisions the entropy allocator hands out. Fixed by the paper's
# design ("assign INT4 to the top-p% highest-entropy tiles and INT3 to the
# rest"); the SPLIT between them is the tunable, not the levels themselves.
TAHQUANT_INT4_BITS = 4
TAHQUANT_INT3_BITS = 3

# Per-tile wire metadata, and the width is FORCED rather than chosen: see the
# module docstring. fp16 scale + fp16 zero-point + 1-bit precision bitmap +
# ceil(log2 tile) pivot index + 1-bit "was the tile rotated" flag.
_SCALE_ZP_BITS = 16
_BITMAP_BITS = 1
_TRANSFORM_FLAG_BITS = 1

# Guards from the paper: epsilon in the L1 normalisation of Eq (1), zeta inside
# the log of Eq (2), rho in the outlier ratio of Eq (3).
_EPS_L1 = 1e-12
_ZETA_LOG = 1e-12
_RHO_RATIO = 1e-9


def tahquant_metadata_bits(tile: int) -> int:
    """Metadata bits per tile, the honest ledger including the transform flag.

    At ``tile = 32`` this is 39, the same width the paper's own 4.41 bits/element
    reduces to, but with the conditional gate paid for. At ``tile = 64`` it is
    40, one bit above what the paper's figure can accommodate, which is the
    arithmetic showing the published accounting omits a field its algorithm
    needs.
    """
    if tile <= 1:
        raise ValueError(f"tah_quant tile must be >= 2 to carry a pivot index; got {tile!r}")
    pivot_bits = int(math.ceil(math.log2(int(tile))))
    return 2 * _SCALE_ZP_BITS + _BITMAP_BITS + pivot_bits + _TRANSFORM_FLAG_BITS


def tahquant_hadamard(size: int, *, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """The ``+/-1`` Sylvester Hadamard matrix, ``H H^T = size * I``.

    Cached per (size, device, dtype): it is a constant both sides of a boundary
    can hardcode, so it never goes on the wire. Sylvester construction needs a
    power of two, which the paper assumes silently and the validator enforces.
    """
    key = (int(size), str(device), str(dtype))
    cached = _HADAMARD_CACHE.get(key)
    if cached is not None:
        return cached
    n = int(size)
    if n < 1 or (n & (n - 1)) != 0:
        raise ValueError(f"tah_quant tile must be a power of two for the Hadamard; got {size!r}")
    h = torch.ones((1, 1), device=device, dtype=dtype)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0)
    _HADAMARD_CACHE[key] = h
    return h


_HADAMARD_CACHE: dict = {}


def tahquant_tile_entropy(tiles: torch.Tensor, valid: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Shannon entropy of the normalised magnitude profile, Eq (1) and Eq (2).

    ``-sum_k p_k log(p_k + zeta)`` with ``p_k = |a_k| / (||a||_1 + eps)``. The
    SIGN is ours: the paper prints Eq (2) without the minus, which inverts it
    against its own prose and against the top-p% rule. See the module docstring.

    Args:
        tiles: ``(..., W)`` fp32, one tile per row.
        valid: optional ``(W,)`` bool marking real channels, for a partial tile.
            Invalid channels contribute no mass and no entropy.

    Returns:
        ``(...)`` fp32 entropy, high for a flat tile and low for a spiky one.
    """
    mag = tiles.abs()
    if valid is not None:
        mag = mag * valid.to(mag.dtype)
    denom = mag.sum(dim=-1, keepdim=True) + _EPS_L1
    p = mag / denom
    terms = p * torch.log(p + _ZETA_LOG)
    if valid is not None:
        terms = terms * valid.to(terms.dtype)
    return -terms.sum(dim=-1)


def _asymmetric_quantize(
    tiles: torch.Tensor,
    *,
    levels: torch.Tensor,
    valid: Optional[torch.Tensor],
    rounding: str,
    u: Optional[torch.Tensor],
) -> torch.Tensor:
    """Per-tile asymmetric affine quantization onto ``levels + 1`` points.

    ``scale = (max - min) / levels`` and ``zero = min``, both per tile, which is
    the two fp16 fields the ledger pays for. ``levels`` is a per-tile tensor
    because the allocator hands different tiles different precisions.

    ``rounding="rn"`` is deterministic round-to-nearest, the paper's implied
    mode and a BIASED estimator. ``rounding="sr"`` rounds stochastically against
    the supplied PRF uniform, which makes ``E[q] = a`` exactly within the tile,
    at the cost of variance. A partial tile's invalid channels are excluded from
    the range and returned as zero.
    """
    if valid is not None:
        mask = valid.to(torch.bool)
        big = torch.finfo(tiles.dtype).max
        lo = torch.where(mask, tiles, torch.full_like(tiles, big)).amin(dim=-1, keepdim=True)
        hi = torch.where(mask, tiles, torch.full_like(tiles, -big)).amax(dim=-1, keepdim=True)
    else:
        lo = tiles.amin(dim=-1, keepdim=True)
        hi = tiles.amax(dim=-1, keepdim=True)

    lv = levels.to(tiles.dtype).unsqueeze(-1)
    scale = ((hi - lo) / lv).clamp_min(1e-12)
    t = (tiles - lo) / scale

    if rounding == "sr":
        floor = torch.floor(t)
        frac = (t - floor).clamp_(0.0, 1.0)
        draw = u if u is not None else torch.rand_like(t)
        idx = floor + (draw.to(t.dtype) < frac).to(t.dtype)
    else:
        idx = torch.round(t)
    idx = idx.clamp_(torch.zeros_like(lv), lv)

    out = idx * scale + lo
    if valid is not None:
        out = torch.where(valid.to(torch.bool), out, torch.zeros_like(out))
    return out


def tahquant_reconstruct(
    x: torch.Tensor,
    *,
    sample_ids: torch.Tensor,
    position_ids: torch.Tensor,
    layer_idx: int,
    global_step: int,
    base_seed: int,
    tile: int = 32,
    int4_frac: float = 0.8,
    tau: float = 2.0,
    rounding: str = "rn",
    subset_k: int = 0,
    direction: int = FORWARD_DIRECTION,
    stats: Optional[dict] = None,
) -> torch.Tensor:
    """Reconstruct ``x`` as the receiver of a TAH-Quant wire would.

    Encoder and decoder are fused, as in every codec here: there is no real
    network, so the point is to make the tensor the training graph sees be
    EXACTLY what a receiver could rebuild from the priced wire, and nothing
    more. Concretely, every quantity used below is either on the ledger (the
    payload, the per-tile scale and zero-point, the precision bit, the pivot
    index, the transform flag) or derivable from the shared PRF key (the channel
    subset, and the stochastic-rounding draw).

    Args:
        x: ``(..., H)`` activations at a boundary, any float dtype.
        sample_ids: ``(N,)`` per-token row identity, for the PRF key.
        position_ids: ``(N,)`` per-token position, for the PRF key.
        layer_idx: boundary index, part of the PRF key.
        global_step: optimizer step, part of the PRF key.
        base_seed: run seed.
        tile: contiguous channels per tile, a power of two.
        int4_frac: fraction of tiles, ranked by entropy WITHIN EACH SAMPLE, that
            receive INT4; the rest receive INT3.
        tau: outlier ratio above which a tile is rotated. ``0`` rotates always,
            ``inf`` never (the paper's own two ablation endpoints).
        rounding: ``rn`` (deterministic, biased) or ``sr`` (unbiased in-tile).
        subset_k: channels sent per token; ``0`` sends the full width.
        direction: forward or backward, keys the stochastic draw apart.
        stats: optional sink; when given, receives the tile counts that make the
            mechanism observable (``tiles``, ``fired``, ``int4``). Without it the
            arm can report a score but not a reason.

    Returns:
        The reconstruction, in ``x``'s original shape and dtype.
    """
    if rounding not in ROUNDING_MODES:
        raise ValueError(f"tah_quant rounding must be one of {ROUNDING_MODES}; got {rounding!r}")
    orig_shape = x.shape
    orig_dtype = x.dtype
    hidden_size = int(orig_shape[-1])
    v = x.reshape(-1, hidden_size).to(torch.float32)
    n_tokens = int(v.shape[0])

    if int(sample_ids.numel()) != n_tokens or int(position_ids.numel()) != n_tokens:
        raise RuntimeError(
            f"tah_quant needs per-token identity: got {n_tokens} rows but "
            f"sample_ids={int(sample_ids.numel())} position_ids={int(position_ids.numel())}"
        )
    if subset_k < 0 or subset_k > hidden_size:
        raise ValueError(f"tah_quant subset_k must be in [0, {hidden_size}]; got {subset_k!r}")

    # ---- the PRF-keyed channel subset, priced at zero because it is derived --
    if subset_k > 0:
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
        width = int(subset_k)
        w = v[keep].reshape(n_tokens, width)
        u_full = None
        if rounding == "sr":
            u_full = prf_token_uniform(
                sample_ids,
                position_ids,
                layer_idx=layer_idx,
                global_step=global_step,
                base_seed=base_seed,
                hidden_size=hidden_size,
                direction=direction,
                device=v.device,
            )[keep].reshape(n_tokens, width)
    else:
        kept_idx = None
        width = hidden_size
        w = v
        u_full = None
        if rounding == "sr":
            u_full = prf_token_uniform(
                sample_ids,
                position_ids,
                layer_idx=layer_idx,
                global_step=global_step,
                base_seed=base_seed,
                hidden_size=hidden_size,
                direction=direction,
                device=v.device,
            )

    # ---- tile the GATHERED width; the final tile may be partial -------------
    g = int(tile)
    n_tiles = (width + g - 1) // g
    padded = n_tiles * g
    valid = None
    if padded != width:
        pad = padded - width
        w = torch.cat([w, torch.zeros((n_tokens, pad), device=w.device, dtype=w.dtype)], dim=1)
        if u_full is not None:
            u_full = torch.cat([u_full, torch.zeros((n_tokens, pad), device=u_full.device, dtype=u_full.dtype)], dim=1)
        valid = torch.ones(padded, device=w.device, dtype=torch.bool)
        valid[width:] = False
        valid = valid.reshape(n_tiles, g)

    tiles = w.reshape(n_tokens, n_tiles, g)
    u_tiles = u_full.reshape(n_tokens, n_tiles, g) if u_full is not None else None
    valid_t = valid.unsqueeze(0) if valid is not None else None

    # ---- 1. entropy-guided bit allocation, ranked WITHIN EACH SAMPLE -------
    ent = tahquant_tile_entropy(tiles, valid_t)  # (N, n_tiles)
    n_int4 = int(round(float(int4_frac) * n_tiles))
    n_int4 = max(0, min(n_tiles, n_int4))
    levels = torch.full((n_tokens, n_tiles), float(2**TAHQUANT_INT3_BITS - 1), device=w.device, dtype=torch.float32)
    if n_int4 > 0:
        top = ent.topk(n_int4, dim=1).indices
        levels.scatter_(1, top, float(2**TAHQUANT_INT4_BITS - 1))

    # ---- 2. conditional pivot swap + Hadamard rotation ---------------------
    mag = tiles.abs()
    if valid_t is not None:
        mag = torch.where(valid_t, mag, torch.zeros_like(mag))
    if g >= 2:
        top2 = mag.topk(2, dim=-1).values
        ratio = top2[..., 0] / (top2[..., 1] + _RHO_RATIO)
    else:  # pragma: no cover - the validator forbids tile < 2
        ratio = torch.zeros_like(mag[..., 0])
    fire = ratio > float(tau)

    pivot = mag.argmax(dim=-1)  # the index the wire spends its 5 bits on
    swapped = _swap_with_first(tiles, pivot)
    h_g = tahquant_hadamard(g, device=w.device, dtype=torch.float32)
    rotated = torch.matmul(swapped, h_g) / math.sqrt(float(g))

    # A rotated tile is dense by construction, so its whole width is real even
    # when the source tile was partial: the padding's zeros have been mixed in.
    q_rot = _asymmetric_quantize(rotated, levels=levels, valid=None, rounding=rounding, u=u_tiles)
    back = torch.matmul(q_rot, h_g.transpose(0, 1)) / math.sqrt(float(g))
    back = _swap_with_first(back, pivot)
    if valid_t is not None:
        back = torch.where(valid_t, back, torch.zeros_like(back))

    q_raw = _asymmetric_quantize(tiles, levels=levels, valid=valid_t, rounding=rounding, u=u_tiles)

    out_tiles = torch.where(fire.unsqueeze(-1), back, q_raw)

    if stats is not None:
        stats["tiles"] = n_tokens * n_tiles
        stats["fired"] = int(fire.sum().item())
        stats["int4"] = int((levels > float(2**TAHQUANT_INT3_BITS - 1)).sum().item())

    # ---- 3. scatter the subset back, carrying the H/k gain ------------------
    flat = out_tiles.reshape(n_tokens, padded)[:, :width]
    if kept_idx is None:
        x_hat = flat
    else:
        gain = float(hidden_size) / float(width)
        x_hat = torch.zeros_like(v)
        x_hat.scatter_(1, kept_idx, flat * gain)
    return x_hat.reshape(orig_shape).to(orig_dtype)


def _swap_with_first(tiles: torch.Tensor, pivot: torch.Tensor) -> torch.Tensor:
    """Swap coordinate 0 with coordinate ``pivot`` in every tile.

    ``P_d`` from the paper, and its own inverse, so the same call undoes it.
    Written as a gather rather than in-place indexing so it composes with
    autograd and stays one kernel over ``(N, n_tiles, G)``.
    """
    g = int(tiles.shape[-1])
    idx = torch.arange(g, device=tiles.device).expand(tiles.shape).clone()
    p = pivot.unsqueeze(-1)
    zero = torch.zeros_like(p)
    idx.scatter_(-1, zero, p)
    idx.scatter_(-1, p, zero)
    return tiles.gather(-1, idx)


class BoundaryTAHQuant(torch.autograd.Function):
    """TAH-Quant on the forward activation and on the backward gradient.

    Both directions run the same codec, which is the house convention here and
    what makes the arm readable against ``prf_mask`` / ``sr_quant`` / ``aq_sgd``.
    The paper instead sends gradients through a naive 6-8 bit fixed-point
    compressor; its Appendix H shows TAH-Quant is the stronger choice at low
    precision, so this deviation is the better-supported direction.

    The backward wire is keyed with ``BACKWARD_DIRECTION`` so a stochastic draw
    is independent of the forward one at the same (token, channel), while the
    CHANNEL SUBSET is deliberately keyed identically in both directions and
    across all passes of one step: the receiver of the gradient must be able to
    place it on the same coordinates the activation came from.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        h: torch.Tensor,
        sample_ids: torch.Tensor,
        position_ids: torch.Tensor,
        layer_idx: int,
        global_step: int,
        base_seed: int,
        tile: int,
        int4_frac: float,
        tau: float,
        rounding: str,
        subset_k: int,
        stats: Optional[dict],
    ) -> torch.Tensor:
        ctx.save_for_backward(sample_ids, position_ids)
        ctx.tahquant_key = (
            int(layer_idx),
            int(global_step),
            int(base_seed),
            int(tile),
            float(int4_frac),
            float(tau),
            str(rounding),
            int(subset_k),
        )
        return tahquant_reconstruct(
            h,
            sample_ids=sample_ids,
            position_ids=position_ids,
            layer_idx=int(layer_idx),
            global_step=int(global_step),
            base_seed=int(base_seed),
            tile=int(tile),
            int4_frac=float(int4_frac),
            tau=float(tau),
            rounding=str(rounding),
            subset_k=int(subset_k),
            direction=FORWARD_DIRECTION,
            stats=stats,
        )

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        sample_ids, position_ids = ctx.saved_tensors
        layer_idx, global_step, base_seed, tile, int4_frac, tau, rounding, subset_k = ctx.tahquant_key
        g_hat = tahquant_reconstruct(
            grad_output,
            sample_ids=sample_ids,
            position_ids=position_ids,
            layer_idx=layer_idx,
            global_step=global_step,
            base_seed=base_seed,
            tile=tile,
            int4_frac=int4_frac,
            tau=tau,
            rounding=rounding,
            subset_k=subset_k,
            direction=BACKWARD_DIRECTION,
        )
        # One None per forward arg after ctx, less the one we return a grad
        # for. Written as (None,) * N so the count is stated once and a new
        # forward argument fails loudly rather than silently dropping a grad.
        return (g_hat,) + (None,) * 11


class ActivationTAHQuant:
    """Installs :class:`BoundaryTAHQuant` at simulated pipeline-stage boundaries.

    Exposes the same ``register`` / ``set_context`` / ``unregister`` /
    ``telemetry`` surface as the masker, the quantizer and the aq_sgd codec, so
    ``CommEffState.per_token_codec`` drives it through the identical engine path.
    Holds no cross-step state, so ``register`` and ``unregister`` may be called
    once per PASS (which the engine does) with no consequence at all.
    """

    def __init__(
        self,
        *,
        tile: int = 32,
        int4_frac: float = 0.8,
        tau: float = 2.0,
        rounding: str = "rn",
        subset_k: int = 0,
        base_seed: int = 0,
        pp_size: int = 8,
        state: Any = None,
    ) -> None:
        # Re-validated here because codecs are constructed outside Hydra too.
        if int(tile) < 2 or (int(tile) & (int(tile) - 1)) != 0:
            raise ValueError(f"tah_quant tile must be a power of two >= 2; got {tile!r}")
        if not 0.0 <= float(int4_frac) <= 1.0:
            raise ValueError(f"tah_quant int4_frac must be in [0, 1]; got {int4_frac!r}")
        if float(tau) < 0.0:
            raise ValueError(f"tah_quant tau must be >= 0 (0 rotates always); got {tau!r}")
        if rounding not in ROUNDING_MODES:
            raise ValueError(f"tah_quant rounding must be one of {ROUNDING_MODES}; got {rounding!r}")
        if int(subset_k) < 0:
            raise ValueError(f"tah_quant subset_k must be >= 0; got {subset_k!r}")
        if int(pp_size) < 2:
            raise ValueError(f"tah_quant pp_size must be >= 2; got {pp_size!r}")

        self.tile = int(tile)
        self.int4_frac = float(int4_frac)
        self.tau = float(tau)
        self.rounding = str(rounding)
        self.subset_k = int(subset_k)
        self.base_seed = int(base_seed)
        self.pp_size = int(pp_size)
        self.state = state

        self._handles: list = []
        self._global_step = 0
        self._sample_ids: Optional[torch.Tensor] = None
        self._position_ids: Optional[torch.Tensor] = None
        self.hidden_size: Optional[int] = None
        self.logical_pp_bits_tah_quant: Optional[float] = None
        self.rotations_fired = 0
        self.tiles_seen = 0
        self.int4_tiles = 0

    @property
    def is_registered(self) -> bool:
        return bool(self._handles)

    def register(self, module: nn.Module) -> None:
        if self._handles:
            return
        # Stateless codec, so the ONLY thing to clear is the per-token context,
        # which must not leak across passes.
        self._sample_ids = None
        self._position_ids = None
        layers = find_decoder_layers(module)
        if layers is None:
            logger.warning("comm_eff: tah_quant found no decoder layers; boundary codec not installed")
            return
        for layer_idx in decoder_boundary_indices(len(layers), self.pp_size):
            self._handles.append(layers[layer_idx].register_forward_hook(self._make_hook(layer_idx)))

    def unregister(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def set_context(
        self,
        *,
        global_step: int,
        sample_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        **_ignored: Any,
    ) -> None:
        """Publish the per-micro-batch token identity the PRF key needs.

        Accepts and ignores the extra kwargs other codecs take, so the engine's
        one call site serves every codec. ``None`` ids are legal: the anchor's
        FRLR path calls it that way and the hook then refuses to fire.
        """
        self._global_step = int(global_step)
        self._sample_ids = sample_ids
        self._position_ids = position_ids

    def _record_bits(self, width: int) -> None:
        """The money ledger: bits per token per boundary actually on the wire."""
        k = self.subset_k if self.subset_k > 0 else int(width)
        n_tiles = (k + self.tile - 1) // self.tile
        payload = float(k) * (self.int4_frac * TAHQUANT_INT4_BITS + (1.0 - self.int4_frac) * TAHQUANT_INT3_BITS)
        self.logical_pp_bits_tah_quant = payload + float(n_tiles) * float(tahquant_metadata_bits(self.tile))

    def telemetry(self) -> dict:
        """The measurements that EXPLAIN the arm, not just score it.

        ``rotation_rate`` is the fraction of tiles whose outlier ratio cleared
        ``tau``, which is the paper's mechanism either engaging or not: near 0
        means the activations have no dominant channel per tile and the arm has
        degenerated to plain asymmetric quantization, near 1 means the gate is
        always on and ``tau`` is doing no work. ``int4_share`` should track
        ``int4_frac`` and is the cheap check that the allocator ran at all.
        """
        out = {
            "tah_quant/tiles_seen": float(self.tiles_seen),
            "tah_quant/rotation_rate": (self.rotations_fired / self.tiles_seen) if self.tiles_seen else 0.0,
            "tah_quant/int4_share": (self.int4_tiles / self.tiles_seen) if self.tiles_seen else 0.0,
        }
        if self.logical_pp_bits_tah_quant is not None:
            out["tah_quant/logical_pp_bits"] = float(self.logical_pp_bits_tah_quant)
        return out

    def _make_hook(self, layer_idx: int):
        codec = self

        def _hook(_mod: nn.Module, _inputs: tuple, output: Any):
            h = output[0] if isinstance(output, tuple) else output
            if not torch.is_tensor(h):
                return output

            state = codec.state
            tag = getattr(state, "path_tag", None) if state is not None else None
            if state is not None:
                eligible = mask_eligible_tags(state)
                assert tag in eligible, (
                    f"tah_quant hook fired on path_tag={tag!r}, which is not in {sorted(map(str, eligible))}. "
                    "The anchor circuit and the dense-view probe must never see the codec."
                )

            sample_ids = codec._sample_ids
            position_ids = codec._position_ids
            if sample_ids is None or position_ids is None:
                raise RuntimeError(
                    "tah_quant hook fired without per-token identity; "
                    "set_context(global_step=..., sample_ids=..., position_ids=...) must precede the forward"
                )
            hidden_size = int(h.shape[-1])
            n_tokens = int(h.reshape(-1, hidden_size).shape[0])
            if int(sample_ids.numel()) != n_tokens:
                raise RuntimeError(f"tah_quant hook got {n_tokens} token rows but {int(sample_ids.numel())} sample_ids")
            if codec.subset_k > hidden_size:
                raise RuntimeError(
                    f"tah_quant subset_k={codec.subset_k} exceeds hidden_size={hidden_size} at boundary {layer_idx}"
                )
            codec.hidden_size = hidden_size
            codec._record_bits(hidden_size)

            stats: dict = {}
            h_tilde = BoundaryTAHQuant.apply(
                h,
                sample_ids.to(h.device),
                position_ids.to(h.device),
                layer_idx,
                codec._global_step,
                codec.base_seed,
                codec.tile,
                codec.int4_frac,
                codec.tau,
                codec.rounding,
                codec.subset_k,
                stats,
            )
            codec.tiles_seen += int(stats.get("tiles", 0))
            codec.rotations_fired += int(stats.get("fired", 0))
            codec.int4_tiles += int(stats.get("int4", 0))
            if state is not None and hasattr(state, "note_mask_application"):
                state.note_mask_application()
            if isinstance(output, tuple):
                return (h_tilde,) + tuple(output[1:])
            return h_tilde

        return _hook
