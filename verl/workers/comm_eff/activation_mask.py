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

from verl.workers.comm_eff.powersgd_activation import init_basis, orthonormalize
from verl.workers.comm_eff.state import (
    TRAIN_TAG,
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
# FRLR residual-norm matching floor: gamma = ||res|| / max(||scatter_J(res_J)||, eps).
_FRLR_NORM_EPS = 1e-8


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
    exact_k: bool = False,
    antithetic: bool = False,
    exact_keep: Optional[int] = None,
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
        exact_k: When ``True`` (issue #89 lever 2, default off) keep EXACTLY
            ``round((1-p)*H)`` channels per token, selected by the per-token PRF
            hash order statistic (the ``k`` channels with the largest hash) —
            RANDOM, never a top-k on activation values. This removes the
            per-token Bernoulli variance so ``comm_eff/mask_ratio == 1 - k/H``
            exactly, with no magnitude bias (the hash is value-independent).
        antithetic: When ``True`` (issue #89 lever 5, default off) the SAME
            uniform draw is used for both steps of an antithetic pair
            (``global_step`` ``2k`` and ``2k+1``) and flipped (``u -> 1-u``) on
            the odd step, so the kept set at ``t+1`` is the antithetic complement
            of ``t``'s: DISJOINT tails, keep FRACTION preserved (this is NOT a
            set complement, which would flip the mask ratio to ``1-p``). The
            within-step mask is unchanged across the old/train/reference forwards
            because only ``global_step`` (never the forward tag) enters the key.
        exact_keep: Optional explicit keep count for the exact-k order statistic
            (requires ``exact_k=True``). Used by the FRLR codec to draw its
            PRF-fresh residual subset ``J`` of exactly ``frlr_k`` channels with
            the SAME key as the baseline mask (``p`` then plays no role in the
            count). ``None`` (default) keeps ``round((1-p)*H)``.

    With ``exact_k=False``, ``antithetic=False`` and ``exact_keep=None`` (the
    defaults) the output is byte-identical to the frozen baseline PRF codec.

    Returns:
        ``(N, hidden_size)`` mask of ``{0, 1}`` in ``dtype`` on ``device``.
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"mask p must be in [0, 1]; got {p}")
    if exact_keep is not None:
        if not exact_k:
            raise ValueError("mask exact_keep requires exact_k=True (it overrides the order-statistic keep count)")
        if not 0 <= int(exact_keep) <= int(hidden_size):
            raise ValueError(f"mask exact_keep must be in [0, hidden_size={hidden_size}]; got {exact_keep}")
    sid = sample_ids.reshape(-1).to(device=device, dtype=torch.int64)
    pos = position_ids.reshape(-1).to(device=device, dtype=torch.int64)
    if sid.numel() != pos.numel():
        raise ValueError(f"sample_ids and position_ids length mismatch: {sid.numel()} vs {pos.numel()}")

    # Antithetic pairing (lever 5). Within an antithetic pair the draw is shared
    # (keyed on the pair index global_step//2) and flipped on the odd step; the
    # off-path (antithetic=False) keys on global_step directly so the key
    # derivation stays byte-identical to the baseline.
    if antithetic:
        step_key = global_step // 2
        flip = (global_step % 2) == 1
    else:
        step_key = global_step
        flip = False

    # Fold the scalar prefix, then the per-token ids, then the channel index.
    # Left-fold equivalence makes this bit-identical to _derive_seed over the
    # full key tuple per (token, channel).
    prefix = _u64_to_i64(_derive_seed((base_seed, layer_idx, step_key)))
    acc = _splitmix64_tensor(sid ^ prefix)  # fold sample_id   -> (N,)
    acc = _splitmix64_tensor(acc ^ pos)  # fold position_id -> (N,)
    channel = torch.arange(hidden_size, device=device, dtype=torch.int64)
    h = _splitmix64_tensor(acc.unsqueeze(1) ^ channel.unsqueeze(0))  # (N, H)

    # (top-53-bit uniform) per (token, channel), in integer space (no float tile).
    hash53 = _logical_rshift(h, 11)
    if flip:
        # Antithetic uniform u -> 1-u, staying in [0, 2**53): preserves the
        # keep-count and lands on the complementary tail on the odd step.
        hash53 = (_PRF_2POW53 - 1) - hash53

    n_tokens = sid.numel()
    if exact_k:
        # Keep EXACTLY round((1-p)*H) channels per token by the hash order
        # statistic (random, not value top-k). scatter is exactly-k safe even
        # under (astronomically unlikely) 53-bit ties. exact_keep (FRLR's J)
        # overrides the count without touching the key.
        keep = int(exact_keep) if exact_keep is not None else int(round((1.0 - p) * hidden_size))
        if keep <= 0:
            return torch.zeros((n_tokens, hidden_size), device=device, dtype=dtype)
        if keep >= hidden_size:
            return torch.ones((n_tokens, hidden_size), device=device, dtype=dtype)
        topk_idx = torch.topk(hash53, keep, dim=-1).indices  # (N, keep)
        mask = torch.zeros((n_tokens, hidden_size), device=device, dtype=dtype)
        mask.scatter_(1, topk_idx, 1.0)
        return mask

    # keep iff (top-53-bit uniform) >= p, done in integer space (no float tile).
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
        exact_k: bool = False,
        antithetic: bool = False,
        p_by_boundary: Optional[list] = None,
        frlr: bool = False,
        frlr_rank: int = 32,
        frlr_k: int = 44,
        frlr_unbiased: bool = False,
        frlr_q_cadence: int = 1,
        anchor_owns_q: bool = False,
        state: Any = None,
    ):
        self.p = float(p)
        self.base_seed = int(base_seed)
        self.pp_size = int(pp_size)
        # Issue #89 codec levers, all default-off so the baseline PRF stays
        # bit-identical. exact_k / antithetic are passed straight to
        # prf_token_mask; p_by_boundary assigns a per-boundary masked fraction
        # (its length must equal the boundary count, checked in register()).
        self.exact_k = bool(exact_k)
        self.antithetic = bool(antithetic)
        if p_by_boundary is None:
            self.p_by_boundary: list = []
        else:
            self.p_by_boundary = [float(v) for v in p_by_boundary]
        for v in self.p_by_boundary:
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"mask p_by_boundary entries must be in [0, 1]; got {v}")
        # boundary_idx -> p, built in register() once the boundaries are known.
        self._p_for_layer: dict[int, float] = {}
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
        # FRLR (issue #89, "32+44+1"): fresh-residual low-rank codec, default
        # off. When on, the boundary transform becomes
        #   h_hat = l + gamma * scatter_J(res_J)      (l = (h@Q)@Q^T, res = h-l)
        # with a step-frozen activation-derived Q (H x frlr_rank), a per-token
        # PRF-fresh EXACT-k residual subset J (frlr_k channels, keyed like the
        # baseline mask INCLUDING global_step), and a DETACHED per-token
        # residual-norm-matching gamma capped at H/k (frlr_unbiased instead
        # uses the constant H/k gain, E[h_hat | h, Q] = h). Payload per token:
        # frlr_rank + frlr_k + 1 values (the +1 is the norm scalar).
        self.frlr = bool(frlr)
        self.frlr_rank = int(frlr_rank)
        self.frlr_k = int(frlr_k)
        self.frlr_unbiased = bool(frlr_unbiased)
        # Slow-Q lever (issue #89): refresh Q only when at least frlr_q_cadence
        # global steps elapsed since the last refresh; 1 = the original
        # every-step refresh (bit-identical). Between refreshes Q stays frozen
        # while the activation sketch keeps accumulating over the full window.
        # Motivation: the first FRLR GPU trial cut codec-view entropy 63% but
        # its reference-KL accelerated (0.005@9 -> 0.33@30) because the
        # per-step activation-refit Q chases the drifting policy; a slow
        # cadence keeps the core stable between refreshes while the fresh
        # per-step PRF residual keeps repairing the stale-Q nullspace.
        self.frlr_q_cadence = int(frlr_q_cadence)
        if self.frlr_rank < 1:
            raise ValueError(f"mask frlr_rank must be >= 1; got {frlr_rank}")
        if self.frlr_k < 1:
            raise ValueError(f"mask frlr_k must be >= 1; got {frlr_k}")
        if self.frlr_q_cadence < 1:
            raise ValueError(f"mask frlr_q_cadence must be >= 1; got {frlr_q_cadence}")
        if self.frlr:
            if self.exact_k or self.antithetic or self.p_by_boundary:
                raise ValueError(
                    "mask frlr=true is mutually exclusive with exact_k/antithetic/p_by_boundary; "
                    "FRLR draws its own PRF-fresh exact-k residual subset J."
                )
            if self.rescale_mode != "none":
                raise ValueError(
                    "mask frlr=true requires the plain-mask rescale path OFF "
                    "(rescale=false, rescale_mode none|auto); FRLR applies its own "
                    f"detached residual-norm matching. Got rescale_mode={self.rescale_mode!r}."
                )
        # Anchor-owned Q (issue #93). When true the FAST path never touches the
        # FRLR basis: it neither folds activations into the sketch nor refreshes
        # Q. Both happen exclusively inside the anchor's dense stale-weight
        # forward, and the refresh fires only when the anchor fires, which is
        # the same governance PowerSGD uses (anchor_owns_q). This puts the Q
        # side channel on the slow circuit, so it stops being charged against
        # the boundary wire budget, and it makes the codec view stationary
        # between anchor fires instead of chasing the policy every step.
        self.anchor_owns_q = bool(anchor_owns_q)
        if self.anchor_owns_q and not self.frlr:
            raise ValueError(
                "mask anchor_owns_q=true requires frlr=true: the plain PRF mask has no "
                "basis Q for the anchor to own (its mask is a PRF of seed/step/layer)."
            )
        # True only while the anchor's clean stale-weight forward is running on
        # the no-hook clone. Routes the sketch gate (see
        # _should_accumulate_frlr_sketch) and makes the boundary hook return the
        # RAW activation so the anchor gradient stays uncompressed.
        self._anchor_sketch_mode = False
        # FRLR runtime state. The per-boundary fp32 basis Q is FROZEN within a
        # global step and refreshed lazily at the first fire of a step that is
        # >= frlr_q_cadence steps past the last refresh, from the activation
        # sketch V = sum h^T (h Q) accumulated over the whole frozen window
        # (warm-started block power iteration, mirroring the PowerSGD
        # projector, whose Q_{t+1} is likewise built from step t's
        # activations). All of this persists across register/unregister
        # cycles, like the PowerSGD basis. _frlr_q_step is the step of the
        # last refresh attempt (bootstrap counts), which anchors the cadence.
        self._frlr_basis: dict[int, torch.Tensor] = {}
        # Staged anchor candidate (issue #93). Under anchor ownership the anchor
        # fires INSIDE train_batch, AFTER this step's old_log_probs were already
        # recomputed. Publishing Q immediately therefore makes the old-logprob and
        # the train forward of the SAME step see DIFFERENT bases, so the PPO ratio
        # deviates from 1 for a reason that is NOT a policy change, and PPO clips
        # it. Measured before this was fixed: actor/pg_clipfrac spiked to 0.19-0.37
        # at exactly every anchor step in a9/a10/c600 and is identically 0 across
        # all 200 steps of the fast-Q arms a7/a8. So the candidate is STAGED here
        # and published only once every PPO minibatch sharing those old_log_probs
        # has completed, which is what the PowerSGD anchor path already does.
        self._pending_frlr_basis: dict[int, torch.Tensor] = {}
        self.frlr_basis_generation = 0
        self.frlr_q_activations = 0
        self._frlr_q_step: dict[int, int] = {}
        self._frlr_sketch: dict[int, torch.Tensor] = {}
        self._frlr_sketched_this_gen: dict[int, int] = {}
        self._frlr_fwd_generation = 0
        self.frlr_q_refreshes = 0
        # Most recent FRLR kept-coords/token (rank + k + 1), for metrics.
        self.frlr_payload_per_token: Optional[float] = None
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
        if self.frlr:
            # One generation per micro-batch: dedupes the FRLR Q sketch against
            # gradient-checkpoint recompute (the recompute re-runs the boundary
            # forward with the SAME context; same pattern as PowerSGD).
            self._frlr_fwd_generation += 1

    # ------------------------------------------------------------------ #
    # FRLR (issue #89): fresh-residual low-rank codec
    # ------------------------------------------------------------------ #
    def _frlr_ensure_basis(self, layer_idx: int, *, hidden_size: int, device: torch.device) -> torch.Tensor:
        """Return the step-frozen fp32 FRLR basis ``Q`` for ``layer_idx``.

        Bootstrap: the deterministic seeded orthonormal frame (same
        construction as the PowerSGD projector's ``init_basis``). Refresh: at
        the FIRST fire of a ``global_step`` at least ``frlr_q_cadence`` steps
        past the last refresh, the activation sketch ``V = sum h^T (h Q)``
        accumulated over the WHOLE frozen window is orthonormalized into the
        new ``Q`` (warm-started block power iteration; PowerSGD likewise
        builds ``Q_{t+1}`` from step ``t``'s activations at end-of-step).
        ``frlr_q_cadence=1`` reproduces the original every-step refresh
        bit-identically; a larger cadence keeps ``Q`` bitwise FROZEN across
        the intermediate steps while the sketch keeps accumulating, so the
        codec view stays stationary between refreshes. Within one
        ``global_step`` the basis is never touched, so the old/train/reference
        forwards and any gradient-checkpoint recompute of the same step see
        the identical ``Q``.
        """
        step = int(self._global_step)
        q = self._frlr_basis.get(layer_idx)
        last_refresh = self._frlr_q_step.get(layer_idx, step)
        if q is None:
            q = init_basis(
                hidden_size=hidden_size,
                rank=self.frlr_rank,
                base_seed=self.base_seed,
                layer_idx=layer_idx,
            ).to(device=device, dtype=torch.float32)
            self._frlr_basis[layer_idx] = q
            self._frlr_q_step[layer_idx] = step
        elif self.anchor_owns_q:
            # Anchor owns Q: the fast path is NOT a Q writer. The cadence branch
            # below is skipped entirely and anchor_update_basis() is the sole
            # refresh, called by the engine only when the anchor fires. Q is
            # therefore bitwise frozen across every fast step in between.
            pass
        elif step != last_refresh and (step - last_refresh) >= self.frlr_q_cadence:
            # Lazy refresh point: >= frlr_q_cadence steps since the last
            # refresh. Consume the sketch accumulated over the whole frozen
            # window (keep the warm-started Q unchanged if no sketch was
            # accumulated). Intermediate step boundaries fall through, leaving
            # Q bitwise frozen and the sketch growing.
            v = self._frlr_sketch.pop(layer_idx, None)
            if v is not None:
                q = orthonormalize(v).to(device=device, dtype=torch.float32)
                self._frlr_basis[layer_idx] = q
                self.frlr_q_refreshes += 1
            self._frlr_q_step[layer_idx] = step
        if q.device != device:
            q = q.to(device=device)
            self._frlr_basis[layer_idx] = q
        return q

    def _should_accumulate_frlr_sketch(self, layer_idx: int, *, grad_enabled: bool) -> bool:
        """True iff this forward should fold ``h`` into the FRLR basis sketch ``V``.

        Gated by (a) ``grad_enabled`` (a forward_only / old-logprob recompute
        runs under ``torch.no_grad()``, so V is built from the gradient-bearing
        forward only); (b) the path tag; and (c) one contribution per
        forward-generation, which dedupes against gradient-checkpoint recompute.

        **Anchor-owns-Q.** In that mode the fast path must NEVER fold into V, so
        off the anchor pass (``_anchor_sketch_mode`` False) this returns False
        unconditionally. Inside the anchor's stale-weight forward we DO
        accumulate, and we bypass the ``path_tag == train`` gate because the
        anchor pass deliberately runs with ``path_tag=None``. Mirrors
        ``PowerSGDActivationCompressor._should_accumulate_sketch``.
        """
        if not grad_enabled:
            return False
        if self.anchor_owns_q:
            if not self._anchor_sketch_mode:
                return False
            return self._frlr_sketched_this_gen.get(layer_idx, -1) != self._frlr_fwd_generation
        tag = getattr(self._state, "path_tag", None) if self._state is not None else TRAIN_TAG
        if tag != TRAIN_TAG:
            return False
        return self._frlr_sketched_this_gen.get(layer_idx, -1) != self._frlr_fwd_generation

    # ------------------------------------------------------------------ #
    # Anchor-owned Q: slow-circuit sketch harvest and refresh (issue #93)
    # ------------------------------------------------------------------ #
    def set_anchor_sketch_mode(self, on: bool) -> None:
        """Toggle FRLR sketch harvesting during the dense anchor pass.

        While true the boundary hooks return the RAW activation (so the anchor
        gradient stays clean and uncompressed) and fold it into the sketch V
        consumed by :meth:`anchor_update_basis`.
        """
        self._anchor_sketch_mode = bool(on)

    def anchor_update_basis(self, *, staged: bool = False, dp_group: Any = None) -> bool:
        """Build ``Q <- orth(V)`` from the anchor's harvested sketch.

        The same warm-started block power iteration the fast path uses at
        ``frlr_q_cadence``, but driven by the anchor refresh: the engine calls
        this once per anchor fire, immediately after the clean stale-weight
        forward has folded slow-net activations into V. Returns True iff at
        least one boundary basis was refreshed.

        **Collective safety.** When the DP group is genuinely multi-rank the raw
        sketches are all-reduced BEFORE ``orth`` so every rank orthonormalizes
        the same pooled V and ends on a bit-identical consensus Q. The reduction
        walks the FIXED ``sorted(self.boundary_indices)`` (a model-geometry
        property, identical on every rank) and zero-fills a boundary a rank
        happens to lack, so all ranks issue the identical sequence of
        collectives. Note that the FAST path has no such sync, so a multi-rank
        FRLR run is consistent only under anchor ownership.
        """
        if not self.frlr:
            return False
        if not self.anchor_owns_q:
            raise RuntimeError(
                "comm_eff.activation_mask: anchor_update_basis() called with anchor_owns_q=false. "
                "The anchor must not write Q unless it owns it, or Q would get two writers."
            )
        boundaries = sorted(self.boundary_indices)
        if not boundaries:
            return False
        do_sync = False
        if torch.distributed.is_initialized():
            try:
                do_sync = torch.distributed.get_world_size(group=dp_group) > 1
            except Exception:
                do_sync = False
        step = int(self._global_step)
        updated = 0
        for idx in boundaries:
            v = self._frlr_sketch.pop(idx, None)
            if do_sync:
                q_prev = self._frlr_basis.get(idx)
                if q_prev is None:
                    raise RuntimeError(
                        "comm_eff.activation_mask anchor-owns-Q: boundary "
                        f"{idx} has no basis to shape a zero sketch from, so the DP "
                        "all-reduce sequence would differ across ranks and hang. The "
                        "anchor forward must fire every boundary before the refresh."
                    )
                if v is None:
                    v = torch.zeros_like(q_prev)
                torch.distributed.all_reduce(v, group=dp_group)
            if v is None:
                continue
            if not torch.isfinite(v).all():
                raise RuntimeError(
                    f"comm_eff.activation_mask anchor-owns-Q: sketch V at boundary {idx} is "
                    "not finite; orth(V) would produce a garbage basis. Refusing to refresh Q."
                )
            device = self._frlr_basis[idx].device if idx in self._frlr_basis else v.device
            q_new = orthonormalize(v).to(device=device, dtype=torch.float32)
            if staged:
                # Live Q untouched; activate_staged_frlr_basis() publishes it.
                self._pending_frlr_basis[idx] = q_new
            else:
                self._frlr_basis[idx] = q_new
                self._frlr_q_step[idx] = step
            self.frlr_q_refreshes += 1
            updated += 1
        # Anything left (a boundary that fired outside `boundary_indices`) would
        # otherwise leak across windows and pollute the next refresh.
        self._frlr_sketch.clear()
        return updated > 0

    def activate_staged_frlr_basis(self) -> bool:
        """Publish the staged anchor candidate as the live basis.

        Called by the engine AFTER every PPO minibatch sharing this step's
        ``old_log_probs`` has completed, so the NEXT step's old-logprob and train
        forwards both see the same new ``Q``. Returns True iff a candidate was
        published. Mirrors ``PowerSGDActivationCompressor.activate_staged_anchor_basis``.
        """
        if not self._pending_frlr_basis:
            return False
        step = int(self._global_step)
        for idx, q in self._pending_frlr_basis.items():
            self._frlr_basis[idx] = q
            self._frlr_q_step[idx] = step
        self._pending_frlr_basis = {}
        self.frlr_basis_generation += 1
        self.frlr_q_activations += 1
        return True

    def discard_staged_frlr_basis(self) -> None:
        """Drop the staged candidate.

        Used when the optimizer update it was derived from did not commit: such a
        candidate must never leak into a later policy pair.
        """
        self._pending_frlr_basis = {}

    def _frlr_transform(self, h: torch.Tensor, *, layer_idx: int) -> torch.Tensor:
        """Apply the FRLR ``rank + k + 1`` reconstruction to a boundary activation.

        ``y = h @ Q`` (rank core coords/token), ``l = y @ Q^T``,
        ``res = h - l``. ``J`` is a per-token PRF-fresh EXACT-``k`` channel
        subset keyed exactly like the baseline mask (INCLUDING
        ``global_step``: fresh across steps, identical within a step and
        across the old/train/reference forwards). Default mode sends one
        residual-norm scalar per token (the ``+1``) and rescales the scattered
        kept residual by the DETACHED
        ``gamma = ||res|| / max(||scatter_J(res_J)||, eps)`` capped at ``H/k``
        (no blow-up on adversarial tokens). ``frlr_unbiased`` instead applies
        the constant ``H/k`` gain, making ``E[h_hat | h, Q] = h``. The whole
        transform is in-graph through ``h`` (``Q``, ``J`` and ``gamma`` are
        constants to autograd), so backward is the exact adjoint, and the same
        PRF key makes the gradient-checkpoint recompute bit-deterministic.
        """
        grad_enabled = torch.is_grad_enabled()
        hidden_size = int(h.shape[-1])
        k = int(self.frlr_k)
        if k > hidden_size:
            raise ValueError(f"comm_eff mask.frlr_k={k} exceeds the hidden size H={hidden_size}")
        orig_shape = h.shape
        m = h.reshape(-1, hidden_size)
        q_fp32 = self._frlr_ensure_basis(layer_idx, hidden_size=hidden_size, device=m.device)
        q_act = q_fp32.to(dtype=m.dtype)
        y = m @ q_act  # (N, r) core payload; Q is a detached buffer, m stays in-graph
        low = y @ q_act.t()  # (N, H) rank-r reconstruction
        res = m - low
        mask_j = prf_token_mask(
            self._sample_ids,
            self._position_ids,
            layer_idx=layer_idx,
            global_step=self._global_step,
            base_seed=self.base_seed,
            hidden_size=hidden_size,
            p=self.p,
            device=m.device,
            dtype=m.dtype,
            exact_k=True,
            exact_keep=k,
        )
        res_j = res * mask_j  # scatter_J(res_J): kept residual channels, zero elsewhere
        cap = float(hidden_size) / float(k)  # 1/q_r with q_r = k/H
        if self.frlr_unbiased:
            m_hat = low + cap * res_j
        else:
            res32 = res.detach().to(torch.float32)
            res_norm = res32.norm(dim=-1, keepdim=True)  # the +1 norm scalar per token
            kept_norm = (res32 * mask_j.to(torch.float32)).norm(dim=-1, keepdim=True)
            gamma = (res_norm / kept_norm.clamp_min(_FRLR_NORM_EPS)).clamp_(max=cap)
            m_hat = low + gamma.to(dtype=m.dtype) * res_j
        with torch.no_grad():
            # Cross-step Q refresh sketch V += h^T (h Q): the gradient-bearing
            # train forward only, at most once per forward generation (dedupe
            # against gradient-checkpoint recompute), mirroring PowerSGD.
            if self._should_accumulate_frlr_sketch(layer_idx, grad_enabled=grad_enabled):
                m32 = m.detach().to(torch.float32)
                contrib = m32.t() @ (m32 @ q_fp32)  # (H, r)
                cur = self._frlr_sketch.get(layer_idx)
                if cur is None:
                    self._frlr_sketch[layer_idx] = contrib
                else:
                    cur.add_(contrib)
                self._frlr_sketched_this_gen[layer_idx] = self._frlr_fwd_generation
            # Payload accounting: rank + k + 1 kept coords/token (the +1 norm
            # scalar is not sent in unbiased mode). 32 + 44 + 1 = 77 of 1536
            # => mask_ratio ~ 0.9499.
            r_eff = int(q_fp32.shape[1])
            kept = r_eff + k + (0 if self.frlr_unbiased else 1)
            self.frlr_payload_per_token = float(kept)
            self.last_mask_ratio[layer_idx] = max(0.0, 1.0 - float(kept) / float(hidden_size))
        return m_hat.reshape(orig_shape)

    def _make_hook(self, layer_idx: int):
        masker = self

        def _hook(_mod: nn.Module, _inputs: tuple, output: Any):
            h = output[0] if isinstance(output, tuple) else output
            if not torch.is_tensor(h):
                return output

            # Anchor stale-forward harvest (anchor-owns-Q, FRLR only). The anchor
            # forward must be CLEAN: its gradient is G_anchor and its activations
            # feed Q, so fold the RAW activation into V (V += hᵀ(hQ)) and return h
            # UNCHANGED with no reconstruction. This runs BEFORE the confinement
            # assert and the per-token identity check on purpose: the anchor pass
            # carries path_tag=None and needs no PRF key (no mask is drawn here).
            if masker._anchor_sketch_mode:
                # Captured BEFORE the no_grad block: inside it is_grad_enabled()
                # is False by construction, which would gate the harvest off
                # unconditionally. Same ordering as the PowerSGD hook.
                harvest_grad_enabled = torch.is_grad_enabled()
                with torch.no_grad():
                    hidden = int(h.shape[-1])
                    if masker.hidden_size is None:
                        masker.hidden_size = hidden
                    q_fp32 = masker._frlr_ensure_basis(layer_idx, hidden_size=hidden, device=h.device)
                    if masker._should_accumulate_frlr_sketch(layer_idx, grad_enabled=harvest_grad_enabled):
                        m32 = h.detach().reshape(-1, hidden).to(torch.float32)
                        contrib = m32.t() @ (m32 @ q_fp32)  # (H, r)
                        cur = masker._frlr_sketch.get(layer_idx)
                        if cur is None:
                            masker._frlr_sketch[layer_idx] = contrib
                        else:
                            cur.add_(contrib)
                        masker._frlr_sketched_this_gen[layer_idx] = masker._frlr_fwd_generation
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

            if masker.frlr:
                # FRLR (issue #89): fresh-residual low-rank reconstruction
                # replaces the plain mask transform. Sets last_mask_ratio and
                # frlr_payload_per_token internally.
                h_tilde = masker._frlr_transform(h, layer_idx=layer_idx)
                if state is not None:
                    if hasattr(state, "note_mask_application"):
                        state.note_mask_application()
                    else:
                        state.mask_applications += 1
                if isinstance(output, tuple):
                    return (h_tilde,) + tuple(output[1:])
                return h_tilde

            # Per-boundary p (lever 4) when configured, else the scalar p.
            p_layer = masker._p_for_layer.get(layer_idx, masker.p) if masker._p_for_layer else masker.p
            mask = prf_token_mask(
                sample_ids,
                position_ids,
                layer_idx=layer_idx,
                global_step=masker._global_step,
                base_seed=masker.base_seed,
                hidden_size=hidden_size,
                p=p_layer,
                device=h.device,
                dtype=h.dtype,
                exact_k=masker.exact_k,
                antithetic=masker.antithetic,
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
            elif masker.rescale_mode == "constant":
                # Inverted-dropout 1/(1-p). With per-boundary p the gain is
                # recomputed for this boundary's p; without it the precomputed
                # scalar gain keeps the baseline byte-identical.
                if masker._p_for_layer:
                    gain = (1.0 / (1.0 - p_layer)) if p_layer < 1.0 else 1.0
                else:
                    gain = masker._rescale_gain
                h_tilde = h * mask * gain
            else:  # "none"
                h_tilde = h * mask
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
        # Lever 4: one masked fraction per boundary. Validate the length against
        # the ACTUAL boundary count here (num_layers is a model property unknown
        # at construction), then map boundary_idx -> p.
        if self.p_by_boundary:
            if len(self.p_by_boundary) != len(self.boundary_indices):
                raise ValueError(
                    f"mask p_by_boundary has {len(self.p_by_boundary)} entries but there are "
                    f"{len(self.boundary_indices)} masked boundaries {self.boundary_indices} "
                    f"(L={len(layers)}, pp_size={self.pp_size}); supply exactly one p per boundary."
                )
            self._p_for_layer = {idx: self.p_by_boundary[i] for i, idx in enumerate(self.boundary_indices)}
        else:
            self._p_for_layer = {}
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
