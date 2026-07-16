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

"""PowerSGD-style pipeline-boundary activation compression.

A shared, frozen, per-layer orthonormal basis ``Q`` (shape ``(H, r)``) projects
each pipeline-boundary block's hidden-state output ``M`` (shape ``(N, H)`` —
``N`` packed tokens × ``H`` hidden dims) onto its rank-``r`` subspace::

    M_hat = (M @ Q) @ Q.T            # forward; Q DETACHED, M in-graph (NO STE)

The boundary therefore transmits only the ``N·r`` projected coordinates
``Y = M @ Q`` plus the communication-free shared ``Q``.

Three properties make this correct:

* **No straight-through.** ``Q`` is detached and ``M`` stays in-graph, so the
  autograd backward of ``M_hat = (M @ Q) @ Qᵀ`` is the exact self-adjoint
  projector ``dL/dM = (dL/dM_hat) Q Qᵀ`` — no STE, no custom autograd Function.
* **Deterministic zero-comm bootstrap.** ``Q_L = orth(randn(H, r))`` seeded by
  ``seed_L = (base_seed·1_000_003 + layer_idx·7919) & 0x7FFFFFFF``, drawn in fp32
  on CPU so it is bit-identical on every rank/device.
* **Optional pre-pair fast calibration.** With ``fast_q_bootstrap=true``, one
  discarded dense no-grad actor prepass on the first trajectory batch builds a
  DP-pooled activation ``Q1`` and atomically publishes it before any compressed
  old/current PPO forward. When calibration is disabled, the deterministic
  seeded basis remains active.
* **Block power iteration, off-graph.** On compressed *train* forwards we
  accumulate ``V += Mᵀ (M Q)`` under ``torch.no_grad()`` (one sketch per boundary
  forward, deduplicated against gradient-checkpoint recompute), then once at
  end-of-actor-update set ``Q ← orth(V)`` in fp32 and clear ``V``. ``Q`` is
  **frozen for the whole global step** so the old-logprob recompute and the
  actor-train forward see the same ``Q_t``.

fp32 QR is REQUIRED: bf16-QR loses orthogonality (``QᵀQ`` drifts from ``I``),
degrades the projector identity ``P² = P``, and is a frequent NaN / ``q_cond``
source. The projection itself runs in the activation dtype.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch
import torch.nn as nn

from verl.workers.comm_eff.boundary import (
    decoder_boundary_indices,
    find_decoder_layers,
)
from verl.workers.comm_eff.state import TRAIN_TAG

logger = logging.getLogger(__name__)

__all__ = [
    "powersgd_layer_seed",
    "init_basis",
    "orthonormalize",
    "PowerSGDActivationCompressor",
]

# Per-layer seed mixing constants. seed_L = (base_seed*MIX_BASE +
# layer_idx*MIX_LAYER) & MASK31. The bit mask keeps the value positive int31 so it is
# a valid torch.Generator manual_seed on every backend.
_SEED_MIX_BASE = 1_000_003
_SEED_MIX_LAYER = 7919
_MASK31 = 0x7FFFFFFF


def powersgd_layer_seed(base_seed: int, layer_idx: int) -> int:
    """Deterministic per-layer basis seed.

    ``seed_L = (base_seed·1_000_003 + layer_idx·7919) & 0x7FFFFFFF``. Pure — no
    side effects. The bit mask keeps it int31-positive so the same value is a legal
    ``torch.Generator.manual_seed`` on CPU and every accelerator.
    """
    return (int(base_seed) * _SEED_MIX_BASE + int(layer_idx) * _SEED_MIX_LAYER) & _MASK31


def orthonormalize(mat: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
    """Return an orthonormal basis spanning the columns of ``mat`` (fp32 QR).

    ``mat`` is ``(H, r)``. Runs ``torch.linalg.qr`` in fp32, normalises the sign
    of ``Q`` against ``R``'s diagonal
    so the basis is deterministic across backends, and falls back to a fresh
    deterministic orthonormal frame if the input is rank-deficient / non-finite
    (so a degenerate sketch never propagates a NaN basis). The result is fp32.
    """
    work = mat.detach().to(torch.float32)
    # Guard a non-finite sketch up front: an all-zero / NaN V would make QR
    # return invalid values. Replace non-finite entries with 0 before the QR; the
    # column-norm check below then catches a fully-empty basis.
    if not torch.isfinite(work).all():
        work = torch.nan_to_num(work, nan=0.0, posinf=0.0, neginf=0.0)
    H, r = work.shape
    # Reduced QR: Q is (H, r) with orthonormal columns, R is (r, r) upper-tri.
    q, rmat = torch.linalg.qr(work, mode="reduced")
    # Sign convention: make the diagonal of R non-negative so Q is unique
    # (QR is unique only up to column signs). Bit-identical across ranks.
    diag = torch.diagonal(rmat, dim1=-2, dim2=-1)
    sign = torch.sign(diag)
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    q = q * sign.unsqueeze(0)
    # Detect a rank-deficient sketch (a near-zero R diagonal ⇒ degenerate
    # column). If any column collapsed, re-seed those columns from a fresh
    # orthonormal frame derived from the existing basis (Gram–Schmidt against a
    # random complement) so Q always has full numerical rank r.
    bad = torch.abs(diag) <= eps
    if bool(bad.any()):
        # Deterministic random complement (seed from the matrix shape so the
        # repair is reproducible regardless of which rank hit the degeneracy).
        gen = torch.Generator(device="cpu").manual_seed((H * 1_000_003 + r) & _MASK31)
        rand = torch.randn(H, r, generator=gen, dtype=torch.float32)
        q_fix, _ = torch.linalg.qr(rand, mode="reduced")
        # q lives on the compute device (cuda); q_fix was built on CPU via the
        # DETERMINISTIC cpu generator + a CPU QR (keep it on CPU so the repair is
        # bit-identical across ranks — a GPU QR could differ in low bits and
        # break the cross-rank Q consensus). Move ONLY the result onto q's device
        # right before torch.where so the degenerate-column repair does not raise
        # a cross-device error. dtype is already fp32 on both.
        q = torch.where(bad.unsqueeze(0), q_fix.to(q.device), q)
    return q


def init_basis(
    *,
    hidden_size: int,
    rank: int,
    base_seed: int,
    layer_idx: int,
) -> torch.Tensor:
    """Deterministic per-layer orthonormal basis ``Q_L = orth(randn(H, r))``.

    Drawn in fp32 on CPU with a ``torch.Generator`` seeded by
    :func:`powersgd_layer_seed`, so the basis is bit-identical on every
    rank/device. ``rank`` is
    clamped to ``hidden_size`` (``r == H`` is the lossless limiting case
    ``M_hat = M``). Returns an fp32 ``(H, min(rank, H))`` tensor on CPU; the
    caller moves it to the activation device/dtype for the forward.
    """
    r = min(int(rank), int(hidden_size))
    seed = powersgd_layer_seed(base_seed, layer_idx)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    g = torch.randn(int(hidden_size), r, generator=gen, dtype=torch.float32)
    return orthonormalize(g)


class PowerSGDActivationCompressor:
    """Installs/clears in-graph PowerSGD projection hooks on boundary blocks.

    The engine drives the codec through a register/context/update lifecycle:

    * ``register(module)`` discovers the boundary decoder blocks, lazily
      bootstraps each block's deterministic basis ``Q`` and installs a
      forward hook that replaces the block output ``M`` with ``M_hat=(M@Q)@Qᵀ``.
    * ``unregister()`` removes the hooks.
    * ``set_context(global_step)`` stamps the trainer step and bumps the
      per-micro-batch "forward generation" used to deduplicate the basis sketch
      against gradient-checkpoint recompute.
    * ``maybe_update_basis()`` runs the block-power-iteration ``Q ← orth(V)`` at
      cadence, AFTER the gradient-bearing actor work — called by the engine's
      end-of-train_batch hook so ``Q`` is frozen across the paired GRPO forwards.
    * Anchor-owned basis refreshes can be staged during PPO minibatches and
      activated only after the complete ``update_actor`` call. This keeps the
      old-logprob recompute and every current-policy minibatch on exactly the
      same live ``Q_t``.

    The basis lives on the worker (one ``Q`` per boundary layer), so it persists
    across steps (warm start) independent of the hook register/unregister cycle.
    """

    def __init__(
        self,
        *,
        rank: int,
        base_seed: int,
        pp_size: int,
        update_cadence: int = 1,
        warm_start: bool = True,
        compress_recompute: bool = True,
        sync_basis: bool = False,
        qr_dtype: str = "fp32",
        reortho_eps: float = 1e-6,
        anchor_owns_q: bool = False,
        anchor_cadence: int = 1,
        fast_q_bootstrap: bool = False,
        state: Any = None,
    ):
        self.rank = int(rank)
        self.base_seed = int(base_seed)
        self.pp_size = int(pp_size)
        self.update_cadence = int(update_cadence)
        self.warm_start = bool(warm_start)
        self.sync_basis = bool(sync_basis)
        # Anchor refresh cadence: the Q broadcast (H*r per boundary) happens once
        # per ``anchor_cadence`` optimizer ticks, so its per-tick amortized cost is
        # ``Σ_boundary H·r / anchor_cadence``. (The per-token Y term N·r dominates,
        # so the exact divisor is a small correction, but use the real cadence.)
        self.anchor_cadence = max(1, int(anchor_cadence))
        # Optional one-time fast-network Q calibration.  The normal
        # anchor-owned-Q path intentionally leaves the fast circuit read-only;
        # this is the sole exception and is allowed only before the first
        # compressed old/current PPO pair.  A discarded dense no-grad actor
        # prepass observes the first real trajectory batch, stages a complete
        # DP-consensus activation basis, and atomically publishes it before the
        # real old-logprob forward.  After that handoff the anchor is again the
        # only Q writer.
        self.fast_q_bootstrap = bool(fast_q_bootstrap)
        # When True the anchor owns Q. The fast net is then a pure
        # read-only consumer: its end-of-step maybe_update_basis is gated OFF (by
        # the engine call site) AND its forward-hook sketch accumulation is gated
        # OFF here (so V never grows on the fast path). Q is updated ONLY by the
        # anchor's slow-net forward (anchor_update_basis) and propagated by
        # broadcast_basis. False keeps the fast-owned-Q path.
        self.anchor_owns_q = bool(anchor_owns_q)
        # Transient toggle: True ONLY inside the anchor's stale-weight forward so
        # the SAME forward hook that is suppressed on the fast path DOES fold the
        # slow-net activations into V. Set/cleared by the engine around the anchor
        # forward. Never persisted.
        self._anchor_sketch_mode = False
        if str(qr_dtype) != "fp32":
            raise ValueError(f"powersgd qr_dtype must be 'fp32'; got {qr_dtype!r}")
        # qr_dtype controls only the orth/QR + sketch math; fp32 is required for
        # orthogonality.
        self.qr_dtype = torch.float32
        self.reortho_eps = float(reortho_eps)
        self._state = state  # CommEffState, for counters

        self._handles: list[Any] = []
        self.boundary_indices: list[int] = []
        self._hidden_size: Optional[int] = None
        # DP process group the basis consensus all-reduces over. None ⇒ world
        # group (correct when world==DP: SP=1, no TP/PP in the training mesh).
        # Bound by the engine via set_dp_group when a narrower
        # DP subgroup is needed.
        self._dp_process_group = None

        # Per-boundary frozen basis Q_t (H, r), fp32, on the activation device.
        # Lazily bootstrapped on first register() once H is known. Persists
        # across steps (warm block power iteration) — NOT cleared by unregister.
        self._basis: dict[int, torch.Tensor] = {}
        # Anchor-owned Q_{t+1} candidate. The anchor can fire inside one of the
        # PPO minibatches, after old_log_probs were already recomputed with Q_t.
        # Publishing directly into _basis there would make that minibatch use a
        # different policy denominator. Keep the candidate isolated until the
        # worker finishes every minibatch in update_actor, then atomically
        # activate it for the NEXT old-logprob/train pair.
        self._pending_anchor_basis: dict[int, torch.Tensor] = {}
        # The fast bootstrap has its own transaction store.  Reusing
        # _pending_anchor_basis would conflate its counters/generation with an
        # anchor fire and could let update_actor's exception boundary discard or
        # activate the wrong candidate.
        self._pending_fast_q_bootstrap_basis: dict[int, torch.Tensor] = {}
        # Monotonic successful handoff generation. This is deliberately distinct
        # from anchor_q_updates: a candidate can be staged and then discarded if
        # the enclosing actor update fails.
        self._anchor_basis_generation = 0
        self._fast_q_bootstrap_observing = False
        self._fast_q_bootstrap_done = False
        self._fast_q_bootstrap_sketch: dict[int, torch.Tensor] = {}
        self._fast_q_bootstrap_sketch_count: dict[int, int] = {}
        self._fast_q_bootstrap_sketched_this_gen: dict[int, int] = {}
        # Explicit one-time counters/communication accounting.  The dense
        # observation element count is the logical PP payload a real pipelined
        # implementation would pay once; sync_elements is the H*r activation-
        # sketch payload per selected boundary pooled over DP once.
        self.fast_q_bootstrap_observations = 0
        self.fast_q_bootstrap_updates = 0
        self.fast_q_bootstrap_activations = 0
        self.fast_q_bootstrap_dense_observation_elements = 0.0
        self.fast_q_bootstrap_sync_elements = 0.0
        self.fast_q_bootstrap_cross_rank_max_rel_dev: Optional[float] = None
        # Per-boundary accumulated sketch V (H, r), fp32. Reset after each
        # orth(V); accumulated under no_grad on compressed train forwards only.
        self._sketch: dict[int, torch.Tensor] = {}
        # Per-boundary count of sketches folded into the current V (for the
        # mean — keeps V's scale independent of how many micro-batches ran).
        self._sketch_count: dict[int, int] = {}

        # Per-forward context (set by the engine before each micro-batch).
        self._global_step = 0
        # Monotonic "forward generation" — bumped once per set_context (i.e. once
        # per micro-batch). Under gradient checkpointing the boundary forward is
        # RECOMPUTED in backward with the SAME context, so we dedupe the sketch
        # by (layer_idx, generation): a layer contributes to V at most once per
        # generation. This guarantees V is accumulated from the ORIGINAL forward
        # only, never double-counted by the recompute.
        self._fwd_generation = 0
        self._sketched_this_gen: dict[int, int] = {}

        # Diagnostics surfaced into metrics by the engine each step:
        #   q_cond            -> max/min singular value of Q (≈1 for orthonormal;
        #                        non-finite ⇒ basis collapse — a hard falsifier).
        #   reconstruction    -> ||M - M_hat|| / ||M|| measured on the last
        #                        compressed forward per boundary (codec health).
        self.last_q_cond: dict[int, float] = {}
        self.last_reconstruction_rel_error: dict[int, float] = {}
        # The logical PP payload actually carried, n·r coordinates per
        # token-layer.
        self.last_y_coords_per_token: int = self.rank

        # Measured inter-stage communication volume (per optimizer
        # tick, summed over boundaries). Reset by the engine each train_batch, then
        # accumulated in the forward hook from the actual codec payload:
        #   bytes_compressed   = Σ_boundary N·r  (Y = M@Q coords sent per fwd) +
        #                        amortized Q-broadcast H·r/cadence at the anchor
        #                        refresh (added in the engine).
        #   bytes_dense_equiv  = Σ_boundary N·H  (the uncompressed activation).
        # "bytes" here counts ELEMENTS (fp count); the dense-vs-compressed RATIO is
        # the reported number and is dtype-invariant. last_* mirrors the most
        # recent tick for the metrics surface.
        self.tick_elems_compressed: float = 0.0
        self.tick_elems_dense_equiv: float = 0.0
        self.last_elems_compressed: float = 0.0
        self.last_elems_dense_equiv: float = 0.0

    # ----------------------------------------------------------------------
    # Byte-counter helpers
    # ----------------------------------------------------------------------
    def reset_tick_comm_counters(self) -> None:
        """Zero the per-tick comm-volume accumulators (engine calls once per
        train_batch before the forward). Pure — no allocation."""
        self.tick_elems_compressed = 0.0
        self.tick_elems_dense_equiv = 0.0

    def add_amortized_q_broadcast_bytes(self) -> None:
        """Add the amortized per-tick Q-broadcast element count.

        The shared basis Q (H×r per boundary) is broadcast once per anchor refresh
        (every ``anchor.cadence`` optimizer ticks), so its per-tick amortized cost is
        ``Σ_boundary H·r / cadence``. The engine calls this on EVERY tick (the
        amortization already divides by cadence) so the running ratio reflects the
        true mean inter-stage volume. No-op if H is unknown."""
        if self._hidden_size is None:
            return
        r = self._effective_rank()
        n_boundaries = len(self.boundary_indices) or 1
        # Amortize the Q broadcast over the ANCHOR refresh cadence (the cadence at
        # which Q is actually re-broadcast in anchor-owned-Q mode). H·r per
        # boundary, divided by the anchor cadence.
        cadence = max(1, int(getattr(self, "anchor_cadence", self.update_cadence)))
        self.tick_elems_compressed += float(n_boundaries) * float(self._hidden_size) * float(r) / float(cadence)

    # ----------------------------------------------------------------------
    # Basis access / bootstrap
    # ----------------------------------------------------------------------
    def _effective_rank(self) -> int:
        if self._hidden_size is None:
            return self.rank
        return min(self.rank, self._hidden_size)

    def _ensure_basis(self, layer_idx: int, *, device, dtype) -> torch.Tensor:
        """Return the frozen fp32 basis for ``layer_idx``, bootstrapping if absent.

        The basis is stored in fp32 on the activation device. The forward casts
        it to the activation dtype for the projection.
        """
        q = self._basis.get(layer_idx, None)
        if q is None:
            q = init_basis(
                hidden_size=int(self._hidden_size),
                rank=self.rank,
                base_seed=self.base_seed,
                layer_idx=layer_idx,
            ).to(device=device, dtype=torch.float32)
            self._basis[layer_idx] = q
        elif q.device != device:
            q = q.to(device=device)
            self._basis[layer_idx] = q
        return q

    # ----------------------------------------------------------------------
    # Per-forward context
    # ----------------------------------------------------------------------
    def set_context(
        self,
        *,
        global_step: int,
    ) -> None:
        """Stamp the trainer step + bump the forward generation for this micro-batch.

        The basis is shared across all tokens. The generation bump deduplicates
        the sketch against gradient-checkpoint recompute.
        """
        self._global_step = int(global_step)
        self._fwd_generation += 1

    # ----------------------------------------------------------------------
    # Forward hook
    # ----------------------------------------------------------------------
    def _make_hook(self, layer_idx: int):
        compressor = self

        def _hook(_mod: nn.Module, _inputs: tuple, output: Any):
            h = output[0] if isinstance(output, tuple) else output
            if not torch.is_tensor(h):
                return output

            # Capture the grad-enabled state at HOOK ENTRY, before we open the
            # no_grad block for diagnostics. The old-logprob recompute runs the
            # WHOLE forward under torch.no_grad() (forward_backward_batch
            # forward_only=True), so grad is disabled there ⇒ no sketch. The
            # gradient-bearing actor-train forward has grad enabled here. Reading
            # is_grad_enabled() inside the no_grad block below would always be
            # False.
            grad_enabled = torch.is_grad_enabled()

            # Record H on first fire (needed to bootstrap the basis).
            hidden_size = h.shape[-1]
            if compressor._hidden_size is None:
                compressor._hidden_size = int(hidden_size)
            elif compressor._hidden_size != int(hidden_size):
                raise RuntimeError(
                    f"comm_eff powersgd: hidden size changed across forwards "
                    f"({compressor._hidden_size} -> {hidden_size}); the shared basis "
                    "assumes a fixed H."
                )

            # Flatten to (N, H): the boundary activation is (1, total_nnz, H) or
            # (total_nnz, H) under rmpad. Project in 2D, restore the shape.
            orig_shape = h.shape
            M = h.reshape(-1, hidden_size)

            # One-time fast-Q observation prepass.  This branch is deliberately
            # UNCOMPRESSED: it returns the raw boundary activation while folding
            # the first real trajectory batch into a private activation sketch
            # V = Mᵀ(MQ0).  The candidate produced from this sketch is not made
            # live in a hook; the engine publishes the complete all-boundary
            # transaction only after this discarded prepass finishes.  Therefore
            # no PPO log-prob is ever computed partly with Q0 and partly with Q1.
            if compressor._fast_q_bootstrap_observing:
                with torch.no_grad():
                    q_fp32 = compressor._ensure_basis(layer_idx, device=M.device, dtype=M.dtype)
                    if compressor._fast_q_bootstrap_sketched_this_gen.get(layer_idx, -1) != compressor._fwd_generation:
                        M32 = M.detach().to(torch.float32)
                        contrib = M32.t() @ (M32 @ q_fp32)
                        cur = compressor._fast_q_bootstrap_sketch.get(layer_idx)
                        if cur is None:
                            compressor._fast_q_bootstrap_sketch[layer_idx] = contrib
                            compressor._fast_q_bootstrap_sketch_count[layer_idx] = 1
                        else:
                            cur.add_(contrib)
                            compressor._fast_q_bootstrap_sketch_count[layer_idx] = (
                                compressor._fast_q_bootstrap_sketch_count.get(layer_idx, 0) + 1
                            )
                        compressor._fast_q_bootstrap_sketched_this_gen[layer_idx] = compressor._fwd_generation
                        compressor.fast_q_bootstrap_dense_observation_elements += float(M32.numel())
                return output

            # Anchor stale-forward harvest. The anchor forward must be
            # CLEAN (uncompressed) — its gradient is G_anchor (→ M) and its
            # activations feed Q. So when _anchor_sketch_mode is on we fold the raw
            # activation into the sketch V (V += Mᵀ(MQ)) but return h UNCHANGED (no
            # M_hat projection). This harvests Q from the slow net without
            # corrupting the clean anchor gradient. grad_enabled is True (the
            # anchor runs forward_only=False), so the sketch lands.
            if compressor._anchor_sketch_mode:
                with torch.no_grad():
                    q_fp32 = compressor._ensure_basis(layer_idx, device=M.device, dtype=M.dtype)
                    if compressor._should_accumulate_sketch(layer_idx, grad_enabled=grad_enabled):
                        M32 = M.detach().float()
                        Y32 = M32 @ q_fp32  # (N, r) in fp32
                        contrib = M32.t() @ Y32  # (H, r)
                        cur = compressor._sketch.get(layer_idx, None)
                        if cur is None:
                            compressor._sketch[layer_idx] = contrib
                            compressor._sketch_count[layer_idx] = 1
                        else:
                            cur.add_(contrib)
                            compressor._sketch_count[layer_idx] = compressor._sketch_count.get(layer_idx, 0) + 1
                        compressor._sketched_this_gen[layer_idx] = compressor._fwd_generation
                # Return the activation UNCHANGED — the anchor forward is clean.
                return output

            q_fp32 = compressor._ensure_basis(layer_idx, device=M.device, dtype=M.dtype)
            # Project in the activation dtype. Q is detached (a buffer,
            # never required grad) and M stays in-graph, so autograd gives the
            # exact self-adjoint projector dL/dM = (dL/dM_hat) Q Qᵀ — NO STE.
            q_act = q_fp32.to(dtype=M.dtype)
            Y = M @ q_act  # (N, r) — the projected coordinates actually "sent"
            M_hat = Y @ q_act.t()  # (N, H) — reconstruction; in-graph through M

            # ---- off-graph diagnostics + basis sketch (no autograd effect) ----
            with torch.no_grad():
                # q_cond: orthonormal Q has all singular values ≈ 1 ⇒ cond ≈ 1.
                # A non-finite cond is the basis-collapse falsifier.
                try:
                    svals = torch.linalg.svdvals(q_fp32.float())
                    smax = float(svals.max().item())
                    smin = float(svals.min().item())
                    q_cond = smax / smin if smin > compressor.reortho_eps else float("inf")
                except Exception:  # pragma: no cover - defensive
                    q_cond = float("inf")
                compressor.last_q_cond[layer_idx] = q_cond

                # reconstruction_rel_error: ||M - M_hat|| / ||M||. Bounded < 1
                # means the codec keeps more signal than it discards.
                M32 = M.detach().float()
                Mhat32 = M_hat.detach().float()
                denom = M32.norm()
                if float(denom.item()) > 0.0:
                    rel = float((M32 - Mhat32).norm().item() / denom.item())
                else:
                    rel = 0.0
                compressor.last_reconstruction_rel_error[layer_idx] = rel
                compressor.last_y_coords_per_token = q_act.shape[1]

                # Accumulate this boundary's inter-stage comm volume
                # for THIS optimizer tick. The codec sends Y = M@Q (N·r coords);
                # the dense baseline would send M (N·H). Counted ONLY on the
                # gradient-bearing fast-train forward (grad_enabled) so the
                # old-logprob recompute / anchor pass do not double-count. (The
                # amortized Q-broadcast term is added once per tick in the engine.)
                if grad_enabled:
                    N = int(M32.shape[0])
                    r_sent = int(q_act.shape[1])
                    Hdim = int(hidden_size)
                    compressor.tick_elems_compressed += float(N) * float(r_sent)
                    compressor.tick_elems_dense_equiv += float(N) * float(Hdim)

                # Block-power-iteration sketch V += Mᵀ (M Q) = Mᵀ Y, OFF the
                # graph, accumulated ONLY on the gradient-bearing actor-train
                # forward (path_tag == train) and at most once per forward
                # generation (so grad-ckpt recompute never double-counts).
                if compressor._should_accumulate_sketch(layer_idx, grad_enabled=grad_enabled):
                    # Mᵀ Y in qr_dtype (fp32 by default). Use the
                    # already-computed Y in fp32.
                    contrib = M32.t() @ Y.detach().to(torch.float32)  # (H, r)
                    cur = compressor._sketch.get(layer_idx, None)
                    if cur is None:
                        compressor._sketch[layer_idx] = contrib
                        compressor._sketch_count[layer_idx] = 1
                    else:
                        cur.add_(contrib)
                        compressor._sketch_count[layer_idx] = compressor._sketch_count.get(layer_idx, 0) + 1
                    compressor._sketched_this_gen[layer_idx] = compressor._fwd_generation

            # Count one projection application.
            state = compressor._state
            if state is not None:
                if hasattr(state, "note_powersgd_application"):
                    state.note_powersgd_application()

            if isinstance(output, tuple):
                return (M_hat.reshape(orig_shape),) + tuple(output[1:])
            return M_hat.reshape(orig_shape)

        return _hook

    def _should_accumulate_sketch(self, layer_idx: int, *, grad_enabled: bool) -> bool:
        """True iff this forward should fold M into the basis sketch V.

        Gated by (a) ``grad_enabled`` (captured at hook entry, before the no_grad
        diagnostics block) — a forward_only / old-logprob recompute pass runs the
        whole forward under ``torch.no_grad()`` so this is False there, which
        means V is built from the gradient-bearing actor-train forward ONLY
        (never the old-logprob recompute); (b) the path tag is
        ``train``; and (c) this layer has not already contributed in the current
        forward generation (dedupe against gradient-checkpoint recompute, which
        reuses the generation set by ``set_context`` and re-runs the boundary
        forward under grad during backward).

        **Anchor-owns-Q.** In that mode the fast path must NEVER fold
        into V (Q is anchor-owned), so on the fast path (``_anchor_sketch_mode``
        False) this returns False unconditionally. Inside the anchor's stale-
        weight forward (``_anchor_sketch_mode`` True) we DO accumulate — and we
        bypass the ``path_tag == train`` gate because the anchor pass deliberately
        runs with ``path_tag=None`` (the generation-dedupe still applies).
        """
        if not grad_enabled:
            return False
        # Anchor-owns-Q routing.
        if self.anchor_owns_q:
            if not self._anchor_sketch_mode:
                # Fast path: NEVER accumulate (Q is owned by the anchor).
                return False
            # Anchor stale-forward: accumulate regardless of path_tag (None here),
            # still deduped per forward-generation against grad-ckpt recompute.
            return self._sketched_this_gen.get(layer_idx, -1) != self._fwd_generation
        # Fast-owns-Q path.
        state = self._state
        tag = getattr(state, "path_tag", None) if state is not None else None
        if tag != TRAIN_TAG:
            return False
        return self._sketched_this_gen.get(layer_idx, -1) != self._fwd_generation

    # ----------------------------------------------------------------------
    # Basis update (block power iteration) — called AFTER the actor backward
    # ----------------------------------------------------------------------
    def maybe_update_basis(self) -> bool:
        """Run ``Q ← orth(V)`` at cadence, AFTER the gradient-bearing actor work.

        Called by the engine's end-of-``train_batch`` hook. Skips non-cadence
        steps, clears the sketch after a successful update, and returns True iff
        Q was updated.

        Because this runs AFTER backward, ``Q`` was frozen for both paired GRPO
        forwards of this step; the update advances ``Q_t → Q_{t+1}`` for the NEXT
        step.

        **Cross-rank consensus codebook.** The
        basis ``Q`` is a SINGLE shared codebook that must differ ONLY per
        layer-boundary and be IDENTICAL on every DP rank. Each rank, however,
        builds its local sketch ``V = Σ Mᵀ(MQ)`` from its OWN data shard (the
        dispatch scatters a different shard per rank), so per-rank ``orth(V)``
        would DIVERGE after the first update. With ``sync_basis=true`` (the
        default) we all-reduce the raw sketches across the DP group BEFORE
        ``orth`` so every rank orthonormalizes the SAME pooled
        ``V_global = Σ_ranks Σ_microbatch Mᵀ(MQ)`` (the global activation
        second-moment projected through ``Q``) → bit-identical consensus ``Q`` on
        every rank, differing only per boundary. DP training is untouched; this
        is just an ``H×r`` all-reduce per boundary at each update.

        Orthonormalization is scale-invariant, so summing the raw per-rank ``V``s
        (rather than averaging) is exactly the pooled direction — no per-rank
        count re-weighting is needed for the basis.

        **Collective safety.** The all-reduce iterates the FIXED
        ``sorted(self.boundary_indices)`` on EVERY rank, contributing a correctly
        shaped ZERO sketch for any boundary a rank happens to be missing locally,
        so all ranks issue the identical sequence of collectives. A rank-relative
        iteration over ``self._sketch`` (different/missing keys, different order)
        would mismatch the collective and hang. All ranks call ``update_actor``
        in lockstep, so a
        symmetric per-boundary collective set is sufficient.
        """
        # Sole-Q-writer invariant: the fast path must never write Q when the
        # anchor owns it. The sole call site (engine_workers.py) is gated on
        # ``fast_owns_q``; reaching here in anchor_owns_q mode means that gate
        # leaked and Q would get two writers.
        assert not getattr(self, "anchor_owns_q", False), (
            "comm_eff.powersgd: maybe_update_basis() entered in anchor_owns_q mode — the "
            "FAST net must NEVER update Q when the anchor owns it. The engine_workers "
            "gate (fast_owns_q) must have leaked."
        )
        gs = int(self._global_step)
        cadence = max(1, int(self.update_cadence))
        # gs <= 0 is the pre-train boundary; never update there.
        if gs <= 0 or (gs % cadence) != 0:
            return False

        do_sync = bool(self.sync_basis) and torch.distributed.is_initialized()
        group = self._dp_group() if do_sync else None

        # GATE THE WHOLE UPDATE SYMMETRICALLY. Without sync, a rank with an empty
        # local sketch simply skips (no collective, safe). WITH sync, every rank
        # MUST walk the identical collective sequence, so we do NOT early-return
        # on an empty local sketch — a rank missing a boundary contributes a zero
        # V for it. (All ranks reach maybe_update_basis in lockstep on the same
        # cadence step, so the boundary_indices set is identical across ranks.)
        if not do_sync and not self._sketch:
            return False

        updated = False
        for layer_idx in self._boundary_for_update():
            V = self._sketch.get(layer_idx, None)
            if V is None:
                if do_sync and self._hidden_size is not None:
                    # Symmetric collective: contribute a zero sketch of the right
                    # shape so this rank issues the same all_reduce as the others.
                    r = self._effective_rank()
                    V = torch.zeros(int(self._hidden_size), r, dtype=torch.float32, device=self._sketch_device())
                else:
                    # No sync + no local sketch for this boundary ⇒ keep Q_t.
                    continue
            Vsum = V.to(torch.float32)
            if do_sync:
                # Pool the RAW sketches across the DP group: V_global = Σ_ranks V.
                # orth is scale-invariant so the SUM gives the pooled direction;
                # every rank now orthonormalizes the identical V_global.
                torch.distributed.all_reduce(Vsum, op=torch.distributed.ReduceOp.SUM, group=group)
            q_new = orthonormalize(Vsum.to(self.qr_dtype), eps=self.reortho_eps)
            self._basis[layer_idx] = q_new.to(device=self._sketch_device(), dtype=torch.float32)
            updated = True

        self._reset_sketch()
        if updated and self._state is not None and hasattr(self._state, "note_powersgd_basis_update"):
            self._state.note_powersgd_basis_update()
        return updated

    # ----------------------------------------------------------------------
    # One-time fast-network activation bootstrap
    # ----------------------------------------------------------------------
    @property
    def fast_q_bootstrap_done(self) -> bool:
        """Whether the one-time fast activation candidate is already live."""

        return bool(self._fast_q_bootstrap_done)

    def fast_q_bootstrap_needed(self) -> bool:
        """True only before this compressor lifecycle's first calibrated pair."""

        return bool(self.fast_q_bootstrap) and not self._fast_q_bootstrap_done

    def reference_basis_ready(self) -> bool:
        """True iff an established Q exists to compress a read-only reference forward.

        The reference-KL forward must never compress against a not-yet-calibrated or
        empty basis. When ``fast_q_bootstrap`` is on this requires the one-time
        calibration to have committed; in every mode it also requires at least one
        boundary basis to be established (a real Q, not a random/empty one). The
        reference forward is ``forward_only`` (grad disabled), so it consumes this Q
        read-only and never advances it. When this returns False the caller falls
        back to a dense reference forward rather than compressing with a random Q.
        """

        if self.fast_q_bootstrap_needed():
            return False
        return bool(self._basis)

    def begin_fast_q_bootstrap_observation(self) -> None:
        """Open the discarded dense observation transaction.

        This method never changes live Q.  The engine calls it immediately before
        a no-grad prepass over the first real old-logprob batch.  Re-entry is a
        hard error: two writers or two overlapping observation passes would make
        the candidate's batch provenance ambiguous.
        """

        if not self.fast_q_bootstrap:
            raise RuntimeError("comm_eff fast-Q bootstrap was requested at runtime but is disabled")
        if not self.anchor_owns_q:
            raise RuntimeError("comm_eff fast-Q bootstrap requires anchor_owns_q=true after the one-time handoff")
        if self._fast_q_bootstrap_done:
            raise RuntimeError("comm_eff fast-Q bootstrap is one-shot and has already committed")
        if self._fast_q_bootstrap_observing or self._pending_fast_q_bootstrap_basis:
            raise RuntimeError("comm_eff fast-Q bootstrap transaction is already in progress")
        self._fast_q_bootstrap_sketch = {}
        self._fast_q_bootstrap_sketch_count = {}
        self._fast_q_bootstrap_sketched_this_gen = {}
        self._fast_q_bootstrap_observing = True

    def finish_fast_q_bootstrap_observation(self) -> None:
        """Close a successful raw-activation prepass without publishing Q."""

        if not self._fast_q_bootstrap_observing:
            raise RuntimeError("comm_eff fast-Q bootstrap observation is not active")
        self._fast_q_bootstrap_observing = False
        self.fast_q_bootstrap_observations += 1

    def stage_fast_q_bootstrap_basis(self) -> bool:
        """DP-pool the private activation sketch and stage a complete Q candidate.

        The live ``_basis`` dictionary is untouched.  With DP>1, every rank first
        performs a symmetric coverage check, then all-reduces the fixed-order
        ``H x r`` sketches and orthonormalizes the identical pooled values.  This
        is the one controlled, one-time payload synchronization requested for the
        fast bootstrap; only a tiny checksum verification follows before commit.
        """

        if self._fast_q_bootstrap_observing:
            raise RuntimeError("comm_eff fast-Q bootstrap observation must finish before staging")
        if self._fast_q_bootstrap_done:
            raise RuntimeError("comm_eff fast-Q bootstrap cannot stage after commit")
        if self._pending_fast_q_bootstrap_basis:
            raise RuntimeError("comm_eff fast-Q bootstrap already has a staged candidate")
        if self._hidden_size is None:
            raise RuntimeError("comm_eff fast-Q bootstrap observed no hidden size")

        boundaries = self._boundary_for_update()
        if not boundaries:
            raise RuntimeError("comm_eff fast-Q bootstrap found no pipeline boundaries")
        dist_ready = torch.distributed.is_initialized()
        group = self._dp_group() if dist_ready else None
        world = torch.distributed.get_world_size(group=group) if dist_ready else 1
        if world > 1 and not self.sync_basis:
            raise RuntimeError(
                "comm_eff fast-Q bootstrap requires sync_basis=true when DP world size is greater than one"
            )

        # Fail symmetrically if any rank missed a boundary.  A local early raise
        # before other ranks enter the V all-reduces could deadlock them.
        device = self._sketch_device()
        coverage = torch.tensor(
            [1 if idx in self._fast_q_bootstrap_sketch else 0 for idx in boundaries],
            dtype=torch.int32,
            device=device,
        )
        if world > 1:
            torch.distributed.all_reduce(coverage, op=torch.distributed.ReduceOp.MIN, group=group)
        missing = [idx for idx, present in zip(boundaries, coverage.tolist(), strict=True) if int(present) == 0]
        if missing:
            raise RuntimeError(
                "comm_eff fast-Q bootstrap did not observe every boundary on every DP rank; "
                f"missing={missing}, boundaries={boundaries}"
            )

        candidate: dict[int, torch.Tensor] = {}
        for layer_idx in boundaries:
            Vsum = self._fast_q_bootstrap_sketch[layer_idx].detach().to(torch.float32).clone()
            if world > 1:
                torch.distributed.all_reduce(Vsum, op=torch.distributed.ReduceOp.SUM, group=group)
            q_new = orthonormalize(Vsum.to(self.qr_dtype), eps=self.reortho_eps)
            candidate[layer_idx] = q_new.to(device=device, dtype=torch.float32)

        expected = set(boundaries)
        if set(candidate) != expected:
            raise RuntimeError(
                "comm_eff fast-Q bootstrap built an incomplete candidate "
                f"(expected={sorted(expected)}, received={sorted(candidate)})"
            )
        self._pending_fast_q_bootstrap_basis = candidate
        self._fast_q_bootstrap_sketch = {}
        self._fast_q_bootstrap_sketch_count = {}
        self._fast_q_bootstrap_sketched_this_gen = {}
        if world > 1:
            self.fast_q_bootstrap_sync_elements += float(
                len(boundaries) * int(self._hidden_size) * self._effective_rank()
            )
        self.fast_q_bootstrap_updates += 1
        return True

    def verify_fast_q_bootstrap_basis_across_ranks(self, *, atol: float = 1e-6) -> Optional[float]:
        """Fail closed unless the complete staged fast-Q candidate agrees over DP."""

        if not torch.distributed.is_initialized():
            return None
        group = self._dp_group()
        world = torch.distributed.get_world_size(group=group)
        if world <= 1:
            self.fast_q_bootstrap_cross_rank_max_rel_dev = 0.0
            return 0.0
        boundaries = self._boundary_for_update()
        if not boundaries:
            return None
        store = self._pending_fast_q_bootstrap_basis
        device = self._sketch_device()
        checksums = []
        for layer_idx in boundaries:
            q = store.get(layer_idx)
            if q is None:
                checksums.append(0.0)
                continue
            checksums.append(float((q.detach().to(torch.float64) * self._ramp_like(q)).sum().item()))
        vec = torch.tensor(checksums, dtype=torch.float64, device=device)
        gathered = [torch.zeros_like(vec) for _ in range(world)]
        torch.distributed.all_gather(gathered, vec, group=group)
        ref = gathered[0]
        max_abs_dev = max((float((item - ref).abs().max().item()) for item in gathered[1:]), default=0.0)
        scale = float(ref.abs().max().item()) or 1.0
        max_rel_dev = max_abs_dev / scale
        self.fast_q_bootstrap_cross_rank_max_rel_dev = max_rel_dev
        if max_rel_dev > atol:
            raise RuntimeError(
                "comm_eff fast-Q bootstrap candidate DIVERGED across DP ranks "
                f"(max_rel_dev={max_rel_dev:.3e} > atol={atol:.1e}); refusing atomic activation"
            )
        return max_rel_dev

    def activate_staged_fast_q_bootstrap_basis(self) -> bool:
        """Atomically publish the verified all-boundary candidate before old_logprob."""

        if not self._pending_fast_q_bootstrap_basis:
            return False
        if self._fast_q_bootstrap_observing or self._fast_q_bootstrap_done:
            raise RuntimeError("comm_eff fast-Q bootstrap candidate is not in a committable state")
        expected = set(self._boundary_for_update())
        received = set(self._pending_fast_q_bootstrap_basis)
        if received != expected:
            raise RuntimeError(
                "comm_eff fast-Q bootstrap staged candidate is incomplete "
                f"(expected={sorted(expected)}, received={sorted(received)})"
            )
        # One dictionary handoff: no boundary can observe a mix of Q0 and Q1.
        self._basis = dict(self._pending_fast_q_bootstrap_basis)
        self._pending_fast_q_bootstrap_basis = {}
        self._fast_q_bootstrap_done = True
        self.fast_q_bootstrap_activations += 1
        return True

    def abort_fast_q_bootstrap(self) -> bool:
        """Discard every private bootstrap buffer while leaving live Q untouched."""

        had_state = bool(
            self._fast_q_bootstrap_observing or self._fast_q_bootstrap_sketch or self._pending_fast_q_bootstrap_basis
        )
        self._fast_q_bootstrap_observing = False
        self._fast_q_bootstrap_sketch = {}
        self._fast_q_bootstrap_sketch_count = {}
        self._fast_q_bootstrap_sketched_this_gen = {}
        self._pending_fast_q_bootstrap_basis = {}
        return had_state

    # ----------------------------------------------------------------------
    # Anchor-owned Q: slow-net update plus broadcast
    # ----------------------------------------------------------------------
    def set_anchor_sketch_mode(self, on: bool) -> None:
        """Toggle activation-sketch collection during the dense anchor pass.

        The anchor owns ``Q`` in the communication-efficient pipeline. While
        this flag is true,
        boundary hooks return raw activations and accumulate the activation
        second-moment sketch used by :meth:`anchor_update_basis`.
        """
        self._anchor_sketch_mode = bool(on)

    def anchor_update_basis(self, *, staged: bool = False) -> bool:
        """Build ``Q <- orth(V)`` from the anchor's slow-net sketch.

        The SAME block-power-iteration math as :meth:`maybe_update_basis` (DP-sync
        of the raw sketch over the DP group, then fp32 ``orth(V)`` per boundary),
        but driven by the anchor refresh (cadence already gated by the engine —
        called only when the anchor fires) instead of the fast end-of-step hook,
        and consuming V built from the slow-net stale-weight forward activations.
        Clears the sketch after. With ``staged=False`` the result replaces live
        ``Q``. With ``staged=True`` it is
        written to ``_pending_anchor_basis`` and the live basis is unchanged;
        :meth:`activate_staged_anchor_basis` publishes it after all PPO
        minibatches finish. Returns True iff any Q candidate was built.

        Collective safety identical to ``maybe_update_basis``: iterate the FIXED
        ``sorted(boundary_indices)`` on every rank, contribute a zero sketch for
        any boundary missing locally, so the all-reduce sequence is symmetric.
        The caller (engine) invokes this on EVERY rank in lockstep on the anchor
        cadence step. The broadcast in :meth:`broadcast_basis` then distributes
        the consensus Q (sync makes it already-identical, broadcast is the
        explicit receipt-checked propagation the invariant requires).
        """
        do_sync = bool(self.sync_basis) and torch.distributed.is_initialized()
        group = self._dp_group() if do_sync else None
        if not do_sync and not self._sketch:
            return False

        target_basis = self._pending_anchor_basis if staged else self._basis
        if staged:
            # Each anchor fire publishes one complete candidate generation.
            # Never mix missing entries from a previous fire into a new one. If
            # an unusually small cadence fires more than once inside one
            # update_actor, define the transaction explicitly as
            # last-candidate-wins and surface that overwrite in diagnostics.
            if self._pending_anchor_basis and self._state is not None:
                if hasattr(self._state, "anchor_q_stage_overwrites"):
                    self._state.anchor_q_stage_overwrites += 1
            self._pending_anchor_basis = {}
            target_basis = self._pending_anchor_basis

        updated = False
        for layer_idx in self._boundary_for_update():
            V = self._sketch.get(layer_idx, None)
            if V is None:
                if do_sync and self._hidden_size is not None:
                    V = torch.zeros(
                        int(self._hidden_size),
                        self._effective_rank(),
                        dtype=torch.float32,
                        device=self._sketch_device(),
                    )
                else:
                    continue
            Vsum = V.to(torch.float32)
            if do_sync:
                # Pool raw sketches across DP: V_global = Σ_ranks V (orth is
                # scale-invariant so SUM gives the pooled direction).
                torch.distributed.all_reduce(Vsum, op=torch.distributed.ReduceOp.SUM, group=group)
            q_new = orthonormalize(Vsum.to(self.qr_dtype), eps=self.reortho_eps)
            target_basis[layer_idx] = q_new.to(device=self._sketch_device(), dtype=torch.float32)
            updated = True

        self._reset_sketch()
        if updated and self._state is not None:
            # Count the anchor-owned Q update (distinct from the fast-path counter).
            if hasattr(self._state, "anchor_q_updates"):
                self._state.anchor_q_updates += 1
        return updated

    def broadcast_basis(self, *, src: int = 0, staged: bool = False) -> Optional[dict]:
        """``dist.broadcast`` the anchor's Q to every DP rank plus receipt.

        Broadcasts each boundary's ``Q`` from DP-group-local rank ``src`` (the
        anchor-owning rank) over the DP group, in the FIXED
        ``sorted(boundary_indices)`` order
        so every rank issues the identical collective sequence. Returns a per-
        boundary receipt dict ``{layer_idx: {src_checksum, recv_checksum,
        changed}}`` so the engine can log a ``[comm_eff][bcast]`` line and assert
        the copy LANDED (recv == src) and CHANGED from the pre-broadcast value
        when the source changed. Returns ``None`` when distributed is unavailable
        / single-rank (broadcast is a trivial no-op there).

        ``staged=True`` broadcasts ``_pending_anchor_basis`` without modifying
        the live basis used by the current PPO update. With ``sync_basis=true``
        the consensus ``orth(V)`` already produced a
        bit-identical Q on every rank, so this broadcast is belt-and-braces — but
        it is the load-bearing positive-receipt mechanism the sole-writer invariant
        requires: it proves every fast/DP rank holds the anchor's Q (a dropped
        broadcast / wrong group would surface as recv != src here).
        """
        if not torch.distributed.is_initialized():
            return None
        group = self._dp_group()
        world = torch.distributed.get_world_size(group=group)
        if world <= 1:
            return None
        # torch.distributed.broadcast's ``src`` is a GLOBAL rank even when a
        # subgroup is supplied. Callers reason in DP-local ranks; translate here
        # so DP group 1 does not accidentally wait for global rank 0.
        global_src = int(src)
        if group is not None and hasattr(torch.distributed, "get_global_rank"):
            global_src = int(torch.distributed.get_global_rank(group, int(src)))
        basis_store = self._pending_anchor_basis if staged else self._basis
        receipts: dict = {}
        for layer_idx in self._boundary_for_update():
            q = basis_store.get(layer_idx, None)
            if q is None:
                # Should not happen post-update, but keep the collective symmetric:
                # every rank must broadcast SOMETHING for this boundary. Seed a
                # deterministic cold-start Q so the shapes/sequence match.
                if self._hidden_size is None:
                    continue
                q = init_basis(
                    hidden_size=int(self._hidden_size),
                    rank=self.rank,
                    base_seed=self.base_seed,
                    layer_idx=layer_idx,
                ).to(device=self._sketch_device(), dtype=torch.float32)
                basis_store[layer_idx] = q
            # Pre-broadcast checksum (what THIS rank held going in).
            candidate_pre = float((q.detach().to(torch.float64) * self._ramp_like(q)).sum().item())
            comparison_q = self._basis.get(layer_idx, q) if staged else q
            pre = float((comparison_q.detach().to(torch.float64) * self._ramp_like(comparison_q)).sum().item())
            q_contig = q.detach().to(torch.float32).contiguous()
            torch.distributed.broadcast(q_contig, src=global_src, group=group)
            # copy_ the received value into the held basis (the receipt: every
            # non-src rank's Q is now bit-equal to src's).
            basis_store[layer_idx] = q_contig.to(device=self._sketch_device(), dtype=torch.float32)
            post_basis = basis_store[layer_idx]
            post = float((post_basis.detach().to(torch.float64) * self._ramp_like(post_basis)).sum().item())
            receipts[layer_idx] = {
                "src_checksum": post,  # after broadcast every rank == src's value
                "recv_checksum": post,
                "pre_checksum": pre,
                "candidate_pre_checksum": candidate_pre,
                "changed": bool(abs(post - pre) > 0.0),
            }
        if receipts and self._state is not None and hasattr(self._state, "anchor_q_broadcasts"):
            self._state.anchor_q_broadcasts += 1
        return receipts

    def activate_staged_anchor_basis(self) -> bool:
        """Publish the staged anchor candidate for the next PPO policy pair.

        The worker calls this exactly once after ``train_mini_batch`` returns,
        when every current-policy minibatch that shares the batch's recomputed
        ``old_log_probs`` has completed. No collective is required here because
        :meth:`broadcast_basis` already made the pending candidate identical on
        every DP rank.
        """

        # Do not introduce a collective at the outer update_actor exception
        # boundary: if one rank raises while another succeeds, a collective in
        # ``finally`` could strand the survivor. The candidate was already
        # broadcast and collectively verified at its in-step anchor fire; this
        # commit is intentionally a local atomic pointer handoff on every rank.
        if not self._pending_anchor_basis:
            return False

        expected = set(self._boundary_for_update())
        received = set(self._pending_anchor_basis)
        if received != expected:
            raise RuntimeError(
                "comm_eff.powersgd: staged anchor Q is incomplete at update_actor exit "
                f"(expected boundaries={sorted(expected)}, received={sorted(received)}). "
                "Refusing a partial basis activation."
            )
        self._basis = dict(self._pending_anchor_basis)
        self._pending_anchor_basis = {}
        self._anchor_basis_generation += 1
        if self._state is not None and hasattr(self._state, "anchor_q_activations"):
            self._state.anchor_q_activations += 1
        return True

    def discard_staged_anchor_basis(self) -> bool:
        """Discard a candidate produced by an actor update that did not commit."""

        had_pending = bool(self._pending_anchor_basis)
        self._pending_anchor_basis = {}
        return had_pending

    @property
    def anchor_basis_generation(self) -> int:
        """Number of staged anchor-Q generations successfully made live."""

        return int(self._anchor_basis_generation)

    def reset_basis_runtime(self) -> None:
        """Clear all non-checkpointed Q/sketch state after loading weights."""

        self._basis.clear()
        self._pending_anchor_basis.clear()
        self._anchor_basis_generation = 0
        self._fast_q_bootstrap_done = False
        self.abort_fast_q_bootstrap()
        self.fast_q_bootstrap_observations = 0
        self.fast_q_bootstrap_updates = 0
        self.fast_q_bootstrap_activations = 0
        self.fast_q_bootstrap_dense_observation_elements = 0.0
        self.fast_q_bootstrap_sync_elements = 0.0
        self.fast_q_bootstrap_cross_rank_max_rel_dev = None
        self._reset_sketch()

    @staticmethod
    def _ramp_like(q: torch.Tensor) -> torch.Tensor:
        """Fixed deterministic index-ramp weighting for a checksum of ``q``.

        Same construction as :meth:`basis_checksums` (sign/permutation/value
        sensitive) but on an arbitrary 2D tensor, in fp64 on ``q``'s device.
        """
        qd = q.detach()
        H, r = qd.shape
        return torch.arange(1, H * r + 1, dtype=torch.float64, device=qd.device).reshape(H, r)

    def _boundary_for_update(self) -> list[int]:
        """The FIXED, sorted boundary set every rank iterates in maybe_update_basis.

        Prefer the registered ``boundary_indices`` (identical on every rank by
        construction — they come from ``decoder_boundary_indices(L, pp_size)`` on
        the same model). Fall back to the sorted union of locally-bootstrapped
        bases / staged candidates / sketches if the compressor was never
        registered.
        """
        if self.boundary_indices:
            return sorted(self.boundary_indices)
        return sorted(
            set(self._basis.keys())
            | set(self._pending_anchor_basis.keys())
            | set(self._pending_fast_q_bootstrap_basis.keys())
            | set(self._sketch.keys())
            | set(self._fast_q_bootstrap_sketch.keys())
        )

    def _sketch_device(self):
        """Device the sketch / basis live on (first available basis or sketch,
        else CPU). Used to place the zero-sketch contribution + the new basis."""
        for d in (
            self._fast_q_bootstrap_sketch,
            self._pending_fast_q_bootstrap_basis,
            self._sketch,
            self._pending_anchor_basis,
            self._basis,
        ):
            for t in d.values():
                return t.device
        return torch.device("cpu")

    def _dp_group(self):
        """The process group the basis is synchronized over.

        The engine binds the actor's data-parallel group with ``set_dp_group``.
        If no group is bound, distributed collectives use the default process
        group, which is valid only when the world group is also the DP group.
        """
        return getattr(self, "_dp_process_group", None)

    def set_dp_group(self, group) -> None:
        """Bind the DP process group the basis consensus all-reduces over.

        Called by the engine with its data-parallel group. ``None`` selects the
        default process group. Pure setter; no collective."""
        self._dp_process_group = group

    def basis_checksums(self, *, staged: bool = False) -> dict:
        """Per-boundary fp64 checksum of the current basis Q.

        Returns ``{layer_idx: float}`` — a deterministic scalar summary of each
        ``Q`` (sum of Q ⊙ a fixed index ramp, in fp64, so sign/permutation/value
        differences all show up). The engine all-gathers these across ranks and
        verifies that ``sync_basis`` produced an identical consensus ``Q`` on
        every rank. Pure read."""
        basis_store = self._pending_anchor_basis if staged else self._basis
        out: dict = {}
        for layer_idx in self._boundary_for_update():
            q = basis_store.get(layer_idx, None)
            if q is None:
                continue
            qd = q.detach().to(torch.float64)
            H, r = qd.shape
            # A fixed deterministic weighting so a permutation/sign flip of Q's
            # columns or any value change moves the checksum.
            ramp = torch.arange(1, H * r + 1, dtype=torch.float64, device=qd.device).reshape(H, r)
            out[layer_idx] = float((qd * ramp).sum().item())
        return out

    def verify_basis_agreement_across_ranks(self, *, atol: float = 1e-6, staged: bool = False) -> Optional[float]:
        """Assert ``Q`` is identical on every DP rank.

        All-gathers a per-boundary checksum VECTOR (built over the FIXED
        ``boundary_indices``, so every rank contributes the same-length, same-order
        vector → the collective is symmetric and cannot deadlock) and asserts the
        max element-wise deviation across ranks is ``<= atol`` (scaled by the
        checksum magnitude). Returns the max relative cross-rank deviation
        (``0.0`` = bit-identical) so the engine can log it, or ``None`` when
        distributed is unavailable / single-rank (the check is trivially true).

        A non-zero result with ``sync_basis=true`` means the consensus all-reduce
        failed to make the basis agree (for example, wrong process group or
        asymmetric sketch). Raises on mismatch rather than silently training
        divergent codebooks.

        MUST be called on EVERY rank in lockstep (it issues an all_gather). The
        caller gates it on a condition identical across ranks (e.g. the first
        successful basis update, which all ranks reach on the same cadence step).
        """
        if not torch.distributed.is_initialized():
            return None
        group = self._dp_group()
        world = torch.distributed.get_world_size(group=group)
        if world <= 1:
            return 0.0
        idxs = self._boundary_for_update()
        if not idxs:
            return None
        sums = self.basis_checksums(staged=staged)
        # Fixed-order vector over the FIXED boundary set (0.0 for any boundary
        # missing locally — should not happen post-update, but keeps it symmetric).
        dev = self._sketch_device()
        vec = torch.tensor([float(sums.get(i, 0.0)) for i in idxs], dtype=torch.float64, device=dev)
        gathered = [torch.zeros_like(vec) for _ in range(world)]
        torch.distributed.all_gather(gathered, vec, group=group)
        ref = gathered[0]
        max_abs_dev = 0.0
        for g in gathered[1:]:
            max_abs_dev = max(max_abs_dev, float((g - ref).abs().max().item()))
        scale = float(ref.abs().max().item()) or 1.0
        max_rel_dev = max_abs_dev / scale
        if max_rel_dev > atol:
            raise RuntimeError(
                "comm_eff.powersgd: basis Q DIVERGED across DP ranks "
                f"(max_rel_dev={max_rel_dev:.3e} > atol={atol:.1e}) despite "
                f"sync_basis={self.sync_basis}. The shared codebook must be "
                "identical on every rank; a non-zero deviation "
                "means the consensus all-reduce used the wrong process group or an "
                "asymmetric sketch. Refusing to train divergent per-rank codebooks."
            )
        return max_rel_dev

    def _reset_sketch(self) -> None:
        self._sketch = {}
        self._sketch_count = {}
        self._sketched_this_gen = {}

    # ----------------------------------------------------------------------
    # Hook lifecycle
    # ----------------------------------------------------------------------
    def register(self, module: nn.Module) -> None:
        """Install projection hooks on the boundary decoder blocks (idempotent)."""
        if self._handles:
            return
        layers = find_decoder_layers(module)
        if layers is None:
            logger.warning(
                "comm_eff.powersgd: could not locate decoder layers on %s; "
                "no projection hooks registered (no-op this pass)",
                type(module).__name__,
            )
            return
        self.boundary_indices = decoder_boundary_indices(len(layers), self.pp_size)
        for idx in self.boundary_indices:
            self._handles.append(layers[idx].register_forward_hook(self._make_hook(idx)))
        logger.info(
            "comm_eff.powersgd: registered projection hooks on boundaries %s "
            "(L=%d, pp_size=%d, rank=%d, warm_start=%s, qr_dtype=%s)",
            self.boundary_indices,
            len(layers),
            self.pp_size,
            self._effective_rank(),
            self.warm_start,
            "fp32",
        )

    def unregister(self) -> None:
        """Remove all projection hooks (basis + sketch persist on the object)."""
        for handle in self._handles:
            handle.remove()
        self._handles = []

    @property
    def is_registered(self) -> bool:
        return bool(self._handles)
