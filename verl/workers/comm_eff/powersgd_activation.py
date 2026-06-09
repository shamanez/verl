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
``Y = M @ Q`` (plus the communication-free shared ``Q``), the identical logical
PP byte budget as the PRF mask at ``p = 1 − r/H`` (``r=102 ≡ p=0.95`` at
``H=2048``).

Three properties make this correct:

* **No straight-through.** ``Q`` is detached and ``M`` stays in-graph, so the
  autograd backward of ``M_hat = (M @ Q) @ Qᵀ`` is the exact self-adjoint
  projector ``dL/dM = (dL/dM_hat) Q Qᵀ`` — no STE, no custom autograd Function.
* **Deterministic zero-comm bootstrap.** ``Q_L = orth(randn(H, r))`` seeded by
  ``seed_L = (base_seed·1_000_003 + layer_idx·7919) & 0x7FFFFFFF``, drawn in fp32
  on CPU so it is bit-identical on every rank/device.
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

# Reuse the mask's boundary-selection + decoder-discovery helpers so the
# PowerSGD codec masks/compresses EXACTLY the same boundary blocks as the PRF
# mask (the matched-budget comparison requires identical boundary placement).
from verl.workers.comm_eff.activation_mask import (
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
# layer_idx*MIX_LAYER) & MASK31. The mask keeps the value positive int31 so it is
# a valid torch.Generator manual_seed on every backend.
_PRF_MIX_BASE = 1_000_003
_PRF_MIX_LAYER = 7919
_MASK31 = 0x7FFFFFFF


def powersgd_layer_seed(base_seed: int, layer_idx: int) -> int:
    """Deterministic per-layer basis seed.

    ``seed_L = (base_seed·1_000_003 + layer_idx·7919) & 0x7FFFFFFF``. Pure — no
    side effects. The mask keeps it int31-positive so the same value is a legal
    ``torch.Generator.manual_seed`` on CPU and every accelerator.
    """
    return (int(base_seed) * _PRF_MIX_BASE + int(layer_idx) * _PRF_MIX_LAYER) & _MASK31


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
    # return garbage. Replace non-finite entries with 0 before the QR; the
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
        # a cross-device error. dtype is already fp32 on both (EXP-25 id-0 hotfix).
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

    Mirrors :class:`verl.workers.comm_eff.activation_mask.ActivationMasker`'s
    lifecycle so the engine can drive it identically:

    * ``register(module)`` discovers the boundary decoder blocks, lazily
      bootstraps each block's deterministic basis ``Q`` and installs a
      forward hook that replaces the block output ``M`` with ``M_hat=(M@Q)@Qᵀ``.
    * ``unregister()`` removes the hooks.
    * ``set_context(global_step, ...)`` stamps the trainer step and bumps the
      per-micro-batch "forward generation" used to deduplicate the basis sketch
      against gradient-checkpoint recompute.
    * ``maybe_update_basis()`` runs the block-power-iteration ``Q ← orth(V)`` at
      cadence, AFTER the gradient-bearing actor work — called by the engine's
      end-of-train_batch hook so ``Q`` is frozen across the paired GRPO forwards.

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
        q_basis: str = "act",
        state: Any = None,
    ):
        self.rank = int(rank)
        self.base_seed = int(base_seed)
        self.pp_size = int(pp_size)
        self.update_cadence = int(update_cadence)
        self.warm_start = bool(warm_start)
        self.compress_recompute = bool(compress_recompute)
        self.sync_basis = bool(sync_basis)
        # EXP-26 Step C: the Q-basis FAMILY (content of the sketch orth(V) consumes
        # at FIXED rank). "act" (default) = the EXP-25 activation-energy basis
        # (V += Mᵀ(MQ)), byte-identical. The RLVR-native families bias V toward
        # GRPO update energy; they require the Step-C sketch-construction the
        # analyst specifies after Step A's H2 finding (gated — see _build_family_sketch).
        self.q_basis = str(q_basis)
        # EXP-25 (R2): when True the ANCHOR owns Q. The fast net is then a pure
        # read-only consumer: its end-of-step maybe_update_basis is gated OFF (by
        # the engine call site) AND its forward-hook sketch accumulation is gated
        # OFF here (so V never grows on the fast path). Q is updated ONLY by the
        # anchor's slow-net forward (anchor_update_basis) and propagated by
        # broadcast_basis. False (default) = EXP-20 fast-owns-Q (byte-identical).
        self.anchor_owns_q = bool(anchor_owns_q)
        # Transient toggle: True ONLY inside the anchor's stale-weight forward so
        # the SAME forward hook that is suppressed on the fast path DOES fold the
        # slow-net activations into V. Set/cleared by the engine around the anchor
        # forward. Never persisted.
        self._anchor_sketch_mode = False
        if str(qr_dtype) not in ("fp32", "bf16"):
            raise ValueError(f"powersgd qr_dtype must be one of (fp32, bf16); got {qr_dtype!r}")
        # qr_dtype controls only the orth/QR + sketch math; fp32 is required for
        # orthogonality. bf16 is a diagnostic knob.
        self.qr_dtype = torch.float32 if str(qr_dtype) == "fp32" else torch.bfloat16
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
        # The logical PP byte budget actually carried, n·r per token-layer; the
        # engine logs comm_eff/logical_pp_bytes_powersgd_y_only against the PRF
        # equivalent for budget-equality checks.
        self.last_y_coords_per_token: int = self.rank

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
        sample_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> None:
        """Stamp the trainer step + bump the forward generation for this micro-batch.

        ``sample_ids`` / ``position_ids`` are accepted for signature-parity with
        the mask (the engine calls both the same way) but the PowerSGD codec does
        not key on token identity — its basis is shared across all tokens. The
        generation bump is what dedupes the sketch against grad-ckpt recompute.
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
            # gradient-bearing actor-train forward has grad enabled here. (Reading
            # is_grad_enabled() inside the no_grad block below would always be
            # False — the bug this fixes.)
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

            # EXP-25 (R2): anchor stale-forward harvest. The anchor forward must be
            # CLEAN (uncompressed) — its gradient is G_anchor (→ M) and its
            # activations feed Q. So when _anchor_sketch_mode is on we fold the raw
            # activation into the sketch V (V += Mᵀ(MQ)) but return h UNCHANGED (no
            # M_hat projection). This harvests Q from the slow net without
            # corrupting the clean anchor gradient. grad_enabled is True (the
            # anchor runs forward_only=False), so the sketch lands.
            if compressor._anchor_sketch_mode:
                with torch.no_grad():
                    q_fp32 = compressor._ensure_basis(layer_idx, device=M.device, dtype=M.dtype)
                    q_act = q_fp32.to(dtype=M.dtype)
                    if compressor._should_accumulate_sketch(layer_idx, grad_enabled=grad_enabled):
                        M32 = M.detach().float()
                        Y32 = (M32 @ q_fp32)  # (N, r) in fp32
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

                # EXP-26 Step A: dump A (the boundary activation M), Â=(A@Q)Qᵀ and
                # the basis Q for THIS boundary, keyed by (global_step,
                # fwd_generation) so the analyst can recompute reconstruction_rel_error
                # from the dumped fp32 tensors and confirm it matches `rel` above (the
                # fp32-dump-fidelity invariant). Gated on the gradient-bearing fast
                # forward only (grad_enabled), so it captures the SAME activations the
                # sketch/codec used — never the old-logprob recompute. Detached/fp32,
                # dump-only. No-op unless a capture writer is attached to the state.
                _w = getattr(compressor._state, "_capture_writer", None) if compressor._state is not None else None
                if _w is not None and grad_enabled and not compressor._anchor_sketch_mode:
                    _tname = f"boundary_{layer_idx}"
                    _stats = {
                        "layer_idx": int(layer_idx),
                        "rank": int(q_act.shape[1]),
                        "fwd_generation": int(compressor._fwd_generation),
                        "reconstruction_rel_error": rel,
                        "q_cond": q_cond,
                    }
                    _w.dump(role="A", target_name=_tname, tensor=M32,
                            global_step=int(compressor._global_step),
                            optimizer_tick=int(compressor._fwd_generation), extra=_stats)
                    _w.dump(role="A_hat", target_name=_tname, tensor=Mhat32,
                            global_step=int(compressor._global_step),
                            optimizer_tick=int(compressor._fwd_generation), extra=_stats)
                    _w.dump(role="Q", target_name=_tname, tensor=q_fp32,
                            global_step=int(compressor._global_step),
                            optimizer_tick=int(compressor._fwd_generation), extra=_stats)

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

            # Count one projection application (mirrors the mask counter).
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

        **EXP-25 (R2) anchor-owns-Q.** In that mode the FAST path must NEVER fold
        into V (Q is anchor-owned), so on the fast path (``_anchor_sketch_mode``
        False) this returns False unconditionally. Inside the anchor's stale-
        weight forward (``_anchor_sketch_mode`` True) we DO accumulate — and we
        bypass the ``path_tag == train`` gate because the anchor pass deliberately
        runs with ``path_tag=None`` (the generation-dedupe still applies).
        """
        if not grad_enabled:
            return False
        # EXP-25 (R2): anchor-owns-Q routing.
        if self.anchor_owns_q:
            if not self._anchor_sketch_mode:
                # Fast path: NEVER accumulate (Q is owned by the anchor).
                return False
            # Anchor stale-forward: accumulate regardless of path_tag (None here),
            # still deduped per forward-generation against grad-ckpt recompute.
            return self._sketched_this_gen.get(layer_idx, -1) != self._fwd_generation
        # Legacy fast-owns-Q (EXP-20): unchanged.
        state = self._state
        tag = getattr(state, "path_tag", None) if state is not None else None
        if tag != TRAIN_TAG:
            return False
        return self._sketched_this_gen.get(layer_idx, -1) != self._fwd_generation

    # ----------------------------------------------------------------------
    # Basis update (block power iteration) — called AFTER the actor backward
    # ----------------------------------------------------------------------
    def maybe_update_basis(self, *, is_clean_step: bool) -> bool:
        """Run ``Q ← orth(V)`` at cadence, AFTER the gradient-bearing actor work.

        Called by the engine's end-of-``train_batch`` hook. Skips the update on a
        clean step (the dense refresh keeps the prior basis, mirroring the mask's
        no-V-no-Q-on-clean rule) and on non-cadence steps. Clears the sketch
        after a successful update. Returns True iff Q was updated.

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
        is just an ``H×r`` all-reduce per boundary at each non-clean update.

        Orthonormalization is scale-invariant, so summing the raw per-rank ``V``s
        (rather than averaging) is exactly the pooled direction — no per-rank
        count re-weighting is needed for the basis.

        **Collective safety (deadlock guard).** The all-reduce iterates the FIXED
        ``sorted(self.boundary_indices)`` on EVERY rank, contributing a correctly
        shaped ZERO sketch for any boundary a rank happens to be missing locally,
        so all ranks issue the identical sequence of collectives. A rank-relative
        iteration over ``self._sketch`` (different/missing keys, different order)
        would mismatch the collective and hang. All ranks call ``update_actor``
        in lockstep, so a
        symmetric per-boundary collective set is sufficient.
        """
        # Fail-closed (R2 sole-Q-writer): the FAST net must NEVER write Q when the
        # anchor owns it. The sole call site (engine_workers.py) is gated on
        # ``fast_owns_q``, so reaching here in anchor_owns_q mode means that gate
        # leaked — Q would get two writers (fast overwrite vs anchor broadcast),
        # silently invalidating the stale-anchor experiment. Crash instead of
        # drifting; this is the durable engine-side equivalent of the probe's
        # ``powersgd_basis_updates == 0`` check.
        assert not getattr(self, "anchor_owns_q", False), (
            "comm_eff.powersgd: maybe_update_basis() entered in anchor_owns_q mode — the "
            "FAST net must NEVER update Q when the anchor owns it (#25 R2 sole-writer "
            "invariant). The engine_workers gate (fast_owns_q) must have leaked."
        )
        if is_clean_step:
            # No V accumulated on a clean step (the train forward ran dense, no
            # hook) — nothing to do; keep Q_t. Clear any stray sketch defensively.
            self._reset_sketch()
            return False
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

        if not self.warm_start:
            # Cold start: re-bootstrap every update from the per-layer seed,
            # then take one orth(V) step from there (diagnostic path). Re-seed
            # the FIXED boundary set so it is symmetric across ranks too.
            for layer_idx in self._boundary_for_update():
                if self._hidden_size is not None:
                    self._basis[layer_idx] = init_basis(
                        hidden_size=int(self._hidden_size),
                        rank=self.rank,
                        base_seed=self.base_seed,
                        layer_idx=layer_idx,
                    ).to(device=self._sketch_device(), dtype=torch.float32)

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
    # EXP-25 (R2): anchor-owned Q — slow-net update + broadcast
    # ----------------------------------------------------------------------
    def set_anchor_sketch_mode(self, on: bool) -> None:
        """Toggle whether the forward hook folds activations into V (anchor pass).

        The engine sets this True around the anchor's stale-weight forward (so the
        SAME projection hook that is suppressed on the fast path harvests the
        slow-net activations into V) and clears it after. Also resets the
        per-generation dedupe so the anchor forward is counted fresh. Pure setter.
        """
        self._anchor_sketch_mode = bool(on)

    def anchor_update_basis(self) -> bool:
        """EXP-25 (R2): ``Q ← orth(V)`` from the ANCHOR's slow-net sketch.

        The SAME block-power-iteration math as :meth:`maybe_update_basis` (DP-sync
        of the raw sketch over the DP group, then fp32 ``orth(V)`` per boundary),
        but driven by the anchor refresh (cadence already gated by the engine —
        called only when the anchor fires) instead of the fast end-of-step hook,
        and consuming V built from the slow-net stale-weight forward activations.
        Clears the sketch after. Returns True iff any Q was updated.

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

        # Without sync and with an empty local sketch, nothing to do (no collective).
        if not do_sync and not self._sketch:
            return False

        updated = False
        for layer_idx in self._boundary_for_update():
            V = self._sketch.get(layer_idx, None)
            if V is None:
                if do_sync and self._hidden_size is not None:
                    r = self._effective_rank()
                    V = torch.zeros(
                        int(self._hidden_size), r, dtype=torch.float32, device=self._sketch_device()
                    )
                else:
                    continue
            Vsum = V.to(torch.float32)
            if do_sync:
                # Pool raw sketches across DP: V_global = Σ_ranks V (orth is
                # scale-invariant so SUM gives the pooled direction).
                torch.distributed.all_reduce(Vsum, op=torch.distributed.ReduceOp.SUM, group=group)
            q_new = orthonormalize(Vsum.to(self.qr_dtype), eps=self.reortho_eps)
            self._basis[layer_idx] = q_new.to(device=self._sketch_device(), dtype=torch.float32)
            updated = True

        self._reset_sketch()
        if updated and self._state is not None:
            # Count the anchor-owned Q update (distinct from the fast-path counter).
            if hasattr(self._state, "anchor_q_updates"):
                self._state.anchor_q_updates += 1
        return updated

    def broadcast_basis(self, *, src: int = 0) -> Optional[dict]:
        """EXP-25 (R2): ``dist.broadcast`` the anchor's Q to every DP rank + receipt.

        Broadcasts each boundary's ``Q`` from rank ``src`` (the anchor-owning
        rank) over the DP group, in the FIXED ``sorted(boundary_indices)`` order
        so every rank issues the identical collective sequence. Returns a per-
        boundary receipt dict ``{layer_idx: {src_checksum, recv_checksum,
        changed}}`` so the engine can log a ``[comm_eff][bcast]`` line and assert
        the copy LANDED (recv == src) and CHANGED from the pre-broadcast value
        when the source changed. Returns ``None`` when distributed is unavailable
        / single-rank (broadcast is a trivial no-op there).

        With ``sync_basis=true`` the consensus ``orth(V)`` already produced a
        bit-identical Q on every rank, so this broadcast is belt-and-braces — but
        it is the load-bearing POSITIVE-RECEIPT mechanism the (R2) invariant
        requires: it proves every fast/DP rank holds the anchor's Q (a dropped
        broadcast / wrong group would surface as recv != src here).
        """
        if not torch.distributed.is_initialized():
            return None
        group = self._dp_group()
        world = torch.distributed.get_world_size(group=group)
        if world <= 1:
            return None
        receipts: dict = {}
        for layer_idx in self._boundary_for_update():
            q = self._basis.get(layer_idx, None)
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
                self._basis[layer_idx] = q
            # Pre-broadcast checksum (what THIS rank held going in).
            pre = float((q.detach().to(torch.float64) * self._ramp_like(q)).sum().item())
            q_contig = q.detach().to(torch.float32).contiguous()
            torch.distributed.broadcast(q_contig, src=src, group=group)
            # copy_ the received value into the held basis (the receipt: every
            # non-src rank's Q is now bit-equal to src's).
            self._basis[layer_idx] = q_contig.to(device=self._sketch_device(), dtype=torch.float32)
            post = float(
                (self._basis[layer_idx].detach().to(torch.float64) * self._ramp_like(self._basis[layer_idx])).sum().item()
            )
            receipts[layer_idx] = {
                "src_checksum": post,  # after broadcast every rank == src's value
                "recv_checksum": post,
                "pre_checksum": pre,
                "changed": bool(abs(post - pre) > 0.0),
            }
        if receipts and self._state is not None and hasattr(self._state, "anchor_q_broadcasts"):
            self._state.anchor_q_broadcasts += 1
        return receipts

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
        bases / sketches if the compressor was never registered (unit tests).
        """
        if self.boundary_indices:
            return sorted(self.boundary_indices)
        return sorted(set(self._basis.keys()) | set(self._sketch.keys()))

    def _sketch_device(self):
        """Device the sketch / basis live on (first available basis or sketch,
        else CPU). Used to place the zero-sketch contribution + the new basis."""
        for d in (self._sketch, self._basis):
            for t in d.values():
                return t.device
        return torch.device("cpu")

    def _dp_group(self):
        """The process group the basis is synchronized over.

        For THIS actor — FSDP, Ulysses SP=1, and NO tensor/pipeline-parallel dim
        in the *training* mesh (the launcher's TP=2 is rollout-only, a separate
        vLLM mesh) — the world process group is the DP group: world_size ==
        data_parallel_size == 4. So the default (world) group is correct here and
        the basis is pooled over exactly the ranks whose data shards we want to
        consensus over. The engine may inject a narrower group via
        ``set_dp_group`` if a future config adds a TP/PP dim to the training mesh
        (in which case the all-reduce MUST reduce over the DP subgroup only, not
        the world). When unset we use the default group.
        """
        return getattr(self, "_dp_process_group", None)

    def set_dp_group(self, group) -> None:
        """Bind the DP process group the basis consensus all-reduces over.

        Called by the engine with its data-parallel group. ``None`` (default)
        ⇒ the world group, which is correct when world == DP (SP=1, no TP/PP in
        the training mesh). Pure setter; no collective."""
        self._dp_process_group = group

    def basis_checksums(self) -> dict:
        """Per-boundary fp64 checksum of the current basis Q.

        Returns ``{layer_idx: float}`` — a deterministic scalar summary of each
        ``Q`` (sum of Q ⊙ a fixed index ramp, in fp64, so sign/permutation/value
        differences all show up). The engine all-gathers these across ranks and
        verifies that ``sync_basis`` produced an identical consensus ``Q`` on
        every rank. Pure read."""
        out: dict = {}
        for layer_idx in self._boundary_for_update():
            q = self._basis.get(layer_idx, None)
            if q is None:
                continue
            qd = q.detach().to(torch.float64)
            H, r = qd.shape
            # A fixed deterministic weighting so a permutation/sign flip of Q's
            # columns or any value change moves the checksum.
            ramp = torch.arange(1, H * r + 1, dtype=torch.float64, device=qd.device).reshape(H, r)
            out[layer_idx] = float((qd * ramp).sum().item())
        return out

    def verify_basis_agreement_across_ranks(self, *, atol: float = 1e-6) -> Optional[float]:
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
        sums = self.basis_checksums()
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
                "identical on every rank (hard-invariant #4); a non-zero deviation "
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
        # EXP-26 Step C: fail LOUD if an unimplemented Q-basis family is selected.
        # "act" (the EXP-25 activation-energy basis) is the only family with a
        # built sketch path today; the RLVR-native families {grad,adv,tail,hybrid,
        # ticket} require the Step-C sketch construction that the analyst specifies
        # from Step A's H2 finding. Silently falling back to "act" would make a
        # Step-C arm a mislabeled control, so crash instead. Steps A and B both run
        # with q_basis="act", so this never blocks them.
        if self.q_basis != "act":
            raise NotImplementedError(
                f"comm_eff.powersgd.q_basis={self.q_basis!r} (EXP-26 Step C) is not yet "
                "implemented — only 'act' (the activation-energy basis) has a built sketch "
                "path. Step C runs ONLY after Step A's H2 finding (Q_act under-captures "
                "GRPO update energy); the analyst names the family's sketch construction "
                "then, and the runner implements it on the exp/* branch before launching "
                "Step C. Steps A and B use q_basis='act'."
            )
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
            "fp32" if self.qr_dtype == torch.float32 else "bf16",
        )

    def unregister(self) -> None:
        """Remove all projection hooks (basis + sketch persist on the object)."""
        for handle in self._handles:
            handle.remove()
        self._handles = []

    @property
    def is_registered(self) -> bool:
        return bool(self._handles)
