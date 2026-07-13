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

"""Per-worker state object for the communication-efficient compression method.

The integration contract is deliberately asymmetric so the disabled path is a
**strict no-op**:

* ``maybe_build_comm_eff_state(config)`` returns ``None`` when
  ``config.enabled`` is false (or the config is absent). No object is
  constructed, **no RNG is drawn**, no buffer is allocated, no forward hook is
  registered. The actor therefore holds ``self._comm_eff_state = None`` for a
  dense GRPO run, and every hook below short-circuits on the ``None`` check.

* Only when ``config.enabled`` is true is a ``CommEffState`` constructed; that
  is where mask RNG, anchor EMA buffers and the spectral workspace get
  allocated lazily by ``build()``.

Because construction is gated, a dense GRPO run with this scaffolding merged
consumes the exact same RNG sequence and issues the exact same collective ops
as one without it.

The instrumented counters (``mask_applications``, ``anchor_backwards``,
``spectral_corrections``) live on the state object. When disabled there is no
state object, so the counters are *absent* rather than zero — which the
caller can treat as equivalent to ``== 0`` (no comm_eff op fired). When enabled
they start at 0 and increment per fired op.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:  # avoid an import cycle at runtime; only needed for type hints
    from verl.workers.config.comm_eff import CommEffConfig

logger = logging.getLogger(__name__)

__all__ = [
    "CommEffState",
    "FastGradRing",
    "GradLagBuffer",
    "maybe_build_comm_eff_state",
    "comm_eff_metrics",
    "resolve_compression_type",
    "PATH_TAGS",
    "TRAIN_TAG",
    "OLD_LOGPROB_TAG",
    "MASK_ELIGIBLE_TAGS",
    "mask_eligible_tags",
]

# The exhaustive set of execution-path tags a comm_eff state can carry. The
# activation mask is allowed to fire on EXACTLY ONE of these (``train``); every
# other tag is an RL-measurement / serving path that must stay uncompressed even
# while masking is enabled. Contamination raises in the mask hook.
#
#   train        -> actor-train forward/backward (the ONLY masked path)
#   rollout      -> vLLM/sglang generation (policy rollouts + eval generation)
#   old_logprob  -> compute_log_prob (old policy log-prob recompute)
#   ref_logprob  -> compute_ref_log_prob (reference policy log-prob)
#   val          -> validation / eval pass (_validate)
#   infer        -> generic infer_batch entrypoint (critic infer, etc.)
#   ckpt         -> checkpoint save / load forward (none expected, tagged for safety)
TRAIN_TAG = "train"
# The old-policy log-prob recompute path. Always present in PATH_TAGS for
# contamination accounting; it is eligible for masking only when
# comm_eff.mask.mask_recompute=true.
OLD_LOGPROB_TAG = "old_logprob"
PATH_TAGS = (
    TRAIN_TAG,
    "rollout",
    OLD_LOGPROB_TAG,
    "ref_logprob",
    "val",
    "infer",
    "ckpt",
)

# The set of execution-path tags the activation mask is allowed to fire on by
# default: only ``train``. ``mask_eligible_tags(state)`` widens this to
# ``{train, old_logprob}`` only when ``state.mask.mask_recompute=True``. ``None``
# (anchor pass) is never eligible, so anchors stay unmasked unconditionally.
MASK_ELIGIBLE_TAGS: frozenset = frozenset({TRAIN_TAG})


def mask_eligible_tags(state: Any) -> frozenset:
    """Return the set of path tags the activation mask is allowed to fire on
    for the given ``state``. Pure read — no side effects, no allocation.

    The default eligibility (``{TRAIN_TAG}``) is widened to
    ``{TRAIN_TAG, OLD_LOGPROB_TAG}`` *only* when both
    ``state.mask.enabled`` and ``state.mask.mask_recompute`` are truthy.
    Anything else (disabled state, missing mask sub-config, ``mask_recompute``
    unset / falsy) returns the singleton default.

    ``None`` (anchor pass) is intentionally NOT in either set: the
    anchor circuit runs unmasked regardless of this flag.
    """
    if state is None:
        return MASK_ELIGIBLE_TAGS
    mask_cfg = getattr(getattr(state, "config", None), "mask", None)
    if mask_cfg is None:
        return MASK_ELIGIBLE_TAGS
    if not bool(getattr(mask_cfg, "enabled", False)):
        return MASK_ELIGIBLE_TAGS
    if not bool(getattr(mask_cfg, "mask_recompute", False)):
        return MASK_ELIGIBLE_TAGS
    return frozenset({TRAIN_TAG, OLD_LOGPROB_TAG})


def _is_enabled(config: Any) -> bool:
    """Read the ``enabled`` flag from a comm_eff config that may be a dataclass,
    an OmegaConf node, a plain dict, or ``None``. Pure read — no side effects."""
    if config is None:
        return False
    if isinstance(config, dict):
        return bool(config.get("enabled", False))
    return bool(getattr(config, "enabled", False))


def resolve_compression_type(config: Any) -> str:
    """Resolve the effective boundary codec from a comm_eff config.

    Returns one of ``{"dense", "prf_mask", "powersgd"}``. Pure read — no side
    effects, no allocation. The resolution is back-compatible:

    * an explicit ``compression_type`` of ``prf_mask`` or ``powersgd`` wins;
    * ``dense`` (the field default) falls back to the mask selector — if the
      mask sub-config is enabled with ``p > 0`` the codec is ``prf_mask``;
      otherwise ``dense``.

    This keeps existing mask configs working while PowerSGD is selected explicitly.
    """
    ctype = getattr(config, "compression_type", "dense") if config is not None else "dense"
    if ctype in ("prf_mask", "powersgd"):
        return ctype
    # ctype == "dense": honor the mask selector for back-compat.
    mask_cfg = getattr(config, "mask", None)
    mask_enabled = bool(getattr(mask_cfg, "enabled", False)) if mask_cfg is not None else False
    if mask_enabled and float(getattr(mask_cfg, "p", 0.0)) > 0.0:
        return "prf_mask"
    return "dense"


class FastGradRing:
    """Fire-aware ring of the fast compressed per-target gradients
    ``G_comp(t)``.

    A fire at tick ``t`` only ever consumes tick ``t − delay_K``, and fires sit
    at ``t ≡ 0 (mod cadence)``, so the ONLY ticks worth storing satisfy
    ``tick ≡ (−delay_K) mod cadence`` (:meth:`tick_retained`), bounding the ring
    at ``delay_K // cadence + 1`` entries. Everything else is
    rejected at push time (the caller also pre-checks :meth:`tick_retained` so
    non-replayable ticks never pay the D2H either).

    Entries are ``tick -> (grads, norms)`` where ``grads`` is
    ``{canon_name: CPU tensor}`` (fp32, detached) and ``norms`` is the matching
    ``{canon_name: float}`` Frobenius norms (computed on-device at extraction so
    the CPU consumer never re-reduces 5 GB just for a denominator). Consumers:
    the geometry probe's within-pair codec error
    ``delta(t) = G_anc_rep(t) - G_comp_ring(t-K)`` and the ``delayed_ef``
    merger's fire-time residual refresh. Pure
    container — no collectives, no RNG, CPU-testable. CPU residency is ASSERTED
    on push (the zero-GPU-memory-growth invariant).
    """

    def __init__(self, delay_K: int, cadence: int = 1):
        assert delay_K >= 0, f"delay_K must be >= 0, got {delay_K}"
        self.delay_K = int(delay_K)
        self.cadence = max(1, int(cadence))
        self._keep_residue = (-self.delay_K) % self.cadence
        self._maxlen = self.delay_K // self.cadence + 1
        self._entries: OrderedDict[int, tuple] = OrderedDict()

    def tick_retained(self, tick: int) -> bool:
        """True iff a future fire or delayed_ef residual refresh can request ``tick``."""
        return (int(tick) % self.cadence) == self._keep_residue

    def push(self, tick: int, grads: dict, norms: Optional[dict] = None) -> bool:
        """Store ``tick``'s per-target G_comp dict. Returns False (storing
        nothing) for a non-retained or duplicate tick. Asserts CPU residency
        and the ≤ ``delay_K // cadence + 1`` entry bound on every push."""
        tick = int(tick)
        if not self.tick_retained(tick) or tick in self._entries:
            return False
        for _name, _t in grads.items():
            assert getattr(_t, "device", None) is not None and _t.device.type == "cpu", (
                f"FastGradRing.push(tick={tick}) received a non-CPU tensor for {_name!r} "
                f"({_t.device}) — the ring must be CPU-resident (zero-GPU-growth invariant)."
            )
            break  # one representative check per push is enough (single extraction site)
        self._entries[tick] = (grads, dict(norms or {}))
        while len(self._entries) > self._maxlen:
            self._entries.popitem(last=False)
        assert len(self._entries) <= self._maxlen, (
            f"FastGradRing blew its bound: {len(self._entries)} > maxlen={self._maxlen} "
            f"(delay_K={self.delay_K} cadence={self.cadence}) — eviction regressed."
        )
        return True

    def get(self, tick: int) -> Optional[tuple]:
        """Exact-tick lookup → ``(grads, norms)`` or None.

        No fallback is used: the m5 / delayed_ef pairing must be the exact
        ``t − delay_K`` entry.
        """
        return self._entries.get(int(tick))

    def pop(self, tick: int) -> None:
        """Drop a consumed entry (fires advance monotonically; ``t − delay_K``
        can never be requested again). Keeps steady-state at ~1 entry."""
        self._entries.pop(int(tick), None)

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def ticks(self) -> list:
        return list(self._entries.keys())


class GradLagBuffer:
    """Rolling buffer of the last ``max_lag`` ticks of per-target
    ``G_comp`` for the m4 lag-autocorrelation ``cos(G_comp(t), G_comp(t−j))``,
    j=1..max_lag.

    Bounded at ``max_lag`` stored entries (default 5) plus the in-flight current
    tick the engine holds during the fire computation. Pushed every tick; entries
    older than ``max_lag`` roll off automatically. Same ``(grads, norms)`` entry
    shape and CPU-residency assert as :class:`FastGradRing`. Pure container,
    CPU-testable.
    """

    def __init__(self, max_lag: int = 5):
        assert 1 <= int(max_lag) <= 5, f"max_lag must be in [1, 5], got {max_lag}"
        self.max_lag = int(max_lag)
        self._entries: OrderedDict[int, tuple] = OrderedDict()

    def push(self, tick: int, grads: dict, norms: Optional[dict] = None) -> bool:
        tick = int(tick)
        if tick in self._entries:
            return False
        for _name, _t in grads.items():
            assert getattr(_t, "device", None) is not None and _t.device.type == "cpu", (
                f"GradLagBuffer.push(tick={tick}) received a non-CPU tensor for {_name!r} "
                f"({_t.device}) — the lag buffer must be CPU-resident (zero-GPU-growth invariant)."
            )
            break
        self._entries[tick] = (grads, dict(norms or {}))
        while len(self._entries) > self.max_lag:
            self._entries.popitem(last=False)
        assert len(self._entries) <= self.max_lag, (
            f"GradLagBuffer blew its bound: {len(self._entries)} > max_lag={self.max_lag}."
        )
        return True

    def get(self, tick: int) -> Optional[tuple]:
        return self._entries.get(int(tick))

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def ticks(self) -> list:
        return list(self._entries.keys())


class CommEffState:
    """Per-worker communication-efficient compression state.

    Constructed **only** when ``comm_eff.enabled=true``. Holds the operation
    counters, mask RNG generator, anchor EMA buffers and spectral workspace.
    The disabled path
    never instantiates this class — see ``maybe_build_comm_eff_state``.
    """

    def __init__(self, config: CommEffConfig):
        # Invariant: never construct a disabled state. The factory enforces it;
        # this assert catches a future caller that forgets to go through it.
        assert _is_enabled(config), (
            "CommEffState must not be constructed when comm_eff.enabled=false; "
            "go through maybe_build_comm_eff_state() so the disabled path stays a no-op."
        )
        self.config = config
        self.enabled = True
        self._built = False
        # Active boundary codec, resolved in build(). Until build() the
        # conservative default is "dense" (no codec). Read by metrics() and the
        # engine to decide which compressor lifecycle to drive.
        self.compression_type = "dense"

        # Operation counters surfaced into training metrics under comm_eff/*.
        self.mask_applications = 0
        self.anchor_backwards = 0
        self.spectral_corrections = 0

        # Anchor-circuit counters. These are guard/falsifier metrics:
        #   anchor_mask_applications  — mask hooks fired DURING the anchor pass.
        #                               MUST stay 0 (the anchor runs
        #                               unmasked even though it's on the train
        #                               path). Captured as a delta around the
        #                               anchor fwd/bwd, not the global counter.
        #   anchor_grad_corrected     — anchor gradients fed THROUGH correct_matrix.
        #                               MUST stay 0 (G_anchor read raw
        #                               into the EMA before any correction).
        #   anchor_rollouts_generated — new rollouts the anchor produced. MUST 0.
        #   anchor_rewards_recomputed — reward recomputations by the anchor. MUST 0.
        #   anchor_optimizer_steps    — optimizer.step() calls on the anchor pass.
        #                               MUST stay 0 (snapshot off the optimizer's
        #                               param group; the anchor only reads grads).
        #   anchor_batch_fraction     — fraction of the rollout-expanded batch the
        #                               anchor backward consumed (1.0 = whole
        #                               batch; <1.0 ⇒ an OOM-bounded subset, which
        #                               the engine logs with a reason).
        self.anchor_mask_applications = 0
        self.anchor_grad_corrected = 0
        self.anchor_rollouts_generated = 0
        self.anchor_rewards_recomputed = 0
        self.anchor_optimizer_steps = 0
        self.anchor_batch_fraction = 1.0
        # Monotonic trainer-step counter the anchor cadence is keyed on (advanced
        # once per actor train_batch).
        self.anchor_step = 0

        # Monotonic optimizer-step counter the spectral cadence is keyed
        # on. Advanced once per actor train_batch by the grad-correction hook
        # (_maybe_comm_eff_grad_correction), in lockstep with anchor_step (both
        # +1 per train_batch), so when anchor.cadence == spectral.cadence the
        # spectral correction fires on EXACTLY the steps the anchor EMA was just
        # refreshed (a fresh basis, never a stale one). Kept independent of
        # anchor_step so spectral cadence still works when the anchor circuit is
        # disabled. Stays 0 on the dense/disabled path (the hook short-circuits
        # on the None/enabled guard before advancing it).
        self.spectral_step = 0

        # Periodic clean-step counter. Incremented once per trainer step
        # whose (global_step % clean_cadence) == 0 while clean_cadence > 0 — i.e.
        # every step on which masking is forced OFF and AdamW takes a step on the
        # true dense gradient. Surfaced as comm_eff/clean_steps so logs can prove
        # the clean step fired at exactly steps clean_cadence, 2*clean_cadence,
        # ... (incremented from the train stamp in update_actor, NOT the old_logprob
        # stamp, so a step is counted once even when mask_recompute also forces the
        # recompute clean). Stays 0 when clean_cadence == 0 (every prior config).
        self.clean_steps = 0
        # The trainer global_step most recently threaded in. -1 = never threaded
        # (e.g. a unit test, or the dense/disabled path). Read by is_clean_step().
        self.global_step = -1

        # The activation masker (first circuit). Constructed in build(); None
        # when the mask sub-config is disabled.
        self.masker = None

        # PowerSGD activation compressor (boundary codec). Constructed
        # in build() ONLY when compression_type == "powersgd"; None otherwise
        # (the disabled path, the dense codec, and the prf_mask codec never
        # touch it). Mutually exclusive with `masker`: a run is either the mask
        # codec or the powersgd codec, never both.
        self.powersgd = None
        # PowerSGD op counters. Cumulative.
        #   powersgd_applications  — projection hooks fired on the train path.
        #   powersgd_basis_updates — orth(V) basis refreshes taken (one per
        #                            non-clean cadence step).
        self.powersgd_applications = 0
        self.powersgd_basis_updates = 0

        # Anchor-owns-Q counters.
        #   anchor_q_updates    — orth(V) Q refreshes the ANCHOR computed from its
        #                         slow-net stale-forward activations (replaces the
        #                         fast net's maybe_update_basis in owns_q mode).
        #   anchor_q_broadcasts — dist.broadcast of (Q, M) from the anchor-owning
        #                         rank to every DP rank (one per refresh).
        self.anchor_q_updates = 0
        self.anchor_q_broadcasts = 0
        # Per-step count of matrices the signed_ema merger no-op'd to
        # G_noisy because M was cold (||M||<=eps). On step 1 == corrected (M cold,
        # NOT zeroed); → 0 after M warms. The silent grad-zeroing falsifier.
        self.merger_coldM_fallbacks = 0
        # Per-step count of ef_powersgd targets whose accumulated
        # error-feedback residual was RESET because the target's logical 2D shape
        # changed (no stale carry across shape change). The shape-aware-residual
        # invariant surfaces this so the probe can prove the reset fired.
        self.residual_reset_on_shape_mismatch = 0
        # Cumulative count of passive family-screen builds (one per
        # anchor refresh that built candidate Q_f for the q_basis_passive families).
        # 0 unless the screen is configured; lets the probe confirm the screen fired.
        self.family_screen_builds = 0
        # Cumulative count of anchor refreshes that replayed a paired
        # (batch[t-delay_K], generator-snapshot) instead of the current batch.
        # 0 unless anchor.replay_paired_batch=true; post-warmup it advances once
        # per anchor fire, so the probe can prove every fire went through replay.
        self.anchor_replay_fires = 0
        # Look-ahead (weight-projection) anchor counters. Both stay 0 unless
        # anchor.lookahead_anchor is active: `lookahead_fires` counts fires that
        # loaded a projected theta_hat into the clone; `lookahead_warmup_fallbacks`
        # counts fires that fell back to the raw stale snapshot because the
        # source ring was not yet warm. Post-warmup every fire increments
        # exactly one of the two, so fires+fallbacks == anchor fires.
        self.lookahead_fires = 0
        self.lookahead_warmup_fallbacks = 0
        # E2 (warmup_mode=no_correct): fires SKIPPED before the projector is
        # ready — the anchor pass (clone fwd/bwd + M update) is not run, so M
        # stays cold and the merger passes fast grads through unchanged. Stays 0
        # in the default stale_correct mode. On the earliest-legal (min_snaps=2)
        # arm this increments exactly once (fire 1), then the projector engages.
        self.warmup_no_correct_skips = 0
        # Pure sliding rank1_relex state. These counters are emitted only when
        # that mode is active, preserving fixed-linear/disabled metric output.
        # rank1_m_ready is the explicit optimizer safety barrier: the correction
        # hook advances cadence but returns before touching any matrix until a
        # complete-window projected anchor has successfully updated M (and Q in
        # the q_only/anchor-owned-Q arm).
        self.rank1_m_ready = False
        self.rank1_q_only_fires = 0
        self.rank1_correction_bypass_ticks = 0
        self.rank1_fires = 0
        self.rank1_history_checkpoints = 0
        self.rank1_history_deltas = 0
        self.rank1_window_span = 0
        self.rank1_prediction_horizon = 0
        self.rank1_evr_mean = 0.0
        self.rank1_r2_mean = 0.0
        self.rank1_zero_motion_tensors = 0
        # Causal sampled-weight probe. Scalar-only and telemetry-only; emitted
        # only when probe.rank1_projection_enabled=true.
        self.rank1_probe_predictions = 0
        self.rank1_probe_resolutions = 0
        self.rank1_probe_pending = 0
        self.rank1_probe_projected_rmse = 0.0
        self.rank1_probe_stale_rmse = 0.0
        self.rank1_probe_skill = 0.0
        self.rank1_probe_direction_cos = 0.0
        # Geometry probe. All None / 0 unless probe.geometry_enabled
        # (off-path parity: the OFF path builds no ring, no buffer, no stash).
        #   fast_grad_ring      — FastGradRing of G_comp(t−K) (≤2 entries, CPU).
        #                         ALSO built (probe-independent) when the
        #                         delayed_ef merger is selected — it feeds δ.
        #   grad_lag_buffer     — GradLagBuffer for m4 (≤5 stored + in-flight ≤6).
        #   geometry_probe_fires— cumulative fires with a complete m1–m7 record.
        #   _probe_fire_stash   — per-fire dict set by the anchor refresh
        #                         (G_anc_rep/G_anc_old CPU fp32 + norms + m7
        #                         stats + replay metadata + loss_mismatch),
        #                         consumed + freed by the end-of-batch hook.
        #   _probe_prev_rep     — previous fire's (grads, norms) of G_anc_rep
        #                         (CPU fp32) for m6 = cos(M_rep(t), M_rep(t−5));
        #                         exactly M_rep at β_anc=0 (1-entry retention).
        self.fast_grad_ring = None
        self.grad_lag_buffer = None
        self.geometry_probe_fires = 0
        self._probe_fire_stash = None
        self._probe_prev_rep = None
        self._probe_prev_rep_tick = -1
        # Optional tensor-capture writer (CommEffState owns it so
        # the anchor / merger / projection hooks can all reach it via the state).
        # None unless comm_eff.capture.enabled — built in build(). Pure I/O sink.
        self._capture_writer = None
        # The single per-train_batch optimizer tick all capture
        # roles key on, stamped at the start of the real fast-path forward (see
        # FSDPEngine.forward_backward_batch). -1 = not yet stamped (fall back to
        # current_optimizer_tick()). Unifying the key across roles is what keeps the
        # max_ticks budget counting OPTIMIZER ticks, not per-forward generations.
        self._capture_tick = -1

        # Whether masking is currently active. Set True only on entry to the
        # actor-train forward/backward (around update_actor) and cleared on
        # exit, so log-prob / ref / infer / val / checkpoint forwards stay clean.
        self.mask_active = False

        # Explicit execution-path tag. Defaults to
        # ``None`` (no path entered). Each entrypoint stamps it before its
        # forward: ``update_actor`` -> "train"; ``compute_log_prob`` ->
        # "old_logprob"; ``compute_ref_log_prob`` -> "ref_logprob"; rollout ->
        # "rollout"; validation -> "val"; ``infer_batch`` -> "infer";
        # checkpoint save/load -> "ckpt". The mask hook asserts the tag is
        # "train" before it fires, so a leak onto any other path raises rather
        # than corrupting the RL-measurement machinery. ``mask_active`` remains
        # the fast gate; ``path_tag`` is the cross-check.
        self.path_tag: Optional[str] = None

        # Per-path mask-application counters. The contract: every key except
        # ``train`` MUST stay 0 for the whole run. They are surfaced into
        # metrics as ``comm_eff/mask_applications/<tag>`` so callers can
        # confirm confinement by KEY PREFIX (no substring false positives).
        self.mask_applications_by_path = {tag: 0 for tag in PATH_TAGS}

        # The spectral filter (third circuit). Constructed in build() when
        # ``comm_eff.spectral.enabled`` is true; None otherwise. Holds the
        # (seeded) anchor-EMA cache and applies the correction formula at the
        # grad-correction hook point. See verl.workers.comm_eff.spectral_filter.
        self.spectral = None

        # FSDP gradient-representation discovery log.
        # The engine's grad-correction hook fills this once, on the first
        # correction, with type(p.grad), the grad container shape, the logical
        # 2D matrix shape, the FSDP wrapping/version, and whether correction ran
        # before/after FSDP gradient reduction and gradient clipping, for >=1
        # target matrix.
        self.fsdp_grad_repr: dict = {}

        # Per-target ||G_proj - G_mask|| / ||G_mask|| from the most recent
        # correction. Logged faithfully (never clamped) because the value is not
        # provably <=1 for arbitrary anchors. Surfaced under comm_eff/spectral/*.
        self.spectral_rel_change: dict = {}

    def build(self, module: Any) -> None:
        """Construct the enabled circuits (mask and/or spectral).

        Idempotent. When ``comm_eff.mask.enabled`` is true this constructs an
        ``ActivationMasker`` (no hooks registered yet — the engine registers
        them only on entry to the train forward and removes them on exit). When
        ``comm_eff.spectral.enabled`` is true this constructs the
        ``SpectralFilter`` with its (optionally seeded) anchor-EMA cache. Anchor
        refresh remains lazy in the engine.
        """
        if self._built:
            return
        # Resolve the active boundary codec once. Exactly one of the
        # mask / powersgd compressors is constructed; `dense` constructs neither.
        self.compression_type = resolve_compression_type(self.config)
        mask_cfg = getattr(self.config, "mask", None)
        if self.compression_type == "prf_mask":
            # Imported lazily so the disabled path never pays the import cost.
            from verl.workers.comm_eff.activation_mask import ActivationMasker

            self.masker = ActivationMasker(
                p=float(mask_cfg.p),
                base_seed=int(getattr(mask_cfg, "seed", 0)),
                pp_size=int(getattr(mask_cfg, "pp_size", 8)),
                rescale=bool(getattr(mask_cfg, "rescale", False)),
                rescale_mode=str(getattr(mask_cfg, "rescale_mode", "auto")),
                state=self,
            )
        elif self.compression_type == "powersgd":
            # Imported lazily so the disabled / mask paths never pay the cost.
            from verl.workers.comm_eff.powersgd_activation import PowerSGDActivationCompressor

            ps_cfg = getattr(self.config, "powersgd", None)
            # anchor-owns-Q lives on the anchor sub-config so it can be
            # set alongside anchor.enabled/cadence/delay_K. Read it here to build the
            # compressor in the right mode (fast Q-update gated off + slow-net Q).
            anc_cfg_for_q = getattr(self.config, "anchor", None)
            anchor_owns_q = bool(getattr(anc_cfg_for_q, "owns_q", False)) if anc_cfg_for_q is not None else False
            self.powersgd = PowerSGDActivationCompressor(
                rank=int(getattr(ps_cfg, "rank", 102)),
                base_seed=int(getattr(ps_cfg, "seed", 0)),
                pp_size=int(getattr(ps_cfg, "pp_size", 8)),
                update_cadence=int(getattr(ps_cfg, "update_cadence", 1)),
                warm_start=bool(getattr(ps_cfg, "warm_start", True)),
                compress_recompute=bool(getattr(ps_cfg, "compress_recompute", True)),
                sync_basis=bool(getattr(ps_cfg, "sync_basis", False)),
                qr_dtype=str(getattr(ps_cfg, "qr_dtype", "fp32")),
                reortho_eps=float(getattr(ps_cfg, "reortho_eps", 1e-6)),
                anchor_owns_q=anchor_owns_q,
                # Live Q-basis family.
                q_basis=str(getattr(ps_cfg, "q_basis", "act")),
                # Passive screen families plus hybrid column split.
                q_basis_passive=list(getattr(ps_cfg, "q_basis_passive", []) or []),
                hybrid_act_cols=int(getattr(ps_cfg, "hybrid_act_cols", -1)),
                hybrid_grad_cols=int(getattr(ps_cfg, "hybrid_grad_cols", -1)),
                # Anchor cadence for the Q-broadcast byte amortization.
                anchor_cadence=int(getattr(anc_cfg_for_q, "cadence", 1)) if anc_cfg_for_q is not None else 1,
                state=self,
            )
            logger.info(
                "comm_eff: powersgd compressor built (rank=%s update_cadence=%s warm_start=%s "
                "compress_recompute=%s sync_basis=%s qr_dtype=%s)",
                self.powersgd.rank,
                self.powersgd.update_cadence,
                self.powersgd.warm_start,
                self.powersgd.compress_recompute,
                self.powersgd.sync_basis,
                getattr(ps_cfg, "qr_dtype", "fp32"),
            )
            print(
                f"[comm_eff] powersgd codec: rank={self.powersgd.rank} "
                f"update_cadence={self.powersgd.update_cadence} warm_start={self.powersgd.warm_start} "
                f"compress_recompute={self.powersgd.compress_recompute} "
                f"sync_basis={self.powersgd.sync_basis} "
                f"qr_dtype={getattr(ps_cfg, 'qr_dtype', 'fp32')} "
                f"anchor_owns_q={self.powersgd.anchor_owns_q}",
                flush=True,
            )

        spec_cfg = getattr(self.config, "spectral", None)
        spec_enabled = bool(getattr(spec_cfg, "enabled", False)) if spec_cfg is not None else False
        if spec_enabled:
            # Imported lazily so the disabled path never pays the import cost.
            from verl.workers.comm_eff.spectral_filter import SpectralFilter

            self.spectral = SpectralFilter(
                beta_anc=float(getattr(spec_cfg, "beta_anc", 0.95)),
                # Storage layer default: gpu.
                ema_device=str(getattr(spec_cfg, "ema_device", "gpu")),
                correction_mode=str(getattr(spec_cfg, "correction_mode", "signed_ema")),
                inject_gamma=float(getattr(spec_cfg, "inject_gamma", 1.0)),
                blend_eta=float(getattr(spec_cfg, "blend_eta", 0.5)),
                signed_ema_alpha=float(getattr(spec_cfg, "signed_ema_alpha", 0.0)),
                # Error-feedback residual knobs (ef_powersgd).
                ef_decay=float(getattr(spec_cfg, "ef_decay", 0.0)),
                ef_clip=float(getattr(spec_cfg, "ef_clip", 0.0)),
                # K-delayed exact codec residual weight (delayed_ef).
                delayed_ef_lambda=float(getattr(spec_cfg, "delayed_ef_lambda", 0.0)),
                # Additive stale-anchor rank-r_sb sub-basis. base_seed = the
                # codec seed (powersgd.seed, identical on every DP rank) so the
                # per-target randomized SVD that builds δ_subbasis is bit-identical
                # across ranks (the multi-rank-agreement invariant).
                delta_subbasis_rank=int(getattr(spec_cfg, "delta_subbasis_rank", 0)),
                delta_subbasis_family=str(getattr(spec_cfg, "delta_subbasis_family", "tail")),
                # Sub-basis weight plus linear decay.
                delta_subbasis_weight=float(getattr(spec_cfg, "delta_subbasis_weight", 1.0)),
                delta_subbasis_decay_steps=int(getattr(spec_cfg, "delta_subbasis_decay_steps", 0)),
                # Hold-then-decay: gamma holds at full weight for hold_steps,
                # THEN decays (0 default ⇒ the existing linear-from-0 schedule).
                delta_subbasis_hold_steps=int(getattr(spec_cfg, "delta_subbasis_hold_steps", 0)),
                base_seed=int(getattr(getattr(self.config, "powersgd", None), "seed", 0) or 0),
                # Zero-mean tunable cross-rank-identical perturbation.
                perturb_sigma=float(getattr(spec_cfg, "perturb_sigma", 0.0)),
                perturb_seed=int(getattr(spec_cfg, "perturb_seed", 0)),
                # Delta-momentum (normalized EMA, stationary gain exactly 1).
                # mu=0 skips the branch. The buffer is built
                # from the DP-mean δ ⇒ cross-rank identical. age_decay fades the held
                # correction by age.
                delta_momentum_mu=float(getattr(spec_cfg, "delta_momentum_mu", 0.0)),
                delta_momentum_age_decay=bool(getattr(spec_cfg, "delta_momentum_age_decay", False)),
                # Adaptive dose (centered gate). mode=off/kappa=0 keeps
                # lambda_t == delayed_ef_lambda. lambda_t = clamp(lambda + kappa*(c_bar - c_t),
                # 0, lambda_cap), built from the DP-mean gm + M_rep ⇒ cross-rank
                # identical.
                adaptive_lambda_mode=str(getattr(spec_cfg, "adaptive_lambda_mode", "off")),
                adaptive_lambda_kappa=float(getattr(spec_cfg, "adaptive_lambda_kappa", 0.0)),
                lambda_cap=float(getattr(spec_cfg, "lambda_cap", 2.0)),
                # When False, skip per-step spectral DIAGNOSTIC overhead
                # (per-matrix rel_change compute+sync+print). Default True =
                # byte-identical. Nothing the optimizer sees changes.
                diagnostics=bool(getattr(spec_cfg, "diagnostics", True)),
            )
            logger.info(
                "comm_eff: spectral filter built (beta_anc=%s ema_device=%s correction_mode=%s "
                "inject_gamma=%s blend_eta=%s signed_ema_alpha=%s)",
                self.spectral.beta_anc,
                self.spectral.ema_device,
                self.spectral.correction_mode,
                self.spectral.inject_gamma,
                self.spectral.blend_eta,
                self.spectral.signed_ema_alpha,
            )
            # Discovery line is string-valued, so it goes to stdout only:
            # reduce_metrics does np.mean on every metric value and crashes on a
            # string. Keep it out of metrics.
            anc_cfg = getattr(self.config, "anchor", None)
            anchor_enabled = bool(getattr(anc_cfg, "enabled", False)) if anc_cfg is not None else False
            isolation_mode = "clone" if anchor_enabled else "n/a (anchor.enabled=false)"
            print(
                f"[comm_eff] spectral storage: ema_device={self.spectral.ema_device} "
                f"correction_mode={self.spectral.correction_mode} "
                f"signed_ema_alpha={self.spectral.signed_ema_alpha} "
                f"ef_decay={self.spectral.ef_decay} ef_clip={self.spectral.ef_clip} "
                f"anchor_backward_isolation_mode={isolation_mode}",
                flush=True,
            )

        # Fire-aware fast-grad ring plus m4 lag buffer. The ring is built
        # when EITHER the geometry probe is on (m5 needs G_comp_ring(t−K))
        # OR the delayed_ef merger is selected (delta refresh needs the same
        # entry); the lag buffer is probe-only (m4). The OFF path constructs
        # NOTHING — flag-OFF parity (both attributes stay None, no allocation).
        anc_cfg_rings = getattr(self.config, "anchor", None)
        probe_cfg = getattr(self.config, "probe", None)
        probe_on = bool(getattr(probe_cfg, "geometry_enabled", False)) if probe_cfg is not None else False
        delayed_ef_on = spec_enabled and str(getattr(spec_cfg, "correction_mode", "signed_ema")) == "delayed_ef"
        if (probe_on or delayed_ef_on) and anc_cfg_rings is not None:
            _ring_delay_K = int(getattr(anc_cfg_rings, "delay_K", 0))
            _ring_cadence = int(getattr(anc_cfg_rings, "cadence", 1))
            self.fast_grad_ring = FastGradRing(delay_K=_ring_delay_K, cadence=_ring_cadence)
            if probe_on:
                self.grad_lag_buffer = GradLagBuffer(max_lag=int(getattr(probe_cfg, "m4_lags", 5)))
            lag_buffer_label = (
                "maxlen=" + str(self.grad_lag_buffer.max_lag) if self.grad_lag_buffer is not None else None
            )
            print(
                f"[geometry-probe] armed: geometry_enabled={probe_on} delayed_ef={delayed_ef_on} "
                f"fast_grad_ring(maxlen={self.fast_grad_ring._maxlen}, delay_K={_ring_delay_K}, "
                f"cadence={_ring_cadence}) "
                # NB `is not None`, not truthiness: GradLagBuffer defines __len__,
                # so an EMPTY (just-armed) buffer is falsy and the build print
                # would lie "None" while the buffer exists (observed on the
                # first launch; functional paths all use `is None`).
                f"lag_buffer={lag_buffer_label} "
                f"out_dir={getattr(probe_cfg, 'out_dir', '') if probe_cfg is not None else ''} "
                f"rank0_only={getattr(probe_cfg, 'rank0_only', True) if probe_cfg is not None else True}",
                flush=True,
            )

        # Build the diagnostic capture writer iff
        # comm_eff.capture.enabled. Strict no-op (None) otherwise — the disabled /
        # non-capture path never touches the filesystem. Lazy import so the
        # disabled path never pays the import cost.
        cap_cfg = getattr(self.config, "capture", None)
        cap_enabled = bool(getattr(cap_cfg, "enabled", False)) if cap_cfg is not None else False
        if cap_enabled:
            from verl.workers.comm_eff.capture import maybe_build_capture_writer

            self._capture_writer = maybe_build_capture_writer(self.config)
            print(
                f"[comm_eff] capture ENABLED: dir={getattr(cap_cfg, 'capture_dir', '') or './captures'} "
                f"max_ticks={getattr(cap_cfg, 'max_ticks', 10)} "
                f"stratified_targets={getattr(cap_cfg, 'stratified_targets', 0)} "
                f"capture_g_dense={getattr(cap_cfg, 'capture_g_dense', False)} "
                f"capture_fresh_anchor={getattr(cap_cfg, 'capture_fresh_anchor', False)} "
                f"dump_dtype={getattr(cap_cfg, 'dump_dtype', 'fp32')}",
                flush=True,
            )
        self._built = True

    def rank1_relex_active(self) -> bool:
        """Whether this state owns the opt-in pure sliding rank1 projector."""
        anchor_cfg = getattr(self.config, "anchor", None)
        return bool(getattr(anchor_cfg, "lookahead_anchor", False)) and (
            str(getattr(anchor_cfg, "lookahead_mode", "disabled")) == "rank1_relex"
        )

    def reset_rank1_runtime(self) -> None:
        """Reset non-checkpointed rank1 state after loading model weights.

        A resumed run must establish a new local base and refill the exact
        checkpoint window. This mirrors a fresh worker even when a test or
        orchestration path loads a checkpoint into an already-used process.
        """
        if not self.rank1_relex_active():
            return
        for name in (
            "_rank1_history",
            "_rank1_projector",
            "_rank1_base_batch",
            "_rank1_base_canary",
            "_rank1_projection_probe",
            "_anchor_replay_ring",
            "_anchor_queue",
            "_anchor_canary_by_tick",
        ):
            if hasattr(self, name):
                delattr(self, name)
        self.anchor_step = 0
        self.spectral_step = 0
        self.anchor_backwards = 0
        self.spectral_corrections = 0
        self.anchor_q_updates = 0
        self.anchor_q_broadcasts = 0
        self.powersgd_basis_updates = 0
        self.anchor_replay_fires = 0
        self.lookahead_fires = 0
        self.lookahead_warmup_fallbacks = 0
        self.warmup_no_correct_skips = 0
        self.rank1_m_ready = False
        self.rank1_q_only_fires = 0
        self.rank1_correction_bypass_ticks = 0
        self.rank1_fires = 0
        self.rank1_history_checkpoints = 0
        self.rank1_history_deltas = 0
        self.rank1_window_span = 0
        self.rank1_prediction_horizon = 0
        self.rank1_evr_mean = 0.0
        self.rank1_r2_mean = 0.0
        self.rank1_zero_motion_tensors = 0
        self.rank1_probe_predictions = 0
        self.rank1_probe_resolutions = 0
        self.rank1_probe_pending = 0
        self.rank1_probe_projected_rmse = 0.0
        self.rank1_probe_stale_rmse = 0.0
        self.rank1_probe_skill = 0.0
        self.rank1_probe_direction_cos = 0.0
        if self.spectral is not None:
            # A resumed rank1 run rewarms from a fresh local checkpoint base.
            # Clear every correction-family history, not only M_anchor, so an
            # ef/delayed-ef/momentum ablation cannot resurrect pre-resume state
            # when rank1_m_ready becomes true again.
            for name in (
                "_anchor",
                "_ef_residual",
                "_delayed_ef_delta",
                "_delta_momentum",
                "_delta_momentum_last_step",
                "_adaptive_lambda_hist",
                "_subbasis_energy_ratios",
            ):
                cache = getattr(self.spectral, name, None)
                if cache is not None and hasattr(cache, "clear"):
                    cache.clear()
            if hasattr(self.spectral, "current_step"):
                self.spectral.current_step = 0
        if self.powersgd is not None:
            getattr(self.powersgd, "_basis", {}).clear()
            getattr(self.powersgd, "_sketch", {}).clear()
            if hasattr(self.powersgd, "clear_family_harvest"):
                self.powersgd.clear_family_harvest()

    def set_path_tag(self, tag: Optional[str]) -> None:
        """Stamp the current execution-path tag.

        ``tag`` must be one of :data:`PATH_TAGS` or ``None`` (clears the tag).
        The mask hook reads this and asserts it equals ``train`` before firing.
        Validating here turns a typo in an entrypoint into an immediate error
        instead of silent mask leakage.
        """
        if tag is not None and tag not in PATH_TAGS:
            raise ValueError(f"unknown comm_eff path tag {tag!r}; expected one of {PATH_TAGS} or None")
        self.path_tag = tag

    def current_optimizer_tick(self) -> int:
        """The optimizer tick the CURRENT train_batch will land on (1-based).

        Capture key. All capture roles (the powersgd-hook A/Â/Q, the merger
        G_comp/G_corr, the anchor M/G_anchor, the parallel G_dense, the delay_K=0
        fresh-anchor probe) key on THIS so they co-locate under one
        ``(global_step, optimizer_tick)`` and the ``max_ticks`` budget counts
        OPTIMIZER ticks (not the hundreds of per-micro-batch forward generations
        the activation hook would otherwise emit, which starved the budget).

        Both ``spectral_step`` (grad-correction hook) and ``anchor_step`` (anchor
        refresh) advance once per ``train_batch`` AFTER/at the top of the batch, so
        DURING the batch's forward+backward they trail by one. The per-batch tick
        is therefore ``max(spectral_step, anchor_step) + 1``. We take the MAX
        because either counter may be inert on a given arm: a no-merger arm
        (``spectral.enabled=false``, e.g. the A1 plain-PowerSGD audit arm) never
        advances ``spectral_step`` (the grad-correction hook early-returns on
        ``spectral is None``) — so keying on ``spectral_step`` alone would collapse
        EVERY tick to 1 and the dumps would overwrite. ``anchor_step`` is the live
        per-batch counter there (it advances at the top of every anchor refresh).
        Symmetrically an anchor-disabled arm keeps advancing ``spectral_step``.
        Pure read.
        """
        # NB ``anchor_step`` is incremented at the TOP of the anchor refresh
        # (before the stamp), so it ALREADY equals N during the batch; while
        # ``spectral_step`` is incremented at the END (grad-correction), so it
        # trails at N-1 during the batch. The batch tick is thus
        # ``max(anchor_step, spectral_step + 1)`` — N from whichever counter is
        # live. (Called only at the stamp sites — anchor-top + the fast forward —
        # where this is exact; later reads go through the stamped capture_tick().)
        ss = int(getattr(self, "spectral_step", 0) or 0)
        as_ = int(getattr(self, "anchor_step", 0) or 0)
        return max(as_, ss + 1)

    def capture_tick(self) -> int:
        """The optimizer tick the CURRENT train_batch's capture dumps key on.

        Returns the value stamped on ``self._capture_tick`` at the start of the
        real fast-path forward (so every role in this batch shares ONE key), or
        falls back to ``current_optimizer_tick()`` if it was never stamped (e.g. a
        unit test, or a code path that did not go through
        ``forward_backward_batch``). Pure read.
        """
        t = int(getattr(self, "_capture_tick", -1) or -1)
        return t if t >= 0 else self.current_optimizer_tick()

    def is_clean_step(self, global_step: Optional[int] = None) -> bool:
        """True iff the given trainer ``global_step`` is a clean step.

        A clean step is one on which masking is forced OFF for the whole step
        and AdamW refreshes its moments on the true dense gradient. The rule is
        ``clean_cadence > 0 and (global_step % clean_cadence) == 0``. When
        ``global_step`` is ``None`` the most-recently-threaded ``self.global_step``
        is used. Pure read — no side effects, no allocation.

        ``clean_cadence`` is read from the config (default 0 ⇒ always False).
        ``global_step <= 0`` is never a clean step: step 0 is the pre-train
        ``val_before_train`` / first-increment boundary (the trainer's first
        train step is global_step=1), and a negative sentinel means "never
        threaded" — masking stays ON in both cases.
        """
        cadence = int(getattr(self.config, "clean_cadence", 0) or 0)
        if cadence <= 0:
            return False
        gs = self.global_step if global_step is None else int(global_step)
        if gs <= 0:
            return False
        return (gs % cadence) == 0

    def should_run_spectral_correction(self, step: Optional[int] = None) -> bool:
        """True iff the spectral grad-correction fires on this optimizer step.

        Mirrors :meth:`is_clean_step` (and ``anchor_should_fire``): a pure
        predicate keyed on the monotonic per-optimizer-step counter
        ``self.spectral_step`` (1-based — the grad-correction hook advances it
        before calling this). The rule is ``(step % cadence) == 0`` with
        ``cadence = comm_eff.spectral.cadence`` (default ``1`` ⇒ always True).

        ``cadence=2`` fires on steps 2, 4, 6, … — aligned with an
        ``anchor.cadence=2`` refresh (both counters advance once per
        ``train_batch``), so the correction always sees a freshly-refreshed
        anchor EMA. ``step <= 0`` never fires (the "never advanced" sentinel /
        pre-train boundary), matching ``is_clean_step``'s ``gs <= 0`` guard.
        Pure read — no side effects, no allocation.
        """
        spec_cfg = getattr(getattr(self, "config", None), "spectral", None)
        cadence = int(getattr(spec_cfg, "cadence", 1)) if spec_cfg is not None else 1
        if cadence < 1:
            return False
        s = self.spectral_step if step is None else int(step)
        if s <= 0:
            return False
        return (s % cadence) == 0

    def note_mask_application(self) -> None:
        """Record one mask-hook fire against the current path tag.

        Called by the activation masker from inside a hook. Increments both the
        aggregate counter (``mask_applications``) and the per-path
        counter for ``self.path_tag``. A fire while the tag is anything other
        than ``train`` is a contamination event; the masker asserts against it
        before calling this, but if the assert is ever disabled (``python -O``)
        the per-path counter still records the leak.
        """
        self.mask_applications += 1
        tag = self.path_tag if self.path_tag in self.mask_applications_by_path else TRAIN_TAG
        self.mask_applications_by_path[tag] += 1

    def note_powersgd_application(self) -> None:
        """Record one PowerSGD projection-hook fire. Called from the
        compressor hook. Pure counter bump — no allocation."""
        self.powersgd_applications += 1

    def note_powersgd_basis_update(self) -> None:
        """Record one PowerSGD block-power-iteration basis refresh.
        Called from ``maybe_update_basis`` after a successful ``orth(V)``."""
        self.powersgd_basis_updates += 1

    def note_family_screen(self, n_families: int = 0) -> None:
        """Record one passive family-screen build (one per anchor
        refresh that built candidate Q_f). Pure counter bump."""
        self.family_screen_builds += 1

    def path_metrics(self) -> dict:
        """Per-path mask-application counters, surfaced under a stable KEY prefix.

        Emits ``comm_eff/mask_applications/<tag>`` for every tag. The only
        nonzero key should be ``.../train``; any other nonzero key is
        the contamination falsifier. Emitting all keys (including the zeros)
        makes the confinement machine-checkable without substring grepping.
        """
        return {f"comm_eff/mask_applications/{tag}": count for tag, count in self.mask_applications_by_path.items()}

    def mask_ratio_metrics(self) -> dict:
        """Return the most-recently-measured masked fraction per boundary layer.

        Surfaced as ``comm_eff/mask_ratio`` (mean across boundaries) plus a
        per-boundary breakdown. Empty when no mask fired this step.
        """
        if self.masker is None or not self.masker.last_mask_ratio:
            return {}
        ratios = self.masker.last_mask_ratio
        mean_ratio = sum(ratios.values()) / len(ratios)
        out = {"comm_eff/mask_ratio": mean_ratio}
        for idx, r in sorted(ratios.items()):
            out[f"comm_eff/mask_ratio/layer_{idx}"] = r
        # Matched-budget metric: PRF kept coords per token = (1-p)*H. This should
        # equal PowerSGD's n*r (=rank) within 1% so the two
        # arms carry the IDENTICAL logical PP byte budget (a confound guard). Use
        # the configured p (exact, not the measured ratio which jitters per draw)
        # and the recorded H.
        H = getattr(self.masker, "hidden_size", None)
        p = float(getattr(self.masker, "p", 0.0))
        if H is not None:
            out["comm_eff/logical_pp_bytes_prf"] = float((1.0 - p) * float(H))
        return out

    def spectral_metrics(self) -> dict:
        """Return spectral-correction metrics for logging.

        Surfaces ONLY NUMERIC values: the per-target
        ``||G_proj - G_mask|| / ||G_mask||`` ratios (faithfully, never clamped)
        plus their mean. Empty when no correction has fired.

        IMPORTANT — these metrics flow into ``actor_output.meta_info["metrics"]``
        and then through ``verl.utils.metric.utils.reduce_metrics``, which does
        ``np.mean(val)`` on EVERY value. A string value (e.g. a flattened FSDP
        discovery field like ``grad_container_type="Tensor"``) makes np.mean
        raise ``UFuncNoLoopError: ufunc 'add' did not contain a loop ... <U59``
        and crashes the trainer's metric-reduction at the end of the step. The
        FSDP gradient-representation discovery log is therefore emitted via
        stdout/logger, not this metrics dict. Reducible metrics must stay numeric.
        """
        if self.spectral is None:
            return {}
        out: dict = {}
        if self.spectral_rel_change:
            vals = list(self.spectral_rel_change.values())
            out["comm_eff/spectral/rel_change_mean"] = sum(vals) / len(vals)
            for name, r in self.spectral_rel_change.items():
                out[f"comm_eff/spectral/rel_change/{name}"] = r
        return out

    def powersgd_metrics(self) -> dict:
        """Return PowerSGD codec health/diagnostic metrics.

        All NUMERIC (the reduce_metrics-must-stay-numeric contract). Empty when
        the powersgd codec is not active. Surfaces:
          comm_eff/powersgd_q_cond                     — mean Q condition number
                                                         (≈1 orthonormal; non-finite
                                                         ⇒ basis collapse falsifier).
          comm_eff/powersgd_q_cond/layer_<i>           — per-boundary breakdown.
          comm_eff/powersgd_reconstruction_rel_error   — mean ||M-M_hat||/||M||
                                                         (must stay < 1.0).
          comm_eff/powersgd_reconstruction_rel_error/layer_<i>
          comm_eff/logical_pp_bytes_powersgd_y_only    — n·r coords/token-layer
                                                         (matched-budget metric
                                                         against PRF q·H bytes).
          comm_eff/powersgd_basis_updates              — cumulative orth(V) refreshes.
        """
        if self.powersgd is None:
            return {}
        out: dict = {}
        qc = getattr(self.powersgd, "last_q_cond", {})
        if qc:
            finite = [v for v in qc.values() if v == v and v not in (float("inf"), float("-inf"))]
            # Report the mean of finite conds; if ANY is non-finite, surface inf
            # for the mean so finiteness checks trip.
            if len(finite) == len(qc):
                out["comm_eff/powersgd_q_cond"] = sum(finite) / len(finite)
            else:
                out["comm_eff/powersgd_q_cond"] = float("inf")
            for idx, v in sorted(qc.items()):
                out[f"comm_eff/powersgd_q_cond/layer_{idx}"] = v
        re = getattr(self.powersgd, "last_reconstruction_rel_error", {})
        if re:
            out["comm_eff/powersgd_reconstruction_rel_error"] = sum(re.values()) / len(re)
            for idx, v in sorted(re.items()):
                out[f"comm_eff/powersgd_reconstruction_rel_error/layer_{idx}"] = v
        # Logical PP byte budget actually carried: n·r coordinate-values per
        # token-layer (Y = M @ Q is the only thing "sent"). This should equal
        # the PRF q·H within 1% (matched budget).
        out["comm_eff/logical_pp_bytes_powersgd_y_only"] = float(
            getattr(self.powersgd, "last_y_coords_per_token", self.powersgd.rank)
        )
        out["comm_eff/powersgd_basis_updates"] = self.powersgd_basis_updates
        out["comm_eff/powersgd_applications"] = self.powersgd_applications
        # Max relative cross-rank deviation of the
        # consensus basis Q after the first update (0.0 = bit-identical on every
        # DP rank). Set once by update_actor's verify_basis_agreement_across_ranks
        # and omitted until the check runs.
        qdev = getattr(self, "_powersgd_q_agreement_dev", None)
        if qdev is not None:
            out["comm_eff/powersgd_q_cross_rank_max_rel_dev"] = float(qdev)
        # Cumulative passive family-screen builds.
        out["comm_eff/family_screen_builds"] = self.family_screen_builds
        # Measured inter-stage communication volume this tick (element
        # counts; the ratio is dtype-invariant). Y=M@Q
        # (N·r) coords + amortized Q-broadcast vs the dense activation (N·H). Only
        # surfaced once a tick has been measured (last_* > 0). The analyst greps
        # comm/bytes_compressed + comm/bytes_dense_equiv (and the ratio) from the
        # metrics jsonl / train.log; the names use the `comm/` namespace.
        ec = float(getattr(self.powersgd, "last_elems_compressed", 0.0))
        ed = float(getattr(self.powersgd, "last_elems_dense_equiv", 0.0))
        if ed > 0.0:
            out["comm/bytes_compressed"] = ec
            out["comm/bytes_dense_equiv"] = ed
            out["comm/bytes_ratio"] = ec / ed
        return out

    def metrics(self) -> dict:
        """Return the comm_eff operation counters for logging.

        All values are numeric; a string here crashes np.mean. The anchor
        counters are emitted unconditionally even on a 0-fire step; the
        contamination / guard counters (anchor_mask_applications, anchor_grad_corrected,
        anchor_rollouts_generated, anchor_rewards_recomputed,
        anchor_optimizer_steps) are the load-bearing falsifiers.
        """
        out = {
            "comm_eff/mask_applications": self.mask_applications,
            "comm_eff/anchor_backwards": self.anchor_backwards,
            "comm_eff/spectral_corrections": self.spectral_corrections,
            "comm_eff/anchor_mask_applications": self.anchor_mask_applications,
            "comm_eff/anchor_grad_corrected": self.anchor_grad_corrected,
            "comm_eff/anchor_rollouts_generated": self.anchor_rollouts_generated,
            "comm_eff/anchor_rewards_recomputed": self.anchor_rewards_recomputed,
            "comm_eff/anchor_optimizer_steps": self.anchor_optimizer_steps,
            "comm_eff/anchor_batch_fraction": self.anchor_batch_fraction,
            # Cumulative count of clean (unmasked) optimizer steps fired.
            # Monotonic; increments at exactly steps clean_cadence, 2*clean_cadence,
            # ... so logs can confirm that the clean cadence fired correctly.
            "comm_eff/clean_steps": self.clean_steps,
            # Monotonic per-optimizer-step counter the spectral cadence is
            # keyed on. Numeric (the reduce_metrics-must-stay-numeric contract).
            # Lets logs confirm spectral_corrections increments only on the
            # cadence steps (spectral_step % spectral.cadence == 0), not every step.
            "comm_eff/spectral_step": self.spectral_step,
            # Anchor-owns-Q counters plus merger cold-M fallbacks.
            "comm_eff/anchor_q_updates": self.anchor_q_updates,
            "comm_eff/anchor_q_broadcasts": self.anchor_q_broadcasts,
            "comm_eff/merger_coldM_fallbacks": self.merger_coldM_fallbacks,
            # ef_powersgd shape-aware residual resets this step.
            "comm_eff/residual_reset_on_shape_mismatch": self.residual_reset_on_shape_mismatch,
            # Cumulative passive family-screen builds.
            "comm_eff/family_screen_builds": self.family_screen_builds,
            # Cumulative paired-replay anchor fires (0 unless
            # anchor.replay_paired_batch=true).
            "comm_eff/anchor_replay_fires": self.anchor_replay_fires,
            # Look-ahead (weight-projection) anchor fires vs warmup fallbacks
            # (both 0 unless anchor.lookahead_anchor is active).
            "comm_eff/lookahead_fires": self.lookahead_fires,
            "comm_eff/lookahead_warmup_fallbacks": self.lookahead_warmup_fallbacks,
            # E2 warmup skips (0 unless warmup_mode=no_correct).
            "comm_eff/warmup_no_correct_skips": self.warmup_no_correct_skips,
            # Cumulative geometry-probe fires with a complete
            # m1–m7 record written (0 unless probe.geometry_enabled).
            "comm_eff/geometry_probe_fires": self.geometry_probe_fires,
        }
        if self.rank1_relex_active():
            out.update(
                {
                    "comm_eff/rank1_m_ready": int(self.rank1_m_ready),
                    "comm_eff/rank1_q_only_fires": self.rank1_q_only_fires,
                    "comm_eff/rank1_correction_bypass_ticks": self.rank1_correction_bypass_ticks,
                    "comm_eff/rank1_fires": self.rank1_fires,
                    "comm_eff/rank1_history_checkpoints": self.rank1_history_checkpoints,
                    "comm_eff/rank1_history_deltas": self.rank1_history_deltas,
                    "comm_eff/rank1_window_span": self.rank1_window_span,
                    "comm_eff/rank1_prediction_horizon": self.rank1_prediction_horizon,
                    "comm_eff/rank1_evr_mean": self.rank1_evr_mean,
                    "comm_eff/rank1_r2_mean": self.rank1_r2_mean,
                    "comm_eff/rank1_zero_motion_tensors": self.rank1_zero_motion_tensors,
                }
            )
            probe_cfg = getattr(self.config, "probe", None)
            if bool(getattr(probe_cfg, "rank1_projection_enabled", False)):
                out.update(
                    {
                        "comm_eff/rank1_probe_predictions": self.rank1_probe_predictions,
                        "comm_eff/rank1_probe_resolutions": self.rank1_probe_resolutions,
                        "comm_eff/rank1_probe_pending": self.rank1_probe_pending,
                        "comm_eff/rank1_probe_projected_rmse": self.rank1_probe_projected_rmse,
                        "comm_eff/rank1_probe_stale_rmse": self.rank1_probe_stale_rmse,
                        "comm_eff/rank1_probe_skill": self.rank1_probe_skill,
                        "comm_eff/rank1_probe_direction_cos": self.rank1_probe_direction_cos,
                    }
                )
        return out


def maybe_build_comm_eff_state(config: Any) -> Optional[CommEffState]:
    """Construct a ``CommEffState`` iff comm_eff is enabled, else return ``None``.

    This is the single gate that guarantees the disabled path is inert: when
    ``config.enabled`` is false (or ``config`` is ``None``/absent) it returns
    ``None`` **without drawing RNG, allocating buffers or registering hooks**.
    Callers store the result and guard every comm_eff op behind a ``None`` /
    ``state.enabled`` check.
    """
    if not _is_enabled(config):
        return None
    state = CommEffState(config)
    logger.info("comm_eff: enabled — constructed CommEffState")
    return state


def comm_eff_metrics(state: Optional[CommEffState]) -> dict:
    """Return comm_eff counters for ``state``, or an empty dict when disabled.

    Centralises the "disabled means no counters" convention so call sites do
    not each re-derive it. Includes the measured ``comm_eff/mask_ratio`` when a
    mask fired this step.
    """
    if state is None:
        return {}
    out = state.metrics()
    out.update(state.path_metrics())
    out.update(state.mask_ratio_metrics())
    out.update(state.spectral_metrics())
    out.update(state.powersgd_metrics())
    return out
