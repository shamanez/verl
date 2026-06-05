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
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:  # avoid an import cycle at runtime; only needed for type hints
    from verl.workers.config.comm_eff import CommEffConfig

logger = logging.getLogger(__name__)

__all__ = [
    "CommEffState",
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
# other tag is an RL-measurement / serving path that must stay byte-identical to
# dense GRPO even while masking is enabled. Contamination is a loud failure in
# the mask hook.
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
    * ``dense`` (the field default) falls back to the LEGACY selector — if the
      mask sub-config is enabled with ``p > 0`` the codec is ``prf_mask``;
      otherwise ``dense``.

    This keeps legacy mask configs working while PowerSGD is selected explicitly.
    """
    ctype = getattr(config, "compression_type", "dense") if config is not None else "dense"
    if ctype in ("prf_mask", "powersgd"):
        return ctype
    # ctype == "dense": honor the legacy mask selector for back-compat.
    mask_cfg = getattr(config, "mask", None)
    mask_enabled = bool(getattr(mask_cfg, "enabled", False)) if mask_cfg is not None else False
    if mask_enabled and float(getattr(mask_cfg, "p", 0.0)) > 0.0:
        return "prf_mask"
    return "dense"


class CommEffState:
    """Per-worker communication-efficient compression state.

    Constructed **only** when ``comm_eff.enabled=true``. Holds the operation
    counters, mask RNG generator, anchor EMA buffers and spectral workspace.
    The disabled path
    never instantiates this class — see ``maybe_build_comm_eff_state``.
    """

    def __init__(self, config: "CommEffConfig"):
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
        # than silently corrupting the RL-measurement machinery. ``mask_active``
        # remains the fast gate; ``path_tag`` is the loud cross-check.
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
        # correction. Logged faithfully (never clamped) per the codex pin: not
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
                f"[comm_eff][EXP-20] powersgd codec: rank={self.powersgd.rank} "
                f"update_cadence={self.powersgd.update_cadence} warm_start={self.powersgd.warm_start} "
                f"compress_recompute={self.powersgd.compress_recompute} "
                f"sync_basis={self.powersgd.sync_basis} "
                f"qr_dtype={getattr(ps_cfg, 'qr_dtype', 'fp32')}",
                flush=True,
            )

        spec_cfg = getattr(self.config, "spectral", None)
        spec_enabled = bool(getattr(spec_cfg, "enabled", False)) if spec_cfg is not None else False
        if spec_enabled:
            # Imported lazily so the disabled path never pays the import cost.
            from verl.workers.comm_eff.spectral_filter import SpectralFilter

            self.spectral = SpectralFilter(
                alpha=float(getattr(spec_cfg, "alpha", 0.3)),
                tau=float(getattr(spec_cfg, "tau", 1e-3)),
                beta_anc=float(getattr(spec_cfg, "beta_anc", 0.95)),
                seed_anchor_cache=bool(getattr(spec_cfg, "seed_anchor_cache", True)),
                anchor_seed=int(getattr(spec_cfg, "anchor_seed", 0)),
                # Storage layer defaults: gpu/full/cache.
                ema_device=str(getattr(spec_cfg, "ema_device", "gpu")),
                svd_mode=str(getattr(spec_cfg, "svd_mode", "full")),
                basis_cache=str(getattr(spec_cfg, "basis_cache", "cache")),
                rank=int(getattr(spec_cfg, "rank", 8)),
                correction_mode=str(getattr(spec_cfg, "correction_mode", "reweight")),
                inject_gamma=float(getattr(spec_cfg, "inject_gamma", 1.0)),
                blend_eta=float(getattr(spec_cfg, "blend_eta", 0.5)),
            )
            logger.info(
                "comm_eff: spectral filter built (alpha=%s tau=%s beta_anc=%s seed_anchor_cache=%s "
                "ema_device=%s svd_mode=%s basis_cache=%s rank=%s)",
                self.spectral.alpha,
                self.spectral.tau,
                self.spectral.beta_anc,
                self.spectral.seed_anchor_cache,
                self.spectral.ema_device,
                self.spectral.svd_mode,
                self.spectral.basis_cache,
                self.spectral.rank,
            )
            # Discovery line is string-valued, so it goes to stdout only:
            # reduce_metrics does np.mean on every metric value and crashes on a
            # string. Keep it out of metrics.
            anc_cfg = getattr(self.config, "anchor", None)
            anchor_enabled = bool(getattr(anc_cfg, "enabled", False)) if anc_cfg is not None else False
            isolation_mode = "clone" if anchor_enabled else "n/a (anchor.enabled=false)"
            print(
                f"[comm_eff][EXP-12] spectral storage: ema_device={self.spectral.ema_device} "
                f"svd_mode={self.spectral.svd_mode} basis_cache={self.spectral.basis_cache} "
                f"rank={self.spectral.rank} seed_anchor_cache={self.spectral.seed_anchor_cache} "
                f"anchor_backward_isolation_mode={isolation_mode}",
                flush=True,
            )
        self._built = True

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
        legacy aggregate counter (``mask_applications``) and the per-path
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

    def path_metrics(self) -> dict:
        """Per-path mask-application counters, surfaced under a stable KEY prefix.

        Emits ``comm_eff/mask_applications/<tag>`` for every tag. The only
        nonzero key should be ``.../train``; any other nonzero key is
        the contamination falsifier. Emitting all keys (including the zeros)
        makes the confinement machine-checkable without substring grepping.
        """
        return {
            f"comm_eff/mask_applications/{tag}": count
            for tag, count in self.mask_applications_by_path.items()
        }

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
        return out

    def metrics(self) -> dict:
        """Return the comm_eff operation counters for logging.

        All values are numeric; a string here crashes np.mean. The anchor
        counters are emitted unconditionally even on a 0-fire step; the
        contamination / guard counters (anchor_mask_applications, anchor_grad_corrected,
        anchor_rollouts_generated, anchor_rewards_recomputed,
        anchor_optimizer_steps) are the load-bearing falsifiers.
        """
        return {
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
        }


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
