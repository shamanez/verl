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
  allocated (lazily, by ``build()``, which later M2 work fills in).

Because construction is gated, a dense GRPO run with this scaffolding merged
consumes the exact same RNG sequence and issues the exact same collective ops
as one without it — the criterion-7 rel-tol-1e-4 parity check holds.

The instrumented counters (``mask_applications``, ``anchor_backwards``,
``spectral_corrections``) live on the state object. When disabled there is no
state object, so the counters are *absent* rather than zero — which the
analyst treats as equivalent to ``== 0`` (no comm_eff op fired). When enabled
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
    "PATH_TAGS",
    "TRAIN_TAG",
    "OLD_LOGPROB_TAG",
    "MASK_ELIGIBLE_TAGS",
    "mask_eligible_tags",
]

# The exhaustive set of execution-path tags a comm_eff state can carry. The
# activation mask is allowed to fire on EXACTLY ONE of these (``train``); every
# other tag is an RL-measurement / serving path that must stay byte-identical to
# dense GRPO even while masking is enabled. EXP-6 makes contamination of those
# paths a *loud* failure (an assert in the mask hook) rather than a counter that
# someone has to remember to grep.
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
# contamination accounting; EXP-9 makes it the ONLY other path the mask is
# permitted to fire on (and only when comm_eff.mask.mask_recompute=true).
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
# default (EXP-5 → EXP-12 contract): only ``train``. EXP-9 widens this to
# ``{train, old_logprob}`` *iff* ``state.mask.mask_recompute=True``; the widen
# is computed at hook-fire time by ``mask_eligible_tags(state)`` so flipping
# the YAML knob does not require restarting the worker. ``None`` (anchor pass,
# GUARD 5) is never eligible — anchors stay unmasked unconditionally.
MASK_ELIGIBLE_TAGS: frozenset = frozenset({TRAIN_TAG})


def mask_eligible_tags(state: Any) -> frozenset:
    """Return the set of path tags the activation mask is allowed to fire on
    for the given ``state``. Pure read — no side effects, no allocation.

    The default eligibility (``{TRAIN_TAG}``) is widened to
    ``{TRAIN_TAG, OLD_LOGPROB_TAG}`` *only* when both
    ``state.mask.enabled`` and ``state.mask.mask_recompute`` are truthy (EXP-9).
    Anything else (disabled state, missing mask sub-config, ``mask_recompute``
    unset / falsy) returns the singleton default, preserving the EXP-5 ⇒ EXP-12
    behavior bit-for-bit.

    ``None`` (anchor pass / GUARD 5) is intentionally NOT in either set: the
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


class CommEffState:
    """Per-worker communication-efficient compression state.

    Constructed **only** when ``comm_eff.enabled=true``. Holds the operation
    counters and (once ``build()`` is implemented by later M2 work) the mask
    RNG generator, anchor EMA buffers and spectral workspace. The disabled path
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

        # Operation counters surfaced into training metrics under comm_eff/*.
        self.mask_applications = 0
        self.anchor_backwards = 0
        self.spectral_corrections = 0

        # EXP-8 anchor-circuit counters. These are the load-bearing falsifiers
        # the analyst greps by NAME (see plan ## Success criteria):
        #   anchor_mask_applications  — mask hooks fired DURING the anchor pass.
        #                               MUST stay 0 (GUARD 5: the anchor runs
        #                               unmasked even though it's on the train
        #                               path). Captured as a delta around the
        #                               anchor fwd/bwd, not the global counter.
        #   anchor_grad_corrected     — anchor gradients fed THROUGH correct_matrix.
        #                               MUST stay 0 (GUARD 6: G_anchor read raw
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

        # EXP-16 monotonic optimizer-step counter the SPECTRAL cadence is keyed
        # on. Advanced once per actor train_batch by the grad-correction hook
        # (_maybe_comm_eff_grad_correction), in lockstep with anchor_step (both
        # +1 per train_batch), so when anchor.cadence == spectral.cadence the
        # spectral correction fires on EXACTLY the steps the anchor EMA was just
        # refreshed (a fresh basis, never a stale one). Kept independent of
        # anchor_step so spectral cadence still works when the anchor circuit is
        # disabled. Stays 0 on the dense/disabled path (the hook short-circuits
        # on the None/enabled guard before advancing it).
        self.spectral_step = 0

        # EXP-14 periodic clean-step counter. Incremented once per trainer step
        # whose (global_step % clean_cadence) == 0 while clean_cadence > 0 — i.e.
        # every step on which masking is forced OFF and AdamW takes a step on the
        # true dense gradient. Surfaced as comm_eff/clean_steps so the analyst can
        # prove the clean step fired at exactly steps clean_cadence, 2*clean_cadence,
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

        # Whether masking is currently active. Set True only on entry to the
        # actor-train forward/backward (around update_actor) and cleared on
        # exit, so log-prob / ref / infer / val / checkpoint forwards stay clean.
        self.mask_active = False

        # Explicit execution-path tag (EXP-6 contamination guard). Defaults to
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
        # metrics as ``comm_eff/mask_applications/<tag>`` so the analyst can
        # confirm confinement by KEY PREFIX (no substring false positives).
        self.mask_applications_by_path = {tag: 0 for tag in PATH_TAGS}

        # The spectral filter (third circuit). Constructed in build() when
        # ``comm_eff.spectral.enabled`` is true; None otherwise. Holds the
        # (seeded) anchor-EMA cache and applies the paper formula at the
        # grad-correction hook point. See verl.workers.comm_eff.spectral_filter.
        self.spectral = None

        # FSDP gradient-representation discovery log (EXP-7 headline deliverable).
        # The engine's grad-correction hook fills this once, on the first
        # correction, with type(p.grad), the grad container shape, the logical
        # 2D matrix shape, the FSDP wrapping/version, and whether correction ran
        # before/after FSDP gradient reduction and gradient clipping, for >=1
        # target matrix. Surfaced into metrics so the analyst greps it.
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
        circuit (EXP-8) allocation is still deferred.
        """
        if self._built:
            return
        mask_cfg = getattr(self.config, "mask", None)
        mask_enabled = bool(getattr(mask_cfg, "enabled", False)) if mask_cfg is not None else False
        if mask_enabled and float(getattr(mask_cfg, "p", 0.0)) > 0.0:
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
                # EXP-8 storage layer (defaults faithful: gpu/full/cache).
                ema_device=str(getattr(spec_cfg, "ema_device", "gpu")),
                svd_mode=str(getattr(spec_cfg, "svd_mode", "full")),
                basis_cache=str(getattr(spec_cfg, "basis_cache", "cache")),
                rank=int(getattr(spec_cfg, "rank", 8)),
                correction_mode=str(getattr(spec_cfg, "correction_mode", "reweight")),
                inject_gamma=float(getattr(spec_cfg, "inject_gamma", 1.0)),
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
            # EXP-12 discovery line (string-valued ⇒ stdout only, NEVER metrics:
            # reduce_metrics does np.mean on every metric value and crashes on a
            # string — the EXP-7 lesson). The analyst greps this for cell 3's
            # "ema_device=cpu + svd_lowrank" confirmation, AND for the EXP-12
            # anchor-backward isolation-mode confirmation (criterion 13).
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
        """EXP-14: True iff the given trainer ``global_step`` is a clean step.

        A clean step is one on which masking is forced OFF for the whole step
        and AdamW refreshes its moments on the true dense gradient. The rule is
        ``clean_cadence > 0 and (global_step % clean_cadence) == 0``. When
        ``global_step`` is ``None`` the most-recently-threaded ``self.global_step``
        is used. Pure read — no side effects, no allocation.

        ``clean_cadence`` is read from the config (default 0 ⇒ always False, so
        every pre-EXP-14 config and the disabled path keep their exact behavior).
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
        """EXP-16: True iff the spectral grad-correction fires on this opt step.

        Mirrors :meth:`is_clean_step` (and ``anchor_should_fire``): a pure
        predicate keyed on the monotonic per-optimizer-step counter
        ``self.spectral_step`` (1-based — the grad-correction hook advances it
        before calling this). The rule is ``(step % cadence) == 0`` with
        ``cadence = comm_eff.spectral.cadence`` (default ``1`` ⇒ always True ⇒
        fire every step ⇒ the pre-EXP-16 behavior, so every prior config and the
        disabled path keep their exact behavior).

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
        the per-path counter still records the leak so the analyst catches it.
        """
        self.mask_applications += 1
        tag = self.path_tag if self.path_tag in self.mask_applications_by_path else TRAIN_TAG
        self.mask_applications_by_path[tag] += 1

    def path_metrics(self) -> dict:
        """Per-path mask-application counters, surfaced under a stable KEY prefix.

        Emits ``comm_eff/mask_applications/<tag>`` for every tag. The analyst
        asserts the only nonzero key is ``.../train``; any other nonzero key is
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
        and crashes the trainer's metric-reduction at the end of the step (this
        is exactly what killed the second EXP-7 spectral_on run before it
        reached global_step=2). So the FSDP gradient-representation DISCOVERY log
        (string-valued container type / placements / fsdp_version / correction
        point) is deliberately NOT emitted here. It is the headline deliverable
        and is surfaced the analyst-greppable way it was designed for: the
        ``[comm_eff][EXP-7][FSDP-DISCOVERY] {...}`` stdout line and the
        ``logger.warning`` record, both written by the engine hook. It also
        stays available in-process on ``state.fsdp_grad_repr`` for any non-metric
        consumer. Reducible metrics must stay numeric.
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

    def metrics(self) -> dict:
        """Return the comm_eff operation counters for logging.

        All values are NUMERIC (the EXP-7 reduce_metrics lesson: a string here
        crashes np.mean). The anchor counters are emitted unconditionally so the
        analyst can grep them by name even on a 0-fire step; the contamination /
        guard counters (anchor_mask_applications, anchor_grad_corrected,
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
            # EXP-14: cumulative count of clean (unmasked) optimizer steps fired.
            # Monotonic; increments at exactly steps clean_cadence, 2*clean_cadence,
            # ... so the analyst can grep that the clean cadence fired correctly.
            "comm_eff/clean_steps": self.clean_steps,
            # EXP-16: monotonic per-optimizer-step counter the spectral cadence is
            # keyed on. Numeric (the reduce_metrics-must-stay-numeric contract).
            # Lets the analyst confirm spectral_corrections increments only on the
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
    return out
