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

"""Per-worker state for the communication-efficient GRPO pipeline.

The factory returns ``None`` while the master switch is off. This preserves the
dense path's RNG, allocation, hook, and collective behavior. An enabled state
owns only PowerSGD, the delayed dense anchor, signed EMA, and rank-1 RELEX.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
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
    "REF_LOGPROB_TAG",
    "MASK_ELIGIBLE_TAGS",
    "mask_eligible_tags",
]

TRAIN_TAG = "train"
OLD_LOGPROB_TAG = "old_logprob"
REF_LOGPROB_TAG = "ref_logprob"
PATH_TAGS = (TRAIN_TAG, OLD_LOGPROB_TAG, REF_LOGPROB_TAG, "ckpt")

# The set of execution-path tags the activation mask (prf_mask codec) is allowed
# to fire on by default: only ``train``. ``mask_eligible_tags(state)`` widens
# this to include ``old_logprob`` when ``state.config.mask.mask_recompute`` is
# truthy and ``ref_logprob`` when ``state.config.mask.mask_reference`` is truthy.
# ``None`` (the anchor pass) is never eligible, so anchors stay unmasked
# unconditionally.
MASK_ELIGIBLE_TAGS: frozenset = frozenset({TRAIN_TAG})


def mask_eligible_tags(state: Any) -> frozenset:
    """Return the path tags the activation mask / sr_quant codec may fire on.

    Pure read (no side effects, no allocation). The default eligibility
    (``{TRAIN_TAG}``) is widened, only when ``state.config.mask.enabled`` is
    truthy OR the codec is ``sr_quant`` (which reuses the mask eligibility
    knobs without requiring ``mask.enabled``), by:

    * ``OLD_LOGPROB_TAG`` when ``state.config.mask.mask_recompute`` is truthy;
    * ``REF_LOGPROB_TAG`` when ``state.config.mask.mask_reference`` is truthy.

    Both widenings are independent and additive. Anything else (disabled state,
    missing mask sub-config, both flags unset / falsy) returns the singleton
    default. ``None`` (the anchor pass) is intentionally in none of the sets:
    the anchor circuit runs unmasked regardless of these flags.
    """
    if state is None:
        return MASK_ELIGIBLE_TAGS
    config = getattr(state, "config", None)
    mask_cfg = getattr(config, "mask", None)
    if mask_cfg is None:
        return MASK_ELIGIBLE_TAGS
    sr_quant_codec = str(getattr(config, "compression_type", "")) == "sr_quant"
    if not (bool(getattr(mask_cfg, "enabled", False)) or sr_quant_codec):
        return MASK_ELIGIBLE_TAGS
    tags = {TRAIN_TAG}
    if bool(getattr(mask_cfg, "mask_recompute", False)):
        tags.add(OLD_LOGPROB_TAG)
    if bool(getattr(mask_cfg, "mask_reference", False)):
        tags.add(REF_LOGPROB_TAG)
    if tags == {TRAIN_TAG}:
        return MASK_ELIGIBLE_TAGS
    return frozenset(tags)


def _is_enabled(config: Any) -> bool:
    if config is None:
        return False
    if isinstance(config, dict):
        return bool(config.get("enabled", False))
    return bool(getattr(config, "enabled", False))


def resolve_compression_type(config: Any) -> str:
    """Resolve the effective boundary codec: ``dense``, ``prf_mask``, ``powersgd`` or ``sr_quant``.

    Pure read (no side effects, no allocation). The resolution is
    back-compatible:

    * an explicit ``compression_type`` of ``prf_mask``, ``powersgd`` or
      ``sr_quant`` wins;
    * ``dense`` (the fall-through) honors the mask selector: a mask sub-config
      enabled with ``p > 0`` resolves to ``prf_mask``, otherwise ``dense``.

    Real ``CommEffConfig`` objects always carry an explicit ``compression_type``
    (defaulting to ``powersgd``), so this only reaches the mask-selector branch
    for lightweight configs that omit the field.
    """

    if config is None:
        return "dense"
    ctype = str(getattr(config, "compression_type", "dense"))
    if ctype in ("prf_mask", "powersgd", "sr_quant"):
        return ctype
    # ctype == "dense": honor the mask selector for back-compat.
    mask_cfg = getattr(config, "mask", None)
    mask_enabled = bool(getattr(mask_cfg, "enabled", False)) if mask_cfg is not None else False
    if mask_enabled and float(getattr(mask_cfg, "p", 0.0)) > 0.0:
        return "prf_mask"
    return "dense"


class CommEffState:
    """Lazily built runtime state for an enabled communication-efficient actor."""

    def __init__(self, config: CommEffConfig):
        assert _is_enabled(config), "construct CommEffState through maybe_build_comm_eff_state()"
        self.config = config
        self.enabled = True
        self._built = False
        self.compression_type = "dense"

        # Execution clocks and hook confinement.
        self.global_step = -1
        self.anchor_step = 0
        self.spectral_step = 0
        self.compression_active = False
        self.path_tag: Optional[str] = None

        # Runtime circuits.
        # The activation masker (prf_mask codec). Constructed in build() only
        # when compression_type resolves to prf_mask; None otherwise (the
        # disabled path, the dense codec, and the powersgd codec never touch it).
        # Mutually exclusive with self.powersgd: a run is either the mask codec
        # or the powersgd codec, never both.
        self.masker = None
        # The activation quantizer (sr_quant codec). Constructed in build() only
        # when compression_type resolves to sr_quant; mutually exclusive with
        # both the masker and the powersgd compressor (exactly one boundary
        # codec object exists per run).
        self.quantizer = None
        self.powersgd = None
        self.spectral = None
        self.fsdp_grad_repr: dict = {}
        self.spectral_rel_change: dict = {}

        # Activation-mask counters. Cumulative. mask_applications is the
        # aggregate hook-fire count; mask_applications_by_path breaks it down per
        # execution-path tag (only .../train should be nonzero; any other
        # nonzero key is a confinement leak).
        self.mask_applications = 0
        self.mask_applications_by_path = {tag: 0 for tag in PATH_TAGS}

        # PowerSGD and Q-transaction counters.
        self.powersgd_applications = 0
        self.powersgd_basis_updates = 0
        self.anchor_q_updates = 0
        self.anchor_q_broadcasts = 0
        self.anchor_q_activations = 0
        self.anchor_q_stage_overwrites = 0
        object.__setattr__(self, "_powersgd_q_agreement_checked", False)
        object.__setattr__(self, "_powersgd_q_agreement_dev", None)

        # Dense-anchor and signed-EMA counters.
        self.anchor_backwards = 0
        self.spectral_corrections = 0
        self.merger_coldM_fallbacks = 0
        self.anchor_grad_corrected = 0
        self.anchor_rollouts_generated = 0
        self.anchor_rewards_recomputed = 0
        self.anchor_optimizer_steps = 0
        self.anchor_replay_fires = 0
        self.anchor_batch_fraction = 0.0
        self.anchor_batch_sequences_global = 0
        self.anchor_update_sequences_global = 0
        self.anchor_batch_prompt_equivalents_global = 0
        self.anchor_update_prompt_equivalents_global = 0
        self.anchor_rollout_n = 0
        # getattr with the real-config default keeps this byte-identical for a
        # full CommEffConfig (anchor.batch_scope defaults to "ppo_minibatch")
        # while tolerating the lightweight prf_mask configs that omit `anchor`.
        _anchor_cfg = getattr(config, "anchor", None)
        self.anchor_batch_scope_rollout = int(getattr(_anchor_cfg, "batch_scope", "ppo_minibatch") == "rollout_batch")

        # Rank-1 RELEX state and scientific counters.
        self.rank1_m_ready = False
        self.rank1_q_only_fires = 0
        self.rank1_warmup_correction_fires = 0
        self.rank1_correction_bypass_ticks = 0
        self.rank1_fires = 0
        self.rank1_history_checkpoints = 0
        self.rank1_history_deltas = 0
        self.rank1_window_span = 0
        self.rank1_prediction_horizon = 0
        self.rank1_evr_mean = 0.0
        self.rank1_r2_mean = 0.0
        self.rank1_zero_motion_tensors = 0

    def build(self, module: Any) -> None:
        """Build PowerSGD and signed EMA once; anchor snapshots remain lazy."""

        del module
        if self._built:
            return
        self.compression_type = resolve_compression_type(self.config)

        if self.compression_type == "prf_mask":
            # Imported lazily so the disabled / dense / powersgd paths never pay
            # the import cost. Mutually exclusive with the powersgd branch below.
            from verl.workers.comm_eff.activation_mask import ActivationMasker

            mask_cfg = getattr(self.config, "mask", None)
            # Anchor ownership of the FRLR basis Q (issue #93). Only FRLR carries
            # a basis, so a plain-PRF arm always resolves to False here; the
            # config validator rejects the plain-mask + owns_q combination.
            mask_anchor_cfg = getattr(self.config, "anchor", None)
            mask_anchor_owns_q = bool(getattr(mask_anchor_cfg, "owns_q", False)) and bool(
                getattr(mask_cfg, "frlr", False)
            )
            self.masker = ActivationMasker(
                p=float(getattr(mask_cfg, "p", 0.0)),
                base_seed=int(getattr(mask_cfg, "seed", 0)),
                pp_size=int(getattr(mask_cfg, "pp_size", 8)),
                rescale=bool(getattr(mask_cfg, "rescale", False)),
                rescale_mode=str(getattr(mask_cfg, "rescale_mode", "auto")),
                exact_k=bool(getattr(mask_cfg, "exact_k", False)),
                antithetic=bool(getattr(mask_cfg, "antithetic", False)),
                p_by_boundary=list(getattr(mask_cfg, "p_by_boundary", []) or []),
                dense_every=int(getattr(mask_cfg, "dense_every", 0) or 0),
                frlr=bool(getattr(mask_cfg, "frlr", False)),
                frlr_rank=int(getattr(mask_cfg, "frlr_rank", 32)),
                frlr_k=int(getattr(mask_cfg, "frlr_k", 44)),
                frlr_unbiased=bool(getattr(mask_cfg, "frlr_unbiased", False)),
                frlr_q_cadence=int(getattr(mask_cfg, "frlr_q_cadence", 1)),
                anchor_owns_q=mask_anchor_owns_q,
                state=self,
            )
            logger.info(
                "comm_eff: prf_mask p=%s pp_size=%s rescale=%s rescale_mode=%s "
                "exact_k=%s antithetic=%s p_by_boundary=%s "
                "frlr=%s frlr_rank=%s frlr_k=%s frlr_unbiased=%s frlr_q_cadence=%s "
                "anchor_owns_q=%s dense_every=%s",
                getattr(mask_cfg, "p", 0.0),
                getattr(mask_cfg, "pp_size", 8),
                getattr(mask_cfg, "rescale", False),
                getattr(mask_cfg, "rescale_mode", "auto"),
                getattr(mask_cfg, "exact_k", False),
                getattr(mask_cfg, "antithetic", False),
                list(getattr(mask_cfg, "p_by_boundary", []) or []),
                getattr(mask_cfg, "frlr", False),
                getattr(mask_cfg, "frlr_rank", 32),
                getattr(mask_cfg, "frlr_k", 44),
                getattr(mask_cfg, "frlr_unbiased", False),
                getattr(mask_cfg, "frlr_q_cadence", 1),
                mask_anchor_owns_q,
                int(getattr(mask_cfg, "dense_every", 0) or 0),
            )

        if self.compression_type == "sr_quant":
            # Imported lazily so the disabled / dense / powersgd / prf_mask
            # paths never pay the import cost. Mutually exclusive with the
            # prf_mask branch above and the powersgd branch below. sr_quant
            # reuses the mask sub-config for eligibility (mask_recompute /
            # mask_reference) and keying (seed / pp_size); mask.p / rescale /
            # exact_k / antithetic / frlr are ignored by this codec.
            from verl.workers.comm_eff.activation_quant import ActivationQuantizer

            mask_cfg = getattr(self.config, "mask", None)
            quant_cfg = getattr(self.config, "quant", None)
            self.quantizer = ActivationQuantizer(
                bits=int(getattr(quant_cfg, "bits", 1)),
                base_seed=int(getattr(mask_cfg, "seed", 0)),
                pp_size=int(getattr(mask_cfg, "pp_size", 8)),
                block_size=int(getattr(quant_cfg, "block_size", 32)),
                rounding=str(getattr(quant_cfg, "rounding", "sr")),
                subset_k=int(getattr(quant_cfg, "subset_k", 0)),
                state=self,
            )
            logger.info(
                "comm_eff: sr_quant bits=%s block_size=%s rounding=%s subset_k=%s pp_size=%s seed=%s "
                "mask_recompute=%s mask_reference=%s",
                getattr(quant_cfg, "bits", 1),
                getattr(quant_cfg, "block_size", 32),
                getattr(quant_cfg, "rounding", "sr"),
                getattr(quant_cfg, "subset_k", 0),
                getattr(mask_cfg, "pp_size", 8),
                getattr(mask_cfg, "seed", 0),
                getattr(mask_cfg, "mask_recompute", False),
                getattr(mask_cfg, "mask_reference", False),
            )

        if self.compression_type == "powersgd":
            from verl.workers.comm_eff.powersgd_activation import PowerSGDActivationCompressor

            cfg = self.config.powersgd
            anchor_cfg = self.config.anchor
            self.powersgd = PowerSGDActivationCompressor(
                rank=int(cfg.rank),
                base_seed=int(cfg.seed),
                pp_size=int(cfg.pp_size),
                update_cadence=int(cfg.update_cadence),
                warm_start=bool(cfg.warm_start),
                compress_recompute=bool(cfg.compress_recompute),
                sync_basis=bool(cfg.sync_basis),
                qr_dtype=str(cfg.qr_dtype),
                reortho_eps=float(cfg.reortho_eps),
                anchor_owns_q=bool(anchor_cfg.owns_q),
                anchor_cadence=int(anchor_cfg.cadence),
                fast_q_bootstrap=bool(cfg.fast_q_bootstrap),
                state=self,
            )
            logger.info(
                "comm_eff: PowerSGD rank=%s sync_basis=%s anchor_owns_q=%s fast_q_bootstrap=%s",
                cfg.rank,
                cfg.sync_basis,
                anchor_cfg.owns_q,
                cfg.fast_q_bootstrap,
            )

        _spectral_cfg = getattr(self.config, "spectral", None)
        if _spectral_cfg is not None and getattr(_spectral_cfg, "enabled", False):
            from verl.workers.comm_eff.spectral_filter import SpectralFilter

            cfg = _spectral_cfg
            self.spectral = SpectralFilter(
                beta_anc=float(cfg.beta_anc),
                ema_device=str(cfg.ema_device),
                signed_ema_alpha=float(cfg.signed_ema_alpha),
                diagnostics=bool(cfg.diagnostics),
            )
            logger.info(
                "comm_eff: signed EMA beta_anc=%s alpha=%s ema_device=%s target_scope=%s",
                cfg.beta_anc,
                cfg.signed_ema_alpha,
                cfg.ema_device,
                cfg.target_scope,
            )

        self._built = True

    def rank1_relex_active(self) -> bool:
        # getattr defaults keep this byte-identical for a full CommEffConfig
        # while returning False for the lightweight prf_mask configs that omit
        # `anchor` (no rank1_relex without an anchor sub-config).
        anchor = getattr(self.config, "anchor", None)
        return (
            bool(getattr(anchor, "lookahead_anchor", False))
            and getattr(anchor, "lookahead_mode", None) == "rank1_relex"
        )

    def reset_anchor_q_runtime(self) -> None:
        """Reset non-checkpointed Q transaction state after a weight load."""

        self.anchor_q_updates = 0
        self.anchor_q_broadcasts = 0
        self.anchor_q_activations = 0
        self.anchor_q_stage_overwrites = 0
        object.__setattr__(self, "_powersgd_q_agreement_checked", False)
        object.__setattr__(self, "_powersgd_q_agreement_dev", None)
        if self.powersgd is not None:
            if hasattr(self.powersgd, "reset_basis_runtime"):
                self.powersgd.reset_basis_runtime()
            else:
                getattr(self.powersgd, "_basis", {}).clear()
                getattr(self.powersgd, "_pending_anchor_basis", {}).clear()
                getattr(self.powersgd, "_sketch", {}).clear()

    def reset_rank1_runtime(self) -> None:
        """Reset local RELEX history after model weights are loaded."""

        if not self.rank1_relex_active():
            return
        for name in (
            "_rank1_history",
            "_rank1_projector",
            "_rank1_base_batch",
            "_rank1_base_canary",
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
        self.anchor_replay_fires = 0
        self.anchor_batch_fraction = 0.0
        self.anchor_batch_sequences_global = 0
        self.anchor_update_sequences_global = 0
        self.anchor_batch_prompt_equivalents_global = 0
        self.anchor_update_prompt_equivalents_global = 0
        self.anchor_rollout_n = 0
        self.anchor_batch_scope_rollout = int(self.config.anchor.batch_scope == "rollout_batch")
        self.reset_anchor_q_runtime()
        self.powersgd_basis_updates = 0
        self.rank1_m_ready = False
        self.rank1_q_only_fires = 0
        self.rank1_warmup_correction_fires = 0
        self.rank1_correction_bypass_ticks = 0
        self.rank1_fires = 0
        self.rank1_history_checkpoints = 0
        self.rank1_history_deltas = 0
        self.rank1_window_span = 0
        self.rank1_prediction_horizon = 0
        self.rank1_evr_mean = 0.0
        self.rank1_r2_mean = 0.0
        self.rank1_zero_motion_tensors = 0
        if self.spectral is not None:
            self.spectral._anchor.clear()

    def set_path_tag(self, tag: Optional[str]) -> None:
        if tag is not None and tag not in PATH_TAGS:
            raise ValueError(f"unknown comm_eff path tag {tag!r}; expected one of {PATH_TAGS} or None")
        self.path_tag = tag

    def should_run_spectral_correction(self) -> bool:
        cadence = int(self.config.spectral.cadence)
        current = self.spectral_step
        return current > 0 and current % cadence == 0

    def note_mask_application(self) -> None:
        """Record one activation-mask hook fire against the current path tag.

        Called by the activation masker from inside a hook. Increments both the
        aggregate counter (``mask_applications``) and the per-path counter for
        ``self.path_tag``. A fire while the tag is anything other than ``train``
        (or ``old_logprob`` under mask_recompute) is a contamination event; the
        masker asserts against it before calling this, but if the assert is ever
        disabled (``python -O``) the per-path counter still records the leak.
        """
        self.mask_applications += 1
        tag = self.path_tag if self.path_tag in self.mask_applications_by_path else TRAIN_TAG
        self.mask_applications_by_path[tag] += 1

    def path_metrics(self) -> dict:
        """Per-path mask-application counters under a stable key prefix.

        Emits ``comm_eff/mask_applications/<tag>`` for every tag; the only
        nonzero key should be ``.../train`` (plus ``.../old_logprob`` under
        mask_recompute). Any other nonzero key is the confinement falsifier.
        The sr_quant codec shares these counters (its hook calls
        ``note_mask_application`` too), so the same falsifier covers it.
        """
        if self.masker is None and self.quantizer is None:
            return {}
        return {f"comm_eff/mask_applications/{tag}": count for tag, count in self.mask_applications_by_path.items()}

    def mask_ratio_metrics(self) -> dict:
        """Most-recently-measured masked fraction per boundary layer.

        Surfaced as ``comm_eff/mask_ratio`` (mean across boundaries) plus a
        per-boundary breakdown and the matched-budget kept-coords/token metric.
        Empty when no mask fired this step.
        """
        if self.masker is None or not getattr(self.masker, "last_mask_ratio", None):
            return {}
        ratios = self.masker.last_mask_ratio
        out = {"comm_eff/mask_ratio": sum(ratios.values()) / len(ratios)}
        for idx, r in sorted(ratios.items()):
            out[f"comm_eff/mask_ratio/layer_{idx}"] = r
        # Matched-budget metric: PRF kept coords per token = (1-p)*H. Compare
        # against PowerSGD's rank for an identical logical PP byte budget.
        # FRLR (issue #89) instead carries rank + k + 1 coords/token (core y,
        # kept residual subset J, one norm scalar), e.g. 32 + 44 + 1 = 77.
        hidden_size = getattr(self.masker, "hidden_size", None)
        p = float(getattr(self.masker, "p", 0.0))
        if bool(getattr(self.masker, "frlr", False)):
            payload = getattr(self.masker, "frlr_payload_per_token", None)
            if payload is not None:
                out["comm_eff/logical_pp_bytes_prf"] = float(payload)
            out["comm_eff/frlr_q_refreshes"] = int(getattr(self.masker, "frlr_q_refreshes", 0))
        elif hidden_size is not None:
            out["comm_eff/logical_pp_bytes_prf"] = float((1.0 - p) * float(hidden_size))
        return out

    def quant_metrics(self) -> dict:
        """sr_quant codec metrics: logical PP bit budget per token per boundary.

        ``comm_eff/logical_pp_bits_sr_quant`` is ``H*bits + n_blocks*16``
        (payload plus one fp16 scale per block; in subset mode
        ``subset_k*bits + subset_k*16/block``), the sr_quant analogue of the
        PRF codec's ``comm_eff/logical_pp_bytes_prf``; the ``_bytes_`` variant
        is the same number divided by 8 for direct budget comparison. Empty
        until the first quant hook fire records the hidden size.
        """
        if self.quantizer is None:
            return {}
        bits_per_token = getattr(self.quantizer, "logical_pp_bits_sr_quant", None)
        if bits_per_token is None:
            return {}
        return {
            "comm_eff/logical_pp_bits_sr_quant": float(bits_per_token),
            "comm_eff/logical_pp_bytes_sr_quant": float(bits_per_token) / 8.0,
        }

    def note_powersgd_application(self) -> None:
        self.powersgd_applications += 1

    def note_powersgd_basis_update(self) -> None:
        self.powersgd_basis_updates += 1

    def spectral_metrics(self) -> dict:
        if self.spectral is None or not self.spectral_rel_change:
            return {}
        values = list(self.spectral_rel_change.values())
        output = {"comm_eff/spectral/rel_change_mean": sum(values) / len(values)}
        output.update(
            {f"comm_eff/spectral/rel_change/{name}": value for name, value in self.spectral_rel_change.items()}
        )
        return output

    def powersgd_metrics(self) -> dict:
        if self.powersgd is None:
            return {}
        output: dict = {
            "comm_eff/powersgd_basis_updates": self.powersgd_basis_updates,
            "comm_eff/powersgd_applications": self.powersgd_applications,
            "comm_eff/logical_pp_bytes_powersgd_y_only": float(
                getattr(self.powersgd, "last_y_coords_per_token", self.powersgd.rank)
            ),
            "comm_eff/fast_q_bootstrap_done": int(getattr(self.powersgd, "fast_q_bootstrap_done", False)),
            "comm_eff/fast_q_bootstrap_observations": int(getattr(self.powersgd, "fast_q_bootstrap_observations", 0)),
            "comm_eff/fast_q_bootstrap_updates": int(getattr(self.powersgd, "fast_q_bootstrap_updates", 0)),
            "comm_eff/fast_q_bootstrap_activations": int(getattr(self.powersgd, "fast_q_bootstrap_activations", 0)),
            "comm_eff/fast_q_bootstrap_dense_observation_elements": float(
                getattr(self.powersgd, "fast_q_bootstrap_dense_observation_elements", 0.0)
            ),
            "comm_eff/fast_q_bootstrap_sync_elements": float(
                getattr(self.powersgd, "fast_q_bootstrap_sync_elements", 0.0)
            ),
        }
        q_conditions = getattr(self.powersgd, "last_q_cond", {})
        if q_conditions:
            finite = [v for v in q_conditions.values() if v == v and v not in (float("inf"), float("-inf"))]
            output["comm_eff/powersgd_q_cond"] = (
                sum(finite) / len(finite) if len(finite) == len(q_conditions) else float("inf")
            )
            output.update(
                {f"comm_eff/powersgd_q_cond/layer_{idx}": value for idx, value in sorted(q_conditions.items())}
            )
        reconstruction = getattr(self.powersgd, "last_reconstruction_rel_error", {})
        if reconstruction:
            output["comm_eff/powersgd_reconstruction_rel_error"] = sum(reconstruction.values()) / len(reconstruction)
            output.update(
                {
                    f"comm_eff/powersgd_reconstruction_rel_error/layer_{idx}": value
                    for idx, value in sorted(reconstruction.items())
                }
            )
        bootstrap_deviation = getattr(self.powersgd, "fast_q_bootstrap_cross_rank_max_rel_dev", None)
        if bootstrap_deviation is not None:
            output["comm_eff/fast_q_bootstrap_cross_rank_max_rel_dev"] = float(bootstrap_deviation)
        q_deviation = getattr(self, "_powersgd_q_agreement_dev", None)
        if q_deviation is not None:
            output["comm_eff/powersgd_q_cross_rank_max_rel_dev"] = float(q_deviation)
        compressed = float(getattr(self.powersgd, "last_elems_compressed", 0.0))
        dense = float(getattr(self.powersgd, "last_elems_dense_equiv", 0.0))
        if dense > 0.0:
            output["comm/bytes_compressed"] = compressed
            output["comm/bytes_dense_equiv"] = dense
            output["comm/bytes_ratio"] = compressed / dense
        return output

    def metrics(self) -> dict:
        output = {
            "comm_eff/mask_applications": self.mask_applications,
            "comm_eff/anchor_backwards": self.anchor_backwards,
            "comm_eff/spectral_corrections": self.spectral_corrections,
            "comm_eff/anchor_grad_corrected": self.anchor_grad_corrected,
            "comm_eff/anchor_rollouts_generated": self.anchor_rollouts_generated,
            "comm_eff/anchor_rewards_recomputed": self.anchor_rewards_recomputed,
            "comm_eff/anchor_optimizer_steps": self.anchor_optimizer_steps,
            "comm_eff/anchor_batch_fraction": self.anchor_batch_fraction,
            "comm_eff/anchor_batch_sequences_global": self.anchor_batch_sequences_global,
            "comm_eff/anchor_update_sequences_global": self.anchor_update_sequences_global,
            "comm_eff/anchor_batch_prompt_equivalents_global": self.anchor_batch_prompt_equivalents_global,
            "comm_eff/anchor_update_prompt_equivalents_global": self.anchor_update_prompt_equivalents_global,
            "comm_eff/anchor_rollout_n": self.anchor_rollout_n,
            "comm_eff/anchor_batch_scope_rollout": self.anchor_batch_scope_rollout,
            "comm_eff/spectral_step": self.spectral_step,
            "comm_eff/anchor_q_updates": self.anchor_q_updates,
            "comm_eff/anchor_q_broadcasts": self.anchor_q_broadcasts,
            "comm_eff/anchor_q_activations": self.anchor_q_activations,
            "comm_eff/anchor_q_stage_overwrites": self.anchor_q_stage_overwrites,
            "comm_eff/merger_coldM_fallbacks": self.merger_coldM_fallbacks,
            "comm_eff/anchor_replay_fires": self.anchor_replay_fires,
        }
        if self.rank1_relex_active():
            output.update(
                {
                    "comm_eff/rank1_m_ready": int(self.rank1_m_ready),
                    "comm_eff/rank1_q_only_fires": self.rank1_q_only_fires,
                    "comm_eff/rank1_warmup_correction_fires": self.rank1_warmup_correction_fires,
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
        return output


def maybe_build_comm_eff_state(config: Any) -> Optional[CommEffState]:
    """Construct state only when the top-level switch is enabled."""

    if not _is_enabled(config):
        return None
    state = CommEffState(config)
    logger.info("comm_eff: enabled — constructed CommEffState")
    return state


def comm_eff_metrics(state: Optional[CommEffState]) -> dict:
    """Return the combined runtime metrics for an active communication-efficient state."""
    if state is None:
        return {}
    output = state.metrics()
    output.update(state.spectral_metrics())
    output.update(state.powersgd_metrics())
    output.update(state.path_metrics())
    output.update(state.mask_ratio_metrics())
    output.update(state.quant_metrics())
    return output
