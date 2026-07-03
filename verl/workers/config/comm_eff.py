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

"""Configuration for the communication-efficient pipeline-adaptation method.

This module defines the ``comm_eff`` config group: the two-circuit compression
method (pipeline activation masking + asynchronous unmasked anchor circuit +
spectral correction of masked gradients). The config defaults to **disabled**,
in which case the integration hooks in the actor train path are strict no-ops
(see ``verl.workers.comm_eff.state`` and the guards in ``engine_workers``,
``engine/base`` and ``engine/fsdp/transformer_impl``).

The nested sub-configs validate the Hydra schema up front, but carry no
behavior while ``enabled=false``.
"""

from dataclasses import dataclass, field

from verl.base_config import BaseConfig

__all__ = [
    "CommEffMaskConfig",
    "CommEffAnchorConfig",
    "CommEffSpectralConfig",
    "CommEffPowerSGDConfig",
    "CommEffCaptureConfig",
    "CommEffWeightTrajConfig",
    "CommEffProbeConfig",
    "CommEffConfig",
]

# The Q-basis families PowerSGD may use. ``act`` is the activation-energy basis
# built from boundary activations; the others are alternate sketch sources. All
# share the same fixed rank ``r`` and therefore the same byte budget.
Q_BASIS_FAMILIES = ("act", "grad", "adv", "tail", "hybrid", "ticket")

# The compression codecs ``comm_eff.compression_type`` may select. Exactly one
# codec is active per run (mutually exclusive). ``dense`` leaves the activation
# path uncompressed (equivalent to ``comm_eff.enabled=false`` for that path);
# ``prf_mask`` is the per-(token, dim) PRF Bernoulli mask; ``powersgd`` is the
# shared frozen-basis PowerSGD-style projector
# ``M_hat = (M @ Q) @ Qᵀ`` at the same logical PP byte budget.
COMPRESSION_TYPES = ("dense", "prf_mask", "powersgd")


@dataclass
class CommEffMaskConfig(BaseConfig):
    """Per-(token, dimension) pipeline-boundary activation masking (inert while disabled).

    A deterministic PRF Bernoulli mask applied in-graph (``h_tilde = h * mask``)
    to the boundary decoder blocks, only on the actor-train forward (and the
    old-logprob recompute when ``mask_recompute``). Each token independently keeps
    ``round((1-p)*H)`` dims; the mask is keyed on each token's stable
    ``(sample_id, position_id)`` so it is packing-invariant across the
    differently-packed forwards. See ``verl.workers.comm_eff.activation_mask``.

    Args:
        enabled (bool): Whether masking runs (still gated by ``comm_eff.enabled``).
        p (float): Masked (zeroed) fraction in ``[0, 1]``; ``comm_eff/mask_ratio``
            tracks it. ``0.0`` means no masking.
        seed (int): Base seed folded into the mask PRF key.
        pp_size (int): Logical pipeline-shard count; the last block of every shard
            except the final one is masked (``L`` from ``model.config``).
            ``L=16, pp_size=8`` -> ``[1, 3, 5, 7, 9, 11, 13]``.
        mask_recompute (bool): When ``True`` the mask also fires on the
            old-logprob recompute, so both gradient-feeding forwards are masked.
            The stable key gives a token the identical mask in both, keeping the
            PPO importance ratio ≈1 at the first inner step. ``False`` (default)
            masks only the train forward.
        rescale (bool): ``False`` (default) writes the raw product ``h * mask``;
            ``True`` applies inverted-dropout ``h * mask / (1 - p)`` so
            ``E[h_tilde] = h`` (requires ``p < 1``). The theory wants the
            ``1/(1-p)`` rescale (unbiased), but the supervised reference omits it
            (the "dropped activations are zeros" reading, and ``1/(1-p)`` is
            bf16-risky at high ``p``); default matches the reference.
    """

    enabled: bool = True
    p: float = 0.95
    seed: int = 0
    pp_size: int = 8
    mask_recompute: bool = False
    rescale: bool = False
    rescale_mode: str = "auto"


@dataclass
class CommEffAnchorConfig(BaseConfig):
    """Asynchronous unmasked-anchor-circuit sub-config (inert while disabled).

    Every ``cadence`` trainer steps, the anchor circuit runs one unmasked
    GRPO-actor-loss forward/backward from a ``delay_K``-stale weight snapshot to
    produce a clean per-target gradient ``G_anchor``. ``G_anchor`` is read RAW
    into the spectral filter's EMA ``M_anchor``. The anchor takes NO optimizer
    step and generates NO rollouts / recomputes NO rewards.

    Args:
        enabled (bool): Whether the anchor circuit runs. Gated by the parent
            ``comm_eff.enabled`` regardless of this value. ``false`` (default) is
            a strict no-op — opt-in only.
        cadence (int): Anchor-refresh cadence in trainer steps. The anchor fires
            when ``(step % cadence) == 0``. Must be ``>= 1``.
        delay_K (int): Staleness of the weight snapshot the anchor forwards
            from, in trainer steps. ``0`` = current weights; ``1`` = the prior
            step's weights. Must be ``>= 0``.
        owns_q (bool): When ``true`` the anchor owns the PowerSGD projection
            basis ``Q``. The fast net becomes a read-only consumer of ``Q`` and
            the anchor computes ``Q <- orth(V)`` from its own unmasked forward
            activations before broadcasting ``Q`` and ``M`` to the DP group.
        replay_paired_batch (bool): When ``true`` the anchor replays the paired
            ``(batch[t-delay_K], generator_snapshot[t-delay_K])`` that produced
            the fast circuit's rollout. This makes the anchor gradient comparable
            to the retained fast gradient at the same batch/weight point. When
            ``false``, the anchor uses the current batch with stale weights.
        snapshot_device (str): Where the anchor weight snapshots live between
            ticks — ``"gpu"`` (default, faithful: detached clones stay on each
            param's device, today's exact behaviour) or ``"cpu"`` (memory-lean:
            the ``delay_K+1`` full bf16 snapshots move off HBM; the clone load
            casts back via ``.to(p.device, p.dtype)``, a byte-preserving round
            trip — numerics-neutral). Applies to BOTH the staleness
            queue and the paired replay ring. Validated against {gpu, cpu}.
        lookahead_anchor (bool): Master flag for the look-ahead (weight-
            projection) anchor — M4. When active (this flag AND a
            non-``disabled`` ``lookahead_mode``), the anchor backward runs from
            a linearly extrapolated weight point ``theta_hat[t]`` instead of the
            raw ``delay_K``-stale snapshot. ``false`` (default) is a strict
            no-op: no snapshot ring, no projector, no extra logs — byte-
            identical to today. See ``verl/workers/comm_eff/lookahead.py``.
        lookahead_mode (str): ``"disabled"`` (default) | ``"fixed_linear"``
            (frozen AsyncPP seed: pure linear extrapolation over the recorded
            snapshot ticks) | ``"learned_linear_with_fixed_linear_cold_start"``
            (same seed plus a small per-block residual trained ONLY from
            retrospective prediction errors — the no-peek invariant). A stray
            ``lookahead_anchor=true`` with ``lookahead_mode=disabled`` (or
            vice-versa) is inert by design.
        lookahead_strength (float): Projection horizon multiplier ``alpha``.
            ``1.0`` (default) projects the full realized horizon (catch-up to
            the current tick); ``<1`` projects a shorter horizon; ``0`` degrades
            to the raw stale weights. Must be ``>= 0``.
        lookahead_rollout_source (str): Which rollouts the anchor consumes when
            the look-ahead projector is on. ``"auto"`` (default) resolves to
            ``"current_step"`` when the projector is active and to
            ``"stale_paired"`` otherwise — so matching rollouts are THE DEFAULT
            whenever weight projection is ON and the knob has zero effect when
            it is OFF. ``"stale_paired"`` = today's exact behavior (the replayed
            ``t-delay_K`` batch in replay mode). ``"current_step"`` = the anchor
            consumes the CURRENT tick's batch — the step-``t`` rollouts that the
            projected ``theta_hat[t]`` corresponds to; requires the projector ON
            (stale-weights + fresh-rollouts is an unsupported ablation).
            ``"self_generate"`` is a RESERVED seam (the anchor generating its
            own rollouts) and is rejected as not implemented.
        warmup_mode (str): What the anchor does at fires BEFORE the look-ahead
            projector is ready (fewer than the required source snapshots
            retained). ``"stale_correct"`` (DEFAULT, today's exact behavior):
            the anchor computes ``M`` from the raw stale ``theta[t-K]`` + paired
            stale rollouts and the merger folds it every ``spectral.cadence``
            ticks — the k-collapse warmup that A0 suffered. ``"no_correct"``:
            do ALL ring/snapshot bookkeeping but SKIP the anchor clone fwd/bwd
            and the ``M`` update entirely, so ``M`` stays cold and the merger's
            cold-M guard passes the fast gradient through UNCHANGED (no
            correction) until the FIRST projected fire. Only meaningful with the
            look-ahead projector on; requires ``owns_q=false`` (a skipped anchor
            pass must not be the sole Q updater). Validated against
            {stale_correct, no_correct}.
        lookahead_min_snapshots (int): Minimum ring snapshots required before the
            projector engages. ``-1`` (DEFAULT): the mode's full source count
            (2 for fixed_linear, 3 for the learned mode) — today's behavior. A
            concrete value in ``[2, mode_n_points]`` lets the projector engage at
            the earliest mathematically-legal fire: ``2`` projects from fire 2
            (the first fire at which two ``>=K``-stale snapshots exist — fire 1
            can NEVER project, a line needs 2 points). Retention is unchanged
            (the ring still holds ``mode_n_points`` for the learned residual);
            only readiness is relaxed. Requires the projector enabled.
    """

    enabled: bool = False
    cadence: int = 20
    delay_K: int = 20
    owns_q: bool = False
    replay_paired_batch: bool = False
    snapshot_device: str = "gpu"
    lookahead_anchor: bool = False
    lookahead_mode: str = "disabled"
    lookahead_strength: float = 1.0
    lookahead_rollout_source: str = "auto"
    warmup_mode: str = "stale_correct"
    lookahead_min_snapshots: int = -1


@dataclass
class CommEffSpectralConfig(BaseConfig):
    """Anchor-guided gradient-correction sub-config (inert while disabled).

    Applies an anchor combiner to selected 2D decoder gradients after the actor
    backward and before ``optimizer.step()``, using the anchor-gradient EMA
    ``M_anchor`` (no SVD / no basis). See
    ``verl.workers.comm_eff.spectral_filter.SpectralFilter``.

    Supported correction modes include ``delayed_ef``, ``ef_powersgd``,
    ``signed_ema``, ``inject``, ``blend`` and ``none``. The delayed error-feedback
    mode uses paired anchor replay to estimate the codec residual at a matching
    batch/weight point::

        G_corr(t) = G_comp(t) + lambda * (M_rep - G_comp_ring(t - K))

    Other modes remain available for controlled comparisons. A cold-M guard
    returns the fast gradient unchanged whenever ``M_anchor`` is unwarmed, so the
            correction path returns the fast gradient unchanged until the anchor
            state is warm.

    Args:
        enabled (bool): Whether anchor-guided correction of masked gradients
            runs. Gated by the parent ``comm_eff.enabled`` regardless of this
            value. ``false`` (default) ⇒ the grad-correction hook is a strict
            no-op and the actor path is identical to dense GRPO.
        beta_anc (float): EMA decay for the anchor-gradient running matrix
            ``M_anchor``. Default ``0.95``.
        target_substr (list[str]): Substrings used to SELECT which named 2D
            parameters receive correction. A parameter is targeted iff its name
            contains one of these substrings AND its logical shape is 2D.
            Defaults select the decoder attention/MLP projection matrices and
            skip norms, biases, embeddings and the lm head.
        max_targets (int): Cap on the number of target matrices corrected per
            step. ``-1`` means no cap. The cap applies to both anchor extraction
            and merger writeback.
        ema_device (str): Where the anchor-gradient EMA ``M_anchor`` is stored
            between refreshes — ``"gpu"`` (default, faithful: kept in HBM) or
            ``"cpu"`` (memory-lean: offloaded to pinned CPU, moved to GPU only
            inside the refresh/correct call and moved back). ``M_anchor`` is
            touched only at refresh, so CPU offload costs one H2D/D2H per refresh,
            not per mini-batch. Validated against {gpu, cpu}.
        correction_mode (str): The anchor combiner the fast-path grad uses.
            ``"none"`` (default) = no correction. ``"delayed_ef"`` applies the
            paired-replay codec residual ``G_comp + lambda*(M_rep - G_comp_ring)``.
            ``"inject"`` and ``"blend"`` are alternate combiners. Validated against
            {none, inject, blend, signed_ema, ef_powersgd, delayed_ef}.
        inject_gamma (float): Injection strength for
            ``correction_mode="inject"``; unused otherwise. Must be ``>= 0``.
        blend_eta (float): Convex-blend weight for ``correction_mode="blend"``;
            validated to ``[0, 1]``. Unused otherwise.
        signed_ema_alpha (float): The signed_ema merger weight ``alpha`` in
            ``G_corr = alpha*G_noisy + (1-alpha)*|G_noisy|*sign(M_anchor)``.
            ``alpha=0`` is the pure sign-merger; ``alpha=1`` returns ``G_noisy``
            unchanged. Validated to ``[0, 1]``. Unused unless
            ``correction_mode=signed_ema``.
        cadence (int): Correction cadence in optimizer steps.
            The grad-correction hook (``_maybe_comm_eff_grad_correction``) fires
            only when ``(spectral_step % cadence) == 0`` on the monotonic
            per-optimizer-step counter, MIRRORING the anchor cadence
            (``anchor_should_fire``) and the clean-step cadence
            (``CommEffState.is_clean_step``). ``1`` (default) fires EVERY step =
            the default behavior. Set ``> 1`` (e.g. ``2``) to align
            correction with a matching ``anchor.cadence`` so the
            correction always uses a freshly-refreshed anchor EMA instead of a
            stale one on the in-between steps. Must be ``>= 1``.
        diagnostics (bool): When ``False``, skip the per-step spectral
            DIAGNOSTIC overhead — the per-matrix ``relative_change()``
            compute+GPU→CPU ``.item()`` sync, the per-matrix/per-merge
            diagnostic prints, and the anchor-fire relevance probe (a diagnostic
            forward). ``True`` (default) = current behavior, byte-identical.
            Neutral: nothing the optimizer sees changes — the optimizer-visible
            ``g_corr`` writeback, the bitwise anchor canary assert, and the
            aggregate W&B counters (``anchor_backwards`` / ``bytes_ratio`` /
            ``merger_coldM_fallbacks`` / ``spectral_corrections``) are preserved
            in both states. Set ``False`` only for runtime efficiency.
    """

    enabled: bool = False
    beta_anc: float = 0.95
    cadence: int = 1
    # When False, skip the per-step spectral DIAGNOSTIC overhead (per-matrix
    # rel_change compute+sync+print and the anchor relevance probe). Default
    # True = current behavior, byte-identical. Neutral: nothing the optimizer
    # sees changes; the canary assert and aggregate counters are preserved.
    diagnostics: bool = True
    target_substr: list = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    # Full coverage by default (-1 = no cap). The cap applies to both anchor
    # extraction and merger writeback.
    max_targets: int = -1
    # EMA storage default: keep tensors on GPU.
    ema_device: str = "gpu"
    # Correction mode (anchor combiner). "none" (default) = no correction.
    # "delayed_ef" applies the paired-replay codec residual; "inject"/"blend" are
    # alternate anchor combiners; "ef_powersgd" is direction-preserving residual
    # feedback; "signed_ema" is the sign-replacement combiner.
    correction_mode: str = "none"
    # Injection strength for correction_mode="inject"; unused otherwise.
    inject_gamma: float = 1.0
    # Convex-blend weight for correction_mode="blend"; validated to [0, 1].
    blend_eta: float = 0.5
    # signed_ema weight alpha in
    # G_corr = alpha*G_noisy + (1-alpha)*|G_noisy|*sign(M_anchor). alpha=0 is the
    # pure sign-merger; alpha=1 returns G_noisy unchanged. Validated to [0, 1].
    # Unused unless correction_mode=signed_ema.
    signed_ema_alpha: float = 0.0
    # Error-feedback PowerSGD merger (direction-preserving).
    # ``correction_mode="ef_powersgd"`` re-injects the PowerSGD reconstruction
    # residual ``e_t = G_dense_proxy - G_comp`` (the off-subspace component the
    # rank-r projection dropped), accumulated with decay, ADDED to G_comp with NO
    # sign term — so the corrected gradient KEEPS G_comp's direction/sign and only
    # restores the dropped magnitude along the off-principal directions. Unlike
    # signed_ema (which REPLACES the sign with sign(M)), this is sign-preserving.
    #
    #   e_t   <- ef_decay * e_{t-1} + (M_anchor - P_Q(M_anchor))         # residual
    #   e_t   <- clip(e_t, ef_clip * ||G_comp||)                         # norm cap
    #   G_corr = G_comp + e_t                                            # NO sign
    #
    # The residual proxy uses the anchor EMA's off-subspace component
    # (M_anchor minus its projection onto the span of G_comp). The merger reduces
    # to plain PowerSGD when ef_decay=0 and ef_clip=0.
    ef_decay: float = 0.0
    # ef_clip: the residual norm cap as a FRACTION of ||G_comp|| (shape-aware,
    # per-matrix). 0.0 ⇒ the residual is fully zeroed ⇒ G_corr == G_comp (the
    # plain-PowerSGD limiting case). A typical live value is ~1.0 (the re-injected
    # residual may not exceed the compressed gradient's own norm). Must be >= 0.
    ef_clip: float = 0.0
    # K-delayed exact codec residual (correction_mode="delayed_ef").
    # G_corr(t) = G_comp(t) + delayed_ef_lambda * (M_rep − G_comp_ring(t−K)), where
    # M_rep is the beta_anc=0 anchor EMA from paired replay and G_comp_ring(t-K) is the fast compressed
    # gradient stored at the identical (batch, θ) tick by the fire-aware
    # FastGradRing. δ refreshes at anchor fires and is HELD between them (the
    # telescoping per-tick injection). 0.0 means G_corr == G_comp exactly. Must
    # be >= 0. Unused unless correction_mode=delayed_ef.
    delayed_ef_lambda: float = 0.0
    # Additive stale-anchor rank-r_sb sub-basis (delayed_ef).
    # When delta_subbasis_rank (r_sb) > 0, the delayed_ef merger ADDS a rank-r_sb
    # low-rank reconstruction of the source S into its correction term:
    #
    #   delta_subbasis = rank_{r_sb}(S)                   # seeded randomized SVD
    #   G_corr(t) = G_comp(t) + lambda * (delta + delta_subbasis)
    #
    # delta_subbasis_family selects S: "tail" (default) uses the
    # act-deflated stale weight gradient = the off-act-principal direction the
    # codec structurally drops; "grad" uses M_rep. The sub-basis enters only the
    # correction term; the forward codec Q is untouched. r_sb = 0 skips the branch.
    # The randomized SVD is per-target seeded so the sub-basis is bit-identical
    # across DP ranks. Must be >= 0. Unused unless correction_mode=delayed_ef.
    delta_subbasis_rank: int = 0
    # The sub-basis source family: "tail" (act-deflated grad) or "grad" (raw
    # stale anchor gradient). Validated to {tail, grad}.
    delta_subbasis_family: str = "tail"
    # Sub-basis weight and optional linear decay. ``delta_subbasis_weight`` scales
    # the additive sub-basis term and ``delta_subbasis_decay_steps`` linearly
    # decays it to zero, optionally after ``delta_subbasis_hold_steps`` steps:
    #
    #   gamma_t      = delta_subbasis_weight * decay_factor
    #   decay_factor = 1.0                              if D <= 0   (constant γ = weight)
    #                = 1.0                              if D > 0 and step < H (HOLD)
    #                = max(0, 1 - (step - H) / D)       else        (linear 1→0, clamped)
    #   G_corr(t)    = G_comp(t) + lambda * (delta + gamma_t * delta_subbasis)
    #
    # The decay knob is a scalar on the deterministic DP-mean sub-basis; it does
    # not alter the seeded SVD. Both fields are validated unconditionally.
    delta_subbasis_weight: float = 1.0
    delta_subbasis_decay_steps: int = 0
    # Hold-then-decay schedule: ``delta_subbasis_hold_steps`` is the number of
    # steps gamma holds at full ``weight`` before the linear decay begins.
    #
    #   decay_factor = 1.0                              if D <= 0          (constant)
    #                = 1.0                              if D > 0, step < H (HOLD shelf)
    #                = max(0, 1 - (step - H) / D)       otherwise          (linear ramp)
    #
    # hold_steps=0 leaves no shelf. Only meaningful when decay_steps > 0. Pure
    # scalar; no RNG or SVD state is changed. Must be >= 0.
    delta_subbasis_hold_steps: int = 0
    # Correction-delta compression rank. A value > 0 reserves config space for
    # compressing the correction before injection; the current path leaves the
    # correction uncompressed at 0. Must be >= 0. Unused unless
    # correction_mode=delayed_ef.
    r_delta: int = 0
    # Optional zero-mean gradient perturbation after the delayed_ef correction.
    # The perturbation is cross-rank-identical and scaled by ||g_corr||:
    #
    #   ξ          = randn(g_corr.shape, gen=seed(perturb_seed, target, step))  # cross-rank-identical
    #   ξ          = ξ / ‖ξ‖                                                     # unit
    #   g_corr     = g_corr + perturb_sigma · ‖g_corr‖ · ξ                       # ‖perturbation‖ = σ·‖g_corr‖
    #
    # perturb_sigma = 0.0 skips the perturbation branch. The seed is a pure
    # function of (perturb_seed, canonical-target-name, current_step) with NO
    # rank/device-local state, so every DP rank draws the SAME ξ (the
    # multi-rank-agreement invariant — else ranks diverge). Fresh per step ⇒
    # zero-mean over training. The perturbation adds no communication. Both must
    # be >= 0 / int and are validated unconditionally.
    perturb_sigma: float = 0.0
    perturb_seed: int = 0
    # Normalized EMA momentum for the delayed_ef correction. The buffer is updated
    # only at refresh ticks; between anchor fires the held correction is reused:
    #
    #   REFRESH: m ← μ·m + (1−μ)·δ      (first fire: m = δ.clone());  correction = m
    #   HELD:    correction = m         (the held buffer; with age_decay below it is
    #                                    scaled by μ**(step − last_refresh_step) so a
    #                                    long hold fades the APPLIED correction → 0)
    #
    # The normalized recurrence has stationary gain 1 for a constant delta.
    # delta_momentum_mu = 0.0 skips the branch. With age decay enabled, a held
    # correction fades by its actual age if refreshes arrive late or stop.
    # The buffer is built from the DP-mean delta and is shape-keyed.
    delta_momentum_mu: float = 0.0
    delta_momentum_age_decay: bool = False
    # Adaptive delayed_ef dose. The constant lambda in
    # g_corr = G_comp + lambda * delta can be replaced by a per-target, per-tick
    # lambda_t based on the deviation from the target's typical agreement:
    #
    #   c_t = cos(G_comp, M_rep)  [mode=cos]   OR   ‖δ‖/‖G_comp‖  [mode=ratio]   # RAW δ
    #   c̄   = running MEDIAN of c_t over a bounded per-target history (last 64)
    #   λ_t = clamp(delayed_ef_lambda + κ·(c̄ − c_t), 0.0, lambda_cap)
    #   g_corr = G_comp + λ_t·correction
    #
    # Centering around the running median keeps the average dose tied to
    # delayed_ef_lambda instead of adding a constant boost. mode in
    # {off, cos, ratio}; kappa >= 0; lambda_cap >= 0.
    adaptive_lambda_mode: str = "off"
    adaptive_lambda_kappa: float = 0.0
    lambda_cap: float = 2.0


@dataclass
class CommEffPowerSGDConfig(BaseConfig):
    """PowerSGD-style pipeline-boundary activation-compression sub-config.

    The codec replaces each boundary block's hidden-state output ``M`` (shape
    ``(N, H)`` — ``N`` packed tokens × ``H`` hidden dims) with its rank-``r``
    projection onto a **shared, frozen, per-layer orthonormal basis** ``Q``
    (shape ``(H, r)``)::

        M_hat = (M @ Q) @ Q.T            # forward; Q detached, M in-graph (NO STE)

    so the pipeline boundary transmits only the ``N·r`` projected coordinates
    ``Y = M @ Q`` (plus the shared, communication-free ``Q``) instead of ``N·H``
    — the **identical logical PP byte budget as the PRF mask at ``p = 1 − r/H``**
    (``r=102 ≡ p=0.95`` at ``H=2048``). Because ``Q`` is detached and ``M`` stays
    in-graph, the backward is the exact self-adjoint projector
    ``dL/dM = (dL/dM_hat) · Q Qᵀ`` with no straight-through estimator.

    The basis is bootstrapped with **zero communication** from a deterministic
    per-layer seed ``seed_L = (base_seed·1_000_003 + layer_idx·7919) & 0x7FFFFFFF``
    in fp32, identical on every rank, and refined by block power iteration: on
    compressed *train* forwards we
    accumulate, OFF the autograd graph, ``V += Mᵀ (M Q)`` (one sketch per
    boundary forward), then once at end-of-actor-update (when not a clean step
    and ``global_step % update_cadence == 0``) set ``Q ← orth(V)`` in fp32 and
    clear ``V``. ``Q`` is **frozen for the entire global step** — the
    old-logprob recompute and the actor-train forward both see ``Q_t``; the
    update to ``Q_{t+1}`` happens only after the gradient-bearing actor work, so
    the GRPO importance ratio starts near 1 with no weight change.

    Inert unless ``comm_eff.enabled=true`` AND ``comm_eff.compression_type ==
    "powersgd"``. Every key is registered regardless of ``compression_type`` so a
    ``prf_mask`` run that passes ``comm_eff.powersgd.rank=...`` still parses.

    Args:
        enabled (bool): Sub-switch; gated by the parent ``comm_eff.enabled`` AND
            by ``compression_type == "powersgd"``. ``True`` (default) so simply
            selecting ``compression_type=powersgd`` activates it; the parent
            gates remain the real master switches.
        rank (int): Retained projection rank ``r``. ``r=102`` matches the PRF
            mask at ``p=0.95`` (``q·H = 0.05·2048 = 102.4``). Must be ``>= 1``.
            ``r == H`` is the lossless limiting case ``M_hat = M``.
        seed (int): Base seed for the deterministic per-layer basis bootstrap
            folded with ``layer_idx``. Identical on every rank ⇒
            zero-communication codebook init.
        pp_size (int): Logical pipeline-shard count; the same boundary-block
            selection as the PRF mask (last block of every shard except the
            final). ``L=16, pp_size=8 -> [1,3,5,7,9,11,13]``. Must be ``>= 1``.
        update_cadence (int): Block-power-iteration basis-update cadence in
            optimizer steps. ``1`` (default) refreshes ``Q`` every (non-clean)
            step. ``> 1`` refreshes only when ``global_step % update_cadence ==
            0`` (a quieter, more stable basis). Must be ``>= 1``.
        warm_start (bool): ``True`` (default) carries ``Q`` across steps (warm
            block power iteration — the standard PowerSGD warm start). ``False``
            re-bootstraps ``Q`` from the per-layer seed every update.
        compress_recompute (bool): When ``True`` (default) the projector also
            fires on the old-logprob recompute, so BOTH gradient-feeding forwards
            see the same frozen ``Q_t`` ⇒ ``ρ ≈ 1`` (the analogue of the mask's
            ``mask_recompute``). ``False`` projects only the actor-train forward
            (old-logprob runs dense), which dense-anchors the importance ratio's
            denominator.
        sync_basis (bool): ``True`` (default — REQUIRED for a correct shared
            codebook under DP). Each DP rank builds its sketch ``V`` from its OWN
            data shard (the dispatch scatters a different shard per rank), so a
            per-rank ``orth(V)`` would DIVERGE the per-boundary ``Q`` across ranks
            after the first update. ``True`` all-reduces the raw sketches over the
            DP group before ``orth`` so every rank orthonormalizes the SAME pooled
            ``V_global = Σ_ranks V`` → a bit-identical consensus ``Q`` on every
            rank, differing only per boundary. The collective is made
            deadlock-safe by iterating the fixed
            ``boundary_indices`` on every rank. ``False`` keeps each rank's basis
            local — only correct if every rank sees identical data, which DP
            does not.
        qr_dtype (str): Dtype for the orthonormalization (``orth``/QR) and the
            stored basis math — ``"fp32"`` (default, required for correctness:
            bf16-QR loses orthogonality, drifts ``QᵀQ`` from ``I``, and is a
            frequent NaN / ``q_cond`` source) or ``"bf16"``. The projection itself
            runs in the activation dtype regardless; only the QR/orth + ``V``
            accumulation are in ``qr_dtype``.
        reortho_eps (float): Floor added under the QR when forming the basis, and
            the singular-value floor used in the ``q_cond`` diagnostic, to keep a
            rank-deficient sketch from producing a non-finite condition number.
            Must be ``> 0``.
    """

    enabled: bool = True
    rank: int = 102
    seed: int = 0
    pp_size: int = 8
    update_cadence: int = 1
    warm_start: bool = True
    compress_recompute: bool = True
    sync_basis: bool = True
    qr_dtype: str = "fp32"
    reortho_eps: float = 1e-6
    # Q-basis family: the content of the sketch consumed by ``orth(V)`` at fixed
    # rank. ``act`` builds V from boundary activations. Other families use
    # alternate GRPO-related sketch sources while keeping the same byte budget.
    q_basis: str = "act"
    # Passive basis families to accumulate inside the anchor pass, off the live
    # Q, fast path, and optimizer. Empty means no passive accumulation.
    q_basis_passive: list = field(default_factory=list)
    # Hybrid family column split at fixed total rank r. ``-1`` means AUTO:
    # ``hybrid_act_cols = ceil(r/2)`` and ``hybrid_grad_cols = r - act``. When
    # both are explicit, they must sum to the rank.
    hybrid_act_cols: int = -1
    hybrid_grad_cols: int = -1


@dataclass
class CommEffCaptureConfig(BaseConfig):
    """Diagnostic tensor-capture sub-config.

    Off by default. When ``enabled=true`` the comm-eff hooks dump fp32 tensors
    such as ``A``, ``(A@Q)Q.T``, ``Q``, ``G_comp``, ``G_corr``, ``M``,
    ``G_anchor``, ``G_dense`` and the fresh-anchor measurement grad under
    ``capture_dir``. Every tensor is detached and dump-only; capture adds I/O
    but no optimizer-visible numerical side effect.

    The two measurement-only PROBES (``capture_g_dense`` = a second uncompressed
    fast backward; ``capture_fresh_anchor`` = a delay_K=0 fresh-anchor grad) are
    the expensive, highest-integration-risk dumps; they are independently gated so
    the audit can be staged. ``delay_K=0`` appears ONLY here as a removed probe —
    it is forbidden as a TRAINING config (Correctness invariant "mandatory anchor
    staleness").

    Args:
        enabled (bool): Master capture switch. ``false`` (default) ⇒ NO dump hook
            fires, NO probe backward runs, NO tensor is written.
        capture_dir (str): Directory the fp32 dumps + the manifest are written to
            on the box. Empty ⇒ ``./captures`` relative to cwd.
        max_ticks (int): Cap on the number of optimizer ticks captured (the audit
            needs only ~5-10). ``<= 0`` ⇒ no cap. Bounds disk + rsync volume.
        stratified_targets (int): If ``> 0``, dump only this many targets per
            layer-type. ``0`` means dump every target.
        capture_g_dense (bool): Run the parallel UNCOMPRESSED fast backward to
            capture ``G_dense`` alongside ``G_comp`` at the SAME step. Detached /
            dump-only — MUST NOT touch the optimizer. ``false`` (default) ⇒ no
            second backward (the highest-OOM-risk probe; gate it on first).
        capture_fresh_anchor (bool): Capture the ``delay_K=0`` fresh-anchor grad as
            a MEASUREMENT probe for the sign-agreement decomposition. Detached /
            dump-only. ``false`` (default).
        fresh_anchor_loss (str): Loss the ``delay_K=0`` fresh-anchor measurement
            probe backward uses: ``"clean_pg"`` (default) or ``"ppo_clip"``.
            Affects only the dump-only fresh-anchor probe, never the optimizer,
            EMA, or K-stale ``G_anchor`` that feeds ``M``.
        dump_dtype (str): Dump precision. ``"fp32"`` (default, required for the
            fidelity invariant — the reconstruction_rel_error recomputed from the
            dump must match the logged scalar within 1e-3). ``"bf16"`` reduces
            dump volume.
    """

    enabled: bool = False
    capture_dir: str = ""
    max_ticks: int = 10
    stratified_targets: int = 0
    capture_g_dense: bool = False
    capture_fresh_anchor: bool = False
    # Loss for the delay_K=0 fresh-anchor probe. Affects only the dump-only probe.
    fresh_anchor_loss: str = "clean_pg"
    dump_dtype: str = "fp32"
    # Disk-volume guard: capture only rank 0 by default. Set False only if
    # per-rank shards are genuinely needed.
    rank0_only: bool = True
    # Skip capture ticks below this value. Useful for avoiding cold-start capture.
    min_tick: int = 0
    # Cloudflare R2 offload for the raw grad/activation .pt dumps: upload-then-
    # delete-local (creds from the env, R2_BUCKET=shamane-pluralis guard). Off =>
    # local-only (byte-identical). For an ACCEPTED raw-grad collection run pair
    # this with max_ticks=0 (no cap) + stratified_targets=0 (every target).
    r2_enabled: bool = False
    r2_delete_local: bool = True
    # Async batched upload (opt-in) for the grad/activation dumps. Off =>
    # synchronous (byte-identical). On => a background worker pool overlaps uploads
    # with compute. The CaptureWriter has no per-step flush cadence (the audit is a
    # bounded handful of ticks); the run-end drain + fail-loud is invoked from the
    # engine teardown (ActorRolloutRefWorker.comm_eff_close -> CaptureWriter.close)
    # with a finite timeout, backstopped best-effort by a bounded atexit handler.
    # See CommEffWeightTrajConfig for the field semantics.
    r2_async: bool = False
    r2_upload_workers: int = 4
    r2_max_staged_gb: float = 80.0
    # ASYNC only: finite timeout (s) for the flush barrier (and run-end close drain)
    # so a slow/hung uploader cannot block forever on queue.join(). <=0 => wait
    # forever (the original unbounded behaviour).
    r2_flush_timeout_s: float = 1800.0


@dataclass
class CommEffWeightTrajConfig(BaseConfig):
    """Full-weight trajectory recorder; off by default.

    A dump-only recorder (``verl.workers.comm_eff.capture.WeightTrajObserver``)
    that saves the model's FULL weight matrices to disk so any offline analysis
    can run on the real weights. At every optimizer tick the engine summons ALL
    floating-point params (the whole model — no subset, no sketch) and hands them
    to the observer, which writes ``full/<snapshot>.pt`` (a ``torch.save`` state
    dict of the actual tensors) + a ``full_manifest.jsonl`` row. The snapshot
    cadence is set by ``per_tick``: ``true`` dumps EVERY optimizer tick
    (``full/tick_<tick>.pt``; e.g. 160 snapshots for batch128/mini64 × 80 steps),
    ``false`` (default) dumps once per training step (``full/step_<gs>.pt``,
    deduped on ``global_step``, gated by ``every_steps``). Either way each row
    records both ``global_step`` and ``tick`` so the per-tick trajectory can be
    subsampled to the per-step one offline. There is NO compression — the tensors
    saved ARE the weights (cast to ``dump_dtype``). Telemetry-only: it reads the
    live weights and feeds NOTHING into the optimizer / EMA / Q.

    The heavy ``.pt`` files are large (a bf16 full-model snapshot is ~3 GB on
    Qwen2.5-1.5B; a per-tick bf16 trajectory ~492 GB). Set ``r2_enabled=true`` to
    upload each snapshot to Cloudflare R2 (bucket ``shamane-pluralis``, creds from
    the env) and delete the local ``.pt`` after a verified upload, so local disk is
    only a staging area (see ``verl.workers.comm_eff.r2_sink``).

    **Independent of ``comm_eff.enabled``.** Built whenever ``enabled=true`` even
    on the plain-GRPO (codec OFF) regime, so the clean-trajectory baseline is
    instrumented. ``enabled=false`` (default) ⇒ no observer, no summon, no I/O:
    the train path is byte-identical (off-path-parity invariant).

    Args:
        enabled (bool): Master switch. ``false`` (default) = strict no-op.
        out_dir (str): Directory for ``full/step_*.pt`` + ``full_manifest.jsonl``.
            Empty ⇒ ``./weights`` at build (launcher pins an absolute run-dir path).
        dump_dtype (str): Full-weight storage precision, ``"bf16"`` (default) or
            ``"fp32"``. bf16 halves disk (~3 GB/step vs ~6 GB/step on Qwen2.5-1.5B);
            fp32 keeps full precision (needed only if the downstream analysis
            differences consecutive steps, where the ~1e-3 per-step update would be
            swamped by bf16's ~4e-3 rounding). Validated to {bf16, fp32}.
        per_tick (bool): Snapshot cadence. ``false`` (default) = one dump per
            training step (deduped on ``global_step``, gated by ``every_steps``).
            ``true`` = dump EVERY optimizer tick (no dedup, ``every_steps`` ignored);
            for batch128/mini64 this is 2 ticks/step ≈ 160 snapshots over 80 steps.
            The per-tick set is a superset of the per-step one (subsample the first
            tick of each ``global_step`` to recover the 80-point trajectory).
        every_steps (int): Per-STEP-mode only. Dump the full weights every N
            training steps. ``1`` (default) = every step. Ignored when ``per_tick``.
            ``>= 1``.
        rank0_only (bool): Dump/write on DP rank 0 only (default; the summoned full
            params are DP-identical). Other ranks build an inactive observer so the
            summon collective stays symmetric.
        r2_enabled (bool): Upload each snapshot to Cloudflare R2 (bucket
            ``shamane-pluralis``, creds from the env) then delete the local ``.pt``
            after a verified upload. ``false`` (default) = keep ``.pt`` files local.
        r2_delete_local (bool): When ``r2_enabled``, delete the local ``.pt`` after a
            VERIFIED upload (``true``, default). ``false`` keeps a local copy too.
        r2_async (bool): Opt-in ASYNC upload mode. ``false`` (default) = synchronous
            (the dump path blocks on each cp -> verify -> manifest -> delete-local,
            byte-identical to before). ``true`` = the observer ENQUEUES each
            snapshot to a background worker pool and returns immediately; the
            observer flushes every ``r2_flush_every_steps`` steps and at run end so
            disk stays bounded and the manifest is checkpointed. Only meaningful when
            ``r2_enabled``.
        r2_flush_every_steps (int): ASYNC mode only. The observer calls
            ``sink.flush()`` (a barrier that drains the queue + fails loud on any
            upload error) whenever ``global_step % r2_flush_every_steps == 0`` (and
            always at observer close). ``10`` (default). Must be ``>= 1``.
        r2_upload_workers (int): ASYNC mode only. Number of background upload worker
            threads (parallel ``aws s3 cp`` streams). ``4`` (default). Must be
            ``>= 1``. More streams approach the aggregate R2 bandwidth ceiling.
        r2_max_staged_gb (float): ASYNC mode only. Disk-backpressure cap, in GiB, on
            the staged (queued + in-flight) bytes. ``upload()`` BLOCKS the producer
            once staged bytes exceed this, so local ``full/`` never overflows the
            box disk even if uploads fall behind compute. ``80`` (default). Must be
            ``> 0``.
    """

    enabled: bool = False
    out_dir: str = ""
    dump_dtype: str = "bf16"
    # Always dump ALL floating-point params (the whole model). There is no subset
    # toggle: the deliverable is the raw full weights.
    per_tick: bool = False
    every_steps: int = 1
    rank0_only: bool = True
    # Cloudflare R2 offload (heavy .pt files): upload-then-delete-local. Creds from
    # the env (R2_BUCKET=shamane-pluralis guard). Off => local-only (byte-identical).
    r2_enabled: bool = False
    r2_delete_local: bool = True
    # Async batched upload (opt-in). Off => synchronous, byte-identical. On => a
    # background worker pool overlaps uploads with compute; the observer flushes
    # every r2_flush_every_steps + at run end so disk stays bounded.
    r2_async: bool = False
    r2_flush_every_steps: int = 10
    r2_upload_workers: int = 4
    r2_max_staged_gb: float = 80.0
    # ASYNC only: finite timeout (s) for the per-step flush barrier (and run-end
    # close drain) so a slow/hung uploader cannot block the optimizer step forever
    # on queue.join(). <=0 => wait forever (the original unbounded behaviour).
    r2_flush_timeout_s: float = 1800.0


@dataclass
class CommEffProbeConfig(BaseConfig):
    """Geometry probe for paired anchor replay; off by default.

    **Telemetry-only by contract.** When ``geometry_enabled=true`` the harness
    measures, at every anchor fire on the paired-replay path, the
    m1-m7 geometry of the generator-consistent anchor gradient ``G_anc_rep``
    against (a) a current-batch stale-weight anchor gradient ``G_anc_old``
    (a second telemetry backward on the SAME stale-loaded clone with the CURRENT
    batch), (b) the live fast compressed gradient ``G_comp(t)``, (c) the
    fire-aware ring entry ``G_comp_ring(t−K)`` (m5 codec error), (d) its own lag
    history (m4), and (e) the previous fire's ``M_rep`` (m6 persistence). One
    JSON line per fire is appended to ``<out_dir>/stepA_fires.jsonl`` with stable
    field names, plus a ``[geometry-probe]`` train-log line.
    Nothing is ever fed to the optimizer, the EMA, the sketch V, or Q
    (``anchor_grad_corrected`` stays 0 — the probe invariant).

    Hard config invariants (validated in ``CommEffConfig.__post_init__``):
    requires ``anchor.enabled`` + ``anchor.replay_paired_batch`` (G_anc_rep IS
    the replay gradient), and the merger must be inert
    (``spectral.correction_mode="none"`` when spectral is enabled) so the
    measured ``G_comp`` is the raw codec output, never a merged gradient.

    Args:
        geometry_enabled (bool): Master probe switch. ``false`` (default) is a
            strict no-op: no ring, no lag buffer, no extra backward, no I/O.
        out_dir (str): Directory for ``stepA_fires.jsonl`` (+ the per-target
            sidecar). Empty (default) resolves to ``./geometry_probe`` at build.
        rank0_only (bool): Stage/compute/write on DP rank 0 only (default). The
            inputs are DP-identical by construction (the anchor grads are
            all-reduced(MEAN); the FSDP-summoned fast grads are the DP-mean), so
            rank 0 suffices; other ranks still run the symmetric collectives
            (the dual backward + DP-reduce) but skip storage and I/O. ``false``
            writes rank-suffixed files (debug only).
        m4_lags (int): Lag depth j=1..m4_lags for the m4 autocorrelation.
            Default 5; bounded to [1, 5] so the CPU lag buffer stays small
            (max_lag stored + the in-flight current).
        per_target_sidecar (bool): Also append the per-target scalar map (the
            196-matrix arrays behind each median) to
            ``<out_dir>/stepA_fires_targets.jsonl``. Scalars only — a few tens
            of KB per fire, not a tensor dump. Default true.
    """

    geometry_enabled: bool = False
    out_dir: str = ""
    rank0_only: bool = True
    m4_lags: int = 5
    per_target_sidecar: bool = True
    # Weight-trajectory FULL-weight recorder (dump-only). Off by default; no
    # observer/summon/IO is built unless weight_traj.enabled. Independent of the
    # geometry probe and of comm_eff.enabled.
    weight_traj: CommEffWeightTrajConfig = field(default_factory=CommEffWeightTrajConfig)


@dataclass
class CommEffConfig(BaseConfig):
    """Top-level config for the communication-efficient compression method.

    The inheritance from BaseConfig provides an omegaconf.DictConfig-like
    interface for a dataclass config and, because it is a structured config,
    OmegaConf rejects unknown ``comm_eff.*`` keys at merge time (typos in the
    mask/anchor/spectral namespaces fail fast instead of being silently
    ignored).

    **Disabled is a strict no-op.** When ``enabled=false`` the integration
    hooks short-circuit before importing any comm_eff machinery: no forward
    hooks are registered, no SVD/EMA buffers are allocated, no extra
    all-reduce is issued, and crucially **no RNG is drawn**. Constructing this
    config (which every actor does, since it defaults disabled) must therefore
    have zero numerical side effects on the dense GRPO path.

    Args:
        enabled (bool): Master switch. ``false`` (default) makes every comm_eff
            hook a no-op. Must be set ``true`` explicitly to activate any
            circuit.
        compression_type (str): Codec selector, one of
            ``{dense, prf_mask, powersgd}`` (mutually exclusive per run).
            ``dense`` (default) = no activation compression. ``prf_mask`` = the
            PRF mask. ``powersgd`` = the shared frozen-basis projector.
            For back-compat the ``mask.enabled`` path still selects the
            mask when ``compression_type`` is left at its ``dense`` default, so
            older mask configs behave unchanged; an explicit
            ``compression_type=powersgd`` selects the PowerSGD codec.
        mask (CommEffMaskConfig): Pipeline activation-masking sub-config.
        anchor (CommEffAnchorConfig): Asynchronous anchor-circuit sub-config.
        spectral (CommEffSpectralConfig): Spectral-correction sub-config.
        powersgd (CommEffPowerSGDConfig): PowerSGD activation-compression
            sub-config.
        clean_cadence (int): Periodic clean (unmasked) optimizer-step cadence,
            in trainer steps. ``0`` (default) = off. When
            ``> 0``, the trainer step ``s`` runs **entirely unmasked** whenever
            ``(s % clean_cadence) == 0`` — both gradient-feeding forwards (the
            ``compute_log_prob`` old-logprob recompute AND the actor-train
            forward) have masking forced OFF for that step, and the normal
            single ``optimizer.step()`` runs on the true dense gradient so
            AdamW's ``m``/``v`` are periodically refreshed (the design's own
            "naive synchronous fix"). It does NOT add a second optimizer step:
            the clean step REPLACES the masked step at that index. The other
            ``clean_cadence - 1`` of every ``clean_cadence`` steps stay masked
            exactly as today (≈90% of the pipeline-boundary savings retained at
            ``clean_cadence=10``). Gated by the parent ``comm_eff.enabled``; on
            the disabled path the threading short-circuits on the ``None`` state
            so there is no numerical side effect. No anchor/spectral machinery is
            touched — the clean step rides the existing dense optimizer path.
    """

    enabled: bool = False
    # Codec selector. Exactly one boundary codec is active per run:
    #   "dense"    -> no activation compression (the boundary forward is
    #                 equivalent to the uncompressed activation path of
    #                 enabled=false). A mask-circuit run is selected
    #                 by mask.enabled, NOT by this field, for back-compat.
    #   "prf_mask" -> the per-(token, dim) PRF Bernoulli mask.
    #   "powersgd" -> the shared frozen-basis projector M_hat=(M@Q)@Qᵀ.
    # Registered with the powersgd block regardless of value so a prf_mask run
    # that passes comm_eff.powersgd.rank=... still parses.
    compression_type: str = "dense"
    mask: CommEffMaskConfig = field(default_factory=CommEffMaskConfig)
    anchor: CommEffAnchorConfig = field(default_factory=CommEffAnchorConfig)
    spectral: CommEffSpectralConfig = field(default_factory=CommEffSpectralConfig)
    powersgd: CommEffPowerSGDConfig = field(default_factory=CommEffPowerSGDConfig)
    # Diagnostic tensor capture. Off by default, with no numerical side effect.
    capture: CommEffCaptureConfig = field(default_factory=CommEffCaptureConfig)
    # Geometry probe. Off by default; no ring/buffer/backward/I-O is built.
    probe: CommEffProbeConfig = field(default_factory=CommEffProbeConfig)
    clean_cadence: int = 0

    def __post_init__(self):
        """Validate comm_eff configuration parameters.

        Validation only — no allocation, no RNG. When ``enabled=false`` this
        must remain free of numerical side effects.
        """
        if not 0.0 <= self.mask.p <= 1.0:
            raise ValueError(f"comm_eff.mask.p must be in [0, 1]; got {self.mask.p}")
        if self.mask.pp_size < 1:
            raise ValueError(f"comm_eff.mask.pp_size must be >= 1; got {self.mask.pp_size}")
        # mask_recompute is a strict bool. Validation here turns a YAML
        # typo ("False" string) or a numeric override into a clear error instead
        # of a silent truthy/falsy surprise that would mis-route masking.
        if not isinstance(self.mask.mask_recompute, bool):
            raise ValueError(
                f"comm_eff.mask.mask_recompute must be a bool; got {type(self.mask.mask_recompute).__name__} "
                f"({self.mask.mask_recompute!r})"
            )
        # rescale is a strict bool (same rationale as the other mask flags). It
        # also requires p < 1 so the 1/(1-p) factor is finite; p==1 with rescale
        # would divide by zero (and p==1 masks everything anyway).
        if not isinstance(self.mask.rescale, bool):
            raise ValueError(
                f"comm_eff.mask.rescale must be a bool; got {type(self.mask.rescale).__name__} ({self.mask.rescale!r})"
            )
        if self.mask.rescale and self.mask.p >= 1.0:
            raise ValueError(
                "comm_eff.mask.rescale=true requires comm_eff.mask.p < 1.0 (the 1/(1-p) "
                f"magnitude-preservation factor is undefined at p>=1); got p={self.mask.p}"
            )
        if not 0.0 <= self.spectral.beta_anc <= 1.0:
            raise ValueError(f"comm_eff.spectral.beta_anc must be in [0, 1]; got {self.spectral.beta_anc}")
        # Spectral-correction cadence. 1 = fire every step. A value < 1 is a
        # config error, not a silent disable; mirrors anchor.cadence >= 1.
        if self.spectral.cadence < 1:
            raise ValueError(f"comm_eff.spectral.cadence must be >= 1; got {self.spectral.cadence}")
        # diagnostics is a strict bool (same rationale as the other comm_eff
        # flags). It gates only DIAGNOSTIC overhead; a YAML typo ("False" string)
        # or a numeric override should fail loudly instead of silently leaving
        # the per-step rel_change syncs / relevance probe on or off.
        if not isinstance(self.spectral.diagnostics, bool):
            raise ValueError(
                f"comm_eff.spectral.diagnostics must be a bool; got "
                f"{type(self.spectral.diagnostics).__name__} ({self.spectral.diagnostics!r})"
            )
        # Anchor cadence/staleness.
        if self.anchor.cadence < 1:
            raise ValueError(f"comm_eff.anchor.cadence must be >= 1; got {self.anchor.cadence}")
        if self.anchor.delay_K < 0:
            raise ValueError(f"comm_eff.anchor.delay_K must be >= 0; got {self.anchor.delay_K}")
        # Paired replay flag is a strict bool; YAML strings and numeric overrides
        # should fail instead of routing the replay ring by accident.
        if not isinstance(self.anchor.replay_paired_batch, bool):
            raise ValueError(
                f"comm_eff.anchor.replay_paired_batch must be a bool; got "
                f"{type(self.anchor.replay_paired_batch).__name__} ({self.anchor.replay_paired_batch!r})"
            )
        # Snapshot storage enum (mirrors spectral.ema_device). "gpu" keeps the
        # snapshots on device; "cpu" moves the delay_K+1 full snapshots off HBM.
        if self.anchor.snapshot_device not in ("gpu", "cpu"):
            raise ValueError(
                f"comm_eff.anchor.snapshot_device must be one of (gpu, cpu); got {self.anchor.snapshot_device!r}"
            )
        # Look-ahead (weight-projection) anchor knobs. The mode / rollout-source
        # whitelists live in verl.workers.comm_eff.lookahead (single source of
        # truth); imported lazily here — validation only, no allocation, no RNG,
        # so the enabled=false path keeps zero numerical side effects.
        from verl.workers.comm_eff.lookahead import (
            LOOKAHEAD_MODES,
            LOOKAHEAD_ROLLOUT_SOURCES,
            lookahead_enabled,
            lookahead_num_source_points,
        )

        if not isinstance(self.anchor.lookahead_anchor, bool):
            raise ValueError(
                f"comm_eff.anchor.lookahead_anchor must be a bool; got "
                f"{type(self.anchor.lookahead_anchor).__name__} ({self.anchor.lookahead_anchor!r})"
            )
        if self.anchor.lookahead_mode not in LOOKAHEAD_MODES:
            raise ValueError(
                f"comm_eff.anchor.lookahead_mode must be one of {LOOKAHEAD_MODES}; "
                f"got {self.anchor.lookahead_mode!r}"
            )
        if not float(self.anchor.lookahead_strength) >= 0.0:
            raise ValueError(
                f"comm_eff.anchor.lookahead_strength must be >= 0 (0 = raw stale weights, "
                f"1 = full catch-up to the current tick); got {self.anchor.lookahead_strength}"
            )
        if self.anchor.lookahead_rollout_source not in LOOKAHEAD_ROLLOUT_SOURCES:
            raise ValueError(
                f"comm_eff.anchor.lookahead_rollout_source must be one of "
                f"{LOOKAHEAD_ROLLOUT_SOURCES}; got {self.anchor.lookahead_rollout_source!r}"
            )
        if self.anchor.lookahead_rollout_source == "self_generate":
            raise ValueError(
                "comm_eff.anchor.lookahead_rollout_source='self_generate' is a RESERVED "
                "seam (the anchor generating its own rollouts) and is NOT implemented. "
                "Use 'auto', 'stale_paired', or 'current_step'."
            )
        if self.anchor.lookahead_rollout_source == "current_step" and not lookahead_enabled(self.anchor):
            raise ValueError(
                "comm_eff.anchor.lookahead_rollout_source='current_step' requires the "
                "look-ahead projector ON (lookahead_anchor=true AND lookahead_mode != "
                "'disabled'): stale weights + fresh rollouts is an unsupported ablation. "
                "Leave it 'auto' to get current_step automatically whenever the projector is on."
            )
        # --- warmup behavior knobs (E2/E3): what the anchor does at fires
        # BEFORE the look-ahead projector is ready, and how early it engages.
        # All additive — the defaults (stale_correct, -1) reproduce today's
        # behavior byte-identically.
        if self.anchor.warmup_mode not in ("stale_correct", "no_correct"):
            raise ValueError(
                f"comm_eff.anchor.warmup_mode must be one of (stale_correct, no_correct); "
                f"got {self.anchor.warmup_mode!r}"
            )
        if self.anchor.warmup_mode == "no_correct":
            # no_correct SKIPS the anchor pass while warming, so M is never set
            # during the wait. That is only coherent when there IS a projector to
            # later set M (else M would never exist — a different ablation:
            # "anchor disabled").
            if not lookahead_enabled(self.anchor):
                raise ValueError(
                    "comm_eff.anchor.warmup_mode='no_correct' requires the look-ahead projector ON "
                    "(lookahead_anchor=true AND lookahead_mode != 'disabled'): the skipped warmup "
                    "leaves M cold, so M is only ever set by the FIRST projected fire. With the "
                    "projector off, M would never exist — that is 'anchor disabled', a different "
                    "ablation. Enable the projector or use warmup_mode='stale_correct'."
                )
            # Skipping the anchor pass means it cannot be the Q updater during the
            # warmup window; owns_q=true would then leave Q at its cold random
            # bootstrap for the whole wait (the step-1..9 blowup A0 suffered).
            if self.anchor.owns_q:
                raise ValueError(
                    "comm_eff.anchor.warmup_mode='no_correct' requires anchor.owns_q=false. With "
                    "owns_q=true the anchor is the ONLY Q updater, but no_correct SKIPS the anchor "
                    "pass during warmup, so Q would stay frozen at its random bootstrap basis for "
                    "the entire wait (the cold-Q blowup). Set anchor.owns_q=false so the FAST net "
                    "owns and refreshes Q (E1)."
                )
        # lookahead_min_snapshots: -1 (mode default) or a concrete count in
        # [2, mode_n_points]. Any non-(-1) value requires the projector on.
        if self.anchor.lookahead_min_snapshots != -1:
            if not lookahead_enabled(self.anchor):
                raise ValueError(
                    "comm_eff.anchor.lookahead_min_snapshots is only meaningful with the look-ahead "
                    "projector ON (lookahead_anchor=true AND lookahead_mode != 'disabled'); got "
                    f"{self.anchor.lookahead_min_snapshots} with the projector off. Leave it -1."
                )
            _n_points = lookahead_num_source_points(self.anchor)
            if not (2 <= self.anchor.lookahead_min_snapshots <= _n_points):
                raise ValueError(
                    f"comm_eff.anchor.lookahead_min_snapshots must be -1 (mode default) or in "
                    f"[2, {_n_points}] for lookahead_mode={self.anchor.lookahead_mode!r}; got "
                    f"{self.anchor.lookahead_min_snapshots}. (Fire 1 can NEVER project — a line "
                    f"needs 2 points — so 2 is the earliest legal value.)"
                )
        # Storage-layer enum.
        if self.spectral.ema_device not in ("gpu", "cpu"):
            raise ValueError(
                f"comm_eff.spectral.ema_device must be one of (gpu, cpu); got {self.spectral.ema_device!r}"
            )
        valid_modes = ("none", "inject", "blend", "signed_ema", "ef_powersgd", "delayed_ef")
        if self.spectral.correction_mode not in valid_modes:
            raise ValueError(
                f"comm_eff.spectral.correction_mode must be one of "
                f"(none, inject, blend, signed_ema, ef_powersgd, delayed_ef); "
                f"got {self.spectral.correction_mode!r}"
            )
        if self.spectral.inject_gamma < 0.0:
            raise ValueError(f"comm_eff.spectral.inject_gamma must be >= 0; got {self.spectral.inject_gamma}")
        # Convex-blend weight. [0, 1]: 0 => pure G_mask, 1 =>
        # scale-matched stale true gradient. Unused unless correction_mode=blend.
        if not 0.0 <= self.spectral.blend_eta <= 1.0:
            raise ValueError(f"comm_eff.spectral.blend_eta must be in [0, 1]; got {self.spectral.blend_eta}")
        # signed_ema merger weight. [0, 1]: 0 => pure |G|*sign(M), 1 =>
        # G_noisy unchanged. Unused unless correction_mode=signed_ema.
        if not 0.0 <= self.spectral.signed_ema_alpha <= 1.0:
            raise ValueError(
                f"comm_eff.spectral.signed_ema_alpha must be in [0, 1]; got {self.spectral.signed_ema_alpha}"
            )
        # Error-feedback residual knobs. decay in [0, 1); clip >= 0. Both are
        # validated unconditionally so typos fail even when the mode is inactive.
        if not 0.0 <= self.spectral.ef_decay < 1.0:
            raise ValueError(f"comm_eff.spectral.ef_decay must be in [0, 1); got {self.spectral.ef_decay}")
        if self.spectral.ef_clip < 0.0:
            raise ValueError(f"comm_eff.spectral.ef_clip must be >= 0; got {self.spectral.ef_clip}")
        # delayed_ef residual weight. >= 0; 0.0 is the identity limiting case.
        # Validated unconditionally so typos fail even when the mode is inactive.
        if self.spectral.delayed_ef_lambda < 0.0:
            raise ValueError(f"comm_eff.spectral.delayed_ef_lambda must be >= 0; got {self.spectral.delayed_ef_lambda}")
        # Additive stale-anchor sub-basis rank. >= 0; 0 disables the branch.
        # Validated unconditionally so typos fail even when the mode is inactive.
        if self.spectral.delta_subbasis_rank < 0:
            raise ValueError(
                f"comm_eff.spectral.delta_subbasis_rank must be >= 0; got {self.spectral.delta_subbasis_rank}"
            )
        if self.spectral.delta_subbasis_family not in ("tail", "grad"):
            raise ValueError(
                "comm_eff.spectral.delta_subbasis_family must be one of (tail, grad); "
                f"got {self.spectral.delta_subbasis_family!r}"
            )
        # Sub-basis weight and decay horizon. Both are validated unconditionally
        # so typos fail even when the branch is inactive.
        if self.spectral.delta_subbasis_weight < 0.0:
            raise ValueError(
                f"comm_eff.spectral.delta_subbasis_weight must be >= 0; got {self.spectral.delta_subbasis_weight}"
            )
        if self.spectral.delta_subbasis_decay_steps < 0:
            raise ValueError(
                "comm_eff.spectral.delta_subbasis_decay_steps must be >= 0; "
                f"got {self.spectral.delta_subbasis_decay_steps}"
            )
        # Hold horizon before the sub-basis decay begins. Only meaningful when
        # decay_steps > 0.
        if self.spectral.delta_subbasis_hold_steps < 0:
            raise ValueError(
                "comm_eff.spectral.delta_subbasis_hold_steps must be >= 0; "
                f"got {self.spectral.delta_subbasis_hold_steps}"
            )
        # Correction-delta compression rank. >= 0; 0 keeps the correction
        # uncompressed.
        if self.spectral.r_delta < 0:
            raise ValueError(f"comm_eff.spectral.r_delta must be >= 0; got {self.spectral.r_delta}")
        # Zero-mean perturbation magnitude and cross-rank-identical RNG seed.
        # Validated unconditionally so typos fail even when the branch is inactive.
        if self.spectral.perturb_sigma < 0.0:
            raise ValueError(f"comm_eff.spectral.perturb_sigma must be >= 0; got {self.spectral.perturb_sigma}")
        # Delta-momentum. delta_momentum_mu is the normalized-EMA decay in [0, 1).
        # delta_momentum_age_decay is a strict bool. Both are validated
        # unconditionally so typos fail even when the branch is inactive.
        if not 0.0 <= self.spectral.delta_momentum_mu < 1.0:
            raise ValueError(
                f"comm_eff.spectral.delta_momentum_mu must be in [0, 1); got {self.spectral.delta_momentum_mu}"
            )
        if not isinstance(self.spectral.delta_momentum_age_decay, bool):
            raise ValueError(
                f"comm_eff.spectral.delta_momentum_age_decay must be a bool; got "
                f"{type(self.spectral.delta_momentum_age_decay).__name__} "
                f"({self.spectral.delta_momentum_age_decay!r})"
            )
        # Adaptive dose. adaptive_lambda_mode is the closed enum {off, cos, ratio};
        # adaptive_lambda_kappa is the non-negative gate gain; lambda_cap is the
        # non-negative upper bound. Validated unconditionally.
        if self.spectral.adaptive_lambda_mode not in ("off", "cos", "ratio"):
            raise ValueError(
                "comm_eff.spectral.adaptive_lambda_mode must be one of (off, cos, ratio); "
                f"got {self.spectral.adaptive_lambda_mode!r}"
            )
        if self.spectral.adaptive_lambda_kappa < 0.0:
            raise ValueError(
                f"comm_eff.spectral.adaptive_lambda_kappa must be >= 0; got {self.spectral.adaptive_lambda_kappa}"
            )
        if self.spectral.lambda_cap < 0.0:
            raise ValueError(f"comm_eff.spectral.lambda_cap must be >= 0; got {self.spectral.lambda_cap}")
        # delayed_ef assumes a generator-consistent anchor feed. Fail at
        # config time instead of silently mixing unmatched batches and weights.
        if self.spectral.enabled and self.spectral.correction_mode == "delayed_ef":
            if not (self.anchor.enabled and self.anchor.replay_paired_batch):
                raise ValueError(
                    "comm_eff.spectral.correction_mode=delayed_ef requires a "
                    "generator-consistent anchor feed: comm_eff.anchor.enabled=true AND "
                    "comm_eff.anchor.replay_paired_batch=true; "
                    f"got enabled={self.anchor.enabled}, replay_paired_batch={self.anchor.replay_paired_batch})."
                )
        # Geometry-probe knobs. Bool flags must be strict bools; m4_lags is
        # bounded so the lag buffer remains small.
        for _bname in ("geometry_enabled", "rank0_only", "per_target_sidecar"):
            _bval = getattr(self.probe, _bname)
            if not isinstance(_bval, bool):
                raise ValueError(f"comm_eff.probe.{_bname} must be a bool; got {type(_bval).__name__} ({_bval!r})")
        if not 1 <= self.probe.m4_lags <= 5:
            raise ValueError(
                f"comm_eff.probe.m4_lags must be in [1, 5] (lag buffer <=6-entry bound); got {self.probe.m4_lags}"
            )
        if self.probe.geometry_enabled:
            # The probe measures paired replay. Without replay there is no
            # generator-consistent anchor gradient to measure.
            if not (self.anchor.enabled and self.anchor.replay_paired_batch):
                raise ValueError(
                    "comm_eff.probe.geometry_enabled=true requires comm_eff.anchor.enabled=true "
                    "AND comm_eff.anchor.replay_paired_batch=true (G_anc_rep is the "
                    "paired-replay gradient; without replay the probe would measure the "
                    f"generator-mismatched feed). Got enabled={self.anchor.enabled}, "
                    f"replay_paired_batch={self.anchor.replay_paired_batch}."
                )
            # The probe's G_comp must be the raw codec output; an active merger
            # would rewrite the live grads before end-of-batch extraction.
            if self.spectral.enabled and self.spectral.correction_mode != "none":
                raise ValueError(
                    "comm_eff.probe.geometry_enabled=true requires an INERT merger: set "
                    "comm_eff.spectral.correction_mode=none (the probe measures raw G_comp; "
                    f"an active merger would corrupt it). Got correction_mode="
                    f"{self.spectral.correction_mode!r}."
                )
        # Weight-trajectory instrument. Validated unconditionally so a
        # typo fails fast even when the instrument is off. It is dump-only and
        # carries NO cross-config dependency (independent of comm_eff.enabled,
        # the anchor, the merger and the geometry probe).
        wt = self.probe.weight_traj
        if not isinstance(wt.enabled, bool):
            raise ValueError(
                f"comm_eff.probe.weight_traj.enabled must be a bool; got {type(wt.enabled).__name__} ({wt.enabled!r})"
            )
        if not isinstance(wt.rank0_only, bool):
            raise ValueError(
                f"comm_eff.probe.weight_traj.rank0_only must be a bool; got "
                f"{type(wt.rank0_only).__name__} ({wt.rank0_only!r})"
            )
        if not isinstance(wt.per_tick, bool):
            raise ValueError(
                f"comm_eff.probe.weight_traj.per_tick must be a bool; got "
                f"{type(wt.per_tick).__name__} ({wt.per_tick!r})"
            )
        if not isinstance(wt.r2_enabled, bool):
            raise ValueError(
                f"comm_eff.probe.weight_traj.r2_enabled must be a bool; got "
                f"{type(wt.r2_enabled).__name__} ({wt.r2_enabled!r})"
            )
        if not isinstance(wt.r2_delete_local, bool):
            raise ValueError(
                f"comm_eff.probe.weight_traj.r2_delete_local must be a bool; got "
                f"{type(wt.r2_delete_local).__name__} ({wt.r2_delete_local!r})"
            )
        if wt.dump_dtype not in ("bf16", "fp32"):
            raise ValueError(
                f"comm_eff.probe.weight_traj.dump_dtype must be one of (bf16, fp32); got {wt.dump_dtype!r}"
            )
        if int(wt.every_steps) < 1:
            raise ValueError(f"comm_eff.probe.weight_traj.every_steps must be >= 1; got {wt.every_steps}")
        # Async-upload knobs. Validated unconditionally (registered regardless of
        # r2_async) so a typo fails fast even when async is off. r2_async is a strict
        # bool; the worker count and flush cadence are >= 1; the staged cap is > 0.
        if not isinstance(wt.r2_async, bool):
            raise ValueError(
                f"comm_eff.probe.weight_traj.r2_async must be a bool; got "
                f"{type(wt.r2_async).__name__} ({wt.r2_async!r})"
            )
        if int(wt.r2_flush_every_steps) < 1:
            raise ValueError(
                f"comm_eff.probe.weight_traj.r2_flush_every_steps must be >= 1; got {wt.r2_flush_every_steps}"
            )
        if int(wt.r2_upload_workers) < 1:
            raise ValueError(
                f"comm_eff.probe.weight_traj.r2_upload_workers must be >= 1; got {wt.r2_upload_workers}"
            )
        if float(wt.r2_max_staged_gb) <= 0.0:
            raise ValueError(
                f"comm_eff.probe.weight_traj.r2_max_staged_gb must be > 0; got {wt.r2_max_staged_gb}"
            )
        # Periodic clean-step cadence. 0 = off. A negative value is a config
        # error, not a silent disable.
        if self.clean_cadence < 0:
            raise ValueError(f"comm_eff.clean_cadence must be >= 0; got {self.clean_cadence}")
        # Codec selector. Validated to the closed enum so a typo
        # (compression_type=powerSGD / powergsd) is an error, not a silent
        # fall-through to dense.
        if self.compression_type not in COMPRESSION_TYPES:
            raise ValueError(
                f"comm_eff.compression_type must be one of {COMPRESSION_TYPES}; got {self.compression_type!r}"
            )
        # PowerSGD block. Validated unconditionally (the keys are
        # registered regardless of compression_type) so a prf_mask run that
        # forwards comm_eff.powersgd.* args still fails fast on a bad value.
        if self.powersgd.rank < 1:
            raise ValueError(f"comm_eff.powersgd.rank must be >= 1; got {self.powersgd.rank}")
        if self.powersgd.pp_size < 1:
            raise ValueError(f"comm_eff.powersgd.pp_size must be >= 1; got {self.powersgd.pp_size}")
        if self.powersgd.update_cadence < 1:
            raise ValueError(f"comm_eff.powersgd.update_cadence must be >= 1; got {self.powersgd.update_cadence}")
        if not isinstance(self.powersgd.warm_start, bool):
            raise ValueError(
                f"comm_eff.powersgd.warm_start must be a bool; got "
                f"{type(self.powersgd.warm_start).__name__} ({self.powersgd.warm_start!r})"
            )
        if not isinstance(self.powersgd.compress_recompute, bool):
            raise ValueError(
                f"comm_eff.powersgd.compress_recompute must be a bool; got "
                f"{type(self.powersgd.compress_recompute).__name__} ({self.powersgd.compress_recompute!r})"
            )
        if not isinstance(self.powersgd.sync_basis, bool):
            raise ValueError(
                f"comm_eff.powersgd.sync_basis must be a bool; got "
                f"{type(self.powersgd.sync_basis).__name__} ({self.powersgd.sync_basis!r})"
            )
        if self.powersgd.qr_dtype not in ("fp32", "bf16"):
            raise ValueError(f"comm_eff.powersgd.qr_dtype must be one of (fp32, bf16); got {self.powersgd.qr_dtype!r}")
        if self.powersgd.reortho_eps <= 0.0:
            raise ValueError(f"comm_eff.powersgd.reortho_eps must be > 0; got {self.powersgd.reortho_eps}")
        # FROZEN-Q footgun guard. With anchor.owns_q=true the anchor is the ONLY
        # Q updater and the fast-path basis update is fail-closed; if the anchor
        # is ALSO off, NOTHING ever updates Q, so the PowerSGD codec runs on its
        # fixed random bootstrap basis (basis_updates=0, reconstruction_rel_error
        # stuck ~0.97) and the run collapses. That is a fixed random projection,
        # not a learning compressed regime. Forbid it: either enable the anchor
        # (so it owns + adapts Q) or set anchor.owns_q=false (fast-owned adaptive
        # Q). (A frozen-basis codec-only regime hit exactly this and silently collapsed.)
        if (
            self.enabled
            and self.compression_type == "powersgd"
            and getattr(self.powersgd, "enabled", True)
            and self.anchor.owns_q
            and not self.anchor.enabled
        ):
            raise ValueError(
                "comm_eff: PowerSGD basis Q has no updater. anchor.owns_q=true (so the "
                "fast-path basis update is fail-closed) but anchor.enabled=false (so the "
                "anchor never refreshes Q). Q would stay frozen at its random bootstrap "
                "basis and the codec collapses. Set anchor.owns_q=false for a fast-owned "
                "adaptive Q, or enable the anchor."
            )
        # Q-basis family. Validated to the closed enum so a typo
        # (q_basis=gradient) is an error, not a silent fall-through to "act".
        if self.powersgd.q_basis not in Q_BASIS_FAMILIES:
            raise ValueError(
                f"comm_eff.powersgd.q_basis must be one of {Q_BASIS_FAMILIES}; got {self.powersgd.q_basis!r}"
            )
        # Passive screen family list. Every entry must be a
        # known family; a typo is an error (a silently-dropped family would make
        # the screen miss an arm). OmegaConf may pass a ListConfig — iterate it.
        for _fam in list(self.powersgd.q_basis_passive):
            if _fam not in Q_BASIS_FAMILIES:
                raise ValueError(
                    f"comm_eff.powersgd.q_basis_passive entries must each be one of {Q_BASIS_FAMILIES}; got {_fam!r}"
                )
        # Hybrid column split. Only meaningful when the hybrid
        # family is requested (live or passive); otherwise the (default -1/-1 AUTO)
        # values are inert. When BOTH are set EXPLICITLY (>= 0) they MUST sum to the
        # rank; the -1 sentinel means AUTO (resolved to ceil(r/2) + (r-act) by the
        # compressor) and is unconstrained. A single explicit value with the other
        # at -1 is a config error (ambiguous).
        _hybrid_used = (self.powersgd.q_basis == "hybrid") or ("hybrid" in list(self.powersgd.q_basis_passive))
        if _hybrid_used:
            _a, _g = self.powersgd.hybrid_act_cols, self.powersgd.hybrid_grad_cols
            if (_a < 0) != (_g < 0):
                raise ValueError(
                    "comm_eff.powersgd.hybrid_act_cols / hybrid_grad_cols must BOTH be -1 "
                    f"(AUTO) or BOTH be >= 0 (explicit); got {_a} / {_g}"
                )
            if _a >= 0 and _g >= 0 and _a + _g != self.powersgd.rank:
                raise ValueError(
                    "comm_eff.powersgd.hybrid_act_cols + hybrid_grad_cols must equal "
                    f"powersgd.rank ({self.powersgd.rank}) when set explicitly + the hybrid "
                    f"family is used; got {_a} + {_g} = {_a + _g}. (Use -1/-1 for AUTO.)"
                )
        # Diagnostic-capture block. Validated unconditionally (the
        # keys are registered regardless of capture.enabled) so a bad dump_dtype /
        # negative cap fails fast even on a non-capture run that forwards them.
        if self.capture.dump_dtype not in ("fp32", "bf16"):
            raise ValueError(
                f"comm_eff.capture.dump_dtype must be one of (fp32, bf16); got {self.capture.dump_dtype!r}"
            )
        if self.capture.fresh_anchor_loss not in ("clean_pg", "ppo_clip"):
            raise ValueError(
                "comm_eff.capture.fresh_anchor_loss must be one of (clean_pg, ppo_clip); "
                f"got {self.capture.fresh_anchor_loss!r}"
            )
        if self.capture.stratified_targets < 0:
            raise ValueError(f"comm_eff.capture.stratified_targets must be >= 0; got {self.capture.stratified_targets}")
        # Capture-path async-upload knobs (mirror the weight_traj ones). Validated
        # unconditionally so a typo fails fast even on a non-capture run.
        if not isinstance(self.capture.r2_async, bool):
            raise ValueError(
                f"comm_eff.capture.r2_async must be a bool; got "
                f"{type(self.capture.r2_async).__name__} ({self.capture.r2_async!r})"
            )
        if int(self.capture.r2_upload_workers) < 1:
            raise ValueError(f"comm_eff.capture.r2_upload_workers must be >= 1; got {self.capture.r2_upload_workers}")
        if float(self.capture.r2_max_staged_gb) <= 0.0:
            raise ValueError(f"comm_eff.capture.r2_max_staged_gb must be > 0; got {self.capture.r2_max_staged_gb}")
