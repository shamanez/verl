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
    "CommEffProbeConfig",
    "CommEffConfig",
]

# The Q-basis families PowerSGD may use (EXP-26 Step C). ``act`` is the existing
# activation-energy basis (block power iteration on the boundary activations);
# the others are RLVR-native candidates that bias Q toward GRPO UPDATE energy.
# All share the SAME fixed rank ``r`` (the byte budget is invariant); only the
# CONTENT of the sketch fed to ``orth(V)`` changes. ``act`` is the byte-identical
# default (EXP-25 substrate) so a run that does not opt into a family is unchanged.
Q_BASIS_FAMILIES = ("act", "grad", "adv", "tail", "hybrid", "ticket")

# The compression codecs ``comm_eff.compression_type`` may select. Exactly one
# codec is active per run (mutually exclusive). ``dense`` is the byte-identical
# off-path (equivalent to ``comm_eff.enabled=false`` for the activation path);
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
        owns_q (bool): EXP-25 (R2) structural inversion. When ``true`` the ANCHOR
            owns the PowerSGD projection basis ``Q``: (a) the fast net's
            ``maybe_update_basis`` + sketch accumulation are gated OFF (the fast
            net is a pure read-only consumer of ``Q``), and (b) the anchor
            computes ``Q ← orth(V)`` from its OWN slow-net stale-weight forward
            ACTIVATIONS (the same ``V += Aᵀ(AQ)`` block-power-iteration the fast
            path used, relocated fast→slow) and ``dist.broadcast``s both ``Q`` and
            the gradient-EMA ``M`` to every DP rank each refresh. ``false``
            (default) keeps the EXP-20 fast-owns-Q behaviour byte-identical
            (Prime Directive). Gated by ``anchor.enabled`` + the powersgd codec.
        replay_paired_batch (bool): EXP-29 on-policy replay. When ``true`` the
            anchor pairs its ``delay_K``-stale weights with the BATCH those same
            weights generated: every ``train_batch`` tick deep-clones the batch
            into a ring (CPU-resident), ONE generator-weight snapshot is taken
            per global step (at its first tick — exactly the weights vLLM
            generated that step's rollouts from), and at fire time the anchor
            replays ``(batch[t-delay_K], gen_snapshot)`` instead of (current
            batch, t-delay_K weights). ``false`` (default) keeps the legacy
            behaviour byte-identical: the anchor forwards the CURRENT tick's
            batch at the stale weights, and no replay ring is constructed.
        snapshot_device (str): Where the anchor weight snapshots live between
            ticks — ``"gpu"`` (default, faithful: detached clones stay on each
            param's device, today's exact behaviour) or ``"cpu"`` (memory-lean:
            the ``delay_K+1`` full bf16 snapshots move off HBM; the clone load
            casts back via ``.to(p.device, p.dtype)``, a byte-preserving round
            trip — numerics-neutral). Applies to BOTH the legacy staleness
            queue and the EXP-29 replay ring. Validated against {gpu, cpu}.
    """

    enabled: bool = False
    cadence: int = 20
    delay_K: int = 20
    owns_q: bool = False
    replay_paired_batch: bool = False
    snapshot_device: str = "gpu"


@dataclass
class CommEffSpectralConfig(BaseConfig):
    """Anchor-guided gradient-correction sub-config (inert while disabled).

    Applies an anchor combiner to selected 2D decoder gradients after the actor
    backward and before ``optimizer.step()``, using the anchor-gradient EMA
    ``M_anchor`` (no SVD / no basis). See
    ``verl.workers.comm_eff.spectral_filter.SpectralFilter``.

    The live correction is the signed-EMA merger (per targeted 2D matrix
    ``G_noisy`` with anchor-EMA ``M_anchor``)::

        M_anchor = beta_anc * M_anchor + (1 - beta_anc) * G_anchor    # EMA
        G_corr   = alpha * G_noisy + (1 - alpha) * |G_noisy| * sign(M_anchor)

    Magnitude from ``G_noisy``, sign from ``M_anchor``. ``inject`` and ``blend``
    are alternate anchor combiners. A cold-M guard returns ``G_noisy`` unchanged
    whenever ``M_anchor`` is unwarmed, so the masked gradient is never silently
    zeroed.

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
            step (keeps the discovery smoke cheap). ``-1`` ⇒ no cap (the EXP-25
            default — full coverage of ALL 196 = 28 layers × 7 projection
            matrices, the set the signed_ema merger corrects; ``max_targets``
            caps BOTH the anchor extraction AND the merger, so a residual cap
            silently drops merger targets). Set ``>= 0`` only as a diagnostic
            throttle, never in production.
        ema_device (str): Where the anchor-gradient EMA ``M_anchor`` is stored
            between refreshes — ``"gpu"`` (default, faithful: kept in HBM) or
            ``"cpu"`` (memory-lean: offloaded to pinned CPU, moved to GPU only
            inside the refresh/correct call and moved back). ``M_anchor`` is
            touched only at refresh, so CPU offload costs one H2D/D2H per refresh,
            not per mini-batch. Validated against {gpu, cpu}.
        correction_mode (str): The anchor combiner the fast-path grad uses.
            ``"none"`` (default) = no correction. The comm-eff SOTA is
            ``"delayed_ef"`` (B2: ``G_comp + lambda*(M_rep - G_comp_ring)``), set
            explicitly by the launcher. ``"inject"``/``"blend"`` are alternate
            combiners. Validated against
            {none, inject, blend, signed_ema, ef_powersgd, delayed_ef}.
        inject_gamma (float): Injection strength for
            ``correction_mode="inject"``; unused otherwise. Must be ``>= 0``.
        blend_eta (float): Convex-blend weight for ``correction_mode="blend"``;
            validated to ``[0, 1]``. Unused otherwise.
        signed_ema_alpha (float): The signed_ema merger weight ``alpha`` in
            ``G_corr = alpha*G_noisy + (1-alpha)*|G_noisy|*sign(M_anchor)``.
            ``alpha=0`` is the SL-validated pure sign-merger; ``alpha=1`` returns
            ``G_noisy`` unchanged. THE swept axis. Validated to ``[0, 1]``.
            Unused unless ``correction_mode=signed_ema``.
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
    """

    enabled: bool = False
    beta_anc: float = 0.95
    cadence: int = 1
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
    # EXP-25: full coverage by default (-1 = no cap). The signed_ema merger must
    # correct ALL 196 weight matrices (the activation compression corrupts weight
    # gradients throughout the network); a residual cap re-creates the EXP-23
    # coverage bug. Caps BOTH the anchor extraction and the merger (one knob).
    max_targets: int = -1
    # EMA storage default: keep tensors on GPU.
    ema_device: str = "gpu"
    # Correction mode (anchor combiner). "none" (default) = no correction; the
    # comm-eff SOTA is "delayed_ef" (B2: G_corr=G_comp+lambda*(M_rep-G_comp_ring)),
    # set explicitly by the launcher. "inject"/"blend" are alternate combiners;
    # "ef_powersgd" is the direction-preserving error-feedback merger; "signed_ema"
    # is the legacy sign-replacement merger (a still-supported but unused mode).
    correction_mode: str = "none"
    # Injection strength for correction_mode="inject"; unused otherwise.
    inject_gamma: float = 1.0
    # Convex-blend weight for correction_mode="blend"; validated to [0, 1].
    blend_eta: float = 0.5
    # EXP-25 (R3): the signed_ema merger weight alpha in
    # G_corr = alpha*G_noisy + (1-alpha)*|G_noisy|*sign(M_anchor). alpha=0 is the
    # SFT-validated pure sign-merger; alpha=1 returns G_noisy unchanged. THE swept
    # axis (id-2). Validated to [0, 1]. Unused unless correction_mode=signed_ema.
    signed_ema_alpha: float = 0.0
    # ---- EXP-26 Step B: error-feedback PowerSGD merger (direction-preserving) ----
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
    # The residual proxy uses the anchor EMA's OFF-subspace component (M_anchor
    # minus its projection onto the span of G_comp), which is exactly the
    # low-rank-compression bias the audit (Step A) measures. The merger reduces to
    # plain PowerSGD (G_corr == G_comp) at the LIMITING setting ef_decay=0 AND
    # ef_clip=0 (Correctness-invariant "EF residual limiting-case identity").
    ef_decay: float = 0.0
    # ef_clip: the residual norm cap as a FRACTION of ||G_comp|| (shape-aware,
    # per-matrix). 0.0 ⇒ the residual is fully zeroed ⇒ G_corr == G_comp (the
    # plain-PowerSGD limiting case). A typical live value is ~1.0 (the re-injected
    # residual may not exceed the compressed gradient's own norm). Must be >= 0.
    ef_clip: float = 0.0
    # ---- EXP-30 B2: K-delayed exact codec residual (correction_mode="delayed_ef") ----
    # G_corr(t) = G_comp(t) + delayed_ef_lambda * (M_rep − G_comp_ring(t−K)), where
    # M_rep is the β_anc=0 anchor EMA (= the latest fire's generator-consistent
    # G_anc_rep, EXP-29 paired replay) and G_comp_ring(t−K) is the fast compressed
    # gradient stored at the identical (batch, θ) tick by the fire-aware
    # FastGradRing. δ refreshes at anchor fires and is HELD between them (the
    # telescoping per-tick injection). 0.0 (default — the OFF/legacy posture) ⇒
    # G_corr == G_comp EXACTLY (the limiting-case-identity invariant); the B2
    # cell sets 1.0 explicitly. Must be >= 0. Unused unless
    # correction_mode=delayed_ef.
    delayed_ef_lambda: float = 0.0
    # ---- EXP-31 Cell D: additive stale-anchor rank-r_sb sub-basis (delayed_ef) ----
    # When delta_subbasis_rank (r_sb) > 0, the delayed_ef merger ADDS a rank-r_sb
    # low-rank reconstruction of the source S into its correction term:
    #
    #   δ_subbasis = rank_{r_sb}(S)                       # seeded randomized SVD
    #   G_corr(t)  = G_comp(t) + λ·(δ_B2 + δ_subbasis)    # forward Q UNCHANGED
    #
    # delta_subbasis_family selects S: "tail" (default) ⇒ S = δ_B2 (the
    # act-deflated stale weight gradient = the off-act-principal direction the
    # codec structurally drops); "grad" ⇒ S = M_rep (the raw stale anchor
    # gradient). The sub-basis enters ONLY the correction δ — the forward/recon
    # codec Q is untouched, so the EXP-26 Step-C dead route is avoided BY
    # CONSTRUCTION. r_sb = 0 (default, OFF) SKIPS the sub-basis branch entirely ⇒
    # G_corr == B2's G_corr bitwise (off-path-parity invariant). The randomized
    # SVD is per-target seeded so δ_subbasis is bit-identical across DP ranks
    # (δ_B2 / M_rep are already DP-mean identical). Must be >= 0. Unused unless
    # correction_mode=delayed_ef.
    delta_subbasis_rank: int = 0
    # The sub-basis source family: "tail" (act-deflated grad, the default) or
    # "grad" (raw stale anchor gradient, the REVISE fallback). Validated to
    # {tail, grad}.
    delta_subbasis_family: str = "tail"
    # ---- EXP-31 Cell D γ-knob: sub-basis WEIGHT + linear DECAY ----------------
    # The over-amplification fix. The Cell D r_sb=2 tail tail (γ=1, full weight)
    # BEAT B2 at step 25 (+0.036) but REGRESSED step 25→50 (−0.031): the constant
    # full-weight sub-basis accelerates early learning but OVER-AMPLIFIES near
    # convergence. ``delta_subbasis_weight`` (γ) scales the ADDITIVE δ_subbasis term
    # and ``delta_subbasis_decay_steps`` (D) linearly DECAYS it to 0 over training,
    # optionally AFTER a HOLD of ``delta_subbasis_hold_steps`` (H) steps at full
    # weight (see that field for the shelf-then-ramp schedule):
    #
    #   γ_t          = delta_subbasis_weight * decay_factor
    #   decay_factor = 1.0                              if D <= 0   (constant γ = weight)
    #                = 1.0                              if D > 0 and step < H (HOLD)
    #                = max(0, 1 - (step - H) / D)       else        (linear 1→0, clamped)
    #   G_corr(t)    = G_comp(t) + λ·(δ_B2 + γ_t·δ_subbasis)
    #
    # weight=1.0, decay_steps=0 (DEFAULTS) ⇒ γ_t == 1.0 always ⇒ EXACTLY the
    # current Cell D behaviour (no regression in the OFF/legacy posture). weight=0
    # ⇒ γ_t==0 ⇒ G_corr == B2's G_corr (δ_subbasis contributes nothing). The
    # decay knob is a SCALAR on the (already deterministic, DP-mean) δ_subbasis —
    # no new RNG, the seeded SVD is untouched. Both validated unconditionally so a
    # typo is loud even on a non-delayed_ef / rank-0 run that forwards them.
    delta_subbasis_weight: float = 1.0
    delta_subbasis_decay_steps: int = 0
    # EXP-31 hold-then-decay schedule: ``delta_subbasis_hold_steps`` (H) is the
    # number of steps γ HOLDS at full ``weight`` before the linear decay (over D
    # steps) begins. Targeted fix to preserve r2's early lead (the HOLD shelf) AND
    # finish clean (the decay ramp). The decay factor becomes a shelf-then-ramp:
    #
    #   decay_factor = 1.0                              if D <= 0          (constant)
    #                = 1.0                              if D > 0, step < H (HOLD shelf)
    #                = max(0, 1 - (step - H) / D)       otherwise          (linear ramp)
    #
    # hold_steps=0 (DEFAULT) ⇒ ``step < 0`` is never true ⇒ the shelf is empty ⇒
    # the formula reduces to the existing ``max(0, 1 - step/D)`` linear-from-0
    # decay EXACTLY (bitwise). hold_steps=25, decay_steps=25 ⇒ γ=weight for steps
    # 0..24 then linear weight→0 over steps 25..50. Only meaningful when D > 0.
    # A pure scalar — no RNG, the seeded SVD is untouched. Must be >= 0; validated
    # unconditionally so a typo is loud even on a non-delayed_ef / rank-0 run.
    delta_subbasis_hold_steps: int = 0
    # ---- EXP-31 Cell C: correction-δ compression rank (SECONDARY savings) ----
    # r_delta > 0 compresses the correction δ to r_delta columns BEFORE injection
    # (the Cell C residual-codec savings cell; forward/recon act-Q untouched).
    # 0 (default, OFF) = δ is injected uncompressed (the B2 / Cell D path). Wired
    # here for config completeness; the Cell C codec is a SEPARATE later change.
    # Must be >= 0. Unused unless correction_mode=delayed_ef.
    r_delta: int = 0
    # ---- EXP-31 surpass lever: zero-mean tunable gradient perturbation ----------
    # The destination-changing (vs path-speeding) lever. AFTER the delayed_ef
    # correction term (``g_corr = G_comp + λ·(δ_B2 + γ_t·δ_subbasis)``) a zero-mean,
    # σ-scaled, cross-rank-IDENTICAL noise is added to bias SGD toward FLATTER
    # minima (SGLD / SAM-style beneficial noise → a better-generalizing greedy
    # mode → potentially beats dense on greedy val):
    #
    #   ξ          = randn(g_corr.shape, gen=seed(perturb_seed, target, step))  # cross-rank-identical
    #   ξ          = ξ / ‖ξ‖                                                     # unit
    #   g_corr     = g_corr + perturb_sigma · ‖g_corr‖ · ξ                       # ‖perturbation‖ = σ·‖g_corr‖
    #
    # perturb_sigma = 0.0 (default, OFF) ⇒ the perturbation branch is SKIPPED
    # ENTIRELY ⇒ ``g_corr`` is the EXACT delayed_ef / Cell-D path bitwise
    # (off-path parity; composes with rank-0 ⇒ bitwise-B2). The seed is a pure
    # function of (perturb_seed, canonical-target-name, current_step) with NO
    # rank/device-local state, so every DP rank draws the SAME ξ (the
    # multi-rank-agreement invariant — else ranks diverge). Fresh per step ⇒
    # zero-mean over training ⇒ E[update] unchanged + exploration. Local, ZERO
    # added communication (the comm-eff substrate is untouched). σ relative to
    # ‖g_corr‖ ⇒ scale-free / tunable. Both must be >= 0 / int; validated
    # unconditionally so a typo is loud even on a non-delayed_ef / σ=0 run.
    perturb_sigma: float = 0.0
    perturb_seed: int = 0
    # ---- EXP-31 L2: δ-MOMENTUM (NORMALIZED EMA, stationary gain EXACTLY 1) -------
    # The "build up the corrections" lever. The codec drops the SAME kind of signal
    # each fire; a fading running EMA of the per-target correction δ turns the
    # persistently-missed direction into a strong steady push. The buffer is updated
    # ONLY at REFRESH ticks (δ is HELD between anchor fires in delayed_ef_matrix —
    # accumulating every tick would re-add the same δ ~5× = the forbidden constant-
    # λ>1 amplification dead-end):
    #
    #   REFRESH: m ← μ·m + (1−μ)·δ      (first fire: m = δ.clone());  correction = m
    #   HELD:    correction = m         (the held buffer; with age_decay below it is
    #                                    scaled by μ**(step − last_refresh_step) so a
    #                                    long hold fades the APPLIED correction → 0)
    #
    # NORMALIZED EMA — mandatory: the stationary gain of m←μ·m+(1−μ)·δ is EXACTLY 1
    # (constant δ ⇒ m→δ), a RE-WEIGHTING not a louder copy. The naive m←μ·m+δ has
    # stationary gain 1/(1−μ) (μ=0.9 ⇒ 10× dose) = the constant-λ>1 ignition dead-
    # end, FORBIDDEN. delta_momentum_mu = 0.0 (default, OFF) ⇒ the momentum branch is
    # SKIPPED ENTIRELY ⇒ correction == δ bitwise (off-path parity; composes with
    # rank-0 + σ=0 ⇒ bitwise-B2). delta_momentum_age_decay (default False) is the
    # async staleness-degrade: when fires arrive late/irregularly the held correction
    # fades by its actual age so the buffer decays to 0 (G_corr → G_comp) when fires
    # stop. The buffer is built from the DP-mean δ (cross-rank identical), lives on
    # the EMA storage device (CPU fp32 detached), shape-keyed reset. μ ∈ [0, 1);
    # age_decay strict bool. Validated unconditionally so a typo is loud even on a
    # non-delayed_ef / μ=0 run that forwards them.
    delta_momentum_mu: float = 0.0
    delta_momentum_age_decay: bool = False
    # ---- EXP-31 L3: ADAPTIVE-DOSE (MEAN-1 CENTERED gate) ------------------------
    # The "trust the slow node adaptively" lever. Some steps the cheap gradient
    # agrees with the trusted correction (compression did fine), others it disagrees
    # badly (compression dropped a lot). Lean MORE on the correction when they
    # disagree — but only via the step-to-step DEVIATION from the typical agreement,
    # never a constant offset. The constant λ in the final g_corr = G_comp + λ·δ is
    # replaced by a per-target, per-tick λ_t:
    #
    #   c_t = cos(G_comp, M_rep)  [mode=cos]   OR   ‖δ‖/‖G_comp‖  [mode=ratio]   # RAW δ
    #   c̄   = running MEDIAN of c_t over a bounded per-target history (last 64)
    #   λ_t = clamp(delayed_ef_lambda + κ·(c̄ − c_t), 0.0, lambda_cap)
    #   g_corr = G_comp + λ_t·correction
    #
    # MEAN-1 CENTERED — mandatory: E[λ_t] ≈ delayed_ef_lambda (=1) because c̄≈median(c_t),
    # so ONLY the deviation (c̄ − c_t) is the lever. The naive λ_t = 1 + κ(1 − cos) is
    # FORBIDDEN: this system has cos(G_comp,M)≈0, so it would pin at a constant 1+κ =
    # a disguised constant-λ>1 ignition dead-end. adaptive_lambda_mode = "off"
    # (default) OR adaptive_lambda_kappa = 0.0 ⇒ λ_t ≡ delayed_ef_lambda (constant) ⇒
    # bitwise B2. lambda_cap bounds the RAW λ so a stale/garbage M can't spike the
    # dose (variable-staleness safety; the ignition trip-wires are the behavioral
    # backstop). c_t / c̄ / λ_t are built from the DP-mean G_comp + M_rep (cross-rank
    # identical) — the per-target history deque carries NO rank-local state. mode ∈
    # {off, cos, ratio}; κ ≥ 0; lambda_cap ≥ 0. Validated unconditionally so a typo
    # is loud even on a non-delayed_ef / off run that forwards them.
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
            re-bootstraps ``Q`` from the per-layer seed every update (cold,
            diagnostic only).
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
            ``boundary_indices`` on every rank. ``False`` (diagnostic only) keeps
            each rank's basis local — only correct if every rank sees identical
            data, which DP does not.
        qr_dtype (str): Dtype for the orthonormalization (``orth``/QR) and the
            stored basis math — ``"fp32"`` (default, REQUIRED for correctness:
            bf16-QR loses orthogonality, drifts ``QᵀQ`` from ``I``, and is a
            frequent NaN / ``q_cond`` source) or ``"bf16"`` (diagnostic only,
            expected to degrade). The projection itself runs in the
            activation dtype regardless; only the QR/orth + ``V`` accumulation are
            in ``qr_dtype``.
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
    # EXP-26 Step C: the Q-basis FAMILY (content of the sketch ``orth(V)`` consumes
    # at FIXED rank). "act" (default) = the EXP-25 activation-energy basis (V is
    # built from the boundary ACTIVATIONS, V += Mᵀ(MQ)) — byte-identical to the
    # locked substrate so a run that does not opt in is unchanged. The RLVR-native
    # families ("grad"/"adv"/"tail"/"hybrid"/"ticket") bias V toward GRPO UPDATE
    # energy and ONLY run if Step A finds Q_act under-captures off-principal update
    # energy (H2). Validated to Q_BASIS_FAMILIES. The byte budget (rank) is held
    # fixed across families — only WHICH directions Q spans changes.
    #
    # This is the LIVE basis the fast/training path consumes (anchor feeds V from
    # the family's statistic). "act" is byte-identical; a non-"act" value is the
    # C2/Step-B LIVE-family training path (the fast path stays a read-only
    # consumer — the owns_q invariant is untouched, only the anchor's V content
    # changes). The compressor fails LOUD on a family it does not implement.
    q_basis: str = "act"
    # EXP-26 Step C1 PASSIVE screen: the families to PASSIVELY accumulate inside the
    # anchor's stale-weight pass (in no_grad, off the live Q / fast path / optimizer)
    # so ONE short run builds candidate bases for ALL families at once. Each family's
    # candidate Q_f is orthonormalized + dumped at the anchor cadence; the judge
    # metrics (update-capture, off-principal preservation) are computed offline
    # against the SAME captured reference grads. Empty (default) ⇒ no passive
    # accumulation (byte-identical). The live ``q_basis`` above is INDEPENDENT — the
    # screen runs with q_basis="act" LIVE while these families accumulate passively.
    # Validated to a subset of Q_BASIS_FAMILIES (minus "act"-only redundancy is OK —
    # "act" passively re-derives the live basis as the control row in the dump).
    q_basis_passive: list = field(default_factory=list)
    # EXP-26 Step C1: the hybrid family column split at FIXED total rank r. ``Q_h =
    # orth([Q_act[:, :hybrid_act_cols], Q_grad_deflated[:, :hybrid_grad_cols]])``.
    # ``-1`` (default) ⇒ AUTO: ``hybrid_act_cols = ceil(r/2)``,
    # ``hybrid_grad_cols = r − act`` (39 + 38 = 77 at the locked r=77 from
    # STEP_C_SPEC.md). When BOTH are set explicitly (>= 0) they MUST sum to the rank
    # (validated in __post_init__ only when "hybrid" is requested, so a non-hybrid
    # run is unconstrained). The compressor reads the resolved values via
    # ``resolved_hybrid_cols(rank)``.
    hybrid_act_cols: int = -1
    hybrid_grad_cols: int = -1


@dataclass
class CommEffCaptureConfig(BaseConfig):
    """EXP-26 Step A: real-gradient geometry-audit tensor-capture sub-config.

    **OFF by default — a strict no-op so the EXP-25 / plain-PowerSGD path is
    byte-identical** (Correctness invariant "off-path parity"). When
    ``enabled=true`` the comm-eff hooks dump fp32 tensors (``A``, ``Â=(A@Q)Qᵀ``,
    ``Q``, projection stats; ``G_comp`` the merger input; ``G_corr`` post-merger
    pre-Adam; ``M``/``G_anchor``; the parallel uncompressed ``G_dense``; and the
    ``delay_K=0`` fresh-anchor measurement grad) keyed by
    ``(global_step, optimizer_tick, target_name, shape, dtype, norm)`` under
    ``capture_dir``. Every dumped tensor is detached / dump-only — the capture
    path adds NO numerical side effect, only I/O (Correctness invariant
    "measurement-only probes never feed the optimizer").

    The two measurement-only PROBES (``capture_g_dense`` = a second uncompressed
    fast backward; ``capture_fresh_anchor`` = a delay_K=0 fresh-anchor grad) are
    the expensive, highest-integration-risk dumps; they are independently gated so
    the audit can be staged. ``delay_K=0`` appears ONLY here as a removed probe —
    it is forbidden as a TRAINING config (Correctness invariant "mandatory anchor
    staleness").

    Args:
        enabled (bool): Master capture switch. ``false`` (default) ⇒ NO dump hook
            fires, NO probe backward runs, NO tensor is written; the training path
            is byte-identical to the EXP-25 substrate.
        capture_dir (str): Directory the fp32 dumps + the manifest are written to
            (on the box; rsynced to ``runs/EXP-26/captures/``). Empty ⇒
            ``./captures`` relative to cwd.
        max_ticks (int): Cap on the number of optimizer ticks captured (the audit
            needs only ~5-10). ``<= 0`` ⇒ no cap. Bounds disk + rsync volume.
        stratified_targets (int): If ``> 0``, dump only this many targets PER
            layer-type (the √2 disagreement was uniform across the 7 matrix types
            in EXP-25, so a stratified subset is defensible — see ## Notes for
            runner). ``0`` ⇒ dump every target (full 196-matrix coverage).
        capture_g_dense (bool): Run the parallel UNCOMPRESSED fast backward to
            capture ``G_dense`` alongside ``G_comp`` at the SAME step. Detached /
            dump-only — MUST NOT touch the optimizer. ``false`` (default) ⇒ no
            second backward (the highest-OOM-risk probe; gate it on first).
        capture_fresh_anchor (bool): Capture the ``delay_K=0`` fresh-anchor grad as
            a MEASUREMENT probe for the sign-agreement decomposition. Detached /
            dump-only. ``false`` (default).
        fresh_anchor_loss (str): Loss the ``delay_K=0`` fresh-anchor MEASUREMENT
            probe backward uses — ``"clean_pg"`` (default, ratio≡1 ``-A·logπ`` like
            the anchor refresh) or ``"ppo_clip"`` (the SAME PPO ratio/clip loss as
            the fast path, against the batch's ``old_log_probs``). EXP-26 Step-C/B
            should-have: ``ppo_clip`` removes the clean-PG-vs-PPO-clip loss-mismatch
            confound the Step-A audit flagged, giving a clean
            ``cos(G_fresh_ppo, G_corr)`` improvement test. Affects ONLY the dump-only
            fresh-anchor probe (never the optimizer, the EMA, or the K-stale
            ``G_anchor`` that feeds ``M``). Validated to {clean_pg, ppo_clip}.
        dump_dtype (str): Dump precision. ``"fp32"`` (default, REQUIRED for the
            fidelity invariant — the reconstruction_rel_error recomputed from the
            dump must match the logged ~0.024 scalar within 1e-3). ``"bf16"`` is a
            volume-saving diagnostic only.
    """

    enabled: bool = False
    capture_dir: str = ""
    max_ticks: int = 10
    stratified_targets: int = 0
    capture_g_dense: bool = False
    capture_fresh_anchor: bool = False
    # EXP-26 Step C/B should-have: loss for the delay_K=0 fresh-anchor probe.
    # "clean_pg" (default, ratio≡1, matches the anchor refresh) or "ppo_clip" (the
    # fast path's PPO ratio/clip loss vs old_log_probs — removes the loss-mismatch
    # confound). Affects ONLY the dump-only probe.
    fresh_anchor_loss: str = "clean_pg"
    dump_dtype: str = "fp32"
    # EXP-26 disk-volume guard: capture ONLY rank 0 (default True). The audit's
    # gradient roles are DP-reduced and Q/A/Â are sync_basis-consensus — identical
    # across ranks — so rank 0 suffices, and writing all 4 ranks blew the box disk
    # (76 GB for ONE arm => torch.save crashed mid-write on the full disk). Set
    # False only if per-rank shards are genuinely needed.
    rank0_only: bool = True
    # EXP-26: skip capture ticks below this (cold-Q warmup). The anchor warms Q at
    # cadence (tick 5/10 for cadence=5); capturing from tick 1 fills the budget with
    # PRE-warm ticks (recon ~0.97), making H2 (Q activation/update-capture)
    # untrustworthy. Set just above the first anchor refresh so captured ticks are
    # POST-warm. 0 (default) = capture from the start (back-compat).
    min_tick: int = 0


@dataclass
class CommEffProbeConfig(BaseConfig):
    """EXP-30 Step-A geometry probe (M-validity discriminator) — OFF by default.

    **Telemetry-only by contract.** When ``geometry_enabled=true`` the harness
    measures, at every anchor fire on the EXP-29 paired-replay substrate, the
    m1–m7 geometry of the generator-consistent anchor gradient ``G_anc_rep``
    against (a) the old-style generator-MISMATCHED anchor gradient ``G_anc_old``
    (a second telemetry backward on the SAME stale-loaded clone with the CURRENT
    batch), (b) the live fast compressed gradient ``G_comp(t)``, (c) the
    fire-aware ring entry ``G_comp_ring(t−K)`` (m5 codec error), (d) its own lag
    history (m4), and (e) the previous fire's ``M_rep`` (m6 persistence). One
    JSON line per fire is appended to ``<out_dir>/stepA_fires.jsonl`` with the
    plan's verbatim field names, plus a ``[geometry-probe]`` train-log line.
    Nothing is ever fed to the optimizer, the EMA, the sketch V, or Q
    (``anchor_grad_corrected`` stays 0 — the probe invariant).

    Hard config invariants (validated in ``CommEffConfig.__post_init__``):
    requires ``anchor.enabled`` + ``anchor.replay_paired_batch`` (G_anc_rep IS
    the replay gradient), and the merger must be inert
    (``spectral.correction_mode="none"`` when spectral is enabled) so the
    measured ``G_comp`` is the raw codec output, never a merged gradient.

    Args:
        geometry_enabled (bool): Master probe switch. ``false`` (default) is a
            strict no-op — no ring, no lag buffer, no extra backward, no I/O.
        out_dir (str): Directory for ``stepA_fires.jsonl`` (+ the per-target
            sidecar). Empty (default) resolves to ``./geometry_probe`` at build.
        rank0_only (bool): Stage/compute/write on DP rank 0 only (default). The
            inputs are DP-identical by construction (the anchor grads are
            all-reduced(MEAN); the FSDP-summoned fast grads are the DP-mean), so
            rank 0 suffices; other ranks still run the symmetric collectives
            (the dual backward + DP-reduce) but skip storage and I/O. ``false``
            writes rank-suffixed files (debug only).
        m4_lags (int): Lag depth j=1..m4_lags for the m4 autocorrelation.
            Default 5; bounded to [1, 5] so the CPU lag buffer respects the
            plan's ≤6-entry bound (max_lag stored + the in-flight current).
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
    have zero numerical side effects, so a dense GRPO run with the scaffolding
    merged is bit-for-bit / rel-tol-1e-4 identical to one without it.

    Args:
        enabled (bool): Master switch. ``false`` (default) makes every comm_eff
            hook a no-op. Must be set ``true`` explicitly to activate any
            circuit.
        compression_type (str): Codec selector, one of
            ``{dense, prf_mask, powersgd}`` (mutually exclusive per run).
            ``dense`` (default) = no activation compression. ``prf_mask`` = the
            existing PRF mask. ``powersgd`` = the shared frozen-basis projector.
            For back-compat the legacy ``mask.enabled`` path still selects the
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
    #                 byte-identical to dense; equivalent to the activation path
    #                 of enabled=false). A legacy mask-circuit run is selected
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
    # EXP-26 Step A diagnostic tensor capture. OFF by default ⇒ no numerical side
    # effect (byte-identical to the EXP-25 substrate).
    capture: CommEffCaptureConfig = field(default_factory=CommEffCaptureConfig)
    # EXP-30 Step A geometry probe (telemetry-only M-validity discriminator).
    # OFF by default ⇒ no ring/buffer/backward/I-O (off-path parity).
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
        # typo ("False" string) or a numeric override into a loud error instead
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
                f"comm_eff.mask.rescale must be a bool; got {type(self.mask.rescale).__name__} "
                f"({self.mask.rescale!r})"
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
        # Anchor cadence/staleness.
        if self.anchor.cadence < 1:
            raise ValueError(f"comm_eff.anchor.cadence must be >= 1; got {self.anchor.cadence}")
        if self.anchor.delay_K < 0:
            raise ValueError(f"comm_eff.anchor.delay_K must be >= 0; got {self.anchor.delay_K}")
        # EXP-29: on-policy replay flag is a strict bool (a YAML "False" string or
        # a numeric override must be loud, not a silent truthy surprise that turns
        # the replay ring on/off by accident — same rationale as mask_recompute).
        if not isinstance(self.anchor.replay_paired_batch, bool):
            raise ValueError(
                f"comm_eff.anchor.replay_paired_batch must be a bool; got "
                f"{type(self.anchor.replay_paired_batch).__name__} ({self.anchor.replay_paired_batch!r})"
            )
        # EXP-29: snapshot storage enum (mirrors spectral.ema_device). "gpu" is
        # the byte-identical legacy default; "cpu" moves the delay_K+1 full
        # snapshots off HBM (numerics-neutral byte-preserving round trip).
        if self.anchor.snapshot_device not in ("gpu", "cpu"):
            raise ValueError(
                f"comm_eff.anchor.snapshot_device must be one of (gpu, cpu); "
                f"got {self.anchor.snapshot_device!r}"
            )
        # Storage-layer enum.
        if self.spectral.ema_device not in ("gpu", "cpu"):
            raise ValueError(f"comm_eff.spectral.ema_device must be one of (gpu, cpu); got {self.spectral.ema_device!r}")
        if self.spectral.correction_mode not in ("none", "inject", "blend", "signed_ema", "ef_powersgd", "delayed_ef"):
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
        # EXP-25 (R3) signed_ema merger weight. [0, 1]: 0 => pure |G|*sign(M)
        # sign-merger, 1 => G_noisy unchanged. Unused unless
        # correction_mode=signed_ema.
        if not 0.0 <= self.spectral.signed_ema_alpha <= 1.0:
            raise ValueError(
                f"comm_eff.spectral.signed_ema_alpha must be in [0, 1]; got {self.spectral.signed_ema_alpha}"
            )
        # EXP-26 Step B: error-feedback residual knobs. decay in [0, 1) (an EMA
        # decay; 1.0 would never forget the residual). clip >= 0 (a norm-cap
        # fraction; 0 ⇒ residual fully zeroed ⇒ G_corr==G_comp, the plain-PowerSGD
        # limiting case). Both validated unconditionally so a typo is loud even on
        # a non-ef_powersgd run that forwards them.
        if not 0.0 <= self.spectral.ef_decay < 1.0:
            raise ValueError(
                f"comm_eff.spectral.ef_decay must be in [0, 1); got {self.spectral.ef_decay}"
            )
        if self.spectral.ef_clip < 0.0:
            raise ValueError(
                f"comm_eff.spectral.ef_clip must be >= 0; got {self.spectral.ef_clip}"
            )
        # EXP-30 B2: the delayed_ef residual weight. >= 0; 0.0 (default) is the
        # exact-identity limiting case. Validated unconditionally so a typo is
        # loud even on a non-delayed_ef run that forwards it.
        if self.spectral.delayed_ef_lambda < 0.0:
            raise ValueError(
                f"comm_eff.spectral.delayed_ef_lambda must be >= 0; got {self.spectral.delayed_ef_lambda}"
            )
        # EXP-31 Cell D: the additive stale-anchor sub-basis rank. >= 0; 0
        # (default) is OFF (the exact B2 path). Validated unconditionally so a
        # typo is loud even on a non-delayed_ef run that forwards it.
        if self.spectral.delta_subbasis_rank < 0:
            raise ValueError(
                f"comm_eff.spectral.delta_subbasis_rank must be >= 0; got {self.spectral.delta_subbasis_rank}"
            )
        if self.spectral.delta_subbasis_family not in ("tail", "grad"):
            raise ValueError(
                "comm_eff.spectral.delta_subbasis_family must be one of (tail, grad); "
                f"got {self.spectral.delta_subbasis_family!r}"
            )
        # EXP-31 Cell D γ-knob: the sub-basis WEIGHT γ (>= 0; 1.0 default = current
        # Cell D, 0 = B2) and its linear-DECAY horizon (>= 0; 0 default = constant
        # γ). Validated unconditionally so a typo is loud even on a non-delayed_ef /
        # rank-0 run that forwards them.
        if self.spectral.delta_subbasis_weight < 0.0:
            raise ValueError(
                f"comm_eff.spectral.delta_subbasis_weight must be >= 0; got {self.spectral.delta_subbasis_weight}"
            )
        if self.spectral.delta_subbasis_decay_steps < 0:
            raise ValueError(
                "comm_eff.spectral.delta_subbasis_decay_steps must be >= 0; "
                f"got {self.spectral.delta_subbasis_decay_steps}"
            )
        # EXP-31 hold-then-decay: the HOLD horizon H (>= 0; 0 default = the existing
        # linear-from-0 decay, bitwise). Only meaningful when decay_steps > 0.
        if self.spectral.delta_subbasis_hold_steps < 0:
            raise ValueError(
                "comm_eff.spectral.delta_subbasis_hold_steps must be >= 0; "
                f"got {self.spectral.delta_subbasis_hold_steps}"
            )
        # EXP-31 Cell C: the correction-δ compression rank. >= 0; 0 (default) is
        # OFF (uncompressed δ = the B2 / Cell D path).
        if self.spectral.r_delta < 0:
            raise ValueError(
                f"comm_eff.spectral.r_delta must be >= 0; got {self.spectral.r_delta}"
            )
        # EXP-31 surpass lever: the zero-mean perturbation magnitude σ (>= 0; 0.0
        # default = OFF ⇒ the exact delayed_ef / B2 path) and its seed (the
        # cross-rank-identical RNG salt). Validated unconditionally so a typo is
        # loud even on a non-delayed_ef / σ=0 run that forwards them.
        if self.spectral.perturb_sigma < 0.0:
            raise ValueError(
                f"comm_eff.spectral.perturb_sigma must be >= 0; got {self.spectral.perturb_sigma}"
            )
        # EXP-31 L2: δ-momentum. delta_momentum_mu is the NORMALIZED-EMA decay μ in
        # [0, 1) (μ=0.0 default = OFF; the stationary gain is EXACTLY 1 for any μ in
        # range; μ=1 would never forget so it is excluded, mirroring ef_decay).
        # delta_momentum_age_decay is a strict bool (a YAML "False" string would
        # silently flip the staleness-degrade behaviour — same rationale as
        # mask_recompute). Validated unconditionally so a typo is loud even on a
        # non-delayed_ef / μ=0 run that forwards them.
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
        # EXP-31 L3: adaptive dose. adaptive_lambda_mode is the closed enum
        # {off, cos, ratio} (a typo must be loud, not a silent fall-through to off);
        # adaptive_lambda_kappa is the gate gain κ >= 0 (0 default = OFF ⇒ λ_t≡λ);
        # lambda_cap is the RAW-λ upper bound >= 0 (the variable-staleness safety
        # clamp). Validated unconditionally so a typo is loud even on a
        # non-delayed_ef / off run that forwards them.
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
            raise ValueError(
                f"comm_eff.spectral.lambda_cap must be >= 0; got {self.spectral.lambda_cap}"
            )
        # EXP-30 B2: delayed_ef merges the VALID (generator-consistent) M_rep by
        # definition — running it on the legacy generator-mismatched feed would
        # re-test the retired object (the falsified #23/#25/#26/#27 dose-response).
        # Fail loud at config time, not silently mid-run.
        if self.spectral.enabled and self.spectral.correction_mode == "delayed_ef":
            if not (self.anchor.enabled and self.anchor.replay_paired_batch):
                raise ValueError(
                    "comm_eff.spectral.correction_mode=delayed_ef requires the EXP-29 "
                    "generator-consistent anchor feed: comm_eff.anchor.enabled=true AND "
                    "comm_eff.anchor.replay_paired_batch=true (the valid-M premise; "
                    f"got enabled={self.anchor.enabled}, replay_paired_batch={self.anchor.replay_paired_batch})."
                )
        # EXP-30 Step A: geometry-probe knobs. geometry_enabled is a strict bool
        # (a YAML "False" string would silently arm a 2nd backward per fire);
        # rank0_only / per_target_sidecar likewise; m4_lags in [1, 5] (the plan's
        # ≤6-entry lag-buffer bound).
        for _bname in ("geometry_enabled", "rank0_only", "per_target_sidecar"):
            _bval = getattr(self.probe, _bname)
            if not isinstance(_bval, bool):
                raise ValueError(
                    f"comm_eff.probe.{_bname} must be a bool; got {type(_bval).__name__} ({_bval!r})"
                )
        if not 1 <= self.probe.m4_lags <= 5:
            raise ValueError(
                f"comm_eff.probe.m4_lags must be in [1, 5] (lag buffer <=6-entry bound); "
                f"got {self.probe.m4_lags}"
            )
        if self.probe.geometry_enabled:
            # The probe measures the EXP-29 replay substrate — G_anc_rep IS the
            # paired-replay gradient. Without replay there is nothing valid to
            # measure (m1 would silently equal m2).
            if not (self.anchor.enabled and self.anchor.replay_paired_batch):
                raise ValueError(
                    "comm_eff.probe.geometry_enabled=true requires comm_eff.anchor.enabled=true "
                    "AND comm_eff.anchor.replay_paired_batch=true (G_anc_rep is the EXP-29 "
                    "paired-replay gradient; without replay the probe would measure the retired "
                    f"generator-mismatched feed). Got enabled={self.anchor.enabled}, "
                    f"replay_paired_batch={self.anchor.replay_paired_batch}."
                )
            # The probe's G_comp must be the RAW codec output — an active merger
            # would rewrite the live grads before the end-of-batch extraction and
            # silently corrupt m1/m2/m4/m5. Step A runs correction INERT.
            if self.spectral.enabled and self.spectral.correction_mode != "none":
                raise ValueError(
                    "comm_eff.probe.geometry_enabled=true requires an INERT merger: set "
                    "comm_eff.spectral.correction_mode=none (Step A measures the raw G_comp; "
                    f"an active merger would corrupt it). Got correction_mode="
                    f"{self.spectral.correction_mode!r}."
                )
        # Periodic clean-step cadence. 0 = off. A negative value is a config
        # error, not a silent disable.
        if self.clean_cadence < 0:
            raise ValueError(f"comm_eff.clean_cadence must be >= 0; got {self.clean_cadence}")
        # Codec selector. Validated to the closed enum so a typo
        # (compression_type=powerSGD / powergsd) is a loud error, not a silent
        # fall-through to dense.
        if self.compression_type not in COMPRESSION_TYPES:
            raise ValueError(
                f"comm_eff.compression_type must be one of {COMPRESSION_TYPES}; "
                f"got {self.compression_type!r}"
            )
        # PowerSGD block. Validated unconditionally (the keys are
        # registered regardless of compression_type) so a prf_mask run that
        # forwards comm_eff.powersgd.* args still fails fast on a bad value.
        if self.powersgd.rank < 1:
            raise ValueError(f"comm_eff.powersgd.rank must be >= 1; got {self.powersgd.rank}")
        if self.powersgd.pp_size < 1:
            raise ValueError(f"comm_eff.powersgd.pp_size must be >= 1; got {self.powersgd.pp_size}")
        if self.powersgd.update_cadence < 1:
            raise ValueError(
                f"comm_eff.powersgd.update_cadence must be >= 1; got {self.powersgd.update_cadence}"
            )
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
            raise ValueError(
                f"comm_eff.powersgd.qr_dtype must be one of (fp32, bf16); got {self.powersgd.qr_dtype!r}"
            )
        if self.powersgd.reortho_eps <= 0.0:
            raise ValueError(f"comm_eff.powersgd.reortho_eps must be > 0; got {self.powersgd.reortho_eps}")
        # EXP-26 Step C: Q-basis family. Validated to the closed enum so a typo
        # (q_basis=gradient) is a loud error, not a silent fall-through to "act".
        if self.powersgd.q_basis not in Q_BASIS_FAMILIES:
            raise ValueError(
                f"comm_eff.powersgd.q_basis must be one of {Q_BASIS_FAMILIES}; "
                f"got {self.powersgd.q_basis!r}"
            )
        # EXP-26 Step C1: the PASSIVE screen family list. Every entry must be a
        # known family; a typo is a loud error (a silently-dropped family would make
        # the screen miss an arm). OmegaConf may pass a ListConfig — iterate it.
        for _fam in list(self.powersgd.q_basis_passive):
            if _fam not in Q_BASIS_FAMILIES:
                raise ValueError(
                    f"comm_eff.powersgd.q_basis_passive entries must each be one of "
                    f"{Q_BASIS_FAMILIES}; got {_fam!r}"
                )
        # EXP-26 Step C1: the hybrid column split. Only meaningful when the hybrid
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
        # EXP-26 Step A: diagnostic-capture block. Validated unconditionally (the
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
            raise ValueError(
                f"comm_eff.capture.stratified_targets must be >= 0; got {self.capture.stratified_targets}"
            )
