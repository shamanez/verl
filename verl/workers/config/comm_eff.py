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
    "CommEffConfig",
]

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
    """

    enabled: bool = False
    cadence: int = 20
    delay_K: int = 20


@dataclass
class CommEffSpectralConfig(BaseConfig):
    """Spectral-correction sub-config (inert while disabled).

    Implements the anchor-EMA -> SVD -> Tikhonov weights -> two-sided projection
    -> alpha blend applied to selected 2D decoder gradients after the actor
    backward and before ``optimizer.step()``. See
    ``verl.workers.comm_eff.spectral_filter.SpectralFilter``.

    The formula (per targeted 2D matrix ``G_mask`` with anchor-EMA ``M_anchor``)::

        M_anchor = beta_anc * M_anchor + (1 - beta_anc) * G_anchor   # EMA
        M_anchor = U S V^T                                           # full thin SVD
        d_i      = s_i / (s_i + tau)                                 # Tikhonov weights
        X        = U^T G_mask V
        G_filt   = U diag(d) X diag(d) V^T                           # two-sided projection
        G_proj   = alpha * G_mask + (1 - alpha) * G_filt             # blend

    At ``alpha=1.0`` this is an exact no-op (``G_proj == G_mask``); at
    ``alpha=0`` it is the pure two-sided Tikhonov projection. The masked
    gradient is never discarded — the anchor supplies geometry, not a
    replacement.

    Args:
        enabled (bool): Whether spectral correction of masked gradients runs.
            Gated by the parent ``comm_eff.enabled`` regardless of this value.
            ``false`` (default) ⇒ the grad-correction hook is a strict no-op and
            the actor path is identical to dense GRPO.
        alpha (float): Blend coefficient in ``[0, 1]``: ``alpha * G_mask +
            (1 - alpha) * G_filt``. ``1.0`` ⇒ no-op; ``0.0`` ⇒ pure projection.
            Default ``0.3``.
        tau (float): Tikhonov damping added to each singular value before
            forming the spectral weight ``d_i = s_i / (s_i + tau)``. Default
            ``1e-3``.
        beta_anc (float): EMA decay for the anchor-gradient running matrix
            ``M_anchor``. Default ``0.95``.
        seed_anchor_cache (bool): When ``true``, populate ``M_anchor`` with a
            fixed deterministic PSD basis (seeded) so the filter runs without
            a live anchor refresh. When ``false`` the anchor EMA starts empty
            and is populated by the live anchor circuit.
        anchor_seed (int): Base seed for the deterministic anchor cache.
        target_substr (list[str]): Substrings used to SELECT which named 2D
            parameters receive correction. A parameter is targeted iff its name
            contains one of these substrings AND its logical shape is 2D.
            Defaults select the decoder attention/MLP projection matrices and
            skip norms, biases, embeddings and the lm head.
        max_targets (int): Cap on the number of target matrices corrected per
            step (keeps the discovery smoke cheap). ``-1`` ⇒ no cap.
        rank (int): Retained low-rank truncation rank. Under ``svd_mode=lowrank``
            it is the ``q`` passed to ``torch.svd_lowrank``; under ``full`` it is
            unused. Must be ``>= 1``.
        damping (float): Legacy alias kept for schema back-compat; the active
            damping knob is ``tau``.
        ema_device (str): Where the anchor-gradient EMA ``M_anchor`` is stored
            between refreshes — ``"gpu"`` (default, faithful: kept in HBM) or
            ``"cpu"`` (memory-lean: offloaded to pinned CPU, moved to GPU only
            inside the refresh/correct call and moved back). ``M_anchor`` is
            touched only at refresh, so CPU offload costs one H2D/D2H per refresh,
            not per mini-batch. Validated against {gpu, cpu}.
        svd_mode (str): How the anchor SVD basis is computed — ``"full"``
            (default, faithful: ``torch.linalg.svd`` full thin SVD) or
            ``"lowrank"`` (memory-lean: ``torch.svd_lowrank(M_anchor, q=rank)``,
            shrinking ``U/S/V`` from ``O(m·k)`` to ``O(m·rank)``). Validated
            against {full, lowrank}.
        basis_cache (str): Whether the ``U/S/V`` basis is computed once per
            refresh and reused across the fast PPO mini-batches — ``"cache"``
            (default, faithful: compute at refresh, store on GPU, reuse) or
            ``"recompute"`` (memory-lean inverse: recompute the SVD on every
            ``correct_matrix``). The basis is touched every fast mini-batch, so
            it stays on-GPU during the refresh window
            regardless. Validated against {cache, recompute}.
        cadence (int): Spectral-correction cadence in optimizer steps.
            The grad-correction hook (``_maybe_comm_eff_grad_correction``) fires
            only when ``(spectral_step % cadence) == 0`` on the monotonic
            per-optimizer-step counter, MIRRORING the anchor cadence
            (``anchor_should_fire``) and the clean-step cadence
            (``CommEffState.is_clean_step``). ``1`` (default) fires EVERY step =
            the default behavior. Set ``> 1`` (e.g. ``2``) to align
            spectral correction with a matching ``anchor.cadence`` so the
            correction always uses a freshly-refreshed anchor basis instead of a
            stale one on the in-between steps. Must be ``>= 1``.
    """

    enabled: bool = False
    alpha: float = 0.3
    tau: float = 1e-3
    beta_anc: float = 0.95
    cadence: int = 1
    seed_anchor_cache: bool = True
    anchor_seed: int = 0
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
    max_targets: int = 4
    rank: int = 8
    damping: float = 1e-6
    # EMA/SVD storage defaults: keep tensors on GPU, use full SVD, and cache the
    # basis per refresh.
    ema_device: str = "gpu"
    svd_mode: str = "full"
    basis_cache: str = "cache"
    # Correction mode. "reweight" applies two-sided Tikhonov reweighting.
    # "inject" adds a scale-matched anchor-EMA complement. "blend" uses
    # G_corr=(1-eta)*G_mask + eta*scale*M_anchor.
    correction_mode: str = "reweight"
    # Injection strength for correction_mode="inject"; unused otherwise.
    inject_gamma: float = 1.0
    # Convex-blend weight for correction_mode="blend"; validated to [0, 1].
    blend_eta: float = 0.5


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
        if self.spectral.rank < 1:
            raise ValueError(f"comm_eff.spectral.rank must be >= 1; got {self.spectral.rank}")
        if not 0.0 <= self.spectral.alpha <= 1.0:
            raise ValueError(f"comm_eff.spectral.alpha must be in [0, 1]; got {self.spectral.alpha}")
        if self.spectral.tau <= 0.0:
            raise ValueError(f"comm_eff.spectral.tau must be > 0; got {self.spectral.tau}")
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
        # Storage-layer enums.
        if self.spectral.ema_device not in ("gpu", "cpu"):
            raise ValueError(f"comm_eff.spectral.ema_device must be one of (gpu, cpu); got {self.spectral.ema_device!r}")
        if self.spectral.svd_mode not in ("full", "lowrank"):
            raise ValueError(f"comm_eff.spectral.svd_mode must be one of (full, lowrank); got {self.spectral.svd_mode!r}")
        if self.spectral.basis_cache not in ("cache", "recompute"):
            raise ValueError(
                f"comm_eff.spectral.basis_cache must be one of (cache, recompute); got {self.spectral.basis_cache!r}"
            )
        if self.spectral.correction_mode not in ("reweight", "inject", "blend"):
            raise ValueError(
                f"comm_eff.spectral.correction_mode must be one of (reweight, inject, blend); "
                f"got {self.spectral.correction_mode!r}"
            )
        if self.spectral.inject_gamma < 0.0:
            raise ValueError(f"comm_eff.spectral.inject_gamma must be >= 0; got {self.spectral.inject_gamma}")
        # Convex-blend weight. [0, 1]: 0 => pure G_mask, 1 =>
        # scale-matched stale true gradient. Unused unless correction_mode=blend.
        if not 0.0 <= self.spectral.blend_eta <= 1.0:
            raise ValueError(f"comm_eff.spectral.blend_eta must be in [0, 1]; got {self.spectral.blend_eta}")
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
