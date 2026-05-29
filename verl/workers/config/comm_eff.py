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

The nested sub-configs (mask / anchor / spectral) are declared here so the
Hydra schema is validated up front (typos in ``comm_eff.mask.*`` etc. are
rejected by OmegaConf's structured-config merge), but they carry no behavior
while ``enabled=false``. They are consumed only by later M2 work that flips
``enabled=true``.
"""

from dataclasses import dataclass, field

from verl.base_config import BaseConfig

__all__ = [
    "CommEffMaskConfig",
    "CommEffAnchorConfig",
    "CommEffSpectralConfig",
    "CommEffConfig",
]


@dataclass
class CommEffMaskConfig(BaseConfig):
    """Pipeline activation-masking sub-config (inert while ``comm_eff.enabled=false``).

    Implements Algorithm A: a deterministic PRF Bernoulli mask applied in-graph
    (``h_tilde = h * mask``, **no** ``1/(1-p)`` rescale) to the hidden-state
    output of pipeline-boundary decoder blocks, **only** inside the actor train
    forward/backward. See ``verl.workers.comm_eff.activation_mask``.

    Args:
        enabled (bool): Whether activation masking runs. Gated by the parent
            ``comm_eff.enabled`` regardless of this value (so the disabled path
            stays a strict no-op). Default ``True`` so that flipping the parent
            ``comm_eff.enabled=true`` activates masking without a second flag;
            set ``false`` to keep masking off while another circuit is enabled.
        p (float): **Masked fraction** in ``[0, 1]`` — the probability an
            element is zeroed (``mask=0``). The measured ``comm_eff/mask_ratio``
            tracks this value. ``0.0`` means no masking. Only consulted when
            ``comm_eff.enabled=true``.
        seed (int): Base seed folded into the mask PRF key. Reproducible across
            ranks and re-runs. Only drawn from when ``comm_eff.enabled=true`` —
            the disabled path never touches RNG.
        pp_size (int): Logical pipeline-shard count used to derive which decoder
            blocks are masked: the decoder blocks are partitioned into
            ``pp_size`` contiguous shards and the last block of every shard
            **except the final shard** is masked. ``L`` is read from
            ``model.config`` (never hardcoded). For ``L=16, pp_size=8`` the
            boundaries are ``[1, 3, 5, 7, 9, 11, 13]``. This is a logical knob,
            not a real pipeline split.
        mask_recompute (bool): When ``True`` (EXP-9), the activation mask is
            ALSO permitted to fire on the ``old_logprob`` recompute path
            (``compute_log_prob``), in addition to the actor-train forward.
            This extends the *fast (masked) circuit* to BOTH gradient-feeding
            forwards in pipeline-parallel RL: the train forward produces
            ``log_prob_current`` and the old_logp recompute produces
            ``old_log_prob`` that enters the PPO importance ratio
            ``r = exp(log_prob_current - old_log_prob)``. Both consume
            pipeline-boundary bandwidth on the FSDP train engine. ``False``
            (default) keeps the EXP-5 / EXP-6 / EXP-7 / EXP-8 / EXP-12 behavior
            where the mask fires ONLY on the train forward. The mask form
            stays ``h_tilde = h * mask`` (no ``1/(1-p)`` rescale); the
            substep counter naturally differs between ``compute_log_prob``
            (one call per trainer step) and the PPO inner loop (N×E calls per
            trainer step), so masks differ between the two paths by design.
            Anchor pass (``path_tag=None``) is unaffected and stays unmasked
            regardless of this flag (GUARD 5).
        consistent_across_forwards (bool): EXP-14 on-policy-consistency fix.
            When ``True`` (default), the PRF mask is realized IDENTICALLY across
            every forward pass that belongs to the SAME global gradient update —
            i.e. the ``compute_log_prob`` old-logprob recompute (when
            ``mask_recompute=true``) AND all micro-batches of the actor-train
            forward at a given ``global_step`` draw the same mask per boundary
            block. The PRF key drops its per-forward ``substep`` component (it is
            held at a fixed sentinel ``0``) so the key is
            ``f(layer_idx, global_step, seq_shard, hidden_size, seed)`` —
            independent of WHICH forward pass fired it. This restores on-policy
            correctness: ``pi_old`` (old_logprob) and ``pi_new`` (train forward)
            are computed under the SAME masked subnetwork, so the PPO importance
            ratio ``r = exp(log_prob_current - old_log_prob)`` is ≈1 at the
            first inner step (as in vanilla on-policy GRPO) instead of being
            corrupted by mask resampling between the two gradient-feeding
            forwards (the EXP-14 test2_cellA grad_norm=771 explosion). The
            ``substep`` counter still ADVANCES per masked forward so metrics and
            the ``consistent_across_forwards=false`` path are byte-unchanged; it
            is simply NOT folded into the PRF key in consistent mode. When
            ``False`` the legacy EXP-5⇒EXP-12 behavior is preserved exactly: the
            advancing ``substep`` keys the mask, so every forward (and every PPO
            mini-batch) gets a distinct mask. ``True`` is the default because a
            consistent mask is the on-policy-correct choice; flip ``false`` only
            for the A/B comparison that isolates the resampling effect.
        rescale (bool): EXP-14 magnitude-preservation knob (a DELIBERATE
            DESIGN-CHANGE candidate — see the warning below). When ``False``
            (default) the mask form is the spec's pure-simulated-pipeline-parallel
            ``h_tilde = h * mask``: a masked (zeroed) boundary activation is
            literally a dropped activation, so at ``p=0.9`` the boundary
            hidden-state RMS collapses to ``sqrt(1-p) ≈ 0.316×`` and the masked
            forward sees a large distribution shift from the weights' training
            regime — the suspected driver of the test2_cellA/cellC step-1
            grad_norm explosion (clean_cadence works precisely because its clean
            step runs the UNMASKED, full-magnitude network). When ``True`` the
            hook applies inverted-dropout-style magnitude preservation
            ``h_tilde = h * mask / (1 - p)`` so the kept activations are scaled up
            to hold ``E[h_tilde] = h`` (the boundary RMS is preserved in
            expectation), at the cost of departing from the "dropped activations
            are zeros" pipeline-parallel analogy. ``DESIGN NOTE``: the method spec
            (CODE_WALKTHROUGH §1 + the "Out of scope" list) explicitly EXCLUDES a
            forward ``1/(1-p)`` rescale, both because it breaks the pure-PP
            interpretation and because it was observed to destabilise bf16 at
            ``p=0.95``. This knob exists to TEST whether the no-rescale magnitude
            collapse — not the importance-sampling inconsistency — dominates the
            grad_norm explosion; treat a PASS here as evidence for a design
            change to be ratified, not as a silently-adopted default. Default
            ``False`` keeps every pre-EXP-14 run byte-identical.
        granularity (str): EXP-14 mask granularity — ``"channel"`` (the new
            DEFAULT) or ``"element"`` (legacy). This is the on-policy-consistency
            fix that supersedes ``consistent_across_forwards``.

            ``"channel"`` draws ONE ``(hidden,)`` keep/zero vector keyed on
            ``(layer, global_step, seed)`` — NO token / substep / sequence-shard
            / packing component — and drops the SAME hidden channels for EVERY
            token at the boundary (``h_tilde = h * mask`` with ``mask`` broadcast
            over the token axis). Because the mask is CONSTANT along the token
            axis, it is IDENTICAL across every forward pass of one global update
            no matter how dynamic-bsz packs / length-sorts / shuffles the tokens
            (the ``compute_log_prob`` old-logprob recompute and the actor-train
            forward run different micro-batch counts — ~21 vs ~28 — so a
            per-element positional mask hits a given token at different flat
            indices in the two phases, corrupting ``log pi_new / pi_old``;
            per-channel makes the two phases bit-identical by construction, like
            a single supervised forward). It is also a faithful model of
            structured pipeline-boundary communication compression (transmit a
            fixed random subset of the hidden channels for all tokens), and the
            measured ``comm_eff/mask_ratio`` still tracks ``p`` (fraction of
            zeroed channels). The ``rescale`` knob (``h * mask / (1-p)``) applies
            unchanged and remains the magnitude-collapse / grad_norm fix.

            ``"element"`` is the legacy per-element mask (an independent draw per
            ``(token, channel)``): packing-dependent, with cross-forward
            consistency only APPROXIMATED by ``consistent_across_forwards``
            holding the substep fixed (which is insufficient — see cellC). Kept
            for A/B comparison. Note: a packing-invariant per-element mask is
            impossible without keying on stable per-token identity (rejected
            plumbing), so ``"channel"`` is the only no-plumbing route to exact
            cross-forward consistency. ``consistent_across_forwards`` is a no-op
            under ``"channel"`` (channel masks are inherently consistent) and
            stays functional only for the ``"element"`` path. Default
            ``"channel"`` so all comm-eff experiments get the consistent mask.
    """

    enabled: bool = True
    p: float = 0.95
    seed: int = 0
    pp_size: int = 8
    mask_recompute: bool = False
    consistent_across_forwards: bool = True
    rescale: bool = False
    granularity: str = "channel"


@dataclass
class CommEffAnchorConfig(BaseConfig):
    """Asynchronous unmasked-anchor-circuit sub-config (inert while disabled).

    The anchor circuit (EXP-8) runs, every ``cadence`` trainer steps, ONE
    unmasked GRPO-actor-loss forward/backward from a ``delay_K``-stale weight
    snapshot to produce a clean per-target gradient ``G_anchor``. ``G_anchor`` is
    read RAW (before any spectral correction) into the anchor-gradient EMA
    ``M_anchor`` — whose decay is ``spectral.beta_anc`` (the EMA is owned by the
    spectral filter, NOT a separate anchor knob; that is why the EXP-4 scaffold's
    ``ema_decay`` is dropped here). The anchor takes NO optimizer step and
    generates NO rollouts / recomputes NO rewards.

    Args:
        enabled (bool): Whether the anchor circuit runs. Gated by the parent
            ``comm_eff.enabled`` regardless of this value. ``false`` (default) is
            a strict no-op — opt-in only.
        cadence (int): Anchor-refresh cadence in trainer steps (paper ``K``). The
            anchor fires when ``(step % cadence) == 0``. Smoke uses ``1`` (fire
            every step); the paper default is ``20``. Must be ``>= 1``.
        delay_K (int): Staleness of the weight snapshot the anchor forwards
            from, in trainer steps. ``0`` = current weights; ``1`` = the prior
            step's weights (smoke). Must be ``>= 0``. Replaces the EXP-4 scaffold
            field ``every_n_steps`` (which was never consumed).
    """

    enabled: bool = False
    cadence: int = 20
    delay_K: int = 20


@dataclass
class CommEffSpectralConfig(BaseConfig):
    """Spectral-correction sub-config (inert while disabled).

    Implements the M2 paper formula (anchor-EMA -> full thin SVD -> Tikhonov
    spectral weights -> two-sided projection -> alpha blend) applied to the
    gradients of selected 2D decoder matrices after the actor backward and
    before ``optimizer.step()``. See
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
            the actor path is identical to EXP-5 / dense GRPO.
        alpha (float): Blend coefficient in ``[0, 1]``: ``alpha * G_mask +
            (1 - alpha) * G_filt``. ``1.0`` ⇒ no-op; ``0.0`` ⇒ pure projection.
            Default ``0.3`` (the EXP-7 operating point).
        tau (float): Tikhonov damping added to each singular value before
            forming the spectral weight ``d_i = s_i / (s_i + tau)``. Default
            ``1e-3``.
        beta_anc (float): EMA decay for the anchor-gradient running matrix
            ``M_anchor``. Default ``0.95``.
        seed_anchor_cache (bool): When ``true``, populate ``M_anchor`` with a
            fixed deterministic PSD basis (seeded) so the filter runs without
            the (not-yet-built, EXP-8) live anchor circuit. The EXP-7 discovery
            smoke uses this. When ``false`` the anchor EMA starts empty and is
            populated by the live anchor circuit.
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
            ``correct_matrix`` as the pre-EXP-8 code did). The basis is touched
            every fast mini-batch, so it stays on-GPU during the refresh window
            regardless. Validated against {cache, recompute}.
    """

    enabled: bool = False
    alpha: float = 0.3
    tau: float = 1e-3
    beta_anc: float = 0.95
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
    # EXP-8 config-driven EMA/SVD storage layer. Defaults stay FAITHFUL
    # (gpu/full/cache) so cell 1 and the EXP-7 contract are numerically unchanged.
    ema_device: str = "gpu"
    svd_mode: str = "full"
    basis_cache: str = "cache"


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
        mask (CommEffMaskConfig): Pipeline activation-masking sub-config.
        anchor (CommEffAnchorConfig): Asynchronous anchor-circuit sub-config.
        spectral (CommEffSpectralConfig): Spectral-correction sub-config.
        clean_cadence (int): EXP-14 periodic clean (unmasked) optimizer-step
            cadence, in trainer steps. ``0`` (default) = off, so the disabled
            path AND every pre-EXP-14 method config stay a strict no-op. When
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
    mask: CommEffMaskConfig = field(default_factory=CommEffMaskConfig)
    anchor: CommEffAnchorConfig = field(default_factory=CommEffAnchorConfig)
    spectral: CommEffSpectralConfig = field(default_factory=CommEffSpectralConfig)
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
        # EXP-9: mask_recompute is a strict bool. Validation here turns a YAML
        # typo ("False" string) or a numeric override into a loud error instead
        # of a silent truthy/falsy surprise that would mis-route masking.
        if not isinstance(self.mask.mask_recompute, bool):
            raise ValueError(
                f"comm_eff.mask.mask_recompute must be a bool; got {type(self.mask.mask_recompute).__name__} "
                f"({self.mask.mask_recompute!r})"
            )
        # EXP-14: consistent_across_forwards is a strict bool (same rationale as
        # mask_recompute — a YAML "False" string or numeric override must fail
        # loud, not silently flip the on-policy-consistency contract).
        if not isinstance(self.mask.consistent_across_forwards, bool):
            raise ValueError(
                "comm_eff.mask.consistent_across_forwards must be a bool; got "
                f"{type(self.mask.consistent_across_forwards).__name__} "
                f"({self.mask.consistent_across_forwards!r})"
            )
        # EXP-14: rescale is a strict bool (same rationale as the other mask
        # flags). It also requires p < 1 so the 1/(1-p) factor is finite; p==1
        # with rescale would divide by zero (and p==1 masks everything anyway).
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
        # EXP-14: mask granularity enum (channel = new packing-invariant default,
        # element = legacy per-element). A typo must fail loud, not silently fall
        # back to a different masking regime.
        if self.mask.granularity not in ("channel", "element"):
            raise ValueError(
                "comm_eff.mask.granularity must be one of (channel, element); "
                f"got {self.mask.granularity!r}"
            )
        if self.spectral.rank < 1:
            raise ValueError(f"comm_eff.spectral.rank must be >= 1; got {self.spectral.rank}")
        if not 0.0 <= self.spectral.alpha <= 1.0:
            raise ValueError(f"comm_eff.spectral.alpha must be in [0, 1]; got {self.spectral.alpha}")
        if self.spectral.tau <= 0.0:
            raise ValueError(f"comm_eff.spectral.tau must be > 0; got {self.spectral.tau}")
        if not 0.0 <= self.spectral.beta_anc <= 1.0:
            raise ValueError(f"comm_eff.spectral.beta_anc must be in [0, 1]; got {self.spectral.beta_anc}")
        # EXP-8 anchor cadence/staleness (replaces the unused EXP-4 ema_decay).
        if self.anchor.cadence < 1:
            raise ValueError(f"comm_eff.anchor.cadence must be >= 1; got {self.anchor.cadence}")
        if self.anchor.delay_K < 0:
            raise ValueError(f"comm_eff.anchor.delay_K must be >= 0; got {self.anchor.delay_K}")
        # EXP-8 storage-layer enums (faithful defaults: gpu/full/cache).
        if self.spectral.ema_device not in ("gpu", "cpu"):
            raise ValueError(f"comm_eff.spectral.ema_device must be one of (gpu, cpu); got {self.spectral.ema_device!r}")
        if self.spectral.svd_mode not in ("full", "lowrank"):
            raise ValueError(f"comm_eff.spectral.svd_mode must be one of (full, lowrank); got {self.spectral.svd_mode!r}")
        if self.spectral.basis_cache not in ("cache", "recompute"):
            raise ValueError(
                f"comm_eff.spectral.basis_cache must be one of (cache, recompute); got {self.spectral.basis_cache!r}"
            )
        # EXP-14 periodic clean-step cadence. 0 = off (strict no-op for the
        # disabled path and every pre-EXP-14 config). A negative value is a
        # config error, not a silent disable.
        if self.clean_cadence < 0:
            raise ValueError(f"comm_eff.clean_cadence must be >= 0; got {self.clean_cadence}")
