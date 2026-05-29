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
        cadence (int): EXP-16 spectral-correction cadence in optimizer steps.
            The grad-correction hook (``_maybe_comm_eff_grad_correction``) fires
            only when ``(spectral_step % cadence) == 0`` on the monotonic
            per-optimizer-step counter, MIRRORING the anchor cadence
            (``anchor_should_fire``) and the clean-step cadence
            (``CommEffState.is_clean_step``). ``1`` (default) fires EVERY step =
            the pre-EXP-16 behavior, so every prior method config and the
            disabled path are a STRICT no-op. Set ``> 1`` (e.g. ``2``) to align
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
        # EXP-16 spectral-correction cadence. 1 = fire every step (the pre-EXP-16
        # behavior, so every prior config and the disabled path stay a strict
        # no-op). A value < 1 is a config error (it would never fire), not a
        # silent disable — mirrors the anchor.cadence >= 1 contract above.
        if self.spectral.cadence < 1:
            raise ValueError(f"comm_eff.spectral.cadence must be >= 1; got {self.spectral.cadence}")
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
