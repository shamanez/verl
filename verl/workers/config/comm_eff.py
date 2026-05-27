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
    """

    enabled: bool = True
    p: float = 0.95
    seed: int = 0
    pp_size: int = 8


@dataclass
class CommEffAnchorConfig(BaseConfig):
    """Asynchronous unmasked-anchor-circuit sub-config (inert while disabled).

    Args:
        enabled (bool): Whether the anchor circuit runs. Gated by the parent
            ``comm_eff.enabled`` regardless of this value.
        every_n_steps (int): Cadence of anchor backward passes.
        ema_decay (float): EMA decay for the anchor circuit's running state.
    """

    enabled: bool = False
    every_n_steps: int = 1
    ema_decay: float = 0.99


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
        rank (int): Retained for back-compat with the EXP-4 scaffold schema
            (full thin SVD is used; low-rank truncation is a later ablation).
        damping (float): Legacy alias kept for schema back-compat; the active
            damping knob is ``tau``.
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
    """

    enabled: bool = False
    mask: CommEffMaskConfig = field(default_factory=CommEffMaskConfig)
    anchor: CommEffAnchorConfig = field(default_factory=CommEffAnchorConfig)
    spectral: CommEffSpectralConfig = field(default_factory=CommEffSpectralConfig)

    def __post_init__(self):
        """Validate comm_eff configuration parameters.

        Validation only — no allocation, no RNG. When ``enabled=false`` this
        must remain free of numerical side effects.
        """
        if not 0.0 <= self.mask.p <= 1.0:
            raise ValueError(f"comm_eff.mask.p must be in [0, 1]; got {self.mask.p}")
        if self.mask.pp_size < 1:
            raise ValueError(f"comm_eff.mask.pp_size must be >= 1; got {self.mask.pp_size}")
        if self.spectral.rank < 1:
            raise ValueError(f"comm_eff.spectral.rank must be >= 1; got {self.spectral.rank}")
        if not 0.0 <= self.spectral.alpha <= 1.0:
            raise ValueError(f"comm_eff.spectral.alpha must be in [0, 1]; got {self.spectral.alpha}")
        if self.spectral.tau <= 0.0:
            raise ValueError(f"comm_eff.spectral.tau must be > 0; got {self.spectral.tau}")
        if not 0.0 <= self.spectral.beta_anc <= 1.0:
            raise ValueError(f"comm_eff.spectral.beta_anc must be in [0, 1]; got {self.spectral.beta_anc}")
        if not 0.0 <= self.anchor.ema_decay <= 1.0:
            raise ValueError(f"comm_eff.anchor.ema_decay must be in [0, 1]; got {self.anchor.ema_decay}")
