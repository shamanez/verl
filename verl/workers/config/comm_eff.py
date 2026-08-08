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

"""Configuration for the communication-efficient GRPO pipeline.

The method has one activation codec and one anchor-gradient merger:

* PowerSGD projects pipeline-boundary activations through a shared rank-r basis.
* A delayed, paired dense anchor owns Q and maintains the dense gradient EMA M.
* ``signed_ema`` combines the compressed fast gradient with the sign of M.
* ``rank1_relex`` projects delayed anchor weights to the current optimizer tick.

The top-level switch remains disabled by default, so dense GRPO is a strict
no-op path. The nested settings become active only when
``comm_eff.enabled=true``.
"""

import math
from dataclasses import dataclass, field

from verl.base_config import BaseConfig

__all__ = [
    "CommEffMaskConfig",
    "CommEffQuantConfig",
    "CommEffAnchorOptResetConfig",
    "CommEffAnchorConfig",
    "CommEffSpectralConfig",
    "CommEffPowerSGDConfig",
    "CommEffProbeConfig",
    "CommEffDCConfig",
    "CommEffConfig",
]

# The compression codecs ``comm_eff.compression_type`` may select. Exactly one
# codec is active per run (mutually exclusive). ``dense`` leaves the activation
# path uncompressed; ``prf_mask`` is the per-(token, dim) PRF Bernoulli mask;
# ``powersgd`` is the shared frozen-basis projector ``A_hat = (A @ Q) @ Qᵀ``;
# ``sr_quant`` is the dense low-bit stochastic-rounding boundary quantizer.
COMPRESSION_TYPES = ("dense", "prf_mask", "powersgd", "sr_quant")


@dataclass
class CommEffMaskConfig(BaseConfig):
    """Per-(token, dimension) pipeline-boundary activation masking (inert while disabled).

    A deterministic PRF Bernoulli mask applied in-graph (``h_tilde = h * mask``)
    to the boundary decoder blocks, only on the actor-train forward (and the
    old-logprob recompute when ``mask_recompute``). Each token independently
    keeps ``round((1-p)*H)`` dims; the mask is keyed on each token's stable
    ``(sample_id, position_id)`` so it is packing-invariant across the
    differently-packed forwards. See ``verl.workers.comm_eff.activation_mask``.

    This codec is selected by ``comm_eff.compression_type='prf_mask'`` (or, for
    back-compat, by ``mask.enabled=true`` with ``p>0`` while
    ``compression_type`` is left at its ``powersgd``/``dense`` default and the
    resolver falls through). It is mutually exclusive with the PowerSGD codec and
    anchor-independent (the anchor pass is never masked; it cannot own ``Q``).

    Args:
        enabled (bool): Whether masking runs (still gated by ``comm_eff.enabled``
            and by the codec resolving to ``prf_mask``).
        p (float): Masked (zeroed) fraction in ``[0, 1]``; ``comm_eff/mask_ratio``
            tracks it. ``0.0`` means no masking.
        seed (int): Base seed folded into the mask PRF key.
        pp_size (int): Logical pipeline-shard count; the last block of every
            shard except the final one is masked. ``L=16, pp_size=8`` ->
            ``[1, 3, 5, 7, 9, 11, 13]``.
        mask_recompute (bool): When ``True`` the mask also fires on the
            old-logprob recompute so both gradient-feeding forwards are masked
            with the identical per-token mask (keeps the PPO importance ratio
            ~1 at the first inner step). ``False`` (default) masks only the
            train forward.
        mask_reference (bool): When ``True`` the mask also fires on the frozen
            reference-policy forward (reference-KL loss), keyed on the SAME
            within-step ``(sample_id, position_id)`` as the policy forwards, so
            KL(current || ref) is a codec-vs-codec quantity (masked-current vs
            masked-reference), directly comparable to the PowerSGD codec's
            ``compress_reference`` circuit. ``False`` (default) leaves the
            reference forward dense (masked-current vs dense-reference).
        dense_every (int): Periodic full-fidelity step (issue #93, default ``0``
            = off). When ``N > 0`` the codec is bypassed entirely on every
            trainer step where ``global_step % N == 0``: the boundary hook
            returns the RAW activation on every path (train, old-logprob
            recompute and reference alike), so that step's forward AND backward
            are uncompressed and the fast circuit takes one ordinary RLVR
            update. The anchor is suppressed on the same step (see
            ``_comm_eff_maybe_anchor_refresh``), because the anchor computes a
            correction to a COMPRESSED gradient and there is no compression
            error to correct on a dense step. Bypassed steps are visible in
            WandB as a FLAT ``comm_eff/mask_applications/<tag>`` counter across
            the step, since the hook never fires. Wire accounting: a dense step
            sends ``H`` numbers per token instead of ``(1-p)*H``, so at
            ``H=1536``, ``k=77`` and ``N=50`` the average boundary payload rises
            from 1232 to about 1699 bits/token, 1.38x. Under the deployment
            premise in ``CLAUDE.md`` that cost is only real if the dense pass
            crosses the constrained link rather than running in the central
            mesh, which is the same accounting already applied to the anchor.
        rescale (bool): ``False`` (default) writes the raw product ``h * mask``;
            ``True`` applies inverted-dropout ``h * mask / (1 - p)`` so
            ``E[h_tilde] = h`` (requires ``p < 1``). Honored only when
            ``rescale_mode == "auto"``.
        rescale_mode (str): Magnitude-restoration scheme applied to ``h*mask``:
            ``none`` (raw product), ``constant`` (``1/(1-p)`` inverted dropout),
            ``rms_match`` (per-token exact RMS match), or ``auto`` (``constant``
            if ``rescale`` else ``none``).
        exact_k (bool): Issue #89 lever 2 (default off). Keep EXACTLY
            ``round((1-p)*H)`` channels per token via the per-token PRF hash
            order statistic (random, not a value top-k), so ``mask_ratio`` equals
            ``1-k/H`` exactly with no per-token Bernoulli variance.
        antithetic (bool): Issue #89 lever 5 (default off). Step ``t+1`` keeps
            the antithetic complement of step ``t`` (shared draw flipped
            ``u->1-u`` across the pair); only the cross-step draw changes, the
            within-step mask is identical across the old/train/reference forwards.
        p_by_boundary (list): Issue #89 lever 4 (default empty = off). A
            per-boundary vector of masked fractions (each in ``[0, 1]``); its
            length must equal the masked-boundary count. Empty means the scalar
            ``p`` applies to every boundary.
        frlr (bool): Issue #89 FRLR lever (default off). Fresh-Residual
            Low-Rank codec ("32+44+1"): each boundary activation ``h`` is
            reconstructed as ``h_hat = l + gamma * scatter_J(res_J)`` where
            ``l = (h @ Q) @ Q^T`` uses a step-frozen activation-derived
            orthonormal ``Q`` (H x frlr_rank, warm-started like the PowerSGD
            projector and refreshed at step boundaries from the previous
            step's activation sketch), ``J`` is a per-token PRF-fresh EXACT-k
            residual channel subset (keyed like the baseline mask INCLUDING
            ``global_step``), and ``gamma`` is a DETACHED per-token
            residual-norm-matching gain capped at ``H/frlr_k``. Payload:
            ``frlr_rank + frlr_k + 1`` values/token (32+44+1 = 77 of 1536,
            mask_ratio ~ 0.9499). Mutually exclusive with
            ``exact_k``/``antithetic``/``p_by_boundary`` and requires the
            plain rescale path off (``rescale=false``, ``rescale_mode``
            ``none``/``auto``).
        frlr_rank (int): FRLR core rank r (columns of ``Q``); default 32.
        frlr_k (int): FRLR per-token residual subset size k; default 44.
        frlr_unbiased (bool): FRLR unbiased mode (default off): apply the
            constant ``H/frlr_k`` gain with no norm matching, so
            ``E[h_hat | h, Q] = h``.
        frlr_q_cadence (int): Issue #89 slow-Q lever (default 1 = the original
            every-step refresh, bit-identical). Refresh the FRLR core ``Q``
            only when ``global_step - last_refresh_step >= frlr_q_cadence`` at
            the lazy refresh point (first hook fire of a new step); between
            refreshes ``Q`` stays FROZEN (bitwise) while the activation sketch
            keeps accumulating, so each refresh consumes the FULL window's
            sketch. Motivation: the first FRLR GPU trial cut codec-view
            entropy 63% but its reference-KL accelerated (0.005@9 -> 0.33@30)
            because the per-step activation-refit ``Q`` chases the drifting
            policy (a non-stationary codec view); a slow cadence keeps the
            core stable between refreshes while the fresh per-step PRF
            residual keeps repairing the stale-Q nullspace.
    """

    enabled: bool = False
    p: float = 0.95
    seed: int = 0
    pp_size: int = 8
    mask_recompute: bool = False
    mask_reference: bool = False
    dense_every: int = 0
    rescale: bool = False
    rescale_mode: str = "auto"
    exact_k: bool = False
    antithetic: bool = False
    p_by_boundary: list = field(default_factory=list)
    frlr: bool = False
    frlr_rank: int = 32
    frlr_k: int = 44
    frlr_unbiased: bool = False
    frlr_q_cadence: int = 1


@dataclass
class CommEffQuantConfig(BaseConfig):
    """Dense stochastic-rounding boundary-activation quantization (inert while disabled).

    The sr_quant codec, selected by ``comm_eff.compression_type='sr_quant'``:
    every pipeline-boundary hidden-state channel is quantized in-graph to
    ``bits`` bits per value with blockwise absmax scales and unbiased stochastic
    rounding (``E[q] = h``), and the upstream gradient at the same boundary is
    quantized the same way on backward (``E[g_hat] = g``, fresh PRF
    ``direction`` subkey). ``rounding='rn'`` swaps in deterministic
    round-to-nearest (biased; the ablation control). See
    ``verl.workers.comm_eff.activation_quant``.

    Knob reuse: sr_quant reuses the ``mask`` sub-config for eligibility and
    keying: ``mask.mask_recompute`` / ``mask.mask_reference`` widen the
    eligible path tags exactly as for prf_mask, and ``mask.seed`` /
    ``mask.pp_size`` provide the PRF base seed and the boundary placement.
    ``mask.p`` / ``rescale*`` / ``exact_k`` / ``antithetic`` / ``frlr*`` are
    IGNORED by sr_quant. Like prf_mask, sr_quant carries no PowerSGD basis, so
    the anchor cannot own ``Q`` under this codec (``anchor.owns_q=false``).

    Args:
        bits (int): Quantization width per channel; ``L = 2**bits`` uniform
            levels span ``[-s, +s]`` per (token, block). Default 1 (sign-like:
            ``q in {-s, +s}`` with ``P(+s) = (h/s + 1)/2``).
        block_size (int): Channels per absmax-scale block within a token
            (QSGD-style bucketing). ``0`` (or ``>= hidden_size``) means one
            whole-token scale. Default 32. Logical PP bits per token per
            boundary: ``hidden_size*bits + ceil(hidden_size/block_size)*16``
            (one fp16 scale per block).
        rounding (str): ``sr`` (default) = unbiased PRF-keyed stochastic
            rounding; ``rn`` = deterministic round-to-nearest to the same level
            grid (no PRF draw, trivially pass-identical, biased: the required
            ablation control).
        subset_k (int): ``0`` (default) = full-width quantization of all ``H``
            channels. ``> 0`` (issue #93 I5, the byte-parity hybrid): per token
            quantize only a PRF-fresh EXACT-``subset_k`` channel subset ``J``
            (drawn with the mask codec's order-statistic machinery, keyed
            identically, so ``J`` is bit-identical across the old/train/ref
            passes of one step), zero elsewhere, rescale by ``H/subset_k``
            (``E[q] = h`` through both the subset draw and the rounding).
            Blocks then span ``subset_k`` consecutive KEPT channels; logical PP
            bits per token per boundary become
            ``subset_k*bits + subset_k*16/block_size`` (pro-rata tail).
    """

    bits: int = 1
    block_size: int = 32
    rounding: str = "sr"
    subset_k: int = 0


# The overwrite modes ``comm_eff.anchor.opt_reset.mode`` may select.
# ``anchor_moments`` writes the (optionally norm-matched) anchor-maintained
# AdamW moments over the fast moments; ``zero`` zeroes both fast moments.
OPT_RESET_MODES = ("anchor_moments", "zero")


@dataclass
class CommEffAnchorOptResetConfig(BaseConfig):
    """Anchor-sourced optimizer-state reset for the fast circuit.

    Under a lossy activation codec the fast AdamW moments are built from
    codec-noised gradients, so compression bias can accumulate in the optimizer
    state itself. When enabled, the anchor circuit maintains parallel fp32 CPU
    moments from its clean, DP-averaged dense replay gradients (the same
    tensors that feed the signed EMA ``M``), and every ``cadence`` optimizer
    ticks — after ``optimizer.step()`` and after any anchor fire on the same
    tick — the fast ``exp_avg``/``exp_avg_sq`` are overwritten with them.

    Args:
        enabled (bool): Off (default) is a strict no-op: no anchor-side moment
            state is allocated and the optimizer is never touched.
        cadence (int): Optimizer ticks between resets (the same
            ``state.anchor_step`` units the anchor cadence counts).
        mode (str): ``anchor_moments`` overwrites ``exp_avg`` with
            ``rho * m_anc`` and ``exp_avg_sq`` with ``rho^2 * v_anc``;
            ``zero`` zeroes both moments.
        beta1 (float): Per-fire EMA coefficient for the anchor ``exp_avg``
            (``m <- beta1*m + (1-beta1)*G_anc``); must be in (0, 1).
        beta2 (float): Per-fire EMA coefficient for the anchor ``exp_avg_sq``
            (``v <- beta2*v + (1-beta2)*G_anc^2``); must be in (0, 1).
        scale_match (bool): When true, ``rho`` matches the global L2 of the
            fast ``exp_avg`` set (``rho = ||exp_avg|| / (||m_anc|| + 1e-12)``);
            false uses ``rho = 1``.
    """

    enabled: bool = False
    cadence: int = 50
    mode: str = "anchor_moments"
    beta1: float = 0.8
    beta2: float = 0.95
    scale_match: bool = True


@dataclass
class CommEffAnchorConfig(BaseConfig):
    """Delayed dense anchor and RELEX weight-projection configuration.

    ``rank1_relex`` fits a per-tensor rank-1 trajectory over exact checkpoints.
    Two checkpoints give the ordinary per-tensor secant; three or more fit the
    rank-1 trajectory using the checkpoints' actual ticks.

    ``lookahead_history_mode`` selects the delta base:

    * ``sliding_window`` (default): keep the last ``lookahead_window_snapshots``
      checkpoints; the base is the oldest snapshot still in the window and it
      advances as the window slides. Bounded memory; tracks local drift.
    * ``growing_fixed_base``: pin the seeded base for the whole run and keep
      appending checkpoints so the base-relative delta history grows
      (RELEX-faithful, a longer denoised lever arm). ``lookahead_max_snapshots``
      caps retention (``-1`` unbounded); one full-model CPU snapshot is kept per
      checkpoint, so memory grows with the run.
    """

    enabled: bool = True
    cadence: int = 20
    delay_K: int = 20
    owns_q: bool = True
    replay_paired_batch: bool = True
    batch_scope: str = "ppo_minibatch"
    snapshot_device: str = "cpu"
    lookahead_anchor: bool = True
    lookahead_mode: str = "rank1_relex"
    lookahead_strength: float = 1.0
    lookahead_rollout_source: str = "auto"
    warmup_mode: str = "stale_correct"
    lookahead_min_snapshots: int = 2
    lookahead_window_snapshots: int = 4
    lookahead_history_mode: str = "sliding_window"
    lookahead_max_snapshots: int = -1
    opt_reset: CommEffAnchorOptResetConfig = field(default_factory=CommEffAnchorOptResetConfig)


@dataclass
class CommEffSpectralConfig(BaseConfig):
    """Dense-anchor EMA and signed-gradient merger configuration."""

    enabled: bool = True
    beta_anc: float = 0.50
    cadence: int = 1
    diagnostics: bool = False
    target_scope: str = "all_floating"
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
    max_targets: int = -1
    ema_device: str = "cpu"
    signed_ema_alpha: float = 0.25


@dataclass
class CommEffPowerSGDConfig(BaseConfig):
    """PowerSGD boundary-activation projection configuration.

    For a boundary activation ``A`` and detached orthonormal basis ``Q``, the
    live forward uses ``A_hat = (A @ Q) @ Q.T``. The rank controls the width of
    the transmitted representation.
    """

    enabled: bool = True
    rank: int = 77
    seed: int = 0
    pp_size: int = 8
    update_cadence: int = 1
    warm_start: bool = True
    compress_recompute: bool = True
    # Run the frozen reference-policy forward (reference-KL loss) through the same
    # compressed PowerSGD circuit as the paired actor forwards, so KL(current||ref)
    # is measured on one shared basis. The reference forward is read-only
    # (forward_only, grad disabled): it consumes the current anchor-owned Q and
    # never folds the sketch or advances Q. Default true (faithful deployment).
    # Set false for a dense-reference control arm.
    compress_reference: bool = True
    sync_basis: bool = True
    qr_dtype: str = "fp32"
    reortho_eps: float = 1e-6
    fast_q_bootstrap: bool = True


@dataclass
class CommEffProbeConfig(BaseConfig):
    """Dense-view probe + adaptive reference-KL coefficient (issue #93, I3).

    Every ``probe_every`` trainer steps the trainer reruns the current batch's
    actor and reference log-prob computations once with every comm_eff codec
    silent (path tag ``None``, ``compression_active`` untouched): a pure
    measurement pass, no backward, no weight change. It logs
    ``probe/kl_dense`` (token-mean ``kl_loss_type`` estimate of
    KL(pi_theta_dense || pi_ref_dense), the same estimator + aggregation as
    ``actor/kl_loss``), ``probe/kl_gain`` (actor/kl_loss / probe/kl_dense, the
    measured G(t)), and ``probe/gap_dense`` (token-mean
    ``rollout_log_probs - dense actor log probs``, the dense-view
    train-inference gap).

    ``ctrl_enabled`` closes the loop: projected dual ascent in log space with
    proportional damping retunes the reference-KL coefficient once per probe,
    ``beta <- clip(beta * exp(ki*e + kp*(e - e_prev)), beta_min, beta_max)``
    with ``e = (kl_dense - c)/c`` and setpoint
    ``c = max(kl_target_floor, kl_target_gain * table(step))``; anti-windup is
    conditional integration (the integral term freezes while pinned at a
    bound). ``beta_0`` is the actor's static ``kl_loss_coef``. See
    ``verl.trainer.ppo.comm_eff_control``.

    Args:
        probe_every (int): Probe cadence in trainer global steps; 0 (default)
            disables the probe entirely (bit-identical trainer path).
        ctrl_enabled (bool): Enable the adaptive KL coefficient controller
            (requires ``probe_every >= 1``). Off (default) leaves the loss on
            the static ``kl_loss_coef``.
        kl_target_table (str): ``"step:value,step:value"`` dense-control
            reference-KL curve (baked from the finished dense run at matched
            steps); linear interpolation with edge clamping. Empty (default)
            means the floor alone sets the target.
        kl_target_floor (float): Setpoint floor ``c_floor`` in nats.
        kl_target_gain (float): Multiplier on the interpolated dense KL
            (``2.0`` = "hold compressed dense-view KL <= 2x dense control").
        ctrl_ki (float): Integral (dual-ascent) gain on ``e_k``.
        ctrl_kp (float): Proportional damping gain on ``e_k - e_{k-1}``.
        ctrl_beta_min (float): Lower projection bound for beta.
        ctrl_beta_max (float): Upper projection bound for beta.
    """

    probe_every: int = 0
    ctrl_enabled: bool = False
    kl_target_table: str = ""
    kl_target_floor: float = 0.005
    kl_target_gain: float = 2.0
    ctrl_ki: float = 0.3
    ctrl_kp: float = 0.1
    ctrl_beta_min: float = 2.0e-4
    ctrl_beta_max: float = 0.05


@dataclass
class CommEffDCConfig(BaseConfig):
    """DC-GRPO advantage shaping (issue #93 4.7b, arXiv 2606.08779).

    Once per trainer step, after advantages exist and before ``update_actor``,
    the driver computes the per-response-token probability discrepancy
    ``delta_t = |exp(old_log_probs) - exp(rollout_log_probs)|`` (the codec-view
    trainer probability vs the sampler's; bounded, ratio-free, so it stays
    numerically alive where importance ratios die) and shapes
    ``A_t <- A_t - lambda * delta_t`` on response tokens only. The dual
    variable then takes one projected ascent step,
    ``lambda <- clip(lambda + eta * (delta_bar - target), 0, lambda_max)``
    with ``delta_bar`` the response-masked mean of ``delta_t``: lambda grows
    while the gap exceeds the target and decays toward 0 below it, regulating
    the GROWTH of the gap without fighting its static part. Zero extra forward
    passes, zero wire cost; logs ``dc/lambda`` (applied) and ``dc/delta_bar``.

    Args:
        enabled (bool): Enable DC advantage shaping. Off (default) is a strict
            trainer no-op (advantages untouched, no metrics).
        eta (float): Dual ascent step size on ``delta_bar - target``.
        target (float): Per-token discrepancy setpoint ``c_gap`` = the measured
            step-1 static floor plus slack. Deliberately NO default magic:
            the -1.0 sentinel is rejected when enabled; measure, then set.
        lambda0 (float): Initial shaping strength lambda.
        lambda_max (float): Upper projection bound for lambda.
    """

    enabled: bool = False
    eta: float = 1.0
    target: float = -1.0
    lambda0: float = 0.05
    lambda_max: float = 1.0


@dataclass
class CommEffConfig(BaseConfig):
    """Top-level communication-efficient method configuration.

    ``enabled=false`` is the dense compatibility path. ``compression_type`` may
    also be set to ``dense`` explicitly for control arms; no alternative lossy
    codec is registered.
    """

    enabled: bool = False
    compression_type: str = "powersgd"
    mask: CommEffMaskConfig = field(default_factory=CommEffMaskConfig)
    quant: CommEffQuantConfig = field(default_factory=CommEffQuantConfig)
    anchor: CommEffAnchorConfig = field(default_factory=CommEffAnchorConfig)
    spectral: CommEffSpectralConfig = field(default_factory=CommEffSpectralConfig)
    powersgd: CommEffPowerSGDConfig = field(default_factory=CommEffPowerSGDConfig)
    probe: CommEffProbeConfig = field(default_factory=CommEffProbeConfig)
    dc: CommEffDCConfig = field(default_factory=CommEffDCConfig)

    def __post_init__(self):
        """Validate without allocating tensors, communicating, or drawing RNG."""

        if not isinstance(self.enabled, bool):
            raise ValueError(f"comm_eff.enabled must be a bool; got {self.enabled!r}")
        if self.compression_type not in COMPRESSION_TYPES:
            raise ValueError(
                f"comm_eff.compression_type must be one of {COMPRESSION_TYPES}; got {self.compression_type!r}"
            )

        self._validate_mask()
        self._validate_quant()
        self._validate_anchor()
        self._validate_spectral()
        self._validate_powersgd()
        self._validate_probe()
        self._validate_dc()
        self._validate_cross_circuit_contract()

    def _validate_mask(self) -> None:
        """Validate the PRF activation-mask sub-config (no allocation, no RNG)."""

        if not isinstance(self.mask.enabled, bool):
            raise ValueError(f"comm_eff.mask.enabled must be a bool; got {self.mask.enabled!r}")
        if not 0.0 <= self.mask.p <= 1.0:
            raise ValueError(f"comm_eff.mask.p must be in [0, 1]; got {self.mask.p}")
        if self.mask.pp_size < 1:
            raise ValueError(f"comm_eff.mask.pp_size must be >= 1; got {self.mask.pp_size}")
        if not isinstance(self.mask.mask_recompute, bool):
            raise ValueError(f"comm_eff.mask.mask_recompute must be a bool; got {self.mask.mask_recompute!r}")
        if not isinstance(self.mask.mask_reference, bool):
            raise ValueError(f"comm_eff.mask.mask_reference must be a bool; got {self.mask.mask_reference!r}")
        if int(self.mask.dense_every) < 0:
            raise ValueError(f"comm_eff.mask.dense_every must be >= 0 (0 = off); got {self.mask.dense_every!r}")
        if not isinstance(self.mask.rescale, bool):
            raise ValueError(f"comm_eff.mask.rescale must be a bool; got {self.mask.rescale!r}")
        if str(self.mask.rescale_mode).lower() not in ("none", "constant", "rms_match", "auto"):
            raise ValueError(
                "comm_eff.mask.rescale_mode must be one of (none, constant, rms_match, auto); "
                f"got {self.mask.rescale_mode!r}"
            )
        if self.mask.rescale and self.mask.p >= 1.0:
            raise ValueError(
                "comm_eff.mask.rescale=true requires comm_eff.mask.p < 1.0 (the 1/(1-p) "
                f"magnitude-preservation factor is undefined at p>=1); got p={self.mask.p}"
            )
        if not isinstance(self.mask.exact_k, bool):
            raise ValueError(f"comm_eff.mask.exact_k must be a bool; got {self.mask.exact_k!r}")
        if not isinstance(self.mask.antithetic, bool):
            raise ValueError(f"comm_eff.mask.antithetic must be a bool; got {self.mask.antithetic!r}")
        pbb = self.mask.p_by_boundary if self.mask.p_by_boundary is not None else []
        try:
            pbb_vals = list(pbb)
        except TypeError:
            raise ValueError(
                f"comm_eff.mask.p_by_boundary must be a list of floats in [0, 1]; got {self.mask.p_by_boundary!r}"
            ) from None
        for v in pbb_vals:
            if isinstance(v, bool) or not isinstance(v, int | float):
                raise ValueError(f"comm_eff.mask.p_by_boundary entries must be numbers in [0, 1]; got {v!r}")
            if not 0.0 <= float(v) <= 1.0:
                raise ValueError(f"comm_eff.mask.p_by_boundary entries must be in [0, 1]; got {v}")
        # FRLR (issue #89) lever.
        for name in ("frlr", "frlr_unbiased"):
            value = getattr(self.mask, name)
            if not isinstance(value, bool):
                raise ValueError(f"comm_eff.mask.{name} must be a bool; got {value!r}")
        for name in ("frlr_rank", "frlr_k", "frlr_q_cadence"):
            value = getattr(self.mask, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"comm_eff.mask.{name} must be an integer >= 1; got {value!r}")
        if self.mask.frlr:
            if self.mask.exact_k or self.mask.antithetic or pbb_vals:
                raise ValueError(
                    "comm_eff.mask.frlr=true is mutually exclusive with "
                    "exact_k/antithetic/p_by_boundary; FRLR draws its own PRF-fresh "
                    "exact-k residual subset J."
                )
            if self.mask.rescale or str(self.mask.rescale_mode).lower() not in ("none", "auto"):
                raise ValueError(
                    "comm_eff.mask.frlr=true requires the plain-mask rescale path OFF "
                    "(rescale=false, rescale_mode none|auto); FRLR applies its own detached "
                    f"residual-norm matching. Got rescale={self.mask.rescale}, "
                    f"rescale_mode={self.mask.rescale_mode!r}."
                )

    def _validate_quant(self) -> None:
        """Validate the sr_quant sub-config (no allocation, no RNG)."""

        bits = self.quant.bits
        if isinstance(bits, bool) or not isinstance(bits, int) or not 1 <= bits <= 16:
            raise ValueError(f"comm_eff.quant.bits must be an integer in [1, 16]; got {bits!r}")
        block_size = self.quant.block_size
        if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size < 0:
            raise ValueError(
                f"comm_eff.quant.block_size must be an integer >= 0 (0 = whole-token scale); got {block_size!r}"
            )
        if str(self.quant.rounding) not in ("sr", "rn"):
            raise ValueError(f"comm_eff.quant.rounding must be one of (sr, rn); got {self.quant.rounding!r}")
        subset_k = self.quant.subset_k
        if isinstance(subset_k, bool) or not isinstance(subset_k, int) or subset_k < 0:
            raise ValueError(f"comm_eff.quant.subset_k must be an integer >= 0 (0 = full-width); got {subset_k!r}")

    def _validate_anchor(self) -> None:
        from verl.workers.comm_eff.lookahead import (
            HISTORY_MODES,
            LOOKAHEAD_MODES,
            LOOKAHEAD_ROLLOUT_SOURCES,
            MODE_GROWING_FIXED_BASE,
            lookahead_enabled,
            rank1_relex_enabled,
        )

        for name in ("enabled", "owns_q", "replay_paired_batch", "lookahead_anchor"):
            value = getattr(self.anchor, name)
            if not isinstance(value, bool):
                raise ValueError(f"comm_eff.anchor.{name} must be a bool; got {value!r}")
        if self.anchor.cadence < 1:
            raise ValueError(f"comm_eff.anchor.cadence must be >= 1; got {self.anchor.cadence}")
        if self.anchor.delay_K < 0:
            raise ValueError(f"comm_eff.anchor.delay_K must be >= 0; got {self.anchor.delay_K}")
        if self.anchor.batch_scope not in ("ppo_minibatch", "rollout_batch"):
            raise ValueError(
                "comm_eff.anchor.batch_scope must be one of (ppo_minibatch, rollout_batch); "
                f"got {self.anchor.batch_scope!r}"
            )
        if self.anchor.snapshot_device != "cpu":
            raise ValueError(f"comm_eff.anchor.snapshot_device must be 'cpu'; got {self.anchor.snapshot_device!r}")
        if self.anchor.lookahead_mode not in LOOKAHEAD_MODES:
            raise ValueError(
                f"comm_eff.anchor.lookahead_mode must be one of {LOOKAHEAD_MODES}; got {self.anchor.lookahead_mode!r}"
            )
        if self.anchor.lookahead_strength < 0.0:
            raise ValueError(f"comm_eff.anchor.lookahead_strength must be >= 0; got {self.anchor.lookahead_strength}")
        if self.anchor.lookahead_rollout_source not in LOOKAHEAD_ROLLOUT_SOURCES:
            raise ValueError(
                "comm_eff.anchor.lookahead_rollout_source must be one of "
                f"{LOOKAHEAD_ROLLOUT_SOURCES}; got {self.anchor.lookahead_rollout_source!r}"
            )
        if self.anchor.lookahead_rollout_source == "current_step" and not lookahead_enabled(self.anchor):
            raise ValueError(
                "comm_eff.anchor.lookahead_rollout_source='current_step' requires active rank1_relex projection"
            )
        if self.anchor.warmup_mode not in ("stale_correct", "q_only"):
            raise ValueError(
                f"comm_eff.anchor.warmup_mode must be one of (stale_correct, q_only); got {self.anchor.warmup_mode!r}"
            )
        if isinstance(self.anchor.lookahead_window_snapshots, bool) or not isinstance(
            self.anchor.lookahead_window_snapshots, int
        ):
            raise ValueError(
                "comm_eff.anchor.lookahead_window_snapshots must be an integer >= 2; "
                f"got {self.anchor.lookahead_window_snapshots!r}"
            )
        if self.anchor.lookahead_window_snapshots < 2:
            raise ValueError(
                f"comm_eff.anchor.lookahead_window_snapshots must be >= 2; got {self.anchor.lookahead_window_snapshots}"
            )
        if isinstance(self.anchor.lookahead_min_snapshots, bool) or not isinstance(
            self.anchor.lookahead_min_snapshots, int
        ):
            raise ValueError(
                "comm_eff.anchor.lookahead_min_snapshots must be -1 or an integer in [2, W]; "
                f"got {self.anchor.lookahead_min_snapshots!r}"
            )
        if self.anchor.lookahead_min_snapshots != -1:
            if not rank1_relex_enabled(self.anchor):
                raise ValueError("comm_eff.anchor.lookahead_min_snapshots is only meaningful with active rank1_relex")
            if not 2 <= self.anchor.lookahead_min_snapshots <= self.anchor.lookahead_window_snapshots:
                raise ValueError(
                    "comm_eff.anchor.lookahead_min_snapshots must be -1 or in "
                    f"[2, {self.anchor.lookahead_window_snapshots}]; got {self.anchor.lookahead_min_snapshots}"
                )
        if self.anchor.lookahead_history_mode not in HISTORY_MODES:
            raise ValueError(
                f"comm_eff.anchor.lookahead_history_mode must be one of {HISTORY_MODES}; "
                f"got {self.anchor.lookahead_history_mode!r}"
            )
        if isinstance(self.anchor.lookahead_max_snapshots, bool) or not isinstance(
            self.anchor.lookahead_max_snapshots, int
        ):
            raise ValueError(
                "comm_eff.anchor.lookahead_max_snapshots must be -1 (unbounded) or an integer "
                f">= lookahead_window_snapshots; got {self.anchor.lookahead_max_snapshots!r}"
            )
        if self.anchor.lookahead_history_mode == MODE_GROWING_FIXED_BASE:
            if not rank1_relex_enabled(self.anchor):
                raise ValueError(
                    "comm_eff.anchor.lookahead_history_mode='growing_fixed_base' requires active rank1_relex"
                )
            if self.anchor.lookahead_max_snapshots != -1 and (
                self.anchor.lookahead_max_snapshots < self.anchor.lookahead_window_snapshots
            ):
                raise ValueError(
                    "comm_eff.anchor.lookahead_max_snapshots must be -1 or "
                    f">= lookahead_window_snapshots={self.anchor.lookahead_window_snapshots}; "
                    f"got {self.anchor.lookahead_max_snapshots}"
                )
        elif self.anchor.lookahead_max_snapshots != -1:
            raise ValueError(
                "comm_eff.anchor.lookahead_max_snapshots is only meaningful with "
                "lookahead_history_mode='growing_fixed_base'; leave it at -1 for sliding_window"
            )
        opt_reset = self.anchor.opt_reset
        for name in ("enabled", "scale_match"):
            value = getattr(opt_reset, name)
            if not isinstance(value, bool):
                raise ValueError(f"comm_eff.anchor.opt_reset.{name} must be a bool; got {value!r}")
        if isinstance(opt_reset.cadence, bool) or not isinstance(opt_reset.cadence, int) or opt_reset.cadence < 1:
            raise ValueError(f"comm_eff.anchor.opt_reset.cadence must be an integer >= 1; got {opt_reset.cadence!r}")
        if opt_reset.mode not in OPT_RESET_MODES:
            raise ValueError(
                f"comm_eff.anchor.opt_reset.mode must be one of {OPT_RESET_MODES}; got {opt_reset.mode!r}"
            )
        for name in ("beta1", "beta2"):
            value = getattr(opt_reset, name)
            if not 0.0 < float(value) < 1.0:
                raise ValueError(f"comm_eff.anchor.opt_reset.{name} must be in (0, 1); got {value}")

    def _validate_spectral(self) -> None:
        for name in ("enabled", "diagnostics"):
            value = getattr(self.spectral, name)
            if not isinstance(value, bool):
                raise ValueError(f"comm_eff.spectral.{name} must be a bool; got {value!r}")
        if not 0.0 <= self.spectral.beta_anc <= 1.0:
            raise ValueError(f"comm_eff.spectral.beta_anc must be in [0, 1]; got {self.spectral.beta_anc}")
        if self.spectral.cadence < 1:
            raise ValueError(f"comm_eff.spectral.cadence must be >= 1; got {self.spectral.cadence}")
        if self.spectral.target_scope not in ("decoder_matrices", "all_floating"):
            raise ValueError(
                "comm_eff.spectral.target_scope must be one of (decoder_matrices, all_floating); "
                f"got {self.spectral.target_scope!r}"
            )
        if self.spectral.max_targets < -1:
            raise ValueError(f"comm_eff.spectral.max_targets must be -1 or >= 0; got {self.spectral.max_targets}")
        if self.spectral.ema_device not in ("gpu", "cpu"):
            raise ValueError(
                f"comm_eff.spectral.ema_device must be one of (gpu, cpu); got {self.spectral.ema_device!r}"
            )
        if not 0.0 <= self.spectral.signed_ema_alpha <= 1.0:
            raise ValueError(
                f"comm_eff.spectral.signed_ema_alpha must be in [0, 1]; got {self.spectral.signed_ema_alpha}"
            )

    def _validate_powersgd(self) -> None:
        for name in (
            "enabled",
            "warm_start",
            "compress_recompute",
            "compress_reference",
            "sync_basis",
            "fast_q_bootstrap",
        ):
            value = getattr(self.powersgd, name)
            if not isinstance(value, bool):
                raise ValueError(f"comm_eff.powersgd.{name} must be a bool; got {value!r}")
        if self.powersgd.rank < 1:
            raise ValueError(f"comm_eff.powersgd.rank must be >= 1; got {self.powersgd.rank}")
        if self.powersgd.pp_size < 1:
            raise ValueError(f"comm_eff.powersgd.pp_size must be >= 1; got {self.powersgd.pp_size}")
        if self.powersgd.update_cadence < 1:
            raise ValueError(f"comm_eff.powersgd.update_cadence must be >= 1; got {self.powersgd.update_cadence}")
        if self.powersgd.qr_dtype != "fp32":
            raise ValueError(f"comm_eff.powersgd.qr_dtype must be 'fp32'; got {self.powersgd.qr_dtype!r}")
        if self.powersgd.reortho_eps <= 0.0:
            raise ValueError(f"comm_eff.powersgd.reortho_eps must be > 0; got {self.powersgd.reortho_eps}")

    def _validate_probe(self) -> None:
        """Validate the dense-view probe / controller sub-config (no allocation)."""

        probe = self.probe
        if isinstance(probe.probe_every, bool) or not isinstance(probe.probe_every, int) or probe.probe_every < 0:
            raise ValueError(f"comm_eff.probe.probe_every must be an integer >= 0 (0 = off); got {probe.probe_every!r}")
        if not isinstance(probe.ctrl_enabled, bool):
            raise ValueError(f"comm_eff.probe.ctrl_enabled must be a bool; got {probe.ctrl_enabled!r}")
        if probe.ctrl_enabled and probe.probe_every < 1:
            raise ValueError(
                "comm_eff.probe.ctrl_enabled=true requires probe_every >= 1: the controller "
                "only updates at probes, so without a probe cadence it would never act."
            )
        if not isinstance(probe.kl_target_table, str):
            raise ValueError(
                f"comm_eff.probe.kl_target_table must be a 'step:value,...' string; got {probe.kl_target_table!r}"
            )
        # Import-light parser shared with the runtime controller; fails here so
        # a malformed table dies at config time, not at the first probe.
        from verl.trainer.ppo.comm_eff_control import parse_kl_target_table

        try:
            parse_kl_target_table(probe.kl_target_table)
        except ValueError as e:
            raise ValueError(f"comm_eff.probe.kl_target_table invalid: {e}") from None
        if probe.kl_target_floor <= 0.0:
            raise ValueError(
                f"comm_eff.probe.kl_target_floor must be > 0 (the setpoint divides the error); "
                f"got {probe.kl_target_floor}"
            )
        if probe.kl_target_gain <= 0.0:
            raise ValueError(f"comm_eff.probe.kl_target_gain must be > 0; got {probe.kl_target_gain}")
        for name in ("ctrl_ki", "ctrl_kp"):
            value = getattr(probe, name)
            if value < 0.0:
                raise ValueError(f"comm_eff.probe.{name} must be >= 0; got {value}")
        if not 0.0 < probe.ctrl_beta_min <= probe.ctrl_beta_max:
            raise ValueError(
                "comm_eff.probe requires 0 < ctrl_beta_min <= ctrl_beta_max; "
                f"got [{probe.ctrl_beta_min}, {probe.ctrl_beta_max}]"
            )

    def _validate_dc(self) -> None:
        """Validate the DC-GRPO advantage-shaping sub-config (no allocation)."""

        dc = self.dc
        if not isinstance(dc.enabled, bool):
            raise ValueError(f"comm_eff.dc.enabled must be a bool; got {dc.enabled!r}")
        if not math.isfinite(dc.eta) or dc.eta < 0.0:
            raise ValueError(f"comm_eff.dc.eta must be finite and >= 0; got {dc.eta}")
        if not math.isfinite(dc.lambda_max) or dc.lambda_max <= 0.0:
            raise ValueError(f"comm_eff.dc.lambda_max must be finite and > 0; got {dc.lambda_max}")
        if not math.isfinite(dc.lambda0) or not 0.0 <= dc.lambda0 <= dc.lambda_max:
            raise ValueError(f"comm_eff.dc.lambda0 must be in [0, lambda_max={dc.lambda_max}]; got {dc.lambda0}")
        if not math.isfinite(dc.target):
            raise ValueError(f"comm_eff.dc.target must be finite; got {dc.target}")
        # target has no default magic on purpose: it is the measured step-1
        # static per-token discrepancy floor plus slack. Reject the sentinel at
        # config time instead of silently pinning lambda at lambda_max.
        if dc.enabled and dc.target < 0.0:
            raise ValueError(
                "comm_eff.dc.enabled=true requires an explicit comm_eff.dc.target >= 0 "
                "(the measured step-1 static per-token discrepancy floor plus slack); "
                f"got {dc.target}"
            )

    def _validate_cross_circuit_contract(self) -> None:
        from verl.workers.comm_eff.lookahead import rank1_relex_enabled

        if not self.enabled:
            return
        # The PLAIN PRF activation mask is anchor-independent and carries no
        # basis at all (its mask is a PRF of seed/step/layer), so the anchor
        # cannot own a Q under it. FRLR is the exception: it does carry a
        # per-boundary basis Q, so the anchor CAN own it (issue #93), and then
        # the Q side channel rides the slow circuit instead of the boundary.
        if self.compression_type == "prf_mask" and self.anchor.owns_q:
            if not self.mask.frlr:
                raise ValueError(
                    "comm_eff.compression_type='prf_mask' requires anchor.owns_q=false unless "
                    "mask.frlr=true: the plain PRF activation mask has no basis Q for the "
                    "anchor to own."
                )
            if not self.anchor.enabled:
                raise ValueError(
                    "comm_eff: mask.frlr with anchor.owns_q=true requires anchor.enabled=true "
                    "so the FRLR basis has an updater (the fast path is gated off as a Q writer)"
                )
        # The SR boundary quantizer is likewise anchor-independent and carries
        # no PowerSGD basis, so the anchor cannot own a Q under this codec.
        if self.compression_type == "sr_quant" and self.anchor.owns_q:
            raise ValueError(
                "comm_eff.compression_type='sr_quant' requires anchor.owns_q=false: the SR "
                "boundary quantizer has no PowerSGD basis Q for the anchor to own."
            )
        # The reset moments are EMAs of the anchor circuit's dense replay
        # gradients, so without a firing anchor they would stay cold forever.
        if self.anchor.opt_reset.enabled and not self.anchor.enabled:
            raise ValueError(
                "comm_eff.anchor.opt_reset.enabled=true requires anchor.enabled=true: the reset "
                "moments are built from the anchor circuit's clean dense replay gradients."
            )
        if self.compression_type == "powersgd" and not self.powersgd.enabled:
            raise ValueError("comm_eff.compression_type='powersgd' requires powersgd.enabled=true")
        if self.compression_type == "powersgd" and self.anchor.owns_q and not self.anchor.enabled:
            raise ValueError(
                "comm_eff: anchor.owns_q=true requires anchor.enabled=true so the PowerSGD basis has an updater"
            )
        if rank1_relex_enabled(self.anchor):
            if not self.anchor.enabled:
                raise ValueError("comm_eff.anchor.lookahead_mode='rank1_relex' requires anchor.enabled=true")
            if not self.anchor.replay_paired_batch:
                raise ValueError(
                    "comm_eff.anchor.lookahead_mode='rank1_relex' requires anchor.replay_paired_batch=true"
                )
            if not self.spectral.enabled:
                raise ValueError("comm_eff.anchor.lookahead_mode='rank1_relex' requires spectral.enabled=true")
            if self.anchor.cadence % self.spectral.cadence != 0:
                raise ValueError("comm_eff.anchor.cadence must be divisible by spectral.cadence for rank1_relex")
            if self.anchor.lookahead_rollout_source == "stale_paired" and self.anchor.lookahead_strength != 0.0:
                raise ValueError(
                    "rank1_relex with nonzero strength requires auto/current_step trajectories; "
                    "stale_paired is reserved for the zero-increment control"
                )
        if self.anchor.warmup_mode == "q_only":
            if not rank1_relex_enabled(self.anchor):
                raise ValueError("comm_eff.anchor.warmup_mode='q_only' requires active rank1_relex")
            if self.compression_type != "powersgd" or not self.powersgd.enabled:
                raise ValueError("comm_eff.anchor.warmup_mode='q_only' requires PowerSGD")
            if not self.anchor.owns_q:
                raise ValueError("comm_eff.anchor.warmup_mode='q_only' requires anchor.owns_q=true")
        # fast_q_bootstrap is a PowerSGD-only feature. It is inert for the
        # prf_mask and sr_quant codecs (no PowerSGD compressor is built), so
        # leaving the launcher default fast_q_bootstrap=true on such an arm must
        # not error; the powersgd path validation below is unchanged.
        if self.powersgd.fast_q_bootstrap and self.compression_type not in ("prf_mask", "sr_quant"):
            if self.compression_type != "powersgd" or not self.powersgd.enabled:
                raise ValueError("comm_eff.powersgd.fast_q_bootstrap=true requires PowerSGD")
            if not self.anchor.owns_q:
                raise ValueError("comm_eff.powersgd.fast_q_bootstrap=true requires anchor.owns_q=true")
            if not self.powersgd.compress_recompute or not self.powersgd.sync_basis:
                raise ValueError(
                    "comm_eff.powersgd.fast_q_bootstrap=true requires compress_recompute=true and sync_basis=true"
                )
