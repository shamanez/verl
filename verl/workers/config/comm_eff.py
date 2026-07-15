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

from dataclasses import dataclass, field

from verl.base_config import BaseConfig

__all__ = [
    "CommEffAnchorConfig",
    "CommEffSpectralConfig",
    "CommEffPowerSGDConfig",
    "CommEffConfig",
]

COMPRESSION_TYPES = ("dense", "powersgd")


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
    sync_basis: bool = True
    qr_dtype: str = "fp32"
    reortho_eps: float = 1e-6
    fast_q_bootstrap: bool = True


@dataclass
class CommEffConfig(BaseConfig):
    """Top-level communication-efficient method configuration.

    ``enabled=false`` is the dense compatibility path. ``compression_type`` may
    also be set to ``dense`` explicitly for control arms; no alternative lossy
    codec is registered.
    """

    enabled: bool = False
    compression_type: str = "powersgd"
    anchor: CommEffAnchorConfig = field(default_factory=CommEffAnchorConfig)
    spectral: CommEffSpectralConfig = field(default_factory=CommEffSpectralConfig)
    powersgd: CommEffPowerSGDConfig = field(default_factory=CommEffPowerSGDConfig)

    def __post_init__(self):
        """Validate without allocating tensors, communicating, or drawing RNG."""

        if not isinstance(self.enabled, bool):
            raise ValueError(f"comm_eff.enabled must be a bool; got {self.enabled!r}")
        if self.compression_type not in COMPRESSION_TYPES:
            raise ValueError(
                f"comm_eff.compression_type must be one of {COMPRESSION_TYPES}; got {self.compression_type!r}"
            )

        self._validate_anchor()
        self._validate_spectral()
        self._validate_powersgd()
        self._validate_cross_circuit_contract()

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
        for name in ("enabled", "warm_start", "compress_recompute", "sync_basis", "fast_q_bootstrap"):
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

    def _validate_cross_circuit_contract(self) -> None:
        from verl.workers.comm_eff.lookahead import rank1_relex_enabled

        if not self.enabled:
            return
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
        if self.powersgd.fast_q_bootstrap:
            if self.compression_type != "powersgd" or not self.powersgd.enabled:
                raise ValueError("comm_eff.powersgd.fast_q_bootstrap=true requires PowerSGD")
            if not self.anchor.owns_q:
                raise ValueError("comm_eff.powersgd.fast_q_bootstrap=true requires anchor.owns_q=true")
            if not self.powersgd.compress_recompute or not self.powersgd.sync_basis:
                raise ValueError(
                    "comm_eff.powersgd.fast_q_bootstrap=true requires compress_recompute=true and sync_basis=true"
                )
