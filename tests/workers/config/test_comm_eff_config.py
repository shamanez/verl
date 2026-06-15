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

"""Unit tests for the comm_eff config group.

Headline assertions:
  1. comm_eff defaults to DISABLED (enabled=false) at every level — the
     top-level config, the actor-nested config, and the YAML-composed config.
  2. The structured schema REJECTS unknown comm_eff.* keys (typos fail fast)
     so the later mask/anchor/spectral keys are validated.
  3. Constructing the disabled config / state is a strict no-op: the state
     factory returns None when disabled (no object, no RNG, no buffers), which
     is the load-bearing invariant behind the rel-tol-1e-4 parity claim.
"""

import os
import unittest

from verl.utils.config import omega_conf_to_dataclass
from verl.workers.comm_eff import maybe_build_comm_eff_state
from verl.workers.config import (
    ActorConfig,
    CommEffAnchorConfig,
    CommEffConfig,
    CommEffMaskConfig,
    CommEffSpectralConfig,
    OptimizerConfig,
)


class TestCommEffConfigDefaults(unittest.TestCase):
    """comm_eff must default to disabled everywhere."""

    def test_default_disabled(self):
        """A bare CommEffConfig is disabled and all circuits are off."""
        cfg = CommEffConfig()
        self.assertFalse(cfg.enabled)
        self.assertFalse(cfg.anchor.enabled)
        self.assertFalse(cfg.spectral.enabled)
        self.assertIsInstance(cfg.mask, CommEffMaskConfig)
        self.assertIsInstance(cfg.anchor, CommEffAnchorConfig)
        self.assertIsInstance(cfg.spectral, CommEffSpectralConfig)

    def test_actor_config_carries_disabled_comm_eff_by_default(self):
        """ActorConfig wires comm_eff and defaults it disabled."""
        config = ActorConfig(
            strategy="fsdp",
            use_dynamic_bsz=True,
            optim=OptimizerConfig(lr=0.1),
            rollout_n=1,
        )
        self.assertIsInstance(config.comm_eff, CommEffConfig)
        self.assertFalse(config.comm_eff.enabled)

    def test_from_dict_default_disabled(self):
        """omega_conf_to_dataclass on a minimal dict yields a disabled config."""
        cfg = omega_conf_to_dataclass(
            {"_target_": "verl.workers.config.CommEffConfig"},
            dataclass_type=CommEffConfig,
        )
        self.assertIsInstance(cfg, CommEffConfig)
        self.assertFalse(cfg.enabled)

    def test_explicit_enabled_false_and_true_roundtrip(self):
        """enabled is honored both ways through the same field."""
        disabled = omega_conf_to_dataclass({"enabled": False}, dataclass_type=CommEffConfig)
        self.assertFalse(disabled.enabled)
        enabled = omega_conf_to_dataclass({"enabled": True}, dataclass_type=CommEffConfig)
        self.assertTrue(enabled.enabled)

    def test_yaml_default_disabled(self):
        """The composed actor YAML defaults comm_eff disabled (registered key,
        reachable as actor_rollout_ref.actor.comm_eff.*)."""
        from hydra import compose, initialize_config_dir

        with initialize_config_dir(config_dir=os.path.abspath("verl/trainer/config/actor")):
            cfg = compose(config_name="dp_actor", overrides=["strategy=fsdp", "ppo_micro_batch_size_per_gpu=128"])

        config = omega_conf_to_dataclass(cfg)
        self.assertIsInstance(config.comm_eff, CommEffConfig)
        self.assertFalse(config.comm_eff.enabled)

    def test_yaml_plain_override_enabled_false(self):
        """The plain (no `+`) override comm_eff.enabled=false composes — i.e. the
        key is registered in the schema."""
        from hydra import compose, initialize_config_dir

        with initialize_config_dir(config_dir=os.path.abspath("verl/trainer/config/actor")):
            cfg = compose(
                config_name="dp_actor",
                overrides=[
                    "strategy=fsdp",
                    "ppo_micro_batch_size_per_gpu=128",
                    "comm_eff.enabled=false",
                ],
            )
        config = omega_conf_to_dataclass(cfg)
        self.assertFalse(config.comm_eff.enabled)

    def test_yaml_plain_override_anchor_replay_knobs(self):
        """EXP-29: the replay knobs are declared in the YAML struct, so a PLAIN
        (no `+`) CLI override composes. This is the dataclass<->YAML drift gate
        the first EXP-29 launch failed on (`Key 'replay_paired_batch' is not in
        struct`) — a new dataclass field MUST be mirrored in actor.yaml or every
        launcher override of it dies in Hydra validation before main."""
        from hydra import compose, initialize_config_dir

        with initialize_config_dir(config_dir=os.path.abspath("verl/trainer/config/actor")):
            cfg = compose(
                config_name="dp_actor",
                overrides=[
                    "strategy=fsdp",
                    "ppo_micro_batch_size_per_gpu=128",
                    "comm_eff.anchor.replay_paired_batch=true",
                    "comm_eff.anchor.snapshot_device=cpu",
                ],
            )
        config = omega_conf_to_dataclass(cfg)
        self.assertTrue(config.comm_eff.anchor.replay_paired_batch)
        self.assertEqual(config.comm_eff.anchor.snapshot_device, "cpu")
        # And the YAML defaults still mirror the dataclass defaults (off path).
        with initialize_config_dir(config_dir=os.path.abspath("verl/trainer/config/actor")):
            cfg_default = compose(
                config_name="dp_actor", overrides=["strategy=fsdp", "ppo_micro_batch_size_per_gpu=128"]
            )
        config_default = omega_conf_to_dataclass(cfg_default)
        self.assertFalse(config_default.comm_eff.anchor.replay_paired_batch)
        self.assertEqual(config_default.comm_eff.anchor.snapshot_device, "gpu")


class TestCommEffConfigSchema(unittest.TestCase):
    """The structured schema must reject unknown comm_eff.* keys."""

    def test_rejects_unknown_top_level_key(self):
        """An unknown comm_eff key (typo) must raise, not be silently dropped."""
        with self.assertRaises(Exception):
            omega_conf_to_dataclass(
                {"enabled": False, "enabledd": True},  # typo'd key
                dataclass_type=CommEffConfig,
            )

    def test_rejects_unknown_nested_mask_key(self):
        """An unknown key under comm_eff.mask must raise."""
        with self.assertRaises(Exception):
            omega_conf_to_dataclass(
                {"enabled": False, "mask": {"p": 0.9, "bogus": 1}},
                dataclass_type=CommEffConfig,
            )

    def test_rejects_unknown_nested_spectral_key(self):
        """An unknown key under comm_eff.spectral must raise."""
        with self.assertRaises(Exception):
            omega_conf_to_dataclass(
                {"enabled": False, "spectral": {"beta_anc": 0.9, "typo": 2}},
                dataclass_type=CommEffConfig,
            )

    def test_post_init_validates_ranges(self):
        """__post_init__ rejects out-of-range mask.p / spectral.signed_ema_alpha."""
        with self.assertRaises(ValueError):
            CommEffConfig(mask=CommEffMaskConfig(p=1.5))
        with self.assertRaises(ValueError):
            CommEffConfig(spectral=CommEffSpectralConfig(signed_ema_alpha=1.5))
        with self.assertRaises(ValueError):
            CommEffConfig(anchor=CommEffAnchorConfig(cadence=0))

    def test_anchor_replay_knob_defaults(self):
        """EXP-29 knobs default OFF / gpu — the byte-identical legacy path."""
        cfg = CommEffConfig()
        self.assertFalse(cfg.anchor.replay_paired_batch)
        self.assertEqual(cfg.anchor.snapshot_device, "gpu")

    def test_anchor_replay_knob_validation(self):
        """replay_paired_batch is a strict bool; snapshot_device is a closed enum."""
        # Valid settings construct fine.
        cfg = CommEffConfig(
            anchor=CommEffAnchorConfig(replay_paired_batch=True, snapshot_device="cpu")
        )
        self.assertTrue(cfg.anchor.replay_paired_batch)
        self.assertEqual(cfg.anchor.snapshot_device, "cpu")
        # A YAML "False" string (truthy!) must be loud, not a silent enable.
        with self.assertRaises(ValueError):
            CommEffConfig(anchor=CommEffAnchorConfig(replay_paired_batch="False"))
        with self.assertRaises(ValueError):
            CommEffConfig(anchor=CommEffAnchorConfig(replay_paired_batch=1))
        # Typo'd device is a loud error, not a fall-through to gpu.
        with self.assertRaises(ValueError):
            CommEffConfig(anchor=CommEffAnchorConfig(snapshot_device="hbm"))

    def test_anchor_replay_rejects_unknown_key(self):
        """The structured schema still rejects a typo'd replay key."""
        with self.assertRaises(Exception):
            omega_conf_to_dataclass(
                {"anchor": {"replay_paired_batches": True}}, dataclass_type=CommEffConfig
            )


class TestCommEffPowerSGDConfig(unittest.TestCase):
    """The compression_type enum and powersgd block must be registered and validated."""

    def test_default_compression_type_dense_powersgd_block_present(self):
        from verl.workers.config import CommEffPowerSGDConfig

        cfg = CommEffConfig()
        self.assertEqual(cfg.compression_type, "dense")
        self.assertIsInstance(cfg.powersgd, CommEffPowerSGDConfig)
        self.assertEqual(cfg.powersgd.rank, 102)
        self.assertEqual(cfg.powersgd.update_cadence, 1)
        self.assertTrue(cfg.powersgd.warm_start)
        self.assertTrue(cfg.powersgd.compress_recompute)
        # The shared codebook must be synced across DP ranks by default.
        self.assertTrue(cfg.powersgd.sync_basis)
        self.assertEqual(cfg.powersgd.qr_dtype, "fp32")

    def test_compression_type_enum_validated(self):
        with self.assertRaises(ValueError):
            CommEffConfig(compression_type="powerSGD")  # typo
        for ok in ("dense", "prf_mask", "powersgd"):
            self.assertEqual(CommEffConfig(compression_type=ok).compression_type, ok)

    def test_powersgd_block_validated(self):
        from verl.workers.config import CommEffPowerSGDConfig

        with self.assertRaises(ValueError):
            CommEffConfig(powersgd=CommEffPowerSGDConfig(rank=0))
        with self.assertRaises(ValueError):
            CommEffConfig(powersgd=CommEffPowerSGDConfig(update_cadence=0))
        with self.assertRaises(ValueError):
            CommEffConfig(powersgd=CommEffPowerSGDConfig(qr_dtype="float16"))

    def test_resolve_compression_type_back_compat(self):
        """resolve_compression_type honors an explicit codec, else falls back to
        the legacy mask selector."""
        from verl.workers.config import CommEffMaskConfig
        from verl.workers.comm_eff.state import resolve_compression_type

        # explicit powersgd wins
        self.assertEqual(
            resolve_compression_type(CommEffConfig(enabled=True, compression_type="powersgd")),
            "powersgd",
        )
        # dense default + mask enabled p>0 => legacy prf_mask
        self.assertEqual(
            resolve_compression_type(CommEffConfig(enabled=True, mask=CommEffMaskConfig(enabled=True, p=0.95))),
            "prf_mask",
        )
        # dense default + mask off => dense
        self.assertEqual(
            resolve_compression_type(CommEffConfig(enabled=True, mask=CommEffMaskConfig(enabled=False))),
            "dense",
        )

    def test_yaml_powersgd_args_compose_the_gotcha(self):
        """The whole launcher arg path (compression_type=powersgd +
        comm_eff.powersgd.rank=102 + a prf_mask run forwarding powersgd.rank)
        composes through the actor YAML — the registered-keys requirement that
        keeps the shared launcher valid for both mask and PowerSGD arms."""
        from hydra import compose, initialize_config_dir

        with initialize_config_dir(config_dir=os.path.abspath("verl/trainer/config/actor")):
            cfg = compose(
                config_name="dp_actor",
                overrides=[
                    "strategy=fsdp",
                    "ppo_micro_batch_size_per_gpu=128",
                    "comm_eff.enabled=true",
                    "comm_eff.compression_type=powersgd",
                    "comm_eff.powersgd.rank=102",
                    "comm_eff.powersgd.warm_start=true",
                    "comm_eff.powersgd.compress_recompute=true",
                    "comm_eff.powersgd.qr_dtype=fp32",
                ],
            )
        config = omega_conf_to_dataclass(cfg)
        self.assertEqual(config.comm_eff.compression_type, "powersgd")
        self.assertEqual(config.comm_eff.powersgd.rank, 102)

        # The mask arm forwards the same powersgd.* args while selecting
        # prf_mask; those keys must still parse.
        with initialize_config_dir(config_dir=os.path.abspath("verl/trainer/config/actor")):
            cfg2 = compose(
                config_name="dp_actor",
                overrides=[
                    "strategy=fsdp",
                    "ppo_micro_batch_size_per_gpu=128",
                    "comm_eff.enabled=true",
                    "comm_eff.compression_type=prf_mask",
                    "comm_eff.mask.p=0.95",
                    "comm_eff.powersgd.rank=102",  # forwarded by the shared launcher
                ],
            )
        config2 = omega_conf_to_dataclass(cfg2)
        self.assertEqual(config2.comm_eff.compression_type, "prf_mask")
        self.assertEqual(config2.comm_eff.powersgd.rank, 102)


class TestCommEffExp30Knobs(unittest.TestCase):
    """EXP-30: geometry-probe + delayed_ef knobs — defaults OFF/legacy, the
    dataclass<->actor.yaml drift gate, and the cross-field validation rules."""

    def test_defaults_off_legacy(self):
        """Every EXP-30 knob defaults OFF/legacy (off-path parity)."""
        from verl.workers.config import CommEffProbeConfig

        cfg = CommEffConfig()
        self.assertIsInstance(cfg.probe, CommEffProbeConfig)
        self.assertFalse(cfg.probe.geometry_enabled)
        self.assertEqual(cfg.probe.out_dir, "")
        self.assertTrue(cfg.probe.rank0_only)
        self.assertEqual(cfg.probe.m4_lags, 5)
        self.assertTrue(cfg.probe.per_target_sidecar)
        # delayed_ef λ=0 = the exact-identity limiting case; the DEFAULT merger is
        # the inert "none" (the SOTA delayed_ef is set explicitly by the launcher,
        # not the dataclass default).
        self.assertEqual(cfg.spectral.delayed_ef_lambda, 0.0)
        self.assertEqual(cfg.spectral.correction_mode, "none")

    def test_correction_mode_enum_extended(self):
        """'none' and 'delayed_ef' are accepted; typos stay loud."""
        from verl.workers.config import CommEffAnchorConfig, CommEffSpectralConfig

        for ok in ("none", "inject", "blend", "signed_ema", "ef_powersgd"):
            cfg = CommEffConfig(spectral=CommEffSpectralConfig(correction_mode=ok))
            self.assertEqual(cfg.spectral.correction_mode, ok)
        # delayed_ef requires the valid-M premise (replay) when spectral is on.
        cfg = CommEffConfig(
            anchor=CommEffAnchorConfig(enabled=True, replay_paired_batch=True),
            spectral=CommEffSpectralConfig(enabled=True, correction_mode="delayed_ef"),
        )
        self.assertEqual(cfg.spectral.correction_mode, "delayed_ef")
        with self.assertRaises(ValueError):
            CommEffConfig(spectral=CommEffSpectralConfig(correction_mode="delayed_EF"))
        with self.assertRaises(ValueError):
            CommEffConfig(spectral=CommEffSpectralConfig(correction_mode="None"))

    def test_delayed_ef_requires_replay(self):
        """delayed_ef on the legacy (generator-mismatched) feed would re-test the
        retired object — must fail loud at config time."""
        from verl.workers.config import CommEffAnchorConfig, CommEffSpectralConfig

        with self.assertRaises(ValueError):
            CommEffConfig(
                anchor=CommEffAnchorConfig(enabled=True, replay_paired_batch=False),
                spectral=CommEffSpectralConfig(enabled=True, correction_mode="delayed_ef"),
            )
        with self.assertRaises(ValueError):
            CommEffConfig(
                anchor=CommEffAnchorConfig(enabled=False, replay_paired_batch=True),
                spectral=CommEffSpectralConfig(enabled=True, correction_mode="delayed_ef"),
            )

    def test_delayed_ef_lambda_validated(self):
        from verl.workers.config import CommEffSpectralConfig

        with self.assertRaises(ValueError):
            CommEffConfig(spectral=CommEffSpectralConfig(delayed_ef_lambda=-0.1))

    def test_probe_requires_replay_and_inert_merger(self):
        """The Step-A posture is validated as a unit: probe needs the EXP-29
        replay substrate AND an inert merger (correction_mode=none)."""
        from verl.workers.config import (
            CommEffAnchorConfig,
            CommEffProbeConfig,
            CommEffSpectralConfig,
        )

        # The sanctioned Step-A shape parses.
        cfg = CommEffConfig(
            anchor=CommEffAnchorConfig(enabled=True, replay_paired_batch=True, snapshot_device="cpu"),
            spectral=CommEffSpectralConfig(enabled=True, correction_mode="none", beta_anc=0.0),
            probe=CommEffProbeConfig(geometry_enabled=True, out_dir="/tmp/probe"),
        )
        self.assertTrue(cfg.probe.geometry_enabled)
        # No replay ⇒ loud.
        with self.assertRaises(ValueError):
            CommEffConfig(
                anchor=CommEffAnchorConfig(enabled=True, replay_paired_batch=False),
                spectral=CommEffSpectralConfig(enabled=True, correction_mode="none"),
                probe=CommEffProbeConfig(geometry_enabled=True),
            )
        # Anchor off ⇒ loud.
        with self.assertRaises(ValueError):
            CommEffConfig(
                anchor=CommEffAnchorConfig(enabled=False, replay_paired_batch=True),
                spectral=CommEffSpectralConfig(enabled=True, correction_mode="none"),
                probe=CommEffProbeConfig(geometry_enabled=True),
            )
        # ACTIVE merger under the probe ⇒ loud (G_comp would be corrupted).
        for live_mode in ("signed_ema", "blend", "ef_powersgd"):
            with self.assertRaises(ValueError):
                CommEffConfig(
                    anchor=CommEffAnchorConfig(enabled=True, replay_paired_batch=True),
                    spectral=CommEffSpectralConfig(enabled=True, correction_mode=live_mode),
                    probe=CommEffProbeConfig(geometry_enabled=True),
                )
        # Spectral fully OFF + probe is fine (no merger to corrupt G_comp).
        cfg2 = CommEffConfig(
            anchor=CommEffAnchorConfig(enabled=True, replay_paired_batch=True),
            spectral=CommEffSpectralConfig(enabled=False, correction_mode="signed_ema"),
            probe=CommEffProbeConfig(geometry_enabled=True),
        )
        self.assertTrue(cfg2.probe.geometry_enabled)

    def test_probe_knob_validation(self):
        from verl.workers.config import CommEffProbeConfig

        # Strict bools (the YAML "False"-string trap).
        with self.assertRaises(ValueError):
            CommEffConfig(probe=CommEffProbeConfig(geometry_enabled="False"))
        with self.assertRaises(ValueError):
            CommEffConfig(probe=CommEffProbeConfig(rank0_only=1))
        with self.assertRaises(ValueError):
            CommEffConfig(probe=CommEffProbeConfig(per_target_sidecar="true"))
        # m4_lags bounded to [1, 5] (the ≤6-entry lag-buffer plan bound).
        with self.assertRaises(ValueError):
            CommEffConfig(probe=CommEffProbeConfig(m4_lags=0))
        with self.assertRaises(ValueError):
            CommEffConfig(probe=CommEffProbeConfig(m4_lags=6))

    def test_probe_rejects_unknown_key(self):
        with self.assertRaises(Exception):
            omega_conf_to_dataclass(
                {"probe": {"geometry_enabld": True}}, dataclass_type=CommEffConfig
            )

    def test_yaml_plain_override_exp30_knobs(self):
        """The dataclass<->actor.yaml drift gate for EXP-30 (the EXP-29
        first-launch killer): every new knob composes as a PLAIN override
        through the actor YAML, and the YAML defaults mirror the dataclass."""
        from hydra import compose, initialize_config_dir

        with initialize_config_dir(config_dir=os.path.abspath("verl/trainer/config/actor")):
            cfg = compose(
                config_name="dp_actor",
                overrides=[
                    "strategy=fsdp",
                    "ppo_micro_batch_size_per_gpu=128",
                    "comm_eff.anchor.enabled=true",
                    "comm_eff.anchor.replay_paired_batch=true",
                    "comm_eff.anchor.snapshot_device=cpu",
                    "comm_eff.spectral.enabled=true",
                    "comm_eff.spectral.correction_mode=none",
                    "comm_eff.spectral.beta_anc=0.0",
                    "comm_eff.spectral.delayed_ef_lambda=1.0",
                    "comm_eff.probe.geometry_enabled=true",
                    "comm_eff.probe.out_dir=/workspace/runs/EXP-30/metrics",
                    "comm_eff.probe.rank0_only=true",
                    "comm_eff.probe.m4_lags=5",
                    "comm_eff.probe.per_target_sidecar=true",
                ],
            )
        config = omega_conf_to_dataclass(cfg)
        self.assertTrue(config.comm_eff.probe.geometry_enabled)
        self.assertEqual(config.comm_eff.probe.out_dir, "/workspace/runs/EXP-30/metrics")
        self.assertEqual(config.comm_eff.probe.m4_lags, 5)
        self.assertEqual(config.comm_eff.spectral.correction_mode, "none")
        self.assertEqual(config.comm_eff.spectral.beta_anc, 0.0)
        self.assertEqual(config.comm_eff.spectral.delayed_ef_lambda, 1.0)
        # And the YAML defaults still mirror the dataclass defaults (off path).
        with initialize_config_dir(config_dir=os.path.abspath("verl/trainer/config/actor")):
            cfg_default = compose(
                config_name="dp_actor", overrides=["strategy=fsdp", "ppo_micro_batch_size_per_gpu=128"]
            )
        config_default = omega_conf_to_dataclass(cfg_default)
        self.assertFalse(config_default.comm_eff.probe.geometry_enabled)
        self.assertEqual(config_default.comm_eff.probe.out_dir, "")
        self.assertTrue(config_default.comm_eff.probe.rank0_only)
        self.assertEqual(config_default.comm_eff.probe.m4_lags, 5)
        self.assertTrue(config_default.comm_eff.probe.per_target_sidecar)
        self.assertEqual(config_default.comm_eff.spectral.delayed_ef_lambda, 0.0)
        # Default merger is the inert "none" (the SOTA delayed_ef is set explicitly
        # by the launcher); the YAML default mirrors the dataclass default.
        self.assertEqual(config_default.comm_eff.spectral.correction_mode, "none")

    def test_yaml_delayed_ef_b2_shape_composes(self):
        """The (gated) B2 cell's exact override set composes — pre-validated now
        so a future GATE-B2-open dispatch cannot die in Hydra."""
        from hydra import compose, initialize_config_dir

        with initialize_config_dir(config_dir=os.path.abspath("verl/trainer/config/actor")):
            cfg = compose(
                config_name="dp_actor",
                overrides=[
                    "strategy=fsdp",
                    "ppo_micro_batch_size_per_gpu=128",
                    "comm_eff.anchor.enabled=true",
                    "comm_eff.anchor.replay_paired_batch=true",
                    "comm_eff.spectral.enabled=true",
                    "comm_eff.spectral.correction_mode=delayed_ef",
                    "comm_eff.spectral.delayed_ef_lambda=1.0",
                    "comm_eff.spectral.beta_anc=0.0",
                ],
            )
        config = omega_conf_to_dataclass(cfg)
        self.assertEqual(config.comm_eff.spectral.correction_mode, "delayed_ef")
        self.assertEqual(config.comm_eff.spectral.delayed_ef_lambda, 1.0)
        self.assertEqual(config.comm_eff.spectral.beta_anc, 0.0)


class TestCommEffStateInert(unittest.TestCase):
    """The disabled state must be a strict no-op (the parity-check invariant)."""

    def test_disabled_state_is_none(self):
        """maybe_build_comm_eff_state returns None when disabled — no object,
        no RNG, no buffers, no hooks."""
        self.assertIsNone(maybe_build_comm_eff_state(CommEffConfig()))
        self.assertIsNone(maybe_build_comm_eff_state(CommEffConfig(enabled=False)))
        self.assertIsNone(maybe_build_comm_eff_state(None))
        self.assertIsNone(maybe_build_comm_eff_state({"enabled": False}))

    def test_enabled_state_is_built_with_zero_counters(self):
        """When enabled, a state exists and its op counters start at exactly 0."""
        state = maybe_build_comm_eff_state(CommEffConfig(enabled=True))
        self.assertIsNotNone(state)
        self.assertTrue(state.enabled)
        m = state.metrics()
        self.assertEqual(m["comm_eff/mask_applications"], 0)
        self.assertEqual(m["comm_eff/anchor_backwards"], 0)
        self.assertEqual(m["comm_eff/spectral_corrections"], 0)

    def test_disabled_dict_metrics_empty(self):
        """comm_eff_metrics(None) is empty (disabled => counters absent)."""
        from verl.workers.comm_eff.state import comm_eff_metrics

        self.assertEqual(comm_eff_metrics(None), {})


if __name__ == "__main__":
    unittest.main()
