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

"""Unit tests for the comm_eff config group (EXP-4 M2 no-op scaffolding).

Headline assertions (machine-checkable by codex-verify / the analyst):
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
        """enabled is honored both ways; this is the Run-A `comm_eff.enabled=false`
        path and the (future) enabled path through the same field."""
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
        key is registered in the schema (Run A uses the plain key)."""
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
                {"enabled": False, "spectral": {"rank": 8, "typo": 2}},
                dataclass_type=CommEffConfig,
            )

    def test_post_init_validates_ranges(self):
        """__post_init__ rejects out-of-range mask.p / spectral.rank."""
        with self.assertRaises(ValueError):
            CommEffConfig(mask=CommEffMaskConfig(p=1.5))
        with self.assertRaises(ValueError):
            CommEffConfig(spectral=CommEffSpectralConfig(rank=0))
        with self.assertRaises(ValueError):
            CommEffConfig(anchor=CommEffAnchorConfig(cadence=0))


class TestCommEffPowerSGDConfig(unittest.TestCase):
    """EXP-20/M6: the compression_type enum + powersgd block must be registered
    (so a launcher arg parses regardless of compression_type — the clean_cadence
    struct-mode gotcha) and validated."""

    def test_default_compression_type_dense_powersgd_block_present(self):
        from verl.workers.config import CommEffPowerSGDConfig

        cfg = CommEffConfig()
        self.assertEqual(cfg.compression_type, "dense")
        self.assertIsInstance(cfg.powersgd, CommEffPowerSGDConfig)
        # Issue VII.1 defaults.
        self.assertEqual(cfg.powersgd.rank, 102)
        self.assertEqual(cfg.powersgd.update_cadence, 1)
        self.assertTrue(cfg.powersgd.warm_start)
        self.assertTrue(cfg.powersgd.compress_recompute)
        self.assertFalse(cfg.powersgd.sync_basis)
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
        the legacy mask selector (so every pre-EXP-20 mask config still runs)."""
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
        bit clean_cadence. Mask-arm and powersgd-arm share ONE launcher, so the
        mask arm DOES forward powersgd.* args; that must parse."""
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

        # The mask arm forwards the SAME powersgd.* args (shared launcher) while
        # selecting prf_mask — must also parse (the gotcha).
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
