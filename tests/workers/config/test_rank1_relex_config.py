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

"""Fail-closed configuration coverage for rank1_relex and q_only warmup."""

import os
from pathlib import Path

import pytest
from jinja2 import Environment

from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config import (
    CommEffAnchorConfig,
    CommEffCaptureConfig,
    CommEffConfig,
    CommEffPowerSGDConfig,
    CommEffProbeConfig,
    CommEffSpectralConfig,
)


def _anchor(*, warmup="q_only", owns_q=True, **kwargs):
    values = dict(
        enabled=True,
        cadence=20,
        delay_K=20,
        owns_q=owns_q,
        replay_paired_batch=True,
        snapshot_device="cpu",
        lookahead_anchor=True,
        lookahead_mode="rank1_relex",
        lookahead_strength=1.0,
        lookahead_rollout_source="auto",
        warmup_mode=warmup,
        lookahead_min_snapshots=-1,
        lookahead_window_snapshots=4,
    )
    values.update(kwargs)
    return CommEffAnchorConfig(**values)


def _config(*, anchor=None, powersgd=None, spectral=None, probe=None, capture=None, compression_type="powersgd"):
    return CommEffConfig(
        enabled=True,
        compression_type=compression_type,
        anchor=anchor or _anchor(),
        powersgd=powersgd or CommEffPowerSGDConfig(q_basis="act", q_basis_passive=[]),
        spectral=spectral or CommEffSpectralConfig(enabled=True, correction_mode="signed_ema"),
        probe=probe or CommEffProbeConfig(),
        capture=capture or CommEffCaptureConfig(),
    )


def test_rank1_defaults_and_canonical_q_only_roundtrip():
    cfg = _config()
    assert cfg.anchor.lookahead_window_snapshots == 4
    assert cfg.anchor.lookahead_min_snapshots == -1
    assert cfg.anchor.warmup_mode == "q_only"
    assert cfg.anchor.owns_q
    assert cfg.powersgd.q_basis == "act"
    assert list(cfg.powersgd.q_basis_passive) == []


def test_rank1_no_correct_fast_owned_q_ablation_is_valid():
    cfg = _config(anchor=_anchor(warmup="no_correct", owns_q=False))
    assert cfg.anchor.warmup_mode == "no_correct"
    assert not cfg.anchor.owns_q


def test_rank1_stale_correct_first_fire_mode_is_valid():
    cfg = _config(anchor=_anchor(warmup="stale_correct"))
    assert cfg.anchor.warmup_mode == "stale_correct"
    assert cfg.anchor.lookahead_min_snapshots == -1


def test_rank1_rejects_unknown_warmup():
    with pytest.raises(ValueError, match="warmup_mode"):
        _config(anchor=_anchor(warmup="skip"))


def test_q_only_is_rank1_only():
    with pytest.raises(ValueError, match="rank1_relex-only"):
        _config(anchor=_anchor(lookahead_mode="fixed_linear"))


@pytest.mark.parametrize("window", [0, 1, True, 3.5])
def test_rank1_window_must_hold_base_and_one_delta(window):
    with pytest.raises(ValueError, match="lookahead_window_snapshots"):
        _config(anchor=_anchor(lookahead_window_snapshots=window))


def test_rank1_w2_secant_fallback_is_valid():
    cfg = _config(anchor=_anchor(lookahead_window_snapshots=2))
    assert cfg.anchor.lookahead_window_snapshots == 2


@pytest.mark.parametrize("minimum", [2, 3, 4])
def test_rank1_progressive_readiness_threshold_is_bounded_by_window(minimum):
    cfg = _config(anchor=_anchor(lookahead_min_snapshots=minimum))
    assert cfg.anchor.lookahead_window_snapshots == 4
    assert cfg.anchor.lookahead_min_snapshots == minimum


@pytest.mark.parametrize("minimum", [0, 1, 5])
def test_rank1_progressive_readiness_threshold_rejects_out_of_range_values(minimum):
    with pytest.raises(ValueError, match="lookahead_min_snapshots"):
        _config(anchor=_anchor(lookahead_min_snapshots=minimum))


@pytest.mark.parametrize("mode", ["disabled", "fixed_linear"])
def test_rank1_window_knob_is_inert_for_other_modes(mode):
    anchor = _anchor(
        lookahead_anchor=mode == "fixed_linear",
        lookahead_mode=mode,
        lookahead_window_snapshots=True,
        warmup="stale_correct",
    )
    cfg = _config(anchor=anchor)
    assert cfg.anchor.lookahead_window_snapshots is True


def test_rank1_requires_current_trajectories_and_exact_replay():
    with pytest.raises(ValueError, match="current trajectories"):
        _config(anchor=_anchor(lookahead_rollout_source="stale_paired"))
    with pytest.raises(ValueError, match="exact delayed"):
        _config(anchor=_anchor(replay_paired_batch=False))


def test_rank1_zero_increment_control_uses_exact_paired_trajectories():
    cfg = _config(
        anchor=_anchor(
            warmup="stale_correct",
            lookahead_strength=0.0,
            lookahead_rollout_source="stale_paired",
            lookahead_min_snapshots=2,
        )
    )
    assert cfg.anchor.lookahead_strength == 0.0
    assert cfg.anchor.lookahead_rollout_source == "stale_paired"


def test_rank1_requires_anchor_and_spectral_M_path():
    with pytest.raises(ValueError, match="anchor.enabled"):
        _config(anchor=_anchor(enabled=False))
    with pytest.raises(ValueError, match="spectral.enabled"):
        _config(spectral=CommEffSpectralConfig(enabled=False))


def test_rank1_requires_active_same_tick_correction_schedule():
    with pytest.raises(ValueError, match="active spectral correction_mode"):
        _config(spectral=CommEffSpectralConfig(enabled=True, correction_mode="none"))
    with pytest.raises(ValueError, match="divisible by spectral.cadence"):
        _config(spectral=CommEffSpectralConfig(enabled=True, cadence=3, correction_mode="signed_ema"))


def test_q_only_requires_enabled_powersgd_and_anchor_owned_Q():
    with pytest.raises(ValueError, match="PowerSGD codec"):
        _config(compression_type="dense")
    with pytest.raises(ValueError, match="PowerSGD codec"):
        _config(powersgd=CommEffPowerSGDConfig(enabled=False))
    with pytest.raises(ValueError, match="owns_q=true"):
        _config(anchor=_anchor(owns_q=False))


def test_q_only_requires_act_Q_without_passive_families():
    with pytest.raises(ValueError, match="q_basis='act'"):
        _config(powersgd=CommEffPowerSGDConfig(q_basis="grad"))
    with pytest.raises(ValueError, match="q_basis_passive"):
        _config(powersgd=CommEffPowerSGDConfig(q_basis="act", q_basis_passive=["grad"]))


def test_q_only_rejects_gradient_dependent_probes_but_allows_passive_capture():
    with pytest.raises(ValueError, match="geometry_enabled"):
        _config(probe=CommEffProbeConfig(geometry_enabled=True))
    with pytest.raises(ValueError, match="gradient-dependent"):
        _config(capture=CommEffCaptureConfig(enabled=True, capture_g_dense=True))
    with pytest.raises(ValueError, match="gradient-dependent"):
        _config(capture=CommEffCaptureConfig(enabled=True, capture_fresh_anchor=True))
    cfg = _config(capture=CommEffCaptureConfig(enabled=True))
    assert cfg.capture.enabled


def test_rank1_projection_probe_requires_rank1_and_bounded_samples():
    cfg = _config(probe=CommEffProbeConfig(rank1_projection_enabled=True, rank1_projection_samples=16))
    assert cfg.probe.rank1_projection_enabled
    assert cfg.probe.rank1_projection_samples == 16

    with pytest.raises(ValueError, match="requires active lookahead_mode='rank1_relex'"):
        _config(
            anchor=_anchor(lookahead_mode="fixed_linear", warmup="stale_correct"),
            probe=CommEffProbeConfig(rank1_projection_enabled=True),
        )
    for samples in (0, 65, True, 3.5):
        with pytest.raises(ValueError, match="rank1_projection_samples"):
            _config(probe=CommEffProbeConfig(rank1_projection_samples=samples))


def test_yaml_plain_overrides_register_all_rank1_launcher_fields():
    from hydra import compose, initialize_config_dir

    overrides = [
        "strategy=fsdp",
        "ppo_micro_batch_size_per_gpu=128",
        "comm_eff.enabled=true",
        "comm_eff.compression_type=powersgd",
        "comm_eff.anchor.enabled=true",
        "comm_eff.anchor.owns_q=true",
        "comm_eff.anchor.replay_paired_batch=true",
        "comm_eff.anchor.lookahead_anchor=true",
        "comm_eff.anchor.lookahead_mode=rank1_relex",
        "comm_eff.anchor.lookahead_strength=0.75",
        "comm_eff.anchor.lookahead_rollout_source=current_step",
        "comm_eff.anchor.lookahead_window_snapshots=5",
        "comm_eff.anchor.warmup_mode=q_only",
        "comm_eff.anchor.lookahead_min_snapshots=-1",
        "comm_eff.spectral.enabled=true",
        "comm_eff.spectral.correction_mode=signed_ema",
        "comm_eff.probe.rank1_projection_enabled=true",
        "comm_eff.probe.rank1_projection_samples=8",
    ]
    with initialize_config_dir(config_dir=os.path.abspath("verl/trainer/config/actor"), version_base=None):
        cfg = compose(config_name="dp_actor", overrides=overrides)
    actor = omega_conf_to_dataclass(cfg)
    assert actor.comm_eff.anchor.lookahead_anchor
    assert actor.comm_eff.anchor.lookahead_mode == "rank1_relex"
    assert actor.comm_eff.anchor.lookahead_strength == 0.75
    assert actor.comm_eff.anchor.lookahead_rollout_source == "current_step"
    assert actor.comm_eff.anchor.lookahead_window_snapshots == 5
    assert actor.comm_eff.probe.rank1_projection_enabled
    assert actor.comm_eff.probe.rank1_projection_samples == 8


def test_rank1_launcher_chat_template_is_byte_identical_to_relex_qwen():
    problem = r"How many vertical asymptotes does $y=2/(x^2+x-6)$ have?"
    legacy_user = problem + " Let's think step by step and output the final answer within \\boxed{}."
    template_path = Path("examples/grpo_trainer/relex_qwen_chat_template.jinja")
    rendered = (
        Environment()
        .from_string(template_path.read_text())
        .render(
            messages=[{"role": "user", "content": legacy_user}],
            add_generation_prompt=True,
        )
    )
    expected = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n{problem}\n"
        "Please reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    assert rendered == expected
