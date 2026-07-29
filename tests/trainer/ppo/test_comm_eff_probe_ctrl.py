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

"""CPU tests for the I3 dense-view probe + adaptive KL coefficient (issue #93).

Covers: setpoint table parsing/interpolation, controller math including
conditional-integration anti-windup at both bounds and e_prev handling, the
dormant LR-brake detector, probe metric computation on synthetic tensors, the
probe_every=0 no-op, and the ppo_loss coefficient delivery (batch-stamped
override + config fallback).
"""

import math

import pytest
import torch

from verl.trainer.ppo.comm_eff_control import (
    DenseKLCoefController,
    LRBrakeDetector,
    compute_probe_metrics,
    interp_kl_target_table,
    parse_kl_target_table,
    should_probe,
)
from verl.utils import tensordict_utils as tu
from verl.workers.config import CommEffConfig, CommEffProbeConfig
from verl.workers.config.actor import ActorConfig
from verl.workers.utils.losses import ppo_loss


# --------------------------------------------------------------------------- #
# setpoint table parsing + interpolation
# --------------------------------------------------------------------------- #
def test_parse_table_basic_and_sorted():
    assert parse_kl_target_table("") == []
    assert parse_kl_target_table("   ") == []
    assert parse_kl_target_table("100:0.003,0:0.001") == [(0, 0.001), (100, 0.003)]
    assert parse_kl_target_table("50:1e-3") == [(50, 0.001)]


@pytest.mark.parametrize(
    "bad",
    ["5", "a:0.1", "5:b", "5:0.1,,10:0.2", "5:0.1,5:0.2", "-3:0.1", "5:-0.1", "5:nan"],
)
def test_parse_table_rejects_malformed(bad):
    with pytest.raises(ValueError):
        parse_kl_target_table(bad)


def test_interp_linear_and_edge_clamped():
    table = parse_kl_target_table("0:0.001,100:0.003")
    assert interp_kl_target_table(table, -5) == pytest.approx(0.001)  # clamp below
    assert interp_kl_target_table(table, 0) == pytest.approx(0.001)
    assert interp_kl_target_table(table, 50) == pytest.approx(0.002)  # linear midpoint
    assert interp_kl_target_table(table, 100) == pytest.approx(0.003)
    assert interp_kl_target_table(table, 900) == pytest.approx(0.003)  # clamp above
    assert interp_kl_target_table([], 42) == 0.0


# --------------------------------------------------------------------------- #
# controller math
# --------------------------------------------------------------------------- #
def _ctrl(**kwargs):
    defaults = dict(beta0=0.001, ki=0.3, kp=0.1, beta_min=2e-4, beta_max=0.05, c_floor=0.005, gain=2.0)
    defaults.update(kwargs)
    return DenseKLCoefController(**defaults)


def test_setpoint_floor_vs_table():
    ctrl = _ctrl(table=parse_kl_target_table("0:0.001,100:0.01"))
    # gain * table(0) = 0.002 < floor 0.005 -> floor wins.
    assert ctrl.setpoint(0) == pytest.approx(0.005)
    # gain * table(100) = 0.02 > floor -> table wins.
    assert ctrl.setpoint(100) == pytest.approx(0.02)
    assert _ctrl().setpoint(12345) == pytest.approx(0.005)  # no table -> floor


def test_update_matches_formula_and_eprev():
    ctrl = _ctrl()
    c = ctrl.setpoint(10)
    kl = 0.008
    e0 = (kl - c) / c
    # First update: e_prev starts at 0.0, so damping contributes kp*e0.
    expected = 0.001 * math.exp(0.3 * e0 + 0.1 * (e0 - 0.0))
    assert ctrl.update(kl, 10) == pytest.approx(expected)
    assert ctrl.e_prev == pytest.approx(e0)
    # Second update at the same reading: damping term vanishes (e == e_prev).
    expected2 = expected * math.exp(0.3 * e0)
    assert ctrl.update(kl, 20) == pytest.approx(expected2)


def test_update_skips_bad_readings():
    ctrl = _ctrl()
    for bad in (float("nan"), float("inf"), 0.0, -1.0):
        assert ctrl.update(bad, 10) == pytest.approx(0.001)
    assert ctrl.e_prev == 0.0  # untouched by measurement failures


def test_clipping_at_both_bounds():
    high = _ctrl()
    for step in range(0, 4000, 25):
        high.update(10.0, step)  # kl far above setpoint
    assert high.beta == pytest.approx(0.05)
    low = _ctrl()
    for step in range(0, 4000, 25):
        low.update(1e-6, step)  # kl far below setpoint
    assert low.beta == pytest.approx(2e-4)


def test_anti_windup_at_upper_bound():
    """Conditional integration: after any amount of saturation at beta_max,
    the first below-setpoint reading must pull beta OFF the bound immediately
    (no accumulated integral to unwind); more saturated steps must not deepen
    the exit."""
    ctrl = _ctrl()
    for step in range(0, 100 * 25, 25):
        ctrl.update(0.02, step)  # sustained e=3 saturation at beta_max
    assert ctrl.at_max
    e_prev = ctrl.e_prev
    kl_low = 0.004  # below the 0.005 floor setpoint
    e = (kl_low - 0.005) / 0.005
    expected = 0.05 * math.exp(0.3 * e + 0.1 * (e - e_prev))
    new_beta = ctrl.update(kl_low, 100 * 25)
    assert 2e-4 < new_beta < 0.05
    assert new_beta == pytest.approx(expected)

    # A controller saturated for 10x longer exits to the SAME beta: the
    # integral state never wound past the bound.
    longer = _ctrl()
    for step in range(0, 1000 * 25, 25):
        longer.update(0.02, step)
    assert longer.update(kl_low, 1000 * 25) == pytest.approx(new_beta)


def test_anti_windup_at_lower_bound():
    ctrl = _ctrl()
    for step in range(0, 100 * 25, 25):
        ctrl.update(0.0025, step)  # sustained e=-0.5 saturation at beta_min
    assert ctrl.at_min
    e_prev = ctrl.e_prev
    kl_high = 0.02  # above setpoint
    e = (kl_high - 0.005) / 0.005
    expected = 2e-4 * math.exp(0.3 * e + 0.1 * (e - e_prev))
    new_beta = ctrl.update(kl_high, 100 * 25)
    assert 2e-4 < new_beta < 0.05
    assert new_beta == pytest.approx(expected)

    longer = _ctrl()
    for step in range(0, 1000 * 25, 25):
        longer.update(0.0025, step)
    assert longer.update(kl_high, 1000 * 25) == pytest.approx(new_beta)


def test_integral_frozen_while_pinned():
    """While pinned at beta_max with the error still positive, the integral
    term is skipped: with e == e_prev (zero damping) beta must stay exactly
    at the bound rather than accumulate beyond it."""
    ctrl = _ctrl(beta0=0.05)  # start pinned at beta_max
    assert ctrl.at_max
    kl = 1.0
    ctrl.update(kl, 0)
    beta_after_two = ctrl.update(kl, 25)  # e == e_prev now
    assert beta_after_two == pytest.approx(0.05)
    assert ctrl.at_max


def test_beta0_projected_into_bounds():
    assert _ctrl(beta0=1.0).beta == pytest.approx(0.05)
    assert _ctrl(beta0=1e-9).beta == pytest.approx(2e-4)


# --------------------------------------------------------------------------- #
# dormant LR brake (detection only)
# --------------------------------------------------------------------------- #
def test_brake_doubling_requires_beta_pinned():
    brake = LRBrakeDetector()
    assert not brake.observe(0.01, 0.0, beta_at_max=False)
    # doubled but beta not pinned -> no trigger
    assert not brake.observe(0.02, 0.0, beta_at_max=False)
    # doubled again AND pinned -> trigger
    assert brake.observe(0.05, 0.0, beta_at_max=True)


def test_brake_gap_slope_acceleration():
    brake = LRBrakeDetector()
    # previous window slope 1e-4/probe, recent window slope 1e-3/probe (10x).
    gaps = [i * 1e-4 for i in range(4)] + [3e-4 + (i + 1) * 1e-3 for i in range(4)]
    fired = [brake.observe(0.001, g, beta_at_max=False) for g in gaps]
    assert fired[-1]
    assert not any(fired[:-1])


def test_brake_flat_previous_slope_never_ratio_trips():
    brake = LRBrakeDetector()
    # previous window strictly flat -> ratio test undefined -> no trigger.
    gaps = [1e-4] * 4 + [1e-4 + (i + 1) * 1e-3 for i in range(4)]
    assert not any(brake.observe(0.001, g, beta_at_max=False) for g in gaps)


def test_brake_nonfinite_readings_reset_history():
    brake = LRBrakeDetector()
    brake.observe(0.01, 0.0, beta_at_max=True)
    brake.observe(float("nan"), float("nan"), beta_at_max=True)
    # 0.04 is not compared against the pre-NaN 0.01 (history was reset).
    assert not brake.observe(0.04, 0.0, beta_at_max=True)


# --------------------------------------------------------------------------- #
# probe metrics on synthetic tensors
# --------------------------------------------------------------------------- #
def _k3_token_mean(logp, ref_logp, mask):
    kl = ref_logp - logp
    kld = torch.exp(kl) - kl - 1
    return (kld * mask).sum() / mask.sum()


def test_probe_metrics_kl_dense_and_gap_dense_values():
    torch.manual_seed(0)
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.bool)
    dense = -torch.rand(2, 4)
    ref = dense - 0.1 * torch.rand(2, 4)
    rollout = dense + 0.05
    out = compute_probe_metrics(
        dense_log_probs=dense,
        dense_ref_log_probs=ref,
        rollout_log_probs=rollout,
        response_mask=mask,
        kl_loss_last=None,
    )
    assert out["probe/kl_dense"] == pytest.approx(_k3_token_mean(dense, ref, mask.float()).item(), rel=1e-5)
    assert out["probe/gap_dense"] == pytest.approx(0.05, rel=1e-5)
    assert math.isnan(out["probe/kl_gain"])  # no actor/kl_loss supplied


def test_probe_metrics_kl_gain_ratio_and_guards():
    mask = torch.ones(1, 4, dtype=torch.bool)
    dense = torch.full((1, 4), -1.0)
    ref = torch.full((1, 4), -1.2)
    out = compute_probe_metrics(
        dense_log_probs=dense,
        dense_ref_log_probs=ref,
        rollout_log_probs=None,
        response_mask=mask,
        kl_loss_last=0.5,
    )
    kl_dense = math.exp(-0.2) + 0.2 - 1
    assert out["probe/kl_dense"] == pytest.approx(kl_dense, rel=1e-5)
    assert out["probe/kl_gain"] == pytest.approx(0.5 / kl_dense, rel=1e-5)
    assert math.isnan(out["probe/gap_dense"])  # rollout log probs absent

    # identical policies: kl_dense == 0 -> NaN-safe division guard
    out_zero = compute_probe_metrics(
        dense_log_probs=dense,
        dense_ref_log_probs=dense.clone(),
        rollout_log_probs=None,
        response_mask=mask,
        kl_loss_last=0.5,
    )
    assert out_zero["probe/kl_dense"] == pytest.approx(0.0, abs=1e-8)
    assert math.isnan(out_zero["probe/kl_gain"])


def test_probe_metrics_no_reference_is_nan_safe():
    mask = torch.ones(1, 4, dtype=torch.bool)
    dense = torch.full((1, 4), -1.0)
    out = compute_probe_metrics(
        dense_log_probs=dense,
        dense_ref_log_probs=None,
        rollout_log_probs=dense + 0.01,
        response_mask=mask,
        kl_loss_last=0.5,
    )
    assert math.isnan(out["probe/kl_dense"])
    assert math.isnan(out["probe/kl_gain"])
    assert out["probe/gap_dense"] == pytest.approx(0.01, rel=1e-4)


# --------------------------------------------------------------------------- #
# probe_every=0 no-op + config validation
# --------------------------------------------------------------------------- #
def test_probe_every_zero_never_fires():
    assert not any(should_probe(0, step) for step in range(1, 500))
    assert not should_probe(None, 25)
    fired = [step for step in range(1, 101) if should_probe(25, step)]
    assert fired == [25, 50, 75, 100]


def test_probe_config_defaults_off_and_validated():
    cfg = CommEffConfig()
    assert cfg.probe.probe_every == 0
    assert cfg.probe.ctrl_enabled is False
    assert cfg.probe.kl_target_table == ""
    with pytest.raises(ValueError, match="ctrl_enabled"):
        CommEffConfig(probe=CommEffProbeConfig(ctrl_enabled=True))
    with pytest.raises(ValueError, match="kl_target_table"):
        CommEffConfig(probe=CommEffProbeConfig(kl_target_table="5:0.1,x"))
    with pytest.raises(ValueError, match="probe_every"):
        CommEffConfig(probe=CommEffProbeConfig(probe_every=-1))
    with pytest.raises(ValueError, match="beta"):
        CommEffConfig(probe=CommEffProbeConfig(ctrl_beta_min=0.1, ctrl_beta_max=0.01))
    # a launch-shaped config passes
    CommEffConfig(probe=CommEffProbeConfig(probe_every=25, ctrl_enabled=True, kl_target_table="0:0.001,600:0.004"))


# --------------------------------------------------------------------------- #
# ppo_loss coefficient delivery: batch-stamped override + config fallback
# --------------------------------------------------------------------------- #
def _make_ppo_loss_case(kl_coef_override=None):
    torch.manual_seed(7)
    prompt_lens = [2, 3]
    resp_lens = [3, 2]
    seq_lens = [p + r for p, r in zip(prompt_lens, resp_lens, strict=True)]
    total = sum(seq_lens)
    max_resp = max(resp_lens)

    def _nested(lens, dtype=torch.long):
        flat = torch.zeros(sum(lens), dtype=dtype)
        offsets = torch.tensor([0, lens[0], sum(lens)])
        return torch.nested.nested_tensor_from_jagged(flat, offsets=offsets)

    log_probs = torch.nested.nested_tensor_from_jagged(
        -torch.rand(total), offsets=torch.tensor([0, seq_lens[0], total])
    )
    response_mask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.bool)
    tensor_dict = {
        "prompts": _nested(prompt_lens),
        "responses": _nested(resp_lens),
        "response_mask": response_mask,
        "old_log_probs": -torch.rand(2, max_resp),
        "advantages": torch.randn(2, max_resp),
        "ref_log_prob": -torch.rand(2, max_resp),
    }
    non_tensor = {"dp_size": 1, "batch_num_tokens": None, "global_batch_size": None}
    if kl_coef_override is not None:
        non_tensor["comm_eff_kl_coef"] = kl_coef_override
    data = tu.get_tensordict(tensor_dict=tensor_dict, non_tensor_dict=non_tensor)
    model_output = {"log_probs": log_probs}
    config = ActorConfig(
        strategy="fsdp",
        rollout_n=1,
        ppo_micro_batch_size=2,
        clip_ratio=0.2,
        use_kl_loss=True,
        kl_loss_coef=0.001,
        kl_loss_type="low_var_kl",
    )
    return config, model_output, data


def test_ppo_loss_falls_back_to_config_coef():
    config, model_output, data = _make_ppo_loss_case()
    _, metrics = ppo_loss(config, model_output, data)
    assert metrics["kl_coef"] == pytest.approx(0.001)


def test_ppo_loss_applies_stamped_coef():
    override = 0.02
    config, model_output, data_plain = _make_ppo_loss_case()
    loss_plain, metrics_plain = ppo_loss(config, model_output, data_plain)
    config2, model_output2, data_stamped = _make_ppo_loss_case(kl_coef_override=override)
    loss_stamped, metrics_stamped = ppo_loss(config2, model_output2, data_stamped)

    assert metrics_stamped["kl_coef"] == pytest.approx(override)
    kl_loss = metrics_plain["kl_loss"].aggregate()
    # identical inputs (same seed): the losses differ by exactly (delta coef) * kl_loss
    expected_delta = (override - 0.001) * float(kl_loss)
    assert float(loss_stamped - loss_plain) == pytest.approx(expected_delta, rel=1e-5)
