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

"""CPU tests for I4 CVC: train the train-inference disagreement down (#93 4.7).

Covers: CE mode through ppo_loss (loss increases by exactly
lambda_eff * token-mean(-logprob), the linear warmup ramp at steps 0/10/20/40,
and the cvc_lambda=0 bit-identical no-op), DC mode through the pure helpers
(masked advantage-shaping arithmetic, dual update rise/decay/clipping), and
the config defaults + validation for both modes.
"""

import math

import pytest
import torch

from verl.trainer.ppo.comm_eff_control import dc_dual_update, dc_shape_advantages
from verl.utils import tensordict_utils as tu
from verl.workers.config import CommEffConfig, CommEffDCConfig
from verl.workers.config.actor import ActorConfig
from verl.workers.utils.losses import cvc_warmup_ramp, ppo_loss
from verl.workers.utils.padding import no_padding_2_padding


# --------------------------------------------------------------------------- #
# CE mode: ppo_loss synthetic-logprob harness
# --------------------------------------------------------------------------- #
def _make_ppo_loss_case(cvc_lambda=0.0, cvc_warmup_steps=20, global_step=None):
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
    if global_step is not None:
        non_tensor["comm_eff_global_step"] = global_step
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
        cvc_lambda=cvc_lambda,
        cvc_warmup_steps=cvc_warmup_steps,
    )
    return config, model_output, data


def _expected_token_mean_ce(model_output, data):
    # token-mean of (-log_prob) over response tokens, computed independently
    # of ppo_loss's agg_loss path.
    resp_log_prob = no_padding_2_padding(model_output["log_probs"], data)
    mask = data["response_mask"].to(bool)
    return float((-resp_log_prob[mask]).mean())


def test_ppo_loss_cvc_lambda_zero_is_bit_identical_noop():
    config, model_output, data = _make_ppo_loss_case(cvc_lambda=0.0, global_step=100)
    loss_off, metrics_off = ppo_loss(config, model_output, data)
    config2, model_output2, data2 = _make_ppo_loss_case(cvc_lambda=0.0, global_step=100)
    # rebuild is seeded identically; the default path must not even log CVC
    loss_again, _ = ppo_loss(config2, model_output2, data2)
    assert "cvc_ce" not in metrics_off and "cvc_lambda" not in metrics_off
    assert torch.equal(loss_off, loss_again)


def test_ppo_loss_cvc_ce_adds_exactly_lambda_times_token_mean():
    lam = 0.003
    config_off, mo_off, data_off = _make_ppo_loss_case(cvc_lambda=0.0, global_step=40)
    loss_off, _ = ppo_loss(config_off, mo_off, data_off)
    config_on, mo_on, data_on = _make_ppo_loss_case(cvc_lambda=lam, cvc_warmup_steps=20, global_step=40)
    loss_on, metrics_on = ppo_loss(config_on, mo_on, data_on)

    expected_ce = _expected_token_mean_ce(mo_on, data_on)
    assert float(metrics_on["cvc_ce"].aggregate()) == pytest.approx(expected_ce, rel=1e-6)
    assert metrics_on["cvc_lambda"] == pytest.approx(lam)  # fully warmed at step 40
    assert float(loss_on - loss_off) == pytest.approx(lam * expected_ce, rel=1e-5)


@pytest.mark.parametrize(
    ("step", "expected_frac"),
    [(0, 0.0), (10, 0.5), (20, 1.0), (40, 1.0)],
)
def test_ppo_loss_cvc_warmup_ramp(step, expected_frac):
    lam = 0.01
    config_off, mo_off, data_off = _make_ppo_loss_case(cvc_lambda=0.0, global_step=step)
    loss_off, _ = ppo_loss(config_off, mo_off, data_off)
    config_on, mo_on, data_on = _make_ppo_loss_case(cvc_lambda=lam, cvc_warmup_steps=20, global_step=step)
    loss_on, metrics_on = ppo_loss(config_on, mo_on, data_on)

    lambda_eff = lam * expected_frac
    assert metrics_on["cvc_lambda"] == pytest.approx(lambda_eff)
    expected_ce = _expected_token_mean_ce(mo_on, data_on)
    # the CE metric is logged even while the ramp holds the coefficient at 0
    assert float(metrics_on["cvc_ce"].aggregate()) == pytest.approx(expected_ce, rel=1e-6)
    if lambda_eff == 0.0:
        assert torch.equal(loss_on, loss_off)
    else:
        assert float(loss_on - loss_off) == pytest.approx(lambda_eff * expected_ce, rel=1e-5)


def test_cvc_warmup_ramp_helper_edges():
    assert cvc_warmup_ramp(0, 5) == 1.0  # no ramp configured
    assert cvc_warmup_ramp(20, None) == 1.0  # unstamped caller counts as warmed
    assert cvc_warmup_ramp(20, -3) == 0.0
    assert cvc_warmup_ramp(20, 5) == pytest.approx(0.25)


# --------------------------------------------------------------------------- #
# DC mode: advantage shaping + dual update
# --------------------------------------------------------------------------- #
def _dc_case():
    torch.manual_seed(11)
    old_log_probs = -torch.rand(2, 4)
    rollout_log_probs = -torch.rand(2, 4)
    advantages = torch.randn(2, 4)
    response_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.bool)
    return advantages, old_log_probs, rollout_log_probs, response_mask


def test_dc_shape_advantages_masked_arithmetic():
    advantages, old_lp, roll_lp, mask = _dc_case()
    dc_lambda = 0.25
    shaped, delta_bar = dc_shape_advantages(
        advantages=advantages,
        old_log_probs=old_lp,
        rollout_log_probs=roll_lp,
        response_mask=mask,
        dc_lambda=dc_lambda,
    )
    delta = (old_lp.exp() - roll_lp.exp()).abs()
    expected = advantages - dc_lambda * delta * mask.float()
    assert torch.allclose(shaped, expected, atol=1e-6)
    # padded (masked-out) positions are untouched
    assert torch.equal(shaped[~mask], advantages[~mask])
    assert delta_bar == pytest.approx(float(delta[mask].mean()), rel=1e-6)


def test_dc_shape_advantages_lambda_zero_leaves_advantages():
    advantages, old_lp, roll_lp, mask = _dc_case()
    shaped, _ = dc_shape_advantages(
        advantages=advantages,
        old_log_probs=old_lp,
        rollout_log_probs=roll_lp,
        response_mask=mask,
        dc_lambda=0.0,
    )
    assert torch.equal(shaped, advantages)


def test_dc_dual_update_rises_and_decays():
    # gap above target: lambda rises by eta * (delta_bar - target)
    assert dc_dual_update(0.05, 0.10, eta=1.0, target=0.02, lambda_max=1.0) == pytest.approx(0.13)
    # gap below target: lambda decays toward 0
    assert dc_dual_update(0.05, 0.01, eta=1.0, target=0.02, lambda_max=1.0) == pytest.approx(0.04)
    # eta scales the step
    assert dc_dual_update(0.05, 0.10, eta=0.5, target=0.02, lambda_max=1.0) == pytest.approx(0.09)


def test_dc_dual_update_clips_at_bounds():
    assert dc_dual_update(0.95, 0.50, eta=1.0, target=0.0, lambda_max=1.0) == 1.0
    assert dc_dual_update(0.01, 0.0, eta=1.0, target=0.5, lambda_max=1.0) == 0.0
    # non-finite reading (empty mask) skips the update
    assert dc_dual_update(0.05, float("nan"), eta=1.0, target=0.02, lambda_max=1.0) == 0.05


# --------------------------------------------------------------------------- #
# config defaults + validation
# --------------------------------------------------------------------------- #
def test_actor_config_cvc_defaults_off_and_validated():
    config = ActorConfig(strategy="fsdp", rollout_n=1, ppo_micro_batch_size=2)
    assert config.cvc_lambda == 0.0
    assert config.cvc_warmup_steps == 20
    with pytest.raises(ValueError, match="cvc_lambda"):
        ActorConfig(strategy="fsdp", rollout_n=1, ppo_micro_batch_size=2, cvc_lambda=-0.1)
    with pytest.raises(ValueError, match="cvc_lambda"):
        ActorConfig(strategy="fsdp", rollout_n=1, ppo_micro_batch_size=2, cvc_lambda=math.inf)
    with pytest.raises(ValueError, match="cvc_warmup_steps"):
        ActorConfig(strategy="fsdp", rollout_n=1, ppo_micro_batch_size=2, cvc_warmup_steps=-1)


def test_dc_config_defaults_off_and_target_required_when_enabled():
    config = CommEffConfig()
    assert config.dc.enabled is False
    assert config.dc.eta == 1.0
    assert config.dc.target == -1.0
    assert config.dc.lambda0 == 0.05
    assert config.dc.lambda_max == 1.0

    # the -1.0 sentinel passes while disabled but is rejected when enabled
    with pytest.raises(ValueError, match="explicit comm_eff.dc.target"):
        CommEffConfig(dc=CommEffDCConfig(enabled=True))
    CommEffConfig(dc=CommEffDCConfig(enabled=True, target=0.02))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"eta": -1.0},
        {"eta": math.nan},
        {"lambda_max": 0.0},
        {"lambda0": -0.01},
        {"lambda0": 1.5, "lambda_max": 1.0},
        {"target": math.nan},
    ],
)
def test_dc_config_rejects_bad_values(kwargs):
    with pytest.raises(ValueError, match="comm_eff.dc"):
        CommEffConfig(dc=CommEffDCConfig(**kwargs))
