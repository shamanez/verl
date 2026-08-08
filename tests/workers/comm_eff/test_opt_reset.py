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

"""Anchor-sourced optimizer-state reset (comm_eff.anchor.opt_reset).

CPU tests for the reservoir hypothesis intervention: the anchor circuit keeps
fp32 CPU AdamW-style moment EMAs of its clean DP-averaged replay gradients,
and every ``cadence`` optimizer ticks the fast AdamW ``exp_avg``/``exp_avg_sq``
are overwritten with ``rho * m_anc`` / ``rho^2 * v_anc`` (mode=anchor_moments)
or zeroed (mode=zero). Asserted here: the per-fire EMA recurrence, the exact
overwrite math against a real ``torch.optim.AdamW`` state, the cadence gate
with its skip-before-first-fire path, the zero mode, the config validator
bounds, and that ``enabled=false`` allocates no state and never touches the
optimizer. These tests say nothing about training stability.
"""

import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from verl.workers.comm_eff.opt_reset import AnchorOptMoments, reset_optimizer_moments

# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _tiny_model_and_stepped_adamw(seed: int = 0):
    """A tiny plain model with a REAL AdamW that has taken one step.

    The step materializes the lazy ``exp_avg`` / ``exp_avg_sq`` / ``step``
    entries in ``optimizer.state`` so the reset has genuine state to overwrite.
    """
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    loss = model(torch.randn(4, 8)).square().mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return model, optimizer


def _fire_moments(model, moments: AnchorOptMoments, *, fires: int, seed: int = 1) -> None:
    torch.manual_seed(seed)
    for _ in range(fires):
        grads = {name: torch.randn_like(param) for name, param in model.named_parameters()}
        moments.update(grads)


def _global_l2_of_exp_avg(optimizer) -> float:
    total = 0.0
    for state in optimizer.state.values():
        t = state["exp_avg"].to(torch.float32)
        total += float(torch.sum(t * t).item())
    return math.sqrt(total)


def _snapshot_opt_state(optimizer) -> dict:
    out = {}
    for param, state in optimizer.state.items():
        out[id(param)] = {key: value.clone() for key, value in state.items() if isinstance(value, torch.Tensor)}
    return out


# --------------------------------------------------------------------------- #
# (a) anchor moment EMA math
# --------------------------------------------------------------------------- #


def test_anchor_moment_ema_math_over_several_fires():
    beta1, beta2 = 0.8, 0.95
    moments = AnchorOptMoments(beta1=beta1, beta2=beta2)
    torch.manual_seed(7)
    shapes = {"w": (3, 5), "b": (7,)}
    m_ref = {name: torch.zeros(shape) for name, shape in shapes.items()}
    v_ref = {name: torch.zeros(shape) for name, shape in shapes.items()}
    for fire in range(4):
        grads = {name: torch.randn(shape) for name, shape in shapes.items()}
        moments.update(grads)
        for name, g in grads.items():
            m_ref[name] = beta1 * m_ref[name] + (1.0 - beta1) * g
            v_ref[name] = beta2 * v_ref[name] + (1.0 - beta2) * g * g
        assert moments.fires == fire + 1
    for name in shapes:
        m, v = moments.get(name)
        assert m.dtype == torch.float32 and m.device.type == "cpu"
        assert v.dtype == torch.float32 and v.device.type == "cpu"
        assert torch.equal(m, m_ref[name])
        assert torch.equal(v, v_ref[name])


def test_anchor_moments_key_by_canonical_name():
    # Same convention as SpectralFilter._anchor: the FSDP wrap infix is
    # stripped on write AND on read, so live and anchor-clone names share keys.
    moments = AnchorOptMoments(beta1=0.8, beta2=0.95)
    moments.update({"model._fsdp_wrapped_module.lin.weight": torch.ones(2, 2)})
    assert moments.get("model.lin.weight") is not None
    assert moments.get("_fsdp_wrapped_module.model.lin.weight") is not None
    assert moments.get("model.other.weight") is None


def test_anchor_moments_reject_out_of_range_betas():
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            AnchorOptMoments(beta1=bad, beta2=0.95)
        with pytest.raises(ValueError):
            AnchorOptMoments(beta1=0.8, beta2=bad)


# --------------------------------------------------------------------------- #
# (b) reset math against a real torch.optim.AdamW state
# --------------------------------------------------------------------------- #


def test_reset_writes_exactly_rho_scaled_anchor_moments():
    model, optimizer = _tiny_model_and_stepped_adamw()
    moments = AnchorOptMoments(beta1=0.8, beta2=0.95)
    _fire_moments(model, moments, fires=3)

    steps_before = {id(p): optimizer.state[p]["step"].clone() for p in optimizer.state}
    fast_norm = _global_l2_of_exp_avg(optimizer)
    anchor_norm = math.sqrt(moments.m_sq_sum())
    rho_expected = fast_norm / (anchor_norm + 1e-12)

    rho = reset_optimizer_moments(
        optimizer,
        list(model.named_parameters()),
        moments=moments,
        mode="anchor_moments",
        scale_match=True,
    )
    assert rho == pytest.approx(rho_expected, rel=1e-6)
    for name, param in model.named_parameters():
        m_full, v_full = moments.get(name)
        state = optimizer.state[param]
        assert torch.allclose(state["exp_avg"], rho * m_full)
        assert torch.allclose(state["exp_avg_sq"], rho * rho * v_full)
        assert (state["exp_avg_sq"] >= 0).all()
        # The AdamW bias-correction clock is never touched.
        assert torch.equal(state["step"], steps_before[id(param)])


def test_reset_without_scale_match_uses_rho_one():
    model, optimizer = _tiny_model_and_stepped_adamw()
    moments = AnchorOptMoments(beta1=0.8, beta2=0.95)
    _fire_moments(model, moments, fires=2)
    rho = reset_optimizer_moments(
        optimizer,
        list(model.named_parameters()),
        moments=moments,
        mode="anchor_moments",
        scale_match=False,
    )
    assert rho == 1.0
    for name, param in model.named_parameters():
        m_full, v_full = moments.get(name)
        assert torch.equal(optimizer.state[param]["exp_avg"], m_full)
        assert torch.equal(optimizer.state[param]["exp_avg_sq"], v_full)


def test_reset_skips_params_without_anchor_entry_and_clamps_v():
    model, optimizer = _tiny_model_and_stepped_adamw()
    moments = AnchorOptMoments(beta1=0.8, beta2=0.95)
    _fire_moments(model, moments, fires=2)
    named = list(model.named_parameters())
    skipped_name, skipped_param = named[-1]
    from verl.workers.comm_eff.spectral_filter import _canon

    del moments._m[_canon(skipped_name)]
    del moments._v[_canon(skipped_name)]
    # A hand-poisoned negative v must come out clamped at >= 0.
    poisoned_name, poisoned_param = named[0]
    moments._v[_canon(poisoned_name)] = -torch.ones_like(poisoned_param)

    before = optimizer.state[skipped_param]["exp_avg"].clone()
    before_sq = optimizer.state[skipped_param]["exp_avg_sq"].clone()
    reset_optimizer_moments(optimizer, named, moments=moments, mode="anchor_moments", scale_match=True)
    assert torch.equal(optimizer.state[skipped_param]["exp_avg"], before)
    assert torch.equal(optimizer.state[skipped_param]["exp_avg_sq"], before_sq)
    assert (optimizer.state[poisoned_param]["exp_avg_sq"] >= 0).all()
    assert torch.equal(optimizer.state[poisoned_param]["exp_avg_sq"], torch.zeros_like(poisoned_param))


def test_reset_matches_moments_through_fsdp_infixed_names():
    # The live module's named_parameters may carry the FSDP wrap infix; the
    # lookup canonicalizes, so the plain-keyed moments still land.
    model, optimizer = _tiny_model_and_stepped_adamw()
    moments = AnchorOptMoments(beta1=0.8, beta2=0.95)
    _fire_moments(model, moments, fires=1)
    infixed = [(f"_fsdp_wrapped_module.{name}", param) for name, param in model.named_parameters()]
    rho = reset_optimizer_moments(optimizer, infixed, moments=moments, mode="anchor_moments", scale_match=False)
    assert rho == 1.0
    for name, param in model.named_parameters():
        m_full, _ = moments.get(name)
        assert torch.equal(optimizer.state[param]["exp_avg"], m_full)


# --------------------------------------------------------------------------- #
# (d) zero mode
# --------------------------------------------------------------------------- #


def test_zero_mode_zeroes_both_moments_and_keeps_step():
    model, optimizer = _tiny_model_and_stepped_adamw()
    moments = AnchorOptMoments(beta1=0.8, beta2=0.95)
    _fire_moments(model, moments, fires=1)
    for param in optimizer.state:
        assert optimizer.state[param]["exp_avg_sq"].abs().sum() > 0
    steps_before = {id(p): optimizer.state[p]["step"].clone() for p in optimizer.state}
    rho = reset_optimizer_moments(
        optimizer,
        list(model.named_parameters()),
        moments=moments,
        mode="zero",
        scale_match=True,
    )
    assert rho is None
    for param, state in optimizer.state.items():
        assert torch.equal(state["exp_avg"], torch.zeros_like(state["exp_avg"]))
        assert torch.equal(state["exp_avg_sq"], torch.zeros_like(state["exp_avg_sq"]))
        assert torch.equal(state["step"], steps_before[id(param)])


def test_unknown_mode_raises():
    model, optimizer = _tiny_model_and_stepped_adamw()
    moments = AnchorOptMoments(beta1=0.8, beta2=0.95)
    _fire_moments(model, moments, fires=1)
    with pytest.raises(ValueError):
        reset_optimizer_moments(
            optimizer,
            list(model.named_parameters()),
            moments=moments,
            mode="bogus",
            scale_match=True,
        )


# --------------------------------------------------------------------------- #
# (c) + (f) engine hook: cadence gate, skip-before-first-fire, disabled no-op
# --------------------------------------------------------------------------- #


def _fake_engine(model, optimizer, *, opt_reset_cfg, anchor_step: int, moments=None):
    """The REAL engine hook bound to a plain-module stand-in.

    Borrows FSDPEngine's methods so the cadence gate, the skip path and the
    disabled no-op are the production code paths, not re-implementations.
    """
    from verl.workers.engine.fsdp.transformer_impl import FSDPEngine

    class _FakeEngine:
        _maybe_comm_eff_opt_reset = FSDPEngine._maybe_comm_eff_opt_reset
        _opt_reset_fsdp1_shard_infos = FSDPEngine._opt_reset_fsdp1_shard_infos
        _opt_reset_reduce_sq_sum = FSDPEngine._opt_reset_reduce_sq_sum

    engine = _FakeEngine()
    engine.module = model
    engine.optimizer = optimizer
    state = SimpleNamespace(
        enabled=True,
        config=SimpleNamespace(anchor=SimpleNamespace(opt_reset=opt_reset_cfg)),
        anchor_step=anchor_step,
        opt_reset_count=0,
        opt_reset_last_rho=0.0,
    )
    if moments is not None:
        state._opt_reset_moments = moments
    engine._comm_eff_state = state
    return engine, state


def _opt_reset_cfg(**overrides):
    cfg = dict(enabled=True, cadence=2, mode="anchor_moments", beta1=0.8, beta2=0.95, scale_match=True)
    cfg.update(overrides)
    return SimpleNamespace(**cfg)


def test_hook_skips_before_first_anchor_fire(capsys):
    model, optimizer = _tiny_model_and_stepped_adamw()
    before = _snapshot_opt_state(optimizer)
    engine, state = _fake_engine(model, optimizer, opt_reset_cfg=_opt_reset_cfg(), anchor_step=2)
    engine._maybe_comm_eff_opt_reset()
    assert "[comm_eff][opt_reset] SKIP" in capsys.readouterr().out
    assert state.opt_reset_count == 0
    assert state.opt_reset_last_rho == 0.0
    for param, snap in ((p, before[id(p)]) for p in optimizer.state):
        for key, value in snap.items():
            assert torch.equal(optimizer.state[param][key], value)


def test_hook_cadence_gate_fires_only_on_multiples():
    model, optimizer = _tiny_model_and_stepped_adamw()
    moments = AnchorOptMoments(beta1=0.8, beta2=0.95)
    _fire_moments(model, moments, fires=1)

    # Off-cadence tick: untouched.
    before = _snapshot_opt_state(optimizer)
    engine, state = _fake_engine(
        model, optimizer, opt_reset_cfg=_opt_reset_cfg(cadence=2), anchor_step=3, moments=moments
    )
    engine._maybe_comm_eff_opt_reset()
    assert state.opt_reset_count == 0
    for param in optimizer.state:
        assert torch.equal(optimizer.state[param]["exp_avg"], before[id(param)]["exp_avg"])

    # On-cadence tick: fires, counts, records rho.
    state.anchor_step = 4
    engine._maybe_comm_eff_opt_reset()
    assert state.opt_reset_count == 1
    assert state.opt_reset_last_rho > 0.0
    for name, param in model.named_parameters():
        m_full, _ = moments.get(name)
        assert torch.allclose(optimizer.state[param]["exp_avg"], state.opt_reset_last_rho * m_full)


def test_hook_zero_mode_does_not_record_rho():
    model, optimizer = _tiny_model_and_stepped_adamw()
    moments = AnchorOptMoments(beta1=0.8, beta2=0.95)
    _fire_moments(model, moments, fires=1)
    engine, state = _fake_engine(
        model, optimizer, opt_reset_cfg=_opt_reset_cfg(mode="zero"), anchor_step=2, moments=moments
    )
    engine._maybe_comm_eff_opt_reset()
    assert state.opt_reset_count == 1
    assert state.opt_reset_last_rho == 0.0
    for param in optimizer.state:
        assert torch.equal(optimizer.state[param]["exp_avg"], torch.zeros_like(optimizer.state[param]["exp_avg"]))


def test_disabled_allocates_no_state_and_never_touches_optimizer():
    model, optimizer = _tiny_model_and_stepped_adamw()
    before = _snapshot_opt_state(optimizer)
    engine, state = _fake_engine(model, optimizer, opt_reset_cfg=_opt_reset_cfg(enabled=False), anchor_step=2)
    for tick in range(1, 7):
        state.anchor_step = tick
        engine._maybe_comm_eff_opt_reset()
    assert not hasattr(state, "_opt_reset_moments")
    assert state.opt_reset_count == 0
    for param in optimizer.state:
        for key, value in before[id(param)].items():
            assert torch.equal(optimizer.state[param][key], value)


# --------------------------------------------------------------------------- #
# (e) config validator
# --------------------------------------------------------------------------- #


def test_config_accepts_default_opt_reset_block():
    from verl.workers.config.comm_eff import (
        CommEffAnchorConfig,
        CommEffAnchorOptResetConfig,
        CommEffConfig,
        CommEffMaskConfig,
    )

    cfg = CommEffConfig()
    assert cfg.anchor.opt_reset.enabled is False
    assert cfg.anchor.opt_reset.cadence == 50
    assert cfg.anchor.opt_reset.mode == "anchor_moments"
    assert cfg.anchor.opt_reset.beta1 == 0.8
    assert cfg.anchor.opt_reset.beta2 == 0.95
    assert cfg.anchor.opt_reset.scale_match is True
    # The optreset arm's exact shape: prf_mask exact-k + enabled reset.
    CommEffConfig(
        enabled=True,
        compression_type="prf_mask",
        mask=CommEffMaskConfig(enabled=True, p=0.95, exact_k=True, rescale_mode="constant"),
        anchor=CommEffAnchorConfig(owns_q=False, opt_reset=CommEffAnchorOptResetConfig(enabled=True)),
    )


def test_config_rejects_bad_cadence_and_mode():
    from verl.workers.config.comm_eff import CommEffAnchorConfig, CommEffAnchorOptResetConfig, CommEffConfig

    with pytest.raises(ValueError, match="comm_eff.anchor.opt_reset.cadence"):
        CommEffConfig(anchor=CommEffAnchorConfig(opt_reset=CommEffAnchorOptResetConfig(cadence=0)))
    with pytest.raises(ValueError, match="comm_eff.anchor.opt_reset.mode"):
        CommEffConfig(anchor=CommEffAnchorConfig(opt_reset=CommEffAnchorOptResetConfig(mode="bogus")))
    for field_name in ("beta1", "beta2"):
        with pytest.raises(ValueError, match=f"comm_eff.anchor.opt_reset.{field_name}"):
            CommEffConfig(anchor=CommEffAnchorConfig(opt_reset=CommEffAnchorOptResetConfig(**{field_name: 1.0})))
    # mode="zero" is the other accepted value.
    CommEffConfig(anchor=CommEffAnchorConfig(opt_reset=CommEffAnchorOptResetConfig(mode="zero")))


def test_config_rejects_opt_reset_without_anchor():
    from verl.workers.config.comm_eff import (
        CommEffAnchorConfig,
        CommEffAnchorOptResetConfig,
        CommEffConfig,
        CommEffMaskConfig,
    )

    with pytest.raises(ValueError, match="requires anchor.enabled=true"):
        CommEffConfig(
            enabled=True,
            compression_type="prf_mask",
            mask=CommEffMaskConfig(enabled=True, p=0.95),
            anchor=CommEffAnchorConfig(
                enabled=False,
                owns_q=False,
                lookahead_anchor=False,
                lookahead_min_snapshots=-1,
                opt_reset=CommEffAnchorOptResetConfig(enabled=True),
            ),
        )


# --------------------------------------------------------------------------- #
# startup resolved line (the launch money-gate grep target)
# --------------------------------------------------------------------------- #


def test_state_build_prints_one_resolved_line_when_enabled(capsys):
    from verl.workers.comm_eff.state import maybe_build_comm_eff_state

    cfg = SimpleNamespace(
        enabled=True,
        anchor=SimpleNamespace(
            opt_reset=SimpleNamespace(
                enabled=True, cadence=50, mode="anchor_moments", beta1=0.8, beta2=0.95, scale_match=True
            )
        ),
    )
    state = maybe_build_comm_eff_state(cfg)
    state.build(None)
    out = capsys.readouterr().out
    expected = "[comm_eff][opt_reset] enabled cadence=50 mode=anchor_moments b1=0.8 b2=0.95 scale_match=True"
    assert out.count(expected) == 1

    # Disabled: the line must NOT print (grep-able absence on control arms).
    cfg_off = SimpleNamespace(
        enabled=True,
        anchor=SimpleNamespace(
            opt_reset=SimpleNamespace(
                enabled=False, cadence=50, mode="anchor_moments", beta1=0.8, beta2=0.95, scale_match=True
            )
        ),
    )
    state_off = maybe_build_comm_eff_state(cfg_off)
    state_off.build(None)
    assert "[comm_eff][opt_reset] enabled" not in capsys.readouterr().out
