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

"""CPU unit tests for the learned signed gate (spectral.signed_gate).

Semantics under test:

    fire tick    G_corr = alpha*G + (1-alpha)*|G|*sign(M)     full strength, fresh M
    held tick    G_corr = (1-w)*G + w*G_signed,  w = rho * decay**age
    at each fire rho <- (1-BETA)*rho + BETA * clamp(mean(sign(M_old)*sign(G_anchor)), 0, 1)

rho starts at 0, so the first interval is exactly the use-once schedule; the
gate earns between-fire dose only from measured agreement, and decay < 1 keeps
the schedule inside the geometric envelope (standing full-strength reuse is
unreachable by construction).
"""

import pytest
import torch

from verl.workers.comm_eff.spectral_filter import GATE_RHO_EMA_BETA, SpectralFilter

ALPHA = 0.25
DECAY = 0.75


def make_filter(**kw):
    defaults = dict(
        beta_anc=0.0,
        ema_device="cpu",
        signed_ema_alpha=ALPHA,
        correction_mode="signed_ema",
        signed_gate="learned",
        signed_gate_decay=DECAY,
        diagnostics=False,
    )
    defaults.update(kw)
    return SpectralFilter(**defaults)


def signed(g, m):
    return ALPHA * g + (1.0 - ALPHA) * g.abs() * torch.sign(m)


def test_gate_off_is_bitwise_identity_and_allocates_nothing():
    # signed_gate="off" must be the historical signed_ema arithmetic exactly,
    # and must never touch the gate stores (no measurement cost on old arms).
    f = make_filter(signed_gate="off")
    a, g = torch.randn(4, 6), torch.randn(4, 6)
    f.update_anchor("w", a)
    out = f.signed_ema_matrix("w", g)
    assert torch.equal(out, signed(g, a))
    f.update_anchor("w", torch.randn(4, 6))
    f.signed_ema_matrix("w", torch.randn(4, 6))
    assert not f._gate_rho and not f._gate_m_version and not f._gate_age
    assert f.signed_gate_refreshed == 0 and f.signed_gate_held == 0


def test_first_interval_is_use_once():
    # rho is unmeasured (0) until the SECOND fire, so every held tick of the
    # first interval is a pass-through: the exact use-once (freshm) dose.
    f = make_filter()
    a1 = torch.randn(4, 6)
    f.update_anchor("w", a1)
    g0 = torch.randn(4, 6)
    assert torch.allclose(f.signed_ema_matrix("w", g0), signed(g0, a1), atol=1e-6)  # fire: full strength
    g1 = torch.randn(4, 6)
    out1 = f.signed_ema_matrix("w", g1)
    assert out1 is g1  # zero weight skips the fp32 round trip entirely
    g2 = torch.randn(4, 6)
    assert f.signed_ema_matrix("w", g2) is g2
    assert f.signed_gate_refreshed == 1 and f.signed_gate_held == 2
    assert f._gate_last_w == 0.0


def test_rho_measured_at_second_fire_and_gates_held_ticks():
    # Second fire grades the held direction against the fresh anchor grad.
    # a2 = 2*a1 agrees in sign everywhere -> rho_raw = 1 -> rho = BETA.
    f = make_filter()
    a1 = torch.randn(4, 6)
    f.update_anchor("w", a1)
    f.signed_ema_matrix("w", torch.randn(4, 6))  # fire tick 1
    f.signed_ema_matrix("w", torch.randn(4, 6))  # held, w=0
    a2 = 2.0 * a1
    f.update_anchor("w", a2)  # rho <- 0.5*0 + 0.5*1
    assert f._gate_rho["w"] == pytest.approx(GATE_RHO_EMA_BETA)
    gf = torch.randn(4, 6)
    assert torch.allclose(f.signed_ema_matrix("w", gf), signed(gf, a2), atol=1e-6)  # fire: full strength
    gh = torch.randn(4, 6)
    w = GATE_RHO_EMA_BETA * DECAY  # age 1
    expect = (1.0 - w) * gh + w * signed(gh, a2)
    assert torch.allclose(f.signed_ema_matrix("w", gh), expect, atol=1e-6)
    assert f._gate_last_w == pytest.approx(w)
    gh2 = torch.randn(4, 6)
    w2 = GATE_RHO_EMA_BETA * DECAY**2  # age 2
    expect2 = (1.0 - w2) * gh2 + w2 * signed(gh2, a2)
    assert torch.allclose(f.signed_ema_matrix("w", gh2), expect2, atol=1e-6)


def test_rho_ema_accumulates_across_fires():
    f = make_filter()
    a = torch.randn(4, 6)
    f.update_anchor("w", a)
    f.signed_ema_matrix("w", a.clone())
    f.update_anchor("w", a)  # agreement 1: rho = 0.5
    f.signed_ema_matrix("w", a.clone())
    f.update_anchor("w", a)  # agreement 1 again: rho = 0.75
    assert f._gate_rho["w"] == pytest.approx(0.75)
    f.signed_ema_matrix("w", a.clone())  # fire tick
    gh = torch.randn(4, 6)
    w = 0.75 * DECAY
    expect = (1.0 - w) * gh + w * signed(gh, a)
    assert torch.allclose(f.signed_ema_matrix("w", gh), expect, atol=1e-6)


def test_anticorrelated_direction_is_graded_to_zero():
    # A held direction the next fire fully disagrees with earns NO dose:
    # rho_raw = mean(sign(a1)*sign(-a1)) = -1, clamped to 0.
    f = make_filter()
    a1 = torch.randn(4, 6)
    f.update_anchor("w", a1)
    f.signed_ema_matrix("w", torch.randn(4, 6))
    f.update_anchor("w", -a1)
    assert f._gate_rho["w"] == pytest.approx(0.0)
    f.signed_ema_matrix("w", torch.randn(4, 6))  # fire tick
    gh = torch.randn(4, 6)
    assert f.signed_ema_matrix("w", gh) is gh  # held: still use-once


def test_age_resets_on_refresh():
    f = make_filter()
    a = torch.randn(4, 6)
    f.update_anchor("w", a)
    f.signed_ema_matrix("w", torch.randn(4, 6))
    f.update_anchor("w", a)  # rho = 0.5
    f.signed_ema_matrix("w", torch.randn(4, 6))  # fire tick
    f.signed_ema_matrix("w", torch.randn(4, 6))  # age 1
    f.signed_ema_matrix("w", torch.randn(4, 6))  # age 2
    assert f._gate_age["w"] == 2
    f.update_anchor("w", a)  # third fire: rho = 0.75
    gf = torch.randn(4, 6)
    assert torch.allclose(f.signed_ema_matrix("w", gf), signed(gf, a), atol=1e-6)  # full strength again
    assert f._gate_age["w"] == 0
    gh = torch.randn(4, 6)
    w = 0.75 * DECAY  # age restarts at 1
    expect = (1.0 - w) * gh + w * signed(gh, a)
    assert torch.allclose(f.signed_ema_matrix("w", gh), expect, atol=1e-6)


def test_standing_reuse_is_unreachable():
    # Even at the rho ceiling (perfect measured agreement forever), the held
    # weight is bounded by decay**age < 1 and vanishes with age: the 3-for-3
    # dead configuration (w = 1 on every tick) cannot be expressed.
    f = make_filter()
    a = torch.randn(4, 6)
    f.update_anchor("w", a)
    f.signed_ema_matrix("w", a.clone())
    for _ in range(20):  # drive rho toward its ceiling
        f.update_anchor("w", a)
        f.signed_ema_matrix("w", a.clone())
    assert f._gate_rho["w"] <= 1.0
    weights = []
    for age in range(1, 6):
        f.signed_ema_matrix("w", torch.randn(4, 6))
        weights.append(f._gate_last_w)
        assert f._gate_last_w == pytest.approx(f._gate_rho["w"] * DECAY**age)
    assert all(w2 < w1 for w1, w2 in zip(weights, weights[1:], strict=False))
    assert max(weights) < 1.0


def test_cold_m_pops_gate_schedule_and_counts_fallback():
    f = make_filter()
    a = torch.randn(4, 6)
    f.update_anchor("w", a)
    f.signed_ema_matrix("w", torch.randn(4, 6))
    assert "w" in f._gate_m_version
    g_new = torch.randn(3, 5)  # logical shape change resets M to zeros
    out = f.signed_ema_matrix("w", g_new)
    assert torch.equal(out, g_new)
    assert f.merger_coldM_fallbacks == 1
    assert "w" not in f._gate_m_version and "w" not in f._gate_age


def test_counter_arithmetic_over_a_two_interval_schedule():
    # cadence-1 simulation, two 4-tick intervals: per target, 2 fire ticks and
    # 6 held ticks. The watcher factors these as n_targets x fires and
    # n_targets x held ticks.
    f = make_filter()
    for _ in range(2):
        f.update_anchor("w", torch.randn(4, 6))
        f.signed_ema_matrix("w", torch.randn(4, 6))  # fire tick
        for _ in range(3):
            f.signed_ema_matrix("w", torch.randn(4, 6))  # held ticks
    assert f.signed_gate_refreshed == 2
    assert f.signed_gate_held == 6
    assert f.merger_coldM_fallbacks == 0


def test_gate_never_touches_delayed_ef_stores():
    f = make_filter()
    f.update_anchor("w", torch.randn(4, 6))
    f.signed_ema_matrix("w", torch.randn(4, 6))
    f.signed_ema_matrix("w", torch.randn(4, 6))
    assert not f._delayed_ef_delta and f.delayed_ef_refreshed == 0 and f.delayed_ef_held == 0


def test_gate_rho_mean_telemetry():
    f = make_filter()
    assert f.gate_rho_mean() == 0.0
    a1, b1 = torch.randn(4, 6), torch.randn(4, 6)
    for name, t in (("w1", a1), ("w2", b1)):
        f.update_anchor(name, t)
        f.signed_ema_matrix(name, torch.randn(4, 6))
    f.update_anchor("w1", a1)  # agreement 1 -> rho 0.5
    f.update_anchor("w2", -b1)  # agreement -1 -> rho 0
    assert f.gate_rho_mean() == pytest.approx(0.25)


def test_constructor_rejects_bad_gate_and_decay():
    with pytest.raises(ValueError):
        make_filter(signed_gate="banana")
    with pytest.raises(ValueError):
        make_filter(signed_gate_decay=1.0)  # decay=1 is standing reuse; refused
    with pytest.raises(ValueError):
        make_filter(signed_gate_decay=-0.1)


def test_config_validator_accepts_and_rejects():
    from verl.workers.config.comm_eff import CommEffConfig, CommEffSpectralConfig

    cfg = CommEffConfig(
        enabled=True,
        spectral=CommEffSpectralConfig(correction_mode="signed_ema", signed_gate="learned", signed_gate_decay=0.75),
    )
    assert cfg.spectral.signed_gate == "learned"

    with pytest.raises(ValueError, match="signed_gate"):
        CommEffConfig(enabled=True, spectral=CommEffSpectralConfig(signed_gate="banana"))
    with pytest.raises(ValueError, match="signed_gate_decay"):
        CommEffConfig(
            enabled=True,
            spectral=CommEffSpectralConfig(signed_gate="learned", signed_gate_decay=1.0),
        )
    # The gate is read only by the signed path; on any other merger it would
    # be a silently ignored knob, i.e. a mislabelled experiment.
    with pytest.raises(ValueError, match="signed_gate"):
        CommEffConfig(
            enabled=True,
            spectral=CommEffSpectralConfig(correction_mode="delayed_ef", signed_gate="learned"),
        )
