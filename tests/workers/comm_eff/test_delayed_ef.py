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

"""CPU unit tests for the delayed_ef merger (spectral.correction_mode).

Semantics under test (the figure semantics, same-tick pairing):

    delta      = M - G_comp          refreshed once per anchor fire, on the
                                     first correction after update_anchor()
    G_corr(t)  = G_comp(t) + lambda * delta

At lambda=1 and beta_anc=0 the fire tick returns the anchor gradient exactly;
between fires the HELD delta is re-applied to each new fast gradient.
"""

import pytest
import torch

from verl.workers.comm_eff.spectral_filter import SpectralFilter


def make_filter(**kw):
    defaults = dict(
        beta_anc=0.0,
        ema_device="cpu",
        signed_ema_alpha=0.25,
        correction_mode="delayed_ef",
        delayed_ef_lambda=1.0,
        diagnostics=False,
    )
    defaults.update(kw)
    return SpectralFilter(**defaults)


def test_lambda_zero_is_bitwise_identity():
    f = make_filter(delayed_ef_lambda=0.0)
    g = torch.randn(4, 6)
    out = f.delayed_ef_matrix("w", g)
    assert out is g  # the exact same object, no fp32 round trip


def test_cold_m_is_a_noop_and_counts_fallback():
    f = make_filter()
    g = torch.randn(4, 6)
    out = f.delayed_ef_matrix("w", g)
    assert torch.equal(out, g)
    assert f.merger_coldM_fallbacks == 1
    assert f.delayed_ef_refreshed == 0 and f.delayed_ef_held == 0


def test_fire_tick_at_lambda1_beta0_returns_anchor_exactly():
    f = make_filter()
    g_anchor = torch.randn(4, 6)
    g_comp = torch.randn(4, 6)
    f.update_anchor("w", g_anchor)  # the fire refreshes M = raw anchor grad
    out = f.delayed_ef_matrix("w", g_comp)
    assert torch.allclose(out, g_anchor, atol=1e-6)
    assert f.delayed_ef_refreshed == 1
    assert f.delayed_ef_held == 0


def test_held_delta_reapplied_between_fires():
    f = make_filter(delayed_ef_lambda=0.5)
    g_anchor = torch.randn(4, 6)
    g1 = torch.randn(4, 6)
    f.update_anchor("w", g_anchor)
    out1 = f.delayed_ef_matrix("w", g1)  # refresh: delta = M - g1
    delta = g_anchor - g1
    assert torch.allclose(out1, g1 + 0.5 * delta, atol=1e-6)
    # No new fire: the SAME delta applies to a different fast gradient.
    g2 = torch.randn(4, 6)
    out2 = f.delayed_ef_matrix("w", g2)
    assert torch.allclose(out2, g2 + 0.5 * delta, atol=1e-6)
    assert f.delayed_ef_refreshed == 1
    assert f.delayed_ef_held == 1


def test_second_fire_refreshes_delta():
    f = make_filter()
    a1, g1 = torch.randn(4, 6), torch.randn(4, 6)
    f.update_anchor("w", a1)
    f.delayed_ef_matrix("w", g1)
    a2, g2 = torch.randn(4, 6), torch.randn(4, 6)
    f.update_anchor("w", a2)  # second fire; beta_anc=0 => M = a2
    out = f.delayed_ef_matrix("w", g2)
    assert torch.allclose(out, a2, atol=1e-6)  # lambda=1: fire tick == anchor grad
    assert f.delayed_ef_refreshed == 2


def test_cad10_schedule_one_stale_application_per_interval():
    # Arm J (delayedef-cad10): corrections on ticks 10,20,30,40 with fires on
    # 20,40. Tick 10 is a cold-M no-op, tick 20 refreshes (G_corr = anchor),
    # tick 30 re-applies the HELD tick-20 delta ONCE to a new fast gradient,
    # tick 40 refreshes again. Fingerprint: held == refreshed (1:1), one
    # coldM fallback from the pre-fire tick.
    f = make_filter()
    g10 = torch.randn(4, 6)
    assert torch.equal(f.delayed_ef_matrix("w", g10), g10)  # tick 10: cold M
    a20, g20 = torch.randn(4, 6), torch.randn(4, 6)
    f.update_anchor("w", a20)  # fire tick 20
    assert torch.allclose(f.delayed_ef_matrix("w", g20), a20, atol=1e-6)
    g30 = torch.randn(4, 6)  # tick 30: held delta from tick 20, 10 ticks stale
    out30 = f.delayed_ef_matrix("w", g30)
    assert torch.allclose(out30, g30 + (a20 - g20), atol=1e-6)
    a40, g40 = torch.randn(4, 6), torch.randn(4, 6)
    f.update_anchor("w", a40)  # fire tick 40
    assert torch.allclose(f.delayed_ef_matrix("w", g40), a40, atol=1e-6)
    g50 = torch.randn(4, 6)  # tick 50: held delta from tick 40
    assert torch.allclose(f.delayed_ef_matrix("w", g50), g50 + (a40 - g40), atol=1e-6)
    # Two completed intervals: one refresh and one held application each.
    assert f.delayed_ef_refreshed == 2
    assert f.delayed_ef_held == 2  # the 1:1 arm fingerprint (arm G 19:1, arm H 0:1)
    assert f.merger_coldM_fallbacks == 1


def test_beta_anc_ema_enters_the_residual():
    # With beta_anc > 0, M after the second fire is an EMA, and the fire-tick
    # correction at lambda=1 returns M (not the raw second anchor grad).
    f = make_filter(beta_anc=0.5)
    a1, a2 = torch.randn(4, 6), torch.randn(4, 6)
    f.update_anchor("w", a1)  # M = 0.5*0 + 0.5*a1
    f.delayed_ef_matrix("w", torch.randn(4, 6))
    f.update_anchor("w", a2)  # M = 0.5*(0.5*a1) + 0.5*a2
    m = 0.25 * a1 + 0.5 * a2
    out = f.delayed_ef_matrix("w", torch.randn(4, 6))
    assert torch.allclose(out, m, atol=1e-5)


def test_shape_change_drops_held_delta_and_falls_back():
    f = make_filter()
    f.update_anchor("w", torch.randn(4, 6))
    f.delayed_ef_matrix("w", torch.randn(4, 6))
    assert "w" in f._delayed_ef_delta
    g_new = torch.randn(3, 5)  # logical shape change
    out = f.delayed_ef_matrix("w", g_new)
    assert torch.equal(out, g_new)
    assert "w" not in f._delayed_ef_delta
    assert f.merger_coldM_fallbacks == 1


def test_correct_matrix_dispatch():
    g_anchor = torch.randn(4, 6)
    g_comp = torch.randn(4, 6)
    fd = make_filter()
    fd.update_anchor("w", g_anchor)
    assert torch.allclose(fd.correct_matrix("w", g_comp), g_anchor, atol=1e-6)
    fs = make_filter(correction_mode="signed_ema", beta_anc=0.0, signed_ema_alpha=0.25)
    fs.update_anchor("w", g_anchor)
    expect = 0.25 * g_comp + 0.75 * g_comp.abs() * torch.sign(g_anchor)
    assert torch.allclose(fs.correct_matrix("w", g_comp), expect, atol=1e-6)
    # And signed_ema mode never touches the delayed_ef stores.
    assert not fs._delayed_ef_delta and fs.delayed_ef_refreshed == 0


def test_fsdp_name_canonicalization_shares_one_delta():
    f = make_filter()
    f.update_anchor("model.layers.0.self_attn.q_proj.weight", torch.randn(4, 6))
    out = f.delayed_ef_matrix("model.layers.0._fsdp_wrapped_module.self_attn.q_proj.weight", torch.randn(4, 6))
    assert f.delayed_ef_refreshed == 1
    assert f.merger_coldM_fallbacks == 0
    assert out.shape == (4, 6)


def test_constructor_rejects_bad_mode_and_lambda():
    with pytest.raises(ValueError):
        make_filter(correction_mode="nope")
    with pytest.raises(ValueError):
        make_filter(delayed_ef_lambda=-0.5)


def test_config_validator_accepts_and_rejects():
    from verl.workers.config.comm_eff import CommEffConfig, CommEffSpectralConfig

    # The 4B delayedef arm shape: delayed_ef at lambda=1 with beta_anc=0.
    cfg = CommEffConfig(
        enabled=True,
        spectral=CommEffSpectralConfig(correction_mode="delayed_ef", delayed_ef_lambda=1.0, beta_anc=0.0),
    )
    assert cfg.spectral.correction_mode == "delayed_ef"

    with pytest.raises(ValueError, match="correction_mode"):
        CommEffConfig(enabled=True, spectral=CommEffSpectralConfig(correction_mode="ring_pairing"))

    with pytest.raises(ValueError, match="delayed_ef_lambda"):
        CommEffConfig(
            enabled=True,
            spectral=CommEffSpectralConfig(correction_mode="delayed_ef", delayed_ef_lambda=-1.0),
        )
