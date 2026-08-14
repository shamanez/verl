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

"""CPU unit tests for the blend merger (spectral.correction_mode=blend).

Semantics under test, the EXP-30 B1 formula ported verbatim:

    scale  = ||G_comp|| / (||M|| + eps)
    G_corr = (1 - eta) * G_comp + eta * scale * M

Convex value merger: no sign transplant, no held residual, and for orthogonal
terms ||G_corr|| = ||G_comp|| * sqrt((1-eta)^2 + eta^2) <= ||G_comp||.
"""

import math

import pytest
import torch

from verl.workers.comm_eff.spectral_filter import SpectralFilter


def make_filter(**kw):
    defaults = dict(
        beta_anc=0.0,
        ema_device="cpu",
        signed_ema_alpha=0.25,
        correction_mode="blend",
        blend_eta=0.3,
        diagnostics=False,
    )
    defaults.update(kw)
    return SpectralFilter(**defaults)


def test_eta_zero_returns_g_comp_exactly():
    f = make_filter(blend_eta=0.0)
    g = torch.randn(4, 6)
    out = f.blend_matrix("w", g)
    assert out is g  # bitwise identity, same object


def test_cold_m_is_a_noop_and_counts_fallback():
    f = make_filter()
    g = torch.randn(4, 6)
    out = f.blend_matrix("w", g)
    assert torch.equal(out, g)
    assert f.merger_coldM_fallbacks == 1


def test_zero_norm_g_is_a_noop():
    f = make_filter()
    f.update_anchor("w", torch.randn(4, 6))
    g = torch.zeros(4, 6)
    out = f.blend_matrix("w", g)
    assert torch.equal(out, g)
    assert f.merger_coldM_fallbacks == 1


def test_eta_one_is_scale_matched_anchor():
    f = make_filter(blend_eta=1.0)
    m = torch.randn(4, 6)
    g = torch.randn(4, 6) * 25.0  # the inflated compressed carrier scale
    f.update_anchor("w", m)
    out = f.blend_matrix("w", g)
    expect = (torch.linalg.norm(g) / (torch.linalg.norm(m) + 1e-12)) * m
    assert torch.allclose(out, expect, atol=1e-4)
    # Norm preserved at the carrier scale.
    assert abs(torch.linalg.norm(out).item() - torch.linalg.norm(g).item()) < 1e-2


def test_formula_exact_at_eta_0p3():
    f = make_filter(blend_eta=0.3)
    m, g = torch.randn(4, 6), torch.randn(4, 6)
    f.update_anchor("w", m)
    out = f.blend_matrix("w", g)
    scale = torch.linalg.norm(g) / (torch.linalg.norm(m) + 1e-12)
    expect = 0.7 * g + 0.3 * scale * m
    assert torch.allclose(out, expect, atol=1e-5)


def test_magnitude_bounded_for_orthogonal_terms():
    # Construct exactly orthogonal G and M; the convex bound must hold:
    # ||G_corr|| = ||G|| * sqrt((1-eta)^2 + eta^2).
    eta = 0.7
    f = make_filter(blend_eta=eta)
    g = torch.zeros(2, 2)
    g[0, 0] = 3.0
    m = torch.zeros(2, 2)
    m[1, 1] = 40.0  # different scale, orthogonal direction
    f.update_anchor("w", m)
    out = f.blend_matrix("w", g)
    expect_norm = 3.0 * math.sqrt((1 - eta) ** 2 + eta**2)
    assert abs(torch.linalg.norm(out).item() - expect_norm) < 1e-4
    assert torch.linalg.norm(out).item() <= 3.0 + 1e-6


def test_finds_anchor_across_fsdp_infix():
    f = make_filter()
    f.update_anchor("model.layers.0.self_attn.q_proj.weight", torch.randn(4, 6))
    g = torch.randn(4, 6)
    out = f.blend_matrix("model.layers.0._fsdp_wrapped_module.self_attn.q_proj.weight", g)
    assert f.merger_coldM_fallbacks == 0
    assert not torch.equal(out, g)  # the blend actually fired


def test_correct_matrix_dispatch_and_isolation():
    m, g = torch.randn(4, 6), torch.randn(4, 6)
    fb = make_filter()
    fb.update_anchor("w", m)
    out = fb.correct_matrix("w", g)
    scale = torch.linalg.norm(g) / (torch.linalg.norm(m) + 1e-12)
    assert torch.allclose(out, 0.7 * g + 0.3 * scale * m, atol=1e-5)
    # blend mode never touches the delayed_ef stores.
    assert not fb._delayed_ef_delta and fb.delayed_ef_refreshed == 0 and fb.delayed_ef_held == 0


def test_no_held_state_between_ticks():
    # Between fires the blend keeps consuming the SAME M against each tick's
    # fresh G; there is no residual carried from earlier ticks.
    f = make_filter()
    m = torch.randn(4, 6)
    f.update_anchor("w", m)
    g1, g2 = torch.randn(4, 6), torch.randn(4, 6)
    f.blend_matrix("w", g1)
    out2 = f.blend_matrix("w", g2)
    scale2 = torch.linalg.norm(g2) / (torch.linalg.norm(m) + 1e-12)
    assert torch.allclose(out2, 0.7 * g2 + 0.3 * scale2 * m, atol=1e-5)


def test_constructor_and_config_validation():
    with pytest.raises(ValueError):
        make_filter(blend_eta=1.5)
    with pytest.raises(ValueError):
        make_filter(blend_eta=-0.1)

    from verl.workers.config.comm_eff import CommEffConfig, CommEffSpectralConfig

    cfg = CommEffConfig(
        enabled=True,
        spectral=CommEffSpectralConfig(correction_mode="blend", blend_eta=0.3, beta_anc=0.0),
    )
    assert cfg.spectral.correction_mode == "blend"
    with pytest.raises(ValueError, match="blend_eta"):
        CommEffConfig(enabled=True, spectral=CommEffSpectralConfig(correction_mode="blend", blend_eta=2.0))
