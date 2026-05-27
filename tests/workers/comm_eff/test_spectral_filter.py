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

"""Unit tests for the comm_eff spectral correction filter (EXP-7, M2).

These cover the formula-correctness invariants the EXP-7 plan requires
codex-verify to gate on, none of which need a GPU or a distributed runtime
(the filter operates on logical 2D matrices; FSDP unsharding is the engine's
job, deliberately decoupled from this module):

* alpha=1.0  => G_proj == G_mask           (max abs diff <= 1e-6), any anchor
* alpha=0    => G_proj == pure two-sided Tikhonov projection (<= 1e-6)
* shape preservation for representative square AND rectangular 2D matrices
* determinism for a fixed seed (seeded anchor cache reproduces the result)
* rel_change is faithful: 0 at alpha=1, strictly >0 at alpha=0.3
"""

import pytest
import torch

from verl.workers.comm_eff.spectral_filter import (
    SpectralFilter,
    spectral_correct,
    tikhonov_weights,
    two_sided_projection,
)

TOL = 1e-6


def _rand_matrix(m, n, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(m, n, generator=g, dtype=torch.float64)


def _rand_anchor(m, n, seed=1):
    """A generic (not necessarily PSD) anchor matrix for the standalone math."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(m, n, generator=g, dtype=torch.float64)


# --------------------------------------------------------------------------- #
# alpha = 1.0  =>  exact no-op
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("shape", [(8, 8), (12, 5), (5, 12), (1, 7), (7, 1)])
def test_alpha_one_is_exact_noop(shape):
    m, n = shape
    g_mask = _rand_matrix(m, n, seed=3)
    anchor = _rand_anchor(m, n, seed=4)
    g_proj = spectral_correct(g_mask, anchor, alpha=1.0, tau=1e-3)
    assert g_proj.shape == g_mask.shape
    assert torch.max(torch.abs(g_proj - g_mask)).item() <= TOL


def test_alpha_one_noop_independent_of_anchor():
    # Two very different anchors must both yield G_proj == G_mask at alpha=1.
    g_mask = _rand_matrix(10, 6, seed=5)
    for s in (0, 1, 99):
        anchor = _rand_anchor(10, 6, seed=s) * (s + 1) * 10.0
        g_proj = spectral_correct(g_mask, anchor, alpha=1.0, tau=1e-3)
        assert torch.max(torch.abs(g_proj - g_mask)).item() <= TOL


# --------------------------------------------------------------------------- #
# alpha = 0  =>  pure two-sided Tikhonov projection
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("shape", [(8, 8), (12, 5), (5, 12)])
def test_alpha_zero_is_pure_two_sided_projection(shape):
    m, n = shape
    g_mask = _rand_matrix(m, n, seed=7)
    anchor = _rand_anchor(m, n, seed=8)
    tau = 1e-3

    g_proj = spectral_correct(g_mask, anchor, alpha=0.0, tau=tau)

    # Reference: recompute the projection independently from the SVD.
    u, s, vh = torch.linalg.svd(anchor, full_matrices=False)
    v = vh.transpose(-1, -2)
    d = tikhonov_weights(s, tau)
    ref = two_sided_projection(g_mask, u, d, v)

    assert g_proj.shape == g_mask.shape
    assert torch.max(torch.abs(g_proj - ref)).item() <= TOL


def test_tikhonov_weights_formula():
    s = torch.tensor([10.0, 1.0, 0.0])
    tau = 1e-3
    d = tikhonov_weights(s, tau)
    expected = s / (s + tau)
    assert torch.allclose(d, expected, atol=0, rtol=0)
    # zero singular value => weight exactly 0 (well-defined, no div-by-zero)
    assert d[-1].item() == 0.0


def test_blend_is_convex_combination():
    # G_proj = alpha*G_mask + (1-alpha)*G_filt must lie exactly on that line.
    m, n = 9, 9
    g_mask = _rand_matrix(m, n, seed=11)
    anchor = _rand_anchor(m, n, seed=12)
    tau = 1e-3
    alpha = 0.3

    g0 = spectral_correct(g_mask, anchor, alpha=0.0, tau=tau)
    ga = spectral_correct(g_mask, anchor, alpha=alpha, tau=tau)
    blended = alpha * g_mask + (1.0 - alpha) * g0
    assert torch.max(torch.abs(ga - blended)).item() <= TOL


# --------------------------------------------------------------------------- #
# shape preservation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("shape", [(16, 16), (32, 8), (8, 32), (1, 5), (5, 1), (1, 1)])
def test_shape_preserved(shape):
    m, n = shape
    g_mask = _rand_matrix(m, n, seed=13)
    anchor = _rand_anchor(m, n, seed=14)
    for alpha in (0.0, 0.3, 1.0):
        g_proj = spectral_correct(g_mask, anchor, alpha=alpha, tau=1e-3)
        assert g_proj.shape == g_mask.shape


def test_non_2d_input_rejected():
    g3d = torch.randn(2, 3, 4)
    anchor = torch.randn(2, 3, 4)
    with pytest.raises(AssertionError):
        spectral_correct(g3d, anchor, alpha=0.3, tau=1e-3)


def test_anchor_shape_mismatch_rejected():
    g = torch.randn(8, 6)
    bad_anchor = torch.randn(6, 8)
    with pytest.raises(AssertionError):
        spectral_correct(g, bad_anchor, alpha=0.3, tau=1e-3)


# --------------------------------------------------------------------------- #
# SpectralFilter: seeded anchor cache + determinism
# --------------------------------------------------------------------------- #
def test_seeded_anchor_is_deterministic():
    g_mask = _rand_matrix(12, 7, seed=20).to(torch.float32)
    f1 = SpectralFilter(alpha=0.3, tau=1e-3, beta_anc=0.95, seed_anchor_cache=True, anchor_seed=42)
    f2 = SpectralFilter(alpha=0.3, tau=1e-3, beta_anc=0.95, seed_anchor_cache=True, anchor_seed=42)
    out1 = f1.correct_matrix("model.layers.0.self_attn.q_proj.weight", g_mask.clone())
    out2 = f2.correct_matrix("model.layers.0.self_attn.q_proj.weight", g_mask.clone())
    assert out1.shape == g_mask.shape
    assert torch.max(torch.abs(out1 - out2)).item() <= TOL


def test_seeded_anchor_shape_matches_target():
    f = SpectralFilter(seed_anchor_cache=True, anchor_seed=0)
    for shape in [(16, 16), (32, 8), (8, 32)]:
        g = torch.randn(*shape, dtype=torch.float32)
        anc = f.ensure_anchor(f"p{shape}", g)
        assert anc.shape == g.shape
        out = f.correct_matrix(f"p{shape}", g)
        assert out.shape == g.shape


def test_alpha_one_noop_through_filter():
    f = SpectralFilter(alpha=1.0, tau=1e-3, seed_anchor_cache=True, anchor_seed=1)
    g = torch.randn(10, 6, dtype=torch.float32)
    out = f.correct_matrix("w", g.clone())
    assert torch.max(torch.abs(out - g)).item() <= TOL
    assert f.relative_change(g, out) == 0.0


def test_rel_change_active_at_alpha_0p3():
    # The EXP-7 operating point: correction must actually fire (rel_change > 0).
    f = SpectralFilter(alpha=0.3, tau=1e-3, seed_anchor_cache=True, anchor_seed=2)
    g = torch.randn(12, 12, dtype=torch.float32)
    out = f.correct_matrix("w", g.clone())
    rel = f.relative_change(g, out)
    assert rel > 0.0  # correction is not a silent no-op
    assert out.shape == g.shape


def test_ema_update_moves_anchor():
    f = SpectralFilter(beta_anc=0.5, seed_anchor_cache=False)
    g_anchor = torch.ones(4, 4, dtype=torch.float32)
    a0 = f.ensure_anchor("w", g_anchor).clone()  # zeros (unseeded)
    a1 = f.update_anchor("w", g_anchor)
    # beta=0.5: M <- 0.5*0 + 0.5*1 = 0.5
    assert torch.allclose(a1, torch.full((4, 4), 0.5), atol=TOL)
    assert not torch.allclose(a0, a1)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
