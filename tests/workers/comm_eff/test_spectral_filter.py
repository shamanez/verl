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

import importlib.util
import pathlib
import sys
import types

import pytest
import torch

# Load spectral_filter by FILE PATH so the heavy verl.__init__ chain (tensordict
# / vllm / ray, absent on the CPU dev box) is not imported — same harness as
# tests/workers/comm_eff/test_grad_correction_hook.py. The module under test is
# pure-torch and FSDP-agnostic by design, so this is sufficient and runs on CPU.
_REPO = pathlib.Path(__file__).resolve().parents[3]
for _pkg in ("verl", "verl.workers", "verl.workers.comm_eff"):
    if _pkg not in sys.modules:
        _m = types.ModuleType(_pkg)
        _m.__path__ = []
        sys.modules[_pkg] = _m
_spec = importlib.util.spec_from_file_location(
    "verl.workers.comm_eff.spectral_filter", _REPO / "verl/workers/comm_eff/spectral_filter.py"
)
_sf = importlib.util.module_from_spec(_spec)
sys.modules["verl.workers.comm_eff.spectral_filter"] = _sf
_spec.loader.exec_module(_sf)

SpectralFilter = _sf.SpectralFilter
compute_basis = _sf.compute_basis
spectral_correct = _sf.spectral_correct
tikhonov_weights = _sf.tikhonov_weights
two_sided_projection = _sf.two_sided_projection

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


# =========================================================================== #
# EXP-8: svd_mode=lowrank reconstruction error is bounded and decreasing in rank
# =========================================================================== #
def _recon_error_at_rank(m_anchor, rank):
    """||M - U_r diag(S_r) V_r^T||_F for the rank-r lowrank basis of M."""
    u, s, v = compute_basis(m_anchor, svd_mode="lowrank", rank=rank)
    recon = u @ torch.diag(s) @ v.transpose(-1, -2)
    return torch.linalg.norm(m_anchor - recon).item()


@pytest.mark.parametrize("shape", [(16, 16), (24, 12), (12, 24)])
def test_lowrank_recon_error_bounded_and_decreasing_in_rank(shape):
    m, n = shape
    k = min(m, n)
    # A matrix with a genuine decaying spectrum so higher rank captures more.
    g = torch.Generator().manual_seed(101)
    a = torch.randn(m, k, generator=g, dtype=torch.float64)
    b = torch.randn(n, k, generator=g, dtype=torch.float64)
    spectrum = torch.logspace(0, -2, steps=k, dtype=torch.float64)  # 1 .. 0.01
    qa, _ = torch.linalg.qr(a)
    qb, _ = torch.linalg.qr(b)
    M = (qa * spectrum.unsqueeze(0)) @ qb.transpose(0, 1)

    errs = [_recon_error_at_rank(M, r) for r in range(1, k + 1)]
    # torch.svd_lowrank is RANDOMIZED, so adjacent ranks are not strictly
    # monotone; assert the trend over WELL-SEPARATED ranks (randomization noise
    # is small relative to a halving of the truncation level). The exact-SVD
    # fallback at q==k makes the full-rank point exact.
    lo_rank_err = errs[0]
    mid_rank_err = errs[(k // 2) - 1] if k >= 2 else errs[0]
    full_rank_err = errs[-1]
    # Bounded: every reconstruction error is finite and the ideal (Eckart-Young)
    # rank-r error is the tail energy, which is <= ||M||; allow randomization
    # slack but it must never exceed ||M|| by more than a small factor.
    norm_M = torch.linalg.norm(M).item()
    assert all(0.0 <= e <= norm_M + 1e-6 for e in errs), f"recon error unbounded: {errs} (||M||={norm_M})"
    # Decreasing trend: mid rank beats low rank; full rank reconstructs ~exactly.
    assert mid_rank_err < lo_rank_err, f"mid-rank not better than low-rank: {errs}"
    assert full_rank_err <= 1e-6, f"full-rank reconstruction not exact: {full_rank_err}"
    assert lo_rank_err > full_rank_err, f"low rank should leave residual: {errs}"


def test_lowrank_correct_matrix_runs_and_preserves_shape():
    f = SpectralFilter(alpha=0.3, tau=1e-3, seed_anchor_cache=True, anchor_seed=3, svd_mode="lowrank", rank=4)
    for shape in [(16, 16), (32, 8), (8, 32)]:
        g = torch.randn(*shape, dtype=torch.float32)
        out = f.correct_matrix(f"w{shape}", g.clone())
        assert out.shape == g.shape
        assert torch.isfinite(out).all()


# =========================================================================== #
# EXP-8: ema_device=cpu round-trip yields the same M_anchor as on-device
# =========================================================================== #
def test_ema_device_cpu_roundtrip_equals_on_device():
    # On CPU-only CI both "gpu" and "cpu" storage resolve to CPU tensors, so the
    # EMA arithmetic must be identical; the test guards the offload-move logic
    # (anchor_on / store-back) does not corrupt values. Same seeds, same grads.
    g1 = torch.randn(8, 6, generator=torch.Generator().manual_seed(7), dtype=torch.float32)
    g2 = torch.randn(8, 6, generator=torch.Generator().manual_seed(8), dtype=torch.float32)

    f_gpu = SpectralFilter(beta_anc=0.9, seed_anchor_cache=False, ema_device="gpu")
    f_cpu = SpectralFilter(beta_anc=0.9, seed_anchor_cache=False, ema_device="cpu")

    for f in (f_gpu, f_cpu):
        f.update_anchor("w", g1)
        f.update_anchor("w", g2)

    m_gpu = f_gpu.anchor_on("w", torch.device("cpu"))
    m_cpu = f_cpu.anchor_on("w", torch.device("cpu"))
    assert torch.allclose(m_gpu, m_cpu, atol=TOL), "cpu-offloaded EMA diverged from on-device EMA"
    # The cpu-storage EMA tensor actually lives on CPU between refreshes.
    assert f_cpu._anchor["w"].device.type == "cpu"


def test_ema_device_cpu_correct_matrix_matches_gpu():
    g_anchor = torch.randn(10, 10, generator=torch.Generator().manual_seed(11), dtype=torch.float32)
    g_mask = torch.randn(10, 10, generator=torch.Generator().manual_seed(12), dtype=torch.float32)

    f_gpu = SpectralFilter(alpha=0.3, tau=1e-3, beta_anc=0.9, seed_anchor_cache=False, ema_device="gpu")
    f_cpu = SpectralFilter(alpha=0.3, tau=1e-3, beta_anc=0.9, seed_anchor_cache=False, ema_device="cpu")
    for f in (f_gpu, f_cpu):
        f.update_anchor("w", g_anchor)

    out_gpu = f_gpu.correct_matrix("w", g_mask.clone())
    out_cpu = f_cpu.correct_matrix("w", g_mask.clone())
    assert torch.allclose(out_gpu, out_cpu, atol=1e-5)


# =========================================================================== #
# EXP-8: basis_cache=recompute is numerically equal to basis_cache=cache
# =========================================================================== #
def test_basis_cache_recompute_equals_cache():
    g_anchor = torch.randn(12, 9, generator=torch.Generator().manual_seed(21), dtype=torch.float32)
    g_mask = torch.randn(12, 9, generator=torch.Generator().manual_seed(22), dtype=torch.float32)

    f_cache = SpectralFilter(alpha=0.3, tau=1e-3, beta_anc=0.9, seed_anchor_cache=False, basis_cache="cache")
    f_recompute = SpectralFilter(alpha=0.3, tau=1e-3, beta_anc=0.9, seed_anchor_cache=False, basis_cache="recompute")
    for f in (f_cache, f_recompute):
        f.update_anchor("w", g_anchor)  # cache mode populates _basis here

    # cache mode reuses the basis stored at the last update_anchor; recompute
    # recomputes SVD inside correct_matrix. Both act on the SAME M_anchor.
    out_cache = f_cache.correct_matrix("w", g_mask.clone())
    out_recompute = f_recompute.correct_matrix("w", g_mask.clone())
    assert torch.allclose(out_cache, out_recompute, atol=1e-5), "cache vs recompute diverged"
    # cache mode actually stored a basis; recompute mode did not.
    assert "w" in f_cache._basis
    assert "w" not in f_recompute._basis


def test_basis_cache_reused_across_correct_calls():
    """Under basis_cache=cache the SAME cached basis serves repeated
    correct_matrix calls between refreshes (the fast-mini-batch reuse path)."""
    f = SpectralFilter(alpha=0.3, tau=1e-3, beta_anc=0.9, seed_anchor_cache=False, basis_cache="cache")
    f.update_anchor("w", torch.randn(8, 8, generator=torch.Generator().manual_seed(31), dtype=torch.float32))
    basis_id = id(f._basis["w"])
    g = torch.randn(8, 8, generator=torch.Generator().manual_seed(32), dtype=torch.float32)
    f.correct_matrix("w", g.clone())
    f.correct_matrix("w", g.clone())
    # No refresh happened between the two corrections => the cached basis object
    # is unchanged (not recomputed per correction).
    assert id(f._basis["w"]) == basis_id


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
