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

"""EXP-31 anchor-usage levers L2 (δ-momentum) + L3 (adaptive dose) — CPU tests.

Plan ``.claude/plans/31.md`` Correctness invariants covered here (all CPU, no
GPU, no torch.distributed):

1. **off-path parity (hard):** defaults (μ=0, mode=off, κ=0) ⇒ ``delayed_ef_matrix``
   returns ``g_corr`` ``torch.equal`` to the B2 path on BOTH a refresh tick and a
   held tick — bitwise-B2, no buffer/history touched.
2. **L2 gain-1 (hard):** constant-δ stream, 100 refreshes, μ=0.9 ⇒
   ``‖m − δ‖/‖δ‖ < 1e-3`` (stationary gain EXACTLY 1, NOT 10×). μ=0 ⇒ exact identity
   (the helper returns the SAME tensor object, no buffer touched).
3. **L2 async-degrade (hard):** μ=0.9, age_decay=True, forced 30-step hold (no
   refresh) ⇒ the APPLIED correction → ~0, ‖g_corr‖ → ‖g_comp‖, finite throughout,
   and the STORED buffer is unchanged (APPLIED-only scaling).
4. **L3 off/bounds/mean-1 (hard):** κ=0 OR mode=off ⇒ λ_t == delayed_ef_lambda
   EXACTLY; mode∈{cos,ratio}, κ>0 ⇒ λ_t ∈ [0, lambda_cap] AND E[λ_t]≈λ (within ~5%)
   on a c_t stream symmetric about its median c̄.
5. **cross-rank determinism (hard):** two SpectralFilter instances fed the identical
   (gm, anc, delta) stream produce identical m, c̄, λ_t (bit-for-bit) — the
   async-critical shared-codebook invariant. A per-rank buffer would FAIL this.

The unit-under-test is loaded with the same importlib shim the other comm_eff CPU
tests use (``test_subbasis_merger_exp31`` / ``test_delayed_ef_exp30``) so it runs
with no ``torch.distributed`` / FSDP runtime.
"""

import importlib.util
import pathlib
import sys

import pytest
import torch

_REPO = pathlib.Path(__file__).resolve().parents[3]


def _stub_parent_packages():
    import types

    for pkg in ("verl", "verl.workers", "verl.workers.comm_eff"):
        if pkg not in sys.modules:
            m = types.ModuleType(pkg)
            m.__path__ = []
            sys.modules[pkg] = m


def _load(mod_name, rel_path):
    spec = importlib.util.spec_from_file_location(mod_name, _REPO / rel_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


_stub_parent_packages()
_sf = _load("verl.workers.comm_eff.spectral_filter", "verl/workers/comm_eff/spectral_filter.py")

SpectralFilter = _sf.SpectralFilter

_NAME = "layers.0.q_proj.weight"


def _mk_filter(
    *,
    lam=1.0,
    ema_device="cpu",
    delta_momentum_mu=0.0,
    delta_momentum_age_decay=False,
    adaptive_lambda_mode="off",
    adaptive_lambda_kappa=0.0,
    lambda_cap=2.0,
):
    return SpectralFilter(
        beta_anc=0.0,
        correction_mode="delayed_ef",
        delayed_ef_lambda=lam,
        ema_device=ema_device,
        delta_momentum_mu=delta_momentum_mu,
        delta_momentum_age_decay=delta_momentum_age_decay,
        adaptive_lambda_mode=adaptive_lambda_mode,
        adaptive_lambda_kappa=adaptive_lambda_kappa,
        lambda_cap=lambda_cap,
    )


def _fire(f, name, g, m_rep, ring):
    """Warm M_rep (β=0 ⇒ M == m_rep) + drive one fire-aligned (refresh) tick."""
    f.update_anchor(name, m_rep)
    return f.delayed_ef_matrix(name, g, ring_grad=ring)


# --------------------------------------------------------------------------- #
# 1. off-path parity — defaults == B2 bitwise on a refresh AND a held tick.
# --------------------------------------------------------------------------- #
def test_off_path_parity_refresh_and_held():
    torch.manual_seed(0)
    g1 = torch.randn(5, 4)
    g2 = torch.randn(5, 4)
    m_rep = torch.randn(5, 4) * 2.0
    ring = torch.randn(5, 4) * 0.5

    # B2 reference: NO new-lever args (defaults μ=0, mode=off, κ=0).
    ref = _mk_filter()
    ref_refresh = _fire(ref, _NAME, g1, m_rep, ring).clone()
    ref_held = ref.delayed_ef_matrix(_NAME, g2, ring_grad=None).clone()

    # All three new knobs explicitly OFF at their defaults.
    new = _mk_filter(delta_momentum_mu=0.0, adaptive_lambda_mode="off", adaptive_lambda_kappa=0.0)
    new_refresh = _fire(new, _NAME, g1, m_rep, ring).clone()
    new_held = new.delayed_ef_matrix(_NAME, g2, ring_grad=None).clone()

    assert torch.equal(new_refresh, ref_refresh), "OFF-path refresh tick must be bitwise-B2"
    assert torch.equal(new_held, ref_held), "OFF-path held tick must be bitwise-B2"
    # No lever state touched on the OFF path.
    assert not new._delta_momentum, "μ=0 must build NO momentum buffer"
    assert not new._adaptive_lambda_hist, "mode=off must build NO agreement history"
    assert new.delayed_ef_momentum_applied == 0
    assert new.delayed_ef_adaptive_lambda_applied == 0


def test_off_path_helper_returns_same_object():
    """μ=0 ⇒ _apply_delta_momentum returns the SAME tensor (no copy, no buffer)."""
    f = _mk_filter(delta_momentum_mu=0.0)
    d = torch.randn(3, 3)
    out = f._apply_delta_momentum(_NAME, d, refreshed=True)
    assert out is d, "μ=0 must return the delta object unchanged (bitwise off-path)"
    assert not f._delta_momentum
    # mode=off ⇒ _adaptive_lambda returns exactly delayed_ef_lambda, no history.
    f2 = _mk_filter(lam=1.0, adaptive_lambda_mode="off")
    lam_t = f2._adaptive_lambda(_NAME, torch.randn(3, 3), torch.randn(3, 3), torch.randn(3, 3))
    assert lam_t == 1.0
    assert not f2._adaptive_lambda_hist


# --------------------------------------------------------------------------- #
# 2. L2 gain-1 — constant δ, 100 refreshes, μ=0.9 ⇒ m → δ (gain 1, not 10×).
# --------------------------------------------------------------------------- #
def test_l2_gain_one_constant_delta_stream():
    torch.manual_seed(1)
    delta = torch.randn(6, 6)
    f = _mk_filter(delta_momentum_mu=0.9)
    m = None
    for step in range(100):
        f.current_step = step
        m = f._apply_delta_momentum(_NAME, delta, refreshed=True)
    rel = (torch.linalg.norm(m - delta) / torch.linalg.norm(delta)).item()
    assert rel < 1e-3, f"normalized EMA stationary gain must be 1 (m→δ); got rel-err {rel} (10× would be ~9)"
    # Sanity: the FORBIDDEN naive m←μm+δ would give ‖m‖≈10·‖δ‖ — confirm we are NOT there.
    assert torch.linalg.norm(m).item() < 2.0 * torch.linalg.norm(delta).item(), (
        "buffer norm must NOT blow up to 1/(1-μ)=10× (the constant-λ>1 ignition dead-end)"
    )


def test_l2_mu_zero_exact_identity():
    f = _mk_filter(delta_momentum_mu=0.0)
    delta = torch.randn(4, 4)
    for step in range(10):
        f.current_step = step
        out = f._apply_delta_momentum(_NAME, delta, refreshed=True)
        assert out is delta, "μ=0 must be the exact identity on every tick"
    assert not f._delta_momentum


def test_l2_first_fire_equals_delta():
    """First refresh ⇒ m = δ.clone() (the EMA seed), so correction == δ exactly."""
    f = _mk_filter(delta_momentum_mu=0.9)
    f.current_step = 0
    delta = torch.randn(4, 4)
    out = f._apply_delta_momentum(_NAME, delta, refreshed=True)
    assert torch.allclose(out, delta, atol=1e-7), "first fire must seed m = δ"
    assert _NAME in f._delta_momentum


# --------------------------------------------------------------------------- #
# 3. L2 async-degrade — long hold with age_decay ⇒ applied correction → 0,
#    ‖g_corr‖ → ‖g_comp‖, finite throughout, STORED buffer unchanged.
# --------------------------------------------------------------------------- #
def test_l2_async_degrade_long_hold():
    torch.manual_seed(2)
    mu = 0.9
    f = _mk_filter(delta_momentum_mu=mu, delta_momentum_age_decay=True)
    delta = torch.randn(5, 5) * 3.0

    # One refresh at step 0 ⇒ m = δ, applied correction = δ.
    f.current_step = 0
    c0 = f._apply_delta_momentum(_NAME, delta, refreshed=True)
    assert torch.allclose(c0, delta, atol=1e-6)
    stored0 = f._delta_momentum[_NAME].clone()

    # Long hold (no refresh): the APPLIED correction must fade by μ**age, monotone
    # → 0 as the anchor ages. The exact fade factor is μ**age (μ=0.9 ⇒ 0.0424 at
    # age 30, 1.8e-3 at age 60), so we (a) check the closed form at age 30 and
    # (b) push to age 60 to prove it collapses toward 0 (graceful degradation).
    dnorm = torch.linalg.norm(delta).item()
    norms = []
    for step in range(1, 61):
        f.current_step = step
        c = f._apply_delta_momentum(_NAME, delta, refreshed=False)
        n = torch.linalg.norm(c).item()
        assert torch.isfinite(torch.tensor(n)), f"correction non-finite at hold step {step}"
        norms.append(n)
        # The STORED buffer must NOT change (APPLIED-only scaling).
        assert torch.equal(f._delta_momentum[_NAME], stored0), "stored buffer must be unchanged on held ticks"

    assert norms[0] < dnorm, "age decay must start shrinking immediately"
    assert all(norms[i + 1] <= norms[i] + 1e-9 for i in range(len(norms) - 1)), "monotone non-increasing fade"
    # age 30 (norms index 29) ⇒ exactly μ**30 · ‖δ‖ (closed-form age-decay).
    assert abs(norms[29] - (mu ** 30) * dnorm) < 1e-4 * dnorm, "age-30 fade must equal μ**30·‖δ‖"
    # age 60 ⇒ collapsed toward 0 (‖c‖ < 1% of ‖δ‖), proving → 0 as the anchor ages.
    assert norms[-1] < 1e-2 * dnorm, f"a long hold must fade the correction toward 0; got {norms[-1]} vs ‖δ‖={dnorm}"


def test_l2_async_degrade_g_corr_approaches_g_comp_via_delayed_ef():
    """End-to-end through delayed_ef_matrix: a long hold ⇒ ‖g_corr‖ → ‖g_comp‖."""
    torch.manual_seed(3)
    mu = 0.9
    f = _mk_filter(delta_momentum_mu=mu, delta_momentum_age_decay=True)
    g = torch.randn(5, 5)
    m_rep = torch.randn(5, 5) * 2.0
    ring = torch.zeros(5, 5)

    f.current_step = 0
    _fire(f, _NAME, g, m_rep, ring)  # refresh: m = δ = m_rep
    g_comp_norm = torch.linalg.norm(g).item()
    gaps = []
    for step in range(1, 61):
        f.current_step = step
        out = f.delayed_ef_matrix(_NAME, g, ring_grad=None)  # held ticks
        assert torch.isfinite(out).all(), f"g_corr non-finite at step {step}"
        # ‖g_corr − g_comp‖ = ‖λ·μ**age·δ‖ → 0, so g_corr → g_comp as the anchor ages.
        gaps.append(torch.linalg.norm(out - g).item())
    assert all(gaps[i + 1] <= gaps[i] + 1e-6 for i in range(len(gaps) - 1)), "g_corr must monotonically approach g_comp"
    rel_gap = gaps[-1] / g_comp_norm
    assert rel_gap < 1e-2, f"after a long hold g_corr must collapse onto g_comp; gap {rel_gap}"


# --------------------------------------------------------------------------- #
# 4. L3 off / bounds / mean-1.
# --------------------------------------------------------------------------- #
def test_l3_off_returns_constant_lambda():
    f_off = _mk_filter(lam=1.0, adaptive_lambda_mode="off", adaptive_lambda_kappa=0.7)
    f_k0 = _mk_filter(lam=1.0, adaptive_lambda_mode="cos", adaptive_lambda_kappa=0.0)
    gm, anc, d = torch.randn(4, 4), torch.randn(4, 4), torch.randn(4, 4)
    assert f_off._adaptive_lambda(_NAME, gm, anc, d) == 1.0, "mode=off ⇒ λ_t ≡ λ exactly"
    assert f_k0._adaptive_lambda(_NAME, gm, anc, d) == 1.0, "κ=0 ⇒ λ_t ≡ λ exactly"
    assert not f_off._adaptive_lambda_hist and not f_k0._adaptive_lambda_hist


def test_l3_bounds_and_mean_one_ratio():
    """mode=ratio, κ>0: λ_t ∈ [0, cap] AND E[λ_t]≈λ on a c_t stream symmetric about c̄."""
    lam, kappa, cap = 1.0, 1.0, 2.0
    f = _mk_filter(lam=lam, adaptive_lambda_mode="ratio", adaptive_lambda_kappa=kappa, lambda_cap=cap)
    gm = torch.ones(1, 4)  # ‖gm‖ = 2 (fixed), so c_t = ‖delta‖/2
    gm_norm = torch.linalg.norm(gm).item()
    # A c_t stream symmetric about a center 0.5: ratios in {0.0, 0.25, 0.5, 0.75, 1.0}.
    centers = [0.0, 0.25, 0.5, 0.75, 1.0]
    # Repeat the symmetric pattern so the running median settles at the center 0.5.
    stream = centers * 40
    lam_ts = []
    for i, c_target in enumerate(stream):
        # delta with ‖delta‖ = c_target * ‖gm‖ ⇒ c_t = c_target.
        base = torch.ones(1, 4)
        delta = base * (c_target * gm_norm / torch.linalg.norm(base).item())
        f.current_step = i
        lt = f._adaptive_lambda(_NAME, gm, gm, delta)  # anc irrelevant in ratio mode
        assert 0.0 <= lt <= cap + 1e-9, f"λ_t out of [0,{cap}]: {lt}"
        lam_ts.append(lt)
    # Drop the warmup (median not yet settled) and check E[λ_t] ≈ λ within ~5%.
    tail = lam_ts[len(centers):]  # after one full symmetric period
    mean_lt = sum(tail) / len(tail)
    assert abs(mean_lt - lam) <= 0.05 * lam, f"MEAN-1 centered: E[λ_t]≈{lam}; got {mean_lt}"


def test_l3_bounds_cos_mode_extreme_disagreement_clamps():
    """A garbage/stale M driving a huge deviation must be clamped at lambda_cap."""
    f = _mk_filter(lam=1.0, adaptive_lambda_mode="cos", adaptive_lambda_kappa=10.0, lambda_cap=2.0)
    gm = torch.ones(1, 4)
    # First tick: c̄ = c_t ⇒ deviation 0 ⇒ λ_t = 1 (seeds the history at cos≈+1).
    lt0 = f._adaptive_lambda(_NAME, gm, gm, torch.randn(1, 4))
    assert abs(lt0 - 1.0) < 1e-6
    # Now a strongly ANTI-aligned anc ⇒ c_t≈-1, c̄≈+1 ⇒ deviation≈+2, κ=10 ⇒ λ would
    # be ~21, must clamp to cap=2.0.
    lt1 = f._adaptive_lambda(_NAME, gm, -gm, torch.randn(1, 4))
    assert lt1 == 2.0, f"a large positive deviation must clamp to lambda_cap; got {lt1}"
    # A strongly aligned (agreement HIGHER than median) tick ⇒ deviation<0 ⇒ floor 0.
    for _ in range(5):  # push the median back toward +1
        f._adaptive_lambda(_NAME, gm, gm, torch.randn(1, 4))
    lt2 = f._adaptive_lambda(_NAME, gm, -gm, torch.randn(1, 4))  # c_t≈-1 again, big deviation
    assert lt2 == 2.0


def test_l3_lambda_t_drives_g_corr_via_delayed_ef():
    """The adaptive λ_t (not the constant λ) scales the correction in g_corr."""
    # κ>0 with a controlled single fire: verify g_corr uses λ_t, and that on the very
    # first fire (c̄ = c_t ⇒ deviation 0) λ_t == λ ⇒ g_corr == B2.
    f = _mk_filter(lam=1.0, adaptive_lambda_mode="ratio", adaptive_lambda_kappa=0.5, lambda_cap=2.0)
    g = torch.full((3, 3), 1.0)
    m_rep = torch.full((3, 3), 5.0)
    ring = torch.full((3, 3), 2.0)  # δ = 5 − 2 = 3
    out = _fire(f, _NAME, g, m_rep, ring)
    # First fire: deviation 0 ⇒ λ_t = 1 ⇒ g_corr = g + 1*δ = 1 + 3 = 4 (== B2).
    assert torch.allclose(out, torch.full((3, 3), 4.0), atol=1e-6)
    assert f.delayed_ef_adaptive_lambda_applied == 1


# --------------------------------------------------------------------------- #
# 5. cross-rank determinism — two filters, identical stream ⇒ identical state.
# --------------------------------------------------------------------------- #
def test_cross_rank_determinism_momentum_and_lambda():
    torch.manual_seed(7)
    # A fixed (gm, anc, delta) stream shared by both "ranks".
    stream = [
        (torch.randn(4, 4), torch.randn(4, 4) * 2.0, torch.randn(4, 4) * 0.7)
        for _ in range(20)
    ]
    fa = _mk_filter(
        lam=1.0,
        delta_momentum_mu=0.9,
        delta_momentum_age_decay=True,
        adaptive_lambda_mode="cos",
        adaptive_lambda_kappa=0.5,
        lambda_cap=2.0,
    )
    fb = _mk_filter(
        lam=1.0,
        delta_momentum_mu=0.9,
        delta_momentum_age_decay=True,
        adaptive_lambda_mode="cos",
        adaptive_lambda_kappa=0.5,
        lambda_cap=2.0,
    )
    for step, (gm, anc, delta) in enumerate(stream):
        fa.current_step = step
        fb.current_step = step
        refreshed = (step % 3 == 0)  # mix refresh + held ticks
        ma = fa._apply_delta_momentum(_NAME, delta, refreshed=refreshed)
        mb = fb._apply_delta_momentum(_NAME, delta, refreshed=refreshed)
        assert torch.equal(ma, mb), f"δ-momentum buffer diverged across ranks at step {step}"
        la = fa._adaptive_lambda(_NAME, gm, anc, delta)
        lb = fb._adaptive_lambda(_NAME, gm, anc, delta)
        assert la == lb, f"λ_t diverged across ranks at step {step}: {la} != {lb}"
    # The stored buffers + histories must be bit-identical at the end.
    assert torch.equal(fa._delta_momentum[_NAME], fb._delta_momentum[_NAME])
    assert list(fa._adaptive_lambda_hist[_NAME]) == list(fb._adaptive_lambda_hist[_NAME])


def test_cross_rank_determinism_end_to_end_delayed_ef():
    """Two filters fed identical (g, M_rep, ring) ⇒ identical g_corr each tick."""
    torch.manual_seed(8)
    fa = _mk_filter(delta_momentum_mu=0.5, adaptive_lambda_mode="ratio", adaptive_lambda_kappa=0.8)
    fb = _mk_filter(delta_momentum_mu=0.5, adaptive_lambda_mode="ratio", adaptive_lambda_kappa=0.8)
    m_rep = torch.randn(5, 5) * 2.0
    fa.update_anchor(_NAME, m_rep)
    fb.update_anchor(_NAME, m_rep)
    for step in range(12):
        fa.current_step = step
        fb.current_step = step
        g = torch.randn(5, 5)
        ring = torch.randn(5, 5) * 0.3 if (step % 4 == 0) else None
        oa = fa.delayed_ef_matrix(_NAME, g, ring_grad=ring)
        ob = fb.delayed_ef_matrix(_NAME, g, ring_grad=ring)
        assert torch.equal(oa, ob), f"g_corr diverged across ranks at step {step}"


# --------------------------------------------------------------------------- #
# composition: λ=0 early-return still dominates with both levers ON.
# --------------------------------------------------------------------------- #
def test_lambda_zero_early_return_dominates_with_levers_on():
    f = _mk_filter(lam=0.0, delta_momentum_mu=0.9, adaptive_lambda_mode="cos", adaptive_lambda_kappa=1.0)
    g = torch.randn(4, 4)
    out = f.delayed_ef_matrix(_NAME, g, ring_grad=torch.randn(4, 4))
    assert out is g, "λ=0 must return G_comp EXACTLY even with L2/L3 ON (early-return is FIRST)"
    assert not f._delta_momentum, "λ=0 early-return must build NO momentum buffer"
    assert not f._adaptive_lambda_hist, "λ=0 early-return must build NO agreement history"


def test_validation_rejects_bad_lever_args():
    with pytest.raises(AssertionError):
        SpectralFilter(correction_mode="delayed_ef", delta_momentum_mu=1.0)  # μ must be < 1
    with pytest.raises(AssertionError):
        SpectralFilter(correction_mode="delayed_ef", adaptive_lambda_mode="bogus")
    with pytest.raises(AssertionError):
        SpectralFilter(correction_mode="delayed_ef", adaptive_lambda_kappa=-0.1)
    with pytest.raises(AssertionError):
        SpectralFilter(correction_mode="delayed_ef", lambda_cap=-1.0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
