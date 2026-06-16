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

"""Additive sub-basis: additive stale-anchor rank-r_sb sub-basis merger — CPU tests.

Correctness invariants
covered here (all CPU, no GPU, no torch.distributed):

* **off-path parity (hard):** ``delta_subbasis_rank=0`` SKIPS the sub-basis branch
  entirely ⇒ ``delayed_ef_matrix`` returns the EXACT delayed_ef ``g_corr`` (byte-compare
  to a separate rank-0 filter on the same inputs) AND the new-knob-OFF default is
  bitwise-identical to the delayed_ef filter that has no sub-basis args at all.
* **limiting-case identity (hard):** ``λ=0`` still returns the ``g_comp`` object
  identity even with the sub-basis ON (the λ=0 early-return is FIRST).
* **sub-basis math:** ``_subbasis_delta(S, r=full)`` reconstructs S to < 1e-5;
  rank-2 on a known rank-2 S is exact; the output is detached / fp32; a degenerate
  source (zero / non-finite / r > min-dim) returns None (the merger folds in the
  plain δ unchanged and counts a SKIP).
* **determinism / multi-rank agreement (hard):** two filters with the same
  ``base_seed`` produce bit-identical δ_subbasis for the same (name, source);
  feeding the SAME DP-mean source on two independent "ranks" yields a
  bit-identical correction (the cross-rank-agreement invariant on a synthetic δ).
* **scale contract (the mean-vs-sum trap):** feeding a SUM-reduced source
  (×world_size) inflates ‖δ_subbasis‖ by world_size (the SVD applies no rescaling)
  — mirrors the delayed-EF scale tests.
* **Q-basis avoidance (structural):** the merger reads/writes only the correction
  δ; it never touches a forward basis. Asserted here by the absence of any forward
  -Q state on the filter and the fact that rank-0 is byte-identical to delayed_ef (the
  forward path is provably untouched). The full forward-Q-checksum gate is the
  on-box probe (needs a GPU).

Perturbation lever (``perturb_sigma`` — zero-mean tunable cross-rank-identical
gradient perturbation) Correctness invariants covered here:

* **off-path parity (hard):** ``perturb_sigma=0`` SKIPS the perturbation branch ⇒
  ``delayed_ef_matrix`` returns the EXACT un-perturbed g_corr (``torch.equal``);
  composed with ``delta_subbasis_rank=0`` it is bitwise delayed_ef.
* **perturbation magnitude:** with σ>0, ``‖g_corr_perturbed − g_corr‖ ≈ σ·‖g_corr‖``
  (ξ is unit-normalized so the injected norm is exactly σ·‖g_corr‖).
* **determinism / cross-rank identity (hard):** two filters with the same
  (perturb_seed, current_step, name) produce the SAME perturbed g_corr (the DP
  consistency guarantee — else ranks descend in different directions and diverge).
* **zero-mean over steps:** averaging the perturbation across many ``current_step``
  values ⇒ ‖mean‖ ≪ σ·‖g‖ (it is zero-mean noise, not a bias).
* **guards:** a non-finite / ~zero-norm g_corr ⇒ the perturbation is a no-op
  (returned unchanged), never a silent garbage gradient.
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
    rank=0,
    family="tail",
    lam=1.0,
    base_seed=0,
    weight=1.0,
    decay_steps=0,
    hold_steps=0,
    perturb_sigma=0.0,
    perturb_seed=0,
):
    return SpectralFilter(
        beta_anc=0.0,
        correction_mode="delayed_ef",
        delayed_ef_lambda=lam,
        ema_device="cpu",
        delta_subbasis_rank=rank,
        delta_subbasis_family=family,
        delta_subbasis_weight=weight,
        delta_subbasis_decay_steps=decay_steps,
        delta_subbasis_hold_steps=hold_steps,
        base_seed=base_seed,
        perturb_sigma=perturb_sigma,
        perturb_seed=perturb_seed,
    )


def _warm_and_fire(f, name, m_rep, ring, g):
    """Warm M_rep (β=0 ⇒ M == m_rep) and apply one fire-aligned tick."""
    f.update_anchor(name, m_rep)
    return f.delayed_ef_matrix(name, g, ring_grad=ring)


# --------------------------------------------------------------------------- #
# off-path parity — rank=0 SKIPS the sub-basis branch ⇒ bitwise delayed_ef.
# --------------------------------------------------------------------------- #
def test_rank0_is_bitwise_delayed_ef():
    torch.manual_seed(0)
    m_rep = torch.randn(6, 5)
    ring = torch.randn(6, 5)
    g = torch.randn(6, 5)

    f_off = _mk_filter(rank=0)
    f_delayed_ef = SpectralFilter(  # the delayed_ef filter — no sub-basis kwargs at all
        beta_anc=0.0, correction_mode="delayed_ef", delayed_ef_lambda=1.0, ema_device="cpu"
    )
    out_off = _warm_and_fire(f_off, _NAME, m_rep.clone(), ring.clone(), g.clone())
    out_delayed_ef = _warm_and_fire(f_delayed_ef, _NAME, m_rep.clone(), ring.clone(), g.clone())

    assert torch.equal(out_off, out_delayed_ef), (
        "delta_subbasis_rank=0 must be BITWISE identical to the delayed_ef merger"
    )
    # The sub-basis branch was skipped entirely: no apply, no energy ratio.
    assert f_off.delayed_ef_subbasis_applied == 0
    assert f_off.delayed_ef_subbasis_skipped == 0
    assert f_off._subbasis_energy_ratios == []


def test_rank0_default_kwargs_match_explicit_delayed_ef():
    """The new knobs DEFAULT off — a filter that never names them == delayed_ef."""
    torch.manual_seed(1)
    m_rep, ring, g = torch.randn(4, 4), torch.randn(4, 4), torch.randn(4, 4)
    # Default construction (no sub-basis args) — delta_subbasis_rank defaults 0.
    f_default = SpectralFilter(beta_anc=0.0, correction_mode="delayed_ef", delayed_ef_lambda=1.0, ema_device="cpu")
    assert f_default.delta_subbasis_rank == 0
    assert f_default.delta_subbasis_family == "tail"
    out = _warm_and_fire(f_default, _NAME, m_rep, ring, g)
    expected = g.to(torch.float32) + 1.0 * (m_rep.to(torch.float32) - ring.to(torch.float32))
    assert torch.allclose(out.to(torch.float32), expected, atol=1e-6)


# --------------------------------------------------------------------------- #
# limiting-case identity — λ=0 returns g_comp object even with sub-basis ON.
# --------------------------------------------------------------------------- #
def test_lambda_zero_identity_with_subbasis_on():
    f = _mk_filter(rank=2, family="tail", lam=0.0)
    g = torch.randn(5, 5)
    out = f.delayed_ef_matrix(_NAME, g, ring_grad=torch.randn(5, 5))
    assert out is g, "λ=0 must return G_comp EXACTLY (same object) even with the sub-basis ON"
    assert f.delayed_ef_subbasis_applied == 0, "the λ=0 early-return precedes the sub-basis branch"


# --------------------------------------------------------------------------- #
# sub-basis math — reconstruction fidelity, rank-2-exact, detached, degenerate.
# --------------------------------------------------------------------------- #
def test_subbasis_full_rank_reconstructs_source():
    torch.manual_seed(2)
    f = _mk_filter(rank=0)  # rank irrelevant; we call the helper directly
    S = torch.randn(8, 6)  # full rank = min(8,6) = 6
    recon = f._subbasis_delta(_NAME, S, r=6)
    assert recon is not None
    rel = (torch.linalg.norm(recon - S) / torch.linalg.norm(S)).item()
    assert rel < 1e-5, f"full-rank sub-basis must reconstruct S to <1e-5; got {rel:.2e}"


def test_subbasis_rank2_exact_on_rank2_source():
    torch.manual_seed(3)
    # Build an EXACT rank-2 matrix: U(10x2) @ V(2x7).
    U = torch.randn(10, 2)
    Vt = torch.randn(2, 7)
    S = U @ Vt  # rank <= 2
    f = _mk_filter(rank=0)
    recon = f._subbasis_delta(_NAME, S, r=2)
    assert recon is not None
    rel = (torch.linalg.norm(recon - S) / torch.linalg.norm(S)).item()
    assert rel < 1e-5, f"rank-2 sub-basis must be EXACT on a rank-2 source; got {rel:.2e}"


def test_subbasis_output_detached_fp32():
    f = _mk_filter(rank=0)
    S = torch.randn(5, 5, dtype=torch.float64, requires_grad=True)
    recon = f._subbasis_delta(_NAME, S, r=2)
    assert recon is not None
    assert not recon.requires_grad, "δ_subbasis must be detached (no autograd history)"
    assert recon.dtype == torch.float32, "δ_subbasis must be fp32"
    assert recon.shape == S.shape


def test_subbasis_degenerate_source_returns_none():
    f = _mk_filter(rank=0)
    # zero source
    assert f._subbasis_delta(_NAME, torch.zeros(4, 4), r=2) is None
    # non-finite source
    bad = torch.full((4, 4), float("nan"))
    assert f._subbasis_delta(_NAME, bad, r=2) is None
    # r > min-dim is clamped (still returns a valid recon, not None) — verify it
    # does not raise and reconstructs within the available rank.
    S = torch.randn(3, 5)
    recon = f._subbasis_delta(_NAME, S, r=99)  # clamped to q=3
    assert recon is not None and recon.shape == S.shape
    # non-2D
    assert f._subbasis_delta(_NAME, torch.randn(4), r=2) is None
    # r<=0
    assert f._subbasis_delta(_NAME, torch.randn(4, 4), r=0) is None


def test_subbasis_degenerate_source_in_merger_skips_and_keeps_delayed_ef():
    """A degenerate δ source folds in the plain delayed_ef δ (never garbage) + counts SKIP.

    Build a tick where δ_delayed_ef = M_rep − ring is exactly ZERO (M_rep == ring): the
    tail source is the zero matrix ⇒ the sub-basis SKIPS, and G_corr == G_comp + δ
    = G_comp (since δ=0) — i.e. the plain delayed_ef result, with a SKIP counted.
    """
    f = _mk_filter(rank=2, family="tail")
    same = torch.full((4, 4), 2.0)
    g = torch.randn(4, 4)
    out = _warm_and_fire(f, _NAME, same.clone(), same.clone(), g.clone())
    assert f.delayed_ef_subbasis_skipped == 1
    assert f.delayed_ef_subbasis_applied == 0
    assert torch.allclose(out.to(torch.float32), g.to(torch.float32), atol=1e-6), (
        "with δ_delayed_ef=0 and a degenerate tail source, G_corr must equal G_comp (plain delayed_ef)"
    )


def test_subbasis_applied_adds_energy_and_counts():
    """A non-degenerate tail source ⇒ G_corr = G_comp + δ + δ_subbasis, applied counted."""
    torch.manual_seed(4)
    f = _mk_filter(rank=2, family="tail")
    m_rep = torch.randn(8, 6)
    ring = torch.randn(8, 6)
    g = torch.randn(8, 6)
    out = _warm_and_fire(f, _NAME, m_rep.clone(), ring.clone(), g.clone())

    delta = (m_rep - ring).to(torch.float32)
    delta_sb = f._subbasis_delta(_NAME, delta, r=2)
    expected = g.to(torch.float32) + 1.0 * (delta + delta_sb)
    assert torch.allclose(out.to(torch.float32), expected, atol=1e-5), (
        "applied sub-basis must give G_corr = G_comp + λ(δ + δ_subbasis)"
    )
    assert f.delayed_ef_subbasis_applied == 1
    assert f.delayed_ef_subbasis_skipped == 0
    assert len(f._subbasis_energy_ratios) == 1
    # The recorded ratio is ‖δ_subbasis‖/‖δ‖.
    exp_ratio = (torch.linalg.norm(delta_sb) / torch.linalg.norm(delta)).item()
    assert abs(f._subbasis_energy_ratios[0] - exp_ratio) < 1e-5


def test_subbasis_grad_family_uses_m_rep_not_delta():
    """family='grad' ⇒ the source is M_rep (raw anchor), NOT the deflated δ."""
    torch.manual_seed(5)
    m_rep = torch.randn(8, 6)
    ring = torch.randn(8, 6)
    g = torch.randn(8, 6)
    f = _mk_filter(rank=2, family="grad")
    out = _warm_and_fire(f, _NAME, m_rep.clone(), ring.clone(), g.clone())

    delta = (m_rep - ring).to(torch.float32)
    delta_sb_from_m = f._subbasis_delta(_NAME, m_rep.to(torch.float32), r=2)
    expected = g.to(torch.float32) + 1.0 * (delta + delta_sb_from_m)
    assert torch.allclose(out.to(torch.float32), expected, atol=1e-5), (
        "family=grad must build the sub-basis from M_rep, not from the deflated δ"
    )


# --------------------------------------------------------------------------- #
# determinism / multi-rank agreement — same seed ⇒ bit-identical δ_subbasis.
# --------------------------------------------------------------------------- #
def test_same_seed_identical_subbasis():
    torch.manual_seed(6)
    S = torch.randn(12, 9)
    f1 = _mk_filter(rank=4, base_seed=0)
    f2 = _mk_filter(rank=4, base_seed=0)
    r1 = f1._subbasis_delta(_NAME, S.clone(), r=4)
    r2 = f2._subbasis_delta(_NAME, S.clone(), r=4)
    assert torch.equal(r1, r2), "same base_seed + same source ⇒ bit-identical δ_subbasis"


def test_cross_rank_identity_on_synthetic_delta():
    """Two independent 'ranks' fed the SAME DP-mean δ produce identical corrections.

    δ_delayed_ef is DP-MEAN identical across ranks by construction; the seed is a pure
    function of (base_seed, name) with no rank-local state. So two filters built
    identically and fed the same (M_rep, ring, g) — the cross-rank stand-in —
    must yield a bit-identical G_corr (the multi-rank-agreement invariant on a
    synthetic δ).
    """
    torch.manual_seed(7)
    m_rep, ring, g = torch.randn(10, 7), torch.randn(10, 7), torch.randn(10, 7)
    rank0 = _mk_filter(rank=2, base_seed=0)
    rank1 = _mk_filter(rank=2, base_seed=0)
    out0 = _warm_and_fire(rank0, _NAME, m_rep.clone(), ring.clone(), g.clone())
    out1 = _warm_and_fire(rank1, _NAME, m_rep.clone(), ring.clone(), g.clone())
    assert torch.equal(out0, out1), "cross-rank δ_subbasis must agree bit-for-bit"


def test_seed_salt_is_per_target_and_reproducible():
    f = _mk_filter(rank=2, base_seed=3)
    s_a = f._subbasis_seed("layers.0.q_proj.weight")
    s_b = f._subbasis_seed("layers.5.down_proj.weight")
    # Different targets ⇒ different seeds (so each gets its own projection).
    assert s_a != s_b
    # FSDP infix is canonicalized away ⇒ the seed is wrap-invariant.
    s_a_wrapped = f._subbasis_seed("layers.0._fsdp_wrapped_module.q_proj.weight")
    assert s_a == s_a_wrapped, "the per-target seed must be FSDP-wrap invariant (canon)"
    # Reproducible across calls.
    assert s_a == f._subbasis_seed("layers.0.q_proj.weight")


# --------------------------------------------------------------------------- #
# scale contract (mean-vs-sum) — a SUM-reduced source inflates the norm.
# --------------------------------------------------------------------------- #
def test_scale_contract_sum_reduced_inflates_by_world_size():
    torch.manual_seed(8)
    world_size = 4
    S_mean = torch.randn(10, 8)
    S_sum = S_mean * world_size  # the SUM-reduced side (the mean-vs-sum trap)
    f = _mk_filter(rank=3)
    sb_mean = f._subbasis_delta(_NAME, S_mean, r=3)
    sb_sum = f._subbasis_delta(_NAME, S_sum, r=3)
    n_mean = torch.linalg.norm(sb_mean).item()
    n_sum = torch.linalg.norm(sb_sum).item()
    # The SVD applies NO rescaling ⇒ a ×world_size source ⇒ ×world_size δ_subbasis.
    assert abs(n_sum / (n_mean + 1e-12) - world_size) < 1e-3, (
        f"a SUM-reduced source must inflate ‖δ_subbasis‖ by world_size; got {n_sum / n_mean:.4f}"
    )


# --------------------------------------------------------------------------- #
# Sub-basis gamma knob — sub-basis WEIGHT + linear DECAY (the over-amplification fix).
# --------------------------------------------------------------------------- #
def test_gamma_defaults_reproduce_current_subbasis_bitwise():
    """weight=1.0 + decay_steps=0 ⇒ γ_t≡1 ⇒ the EXACT pre-γ sub-basis δ_subbasis path.

    A filter that names the new knobs at their DEFAULTS must be byte-identical to a
    filter that does NOT name them at all (the OLD sub-basis constructor) — the γ-knob
    is a pure extension with a no-op default.
    """
    torch.manual_seed(40)
    m_rep, ring, g = torch.randn(8, 6), torch.randn(8, 6), torch.randn(8, 6)
    # OLD-style constructor: rank=2 tail, NO weight/decay args at all.
    f_old = SpectralFilter(
        beta_anc=0.0,
        correction_mode="delayed_ef",
        delayed_ef_lambda=1.0,
        ema_device="cpu",
        delta_subbasis_rank=2,
        delta_subbasis_family="tail",
        base_seed=0,
    )
    # NEW constructor with the DEFAULT γ-knobs explicit.
    f_new = _mk_filter(rank=2, family="tail", weight=1.0, decay_steps=0, base_seed=0)
    assert f_old.delta_subbasis_weight == 1.0 and f_old.delta_subbasis_decay_steps == 0
    out_old = _warm_and_fire(f_old, _NAME, m_rep.clone(), ring.clone(), g.clone())
    out_new = _warm_and_fire(f_new, _NAME, m_rep.clone(), ring.clone(), g.clone())
    assert torch.equal(out_old, out_new), (
        "weight=1.0 + decay_steps=0 must reproduce the pre-γ sub-basis δ_subbasis path BITWISE"
    )
    assert f_new.delayed_ef_subbasis_applied == 1
    # γ_t at the default step (0) is exactly the weight (1.0).
    assert f_new._subbasis_gamma() == 1.0


def test_gamma_weight_zero_is_bitwise_delayed_ef():
    """weight=0 ⇒ γ_t==0 ⇒ the sub-basis branch is skipped ⇒ correction == δ_delayed_ef (= delayed_ef baseline)."""
    torch.manual_seed(41)
    m_rep, ring, g = torch.randn(7, 5), torch.randn(7, 5), torch.randn(7, 5)
    f_w0 = _mk_filter(rank=2, family="tail", weight=0.0, decay_steps=0)
    f_delayed_ef = SpectralFilter(  # delayed_ef filter, no sub-basis at all
        beta_anc=0.0, correction_mode="delayed_ef", delayed_ef_lambda=1.0, ema_device="cpu"
    )
    out_w0 = _warm_and_fire(f_w0, _NAME, m_rep.clone(), ring.clone(), g.clone())
    out_delayed_ef = _warm_and_fire(f_delayed_ef, _NAME, m_rep.clone(), ring.clone(), g.clone())
    assert torch.equal(out_w0, out_delayed_ef), "weight=0 must be BITWISE identical to delayed_ef"
    # γ_t==0 ⇒ the SVD was never computed: no apply, no skip, no energy ratio.
    assert f_w0.delayed_ef_subbasis_applied == 0
    assert f_w0.delayed_ef_subbasis_skipped == 0
    assert f_w0._subbasis_energy_ratios == []
    assert f_w0._subbasis_gamma() == 0.0


def test_gamma_decay_schedule_linear_clamped():
    """γ_t = weight at step 0; = 0 at step==decay_steps; linear between; clamped ≥0 past."""
    f = _mk_filter(rank=2, weight=1.0, decay_steps=50)
    # step 0 ⇒ γ == weight.
    f.current_step = 0
    assert abs(f._subbasis_gamma() - 1.0) < 1e-9
    # step == decay_steps ⇒ γ == 0.
    f.current_step = 50
    assert abs(f._subbasis_gamma() - 0.0) < 1e-9
    # halfway ⇒ γ == weight/2 (linear).
    f.current_step = 25
    assert abs(f._subbasis_gamma() - 0.5) < 1e-9
    # quarter ⇒ 0.75.
    f.current_step = 10
    assert abs(f._subbasis_gamma() - (1.0 - 10.0 / 50.0)) < 1e-9
    # past the horizon ⇒ clamped at 0 (never negative).
    f.current_step = 75
    assert f._subbasis_gamma() == 0.0
    f.current_step = 10_000
    assert f._subbasis_gamma() == 0.0


def test_gamma_decay_scales_weight():
    """With weight=0.6, the linear schedule scales THAT weight (γ = 0.6·decay_factor)."""
    f = _mk_filter(rank=2, weight=0.6, decay_steps=40)
    f.current_step = 0
    assert abs(f._subbasis_gamma() - 0.6) < 1e-9
    f.current_step = 20  # halfway ⇒ 0.6 * 0.5 = 0.3
    assert abs(f._subbasis_gamma() - 0.3) < 1e-9
    f.current_step = 40
    assert abs(f._subbasis_gamma() - 0.0) < 1e-9


def test_gamma_decay_steps_zero_is_constant_weight():
    """decay_steps=0 ⇒ decay_factor≡1 ⇒ γ_t == weight at EVERY step (no decay)."""
    f = _mk_filter(rank=2, weight=0.5, decay_steps=0)
    for s in (0, 1, 7, 49, 50, 1000):
        f.current_step = s
        assert abs(f._subbasis_gamma() - 0.5) < 1e-9, f"constant γ broke at step {s}"


def test_gamma_constant_half_dose_formula():
    """weight=0.5, decay_steps=0 ⇒ G_corr = G_comp + λ(δ_delayed_ef + 0.5·δ_subbasis)."""
    torch.manual_seed(42)
    m_rep, ring, g = torch.randn(8, 6), torch.randn(8, 6), torch.randn(8, 6)
    f = _mk_filter(rank=2, family="tail", weight=0.5, decay_steps=0)
    out = _warm_and_fire(f, _NAME, m_rep.clone(), ring.clone(), g.clone())
    delta = (m_rep - ring).to(torch.float32)
    delta_sb = f._subbasis_delta(_NAME, delta, r=2)
    expected = g.to(torch.float32) + 1.0 * (delta + 0.5 * delta_sb)
    assert torch.allclose(out.to(torch.float32), expected, atol=1e-5), (
        "constant half-dose must give G_corr = G_comp + λ(δ_delayed_ef + 0.5·δ_subbasis)"
    )
    assert f.delayed_ef_subbasis_applied == 1
    # The logged energy ratio reflects the EFFECTIVE (γ-scaled) sub-basis norm.
    exp_ratio = (0.5 * torch.linalg.norm(delta_sb) / torch.linalg.norm(delta)).item()
    assert abs(f._subbasis_energy_ratios[0] - exp_ratio) < 1e-5


def test_gamma_decay_midrun_formula_uses_current_step():
    """At step=decay_steps/2, the applied weight is weight/2 (decay reads current_step)."""
    torch.manual_seed(43)
    m_rep, ring, g = torch.randn(8, 6), torch.randn(8, 6), torch.randn(8, 6)
    f = _mk_filter(rank=2, family="tail", weight=1.0, decay_steps=50)
    f.update_anchor(_NAME, m_rep.clone())
    f.current_step = 25  # γ_t = 1.0 * (1 - 25/50) = 0.5
    out = f.delayed_ef_matrix(_NAME, g.clone(), ring_grad=ring.clone())
    delta = (m_rep - ring).to(torch.float32)
    delta_sb = f._subbasis_delta(_NAME, delta, r=2)
    expected = g.to(torch.float32) + 1.0 * (delta + 0.5 * delta_sb)
    assert torch.allclose(out.to(torch.float32), expected, atol=1e-5), (
        "mid-run γ-decay must apply weight*(1 - step/decay_steps) to δ_subbasis"
    )


def test_gamma_decay_past_horizon_is_delayed_ef():
    """Past decay_steps the sub-basis vanishes ⇒ G_corr == delayed_ef (γ_t clamped to 0)."""
    torch.manual_seed(44)
    m_rep, ring, g = torch.randn(7, 5), torch.randn(7, 5), torch.randn(7, 5)
    f = _mk_filter(rank=2, family="tail", weight=1.0, decay_steps=10)
    f.update_anchor(_NAME, m_rep.clone())
    f.current_step = 30  # well past the horizon ⇒ γ_t = 0
    out = f.delayed_ef_matrix(_NAME, g.clone(), ring_grad=ring.clone())
    expected = g.to(torch.float32) + 1.0 * (m_rep.to(torch.float32) - ring.to(torch.float32))
    assert torch.allclose(out.to(torch.float32), expected, atol=1e-6), (
        "past the decay horizon the correction must reduce to δ_delayed_ef (= delayed_ef baseline)"
    )
    # γ_t==0 ⇒ skipped branch (no SVD): no apply / skip / ratio.
    assert f.delayed_ef_subbasis_applied == 0
    assert f.delayed_ef_subbasis_skipped == 0


def test_gamma_negative_weight_rejected():
    """delta_subbasis_weight < 0 is asserted in the filter ctor (mirrors config validation)."""
    with pytest.raises(AssertionError):
        _mk_filter(rank=2, weight=-0.1)


def test_gamma_negative_decay_steps_rejected():
    with pytest.raises(AssertionError):
        _mk_filter(rank=2, decay_steps=-5)


# --------------------------------------------------------------------------- #
# anchor-usage hold-then-decay — γ holds at full weight for hold_steps, THEN decays.
# --------------------------------------------------------------------------- #
def test_hold_steps_zero_reproduces_linear_from_zero_decay_bitwise():
    """hold_steps=0 ⇒ the EXISTING linear-from-step-0 decay, gamma BITWISE-identical.

    The load-bearing regression guard: introducing the HOLD shelf must NOT perturb
    the legacy schedule. A filter with hold_steps=0 (explicit) AND a filter that
    NEVER names hold_steps must produce the EXACT same γ_t as the legacy
    ``max(0, 1 - step/decay_steps)`` formula at EVERY step — float-equal, not
    approx. (Bitwise on the schedule is what makes the prior γ-decay50 run
    reproducible under the new code.)
    """
    f_explicit = _mk_filter(rank=2, weight=1.0, decay_steps=50, hold_steps=0)
    # A filter that NEVER names hold_steps (the default must be 0).
    f_unspecified = SpectralFilter(
        beta_anc=0.0,
        correction_mode="delayed_ef",
        delayed_ef_lambda=1.0,
        ema_device="cpu",
        delta_subbasis_rank=2,
        delta_subbasis_family="tail",
        delta_subbasis_weight=1.0,
        delta_subbasis_decay_steps=50,
        base_seed=0,
    )
    assert f_unspecified.delta_subbasis_hold_steps == 0, "hold_steps must DEFAULT to 0"
    # Sweep the whole horizon + past it. The legacy formula is the reference.
    for s in (0, 1, 7, 24, 25, 26, 37, 49, 50, 51, 75, 10_000):
        # legacy reference: weight * max(0, 1 - s/decay_steps)
        legacy = 1.0 * max(0.0, 1.0 - (float(s) / 50.0))
        f_explicit.current_step = s
        f_unspecified.current_step = s
        g_explicit = f_explicit._subbasis_gamma()
        g_unspecified = f_unspecified._subbasis_gamma()
        # Bitwise (float ==), not approx — the formula must collapse exactly.
        assert g_explicit == legacy, f"hold_steps=0 broke legacy decay at step {s}: {g_explicit} != {legacy}"
        assert g_unspecified == legacy, f"default hold_steps broke legacy decay at step {s}"
        assert g_explicit == g_unspecified


def test_hold25_decay25_schedule():
    """hold_steps=25, decay_steps=25 ⇒ γ=weight for steps 0–24, then linear→0 over 25–50.

    Per the task spec: γ_factor=1.0 at steps 0,10,24; =1.0 at the step-25 boundary
    then ramps; =0.5 at step ~37-38; =0.0 at step 50; clamped ≥0 past 50.
    """
    f = _mk_filter(rank=2, weight=1.0, decay_steps=25, hold_steps=25)
    # The HOLD shelf: γ == weight (1.0) for every step strictly < 25.
    for s in (0, 10, 24):
        f.current_step = s
        assert f._subbasis_gamma() == 1.0, f"HOLD shelf broke at step {s}"
    # Step 25 is the boundary: s < h is False (25 < 25 is False) ⇒ ramp begins;
    # decay_factor = 1 - (25-25)/25 = 1.0 exactly (the ramp STARTS at full weight).
    f.current_step = 25
    assert abs(f._subbasis_gamma() - 1.0) < 1e-12, "ramp must START at full weight at the hold boundary"
    # One step into the ramp: γ = 1 - (26-25)/25 = 0.96.
    f.current_step = 26
    assert abs(f._subbasis_gamma() - (1.0 - 1.0 / 25.0)) < 1e-9
    # Halfway through the ramp (~step 37-38): γ ≈ 0.5.
    f.current_step = 37  # 1 - (37-25)/25 = 1 - 12/25 = 0.52
    assert abs(f._subbasis_gamma() - 0.52) < 1e-9
    f.current_step = 38  # 1 - 13/25 = 0.48 — straddles 0.5 between steps 37 and 38
    assert abs(f._subbasis_gamma() - 0.48) < 1e-9
    # The midpoint step (s=37.5 → take s=37 just above 0.5, s=38 just below): both bracket 0.5.
    # End of the ramp: γ == 0 at step 50 (h + d = 25 + 25).
    f.current_step = 50
    assert abs(f._subbasis_gamma() - 0.0) < 1e-12, "γ must reach 0 at step h+d=50"
    # Clamped ≥ 0 past the horizon.
    f.current_step = 51
    assert f._subbasis_gamma() == 0.0
    f.current_step = 10_000
    assert f._subbasis_gamma() == 0.0


def test_hold25_decay25_applied_in_merger_during_hold_and_ramp():
    """The hold-then-decay γ is the ACTUAL scalar the merger applies to δ_subbasis.

    During the HOLD (step < 25) the merger applies the full δ_subbasis (γ=1); during
    the ramp (step 37) it applies γ=0.52·δ_subbasis — verified against the merger's
    own output, not just _subbasis_gamma.
    """
    torch.manual_seed(45)
    m_rep, ring, g = torch.randn(8, 6), torch.randn(8, 6), torch.randn(8, 6)

    # In the HOLD window (step 10): G_corr = G_comp + λ(δ_delayed_ef + 1.0·δ_subbasis).
    f_hold = _mk_filter(rank=2, family="tail", weight=1.0, decay_steps=25, hold_steps=25)
    f_hold.update_anchor(_NAME, m_rep.clone())
    f_hold.current_step = 10
    out_hold = f_hold.delayed_ef_matrix(_NAME, g.clone(), ring_grad=ring.clone())
    delta = (m_rep - ring).to(torch.float32)
    delta_sb = f_hold._subbasis_delta(_NAME, delta, r=2)
    exp_hold = g.to(torch.float32) + 1.0 * (delta + 1.0 * delta_sb)
    assert torch.allclose(out_hold.to(torch.float32), exp_hold, atol=1e-5), (
        "during the HOLD shelf the merger must apply the FULL δ_subbasis (γ=1)"
    )

    # In the RAMP (step 37): γ = 0.52.
    f_ramp = _mk_filter(rank=2, family="tail", weight=1.0, decay_steps=25, hold_steps=25)
    f_ramp.update_anchor(_NAME, m_rep.clone())
    f_ramp.current_step = 37
    out_ramp = f_ramp.delayed_ef_matrix(_NAME, g.clone(), ring_grad=ring.clone())
    exp_ramp = g.to(torch.float32) + 1.0 * (delta + 0.52 * delta_sb)
    assert torch.allclose(out_ramp.to(torch.float32), exp_ramp, atol=1e-5), (
        "during the ramp the merger must apply γ=weight·(1-(step-hold)/decay)·δ_subbasis"
    )
    # The logged energy ratio reflects the EFFECTIVE (γ-scaled) sub-basis norm.
    exp_ratio = (0.52 * torch.linalg.norm(delta_sb) / torch.linalg.norm(delta)).item()
    assert abs(f_ramp._subbasis_energy_ratios[0] - exp_ratio) < 1e-5


def test_hold_then_decay_with_decay_steps_zero_is_constant():
    """hold_steps is INERT when decay_steps=0 (constant γ — the d<=0 branch wins first)."""
    f = _mk_filter(rank=2, weight=0.7, decay_steps=0, hold_steps=25)
    for s in (0, 10, 24, 25, 26, 100):
        f.current_step = s
        assert abs(f._subbasis_gamma() - 0.7) < 1e-12, (
            f"decay_steps=0 ⇒ constant weight regardless of hold_steps; broke at step {s}"
        )


def test_hold_steps_scales_with_weight():
    """The hold shelf holds at WEIGHT (not 1.0) when weight != 1.0; ramp scales it too."""
    f = _mk_filter(rank=2, weight=0.6, decay_steps=25, hold_steps=25)
    f.current_step = 10  # HOLD ⇒ γ == weight == 0.6
    assert abs(f._subbasis_gamma() - 0.6) < 1e-12
    f.current_step = 25  # ramp start ⇒ 0.6 * 1.0 == 0.6
    assert abs(f._subbasis_gamma() - 0.6) < 1e-9
    f.current_step = 37  # 0.6 * (1 - 12/25) = 0.6 * 0.52 = 0.312
    assert abs(f._subbasis_gamma() - 0.6 * 0.52) < 1e-9
    f.current_step = 50  # 0.6 * 0 == 0
    assert abs(f._subbasis_gamma() - 0.0) < 1e-12


def test_gamma_negative_hold_steps_rejected():
    with pytest.raises(AssertionError):
        _mk_filter(rank=2, hold_steps=-3)


# --------------------------------------------------------------------------- #
# Perturbation lever — zero-mean tunable cross-rank-identical perturbation.
# --------------------------------------------------------------------------- #
def _unperturbed_g_corr(name, m_rep, ring, g, *, rank=0, family="tail", base_seed=0):
    """The g_corr a σ=0 filter (same delayed_ef config) produces — the reference."""
    f = _mk_filter(rank=rank, family=family, base_seed=base_seed, perturb_sigma=0.0)
    return _warm_and_fire(f, name, m_rep.clone(), ring.clone(), g.clone())


def test_perturb_sigma_zero_is_bitwise_delayed_ef():
    """perturb_sigma=0 + rank=0 ⇒ the perturbation branch is SKIPPED ⇒ bitwise delayed_ef."""
    torch.manual_seed(50)
    m_rep, ring, g = torch.randn(6, 5), torch.randn(6, 5), torch.randn(6, 5)
    f_off = _mk_filter(rank=0, perturb_sigma=0.0, perturb_seed=7)
    f_delayed_ef = SpectralFilter(  # delayed_ef filter — NO perturb/sub-basis kwargs at all
        beta_anc=0.0, correction_mode="delayed_ef", delayed_ef_lambda=1.0, ema_device="cpu"
    )
    out_off = _warm_and_fire(f_off, _NAME, m_rep.clone(), ring.clone(), g.clone())
    out_delayed_ef = _warm_and_fire(f_delayed_ef, _NAME, m_rep.clone(), ring.clone(), g.clone())
    assert torch.equal(out_off, out_delayed_ef), (
        "perturb_sigma=0 (rank 0) must be BITWISE identical to the delayed_ef merger"
    )
    # The perturbation never fired.
    assert f_off.delayed_ef_perturb_applied == 0


def test_perturb_sigma_zero_is_bitwise_subbasis():
    """perturb_sigma=0 with the sub-basis ON ⇒ EXACT Cell-D path (perturb is a no-op).

    A filter that names perturb_sigma=0 must be byte-identical to the OLD Cell-D
    constructor that never names the perturb knobs at all — the lever is a pure
    extension with a no-op default, even on the sub-basis path.
    """
    torch.manual_seed(51)
    m_rep, ring, g = torch.randn(8, 6), torch.randn(8, 6), torch.randn(8, 6)
    f_old = SpectralFilter(  # OLD Cell-D constructor: rank=2, NO perturb kwargs.
        beta_anc=0.0,
        correction_mode="delayed_ef",
        delayed_ef_lambda=1.0,
        ema_device="cpu",
        delta_subbasis_rank=2,
        delta_subbasis_family="tail",
        base_seed=0,
    )
    f_new = _mk_filter(rank=2, family="tail", base_seed=0, perturb_sigma=0.0)
    out_old = _warm_and_fire(f_old, _NAME, m_rep.clone(), ring.clone(), g.clone())
    out_new = _warm_and_fire(f_new, _NAME, m_rep.clone(), ring.clone(), g.clone())
    assert torch.equal(out_old, out_new), "perturb_sigma=0 must reproduce the pre-perturb Cell-D path BITWISE"
    assert f_new.delayed_ef_perturb_applied == 0


def test_perturb_lambda_zero_identity_with_perturb_on():
    """λ=0 returns the g_comp OBJECT even with σ>0 (the λ=0 early-return is FIRST)."""
    f = _mk_filter(rank=0, lam=0.0, perturb_sigma=0.2)
    g = torch.randn(5, 5)
    out = f.delayed_ef_matrix(_NAME, g, ring_grad=torch.randn(5, 5))
    assert out is g, "λ=0 must return G_comp EXACTLY (same object) even with σ>0"
    assert f.delayed_ef_perturb_applied == 0, "λ=0 early-return precedes the perturbation"


@pytest.mark.parametrize("sigma", [0.05, 0.10, 0.20])
def test_perturb_magnitude_is_sigma_times_gcorr_norm(sigma):
    """‖g_corr_perturbed − g_corr‖ ≈ σ·‖g_corr‖ (ξ is unit-normalized)."""
    torch.manual_seed(52)
    m_rep, ring, g = torch.randn(16, 12), torch.randn(16, 12), torch.randn(16, 12)
    base = _unperturbed_g_corr(_NAME, m_rep, ring, g)  # the σ=0 reference g_corr
    f = _mk_filter(rank=0, perturb_sigma=sigma, perturb_seed=3)
    out = _warm_and_fire(f, _NAME, m_rep.clone(), ring.clone(), g.clone())
    base_norm = torch.linalg.norm(base.to(torch.float32)).item()
    delta_norm = torch.linalg.norm((out - base).to(torch.float32)).item()
    expected = sigma * base_norm
    # ξ is exactly unit-normalized ⇒ the injected norm is σ·‖g_corr‖ to fp32 tol.
    assert abs(delta_norm - expected) / (expected + 1e-12) < 1e-4, (
        f"σ={sigma}: ‖perturbation‖={delta_norm:.6g} must equal σ·‖g_corr‖={expected:.6g}"
    )
    assert f.delayed_ef_perturb_applied == 1


def test_perturb_cross_rank_identity():
    """Same (perturb_seed, current_step, name) ⇒ bit-identical perturbed g_corr.

    The DP-consistency guarantee: ξ is seeded by a pure function of (perturb_seed,
    canonical-name, current_step) with NO rank/device-local state, so two filters
    standing in for two DP ranks (fed the same DP-mean inputs) draw the SAME ξ and
    produce a bit-identical perturbed g_corr. A per-rank ξ would diverge the ranks.
    """
    torch.manual_seed(53)
    m_rep, ring, g = torch.randn(10, 7), torch.randn(10, 7), torch.randn(10, 7)
    rank0 = _mk_filter(rank=0, perturb_sigma=0.1, perturb_seed=11)
    rank1 = _mk_filter(rank=0, perturb_sigma=0.1, perturb_seed=11)
    # Same step on both "ranks" (the engine sets current_step from global_step).
    rank0.current_step = 4
    rank1.current_step = 4
    rank0.update_anchor(_NAME, m_rep.clone())
    rank1.update_anchor(_NAME, m_rep.clone())
    out0 = rank0.delayed_ef_matrix(_NAME, g.clone(), ring_grad=ring.clone())
    out1 = rank1.delayed_ef_matrix(_NAME, g.clone(), ring_grad=ring.clone())
    assert torch.equal(out0, out1), "cross-rank perturbed g_corr must agree bit-for-bit"


def test_perturb_cross_rank_identity_with_subbasis_on():
    """Cross-rank identity also holds when the sub-basis is ON (both terms DP-mean)."""
    torch.manual_seed(54)
    m_rep, ring, g = torch.randn(12, 9), torch.randn(12, 9), torch.randn(12, 9)
    r0 = _mk_filter(rank=3, family="tail", perturb_sigma=0.15, perturb_seed=2, base_seed=0)
    r1 = _mk_filter(rank=3, family="tail", perturb_sigma=0.15, perturb_seed=2, base_seed=0)
    r0.current_step = r1.current_step = 9
    r0.update_anchor(_NAME, m_rep.clone())
    r1.update_anchor(_NAME, m_rep.clone())
    out0 = r0.delayed_ef_matrix(_NAME, g.clone(), ring_grad=ring.clone())
    out1 = r1.delayed_ef_matrix(_NAME, g.clone(), ring_grad=ring.clone())
    assert torch.equal(out0, out1), "cross-rank identity must hold with the sub-basis ON"


def test_perturb_different_steps_give_different_noise():
    """A fresh seed per step ⇒ the perturbation DIFFERS across steps (not a fixed bias)."""
    torch.manual_seed(55)
    m_rep, ring, g = torch.randn(10, 8), torch.randn(10, 8), torch.randn(10, 8)
    base = _unperturbed_g_corr(_NAME, m_rep, ring, g)

    def _pert_at_step(step):
        f = _mk_filter(rank=0, perturb_sigma=0.1, perturb_seed=0)
        f.current_step = step
        f.update_anchor(_NAME, m_rep.clone())
        out = f.delayed_ef_matrix(_NAME, g.clone(), ring_grad=ring.clone())
        return (out - base).to(torch.float32)

    p1, p2 = _pert_at_step(1), _pert_at_step(2)
    assert not torch.allclose(p1, p2, atol=1e-6), (
        "consecutive steps must draw DIFFERENT ξ (fresh-per-step ⇒ zero-mean noise)"
    )


def test_perturb_different_seeds_give_different_noise():
    """Different perturb_seed at the SAME step ⇒ different ξ (the seed is load-bearing)."""
    torch.manual_seed(56)
    m_rep, ring, g = torch.randn(10, 8), torch.randn(10, 8), torch.randn(10, 8)
    base = _unperturbed_g_corr(_NAME, m_rep, ring, g)

    def _pert_with_seed(seed):
        f = _mk_filter(rank=0, perturb_sigma=0.1, perturb_seed=seed)
        f.current_step = 3
        f.update_anchor(_NAME, m_rep.clone())
        return (f.delayed_ef_matrix(_NAME, g.clone(), ring_grad=ring.clone()) - base).to(torch.float32)

    assert not torch.allclose(_pert_with_seed(0), _pert_with_seed(99), atol=1e-6)


def test_perturb_zero_mean_over_steps():
    """Averaging the perturbation across many steps ⇒ ‖mean‖ ≪ σ·‖g_corr‖ (zero-mean).

    Each step's perturbation has norm exactly σ·‖g_corr‖ but a fresh isotropic
    direction; the Monte-Carlo mean of N independent unit directions has norm
    ~1/√N, so the mean perturbation norm is ~σ·‖g_corr‖/√N ≪ σ·‖g_corr‖. We assert
    the mean is a small fraction of a single-step perturbation (well under 30% at
    N=200 — the 1/√N decay) to prove the noise is unbiased, not a drift.
    """
    torch.manual_seed(57)
    m_rep, ring, g = torch.randn(12, 10), torch.randn(12, 10), torch.randn(12, 10)
    sigma = 0.1
    base = _unperturbed_g_corr(_NAME, m_rep, ring, g)
    base_norm = torch.linalg.norm(base.to(torch.float32)).item()
    single_step_norm = sigma * base_norm  # every step's perturbation has this norm

    N = 200
    acc = torch.zeros_like(base, dtype=torch.float32)
    for step in range(N):
        f = _mk_filter(rank=0, perturb_sigma=sigma, perturb_seed=0)
        f.current_step = step
        f.update_anchor(_NAME, m_rep.clone())
        out = f.delayed_ef_matrix(_NAME, g.clone(), ring_grad=ring.clone())
        acc += (out - base).to(torch.float32)
    mean_norm = torch.linalg.norm(acc / N).item()
    # 1/√N law: mean_norm ≈ single_step_norm/√N ≈ single_step_norm·0.071 at N=200.
    # Assert it is < 30% of a single step (a loose bound that still proves unbiased).
    assert mean_norm < 0.30 * single_step_norm, (
        f"mean perturbation norm {mean_norm:.6g} must be ≪ a single step "
        f"{single_step_norm:.6g} (zero-mean over steps); got ratio "
        f"{mean_norm / (single_step_norm + 1e-12):.3f}"
    )


def test_apply_perturbation_zero_norm_g_unchanged():
    """A ~zero-norm g ⇒ _apply_perturbation returns it UNCHANGED (no garbage grad)."""
    f = _mk_filter(rank=0, perturb_sigma=0.2)
    z = torch.zeros(5, 5)
    out = f._apply_perturbation(_NAME, z)
    assert torch.equal(out, z), "zero-norm g must be returned unchanged"
    assert f.delayed_ef_perturb_applied == 0, "no application counted for a no-op"


def test_apply_perturbation_nonfinite_g_unchanged():
    """A non-finite g ⇒ _apply_perturbation returns it UNCHANGED (guarded)."""
    f = _mk_filter(rank=0, perturb_sigma=0.2)
    bad = torch.full((4, 4), float("nan"))
    out = f._apply_perturbation(_NAME, bad)
    assert out is bad, "non-finite g must be returned unchanged (same object)"
    assert f.delayed_ef_perturb_applied == 0
    inf = torch.full((4, 4), float("inf"))
    out2 = f._apply_perturbation(_NAME, inf)
    assert out2 is inf
    assert f.delayed_ef_perturb_applied == 0


def test_apply_perturbation_sigma_zero_returns_same_object():
    """σ=0 ⇒ _apply_perturbation is a strict no-op (same object, no count)."""
    f = _mk_filter(rank=0, perturb_sigma=0.0)
    g = torch.randn(5, 5)
    out = f._apply_perturbation(_NAME, g)
    assert out is g, "σ=0 must return the SAME g object (strict no-op)"
    assert f.delayed_ef_perturb_applied == 0


def test_apply_perturbation_output_detached_and_dtype_preserved():
    """The perturbed g_corr is detached and keeps g's dtype (g is already detached)."""
    f = _mk_filter(rank=0, perturb_sigma=0.1)
    f.current_step = 1
    g = torch.randn(6, 6, dtype=torch.float32)
    out = f._apply_perturbation(_NAME, g)
    assert not out.requires_grad, "perturbed g_corr must carry no autograd history"
    assert out.dtype == g.dtype
    assert out.shape == g.shape


def test_perturb_seed_is_per_target_step_and_reproducible():
    """_perturb_seed is FSDP-wrap invariant, per-target, per-step, reproducible."""
    f = _mk_filter(rank=0, perturb_sigma=0.1, perturb_seed=5)
    f.current_step = 7
    s_a = f._perturb_seed("layers.0.q_proj.weight")
    s_b = f._perturb_seed("layers.5.down_proj.weight")
    assert s_a != s_b, "different targets ⇒ different perturbation seeds"
    # FSDP infix canonicalized away ⇒ wrap-invariant (cross-rank, same param).
    assert s_a == f._perturb_seed("layers.0._fsdp_wrapped_module.q_proj.weight")
    # Reproducible across calls at the same step.
    assert s_a == f._perturb_seed("layers.0.q_proj.weight")
    # Different step ⇒ different seed (freshness).
    f.current_step = 8
    assert s_a != f._perturb_seed("layers.0.q_proj.weight")


def test_perturb_negative_sigma_rejected():
    """perturb_sigma < 0 is asserted in the filter ctor (mirrors config validation)."""
    with pytest.raises(AssertionError):
        _mk_filter(rank=0, perturb_sigma=-0.1)


def test_perturb_composes_with_subbasis_additively():
    """σ>0 on the sub-basis path adds σ·‖g_corr‖ noise ON TOP of the Cell-D g_corr.

    The perturbation is applied to the FULL g_corr (= G_comp + λ(δ_delayed_ef + δ_subbasis)),
    so ‖perturbed − Cell-D-g_corr‖ ≈ σ·‖Cell-D-g_corr‖ — the lever composes with the
    sub-basis, it does not replace it.
    """
    torch.manual_seed(58)
    m_rep, ring, g = torch.randn(14, 10), torch.randn(14, 10), torch.randn(14, 10)
    sigma = 0.1
    base = _unperturbed_g_corr(_NAME, m_rep, ring, g, rank=2, family="tail")
    f = _mk_filter(rank=2, family="tail", perturb_sigma=sigma, perturb_seed=1)
    out = _warm_and_fire(f, _NAME, m_rep.clone(), ring.clone(), g.clone())
    base_norm = torch.linalg.norm(base.to(torch.float32)).item()
    delta_norm = torch.linalg.norm((out - base).to(torch.float32)).item()
    assert abs(delta_norm - sigma * base_norm) / (sigma * base_norm + 1e-12) < 1e-4
    assert f.delayed_ef_subbasis_applied == 1, "the sub-basis still fired (compose, not replace)"
    assert f.delayed_ef_perturb_applied == 1


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
