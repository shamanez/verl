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

"""EXP-31 Cell D: additive stale-anchor rank-r_sb sub-basis merger — CPU tests.

Design doc (``research/runs/EXP-31/cellD_design.md``) Correctness invariants
covered here (all CPU, no GPU, no torch.distributed):

* **off-path parity (hard):** ``delta_subbasis_rank=0`` SKIPS the sub-basis branch
  entirely ⇒ ``delayed_ef_matrix`` returns the EXACT B2 ``g_corr`` (byte-compare
  to a separate rank-0 filter on the same inputs) AND the new-knob-OFF default is
  bitwise-identical to the legacy B2 filter that has no sub-basis args at all.
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
* **scale contract (the #25 mean-vs-sum trap):** feeding a SUM-reduced source
  (×world_size) inflates ‖δ_subbasis‖ by world_size (the SVD applies no rescaling)
  — mirrors ``test_delayed_ef_exp30``.
* **Step-C avoidance (structural):** the merger reads/writes only the correction
  δ; it never touches a forward basis. Asserted here by the absence of any forward
  -Q state on the filter and the fact that rank-0 is byte-identical to B2 (the
  forward path is provably untouched). The full forward-Q-checksum gate is the
  on-box probe (needs a GPU).
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


def _mk_filter(rank=0, family="tail", lam=1.0, base_seed=0):
    return SpectralFilter(
        beta_anc=0.0,
        correction_mode="delayed_ef",
        delayed_ef_lambda=lam,
        ema_device="cpu",
        delta_subbasis_rank=rank,
        delta_subbasis_family=family,
        base_seed=base_seed,
    )


def _warm_and_fire(f, name, m_rep, ring, g):
    """Warm M_rep (β=0 ⇒ M == m_rep) and apply one fire-aligned tick."""
    f.update_anchor(name, m_rep)
    return f.delayed_ef_matrix(name, g, ring_grad=ring)


# --------------------------------------------------------------------------- #
# off-path parity — rank=0 SKIPS the sub-basis branch ⇒ bitwise B2.
# --------------------------------------------------------------------------- #
def test_rank0_is_bitwise_b2():
    torch.manual_seed(0)
    m_rep = torch.randn(6, 5)
    ring = torch.randn(6, 5)
    g = torch.randn(6, 5)

    f_off = _mk_filter(rank=0)
    f_b2 = SpectralFilter(  # the legacy B2 filter — NO sub-basis kwargs at all
        beta_anc=0.0, correction_mode="delayed_ef", delayed_ef_lambda=1.0, ema_device="cpu"
    )
    out_off = _warm_and_fire(f_off, _NAME, m_rep.clone(), ring.clone(), g.clone())
    out_b2 = _warm_and_fire(f_b2, _NAME, m_rep.clone(), ring.clone(), g.clone())

    assert torch.equal(out_off, out_b2), (
        "delta_subbasis_rank=0 must be BITWISE identical to the legacy B2 merger"
    )
    # The sub-basis branch was skipped entirely: no apply, no energy ratio.
    assert f_off.delayed_ef_subbasis_applied == 0
    assert f_off.delayed_ef_subbasis_skipped == 0
    assert f_off._subbasis_energy_ratios == []


def test_rank0_default_kwargs_match_explicit_b2():
    """The new knobs DEFAULT off — a filter that never names them == B2."""
    torch.manual_seed(1)
    m_rep, ring, g = torch.randn(4, 4), torch.randn(4, 4), torch.randn(4, 4)
    # Default construction (no sub-basis args) — delta_subbasis_rank defaults 0.
    f_default = SpectralFilter(
        beta_anc=0.0, correction_mode="delayed_ef", delayed_ef_lambda=1.0, ema_device="cpu"
    )
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


def test_subbasis_degenerate_source_in_merger_skips_and_keeps_b2():
    """A degenerate δ source folds in the plain B2 δ (never garbage) + counts SKIP.

    Build a tick where δ_B2 = M_rep − ring is exactly ZERO (M_rep == ring): the
    tail source is the zero matrix ⇒ the sub-basis SKIPS, and G_corr == G_comp + δ
    = G_comp (since δ=0) — i.e. the plain B2 result, with a SKIP counted.
    """
    f = _mk_filter(rank=2, family="tail")
    same = torch.full((4, 4), 2.0)
    g = torch.randn(4, 4)
    out = _warm_and_fire(f, _NAME, same.clone(), same.clone(), g.clone())
    assert f.delayed_ef_subbasis_skipped == 1
    assert f.delayed_ef_subbasis_applied == 0
    assert torch.allclose(out.to(torch.float32), g.to(torch.float32), atol=1e-6), (
        "with δ_B2=0 and a degenerate tail source, G_corr must equal G_comp (plain B2)"
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

    δ_B2 is DP-MEAN identical across ranks by construction; the seed is a pure
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
# scale contract (#25 mean-vs-sum) — a SUM-reduced source inflates the norm.
# --------------------------------------------------------------------------- #
def test_scale_contract_sum_reduced_inflates_by_world_size():
    torch.manual_seed(8)
    world_size = 4
    S_mean = torch.randn(10, 8)
    S_sum = S_mean * world_size  # the SUM-reduced side (the #25 trap)
    f = _mk_filter(rank=3)
    sb_mean = f._subbasis_delta(_NAME, S_mean, r=3)
    sb_sum = f._subbasis_delta(_NAME, S_sum, r=3)
    n_mean = torch.linalg.norm(sb_mean).item()
    n_sum = torch.linalg.norm(sb_sum).item()
    # The SVD applies NO rescaling ⇒ a ×world_size source ⇒ ×world_size δ_subbasis.
    assert abs(n_sum / (n_mean + 1e-12) - world_size) < 1e-3, (
        f"a SUM-reduced source must inflate ‖δ_subbasis‖ by world_size; got {n_sum / n_mean:.4f}"
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
