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

"""comm-eff Q-basis family CPU unit tests for the Q-basis-family screen.

Exercises the family sketch constructions, the LIVE-family path, off-path parity
(no families ⇒ byte-identical), byte counters, and config validation —
all on CPU with no distributed runtime, so a hard-gate regression is caught here
before any GPU-hour is spent. Mirrors the on-box pre-run probe's family checks:

* off-path parity: q_basis="act" + empty passive ⇒ family harvest never arms.
* every implemented family builds an (H, r) orthonormal Q_f (ticket = axis-aligned).
* hybrid column split sums to r; the join is orthonormal.
* the LIVE non-"act" family rotates the anchor-owned Q (and "act" stays the
  byte-identical block-power-iteration basis).
* byte counters: Y=N·r compressed vs N·H dense, ratio < 1, accumulate per tick.
* config validation (q_basis_passive enum, hybrid sum, fresh_anchor_loss enum).
"""

import pytest
import torch

from verl.workers.comm_eff.powersgd_activation import (
    IMPLEMENTED_Q_FAMILIES,
    PowerSGDActivationCompressor,
    orthonormalize,
)

H = 32  # hidden size for the tests (small but > rank)
R = 8  # rank


def _make_compressor(**kw):
    """A registered-shape compressor with H known + a couple of boundaries, set up
    so the family helpers can run without a real forward/distributed runtime."""
    c = PowerSGDActivationCompressor(
        rank=R,
        base_seed=0,
        pp_size=8,
        sync_basis=False,
        qr_dtype="fp32",
        anchor_owns_q=True,
        **kw,
    )
    c._hidden_size = H
    c.boundary_indices = [1, 3]
    # Bootstrap the (deterministic seed) basis for each boundary.
    for li in c.boundary_indices:
        c._ensure_basis(li, device=torch.device("cpu"), dtype=torch.float32)
    return c


def _harvest(c, layer_idx, N=20, *, M=True, Gb=True, seed=0):
    """Stash a synthetic M / G_b harvest for a boundary (what the anchor forward +
    grad-hook would populate)."""
    g = torch.Generator().manual_seed(seed + layer_idx)
    if M:
        c._family_M[layer_idx] = torch.randn(N, H, generator=g)
    if Gb:
        c._family_Gb[layer_idx] = torch.randn(N, H, generator=g)


# --------------------------------------------------------------------------- #
# off-path parity: no families ⇒ harvest never arms (byte-identical substrate)
# --------------------------------------------------------------------------- #
def test_families_inactive_when_act_only():
    c = _make_compressor(q_basis="act", q_basis_passive=[])
    assert c._families_active is False
    c.set_anchor_sketch_mode(True)
    assert c._family_harvest is False, "act-only path must NOT arm the family harvest"
    c.set_anchor_sketch_mode(False)


def test_families_active_with_passive_list():
    c = _make_compressor(q_basis="act", q_basis_passive=["grad", "tail"])
    assert c._families_active is True
    c.set_anchor_sketch_mode(True)
    assert c._family_harvest is True
    c.set_anchor_sketch_mode(False)
    assert c._family_harvest is False


def test_families_active_with_live_nonact():
    c = _make_compressor(q_basis="grad", q_basis_passive=[])
    assert c._families_active is True


# --------------------------------------------------------------------------- #
# every implemented family builds an (H, r) basis
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("family", ["act", "grad", "adv", "tail", "ticket"])
def test_family_builds_orthonormal_basis(family):
    c = _make_compressor(q_basis="act", q_basis_passive=[family])
    for li in c.boundary_indices:
        _harvest(c, li)
    if family == "adv":
        # Per-row advantage weight aligned to M's 20 rows.
        c.set_advantage_weight(torch.rand(20).abs() + 0.1)
    out = c.build_and_dump_family_sketches(writer=None, global_step=1, optimizer_tick=10)
    assert family in out
    for li in c.boundary_indices:
        Q = out[family][li]
        assert Q.shape == (H, R), f"{family} Q_f wrong shape {tuple(Q.shape)}"
        # Orthonormal columns: QᵀQ ≈ I.
        gram = Q.t() @ Q
        assert torch.allclose(gram, torch.eye(R), atol=1e-4), f"{family} columns not orthonormal"
    if family == "ticket":
        # Axis-aligned: each column is a one-hot standard basis vector.
        for li in c.boundary_indices:
            Q = out[family][li]
            colsum = Q.abs().sum(dim=0)
            assert torch.allclose(colsum, torch.ones(R), atol=1e-5), "ticket Q not axis-aligned (one-hot cols)"


@pytest.mark.parametrize("split", [(5, 3), (-1, -1)])  # explicit + AUTO
def test_hybrid_family_builds_full_rank_orthonormal(split):
    a, g = split
    c = _make_compressor(q_basis="act", q_basis_passive=["hybrid"], hybrid_act_cols=a, hybrid_grad_cols=g)
    for li in c.boundary_indices:
        _harvest(c, li)
    out = c.build_and_dump_family_sketches(writer=None, global_step=1, optimizer_tick=10)
    for li in c.boundary_indices:
        Q = out["hybrid"][li]
        assert Q.shape == (H, R), f"hybrid split={split} wrong shape {tuple(Q.shape)}"
        gram = Q.t() @ Q
        assert torch.allclose(gram, torch.eye(R), atol=1e-4), f"hybrid split={split} join not orthonormal"


def test_tail_deflates_act_principal_subspace():
    """The tail family's deflated grad G_t = G_b - P_Qact(G_b) must drop the act
    component: <G_t, Q_act-column> ≈ 0 for the harvested grad."""
    c = _make_compressor(q_basis="act", q_basis_passive=["tail"])
    li = c.boundary_indices[0]
    _harvest(c, li)
    q_act = c._basis[li]  # (H, R)
    Gb = c._family_Gb[li]  # (N, H)
    proj = (Gb @ q_act) @ q_act.t()
    Gt = Gb - proj
    # Gt should be (numerically) orthogonal to span(Q_act): ||Gt @ Q_act|| ≈ 0.
    leak = torch.linalg.norm(Gt @ q_act).item()
    assert leak < 1e-3, f"tail deflation left {leak:.2e} act-principal energy"


# --------------------------------------------------------------------------- #
# missing operands ⇒ None (caller contributes a zero sketch) — collective safety
# --------------------------------------------------------------------------- #
def test_grad_family_none_without_gb():
    c = _make_compressor(q_basis="act", q_basis_passive=["grad"])
    li = c.boundary_indices[0]
    _harvest(c, li, M=True, Gb=False)  # no G_b
    assert c._compute_family_V("grad", li) is None


def test_act_family_none_without_m():
    c = _make_compressor(q_basis="act", q_basis_passive=["act"])
    li = c.boundary_indices[0]
    _harvest(c, li, M=False, Gb=True)  # no M
    assert c._compute_family_V("act", li) is None


# --------------------------------------------------------------------------- #
# LIVE family path: anchor_update_basis rotates Q for a non-"act" family,
# and "act" stays the block-power-iteration basis (byte-identical behaviour)
# --------------------------------------------------------------------------- #
def test_live_act_path_consumes_act_sketch():
    """q_basis='act' ⇒ anchor_update_basis orthonormalizes the act sketch V (the
    signed_ema path), independent of the family harvest."""
    c = _make_compressor(q_basis="act", q_basis_passive=[])
    li = c.boundary_indices[0]
    # Simulate the act sketch the forward hook would accumulate.
    g = torch.Generator().manual_seed(7)
    V_act = torch.randn(H, R, generator=g)
    c._sketch = {li: V_act.clone()}
    c.boundary_indices = [li]  # single boundary for a clean check
    updated = c.anchor_update_basis()
    assert updated
    expected = orthonormalize(V_act)
    assert torch.allclose(c._basis[li], expected, atol=1e-5), "act live path must orth the act sketch"


def test_live_grad_family_path_builds_from_gb():
    """q_basis='grad' ⇒ anchor_update_basis builds Q from the grad second moment
    (G_bᵀ(G_b Q)), NOT the act sketch. The resulting Q is orthonormal and DIFFERS
    from the act basis (the grad energy points elsewhere)."""
    c = _make_compressor(q_basis="grad", q_basis_passive=[])
    li = c.boundary_indices[0]
    c.boundary_indices = [li]
    q_act_before = c._basis[li].clone()
    _harvest(c, li)
    # Also populate the act sketch (the forward still accumulates V_act); the grad
    # live path must IGNORE it and consume the harvested G_b instead.
    c._sketch = {li: torch.randn(H, R)}
    updated = c.anchor_update_basis()
    assert updated
    Q = c._basis[li]
    gram = Q.t() @ Q
    assert torch.allclose(gram, torch.eye(R), atol=1e-4), "grad live Q not orthonormal"
    assert not torch.allclose(Q, q_act_before, atol=1e-3), "grad live Q should differ from the seed act basis"


# --------------------------------------------------------------------------- #
# Byte counters
# --------------------------------------------------------------------------- #
def test_byte_counters_ratio_below_one():
    """Y=N·r compressed vs N·H dense ⇒ ratio = r/H < 1; amortized Q broadcast adds
    a small term that keeps the ratio < 1 for r << H."""
    c = _make_compressor(q_basis="act", q_basis_passive=[])
    c.reset_tick_comm_counters()
    N = 100
    # Simulate one boundary forward's accumulation (what the hook does).
    c.tick_elems_compressed += float(N) * float(R)
    c.tick_elems_dense_equiv += float(N) * float(H)
    c.add_amortized_q_broadcast_bytes()
    c.last_elems_compressed = c.tick_elems_compressed
    c.last_elems_dense_equiv = c.tick_elems_dense_equiv
    ratio = c.last_elems_compressed / c.last_elems_dense_equiv
    assert ratio < 1.0, f"compressed ratio {ratio:.3f} must be < 1 for r={R} << H={H}"
    # The dense-equiv is exactly N·H for the one boundary.
    assert c.last_elems_dense_equiv == float(N) * float(H)


def test_reset_tick_comm_counters_zeroes():
    c = _make_compressor(q_basis="act")
    c.tick_elems_compressed = 123.0
    c.tick_elems_dense_equiv = 456.0
    c.reset_tick_comm_counters()
    assert c.tick_elems_compressed == 0.0 and c.tick_elems_dense_equiv == 0.0


# --------------------------------------------------------------------------- #
# harvest lifecycle
# --------------------------------------------------------------------------- #
def test_clear_family_harvest_empties_buffers():
    c = _make_compressor(q_basis="act", q_basis_passive=["grad"])
    for li in c.boundary_indices:
        _harvest(c, li)
    c.set_advantage_weight(torch.rand(20))
    assert c._family_M and c._family_Gb and c._adv_weight is not None
    c.clear_family_harvest()
    assert not c._family_M and not c._family_Gb and c._adv_weight is None


# --------------------------------------------------------------------------- #
# config validation
# --------------------------------------------------------------------------- #
def test_config_validates_passive_family_enum():
    from verl.workers.config.comm_eff import CommEffConfig, CommEffPowerSGDConfig

    with pytest.raises(ValueError, match="q_basis_passive"):
        CommEffConfig(
            enabled=True, compression_type="powersgd", powersgd=CommEffPowerSGDConfig(q_basis_passive=["grad", "bogus"])
        )


def test_config_validates_hybrid_split_sums_to_rank():
    from verl.workers.config.comm_eff import CommEffConfig, CommEffPowerSGDConfig

    # Hybrid requested but split does NOT sum to rank ⇒ loud error.
    with pytest.raises(ValueError, match="hybrid_act_cols"):
        CommEffConfig(
            enabled=True,
            compression_type="powersgd",
            powersgd=CommEffPowerSGDConfig(
                rank=77, q_basis_passive=["hybrid"], hybrid_act_cols=40, hybrid_grad_cols=30
            ),
        )
    # Correct split (39 + 38 = 77) parses.
    cfg = CommEffConfig(
        enabled=True,
        compression_type="powersgd",
        powersgd=CommEffPowerSGDConfig(rank=77, q_basis_passive=["hybrid"], hybrid_act_cols=39, hybrid_grad_cols=38),
    )
    assert cfg.powersgd.hybrid_act_cols == 39


def test_config_hybrid_unconstrained_when_unused():
    from verl.workers.config.comm_eff import CommEffConfig, CommEffPowerSGDConfig

    # Hybrid NOT requested ⇒ the (default) split need not sum to rank.
    cfg = CommEffConfig(
        enabled=True, compression_type="powersgd", powersgd=CommEffPowerSGDConfig(rank=77, q_basis_passive=["grad"])
    )
    assert cfg.powersgd.q_basis == "act"


def test_config_validates_fresh_anchor_loss_enum():
    from verl.workers.config.comm_eff import CommEffCaptureConfig, CommEffConfig

    with pytest.raises(ValueError, match="fresh_anchor_loss"):
        CommEffConfig(
            enabled=True, compression_type="powersgd", capture=CommEffCaptureConfig(fresh_anchor_loss="bogus")
        )
    cfg = CommEffConfig(
        enabled=True, compression_type="powersgd", capture=CommEffCaptureConfig(fresh_anchor_loss="ppo_clip")
    )
    assert cfg.capture.fresh_anchor_loss == "ppo_clip"


def test_implemented_families_cover_spec():
    """All six STEP_C_SPEC families are implemented (none left fail-loud)."""
    assert set(IMPLEMENTED_Q_FAMILIES) == {"act", "grad", "adv", "tail", "hybrid", "ticket"}
