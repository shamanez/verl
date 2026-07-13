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

"""CPU unit tests for the look-ahead (weight-projection) anchor module.

Loads ``lookahead.py`` by file path (it depends only on torch + stdlib, so no
package stubs are needed). Pins the M4 fixed-linear projector invariants:

1. **Generalized horizon math** — ``project(sources, ticks, fire_tick)``
   recovers a synthetic LINEAR weight trajectory EXACTLY at any ``(h, g)``,
   including ``h != g`` (the original naive-linear bug: the frozen seed
   ``(2, -1)`` lands on the fire tick only when cadence == delay_K). This is the
   RLVR-linearity paper's Eq 4 weight-space extrapolation (arXiv:2601.04537).
2. **Seed reduction** — at ``h == g`` the generalized coefficients equal the
   frozen AsyncPP seed, and the tick-less fallback produces the identical
   ``theta_hat`` (operating-point behavior unchanged).
3. **Exclusion set** — non-target and non-2D params take ``S0`` verbatim
   (the LayerNorm/embedding exclusion), by reference.
4. **True-tick ring** — eviction bound, newest-first ``sources()``,
   ``get()`` exact-match lookup, same-tick overwrite, min_points relaxation.
5. **Rollout-source resolver** — ``auto`` resolves by projector state;
   explicit values pass through untouched.
"""

import importlib.util
import pathlib
import sys
import types

import pytest
import torch

_REPO = pathlib.Path(__file__).resolve().parents[3]


def _load(mod_name, rel_path):
    spec = importlib.util.spec_from_file_location(mod_name, _REPO / rel_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


_la = _load("verl_workers_comm_eff_lookahead_testonly", "verl/workers/comm_eff/lookahead.py")

TARGETS = ("q_proj", "o_proj")


def _cfg(mode="fixed_linear", enabled=True, strength=1.0, rollout_source="auto"):
    """Bare attribute bag standing in for CommEffAnchorConfig."""
    return types.SimpleNamespace(
        lookahead_anchor=enabled,
        lookahead_mode=mode,
        lookahead_strength=strength,
        lookahead_rollout_source=rollout_source,
    )


def _snap_at(tick, base, drift):
    """Synthetic LINEAR trajectory W(t) = base + t * drift, per param."""
    return {name: (base[name] + float(tick) * drift[name]) for name in base}


def _make_params(d=4):
    torch.manual_seed(0)
    base = {
        "layers.0.q_proj.weight": torch.randn(d, d),
        "layers.0.o_proj.weight": torch.randn(d, d),
        "model.norm.weight": torch.randn(d),  # 1-D -> excluded
        "embed_tokens.weight": torch.randn(d, d),  # 2-D non-target -> excluded
    }
    drift = {name: torch.randn_like(t) * 0.01 for name, t in base.items()}
    return base, drift


# =========================================================================== #
# 1 + 2. Generalized horizon math + seed reduction
# =========================================================================== #
@pytest.mark.parametrize(
    "ticks,fire_tick",
    [
        ([30, 20], 40),  # h=10 == g=10: the seed operating point (20/20-like)
        ([30, 20], 45),  # h=15 != g=10: the case the frozen seed got WRONG
        ([25, 20], 45),  # h=20, g=5 (cadence 5 / delay 20)
        ([30, 20], 30),  # h=0: theta_hat == S0 (no forward motion)
    ],
)
def test_project_lands_on_fire_tick_for_linear_trajectory(ticks, fire_tick):
    base, drift = _make_params()
    proj = _la.LookaheadProjector(_cfg(), TARGETS)
    sources = [_snap_at(ticks[0], base, drift), _snap_at(ticks[1], base, drift)]
    theta_hat, excluded, info = proj.project(sources, ticks=ticks, fire_tick=fire_tick)
    assert info["h"] == fire_tick - ticks[0] and info["g"] == ticks[0] - ticks[1]
    expected = _snap_at(fire_tick, base, drift)
    for name in ("layers.0.q_proj.weight", "layers.0.o_proj.weight"):
        torch.testing.assert_close(theta_hat[name], expected[name], rtol=1e-5, atol=1e-5)
    # Excluded params take S0 VERBATIM (by reference — no copy, no math).
    for name in ("model.norm.weight", "embed_tokens.weight"):
        assert name in excluded
        assert theta_hat[name] is sources[0][name]


def test_seed_fallback_equals_generalized_at_h_eq_g():
    base, drift = _make_params()
    proj = _la.LookaheadProjector(_cfg(), TARGETS)
    sources = [_snap_at(30, base, drift), _snap_at(20, base, drift)]
    hat_general, _, info = proj.project(sources, ticks=[30, 20], fire_tick=40)
    hat_seed, _, info_seed = proj.project(sources)  # tick-less fallback
    assert info["coeffs"] == pytest.approx(tuple(_la.FIXED_LINEAR_COEFFS))
    assert info_seed["h"] is None and info_seed["g"] is None
    for name in hat_general:
        torch.testing.assert_close(hat_general[name], hat_seed[name], rtol=0, atol=0)


def test_strength_scales_the_horizon():
    base, drift = _make_params()
    proj = _la.LookaheadProjector(_cfg(strength=0.5), TARGETS)
    sources = [_snap_at(30, base, drift), _snap_at(20, base, drift)]
    theta_hat, _, _ = proj.project(sources, ticks=[30, 20], fire_tick=40)
    # alpha=0.5 over h=g=10 projects HALFWAY: W(35).
    expected = _snap_at(35, base, drift)
    torch.testing.assert_close(
        theta_hat["layers.0.q_proj.weight"], expected["layers.0.q_proj.weight"], rtol=1e-5, atol=1e-5
    )


def test_project_asserts_on_regressed_tick_order():
    base, drift = _make_params()
    proj = _la.LookaheadProjector(_cfg(), TARGETS)
    sources = [_snap_at(20, base, drift), _snap_at(30, base, drift)]
    with pytest.raises(AssertionError, match="strictly decreasing"):
        proj.project(sources, ticks=[20, 30], fire_tick=40)
    with pytest.raises(AssertionError, match="BACKWARD"):
        proj.project(
            [_snap_at(30, base, drift), _snap_at(20, base, drift)], ticks=[30, 20], fire_tick=25
        )


# =========================================================================== #
# 3. compute_theta_hat exclusion / defensive fallbacks
# =========================================================================== #
def test_compute_theta_hat_missing_source_falls_back_to_stale():
    base, drift = _make_params()
    s0 = _snap_at(30, base, drift)
    s1 = {k: v for k, v in _snap_at(20, base, drift).items() if "q_proj" not in k}
    theta_hat, excluded = _la.compute_theta_hat([s0, s1], (2.0, -1.0, 0.0), target_substrs=TARGETS)
    # q_proj has no S1 -> raw stale, listed excluded; o_proj still projects.
    assert "layers.0.q_proj.weight" in excluded
    assert theta_hat["layers.0.q_proj.weight"] is s0["layers.0.q_proj.weight"]
    assert "layers.0.o_proj.weight" not in excluded


# =========================================================================== #
# 4. True-tick ring
# =========================================================================== #
def test_ring_true_tick_keying_bound_and_sources():
    ring = _la.LookaheadSnapshotRing(n_points=2)
    assert not ring.ready() and ring.sources() == (None, None)
    ring.push(10, {"w": torch.tensor(10.0)})
    assert not ring.ready()  # 1 of 2
    ring.push(20, {"w": torch.tensor(20.0)})
    ring.push(30, {"w": torch.tensor(30.0)})  # evicts tick 10
    assert ring.ticks == [20, 30] and ring.ready()
    snaps, ticks = ring.sources()
    assert ticks == [30, 20]  # newest-first
    assert float(snaps[0]["w"]) == 30.0 and float(snaps[1]["w"]) == 20.0
    assert ring.get(20) is not None and ring.get(10) is None
    # Same-tick re-push (warmup fallback) overwrites in place — no growth.
    ring.push(30, {"w": torch.tensor(31.0)})
    assert ring.ticks == [20, 30] and float(ring.get(30)["w"]) == 31.0


def test_lookahead_min_points_helper():
    """lookahead_min_snapshots resolution: -1 -> mode n_points; concrete pass-through; disabled -> 0."""
    fixed = _cfg(mode="fixed_linear")
    assert _la.lookahead_num_source_points(fixed) == 2
    assert _la.lookahead_min_points(fixed) == 2  # min_snapshots defaults to -1
    fixed.lookahead_min_snapshots = 2
    assert _la.lookahead_min_points(fixed) == 2  # concrete value passes through
    # Disabled -> 0 regardless of the knob.
    off = _cfg(enabled=False)
    off.lookahead_min_snapshots = 2
    assert _la.lookahead_min_points(off) == 0


def test_ring_min_points_ready_early_but_retains_full():
    """min_points relaxes readiness (project at fire 2) while retention stays n_points.

    Pure container behavior at n_points=3, min_points=2: ready() at 2 snapshots;
    sources() then returns the 2 newest (compute_theta_hat handles the 2-source
    case); once a 3rd arrives, retention is still bounded at 3.
    """
    ring = _la.LookaheadSnapshotRing(n_points=3, min_points=2)
    assert not ring.ready()  # 0 of 2
    ring.push(0, {"w": torch.tensor(0.0)})
    assert not ring.ready()  # 1 of 2
    ring.push(20, {"w": torch.tensor(20.0)})
    assert ring.ready()  # 2 of 2 -> the earliest legal (fire 2) projection
    snaps, ticks = ring.sources()
    assert ticks == [20, 0] and len(snaps) == 2  # 2 sources legal
    ring.push(40, {"w": torch.tensor(40.0)})  # 3rd point; retention bound = n_points=3
    assert ring.ticks == [0, 20, 40]
    snaps, ticks = ring.sources()
    assert ticks == [40, 20, 0] and len(snaps) == 3
    ring.push(60, {"w": torch.tensor(60.0)})  # evicts oldest -> still bounded at 3
    assert ring.ticks == [20, 40, 60]


def test_ring_min_points_bounds_asserted():
    """min_points must be in [2, n_points]."""
    with pytest.raises(AssertionError):
        _la.LookaheadSnapshotRing(n_points=3, min_points=1)
    with pytest.raises(AssertionError):
        _la.LookaheadSnapshotRing(n_points=2, min_points=3)
    # Default (min_points=None) reproduces n_points readiness exactly.
    ring = _la.LookaheadSnapshotRing(n_points=3)
    assert ring.min_points == 3
    ring.push(0, {"w": torch.tensor(0.0)})
    ring.push(20, {"w": torch.tensor(20.0)})
    assert not ring.ready()  # 2 of 3 -> NOT ready under the default threshold


# =========================================================================== #
# 5. Rollout-source resolver
# =========================================================================== #
def test_resolver_auto_follows_projector_state():
    assert _la.resolve_lookahead_rollout_source(_cfg(enabled=True)) == "current_step"
    assert _la.resolve_lookahead_rollout_source(_cfg(enabled=False)) == "stale_paired"
    assert _la.resolve_lookahead_rollout_source(_cfg(mode="disabled")) == "stale_paired"
    assert _la.resolve_lookahead_rollout_source(None) == "stale_paired"


def test_resolver_explicit_values_pass_through():
    assert _la.resolve_lookahead_rollout_source(_cfg(rollout_source="stale_paired")) == "stale_paired"
    assert _la.resolve_lookahead_rollout_source(_cfg(rollout_source="current_step")) == "current_step"


# =========================================================================== #
# Predicates
# =========================================================================== #
def test_enable_predicates_require_both_flags():
    assert _la.lookahead_enabled(_cfg(enabled=True, mode="fixed_linear"))
    assert not _la.lookahead_enabled(_cfg(enabled=False, mode="fixed_linear"))
    assert not _la.lookahead_enabled(_cfg(enabled=True, mode="disabled"))
    assert not _la.lookahead_enabled(None)
    assert _la.lookahead_num_source_points(_cfg(mode="fixed_linear")) == 2
    assert _la.lookahead_num_source_points(_cfg(enabled=False)) == 0
