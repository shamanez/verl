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
package stubs are needed). Pins the M4 projector invariants:

1. **Generalized horizon math** — ``project(sources, ticks, fire_tick)``
   recovers a synthetic LINEAR weight trajectory EXACTLY at any ``(h, g)``,
   including ``h != g`` (the original naive-linear bug: the frozen seed
   ``(2, -1)`` lands on the fire tick only when cadence == delay_K).
2. **Seed reduction** — at ``h == g`` the generalized coefficients equal the
   frozen AsyncPP seed, and the tick-less fallback produces the identical
   ``theta_hat`` (operating-point behavior unchanged).
3. **Exclusion set** — non-target and non-2D params take ``S0`` verbatim
   (the LayerNorm/embedding exclusion), by reference.
4. **True-tick ring** — eviction bound, newest-first ``sources()``,
   ``get()`` exact-match lookup, same-tick overwrite.
5. **Rollout-source resolver** — ``auto`` resolves by projector state;
   explicit values pass through untouched.
6. **Learned mode** — cold start is byte-identical to fixed-linear; the
   retrospective residual is bounded and folded in.
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
    # Learned mode: n_points=3; default (-1) -> 3; explicit 2 -> 2.
    learned = _cfg(mode="learned_linear_with_fixed_linear_cold_start")
    assert _la.lookahead_num_source_points(learned) == 3
    assert _la.lookahead_min_points(learned) == 3  # min_snapshots defaults to -1
    learned.lookahead_min_snapshots = 2
    assert _la.lookahead_min_points(learned) == 2
    # Fixed mode: n_points=2; default -> 2.
    fixed = _cfg(mode="fixed_linear")
    assert _la.lookahead_min_points(fixed) == 2
    # Disabled -> 0 regardless of the knob.
    off = _cfg(enabled=False)
    off.lookahead_min_snapshots = 2
    assert _la.lookahead_min_points(off) == 0


def test_ring_min_points_ready_early_but_retains_full():
    """min_points relaxes readiness (project at fire 2) while retention stays n_points.

    Learned mode: n_points=3, min_points=2. ready() at 2 snapshots; sources()
    then returns the 2 newest (compute_theta_hat handles s2=None); once a 3rd
    arrives, retention is still bounded at 3 and sources() returns all 3.
    """
    ring = _la.LookaheadSnapshotRing(n_points=3, keep_theta_hat=True, min_points=2)
    assert not ring.ready()  # 0 of 2
    ring.push(0, {"w": torch.tensor(0.0)})
    assert not ring.ready()  # 1 of 2
    ring.push(20, {"w": torch.tensor(20.0)})
    assert ring.ready()  # 2 of 2 -> the earliest legal (fire 2) projection
    snaps, ticks = ring.sources()
    assert ticks == [20, 0] and len(snaps) == 2  # 2 sources legal (s2=None)
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


def test_ring_keeps_prev_theta_hat_only_when_asked():
    ring = _la.LookaheadSnapshotRing(n_points=2, keep_theta_hat=False)
    ring.set_prev_theta_hat(40, {"w": torch.tensor(0.0)})
    assert ring.prev_theta_hat() == (None, -1)
    ring = _la.LookaheadSnapshotRing(n_points=3, keep_theta_hat=True)
    ring.set_prev_theta_hat(40, {"w": torch.tensor(0.0)})
    hat, tick = ring.prev_theta_hat()
    assert hat is not None and tick == 40
    assert ring.total_retained() == 1  # counted in the memory report


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
# 6. Learned mode
# =========================================================================== #
def test_learned_cold_start_is_byte_identical_to_fixed_linear():
    base, drift = _make_params()
    sources = [_snap_at(30, base, drift), _snap_at(20, base, drift), _snap_at(10, base, drift)]
    fixed = _la.LookaheadProjector(_cfg(mode="fixed_linear"), TARGETS)
    learned = _la.LookaheadProjector(
        _cfg(mode="learned_linear_with_fixed_linear_cold_start"), TARGETS
    )
    hat_f, _, _ = fixed.project(sources[:2], ticks=[30, 20], fire_tick=40)
    hat_l, _, _ = learned.project(sources, ticks=[30, 20, 10], fire_tick=40)
    for name in hat_f:
        torch.testing.assert_close(hat_f[name], hat_l[name], rtol=0, atol=0)


def test_learned_residual_update_is_bounded_and_folded_in():
    base, drift = _make_params()
    learned = _la.LookaheadProjector(
        _cfg(mode="learned_linear_with_fixed_linear_cold_start"), TARGETS
    )
    name = "layers.0.q_proj.weight"
    theta_hat_prev = {name: base[name].clone()}
    # Huge retrospective error -> the residual must CLIP at 1e-3, not blow up.
    theta_true_prev = {name: base[name] + 5.0}
    residual = learned.update_from_retrospective(theta_true_prev, theta_hat_prev)
    assert residual[name] == pytest.approx(1.0e-3)
    vec = learned.residual_vector()
    assert vec.numel() == 1 and float(vec[0]) == pytest.approx(1.0e-3)
    # project() folds the residual into targets only.
    sources = [_snap_at(30, base, drift), _snap_at(20, base, drift), _snap_at(10, base, drift)]
    hat, _, _ = learned.project(sources, ticks=[30, 20, 10], fire_tick=40)
    fixed = _la.LookaheadProjector(_cfg(mode="fixed_linear"), TARGETS)
    hat_f, _, _ = fixed.project(sources[:2], ticks=[30, 20], fire_tick=40)
    torch.testing.assert_close(hat[name], hat_f[name] + 1.0e-3, rtol=1e-6, atol=1e-7)


# =========================================================================== #
# 6b. Step-scale learned mode (learned_step_scale_with_fixed_linear_cold_start)
# =========================================================================== #
_SCALE = "learned_step_scale_with_fixed_linear_cold_start"


def test_step_scale_cold_start_is_byte_identical_to_fixed_linear():
    """beta defaults to 1.0 (un-updated) => the step equals the fixed-linear step."""
    base, drift = _make_params()
    sources = [_snap_at(30, base, drift), _snap_at(20, base, drift)]
    fixed = _la.LookaheadProjector(_cfg(mode="fixed_linear"), TARGETS)
    scale = _la.LookaheadProjector(_cfg(mode=_SCALE), TARGETS)
    hat_f, _, _ = fixed.project(sources, ticks=[30, 20], fire_tick=40)
    hat_s, _, _ = scale.project(sources, ticks=[30, 20], fire_tick=40)
    for name in hat_f:
        torch.testing.assert_close(hat_f[name], hat_s[name], rtol=0, atol=0)


def test_step_scale_beta_scales_the_projection_step():
    """A learned beta != 1 rescales the step: beta=1.5 over h=g=10 projects to W(45)."""
    base, drift = _make_params()
    proj = _la.LookaheadProjector(_cfg(mode=_SCALE), TARGETS)
    name = "layers.0.q_proj.weight"
    proj._beta[name] = 1.5  # manually set a learned scale for this block
    sources = [_snap_at(30, base, drift), _snap_at(20, base, drift)]
    hat, _, _ = proj.project(sources, ticks=[30, 20], fire_tick=40)
    # eff = beta*base_scale = 1.5*(alpha*h/g=1.0) => (1+1.5)*W(30) - 1.5*W(20) = W(45).
    torch.testing.assert_close(hat[name], _snap_at(45, base, drift)[name], rtol=1e-5, atol=1e-5)
    # A block WITHOUT a learned beta still uses 1.0 => lands on the fire tick W(40).
    torch.testing.assert_close(
        hat["layers.0.o_proj.weight"], _snap_at(40, base, drift)["layers.0.o_proj.weight"], rtol=1e-5, atol=1e-5
    )


def test_step_scale_update_direction_under_and_over_shoot():
    """rho>0 (truth further along the step) grows beta; rho<0 shrinks it."""
    name = "layers.0.q_proj.weight"
    s0 = torch.zeros(2, 2)
    step = torch.ones(2, 2)  # the step the prior fire took: theta_hat - S0
    theta_hat_prev = {name: s0 + step}
    s0_prev = {name: s0}
    # Under-projection: truth is 1.5*step (further) => rho = <0.5*step, step>/||step||^2 = 0.5.
    proj = _la.LookaheadProjector(_cfg(mode=_SCALE), TARGETS)
    proj.update_from_retrospective({name: s0 + 1.5 * step}, theta_hat_prev, s0_prev=s0_prev)
    assert proj._beta[name] == pytest.approx(1.0 + 0.2 * 0.5)  # beta_lr=0.2 => 1.1
    # Over-projection: truth is 0.5*step (short) => rho = -0.5 => beta = 0.9.
    proj2 = _la.LookaheadProjector(_cfg(mode=_SCALE), TARGETS)
    proj2.update_from_retrospective({name: s0 + 0.5 * step}, theta_hat_prev, s0_prev=s0_prev)
    assert proj2._beta[name] == pytest.approx(1.0 - 0.2 * 0.5)  # 0.9


def test_step_scale_beta_is_trust_region_bounded():
    """Repeated huge over/under-shoot saturates at [beta_min, beta_max] = [0, 2] (rho clipped per fire)."""
    name = "layers.0.q_proj.weight"
    s0 = torch.zeros(2, 2)
    step = torch.ones(2, 2)
    theta_hat_prev = {name: s0 + step}
    s0_prev = {name: s0}
    up = _la.LookaheadProjector(_cfg(mode=_SCALE), TARGETS)
    down = _la.LookaheadProjector(_cfg(mode=_SCALE), TARGETS)
    for _ in range(20):
        up.update_from_retrospective({name: s0 + 100.0 * step}, theta_hat_prev, s0_prev=s0_prev)
        down.update_from_retrospective({name: s0 - 100.0 * step}, theta_hat_prev, s0_prev=s0_prev)
    assert up._beta[name] == pytest.approx(2.0)  # beta_max
    assert down._beta[name] == pytest.approx(0.0)  # beta_min (degrades to raw stale)


def test_step_scale_no_step_leaves_beta_untouched():
    """A block whose prior fire took no step (theta_hat == S0) has undefined rho -> skip."""
    name = "layers.0.q_proj.weight"
    s0 = torch.zeros(2, 2)
    proj = _la.LookaheadProjector(_cfg(mode=_SCALE), TARGETS)
    proj.update_from_retrospective({name: s0 + 5.0}, {name: s0.clone()}, s0_prev={name: s0.clone()})
    assert name not in proj._beta  # no update recorded


def test_step_scale_update_is_noop_without_s0_prev():
    """No s0_prev (warmup / no prior step available) -> beta unchanged."""
    name = "layers.0.q_proj.weight"
    proj = _la.LookaheadProjector(_cfg(mode=_SCALE), TARGETS)
    proj.update_from_retrospective({name: torch.ones(2, 2)}, {name: torch.zeros(2, 2)}, s0_prev=None)
    assert proj._beta == {}


def test_step_scale_residual_vector_reports_beta():
    """residual_vector() returns the per-block beta (for the cross-rank determinism probe)."""
    proj = _la.LookaheadProjector(_cfg(mode=_SCALE), TARGETS)
    assert proj.residual_vector().numel() == 0  # empty until first update
    proj._beta["layers.0.q_proj.weight"] = 1.3
    proj._beta["layers.0.o_proj.weight"] = 0.7
    vec = proj.residual_vector()
    assert vec.numel() == 2  # sorted by name: o_proj ('o') before q_proj ('q')
    assert float(vec[0]) == pytest.approx(0.7) and float(vec[1]) == pytest.approx(1.3)


def test_step_scale_recovers_true_scale_on_a_nonlinear_trajectory():
    """End-to-end: on a trajectory the fixed step under-shoots, beta climbs toward the correction.

    Build snapshots where the realized displacement over the horizon is 1.4x the
    fixed-linear step, so fixed_linear under-projects every fire; beta must grow
    (toward, but bounded by the trust region / lr, not instantly to 1.4).
    """
    name = "layers.0.q_proj.weight"
    s1 = torch.zeros(2, 2)
    s0 = torch.ones(2, 2)  # trajectory dir d = S0 - S1 = ones; fixed step (h=g) = d
    proj = _la.LookaheadProjector(_cfg(mode=_SCALE), TARGETS)
    # Prior fire projected with beta=1: theta_hat = S0 + 1*d = 2*ones.
    theta_hat_prev = {name: s0 + (s0 - s1)}
    # Truth landed at S0 + 1.4*d (the trajectory sped up) -> under-projection.
    theta_true_prev = {name: s0 + 1.4 * (s0 - s1)}
    proj.update_from_retrospective(theta_true_prev, theta_hat_prev, s0_prev={name: s0})
    assert proj._beta[name] > 1.0  # grew toward the true scale
    assert proj._beta[name] <= 2.0  # still inside the trust region


# =========================================================================== #
# Predicates
# =========================================================================== #
def test_enable_predicates_require_both_flags():
    assert _la.lookahead_enabled(_cfg(enabled=True, mode="fixed_linear"))
    assert not _la.lookahead_enabled(_cfg(enabled=False, mode="fixed_linear"))
    assert not _la.lookahead_enabled(_cfg(enabled=True, mode="disabled"))
    assert not _la.lookahead_enabled(None)
    assert _la.lookahead_num_source_points(_cfg(mode="fixed_linear")) == 2
    assert _la.lookahead_num_source_points(_cfg(mode="learned_linear_with_fixed_linear_cold_start")) == 3
    assert _la.lookahead_num_source_points(_cfg(enabled=False)) == 0
    # Step-scale mode: enabled, learns per-block state, but FIRST-ORDER (2 points).
    assert _la.lookahead_enabled(_cfg(mode=_SCALE))
    assert _la.lookahead_learns(_cfg(mode=_SCALE))
    assert _la.lookahead_num_source_points(_cfg(mode=_SCALE)) == 2
    # fixed_linear enables the projector but trains nothing.
    assert not _la.lookahead_learns(_cfg(mode="fixed_linear"))
