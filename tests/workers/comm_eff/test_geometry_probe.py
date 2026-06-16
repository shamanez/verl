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

"""EXP-30 Step-A geometry probe — CPU unit tests (no GPU / distributed / FSDP).

Covers the plan's Correctness invariants that are CPU-checkable:

* fire-aware fast-grad ring bounds (≤ delay_K//cadence + 1 = 2 entries at the
  locked cadence=5/delay_K=5) + exact-tick lookup + CPU residency;
* m4 lag-buffer bounds (≤ 5 stored + in-flight = the plan's ≤6-entry bound);
* δ scale-consistency — the #25 mean-vs-sum trap: ``delta_stats_over_targets``
  applies NO hidden rescaling, so mean+mean reduction is world-size invariant
  and a sum-reduced side inflates the ratio by exactly (world−1) on the
  identical-objects construction;
* β_anc=0 semantics: ``update_anchor`` at β=0 yields M_rep == G_anc_rep EXACTLY
  (no bias-correction division, no (1−β) scaling surprise);
* m1–m7 record assembly: verbatim field names, warmup nulls, known cosines;
* off-path parity: probe disabled ⇒ no ring, no lag buffer, zero counters.
"""

import importlib.util
import json
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
_st = _load("verl.workers.comm_eff.state", "verl/workers/comm_eff/state.py")
_an = _load("verl.workers.comm_eff.anchor", "verl/workers/comm_eff/anchor.py")

FastGradRing = _st.FastGradRing
GradLagBuffer = _st.GradLagBuffer
SpectralFilter = _sf.SpectralFilter


def _grads(val, shape=(4, 4)):
    t = torch.full(shape, float(val), dtype=torch.float32)
    return {"layer.q_proj.weight": t}


# --------------------------------------------------------------------------- #
# FastGradRing — fire-aware retention, bounds, exact lookup, CPU residency.
# --------------------------------------------------------------------------- #
def test_fast_grad_ring_retention_and_bounds():
    ring = FastGradRing(delay_K=5, cadence=5)
    assert ring._maxlen == 2  # delay_K // cadence + 1 — the plan's ≤2-entry bound
    stored = []
    for tick in range(1, 41):
        ok = ring.push(tick, _grads(tick), {"layer.q_proj.weight": float(tick)})
        # Only ticks ≡ (−5) mod 5 == 0 mod 5 are retained — the fire ticks.
        assert ok == (tick % 5 == 0), f"tick={tick} retained={ok}"
        if ok:
            stored.append(tick)
        assert len(ring) <= 2, f"ring blew the 2-entry bound at tick {tick}"
    # Exact-tick lookup only — no fallback (a near-miss would corrupt m5).
    assert ring.get(40) is not None
    assert ring.get(39) is None
    grads, norms = ring.get(40)
    assert torch.equal(grads["layer.q_proj.weight"], torch.full((4, 4), 40.0))
    assert norms["layer.q_proj.weight"] == 40.0
    # Duplicate push refused.
    assert ring.push(40, _grads(40)) is False
    # pop drops a consumed entry.
    ring.pop(40)
    assert ring.get(40) is None


def test_fast_grad_ring_rejects_non_cpu():
    ring = FastGradRing(delay_K=5, cadence=5)
    bad = {"layer.q_proj.weight": torch.empty(2, 2, device="meta")}
    with pytest.raises(AssertionError, match="CPU-resident"):
        ring.push(5, bad)


def test_fast_grad_ring_cadence1_legacy_shape():
    # cadence=1 retains every tick, bounded at delay_K+1 (the c512128 pattern).
    ring = FastGradRing(delay_K=3, cadence=1)
    assert ring._maxlen == 4
    for tick in range(1, 20):
        assert ring.push(tick, _grads(tick)) is True
        assert len(ring) <= 4


# --------------------------------------------------------------------------- #
# GradLagBuffer — rolling ≤5-stored bound (≤6 with the in-flight current).
# --------------------------------------------------------------------------- #
def test_grad_lag_buffer_bounds_and_rolloff():
    lag = GradLagBuffer(max_lag=5)
    for tick in range(1, 30):
        assert lag.push(tick, _grads(tick)) is True
        assert len(lag) <= 5, f"lag buffer blew the 5-entry bound at tick {tick}"
    # At tick t=29 pushed, the window is {25..29}; a fire at 30 reads 25..29.
    assert lag.ticks == [25, 26, 27, 28, 29]
    assert lag.get(24) is None
    assert lag.get(25) is not None
    with pytest.raises(AssertionError):
        GradLagBuffer(max_lag=6)  # would break the plan's ≤6-entry bound
    with pytest.raises(AssertionError, match="CPU-resident"):
        GradLagBuffer(max_lag=5).push(1, {"w": torch.empty(2, 2, device="meta")})


# --------------------------------------------------------------------------- #
# grad_summary_stats — m7 stable rank + top-1% energy mass.
# --------------------------------------------------------------------------- #
def test_grad_summary_stats_known_matrices():
    # Rank-1 matrix: stable rank == 1 (fro == sigma1).
    u = torch.arange(1.0, 9.0).reshape(8, 1)
    v = torch.arange(1.0, 7.0).reshape(1, 6)
    s = _an.grad_summary_stats(u @ v)
    assert s["stable_rank"] == pytest.approx(1.0, abs=1e-3)
    # Identity_8: fro² = 8, sigma1 = 1 ⇒ stable rank == 8.
    s = _an.grad_summary_stats(torch.eye(8))
    assert s["stable_rank"] == pytest.approx(8.0, rel=1e-2)
    # Single-spike 10×10 (k = max(1, round(1)) = 1): top-1% mass ≈ spike²/fro².
    m = torch.full((10, 10), 0.01)
    m[3, 7] = 100.0
    s = _an.grad_summary_stats(m)
    assert s["top1pct_mass"] == pytest.approx(1.0, abs=1e-5)
    # Zero matrix: all-zero stats, no NaN/crash.
    s = _an.grad_summary_stats(torch.zeros(4, 4))
    assert s == {"fro": 0.0, "sigma1": 0.0, "stable_rank": 0.0, "top1pct_mass": 0.0}


# --------------------------------------------------------------------------- #
# paired_cosine / cos_over_targets — known geometry + zero-vector None.
# --------------------------------------------------------------------------- #
def test_paired_cosine_known_values():
    a = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    b = torch.tensor([[0.0, 1.0], [0.0, 0.0]])
    assert _an.paired_cosine(a, a) == pytest.approx(1.0)
    assert _an.paired_cosine(a, -a) == pytest.approx(-1.0)
    assert _an.paired_cosine(a, b) == pytest.approx(0.0)
    assert _an.paired_cosine(a, torch.zeros_like(a)) is None  # undefined, not 0.0
    # Cached norms are honored (no re-reduction).
    assert _an.paired_cosine(a, a, norm_a=1.0, norm_b=1.0) == pytest.approx(1.0)


def test_cos_over_targets_intersection_and_medians():
    a = {"x": torch.ones(2, 2), "y": torch.ones(2, 2), "only_a": torch.ones(2, 2)}
    b = {"x": torch.ones(2, 2), "y": -torch.ones(2, 2)}
    out = _an.cos_over_targets(a, b)
    assert set(out.keys()) == {"x", "y"}
    assert out["x"] == pytest.approx(1.0)
    assert out["y"] == pytest.approx(-1.0)
    assert _an.matrix_median(out) == pytest.approx(0.0)
    assert _an.matrix_median({"x": None}) is None  # all-None ⇒ None, not crash
    assert _an.matrix_median({"x": None, "y": 0.5}) == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# δ scale-consistency — the #25 mean-vs-sum trap (hard correctness invariant).
# --------------------------------------------------------------------------- #
def test_delta_scale_consistency_mean_vs_sum():
    """``delta_stats_over_targets`` is pure linear algebra (NO hidden rescaling),
    so the m5 ratio is correct iff BOTH feeds are DP-MEAN-reduced under the same
    loss normalization. Synthetic multi-rank case:

    * correct pipeline (mean+mean) is world-size INVARIANT;
    * the trap (SUM-reduced anchor vs MEAN-reduced fast grad) on the
      identical-global-objects construction (true δ = 0) yields a ratio of
      exactly (world − 1) — i.e. m5/GATE-B2 off by a world-size factor.
    """
    name = "layer.q_proj.weight"
    torch.manual_seed(0)
    per_rank_rep = [torch.randn(6, 6) for _ in range(4)]

    def mean(ts):
        return torch.stack(ts).mean(dim=0)

    def sum_(ts):
        return torch.stack(ts).sum(dim=0)

    for world in (2, 4):
        ranks_rep = per_rank_rep[:world]
        # Construct per-rank fast grads so the GLOBAL means are IDENTICAL
        # objects (codec error exactly zero): comp_r = rep_{(r+1) % world}
        # permutes the ranks WITHIN this world, leaving the mean unchanged.
        ranks_comp = [ranks_rep[(r + 1) % world] for r in range(world)]
        rep_mean = {name: mean(ranks_rep)}
        comp_mean = {name: mean(ranks_comp)}
        ratio, _cos = _an.delta_stats_over_targets(rep_mean, comp_mean)
        # True δ == 0 ⇒ ratio ~0 under the correct mean+mean pipeline, for
        # EVERY world size (invariance; fp32 summation-order noise only).
        assert ratio[name] == pytest.approx(0.0, abs=1e-6), f"world={world}"

        # The trap: anchor side SUM-reduced (the #25 bug) while comp is MEAN.
        rep_sum = {name: sum_(ranks_rep)}
        ratio_bug, _ = _an.delta_stats_over_targets(rep_sum, comp_mean)
        # sum = world*mean ⇒ δ_bug = (world−1)*global ⇒ ratio == world−1 exactly.
        assert ratio_bug[name] == pytest.approx(world - 1.0, rel=1e-5), (
            f"world={world}: the mean-vs-sum trap must inflate the m5 ratio by "
            f"exactly world−1 on the identical-objects construction"
        )


def test_delta_stats_known_geometry():
    name = "w"
    ring = {name: torch.ones(3, 3)}
    rep = {name: 2.0 * torch.ones(3, 3)}
    ratio, cos = _an.delta_stats_over_targets(rep, ring)
    assert ratio[name] == pytest.approx(1.0)  # ‖δ‖ = ‖ring‖
    assert cos[name] == pytest.approx(1.0)  # δ ∥ ring
    # rep == ring ⇒ δ = 0 ⇒ ratio 0, cos undefined (None).
    ratio, cos = _an.delta_stats_over_targets({name: torch.ones(3, 3)}, ring)
    assert ratio[name] == pytest.approx(0.0)
    assert cos[name] is None


# --------------------------------------------------------------------------- #
# β_anc = 0 semantics — M_rep == G_anc_rep EXACTLY (plan unit-test mandate).
# --------------------------------------------------------------------------- #
def test_beta_anc_zero_yields_latest_fire_exactly():
    f = SpectralFilter(beta_anc=0.0, correction_mode="none")
    g1 = torch.randn(5, 5)
    out1 = f.update_anchor("layer.q_proj.weight", g1)
    # EXACT: no bias-correction division, no (1−β) scaling surprise — bitwise.
    assert torch.equal(out1, g1.to(torch.float32))
    assert torch.equal(f._anchor["layer.q_proj.weight"], g1.to(torch.float32))
    # A second fire fully REPLACES (zero memory of the first).
    g2 = torch.randn(5, 5)
    f.update_anchor("layer.q_proj.weight", g2)
    assert torch.equal(f._anchor["layer.q_proj.weight"], g2.to(torch.float32))


# --------------------------------------------------------------------------- #
# geometry_fire_record — verbatim field names + warmup nulls + known values.
# --------------------------------------------------------------------------- #
_PLAN_FIELDS = [
    "step", "tick", "warmup_fallback",
    "m1_matrix_median", "m2_matrix_median", "m3_matrix_median",
    "m4_j1", "m4_j2", "m4_j3", "m4_j4", "m4_j5",
    "m5_ratio_matrix_median", "m5_cos_matrix_median",
    "m6_matrix_median", "m7_stable_rank_median", "m7_top1pct_mass_median",
    "loss_mismatch_nats",
]


def _mk_fire_inputs(n_targets=3, d=4):
    names = [f"layers.{i}.q_proj.weight" for i in range(n_targets)]
    g = {n: torch.ones(d, d) for n in names}
    norms = {n: float(torch.linalg.norm(g[n]).item()) for n in names}
    stats = {n: _an.grad_summary_stats(g[n]) for n in names}
    return names, g, norms, stats


def test_geometry_fire_record_warmup_nulls_and_field_names():
    names, g, norms, stats = _mk_fire_inputs()
    record, per_target = _an.geometry_fire_record(
        step=3, tick=5, warmup_fallback=True, fire_index=1,
        g_comp=g, g_comp_norms=norms,
        rep=g, rep_norms=norms, old=g, old_norms=norms, rep_stats=stats,
        lag_entries={5 - j: None for j in range(1, 6)},  # no lag history yet
        ring_entry=None, ring_tick=0, prev_rep=None,
        loss_mismatch_nats=0.009, used_tick=5, batch_gs=3,
        realized_weight_delay=0, m4_lags=5,
    )
    for field in _PLAN_FIELDS:
        assert field in record, f"plan contract field {field!r} missing from the JSONL record"
    # Warmup: missing structures are JSON null, never fabricated numbers.
    assert record["m5_ratio_matrix_median"] is None
    assert record["m5_cos_matrix_median"] is None
    assert record["m6_matrix_median"] is None
    for j in range(1, 6):
        assert record[f"m4_j{j}"] is None
    assert record["warmup_fallback"] is True
    assert record["n_targets"] == 3
    # Identical inputs ⇒ m1/m2/m3 all exactly 1.
    assert record["m1_matrix_median"] == pytest.approx(1.0)
    assert record["m2_matrix_median"] == pytest.approx(1.0)
    assert record["m3_matrix_median"] == pytest.approx(1.0)
    assert record["loss_mismatch_nats"] == pytest.approx(0.009)
    # JSON-serializable end to end (the appender contract).
    json.dumps(record)
    json.dumps(per_target)


def test_geometry_fire_record_post_warmup_known_cosines():
    names, g, norms, stats = _mk_fire_inputs()
    rep = {n: 2.0 * torch.ones(4, 4) for n in names}  # ∥ g_comp, 2× magnitude
    rep_norms = {n: float(torch.linalg.norm(rep[n]).item()) for n in names}
    old = {n: -torch.ones(4, 4) for n in names}  # anti-parallel
    old_norms = {n: float(torch.linalg.norm(old[n]).item()) for n in names}
    ring_grads = {n: torch.ones(4, 4) for n in names}
    ring_norms = {n: float(torch.linalg.norm(ring_grads[n]).item()) for n in names}
    lag_entry = ({n: torch.ones(4, 4) for n in names}, dict(norms))
    prev = ({n: torch.ones(4, 4) for n in names}, dict(norms))

    record, per_target = _an.geometry_fire_record(
        step=7, tick=10, warmup_fallback=False, fire_index=2,
        g_comp=g, g_comp_norms=norms,
        rep=rep, rep_norms=rep_norms, old=old, old_norms=old_norms, rep_stats=stats,
        lag_entries={10 - j: lag_entry for j in range(1, 6)},
        ring_entry=(ring_grads, ring_norms), ring_tick=5,
        prev_rep=prev, loss_mismatch_nats=0.01,
        used_tick=5, batch_gs=3, realized_weight_delay=5, m4_lags=5,
    )
    assert record["m1_matrix_median"] == pytest.approx(1.0)   # rep ∥ g_comp
    assert record["m2_matrix_median"] == pytest.approx(-1.0)  # old anti ∥
    assert record["m3_matrix_median"] == pytest.approx(-1.0)
    for j in range(1, 6):
        assert record[f"m4_j{j}"] == pytest.approx(1.0)
    # δ = rep − ring = ones ⇒ ratio 1, cos(δ, ring) = 1.
    assert record["m5_ratio_matrix_median"] == pytest.approx(1.0)
    assert record["m5_cos_matrix_median"] == pytest.approx(1.0)
    assert record["m6_matrix_median"] == pytest.approx(1.0)
    assert record["ring_tick_consumed"] == 5
    assert per_target[names[0]]["m1"] == pytest.approx(1.0)


def test_append_jsonl_roundtrip(tmp_path):
    p = tmp_path / "sub" / "stepA_fires.jsonl"
    _an.append_jsonl(str(p), {"tick": 5, "m1_matrix_median": 0.25})
    _an.append_jsonl(str(p), {"tick": 10, "m1_matrix_median": None})
    rows = [json.loads(line) for line in p.read_text().splitlines()]
    assert rows[0]["tick"] == 5 and rows[0]["m1_matrix_median"] == 0.25
    assert rows[1]["m1_matrix_median"] is None


# --------------------------------------------------------------------------- #
# Off-path parity + build wiring on the state object.
# --------------------------------------------------------------------------- #
class _Duck:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def _mk_config(probe_on=False, correction_mode="none", spectral_enabled=True):
    return _Duck(
        enabled=True,
        compression_type="dense",
        mask=None,
        clean_cadence=0,
        anchor=_Duck(enabled=True, cadence=5, delay_K=5, owns_q=False,
                     replay_paired_batch=True, snapshot_device="cpu"),
        spectral=_Duck(enabled=spectral_enabled, beta_anc=0.0, cadence=1,
                       correction_mode=correction_mode, inject_gamma=1.0,
                       blend_eta=0.3, signed_ema_alpha=0.0, ef_decay=0.0,
                       ef_clip=0.0, delayed_ef_lambda=0.0, ema_device="cpu",
                       max_targets=-1),
        capture=None,
        probe=_Duck(geometry_enabled=probe_on, out_dir="", rank0_only=True,
                    m4_lags=5, per_target_sidecar=True),
    )


def test_state_off_path_parity_no_probe_structures():
    state = _st.CommEffState(_mk_config(probe_on=False, correction_mode="signed_ema"))
    state.build(None)
    assert state.fast_grad_ring is None
    assert state.grad_lag_buffer is None
    assert state._probe_fire_stash is None
    assert state._probe_prev_rep is None
    m = state.metrics()
    assert m["comm_eff/geometry_probe_fires"] == 0


def test_state_build_arms_probe_structures():
    state = _st.CommEffState(_mk_config(probe_on=True, correction_mode="none"))
    state.build(None)
    assert isinstance(state.fast_grad_ring, FastGradRing)
    assert state.fast_grad_ring._maxlen == 2
    assert state.fast_grad_ring.delay_K == 5 and state.fast_grad_ring.cadence == 5
    assert isinstance(state.grad_lag_buffer, GradLagBuffer)
    assert state.grad_lag_buffer.max_lag == 5


def test_state_build_arms_ring_for_delayed_ef_without_probe():
    state = _st.CommEffState(_mk_config(probe_on=False, correction_mode="delayed_ef"))
    state.build(None)
    assert isinstance(state.fast_grad_ring, FastGradRing)  # δ needs the ring
    assert state.grad_lag_buffer is None  # the lag buffer is probe-only


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
