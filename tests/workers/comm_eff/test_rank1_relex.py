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

"""CPU coverage for pure sliding rank1_relex checkpoint projection."""

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


_la = _load("verl_workers_comm_eff_rank1_relex_testonly", "verl/workers/comm_eff/lookahead.py")


def _cfg(
    *,
    window=4,
    min_snapshots=-1,
    strength=1.0,
    mode="rank1_relex",
    enabled=True,
    rollout_source="auto",
):
    return types.SimpleNamespace(
        lookahead_anchor=enabled,
        lookahead_mode=mode,
        lookahead_strength=strength,
        lookahead_rollout_source=rollout_source,
        lookahead_min_snapshots=min_snapshots,
        lookahead_window_snapshots=window,
    )


def _snapshot(tick, *, base, direction, residual=None, dtype=torch.float32):
    target = base + float(tick) * direction
    if residual is not None:
        target = target + residual
    return {
        "layers.0.q_proj.weight": target.to(dtype),
        "layers.0.o_proj.weight": (0.5 * target).to(dtype),
        "layers.0.q_proj.bias": (
            torch.tensor([0.1, -0.2], dtype=torch.float32)
            + float(tick) * torch.tensor([0.01, 0.03], dtype=torch.float32)
        ).to(dtype),
        "model.norm.weight": (torch.arange(target.shape[0], dtype=torch.float32) + 0.01 * float(tick)).to(dtype),
        "embed_tokens.weight": torch.full_like(target, float(tick)),
    }


def _linear_history(ticks, *, dtype=torch.float32):
    base = torch.tensor([[0.2, -0.3], [0.5, 0.7]], dtype=torch.float32)
    # Matrix-rank two: rank-1 refers to the temporal trajectory, not matrix rank.
    direction = torch.tensor([[0.03, -0.02], [0.01, 0.04]], dtype=torch.float32)
    assert torch.linalg.matrix_rank(direction) == 2
    return [_snapshot(t, base=base, direction=direction, dtype=dtype) for t in ticks], base, direction


def test_rank1_mode_uses_complete_configured_window():
    cfg = _cfg(window=5)
    assert _la.lookahead_enabled(cfg)
    assert _la.rank1_relex_enabled(cfg)
    assert _la.lookahead_num_source_points(cfg) == 5
    assert _la.lookahead_min_points(cfg) == 5


def test_rank1_enable_predicates_require_master_flag_and_supported_mode():
    assert _la.lookahead_enabled(_cfg())
    assert _la.rank1_relex_enabled(_cfg())
    assert not _la.lookahead_enabled(_cfg(enabled=False))
    assert not _la.rank1_relex_enabled(_cfg(enabled=False))
    assert not _la.lookahead_enabled(_cfg(mode="disabled"))
    assert not _la.lookahead_enabled(_cfg(mode="unknown"))
    assert not _la.lookahead_enabled(None)
    assert _la.lookahead_num_source_points(_cfg(enabled=False)) == 0
    assert _la.lookahead_min_points(_cfg(enabled=False)) == 0


def test_rollout_source_auto_tracks_rank1_state_and_explicit_values_pass_through():
    assert _la.resolve_lookahead_rollout_source(_cfg()) == "current_step"
    assert _la.resolve_lookahead_rollout_source(_cfg(enabled=False)) == "stale_paired"
    assert _la.resolve_lookahead_rollout_source(_cfg(mode="disabled")) == "stale_paired"
    assert _la.resolve_lookahead_rollout_source(None) == "stale_paired"
    assert _la.resolve_lookahead_rollout_source(_cfg(rollout_source="stale_paired")) == "stale_paired"
    assert _la.resolve_lookahead_rollout_source(_cfg(rollout_source="current_step")) == "current_step"


def test_rank1_progressive_mode_resolves_explicit_minimum_without_shrinking_window():
    cfg = _cfg(window=4, min_snapshots=2)
    assert _la.lookahead_num_source_points(cfg) == 4
    assert _la.lookahead_min_points(cfg) == 2


def test_w4_exact_schedule_and_sliding_rebase():
    history = _la.Rank1SnapshotHistory(4)
    assert history.seed_base(1, {"w": torch.tensor([1.0])})
    assert history.ticks == [1] and not history.ready()
    for tick in (19, 39, 59):
        assert history.admit_exact(tick, {"w": torch.tensor([float(tick)])})
    assert history.ready()
    snapshots, ticks = history.sources()
    assert ticks == [1, 19, 39, 59]
    assert len(snapshots) == 4
    assert history.admit_exact(79, {"w": torch.tensor([79.0])})
    assert history.ticks == [19, 39, 59, 79]
    assert history.peak_retained == 4


def test_w4_progressive_projects_with_two_then_three_then_four_checkpoints():
    cfg = _cfg(window=4, min_snapshots=2)
    history = _la.Rank1SnapshotHistory(4, min_snapshots=2)
    snapshots, base, direction = _linear_history([1, 19, 39, 59, 79])
    projector = _la.Rank1RelexProjector(cfg, chunk_numel=2)

    assert history.seed_base(1, snapshots[0])
    assert not history.ready()
    assert history.sources() == (None, None)

    expected_stages = [
        (19, snapshots[1], 39, 2, "two_checkpoint_secant"),
        (39, snapshots[2], 59, 3, "rank1_ols"),
        (59, snapshots[3], 79, 4, "rank1_ols"),
    ]
    for source_tick, snapshot, target_tick, checkpoint_count, fit_kind in expected_stages:
        assert history.admit_exact(source_tick, snapshot)
        assert history.ready()
        sources, ticks = history.sources()
        assert len(sources) == checkpoint_count
        assert len(ticks) == checkpoint_count

        projected, info = projector.project(sources, ticks, target_tick=target_tick)
        torch.testing.assert_close(
            projected["layers.0.q_proj.weight"],
            base + float(target_tick) * direction,
            rtol=1e-5,
            atol=1e-6,
        )
        assert info["checkpoint_count"] == checkpoint_count
        assert info["delta_count"] == checkpoint_count - 1
        assert info["fit_kind"] == fit_kind

    # The target remains W=4: once full, each new exact transfer slides rather
    # than growing the retained history beyond four model checkpoints.
    assert history.admit_exact(79, snapshots[4])
    sources, ticks = history.sources()
    assert ticks == [19, 39, 59, 79]
    assert len(sources) == 4
    assert history.peak_retained == 4


def test_w2_is_explicit_per_tensor_two_checkpoint_secant():
    ticks = [3, 11]
    target_tick = 19
    snapshots, _base, _direction = _linear_history(ticks)
    projector = _la.Rank1RelexProjector(_cfg(window=2), chunk_numel=2)

    projected, info = projector.project(snapshots, ticks, target_tick)

    ratio = float(target_tick - ticks[-1]) / float(ticks[-1] - ticks[0])
    for name, latest in snapshots[-1].items():
        expected = latest + ratio * (latest - snapshots[0][name])
        torch.testing.assert_close(projected[name], expected, rtol=1e-5, atol=1e-6)
    assert info["fit_kind"] == "two_checkpoint_secant"
    assert info["checkpoint_count"] == 2
    assert info["delta_count"] == 1
    assert info["evr_mean"] == pytest.approx(1.0)
    assert info["r2_mean"] == pytest.approx(1.0)


def test_w2_history_fires_then_slides_on_each_new_exact_checkpoint():
    history = _la.Rank1SnapshotHistory(2)
    assert history.seed_base(1, {"w": torch.tensor([1.0])})
    assert history.admit_exact(19, {"w": torch.tensor([19.0])})
    assert history.ready() and history.ticks == [1, 19]
    assert history.admit_exact(39, {"w": torch.tensor([39.0])})
    assert history.ticks == [19, 39]
    assert history.peak_retained == 2


def test_seed_base_first_call_wins_without_scanning_unused_later_snapshots():
    history = _la.Rank1SnapshotHistory(4)
    base = {"w": torch.tensor([1.0])}
    assert history.seed_base(1, base)
    # The engine offers every later generator snapshot to seed_base. They are
    # not history admissions and must return before schema/finite validation.
    assert not history.seed_base(3, {"malformed": None})
    assert history.ticks == [1]
    assert history.latest()[0] is base


def test_canonical_fire_schedule_projects_to_current_generator_ticks():
    history = _la.Rank1SnapshotHistory(4)
    snapshots, base, direction = _linear_history([1, 19, 39, 59, 79])
    assert history.seed_base(1, snapshots[0])

    # Tick 20 is the replay warmup fallback and must not be admitted as a new
    # checkpoint. Ticks 40/60/80 carry exact generator transfers 19/39/59.
    assert history.ticks == [1] and not history.ready()
    for fire_tick, source_tick, snapshot in zip((40, 60), (19, 39), snapshots[1:3], strict=True):
        assert history.admit_exact(source_tick, snapshot), fire_tick
        assert not history.ready()
    assert history.admit_exact(59, snapshots[3])
    window, ticks = history.sources()
    projected, info = _la.Rank1RelexProjector(_cfg()).project(window, ticks, target_tick=79)
    assert ticks == [1, 19, 39, 59]
    assert info["prediction_horizon"] == 20
    torch.testing.assert_close(projected["layers.0.q_proj.weight"], base + 79.0 * direction, rtol=1e-5, atol=1e-6)

    # Tick 100 admits generator tick 79, drops the original local base, and
    # projects one communication-delay horizon to generator tick 99.
    assert history.admit_exact(79, snapshots[4])
    window, ticks = history.sources()
    projected, info = _la.Rank1RelexProjector(_cfg()).project(window, ticks, target_tick=99)
    assert ticks == [19, 39, 59, 79]
    assert info["prediction_horizon"] == 20
    torch.testing.assert_close(projected["layers.0.q_proj.weight"], base + 99.0 * direction, rtol=1e-5, atol=1e-6)


def test_w5_produces_four_deltas():
    ticks = [1, 5, 12, 30, 55]
    snapshots, _base, _direction = _linear_history(ticks)
    _projected, info = _la.Rank1RelexProjector(_cfg(window=5)).project(snapshots, ticks, target_tick=70)
    assert info["checkpoint_count"] == 5
    assert info["delta_count"] == 4


def test_duplicate_and_out_of_order_exact_transfers_are_excluded():
    history = _la.Rank1SnapshotHistory(4)
    history.seed_base(1, {"w": torch.tensor([1.0])})
    history.admit_exact(19, {"w": torch.tensor([19.0])})
    assert not history.admit_exact(19, {"w": torch.tensor([-1.0])})
    assert not history.admit_exact(18, {"w": torch.tensor([-1.0])})
    assert history.ticks == [1, 19]
    assert history.latest()[0]["w"].item() == 19.0


@pytest.mark.parametrize("failure", ["key", "shape", "dtype", "nonfinite", "noncontiguous"])
def test_history_rejects_malformed_exact_transfer_before_q_only_can_use_it(failure):
    history = _la.Rank1SnapshotHistory(4)
    base = {
        "w": torch.ones(3, 2, dtype=torch.float32),
        "norm": torch.ones(3, dtype=torch.float32),
    }
    assert history.seed_base(1, base)
    malformed = {name: tensor.clone() for name, tensor in base.items()}
    if failure == "key":
        malformed.pop("norm")
        match = "key mismatch"
    elif failure == "shape":
        malformed["norm"] = torch.ones(4)
        match = "shape mismatch"
    elif failure == "dtype":
        malformed["norm"] = malformed["norm"].to(torch.float64)
        match = "dtype mismatch"
    elif failure == "nonfinite":
        malformed["norm"][0] = float("nan")
        match = "non-finite"
    else:
        malformed["w"] = torch.ones(2, 3).t()
        assert not malformed["w"].is_contiguous()
        match = "non-contiguous"

    with pytest.raises(_la.Rank1ProjectionError, match=match):
        history.admit_exact(19, malformed)
    assert history.ticks == [1]
    assert history.latest()[0] is base


def test_duplicate_or_out_of_order_transfer_after_readiness_fails_closed():
    history = _la.Rank1SnapshotHistory(4)
    history.seed_base(1, {"w": torch.tensor([1.0])})
    for tick in (19, 39, 59):
        history.admit_exact(tick, {"w": torch.tensor([float(tick)])})
    assert history.ready()

    # Once ready, the timeline guard runs before any expensive payload scan;
    # this malformed duplicate must still produce the deterministic tick error.
    with pytest.raises(_la.Rank1ProjectionError, match="strictly newer exact transfer"):
        history.admit_exact(59, {"malformed": None})
    with pytest.raises(_la.Rank1ProjectionError, match="strictly newer exact transfer"):
        history.admit_exact(39, {"w": torch.tensor([39.0])})
    assert history.ticks == [1, 19, 39, 59]


def test_resume_history_starts_from_a_fresh_local_base():
    old = _la.Rank1SnapshotHistory(4)
    old.seed_base(1, {"w": torch.tensor([1.0])})
    for tick in (19, 39, 59):
        old.admit_exact(tick, {"w": torch.tensor([float(tick)])})
    assert old.ready()
    resumed = _la.Rank1SnapshotHistory(4)
    resumed.seed_base(101, {"w": torch.tensor([101.0])})
    assert resumed.ticks == [101]
    assert not resumed.ready()


def test_irregular_actual_ticks_project_every_parameter_tensor_independently():
    ticks = [1, 7, 20, 31]
    target_tick = 45
    snapshots, base, direction = _linear_history(ticks)
    projected, info = _la.Rank1RelexProjector(_cfg(), chunk_numel=3).project(snapshots, ticks, target_tick)
    torch.testing.assert_close(
        projected["layers.0.q_proj.weight"],
        base + float(target_tick) * direction,
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(
        projected["layers.0.q_proj.bias"],
        torch.tensor([0.1, -0.2]) + float(target_tick) * torch.tensor([0.01, 0.03]),
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(
        projected["model.norm.weight"],
        torch.arange(2, dtype=torch.float32) + 0.01 * float(target_tick),
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(
        projected["embed_tokens.weight"],
        torch.full_like(projected["embed_tokens.weight"], float(target_tick)),
        rtol=1e-5,
        atol=1e-6,
    )
    assert info["targets_projected"] == len(snapshots[-1])
    assert info["history_ticks"] == tuple(ticks)
    assert info["window_span"] == 30
    assert info["prediction_horizon"] == 14
    assert info["evr_mean"] == pytest.approx(1.0, abs=1e-5)
    assert info["r2_mean"] == pytest.approx(1.0, abs=1e-5)


def test_nonfloating_tensor_is_pinned_to_latest_exact_checkpoint():
    ticks = [1, 7, 20, 31]
    snapshots, _base, _direction = _linear_history(ticks)
    for tick, snapshot in zip(ticks, snapshots, strict=True):
        snapshot["synthetic.position_ids"] = torch.tensor([tick, tick + 1], dtype=torch.int64)

    projected, info = _la.Rank1RelexProjector(_cfg()).project(snapshots, ticks, 45)

    assert projected["synthetic.position_ids"] is snapshots[-1]["synthetic.position_ids"]
    assert info["targets_projected"] == len(snapshots[-1]) - 1
    assert info["nonfloating_tensors_passthrough"] == 1


@pytest.mark.parametrize("chunk_numel", [1, 2, 3, 4, 17])
def test_chunk_boundaries_match_dense_linear_oracle(chunk_numel):
    ticks = [1, 7, 20, 31]
    snapshots, _base, _direction = _linear_history(ticks)
    tensors = [snapshot["layers.0.q_proj.weight"] for snapshot in snapshots]
    projected, stats = _la.project_rank1_tensor(
        tensors,
        ticks,
        45,
        chunk_numel=chunk_numel,
    )
    dense = torch.stack([(tensor - tensors[0]).flatten() for tensor in tensors[1:]])
    u, singular, vh = torch.linalg.svd(dense, full_matrices=False)
    coeff = u[:, 0] * singular[0]
    t = torch.tensor(ticks[1:], dtype=torch.float64)
    c = coeff.to(torch.float64)
    slope = torch.sum((t - t.mean()) * (c - c.mean())) / torch.sum((t - t.mean()).square())
    expected = tensors[-1] + float(slope.item()) * (45 - ticks[-1]) * vh[0].reshape_as(tensors[-1])
    torch.testing.assert_close(projected, expected, rtol=1e-5, atol=1e-6)
    assert stats["delta_count"] == 3


@pytest.mark.parametrize("strength", [0.0, 0.25, 1.0])
def test_projection_strength_scales_only_the_pinned_increment(strength):
    ticks = [1, 7, 20, 31]
    snapshots, _base, direction = _linear_history(ticks)
    tensors = [snapshot["layers.0.q_proj.weight"] for snapshot in snapshots]
    projected, _stats = _la.project_rank1_tensor(tensors, ticks, 45, strength=strength, chunk_numel=2)
    expected = tensors[-1] + strength * float(45 - ticks[-1]) * direction
    torch.testing.assert_close(projected, expected, rtol=1e-5, atol=1e-6)


def test_anchor_pinning_preserves_latest_off_subspace_residual():
    ticks = [1, 7, 20, 31]
    snapshots, _base, _direction = _linear_history(ticks)
    residual = torch.tensor([[0.4, -0.1], [-0.2, 0.3]])
    snapshots[-1]["layers.0.q_proj.weight"] += residual
    tensors = [snapshot["layers.0.q_proj.weight"] for snapshot in snapshots]
    projected, _stats = _la.project_rank1_tensor(tensors, ticks, 45, chunk_numel=2)

    dense = torch.stack([(tensor - tensors[0]).flatten() for tensor in tensors[1:]])
    u, singular, vh = torch.linalg.svd(dense, full_matrices=False)
    coeff = u[:, 0] * singular[0]
    t = torch.tensor(ticks[1:], dtype=torch.float64)
    c = coeff.to(torch.float64)
    slope = torch.sum((t - t.mean()) * (c - c.mean())) / torch.sum((t - t.mean()).square())
    expected_pinned = tensors[-1] + float(slope.item()) * 14.0 * vh[0].reshape_as(tensors[-1])
    torch.testing.assert_close(projected, expected_pinned, rtol=1e-5, atol=1e-6)
    # Original RELEX's base-pinned reconstruction is intentionally different.
    intercept = c.mean() - slope * t.mean()
    base_pinned = tensors[0] + float((slope * 45.0 + intercept).item()) * vh[0].reshape_as(tensors[0])
    assert not torch.allclose(projected, base_pinned)


def test_bf16_restoration_and_zero_motion_identity():
    ticks = [1, 7, 20, 31]
    snapshots, _base, _direction = _linear_history(ticks, dtype=torch.bfloat16)
    tensors = [snapshot["layers.0.q_proj.weight"] for snapshot in snapshots]
    projected, _stats = _la.project_rank1_tensor(tensors, ticks, 45, chunk_numel=3)
    assert projected.dtype == torch.bfloat16

    still = [torch.ones(3, 3, dtype=torch.bfloat16) for _ in ticks]
    unchanged, stats = _la.project_rank1_tensor(still, ticks, 45, chunk_numel=2)
    assert unchanged is still[-1]
    assert stats["zero_motion"]
    assert stats["evr"] == 0.0 and stats["slope"] == 0.0


@pytest.mark.parametrize(
    "ticks,target",
    [
        ([1, 7, 7, 31], 45),
        ([1, 20, 7, 31], 45),
        ([1, 7, 20, 31], 31),
        ([1, 7, 20, 31.5], 45),
    ],
)
def test_invalid_timestamps_fail_closed(ticks, target):
    snapshots, _base, _direction = _linear_history([1, 7, 20, 31])
    tensors = [snapshot["layers.0.q_proj.weight"] for snapshot in snapshots]
    with pytest.raises(_la.Rank1ProjectionError, match="tick|strictly increasing|newer"):
        _la.project_rank1_tensor(tensors, ticks, target)


def test_public_tensor_helper_rejects_dtype_mismatch_and_nonfloating_history():
    tensors = [torch.ones(2, dtype=torch.float32), torch.ones(2, dtype=torch.float64)]
    with pytest.raises(_la.Rank1ProjectionError, match="dtype mismatch"):
        _la.project_rank1_tensor(tensors, [1, 2], 3)

    integer_tensors = [torch.ones(2, dtype=torch.int64), torch.ones(2, dtype=torch.int64)]
    with pytest.raises(_la.Rank1ProjectionError, match="floating point"):
        _la.project_rank1_tensor(integer_tensors, [1, 2], 3)


def test_missing_and_mismatched_targets_fail_without_mutating_sources():
    ticks = [1, 7, 20, 31]
    snapshots, _base, _direction = _linear_history(ticks)
    originals = [snapshot["layers.0.q_proj.weight"].clone() for snapshot in snapshots]
    snapshots[2].pop("layers.0.q_proj.weight")
    with pytest.raises(_la.Rank1ProjectionError, match="key mismatch"):
        _la.Rank1RelexProjector(_cfg()).project(snapshots, ticks, 45)
    for original, snapshot in zip(originals[:2], snapshots[:2], strict=True):
        torch.testing.assert_close(snapshot["layers.0.q_proj.weight"], original)

    snapshots, _base, _direction = _linear_history(ticks)
    snapshots[2]["layers.0.q_proj.weight"] = torch.ones(3, 3)
    with pytest.raises(_la.Rank1ProjectionError, match="shape mismatch"):
        _la.Rank1RelexProjector(_cfg()).project(snapshots, ticks, 45)


@pytest.mark.parametrize("failure", ["non_tensor", "shape", "dtype", "nonfinite"])
def test_malformed_parameter_checkpoint_fails_before_projection(failure):
    ticks = [1, 7, 20, 31]
    snapshots, _base, _direction = _linear_history(ticks)
    if failure == "non_tensor":
        snapshots[-1]["model.norm.weight"] = None
        match = "must be a torch.Tensor"
    elif failure == "shape":
        snapshots[-1]["model.norm.weight"] = torch.ones(3)
        match = "shape mismatch"
    elif failure == "dtype":
        snapshots[-1]["model.norm.weight"] = snapshots[-1]["model.norm.weight"].to(torch.float64)
        match = "dtype mismatch"
    else:
        snapshots[-1]["model.norm.weight"][0] = float("nan")
        match = "non-finite"
    with pytest.raises(_la.Rank1ProjectionError, match=match):
        _la.Rank1RelexProjector(_cfg(), chunk_numel=1).project(snapshots, ticks, 45)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_checkpoint_values_fail_closed(bad):
    ticks = [1, 7, 20, 31]
    snapshots, _base, _direction = _linear_history(ticks)
    snapshots[1]["layers.0.q_proj.weight"][0, 0] = bad
    with pytest.raises(_la.Rank1ProjectionError, match="non-finite"):
        _la.Rank1RelexProjector(_cfg()).project(snapshots, ticks, 45)


def test_eigensolver_failure_is_wrapped_in_dedicated_error(monkeypatch):
    ticks = [1, 7, 20, 31]
    snapshots, _base, _direction = _linear_history(ticks)
    tensors = [snapshot["layers.0.q_proj.weight"] for snapshot in snapshots]

    def fail(_gram):
        raise RuntimeError("synthetic eigh failure")

    monkeypatch.setattr(_la.torch.linalg, "eigh", fail)
    with pytest.raises(_la.Rank1ProjectionError, match="eigensolver failed"):
        _la.project_rank1_tensor(tensors, ticks, 45)


def test_eigenvector_sign_flip_leaves_prediction_unchanged(monkeypatch):
    ticks = [1, 7, 20, 31]
    snapshots, _base, _direction = _linear_history(ticks)
    tensors = [snapshot["layers.0.q_proj.weight"] for snapshot in snapshots]
    reference, _stats = _la.project_rank1_tensor(tensors, ticks, 45)
    original_eigh = torch.linalg.eigh

    def flipped(gram):
        values, vectors = original_eigh(gram)
        vectors = vectors.clone()
        vectors[:, -1].neg_()
        return values, vectors

    monkeypatch.setattr(_la.torch.linalg, "eigh", flipped)
    actual, _stats = _la.project_rank1_tensor(tensors, ticks, 45)
    torch.testing.assert_close(actual, reference)


def test_distributed_q_only_and_full_receipt_contracts():
    q = {0: {"changed": True}}
    m = {"layers.0.q_proj.weight": {"changed": True}}
    _la.validate_rank1_broadcast_receipts(
        q_only=True, dp_multi=True, q_receipts=q, m_receipts=None, spectral_enabled=True
    )
    _la.validate_rank1_broadcast_receipts(
        q_only=False, dp_multi=True, q_receipts=q, m_receipts=m, spectral_enabled=True
    )
    with pytest.raises(RuntimeError, match="must not broadcast M"):
        _la.validate_rank1_broadcast_receipts(
            q_only=True, dp_multi=True, q_receipts=q, m_receipts=m, spectral_enabled=True
        )
    with pytest.raises(RuntimeError, match="must not broadcast M"):
        _la.validate_rank1_broadcast_receipts(
            q_only=True, dp_multi=False, q_receipts=None, m_receipts={}, spectral_enabled=True
        )
    with pytest.raises(RuntimeError, match="Q broadcast"):
        _la.validate_rank1_broadcast_receipts(
            q_only=True, dp_multi=True, q_receipts=None, m_receipts=None, spectral_enabled=True
        )
    with pytest.raises(RuntimeError, match="M broadcast"):
        _la.validate_rank1_broadcast_receipts(
            q_only=False, dp_multi=True, q_receipts=q, m_receipts=None, spectral_enabled=True
        )
